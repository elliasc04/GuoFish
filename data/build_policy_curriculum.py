"""Phase 3 policy fine-tuning curriculum builder (single-stage).

Streams 3,000,000 valid positions from bingbangboom/stockfish-evaluation-SAN,
parses each FEN to our 68-token layout, resolves the SAN `best_move` to a
UCI move via python-chess, and writes a one-hot target over the 4096-move
vocabulary (from_square*64 + to_square).

Output: data/processed/policy_singlepv_3M.pt
        {'tokens': int8 [N, 68], 'targets': int16 [N]}

Targets are class indices in [0, 4095]; feed them straight to
F.cross_entropy(logits, targets.long(), label_smoothing=...) at train time.
A dense one-hot would be ~8000x larger for no information gain.
"""
from pathlib import Path
import sys
import time

import chess
import torch
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError as e:
    raise ImportError("Install the Hugging Face `datasets` package: pip install datasets") from e

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.pgn_parallel import _board_to_tokens, SEQ_LENGTH

OUT_DIR = _PROJECT_ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "policy_singlepv_3M.pt"

REPO = "bingbangboom/stockfish-evaluation-SAN"
TARGET_COUNT = 3_000_000
VOCAB_MOVES = 4096
LOG_EVERY = 100_000


def san_to_move_index(board: chess.Board, san: str) -> int:
    """Push SAN against board to get the legal Move, then encode as from*64+to."""
    mv = board.parse_san(san)
    return mv.from_square * 64 + mv.to_square


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Single-PV curriculum: {REPO} ===")
    print(f"Target positions: {TARGET_COUNT:,}")
    try:
        ds = load_dataset(REPO, split="train", streaming=True)
    except Exception as e:
        print(f"[ERROR] Failed to stream {REPO}: {e}")
        raise

    tokens_buf = torch.zeros((TARGET_COUNT, SEQ_LENGTH), dtype=torch.int8)
    targets_buf = torch.zeros((TARGET_COUNT,), dtype=torch.int16)

    kept = 0
    scanned = 0
    skipped_fen = 0
    skipped_san = 0
    t_start = time.time()
    t_window = t_start

    pbar = tqdm(total=TARGET_COUNT, desc="SAN->UCI", unit="pos")
    for row in ds:
        scanned += 1
        fen = row.get("fen")
        san = row.get("best_move")
        if not fen or not san:
            skipped_fen += 1
            continue

        try:
            board = chess.Board(fen)
        except Exception:
            skipped_fen += 1
            continue

        try:
            idx = san_to_move_index(board, san)
        except Exception:
            skipped_san += 1
            continue

        try:
            toks = _board_to_tokens(board)
        except Exception:
            skipped_fen += 1
            continue

        tokens_buf[kept] = torch.tensor(toks, dtype=torch.int8)
        targets_buf[kept] = idx
        kept += 1
        pbar.update(1)

        if kept % LOG_EVERY == 0:
            now = time.time()
            window_rate = LOG_EVERY / max(now - t_window, 1e-9)
            overall_rate = kept / max(now - t_start, 1e-9)
            print(
                f"  [{kept:,}/{TARGET_COUNT:,}] scanned={scanned:,} "
                f"window={window_rate:,.0f} pos/s overall={overall_rate:,.0f} pos/s "
                f"skipped(fen={skipped_fen:,}, san={skipped_san:,})"
            )
            t_window = now

        if kept >= TARGET_COUNT:
            break
    pbar.close()

    if kept < TARGET_COUNT:
        print(f"[WARN] only collected {kept:,}/{TARGET_COUNT:,} (stream exhausted)")
        tokens_buf = tokens_buf[:kept].contiguous()
        targets_buf = targets_buf[:kept].contiguous()

    assert tokens_buf.shape == (kept, SEQ_LENGTH), f"tokens shape {tokens_buf.shape}"
    assert targets_buf.shape == (kept,), f"targets shape {targets_buf.shape}"
    assert targets_buf.min().item() >= 0 and targets_buf.max().item() < VOCAB_MOVES, \
        f"targets out of range: [{targets_buf.min().item()}, {targets_buf.max().item()}]"

    elapsed = time.time() - t_start
    print(
        f"\nDone. kept={kept:,}  scanned={scanned:,}  "
        f"skipped(fen={skipped_fen:,}, san={skipped_san:,})  "
        f"elapsed={elapsed/60:.1f} min  avg={kept/max(elapsed,1e-9):,.0f} pos/s"
    )
    print(f"  tokens: {tuple(tokens_buf.shape)}  targets: {tuple(targets_buf.shape)}")

    torch.save({"tokens": tokens_buf, "targets": targets_buf}, OUT_PATH)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
