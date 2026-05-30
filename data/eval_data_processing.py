"""Produce a value-head fine-tuning dataset from `data/chessData.csv`.

The source CSV contains (FEN, Stockfish-depth-22 evaluation) pairs. This
script converts each row into the same 68-token board encoding used by
`data_processing.py` and writes a `.pt` file with the same dict layout
expected by `training/train.py`:

    {
      'tokens': long  (N, 68),
      'moves':  long  (N,),     # filled with MOVE_IGNORE sentinel
      'values': float (N,)      # Stockfish eval mapped to [-1, 1]
    }

The `moves` column has no ground truth here (the dataset is positions +
evals, not games), so it's filled with the sentinel `-100`. To continue
training a model with this dataset, set:

    policy_criterion = nn.CrossEntropyLoss(ignore_index=-100)

in train.py — this skips policy loss on these positions while still
updating the value head (the actual goal of this dataset).
"""
from pathlib import Path
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import torch


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data import csv_parallel, pgn_parallel

CSV_PATH = _PROJECT_ROOT / "data" / "chessData.csv"
OUT_DIR = _PROJECT_ROOT / "data" / "processed"

# Default subset cap (matches the Phase 1 fine-tune dataset). Override with
# --full to process the entire CSV, or --max-positions N for a custom cap.
DEFAULT_MAX_POSITIONS = 1_500_000
DEFAULT_OUT_NAME = "stockfish_eval_dataset.pt"
FULL_OUT_NAME = "stockfish_eval_dataset_full.pt"

# Rows per worker task. ~5k keeps per-task overhead small while letting
# imap_unordered surface progress at a reasonable cadence on 13M rows.
CHUNK_SIZE = 5000


def iter_csv_chunks(path: Path, chunk_size: int, max_positions: int | None = None):
    """Yield lists of (fen, eval_str) tuples, `chunk_size` rows each.

    Stops early once `max_positions` rows have been yielded (if set).
    """
    total = 0
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # drop header
        chunk: list[tuple[str, str]] = []
        for row in reader:
            if max_positions is not None and total >= max_positions:
                break
            if len(row) < 2:
                continue
            chunk.append((row[0], row[1]))
            total += 1
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert chessData.csv (FEN, Stockfish-eval) into a "
                    "tokenized .pt dataset for value-head fine-tuning.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--full",
        action="store_true",
        help=f"Process the entire CSV. Defaults output to {FULL_OUT_NAME}.",
    )
    g.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help=f"Cap the number of positions written. Default cap: "
             f"{DEFAULT_MAX_POSITIONS:,}.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path. Defaults depend on mode "
             f"({DEFAULT_OUT_NAME} for subset, {FULL_OUT_NAME} for --full).",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    if args.full:
        max_positions: int | None = None
        default_out_name = FULL_OUT_NAME
    elif args.max_positions is not None:
        max_positions = args.max_positions
        default_out_name = DEFAULT_OUT_NAME
    else:
        max_positions = DEFAULT_MAX_POSITIONS
        default_out_name = DEFAULT_OUT_NAME

    out_path = Path(args.out) if args.out else OUT_DIR / default_out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    num_workers = max(1, (os.cpu_count() or 2) - 1)
    cap_str = f"{max_positions:,}" if max_positions is not None else "unlimited (full CSV)"
    print(f"Parsing {CSV_PATH.name} with {num_workers} workers (cap: {cap_str})")
    print(f"Output: {out_path}")
    t0 = time.time()

    tokens_chunks: list[torch.Tensor] = []
    values_chunks: list[torch.Tensor] = []
    processed_positions = 0
    processed_chunks = 0

    with Pool(num_workers) as pool:
        for result in pool.imap_unordered(
            csv_parallel.parse_csv_chunk,
            iter_csv_chunks(CSV_PATH, CHUNK_SIZE, max_positions),
            chunksize=1,
        ):
            if result is None:
                continue
            t, v = result
            tokens_chunks.append(t)
            values_chunks.append(v)
            processed_positions += v.size(0)
            processed_chunks += 1
            if processed_chunks % 50 == 0:
                print(
                    f"  {processed_positions:,} positions in "
                    f"{time.time() - t0:.1f}s"
                )

    if not values_chunks:
        print("ERROR: no rows parsed.")
        sys.exit(1)

    tokens_tensor = torch.cat(tokens_chunks, dim=0).to(torch.long)
    values_tensor = torch.cat(values_chunks, dim=0).to(torch.float32)
    moves_tensor = torch.full(
        (tokens_tensor.size(0),), csv_parallel.MOVE_IGNORE, dtype=torch.long
    )

    print(f"\nParsed {tokens_tensor.size(0):,} positions in {time.time() - t0:.1f}s")
    print(f"Tokens shape: {tokens_tensor.shape}")
    print(f"Moves shape:  {moves_tensor.shape} (all = {csv_parallel.MOVE_IGNORE})")
    print(f"Values shape: {values_tensor.shape}")

    print("\nSanity check:")
    print(f"  Sequence length: {tokens_tensor.shape[1]} (expected: {pgn_parallel.SEQ_LENGTH})")
    print(f"  Vocab size:      {pgn_parallel.VOCAB_SIZE}")
    print(
        f"  Token range:     [{tokens_tensor.min().item()}, "
        f"{tokens_tensor.max().item()}]"
    )
    print(
        f"  Value range:     [{values_tensor.min().item():.4f}, "
        f"{values_tensor.max().item():.4f}]"
    )
    print(
        f"  Value mean/std:  {values_tensor.mean().item():.4f} / "
        f"{values_tensor.std().item():.4f}"
    )

    torch.save(
        {
            "tokens": tokens_tensor,
            "moves": moves_tensor,
            "values": values_tensor,
        },
        out_path,
    )
    size_gb = out_path.stat().st_size / 1024 ** 3
    print(f"\nSaved tensors to {out_path} ({size_gb:.2f} GB)")
