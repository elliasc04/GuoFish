"""Estimate player ELO from a PGN file via ACPL using our Guofish2 MCTS evaluator.

For each game, we compare q_before (target player's eval of the position they faced)
against q_after (eval of the position they handed back, with sign flipped). The
difference, converted to centipawns, is the per-move CPL. We average over the
target player's moves and map to an ELO estimate.

Usage:
    python acpl_elo_estimator.py [--pgn chess_com_games_2026-05-20.pgn] [--target USERNAME]
"""

# Pin intra-op threads to 1 before any torch ops — six concurrent ParallelMCTS
# instances each doing CPU-side tree work will otherwise fan out across cores
# and collide. cudnn.benchmark is a no-op for this model (no convolutions) but
# included per spec.
import torch
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True

import argparse
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.pgn

from core.mctsv1 import ParallelMCTS
from playing.v2.playv2 import load_model


MODEL_PATH = Path("models/guofish2_25.6M_54.8p.pt")
NUM_SIMULATIONS = 800
MAX_CONCURRENT_GAMES = 12  # Sized to VRAM: 6 × 25.6M FP16 ≈ 300MB
# Workers per game. With min_batch_size=1, each worker holds at most one
# in-flight forward request. GPU utilization scales with total in-flight requests
# across all games (MAX_CONCURRENT_GAMES * MCTS_WORKERS_PER_GAME). We have lots
# of headroom (CPU 28%, VRAM 2.2/12GB), so push higher.
MCTS_WORKERS_PER_GAME = 8
# Per-game transposition cache. Each entry is a [4096] FP16 policy tensor (~8KB);
# 500K × 8KB × 6 games would balloon RAM into the tens of GB. 50K covers
# in-search transpositions while keeping per-game footprint under ~400MB.
CACHE_SIZE_PER_GAME = 50_000

# AlphaZero-style Q → centipawn conversion (Lc0 calibration constant).
Q_TO_CP_SCALE = 290.6806
CPL_CAP = 500.0
WINNING_STATE_THRESHOLD = 300  # cp; above this, scale CPL down

# Per-thread storage so each worker thread initializes its own model + MCTS once
# and reuses across games it processes.
_thread_local = threading.local()


def get_thread_mcts(device: torch.device) -> ParallelMCTS:
    """Lazy-initialize one model + ParallelMCTS per worker thread."""
    mcts = getattr(_thread_local, "mcts", None)
    if mcts is not None:
        return mcts

    model = load_model(MODEL_PATH, device)
    mcts = ParallelMCTS(model, device, num_workers=MCTS_WORKERS_PER_GAME,
                        cache_size=CACHE_SIZE_PER_GAME)

    # Override batch-collection thresholds. The default `min_batch_size = max(4,
    # num_workers)` plus a 100ms timeout was tuned for single-game search and
    # stalls multi-game throughput. We want each evaluator to batch within its
    # own 8 workers (amortizing kernel launch overhead) but bail quickly if
    # workers are mid-Python-work — hence min_batch=4 with a 2ms cap. Batches
    # of 4 typically form in <1ms with 8 active workers per game.
    mcts.evaluator.min_batch_size = 4
    mcts.evaluator.batch_timeout = 0.002

    _thread_local.mcts = mcts
    _thread_local.model = model
    return mcts


def q_to_cp(q: float) -> int:
    """Map Q ∈ [-1, 1] to centipawns via atanh. Clamps to avoid infinity at ±1."""
    q_clamped = max(-0.999, min(0.999, q))
    return int(Q_TO_CP_SCALE * math.atanh(q_clamped))


def compute_move_cpl(cp_before: int, cp_after: int) -> float:
    """Per-move CPL with winning-state scaling and a hard cap.

    cp_before / cp_after are both from the *target player's* perspective. A move
    that makes their position worse (lower cp) produces positive CPL.
    """
    cpl = max(0, cp_before - cp_after)

    # In already-won/lost positions, the engine readily trades centipawns that
    # don't change the outcome. Scale those down so they don't dominate ACPL.
    if abs(cp_before) > WINNING_STATE_THRESHOLD:
        cpl *= WINNING_STATE_THRESHOLD / abs(cp_before)

    return min(float(cpl), CPL_CAP)


def evaluate_game(game: chess.pgn.Game, target_color: chess.Color,
                  device: torch.device) -> dict:
    """Walk the mainline, evaluate target player's moves, return per-game stats.

    Tree-reuse contract: we call mcts.apply_move() after every push (both target
    and opponent moves) so the persistent tree tracks the actual game line. The
    pre-move search re-uses the subtree rooted at the post-opponent-move
    position; the post-move search re-uses the subtree we just descended into.
    """
    mcts = get_thread_mcts(device)
    mcts.reset()  # Fresh tree per game — prior game's tree is irrelevant
    mcts.cache.clear()  # Also clear NN cache to free VRAM tensors

    board = game.board()
    total_cpl = 0.0
    moves_evaluated = 0

    for move in game.mainline_moves():
        if board.turn != target_color:
            # Opponent's move: advance tree only, no search.
            board.push(move)
            mcts.apply_move(move)
            continue

        # Pre-move search: evaluate the position the target faced.
        # last_root_q is from the side-to-move-at-root perspective — i.e. the
        # target player's perspective, which is what we want.
        mcts.search(board, num_simulations=NUM_SIMULATIONS)
        q_before = mcts.last_root_q
        cp_before = q_to_cp(q_before)

        # Apply the move the target actually played.
        board.push(move)
        mcts.apply_move(move)

        # Post-move search: evaluate the position handed to the opponent.
        # last_root_q is now from the opponent's perspective, so negate to get
        # the target's perspective on the resulting position.
        mcts.search(board, num_simulations=NUM_SIMULATIONS)
        q_after = -mcts.last_root_q
        cp_after = q_to_cp(q_after)

        total_cpl += compute_move_cpl(cp_before, cp_after)
        moves_evaluated += 1

    acpl = total_cpl / moves_evaluated if moves_evaluated > 0 else 0.0
    estimated_elo = max(100, int(3200 - acpl * 15))

    return {
        "white": game.headers.get("White", "?"),
        "black": game.headers.get("Black", "?"),
        "result": game.headers.get("Result", "*"),
        "target_color": "White" if target_color == chess.WHITE else "Black",
        "moves_evaluated": moves_evaluated,
        "acpl": acpl,
        "estimated_elo": estimated_elo,
    }


def evaluate_game_task(idx: int, game: chess.pgn.Game, target_color: chess.Color,
                        device: torch.device) -> tuple[int, dict]:
    """ThreadPool wrapper that catches errors so one bad game doesn't kill the pool."""
    t0 = time.time()
    try:
        stats = evaluate_game(game, target_color, device)
        stats["elapsed"] = time.time() - t0
        return idx, stats
    except Exception as e:
        return idx, {"error": f"{type(e).__name__}: {e}", "elapsed": time.time() - t0}


def determine_target_color(game: chess.pgn.Game, target_name: str) -> chess.Color | None:
    """Match the target username against White/Black headers (case-insensitive)."""
    white = game.headers.get("White", "").lower()
    black = game.headers.get("Black", "").lower()
    t = target_name.lower()
    if t in white:
        return chess.WHITE
    if t in black:
        return chess.BLACK
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", type=Path, default=Path("chess_com_games_2026-05-19.pgn"),
                        help="Path to PGN file to evaluate")
    parser.add_argument("--target", type=str, default="POTUS_Official",
                        help="Player name to evaluate (substring match, case-insensitive). "
                             "If omitted, evaluates White in every game.")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Cap on games to evaluate")
    args = parser.parse_args()

    if not args.pgn.exists():
        print(f"Error: PGN file not found: {args.pgn}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("Error: this script requires CUDA (6 concurrent model instances).",
              file=sys.stderr)
        return 1
    device = torch.device("cuda")

    # Parse all games up front; PGN parsing is fast and lets the executor schedule freely.
    games: list[tuple[chess.pgn.Game, chess.Color]] = []
    with open(args.pgn, encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if args.target is not None:
                color = determine_target_color(game, args.target)
                if color is None:
                    continue
            else:
                color = chess.WHITE
            games.append((game, color))
            if args.max_games is not None and len(games) >= args.max_games:
                break

    if not games:
        print(f"No games found in {args.pgn}" +
              (f" matching target '{args.target}'" if args.target else ""))
        return 0

    print(f"Evaluating {len(games)} games with {MAX_CONCURRENT_GAMES} concurrent "
          f"MCTS instances ({NUM_SIMULATIONS} sims/move)...\n")

    all_results: list[dict] = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_GAMES) as pool:
        futures = [
            pool.submit(evaluate_game_task, i, game, color, device)
            for i, (game, color) in enumerate(games)
        ]

        for future in as_completed(futures):
            idx, stats = future.result()
            if "error" in stats:
                print(f"[Game {idx + 1}] ERROR ({stats['elapsed']:.1f}s): {stats['error']}")
                continue

            print(
                f"[Game {idx + 1}/{len(games)}] "
                f"{stats['white']} vs {stats['black']} ({stats['result']}) "
                f"-- eval as {stats['target_color']} | "
                f"moves={stats['moves_evaluated']} "
                f"ACPL={stats['acpl']:.1f} "
                f"Est.ELO={stats['estimated_elo']} "
                f"({stats['elapsed']:.1f}s)"
            )
            all_results.append(stats)

    elapsed = time.time() - start_time

    if all_results:
        overall_acpl = sum(r["acpl"] for r in all_results) / len(all_results)
        overall_elo = max(100, int(3200 - overall_acpl * 15))
        print("\n" + "=" * 70)
        print(f"Aggregate over {len(all_results)} games "
              f"(wall time {elapsed:.1f}s, {elapsed / len(all_results):.1f}s/game avg)")
        print(f"  Mean ACPL: {overall_acpl:.1f}")
        print(f"  Estimated ELO: {overall_elo}")
        print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
