"""Estimate player ELO from a PGN file via ACPL, using Stockfish as the evaluator.

Mirrors acpl_elo_estimator.py's methodology — pre/post move evals on the target
player's moves, winning-state scaling, capped CPL, same ELO mapping — but
replaces the Guofish2/MCTS evaluator with Stockfish. Useful as a ground-truth
comparison against the Guofish-based estimate.

Usage:
    python acpl_elo_estimator_sf.py [--pgn ...] [--target USERNAME] [--depth 18]
"""

import argparse
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.engine
import chess.pgn


STOCKFISH_PATH = Path("stockfish-windows-x86-64-avx2.exe").resolve()
DEFAULT_PGN = Path("chess_com_games_2026-05-19.pgn")
DEFAULT_PLAYER = "POTUS_Official"
DEFAULT_DEPTH = 18  # Lichess cloud analysis uses depth 18; reasonable strength/speed balance
DEFAULT_CONCURRENCY = max(2, (os.cpu_count() or 8) - 2)  # Leave 2 cores for OS / main thread
SF_THREADS_PER_INSTANCE = 1  # 1 thread × N instances scales better than N threads × 1 instance
SF_HASH_MB = 64  # Per-instance; N instances * 64MB stays well under typical RAM

CPL_CAP = 500.0
WINNING_STATE_THRESHOLD = 300  # cp; above this, scale CPL down
MATE_SCORE_CP = 10_000  # Treat mate as a large cp (cap will clamp the CPL anyway)

# Per-thread Stockfish handle so each worker thread keeps a long-lived engine
# process and avoids respawning per game.
_thread_local = threading.local()


def get_thread_engine() -> chess.engine.SimpleEngine:
    """Lazy-spawn one Stockfish process per worker thread."""
    engine = getattr(_thread_local, "engine", None)
    if engine is not None:
        return engine

    engine = chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH))
    engine.configure({
        "Threads": SF_THREADS_PER_INSTANCE,
        "Hash": SF_HASH_MB,
    })
    _thread_local.engine = engine
    return engine


def score_to_cp(score: chess.engine.PovScore, perspective: chess.Color) -> int:
    """Convert a Stockfish PovScore to an integer centipawn value from `perspective`'s POV.

    Mates collapse to ±MATE_SCORE_CP. We don't scale by mate distance because the
    CPL cap and winning-state scaling already swallow extreme magnitudes.
    """
    pov = score.pov(perspective)
    if pov.is_mate():
        mate_in = pov.mate()
        return MATE_SCORE_CP if mate_in > 0 else -MATE_SCORE_CP
    cp = pov.score()
    return cp if cp is not None else 0


def compute_move_cpl(cp_before: int, cp_after: int) -> float:
    """Per-move CPL. Same formula as the Guofish script for apples-to-apples comparison."""
    cpl = max(0, cp_before - cp_after)
    if abs(cp_before) > WINNING_STATE_THRESHOLD:
        cpl *= WINNING_STATE_THRESHOLD / abs(cp_before)
    return min(float(cpl), CPL_CAP)


def evaluate_game(game: chess.pgn.Game, target_color: chess.Color, depth: int) -> dict:
    """Walk the mainline, evaluate target's moves via Stockfish, return per-game stats."""
    engine = get_thread_engine()
    limit = chess.engine.Limit(depth=depth)

    board = game.board()
    total_cpl = 0.0
    moves_evaluated = 0

    for move in game.mainline_moves():
        if board.turn != target_color:
            board.push(move)
            continue

        # Pre-move eval from the target player's perspective.
        info_before = engine.analyse(board, limit)
        cp_before = score_to_cp(info_before["score"], target_color)

        board.push(move)

        # Post-move eval — turn has flipped, but we still want target's perspective.
        info_after = engine.analyse(board, limit)
        cp_after = score_to_cp(info_after["score"], target_color)

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
                       depth: int) -> tuple[int, dict]:
    t0 = time.time()
    try:
        stats = evaluate_game(game, target_color, depth)
        stats["elapsed"] = time.time() - t0
        return idx, stats
    except Exception as e:
        return idx, {"error": f"{type(e).__name__}: {e}", "elapsed": time.time() - t0}


def determine_target_color(game: chess.pgn.Game, target_name: str) -> chess.Color | None:
    white = game.headers.get("White", "").lower()
    black = game.headers.get("Black", "").lower()
    t = target_name.lower()
    if t in white:
        return chess.WHITE
    if t in black:
        return chess.BLACK
    return None


def shutdown_thread_engines(pool: ThreadPoolExecutor):
    """Quit each worker thread's Stockfish process so we don't leak subprocesses.

    submit() to every worker so the call hits each thread's local engine. We
    need pool.submit because thread-locals live on the worker, not the main thread.
    """
    def _quit():
        engine = getattr(_thread_local, "engine", None)
        if engine is not None:
            engine.quit()
            _thread_local.engine = None
    # Submit max_workers tasks; barrier ensures each worker picks one up.
    barrier = threading.Barrier(pool._max_workers)
    def _wait_then_quit():
        try:
            barrier.wait(timeout=5.0)
        except threading.BrokenBarrierError:
            pass
        _quit()
    futures = [pool.submit(_wait_then_quit) for _ in range(pool._max_workers)]
    for f in futures:
        try:
            f.result(timeout=10.0)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", type=Path, default=DEFAULT_PGN,
                        help="Path to PGN file to evaluate")
    parser.add_argument("--target", type=str, default=DEFAULT_PLAYER,
                        help="Player name to evaluate (substring match, case-insensitive). "
                             "If omitted, evaluates White in every game.")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help=f"Stockfish search depth per move (default: {DEFAULT_DEPTH})")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Concurrent Stockfish instances (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--cpu-cap", type=int, default=None,
                        help="Cap total CPU usage to roughly this percent by auto-sizing "
                             "concurrency. E.g. --cpu-cap 50 on a 16-core box runs ~8 "
                             "instances. Overrides --concurrency when set. Each instance is "
                             "pinned to 1 Stockfish thread, so cores-used scales linearly.")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Cap on games to evaluate")
    args = parser.parse_args()

    # CPU-cap takes precedence over explicit --concurrency. Mapped to integer
    # instance count via logical-core count, matching Task Manager's reporting.
    if args.cpu_cap is not None:
        if not (1 <= args.cpu_cap <= 100):
            print(f"Error: --cpu-cap must be in 1..100, got {args.cpu_cap}", file=sys.stderr)
            return 1
        logical_cores = os.cpu_count() or 8
        effective_concurrency = max(1, logical_cores * args.cpu_cap // 100)
        if effective_concurrency != args.concurrency:
            print(f"--cpu-cap {args.cpu_cap}%: using {effective_concurrency} concurrent "
                  f"instances (of {logical_cores} logical cores)")
        args.concurrency = effective_concurrency

    if not args.pgn.exists():
        print(f"Error: PGN file not found: {args.pgn}", file=sys.stderr)
        return 1

    if not STOCKFISH_PATH.exists():
        print(f"Error: Stockfish executable not found at {STOCKFISH_PATH}", file=sys.stderr)
        return 1

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

    print(f"Evaluating {len(games)} games with {args.concurrency} concurrent "
          f"Stockfish instances (depth={args.depth})...\n")

    all_results: list[dict] = []
    start_time = time.time()
    pool = ThreadPoolExecutor(max_workers=args.concurrency)

    try:
        futures = [
            pool.submit(evaluate_game_task, i, game, color, args.depth)
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
    finally:
        shutdown_thread_engines(pool)
        pool.shutdown(wait=True)

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
