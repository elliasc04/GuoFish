"""Measure how many previously drawn-while-winning games are now decisive.

Pairs each original winning-repetition draw against its replay in the retest
PGN and reports whether GuoFish now converts it.

The retest book replays every flagged opening with GuoFish on BOTH colours, so
only the replay where GuoFish is on the SAME side as the original draw is a like
-for-like re-test -- the opposite-colour game is a different scenario (GuoFish
plays the side that originally drew *against* it). We therefore match on
(opening, GuoFish side), not opening alone.

Original winning-draws are recovered with drawn_when_winning.analyze_draws (needs
Stockfish); use the same --threshold/--depth/--window that built the retest book
so the set matches. The retest PGN may be partial (an overnight run in progress);
unfinished/again-not-played pairs are reported as pending.

Usage:
    python benchmarking/engine/util/redraw_conversion.py \
        --original benchmarking/engine/games/v4/guofishv4_2_2.5_80_2800sf.pgn \
        --retest   benchmarking/engine/games/v4/guofishv4_redrawn_games_2_2.5_80_2800sf.pgn \
        --depth 16
"""

import argparse
import os
import sys

import chess
import chess.pgn

from drawn_when_winning import (
    analyze_draws, guofish_color, opening_key, DEFAULT_STOCKFISH,
    DEFAULT_OPENING_PLIES, MATE_SCORE_CP,
)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_ORIGINAL = os.path.join(
    REPO_ROOT, "benchmarking", "engine", "games", "v4",
    "guofishv4_2_2.5_80_2800sf.pgn")
DEFAULT_RETEST = os.path.join(
    REPO_ROOT, "benchmarking", "engine", "games", "v4",
    "guofishv4_reps15k_2800sf.pgn")

FINISHED = {"1-0", "0-1", "1/2-1/2"}


def gf_outcome(result: str, gf_color: chess.Color) -> str | None:
    """GuoFish-perspective outcome ('win'/'loss'/'draw') or None if unfinished."""
    if result == "1/2-1/2":
        return "draw"
    if result == "1-0":
        return "win" if gf_color == chess.WHITE else "loss"
    if result == "0-1":
        return "win" if gf_color == chess.BLACK else "loss"
    return None


def index_retest(path: str, plies: int) -> dict[tuple, dict]:
    """Map (opening_key, gf_side) -> {round, outcome} for finished retest games."""
    out: dict[tuple, dict] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            result = game.headers.get("Result", "*")
            if result not in FINISHED:
                continue  # unfinished tail of an in-progress run
            gf_color = guofish_color(game)
            key = opening_key(game, plies)
            if gf_color is None or key is None:
                continue
            side = "White" if gf_color == chess.WHITE else "Black"
            out[(key, side)] = {
                "round": game.headers.get("Round", "?"),
                "outcome": gf_outcome(result, gf_color),
            }
    return out


def fmt_eval(cp: int | None) -> str:
    if cp is None:
        return "n/a"
    if abs(cp) >= MATE_SCORE_CP - 1000:
        return "#" if cp > 0 else "-#"
    return f"{cp/100:+.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", default=DEFAULT_ORIGINAL,
                    help=f"PGN of the original run (default: {DEFAULT_ORIGINAL})")
    ap.add_argument("--retest", default=DEFAULT_RETEST,
                    help=f"PGN of the retest run, may be partial (default: {DEFAULT_RETEST})")
    ap.add_argument("--stockfish", default=DEFAULT_STOCKFISH, help="Stockfish executable.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Winning eval threshold in pawns (default: 0.5).")
    ap.add_argument("--depth", type=int, default=18, help="Stockfish depth (default: 18).")
    ap.add_argument("--window", type=int, default=20,
                    help="Plies from game end to scan for peak eval (default: 20).")
    ap.add_argument("--threads", type=int, default=1, help="Stockfish Threads.")
    ap.add_argument("--hash", type=int, default=256, help="Stockfish Hash MB.")
    ap.add_argument("--plies", type=int, default=DEFAULT_OPENING_PLIES,
                    help=f"Opening length to match on (default: {DEFAULT_OPENING_PLIES}).")
    args = ap.parse_args()

    for path in (args.original, args.retest):
        if not os.path.exists(path):
            sys.exit(f"ERROR: file not found: {path}")
    if not os.path.exists(args.stockfish):
        sys.exit(f"ERROR: Stockfish not found at {args.stockfish}")

    # Original winning-repetition draws: the gold set, keyed by (opening, side).
    analysis = analyze_draws(
        args.original, args.stockfish, threshold_cp=round(args.threshold * 100),
        depth=args.depth, window=args.window, threads=args.threads,
        hash_mb=args.hash, opening_plies=args.plies)
    winning = [r for r in analysis.winning if r.opening is not None]

    retest = index_retest(args.retest, args.plies)

    # Match each original winning-draw to its same-side replay.
    rows = []
    tally = {"win": 0, "draw": 0, "loss": 0, "pending": 0}
    for r in sorted(winning, key=lambda x: -(x.peak_cp or 0)):
        hit = retest.get((r.opening, r.side))
        outcome = hit["outcome"] if hit else None
        tally["pending" if outcome is None else outcome] += 1
        rows.append((r, hit, outcome))

    finished = tally["win"] + tally["draw"] + tally["loss"]

    print(f"Original winning-repetition draws: {len(winning)} "
          f"(threshold +{args.threshold:.2f}, depth {args.depth})")
    print(f"Retest finished games indexed:     {len(retest)}")
    print(f"Same-side replays available:       {finished} of {len(winning)} "
          f"({tally['pending']} still pending)\n")

    print("| Orig round | GoFish side | Orig peak eval | Retest round | Now |")
    print("|------------|-------------|----------------|--------------|-----|")
    label = {"win": "WON", "draw": "drew", "loss": "LOST", None: "pending"}
    for r, hit, outcome in rows:
        rr = hit["round"] if hit else "-"
        print(f"| {r.round} | {r.side} | {fmt_eval(r.peak_cp)} | {rr} | {label[outcome]} |")

    print()
    if finished:
        print(f"Of {finished} replayed winning-draws so far: "
              f"**{tally['win']} now WON**, {tally['draw']} still drawn, "
              f"{tally['loss']} now lost "
              f"({100*tally['win']/finished:.0f}% conversion).")
    else:
        print("No same-side replays have finished yet.")
    if tally["pending"]:
        print(f"{tally['pending']} winning-draw(s) have no finished same-side replay yet "
              "(partial run -- re-run when the overnight job completes).")


if __name__ == "__main__":
    main()
