"""Extract the opening-book lines that produced drawn games, into a new PGN.

Workflow: the benchmark (cutechess) opened every game from assets/8moves_v3.pgn
with plies=16 (8 full moves). To retest only the games that drew, we take each
drawn game's first 16 plies, match them back to the exact book line, and write
the unique matches out as a fresh opening book. Feed that to a new cutechess run
to replay just those openings (both colours, via -repeat) and see how many of
the previously-drawn games now convert.

Matching is by the UCI move sequence of the first N plies, so move-order/notation
differences don't matter. Output entries are written verbatim from the book (same
headers/ECO), so the result is a drop-in opening book.

By default it selects every drawn game. With --winning-only it instead selects
just the repetition/50-move draws where GuoFish was actually winning (peak eval
>= --threshold), reusing drawn_when_winning.analyze_draws (needs Stockfish) --
i.e. exactly the games the repetition fix is meant to rescue.

Usage:
    # all draws
    python benchmarking/engine/util/extract_draw_openings.py \
        --games benchmarking/engine/games/v4/guofishv4_2_2.5_80_2800sf.pgn \
        --out benchmarking/engine/games/v4/draw_openings_2800sf.pgn

    # only the games drawn from a winning position
    python benchmarking/engine/util/extract_draw_openings.py --winning-only \
        --out benchmarking/engine/games/v4/winning_draw_openings_2800sf.pgn
"""

import argparse
import os
import sys

import chess
import chess.pgn

# Sibling util module; the script's own directory is on sys.path when run.
from drawn_when_winning import analyze_draws, opening_key, DEFAULT_STOCKFISH


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_GAMES = os.path.join(
    REPO_ROOT, "benchmarking", "engine", "games", "v4",
    "guofishv4_2_2.5_80_2800sf.pgn",
)
DEFAULT_BOOK = os.path.join(REPO_ROOT, "assets", "8moves_v3.pgn")
DEFAULT_OUT = os.path.join(
    REPO_ROOT, "benchmarking", "engine", "games", "v4", "draw_openings_2800sf.pgn"
)
OPENING_PLIES = 16  # cutechess used plies=16 (8 full moves)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", default=DEFAULT_GAMES,
                    help=f"Played-games PGN to scan (default: {DEFAULT_GAMES})")
    ap.add_argument("--book", default=DEFAULT_BOOK,
                    help=f"Opening book PGN to pull lines from (default: {DEFAULT_BOOK})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Output opening-book PGN (default: {DEFAULT_OUT})")
    ap.add_argument("--result", default="1/2-1/2",
                    help="Game result to select openings for (default: 1/2-1/2 = draws). "
                         "Ignored when --winning-only is set.")
    ap.add_argument("--plies", type=int, default=OPENING_PLIES,
                    help=f"Opening length in half-moves to match on (default: {OPENING_PLIES})")
    # --- Winning-only mode (uses Stockfish via drawn_when_winning.analyze_draws) ---
    ap.add_argument("--winning-only", action="store_true",
                    help="Select only repetition/50-move draws where GuoFish was winning "
                         "(peak eval >= --threshold). Requires Stockfish.")
    ap.add_argument("--stockfish", default=DEFAULT_STOCKFISH,
                    help="Stockfish executable (winning-only mode).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Winning eval threshold in pawns, GuoFish POV (default: 0.5).")
    ap.add_argument("--depth", type=int, default=18,
                    help="Stockfish search depth per position (winning-only mode, default: 18).")
    ap.add_argument("--window", type=int, default=20,
                    help="Plies from game end to scan for peak eval (winning-only mode, default: 20).")
    ap.add_argument("--threads", type=int, default=1, help="Stockfish Threads (winning-only mode).")
    ap.add_argument("--hash", type=int, default=256, help="Stockfish Hash MB (winning-only mode).")
    args = ap.parse_args()

    for path in (args.games, args.book):
        if not os.path.exists(path):
            sys.exit(f"ERROR: file not found: {path}")

    # === 1) Collect the opening keys of the selected games ===
    # Preserve first-seen order so the output is stable. One opening may back two
    # games (both colours) -- we dedupe to unique openings.
    needed: dict[tuple[str, ...], None] = {}

    if args.winning_only:
        if not os.path.exists(args.stockfish):
            sys.exit(f"ERROR: Stockfish not found at {args.stockfish}")
        analysis = analyze_draws(
            args.games, args.stockfish, threshold_cp=round(args.threshold * 100),
            depth=args.depth, window=args.window, threads=args.threads,
            hash_mb=args.hash, opening_plies=args.plies)
        no_key = 0
        for rec in analysis.winning:
            if rec.opening is None:
                no_key += 1
                continue
            needed.setdefault(rec.opening, None)
        print(f"Analyzed {analysis.total_games} games; {len(analysis.records)} "
              f"repetition/50-move draws; {len(analysis.winning)} winning "
              f"(peak >= +{args.threshold:.2f}); {len(needed)} unique openings"
              + (f" ({no_key} too short to key)" if no_key else "") + ".")
    else:
        total = selected = no_key = 0
        with open(args.games, "r", encoding="utf-8", errors="replace") as fh:
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                total += 1
                if game.headers.get("Result", "") != args.result:
                    continue
                selected += 1
                key = opening_key(game, args.plies)
                if key is None:
                    no_key += 1
                    continue
                needed.setdefault(key, None)
        print(f"Scanned {total} games; {selected} with result {args.result!r}; "
              f"{len(needed)} unique openings"
              + (f" ({no_key} too short to key)" if no_key else "") + ".")

    if not needed:
        sys.exit("No openings to extract.")

    # === 2) Walk the book, grabbing the verbatim entry for each needed key ===
    matched: dict[tuple[str, ...], chess.pgn.Game] = {}
    book_scanned = 0
    with open(args.book, "r", encoding="utf-8", errors="replace") as fh:
        while len(matched) < len(needed):
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            book_scanned += 1
            key = opening_key(game, args.plies)
            if key in needed and key not in matched:
                matched[key] = game

    print(f"Matched {len(matched)}/{len(needed)} openings in the book "
          f"(scanned {book_scanned} book lines).")

    missing = [k for k in needed if k not in matched]
    if missing:
        print(f"WARNING: {len(missing)} opening(s) not found in the book "
              "(different book or --plies mismatch?):", file=sys.stderr)
        for k in missing[:5]:
            print("  " + " ".join(k), file=sys.stderr)

    # === 3) Write matches out as a new opening book, in book order ===
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    written = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        # Emit in first-seen (drawn-game) order for traceability.
        for key in needed:
            game = matched.get(key)
            if game is None:
                continue
            print(game, file=fh, end="\n\n")
            written += 1

    print(f"Wrote {written} openings to {args.out}")


if __name__ == "__main__":
    main()
