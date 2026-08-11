#!/usr/bin/env python
"""C11b — did the 2687.7 anchor's games contain OPENING-BOOK moves?

    python tools/anchor_book_audit.py
    python tools/anchor_book_audit.py --pgn <other.pgn> --engine "Guofish5-10.9M"

WHY THIS EXISTS
===============
C11b turns the opening book on by default, and Gate 5 has to be held identical
to the 2687.7 anchor — whatever the anchor turns out to have run with. The v5
wrapper defaulted the book ON and the recorded anchor command passes no
disabling flag, so the anchor plausibly INCLUDES book moves. Nobody knows, and
"plausibly" is not a basis for a strength comparison.

The brief settles it empirically rather than by reconstructing intent: replay
the anchor PGNs and ask, for each of the engine's own moves past the Cutechess
opening cutoff, whether that move is in `assets/gm2001.bin` for the position it
was played in — and, more sharply, whether it is the entry the v5 wrapper's
selection rule would have returned.

WHAT A HIT DOES AND DOES NOT PROVE
==================================
A single move agreeing with the book proves nothing: an opening book contains
good moves and so does a 2687-rated engine, so agreement is the null hypothesis
in the first few plies of a mainline. What is diagnostic is the SHAPE:

  * a RUN of consecutive moves each in book, ending abruptly — that is a book
    playing until it runs out, which is what a book does;
  * agreement rates far above what the same engine achieves on non-book
    positions;
  * agreement holding on positions where the book's top entry is NOT the
    obvious move.

So this reports the run lengths and the per-ply agreement curve, not a single
percentage, and it draws no conclusion the numbers do not support. The verdict
line at the end says which of the two shapes the data has, or says it is
ambiguous.

CUTECHESS SUPPLIES THE FIRST `--plies` MOVES. The anchor command used
`-openings file="assets/8moves_v3.pgn" format=pgn order=sequential plies=16`,
so plies 1-16 came from the opening file and belong to NEITHER engine. They are
skipped by default; a book hit there says nothing about the engine at all.

Reads PGNs and the book. Writes nothing but its report (Global Rule 2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chess  # noqa: E402
import chess.pgn  # noqa: E402
import chess.polyglot  # noqa: E402

DEFAULT_PGN = (REPO_ROOT / "benchmarking" / "engine" / "games" / "v5"
               / "guofishv5_10.9M_2ksims_2600sf_fixednodes.pgn")
DEFAULT_BOOK = REPO_ROOT / "assets" / "gm2001.bin"
# The anchor command: `-openings ... plies=16`.
DEFAULT_PLIES = 16


def audit(pgn_path: Path, book_path: Path, engine_name: str, plies: int) -> dict:
    reader = chess.polyglot.open_reader(str(book_path))
    try:
        games = 0
        # Per-ply-offset agreement, where offset 0 is the first move the engine
        # actually chose (ply `plies` + 1 of the game).
        by_offset: dict[int, list[int]] = {}
        runs: list[int] = []
        engine_moves = 0
        in_book = 0
        top_entry_matches = 0

        with pgn_path.open(encoding="utf-8", errors="replace") as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                white = game.headers.get("White", "")
                black = game.headers.get("Black", "")
                if engine_name not in (white, black):
                    continue
                games += 1
                engine_is_white = engine_name == white

                board = game.board()
                ply = 0
                run = 0
                # COUNTED IN THE ENGINE'S OWN MOVES, not in plies. Plies
                # alternate sides, so a ply-based offset gives the White games
                # the even numbers and the Black games the odd ones, halves
                # every sample count, and makes "the engine's first move past
                # the cutoff" mean two different things depending on colour.
                offset = 0
                for move in game.mainline_moves():
                    ply += 1
                    mover_is_white = board.turn == chess.WHITE
                    # Skip the Cutechess-supplied opening entirely: those moves
                    # were played by the opening FILE, not by either engine.
                    if ply <= plies or mover_is_white != engine_is_white:
                        board.push(move)
                        continue

                    entries = list(reader.find_all(board))
                    hit = any(e.move == move for e in entries)
                    engine_moves += 1
                    by_offset.setdefault(offset, []).append(1 if hit else 0)
                    if hit:
                        in_book += 1
                        run += 1
                        heaviest = max(entries, key=lambda e: e.weight)
                        if heaviest.move == move:
                            top_entry_matches += 1
                    else:
                        if run:
                            runs.append(run)
                        run = 0
                    offset += 1
                    board.push(move)
                if run:
                    runs.append(run)
    finally:
        reader.close()

    return {
        "pgn": str(pgn_path),
        "book": str(book_path),
        "engine": engine_name,
        "opening_plies_skipped": plies,
        "games": games,
        "engine_moves_examined": engine_moves,
        "moves_in_book": in_book,
        "moves_matching_the_top_weighted_entry": top_entry_matches,
        "agreement": (in_book / engine_moves) if engine_moves else 0.0,
        "top_entry_agreement": (top_entry_matches / engine_moves) if engine_moves else 0.0,
        "book_runs": sorted(runs, reverse=True)[:20],
        "longest_run": max(runs) if runs else 0,
        # The share of examined moves that sit inside a run of 3 or more. This
        # is the statistic that separates "a book was playing" from "a strong
        # engine and a book agree about good moves": coincidental agreement is
        # scattered, a book's is contiguous.
        "share_in_runs_of_3_or_more": (
            sum(r for r in runs if r >= 3) / engine_moves) if engine_moves else 0.0,
        "by_offset": {k: {"n": len(v), "hits": sum(v), "rate": sum(v) / len(v)}
                      for k, v in sorted(by_offset.items())[:12]},
    }


def verdict(result: dict) -> str:
    """The shape the numbers have, stated without overreach.

    A book that was ON leaves runs: consecutive in-book moves from the first
    engine ply until the line leaves the book. A book that was OFF leaves
    scattered agreement that decays with ply and never forms long runs, because
    a strong engine and a book agree on good moves in known positions and stop
    agreeing as the position leaves theory.
    """
    if not result["engine_moves_examined"]:
        return ("INCONCLUSIVE: no engine moves past the opening cutoff were found. "
                "Check --engine against the PGN's White/Black headers.")
    agreement = result["agreement"]
    first = result["by_offset"].get(0, {}).get("rate", 0.0)
    contiguous = result["share_in_runs_of_3_or_more"]

    # THE QUESTION IS NOT "WAS THE BOOK OPEN" — that is unknowable from a PGN —
    # BUT "DID THE BOOK MOVE THE ANCHOR", which is the only thing Gate 5 needs.
    # So the thresholds are about INFLUENCE, and a single long run among
    # thousands of moves is noise rather than evidence.
    if agreement >= 0.5 and first >= 0.8 and contiguous >= 0.2:
        return ("THE BOOK PLAYED A MATERIAL PART OF THESE GAMES: agreement is "
                "high, the engine's first move past the cutoff agrees nearly "
                "always, and the hits are contiguous — a book playing until it "
                "runs out. Gate 5 must be run with the book ON to be comparable "
                "with this anchor.")
    if agreement <= 0.02 and contiguous <= 0.01:
        return (f"THE BOOK DID NOT MATERIALLY AFFECT THIS ANCHOR: {agreement:.2%} of "
                f"the engine's moves past the opening cutoff appear in the book at "
                f"all, and {contiguous:.2%} sit in a run of three or more. At this "
                f"level the agreement is what two strong move-choosers produce by "
                f"coincidence, and gm2001.bin's coverage has evidently run out "
                f"inside the Cutechess-supplied opening. Gate 5 is therefore "
                f"comparable to this anchor whether the book is on or off — but "
                f"run it ON with BookSeed=0, which is both the v5 default and the "
                f"reproducible setting.")
    return ("AMBIGUOUS: the agreement is neither coincidental nor clearly "
            "run-shaped. Read `by_offset` and `book_runs` before deciding; a book "
            "that ran out partway past the cutoff would look like this.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pgn", type=Path, default=DEFAULT_PGN)
    parser.add_argument("--book", type=Path, default=DEFAULT_BOOK)
    parser.add_argument("--engine", type=str, default="Guofish5-10.9M",
                        help="the PGN White/Black header naming the engine under audit")
    parser.add_argument("--plies", type=int, default=DEFAULT_PLIES,
                        help="Cutechess-supplied opening plies to skip "
                             "(the anchor command used plies=16)")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    for path in (args.pgn, args.book):
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 2

    result = audit(args.pgn, args.book, args.engine, args.plies)
    result["verdict"] = verdict(result)

    print(f"pgn    : {result['pgn']}")
    print(f"book   : {result['book']}")
    print(f"engine : {result['engine']}  (skipping the first {args.plies} plies, "
          f"which Cutechess supplied)")
    print(f"games  : {result['games']}")
    print(f"engine moves examined      : {result['engine_moves_examined']}")
    print(f"  in the book              : {result['moves_in_book']} "
          f"({result['agreement']:.2%})")
    print(f"  == the top-weight entry  : {result['moves_matching_the_top_weighted_entry']} "
          f"({result['top_entry_agreement']:.2%})")
    print(f"longest consecutive in-book run: {result['longest_run']}")
    print(f"longest runs: {result['book_runs']}")
    print(f"share of moves inside a run of 3 or more: "
          f"{result['share_in_runs_of_3_or_more']:.3%}")
    print("agreement by ENGINE-MOVE offset past the cutoff:")
    for offset, row in result["by_offset"].items():
        print(f"  +{offset:<3d} {row['hits']:4d}/{row['n']:<5d} {row['rate']:.2%}")
    print()
    print(result["verdict"])

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=1), encoding="utf-8",
                                 newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
