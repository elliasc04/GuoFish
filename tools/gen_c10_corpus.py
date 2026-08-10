#!/usr/bin/env python
"""C10 — the shared position corpus for Gate 2 and Gate 2b.

WHY ONE CORPUS AND NOT TWO
==========================
Gate 2 asks whether the C++ gather+softmax reproduces ATen's over recorded
logits. Gate 2b asks whether an engine built on that gather picks the same move
as the reference. They are the same claim at two magnifications, and running them
over different position sets would make the second unable to explain the first:
a Gate 2b disagreement whose position Gate 2 never measured is a disagreement
with nowhere to look. So both read this file, both record its SHA-256, and the
consuming tests refuse a mismatch.

WHY THE POSITIONS COME FROM GAMES AND NOT FROM A GENERATOR
==========================================================
Scope's data table (PART 2 of the port chunks) puts C10 Gate 2b under
"game-realistic positions -- ~500 sampled from the benchmark PGNs", and the ~400
v5 benchmark games are already established as sufficient for that (400 games x
~80 plies is ~32k positions against a requirement of ~500). Constructed
positions would bias the corpus toward whatever the author thinks a midgame
looks like, and the specific thing under test -- whether two softmax reduction
orders ever disagree about which move is best -- is a property of the prior
DISTRIBUTION at real positions, which is exactly what a constructed set gets
wrong.

WHAT IS FILTERED OUT, AND WHAT DELIBERATELY IS NOT
==================================================
Removed: positions with no legal moves (nothing to choose), positions with one
legal move (nothing to disagree about), and positions already over. That is all.

NOT removed -- and these are the ones a tidier filter would have dropped:

  * positions in check. C5's sampler excluded them because a forced mate inside
    5,000 simulations wasted a whole reference run; here a run is cheap and a
    check is an ordinary position that the engine will meet constantly.
  * positions with a lopsided evaluation. A position where one move is obviously
    best is a position where the two engines SHOULD agree, and a corpus that
    kept only hard positions would be measuring something other than the >= 99%
    the acceptance criterion states.
  * endgames, openings, repetition-carrying positions. The corpus is sampled
    across the whole game so that the agreement rate is over the distribution
    the engine actually plays, not over a slice of it.

The sample is several plies per game rather than one, because 500 positions from
~400 games otherwise means taking nearly every game and the ply choice stops
being a choice. Plies are drawn without replacement per game from a seeded
Generator, so the corpus is reproducible from (seed, pgn list) alone.

Usage:
    python tools/gen_c10_corpus.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import chess
import chess.pgn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = REPO_ROOT / "golden" / "c10_corpus.json"

# The benchmark games, most-recent first. More than one file because a single
# 400-game PGN sampled at four plies per game would draw from ~400 distinct
# openings but only one opponent and one time control; these differ in both.
DEFAULT_PGNS = (
    REPO_ROOT / "benchmarking" / "engine" / "games" / "v5" /
    "guofishv5_10.9M_2ksims_2600sf_fixednodes.pgn",
    REPO_ROOT / "benchmarking" / "engine" / "games" / "v5" /
    "guofishv5_10.9M_2ksims_2600sf.pgn",
    REPO_ROOT / "benchmarking" / "engine" / "games" / "v5" /
    "guofishv5_10.9M_800sims_2600sf.pgn",
)


def fen_of(board: chess.Board) -> str:
    """Canonical FEN: ep square present after any double push, per the FEN spec.

    NOT `Board.fen()`. The default omits the ep square when no legal capture
    exists, which moves token 66 and therefore the nn_key on ~3% of positions --
    C2 measured 3,203 of 3,610 corpus positions carrying an ep square that
    `setFen` would have dropped. The C++ side takes the RAW ep field from the FEN
    text, so the text has to carry it. See DECISIONS.md, C2.
    """
    return board.fen(en_passant="fen")


def sample(pgn_paths, *, target: int, seed: int, per_game: int, min_ply: int):
    """Deterministic (fen, ply, game index, source) tuples, de-duplicated.

    Each PGN gets its own quota. Draining the first file before touching the
    second would defeat the reason there are three: the first alone holds enough
    games for 500 positions, and a corpus drawn entirely from one opponent and
    one time control is narrower than the one the docstring above claims.
    """
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    seen: set[str] = set()
    game_index = 0
    quota = -(-target // len(pgn_paths))  # ceiling division

    for file_number, pgn_path in enumerate(pgn_paths, start=1):
        # The last file also absorbs whatever the earlier ones came up short of,
        # so a thin PGN cannot silently shrink the corpus.
        stop_at = min(target, quota * file_number) if file_number < len(pgn_paths) else target
        if not pgn_path.exists():
            raise FileNotFoundError(f"benchmark PGN not found: {pgn_path}")
        with open(pgn_path, encoding="utf-8", errors="replace") as handle:
            while len(out) < stop_at:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                game_index += 1
                moves = list(game.mainline_moves())
                if len(moves) <= min_ply + 2:
                    continue

                # Without replacement, so one game cannot contribute the same
                # position twice through two draws of the same ply.
                choices = rng.choice(np.arange(min_ply, len(moves)),
                                     size=min(per_game, len(moves) - min_ply),
                                     replace=False)
                board = game.board()
                wanted = set(int(c) for c in choices)
                for i, move in enumerate(moves):
                    if i in wanted:
                        legal = board.legal_moves.count()
                        # One legal move is a position with nothing to disagree
                        # about; zero is a finished game. Everything else stays,
                        # checks included -- see the module docstring.
                        if legal >= 2 and not board.is_game_over():
                            fen = fen_of(board)
                            if fen not in seen:
                                seen.add(fen)
                                out.append({
                                    "fen": fen,
                                    "ply": i,
                                    "game": game_index,
                                    "source": pgn_path.name,
                                    "legal_moves": legal,
                                })
                    board.push(move)
                    if len(out) >= stop_at:
                        break

    return out[:target]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--positions", type=int, default=500,
                        help="the acceptance criteria say ~500")
    parser.add_argument("--per-game", type=int, default=4)
    parser.add_argument("--min-ply", type=int, default=8,
                        help="skip the book plies every game shares")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(f"refusing to overwrite {args.out} without --force", file=sys.stderr)
        return 2

    positions = sample(DEFAULT_PGNS, target=args.positions, seed=args.seed,
                       per_game=args.per_game, min_ply=args.min_ply)
    if len(positions) < args.positions:
        print(f"ERROR: sampled only {len(positions)} of {args.positions} positions",
              file=sys.stderr)
        return 1

    payload = {
        "provenance": {
            "generator": "tools/gen_c10_corpus.py",
            "python": platform.python_version(),
            "python_chess": chess.__version__,
            "platform": platform.platform(),
            "seed": args.seed,
            "per_game": args.per_game,
            "min_ply": args.min_ply,
            "pgns": [{"name": p.name, "sha256": sha256_of(p)} for p in DEFAULT_PGNS],
        },
        "positions": positions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8",
                        newline="\n")

    by_source: dict[str, int] = {}
    for p in positions:
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
    plies = np.array([p["ply"] for p in positions])
    legal = np.array([p["legal_moves"] for p in positions])
    print(f"wrote {args.out}  ({len(positions)} positions)")
    print(f"  sources     : {by_source}")
    print(f"  ply         : min {plies.min()} median {int(np.median(plies))} max {plies.max()}")
    print(f"  legal moves : min {legal.min()} mean {legal.mean():.1f} max {legal.max()}")
    print(f"  sha256      : {sha256_of(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
