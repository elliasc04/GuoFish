"""Generate golden/movegen.jsonl — the reference legal-move data for C1 parity.

`python-chess` is the *reference implementation* here (Global Rule 2). The C++
movegen is judged against this file; this file is never regenerated to agree
with the C++ movegen. If the two disagree, the C++ side is wrong until proven
otherwise.

Usage:
    python tools/gen_movegen_golden.py                    # full run, ~100k FENs
    python tools/gen_movegen_golden.py --max-games 50     # smoke run
    python tools/gen_movegen_golden.py --target 250000 --seed 7

Positions come from the engine's own benchmark games under
`benchmarking/engine/games`, so the corpus is the distribution the port will
actually be asked about. Every position of every mainline is harvested, tagged,
and then *quota-sampled* so that the edge cases a movegen actually gets wrong
are over-represented relative to their natural frequency (see "Sampling").

Output is `golden/movegen.jsonl`, one object per line:

    {"fen": "<FEN>", "moves": ["<uci>", ...]}

sorted by FEN, with each move list sorted by UCI string. Both sorts are there so
that regenerating with the same seed produces a byte-identical file and a real
change shows up as a readable diff.

-------------------------------------------------------------------------------
Sampling
-------------------------------------------------------------------------------
Five buckets are tagged. Four are the categories the C1 brief names; the fifth
is the reason en passant is on that list at all:

  castle     a castling move is legal right now
  promo      a promotion is legal right now (all four pieces)
  ep         a legal en-passant capture is available
  check      the side to move is in check
  ep_pin     the FEN carries an en-passant target square, a pawn of the side to
             move attacks it, and capturing is nevertheless *illegal* — almost
             always the horizontal pin, where removing both pawns from the rank
             exposes the king. This is the classic movegen bug, and it is only
             a test if the FEN still carries the ep square (see below).

Buckets are filled rarest-first from a combined budget of `--edge-fraction` of
the dataset, with each bucket's unused share spilling forward to the next. In
practice the rare buckets are taken whole and the common ones absorb the rest;
the quota machinery only starts binding on a corpus much larger than this one.
The remainder of the target is filled uniformly at random from everything not
already chosen, so the file is not *only* edge cases — a movegen that is wrong
about quiet positions has to fail here too.

-------------------------------------------------------------------------------
Two choices worth knowing about
-------------------------------------------------------------------------------
**FENs are emitted with the en-passant square always set after a double pawn
push** (`en_passant="fen"`, the FEN spec's rule), not python-chess's default
`en_passant="legal"`, which omits the square unless a capture is actually
available. The default would silently erase every `ep_pin` position: the trap
FEN above would be written with `-` in the ep field and a buggy movegen would
have nothing to trip over. Emitting the square keeps the trap, and costs
nothing — python-chess reads a set-but-uncapturable ep square back correctly.

**Positions are deduplicated on the full emitted FEN**, halfmove and fullmove
counters included. Those counters do not affect legal move generation, so this
admits some pairs whose move lists are identical. It is the conservative
direction: dropping the counters would mean choosing one arbitrary clock value
for a position and never testing the parser on the others.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import chess
import chess.pgn

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PGN_DIR = REPO_ROOT / "benchmarking" / "engine" / "games"
DEFAULT_OUT = REPO_ROOT / "golden" / "movegen.jsonl"

DEFAULT_TARGET = 100_000
DEFAULT_SEED = 20260805
DEFAULT_EDGE_FRACTION = 0.60

# Bit per bucket, packed into one int so the position table stays small.
CASTLE, PROMO, EP, CHECK, EP_PIN = 1, 2, 4, 8, 16
BUCKETS = (("castle", CASTLE), ("promo", PROMO), ("ep", EP), ("check", CHECK), ("ep_pin", EP_PIN))

BACK_RANK_PAWNS = {chess.WHITE: chess.BB_RANK_7, chess.BLACK: chess.BB_RANK_2}


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------


def tag(board):
    """Bucket mask for `board`.

    Deliberately avoids generating the full legal move list: this runs on every
    position of every game, while moves are only generated for the ~100k that
    survive sampling. Each check below is either a bitboard test or a generator
    restricted to the pieces that could possibly produce the move in question.
    """
    mask = 0

    if board.is_check():
        mask |= CHECK

    if board.castling_rights and any(board.generate_castling_moves()):
        # generate_castling_moves already applies the full legality rules
        # (empty path, king not in/through/into check), so `any` is enough.
        mask |= CASTLE

    ready = board.pawns & board.occupied_co[board.turn] & BACK_RANK_PAWNS[board.turn]
    if ready and any(m.promotion for m in board.generate_legal_moves(from_mask=ready)):
        mask |= PROMO

    if board.ep_square is not None:
        if board.has_legal_en_passant():
            mask |= EP
        elif board.pawns & board.occupied_co[board.turn] & chess.BB_PAWN_ATTACKS[not board.turn][board.ep_square]:
            # A pawn attacks the ep square but may not take it. That is the
            # position a naive movegen gets wrong.
            mask |= EP_PIN

    return mask


# ---------------------------------------------------------------------------
# Harvest
# ---------------------------------------------------------------------------


def harvest(pgn_paths, max_games=None, log=sys.stderr):
    """Replay every mainline and return {fen: bucket mask} plus a stats Counter."""
    positions = {}
    stats = Counter()

    for path in pgn_paths:
        if max_games is not None and stats["games"] >= max_games:
            break
        with open(path, encoding="utf-8", errors="replace") as handle:
            while max_games is None or stats["games"] < max_games:
                try:
                    game = chess.pgn.read_game(handle)
                except Exception:  # a malformed game must not kill the run
                    stats["unreadable_games"] += 1
                    break
                if game is None:
                    break
                stats["games"] += 1

                board = game.board()
                record(board, positions, stats)
                for move in game.mainline_moves():
                    if not board.is_legal(move):
                        stats["illegal_moves_skipped"] += 1
                        break
                    board.push(move)
                    record(board, positions, stats)

        print(
            f"  {path.relative_to(DEFAULT_PGN_DIR.parent) if DEFAULT_PGN_DIR in path.parents else path.name}"
            f"  games={stats['games']} unique={len(positions)}",
            file=log,
        )

    return positions, stats


def record(board, positions, stats):
    stats["plies"] += 1
    fen = board.fen(en_passant="fen")
    if fen in positions:
        return
    positions[fen] = tag(board)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def select(positions, target, seed, edge_fraction):
    """Quota-sample `target` FENs, rarest bucket first, then fill at random."""
    rng = random.Random(seed)

    if len(positions) <= target:
        print(
            f"warning: corpus holds {len(positions)} unique positions, at or below the "
            f"target of {target}; taking all of them",
            file=sys.stderr,
        )
        return sorted(positions), {name: 0 for name, _ in BUCKETS}

    pools = {name: sorted(f for f, m in positions.items() if m & bit) for name, bit in BUCKETS}

    chosen = set()
    taken = {}
    budget = int(target * edge_fraction)
    order = sorted(pools, key=lambda name: len(pools[name]))

    for i, name in enumerate(order):
        share = budget // (len(order) - i)
        candidates = [f for f in pools[name] if f not in chosen]
        n = min(len(candidates), share)
        # rng.sample on a sorted list keeps the draw reproducible across runs.
        chosen.update(candidates if n == len(candidates) else rng.sample(candidates, n))
        taken[name] = n
        budget -= n  # whatever a rare bucket leaves unspent flows to the next

    fill = sorted(set(positions) - chosen)
    chosen.update(rng.sample(fill, min(target - len(chosen), len(fill))))

    return sorted(chosen), taken


# ---------------------------------------------------------------------------
# Move generation and output
# ---------------------------------------------------------------------------


def legal_uci(board):
    """All legal moves as sorted UCI strings.

    `board.uci()` rather than `move.uci()`: on a standard board python-chess
    already normalises castling to e1g1/e1c1, but the raw `Move` is the king-
    takes-rook encoding whenever `board.chess960` is set, and this is the call
    that is correct either way. Promotions need no special handling — the
    generator yields one move per piece, so all four reach the output.
    """
    return sorted(board.uci(m) for m in board.legal_moves)


def write(fens, out_path, log=sys.stderr):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()

    # newline="" keeps Windows from writing \r\n into a file that is compared
    # byte-for-byte across platforms.
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        for i, fen in enumerate(fens):
            moves = legal_uci(chess.Board(fen))
            stats["moves"] += len(moves)
            if not moves:
                stats["terminal"] += 1
            handle.write(json.dumps({"fen": fen, "moves": moves}) + "\n")
            if (i + 1) % 25_000 == 0:
                print(f"  wrote {i + 1}/{len(fens)}", file=log)

    return stats


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pgn-dir", type=Path, default=DEFAULT_PGN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--edge-fraction",
        type=float,
        default=DEFAULT_EDGE_FRACTION,
        help="upper bound on the share of the dataset given to the tagged buckets",
    )
    parser.add_argument("--max-games", type=int, default=None, help="stop early; for smoke runs")
    args = parser.parse_args()

    if not 0.0 <= args.edge_fraction <= 1.0:
        parser.error("--edge-fraction must be in [0, 1]")

    paths = sorted(args.pgn_dir.rglob("*.pgn"))
    if not paths:
        parser.error(f"no .pgn files under {args.pgn_dir}")

    print(f"reading {len(paths)} pgn files from {args.pgn_dir}", file=sys.stderr)
    positions, stats = harvest(paths, args.max_games)
    if not positions:
        parser.error("no positions harvested")

    fens, taken = select(positions, args.target, args.seed, args.edge_fraction)
    print(f"\ngenerating moves for {len(fens)} positions", file=sys.stderr)
    out = write(fens, args.out)

    # Bucket density in the corpus vs. in the sample: the point of the exercise.
    selected = set(fens)
    print()
    print(f"source     : {args.pgn_dir}  ({len(paths)} files, {stats['games']} games)")
    print(f"positions  : {stats['plies']} plies -> {len(positions)} unique -> {len(fens)} sampled")
    print(f"seed       : {args.seed}   edge budget: {args.edge_fraction:.0%}")
    print(f"output     : {args.out}")
    print()
    print("| bucket | in corpus | corpus % | in sample | sample % | enrichment |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, bit in BUCKETS:
        pool = {f for f, m in positions.items() if m & bit}
        got = len(pool & selected)
        c_pct = 100.0 * len(pool) / len(positions)
        s_pct = 100.0 * got / len(fens)
        gain = f"{s_pct / c_pct:.1f}x" if c_pct else "-"
        print(f"| {name} | {len(pool)} | {c_pct:.2f}% | {got} | {s_pct:.2f}% | {gain} |")
    print()
    print(f"moves written : {out['moves']} ({out['moves'] / len(fens):.1f} per position)")
    print(f"terminal      : {out['terminal']} positions with no legal move")
    for key in ("unreadable_games", "illegal_moves_skipped"):
        if stats[key]:
            print(f"{key} : {stats[key]}")


if __name__ == "__main__":
    main()
