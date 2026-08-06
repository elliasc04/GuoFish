"""Generate golden/keys.jsonl and golden/keys_adversarial.jsonl — the C3 key data.

`python-chess` plus `core.mctsv4` are the *reference implementation* (Global Rule 2).
The C++ `nn_key`/`rep_key` are judged against these files; these files are never
regenerated to agree with C++.

Usage:
    python tools/gen_key_golden.py                 # reads golden/movegen.jsonl
    python tools/gen_key_golden.py --limit 5000    # smoke run
    python tools/gen_key_golden.py --force         # allow overwriting existing output

-------------------------------------------------------------------------------
The two keys
-------------------------------------------------------------------------------
Scope §2.4 defines them, and stresses that they are *two of three* coexisting
conventions in the Python engine — the third being python-chess's polyglot
Zobrist, which is used for tree reuse and must never be confused with either:

  nn_key   hash of the 68-token sequence itself (the NN cache key)
  rep_key  position identity under `Board._transposition_key()`'s
           **legal-en-passant** rule (repetition / 50-move detection)

`nn_key` hashes `core.mctsv4.board_to_tokens` output directly rather than
patching a Zobrist hash with an ep correction. That is scope §2.4's explicit
choice: "the key cannot be coarser than the network's input if it *is* the
network's input." Two consequences fall out for free and are pinned by the
adversarial pairs below:

  * the halfmove clock cannot enter the key, because it is not a token — so
    clock-twins share an `nn_key`, which is correct and must stay that way;
  * en-passant twins cannot share a key, because token 66 is written from
    `board.ep_square is not None` unconditionally.

Note the three ep rules that must not be conflated. Token 66 (and therefore
`nn_key`) uses the **raw** ep square. `rep_key` uses the **legal-capture** rule.
Polyglot Zobrist uses a **third** rule — pseudo-legal adjacency, whose own
source says "legality of the potential capture is irrelevant" — and so is wrong
for both purposes. There are positions in the C1 corpus where all three differ.

-------------------------------------------------------------------------------
Hash definition (the C++ side must reproduce this exactly)
-------------------------------------------------------------------------------
FNV-1a, 64-bit: offset basis 0xcbf29ce484222325, prime 0x100000001b3, over a
byte string that begins with a domain tag. Chosen over the obvious alternative
of reusing polyglot Zobrist because a Zobrist realisation of either rule would
have to *re-derive* the ep handling, which is the exact silent-divergence this
chunk exists to prevent; hashing an explicit byte serialisation makes the rule
visible in one place. FNV-1a is a few lines of dependency-free C++ (Global
Rule 7) and needs no shared random table.

  nn_key  : b"guofish/nn_key/v1\\0"  + the 68 tokens as little-endian int32
  rep_key : b"guofish/rep_key/v1\\0" + pawns, knights, bishops, rooks, queens,
            kings, occupied[WHITE], occupied[BLACK] as little-endian uint64;
            side to move as one byte (1 = white); clean_castling_rights() as
            little-endian uint64; then one byte for en passant, 0xFF when the
            legal-ep rule says "none", else the square index 0-63.

`nn_key`'s payload is the byte image of the `int32[68]` token buffer C2 already
produces, so C++ hashes the buffer it is holding with no repacking.

The domain tags are not decoration. They guarantee `nn_key(p) != rep_key(p)`
for every position, so a swapped key can never coincidentally compare equal —
the runtime half of "two keys, never mixed", complementing the compile-time
strong typedefs C3 requires.

-------------------------------------------------------------------------------
Output
-------------------------------------------------------------------------------
`golden/keys.jsonl` — one line per FEN of the shared C1 corpus
(`golden/movegen.jsonl`), the corpus C2's tokens were also built from:

    {"fen": "<FEN>", "nn_key": <uint64>, "rep_key": <uint64>}

`golden/keys_adversarial.jsonl` — one line per hand-built *pair*. A pair is the
unit of meaning here, so it is also the unit of storage; a test cannot mis-pair
records that arrive already paired, and the expected relation travels with the
data instead of being hardcoded in the test:

    {"pair_id": "...", "kind": "...", "desc": "...",
     "expect_nn": "same"|"differ", "expect_rep": "same"|"differ",
     "a": {"fen": ..., "nn_key": ..., "rep_key": ...},
     "b": {...}}

FENs are emitted with the en-passant square always set after a double pawn push
(`en_passant="fen"`), matching `golden/movegen.jsonl`. python-chess's default
(`en_passant="legal"`) omits the square unless a capture is available, which
would write the ep-twin pairs out as two identical FENs carrying two different
`nn_key` values — golden data no implementation can satisfy. `--self-check`
(on by default) refuses to write that, and every other shape of it.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import chess
import chess.polyglot
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.mctsv4 import board_to_tokens  # noqa: E402

DEFAULT_CORPUS = REPO_ROOT / "golden" / "movegen.jsonl"
DEFAULT_OUT = REPO_ROOT / "golden" / "keys.jsonl"
DEFAULT_ADV_OUT = REPO_ROOT / "golden" / "keys_adversarial.jsonl"

SEQ_LENGTH = 68
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = 0xFFFFFFFFFFFFFFFF

NN_TAG = b"guofish/nn_key/v1\0"
REP_TAG = b"guofish/rep_key/v1\0"

EP_NONE = 0xFF  # rep_key's "no legal en passant" sentinel


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def fnv1a64(data):
    h = FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * FNV_PRIME) & MASK64
    return h


def nn_payload(board):
    """The exact byte string `nn_key` hashes (scope §2.4)."""
    tokens = np.asarray(board_to_tokens(board)).astype("<i4")
    assert tokens.shape == (SEQ_LENGTH,), f"tokenizer returned {tokens.shape}"
    return NN_TAG + tokens.tobytes()


def rep_payload(board):
    """The exact byte string `rep_key` hashes.

    Mirrors the tuple python-chess builds in `Board._transposition_key()`:
        (pawns, knights, bishops, rooks, queens, kings,
         occupied_co[WHITE], occupied_co[BLACK], turn, clean_castling_rights(),
         ep_square if has_legal_en_passant() else None)
    """
    ep = board.ep_square if board.has_legal_en_passant() else None
    return REP_TAG + struct.pack(
        "<8QBQB",
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        1 if board.turn else 0,
        board.clean_castling_rights(),
        EP_NONE if ep is None else ep,
    )


def nn_key(board):
    return fnv1a64(nn_payload(board))


def rep_key(board):
    return fnv1a64(rep_payload(board))


def fen_of(board):
    """Canonical FEN: ep square present after any double push, per the FEN spec."""
    return board.fen(en_passant="fen")


def record(board):
    return {"fen": fen_of(board), "nn_key": nn_key(board), "rep_key": rep_key(board)}


# ---------------------------------------------------------------------------
# Adversarial pairs
# ---------------------------------------------------------------------------

SAME, DIFFER = "same", "differ"

# Corpus positions used as bases for constructed twins. The three ep_pin FENs
# are the only positions in the whole C1 corpus where the raw-ep rule, the
# legal-ep rule and polyglot's pseudo-legal rule all disagree; none of them is
# reachable by a short opening line, so they are carried as literals.
EP_PIN_1 = "3rkb1r/6pp/1R3p2/R2Pp3/1N2b3/2B1P3/1P1K1P1P/8 w k e6 0 27"
EP_PIN_2 = "8/5r2/8/8/4NpP1/4pB2/2R4k/1K6 b - g3 0 78"
EP_PIN_3 = "8/8/2B5/1Kp1r3/1PPp4/p7/Pk5R/8 b - c3 0 64"
IN_CHECK = "1B1r4/k4p2/pR1b1qp1/2pP2R1/2Pnp2r/3PB3/5P2/3Q1KN1 b - - 0 27"
ENDGAME = "1B5r/P2K4/8/r1N2k2/8/8/8/8 w - - 36 95"
MIDGAME = "1B2r1k1/1pq2pb1/p2p1n1p/P2P1bp1/8/1QN3PP/1P3PB1/R3R1K1 b - - 0 21"
EP_LEGAL = "1Q6/5pk1/3r1qp1/p2P3p/4pP1P/1P2P3/P1r5/3RK2R b - f3 0 29"


def line(*sans):
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def without_ep(board):
    twin = board.copy()
    twin.ep_square = None
    return twin


def with_clock(board, halfmove_clock):
    twin = board.copy()
    twin.halfmove_clock = halfmove_clock
    return twin


def recastled(fen, castling):
    parts = chess.Board(fen).fen(en_passant="fen").split(" ")
    parts[2] = castling
    return chess.Board(" ".join(parts))


def flipped_turn(fen):
    parts = chess.Board(fen).fen(en_passant="fen").split(" ")
    parts[1] = "b" if parts[1] == "w" else "w"
    return chess.Board(" ".join(parts))


def build_pairs():
    """~20 constructed pairs. Each carries the relation it is meant to prove."""
    P = []

    def add(pair_id, kind, desc, a, b, expect_nn, expect_rep):
        P.append((pair_id, kind, desc, a, b, expect_nn, expect_rep))

    # --- en-passant twins -------------------------------------------------
    # Raw-ep vs legal-ep: nn_key must always split these; rep_key splits them
    # only when the capture is actually legal.
    add("ep_no_capturer", "ep_twin",
        "1.e4 - ep square set, no black pawn able to capture",
        line("e4"), without_ep(line("e4")), DIFFER, SAME)

    add("ep_legal_white_to_move", "ep_twin",
        "1.e4 a6 2.e5 d5 - white to move, exd6 e.p. is legal",
        line("e4", "a6", "e5", "d5"), without_ep(line("e4", "a6", "e5", "d5")),
        DIFFER, DIFFER)

    add("ep_legal_black_to_move", "ep_twin",
        "1.Nf3 c5 2.Nc3 c4 3.d4 - black to move, cxd3 e.p. is legal",
        line("Nf3", "c5", "Nc3", "c4", "d4"),
        without_ep(line("Nf3", "c5", "Nc3", "c4", "d4")), DIFFER, DIFFER)

    add("ep_black_push_no_capturer", "ep_twin",
        "1.a3 d5 - white to move, ep square set, no white pawn adjacent",
        line("a3", "d5"), without_ep(line("a3", "d5")), DIFFER, SAME)

    for i, fen in enumerate((EP_PIN_1, EP_PIN_2, EP_PIN_3), start=1):
        add(f"ep_pinned_{i}", "ep_twin",
            "corpus position: a pawn attacks the ep square but the capture is "
            "illegal, so the legal-ep rule treats it as ep-less while token 66 "
            "does not (polyglot Zobrist disagrees with both)",
            chess.Board(fen), without_ep(chess.Board(fen)), DIFFER, SAME)

    # --- halfmove-clock twins ---------------------------------------------
    # C3 acceptance pins this: the network never sees the clock, so both keys
    # must ignore it. The test exists so nobody later "fixes" it.
    add("clock_start", "clock_twin", "start position, clock 0 vs 98",
        chess.Board(), with_clock(chess.Board(), 98), SAME, SAME)

    add("clock_midgame", "clock_twin", "quiet midgame, clock 0 vs 99",
        chess.Board(MIDGAME), with_clock(chess.Board(MIDGAME), 99), SAME, SAME)

    add("clock_endgame", "clock_twin", "7-piece endgame, clock 36 vs 3",
        chess.Board(ENDGAME), with_clock(chess.Board(ENDGAME), 3), SAME, SAME)

    add("clock_in_check", "clock_twin", "side to move in check, clock 0 vs 75",
        chess.Board(IN_CHECK), with_clock(chess.Board(IN_CHECK), 75), SAME, SAME)

    add("clock_with_ep", "clock_twin",
        "legal ep available, clock 0 vs 40 - the clock must not leak into "
        "either key even when the ep field is live",
        chess.Board(EP_LEGAL), with_clock(chess.Board(EP_LEGAL), 40), SAME, SAME)

    # --- transpositions ---------------------------------------------------
    # Genuinely identical positions reached by different move orders. Every
    # line ends on a move that leaves the same ep state, or the pair would be
    # testing the ep rule rather than transposition.
    add("transpose_qgd", "transposition", "1.d4 Nf6 2.c4 e6 = 1.c4 Nf6 2.d4 e6",
        line("d4", "Nf6", "c4", "e6"), line("c4", "Nf6", "d4", "e6"), SAME, SAME)

    add("transpose_kia", "transposition", "1.Nf3 Nf6 2.g3 g6 = 1.g3 Nf6 2.Nf3 g6",
        line("Nf3", "Nf6", "g3", "g6"), line("g3", "Nf6", "Nf3", "g6"), SAME, SAME)

    add("transpose_d4d5", "transposition",
        "1.d4 d5 2.Nf3 Nf6 3.g3 = 1.Nf3 Nf6 2.d4 d5 3.g3",
        line("d4", "d5", "Nf3", "Nf6", "g3"), line("Nf3", "Nf6", "d4", "d5", "g3"),
        SAME, SAME)

    add("transpose_knights", "transposition",
        "1.Nc3 Nc6 2.Nf3 Nf6 = 1.Nf3 Nf6 2.Nc3 Nc6",
        line("Nc3", "Nc6", "Nf3", "Nf6"), line("Nf3", "Nf6", "Nc3", "Nc6"), SAME, SAME)

    add("transpose_with_ep_set", "transposition",
        "1.d4 e6 2.c4 d5 = 1.c4 e6 2.d4 d5 - both arrive with the same ep "
        "square set, so the keys must match despite a live ep field",
        line("d4", "e6", "c4", "d5"), line("c4", "e6", "d4", "d5"), SAME, SAME)

    # --- near misses ------------------------------------------------------
    # These look like transpositions and are not. Without them a movegen that
    # drops the ep square entirely would pass every pair above.
    add("near_miss_ep_file", "near_miss",
        "1.d4 Nf6 2.c4 vs 1.c4 Nf6 2.d4 - identical placement, castling and "
        "side to move; the ep file differs (c3 vs d3), so nn_key must differ "
        "while rep_key must not (neither ep capture is legal)",
        line("d4", "Nf6", "c4"), line("c4", "Nf6", "d4"), DIFFER, SAME)

    add("castling_kq_vs_k", "near_miss",
        "start position, KQkq vs Kkq - one castling right withdrawn",
        chess.Board(), recastled(chess.STARTING_FEN, "Kkq"), DIFFER, DIFFER)

    add("castling_all_vs_none", "near_miss",
        "start position, KQkq vs no castling rights at all",
        chess.Board(), recastled(chess.STARTING_FEN, "-"), DIFFER, DIFFER)

    add("castling_white_vs_black", "near_miss",
        "start position, KQ vs kq - same count of rights, different side",
        recastled(chess.STARTING_FEN, "KQ"), recastled(chess.STARTING_FEN, "kq"),
        DIFFER, DIFFER)

    add("side_to_move", "near_miss",
        "start position, white to move vs black to move",
        chess.Board(), flipped_turn(chess.STARTING_FEN), DIFFER, DIFFER)

    return [
        {
            "pair_id": pid,
            "kind": kind,
            "desc": desc,
            "expect_nn": exp_nn,
            "expect_rep": exp_rep,
            "a": record(a),
            "b": record(b),
        }
        for pid, kind, desc, a, b, exp_nn, exp_rep in P
    ]


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


def check_pair(pair):
    """Every way this data could be self-contradictory, checked before writing."""
    pid = pair["pair_id"]
    for role in ("a", "b"):
        side = pair[role]
        board = chess.Board(side["fen"])  # must parse
        if fen_of(board) != side["fen"]:
            raise AssertionError(f"{pid}.{role}: FEN is not canonical: {side['fen']}")
        # The keys must be recoverable from the FEN alone. This is what the C++
        # side is given, and the only thing it is given.
        if nn_key(board) != side["nn_key"] or rep_key(board) != side["rep_key"]:
            raise AssertionError(
                f"{pid}.{role}: keys do not survive a FEN round trip -- the "
                f"stored FEN describes a different position than the one hashed"
            )
        if side["nn_key"] == side["rep_key"]:
            raise AssertionError(f"{pid}.{role}: nn_key == rep_key, domain tags failed")

    for key, expected in (("nn_key", pair["expect_nn"]), ("rep_key", pair["expect_rep"])):
        equal = pair["a"][key] == pair["b"][key]
        if equal != (expected == SAME):
            raise AssertionError(
                f"{pid}: declared {key} {expected}, but the pair is "
                f"{'equal' if equal else 'unequal'}\n  a: {pair['a']['fen']}\n  b: {pair['b']['fen']}"
            )

    if pair["a"]["fen"] == pair["b"]["fen"] and (
        pair["expect_nn"] == DIFFER or pair["expect_rep"] == DIFFER
    ):
        raise AssertionError(f"{pid}: identical FENs cannot have differing keys")


def check_no_conflicts(rows, pairs):
    """One FEN must never map to two different key values, anywhere in the set."""
    seen = {}
    everything = list(rows) + [p[role] for p in pairs for role in ("a", "b")]
    for r in everything:
        prior = seen.setdefault(r["fen"], r)
        if (prior["nn_key"], prior["rep_key"]) != (r["nn_key"], r["rep_key"]):
            raise AssertionError(
                f"conflicting keys for the same FEN: {r['fen']}\n"
                f"  {prior['nn_key']}/{prior['rep_key']} vs {r['nn_key']}/{r['rep_key']}"
            )


def check_partition_matches_python(rows):
    """`nn_key` must induce the same equivalence classes as `make_cache_key`.

    C3's constraint on `nn_key` is partition equality with the post-fix Python
    key, not value equality — `make_cache_key` returns the tuple
    `(polyglot_zobrist, board.ep_square)`, so there is no value to match. What
    the cache actually depends on is which positions compare equal, and on that
    the two agree:

      tokens equal => Python key equal.  Exact. Same tokens fix placement, side,
        the four castling flags and the ep file; the ep rank is implied by side
        to move, so the raw ep square matches too, and every input to the
        Zobrist is then equal.
      Python key equal => tokens equal.  Holds unless two distinct positions
        collide in the 64-bit Zobrist. So if this check ever fails it is
        Python's key that is coarser, not ours -- the same failure mode the ep
        correction was introduced to fix, surviving only as a hash collision.

    Asserted here rather than argued, so `tests/test_c3_keys.py` only has to
    compare C++ against the golden file.
    """
    python_first, ours_first = {}, {}
    python_part, ours_part = [], []
    for i, row in enumerate(rows):
        board = chess.Board(row["fen"])
        python = (chess.polyglot.zobrist_hash(board), board.ep_square)
        python_part.append(python_first.setdefault(python, i))
        ours_part.append(ours_first.setdefault(row["nn_key"], i))

    if python_part != ours_part:
        i = next(i for i, (a, b) in enumerate(zip(python_part, ours_part)) if a != b)
        raise AssertionError(
            f"nn_key does not partition the corpus the way make_cache_key does; "
            f"first divergence at row {i}: {rows[i]['fen']}"
        )
    return len(python_first)


class CollisionCheck:
    """Separates two key equalities that look identical in the output file.

    Positions differing only in the halfmove/fullmove counters *must* share both
    keys — neither payload contains a clock — and the corpus is deduplicated on
    the full FEN, so it holds many such groups. A genuine FNV-1a-64 collision
    would be indistinguishable in the file but is astronomically unlikely, so
    seeing one means the hash is broken (a truncation or byte-order slip), not
    that we got unlucky. Distinguished by keeping a strong digest of the payload
    each key was computed from.
    """

    def __init__(self, name):
        self.name = name
        self.digests = {}
        self.shared = 0  # positions that share a key with an earlier position

    def add(self, key, payload):
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        prior = self.digests.setdefault(key, digest)
        if prior is digest:
            return
        self.shared += 1
        if prior != digest:
            raise AssertionError(
                f"{self.name}: genuine hash collision on {key:#018x} -- two "
                f"different payloads produced the same key. The hash is wrong."
            )

    @property
    def distinct(self):
        return len(self.digests)


# ---------------------------------------------------------------------------


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def refuse_to_clobber(path, force):
    if path.exists() and not force:
        sys.exit(
            f"error: {path} already exists.\n"
            "Global Rule 1 forbids regenerating golden data. If it is genuinely "
            "stale, delete it deliberately or pass --force."
        )


def main():
    parser = argparse.ArgumentParser(description="Generate the C3 golden key data.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--adv-out", type=Path, default=DEFAULT_ADV_OUT)
    parser.add_argument("--limit", type=int, default=None, help="cap corpus rows; for smoke runs")
    parser.add_argument("--force", action="store_true", help="overwrite existing golden files")
    parser.add_argument("--no-self-check", dest="self_check", action="store_false")
    args = parser.parse_args()

    refuse_to_clobber(args.out, args.force)
    refuse_to_clobber(args.adv_out, args.force)

    if not args.corpus.exists():
        sys.exit(f"error: corpus {args.corpus} not found -- run tools/gen_movegen_golden.py first")

    print("building adversarial pairs", file=sys.stderr)
    pairs = build_pairs()
    if args.self_check:
        for pair in pairs:
            check_pair(pair)

    print(f"reading corpus FENs from {args.corpus}", file=sys.stderr)
    fens = []
    with open(args.corpus, encoding="utf-8") as handle:
        for line_text in handle:
            if line_text.strip():
                fens.append(json.loads(line_text)["fen"])
    if args.limit is not None:
        fens = fens[: args.limit]

    print(f"keying {len(fens)} corpus positions", file=sys.stderr)
    rows = []
    nn_check, rep_check = CollisionCheck("nn_key"), CollisionCheck("rep_key")
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        if args.self_check and fen_of(board) != fen:
            raise AssertionError(f"corpus FEN is not canonical: {fen}")
        nn_bytes, rep_bytes = nn_payload(board), rep_payload(board)
        row = {"fen": fen, "nn_key": fnv1a64(nn_bytes), "rep_key": fnv1a64(rep_bytes)}
        if args.self_check:
            nn_check.add(row["nn_key"], nn_bytes)
            rep_check.add(row["rep_key"], rep_bytes)
        rows.append(row)
        if (i + 1) % 25_000 == 0:
            print(f"  {i + 1}/{len(fens)}", file=sys.stderr)

    classes = None
    if args.self_check:
        check_no_conflicts(rows, pairs)
        print("checking nn_key partitions the corpus as make_cache_key does", file=sys.stderr)
        classes = check_partition_matches_python(rows)

    write_jsonl(rows, args.out)
    write_jsonl(pairs, args.adv_out)

    kinds = {}
    for pair in pairs:
        kinds[pair["kind"]] = kinds.get(pair["kind"], 0) + 1

    print()
    print(f"corpus      : {args.corpus}")
    print(f"output      : {args.out}  ({len(rows)} positions)")
    print(f"              {args.adv_out}  ({len(pairs)} pairs)")
    print(f"self-check  : {'on' if args.self_check else 'OFF'}")
    print()
    print("| pair kind | count |")
    print("|---|---:|")
    for kind, count in sorted(kinds.items()):
        print(f"| {kind} | {count} |")
    print()
    if args.self_check:
        print(f"distinct nn_key  : {nn_check.distinct}/{len(rows)}")
        print(f"distinct rep_key : {rep_check.distinct}/{len(rows)}")
        print(
            f"shared keys      : {nn_check.shared} nn_key, {rep_check.shared} rep_key -- "
            "all verified as positions with identical payloads (same board, "
            "different clocks), not hash collisions."
        )
        print("true collisions  : 0 for both (asserted; a nonzero count aborts the run)")
        print(
            f"partition        : nn_key induces the same {classes} equivalence classes as "
            "Python's make_cache_key (asserted) -- so the C3 test only has to compare "
            "C++ against this file."
        )


if __name__ == "__main__":
    main()
