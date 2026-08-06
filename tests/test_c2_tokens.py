"""C2 acceptance tests — 68-token board encoding parity with python-chess.

`golden/tokens.npz` is the reference (Global Rule 2): it was produced by
`tools/gen_token_golden.py` from `core.mctsv4.board_to_tokens`, and it is never
regenerated to agree with the C++ side. If the two disagree, the C++ side is
wrong.

Why this chunk gets a suite rather than one `assert_array_equal`
----------------------------------------------------------------
Tokenization has no failure mode that looks like a crash. Every wrong encoding
this test could catch is a well-formed 68-vector of plausible integers, and the
network answers it with a confident evaluation and a full set of policy priors.
The only thing standing between a one-token bug and a silently mis-playing
engine is this file.

So the suite has four layers:

  * **Parity** compares all 100k positions through both entry points — the
    single-position `tokens(fen)` and the batched `TokenBatch.fill()` — because
    they are separate code paths into the same encoder and only one of them is
    what the search will use.
  * **Corpus coverage** asserts the golden file actually contains the cases the
    parity run would otherwise pass vacuously. In particular it counts the
    en-passant positions *with no capturer available*, which is the trap the
    brief names: chess-library's `setFen` discards the ep square on exactly
    those, so an implementation built on it would be wrong on 3.2% of the
    corpus and right everywhere else.
  * **Targeted** cases pin each field with hand-auditable literals. Every value
    below was cross-checked against `core.mctsv4.board_to_tokens` on
    python-chess 1.11.2 and is short enough to verify against the FEN by hand.
  * **Buffer semantics** check that the batch path really does write into
    C++-owned memory rather than handing back a copy.

Nothing here imports `chess`. The reference's answers reach this file only
through the golden data and through the literals above, so the tests cannot
drift into re-deriving the expectation from the same library that produced it.

Pointing the parity test at a different file
--------------------------------------------
``GUOFISH_GOLDEN_TOKENS=/path/to/other.npz`` overrides the input. This exists so
the diagnostic output can be exercised against a deliberately corrupted copy
without writing to ``golden/`` (Global Rules 1 and 2); see DECISIONS.md, "C2 /
mutation check". The corpus-coverage tests skip when the override is set.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = REPO_ROOT / "golden" / "tokens.npz"
GOLDEN_ENV = "GUOFISH_GOLDEN_TOKENS"

SEQ_LENGTH = 68

# Token contract, restated here rather than imported from the module under test.
# A suite that read these off `guofish_core` would agree with itself no matter
# what the module decided they were.
TOKEN_EMPTY = 0
TOKEN_WHITE_TO_MOVE = 13
TOKEN_BLACK_TO_MOVE = 14
TOKEN_CASTLING_BASE = 15
TOKEN_EP_NONE = 31
TOKEN_EP_BASE = 32
TOKEN_CLS = 40

IDX_SIDE_TO_MOVE = 64
IDX_CASTLING = 65
IDX_EN_PASSANT = 66
IDX_CLS = 67

# How many mismatching positions are reported in full before the report is
# truncated. A systematically wrong field produces all 100k of them and the
# first few are enough to name the bug.
MAX_REPORTED = 10

# Rows per TokenBatch.fill() call in the parity sweep. Deliberately not the
# whole corpus: a single fill starting at row 0 would never exercise
# `row_offset`, which is the argument a real dispatcher packing several
# positions into one network batch actually uses.
BATCH_ROWS = 4096


# ---------------------------------------------------------------------------
# FEN reading, for the coverage claims only
#
# Enough of a FEN parser to answer "does this corpus contain the hard cases",
# and deliberately not enough to be a second tokenizer. It never produces a
# token; if it were wrong, the coverage tests would fail, not the parity test.
# ---------------------------------------------------------------------------


def board_squares(fen):
    """The placement field as a list of 64 chars ('.' for empty), a1 at index 0."""
    squares = ["."] * 64
    square = 56
    for char in fen.split()[0]:
        if char.isdigit():
            square += int(char)
        elif char == "/":
            square -= 16
        else:
            squares[square] = char
            square += 1
    return squares


def ep_square_of(fen):
    """(file, rank) of the FEN's en-passant target, both 0-based, or None."""
    field = fen.split()[3]
    if field == "-":
        return None
    return ord(field[0]) - ord("a"), int(field[1]) - 1


def ep_capturer_is_ready(fen):
    """True when a pawn of the side to move is placed to play the ep capture.

    This is the distinction chess-library's `setFen` makes and python-chess does
    not. It is only ever used to *count* positions: the encoding must be
    identical either way, and `test_en_passant_is_emitted_without_a_capturer`
    is what asserts that.

    Placement only — a pinned pawn is still counted, so this slightly
    over-counts the ready side and under-counts the trap side. That is the safe
    direction for a coverage floor.
    """
    ep = ep_square_of(fen)
    if ep is None:
        return False

    ep_file, ep_rank = ep
    squares = board_squares(fen)

    if fen.split()[1] == "w":
        pawn, rank = "P", ep_rank - 1   # ep on rank 6, White pawns capture from rank 5
    else:
        pawn, rank = "p", ep_rank + 1   # ep on rank 3, Black pawns capture from rank 4

    if not 0 <= rank < 8:
        return False

    return any(
        squares[rank * 8 + f] == pawn
        for f in (ep_file - 1, ep_file + 1)
        if 0 <= f < 8
    )


def golden_path():
    override = os.environ.get(GOLDEN_ENV)
    return Path(override) if override else DEFAULT_GOLDEN


def using_default_golden():
    return not os.environ.get(GOLDEN_ENV)


# ---------------------------------------------------------------------------
# Failure reporting
#
# Module level and dependency-free so that the diagnostic can be unit tested
# directly (test_diagnostic_* below) instead of only being seen once something
# is already broken.
# ---------------------------------------------------------------------------

# What each index means, so a report names the field rather than a number.
FIELD_NAMES = {
    IDX_SIDE_TO_MOVE: "side to move",
    IDX_CASTLING: "castling rights",
    IDX_EN_PASSANT: "en-passant file",
    IDX_CLS: "CLS",
}


def describe_index(index):
    if index in FIELD_NAMES:
        return f"[{index}] {FIELD_NAMES[index]}"
    file_name = "abcdefgh"[index % 8]
    return f"[{index}] square {file_name}{index // 8 + 1}"


def format_mismatch(source, position, fen, cpp, ref, path_label="tokens"):
    """A full account of one disagreeing position.

    Reports the FEN, both 68-element arrays in full, and every index at which
    they differ with both values. `.tolist()` rather than NumPy's repr on
    purpose: NumPy elides the middle of a 68-element array with `...`, and the
    elided region is squares 24..47, which is most of the board.
    """
    cpp = np.asarray(cpp)
    ref = np.asarray(ref)

    lines = [
        f"{source}[{position}]  via {path_label}",
        f"  FEN      : {fen}",
        f"  expected : {ref.tolist()}",
        f"  C++      : {cpp.tolist()}",
    ]

    if cpp.shape != ref.shape:
        lines.append(f"  SHAPE    : C++ {cpp.shape} vs expected {ref.shape}")
        return "\n".join(lines)

    differing = np.nonzero(cpp != ref)[0]
    lines.append(f"  {len(differing)} differing index(es):")
    for index in differing.tolist():
        lines.append(
            f"    {describe_index(index):<28} expected {int(ref[index]):3d}  got {int(cpp[index]):3d}"
        )

    if IDX_EN_PASSANT in differing.tolist():
        got, want = int(cpp[IDX_EN_PASSANT]), int(ref[IDX_EN_PASSANT])
        if got == TOKEN_EP_NONE and TOKEN_EP_BASE <= want < TOKEN_EP_BASE + 8:
            lines.append(
                "  EN PASSANT DISCARDED: the FEN carries an ep target and the encoder emitted "
                f"{TOKEN_EP_NONE} (none). This is what chess-library's setFen() does when no "
                "legal ep capture exists; the encoding must follow the FEN field, not the "
                "capture's legality."
            )

    if IDX_CASTLING in differing.tolist():
        got, want = int(cpp[IDX_CASTLING]), int(ref[IDX_CASTLING])
        lines.append(
            f"  CASTLING MASK: expected {want - TOKEN_CASTLING_BASE:04b}, got "
            f"{got - TOKEN_CASTLING_BASE:04b} (bits are WK WQ BK BQ, high to low)"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# One pass over the corpus, shared by every test that needs it
# ---------------------------------------------------------------------------


class Scan:
    """Results of a single pass over the golden file."""

    def __init__(self):
        self.path = None
        self.positions = 0
        self.reports = []           # formatted diagnostics, capped at MAX_REPORTED
        self.mismatches = 0         # positions where tokens(fen) disagrees
        self.batch_mismatches = 0   # positions where TokenBatch.fill disagrees
        self.batch_reports = []
        # Corpus coverage, counted from the FENs and the reference tokens only.
        self.with_ep_square = 0
        self.with_ep_no_capturer = 0
        self.distinct_castling_tokens = set()
        self.distinct_ep_tokens = set()
        self.distinct_side_tokens = set()
        self.distinct_piece_tokens = set()


@pytest.fixture(scope="module")
def golden():
    path = golden_path()
    if not path.exists():
        pytest.fail(
            f"golden data missing: {path}\n"
            "It is generated by tools/gen_token_golden.py from python-chess and is "
            "not produced by this chunk (Global Rule 2).",
            pytrace=False,
        )

    with np.load(path, allow_pickle=True) as archive:
        fens = [str(f) for f in archive["fens"]]
        tokens = np.asarray(archive["tokens"])

    if len(fens) != len(tokens):
        pytest.fail(
            f"{path} holds {len(fens)} FENs but {len(tokens)} token rows; the reference "
            "file is internally inconsistent and nothing can be judged against it.",
            pytrace=False,
        )

    return path, fens, tokens


@pytest.fixture(scope="module")
def scan(golden):
    path, fens, reference = golden

    result = Scan()
    result.path = path
    result.positions = len(fens)

    # --- single-position path, and coverage counting ---
    for index, fen in enumerate(fens):
        expected = reference[index]

        if fen.split()[3] != "-":
            result.with_ep_square += 1
            if not ep_capturer_is_ready(fen):
                result.with_ep_no_capturer += 1

        result.distinct_castling_tokens.add(int(expected[IDX_CASTLING]))
        result.distinct_ep_tokens.add(int(expected[IDX_EN_PASSANT]))
        result.distinct_side_tokens.add(int(expected[IDX_SIDE_TO_MOVE]))
        result.distinct_piece_tokens.update(int(t) for t in expected[:64])

        actual = guofish_core.tokens(fen)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            result.mismatches += 1
            if len(result.reports) < MAX_REPORTED:
                result.reports.append(
                    format_mismatch(path.name, index, fen, actual, expected, "tokens(fen)")
                )

    # --- batch path, in chunks, exercising row_offset ---
    batch = guofish_core.TokenBatch(BATCH_ROWS)
    view = batch.view()

    for start in range(0, len(fens), BATCH_ROWS // 2):
        chunk = fens[start:start + BATCH_ROWS // 2]
        # Alternating offsets so rows are written at both 0 and a non-zero
        # start, and so a stale row from the previous chunk would be read back.
        offset = 0 if (start // (BATCH_ROWS // 2)) % 2 == 0 else BATCH_ROWS // 2
        written = batch.fill(chunk, offset)
        assert written == len(chunk)

        actual = view[offset:offset + written]
        expected = reference[start:start + written]
        if not np.array_equal(actual, expected):
            for row in np.nonzero((actual != expected).any(axis=1))[0].tolist():
                result.batch_mismatches += 1
                if len(result.batch_reports) < MAX_REPORTED:
                    result.batch_reports.append(
                        format_mismatch(
                            path.name, start + row, fens[start + row],
                            actual[row], expected[row], f"TokenBatch.fill(row_offset={offset})",
                        )
                    )

    return result


# ---------------------------------------------------------------------------
# Parity — the acceptance criterion
# ---------------------------------------------------------------------------


def test_tokens_match_golden(scan):
    """Every position: the C++ 68-token array equals the reference exactly."""
    assert scan.positions > 0, f"{scan.path} contained no positions"

    if scan.mismatches:
        shown = len(scan.reports)
        truncated = "" if shown == scan.mismatches else f" (showing the first {shown})"
        pytest.fail(
            f"{scan.mismatches} of {scan.positions} positions disagree with "
            f"{scan.path}{truncated}:\n\n" + "\n\n".join(scan.reports),
            pytrace=False,
        )


def test_batch_tokens_match_golden(scan):
    """The batched path agrees too — it is the one the evaluator will use.

    Separate from the test above rather than folded into it: `tokens()` and
    `TokenBatch.fill()` reach the same encoder through different plumbing (a
    fresh NumPy array per call vs. a row of a C++-owned buffer, GIL held vs.
    GIL released), and a bug in the plumbing would show up in exactly one.
    """
    if scan.batch_mismatches:
        shown = len(scan.batch_reports)
        truncated = "" if shown == scan.batch_mismatches else f" (showing the first {shown})"
        pytest.fail(
            f"{scan.batch_mismatches} of {scan.positions} positions disagree with "
            f"{scan.path} via the batch path{truncated}:\n\n" + "\n\n".join(scan.batch_reports),
            pytrace=False,
        )


def test_reference_rows_are_the_right_shape(golden):
    """A guard on the reference itself: 68 int columns, or nothing below means much."""
    path, fens, reference = golden
    assert reference.ndim == 2, f"{path}: expected a 2D token array, got shape {reference.shape}"
    assert reference.shape[1] == SEQ_LENGTH, (
        f"{path}: expected {SEQ_LENGTH} columns, got {reference.shape[1]}"
    )
    assert np.issubdtype(reference.dtype, np.integer), f"{path}: token dtype is {reference.dtype}"
    assert guofish_core.SEQ_LENGTH == SEQ_LENGTH


# ---------------------------------------------------------------------------
# Corpus coverage — does the parity run actually exercise the traps?
# ---------------------------------------------------------------------------

requires_default_golden = pytest.mark.skipif(
    not using_default_golden(),
    reason=f"{GOLDEN_ENV} overrides the corpus; coverage claims apply to golden/tokens.npz only",
)


@requires_default_golden
def test_golden_corpus_is_not_truncated(scan):
    """Guards against a green run over a stub file."""
    assert scan.positions >= 50_000, (
        f"{scan.path} holds {scan.positions} positions; the C2 corpus is 100k. "
        "A truncated file would let parity pass without covering the corpus."
    )


@requires_default_golden
def test_corpus_covers_en_passant_without_a_capturer(scan):
    """THE TRAP. The parity run must contain positions no ep capture can act on.

    Without these, an implementation that read the ep square off a
    `chess::Board` — which discards it when no legal ep capture exists — would
    pass the parity test outright.
    """
    assert scan.with_ep_square >= 1_000, (
        f"only {scan.with_ep_square} positions carry an en-passant target in {scan.path}"
    )
    assert scan.with_ep_no_capturer >= 1_000, (
        f"only {scan.with_ep_no_capturer} of {scan.with_ep_square} en-passant positions in "
        f"{scan.path} have no pawn placed to capture; the parity run is not exercising the "
        "case where chess-library and python-chess disagree"
    )


@requires_default_golden
def test_en_passant_is_emitted_without_a_capturer(golden):
    """Every ep-carrying FEN encodes its file, capturer or not, and by file.

    Asserted against the *FEN* rather than against the golden tokens, so this
    still fails if the reference and the C++ side were somehow wrong together.
    """
    _, fens, _ = golden

    trap_positions = 0
    for fen in fens:
        ep = ep_square_of(fen)
        actual = guofish_core.tokens(fen)
        emitted = int(actual[IDX_EN_PASSANT])

        if ep is None:
            assert emitted == TOKEN_EP_NONE, f"{fen}: no ep target but index 66 is {emitted}"
            continue

        expected = TOKEN_EP_BASE + ep[0]
        assert TOKEN_EP_BASE <= emitted < TOKEN_EP_BASE + 8, (
            f"{fen}: carries an ep target on file {'abcdefgh'[ep[0]]} but index 66 is "
            f"{emitted}, outside the {TOKEN_EP_BASE}..{TOKEN_EP_BASE + 7} range"
        )
        assert emitted == expected, (
            f"{fen}: ep target is on file {'abcdefgh'[ep[0]]}, expected {expected}, got {emitted}"
        )

        if not ep_capturer_is_ready(fen):
            trap_positions += 1

    assert trap_positions >= 1_000, (
        f"only {trap_positions} en-passant positions with no capturer were checked"
    )


@requires_default_golden
@pytest.mark.parametrize(
    "attribute,label,minimum",
    [
        ("distinct_castling_tokens", "distinct castling tokens", 16),
        ("distinct_ep_tokens", "distinct en-passant tokens", 9),
        ("distinct_side_tokens", "distinct side-to-move tokens", 2),
        ("distinct_piece_tokens", "distinct square tokens", 13),
    ],
)
def test_corpus_covers_every_token_value(scan, attribute, label, minimum):
    """Each field's full value range appears in the parity run.

    16 castling tokens is every 4-bit mask; 9 ep tokens is "none" plus all eight
    files; 13 square tokens is empty plus 12 pieces. A corpus missing one of
    these would let a single mis-mapped value through.
    """
    seen = getattr(scan, attribute)
    assert len(seen) >= minimum, (
        f"only {len(seen)} {label} appear in {scan.path} (expected {minimum}): {sorted(seen)}"
    )


# ---------------------------------------------------------------------------
# Targeted cases
#
# Every literal below was cross-checked against core.mctsv4.board_to_tokens on
# python-chess 1.11.2 and is short enough to audit against the FEN by hand.
# ---------------------------------------------------------------------------


def test_startpos():
    """The whole 68-vector, written out. Squares are a1=0 .. h8=63."""
    assert guofish_core.tokens(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ).tolist() == [
        4, 2, 3, 5, 6, 3, 2, 4,          # a1..h1  R N B Q K B N R
        1, 1, 1, 1, 1, 1, 1, 1,          # a2..h2  White pawns
        0, 0, 0, 0, 0, 0, 0, 0,          # a3..h3
        0, 0, 0, 0, 0, 0, 0, 0,          # a4..h4
        0, 0, 0, 0, 0, 0, 0, 0,          # a5..h5
        0, 0, 0, 0, 0, 0, 0, 0,          # a6..h6
        7, 7, 7, 7, 7, 7, 7, 7,          # a7..h7  Black pawns
        10, 8, 9, 11, 12, 9, 8, 10,      # a8..h8  r n b q k b n r
        13,                              # White to move
        30,                              # 15 + 0b1111, all four rights
        31,                              # no en-passant target
        40,                              # CLS
    ]


def test_every_piece_type_maps_to_its_own_token():
    """White is piece_type, Black is piece_type + 6, in python-chess's order.

    Not a legal position — pawns and kings share the back ranks — which is the
    point: it puts one of each of the twelve pieces on a known square so the
    whole mapping is pinned by a single array, and python-chess encodes it
    without complaint.
    """
    tokens = guofish_core.tokens("qrbnkp2/8/8/8/8/8/8/QRBNKP2 w - - 0 1")

    assert tokens[:6].tolist() == [5, 4, 3, 2, 6, 1], "White Q R B N K P on a1..f1"
    assert tokens[56:62].tolist() == [11, 10, 9, 8, 12, 7], "Black q r b n k p on a8..f8"
    assert tokens[6] == TOKEN_EMPTY and tokens[62] == TOKEN_EMPTY


def test_square_indexing_is_a1_zero_h8_sixtythree():
    """The one ordering the encoding cannot survive getting backwards."""
    tokens = guofish_core.tokens("k7/8/8/8/8/8/8/7K w - - 0 1")

    assert tokens[7] == 6, "White king on h1 is square 7"
    assert tokens[56] == 12, "Black king on a8 is square 56"
    assert (tokens[:64] == 0).sum() == 62


@pytest.mark.parametrize(
    "fen,expected",
    [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", TOKEN_WHITE_TO_MOVE),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", TOKEN_BLACK_TO_MOVE),
    ],
)
def test_side_to_move(fen, expected):
    assert guofish_core.tokens(fen)[IDX_SIDE_TO_MOVE] == expected


@pytest.mark.parametrize(
    "name,fen,expected",
    [
        # One bit at a time, so a transposed mask cannot pass. The bit order is
        # (White_K << 3) | (White_Q << 2) | (Black_K << 1) | (Black_Q << 0).
        ("none",       "4k3/8/8/8/8/8/8/4K3 w - - 0 1",           15 + 0b0000),
        ("black_q",    "r3k3/8/8/8/8/8/8/4K3 b q - 0 1",          15 + 0b0001),
        ("black_k",    "4k2r/8/8/8/8/8/8/4K3 b k - 0 1",          15 + 0b0010),
        ("white_q",    "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",          15 + 0b0100),
        ("white_k",    "4k3/8/8/8/8/8/8/4K2R w K - 0 1",          15 + 0b1000),
        ("all",        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",    15 + 0b1111),
    ],
)
def test_castling_bit_order(name, fen, expected):
    assert guofish_core.tokens(fen)[IDX_CASTLING] == expected, name


def test_castling_rights_are_cleaned_against_the_position():
    """`board_to_tokens` asks python-chess, and python-chess checks the board.

    It calls `has_kingside_castling_rights()` rather than reading the FEN's
    letters, and those methods discard a claimed right when the king is not on
    its home square or the corner rook is gone. Both FENs below claim "KQkq";
    the reference encodes neither at 30.

    Every FEN in golden/tokens.npz is self-consistent, so the parity run alone
    would pass with the letters copied straight through. These two are what
    fails in that case.
    """
    # White king on d1: White loses both rights, Black keeps both -> 0b0011.
    assert guofish_core.tokens("r3k2r/8/8/8/8/8/8/R2K3R w KQkq - 0 1")[IDX_CASTLING] == 15 + 0b0011

    # No White rook on a1: White keeps only kingside -> 0b1011.
    assert guofish_core.tokens("r3k2r/8/8/8/8/8/8/4K2R w KQkq - 0 1")[IDX_CASTLING] == 15 + 0b1011


@pytest.mark.parametrize(
    "fen,expected",
    [
        ("4k3/8/p7/8/8/8/8/4K3 w - a6 0 1", TOKEN_EP_BASE + 0),
        ("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1", TOKEN_EP_BASE + 4),
        ("4k3/8/8/8/7P/8/8/4K3 b - h3 0 1", TOKEN_EP_BASE + 7),
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", TOKEN_EP_NONE),
    ],
)
def test_en_passant_file(fen, expected):
    assert guofish_core.tokens(fen)[IDX_EN_PASSANT] == expected


def test_en_passant_is_encoded_with_no_pawn_able_to_capture():
    """The single most important line in this file.

    Both FENs carry `e3`. In the first, Black's f4 pawn can play exf3 e.p.; in
    the second there is no Black pawn on the board at all. python-chess reports
    `ep_square = e3` for both and the encoding is therefore identical.
    chess-library's `setFen` reports NO_SQ for the second, which would encode as
    31 — a different network input for the same trained contract.
    """
    ready = guofish_core.tokens("4k3/8/8/8/4Pp2/8/8/4K3 b - e3 0 1")
    trap = guofish_core.tokens("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1")

    assert int(ready[IDX_EN_PASSANT]) == TOKEN_EP_BASE + 4
    assert int(trap[IDX_EN_PASSANT]) == TOKEN_EP_BASE + 4, (
        "the ep token was dropped on a position with no capturer; the encoding follows the "
        "FEN's ep field, never the legality of the capture"
    )


def test_cls_token_is_always_last():
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "4k3/8/8/8/8/8/8/4K3 b - - 99 250",
        "7k/5KQ1/8/8/8/8/8/8 b - - 0 1",
    ]:
        tokens = guofish_core.tokens(fen)
        assert len(tokens) == SEQ_LENGTH
        assert tokens[IDX_CLS] == TOKEN_CLS


def test_move_counters_do_not_reach_the_encoding():
    """The halfmove clock and fullmove number are not part of the contract."""
    a = guofish_core.tokens("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    b = guofish_core.tokens("4k3/8/8/8/8/8/8/4K3 w - - 99 250")
    assert np.array_equal(a, b)


def test_output_is_int32():
    """The dispatcher's buffer is int32; a returned int64 would be a silent copy."""
    assert guofish_core.tokens("4k3/8/8/8/8/8/8/4K3 w - - 0 1").dtype == np.int32


# ---------------------------------------------------------------------------
# Buffer semantics — the batch path must not be handing back copies
# ---------------------------------------------------------------------------


def test_batch_view_is_the_right_shape_and_type():
    batch = guofish_core.TokenBatch(8)
    view = batch.view()

    assert batch.capacity == 8
    assert view.shape == (8, SEQ_LENGTH)
    assert view.dtype == np.int32


def test_batch_view_aliases_cpp_memory():
    """Two views of one batch must be the same memory, not two snapshots."""
    batch = guofish_core.TokenBatch(4)
    first = batch.view()
    second = batch.view()

    assert not first.flags.owndata, "view() returned an array that owns its data — that is a copy"
    assert first.__array_interface__["data"][0] == second.__array_interface__["data"][0]

    first[2, 5] = 12345
    assert second[2, 5] == 12345


def test_fill_is_visible_through_a_view_taken_beforehand():
    """A view handed out before fill() must see what fill() writes.

    This is the property the evaluator depends on: it takes one view at startup
    and hands the same array to the network every batch.
    """
    batch = guofish_core.TokenBatch(2)
    view = batch.view()
    view[:] = -1

    written = batch.fill(["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"])

    assert written == 1
    assert view[0, IDX_CLS] == TOKEN_CLS
    assert view[0, IDX_SIDE_TO_MOVE] == TOKEN_WHITE_TO_MOVE
    assert (view[1] == -1).all(), "fill() wrote outside the rows it was given"


def test_fill_honours_the_row_offset():
    batch = guofish_core.TokenBatch(4)
    view = batch.view()
    view[:] = -1

    assert batch.fill(["4k3/8/8/8/8/8/8/4K3 w - - 0 1"], 2) == 1

    assert (view[0] == -1).all() and (view[1] == -1).all()
    assert view[2, IDX_CLS] == TOKEN_CLS
    assert (view[3] == -1).all()


def test_fill_accepts_any_iterable():
    """A generator must be consumed once, here, not re-entered without the GIL."""
    batch = guofish_core.TokenBatch(4)
    fens = ["4k3/8/8/8/8/8/8/4K3 w - - 0 1", "4k3/8/8/8/8/8/8/4K3 b - - 0 1"]

    assert batch.fill(iter(fens)) == 2
    assert batch.view()[0, IDX_SIDE_TO_MOVE] == TOKEN_WHITE_TO_MOVE
    assert batch.view()[1, IDX_SIDE_TO_MOVE] == TOKEN_BLACK_TO_MOVE


def test_fill_matches_the_single_position_path():
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1",
        "r3k2r/8/8/8/8/8/8/R2K3R w KQkq - 0 1",
        "7k/5KQ1/8/8/8/8/8/8 b - - 0 1",
    ]
    batch = guofish_core.TokenBatch(len(fens))
    batch.fill(fens)
    view = batch.view()

    for row, fen in enumerate(fens):
        assert np.array_equal(view[row], guofish_core.tokens(fen)), fen


def test_fill_rejects_a_batch_that_does_not_fit():
    batch = guofish_core.TokenBatch(2)
    fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"

    with pytest.raises(ValueError):
        batch.fill([fen] * 3)
    with pytest.raises(ValueError):
        batch.fill([fen, fen], 1)


def test_fill_rejects_a_non_string_element():
    batch = guofish_core.TokenBatch(2)
    with pytest.raises(TypeError):
        batch.fill(["4k3/8/8/8/8/8/8/4K3 w - - 0 1", 7])


def test_zero_capacity_is_refused():
    with pytest.raises(ValueError):
        guofish_core.TokenBatch(0)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen",
    [
        "",
        "garbage",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",   # bad side to move
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq zz 0 1",  # bad ep square
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e9 0 1",  # ep rank out of range
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Kx - 0 1",     # bad castling letter
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1",            # seven ranks
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/8 w KQkq - 0 1",  # nine ranks
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNRR w KQkq - 0 1",  # nine files on a rank
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN w KQkq - 0 1",    # seven files on a rank
        "rnbqkbnr/pppppppp/44/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # two adjacent digits
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBXR w KQkq - 0 1",   # bad piece letter
        "~nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",   # '~' not after a piece
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 x",  # a seventh field
    ],
)
def test_unusable_fen_raises_value_error(fen):
    with pytest.raises(ValueError):
        guofish_core.tokens(fen)


def test_a_rejected_fen_does_not_poison_the_next_call():
    with pytest.raises(ValueError):
        guofish_core.tokens("garbage")

    assert guofish_core.tokens("4k3/8/8/8/8/8/8/4K3 w - - 0 1")[IDX_CLS] == TOKEN_CLS


def test_a_rejected_fen_inside_a_batch_raises():
    batch = guofish_core.TokenBatch(4)
    view = batch.view()
    view[:] = -1

    with pytest.raises(ValueError):
        batch.fill(["4k3/8/8/8/8/8/8/4K3 w - - 0 1", "garbage"])

    # Documented behaviour, not an accident: rows written before the bad FEN
    # keep their new contents and the rest keep their old ones. Asserted so a
    # change to it is a deliberate one.
    assert view[0, IDX_CLS] == TOKEN_CLS
    assert (view[1] == -1).all()

    # And the batch is still usable afterwards.
    assert batch.fill(["4k3/8/8/8/8/8/8/4K3 b - - 0 1"], 3) == 1
    assert view[3, IDX_SIDE_TO_MOVE] == TOKEN_BLACK_TO_MOVE


def test_a_fen_missing_its_trailing_fields_is_accepted():
    """python-chess defaults everything after the placement, so this does too.

    `tokens()` and `legal_moves()` are fed from the same call sites; two
    different ideas of what parses would be a bug that shows up in only one.
    """
    bare = guofish_core.tokens("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")

    assert bare[IDX_SIDE_TO_MOVE] == TOKEN_WHITE_TO_MOVE
    assert bare[IDX_CASTLING] == TOKEN_CASTLING_BASE
    assert bare[IDX_EN_PASSANT] == TOKEN_EP_NONE
    assert bare[IDX_CLS] == TOKEN_CLS


def test_kingless_positions_are_encoded_rather_than_refused():
    """Unlike `legal_moves`, which cannot answer without a king.

    python-chess encodes a kingless board without complaint — it has no castling
    rights and nothing else about the encoding depends on a king — so refusing
    here would be a divergence, not a safety check.
    """
    tokens = guofish_core.tokens("8/8/8/8/8/8/8/8 w - - 0 1")

    assert (tokens[:64] == 0).all()
    assert tokens[IDX_CASTLING] == TOKEN_CASTLING_BASE
    assert tokens[IDX_CLS] == TOKEN_CLS


# ---------------------------------------------------------------------------
# The diagnostic itself
#
# The brief requires that a corrupted golden value produces the FEN, both
# arrays and every differing index. These exercise the formatter directly, so
# the requirement is covered by the suite and not only by a one-off manual
# drill. See DECISIONS.md, "C2 / mutation check".
# ---------------------------------------------------------------------------


def reference_row():
    row = np.zeros(SEQ_LENGTH, dtype=np.int32)
    row[IDX_SIDE_TO_MOVE] = TOKEN_WHITE_TO_MOVE
    row[IDX_CASTLING] = TOKEN_CASTLING_BASE
    row[IDX_EN_PASSANT] = TOKEN_EP_NONE
    row[IDX_CLS] = TOKEN_CLS
    return row


def test_diagnostic_reports_every_differing_index():
    fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    ref = reference_row()
    ref[4] = 6
    ref[60] = 12

    cpp = ref.copy()
    cpp[4] = 0        # a piece that went missing
    cpp[IDX_CLS] = 0  # and a corrupted CLS

    report = format_mismatch("tokens.npz", 17, fen, cpp, ref)

    assert "tokens.npz[17]" in report
    assert fen in report
    assert "2 differing index(es)" in report
    assert "[4] square e1" in report
    assert "[67] CLS" in report
    assert "expected   6  got   0" in report
    assert "expected  40  got   0" in report
    # Both arrays in full, not NumPy's elided repr.
    assert report.count("...") == 0
    assert str(ref.tolist()) in report
    assert str(cpp.tolist()) in report


def test_diagnostic_names_the_en_passant_trap():
    ref = reference_row()
    ref[IDX_EN_PASSANT] = TOKEN_EP_BASE + 3
    cpp = ref.copy()
    cpp[IDX_EN_PASSANT] = TOKEN_EP_NONE

    report = format_mismatch("tokens.npz", 0, "fen-here", cpp, ref)

    assert "[66] en-passant file" in report
    assert "EN PASSANT DISCARDED" in report
    assert "setFen()" in report


def test_diagnostic_spells_out_the_castling_mask():
    ref = reference_row()
    ref[IDX_CASTLING] = TOKEN_CASTLING_BASE + 0b1000   # White kingside only
    cpp = ref.copy()
    cpp[IDX_CASTLING] = TOKEN_CASTLING_BASE + 0b0001   # Black queenside only — a mirrored mask

    report = format_mismatch("tokens.npz", 0, "fen-here", cpp, ref)

    assert "[65] castling rights" in report
    assert "expected 1000, got 0001" in report
    assert "WK WQ BK BQ" in report


def test_diagnostic_reports_a_wrong_shape():
    ref = reference_row()
    report = format_mismatch("tokens.npz", 0, "fen-here", np.zeros(4, dtype=np.int32), ref)

    assert "SHAPE" in report
    assert "(4,)" in report and "(68,)" in report


def test_diagnostic_index_labels_cover_the_board():
    assert describe_index(0) == "[0] square a1"
    assert describe_index(7) == "[7] square h1"
    assert describe_index(56) == "[56] square a8"
    assert describe_index(63) == "[63] square h8"
    assert describe_index(64) == "[64] side to move"
    assert describe_index(66) == "[66] en-passant file"
