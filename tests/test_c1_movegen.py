"""C1 acceptance tests — legal move generation parity with python-chess.

`golden/movegen.jsonl` is the reference (Global Rule 2): it was produced by
`tools/gen_movegen_golden.py` from python-chess, and it is never regenerated to
agree with the C++ side. If the two disagree, the C++ side is wrong.

The suite has three layers, because "the lists are equal" on its own is a weak
statement about a movegen:

  * **Parity** streams every record in the golden file and compares
    ``guofish_core.legal_moves(fen)`` to the reference list *including order*.
  * **Corpus coverage** asserts the golden file actually contains the cases the
    parity test would otherwise pass vacuously — castling, promotions, en
    passant, terminal positions. A parity run over 100k quiet middlegames would
    be green with the castling normalisation deleted.
  * **Targeted** cases pin the specific behaviours the C1 brief names, with
    hand-auditable expected lists rather than a comparison against a library.
    Every literal below was cross-checked against python-chess 1.11.2 and is
    short enough to verify by hand from the FEN.

Nothing here imports `chess`. The reference's answers reach this file only
through the golden data, so the tests cannot drift into re-deriving the
expectation from the same library that produced it.

Pointing the parity test at a different file
--------------------------------------------
``GUOFISH_GOLDEN_MOVEGEN=/path/to/other.jsonl`` overrides the input. This exists
so the diagnostic output can be exercised against a deliberately corrupted copy
without writing to ``golden/`` (Global Rules 1 and 2); see DECISIONS.md, "C1 /
mutation check". The corpus-coverage tests skip when the override is set.
"""

import json
import os
from pathlib import Path

import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = REPO_ROOT / "golden" / "movegen.jsonl"
GOLDEN_ENV = "GUOFISH_GOLDEN_MOVEGEN"

# How many mismatching positions are reported in full before the report is
# truncated. A systematically wrong normalisation produces tens of thousands of
# them and the first few are enough to name the bug.
MAX_REPORTED = 10

# A record whose FEN's 4th field is not "-" carries an en-passant target square.
# `tools/gen_movegen_golden.py` writes the square whenever a double pawn push
# just happened (FEN spec semantics), not only when a capture is legal, so this
# counts the en-passant *traps* as well as the playable captures.
EP_FIELD = 3

# What chess-library would emit if the normalisation in cpp/movegen.hpp were
# removed: king-takes-rook.
#
# A *hint* for the failure report, not a proof. From the king's own square these
# strings can only be a de-normalised castle, but the same string can be a
# perfectly ordinary rook or queen move made by whoever else stands on e1/e8 —
# a black queen on e1 really can play e1a1. The report says which it looks like;
# the test that decides is the exact list comparison.
KING_TAKES_ROOK_UCI = frozenset({"e1h1", "e1a1", "e8h8", "e8a8"})


def king_square_of_side_to_move(fen):
    """The side-to-move's king square as a UCI square name, or None."""
    fields = fen.split()
    target = "K" if fields[1] == "w" else "k"

    for rank_index, row in enumerate(fields[0].split("/")):
        file_index = 0
        for char in row:
            if char.isdigit():
                file_index += int(char)
            else:
                if char == target:
                    return chr(ord("a") + file_index) + str(8 - rank_index)
                file_index += 1
    return None


def has_legal_castling(fen, moves):
    """True when `moves` contains a castle for the side to move.

    Matching on the literal strings e1g1/e1c1/e8g8/e8c8 is not good enough: the
    e-file back-rank square is not always the king's, so those are also legal
    rook and queen moves in plenty of positions. A castle is the only *king*
    move that crosses two files on its own rank, so the king square is looked up
    from the FEN and the test is anchored there.
    """
    if fen.split()[2] == "-":
        return False

    king = king_square_of_side_to_move(fen)
    if king is None:
        return False

    return any(
        len(move) == 4
        and move[:2] == king
        and move[3] == king[1]
        and abs(ord(move[2]) - ord(king[0])) == 2
        for move in moves
    )


def golden_path():
    override = os.environ.get(GOLDEN_ENV)
    return Path(override) if override else DEFAULT_GOLDEN


def using_default_golden():
    return not os.environ.get(GOLDEN_ENV)


def iter_golden(path):
    """Yield (lineno, fen, moves). Streamed: the full file is ~27 MB of JSON."""
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            yield lineno, record["fen"], record["moves"]


# ---------------------------------------------------------------------------
# Failure reporting
#
# Module level and dependency-free so that the diagnostic can be unit tested
# directly (test_diagnostic_* below) instead of only being seen when something
# is already broken.
# ---------------------------------------------------------------------------


def format_mismatch(source, lineno, fen, cpp, ref):
    """A full account of one disagreeing position.

    Reports the FEN, both lists, and the exact set difference in both
    directions. When the sets agree the lists differ only in order, which is a
    different bug with a different fix, so it is called out separately along
    with the first index at which they diverge.
    """
    missing = sorted(set(ref) - set(cpp))
    extra = sorted(set(cpp) - set(ref))

    lines = [
        f"{source}:{lineno}",
        f"  FEN                  : {fen}",
        f"  C++   ({len(cpp):3d} moves)   : {cpp}",
        f"  python ({len(ref):3d} moves)  : {ref}",
        f"  missing from C++     : {missing}",
        f"  extra in C++         : {extra}",
    ]

    duplicates = sorted({m for m in cpp if cpp.count(m) > 1})
    if duplicates:
        lines.append(f"  DUPLICATED in C++    : {duplicates}")

    if not missing and not extra:
        first = next((i for i, (a, b) in enumerate(zip(cpp, ref)) if a != b), None)
        lines.append(
            "  ORDER ONLY           : same set of moves, different order"
            + (f"; first difference at index {first}: C++ {cpp[first]!r} vs python {ref[first]!r}" if first is not None else "")
        )

    banned = sorted(set(cpp) & KING_TAKES_ROOK_UCI)
    if banned:
        lines.append(
            f"  CASTLING NOT NORMALISED: {banned} is chess-library's king-takes-rook "
            "encoding; standard UCI wants e1g1/e1c1/e8g8/e8c8"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# One pass over the corpus, shared by every test that needs it
# ---------------------------------------------------------------------------


class Scan:
    """Results of a single streaming pass over the golden file."""

    def __init__(self):
        self.path = None
        self.records = 0
        self.reports = []       # formatted diagnostics, capped at MAX_REPORTED
        self.mismatches = 0
        self.unsorted = []      # positions where the C++ list is not in canonical order
        self.duplicated = []    # positions where the C++ list repeats a move
        # Corpus coverage, counted from the *reference* lists.
        self.with_castling = 0
        self.with_promotion = 0
        self.with_ep_square = 0
        self.terminal = 0
        self.total_moves = 0


@pytest.fixture(scope="module")
def scan():
    path = golden_path()
    if not path.exists():
        pytest.fail(
            f"golden data missing: {path}\n"
            "It is generated by tools/gen_movegen_golden.py from python-chess and is "
            "not produced by this chunk (Global Rule 2)."
        )

    result = Scan()
    result.path = path

    for lineno, fen, ref in iter_golden(path):
        result.records += 1
        result.total_moves += len(ref)

        if not ref:
            result.terminal += 1
        if any(len(m) == 5 for m in ref):
            result.with_promotion += 1
        if has_legal_castling(fen, ref):
            result.with_castling += 1
        if fen.split()[EP_FIELD] != "-":
            result.with_ep_square += 1

        cpp = guofish_core.legal_moves(fen)

        if cpp != ref:
            result.mismatches += 1
            if len(result.reports) < MAX_REPORTED:
                result.reports.append(format_mismatch(path, lineno, fen, cpp, ref))

        # Checked independently of the comparison above: if the golden file were
        # ever wrong in the same way, parity would still pass and these would not.
        if cpp != sorted(cpp) and len(result.unsorted) < MAX_REPORTED:
            result.unsorted.append(f"{path}:{lineno}  {fen}\n    {cpp}")
        if len(set(cpp)) != len(cpp) and len(result.duplicated) < MAX_REPORTED:
            result.duplicated.append(f"{path}:{lineno}  {fen}\n    {cpp}")

    return result


# ---------------------------------------------------------------------------
# Parity — the acceptance criterion
# ---------------------------------------------------------------------------


def test_movegen_matches_golden(scan):
    """Every position: the C++ move list equals the reference, order included."""
    assert scan.records > 0, f"{scan.path} contained no records"

    if scan.mismatches:
        shown = len(scan.reports)
        truncated = "" if shown == scan.mismatches else f" (showing the first {shown})"
        pytest.fail(
            f"{scan.mismatches} of {scan.records} positions disagree with "
            f"{scan.path}{truncated}:\n\n" + "\n\n".join(scan.reports),
            pytrace=False,
        )


def test_output_is_in_canonical_order(scan):
    """Canonical order is byte-wise lexicographic UCI order, corpus-wide.

    Independent of the golden comparison on purpose. PUCT resolves ties by child
    order, so an ordering that merely happens to match the reference file is not
    the same guarantee as an ordering that is sorted by construction.
    """
    assert not scan.unsorted, (
        f"{len(scan.unsorted)} position(s) returned moves out of canonical order:\n"
        + "\n".join(scan.unsorted)
    )


def test_output_has_no_duplicate_moves(scan):
    """Two internal moves must never format to the same UCI string."""
    assert not scan.duplicated, (
        f"{len(scan.duplicated)} position(s) returned a duplicated move:\n"
        + "\n".join(scan.duplicated)
    )


# ---------------------------------------------------------------------------
# Corpus coverage — does the parity run actually exercise the traps?
# ---------------------------------------------------------------------------

requires_default_golden = pytest.mark.skipif(
    not using_default_golden(),
    reason=f"{GOLDEN_ENV} overrides the corpus; coverage claims apply to golden/movegen.jsonl only",
)


@requires_default_golden
def test_golden_corpus_is_not_truncated(scan):
    """Guards against a green run over a stub file."""
    assert scan.records >= 50_000, (
        f"{scan.path} holds {scan.records} positions; the C1 corpus is ~100k. "
        "A truncated file would let parity pass without covering the corpus."
    )
    assert scan.total_moves > 0


@requires_default_golden
@pytest.mark.parametrize(
    "attribute,label,minimum",
    [
        ("with_castling", "positions where castling is legal", 1_000),
        ("with_promotion", "positions where a promotion is legal", 1_000),
        ("with_ep_square", "positions carrying an en-passant target square", 1_000),
        ("terminal", "terminal positions (no legal move)", 1),
    ],
)
def test_golden_corpus_covers_the_edge_cases(scan, attribute, label, minimum):
    """Each trap the C1 brief names must be represented in the parity run.

    Without this, deleting the castling normalisation could still produce a
    green suite on a corpus that happened to contain no castling positions.
    """
    count = getattr(scan, attribute)
    assert count >= minimum, (
        f"only {count} {label} in {scan.path} (expected at least {minimum}); "
        "the parity run is not exercising this case"
    )


# ---------------------------------------------------------------------------
# Targeted cases
#
# Expected lists are literals, cross-checked against python-chess 1.11.2 and
# short enough to audit against the FEN by hand.
# ---------------------------------------------------------------------------


def test_castling_is_normalised_to_the_kings_destination():
    """White with both rights: e1g1 and e1c1, never e1h1 / e1a1."""
    moves = guofish_core.legal_moves("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")

    assert moves == [
        "a1b1", "a1c1", "a1d1", "a2a3", "a2a4", "b2b3", "b2b4", "c2c3", "c2c4",
        "d2d3", "d2d4", "e1c1", "e1d1", "e1f1", "e1g1", "e2e3", "e2e4", "f2f3",
        "f2f4", "g2g3", "g2g4", "h1f1", "h1g1", "h2h3", "h2h4",
    ]
    assert not KING_TAKES_ROOK_UCI.intersection(moves)


def test_castling_is_normalised_for_black():
    moves = guofish_core.legal_moves("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1")

    assert moves == [
        "a7a5", "a7a6", "a8b8", "a8c8", "a8d8", "b7b5", "b7b6", "c7c5", "c7c6",
        "d7d5", "d7d6", "e7e5", "e7e6", "e8c8", "e8d8", "e8f8", "e8g8", "f7f5",
        "f7f6", "g7g5", "g7g6", "h7h5", "h7h6", "h8f8", "h8g8",
    ]
    assert not KING_TAKES_ROOK_UCI.intersection(moves)


def test_castling_through_an_attacked_square_is_illegal():
    """The g2 bishop covers f1, so O-O is out while O-O-O stands."""
    moves = guofish_core.legal_moves("r3k2r/8/8/8/8/8/6b1/R3K2R w KQkq - 0 1")

    assert moves == [
        "a1a2", "a1a3", "a1a4", "a1a5", "a1a6", "a1a7", "a1a8", "a1b1", "a1c1",
        "a1d1", "e1c1", "e1d1", "e1d2", "e1e2", "e1f2", "h1f1", "h1g1", "h1h2",
        "h1h3", "h1h4", "h1h5", "h1h6", "h1h7", "h1h8",
    ]
    assert "e1g1" not in moves
    assert "e1c1" in moves


def test_promotions_emit_all_four_pieces():
    """Both the push and the capture promote to q, r, b and n."""
    moves = guofish_core.legal_moves("1n6/P6k/8/8/8/8/7K/8 w - - 0 1")

    assert moves == [
        "a7a8b", "a7a8n", "a7a8q", "a7a8r",
        "a7b8b", "a7b8n", "a7b8q", "a7b8r",
        "h2g1", "h2g2", "h2g3", "h2h1", "h2h3",
    ]


def test_en_passant_capture_is_generated():
    moves = guofish_core.legal_moves("8/8/8/2Pp4/8/8/K6k/8 w - d6 0 1")

    assert moves == ["a2a1", "a2a3", "a2b1", "a2b2", "a2b3", "c5c6", "c5d6"]


def test_en_passant_is_refused_when_it_exposes_the_king():
    """The horizontal pin: cxd6 e.p. clears rank 5 and Ka5 meets Rh5.

    The FEN still carries the ep square, which is the only reason this is a
    test — see the note on `en_passant="fen"` in tools/gen_movegen_golden.py.
    """
    moves = guofish_core.legal_moves("8/8/8/K1Pp3r/8/8/8/7k w - d6 0 1")

    assert moves == ["a5a4", "a5a6", "a5b4", "a5b5", "a5b6", "c5c6"]
    assert "c5d6" not in moves


def test_checkmate_has_no_legal_moves():
    assert guofish_core.legal_moves("7k/5KQ1/8/8/8/8/8/8 b - - 0 1") == []


def test_stalemate_has_no_legal_moves():
    assert guofish_core.legal_moves("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1") == []


def test_startpos():
    moves = guofish_core.legal_moves(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )

    assert len(moves) == 20
    assert moves == sorted(moves)
    assert moves[:4] == ["a2a3", "a2a4", "b1a3", "b1c3"]


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen",
    [
        "",
        "garbage",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",  # bad side to move
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq zz 0 1",  # bad ep square
        # Kingless and one-king positions have to be refused *before* the FEN
        # reaches chess-library: its parser calls kingSq() while building the
        # castling paths, which asserts on an empty king bitboard. Under the
        # ASan/debug build a missing guard here aborts the interpreter rather
        # than raising. See DECISIONS.md, "C1 / rejecting a FEN early enough".
        "8/8/8/8/8/8/8/8 w - - 0 1",  # no kings at all
        "8/8/8/8/8/8/8/K7 w - - 0 1",  # White only
        "7k/8/8/8/8/8/8/8 w - - 0 1",  # Black only
        "KK6/8/8/8/8/8/8/7k w - - 0 1",  # two White kings
        "1k6/8/8/8/8/8/8/K6k w - - 0 1",  # two Black kings
    ],
)
def test_unusable_fen_raises_value_error(fen):
    with pytest.raises(ValueError):
        guofish_core.legal_moves(fen)


def test_a_rejected_fen_does_not_poison_the_next_call():
    with pytest.raises(ValueError):
        guofish_core.legal_moves("garbage")

    assert guofish_core.legal_moves("7k/5KQ1/8/8/8/8/8/8 b - - 0 1") == []
    assert len(guofish_core.legal_moves("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")) == 20


# ---------------------------------------------------------------------------
# The diagnostic itself
#
# The brief requires that a corrupted golden value produces a set difference,
# not a bare "lists differ". These exercise that formatter directly, so the
# requirement is covered by the suite and not only by a one-off manual drill.
# ---------------------------------------------------------------------------


def test_castling_detector_is_anchored_on_the_king_square():
    """e1g1 is not proof of a castle, and the coverage counter must know that.

    Both positions below are real: the first has a Black queen on e1 and the
    second a White rook there, and both generate all four of the strings a
    naive string match would read as castling. If `has_legal_castling` counted
    those, `test_golden_corpus_covers_the_edge_cases` would report castling
    coverage the parity run does not actually have.
    """
    lookalikes = {"e1g1", "e1c1", "e1a1", "e1h1"}

    # Black queen on e1, Black king on c4, no castling rights for anyone.
    fen = "1N6/2p2K2/8/5p1P/2k2b2/2P5/8/4q3 b - - 1 49"
    moves = guofish_core.legal_moves(fen)
    assert lookalikes.issubset(moves)
    assert king_square_of_side_to_move(fen) == "c4"
    assert not has_legal_castling(fen, moves)

    # White rook on e1, White king on d4 — and Black still holds "kq", so the
    # empty-castling-field shortcut is not what rejects this one.
    fen = "r3k2r/8/8/8/3KP3/8/8/4R3 w kq - 0 1"
    moves = guofish_core.legal_moves(fen)
    assert lookalikes.issubset(moves)
    assert king_square_of_side_to_move(fen) == "d4"
    assert not has_legal_castling(fen, moves)


def test_castling_detector_finds_a_real_castle():
    for fen, king in [
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1"),
        ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", "e8"),
    ]:
        moves = guofish_core.legal_moves(fen)
        assert king_square_of_side_to_move(fen) == king
        assert has_legal_castling(fen, moves)

    # Rights on paper, but the g2 bishop stops O-O and O-O-O is still there,
    # so this must still count.
    fen = "r3k2r/8/8/8/8/8/6b1/R3K2R w KQkq - 0 1"
    assert has_legal_castling(fen, guofish_core.legal_moves(fen))

    # Rights present, no castle available: the king has already been checked
    # off the back rank in the second FEN above's White half.
    fen = "r3k2r/8/8/8/8/8/8/R2K3R w kq - 0 1"
    moves = guofish_core.legal_moves(fen)
    assert king_square_of_side_to_move(fen) == "d1"
    assert not has_legal_castling(fen, moves)


def test_diagnostic_reports_the_set_difference():
    fen = "8/8/8/2Pp4/8/8/K6k/8 w - d6 0 1"
    cpp = ["a2a1", "a2a3", "c5c6", "c5d6"]
    ref = ["a2a1", "a2a3", "a2b1", "c5c6"]

    report = format_mismatch("golden/movegen.jsonl", 42, fen, cpp, ref)

    assert "golden/movegen.jsonl:42" in report
    assert fen in report
    assert "'c5d6'" in report and "'a2b1'" in report
    assert "missing from C++     : ['a2b1']" in report
    assert "extra in C++         : ['c5d6']" in report


def test_diagnostic_distinguishes_an_ordering_bug():
    cpp = ["a2a3", "a2a1"]
    ref = ["a2a1", "a2a3"]

    report = format_mismatch("golden/movegen.jsonl", 7, "fen-here", cpp, ref)

    assert "missing from C++     : []" in report
    assert "extra in C++         : []" in report
    assert "ORDER ONLY" in report
    assert "first difference at index 0" in report


def test_diagnostic_names_the_castling_encoding():
    cpp = ["e1h1"]
    ref = ["e1g1"]

    report = format_mismatch("golden/movegen.jsonl", 1, "fen-here", cpp, ref)

    assert "CASTLING NOT NORMALISED" in report
    assert "e1h1" in report


def test_diagnostic_reports_duplicates():
    cpp = ["a2a3", "a2a3"]
    ref = ["a2a3"]

    report = format_mismatch("golden/movegen.jsonl", 1, "fen-here", cpp, ref)

    assert "DUPLICATED in C++    : ['a2a3']" in report
