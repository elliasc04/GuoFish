"""C5 pin tests — chess-library's en-passant behaviour, and c_puct's log.

Neither of these tests a thing C5 *implements*. Both pin an assumption C5 rests
on, so that if the assumption stops holding, a named test fails instead of a
5,000-node bit-exactness comparison diverging at some arbitrary depth with no
indication of why.

-------------------------------------------------------------------------------
1. The en-passant pin
-------------------------------------------------------------------------------
Token 66 — and therefore `nn_key`, and therefore every replay-dump lookup the
search performs — is written from python-chess's `board.ep_square`, which is set
after ANY double pawn push with no test for whether a capture is available. Call
that the RAW rule. It is the trained contract (C2) and it is not negotiable.

chess-library does not implement that rule anywhere. At the pinned revision its
`makeMove` reads:

    if (Square::value_distance(move.to(), move.from()) == 16) {
        Bitboard ep_mask = attacks::pawn(stm_, move.to().ep_square());
        if (ep_mask & pieces(PAWN, ~stm_)) {              // pseudo-legal adjacency
            if constexpr (EXACT) { ... isEpSquareValid ... }   // legal capture
            if (found != 0) ep_sq_ = move.to().ep_square();
        }
    }

so `makeMove<false>` (the default) uses pseudo-legal adjacency and
`makeMove<true>` uses the legal rule. **Neither is the raw rule**, and `setFen`
applies the legal rule again on the way in. Four conventions, three of them
wrong for this purpose, and choosing any of them costs nothing at runtime and
corrupts every key.

The tests below assert what each convention answers on positions constructed to
separate them. They are written to fail on an upstream change rather than to
document a preference: if a future re-pin makes `makeMove<false>` set ep
unconditionally, `test_default_makemove_is_pseudo_legal_adjacency` fails, and
whoever did the re-pin reads this docstring instead of debugging Gate 1.

`unmakeMove` restoring ep exactly is pinned too. The search relies on it — its
own raw-ep stack is independent, but the library's board still has to come back
to the same state or movegen diverges on the way back up the tree.

-------------------------------------------------------------------------------
2. The c_puct pin
-------------------------------------------------------------------------------
The brief requires `c_puct(n)` to be exposed from C++ so a harness can put the
same log on both sides of the equivalence comparison, rather than discovering
that CPython's `math.log` and this translation unit's `std::log` disagree in the
last ulp.

That exposure exists (`guofish_core.c_puct`). What this test establishes is
whether the difference it guards against is real *on this toolchain*, because
the answer decides something about Global Rule 2: if the two logs agree
bit-for-bit, the golden generator can stay pure Python and the golden trees owe
nothing to the implementation under test. As of C5 they do agree — both CPython
and the MSVC build resolve `log` through the same UCRT — so the generator was
left alone. If this test ever fails, that reasoning is void and
tools/gen_gate1_golden.py has to route c_puct through this binding.

Nothing here reads a golden file or imports `chess`.
"""

import math

import pytest

import guofish_core

# The revision cpp/search.hpp's en-passant reasoning was written against, and
# the one CMakeLists.txt pins. Read out of the build rather than restated, so
# this cannot pass against a build of some other revision.
PINNED_REVISION = "53e6a841dcda7059a2af363d85f785ef1817304a"


# ---------------------------------------------------------------------------
# Positions
#
# Each is one double pawn push, chosen so that the four conventions separate.
# ---------------------------------------------------------------------------

# No black pawn anywhere near e4: nothing can ever capture en passant.
# raw = e3, adjacency = none, legal = none.
LONELY = ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "e2e4", "e3")

# A black pawn on d4 sits beside the pushed pawn and the capture is legal.
# All four conventions agree here; this is the control.
CAPTURABLE = ("4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1", "e2e4", "e3")

# A black pawn on d4 is adjacent, and the capture is pseudo-legal, but playing
# it would empty the whole of rank 4 between the black king on a4 and the white
# rook on h4 — the ep capture removes BOTH pawns from that rank — so it is
# illegal. raw = e3, adjacency = e3, legal = none.
#
# Note whose king matters: the capture's legality is judged against the
# CAPTURER's king, so the skewer has to be aimed at Black. Aiming it at White
# (the obvious first draft) leaves the capture perfectly legal and the position
# fails to separate the two conventions at all.
PINNED_CAPTURER = ("8/8/8/8/k2p3R/8/4P3/4K3 w - - 0 1", "e2e4", "e3")

# A position whose FEN already CARRIES an ep square with no legal capture, so
# the setFen filter can be observed without making a move first. The move is
# just something legal to satisfy the probe's signature.
EP_ALREADY_IN_FEN = ("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1", "e8d8", "-")


def _probe(spec):
    fen, uci, _ = spec
    return guofish_core.ep_pin_probe(fen, uci)


# ---------------------------------------------------------------------------
# The pin
# ---------------------------------------------------------------------------


def test_build_is_the_pinned_chess_library_revision():
    """If this fails, every other assertion in this file is about a different
    library and the C5 en-passant reasoning has to be re-derived."""
    assert guofish_core.CHESS_LIBRARY_PIN == PINNED_REVISION, (
        f"chess-library is pinned to {guofish_core.CHESS_LIBRARY_PIN}, but "
        f"cpp/search.hpp's en-passant reasoning was written against "
        f"{PINNED_REVISION}. Re-read this file's docstring before changing the "
        f"expected values below."
    )


@pytest.mark.parametrize("spec", [LONELY, CAPTURABLE, PINNED_CAPTURER],
                         ids=["lonely", "capturable", "pinned_capturer"])
def test_search_state_carries_the_raw_en_passant_square(spec):
    """THE REQUIREMENT. A double push always sets the ep square, unconditionally.

    This is python-chess's rule by construction and it is what token 66 needs.
    All three positions must answer the same way, which is the whole point:
    the raw rule does not look at the opponent's pawns at all.
    """
    fen, uci, expected = spec
    probe = guofish_core.ep_pin_probe(fen, uci)
    assert probe["raw_ep"] == expected, (
        f"the search derived ep {probe['raw_ep']} after {uci}, expected "
        f"{expected}. Token 66 and every nn_key downstream are now wrong."
    )


def test_default_makemove_is_pseudo_legal_adjacency():
    """`makeMove<false>` sets ep only when an enemy pawn stands beside the push.

    Pinned because it is the DEFAULT: a search that called makeMove and then
    read enpassantSq() would silently take this rule, and it is wrong for
    token 66 on the majority of double pushes.
    """
    assert _probe(LONELY)["library_ep_default"] == "-", (
        "makeMove<false> now sets an en-passant square with no enemy pawn "
        "adjacent. The library's convention changed; see this file's docstring."
    )
    assert _probe(CAPTURABLE)["library_ep_default"] == "e3"
    # Adjacent but pinned: the pseudo-legal rule does not care about legality.
    assert _probe(PINNED_CAPTURER)["library_ep_default"] == "e3", (
        "makeMove<false> now applies a legality test. It used to apply only "
        "pseudo-legal adjacency; see this file's docstring."
    )


def test_exact_makemove_is_the_legal_capture_rule():
    """`makeMove<true>` additionally requires the capture to be legal.

    This is the convention that looks most 'correct' and is still not the raw
    rule — it differs from it on both LONELY and PINNED_CAPTURER.
    """
    assert _probe(LONELY)["library_ep_exact"] == "-"
    assert _probe(CAPTURABLE)["library_ep_exact"] == "e3"
    assert _probe(PINNED_CAPTURER)["library_ep_exact"] == "-", (
        "makeMove<true> no longer filters an en-passant capture that would "
        "leave the mover's king in check; see this file's docstring."
    )


def test_the_three_library_conventions_actually_differ():
    """The reason cpp/search.hpp carries its own state, stated as one assertion.

    If these ever collapse into agreement the search could stop carrying its own
    ep square — but nothing should *assume* they have, so this fails loudly on
    the day it happens rather than leaving dead complexity unexplained.
    """
    lonely = _probe(LONELY)
    pinned = _probe(PINNED_CAPTURER)

    # raw vs pseudo-legal adjacency
    assert lonely["raw_ep"] != lonely["library_ep_default"]
    # pseudo-legal adjacency vs legal capture
    assert pinned["library_ep_default"] != pinned["library_ep_exact"]
    # raw vs legal capture
    assert pinned["raw_ep"] != pinned["library_ep_exact"]


def test_setfen_filters_the_en_passant_square_out_of_the_board():
    """`setFen` re-applies the legal rule, which is why Board::fen()/setFen() is
    a prohibited round trip for search state.

    The FEN says e3; chess-library's board does not, because no capture is
    available. A search that round-tripped a position through the library would
    lose the square, and with it token 66.
    """
    probe = _probe(EP_ALREADY_IN_FEN)
    assert probe["fen_ep"] == "e3"
    assert probe["library_ep_after_setfen"] == "-", (
        "setFen no longer filters the en-passant square. If it now keeps it "
        "verbatim, cpp/tokens.hpp's reason for parsing FENs directly has "
        "changed and DECISIONS.md's C2 entry needs revisiting."
    )
    # And the thing that makes the C5 root correct: the search's own state kept
    # the square the library discarded, so the ROOT position tokenizes the way
    # Python tokenized it. Without this every run's very first dump lookup
    # would miss on any position whose FEN carries an unplayable ep square —
    # 3,203 of the 3,610 ep positions in the C1 corpus.
    assert probe["raw_ep_before"] == "e3"
    assert probe["token_66"] == 31, (
        "after 1...Kd8 there is no ep square, so token 66 must be TOKEN_EP_NONE"
    )


@pytest.mark.parametrize("spec", [LONELY, CAPTURABLE, PINNED_CAPTURER],
                         ids=["lonely", "capturable", "pinned_capturer"])
def test_unmake_restores_both_en_passant_states(spec):
    """`unmakeMove` restores the library's ep square, and the search restores its
    own. The search descends and unwinds thousands of times per search; a leak in
    either direction would corrupt every position after the first."""
    fen, uci, _ = spec
    probe = guofish_core.ep_pin_probe(fen, uci)
    assert probe["library_ep_unmade"] == probe["library_ep_after_setfen"]
    assert probe["raw_ep_unmade"] == probe["raw_ep_before"]


def test_token_66_follows_the_raw_square():
    """The end of the chain: raw ep square -> token 66 -> nn_key.

    TOKEN_EP_BASE is 32 and the file is added, so e3 is 32 + 4 = 36 and 'no ep'
    is 31. Asserting the token rather than only the square is what connects this
    file to the thing that actually breaks — a replay-dump miss.
    """
    assert _probe(LONELY)["token_66"] == 36
    assert _probe(CAPTURABLE)["token_66"] == 36
    assert _probe(PINNED_CAPTURER)["token_66"] == 36

    # And two positions differing ONLY in whether the push happened must not
    # share a key, which is the property the dump lookup depends on.
    pushed = _probe(LONELY)["nn_key"]
    quiet = guofish_core.nn_key("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1")
    assert pushed != quiet


# ---------------------------------------------------------------------------
# c_puct
# ---------------------------------------------------------------------------


def _python_c_puct(parent_visits, c_init=1.43, c_base=19652.0):
    """SearchParams.c_puct, transcribed, association included.

    Python evaluates `parent_visits + self.c_base + 1.0` left to right, and
    floating-point addition is not associative, so the grouping is part of the
    reference and not a detail.
    """
    return c_init + math.log((parent_visits + c_base + 1.0) / c_base)


def test_c_puct_matches_python_bit_for_bit():
    """The pin that keeps Global Rule 2 clean.

    Swept over the whole range a 5,000-simulation search reaches at the parent,
    including the fractional values virtual loss produces (a parent mid-descent
    carries N + 1*2.5 effective visits, so the argument is not an integer).

    If this fails, the golden generator must be changed to call
    guofish_core.c_puct, and DECISIONS.md's claim that the golden trees owe
    nothing to the implementation under test stops being true.
    """
    mismatches = []
    for n in range(0, 20001):
        for offset in (0.0, 2.5):
            parent_visits = n + offset
            ours = guofish_core.c_puct(parent_visits)
            theirs = _python_c_puct(parent_visits)
            if ours != theirs:
                mismatches.append((parent_visits, ours, theirs))
                if len(mismatches) >= 5:
                    break
        if len(mismatches) >= 5:
            break

    assert not mismatches, (
        "C++ std::log and CPython math.log disagree on this toolchain:\n" +
        "\n".join(f"  N={n!r}: c++={a!r} python={b!r} (delta {a - b:.3e})"
                  for n, a, b in mismatches) +
        "\nThe golden generator must now route c_puct through guofish_core."
    )


def test_c_puct_is_the_reference_formula_not_a_lookalike():
    """Guards the two ways this could be subtly wrong: a dropped +1, or the log
    base. Both would pass a smoke test at one value and fail Gate 1 everywhere."""
    # At N = 0 the log term is log((c_base + 1) / c_base), which is small but
    # not zero. A dropped +1 would make it exactly zero.
    assert guofish_core.c_puct(0.0) != 1.43
    assert guofish_core.c_puct(0.0) == pytest.approx(1.43 + math.log(19653.0 / 19652.0))

    # The documented growth: +0.04 at N=800, +0.57 at N=15000.
    assert guofish_core.c_puct(800.0) - 1.43 == pytest.approx(0.04, abs=0.005)
    assert guofish_core.c_puct(15000.0) - 1.43 == pytest.approx(0.57, abs=0.01)


def test_c_factor_defaults_to_a_true_no_op():
    """`c_factor` does not exist in the reference. Scope 3 adds it as a tunable
    defaulting to 1.0, and the requirement is that the default cannot change the
    base math — so the default instance must agree with the free function
    bit-for-bit, not approximately."""
    default = guofish_core.ReplaySearchDouble(
        guofish_core.SearchConfig(arena_capacity=64))
    scaled = guofish_core.ReplaySearchDouble(
        guofish_core.SearchConfig(c_factor=2.0, arena_capacity=64))
    for n in (0.0, 1.0, 2.5, 800.0, 19652.0):
        assert default.c_puct(n) == guofish_core.c_puct(n)
        assert scaled.c_puct(n) == 2.0 * guofish_core.c_puct(n)
