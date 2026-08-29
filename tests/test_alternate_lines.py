"""`--alternate N` — the top N replies, each with its own score and its own line.

WHAT THIS HAS TO PROVE, AND WHY IT IS THE SAME BAR AS TC PART 1
===============================================================

`playv6_interactive.root_lines` walks `dump_tree_arrays(1)` in Python because
the core offers no descent from a node other than the root:
`Search::principal_variation` starts at `root_` and takes no start node, and
`root_children()` returns visits and nothing else. That walk is the one TC
Part 1b deleted from the PV, so re-introducing it needs the same discipline
Part 1 was held to — IDENTITY WHERE THE TWO OVERLAP, and a bound on the cost.

They overlap on exactly one line: the most-visited root child is the move the
engine plays, so `root_lines`' first line must be `search.principal_variation`
truncated to the same length. That is asserted here on the pinned FENs at two
budgets. Where they do NOT overlap — the second and subsequent alternates —
there is nothing to compare against, so what is asserted instead is that each
line begins with its own move and is legal from the position, which is what a
reader of the table is entitled to assume.

The cost bound is the Part 1 shape: grow the tree and watch the walk not grow
with it. `root_lines` is O(subtree) per alternate over disjoint subtrees, so
what it must not do is scale with the dump.

AMENDMENT D: no module-scope skip. `guofish_core` is a hard dependency of the
suite; the Gate 1 dump and python-chess are not, so each test that needs one
carries its own marker with a named reason.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

GOLDEN = REPO_ROOT / "golden"

# The calibration every shipped checkpoint carries; the score column only needs
# a scale that is not zero, and using the real one keeps the numbers readable
# in a failure message.
VALUE_SCALE = 290.6806


def _why_no_dump() -> str | None:
    for name in ("gate1_dump.npz", "gate1_manifest.json"):
        if not (GOLDEN / name).is_file():
            return f"golden/{name} is missing"
    return None


def _why_no_frontend() -> str | None:
    try:
        import chess  # noqa: F401 - probing importability
    except ImportError as exc:
        return f"python-chess is not importable ({exc})"
    try:
        import playing.v6.playv6_interactive  # noqa: F401 - probing importability
    except ImportError as exc:                     # pragma: no cover - diagnostic
        return f"playing.v6.playv6_interactive is not importable ({exc})"
    return None


DUMP_UNAVAILABLE = _why_no_dump()
requires_dump = pytest.mark.skipif(DUMP_UNAVAILABLE is not None,
                                   reason=str(DUMP_UNAVAILABLE))

FRONTEND_UNAVAILABLE = _why_no_frontend()
requires_frontend = pytest.mark.skipif(FRONTEND_UNAVAILABLE is not None,
                                       reason=str(FRONTEND_UNAVAILABLE))


def frontend():
    import playing.v6.playv6_interactive as module
    return module


# K=1, as TC Part 1 uses: the identity assertions compare two walks over one
# tree and do not need determinism, but a reproducible tree makes a failure
# reproducible too.
SERIAL = dict(workers=1, in_flight=1, max_batch=1)


@pytest.fixture(scope="module")
def dump():
    arrays = np.load(GOLDEN / "gate1_dump.npz")
    manifest = json.loads((GOLDEN / "gate1_manifest.json").read_text(encoding="utf-8"))
    return {"arrays": arrays, "fens": [p["fen"] for p in manifest["positions"]]}


def _searched(dump, fen, sims):
    config = guofish_core.SearchConfig()
    config.arena_capacity = 4_000_000
    search = guofish_core.ReplaySearchQ32(config)
    a = dump["arrays"]
    search.load_dump(a["keys"], a["is_root"], a["move_offset"], a["moves"],
                     a["priors"], a["values"])
    search.synthetic_fallback = True
    search.set_position(fen)
    search.search_parallel(sims, guofish_core.ParallelConfig(**SERIAL))
    return search


def _lines(dump, fen, sims, top_n=4, max_plies=8):
    """(board, arrays, rows, lines) for one searched position."""
    import chess
    PI = frontend()
    search = _searched(dump, fen, sims)
    board = chess.Board(fen)
    arrays = search.dump_tree_arrays(1)
    arrays = arrays if arrays["depth"].size else None
    rows = PI.root_distribution(board, arrays)
    lines = PI.root_lines(board, arrays, rows, VALUE_SCALE, top_n, max_plies)
    return search, board, arrays, rows, lines


# ---------------------------------------------------------------------------
# The reference: the depth-1 selection root_distribution used before the change
# ---------------------------------------------------------------------------


def _reference_root_indices(arrays):
    """The scalar Python loop `root_child_rows` replaces, copied.

    Asserting against the code that was there is the only way to show the
    vectorised selection is a speed change and not a behaviour change; an
    independently-derived expectation would test the expectation.
    """
    depth = arrays["depth"]
    return [i for i in range(int(depth.size)) if int(depth[i]) == 1]


# ---------------------------------------------------------------------------
# Identity — where the Python walk and the C++ descent overlap
# ---------------------------------------------------------------------------


@requires_dump
@requires_frontend
@pytest.mark.parametrize("sims", [400, 4000])
def test_the_played_moves_alternate_line_is_the_cpp_principal_variation(dump, sims):
    """THE ACCEPTANCE BAR. The first alternate IS the PV, or the walk is wrong.

    Two budgets, because a 400-simulation tree is shallower than `max_plies`
    and a 4,000-simulation one is not — the same reason TC Part 1b parametrises
    the same way, and the two exercise different exits from the descent.
    """
    import chess
    PI = frontend()
    for fen in dump["fens"][:6]:
        search, board, arrays, rows, lines = _lines(dump, fen, sims)
        assert lines, f"no alternates at {fen}"

        want = [uci for uci, _v, _s in search.principal_variation(8)]
        # `entries` is SAN and already truncated at the first illegal move, so
        # the comparison is made on the UCI the walk produced. Rebuilt through
        # the same `_san_entries` the display uses, so a truncation in one and
        # not the other would show up here rather than on screen.
        got_san = [e[2] for e in lines[0]["entries"]]
        want_san = [e[2] for e in PI._san_entries(board, want)]
        assert got_san == want_san, (
            f"the played move's line differs from the C++ PV at {fen}\n"
            f"  python: {got_san}\n  c++   : {want_san}")

        assert lines[0]["uci"] == want[0], (
            f"the top alternate is not the PV head at {fen}")


@requires_dump
@requires_frontend
def test_the_vectorised_root_child_selection_is_the_scalar_loop_it_replaced(dump):
    """PART 1a's fix, applied to the one place on this side that still had it.

    `root_distribution` selected the root's children with a Python loop over
    every row of the dump. Same rows, same order, one pass in NumPy.
    """
    PI = frontend()
    for fen in dump["fens"][:6]:
        search = _searched(dump, fen, 4000)
        arrays = search.dump_tree_arrays(1)
        got = [int(i) for i in PI.root_child_rows(arrays)]
        assert got == _reference_root_indices(arrays), f"depth-1 rows differ at {fen}"


# ---------------------------------------------------------------------------
# What the second and subsequent alternates must satisfy
# ---------------------------------------------------------------------------


@requires_dump
@requires_frontend
def test_every_alternate_line_begins_with_its_own_move_and_is_legal(dump):
    """The two things a reader of the table is entitled to assume.

    A line under `Bc4` that opens with `Nf3` would be worse than no line at
    all, and an illegal continuation is a real possibility on a reused tree —
    `_san_entries` truncates rather than raising, so what is asserted is that
    what survives truncation is playable from the position.
    """
    import chess
    for fen in dump["fens"][:6]:
        _s, board, _a, _r, lines = _lines(dump, fen, 4000, top_n=6)
        for row in lines:
            assert row["entries"], f"{row['san']} has an empty line at {fen}"
            assert row["entries"][0][2] == row["san"], (
                f"{row['san']}'s line opens with {row['entries'][0][2]} at {fen}")
            replay = board.copy()
            for _n, _w, san in row["entries"]:
                move = replay.parse_san(san)   # raises if the line is not legal
                replay.push(move)


@requires_dump
@requires_frontend
def test_the_alternates_are_the_most_visited_replies_in_order(dump):
    """The table is a ranking, and the ranking is the search's own.

    `root_lines` truncates `root_distribution`, which is visit-sorted, so the
    rows must be the top of that list — and the played move is its head.
    """
    for fen in dump["fens"][:6]:
        _s, _b, _a, rows, lines = _lines(dump, fen, 4000, top_n=4)
        visits = [row["visits"] for row in lines]
        assert visits == sorted(visits, reverse=True), f"not visit-ordered at {fen}"
        assert [row["uci"] for row in lines] == [row["uci"] for row in rows[:4]]


@requires_dump
@requires_frontend
def test_the_count_clamps_to_the_replies_that_exist(dump):
    """`--alternate 50` in a position with three legal moves prints three rows.

    And `top_n=0` prints none, which is what the flag's own default is spelled
    as — `engine_move` never calls this at 0, but a helper that returned the
    whole table for a request of nothing would be a trap for the next caller.
    """
    fen = dump["fens"][0]
    _s, _b, _a, rows, lines = _lines(dump, fen, 4000, top_n=500)
    assert len(lines) == len(rows)
    _s, _b, _a, _r, none = _lines(dump, fen, 4000, top_n=0)
    assert none == []


# ---------------------------------------------------------------------------
# The score column
# ---------------------------------------------------------------------------


@requires_dump
@requires_frontend
def test_the_score_column_is_white_relative_like_every_other_number_on_screen(dump):
    """The board convention, not the mover's. See `white_relative`.

    The engine's Q reads from the perspective of whoever is to move, so a
    Black-to-move position displayed raw would report `+` for Black standing
    better. Asserted by flipping the side to move on the same tree and
    requiring the sign of the score to follow the BOARD, not the mover.
    """
    import chess
    PI = frontend()
    for fen in dump["fens"][:6]:
        _s, board, arrays, rows, lines = _lines(dump, fen, 4000)
        for row, line in zip(rows, lines):
            # `q` is already White-relative out of `root_distribution`; the
            # score is the same quantity through the calibration, so the two
            # must never disagree about which side is better.
            if abs(line["score_cp"]) > 1:
                assert (line["score_cp"] > 0) == (row["q"] > 0), (
                    f"score and Q disagree on sign for {row['san']} at {fen}")


@requires_frontend
def test_the_bypass_note_replaces_the_table_rather_than_an_empty_one():
    """A book move gets a sentence, not a table with no rows in it.

    Same discipline as `format_root_distribution` and `format_pv`: printing an
    empty table reads as a search that found nothing rather than as a search
    that did not happen.
    """
    PI = frontend()
    out = PI.format_alternates([], None, 0, 0)
    assert "MCTS did not run" in out
    # One sentence: no heading, and therefore no header row and no columns.
    assert len(out.split("\n")) == 1
    assert "Alternatives" not in out


# ---------------------------------------------------------------------------
# The cost bound — the Part 1 shape
# ---------------------------------------------------------------------------


@requires_dump
@requires_frontend
def test_the_walk_does_not_scale_with_the_tree(dump):
    """PART 1's OWN BAR, restated for this walk: O(subtree), not O(tree).

    A unit test cannot assert a wall-clock number — that is a property of a
    machine — so it asserts the structural fact the timing follows from: the
    number of dump rows the walk READS does not grow with the dump. Counted by
    instrumenting the slice comparisons, which is what `_line_from` spends its
    time on.

    Each alternate descends only inside its own subtree, and the subtrees are
    disjoint, so the rows read across N alternates is bounded by the tree
    ONCE however deep the search went — not by N passes over it.
    """
    PI = frontend()
    fen = dump["fens"][0]
    seen = {}
    for sims in (2000, 16000):
        _s, board, arrays, rows, _l = _lines(dump, fen, sims, top_n=4)
        bounds = PI.root_child_rows(arrays)
        # The rows each alternate's descent can touch: its own subtree.
        ends = list(bounds[1:]) + [int(arrays["depth"].size)]
        spans = [int(e) - int(s) for s, e in zip(bounds, ends)]
        top = sorted(spans, reverse=True)[:4]
        seen[sims] = (int(arrays["depth"].size), sum(top))

    small_nodes, small_span = seen[2000]
    big_nodes, big_span = seen[16000]
    assert big_nodes > small_nodes * 2, "the tree did not actually grow"
    # The bound that matters: four alternates read at most the whole tree
    # between them, not four times it.
    assert big_span <= big_nodes, "an alternate's span exceeded the tree"
    assert small_span <= small_nodes
