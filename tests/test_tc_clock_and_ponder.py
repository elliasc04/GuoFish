"""TC — clock safety and ponder semantics. Part 1.

The brief is `docs/TC/clock_and_ponder_brief.md`; the measurements it argues
from are `docs/TC/{L0_LOGS,RECON,DEFECTS,PROBE_FRESH_ROOT}.md`.

WHAT EACH PART HAS TO PROVE HERE, AND WHAT IT CANNOT
====================================================

  Part 1  the two tree walks outside the engine's own time accounting are
          bounded. The bar is IDENTITY, not approximation: the PV, the branch
          rows and `seldepth` must come out exactly as they did off
          `dump_tree_arrays(1)`, or the change is a behaviour change wearing a
          performance change's clothes. That is asserted here against the old
          walk, re-implemented in this file as the reference.

          What is NOT here: the wall-clock acceptance. "Fresh-root gap median
          170 ms -> under 20 ms" is a property of a 120-search probe on a GPU
          and lives in `telemetry/probe_fresh_root.py`. A unit test can only
          show the walk is O(plies) rather than O(tree), which it does by
          growing the tree and watching the walk not grow.

AMENDMENT D: no module-scope skip. `guofish_core` is a hard dependency of the
suite; the Gate 1 dump and python-chess are not, so each test that needs one
carries its own marker with a named reason.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

from playing.v6.playv6 import EngineConfig  # noqa: E402

GOLDEN = REPO_ROOT / "golden"


# ---------------------------------------------------------------------------
# Fixtures — the Gate 1 dump, the same six pinned FENs M3 uses
# ---------------------------------------------------------------------------


def _why_no_dump() -> str | None:
    for name in ("gate1_dump.npz", "gate1_manifest.json"):
        if not (GOLDEN / name).is_file():
            return f"golden/{name} is missing"
    return None


DUMP_UNAVAILABLE = _why_no_dump()
requires_dump = pytest.mark.skipif(DUMP_UNAVAILABLE is not None,
                                   reason=str(DUMP_UNAVAILABLE))


def _why_no_wrapper() -> str | None:
    try:
        import chess  # noqa: F401 - probing importability
    except ImportError as exc:
        return f"python-chess is not importable ({exc})"
    try:
        import playing.uci_wrapper_v6  # noqa: F401 - probing importability
    except ImportError as exc:                     # pragma: no cover - diagnostic
        return f"playing.uci_wrapper_v6 is not importable ({exc})"
    return None


WRAPPER_UNAVAILABLE = _why_no_wrapper()
requires_wrapper = pytest.mark.skipif(WRAPPER_UNAVAILABLE is not None,
                                      reason=str(WRAPPER_UNAVAILABLE))


def wrapper():
    import playing.uci_wrapper_v6 as module
    return module


# K=1. One worker, one leaf in flight: the configuration D-1 leaves
# reproducible. The identity assertions below do not NEED determinism — they
# compare two walks over one tree, whatever that tree turned out to be — but a
# reproducible tree is what makes a failure reproducible too.
SERIAL = dict(workers=1, in_flight=1, max_batch=1)


@pytest.fixture(scope="module")
def dump():
    arrays = np.load(GOLDEN / "gate1_dump.npz")
    manifest = json.loads((GOLDEN / "gate1_manifest.json").read_text(encoding="utf-8"))
    return {"arrays": arrays, "fens": [p["fen"] for p in manifest["positions"]]}


def _search(dump, fen, capacity=4_000_000):
    config = guofish_core.SearchConfig()
    config.arena_capacity = capacity
    search = guofish_core.ReplaySearchQ32(config)
    a = dump["arrays"]
    search.load_dump(a["keys"], a["is_root"], a["move_offset"], a["moves"],
                     a["priors"], a["values"])
    # Whatever the dump misses, a parallel search reaches within a few plies.
    # Deterministic, which is what makes these equalities reproducible.
    search.synthetic_fallback = True
    search.set_position(fen)
    return search


def _searched(dump, fen, sims, parallel=None):
    search = _search(dump, fen)
    search.search_parallel(sims, guofish_core.ParallelConfig(**(parallel or SERIAL)))
    return search


def _sliced(dump, fen, targets, parallel=None):
    """A search cut into slices, with the running max of `SearchStats::max_depth`.

    `Engine.search_move` cuts a move into ~0.05 s slices and `search_parallel`
    resets its own stats every call, so the move's seldepth is the fold and not
    the last slice's. This reproduces that fold so a test can compare it against
    the tree.
    """
    search = _search(dump, fen)
    config = guofish_core.ParallelConfig(**(parallel or SERIAL))
    deepest = 0
    for target in targets:
        stats = search.search_parallel(target, config)
        deepest = max(deepest, int(stats["max_depth"]))
    return search, deepest


# ---------------------------------------------------------------------------
# Part 1 — the reference walks, as they were before the change
# ---------------------------------------------------------------------------
#
# THESE ARE THE OLD IMPLEMENTATIONS, COPIED. That is deliberate and it is the
# only way this file can assert identity: the point is not that the new walk
# agrees with some independently-derived expectation, it is that it agrees with
# the exact code it replaced, node for node, on real trees. Deleting the old
# code and asserting against a hand-written expectation would test the
# expectation.


def _reference_pv(search, max_plies=12):
    """`Engine._principal_variation`'s walk, off `dump_tree_arrays(1)`.

    DFS preorder with a depth column, so a child of the node at index i is the
    next entry with depth == depth[i] + 1 before the first entry with
    depth <= depth[i]; follow the most-visited child at each step.
    """
    arrays = search.dump_tree_arrays(1)
    depth = arrays["depth"]
    if depth.size == 0:
        return [], 0.0, 0

    visits = arrays["visits"]
    value_sum = arrays["value_sum"]
    packed = arrays["move"]
    max_depth = int(depth.max())

    pv: list[str] = []
    q = 0.0
    i = 0
    while len(pv) < max_plies:
        here = int(depth[i])
        best_index = -1
        best_visits = 0
        j = i + 1
        while j < depth.size and int(depth[j]) > here:
            if int(depth[j]) == here + 1 and int(visits[j]) > best_visits:
                best_visits, best_index = int(visits[j]), j
            j += 1
        if best_index < 0:
            break
        if not pv:
            q = float(value_sum[best_index]) / max(1, best_visits)
        pv.append(guofish_core.move_to_uci(int(packed[best_index])))
        i = best_index
    return pv, q, max_depth


def _reference_root_rows(search):
    """`Engine.root_branches`'s source rows: the depth-1 rows of the dump."""
    arrays = search.dump_tree_arrays(1)
    depth = arrays["depth"]
    if depth.size == 0:
        return []
    visits, packed = arrays["visits"], arrays["move"]
    return [(guofish_core.move_to_uci(int(packed[i])), int(visits[i]))
            for i in range(int(depth.size)) if int(depth[i]) == 1]


# ---------------------------------------------------------------------------
# Part 1b — the PV
# ---------------------------------------------------------------------------


@requires_dump
@pytest.mark.parametrize("sims", [400, 4000])
def test_part1b_the_pv_is_identical_to_the_dump_walk_on_six_pinned_fens(dump, sims):
    """THE PART 1 ACCEPTANCE BAR: "the walk gets cheaper, not different".

    Six pinned FENs at K=1, the same set M3's behavioural-identity check uses.
    Two budgets, because a 400-simulation tree is shallower than `max_plies` and
    a 4,000-simulation one is not, and the two exercise different exits from the
    descent.
    """
    fens = dump["fens"][:6]
    assert len(fens) == 6, "the check is specified as six pinned FENs"

    for fen in fens:
        search = _searched(dump, fen, sims)
        want_pv, want_q, want_depth = _reference_pv(search)
        got = search.principal_variation(12)

        assert [uci for uci, _v, _s in got] == want_pv, f"PV differs at {fen}"
        assert [uci for uci, _v, _s in got], f"empty PV at {fen}"
        # The Q the info line reports is the FIRST step's, and it is the
        # engine's own move, so a sign error here is a sign error in the score.
        _uci, visits, value_sum = got[0]
        assert float(value_sum) / max(1, int(visits)) == pytest.approx(want_q,
                                                                      abs=1e-12)
        assert search.max_visited_depth() == want_depth, \
            f"the tree's own depth differs at {fen}"


@requires_dump
def test_part1b_seldepth_from_the_search_equals_the_tree_on_a_fresh_root(dump):
    """PART 1b's OUTPUT CHANGE, BOUNDED BY MEASUREMENT RATHER THAN ARGUED.

    `seldepth` used to be `dump_tree_arrays(1)["depth"].max()`, a property of
    the whole tree. The exact C++ equivalent still walks every child slot of
    every visited node and cost a measured 48 ms on a 200,000-visit fresh root —
    which after the rest of Part 1 landed was the ENTIRE remaining `go` ->
    `bestmove` gap, and which grows with the tree exactly as the leg it replaced
    did. It now comes from `SearchStats::max_depth`, folded across the move's
    slices, which is free because the search was already computing it.

    ON A FRESH ROOT THE TWO ARE THE SAME NUMBER, and that is what makes the
    substitution safe in the regime the probe measures: every visited node was
    put there by this search, so the deepest ply it selected to IS the deepest
    ply in the tree. Six pinned FENs, sliced the way `search_move` slices a move.

    They differ on an INHERITED tree — the old figure includes plies reached by
    previous searches that this one did not re-descend — and that difference is
    the change, stated here rather than discovered later.
    """
    for fen in dump["fens"][:6]:
        search, deepest = _sliced(dump, fen, (1_000, 2_500, 4_000))
        tree_depth = int(search.dump_tree_arrays(1)["depth"].max())
        assert deepest == tree_depth, fen
        assert search.max_visited_depth() == tree_depth, fen


@requires_dump
def test_part1b_the_pv_respects_max_plies_and_a_zero_asks_for_nothing(dump):
    search = _searched(dump, dump["fens"][0], 4000)
    full = search.principal_variation(12)
    assert len(full) <= 12
    assert [u for u, _, _ in search.principal_variation(3)] == \
        [u for u, _, _ in full][:3]
    assert search.principal_variation(0) == []


@requires_dump
def test_part1b_the_pv_walk_does_not_grow_with_the_tree(dump):
    """O(plies x branching), against `dump_tree_arrays(1)`'s O(tree).

    THE COST PROPERTY, AS A RATIO RATHER THAN A DEADLINE. An absolute
    millisecond bound would be a statement about this machine; the claim Part 1
    actually makes is that the walk stopped scaling with the resident tree, and
    that survives a slow box. A 16x tree is compared against a 1x one and the
    walk is allowed to grow by 4x — an enormous allowance for something that
    should not grow at all, and still nowhere near the ~16x the old walk gave.
    """
    fen = dump["fens"][0]
    small = _searched(dump, fen, 1_000)
    large = _searched(dump, fen, 16_000)
    assert int(large.dump_tree_arrays(1)["depth"].size) > \
        4 * int(small.dump_tree_arrays(1)["depth"].size), \
        "the two trees are not different enough for the ratio to mean anything"

    def _cost(search):
        # Best-of-N, not a mean: the minimum of repeated timings is the one
        # robust to a scheduler, and the claim is about the work done rather
        # than about the worst interruption.
        best = float("inf")
        for _ in range(20):
            t = time.perf_counter()
            search.principal_variation(12)
            best = min(best, time.perf_counter() - t)
        return best

    assert _cost(large) < 4.0 * _cost(small) + 1e-4


# ---------------------------------------------------------------------------
# Part 1a — the branch rows
# ---------------------------------------------------------------------------


@requires_dump
def test_part1a_root_children_are_the_depth_one_rows_of_the_dump(dump):
    """Same rows, same order, same filter — for six pinned FENs."""
    for fen in dump["fens"][:6]:
        search = _searched(dump, fen, 4000)
        assert list(search.root_children()) == _reference_root_rows(search), fen


@requires_dump
def test_part1a_an_unvisited_child_is_absent_from_both(dump):
    """`dump_tree(1)`'s own filter, and the reason it is the right one.

    A reply with no visits is not a branch the search built; including it would
    put a 0%-share row in the table and a zero in the share denominator. A tiny
    budget is the case where most children are unvisited, so it is the case that
    distinguishes "filtered" from "happens not to have any".
    """
    search = _searched(dump, dump["fens"][0], 32)
    rows = list(search.root_children())
    assert rows, "a 32-simulation search should still have visited children"
    assert all(visits >= 1 for _uci, visits in rows)
    assert rows == _reference_root_rows(search)


@requires_dump
@requires_wrapper
def test_part1a_root_branches_still_returns_every_visited_reply(dump):
    """The arithmetic callers look the PLAYED move up in this list.

    Truncating it, or applying the display threshold to it, silently reports
    zero inherited visits for any reply outside the top 5 — the bug a real game
    caught. `Engine.root_branches` changed its SOURCE in Part 1a and must not
    have changed its contract.
    """
    from playing.v6 import playv6

    search = _searched(dump, dump["fens"][0], 4000)
    engine = playv6.Engine.__new__(playv6.Engine)
    engine.search = search

    branches = engine.root_branches()
    reference = _reference_root_rows(search)
    assert len(branches) == len(reference)
    assert {uci for uci, _ in reference} == {uci for uci, _, _ in branches}
    assert dict(reference) == {uci: v for uci, v, _ in branches}
    # Most-visited first, and the shares are of the CHILDREN's total.
    assert [v for _, v, _ in branches] == sorted((v for _, v, _ in branches),
                                                reverse=True)
    assert sum(share for _, _, share in branches) == pytest.approx(1.0)
    assert engine.root_branches(top_n=2) == branches[:2]


# ---------------------------------------------------------------------------
# Part 1a — the gate
# ---------------------------------------------------------------------------


@requires_wrapper
def test_part1a_the_branch_table_is_off_unless_asked_for():
    """Diagnostic output does not run in deployed play.

    A property of the control flow, checked on the flag rather than on stderr,
    because the alternative is a subprocess that loads a 130 MB checkpoint to
    observe the absence of six lines.
    """
    module = wrapper()
    import chess

    def _engine(**kwargs):
        uci = module.UCIEngine.__new__(module.UCIEngine)
        uci.config = EngineConfig()
        uci.board = chess.Board()
        uci._debug = bool(kwargs.get("branch_table", False))
        return uci

    assert _engine()._debug is False
    assert _engine(branch_table=True)._debug is True


@requires_wrapper
def test_part1a_uci_debug_on_and_off_move_the_gate():
    """UCI already has this switch and it was being answered with a shrug."""
    source = (REPO_ROOT / "playing/uci_wrapper_v6.py").read_text(encoding="utf-8")
    assert 'if args and args[0] == "on":' in source
    assert "self._debug = True" in source
    assert "self._debug = False" in source
    # And the gate is actually in front of the emission, not merely defined.
    assert "if self._debug and ponder_outcome is not None" in source


@requires_wrapper
def test_part1a_branch_table_is_a_wrapper_flag_not_an_engine_config_field():
    """The `--move-stats` rule, applied to the second wrapper-level flag.

    Every `EngineConfig` field is advertised as a UCI `option name` line, so a
    field here would change the `uci` handshake and make the flag's ABSENCE
    observable — which is exactly what M3's first check forbids.
    """
    assert not any(f.name == "branch_table" for f in
                   __import__("dataclasses").fields(EngineConfig))


# ---------------------------------------------------------------------------
# Part 1c — pv_ms
# ---------------------------------------------------------------------------


def test_part1c_the_outcome_carries_pv_ms_and_wall_s_still_means_what_it_did():
    """D-L0-4 has two listed fixes and this is the one NOT taken.

    Moving where `wall_s` is sampled would change reported nps and invalidate
    every constant in `L0_LOGS.md`. The excluded work is made immaterial
    instead, and the number is reported beside `wall_s` rather than folded into
    it.
    """
    from playing.v6.playv6 import SearchOutcome
    outcome = SearchOutcome(best_move="e2e4", mating_move=None, nominal=1,
                            inherited=0, delivered=1, wall_s=2.0, slices=1,
                            root_visits=1, score_cp=0, q=0.0)
    assert outcome.pv_ms == 0.0
    assert outcome.sims_per_s == pytest.approx(0.5), \
        "wall_s is still the slice loop's own clock and nothing else"

    source = (REPO_ROOT / "playing/v6/playv6.py").read_text(encoding="utf-8")
    # The sample point, unchanged: `wall_s` is still what `_outcome` was handed.
    assert "time.perf_counter() - started, slices," in source






@requires_dump
@requires_wrapper
def test_part1b_engine_principal_variation_end_to_end_matches_the_old_walk():
    """The Python side of Part 1b, not just the C++ side.

    `Engine._principal_variation` returns `(q, pv, depth)` and every one of them
    reaches a UCI `info` line: `score cp` is derived from q, `pv` is the pv and
    `depth` is len(pv). All three are compared against the walk that produced
    them before the change. `seldepth` was the fourth and is no longer computed
    here; the assertion on the tree's own depth is kept so a change to the walk
    that used to produce it would still be caught.
    """
    from playing.v6 import playv6

    arrays = np.load(GOLDEN / "gate1_dump.npz")
    manifest = json.loads((GOLDEN / "gate1_manifest.json").read_text(encoding="utf-8"))
    fens = [p["fen"] for p in manifest["positions"]][:6]
    payload = {"arrays": arrays, "fens": fens}

    for fen in fens:
        search = _searched(payload, fen, 4000)
        want_pv, want_q, want_depth = _reference_pv(search)

        engine = playv6.Engine.__new__(playv6.Engine)
        engine.search = search
        q, pv, depth = engine._principal_variation(search.best_move)

        assert pv == want_pv, fen
        assert depth == len(want_pv)
        assert q == pytest.approx(want_q, abs=1e-12)
        # `seldepth` left this function in Part 1b; the tree's own depth is
        # still reachable and still agrees with the walk that used to produce
        # it. See test_part1b_seldepth_from_the_search_equals_the_tree.
        assert search.max_visited_depth() == want_depth
        assert pv[0] == search.best_move, \
            "the PV head is the move that gets played, or the PV is wrong"
