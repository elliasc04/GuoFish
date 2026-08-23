"""TC — clock safety and ponder semantics. Parts 1, 2 and 4.

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

  Part 2  the deadline exists on the pondered path (D-L0-7) and is armed from
          the `go`/`ponderhit` instant rather than from whenever the planner got
          round to running (Part 2b). Both are planner-level facts and are
          asserted directly; the end-to-end invariant
          `go -> bestmove <= deadline + reserve` is a property of a match and
          lives in `telemetry/check_deadline_invariant.py`.

          The `4YCsGtQ8` move-85 case is here as a planner assertion with the
          real clock off the wire, because that is the part of it that is
          deterministic. The replay is `telemetry/replay_lichess_game.py`.

  Part 4  `PonderMaxSims`/`PonderDecay` resolve to the pinned ceiling and the
          arena that follows from it, and the ponder's target is
          additive-with-a-ceiling rather than unbounded.

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


# ---------------------------------------------------------------------------
# Part 2 — the deadline on the pondered path
# ---------------------------------------------------------------------------


def _uci_for_plan(config: EngineConfig, root_visits: int = 0):
    import chess
    module = wrapper()
    uci = module.UCIEngine.__new__(module.UCIEngine)
    uci.config = config
    uci.engine = type("_E", (), {"ready": True,
                                 "search": type("_S", (), {"root_visits": root_visits})()})()
    uci.board = chess.Board()
    return uci, module


@requires_wrapper
def test_part2a_a_ponderhit_with_a_node_budget_now_arms_the_deadline():
    """D-L0-7, as a regression test.

    This returned `None` where the deadline goes, on the branch deployment
    actually takes: python-chess puts `nodes` on the ponder line whenever the
    Limit carries one, and lichess-bot's `go_commands.nodes` makes it always. So
    the backstop `config.yml` documents was absent on 53% of moves, and absent
    on exactly the moves where the tree is largest and the nps lowest.

    BOTH BOUNDS, not one instead of the other: the node budget is unchanged and
    is still `current + N` fresh simulations. Whichever binds first ends the
    move, which is what `_plan` and `config.yml` already intend.
    """
    uci, module = _uci_for_plan(EngineConfig(), root_visits=30_000)
    params = module.GoParams(["ponder", "wtime", "300000", "btime", "300000",
                              "nodes", "25000"])
    budget, deadline, nominal, source, note = uci._plan_after_ponderhit(params)

    assert source == "ponderhit"
    assert nominal == 25_000
    assert budget - 30_000 == 25_000, "the node budget is unchanged"
    assert deadline is not None, "D-L0-7: the pondered path had no clock"
    assert deadline > time.monotonic(), "an already-expired deadline is not a backstop"
    assert "clock" in note


@requires_wrapper
def test_part2a_a_ponderhit_with_no_clock_still_has_no_deadline():
    """The other side of it. A `go ponder` carrying no clock is legal UCI and
    means the GUI gave the engine nothing to time against; inventing one would
    be worse than the defect."""
    uci, module = _uci_for_plan(EngineConfig(), root_visits=30_000)
    params = module.GoParams(["ponder", "nodes", "25000"])
    _budget, deadline, _nominal, source, _note = uci._plan_after_ponderhit(params)
    assert (source, deadline) == ("ponderhit", None)


@requires_wrapper
def test_part2a_a_fixed_budget_arm_still_ignores_the_clock_after_a_hit():
    """A fixed-budget run ignores the clock everywhere else and must ignore it
    here too, or a pondered move is the one move in the match decided on time."""
    uci, module = _uci_for_plan(EngineConfig(fixed_sims=4_000), root_visits=9_000)
    params = module.GoParams(["ponder", "wtime", "300000", "btime", "300000"])
    budget, deadline, nominal, source, _note = uci._plan_after_ponderhit(params)
    assert (source, deadline, nominal) == ("fixed", None, 4_000)
    assert budget == 9_000 + 4_000


@requires_wrapper
@pytest.mark.parametrize("argv,label", [
    (["ponder", "wtime", "300000", "btime", "300000", "nodes", "25000"], "nodes"),
    (["ponder", "wtime", "300000", "btime", "300000"], "timed"),
])
def test_part2b_the_ponderhit_deadline_is_absolute_not_relative(argv, label):
    """PART 2b, AND IT IS THE HALF THAT IS EASY TO GET WRONG.

    The pre-search leg is consumed BEFORE `_plan_after_ponderhit` runs — a
    measured median of 343 ms on the pondered path and a maximum of 5,851 ms. A
    deadline computed as `now + allotment` does not charge that leg, so it leaks
    straight through the guard; the allotment has to run from the `ponderhit`
    instant.

    Asserted by handing the planner a `since` two seconds in the past and
    watching the deadline come back two seconds earlier. A relative
    implementation is indifferent to `since` and fails this by exactly the
    amount it leaks.
    """
    uci, module = _uci_for_plan(EngineConfig(), root_visits=30_000)
    params = module.GoParams(argv)

    now = time.monotonic()
    _b, from_now, _n, _s, _note = uci._plan_after_ponderhit(params, since=now)
    _b, from_earlier, _n, _s, _note = uci._plan_after_ponderhit(params,
                                                               since=now - 2.0)
    assert from_now is not None and from_earlier is not None, label
    assert from_now - from_earlier == pytest.approx(2.0, abs=5e-3), label


@requires_wrapper
def test_part2b_the_ordinary_path_arms_from_the_go_line_too():
    """Done identically in both planners rather than only where it hurts.

    On this path the leg is 4 ms at the fresh-root median, so the leak is small
    — but `set_position` REBUILDS THE TREE on a ponder miss and that is the same
    path, so "small" is a statement about the median and not about the tail.
    """
    uci, module = _uci_for_plan(EngineConfig(), root_visits=0)
    params = module.GoParams(["wtime", "300000", "btime", "300000",
                              "nodes", "25000"])
    now = time.monotonic()
    _b, from_now, _n, _s, _note = uci._plan(params, since=now)
    _b, from_earlier, _n, _s, _note = uci._plan(params, since=now - 2.0)
    assert from_now - from_earlier == pytest.approx(2.0, abs=5e-3)


@requires_wrapper
def test_part2_move_85_of_4YCsGtQ8_would_have_been_cut():
    """THE GAME, AT THE PLANNER LEVEL.

    `4YCsGtQ8`, 2026-08-17, lost `outoftime` at move 85. From the wire: a
    ponderhit with 4,283 ms on our clock and a 3,000 ms increment, which spent
    7,164 ms searching because no deadline was armed. The two moves in the same
    sequence where the ponder MISSED took the ordinary path, were timed, and
    ended at `reason=time` under three seconds.

    `_allot` on those numbers: 4.283/30 + 0.8x3.0 = 2.543 s, capped by the
    40%-of-clock guard at 0.4x4.283 = 1.713 s, less the 100 ms engine reserve.
    The bar is not the exact figure — it is that the deadline exists, is under
    the 7.164 s the move actually took, and is inside the 40% guard the engine
    already documents and was violating on this move.

    The end-to-end version, driving the engine over pipes from the recorded
    positions, is `telemetry/replay_lichess_game.py`.
    """
    uci, module = _uci_for_plan(EngineConfig(), root_visits=37_356)
    params = module.GoParams(["ponder", "wtime", "4283", "btime", "311140",
                              "winc", "3000", "binc", "3000", "nodes", "200000"])
    now = time.monotonic()
    _b, deadline, _n, source, _note = uci._plan_after_ponderhit(params, since=now)

    assert source == "ponderhit"
    assert deadline is not None
    allotted = deadline - now
    assert allotted < 7.164, "the move that lost the game must be cut short"
    assert allotted <= 0.4 * 4.283, "the 40%-of-clock guard must bind"
    assert allotted == pytest.approx(0.4 * 4.283 - 0.1, abs=1e-3)


@requires_wrapper
def test_part2c_the_reserve_is_the_engines_own_move_overhead_and_is_recorded():
    """PART 2c, AND THE ANSWER IS THAT THE RESERVE IS ALREADY THERE.

    The unprotected residual after Part 1 is `bestmove` -> move POSTed: 630 ms
    median, 706 ms p95, flat across regimes because it is network rather than
    tree. The brief asks for it to be reserved at p95 AND checked against
    lichess-bot's own `move_overhead` so the margin is not budgeted twice.

    It is budgeted twice if the engine adds 706 ms of its own:
    `game_clock_time` subtracts `pre_move_time + move_overhead` — 2,000 ms in
    the deployed config — from `wtime`/`btime` before the `go` line is written,
    so `clock_before_ms` is ALREADY net of it and covers the p95 leg 2.8x over.
    What the engine adds on top is `MoveOverhead`, 100 ms, which `_allot`
    subtracts from every allotment on every path.

    So the change here is not a new constant. It is that the reserve is now
    RECORDED per move, so the invariant can be checked rather than assumed.
    """
    uci, module = _uci_for_plan(EngineConfig(move_overhead_ms=100), root_visits=0)
    params = module.GoParams(["wtime", "60000", "btime", "60000", "movetime", "5000"])
    assert uci._allot(params) == pytest.approx(4.9), \
        "movetime is taken literally minus the engine's own reserve"

    facts = uci._clock_facts(params, deadline=time.monotonic() + 4.9,
                             since=time.monotonic())
    assert facts["reserve_ms"] == 100.0
    assert facts["clock_before_ms"] == 60_000
    assert facts["deadline_from_go_ms"] == pytest.approx(4900.0, abs=5.0)


@requires_wrapper
def test_part2_the_move_stats_record_carries_what_the_invariant_needs():
    """Thousands of samples instead of counting rare events.

    The base rate of a flag is 1 in 80 games, so the acceptance is the
    INVARIANT — `go -> bestmove <= deadline + reserve` on every move — and that
    needs three numbers per move that nothing was recording.
    """
    from telemetry.move_stats import build_record
    from playing.v6.playv6 import SearchOutcome

    outcome = SearchOutcome(best_move="e2e4", mating_move=None, nominal=200_000,
                            inherited=0, delivered=200_000, wall_s=12.0,
                            slices=240, root_visits=200_000, score_cp=15, q=0.05)
    record = build_record(
        ply=1, side="w", fen="startpos", outcome=outcome, checkpoints={},
        timings={"go_to_bestmove_ms": 12_100.0, "pv_ms": 2.1,
                 "deadline_hit": False, "clock_before_ms": 300_000,
                 "deadline_from_go_ms": 9_900.0, "reserve_ms": 100.0},
        ponder={}, position={}, config={})
    for key in ("go_to_bestmove_ms", "pv_ms", "deadline_hit", "clock_before_ms",
                "deadline_from_go_ms", "reserve_ms"):
        assert key in record, key


# ---------------------------------------------------------------------------
# Part 4 — the ponder ceiling and the arena
# ---------------------------------------------------------------------------


def test_part4a_the_deployed_ceiling_no_longer_comes_from_a_dead_decay():
    """D-L0-1. `PonderDecay` does not decay anything.

    `apply_move`'s `from_ponder` is never passed by any production caller, so
    C8's inheritance decay is unreachable. What the deployed `PonderDecay: 0.3`
    actually did was set the ponder ceiling to `SimCap / 0.3` = 666,667 — and it
    BOUND: `ponder_sims` p99 and max are both exactly that — and commit a
    3.78 GiB arena for it.

    Pinning the ceiling says the same thing without routing it through a knob
    that does not do what its name says.
    """
    dead = EngineConfig(sim_cap=200_000, ponder_decay=0.3)
    assert dead.ponder_max_sims_resolved == 666_667

    pinned = EngineConfig(sim_cap=200_000, ponder_decay=1.0,
                          ponder_max_sims=200_000)
    assert pinned.ponder_max_sims_resolved == 200_000
    assert pinned.coupling_holds, \
        "at decay 1.0 a ponder capped at the move budget cannot outvote it"


def test_part4b_the_arena_follows_the_ceiling_down():
    """3.78 GiB -> 1.74 GiB, and `coupling_holds` becomes a real statement.

    `arena_nodes = 60 x (sims_per_move + ponder_max_sims)` at 78 B/node, both
    ping-pong arenas included (RECON R2, measured against the running engine).
    """
    dead = EngineConfig(sim_cap=200_000, ponder_decay=0.3)
    pinned = EngineConfig(sim_cap=200_000, ponder_decay=1.0,
                          ponder_max_sims=200_000)

    assert dead.arena_nodes == 60 * (200_000 + 666_667)
    assert pinned.arena_nodes == 60 * (200_000 + 200_000) == 24_000_000
    # MiB, the launcher's own unit.
    assert pinned.arena_bytes / 2 ** 20 == pytest.approx(1785.3, abs=0.5)
    assert dead.arena_bytes / 2 ** 20 == pytest.approx(3868.1, abs=0.5)
    assert pinned.arena_bytes < 0.47 * dead.arena_bytes


@requires_wrapper
def test_part4b_the_ponder_target_is_additive_with_a_ceiling():
    """D-L0-2. `current + cap` with no ceiling is how the tree ran away.

    Measured in ordinary rated play: 2,454,476 resident root visits, against
    the 866,667 the deployed arena was sized for — and then `arena_exhausted`,
    and then a move played on ONE delivered simulation.

    Two regimes, and the ceiling has to do the right thing in both: below it the
    ponder is unchanged and still extends the tree, at it the ponder stops.
    """
    config = EngineConfig(sim_cap=200_000, ponder_decay=1.0,
                          ponder_max_sims=200_000)
    assert config.tree_ceiling == 400_000

    uci, module = _uci_for_plan(config, root_visits=50_000)
    params = module.GoParams(["ponder", "wtime", "300000", "btime", "300000",
                              "nodes", "200000"])
    budget, deadline, _n, source, note = uci._plan(params)
    assert (source, deadline) == ("ponder", None)
    assert budget == 250_000, "below the ceiling the ponder is unchanged"
    assert "CAPPED" not in note

    uci, module = _uci_for_plan(config, root_visits=300_000)
    budget, _d, _n, _s, note = uci._plan(params)
    assert budget == 400_000, "the ponder stops at the tree ceiling"
    assert "CAPPED" in note


@requires_wrapper
def test_part4b_the_ponder_still_extends_beyond_what_is_resident():
    """NOT PURELY ABSOLUTE, and the distinction is the whole of 4b.

    A `min(cap, ceiling)` target would deliver zero whenever the previous move
    already reached the ceiling, and then the engine does not think during the
    opponent's turn at all — which is free time and the one budget nothing else
    competes for.
    """
    config = EngineConfig(sim_cap=200_000, ponder_decay=1.0,
                          ponder_max_sims=200_000)
    uci, module = _uci_for_plan(config, root_visits=199_000)
    params = module.GoParams(["ponder", "nodes", "200000"])
    budget, _d, _n, _s, _note = uci._plan(params)
    assert budget > 199_000, "a ponder that delivers nothing is not a ponder"


@requires_wrapper
@pytest.mark.parametrize("floor,current,expected_delivered", [
    # The default: today's behaviour, exactly, on every tree size.
    (1.0, 0, 200_000),
    (1.0, 316_376, 200_000),
    (1.0, 2_454_476, 200_000),
    # Absolute: a hit draws against the move's allocation instead of adding to
    # it, and delivers nothing once the ponder already got there.
    (0.0, 0, 200_000),
    (0.0, 150_000, 50_000),
    (0.0, 316_376, 0),
    # Floored: cost-neutral-ish, and never zero.
    (0.5, 316_376, 100_000),
    (0.5, 0, 200_000),
])
def test_part3_the_floor_spans_both_designs(floor, current, expected_delivered):
    """PART 3, AS ONE KNOB RATHER THAN A CODE CHANGE.

    `target = max(N, current + floor x N)`. 1.0 is `current + N` and is what
    ships today; 0.0 is `max(current, N)` and is the absolute target the brief
    argues for; anything between floors the delivery at a fraction of N.

    The choice between them is the operator's — it is the only part of this
    change that can lose ELO and the only one whose acceptance needs a match —
    so it is a config value with the old behaviour as its default, not a new
    default with the old behaviour as an escape hatch.

    `max(N, ...)` on the low side matters: a hit arriving on a tree SHORTER than
    the budget must still be brought up to it. That is the (0.0, 0) row and the
    (0.0, 150_000) row.
    """
    config = EngineConfig(sim_cap=200_000, ponderhit_floor=floor)
    uci, module = _uci_for_plan(config, root_visits=current)
    params = module.GoParams(["ponder", "wtime", "300000", "btime", "300000",
                              "nodes", "200000"])
    target, deadline, nominal, source, _note = uci._plan_after_ponderhit(params)

    assert (source, nominal) == ("ponderhit", 200_000), \
        "`nominal` stays the ASK; `delivered` becomes what the tree still needed"
    assert deadline is not None, "Part 2 holds whatever Part 3 is set to"
    assert max(0, target - current) == expected_delivered


def test_part3_the_default_is_the_behaviour_that_ships_today():
    """Stated as a test because it is the load-bearing claim of the delivery.

    Parts 1, 2 and 4 are safety and a bounded tree and are strength-neutral by
    construction. Part 3 is a strength change whose acceptance is a match that
    has not been run. Shipping it inert is what keeps the first three
    acceptable on their own evidence.
    """
    assert EngineConfig().ponderhit_floor == 1.0
    assert EngineConfig().tree_max_visits is None


def test_part3_refuses_a_floor_outside_the_unit_interval():
    """1.0 is `current + N` and 0.0 is `max(current, N)`; outside is neither."""
    from playing.v6.playv6 import ConfigError
    with pytest.raises(ConfigError, match="ponderhit_floor"):
        EngineConfig(ponderhit_floor=-0.1)
    with pytest.raises(ConfigError, match="ponderhit_floor"):
        EngineConfig(ponderhit_floor=1.5)


def test_part4b_refuses_a_ceiling_under_the_move_budget():
    """A fresh root would be unable to reach its own target."""
    from playing.v6.playv6 import ConfigError
    with pytest.raises(ConfigError, match="tree_max_visits"):
        EngineConfig(sim_cap=200_000, tree_max_visits=100_000)


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
