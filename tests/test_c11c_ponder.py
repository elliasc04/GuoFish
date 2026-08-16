"""C11c — pondering: the mutable deadline, graceful arena exhaustion, and the
UCI state machine driven through a real subprocess pipe.

WHY THE PIPE IS NOT NEGOTIABLE
==============================
C11's deadlock — `import torch` never returning while another thread blocked in
`sys.stdin.readline()` — was an initialisation-order bug that fired ONLY through
a pipe and never at a terminal. Pondering is the first feature that makes search
run concurrently with stdin reading for an unbounded period, so it inherits that
precedent even though it re-imports nothing: `go ponder` at a terminal and `go
ponder` behind a pipe are different programs, and the second is the one Cutechess
and lichess-bot run. Every state-machine assertion below therefore drives a real
`playing/uci_wrapper_v6.py` subprocess over real pipes. A ponder test that passed
in process would have tested the wrong thing.

THREE LAYERS, AND WHAT EACH ONE IS FOR
======================================
  configuration   Pure Python, no GPU, no subprocess. The `ponder_max_sims` /
                  `ponder_decay` coupling and the arena formula, asserted at the
                  computed defaults. Fast, so a broken derivation fails in
                  milliseconds rather than after a model load.

  core            The C++ mechanisms, against the Gate 1 replay dump and the
                  stand-in evaluator — no GPU and no checkpoint. This is where
                  requirements 1 and 3 are actually pinned: the deadline armed
                  from ANOTHER THREAD mid-search, and an arena driven to
                  exhaustion on purpose. Both assert the tree survives, which is
                  the property that distinguishes degradation from damage:
                  `vloss_total == 0` and `conservation_failures == 0`.

  protocol        One engine subprocess, shared across the state-machine tests
                  because each start pays a checkpoint load and a CUDA graph
                  capture. Every legal transition and every illegal one from
                  requirement 4.

AMENDMENT D: no module-scope skip. `guofish_core` is a hard dependency of the
whole suite; `python-chess`, a CUDA device and a checkpoint are not, and each
test that needs one carries its own marker with a named reason. The protocol
layer therefore SKIPS rather than fails on a machine with no GPU, and says so
per test rather than silently collecting nothing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

from playing.v6 import playv6  # noqa: E402
from playing.v6.playv6 import EngineConfig  # noqa: E402

GOLDEN = REPO_ROOT / "golden"
ENGINE = REPO_ROOT / "playing" / "uci_wrapper_v6.py"

# A short Ruy Lopez, so a ponder has a real middlegame tree to build and the
# reuse checks have something to inherit. Deliberately past the depth a book
# would answer, though the protocol engine runs with the book off anyway.
OPENING = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]


# ---------------------------------------------------------------------------
# Layer 1 — the configuration derivation
# ---------------------------------------------------------------------------


def test_the_ponder_cap_and_the_decay_are_coupled_at_the_computed_default():
    """`decay x ponder_max_sims <= sims_per_move`, which is the whole point.

    C8's inheritance decay exists so a fresh search can overturn a verdict the
    ponder inherited. That only works while the DECAYED inherited weight is
    comparable to the fresh budget — and setting the two independently is
    exactly how the Python reference reached 60,000 inherited simulations
    against 2,000 fresh ones, which scope E6 records as the over-commit defect
    this port is fixing rather than reproducing.

    Asserted across the decay range rather than at one point, because the
    computed default is a FUNCTION of the decay and a formula that held only at
    1.0 would be a coincidence.
    """
    for decay in (1.0, 0.75, 0.5, 0.25, 0.1):
        config = EngineConfig(ponder_decay=decay)
        assert config.coupling_holds, (
            f"at ponder_decay={decay} the computed ponder cap "
            f"{config.ponder_max_sims_resolved} violates "
            f"decay x cap <= sims_per_move ({config.sims_per_move})")
        # The equality the formula is derived from, to within the round-up.
        assert config.ponder_max_sims_resolved == math.ceil(
            config.sims_per_move / decay)


def test_a_pinned_ponder_cap_that_breaks_the_coupling_is_reported_not_refused():
    """An operator may overrule it; the engine must say so rather than object.

    Refusing would make the knob unusable for the one experiment that could
    justify changing it, and the consequence of a violation is a weaker search
    rather than a broken one. So `coupling_holds` goes false, the startup line
    prints VIOLATED, and the engine runs.
    """
    config = EngineConfig(ponder_decay=1.0, ponder_max_sims=10 * EngineConfig.sim_cap)
    assert not config.coupling_holds
    ponder_line = [l for l in config.describe() if l.startswith("[config] ponder")]
    assert len(ponder_line) == 1, config.describe()
    assert "VIOLATED" in ponder_line[0], ponder_line[0]


def test_the_startup_line_reports_both_the_cap_and_the_decay():
    """Requirement 3b: "Log both values together at startup so a mismatched pair
    is visible before a game rather than after one." """
    ponder_line = [l for l in EngineConfig().describe()
                   if l.startswith("[config] ponder")][0]
    assert "ponder_max_sims=" in ponder_line
    assert "ponder_decay=" in ponder_line
    assert "sims_per_move=" in ponder_line
    assert "coupling:" in ponder_line


def test_the_arena_default_is_sixty_nodes_per_simulation_of_both_budgets():
    """`arena_nodes = 60 x (sims_per_move + ponder_max_sims)`.

    60 is the conservative low-sim end of the measured curve (40 nodes/sim from
    C8's 2,000- and 8,000-simulation runs) times a 1.5 safety factor for
    branching variance — NOT the 32.4 the 37,000-simulation playtesting run
    showed, because the curve is sublinear and fitting its cheap end would size
    the arena for the budget least likely to need the headroom.
    """
    for decay in (1.0, 0.25):
        config = EngineConfig(ponder_decay=decay)
        assert config.arena_nodes == 60 * (config.sims_per_move
                                           + config.ponder_max_sims_resolved)


def test_pondering_is_off_unless_a_caller_asks_for_it():
    """The ENGINE default is off; only `playing/run_guofishv6.bat` opts in.

    That split is what keeps the benches honest. Every measuring caller —
    the cutechess `command.txt` runs, `tools/smoke_c11.py`,
    `tools/uci_conform_c11.py` — invokes `uci_wrapper_v6.py` directly and
    passes no `--ponder`, so they measure the ponder-off engine the throughput
    anchors were recorded against. A default flipped here would make every one
    of them spend the opponent's clock without saying so, and report a sims/s
    divided by a wall time that never included it.

    The competitive launcher turning it on is the whole point and is not a
    reason to move the default: one caller wanting a knob is not the knob's
    default.
    """
    import argparse
    assert EngineConfig.ponder is False
    assert EngineConfig().ponder is False

    parser = argparse.ArgumentParser()
    playv6.add_config_arguments(parser)
    bare = playv6.config_from_args(parser.parse_args([]))
    assert bare.ponder is False, (
        "a bare uci_wrapper_v6.py invocation must not ponder; every bench in "
        "benchmarking/engine/games/ is launched that way")
    asked = playv6.config_from_args(parser.parse_args(["--ponder"]))
    assert asked.ponder is True


def test_a_pinned_fixed_budget_does_not_shrink_the_arena():
    """The arena is sized from `sim_cap`, NOT from the budget actually pinned.

    This is the invariant the `--sims`/`--fixed-sims` collapse rests on. Before
    it, `--sims N` set `default_sims`, which feeds nothing in the derivation, so
    the arena stayed at `sim_cap`'s. After it, `--sims N` sets `fixed_sims`,
    which DOES feed `sims_per_move` — and a naive formula would have quietly
    re-sized every fixed-budget run's arena downward by the ratio of the budget
    to the cap.

    That matters because the arena's real consumer is the ACCUMULATING tree, not
    one search. DECISIONS.md's C12b entry is the record: a reuse arm of seven
    20,000-simulation searches on one tree needed 1,436,625 nodes and exhausted
    a 1.2M arena sized off one move's allowance. `playv6 --selfplay` accumulates
    the same way, so a shrunk arena would surface as `arena_exhausted` mid-run
    rather than as an error at startup.
    """
    reference = EngineConfig().arena_nodes
    for budget in (2_000, 4_000, 16_000):
        config = EngineConfig(fixed_sims=budget)
        assert config.arena_nodes == reference, (
            f"fixed_sims={budget} resized the arena to {config.arena_nodes:,} "
            f"from {reference:,}; the sizing must follow sim_cap")
    # It still follows `sim_cap`, which is the knob that IS allowed to move it.
    assert EngineConfig(sim_cap=50_000).arena_nodes < reference


def test_the_ponder_ceiling_still_follows_the_pinned_budget():
    """The other half of the same split, and the reason it is a split at all.

    Widening the ARENA to `sim_cap` is free headroom. Widening the PONDER
    CEILING to `sim_cap` would not be: a ponder in a `--sims 2000` run could
    then spend 60,000 simulations and hand the next 2,000-simulation search a
    tree it cannot outvote. That is scope E6's over-commit defect verbatim, so
    `ponder_max_sims_resolved` stays tied to `sims_per_move` while
    `arena_sizing_terms` does not.
    """
    config = EngineConfig(fixed_sims=2_000)
    assert config.sims_per_move == 2_000
    assert config.ponder_max_sims_resolved == 2_000
    assert config.coupling_holds
    # The arena, meanwhile, is sized from the cap and is unmoved.
    assert config.arena_sizing_terms == (config.sim_cap, config.sim_cap)


def test_an_explicit_ponder_ceiling_is_honoured_by_the_arena_sizing():
    """An operator who named `ponder_max_sims` meant it, so the arena is sized
    for the ponder that will actually run rather than for the computed one."""
    config = EngineConfig(ponder_max_sims=5_000)
    assert config.arena_sizing_terms == (config.sim_cap, 5_000)
    assert config.arena_nodes == 60 * (config.sim_cap + 5_000)


def test_the_startup_line_reports_the_arena_and_its_memory_footprint():
    """Requirement 3c: "the engine prints the resolved value and its memory
    footprint at startup — `arena: 2 x 600k nodes (48 MB)`".

    BOTH arenas, because both are live: `apply_move` compacts the surviving
    subtree into the standby arena and swaps, so peak occupancy is the moment
    the two overlap and reporting one would halve the number.
    """
    config = EngineConfig()
    memory = [l for l in config.describe() if l.startswith("[config] memory")][0]
    assert f"2 x {config.arena_nodes:,} nodes" in memory, memory
    assert "MB total" in memory, memory
    # Sanity on the width itself: ~39 B/node in the SoA layout, doubled.
    per_node = config.arena_bytes / (2 * config.arena_nodes)
    assert 30 <= per_node <= 50, f"{per_node} B/node is not the SoA layout"


def test_pinning_either_field_overrides_its_computed_default():
    explicit = EngineConfig(arena_capacity=123_456, ponder_max_sims=789)
    assert explicit.arena_nodes == 123_456
    assert explicit.ponder_max_sims_resolved == 789
    assert explicit.to_search_config().arena_capacity == 123_456
    # And the log says which of the two it was, so a run's own artifact
    # distinguishes "the operator chose this" from "the formula produced it".
    memory = [l for l in explicit.describe() if l.startswith("[config] memory")][0]
    assert "explicit" in memory, memory


def test_an_unset_arena_capacity_still_reaches_the_core_as_a_number():
    """`None` means "compute it", and the core takes an integer.

    The failure this pins is a `None` travelling into `SearchConfig` and being
    coerced to 0 — an arena that can hold nothing, which would surface as a
    root-expansion failure on the first move rather than as a configuration
    error at startup.
    """
    config = EngineConfig()
    assert config.arena_capacity is None
    assert config.to_search_config().arena_capacity == config.arena_nodes > 1024


@pytest.mark.parametrize("changes", [
    {"arena_capacity": 16},        # still refused when PINNED below the floor
    {"ponder_max_sims": 0},
    {"ponder_max_sims": -1},
])
def test_an_unrunnable_ponder_configuration_is_refused(changes):
    with pytest.raises(playv6.ConfigError):
        EngineConfig(**changes)


# ---------------------------------------------------------------------------
# Layer 1b — the outcome's two simulation counts
# ---------------------------------------------------------------------------


def test_ponder_and_search_simulations_are_reported_separately():
    """Requirement 2: "Telemetry reports `ponder_sims` and `search_sims`
    separately — never one merged figure."

    Budget accounting starts at the `ponderhit` instant, so a merged count would
    make the engine look 2-3x faster than it is on a hit and would make its
    throughput incomparable with the ponder-off arm Gate 5 is measured against.
    """
    outcome = playv6.SearchOutcome(
        best_move="e2e4", mating_move=None, nominal=None, inherited=5000,
        delivered=13_882, wall_s=1.0, slices=20, root_visits=18_882,
        score_cp=58, q=0.1, ponder_sims=4999, ponder_wall_s=0.34)
    assert outcome.search_sims == 13_882
    assert outcome.ponder_sims == 4999
    assert outcome.total_sims == 18_881
    # THE HEADLINE RATE DIVIDES POST-HIT WORK BY POST-HIT WALL TIME. Folding the
    # ponder in either numerator or denominator would be the C11 defect again.
    assert outcome.sims_per_s == pytest.approx(13_882.0)

    line = outcome.telemetry()
    assert "ponder_sims=4999" in line
    assert "search_sims=13882" in line
    assert "total_sims=18881" in line


def test_a_degraded_move_says_so_loudly_in_its_own_telemetry():
    """Requirement 3: "A silent degradation that only shows up as unexplained
    weakness is worse than a crash, because nobody investigates it." """
    outcome = playv6.SearchOutcome(
        best_move="e2e4", mating_move=None, nominal=None, inherited=0,
        delivered=1206, wall_s=0.4, slices=3, root_visits=1207, score_cp=0,
        q=0.0, arena_exhausted=True, arena_exhausted_at=39_982,
        arena_high_water=39_982, arena_capacity=40_000)
    line = outcome.telemetry()
    assert "ARENA_EXHAUSTED" in line
    assert "39982" in line
    assert "NOT admissible as a benchmark row" in line
    assert outcome.arena_utilisation == pytest.approx(39_982 / 40_000)


def test_a_healthy_move_carries_no_degradation_notice():
    """The notice must be absent when nothing degraded, or it is noise nobody
    reads and therefore nobody notices when it matters."""
    outcome = playv6.SearchOutcome(
        best_move="e2e4", mating_move=None, nominal=4000, inherited=0,
        delivered=4000, wall_s=0.3, slices=3, root_visits=4000, score_cp=0,
        q=0.0, arena_high_water=135_097, arena_capacity=1_200_000)
    assert "ARENA_EXHAUSTED" not in outcome.telemetry()
    assert not outcome.arena_exhausted


# ---------------------------------------------------------------------------
# Layer 2 — the C++ mechanisms, on the replay dump (no GPU, no checkpoint)
# ---------------------------------------------------------------------------


def _why_no_dump() -> str | None:
    for name in ("gate1_dump.npz", "gate1_manifest.json"):
        if not (GOLDEN / name).is_file():
            return f"golden/{name} is missing"
    return None


DUMP_UNAVAILABLE = _why_no_dump()
requires_dump = pytest.mark.skipif(DUMP_UNAVAILABLE is not None,
                                   reason=str(DUMP_UNAVAILABLE))


@pytest.fixture(scope="module")
def dump():
    """The Gate 1 replay dump plus a middlegame FEN to search from.

    Read-only, and used here only as a source of network answers so that the
    deadline and exhaustion mechanisms can be exercised without a GPU. The
    stand-in evaluator fills whatever the dump misses, which a parallel search
    reaches within a few plies — that is leaf parallelism working, and nothing
    in this file compares a tree against Python.
    """
    arrays = np.load(GOLDEN / "gate1_dump.npz")
    manifest = json.loads((GOLDEN / "gate1_manifest.json").read_text(encoding="utf-8"))
    return {"arrays": arrays, "fen": manifest["positions"][0]["fen"]}


def _search(dump, capacity):
    config = guofish_core.SearchConfig()
    config.arena_capacity = capacity
    search = guofish_core.ReplaySearchQ32(config)
    a = dump["arrays"]
    search.load_dump(a["keys"], a["is_root"], a["move_offset"], a["moves"],
                     a["priors"], a["values"])
    search.synthetic_fallback = True
    search.set_position(dump["fen"])
    return search


PARALLEL = dict(workers=2, in_flight=8, max_batch=64)


@requires_dump
def test_a_deadline_armed_before_the_search_ends_it_on_time(dump):
    search = _search(dump, 4_000_000)
    search.set_deadline_in(0.15)
    started = time.perf_counter()
    search.search_parallel(2_000_000, guofish_core.ParallelConfig(**PARALLEL))
    took = time.perf_counter() - started
    stats = search.parallel_stats()

    assert stats["deadline_hit"] is True
    assert 0.15 <= took < 0.60, f"{took:.3f}s against a 0.15s deadline"
    assert 0 < stats["delivered"] < stats["requested"]
    assert search.best_move is not None


@requires_dump
def test_the_deadline_is_settable_from_another_thread_mid_search(dump):
    """REQUIREMENT 1, and the reason the whole mechanism exists.

    `ponderhit` must convert an infinite-budget search into a timed one WITHOUT
    stopping and restarting it, because restarting discards the tree and the
    tree is the entire point of having pondered. So the arming thread here is
    not the searching thread, and the assertion afterwards is that the tree
    survived the conversion.
    """
    search = _search(dump, 4_000_000)
    parallel = guofish_core.ParallelConfig(**PARALLEL)

    def arm():
        time.sleep(0.20)
        search.set_deadline_in(0.10)

    armer = threading.Thread(target=arm)
    started = time.perf_counter()
    armer.start()
    search.search_parallel(2_000_000, parallel)
    took = time.perf_counter() - started
    armer.join()

    stats = search.parallel_stats()
    assert stats["deadline_hit"] is True
    # 0.20 to arm plus 0.10 of allotted time. The upper bound is loose because
    # a slice's in-flight batch still has to drain; it is tight enough that a
    # search running to its 2,000,000 budget instead would fail it by minutes.
    assert 0.30 <= took < 1.0, f"{took:.3f}s for a 0.20+0.10 conversion"

    delivered = stats["delivered"]
    assert 0 < delivered < stats["requested"]

    # THE TREE SURVIVED. A second call continues from where the first stopped
    # rather than starting over, which is what makes a ponderhit worth having.
    before = search.root_visits
    assert before >= delivered
    search.clear_deadline()
    search.set_deadline_in(0.10)
    search.search_parallel(2_000_000, parallel)
    assert search.root_visits > before

    audit = search.audit()
    assert audit["vloss_total"] == 0
    assert audit["conservation_failures"] == 0


@requires_dump
def test_clearing_the_deadline_returns_the_search_to_its_node_budget(dump):
    """The deadline outlives the `search_parallel` that observed it BY DESIGN —
    the host arms it once and the slice loop enters the search many times — so
    the disarm has to be a real operation and not a side effect."""
    search = _search(dump, 4_000_000)
    assert search.deadline_armed is False
    search.set_deadline_in(5.0)
    assert search.deadline_armed is True
    assert 0.0 < search.deadline_remaining_s <= 5.0
    search.clear_deadline()
    assert search.deadline_armed is False
    assert search.deadline_remaining_s == 0.0

    search.search_parallel(2000, guofish_core.ParallelConfig(**PARALLEL))
    stats = search.parallel_stats()
    assert stats["deadline_hit"] is False
    assert stats["delivered"] == stats["requested"], (
        "with no deadline armed the search must still deliver its whole budget")


@requires_dump
def test_an_already_expired_deadline_is_answered_rather_than_refused(dump):
    """A `ponderhit` on a flagged clock. Zero allotted time is a legitimate
    instruction — answer with the best move the ponder already found — and not
    an error to raise out of the middle of a game."""
    search = _search(dump, 4_000_000)
    search.search_parallel(500, guofish_core.ParallelConfig(**PARALLEL))
    assert search.best_move is not None

    search.set_deadline_in(-1.0)
    search.search_parallel(500_000, guofish_core.ParallelConfig(**PARALLEL))
    assert search.parallel_stats()["deadline_hit"] is True
    assert search.best_move is not None


@requires_dump
def test_an_undersized_arena_degrades_and_does_not_throw(dump):
    """REQUIREMENT 3, and the acceptance criterion in the brief's own words:
    "assert the search returns a legal best move with `arena_exhausted` set and
    `delivered < requested` — no exception, no hang, no null bestmove".

    The two audit assertions are what separate degradation from damage. A
    virtual loss left applied on an abandoned path is a permanent bias the tree
    never recovers from, and a conservation failure means the visit counts no
    longer describe the tree they are attached to. Either would make "the search
    returned a slightly weaker move" a false description of what happened.
    """
    search = _search(dump, 40_000)
    search.search_parallel(4000, guofish_core.ParallelConfig(**PARALLEL))
    stats = search.parallel_stats()

    assert stats["arena_exhausted"] is True
    assert stats["delivered"] < stats["requested"]
    assert stats["delivered"] > 0, "it degraded before doing any work at all"
    assert stats["arena_exhausted_at"] > 0
    assert search.best_move is not None and search.best_move != "0000"

    audit = search.audit()
    assert audit["vloss_total"] == 0, "virtual loss stranded on an abandoned path"
    assert audit["conservation_failures"] == 0, (
        f"the degraded tree does not conserve visits: {audit}")


@requires_dump
def test_a_generous_arena_delivers_the_whole_budget_and_flags_nothing(dump):
    """The control for the test above. A mechanism that fired unconditionally
    would pass every assertion there while breaking every search."""
    search = _search(dump, 1_200_000)
    search.search_parallel(4000, guofish_core.ParallelConfig(**PARALLEL))
    stats = search.parallel_stats()

    assert stats["arena_exhausted"] is False
    assert stats["arena_exhausted_at"] == 0
    assert stats["delivered"] == stats["requested"]
    assert 0 < stats["arena_high_water"] < stats["arena_capacity"]


@requires_dump
def test_a_degraded_search_can_be_continued_after_the_arena_is_recycled(dump):
    """Degradation must not poison the tree for the rest of the game.

    The nodes that failed to expand were left Unexpanded — not marked
    expanded-with-no-children, which is the terminal/expanded confusion C6 made
    unrepresentable — so a later search with room available simply expands them.
    """
    search = _search(dump, 40_000)
    search.search_parallel(4000, guofish_core.ParallelConfig(**PARALLEL))
    assert search.parallel_stats()["arena_exhausted"] is True
    degraded_visits = search.root_visits

    # A fresh position recycles the arena, exactly as a new game would.
    search.set_position(dump["fen"])
    search.search_parallel(1000, guofish_core.ParallelConfig(**PARALLEL))
    stats = search.parallel_stats()
    assert stats["arena_exhausted"] is False
    assert stats["delivered"] == stats["requested"]
    assert search.root_visits == 1000
    assert degraded_visits > 0


@requires_dump
def test_a_benchmark_harness_rejects_a_degraded_row_rather_than_publishing_it(dump):
    """The second half of the graceful-degradation acceptance criterion.

    `delivered == requested` is the admission test every C10 BENCH table
    applies. C11c makes a short row legitimate, which is precisely why the
    outcome has to carry a REASON: a harness that only saw the shortfall could
    not tell a degraded search from a broken one, and the safe reading of an
    unexplained short row is to reject it.
    """
    def admit(stats: dict) -> tuple[bool, str]:
        """The rule a BENCH harness applies. Extracted here so the test asserts
        the rule and not a paraphrase of it."""
        if stats["arena_exhausted"]:
            return False, (f"arena exhausted at {stats['arena_exhausted_at']} "
                           f"nodes; delivered {stats['delivered']} of "
                           f"{stats['requested']}")
        if stats["delivered"] != stats["requested"]:
            return False, "delivered != requested with no reason given"
        return True, "delivered == requested"

    degraded = _search(dump, 40_000)
    degraded.search_parallel(4000, guofish_core.ParallelConfig(**PARALLEL))
    admitted, why = admit(degraded.parallel_stats())
    assert not admitted, "a degraded row was published as if it were sound"
    assert "arena exhausted" in why

    healthy = _search(dump, 1_200_000)
    healthy.search_parallel(4000, guofish_core.ParallelConfig(**PARALLEL))
    admitted, why = admit(healthy.parallel_stats())
    assert admitted, f"a sound row was rejected: {why}"


@requires_dump
def test_the_arena_high_water_tracks_the_predicted_sixty_nodes_per_simulation(dump):
    """REQUIREMENT 3d: "Verify the formula rather than trusting it."

    The assertion is deliberately one-sided and loose. What matters is that the
    60 nodes/sim prediction is an OVER-estimate — a safety factor that turned
    out to be a shortfall is the failure mode — so the test pins the upper bound
    and reports the measured ratio. The lower bound only catches a high-water
    figure that has stopped being read at all.
    """
    search = _search(dump, 4_000_000)
    sims = 8000
    search.search_parallel(sims, guofish_core.ParallelConfig(**PARALLEL))
    stats = search.parallel_stats()
    assert stats["arena_exhausted"] is False

    per_sim = stats["arena_high_water"] / stats["delivered"]
    assert 5.0 < per_sim < 60.0, (
        f"{per_sim:.1f} nodes/sim measured at {sims} sims. The arena formula "
        f"budgets 60 nodes/sim; a measurement at or above that means the 1.5x "
        f"safety factor has been consumed and the sizing needs revisiting.")


# ---------------------------------------------------------------------------
# Layer 3 — the UCI state machine, through a real subprocess pipe
# ---------------------------------------------------------------------------


def _why_no_engine() -> str | None:
    """The reason the protocol layer cannot run here, or None.

    Checked in the order a failure would be confusing in: a missing dependency
    first, then the GPU the v6 surface has no CPU fallback for, then the
    checkpoint.
    """
    try:
        import chess  # noqa: F401
    except ImportError as exc:
        return f"python-chess is not importable ({exc})"
    try:
        import torch
    except ImportError as exc:
        return f"torch is not importable ({exc})"
    if not torch.cuda.is_available():
        return ("CUDA is not available; the v6 surface runs the graphed CUDA "
                "evaluator and has no measured CPU path")
    if not (playv6.DEFAULT_MODEL).is_file():
        return f"the default checkpoint {playv6.DEFAULT_MODEL} is absent"
    return None


ENGINE_UNAVAILABLE = _why_no_engine()
requires_engine = pytest.mark.skipif(ENGINE_UNAVAILABLE is not None,
                                     reason=str(ENGINE_UNAVAILABLE))

# The first `isready` pays the checkpoint load and the CUDA graph capture.
LOAD_TIMEOUT = 300.0
GO_TIMEOUT = 90.0


class Pipe:
    """A UCI engine over real pipes, with BOTH streams drained continuously.

    Modelled on `tools/uci_conform_c11.py`'s EngineProcess and reimplemented
    here rather than imported so that the test suite does not depend on a
    `tools/` script staying importable. One reader thread per stream for the
    process's whole life: a `read_until` that started its own reader would race
    every other one for the same file object and lose whatever the loser had
    already buffered — and the lost line is always the `bestmove`.

    stderr is drained for a harder reason than tidiness. The wrapper writes a
    configuration block per `isready` and a telemetry line per move, and a full
    pipe buffer would block the engine mid-search. That hang would look exactly
    like the engine bug these tests exist to detect.
    """

    def __init__(self, *args: str):
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(ENGINE), *args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", cwd=str(REPO_ROOT), bufsize=1)
        self.stderr_lines: list[str] = []
        self._out: "queue.Queue[str | None]" = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stdout(self) -> None:
        for raw in self.proc.stdout:
            self._out.put(raw.rstrip("\n"))
        self._out.put(None)

    def _drain_stderr(self) -> None:
        for raw in self.proc.stderr:
            with self._lock:
                self.stderr_lines.append(raw.rstrip("\n"))

    def stderr_text(self) -> str:
        with self._lock:
            return "\n".join(self.stderr_lines)

    def stderr_mark(self) -> int:
        with self._lock:
            return len(self.stderr_lines)

    def stderr_since(self, mark: int) -> str:
        with self._lock:
            return "\n".join(self.stderr_lines[mark:])

    def send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, predicate, timeout: float) -> list[str]:
        collected: list[str] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"nothing matched within {timeout:.1f}s; saw "
                    f"{len(collected)} line(s), last: {collected[-5:]}")
            try:
                line = self._out.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            if line is None:
                raise TimeoutError(f"engine stdout closed; saw {collected[-5:]}")
            collected.append(line)
            if predicate(line):
                return collected

    def bestmove(self, timeout: float = GO_TIMEOUT) -> tuple[str, list[str]]:
        lines = self.read_until(lambda l: l.startswith("bestmove"), timeout)
        for line in lines:
            if line.startswith("bestmove"):
                return line.split()[1], lines
        raise AssertionError("unreachable")

    def drain_stdout(self, quiet_for: float = 0.35) -> list[str]:
        """Whatever is queued, once nothing new has arrived for `quiet_for`.

        Used to assert the ABSENCE of output — that a pondering engine has not
        sent a `bestmove` — which is the one thing `read_until` cannot express.
        """
        seen: list[str] = []
        while True:
            try:
                line = self._out.get(timeout=quiet_for)
            except queue.Empty:
                return seen
            if line is None:
                return seen
            seen.append(line)

    def close(self, grace: float = 30.0) -> int:
        try:
            self.send("quit")
        except (BrokenPipeError, OSError):
            pass
        try:
            return self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return -9


def _boot(*args: str) -> Pipe:
    pipe = Pipe(*args)
    pipe.send("uci")
    pipe.read_until(lambda l: l.strip() == "uciok", 90.0)
    pipe.send("isready")
    pipe.read_until(lambda l: l.strip() == "readyok", LOAD_TIMEOUT)
    return pipe


@pytest.fixture(scope="module")
def engine():
    """ONE engine subprocess for the whole protocol layer.

    Each start pays a checkpoint load and a CUDA graph capture, so a fixture per
    test would spend minutes proving the same startup path works. The tests
    below are written to leave the engine in a known state — every one of them
    ends with a `bestmove` consumed and no search in flight — so sharing is safe
    and a test that broke that would surface as the NEXT test timing out, which
    is a failure worth having rather than one to design away.

    BOOK AND SYZYGY OFF. Both default on and both bypass MCTS, and half the
    assertions here are statements about the search: that a ponder delivered
    simulations, that a hit retained them. With the book on, the Ruy Lopez these
    tests drive is exactly the line a book covers.
    """
    if ENGINE_UNAVAILABLE is not None:                  # pragma: no cover
        pytest.skip(ENGINE_UNAVAILABLE)
    pipe = _boot("--no-book", "--no-syzygy", "--ponder")
    try:
        yield pipe
    finally:
        if pipe.proc.poll() is None:
            pipe.close()
        if pipe.proc.poll() is None:                    # pragma: no cover
            pipe.proc.kill()


def _position(engine, moves=OPENING) -> None:
    engine.send("position startpos moves " + " ".join(moves))


def _legal(uci: str, moves=OPENING) -> bool:
    import chess
    board = chess.Board()
    for played in moves:
        board.push(chess.Move.from_uci(played))
    return uci != "0000" and chess.Move.from_uci(uci) in board.legal_moves


def _info_strings(lines: list[str]) -> list[str]:
    return [l for l in lines if l.startswith("info string")]


def _field(lines: list[str], key: str):
    """The last value of `key=` across the info strings, or None."""
    found = None
    for line in _info_strings(lines):
        for token in line.split():
            if token.startswith(key + "="):
                found = token[len(key) + 1:]
    return found


# --- legal transitions -----------------------------------------------------


@requires_engine
def test_go_ponder_does_not_answer_until_the_gui_speaks(engine):
    """THE ONE PROTOCOL RULE THE WHOLE STATE MACHINE IS SHAPED BY.

    An engine that is pondering must not send `bestmove` until `ponderhit` or
    `stop`. Not when the simulation ceiling is reached, not when the arena
    fills, not when a mate is found. This asserts the absence of output, which
    is why it drains rather than reads.
    """
    _position(engine)
    engine.send("go ponder wtime 30000 btime 30000 winc 300 binc 300")
    quiet = engine.drain_stdout(quiet_for=1.5)
    assert not any(l.startswith("bestmove") for l in quiet), (
        f"the engine answered a `go ponder` before the GUI spoke: {quiet}")

    engine.send("stop")
    uci, _ = engine.bestmove()
    assert _legal(uci), uci


@requires_engine
def test_ponderhit_retains_the_ponder_simulations_and_counts_only_post_hit_work(engine):
    """ACCEPTANCE: "after `go ponder` -> `ponderhit`, the tree retains its ponder
    simulations and the reported `search_sims` counts only post-hit work."

    `inherited` is the root's visit count when the post-hit search STARTED, so
    `inherited >= ponder_sims` is the tree having survived the conversion. If
    `ponderhit` had stopped and restarted the search, `inherited` would be 1.
    """
    _position(engine)
    engine.send("go ponder wtime 30000 btime 30000 winc 300 binc 300")
    time.sleep(1.0)
    engine.send("ponderhit")
    uci, lines = engine.bestmove()

    assert _legal(uci), uci
    ponder_sims = int(_field(lines, "ponder_sims"))
    search_sims = int(_field(lines, "search_sims"))
    total_sims = int(_field(lines, "total_sims"))
    inherited = int(_field(lines, "inherited"))

    assert ponder_sims > 0, "the ponder phase delivered nothing in 1.0s"
    assert search_sims > 0, "the post-hit search delivered nothing"
    assert total_sims == ponder_sims + search_sims
    assert inherited >= ponder_sims, (
        f"the tree was NOT retained across the ponderhit: the post-hit search "
        f"inherited {inherited} visits after a ponder of {ponder_sims}")
    # `search_sims` is post-hit work only. It is the `nodes` a GUI sees and the
    # numerator of the reported nps.
    assert search_sims < total_sims


@requires_engine
def test_a_ponder_miss_discards_the_tree_and_starts_the_next_search_clean(engine):
    """ACCEPTANCE: "`stop` -> new `position` -> `go` yields a legal move, no
    leak, no stale-tree contamination. Assert the new search's root visit count
    begins from a clean state."

    The discontinuity is deliberately a DIFFERENT line rather than a longer one:
    `Engine.set_position` extends the tree when the new move list is the old one
    plus moves, which is right and is not what a miss looks like. A miss is the
    opponent having played something else, so the new line diverges and the tree
    must be rebuilt.
    """
    _position(engine)
    engine.send("go ponder wtime 30000 btime 30000")
    time.sleep(0.8)
    engine.send("stop")
    uci, _ = engine.bestmove()
    assert _legal(uci), uci

    # A line that diverges from the pondered one at ply 2.
    missed = ["e2e4", "c7c5", "g1f3", "d7d6"]
    engine.send("position startpos moves " + " ".join(missed))
    engine.send("go nodes 900")
    uci, lines = engine.bestmove()

    assert _legal(uci, missed), uci
    inherited = int(_field(lines, "inherited"))
    assert inherited == 0, (
        f"the search after a ponder miss inherited {inherited} visits; the "
        f"pondered tree contaminated a position it does not describe")
    delivered = int(_field(lines, "delivered"))
    assert delivered == 899, (
        f"a clean root should deliver budget-1 simulations, got {delivered}")
    assert int(_field(lines, "ponder_sims")) == 0


@requires_engine
def test_a_pondering_engine_emits_no_info_lines(engine):
    """Requirement 5: suppress or throttle `info` during ponder.

    Defence in depth rather than correctness — the switch-interval fix is
    already mandatory — but info formatting is pure-Python work holding the GIL,
    GUIs discard ponder info, and it buys nothing. Asserted on the search phase
    only: the `bestmove` that answers the eventual `stop` is allowed its final
    line, because by then the ponder is over.
    """
    _position(engine)
    engine.send("go ponder wtime 30000 btime 30000")
    during = engine.drain_stdout(quiet_for=1.5)
    assert not during, f"the engine emitted output while pondering: {during}"
    engine.send("stop")
    engine.bestmove()


# --- illegal transitions ---------------------------------------------------


@requires_engine
def test_ponderhit_with_no_ponder_in_flight_is_answered_not_acted_on(engine):
    """Requirement 4's first illegal transition.

    Answered LOUDLY and ignored. Acting on it would apply a clock the GUI never
    offered to a search that is not a ponder; dropping it silently would hide a
    GUI bug. The engine must still work afterwards, which the `go` below checks.
    """
    mark = engine.stderr_mark()
    engine.send("ponderhit")
    time.sleep(0.5)
    text = engine.stderr_since(mark)
    assert "ponderhit" in text and "no ponder in flight" in text, text
    assert not engine.drain_stdout(quiet_for=0.3), (
        "a stray ponderhit produced protocol output")

    _position(engine)
    engine.send("go nodes 600")
    uci, _ = engine.bestmove()
    assert _legal(uci), uci


@requires_engine
def test_ponderhit_arriving_immediately_after_go_ponder_is_not_lost(engine):
    """The stdin-ordering race, which is the one an in-process test cannot see.

    The main thread is an unbounded distance behind stdin, so a `ponderhit` sent
    with no delay can arrive before the ponder has begun. C11's conformance run
    already tests `stop` IMMEDIATELY after `go` for the same reason; pondering
    adds a second flag that has to survive the same ordering, and the fix is
    that the READER sets the pondering state when it sees the `go ponder` line.

    A dropped ponderhit here would show as a hang, not as a wrong answer.
    """
    _position(engine)
    engine.send("go ponder wtime 8000 btime 8000")
    engine.send("ponderhit")
    started = time.perf_counter()
    uci, lines = engine.bestmove()
    took = time.perf_counter() - started

    assert _legal(uci), uci
    # 8 s on the clock over 30 nominal moves is ~270 ms; the bound is loose
    # enough for a slow first slice and far below the ponder ceiling a dropped
    # ponderhit would have run to.
    assert took < 20.0, f"{took:.1f}s — the ponderhit looks to have been dropped"
    assert int(_field(lines, "search_sims")) > 0


@requires_engine
def test_stop_during_ponder_answers_promptly(engine):
    """`stop` on the miss critical path. C11's baseline was 7-109 ms.

    This now sits on the path where the opponent has ALREADY moved and the
    engine is burning its own clock until it answers, which is why C11c aborts
    the running slice through the mutable deadline instead of waiting the slice
    out. `tools/bench_c11c_ponder.py` reports the distribution; this asserts the
    ceiling.
    """
    _position(engine)
    engine.send("go ponder wtime 30000 btime 30000")
    time.sleep(0.8)
    started = time.perf_counter()
    engine.send("stop")
    uci, _ = engine.bestmove()
    took = time.perf_counter() - started

    assert _legal(uci), uci
    assert took < 1.0, f"stop during ponder took {took * 1000:.0f} ms"


@requires_engine
def test_a_position_arriving_mid_ponder_is_answered_and_ends_the_ponder(engine):
    """Requirement 8: ponder must not survive a position discontinuity.

    The `go ponder` is still owed exactly one `bestmove` — the GUI discards it —
    and the `position` must then be applied. Without this the command would sit
    in the queue until the ponder spent its whole simulation ceiling, which is a
    hang of seconds that only a piped test can observe.
    """
    _position(engine)
    engine.send("go ponder wtime 30000 btime 30000")
    time.sleep(0.6)
    mark = engine.stderr_mark()
    started = time.perf_counter()
    engine.send("position startpos moves e2e4 e7e5")
    uci, _ = engine.bestmove()
    took = time.perf_counter() - started

    assert _legal(uci), uci
    assert took < 2.0, f"the mid-ponder position took {took:.1f}s to answer"
    assert "mid-ponder" in engine.stderr_since(mark)

    # And the new position is the one now in force.
    engine.send("go nodes 700")
    uci, _ = engine.bestmove()
    assert _legal(uci, ["e2e4", "e7e5"]), uci


@requires_engine
def test_ucinewgame_mid_ponder_is_answered_and_discards_the_tree(engine):
    """Requirement 8's other half. `ucinewgame` drops the tree AND the cache, so
    a ponder still running inside the search would be reading both."""
    _position(engine)
    engine.send("go ponder wtime 30000 btime 30000")
    time.sleep(0.6)
    started = time.perf_counter()
    engine.send("ucinewgame")
    uci, _ = engine.bestmove()
    assert _legal(uci), uci
    assert time.perf_counter() - started < 2.0

    engine.send("position startpos moves e2e4 e7e5 g1f3")
    engine.send("go nodes 700")
    uci, lines = engine.bestmove()
    assert _legal(uci, ["e2e4", "e7e5", "g1f3"]), uci
    assert int(_field(lines, "inherited")) == 0, (
        "ucinewgame did not discard the pondered tree")


@requires_engine
def test_exactly_one_bestmove_is_emitted_per_go_including_on_a_miss(engine):
    """Requirement 4: "`bestmove` is emitted exactly once per `go`, including on
    a miss where the GUI discards it."

    Two `bestmove` lines for one `go` desynchronise a GUI for the rest of the
    game, and the failure is silent until the next move is attributed to the
    wrong position.
    """
    _position(engine)
    engine.send("go ponder wtime 30000 btime 30000")
    time.sleep(0.5)
    engine.send("stop")
    uci, lines = engine.bestmove()
    assert _legal(uci), uci
    assert sum(1 for l in lines if l.startswith("bestmove")) == 1

    # A second `stop` after the move is already out must add nothing.
    engine.send("stop")
    extra = engine.drain_stdout(quiet_for=0.5)
    assert not any(l.startswith("bestmove") for l in extra), (
        f"a redundant stop produced a second bestmove: {extra}")


@requires_engine
def test_isready_is_answered_while_pondering(engine):
    """A GUI sends `isready` freely and blocks on `readyok`. During a ponder the
    main thread is inside the search, so the reader thread has to answer — the
    same path C11 built for `isready` under `go infinite`, exercised here under
    the unbounded search pondering introduces."""
    _position(engine)
    engine.send("go ponder wtime 30000 btime 30000")
    time.sleep(0.4)
    started = time.perf_counter()
    engine.send("isready")
    engine.read_until(lambda l: l.strip() == "readyok", 10.0)
    assert time.perf_counter() - started < 5.0

    engine.send("stop")
    uci, _ = engine.bestmove()
    assert _legal(uci), uci


@requires_engine
def test_the_ponder_ceiling_comes_from_ponder_max_sims(engine):
    """Requirement 3b: the cap is the PRIMARY bound on an unbounded search,
    with graceful arena exhaustion as the backstop rather than the mechanism.

    `PonderMaxSims` is set to a value NOTHING ELSE IN THE CONFIGURATION COULD
    PRODUCE, which is the only way to prove which knob is in force. At the
    shipping defaults the computed cap happens to equal `SimCap` — both derive
    from `sims_per_move` at `ponder_decay = 1.0` — so a test written against the
    defaults would pass with either bound wired up.

    Read off the engine's own `[go]` plan line rather than by counting
    simulations: the point is which number the engine CHOSE, and a ponder that
    stopped early for an unrelated reason would satisfy a count-based check
    while the wrong bound was in force.
    """
    probe = 1234
    engine.send(f"setoption name PonderMaxSims value {probe}")
    try:
        mark = engine.stderr_mark()
        _position(engine)
        engine.send("go ponder wtime 30000 btime 30000")
        time.sleep(0.4)
        engine.send("stop")
        engine.bestmove()

        plan = [l for l in engine.stderr_since(mark).splitlines()
                if l.startswith("[go] ponder:")]
        assert plan, engine.stderr_since(mark)
        # The CEILING CLAUSE ONLY. Matching against the whole line would also
        # see `root at N`, and N is whatever the shared engine's tree has
        # accumulated by this point in the module — so a root that happened to
        # sit at 60,000 visits would fail this for a reason that has nothing to
        # do with which knob set the bound.
        ceiling = plan[0].split("up to ")[1].split(" new sims")[0]
        assert ceiling == str(probe), (
            f"the ponder ceiling was {ceiling}, not the PonderMaxSims "
            f"{probe} that was set. SimCap is {EngineConfig.sim_cap}. Full "
            f"plan line: {plan[0]}")
    finally:
        # 0 is the UCI spelling of "unset", which restores the computed
        # default for every test after this one.
        engine.send("setoption name PonderMaxSims value 0")


@requires_engine
def test_quit_during_a_ponder_exits_cleanly():
    """`quit` must not be swallowed by the ponder's idle wait.

    ITS OWN PROCESS, because the test consumes it. Using the shared fixture
    would make every other test in the module depend on this one running last,
    which is true today only because nothing reorders the file — a dependency
    on collection order is not a thing to leave lying around, and it costs one
    checkpoint load to remove.

    A hang here is the failure C11's deadlock taught this project to test for
    through a pipe: the process still exists, still holds the GPU, and the
    harness waits forever.
    """
    if ENGINE_UNAVAILABLE is not None:                  # pragma: no cover
        pytest.skip(ENGINE_UNAVAILABLE)
    pipe = _boot("--no-book", "--no-syzygy", "--ponder")
    try:
        _position(pipe)
        pipe.send("go ponder wtime 30000 btime 30000")
        time.sleep(0.5)
        started = time.perf_counter()
        code = pipe.close(grace=30.0)
        took = time.perf_counter() - started

        assert code == 0, f"exit code {code} after quit during a ponder"
        assert took < 25.0, f"quit during a ponder took {took:.1f}s"
    finally:
        if pipe.proc.poll() is None:                    # pragma: no cover
            pipe.proc.kill()


# --- the degradation path, through the pipe --------------------------------


@requires_engine
def test_an_undersized_arena_still_plays_a_legal_move_over_the_protocol():
    """The graceful-degradation criterion end to end, in its own process.

    Its own process because `ArenaCapacity` is a `SearchConfig` field and the
    engine would otherwise have to be rebuilt mid-fixture, and because a
    deliberately broken engine has no business being shared with the tests that
    assert healthy behaviour.

    32,768 nodes is ~1,000 simulations at the measured ~32 nodes/sim, so a
    2,000-node search must degrade — and must still answer.
    """
    if ENGINE_UNAVAILABLE is not None:                  # pragma: no cover
        pytest.skip(ENGINE_UNAVAILABLE)
    pipe = _boot("--no-book", "--no-syzygy", "--arena-capacity", "32768")
    try:
        _position(pipe)
        pipe.send("go nodes 2000")
        uci, lines = pipe.bestmove()

        assert _legal(uci), f"a degraded search returned {uci!r}"
        assert _field(lines, "arena_exhausted") == "true", lines
        delivered = int(_field(lines, "delivered"))
        assert 0 < delivered < 2000, (
            f"delivered {delivered}; a degraded search must deliver less than "
            f"its budget and more than nothing")

        notice = [l for l in _info_strings(lines) if "arena exhausted at" in l]
        assert notice, (
            "the degradation was silent on stdout. Requirement 3: 'A silent "
            "degradation that only shows up as unexplained weakness is worse "
            "than a crash, because nobody investigates it.'")
        assert "NOT admissible as a benchmark row" in notice[0]

        # NO EXCEPTION. The wrapper's error path prints a traceback and an
        # `UNSEARCHED` info line; both absent means the search returned
        # normally rather than being rescued.
        assert "Traceback" not in pipe.stderr_text()
        assert not any("UNSEARCHED" in l for l in lines)

        # AND IT PLAYS ON. A degraded move must not end the game.
        pipe.send("position startpos moves " + " ".join(OPENING + [uci]))
        pipe.send("go nodes 500")
        second, _ = pipe.bestmove()
        assert second != "0000"
    finally:
        if pipe.proc.poll() is None:
            pipe.close()
        if pipe.proc.poll() is None:                    # pragma: no cover
            pipe.proc.kill()
