"""C9 acceptance — W workers, K in flight, and the invariants that replace equality.

This chunk is the one where "looks fine" is least trustworthy, so its acceptance
is three layers with three different standards of proof, and they are not
interchangeable.

Layer 1 — regression, and it is an EQUALITY
-------------------------------------------
``W=1, K=1`` runs the whole C9 machinery — the MPSC queue, the dispatcher, the
PENDING claim, the outstanding-leaf throttle — and must still reproduce the Gate
1 golden trees bit for bit, on both corpora and at both virtual-loss magnitudes.
Visit counts equal as integers, ``value_sum`` identical as a 64-bit pattern,
priors identical as 32-bit patterns. Nothing from single-threaded C5/C6 is
allowed to regress, and because the descent state had to be split out of the
search object to make W threads possible, this is also the test of that split.

``synthetic_evaluations`` is asserted to be 0 for every layer-1 run. See below.

Layer 2 — reproducibility, and it is an EQUALITY BETWEEN TWO C++ RUNS
---------------------------------------------------------------------
``W=1, K=8`` run twice must produce bit-identical trees. Descent is
single-threaded, the queue is FIFO, and the dispatcher drains only when the
worker cannot proceed, so the interleaving is a function of the search state
rather than of the scheduler. If this is not reproducible there is a race, and
the C9 brief is explicit that nothing proceeds until it passes: a race that shows
at W=1, K=8 is far worse at W=8, K=8 and far harder to find there.

``test_layer2_the_harness_has_teeth`` establishes that this is not vacuous, by
showing that the same comparison FAILS at W=4 — where scheduling really does
decide the tree.

Layer 3 — conservation, because equality is not available
---------------------------------------------------------
Under the production configuration the run-to-run tree is not reproducible and
should not be: which thread reaches a leaf first determines where virtual loss
sits, and therefore the tree's shape. So the assertions become exact invariants
instead:

* every expanded node has ``visits == 1 + sum(children visits)``;
* the total virtual-loss count over the whole arena returns to exactly 0;
* delivered simulations equal the requested budget exactly;
* the root visit distribution is stable across 10 runs within a stated tolerance.

The first three are EXACT, with no epsilon anywhere, and that is only possible
because C9 moves ``value_sum`` to the Q32 ``atomic<int64>`` accumulator and
because virtual loss is an integer count applied at read time. Under a
floating-point accumulator "the subtree sums match" would be a statement about
rounding.

Why layers 2 and 3 use a stand-in evaluator, and layer 1 does not
-----------------------------------------------------------------
The Gate 1 dump holds exactly the positions the SERIAL Python reference
evaluated. That is what makes a dump miss the strongest test in C5 — the search
can only stay inside the dump if it walks the same tree. C9 breaks the premise on
purpose: with K in-flight paths, virtual loss steers descents onto branches the
serial reference never opened, which is leaf parallelism working rather than
failing. Measured here: the first miss arrives about five plies in at W=1, K=8.

Regenerating a dump wide enough is not available, because the set of positions a
parallel search reaches depends on the scheduling of the run under test — so
producing it would mean running the implementation under test to decide what the
reference should contain, which is circular in exactly the way Global Rule 2
exists to prevent.

So layer 1, the only layer that compares against Python, runs with the fallback
OFF and asserts the counter is 0. Layers 2 and 3 assert reproducibility and
conservation, which are not claims about what the network said, and run with it
on. Real dump entries are still used wherever they exist; the tests report how
many holes were filled.

Environment overrides (Amendment B)
-----------------------------------
``GUOFISH_GOLDEN_GATE1_*`` for the quiet files and ``GUOFISH_GOLDEN_C6_*`` for
the terminal ones, shared with C5 and C6, so a mutation drill runs against
corrupted copies in a scratch directory and ``golden/`` is never written to.

Nothing here imports ``chess`` or ``torch``. The reference reaches this file only
through the golden files.
"""

import json
import os
import statistics
from pathlib import Path

import numpy as np
import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent

QUIET = {
    "trees": (REPO_ROOT / "golden" / "gate1_trees.npz", "GUOFISH_GOLDEN_GATE1_TREES"),
    "dump": (REPO_ROOT / "golden" / "gate1_dump.npz", "GUOFISH_GOLDEN_GATE1_DUMP"),
    "manifest": (REPO_ROOT / "golden" / "gate1_manifest.json",
                 "GUOFISH_GOLDEN_GATE1_MANIFEST"),
}
TERMINAL = {
    "trees": (REPO_ROOT / "golden" / "gate1_terminal_trees.npz", "GUOFISH_GOLDEN_C6_TREES"),
    "dump": (REPO_ROOT / "golden" / "gate1_terminal_dump.npz", "GUOFISH_GOLDEN_C6_DUMP"),
    "manifest": (REPO_ROOT / "golden" / "gate1_terminal_manifest.json",
                 "GUOFISH_GOLDEN_C6_MANIFEST"),
}

REQUIRED_VIRTUAL_LOSSES = (0.0, 2.5)

# Layer 3's stability criterion, and it is a RATIO rather than an absolute bound.
#
# The obvious formulation — "the top move's share moves less than X points across
# ten runs" — is not measurable in C9 and picking an X that happened to pass
# would be the worst kind of green. Two things defeat it:
#
#   1. On a contested position the top two root moves are within a few percent of
#      each other, so which one leads after 2,000 simulations is decided by a few
#      hundred visits and moves run to run. That is correct MCTS behaviour under
#      any parallelism, not a defect.
#   2. C9 runs on the replay evaluator, and a parallel search leaves the Python
#      dump's coverage almost immediately, so 10-30% of its leaves are answered by
#      the stand-in evaluator. Two runs that open different branches therefore
#      disagree partly because they consulted a different evaluator, not because
#      of concurrency. Measured (tools/bench_c9.py): run-to-run distance tracks
#      the stand-in's SHARE far more closely than it tracks the simulation count.
#
# What IS measurable, and is the substantive question, is whether the answer is
# determined by the configuration or by the scheduler. So the criterion compares
# two distances on the same footing:
#
#   run-to-run     how far two runs of the SAME configuration land from each other
#   vs serial      how far this configuration lands from W=1/K=1, its own serial
#                  ground truth
#
# and requires the first to be comfortably smaller than the second. It says: the
# tree this configuration builds is a property of its virtual-loss exposure, and
# scheduling perturbs it by less than that exposure already does. Both distances
# carry the stand-in contamination roughly equally, so the ratio is far more
# robust than either number alone.
#
# Measured by this test at W=4/K=8 over 6 quiet positions x 6 repeats: run-to-run
# 3.9%, vs-serial 25.3%, ratio 0.15. BENCH.md's grid reports the same two
# quantities over 8 positions and quotes a larger mean run-to-run figure (11.3%)
# because a couple of its positions are contested enough for the top two moves to
# trade places; the RATIO is what is asserted here precisely because it survives
# that, and the tolerance is set well above both.
ROOT_STABILITY_RATIO = 0.75

# The total-variation distance between root visit distributions at which the two
# would be describing different searches. Python's serial 58/28/11/3 flattening
# to 37/29/23/11 at 32 outstanding is a distance of 21 points, and that is the
# effect the whole W x K sizing exercise is about — so a run-to-run distance at
# or above it would mean scheduling alone moves the answer as much as the entire
# parallelism decision does.
PYTHON_FLATTENING_LANDMARK = 0.21

# The sweep the brief asks to be reported, with W=1/K=1 as the reference row.
SWEEP_W = (2, 4, 6, 8)
SWEEP_K = (4, 8, 16)


# ---------------------------------------------------------------------------
# Golden loading — the same accessors C5 and C6 use
# ---------------------------------------------------------------------------


def _path(spec):
    default, env = spec
    return Path(os.environ.get(env, default))


def _load_json(spec, what):
    path = _path(spec)
    if not path.exists():
        pytest.fail(
            f"{what} missing: {path}\n"
            "Global Rule 2: regenerate with `python tools/gen_gate1_golden.py`, which "
            "runs the Python reference. It is never produced from C++ output.")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_npz(spec, what):
    path = _path(spec)
    if not path.exists():
        pytest.fail(
            f"{what} missing: {path}\n"
            "Global Rule 2: regenerate with `python tools/gen_gate1_golden.py`, which "
            "runs the Python reference.")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def corpora():
    return {
        "quiet": {
            "manifest": _load_json(QUIET["manifest"], "Gate 1 quiet manifest"),
            "dump": _load_npz(QUIET["dump"], "Gate 1 quiet replay dump"),
            "trees": _load_npz(QUIET["trees"], "Gate 1 quiet reference trees"),
        },
        "terminal": {
            "manifest": _load_json(TERMINAL["manifest"], "C6 terminal manifest"),
            "dump": _load_npz(TERMINAL["dump"], "C6 terminal replay dump"),
            "trees": _load_npz(TERMINAL["trees"], "C6 terminal reference trees"),
        },
    }


def _golden_run(trees, index):
    """The per-node arrays for one recorded run, sliced out of the CSR block."""
    begin = int(trees["run_offset"][index])
    end = int(trees["run_offset"][index + 1])
    out = {name: trees[name][begin:end]
           for name in ("depth", "move", "visits", "value_sum", "prior", "children")}
    size = end - begin
    out["terminal"] = (trees["terminal"][begin:end] if "terminal" in trees
                       else np.zeros(size, dtype=np.uint8))
    out["terminal_value"] = (trees["terminal_value"][begin:end] if "terminal_value" in trees
                             else np.zeros(size, dtype=np.float32))
    return out


def _history(corpus, run):
    """The pre-root game history for a run, from the POSITION it belongs to.

    It lives on the position rather than on the run because it is a property of
    the position — several runs (two virtual-loss magnitudes, full-tree and not)
    share one. The threefold cases are the ones that care: their root repetition
    history is primed by a shuffle, so a search handed an empty history counts
    one occurrence where the reference counts three, never claims the draw, and
    walks off into positions the reference never evaluated. That surfaces as a
    replay-dump miss several plies deeper than the actual mistake.
    """
    positions = corpus["manifest"].get("positions") or []
    index = run.get("position")
    if index is not None and index < len(positions):
        return positions[index].get("history", [])
    return run.get("history", [])


def _capacity(corpus):
    """Arena capacity from the GOLDEN data, not from a guess.

    Every visited node is expanded and an expansion allocates exactly its child
    count, so ``1 + sum(children)`` is the exact node total for a serial run. The
    parallel runs explore differently and can allocate more, so layers 2 and 3
    get headroom on top; layer 1 gets the exact figure, which is what makes an
    over-allocating C++ expansion show up as a failure rather than being absorbed
    by slack.
    """
    trees, manifest = corpus["trees"], corpus["manifest"]
    return max(1 + int(_golden_run(trees, i)["children"].sum())
               for i in range(len(manifest["runs"])))


def _build(corpus, virtual_loss, capacity, *, accumulator="double", synthetic=False,
           cache_slots=0, max_tree_depth=80):
    # `max_tree_depth` is per-RUN in the terminal corpus, not per-corpus: the
    # depth-cap positions were generated at a cap of 6 or 7 because the cap is
    # unreachable at 80 in 5,000 simulations (repetition or the fifty-move rule
    # fires first). A run at 7 executes the same code a run at 80 would, so the
    # cap has to come from the manifest rather than from the default — and a test
    # that used the default here would not fail cleanly, it would fail as a
    # replay-dump miss several plies deeper than the reference ever went.
    config = guofish_core.SearchConfig(virtual_loss=virtual_loss, arena_capacity=capacity,
                                       cache_slots=cache_slots,
                                       max_tree_depth=max_tree_depth)
    cls = (guofish_core.ReplaySearchDouble if accumulator == "double"
           else guofish_core.ReplaySearchQ32)
    search = cls(config)
    dump = corpus["dump"]
    search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                     dump["moves"], dump["priors"], dump["values"])
    search.synthetic_fallback = synthetic
    return search


def _tree_bits(search, min_visits=1):
    """The comparison surface: ints as ints, floats as bit patterns.

    ``==`` on floats is the wrong operator twice over — it calls +0.0 and -0.0
    equal and no NaN equal to itself — and a difference of one ulp prints
    identically in decimal, so the bit pattern is what a failure has to report.
    """
    arrays = search.dump_tree_arrays(min_visits)
    return {
        "depth": arrays["depth"],
        "move": arrays["move"],
        "visits": arrays["visits"],
        "value_sum": arrays["value_sum"].view(np.uint64),
        "prior": arrays["prior"].view(np.uint32),
        "children": arrays["children"],
        "terminal": arrays["terminal"],
        "terminal_value": arrays["terminal_value"].view(np.uint32),
    }


def _paths_from_dfs(depth, move):
    """Move path from the root for every node, from the DFS depth column."""
    paths, stack = [], []
    for d, m in zip(depth, move):
        del stack[int(d):]
        stack.append(guofish_core.move_to_uci(int(m)) if int(d) > 0 else "(root)")
        paths.append(" ".join(stack[1:]) if len(stack) > 1 else "(root)")
    return paths


def _first_divergence(label_a, a, label_b, b):
    """The first differing node in DFS order, with the path and both values.

    A bare "trees differ" is useless, and the C5 brief says so. This is the same
    reporting discipline, reused: locate the node, name the field, print both
    sides including raw bit patterns for the float columns.
    """
    n = min(len(a["visits"]), len(b["visits"]))
    paths = _paths_from_dfs(a["depth"][:n], a["move"][:n])
    for i in range(n):
        for field in ("depth", "move", "visits", "children", "terminal",
                      "value_sum", "prior", "terminal_value"):
            av, bv = a[field][i], b[field][i]
            if av != bv:
                extra = ""
                if field in ("value_sum", "prior", "terminal_value"):
                    dtype = np.float64 if field == "value_sum" else np.float32
                    extra = (f"\n    as float: {label_a}={np.array([av]).view(dtype)[0]!r} "
                             f"{label_b}={np.array([bv]).view(dtype)[0]!r}")
                return (f"first divergence at DFS node {i}\n"
                        f"  path   : {paths[i]}\n"
                        f"  field  : {field}\n"
                        f"  {label_a:<7}: {av}\n"
                        f"  {label_b:<7}: {bv}{extra}")
    if len(a["visits"]) != len(b["visits"]):
        return (f"node counts differ: {label_a}={len(a['visits'])} "
                f"{label_b}={len(b['visits'])}; the first {n} nodes agree")
    return None


def _root_children_visits(search):
    arrays = search.dump_tree_arrays(0)
    return np.sort(arrays["visits"][arrays["depth"] == 1])[::-1]


def _root_distribution(search):
    """Root children's visit shares, in CANONICAL MOVE ORDER.

    Canonical order rather than sorted by visits, so two runs are compared move
    against move. Sorting first would call two completely different move sets
    identical whenever they happened to have the same shape, which is exactly the
    failure a stability test must not have.
    """
    arrays = search.dump_tree_arrays(0)
    at_depth_one = arrays["depth"] == 1
    visits = arrays["visits"][at_depth_one].astype(np.float64)
    order = np.argsort(arrays["move"][at_depth_one])
    visits = visits[order]
    total = visits.sum()
    return visits / total if total > 0 else visits


def _total_variation(p, q):
    n = min(len(p), len(q))
    return 0.5 * float(np.abs(p[:n] - q[:n]).sum())


# ---------------------------------------------------------------------------
# Layer 1 — W=1, K=1 still passes Gate 1 bit-exactly
# ---------------------------------------------------------------------------


def _layer1_cases():
    """One parameter per recorded run, read from the manifests at COLLECTION time.

    Read directly rather than through the `corpora` fixture because pytest needs
    the parameter list before fixtures exist. The alternative — parametrising over
    a fixed range and skipping the tail — was what this file did first, and it
    produced 294 skips that no report could account for. Amendment D exists
    because exactly that pattern hid a 242-test hole on Linux for a whole chunk,
    so the ids here are the real runs and nothing else.

    A missing golden file yields ONE parameter that fails with the regeneration
    instructions, rather than silently yielding zero tests.
    """
    out = []
    for name, spec in (("quiet", QUIET), ("terminal", TERMINAL)):
        path = _path(spec["manifest"])
        if not path.exists():
            out.append(pytest.param(name, -1, id=f"{name}-manifest-missing"))
            continue
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for index, run in enumerate(manifest["runs"]):
            tag = "full" if run.get("full_tree") else "vis"
            out.append(pytest.param(
                name, index,
                id=f"{name}{index}-vl{run['virtual_loss']}-{tag}"))
    return out


@pytest.mark.parametrize("corpus_name,index", _layer1_cases())
def test_layer1_w1_k1_is_bit_exact_against_gate1(corpora, corpus_name, index):
    """The whole C9 machinery, one worker, one path in flight, still bit-exact.

    This is not a re-run of C5's test with a different entry point. The descent
    state had to be lifted out of the search object into a per-thread struct to
    make W threads possible at all, and every simulation here goes through the
    queue, the dispatcher and the PENDING claim. If the split moved so much as a
    scratch buffer's lifetime, this fails.
    """
    if index < 0:
        pytest.fail(
            f"the {corpus_name} manifest is missing. Global Rule 2: regenerate with "
            "`python tools/gen_gate1_golden.py`, which runs the Python reference.")
    corpus = corpora[corpus_name]
    run = corpus["manifest"]["runs"][index]

    capacity = _capacity(corpus)
    search = _build(corpus, run["virtual_loss"], capacity, synthetic=False,
                    max_tree_depth=run.get("max_tree_depth", 80))
    search.set_position(run["fen"], _history(corpus, run))
    stats = search.search_parallel(
        int(run["sims"]),
        guofish_core.ParallelConfig(workers=1, in_flight=1,
                                    affinity="none"))

    assert stats["synthetic_evaluations"] == 0, (
        "a W=1/K=1 run left the replay dump, which means it did not walk the serial "
        "reference's tree — the whole premise of layer 1")

    golden = _golden_run(corpus["trees"], index)
    min_visits = 0 if run.get("full_tree") else 1
    got = _tree_bits(search, min_visits)
    want = {
        "depth": golden["depth"],
        "move": golden["move"],
        "visits": golden["visits"],
        "value_sum": golden["value_sum"].astype(np.float64).view(np.uint64),
        "prior": golden["prior"].astype(np.float32).view(np.uint32),
        "children": golden["children"],
        "terminal": golden["terminal"],
        "terminal_value": golden["terminal_value"].astype(np.float32).view(np.uint32),
    }
    diff = _first_divergence("python", want, "c++", got)
    assert diff is None, (
        f"{corpus_name} run {index} (vl={run['virtual_loss']}, sims={run['sims']}) "
        f"diverged from the Gate 1 reference under W=1/K=1\n{diff}")

    assert int(run["root_visits"]) == stats["root_visits"]
    assert run["best_move"] == stats["best_move"]


def test_layer1_covers_both_corpora_and_both_virtual_losses(corpora):
    """The parametrisation above is data-driven; this pins what the data must contain."""
    for name in ("quiet", "terminal"):
        runs = corpora[name]["manifest"]["runs"]
        magnitudes = {r["virtual_loss"] for r in runs}
        assert set(REQUIRED_VIRTUAL_LOSSES) <= magnitudes, (
            f"the {name} corpus does not cover both virtual-loss magnitudes: {magnitudes}")
        assert len(runs) >= 20


# ---------------------------------------------------------------------------
# Layer 2 — W=1, K>1 is bit-reproducible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("virtual_loss", REQUIRED_VIRTUAL_LOSSES)
@pytest.mark.parametrize("in_flight", (2, 8, 16))
def test_layer2_w1_multiple_in_flight_is_bit_reproducible(corpora, virtual_loss, in_flight):
    """Two runs of W=1, K>1 must produce bit-identical trees.

    This is the only clean test of the in-flight machinery. Everything that makes
    the parallel path different from the serial one is exercised — leaves are
    submitted rather than expanded in place, virtual loss survives the handoff,
    the dispatcher expands a batch in FIFO order, the PENDING claim collides with
    the worker's own earlier submissions — and yet the result is a function of
    the search state alone, because the dispatcher drains only when the worker
    cannot proceed.

    At virtual loss 0.0 the collisions are the interesting part: the second
    descent scores every child exactly as the first did, re-selects the same
    leaf, finds it PENDING, and discards. K=8 therefore degenerates to batches of
    one — visible in the reported mean batch size — and that is correct rather
    than a failure to parallelise.
    """
    corpus = corpora["quiet"]
    run = next(r for r in corpus["manifest"]["runs"]
               if r["virtual_loss"] == virtual_loss and not r.get("full_tree"))
    capacity = _capacity(corpus) * 2

    trees, reports = [], []
    for _ in range(2):
        search = _build(corpus, virtual_loss, capacity, synthetic=True)
        search.set_position(run["fen"], _history(corpus, run))
        stats = search.search_parallel(
            int(run["sims"]),
            guofish_core.ParallelConfig(workers=1, in_flight=in_flight,
                                        affinity="none"))
        trees.append(_tree_bits(search))
        reports.append((stats, search.parallel_stats(), search.audit()))

    diff = _first_divergence("run1", trees[0], "run2", trees[1])
    stats0, par0, _ = reports[0]
    assert diff is None, (
        f"W=1/K={in_flight} at virtual loss {virtual_loss} is not reproducible; that is a "
        f"race, and the C9 brief says nothing proceeds until it is fixed.\n{diff}\n"
        f"  batches={par0['batches']} mean_batch={par0['mean_batch']:.2f} "
        f"collisions={par0['select_collisions']}")

    for stats, par, audit in reports:
        assert par["delivered"] == par["requested"]
        assert audit["vloss_total"] == 0
        assert audit["conservation_failures"] == 0
        assert par["largest_batch"] <= in_flight

    print(f"\n  W=1 K={in_flight} vl={virtual_loss}: batches={par0['batches']} "
          f"mean_batch={par0['mean_batch']:.2f} largest={par0['largest_batch']} "
          f"collisions={par0['select_collisions']} "
          f"stand-in evals={stats0['synthetic_evaluations']}")


def test_layer2_the_harness_has_teeth(corpora):
    """The same comparison must FAIL at W=4, or layer 2 proves nothing.

    A reproducibility test that cannot detect irreproducibility is worse than no
    test: it reports green whether or not the machinery works. At four workers
    the interleaving really is decided by the scheduler, so two runs must differ
    — and if they do not, the workers are not running concurrently and every
    other assertion in this file is about a serial engine wearing a parallel
    name.
    """
    corpus = corpora["quiet"]
    run = next(r for r in corpus["manifest"]["runs"]
               if r["virtual_loss"] == 2.5 and not r.get("full_tree"))
    capacity = _capacity(corpus) * 3

    attempts = 5
    for _ in range(attempts):
        trees = []
        for _ in range(2):
            search = _build(corpus, 2.5, capacity, synthetic=True)
            search.set_position(run["fen"], _history(corpus, run))
            search.search_parallel(int(run["sims"]),
                                   guofish_core.ParallelConfig(workers=4, in_flight=8))
            trees.append(_tree_bits(search))
        if _first_divergence("run1", trees[0], "run2", trees[1]) is not None:
            return
    pytest.fail(
        f"{attempts} pairs of W=4/K=8 runs all produced identical trees. Either the workers "
        "are not actually running concurrently, or the dispatcher is serialising them "
        "completely — and in either case the layer 2 reproducibility test above is vacuous.")


# ---------------------------------------------------------------------------
# Layer 3 — exact conservation under the production configuration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def production_runs(corpora):
    """Ten runs of the production configuration on one quiet position.

    Built once and shared, because the assertions below are several views of the
    same ten runs and re-running them per test would cost ten seconds a test for
    no extra coverage.
    """
    corpus = corpora["quiet"]
    run = next(r for r in corpus["manifest"]["runs"]
               if r["virtual_loss"] == 2.5 and not r.get("full_tree"))
    capacity = _capacity(corpus) * 3

    out = []
    for _ in range(10):
        # Q32 IS THE POINT. The exact conservation invariants below are only
        # available because value_sum is an integer accumulator: `fetch_add` on
        # int64 is associative, so the sum does not depend on interleaving.
        search = _build(corpus, 2.5, capacity, accumulator="q32", synthetic=True)
        search.set_position(run["fen"], _history(corpus, run))
        stats = search.search_parallel(int(run["sims"]),
                                       guofish_core.ParallelConfig(workers=4, in_flight=8))
        out.append({
            "stats": stats,
            "parallel": search.parallel_stats(),
            "audit": search.audit(),
            "root_children": _root_children_visits(search),
            "requested": int(run["sims"]),
        })
    return out


def test_layer3_delivered_simulations_equal_the_budget_exactly(production_runs):
    """Not "about right" — exactly.

    Python's ``stats['simulations'] += 1`` was an unsynchronized read-modify-write
    across 32 threads over exactly this quantity, so its throughput figures could
    not be trusted to the last few percent. Here a simulation is claimed with a
    ``fetch_add`` before it starts and counted with a ``fetch_add`` when it is
    backed up, and a discarded descent hands its claim back — so the two numbers
    are equal or something is wrong.
    """
    for i, run in enumerate(production_runs):
        par, stats = run["parallel"], run["stats"]
        assert par["delivered"] == par["requested"], (
            f"run {i}: delivered {par['delivered']} of {par['requested']} requested")
        assert stats["root_visits"] == run["requested"], (
            f"run {i}: the root ended at {stats['root_visits']} visits, not {run['requested']}")
        # Every delivered simulation was resolved either by a worker (terminal,
        # depth cap) or by the dispatcher (evaluated leaf), and by nothing else.
        assert par["queued_leaves"] + par["worker_terminals"] == par["delivered"]
        assert stats["simulations"] == par["delivered"]


def test_layer3_virtual_loss_returns_to_exactly_zero_everywhere(production_runs):
    """Not "close to zero" — the counts are integers, so the total is 0 or it is a leak.

    The scan is FLAT over the arena rather than a traversal of the tree: a loss
    stranded on a node the tree no longer reaches is precisely the failure this
    is looking for, and a traversal would walk right past it.
    """
    for i, run in enumerate(production_runs):
        audit = run["audit"]
        assert audit["vloss_total"] == 0, (
            f"run {i}: {audit['vloss_total']} virtual loss left applied across "
            f"{audit['vloss_nonzero_nodes']} nodes after the search ended")
        assert audit["vloss_nonzero_nodes"] == 0


def test_layer3_subtree_visit_sums_match(production_runs):
    """visits(node) == 1 + sum(visits(children)) for every expanded node.

    The 1 is the simulation that expanded it. Nothing else can stop at an
    expanded node: the descent loop exits only at a node that is unexpanded,
    terminal or at the depth cap, and a depth-capped node is at MAX_TREE_DEPTH
    and therefore never expanded.
    """
    for i, run in enumerate(production_runs):
        audit = run["audit"]
        assert audit["conservation_failures"] == 0, (
            f"run {i}: {audit['conservation_failures']} expanded nodes do not conserve "
            f"visits; first is node {audit['first_bad_node']}, which has "
            f"{audit['first_bad_actual']} visits where its children sum to "
            f"{audit['first_bad_expected'] - 1} (+1 for the expanding simulation)")
        assert audit["root_visits"] == audit["root_children_visits"] + 1


@pytest.fixture(scope="module")
def stability_measurement(corpora):
    """Run-to-run distance against distance-from-serial, over several positions.

    Both quantities come from the same runs, on the same positions, with the same
    stand-in coverage, which is what makes their RATIO meaningful where neither
    number alone is. See ROOT_STABILITY_RATIO.
    """
    corpus = corpora["quiet"]
    cases = [r for r in corpus["manifest"]["runs"]
             if r["virtual_loss"] == 2.5 and not r.get("full_tree")][:6]
    capacity = _capacity(corpus) * 4
    sims = 2000
    repeats = 6

    rows = []
    for case in cases:
        # W=1/K=1 is deterministic, so one run IS the reference.
        reference_search = _build(corpus, 2.5, capacity, accumulator="q32", synthetic=True)
        reference_search.set_position(case["fen"], _history(corpus, case))
        reference_search.search_parallel(
            sims, guofish_core.ParallelConfig(workers=1, in_flight=1,
                                              affinity="none"))
        reference = _root_distribution(reference_search)

        distributions, best_moves, stand_in = [], [], []
        for _ in range(repeats):
            search = _build(corpus, 2.5, capacity, accumulator="q32", synthetic=True)
            search.set_position(case["fen"], _history(corpus, case))
            stats = search.search_parallel(
                sims, guofish_core.ParallelConfig(workers=4, in_flight=8))
            assert search.audit()["vloss_total"] == 0
            distributions.append(_root_distribution(search))
            best_moves.append(stats["best_move"])
            stand_in.append(stats["synthetic_evaluations"] / max(1, stats["expansions"]))

        run_to_run = max(_total_variation(a, b)
                         for i, a in enumerate(distributions)
                         for b in distributions[i + 1:])
        vs_serial = statistics.fmean(_total_variation(d, reference) for d in distributions)
        rows.append({
            "run_to_run": run_to_run,
            "vs_serial": vs_serial,
            "stand_in": statistics.fmean(stand_in),
            "best_moves": best_moves,
        })
    return rows


def test_layer3_root_visit_distribution_is_stable_across_runs(stability_measurement):
    """Scheduling must perturb the answer less than the configuration itself does.

    Run-to-run variation here is correct behaviour, not noise to be minimised
    away: thread scheduling decides where virtual loss sits and therefore which
    branch the next descent avoids. What must not happen is scheduling deciding
    the answer — and the way to say that without an arbitrary constant is to
    compare it against the distance the configuration has already moved from its
    own serial ground truth.
    """
    run_to_run = statistics.fmean(row["run_to_run"] for row in stability_measurement)
    vs_serial = statistics.fmean(row["vs_serial"] for row in stability_measurement)
    stand_in = statistics.fmean(row["stand_in"] for row in stability_measurement)
    ratio = run_to_run / vs_serial if vs_serial > 0 else float("inf")

    print(f"\n  over {len(stability_measurement)} positions at W=4/K=8, 2000 sims:")
    print(f"    worst run-to-run TV : {100 * run_to_run:.1f}%")
    print(f"    TV vs W=1/K=1       : {100 * vs_serial:.1f}%")
    print(f"    ratio               : {ratio:.2f}  (tolerance {ROOT_STABILITY_RATIO})")
    print(f"    stand-in share      : {100 * stand_in:.1f}% of expansions")
    unstable = sum(len(set(row["best_moves"])) > 1 for row in stability_measurement)
    print(f"    positions whose best move changed across runs: "
          f"{unstable}/{len(stability_measurement)}")

    assert ratio <= ROOT_STABILITY_RATIO, (
        f"thread scheduling moved the root distribution {100 * run_to_run:.1f}% against the "
        f"{100 * vs_serial:.1f}% this configuration already sits from serial (ratio "
        f"{ratio:.2f} > {ROOT_STABILITY_RATIO}). The answer is being decided by the "
        "scheduler rather than by the virtual-loss setting.")
    assert run_to_run < PYTHON_FLATTENING_LANDMARK, (
        f"run-to-run distance {100 * run_to_run:.1f}% is at or above the "
        f"{100 * PYTHON_FLATTENING_LANDMARK:.0f}% that separates Python's serial root "
        "distribution from its 32-outstanding one — scheduling alone would be moving the "
        "answer as much as the entire parallelism decision does")


def test_root_flattening_tracks_outstanding_leaves_not_worker_count(corpora):
    """The diagnostic the brief asks for: does the new model carry a tax of its own?

    Scope 2.2 predicts that virtual-loss distortion is a function of the
    IN-FLIGHT LEAF COUNT and no longer of the thread count — that is the whole
    reason ``max_outstanding`` replaces the worker count as the governing knob.
    This is that prediction as a test.

    W=4/K=8 and W=1/K=32 hold the same 32 leaves in flight and therefore carry the
    same virtual loss, but the second has no concurrency at all and is
    deterministic. If the concurrent one were flatter, the difference would be a
    tax belonging to the parallelism model rather than to virtual loss, and it
    would be the thing to fix. Measured over the quiet corpus: 28.6% against
    34.3% distance from serial, i.e. the concurrent configuration is if anything
    slightly LESS flattened.
    """
    corpus = corpora["quiet"]
    cases = [r for r in corpus["manifest"]["runs"]
             if r["virtual_loss"] == 2.5 and not r.get("full_tree")][:6]
    capacity = _capacity(corpus) * 4
    sims = 2000

    concurrent, control = [], []
    for case in cases:
        reference_search = _build(corpus, 2.5, capacity, accumulator="q32", synthetic=True)
        reference_search.set_position(case["fen"], _history(corpus, case))
        reference_search.search_parallel(
            sims, guofish_core.ParallelConfig(workers=1, in_flight=1,
                                              affinity="none"))
        reference = _root_distribution(reference_search)

        for workers, in_flight, sink in ((4, 8, concurrent), (1, 32, control)):
            search = _build(corpus, 2.5, capacity, accumulator="q32", synthetic=True)
            search.set_position(case["fen"], _history(corpus, case))
            search.search_parallel(
                sims,
                guofish_core.ParallelConfig(
                    workers=workers, in_flight=in_flight,
                    affinity=("none" if workers == 1
                              else "pcore_physical")))
            sink.append(_total_variation(_root_distribution(search), reference))

    concurrent_tv = statistics.fmean(concurrent)
    control_tv = statistics.fmean(control)
    print(f"\n  distance from W=1/K=1 at 32 outstanding leaves, over {len(cases)} positions:")
    print(f"    W=4 K=8  (concurrent)   : {100 * concurrent_tv:.1f}%")
    print(f"    W=1 K=32 (deterministic): {100 * control_tv:.1f}%")
    print(f"    excess attributable to concurrency: "
          f"{100 * (concurrent_tv - control_tv):+.1f} pp")

    assert concurrent_tv <= control_tv + PYTHON_FLATTENING_LANDMARK / 2, (
        f"W=4/K=8 sits {100 * concurrent_tv:.1f}% from serial where W=1/K=32 — the same "
        f"32 leaves in flight, no concurrency — sits {100 * control_tv:.1f}%. The excess is "
        "a concurrency tax the parallelism model is adding on top of virtual loss, which "
        "scope 2.2 predicts should not exist.")


# ---------------------------------------------------------------------------
# The W x K grid — conservation everywhere, not only at the default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workers", SWEEP_W)
@pytest.mark.parametrize("in_flight", SWEEP_K)
def test_conservation_holds_across_the_sweep(corpora, workers, in_flight):
    """Every cell of the reported grid must conserve, not only the shipping default.

    The grid in BENCH.md is a measurement, and a measurement taken from a
    configuration that silently loses simulations or strands virtual loss is not
    evidence of anything. This is the assertion that makes the table admissible.
    """
    corpus = corpora["quiet"]
    run = next(r for r in corpus["manifest"]["runs"]
               if r["virtual_loss"] == 2.5 and not r.get("full_tree"))
    # Headroom scales with outstanding leaves: more in-flight paths open more
    # branches, so a parallel run allocates more nodes than the serial reference.
    capacity = _capacity(corpus) * 4

    search = _build(corpus, 2.5, capacity, accumulator="q32", synthetic=True)
    search.set_position(run["fen"], _history(corpus, run))
    stats = search.search_parallel(
        int(run["sims"]),
        guofish_core.ParallelConfig(workers=workers, in_flight=in_flight, max_batch=128))
    par, audit = search.parallel_stats(), search.audit()

    assert par["delivered"] == par["requested"]
    assert audit["vloss_total"] == 0
    assert audit["conservation_failures"] == 0
    assert stats["root_visits"] == int(run["sims"])
    assert par["largest_batch"] <= workers * in_flight
    assert par["max_outstanding"] == workers * in_flight
    assert int(par["outstanding_at_drain"].max()) <= workers * in_flight


# ---------------------------------------------------------------------------
# The dispatcher's contract
# ---------------------------------------------------------------------------


def test_the_dispatcher_has_no_minimum_batch_floor(corpora):
    """Batches smaller than max_outstanding must occur, and at VL 0 they must be tiny.

    Both of the Python collector's measured pathologies are absent here and this
    is the test that says so. The lockstep one was ``min_batch == worker count``,
    which made the evaluator wait for every worker and then serially wake all of
    them; the starved one was a straggler timeout that collapsed batches to 2
    with every one eating a full 10 ms deadline. This dispatcher has neither a
    floor nor a clock: it takes whatever is there once no search thread can add
    to it.

    At virtual loss 0 with K=8 that is a batch of ONE, because the second descent
    re-selects the first descent's leaf. Seeing batches of one here is the
    positive evidence that no floor exists.
    """
    corpus = corpora["quiet"]
    run = next(r for r in corpus["manifest"]["runs"]
               if r["virtual_loss"] == 0.0 and not r.get("full_tree"))
    search = _build(corpus, 0.0, _capacity(corpus) * 2, synthetic=True)
    search.set_position(run["fen"], _history(corpus, run))
    search.search_parallel(int(run["sims"]),
                           guofish_core.ParallelConfig(workers=1, in_flight=8,
                                                       affinity="none"))
    par = search.parallel_stats()
    sizes = par["batch_sizes"]
    assert len(sizes) == par["batches"]
    assert int(sizes.min()) == 1, (
        f"no batch of size 1 occurred at virtual loss 0 with K=8 (smallest was "
        f"{int(sizes.min())}); the dispatcher appears to be waiting for a minimum batch")
    assert par["select_collisions"] > 0, (
        "virtual loss 0 with K=8 produced no select collisions, which cannot happen if the "
        "worker really is descending with 8 paths allowed in flight")


def test_max_batch_is_a_ceiling(corpora):
    """A max_batch below the outstanding count must actually cap the drain."""
    corpus = corpora["quiet"]
    run = next(r for r in corpus["manifest"]["runs"]
               if r["virtual_loss"] == 2.5 and not r.get("full_tree"))
    search = _build(corpus, 2.5, _capacity(corpus) * 3, accumulator="q32", synthetic=True)
    search.set_position(run["fen"], _history(corpus, run))
    search.search_parallel(int(run["sims"]),
                           guofish_core.ParallelConfig(workers=4, in_flight=8, max_batch=5))
    par = search.parallel_stats()
    assert par["largest_batch"] <= 5
    assert int(par["batch_sizes"].max()) <= 5
    assert par["delivered"] == par["requested"]
    assert search.audit()["vloss_total"] == 0


def test_select_collisions_fall_as_virtual_loss_rises(corpora):
    """The VL-too-low signal scope 2.2 asks to be instrumented.

    Virtual loss exists to keep concurrent descents off each other's branches, so
    a rising collision count means it is set too low. The direction is the claim
    worth pinning: at magnitude 0 it does nothing by construction and every extra
    in-flight path collides; at 2.5 it should be doing its job.
    """
    corpus = corpora["quiet"]
    counts = {}
    for virtual_loss in (0.0, 2.5):
        run = next(r for r in corpus["manifest"]["runs"]
                   if r["virtual_loss"] == virtual_loss and not r.get("full_tree"))
        search = _build(corpus, virtual_loss, _capacity(corpus) * 2, synthetic=True)
        search.set_position(run["fen"], _history(corpus, run))
        search.search_parallel(
            int(run["sims"]),
            guofish_core.ParallelConfig(workers=1, in_flight=8,
                                        affinity="none"))
        counts[virtual_loss] = search.parallel_stats()["select_collisions"]
    print(f"\n  select collisions at W=1/K=8: vl 0.0 -> {counts[0.0]}, vl 2.5 -> {counts[2.5]}")
    assert counts[2.5] < counts[0.0], (
        f"virtual loss 2.5 did not reduce select collisions against 0.0 ({counts}); either "
        "virtual loss is not reaching selection or the collision counter is wrong")


# ---------------------------------------------------------------------------
# Thread affinity
# ---------------------------------------------------------------------------


def test_an_unknown_affinity_policy_is_rejected(corpora):
    """An unrecognised policy must raise, never silently degrade to "none".

    The policy crosses the boundary as a string rather than as a bound enum
    (`py::enum_` is the one binding shape in this module that leaks its
    registration at import — see DECISIONS.md, C9), so the type system is no
    longer doing this check and the constructor has to. Silently falling back to
    "none" would make a BENCH.md row claim an affinity it never applied, which is
    exactly the class of untrue-provenance the project keeps tripping over.
    """
    with pytest.raises(ValueError) as excinfo:
        guofish_core.ParallelConfig(workers=2, in_flight=4, affinity="pcore-physical")
    message = str(excinfo.value)
    assert "pcore-physical" in message
    for name in guofish_core.AFFINITY_POLICIES:
        assert name in message, "the error should name every valid policy"

    # And the round trip holds, so a report cannot disagree with the request.
    for name in guofish_core.AFFINITY_POLICIES:
        assert guofish_core.ParallelConfig(affinity=name).affinity == name


def test_topology_is_reported_or_honestly_absent():
    """Whatever the platform says, and never a guess.

    A kernel that does not expose the hybrid split (WSL2 does not expose
    cpu_capacity) must report hybrid=False and say why in `source`, rather than
    inferring a layout. Pinning threads to an inferred layout is worse than not
    pinning them, because the resulting BENCH.md row would claim something
    untrue.
    """
    topo = guofish_core.ReplaySearchQ32.topology()
    assert topo["source"]
    assert isinstance(topo["hybrid"], bool)
    for key in ("pcore_physical", "pcore_all", "ecore_all", "all_logical"):
        assert all(isinstance(cpu, int) and cpu >= 0 for cpu in topo[key])
    if topo["all_logical"]:
        assert set(topo["pcore_all"]).isdisjoint(topo["ecore_all"])
        assert set(topo["pcore_physical"]) <= set(topo["pcore_all"])
        assert len(set(topo["all_logical"])) == len(topo["all_logical"])
    if topo["hybrid"]:
        assert topo["ecore_all"], "hybrid was reported with no efficiency cores"
    print(f"\n  topology: {topo['source']}\n"
          f"    hybrid={topo['hybrid']} hw={topo['hardware_concurrency']}\n"
          f"    P-core physical={topo['pcore_physical']}\n"
          f"    P-core all     ={topo['pcore_all']}\n"
          f"    E-core all     ={topo['ecore_all']}")


@pytest.mark.parametrize("policy_name",
                         ("none", "pcore_physical", "pcore_smt", "all_logical"))
def test_affinity_policies_pin_where_they_say(corpora, policy_name):
    """Reported pinning must match the policy, and "none" must pin nothing."""
    assert policy_name in guofish_core.AFFINITY_POLICIES
    policy = policy_name
    topo = guofish_core.ReplaySearchQ32.topology()
    corpus = corpora["quiet"]
    run = next(r for r in corpus["manifest"]["runs"]
               if r["virtual_loss"] == 2.5 and not r.get("full_tree"))

    search = _build(corpus, 2.5, _capacity(corpus) * 2, accumulator="q32", synthetic=True)
    search.set_position(run["fen"], _history(corpus, run))
    # A short run: this is about where the threads went, not about the tree.
    search.search_parallel(500, guofish_core.ParallelConfig(workers=2, in_flight=8,
                                                            affinity=policy))
    par = search.parallel_stats()
    assert par["affinity"] == policy_name
    assert len(par["pinned_cpus"]) == 2

    allowed = {
        "none": None,
        "pcore_physical": set(topo["pcore_physical"]),
        "pcore_smt": set(topo["pcore_all"]),
        "all_logical": set(topo["all_logical"]),
    }[policy_name]
    if allowed is None:
        assert par["pinned_cpus"] == [-1, -1]
    else:
        for cpu in par["pinned_cpus"]:
            # -1 means the platform refused the request, which is reported
            # rather than thrown — see run_workers.
            assert cpu == -1 or cpu in allowed, (
                f"{policy_name} pinned a worker to CPU {cpu}, which is not in {sorted(allowed)}")
    assert search.audit()["vloss_total"] == 0


# ---------------------------------------------------------------------------
# GIL acquisition — the prediction, not just the histogram
# ---------------------------------------------------------------------------


def test_dispatcher_gil_acquisition_stays_near_the_uncontended_floor(corpora):
    """Scope 2.1's prediction, tested on the real C9 thread topology.

    C0b established that the pybind11 boundary itself costs ~60 ns uncontended,
    so all of the GIL cost scope 2.1 projects is contention and none of it is the
    mechanism. During a C9 search the only Python-relevant thread is the
    dispatcher: the W search threads touch no Python at all, and the caller has
    released the GIL for the duration of search_parallel. The prediction is
    therefore that acquisition stays near the floor, and that an excursion toward
    milliseconds means Python bytecode is running that should not be.

    The assertion is deliberately loose — a millisecond, four orders above the
    floor — because this is a prediction test rather than a performance budget,
    and the number that matters is the one printed. C10 tightens it to the 200 us
    p99 trigger for C++-side info emission.
    """
    corpus = corpora["quiet"]
    run = next(r for r in corpus["manifest"]["runs"]
               if r["virtual_loss"] == 2.5 and not r.get("full_tree"))
    search = _build(corpus, 2.5, _capacity(corpus) * 3, accumulator="q32", synthetic=True)
    search.set_position(run["fen"], _history(corpus, run))

    probe = guofish_core.GilProbe()
    search.set_batch_hook(probe)
    search.search_parallel(int(run["sims"]),
                           guofish_core.ParallelConfig(workers=4, in_flight=8))
    search.set_batch_hook(None)

    par = search.parallel_stats()
    samples = np.sort(par["hook_wait_ns_samples"])
    assert probe.acquisitions == par["batches"], (
        "the hook did not run once per batch; the measurement is not of what it claims")
    assert len(samples) == par["batches"]
    assert probe.rows == par["queued_leaves"]

    def pct(p):
        return float(samples[min(len(samples) - 1, int(p * len(samples)))]) if len(samples) else 0.0

    median, p99, worst = pct(0.50), pct(0.99), float(samples[-1])
    print(f"\n  dispatcher GIL acquisition over {len(samples)} batches: "
          f"median={median:.0f} ns  p99={p99:.0f} ns  max={worst:.0f} ns  "
          f"total={par['hook_wait_ns'] / 1e6:.3f} ms of {par['wall_ns'] / 1e6:.1f} ms wall")
    assert p99 < 1_000_000, (
        f"dispatcher GIL acquisition p99 is {p99 / 1000:.1f} us, which is in the contended "
        "regime. Scope 2.1's prediction is that nothing competes for the interpreter during a "
        "search — find the Python bytecode that is running and move it to C++ rather than "
        "tuning around it.")
    assert search.audit()["vloss_total"] == 0


# ---------------------------------------------------------------------------
# Failures stay loud, and the C8 seam still holds
# ---------------------------------------------------------------------------


def test_a_dump_miss_is_still_a_named_failure_from_a_worker_thread(corpora):
    """An exception raised on the dispatcher must reach the caller with its message.

    C5's dump-miss path is a test in its own right — it is what audits in-search
    tokenization for free — and moving expansion onto another thread is exactly
    the kind of change that turns a named failure into a silent one or a crash.
    The exception has to cross the thread boundary, the search has to unwind
    every in-flight leaf on the way out, and the message has to survive.
    """
    corpus = corpora["quiet"]
    run = next(r for r in corpus["manifest"]["runs"]
               if r["virtual_loss"] == 2.5 and not r.get("full_tree"))
    search = _build(corpus, 2.5, _capacity(corpus) * 2, synthetic=False)
    search.set_position(run["fen"], _history(corpus, run))

    with pytest.raises(RuntimeError) as excinfo:
        search.search_parallel(int(run["sims"]),
                               guofish_core.ParallelConfig(workers=1, in_flight=8,
                                                           affinity="none"))
    message = str(excinfo.value)
    assert "replay dump miss" in message
    assert "nn_key" in message and "fen" in message and "path" in message

    # And the tree must not be left holding virtual loss. A failed search that
    # strands in-flight losses would make every later assertion in a suite
    # meaningless for a reason unrelated to its own subject.
    assert search.audit()["vloss_total"] == 0


def test_conservation_survives_apply_move(corpora):
    """C8's seam under C9's engine: search, play, search again, still exact.

    Tree reuse and concurrency are the two features most likely to interact
    badly, because compaction rewrites every index in the arena while the
    invariants above are stated in terms of those indices. The compaction itself
    asserts that no node arrives carrying virtual loss, which is only true if the
    parallel search really did repay everything.
    """
    corpus = corpora["quiet"]
    run = next(r for r in corpus["manifest"]["runs"]
               if r["virtual_loss"] == 2.5 and not r.get("full_tree"))
    search = _build(corpus, 2.5, _capacity(corpus) * 3, accumulator="q32", synthetic=True)
    search.set_position(run["fen"], _history(corpus, run))

    config = guofish_core.ParallelConfig(workers=4, in_flight=8)
    for ply in range(4):
        # The budget is a ROOT VISIT TARGET, not an increment: after a reused
        # promotion the root already carries inherited visits, so a second call
        # at the same number is a no-op by design. Each ply therefore asks for
        # more than the last, which is also what makes the compaction run on a
        # tree that actually grew.
        result = search.search_parallel(800 * (ply + 1), config)
        audit = search.audit()
        assert audit["vloss_total"] == 0, f"ply {ply}: virtual loss stranded"
        assert audit["conservation_failures"] == 0, f"ply {ply}: visits do not conserve"

        move = result["best_move"]
        assert move is not None, f"ply {ply}: no best move"
        search.apply_move(move)
        # The compaction asserts that no node arrives carrying virtual loss,
        # which is only true if the parallel search really did repay everything.
        assert search.audit()["vloss_total"] == 0, f"ply {ply}: apply_move left virtual loss"


def test_q32_and_double_accumulators_agree_on_the_tree_shape(corpora):
    """W=1/K=1, both accumulators, same visit counts and same move choice.

    Q32 resolves 2.3e-10 against a network that emits bf16's eight mantissa bits,
    so the two accumulators cannot disagree about which child PUCT prefers on any
    position where the margin is real. What this pins is that the production
    accumulator did not change the SEARCH, only the representation of its sums —
    the visit counts are integers on both sides and must be identical.
    """
    corpus = corpora["quiet"]
    run = next(r for r in corpus["manifest"]["runs"]
               if r["virtual_loss"] == 2.5 and not r.get("full_tree"))
    capacity = _capacity(corpus)

    out = {}
    for accumulator in ("double", "q32"):
        search = _build(corpus, 2.5, capacity, accumulator=accumulator, synthetic=False)
        search.set_position(run["fen"], _history(corpus, run))
        stats = search.search_parallel(
            int(run["sims"]),
            guofish_core.ParallelConfig(workers=1, in_flight=1,
                                        affinity="none"))
        arrays = search.dump_tree_arrays(1)
        out[accumulator] = (stats, arrays, search.audit())

    assert np.array_equal(out["double"][1]["visits"], out["q32"][1]["visits"])
    assert np.array_equal(out["double"][1]["move"], out["q32"][1]["move"])
    assert out["double"][0]["best_move"] == out["q32"][0]["best_move"]
    for accumulator in ("double", "q32"):
        assert out[accumulator][2]["conservation_failures"] == 0
        assert out[accumulator][2]["vloss_total"] == 0
        assert out[accumulator][0]["synthetic_evaluations"] == 0

    # Q32's stated resolution is 2^-32; the accumulated difference over a 5,000
    # simulation tree is bounded by that times the visit count, which is what
    # this checks rather than an arbitrary epsilon.
    visits = out["double"][1]["visits"].astype(np.float64)
    delta = np.abs(out["double"][1]["value_sum"] - out["q32"][1]["value_sum"])
    assert np.all(delta <= visits * 2.0 ** -32 + 1e-12), (
        f"Q32 and double value sums differ by more than the fixed-point resolution allows; "
        f"worst is {float((delta / np.maximum(visits, 1)).max()):.3e} per visit")


def test_search_parallel_rejects_impossible_configurations(corpora):
    """W, K and max_batch below 1 are programming errors, not degenerate cases."""
    corpus = corpora["quiet"]
    run = corpus["manifest"]["runs"][0]
    search = _build(corpus, 0.0, _capacity(corpus), synthetic=True)
    search.set_position(run["fen"], _history(corpus, run))
    for kwargs in ({"workers": 0}, {"in_flight": 0}, {"max_batch": 0}):
        base = {"workers": 1, "in_flight": 1, "max_batch": 8}
        base.update(kwargs)
        with pytest.raises(ValueError):
            search.search_parallel(10, guofish_core.ParallelConfig(**base))


# ---------------------------------------------------------------------------
# The sanitizer's own credentials
# ---------------------------------------------------------------------------


def test_thread_sanitizer_can_actually_fail():
    """Under a TSan build, a deliberate race MUST be reported.

    A clean ThreadSanitizer run is the C9 brief's mandatory acceptance evidence,
    and it is evidence only if the sanitizer can be shown to fail. This is the
    same argument the mutation drill makes about golden data: a test that cannot
    fail is not a test, and "TSan reported nothing" and "TSan was not looking"
    produce identical logs.

    So this runs ``guofish_core.race_probe`` — four threads incrementing a plain
    int — in a SUBPROCESS with ``halt_on_error=1``, and requires a non-zero exit.
    A subprocess because a TSan report is written to stderr by the runtime rather
    than raised into Python, so the exit code is the only thing an assertion can
    reach.

    On a non-TSan build this asserts the opposite: the probe runs to completion,
    which is what makes the test meaningful on both platforms instead of a skip
    that hides a hole. (Amendment D: no skip goes unexplained.)
    """
    import subprocess
    import sys

    script = (
        "import guofish_core; "
        "guofish_core.race_probe(200000, 4); "
        "print('completed')"
    )
    env = dict(os.environ)
    env["TSAN_OPTIONS"] = "halt_on_error=1:exitcode=66"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, env=env, timeout=300)

    # The assertion is on the REPORT, not on the exit code. LeakSanitizer sets a
    # non-zero exit at interpreter shutdown on the ASan build (CPython and
    # pybind11 both hold interpreter-lifetime allocations), so a return code
    # says "some sanitizer objected to something" and not "a race was found".
    if guofish_core.TSAN:
        assert "data race" in result.stderr, (
            "this is a ThreadSanitizer build, but a deliberate four-thread race on a plain "
            "int produced no data-race report. The sanitizer is not instrumenting this "
            "module, so the C9 acceptance run proves nothing.\n"
            f"  returncode: {result.returncode}\n  stdout: {result.stdout!r}\n"
            f"  stderr tail: {result.stderr[-2000:]!r}")
        assert result.returncode != 0, (
            "the race was reported but halt_on_error=1 did not stop the process, so a race "
            "in the engine would not fail a run either.\n"
            f"  stderr tail: {result.stderr[-2000:]!r}")
    else:
        assert "data race" not in result.stderr, (
            "a data race was reported by a build that is not a ThreadSanitizer build")
        assert "completed" in result.stdout, (
            "the race probe should run to completion on a build without ThreadSanitizer.\n"
            f"  returncode: {result.returncode}\n  stdout: {result.stdout!r}\n"
            f"  stderr tail: {result.stderr[-2000:]!r}")


def test_serial_search_reports_no_parallel_statistics(corpora):
    """search() must leave the C9 counters untouched, so a report cannot confuse the two."""
    corpus = corpora["quiet"]
    run = corpus["manifest"]["runs"][0]
    search = _build(corpus, run["virtual_loss"], _capacity(corpus), synthetic=False)
    search.set_position(run["fen"], _history(corpus, run))
    search.search(200)
    par = search.parallel_stats()
    assert par["delivered"] == 0 and par["requested"] == 0 and par["batches"] == 0
    assert par["workers"] == 0
    assert search.audit()["vloss_total"] == 0
