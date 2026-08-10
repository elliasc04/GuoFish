"""C10 Gate 2b — the differential corpus, and the live-evaluator smoke position.

WHAT THIS TEST IS FOR, GIVEN THAT GATE 2 ALREADY PASSED
=======================================================
Gate 2 says the C++ gather reproduces ATen's priors to 1e-6 with no ordering
inversions. That is a statement about one function. It says nothing about whether
an engine built on that function plays the same chess, and there are two
deliberate divergences between the two engines that Gate 2 cannot see:

  1. THE SOFTMAX UNIFICATION. The reference softmaxes root priors on the GPU
     (`_expand_root` hands `expand()` a CUDA tensor) and interior priors on the
     CPU (`BatchedEvaluator`'s bulk `.cpu()`), so it has two answers for one
     position. Production has one — every node, root included, down one path
     through one cache. Scope §2.5 says Gate 2b arbitrates that choice, and this
     is the arbitration.

  2. THE TIE-BREAK ORDER. About 1% of PUCT selection steps are exact ties, mostly
     unvisited siblings at identical `fpu + c*P*sqrt(N)` plus the promotion
     collisions. Python resolves them by python-chess's generation order via dict
     insertion; C++ resolves them by canonical `(from, to, promotion)` order.
     Scope §2.6 chose canonical deliberately and made Gate 2b the evidence that
     the choice does not move search behaviour — which is why the reference run
     behind this file is UNPATCHED Python, with `GATE1_CANONICAL_ORDER` off.

So this is the substantive assurance of the chunk and Gate 2 is the diagnostic
underneath it.

THE TWO CRITERIA, AND WHY THE SECOND IS NOT A SOFTENING OF THE FIRST
====================================================================
    >= 99% move agreement, and every disagreement traced to a top-two visit
    margin of under 2%.

The second is a claim about the KIND of disagreement, not an excuse for its
number. A disagreement at a 40% margin would mean the two engines disagree about
which move is good; a disagreement at a 0.3% margin means they agree about the
position and split a coin-flip differently, which is what two numerically
distinct engines are expected to do. Both are asserted, and every disagreement is
printed with both moves and both margins, so the claim can be read rather than
taken.

WHY THE SIMULATION BUDGET IS 1,600
==================================
Because the budget sets a ceiling on the disagreement rate before the engines are
compared at all. A disagreement can only happen where there is something to
disagree about, and near-ties resolve as visits accumulate: over this corpus, the
share of positions the reference decides by under 2% falls from 12% at 400
simulations to 4.6% (23/500) at 1,600. Below that the gate would be measuring how
undecided the reference was rather than whether the two engines agree.

It costs ~63 minutes per engine over the 500 positions, which makes this file the
dominant term in the suite's runtime. That is stated in BENCH.md rather than
worked around; the acceptance criterion names ~500 positions and a fixed budget,
and both are met at full size.

BOTH ENGINES RUN THE CONFIGURATION THE MANIFEST RECORDS
=======================================================
Not each side's defaults — see `_config_from`. The first run of this test took
`SearchConfig()`'s defaults against the reference's and they differ in
`virtual_loss` (0.0 against 2.5), which produced twelve *decisive* move
disagreements at margins up to 63%. It looked exactly like a broken evaluator. It
was two engines running two different searches.

WHAT RUNS HERE AND WHAT DOES NOT
================================
The Python reference does NOT run here — Amendment A. It ran once, in
tools/gen_c10_gate2b_golden.py, and reaches this file only through
`golden/c10_gate2b.json`. What runs here is the C++ engine on the live
evaluator, which needs torch, CUDA and the checkpoint; where any of those is
missing every test in this file skips individually, with the reason naming which
one. There are no module-scope skips (Amendment D).
"""

import hashlib
import json
import os
from pathlib import Path
import time

import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent

GOLDEN = (REPO_ROOT / "golden" / "c10_gate2b.json", "GUOFISH_GOLDEN_C10_GATE2B")
MANIFEST = (REPO_ROOT / "golden" / "c10_gate2b_manifest.json",
            "GUOFISH_GOLDEN_C10_GATE2B_MANIFEST")
CORPUS = REPO_ROOT / "golden" / "c10_corpus.json"
MODEL = REPO_ROOT / "models" / "guofish5_20M" / "v5_10.9M_best.pt"

# The acceptance criteria, stated once.
MIN_AGREEMENT = 0.99
MAX_DISAGREEMENT_MARGIN = 0.02

# Criterion 3. The engine must find the king move; the position is a study in
# which every other continuation loses material or the game.
SMOKE_FEN = "8/2R5/4R2p/5pp1/2N1k3/6Pb/r4r1P/6K1 b - - 11 46"
SMOKE_MOVE = "e4f3"  # Kf3
SMOKE_SIMS = 7000

# Matches the reference run's cache_size: large enough that nothing is evicted
# over a 500-position sweep, so neither side's answer depends on the order the
# corpus was visited in.
CACHE_SLOTS = 400_000


def _path(spec):
    default, env = spec
    return Path(os.environ.get(env, default))


# --- availability, resolved once and reported per test (Amendment D) -------

def _why_unavailable() -> str | None:
    """The reason the live evaluator cannot run here, or None.

    A guarded import plus a skipif marker, never a module-scope skip: the C7
    incident was a module-scope `importorskip` silently taking 242 tests out of
    the Linux run while Rule 3 reported a pass. Every test below is individually
    marked, so a skipped platform reports the count and the reason.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return f"torch is not importable ({type(exc).__name__}: {exc})"
    if not torch.cuda.is_available():
        return ("CUDA is not available; the golden data records a CUDA forward pass "
                "and playv5.load_model quantises to int8 on CPU, so a CPU run would "
                "be comparing a different network")
    if not MODEL.exists():
        return f"the v5 checkpoint is missing ({MODEL})"
    if not _path(GOLDEN).exists():
        return (f"missing golden file {_path(GOLDEN)}; generate it with "
                f"python tools/gen_c10_gate2b_golden.py --sims 1600")
    return None


UNAVAILABLE = _why_unavailable()
requires_live = pytest.mark.skipif(UNAVAILABLE is not None, reason=str(UNAVAILABLE))


@pytest.fixture(scope="module")
def golden():
    return json.loads(_path(GOLDEN).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest():
    return json.loads(_path(MANIFEST).read_text(encoding="utf-8"))


SEARCH_CONFIG_FIELDS = ("c_init", "c_base", "fpu_root", "fpu_tree",
                        "policy_temperature", "virtual_loss", "max_tree_depth")


def _config_from(manifest):
    """Build the C++ SearchConfig from the REFERENCE's recorded configuration.

    THIS IS NOT DEFENSIVENESS; IT IS THE FIX FOR A BUG THIS TEST ALREADY HAD.
    The first Gate 2b run took `SearchConfig()`'s defaults on the C++ side and
    the reference's defaults on the other. They differ in one field:
    `SearchConfig.virtual_loss` is 0.0 (deliberately — C5 was certified at VL 0
    and a later chunk must not change that by default) while
    `mctsv4.VIRTUAL_LOSS` is 2.5. Nothing complained, and the run produced 12
    decisive move disagreements that read as a numerical failure and were two
    engines running two different searches.

    Virtual loss is not neutral at one worker, which is why the mismatch was
    plausible: `MCTSNode.select_child` takes `parent_visits` from
    `effective_visits` — `visit_count + vloss_count * VIRTUAL_LOSS` — so the
    descent's own in-flight loss inflates `sqrt(N)` in the exploration term for
    every child of every node on the current path.

    So the configuration travels in the manifest and is applied here, and
    `test_the_configurations_match` asserts every field arrived. A parameter
    added to one side and not the other is now a failure, not a silent
    divergence.
    """
    recorded = manifest["search_config"]
    config = guofish_core.SearchConfig()
    config.c_init = recorded["c_init"]
    config.c_base = recorded["c_base"]
    config.fpu_root = recorded["fpu_root"]
    config.fpu_tree = recorded["fpu_tree"]
    config.virtual_loss = recorded["virtual_loss"]
    config.max_tree_depth = recorded["max_tree_depth"]
    config.cache_slots = CACHE_SLOTS
    return config


@pytest.fixture(scope="module")
def live(manifest):
    """The live evaluator and a search wired to it, for the whole module.

    max_batch is 256 so the same evaluator serves both the W=1/K=1 differential
    run (which never fills more than one row) and the production-config smoke
    run. Torn down explicitly: the buffers are page-locked, and the unregister
    has to happen before C++ frees them.
    """
    from playing.v6 import evaluator as live_evaluator

    evaluator = live_evaluator.build(256)
    search = guofish_core.ReplaySearchDouble(_config_from(manifest))
    search.set_evaluator(evaluator.core)
    try:
        yield search, evaluator
    finally:
        search.set_evaluator(None)
        evaluator.close()


def _root_child_visits(search):
    """{uci: visits} over the root's children, from the arena rather than a tally.

    `dump_tree_arrays(0)` emits every node including unvisited children, so the
    margin below is computed over the same denominator the reference's
    `root.children` gives — a search that spent visits on terminal fast paths
    has fewer to distribute, and a raw difference would read as more decisive
    than it was.
    """
    arrays = search.dump_tree_arrays(0)
    return {guofish_core.move_to_uci(int(move)): int(visits)
            for depth, move, visits in zip(arrays["depth"], arrays["move"], arrays["visits"])
            if depth == 1}


def _margin(visits: dict) -> float:
    counts = sorted(visits.values(), reverse=True)
    total = sum(counts)
    if total == 0:
        return 0.0
    if len(counts) == 1:
        return 1.0
    return (counts[0] - counts[1]) / total


@pytest.fixture(scope="module")
def differential(live, golden):
    """Run the C++ engine over every recorded position. One sweep, four tests.

    W=1, K=1: one search thread, one in-flight path. That removes virtual loss
    from the comparison entirely — with a single simulation in flight the loss is
    applied during a descent and repaid before the next one starts, so it never
    influences a selection on either side — and it is the configuration the
    criterion's "1 thread" names.
    """
    search, _ = live
    sims = golden["sims"]
    config = guofish_core.ParallelConfig(workers=1, in_flight=1, max_batch=1)

    results = []
    started = time.perf_counter()
    total = len(golden["records"])
    for index, record in enumerate(golden["records"]):
        # Progress, because this sweep is the dominant term in the suite's
        # runtime and a silent hour is indistinguishable from a hung one.
        #
        # RUN WITH `-s` TO SEE IT LIVE. pytest's default fd-level capture
        # swallows stdout AND stderr until the test ends, so there is no stream a
        # plain print can reach mid-run; `sys.__stderr__` does not escape it
        # either. Without `-s` these lines surface only in the captured output of
        # a failing test, which is exactly when they are least useful. The
        # alternative — writing to a side file — was rejected as more machinery
        # than a documented flag.
        if index and index % 25 == 0:
            rate = index / (time.perf_counter() - started)
            print(f"  gate2b {index}/{total}  {rate:.2f} pos/s  "
                  f"eta {(total - index) / rate:.0f}s", flush=True)
        search.set_position(record["fen"])
        stats = search.search_parallel(sims, config)
        visits = _root_child_visits(search)
        results.append({
            "fen": record["fen"],
            "reference_move": record["best_move"],
            "engine_move": stats["best_move"],
            "reference_margin": record["margin"],
            "engine_margin": _margin(visits),
            "reference_visits": record["visits"],
            "engine_visits": visits,
            "root_visits": stats["root_visits"],
            "mating_move": stats["mating_move"],
        })
    return results, time.perf_counter() - started


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@requires_live
def test_the_golden_run_was_the_reference_unpatched(manifest):
    """The configuration the golden was produced under, asserted field by field.

    `canonical_order_patch` is the one that matters. Gate 1 ran the reference
    with `GATE1_CANONICAL_ORDER = True` so both sides gathered in the same order;
    a Gate 2b golden produced under that patch would look identical and would be
    measuring the wrong thing, because the ordering choice is one of the two
    divergences this gate exists to arbitrate.
    """
    assert manifest["generator"] == "tools/gen_c10_gate2b_golden.py"
    assert manifest["canonical_order_patch"] is False
    assert manifest["dirichlet"] is False
    assert manifest["tablebase"] is None
    assert manifest["workers"] == 1
    assert manifest["device"] == "cuda"
    assert manifest["python"].startswith("3.13"), manifest["python"]
    assert manifest["python_chess"] == "1.11.2", manifest["python_chess"]
    assert manifest["corpus"]["positions"] >= 500, (
        f"the criterion says ~500 positions; the golden has "
        f"{manifest['corpus']['positions']}")


@requires_live
def test_the_configurations_match(manifest, live):
    """Both engines ran the same search, field by field.

    The differential is meaningless otherwise, and "meaningless" here does not
    mean noisy — it means confidently wrong. A single mismatched field
    (`virtual_loss`, in the first run of this test) produced twelve decisive
    disagreements at margins up to 63%, which looks exactly like a broken
    evaluator and is not one.

    `policy_temperature` is checked as a value rather than applied: the C++ side
    has no such knob because 1.0 is a no-op and the reference guards the divide
    so that the default path is bit-identical to no divide at all. A non-default
    value here would mean the C++ engine cannot reproduce the run and must say
    so rather than silently ignore it.
    """
    recorded = manifest["search_config"]
    missing = [f for f in SEARCH_CONFIG_FIELDS if f not in recorded]
    assert not missing, (
        f"the manifest is missing {missing}. Every search parameter both engines "
        f"have must travel in the golden file; a field that defaults on each side "
        f"independently is how the two engines end up running different searches.")

    search, _ = live
    assert recorded["policy_temperature"] == 1.0, (
        f"the reference ran at policy_temperature={recorded['policy_temperature']}, "
        f"which the C++ engine has no knob for (1.0 is a no-op and is the only "
        f"value it can reproduce).")
    # c_puct folds c_init and c_base together, so comparing it at several parent
    # visit counts checks both at once, through the function that actually reads
    # them rather than through the fields.
    for parent_visits in (1, 4, 64, 1024, 20000):
        expected = guofish_core.c_puct(parent_visits, recorded["c_init"],
                                       recorded["c_base"])
        assert search.c_puct(parent_visits) == expected, (
            f"c_puct({parent_visits}) disagrees with the recorded c_init/c_base")


@requires_live
def test_the_corpus_is_the_one_gate_2_measured(manifest):
    if _path(GOLDEN) != GOLDEN[0]:
        pytest.skip("golden override in effect; corpus digest not comparable")
    digest = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    assert digest == manifest["corpus"]["sha256"], (
        "golden/c10_corpus.json has changed since the Gate 2b golden was generated")


@requires_live
def test_the_engine_and_the_reference_ran_the_same_checkpoint(manifest):
    """A differential test against a different network measures nothing."""
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert digest == manifest["model"]["sha256"], (
        f"the checkpoint on disk is not the one the reference ran.\n"
        f"  manifest: {manifest['model']['sha256']}\n  on disk : {digest}")


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@requires_live
def test_every_search_delivered_the_budget(differential, golden):
    """Before agreement is compared, both sides must have run the same search.

    The root ends at exactly `sims` visits unless the depth-1 mate short-circuit
    fired, which ends the search early on both sides by design — so those
    positions are exempted by name rather than by tolerance.
    """
    results, _ = differential
    sims = golden["sims"]
    short = [r for r in results if r["mating_move"] is not None]
    for record in results:
        if record["mating_move"] is not None:
            continue
        assert record["root_visits"] == sims, (
            f"{record['fen']} ran {record['root_visits']} simulations, not {sims}")
    if short:
        print(f"\n{len(short)} positions ended on the depth-1 mate short-circuit "
              f"(exempt from the budget check, as the reference is)")


@requires_live
def test_the_engines_agree_on_at_least_99_percent_of_moves(differential, golden, capsys):
    """Criterion 1, with every disagreement printed as the criterion requires."""
    results, seconds = differential
    disagreements = [r for r in results if r["engine_move"] != r["reference_move"]]
    agreed = len(results) - len(disagreements)
    rate = agreed / len(results)

    with capsys.disabled():
        print()
        print(f"Gate 2b — {len(results)} positions at {golden['sims']} sims, "
              f"W=1 K=1, live evaluator, {seconds:.0f}s")
        print(f"  agreement: {agreed}/{len(results)} = {rate:.4%}  "
              f"(criterion >= {MIN_AGREEMENT:.0%})")
        if disagreements:
            print(f"  {len(disagreements)} disagreements:")
            for record in disagreements:
                ref_move, eng_move = record["reference_move"], record["engine_move"]
                print(f"    {record['fen']}")
                print(f"      reference {ref_move:6s} "
                      f"(its visits: {record['reference_visits'].get(ref_move, 0)}"
                      f" vs {record['reference_visits'].get(eng_move, 0)} for {eng_move})"
                      f"  margin {record['reference_margin']:.4%}")
                print(f"      engine    {eng_move:6s} "
                      f"(its visits: {record['engine_visits'].get(eng_move, 0)}"
                      f" vs {record['engine_visits'].get(ref_move, 0)} for {ref_move})"
                      f"  margin {record['engine_margin']:.4%}")
        else:
            print("  no disagreements")

    assert rate >= MIN_AGREEMENT, (
        f"move agreement {rate:.4%} is below the {MIN_AGREEMENT:.0%} criterion "
        f"({len(disagreements)} disagreements over {len(results)} positions); "
        f"each one is printed above with both moves and both margins")


@requires_live
def test_every_disagreement_is_a_near_tie(differential, capsys):
    """Criterion 2: a disagreement must be a coin-flip, not a difference of opinion.

    Measured on BOTH sides. The reference's margin says how close the decision
    was in the tree that produced the recorded answer; the engine's says the same
    about its own. Requiring only the reference's would let an engine that is
    decisively wrong pass whenever the reference happened to be undecided.
    """
    results, _ = differential
    disagreements = [r for r in results if r["engine_move"] != r["reference_move"]]

    with capsys.disabled():
        if disagreements:
            worst_ref = max(r["reference_margin"] for r in disagreements)
            worst_eng = max(r["engine_margin"] for r in disagreements)
            print(f"  widest disagreement margin: reference {worst_ref:.4%}, "
                  f"engine {worst_eng:.4%}  (criterion < {MAX_DISAGREEMENT_MARGIN:.0%})")

    decisive = [r for r in disagreements
                if r["reference_margin"] >= MAX_DISAGREEMENT_MARGIN
                or r["engine_margin"] >= MAX_DISAGREEMENT_MARGIN]
    if decisive:
        lines = [f"  {r['fen']}\n"
                 f"    reference {r['reference_move']} (margin {r['reference_margin']:.4%}), "
                 f"engine {r['engine_move']} (margin {r['engine_margin']:.4%})"
                 for r in decisive]
        pytest.fail(
            f"{len(decisive)} disagreement(s) are not near-ties: one side chose its move "
            f"by a margin of {MAX_DISAGREEMENT_MARGIN:.0%} or more, which is a difference "
            f"of opinion about the position rather than a coin flip split two ways.\n"
            + "\n".join(lines))


@requires_live
def test_the_margin_distribution_explains_the_agreement_rate(differential, capsys):
    """Reported, not asserted: how much of the corpus was ever in doubt.

    Agreement is bounded above by the fraction of positions whose top-two margin
    leaves room to differ. Printing that fraction is what makes the agreement
    rate interpretable — a 99% agreement over a corpus where 30% of positions are
    near-ties would be a much stronger result than the same number over a corpus
    where 1% are.
    """
    results, _ = differential
    near = [r for r in results if r["reference_margin"] < MAX_DISAGREEMENT_MARGIN]
    disagreed_near = [r for r in near if r["engine_move"] != r["reference_move"]]
    with capsys.disabled():
        print(f"  positions the reference decided by < {MAX_DISAGREEMENT_MARGIN:.0%}: "
              f"{len(near)}/{len(results)} = {len(near) / len(results):.2%}")
        print(f"  of those, disagreed: {len(disagreed_near)}/{len(near)}")
        print(f"  disagreements outside that set: "
              f"{sum(1 for r in results if r['engine_move'] != r['reference_move']) - len(disagreed_near)}")


# ---------------------------------------------------------------------------
# Criterion 3 — the smoke position
# ---------------------------------------------------------------------------

@requires_live
def test_the_smoke_position_plays_kf3_serially(live, capsys):
    """`8/2R5/4R2p/5pp1/2N1k3/6Pb/r4r1P/6K1 b - - 11 46` must play Kf3 at 7,000 sims.

    Run at W=1/K=1 first because that configuration is deterministic: the
    dispatcher drains only when no search thread can proceed, so the handoff
    point is a function of the search state rather than of the scheduler (C9,
    acceptance layer 2). A failure here is reproducible; a failure at the
    production configuration might not be.
    """
    search, _ = live
    config = guofish_core.ParallelConfig(workers=1, in_flight=1, max_batch=1)
    search.set_position(SMOKE_FEN)
    started = time.perf_counter()
    stats = search.search_parallel(SMOKE_SIMS, config)
    elapsed = time.perf_counter() - started
    visits = _root_child_visits(search)
    ranked = sorted(visits.items(), key=lambda kv: -kv[1])[:5]

    with capsys.disabled():
        print()
        print(f"Smoke — {SMOKE_FEN}")
        print(f"  W=1 K=1, {SMOKE_SIMS} sims in {elapsed:.1f}s -> {stats['best_move']}")
        print(f"  top root children: {ranked}")

    assert stats["best_move"] == SMOKE_MOVE, (
        f"the smoke position played {stats['best_move']}, not {SMOKE_MOVE} (Kf3), at "
        f"{SMOKE_SIMS} simulations. Root children by visits: {ranked}")


@requires_live
def test_the_smoke_position_plays_kf3_at_the_production_configuration(live, capsys):
    """The same position through the shipping path: batches, not one leaf at a time.

    W=1 keeps the descent single-threaded while K=32 lets the dispatcher form
    real batches, so this exercises `prepare_live_batch` — the cache probe, the
    row assignment and the one-crossing-per-batch discipline — which the serial
    run above never reaches. A batching bug that reordered rows would give the
    right answer here only by accident.
    """
    search, evaluator = live
    config = guofish_core.ParallelConfig(workers=1, in_flight=32, max_batch=256)
    search.set_position(SMOKE_FEN)
    started = time.perf_counter()
    stats = search.search_parallel(SMOKE_SIMS, config)
    elapsed = time.perf_counter() - started
    eval_stats = search.eval_stats()

    with capsys.disabled():
        print(f"  W=1 K=32, {SMOKE_SIMS} sims in {elapsed:.1f}s -> {stats['best_move']}")
        print(f"  {eval_stats['batches']} boundary crossings, {eval_stats['rows']} rows "
              f"({eval_stats['mean_rows']:.1f}/batch), "
              f"{eval_stats['cache_skipped']} answered by the cache")
        print(f"  switch interval {evaluator.switch_interval} "
              f"(was {evaluator.switch_interval_before}), pinned={evaluator.pinned}")

    assert stats["best_move"] == SMOKE_MOVE, (
        f"at the production configuration the smoke position played "
        f"{stats['best_move']}, not {SMOKE_MOVE} (Kf3)")


# ---------------------------------------------------------------------------
# The boundary's own discipline
# ---------------------------------------------------------------------------

@requires_live
def test_the_switch_interval_is_set_before_any_search_thread(live):
    """Scope §2.1's mandatory setting, checked on the object that imposes it.

    Constructing the evaluator is what sets it, and the evaluator is constructed
    before any search exists — so this asserts the ordering by asserting the
    place. At the default 5 ms interval C0b measured the dispatcher's contended
    acquire wait at a median of 15.25 ms on Windows.
    """
    _, evaluator = live
    import sys as _sys
    assert evaluator.switch_interval == pytest.approx(guofish_core.DEFAULT_SWITCH_INTERVAL)
    assert _sys.getswitchinterval() == pytest.approx(guofish_core.DEFAULT_SWITCH_INTERVAL)


@requires_live
def test_the_buffers_are_zero_copy(live):
    """C++ writes the token buffer; Python must be reading that memory, not a copy.

    `.base` being a capsule rather than None is pybind11's own statement that the
    array was built around a foreign pointer with NPY_ARRAY_OWNDATA clear — the
    same check C0 used to establish the boundary in the first place.
    """
    _, evaluator = live
    for view, dtype, shape in ((evaluator.core.input_view(), "int32", (256, 68)),
                               (evaluator.core.policy_view(), "uint16", (256, 4096)),
                               (evaluator.core.value_view(), "float32", (256,))):
        assert view.base is not None, "the view owns its data; it is a copy"
        assert view.dtype.name == dtype
        assert view.shape == shape


@requires_live
def test_the_dispatcher_barely_waits_for_the_gil(live, capsys):
    """C10's contingency trigger, evaluated rather than assumed.

    The brief makes C++-side `info` emission conditional on this measurement: it
    becomes mandatory if the acquire-wait p99 exceeds 200 us, or if the total
    exceeds 1% of the move's wall time. This is the in-suite check that the
    condition still does not hold; BENCH.md carries the full histogram, including
    the adversarial case with a competing pure-Python thread, which is the one
    the threshold was written for.

    `call_ns` is deliberately not compared against anything. Torch re-acquires
    the GIL inside a call, so a handoff wait lands in the middle of it — only the
    wait measured from outside the acquire scope is contention.
    """
    search, _ = live
    config = guofish_core.ParallelConfig(workers=1, in_flight=32, max_batch=256)
    search.set_position(SMOKE_FEN)
    started = time.perf_counter()
    search.search_parallel(2000, config)
    wall_ns = (time.perf_counter() - started) * 1e9
    stats = search.eval_stats()

    samples = sorted(int(s) for s in stats["acquire_wait_ns_samples"])
    assert samples, "no acquire-wait samples; the histogram is not being collected"
    p99 = samples[min(len(samples) - 1, int(0.99 * len(samples)))]
    share = stats["acquire_wait_ns"] / wall_ns

    with capsys.disabled():
        print()
        print(f"Acquire wait over {len(samples)} boundary crossings")
        print(f"  p50 {samples[len(samples) // 2] / 1000:.2f} us  "
              f"p99 {p99 / 1000:.2f} us  max {max(samples) / 1000:.2f} us")
        print(f"  total {stats['acquire_wait_ns'] / 1e6:.2f} ms = {share:.4%} of wall "
              f"(triggers: p99 > 200 us, or share > 1%)")

    assert p99 <= 200_000, (
        f"acquire-wait p99 is {p99 / 1000:.1f} us, over the 200 us trigger. C10 requires "
        f"C++-side `info` emission once this holds — see the brief.")
    assert share <= 0.01, (
        f"acquire wait is {share:.2%} of the move's wall time, over the 1% trigger. "
        f"C10 requires C++-side `info` emission once this holds.")
