"""C10b — the graphed forward: Gate 2 re-run, the pad contract, and Gate 1's floor.

WHAT CHANGED, AND THEREFORE WHAT HAS TO BE RE-ESTABLISHED
=========================================================
C10b replaces ~110 host ATen submissions per batch with one `cudaGraphLaunch`.
That is a change to HOW the forward is executed, not to what it computes — but
"not to what it computes" is a claim, and CUDA graph capture can genuinely
change kernel selection. So the C10b brief requires Gate 2 re-run on the graphed
forward, and this file is that re-run plus the three properties graphs
introduced that nothing else covers:

  1. CAPTURE FIDELITY. The graphed forward's bf16 policy bits and its values are
     compared against the eager forward's, at every captured shape, over the
     whole 500-position Gate 2 corpus. Not "close" — identical. If this test
     ever fails, Gate 2's re-pass below is meaningless whatever number it
     prints, because the two would then be measuring different networks.

  2. THE PAD CONTRACT. Fixed shapes mean a batch of 21 runs at shape 24, and the
     three rows nobody asked about produce priors for a position nobody is
     searching. `test_a_padded_rows_prior_cannot_reach_an_expansion` fails if
     one of them can: the evaluator stamps every unused policy row with bf16 NaN
     and the search must produce the same tree it produces without the stamp.
     Asserting the pad outputs are "small" would prove nothing — they are a real
     position's real priors, and they look exactly like the answer.

  3. GATE 1's FLOOR. The replay evaluator must not route through graphs. It is a
     different object with no torch in it at all, and the acceptance is that
     this stays structurally true rather than remaining true by habit.

WHAT THIS FILE DOES NOT ASSERT, AND WHY
=======================================
It does not compare the graphed engine's SEARCH output against Gate 2b's golden.
Gate 2b runs `max_batch=1`, `1` is a captured shape, and capture is bit-identical
at every shape — so the graphed engine evaluates that gate's every batch through
exactly the kernels C10 measured it through. There is no new claim to test and a
two-hour re-run would produce the same numbers; the reasoning is recorded in
DECISIONS.md, C10b, with the bit-identity above as its evidence.

Per Amendment D there are no module-scope skips. Every test is individually
marked and the reason names which of torch / CUDA / the checkpoint is missing.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent

GOLDEN = (REPO_ROOT / "golden" / "c10_gate2.npz", "GUOFISH_GOLDEN_C10_GATE2")
MODEL = REPO_ROOT / "models" / "guofish5_20M" / "v5_10.9M_best.pt"

# Gate 2's criterion, restated once. Identical to tests/test_c10_gate2.py's,
# deliberately: a re-run under a different bound would not be a re-run.
MAX_ABS_DELTA = 1e-6

COLUMNS = ("priors_cpu_pychess", "priors_gpu_pychess", "priors_cpu_libchess")

# The Gate 2 golden was generated in chunks of `--batch 128` (tools/
# gen_c10_gate2_golden.py), so positions [0, 384) were evaluated at shape 128
# and the remaining 116 at shape 116. Only the first group can be compared
# against the golden as a statement about CAPTURE rather than about batch width;
# `test_the_shape_term_belongs_to_torch_and_not_to_the_graph` handles the rest.
GOLDEN_CHUNK = 128
COMPARABLE = 384

# A quiet midgame with a wide move list, for the search-level tests.
LEAK_FEN = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"


def _why_unavailable() -> str | None:
    """The reason the graphed evaluator cannot run here, or None.

    A guarded import plus a skipif marker, never a module-scope skip — the C7
    incident took 242 tests out of the Linux run while Rule 3 reported a pass.
    """
    try:
        import torch
    except ImportError as exc:
        return f"torch is not importable ({exc})"
    if not torch.cuda.is_available():
        return "no CUDA device; a CUDA graph has no CPU equivalent to fall back to"
    if not MODEL.exists():
        return f"the v5 checkpoint is missing ({MODEL})"
    return None


UNAVAILABLE = _why_unavailable()
requires_graphs = pytest.mark.skipif(UNAVAILABLE is not None, reason=str(UNAVAILABLE))


@pytest.fixture(scope="module")
def golden():
    path = Path(os.environ.get(GOLDEN[1], GOLDEN[0]))
    if not path.exists():
        pytest.fail(f"missing golden file {path}; see tools/gen_c10_gate2_golden.py")
    return np.load(path, allow_pickle=True)


@pytest.fixture(scope="module")
def evaluator():
    """One graphed evaluator for the module. Capture is seconds; do it once."""
    from playing.v6 import evaluator as live_evaluator

    built = live_evaluator.build(GOLDEN_CHUNK)
    try:
        yield built
    finally:
        built.close()


@pytest.fixture(scope="module")
def corpus_tokens(golden):
    return np.stack([guofish_core.tokens(str(fen)) for fen in golden["fens"]])


def _run(evaluator, tokens, start, count):
    """One batch through the PRODUCTION callback, results copied out.

    Through `_evaluate` rather than around it: the thing under test is the path
    the dispatcher takes, including the pad refill and the narrowing, not a
    convenient re-spelling of it.
    """
    evaluator._input_np[:count] = tokens[start:start + count]
    evaluator._evaluate(count)
    return evaluator._policy_np[:count].copy(), evaluator._value_np[:count].copy()


# ---------------------------------------------------------------------------
# 1. Capture fidelity
# ---------------------------------------------------------------------------

@requires_graphs
def test_the_captured_shapes_cover_every_batch_the_dispatcher_can_hand_over(evaluator):
    """`pad_to` is total on [1, max_batch], and never rounds down.

    The contract has one hole worth testing for: a count with no captured shape
    at or above it. `resolve_sizes` closes it by always including `max_batch`,
    and this is the assertion that it did.
    """
    sizes = evaluator.graph_sizes
    assert sizes == tuple(sorted(sizes)), sizes
    assert sizes[-1] == evaluator.max_batch
    for count in range(1, evaluator.max_batch + 1):
        padded = evaluator.padded_batch(count)
        assert padded >= count
        assert padded in sizes
        # And it is the SMALLEST such shape, or the ladder is being wasted.
        assert min(s for s in sizes if s >= count) == padded


@requires_graphs
def test_the_graphed_forward_is_bit_identical_to_the_eager_one(evaluator, corpus_tokens,
                                                              capsys):
    """The claim the rest of the file rests on, over the whole Gate 2 corpus.

    For every captured shape, run the corpus through the graph and through the
    un-captured forward at the SAME shape and compare raw bf16 bit patterns. The
    same shape matters: comparing shape 24 against shape 32 measures cuBLAS
    tiling, which is a real effect and a different one (see the shape-term test
    below).
    """
    import torch

    graph = evaluator.graph
    lines = []
    for size in graph.sizes:
        policy_bad = value_bad = words = 0
        for start in range(0, len(corpus_tokens) - size + 1, size):
            got_policy, got_value = _run(evaluator, corpus_tokens, start, size)
            want_policy, want_value = graph.eager(size)
            torch.cuda.synchronize()
            policy_bad += int((got_policy !=
                               want_policy.view(torch.uint16).cpu().numpy()).sum())
            value_bad += int((got_value != want_value.float().cpu().numpy()).sum())
            words += got_policy.size
        lines.append(f"  shape {size:>4}: {policy_bad} of {words:,} policy words differ, "
                     f"{value_bad} values")
        assert policy_bad == 0, (
            f"capture changed the forward at shape {size}: {policy_bad} of {words} bf16 "
            f"policy words differ from the eager forward. Gate 2's re-pass below is not "
            f"interpretable until this is zero.")
        assert value_bad == 0, f"capture changed {value_bad} values at shape {size}"
    with capsys.disabled():
        print("\nC10b capture fidelity (graphed vs eager, same shape, 500 positions)")
        print("\n".join(lines))


# ---------------------------------------------------------------------------
# 2. Gate 2, re-run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graphed_priors(evaluator, corpus_tokens, golden):
    """The production gather over logits the GRAPH produced.

    `guofish_core.gather_softmax` is the function the dispatcher calls, so this
    is the shipping path end to end: graph replay -> bf16 policy buffer -> the
    C++ gather -> priors in canonical order.
    """
    offsets = golden["move_offset"]
    fens = [str(f) for f in golden["fens"]]
    priors = np.zeros(int(offsets[-1]), dtype=np.float32)
    for start in range(0, COMPARABLE, GOLDEN_CHUNK):
        rows, _ = _run(evaluator, corpus_tokens, start, GOLDEN_CHUNK)
        for i in range(GOLDEN_CHUNK):
            begin, end = int(offsets[start + i]), int(offsets[start + i + 1])
            _, row = guofish_core.gather_softmax(fens[start + i], rows[i])
            priors[begin:end] = row
    return priors, int(offsets[COMPARABLE])


@requires_graphs
@pytest.mark.parametrize("column", COLUMNS)
def test_gate_2_re_passes_on_the_graphed_forward(graphed_priors, golden, column, capsys):
    """Criterion 1a, re-run: max |graphed prior - ATen prior| <= 1e-6.

    Every number here should reproduce C10a's table cell for cell, because the
    logits are bit-identical to the ones the golden recorded. That it does is the
    point: the gate is re-run rather than argued away.
    """
    priors, end = graphed_priors
    deltas = np.abs(priors[:end].astype(np.float64) - golden[column][:end].astype(np.float64))
    with capsys.disabled():
        print(f"  graphed vs {column:22s}: n={deltas.size:6d} "
              f"exact={int((deltas == 0).sum()):6d} max={deltas.max():.3e} "
              f"p99={np.percentile(deltas, 99):.3e} >1e-6={int((deltas > 1e-6).sum())}")
    assert deltas.max() <= MAX_ABS_DELTA, (
        f"the graphed forward fails Gate 2 against {column}: max delta "
        f"{deltas.max():.6e} over the {MAX_ABS_DELTA:.0e} bound. The C10b brief is "
        f"explicit that a graphed forward failing Gate 2 is rejected regardless of speed.")


@requires_graphs
@pytest.mark.parametrize("column", COLUMNS)
def test_the_graphed_forward_introduces_no_prior_ordering_inversions(graphed_priors, golden,
                                                                     column, capsys):
    """Criterion 1b, and the half that decides which move gets searched.

    PUCT reads priors only through comparisons, so a pair whose order the
    graphed forward reverses changes which child is selected and the trees
    diverge structurally from there.
    """
    priors, _ = graphed_priors
    offsets = golden["move_offset"]
    reference = golden[column]
    inversions = 0
    for i in range(COMPARABLE):
        begin, end = int(offsets[i]), int(offsets[i + 1])
        ref = reference[begin:end].astype(np.float64)
        got = priors[begin:end].astype(np.float64)
        sign_ref = np.sign(ref[:, None] - ref[None, :])
        sign_got = np.sign(got[:, None] - got[None, :])
        inversions += int((np.triu(sign_ref * sign_got, 1) < 0).sum())
    with capsys.disabled():
        print(f"  ordering vs {column:22s}: {inversions} inversions")
    assert inversions == 0, f"{inversions} prior-ordering inversions against {column}"


@requires_graphs
def test_the_shape_term_belongs_to_torch_and_not_to_the_graph(evaluator, corpus_tokens,
                                                              capsys):
    """Padding changes which kernel evaluates a row. That is cuBLAS, not capture.

    Batches of 4, 5 and 6 rows are the only widths on this device where padding
    up to the next captured shape moves the logits at all — by exactly one bf16
    ulp — and the same move happens on the EAGER forward with no graph anywhere
    near it. So the test is a comparison of two comparisons: graph-at-shape-N
    against eager-at-shape-N (must be zero, the capture term) and eager-at-N
    against eager-at-M (may be nonzero, the shape term).

    This is reported rather than bounded because it is not this port's to fix,
    and because C10b makes it SMALLER rather than larger: before graphs the
    engine evaluated at whatever width the dispatcher happened to drain, i.e. at
    up to `max_batch` distinct shapes; now there are `len(sizes)`.
    """
    import torch

    graph = evaluator.graph
    rows = []
    for count in (1, 3, 4, 5, 6, 7, 8, 21, 24, 31):
        padded = graph.pad_to(count)
        got, _ = _run(evaluator, corpus_tokens, 0, count)
        capture_term = int((got != graph.eager(padded)[0][:count]
                            .view(torch.uint16).cpu().numpy()).sum())
        narrow = graph.eager(count)[0].float().cpu().numpy()
        wide = graph.eager(padded)[0][:count].float().cpu().numpy()
        shape_term = float(np.abs(narrow - wide).max())
        rows.append((count, padded, capture_term, shape_term))
        assert capture_term == 0, (
            f"the graph disagrees with the eager forward at shape {padded} "
            f"({capture_term} words); that IS a capture defect")
    with capsys.disabled():
        print("\nC10b shape term (padding changes the cuBLAS tile; capture does not)")
        print("  count  shape  capture term  max |dlogit| from the shape alone")
        for count, padded, capture, shape in rows:
            print(f"  {count:>5}  {padded:>5}  {capture:>12}  {shape:.4g}")


# ---------------------------------------------------------------------------
# 3. The pad contract
# ---------------------------------------------------------------------------

@requires_graphs
def test_the_padding_rows_hold_the_pad_position_and_not_the_previous_batch(evaluator,
                                                                          corpus_tokens):
    """The tail of the static input is the fixed pad position, always.

    Not for correctness — the rows are never read — but because a padded batch
    must cost the same every time. Leaving a previous, larger batch's tokens in
    the tail would make a 3-row batch's timing depend on what ran before it, and
    would eventually feed the network a token id from a freed position.
    """
    import torch

    graph = evaluator.graph
    _run(evaluator, corpus_tokens, 0, evaluator.max_batch)   # dirty the whole buffer
    _run(evaluator, corpus_tokens, 0, 3)                     # then a small one
    padded = graph.pad_to(3)
    tail = graph.tokens[3:padded]
    expected = graph._pad_row.expand_as(tail)
    assert torch.equal(tail, expected), (
        "the pad rows still hold a previous batch's tokens; a padded batch's cost is "
        "then a function of history")


@requires_graphs
def test_a_padded_rows_prior_cannot_reach_an_expansion(evaluator):
    """The acceptance criterion, as a search that has to survive sabotage.

    `poison_unused_rows` stamps every policy row from `count` to `max_batch`
    with bf16 NaN after each batch. If any expansion read a row it was not
    given, the softmax over that row would produce NaN priors and the tree would
    change — so an IDENTICAL tree is proof the padded rows are ignored, not
    merely small. This is the test the brief asks to fail if a padded row's
    prior leaks; the poison is what makes a leak loud instead of plausible.
    """
    config = guofish_core.SearchConfig(virtual_loss=2.5, arena_capacity=400_000)
    config.cache_slots = 200_000
    parallel = guofish_core.ParallelConfig(workers=1, in_flight=24,
                                           max_batch=evaluator.max_batch)

    trees = []
    for poison in (False, True):
        evaluator.poison_unused_rows = poison
        search = guofish_core.ReplaySearchQ32(config)
        search.set_evaluator(evaluator.core)
        try:
            search.set_position(LEAK_FEN)
            stats = search.search_parallel(3000, parallel)
            arrays = search.dump_tree_arrays(0)
            at_root = arrays["depth"] == 1
            trees.append((stats["best_move"],
                          arrays["move"][at_root].copy(),
                          arrays["visits"][at_root].copy()))
        finally:
            search.set_evaluator(None)
            evaluator.poison_unused_rows = False

    clean, poisoned = trees
    assert clean[0] == poisoned[0], (
        f"poisoning the unused policy rows changed the best move "
        f"({clean[0]} -> {poisoned[0]}); a padded row's prior reached an expansion")
    assert np.array_equal(clean[1], poisoned[1]), "the root's move list changed"
    assert np.array_equal(clean[2], poisoned[2]), (
        "the root visit distribution changed when the unused rows were poisoned; the "
        "search read a row it was not given")
    # AND THE POISON REALLY WAS WRITTEN. Without this the test passes whether or
    # not `poison_unused_rows` does anything, which would make it evidence of
    # nothing. The poisoned sweep ran last and never filled the buffer (24 leaves
    # in flight against 128 rows), so the final row must still be all NaN.
    from playing.v6.evaluator import POISON_BITS

    assert (evaluator._policy_np[-1] == POISON_BITS).all(), (
        "the unused rows were never poisoned, so this test could not have failed "
        "however badly a padded row leaked")


@requires_graphs
def test_the_callback_allocates_no_device_memory(evaluator, corpus_tokens):
    """Property 2 of graph capture: the hot path is allocation-free.

    Measured rather than reviewed. `memory_allocated` is exact — it is the
    caching allocator's own ledger — so a single fresh tensor anywhere in the
    callback moves it. A temporary that is allocated and freed within the call
    would not show, which is why the loop runs many batches at many shapes: the
    allocator's high-water mark would still climb.
    """
    import torch

    counts = [1, 3, 8, 16, 21, 24, 31, 32, 48, 64, 96, 127, 128]
    for count in counts:                       # warm every shape first
        _run(evaluator, corpus_tokens, 0, min(count, evaluator.max_batch))
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    peak_before = torch.cuda.max_memory_allocated()
    for _ in range(8):
        for count in counts:
            _run(evaluator, corpus_tokens, 0, min(count, evaluator.max_batch))
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, (
        f"the callback leaked {torch.cuda.memory_allocated() - before} bytes of device "
        f"memory over 104 batches")
    assert torch.cuda.max_memory_allocated() == peak_before, (
        f"the callback allocated a temporary: peak rose by "
        f"{torch.cuda.max_memory_allocated() - peak_before} bytes")


# ---------------------------------------------------------------------------
# 4. Gate 1's floor
# ---------------------------------------------------------------------------

def test_the_replay_evaluator_never_routes_through_a_graph():
    """Gate 1 equivalence is untouched: the replay path has no torch in it.

    This runs on EVERY platform and needs no GPU, which is the point — it is the
    assertion that the equivalence build cannot acquire a graph by accident. A
    search with no evaluator installed uses `ReplayDump`, and `playing.v6` is
    not imported by anything it touches.
    """
    import sys

    for module in ("playing.v6.graphs", "playing.v6.evaluator"):
        sys.modules.pop(module, None)
    search = guofish_core.ReplaySearchDouble(guofish_core.SearchConfig())
    # The C++ surface has no capture, no shapes and no padding to configure:
    # graphs live entirely on the Python side of the boundary, which is what
    # makes "the replay path does not route through graphs" structural.
    for attribute in ("capture", "graph", "pad_to", "replay"):
        assert not hasattr(search, attribute), (
            f"guofish_core's search grew a `{attribute}` member; graphs were supposed to "
            f"stay on the Python side of the evaluation boundary")
    assert "playing.v6.graphs" not in sys.modules, (
        "importing guofish_core or building a replay search pulled in the graph module")


@requires_graphs
def test_turning_graphs_off_restores_the_eager_callback(corpus_tokens):
    """`graphs=False` is a real fallback, not a flag that changes nothing.

    It exists so that a machine where capture fails still runs, and so that the
    C10b benchmarks can measure eager against graphed in one session. Both halves
    are asserted: no graph, and the same answers.
    """
    import torch

    from playing.v6 import evaluator as live_evaluator

    model, device = live_evaluator.load_default_model()
    eager = live_evaluator.TorchEvaluator(model, device, 32, graphs=False)
    graphed = live_evaluator.TorchEvaluator(model, device, 32, graphs=True)
    try:
        assert eager.graph is None
        assert eager.graph_sizes == ()
        assert eager.padded_batch(21) == 21
        assert graphed.graph is not None
        # Only the exactly-captured widths are comparable: at any other width the
        # graphed path pads and the eager one does not, and the difference that
        # produces is the shape term, measured elsewhere in this file.
        for count in (1, 8, 16, 32):
            assert graphed.padded_batch(count) == count, count
            want_policy, want_value = _run(eager, corpus_tokens, 0, count)
            got_policy, got_value = _run(graphed, corpus_tokens, 0, count)
            torch.cuda.synchronize()
            assert np.array_equal(got_policy, want_policy), (
                f"graphed and eager disagree at the exactly captured shape {count}")
            assert np.array_equal(got_value, want_value), (
                f"graphed and eager values disagree at shape {count}")
    finally:
        eager.close()
        graphed.close()
