"""C11b — policy temperature: the Gate 2 extension and the four invariants.

WHY THIS IS A NEW FILE RATHER THAN AN EXTENSION OF test_c10_gate2.py
====================================================================
Global Rule 1 is hard from C1 onward: no existing test file may be modified. The
C11b brief asks for Gate 2 to be "extended" over T in {0.7, 1.0, 1.5}, and the
rule and the brief are reconciled by extending the GATE rather than the FILE.
`test_c10_gate2.py` is untouched and still measures exactly what it measured at
C10 — which is worth something in itself, because the T = 1.0 column of this
file's golden data was verified bit-identical to that file's golden before it
was written, so the two are measuring one function at one temperature and
agreeing.

WHAT THIS FILE ADDS TO GATE 2
=============================
The same two criteria — max |C++ - ATen| <= 1e-6, and ZERO prior-ordering
inversions — over the same 500 positions and the same three reference columns,
at three temperatures instead of one. Nine cells per criterion.

Plus one number the brief asks to be measured rather than assumed:

    THE SMALLEST NON-ZERO INTER-PRIOR GAP, PER TEMPERATURE.

Gate 2's ordering criterion is only a MEANINGFUL check on a correct
implementation while the corpus' closest prior pair is separated by more than
the magnitude bound. C10b measured that gap at 1.927e-06 — 1.93x the 1e-6 bound,
which is not much headroom — and the brief predicted flattening (T > 1) would
compress it toward the bound.

**The measurement came out the other way round, and the direction is the finding.**
See `test_the_minimum_inter_prior_gap_is_measured_at_each_temperature` for the
numbers and for why sharpening, not flattening, is what closes the gap.

WHAT THIS FILE DOES NOT NEED
============================
No torch, no GPU, no model, no python-chess. The Gate 2 half reads recorded
logits and recorded ATen answers out of `golden/c11b_gate2_temp.npz`
(Amendment A); the invariants half drives the real search through a
`guofish_core.LiveEvaluator` whose callback is nine lines of NumPy. That is a
deliberate choice and not a convenience: the root-path and clear-on-change
invariants are the two most important things this chunk has to prove, and
proving them on a synthetic evaluator means they are proved on every platform
the suite runs on rather than skipped wherever CUDA is absent (Amendment D).
Every test in this file runs everywhere. There are no skips and no module-scope
guards.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent

DUMP = (REPO_ROOT / "golden" / "c11b_gate2_temp.npz", "GUOFISH_GOLDEN_C11B_GATE2_TEMP")
MANIFEST = (REPO_ROOT / "golden" / "c11b_gate2_temp_manifest.json",
            "GUOFISH_GOLDEN_C11B_GATE2_TEMP_MANIFEST")
GATE2_DUMP = REPO_ROOT / "golden" / "c10_gate2.npz"
CORPUS = REPO_ROOT / "golden" / "c10_corpus.json"

# Gate 2's criterion, unchanged. Stated here so that a reader can see it is the
# SAME number and not a temperature-adjusted one.
MAX_ABS_DELTA = 1e-6

TEMPERATURES = (0.7, 1.0, 1.5)
COLUMNS = ("priors_cpu_pychess", "priors_gpu_pychess", "priors_cpu_libchess")

COLUMN_MEANING = {
    "priors_cpu_pychess": "reference INTERIOR path (ATen CPU, python-chess order)",
    "priors_gpu_pychess": "reference ROOT path (ATen CUDA, python-chess order)",
    "priors_cpu_libchess": "ATen CPU, chess-library order (permutation removed)",
}

# C10's measured max delta against `priors_cpu_libchess` at T = 1.0. The brief
# predicts the error scales as ~1/T, so 0.7 should land near 1.43x this and 1.5
# near 0.67x it. Stated so a surprise is visible AS a surprise rather than as a
# number nobody had an expectation for.
C10_MAX_DELTA_AT_T1 = 2.682e-07


def _path(spec):
    default, env = spec
    return Path(os.environ.get(env, default))


def _overridden(spec) -> bool:
    return spec[1] in os.environ


def suffix(temperature: float) -> str:
    """Matches tools/gen_c11b_gate2_temp_golden.py::suffix."""
    return f"t{int(round(temperature * 100)):03d}"


@pytest.fixture(scope="module")
def golden():
    path = _path(DUMP)
    if not path.exists():
        pytest.fail(
            f"missing golden file {path}. Generate it with\n"
            f"    python tools/gen_c11b_gate2_temp_golden.py\n"
            f"It is produced by the Python reference over golden/c10_gate2.npz's "
            f"recorded logits (Global Rule 2); it cannot be reconstructed from C++ "
            f"output.")
    return np.load(path, allow_pickle=True)


@pytest.fixture(scope="module")
def manifest():
    path = _path(MANIFEST)
    if not path.exists():
        pytest.fail(f"missing manifest {path}; see tools/gen_c11b_gate2_temp_golden.py")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def measured(golden):
    """{T: (priors, mismatched_move_lists)}, from the PRODUCTION gather.

    `guofish_core.gather_softmax` is the function `expand_from_live_row` calls,
    with the temperature argument the search passes it — not a restatement. Three
    temperatures over 500 positions is about a second, and every test below
    reads it, so it is built once for the module.
    """
    fens = golden["fens"]
    offsets = golden["move_offset"]
    moves = golden["moves"]
    source = np.load(GATE2_DUMP, allow_pickle=True)
    logits = source["logits"]

    out = {}
    for temperature in TEMPERATURES:
        priors = np.zeros(int(offsets[-1]), dtype=np.float32)
        mismatched = []
        for i, fen in enumerate(fens):
            begin, end = int(offsets[i]), int(offsets[i + 1])
            cpp_moves, cpp_priors = guofish_core.gather_softmax(
                str(fen), logits[i], temperature)
            if list(cpp_moves) != [str(m) for m in moves[begin:end]]:
                mismatched.append((str(fen), list(cpp_moves),
                                   [str(m) for m in moves[begin:end]]))
            priors[begin:end] = cpp_priors
        out[temperature] = (priors, mismatched)
    return out


def _deltas(measured_priors, reference):
    return np.abs(measured_priors.astype(np.float64) - reference.astype(np.float64))


def _distribution(deltas):
    return {
        "count": int(deltas.size),
        "exact": int((deltas == 0).sum()),
        "max": float(deltas.max()),
        "mean": float(deltas.mean()),
        "p50": float(np.percentile(deltas, 50)),
        "p90": float(np.percentile(deltas, 90)),
        "p99": float(np.percentile(deltas, 99)),
        "p99.9": float(np.percentile(deltas, 99.9)),
        "over_1e-7": int((deltas > 1e-7).sum()),
        "over_1e-6": int((deltas > 1e-6).sum()),
    }


def _format(name, dist):
    return (f"{name:24s} n={dist['count']:6d}  exact={dist['exact']:6d}  "
            f"mean={dist['mean']:.3e}  p50={dist['p50']:.3e}  p90={dist['p90']:.3e}  "
            f"p99={dist['p99']:.3e}  p99.9={dist['p99.9']:.3e}  max={dist['max']:.3e}  "
            f">1e-7={dist['over_1e-7']:5d}  >1e-6={dist['over_1e-6']}")


def _order_violations(measured_priors, reference, offsets):
    """Pairs whose relative order the port got wrong, and pairs it collapsed.

    Transcribed from tests/test_c10_gate2.py so the extension measures ordering
    the same way the gate it extends does. An INVERSION is a pair the reference
    ranks one way and C++ ranks the other — the failure the criterion names,
    because PUCT reads priors only through comparisons. A COLLAPSE is a pair the
    reference separates and C++ ties, which hands the decision to child ordering
    instead of to the network.
    """
    inversions = []
    collapses = []
    for i in range(len(offsets) - 1):
        begin, end = int(offsets[i]), int(offsets[i + 1])
        ref = reference[begin:end].astype(np.float64)
        got = measured_priors[begin:end].astype(np.float64)
        sign_ref = np.sign(ref[:, None] - ref[None, :])
        sign_got = np.sign(got[:, None] - got[None, :])
        bad = np.argwhere(np.triu(sign_ref * sign_got, 1) < 0)
        tied = np.argwhere(np.triu((sign_ref != 0) & (sign_got == 0), 1))
        for a, b in bad:
            inversions.append((i, int(a), int(b), ref[a], ref[b], got[a], got[b]))
        for a, b in tied:
            collapses.append((i, int(a), int(b), ref[a], ref[b], got[a]))
    return inversions, collapses


def _smallest_nonzero_gap(priors, offsets):
    """(gap, position). Measured exactly as tools/drill_c10_gate2.py measures it."""
    best, where = float("inf"), -1
    for i in range(len(offsets) - 1):
        begin, end = int(offsets[i]), int(offsets[i + 1])
        chunk = np.sort(priors[begin:end].astype(np.float64))
        gaps = np.diff(chunk)
        nonzero = gaps[gaps > 0]
        if nonzero.size == 0:
            continue
        smallest = float(nonzero.min())
        if smallest < best:
            best, where = smallest, i
    return best, where


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_the_temperature_golden_records_the_reference_it_came_from(manifest):
    """Amendment A, one link further down the chain than Gate 2's own.

    No forward pass happens in this file's generator: the logits come from
    `golden/c10_gate2.npz`, so the provenance that matters is that file's digest
    and, through its manifest, the checkpoint and autocast dtype that produced
    it. Both travel here rather than being re-derived.
    """
    assert manifest["generator"] == "tools/gen_c11b_gate2_temp_golden.py"
    assert manifest["python"].startswith("3.13"), manifest["python"]
    assert manifest["python_chess"] == "1.11.2", manifest["python_chess"]
    assert manifest["autocast_dtype"] == "torch.bfloat16"
    assert manifest["policy_size"] == guofish_core.POLICY_SIZE
    assert list(manifest["temperatures"]) == list(TEMPERATURES)
    assert len(manifest["source_model"]["sha256"]) == 64
    assert manifest["corpus"]["positions"] >= 500, (
        f"the acceptance criterion says ~500 positions; the manifest records "
        f"{manifest['corpus']['positions']}")


def test_the_identity_temperature_reproduces_gate_2s_own_golden(golden, manifest):
    """The T = 1.0 columns ARE Gate 2's columns, bit for bit.

    This is what makes the other six columns trustworthy. The generator computes
    all three temperatures through one function; if that function reproduces the
    C10 reference exactly at T = 1.0 — same ATen, same devices, same reduction
    orders, same permutation back to canonical — then the only thing that differs
    at T != 1.0 is the divide the brief asked for. Without this check, a
    transcription error would show up as a Gate 2 delta at every temperature and
    be indistinguishable from a C++ defect.

    NOT SKIPPED UNDER A GOLDEN OVERRIDE, unlike the corpus-digest check below,
    and the difference is worth stating because the first draft got it wrong and
    the drill caught it. A corpus digest cannot match a corrupted copy, so
    comparing it says nothing. This check compares the override's T = 1.0
    columns against `golden/c10_gate2.npz`, WHICH THE DRILL NEVER TOUCHES — so a
    corrupted copy is exactly what it should be able to reject, and skipping
    here would leave the check undrillable. `break-the-identity` in
    tools/drill_c11b_temperature.py is the mutation that established this.
    """
    assert "bit-identical" in manifest["identity_self_check"]
    source = np.load(GATE2_DUMP, allow_pickle=True)
    for column in COLUMNS:
        mine = golden[f"{column}_{suffix(1.0)}"]
        theirs = source[column]
        assert np.array_equal(mine.view(np.uint32), theirs.view(np.uint32)), (
            f"the T=1.0 column `{column}` is not bit-identical to "
            f"golden/c10_gate2.npz's. The temperature golden was generated from a "
            f"different reference than Gate 2's, so nothing in this file is "
            f"comparable to C10's recorded figures.")


def test_the_corpus_is_the_one_gate_2_measured(manifest):
    if _overridden(DUMP) or _overridden(MANIFEST):
        pytest.skip("golden override in effect (mutation drill); corpus digest not comparable")
    digest = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    assert digest == manifest["corpus"]["sha256"], (
        f"golden/c10_corpus.json has changed since the temperature golden was "
        f"generated.\n  manifest: {manifest['corpus']['sha256']}\n  on disk : {digest}")


# ---------------------------------------------------------------------------
# The gate, at three temperatures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_the_move_lists_agree(measured, temperature):
    """Temperature must not touch WHICH moves exist, only their priors.

    Trivially true by construction — the divide happens after the gather — and
    checked anyway at every temperature, because the gather is positional and a
    move-list difference would make every delta below a comparison of unrelated
    moves that still produces a number.
    """
    _, mismatched = measured[temperature]
    if mismatched:
        fen, cpp, ref = mismatched[0]
        pytest.fail(
            f"at T={temperature}, {len(mismatched)} positions have a different "
            f"canonical move list.\nfirst: {fen}\n  C++   : {cpp}\n  golden: {ref}")


@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize("column", COLUMNS)
def test_the_max_absolute_delta_is_within_the_gate(measured, golden, temperature,
                                                   column, capsys):
    """Criterion 1a at each T: max |C++ prior - ATen prior| <= 1e-6.

    The SAME bound at every temperature. It is not scaled by T, because the
    criterion is a statement about how far apart two implementations of one
    function are allowed to be, and that does not become more forgiving because
    the function's input was divided first.
    """
    priors, _ = measured[temperature]
    deltas = _deltas(priors, golden[f"{column}_{suffix(temperature)}"])
    dist = _distribution(deltas)

    with capsys.disabled():
        print()
        print(f"Gate 2 (C11b) T={temperature:g} vs {column}")
        print(f"  ({COLUMN_MEANING[column]})")
        print(f"  {_format('|C++ - ATen|', dist)}")

    assert dist["max"] <= MAX_ABS_DELTA, (
        f"at T={temperature}, max absolute delta {dist['max']:.6e} exceeds the "
        f"{MAX_ABS_DELTA:.0e} gate against {column} ({COLUMN_MEANING[column]}); "
        f"{dist['over_1e-6']} of {dist['count']} priors are over the bound")


@pytest.mark.parametrize("temperature", TEMPERATURES)
@pytest.mark.parametrize("column", COLUMNS)
def test_there_are_no_prior_ordering_inversions(measured, golden, temperature,
                                                column, capsys):
    """Criterion 1b at each T, and the half that decides which move gets searched.

    This is the criterion the gap measurement below bears on. At T = 0.7 the
    corpus' closest prior pair is separated by 5.88e-08 while the measured error
    is ~3.8e-07 — an error six times the gap — so zero inversions here is a
    stronger result at T = 0.7 than it is at T = 1.0, not a weaker one. If it
    ever fails, the gap table is where to look first.
    """
    priors, _ = measured[temperature]
    inversions, collapses = _order_violations(
        priors, golden[f"{column}_{suffix(temperature)}"], golden["move_offset"])

    with capsys.disabled():
        print(f"  ordering T={temperature:<4g} vs {column:22s}: {len(inversions)} "
              f"inversions, {len(collapses)} collapsed pairs")

    if collapses:
        i, a, b, ref_a, ref_b, got = collapses[0]
        print(f"NOTE: {len(collapses)} pairs the reference separates and C++ ties at "
              f"T={temperature}; first at position {i}, moves {a}/{b}: reference "
              f"{ref_a!r} vs {ref_b!r}, C++ both {got!r}")

    if inversions:
        lines = []
        for i, a, b, ref_a, ref_b, got_a, got_b in inversions[:10]:
            lines.append(f"  position {i} moves {a}/{b}: "
                         f"reference {ref_a:.9e} vs {ref_b:.9e}, "
                         f"C++ {got_a:.9e} vs {got_b:.9e}")
        pytest.fail(
            f"{len(inversions)} prior-ordering inversions at T={temperature} against "
            f"{column}. PUCT reads priors only through comparisons, so an inversion "
            f"changes which child is selected.\n" + "\n".join(lines))


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_the_priors_are_a_distribution(measured, golden, temperature):
    """Non-negative and summing to 1 at every temperature.

    Catches the class of bug the delta comparison cannot: a divide applied to
    some entries and not others still lands close to the reference on most
    priors while breaking the sum. Renormalisation is what makes softmax(l/T) a
    distribution at all, so this is the check that the divide happened BEFORE
    the softmax rather than after it.
    """
    priors, _ = measured[temperature]
    offsets = golden["move_offset"]
    assert (priors >= 0).all()
    for i in range(len(offsets) - 1):
        begin, end = int(offsets[i]), int(offsets[i + 1])
        total = float(priors[begin:end].astype(np.float64).sum())
        assert abs(total - 1.0) < 1e-5, (
            f"at T={temperature}, position {i} priors sum to {total!r}")


# ---------------------------------------------------------------------------
# The two numbers the brief asks to be measured rather than assumed
# ---------------------------------------------------------------------------

def test_the_error_scales_roughly_as_one_over_t(measured, golden, capsys):
    """The brief's prediction, stated first and then measured.

    Dividing by T = 0.7 amplifies the logits 1.43x, so the absolute error on the
    resulting priors should scale about the same way: ~3.8e-07 at T = 0.7 and
    ~1.8e-07 at T = 1.5 against C10's 2.682e-07 at T = 1.0. Reported rather than
    asserted tightly — a factor-of-two band around the prediction is the
    assertion, because the point is to make a SURPRISE visible as a surprise,
    and a tight bound on a heuristic would only produce false alarms.
    """
    rows = []
    for temperature in TEMPERATURES:
        priors, _ = measured[temperature]
        deltas = _deltas(priors, golden[f"priors_cpu_libchess_{suffix(temperature)}"])
        predicted = C10_MAX_DELTA_AT_T1 / temperature
        rows.append((temperature, float(deltas.max()), predicted))

    with capsys.disabled():
        print()
        print("Error scaling with temperature (vs priors_cpu_libchess, "
              "the permutation-free column)")
        print(f"  {'T':>5}  {'predicted ~2.682e-07/T':>24}  {'measured':>12}  ratio")
        for temperature, got, predicted in rows:
            print(f"  {temperature:5g}  {predicted:24.3e}  {got:12.3e}  "
                  f"{got / predicted:.2f}x")

    for temperature, got, predicted in rows:
        assert 0.5 * predicted <= got <= 2.0 * predicted, (
            f"at T={temperature} the max delta is {got:.3e}, outside the "
            f"[0.5x, 2x] band around the predicted {predicted:.3e}. The ~1/T "
            f"scaling is a claim about how the divide propagates through the "
            f"softmax; a departure this large means it does not, and the reason "
            f"needs finding before the number is trusted.")


def test_the_minimum_inter_prior_gap_is_measured_at_each_temperature(
        measured, golden, manifest, capsys):
    """THE BRIEF'S SECOND MEASUREMENT, AND IT CAME OUT THE OTHER WAY ROUND.

    The brief predicted: "Sharpening (T < 1) spreads priors and widens gaps;
    flattening (T > 1) compresses them", and asked whether the T = 1.5 gap
    approaches the 1e-6 bound. Measured over the 500-position corpus:

        T = 0.7   5.884e-08   0.06x the bound    <-- BELOW IT
        T = 1.0   1.927e-06   1.93x the bound        (reproduces C10b exactly)
        T = 1.5   1.567e-05  15.67x the bound

    The direction is inverted, and the reason is that this statistic is a
    MINIMUM over the whole corpus, so it is set by the BOTTOM of each position's
    prior distribution rather than the top. Flattening pulls every prior toward
    1/n, which lifts the near-zero tail up and spreads it out in absolute terms;
    sharpening drives that same tail toward zero, where the absolute differences
    between adjacent tiny priors collapse. The brief's intuition holds for the
    top few moves — where sharpening does widen the gaps — but the top few are
    never where the corpus minimum lives.

    WHAT THAT MEANS FOR THE GATE, since a crossed threshold is a
    report-and-stop condition rather than a bound to loosen quietly:

    * Gate 2's ordering criterion is NOT weakened. It is strengthened. With a
      gap of 5.88e-08 and a measured error of ~3.8e-07 at T = 0.7, an incorrect
      implementation has ample room to invert a pair while staying far inside
      the magnitude bound. Zero inversions at T = 0.7 is therefore a claim the
      magnitude bound does not imply — which is exactly the independence C10's
      mutation drill went looking for and could not find at T = 1.0.

    * `drill_c10_gate2.py`'s `invert-inside-tolerance` construction is not
      threatened. It needs `0.5 * gap + ulp < 1e-6` to build an inversion the
      magnitude check cannot see; at T = 1.0 that holds with almost no room
      (9.6e-07 against 1e-06), and at T = 0.7 it holds easily. Nothing there
      needs revisiting. The temperature at which the construction becomes
      IMPOSSIBLE is a high one, and 1.5 is not it.

    Reported, and asserted only where an assertion means something: the T = 1.0
    row must still reproduce C10b's recorded 1.927e-06, because that is a
    regression check on the identity path.
    """
    c10b_recorded = 1.927e-06
    rows = []
    for temperature in TEMPERATURES:
        priors, _ = measured[temperature]
        cpp_gap, cpp_where = _smallest_nonzero_gap(priors, golden["move_offset"])
        ref_gap, _ = _smallest_nonzero_gap(
            golden[f"priors_cpu_pychess_{suffix(temperature)}"], golden["move_offset"])
        rows.append((temperature, cpp_gap, ref_gap, cpp_where))

    with capsys.disabled():
        print()
        print("Smallest non-zero inter-prior gap on the corpus, per temperature")
        print(f"  (Gate 2's magnitude bound is {MAX_ABS_DELTA:.1e}; C10b recorded "
              f"{c10b_recorded:.3e} at T=1.0)")
        print(f"  {'T':>5}  {'C++':>12}  {'ATen ref':>12}  {'x bound':>9}  position")
        for temperature, cpp_gap, ref_gap, where in rows:
            print(f"  {temperature:5g}  {cpp_gap:12.3e}  {ref_gap:12.3e}  "
                  f"{cpp_gap / MAX_ABS_DELTA:8.2f}x  {where}")
        print("  -> sharpening COMPRESSES this minimum and flattening WIDENS it, "
              "which is the")
        print("     opposite of the brief's prediction. The statistic lives in the "
              "near-zero tail,")
        print("     not among the top moves. See this test's docstring.")

    identity = next(r for r in rows if r[0] == 1.0)
    assert identity[1] == pytest.approx(c10b_recorded, rel=1e-3), (
        f"the T=1.0 minimum gap is {identity[1]:.6e}, not the {c10b_recorded:.3e} "
        f"C10b recorded. The identity path is supposed to be bit-identical to the "
        f"pre-C11b one, so this is a regression in the gather, not a temperature "
        f"effect.")

    # The manifest's table and this run must agree; a generator that measured a
    # different corpus than the tests read is the failure that check exists for.
    for temperature, cpp_gap, ref_gap, _ in rows:
        recorded = manifest["min_nonzero_prior_gap"][f"{temperature:g}"]["min_nonzero_gap"]
        assert ref_gap == pytest.approx(recorded, rel=1e-9), (
            f"at T={temperature} the manifest records a minimum gap of {recorded:.6e} "
            f"and the golden columns give {ref_gap:.6e}")


# ---------------------------------------------------------------------------
# Invariant 1 — T = 1.0 is the temperature-absent path, bit for bit
# ---------------------------------------------------------------------------

def test_t_equals_one_is_bit_identical_to_the_temperature_absent_path(golden):
    """The cheapest possible regression guard on the whole feature.

    Division by 1.0f is exact in IEEE-754, so this holds automatically — and it
    is asserted anyway, over all 15,036 priors, on RAW BIT PATTERNS rather than
    with a tolerance. What it protects is everything measured before C11b: Gate
    1, Gate 2's recorded distributions, the frozen POLICY_TEMPERATURE = 1.0
    configuration, and the C10b graph benchmarks all stand or fall on the claim
    that the default path did not move. `apply_policy_temperature` returns early
    at exactly 1.0f, so the two paths are the same instructions and not merely
    the same answer; this is the test that says so.
    """
    fens = golden["fens"]
    offsets = golden["move_offset"]
    source = np.load(GATE2_DUMP, allow_pickle=True)
    logits = source["logits"]

    for i, fen in enumerate(fens):
        absent_moves, absent = guofish_core.gather_softmax(str(fen), logits[i])
        explicit_moves, explicit = guofish_core.gather_softmax(str(fen), logits[i], 1.0)
        assert list(absent_moves) == list(explicit_moves)
        assert np.array_equal(np.asarray(absent).view(np.uint32),
                              np.asarray(explicit).view(np.uint32)), (
            f"position {i} ({fen}) differs between the temperature-absent call and "
            f"the explicit T=1.0 call. Division by 1.0f is exact, so any difference "
            f"here means the guard in apply_policy_temperature is not doing what it "
            f"claims and every pre-C11b measurement is invalidated.")

    del offsets  # the comparison is per-position; the CSR offsets are not needed


def test_a_temperature_of_zero_or_less_is_refused_everywhere():
    """It DIVIDES, so 0 is a division by zero and a negative inverts the policy.

    Three independent layers refuse it and all three are checked, because each
    one is the only one some caller passes through: the gather (a test or tool
    calling it directly), the search constructor (anything building a
    SearchConfig), and `EngineConfig.validate` (a `setoption` from a GUI).
    """
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    row = np.zeros(guofish_core.POLICY_SIZE, dtype=np.uint16)

    for bad in (0.0, -1.0, -0.5):
        with pytest.raises(ValueError, match="must be > 0"):
            guofish_core.gather_softmax(fen, row, bad)
        with pytest.raises(ValueError, match="must be > 0"):
            guofish_core.ReplaySearchQ32(
                guofish_core.SearchConfig(policy_temperature=bad))

    from playing.v6 import playv6
    for bad in (0.0, -1.0):
        with pytest.raises(playv6.ConfigError, match="must be > 0"):
            playv6.EngineConfig(policy_temperature=bad)


def test_the_replay_path_refuses_a_temperature_it_could_only_ignore():
    """A search with no live evaluator replays priors that are already softmaxed.

    Setting a temperature there is accepted-and-ignored — the same shape of
    defect C11 removed from the Python surface, one layer down. It throws at
    search entry instead, once per search rather than per node.
    """
    search = guofish_core.ReplaySearchQ32(
        guofish_core.SearchConfig(policy_temperature=0.8, arena_capacity=4096))
    # A one-entry dump with a key nothing will ever look up. It exists only to
    # get past the "no evaluation source at all" check, which is a different and
    # more fundamental complaint and is reported first — this test is about the
    # search having a source that temperature cannot reach, not about it having
    # none.
    search.load_dump(
        keys=np.zeros(1, dtype=np.uint64),
        is_root=np.zeros(1, dtype=np.uint8),
        move_offset=np.array([0, 1], dtype=np.uint64),
        moves=np.zeros(1, dtype=np.uint16),
        priors=np.ones(1, dtype=np.float32),
        values=np.zeros(1, dtype=np.float64))
    search.set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    with pytest.raises(ValueError, match="no live evaluator"):
        search.search(4)


# ---------------------------------------------------------------------------
# Invariant 2 — the ROOT is softmaxed at the same temperature as the interior
# ---------------------------------------------------------------------------

class _SyntheticNetwork:
    """The callback half of a `LiveEvaluator` whose network is a hash.

    A CLASS RATHER THAN A CLOSURE, so the reference cycle it necessarily forms
    can be broken on purpose. `LiveEvaluator` holds its callback and the
    callback must reach the evaluator's buffers, so evaluator -> callback ->
    evaluator is unavoidable — and pybind11 instances are not traversable by
    Python's cycle collector, so `gc` cannot break it either. Left alone, every
    evaluator a test builds survives to interpreter exit and LeakSanitizer
    reports it with `guofish_core` in the stack.

    That matters beyond tidiness: README_BUILD.md's leak discriminator is "no
    leaked allocation's stack mentions guofish_core", and it has read clean for
    eleven chunks. A test fixture that quietly breaks it costs the project a
    standing check. `_release_synthetic_evaluators` below clears
    `self.evaluator` after every test, which drops the only strong reference
    back and lets the evaluator die deterministically.

    `switch_interval=0` at construction leaves the interpreter's setting alone:
    this is a correctness fixture, not a throughput one, and a test has no
    business changing a process-global scheduler knob out from under whatever
    runs next.
    """

    def __init__(self):
        self.evaluator = None

    def __call__(self, count):
        tokens = self.evaluator.input_view()
        policy = self.evaluator.policy_view()
        value = self.evaluator.value_view()
        for row in range(count):
            seed = int(np.asarray(tokens[row], dtype=np.int64).sum() & 0x7FFFFFFF)
            rng = np.random.default_rng(seed)
            # float32 -> bf16 by truncation, which is what the bit pattern is.
            floats = rng.normal(0.0, 3.0, size=guofish_core.POLICY_SIZE).astype(np.float32)
            policy[row, :] = (floats.view(np.uint32) >> 16).astype(np.uint16)
            value[row] = np.float32(np.tanh(seed % 1000 / 1000.0 - 0.5))


_LIVE_NETWORKS: list = []


@pytest.fixture(autouse=True)
def _release_synthetic_evaluators():
    """Break every fixture cycle this module built, after every test."""
    yield
    for network in _LIVE_NETWORKS:
        network.evaluator = None
    _LIVE_NETWORKS.clear()


def _synthetic_evaluator(max_batch: int = 8):
    """A LiveEvaluator whose network is a hash. No torch, no CUDA, no model.

    The callback reads the 68 int32 tokens C++ wrote for each row, derives a
    deterministic 4096-wide bf16 policy row from them, and writes a value. That
    is all `expand_from_live_row` needs, and it means the invariants this
    section proves are proved on every platform the suite runs on rather than
    wherever CUDA happens to be present (Amendment D).
    """
    network = _SyntheticNetwork()
    evaluator = guofish_core.LiveEvaluator(max_batch, network, 0.0)
    network.evaluator = evaluator
    _LIVE_NETWORKS.append(network)
    return evaluator


def _root_child_priors(search):
    """{uci: prior} over the root's children, out of the arena."""
    arrays = search.dump_tree_arrays(0)
    return {guofish_core.move_to_uci(int(move)): float(prior)
            for depth, move, prior in zip(arrays["depth"], arrays["move"],
                                          arrays["prior"])
            if depth == 1}


ROOT_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_the_root_is_expanded_at_the_configured_temperature(temperature):
    """C11b requirement 3, and the defect it exists to prevent from returning.

    The reference softmaxes the root on the GPU (`_expand_root` hands `expand()`
    a CUDA tensor) and every interior node on the CPU, so it has two answers for
    one position — the split C10 measured at 1.9e-9 and unified away by routing
    the root through `evaluate_and_expand` like everything else. A temperature
    threaded into the interior path alone would re-create that inconsistency
    under a new name and at a magnitude thousands of times larger.

    So this asserts the strongest available statement: the root's stored child
    priors are EXACTLY the priors `gather_softmax` produces for the root position
    at the configured T — same function, same temperature, no tolerance. If the
    root were expanded at T = 1.0 while the config said 0.7, this fails at 0.7
    and 1.5 and passes only at the identity, which is precisely the signature of
    the defect.
    """
    evaluator = _synthetic_evaluator()
    search = guofish_core.ReplaySearchQ32(guofish_core.SearchConfig(
        policy_temperature=temperature, arena_capacity=1 << 16))
    search.set_evaluator(evaluator)
    try:
        search.set_position(ROOT_FEN)
        # One simulation: enough to force the root expansion, few enough that the
        # root's own policy row is the first and only row the evaluator sees.
        search.search(1)
        stored = _root_child_priors(search)

        row = np.asarray(evaluator.policy_view()[0]).copy()
        moves, expected = guofish_core.gather_softmax(ROOT_FEN, row, temperature)
    finally:
        search.set_evaluator(None)

    assert set(stored) == set(moves), (
        f"the root's children are not the position's canonical move list: "
        f"stored {sorted(stored)}, expected {sorted(moves)}")
    for uci, want in zip(moves, np.asarray(expected)):
        assert np.float32(stored[uci]).view(np.uint32) == np.float32(want).view(np.uint32), (
            f"root child {uci} carries prior {stored[uci]!r} but the gather at "
            f"T={temperature} produces {float(want)!r}. The root is not being "
            f"softmaxed at the configured temperature.")


def _interior_child_priors(search, uci):
    """{uci: prior} for the children of the ROOT'S CHILD reached by `uci`.

    `dump_tree_arrays` is DFS preorder with a depth column, so the children of
    the depth-1 entry for `uci` are the depth-2 entries that follow it before
    the next entry at depth <= 1. Reading the interior node's priors straight
    out of the arena is what makes the comparison below a statement about the
    EXPANSION rather than about `apply_move`'s promotion machinery.
    """
    arrays = search.dump_tree_arrays(0)
    depth, move, prior = arrays["depth"], arrays["move"], arrays["prior"]
    for i in range(len(depth)):
        if depth[i] == 1 and guofish_core.move_to_uci(int(move[i])) == uci:
            out = {}
            j = i + 1
            while j < len(depth) and depth[j] > 1:
                if depth[j] == 2:
                    out[guofish_core.move_to_uci(int(move[j]))] = float(prior[j])
                j += 1
            return out
    return {}


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_the_root_and_the_interior_are_sharpened_identically(temperature):
    """ONE position, expanded both ways, compared on bit patterns.

    The reference's root/interior split is only visible when a single position
    is expanded as a root AND reached as an interior node, which is what this
    does: search from ROOT_FEN until some child has been expanded, read that
    child's stored priors out of the arena, then root a second search at the
    same position and read ITS children. Same synthetic network (the callback
    seeds off the token row, so one position always yields one policy row), same
    temperature — so the two must agree exactly.

    Compared with `==` on the raw bit patterns and not with a tolerance, because
    "close" is precisely what the reference's two paths already are: 1.9e-9
    apart across 6 of 37 priors, enough to flip a best move at 200 sims once the
    root position recurs four plies down. C10 refused to accept that and
    unified the paths; this is the test that keeps the temperature from
    re-introducing it.

    The child is chosen as the MOST-VISITED one rather than named, so the test
    does not depend on the synthetic network happening to like a particular
    move — which it did not, the first time this was written against a hardcoded
    `g8f6` that drew zero visits.
    """
    import chess

    evaluator = _synthetic_evaluator()
    search = guofish_core.ReplaySearchQ32(guofish_core.SearchConfig(
        policy_temperature=temperature, arena_capacity=1 << 16))
    search.set_evaluator(evaluator)
    try:
        search.set_position(ROOT_FEN)
        search.search(600)
        arrays = search.dump_tree_arrays(0)
        visits, child_uci = max(
            (int(v), guofish_core.move_to_uci(int(m)))
            for d, m, v in zip(arrays["depth"], arrays["move"], arrays["visits"])
            if d == 1)
        as_interior = _interior_child_priors(search, child_uci)
    finally:
        search.set_evaluator(None)

    assert visits > 0 and as_interior, (
        f"no root child was expanded in 600 simulations (best was {child_uci} with "
        f"{visits} visits), so there is no interior expansion to compare against")

    board = chess.Board(ROOT_FEN)
    board.push(chess.Move.from_uci(child_uci))
    child_fen = board.fen(en_passant="fen")

    evaluator = _synthetic_evaluator()
    search = guofish_core.ReplaySearchQ32(guofish_core.SearchConfig(
        policy_temperature=temperature, arena_capacity=1 << 16))
    search.set_evaluator(evaluator)
    try:
        search.set_position(child_fen)
        search.search(1)
        as_root = _root_child_priors(search)
    finally:
        search.set_evaluator(None)

    assert set(as_root) == set(as_interior), (
        f"{child_fen} has different children as a root ({sorted(as_root)}) and as "
        f"an interior node ({sorted(as_interior)})")
    for uci in as_root:
        assert np.float32(as_root[uci]).view(np.uint32) == \
               np.float32(as_interior[uci]).view(np.uint32), (
            f"at T={temperature}, move {uci} carries prior {as_root[uci]!r} when "
            f"{child_fen} is a root and {as_interior[uci]!r} when it is an interior "
            f"node. That is the reference's root/interior split, re-created by the "
            f"temperature — the exact defect C10 unified away.")


# ---------------------------------------------------------------------------
# Invariant 3 — changing the temperature drops the tree AND the cache
# ---------------------------------------------------------------------------

class _EvaluatorStub:
    """What `Engine.reconfigure` needs of an evaluator: a `.core`.

    Standing one of these up lets the clear-on-change invariant be tested on the
    real `Engine.reconfigure` — the production code path a `setoption` takes —
    without a checkpoint, CUDA or torch. `ensure_ready` is bypassed by setting
    `_ready` directly, which is the only thing being faked here; everything
    below `reconfigure` is the shipping object.
    """

    def __init__(self, core):
        self.core = core
        self.max_batch = core.max_batch

    def close(self):
        """`Engine.close()` calls this. Nothing to release: the LiveEvaluator's
        buffers are C++-owned and freed when the last reference to it goes."""


def _ready_engine(temperature: float):
    from playing.v6 import playv6

    config = playv6.EngineConfig(policy_temperature=temperature,
                                 cache_entries=4096, arena_capacity=1 << 16)
    engine = playv6.Engine(config)
    evaluator = _synthetic_evaluator()
    engine.evaluator = _EvaluatorStub(evaluator)
    engine.search = guofish_core.ReplaySearchQ32(config.to_search_config())
    engine.search.set_evaluator(evaluator)
    engine.value_scale = 100.0
    engine._ready = True
    return engine, evaluator


def test_changing_the_temperature_drops_both_the_tree_and_the_cache(capsys):
    """C11b requirement 4, on the production `Engine.reconfigure`.

    A `setoption name PolicyTemperature value 0.7` between moves — no
    `ucinewgame`, which is the ordinary Cutechess sequence — must not leave a
    tree whose children carry priors sharpened at the old temperature while
    every new expansion uses the new one. THE CACHE IS THE OBVIOUS HALF, because
    it is visibly full of post-softmax prior vectors; the TREE is the half that
    gets missed, and it is the half that survives a move.

    Both are asserted after the change, and both are asserted to have been
    non-empty before it — a test that cleared an already-empty cache and an
    already-empty tree would pass against an engine that does nothing.
    """
    engine, _ = _ready_engine(1.0)
    try:
        engine.set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", [])
        engine.search.search(400)

        nodes_before = int(engine.search.nodes)
        cache_before = int(engine.search.cache_stats()["size"])
        root_before = int(engine.search.root_visits)
        search_before = engine.search

        with capsys.disabled():
            print()
            print(f"before PolicyTemperature 1.0 -> 0.7: nodes={nodes_before} "
                  f"root_visits={root_before} cache_entries={cache_before}")

        assert nodes_before > 1, "the tree is empty before the change; nothing to drop"
        assert cache_before > 0, "the cache is empty before the change; nothing to flush"

        engine.reconfigure(engine.config.replace(policy_temperature=0.7))

        assert engine.search is not search_before, (
            "reconfigure did not rebuild the search object, so the arena and the "
            "cache that hold old-temperature priors are both still live")
        nodes_after = int(engine.search.nodes)
        cache_after = int(engine.search.cache_stats()["size"])

        with capsys.disabled():
            print(f"after : nodes={nodes_after} cache_entries={cache_after} "
                  f"base_fen={engine._base_fen!r}")

        assert cache_after == 0, (
            f"the transposition cache still holds {cache_after} entries after the "
            f"temperature changed. Every one is a post-softmax prior vector computed "
            f"at the OLD temperature, and a hit would expand a node at a sharpness "
            f"nothing in the configuration says.")
        assert nodes_after <= 1, (
            f"the search tree still holds {nodes_after} nodes after the temperature "
            f"changed. Its children carry priors materialised at expansion time under "
            f"the OLD temperature; new expansions would use the new one, and one tree "
            f"would be running two temperatures.")
        assert engine._base_fen is None and engine._moves == [], (
            "the position was not cleared, so the next `position` command would try "
            "to extend a tree that no longer exists")
    finally:
        engine.close()


def test_a_temperature_change_is_what_forces_the_rebuild():
    """The field is in `_SEARCH_CONFIG_FIELDS`, checked through the predicate.

    Structural counterpart to the behavioural test above. It is worth having
    both: the behavioural one proves the drop happens, this one names the reason
    it happens, so a future edit that removed the field from the tuple fails
    with a message about the tuple rather than with a mysterious prior mismatch
    several chunks later.
    """
    from playing.v6 import playv6

    assert "policy_temperature" in playv6._SEARCH_CONFIG_FIELDS
    base = playv6.EngineConfig()
    assert playv6._search_config_differs(base, base.replace(policy_temperature=0.7))
    assert not playv6._search_config_differs(base, base.replace(policy_temperature=1.0))


def test_the_python_surface_no_longer_refuses_a_temperature():
    """C11b's mandate, and the one thing that must NOT have moved with it.

    `policy_temperature` leaves `UNSUPPORTED_IN_CORE` because it is implemented.
    Dirichlet noise stays, because the arena still stores one prior per child
    and nothing preserves the untouched network distribution the reference's C3b
    fix derives noise from. The consequence — the C++ engine cannot generate
    self-play training data — is recorded in DECISIONS.md and in playv6's module
    docstring, not left to be discovered.
    """
    from playing.v6 import playv6

    assert "policy_temperature" not in playv6.UNSUPPORTED_IN_CORE
    assert "dirichlet_epsilon" in playv6.UNSUPPORTED_IN_CORE

    accepted = playv6.EngineConfig(policy_temperature=0.7)
    assert accepted.to_search_config().policy_temperature == pytest.approx(0.7)

    with pytest.raises(playv6.ConfigError, match="dirichlet_epsilon"):
        playv6.EngineConfig(dirichlet_epsilon=0.25)
