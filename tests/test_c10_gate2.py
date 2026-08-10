"""C10 Gate 2 — the C++ gather+softmax against ATen, over recorded logits.

WHAT THE BOUND IS, AND WHY IT IS NOT A ULP COUNT
================================================
The chunk brief originally asked for <= 4 ULP. That is unachievable by a correct
implementation and the criterion was replaced, before any code was written, with

    max absolute delta <= 1e-6, and ZERO prior-ordering inversions.

The reason is measurable and is measured here. `torch.softmax` is not
permutation-invariant: over 200 sampled permutations of one node's legal logits,
scope §2.6 found only 109 reproducing bit-identical priors, max delta 3e-7. The
two sides reduce in different orders by construction — Python gathers in
python-chess's generation order, C++ in chess-library's — so permutation ALONE
puts ~3e-7 on the table before the softmax implementations are even compared.
``test_the_permutation_alone_costs_this_much`` measures that term separately, out
of the golden file's third column, so the claim is not taken on trust.

WHY "ZERO INVERSIONS" IS THE LOAD-BEARING HALF
==============================================
PUCT reads priors through comparisons. A prior that is 3e-7 low changes a score
by 3e-7 and changes nothing; a prior that has swapped places with its neighbour
changes which child is selected, and from there the trees diverge structurally.
So the magnitude bound is the sanity check and the ORDERING is the property that
matters — which is exactly what the mutation drill (tools/drill_c10_gate2.py)
attacks: it swaps the two closest-scored priors in a scratch copy of the
reference, leaving the magnitude bound comfortably satisfied, and
``test_there_are_no_prior_ordering_inversions`` has to fail anyway.

THE THREE REFERENCE COLUMNS
===========================
``priors_cpu_pychess``   the reference's INTERIOR path: ATen CPU softmax.
``priors_gpu_pychess``   the reference's ROOT path: ATen CUDA softmax.
``priors_cpu_libchess``  ATen CPU softmax reduced in chess-library's order.

C++ is compared against all three, because the first two are the reference
disagreeing with ITSELF — the device split C5 found and C10 unifies away — and a
gate that picked one of them would be picking a side in an inconsistency rather
than measuring a port. ``test_the_reference_disagrees_with_itself`` reports the
size of that split as a first-class number; it is the divergence C10 accepts by
design (DECISIONS.md, C10) and Gate 2b is what arbitrates it.

WHAT THIS FILE DOES NOT NEED
============================
No torch, no GPU, no model, no python-chess. The logits were recorded once by the
Python reference (tools/gen_c10_gate2_golden.py) and reach this test only through
the golden file, per Amendment A. That is also why it runs identically on Linux
and Windows and contributes no skips to either.

``GUOFISH_GOLDEN_C10_GATE2`` and ``GUOFISH_GOLDEN_C10_GATE2_MANIFEST`` override
the inputs, so the mutation drill can point the suite at a corrupted COPY in a
scratch directory and never touch ``golden/`` (Amendment B).
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent

DUMP = (REPO_ROOT / "golden" / "c10_gate2.npz", "GUOFISH_GOLDEN_C10_GATE2")
MANIFEST = (REPO_ROOT / "golden" / "c10_gate2_manifest.json",
            "GUOFISH_GOLDEN_C10_GATE2_MANIFEST")
CORPUS = REPO_ROOT / "golden" / "c10_corpus.json"

# The acceptance criterion, stated once.
MAX_ABS_DELTA = 1e-6

COLUMNS = ("priors_cpu_pychess", "priors_gpu_pychess", "priors_cpu_libchess")

COLUMN_MEANING = {
    "priors_cpu_pychess": "reference INTERIOR path (ATen CPU, python-chess order)",
    "priors_gpu_pychess": "reference ROOT path (ATen CUDA, python-chess order)",
    "priors_cpu_libchess": "ATen CPU, chess-library order (permutation removed)",
}


def _path(spec):
    default, env = spec
    return Path(os.environ.get(env, default))


def _overridden(spec) -> bool:
    return spec[1] in os.environ


@pytest.fixture(scope="module")
def golden():
    path = _path(DUMP)
    if not path.exists():
        pytest.fail(
            f"missing golden file {path}. Generate it with\n"
            f"    python tools/gen_c10_gate2_golden.py\n"
            f"It is produced by the Python reference only (Global Rule 2); it "
            f"cannot be reconstructed from C++ output.")
    return np.load(path, allow_pickle=True)


@pytest.fixture(scope="module")
def manifest():
    path = _path(MANIFEST)
    if not path.exists():
        pytest.fail(f"missing manifest {path}; see tools/gen_c10_gate2_golden.py")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def measured(golden):
    """Run the production gather over every recorded row, once.

    `guofish_core.gather_softmax` is the function the dispatcher calls, not a
    restatement of it, so this measures the shipping code path. Module-scoped
    because 500 positions is a second of work and four tests read it.
    """
    fens = golden["fens"]
    offsets = golden["move_offset"]
    logits = golden["logits"]
    moves = golden["moves"]

    priors = np.zeros(int(offsets[-1]), dtype=np.float32)
    mismatched = []
    for i, fen in enumerate(fens):
        begin, end = int(offsets[i]), int(offsets[i + 1])
        cpp_moves, cpp_priors = guofish_core.gather_softmax(str(fen), logits[i])
        if list(cpp_moves) != [str(m) for m in moves[begin:end]]:
            mismatched.append((str(fen), list(cpp_moves), [str(m) for m in moves[begin:end]]))
        priors[begin:end] = cpp_priors
    return priors, mismatched


def _deltas(measured_priors, reference):
    return np.abs(measured_priors.astype(np.float64) - reference.astype(np.float64))


def _distribution(deltas):
    """Every number the criterion asks to be reported, not just the max."""
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

    An INVERSION is a pair the reference ranks one way and C++ ranks the other.
    That is the failure the criterion names, because PUCT reads priors only
    through comparisons.

    A COLLAPSE is a pair the reference separates and C++ ties. It is not an
    inversion — no order is reversed — but it hands the decision to the child
    ordering instead of to the network, so it is counted and reported rather
    than ignored. It should be zero: two distinct bf16 logits produce two
    distinct exp() values at this scale.
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


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_the_golden_file_records_the_reference_it_came_from(manifest):
    """Amendment A: the interpreter, the library and the checkpoint are pinned.

    A Gate 2 delta is a claim about two implementations of one function. If the
    file could have been produced by a different checkpoint, or a different
    torch, the claim is about something else — so the provenance is asserted to
    be present rather than assumed to have been recorded.
    """
    assert manifest["generator"] == "tools/gen_c10_gate2_golden.py"
    assert manifest["python"].startswith("3.13"), manifest["python"]
    assert manifest["python_chess"] == "1.11.2", manifest["python_chess"]
    assert manifest["autocast_dtype"] == "torch.bfloat16"
    assert manifest["policy_size"] == guofish_core.POLICY_SIZE
    assert len(manifest["model"]["sha256"]) == 64
    assert manifest["corpus"]["positions"] >= 500, (
        f"the acceptance criterion says ~500 positions; the manifest records "
        f"{manifest['corpus']['positions']}")


def test_the_corpus_is_the_one_gate_2b_runs(manifest):
    """Gate 2 and Gate 2b must measure the same positions.

    A Gate 2b disagreement whose position Gate 2 never gathered is a
    disagreement with nowhere to look, so the two share one corpus file and the
    manifest records its digest. Skipped only when the drill has redirected the
    golden file, because a corrupted copy is not expected to match anything.
    """
    if _overridden(DUMP) or _overridden(MANIFEST):
        pytest.skip("golden override in effect (mutation drill); corpus digest not comparable")
    digest = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    assert digest == manifest["corpus"]["sha256"], (
        f"golden/c10_corpus.json has changed since the Gate 2 golden was "
        f"generated.\n  manifest: {manifest['corpus']['sha256']}\n  on disk : {digest}\n"
        f"Regenerate both goldens, or restore the corpus.")


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_the_move_lists_agree(measured, golden):
    """Before any prior is compared, the two sides must agree on the moves.

    The gather is positional: prior k belongs to move k. If chess-library and
    python-chess disagreed about the legal move set, or about canonical order,
    every delta below would be comparing unrelated moves and would still produce
    a number. This is the C1 regression check standing in front of the gate.
    """
    _, mismatched = measured
    if mismatched:
        fen, cpp, ref = mismatched[0]
        pytest.fail(
            f"{len(mismatched)} positions have a different canonical move list.\n"
            f"first: {fen}\n  C++   : {cpp}\n  golden: {ref}")


@pytest.mark.parametrize("column", COLUMNS)
def test_the_max_absolute_delta_is_within_the_gate(measured, golden, column, capsys):
    """Criterion 1a: max |C++ prior - ATen prior| <= 1e-6, distribution reported."""
    priors, _ = measured
    deltas = _deltas(priors, golden[column])
    dist = _distribution(deltas)

    with capsys.disabled():
        print()
        print(f"Gate 2 error distribution vs {column}")
        print(f"  ({COLUMN_MEANING[column]})")
        print(f"  {_format('|C++ - ATen|', dist)}")

    assert dist["max"] <= MAX_ABS_DELTA, (
        f"max absolute delta {dist['max']:.6e} exceeds the {MAX_ABS_DELTA:.0e} gate "
        f"against {column} ({COLUMN_MEANING[column]}); {dist['over_1e-6']} of "
        f"{dist['count']} priors are over the bound")


@pytest.mark.parametrize("column", COLUMNS)
def test_there_are_no_prior_ordering_inversions(measured, golden, column, capsys):
    """Criterion 1b, and the half that decides which move gets searched.

    Reported before it is asserted: a collapse count of zero is as much a part of
    the result as an inversion count of zero, and neither is visible from the
    magnitude distribution.
    """
    priors, _ = measured
    inversions, collapses = _order_violations(priors, golden[column], golden["move_offset"])

    with capsys.disabled():
        print(f"  ordering vs {column:22s}: {len(inversions)} inversions, "
              f"{len(collapses)} collapsed pairs")

    if collapses:
        i, a, b, ref_a, ref_b, got = collapses[0]
        print(f"NOTE: {len(collapses)} pairs the reference separates and C++ ties; "
              f"first at position {i}, moves {a}/{b}: reference {ref_a!r} vs {ref_b!r}, "
              f"C++ both {got!r}")

    if inversions:
        lines = []
        for i, a, b, ref_a, ref_b, got_a, got_b in inversions[:10]:
            lines.append(f"  position {i} moves {a}/{b}: "
                         f"reference {ref_a:.9e} vs {ref_b:.9e}, "
                         f"C++ {got_a:.9e} vs {got_b:.9e}")
        pytest.fail(
            f"{len(inversions)} prior-ordering inversions against {column}. PUCT reads "
            f"priors only through comparisons, so an inversion changes which child is "
            f"selected — this is the failure the criterion names.\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# What the numbers are made of
# ---------------------------------------------------------------------------

def test_the_permutation_alone_costs_this_much(golden, capsys):
    """The 1e-6 bound's justification, measured rather than asserted.

    Both columns are ATen. They differ only in the order the softmax reduced in —
    python-chess's generation order against chess-library's. Whatever shows up
    here is the floor no correct C++ implementation can get under, and it is why
    a ULP-tight bound was rejected.
    """
    deltas = _deltas(golden["priors_cpu_libchess"], golden["priors_cpu_pychess"])
    dist = _distribution(deltas)
    with capsys.disabled():
        print()
        print("Where the delta comes from (both sides ATen; neither is C++)")
        print(f"  {_format('permutation only', dist)}")
    assert dist["max"] <= MAX_ABS_DELTA, (
        f"the permutation term alone is {dist['max']:.3e}, at or over the gate. "
        f"The gate cannot then be a statement about this port.")


def test_the_reference_disagrees_with_itself(golden, capsys):
    """The device split C5 found and C10 unifies away — as a number.

    `_expand_root` softmaxes on the GPU, `BatchedEvaluator` on the CPU, so the
    reference has two answers for one position depending on where it is expanded.
    C++ has one (DECISIONS.md, C10). This test does not fail on the split; it
    records its size, because "the port diverges from the reference at roots" is
    only a defensible sentence with the magnitude attached.
    """
    deltas = _deltas(golden["priors_gpu_pychess"], golden["priors_cpu_pychess"])
    dist = _distribution(deltas)
    with capsys.disabled():
        print(f"  {_format('reference root vs int.', dist)}")
        print(f"  -> C10 unifies on ONE path; this is the divergence it accepts at roots.")
    assert dist["max"] <= MAX_ABS_DELTA, (
        f"the reference's own root/interior split is {dist['max']:.3e}, at or over the "
        f"gate C++ is held to. Unifying is then not a refinement but a change of "
        f"answer, and Gate 2b's arbitration is the only evidence left.")


def test_the_priors_are_a_distribution(measured, golden):
    """Every position's C++ priors are non-negative and sum to 1 in float32.

    Cheap, and it catches the class of gather bug the delta comparison cannot: a
    permutation that drops or duplicates an entry still lands close to the
    reference on most priors while breaking the sum.
    """
    priors, _ = measured
    offsets = golden["move_offset"]
    assert (priors >= 0).all()
    for i in range(len(offsets) - 1):
        begin, end = int(offsets[i]), int(offsets[i + 1])
        total = float(priors[begin:end].astype(np.float64).sum())
        assert abs(total - 1.0) < 1e-5, f"position {i} priors sum to {total!r}"


def test_the_gather_reads_the_recorded_bits_and_not_a_conversion(golden):
    """bf16 -> float is a shift, and the port had better be doing exactly that.

    Constructed rather than sampled: a row whose only non-zero entries are the
    legal moves' own indices, each holding a distinct bf16 pattern. If the
    widening rounded, or if the gather indexed by anything other than
    `from_square * 64 + to_square`, the resulting priors would not be the softmax
    of the values that were planted.
    """
    fen = str(golden["fens"][0])
    offsets = golden["move_offset"]
    moves = [str(m) for m in golden["moves"][int(offsets[0]):int(offsets[1])]]

    # bf16 patterns for a spread of small integers, planted at each legal move's
    # policy index in canonical order.
    row = np.zeros(guofish_core.POLICY_SIZE, dtype=np.uint16)
    planted = np.linspace(-4.0, 4.0, len(moves), dtype=np.float32)
    as_bf16 = (planted.view(np.uint32) >> 16).astype(np.uint16)
    for uci, bits in zip(moves, as_bf16):
        from_square = (ord(uci[0]) - ord("a")) + 8 * (ord(uci[1]) - ord("1"))
        to_square = (ord(uci[2]) - ord("a")) + 8 * (ord(uci[3]) - ord("1"))
        row[from_square * 64 + to_square] = bits

    got_moves, got_priors = guofish_core.gather_softmax(fen, row)
    assert list(got_moves) == moves

    # The truncation the planting performed is what bf16 IS, so recovering the
    # float from the bits is exact and the expected softmax is unambiguous.
    exact = (as_bf16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    # Promotions collide on one policy index: four moves, one planted value, and
    # whichever was written last wins. Re-read the row rather than reuse
    # `exact` so the expectation matches what is actually in memory.
    seen = []
    for uci in moves:
        from_square = (ord(uci[0]) - ord("a")) + 8 * (ord(uci[1]) - ord("1"))
        to_square = (ord(uci[2]) - ord("a")) + 8 * (ord(uci[3]) - ord("1"))
        seen.append(np.uint32(row[from_square * 64 + to_square]) << 16)
    exact = np.array(seen, dtype=np.uint32).view(np.float32).astype(np.float64)

    shifted = np.exp(exact - exact.max())
    expected = shifted / shifted.sum()
    assert np.abs(np.asarray(got_priors, dtype=np.float64) - expected).max() < 1e-6
