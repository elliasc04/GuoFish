"""Amendment B mutation drill for C10 Gate 2 — does the gate have teeth?

THE BRIEF'S MUTATION, AND WHY IT IS THE INTERESTING ONE
=======================================================
    "In a scratch copy of the Gate 2 reference logits, flip two
     identically-scored priors. Verify the Gate 2 test strictly catches the
     prior-ordering inversion and fails."

Read literally, swapping two EXACTLY equal priors is a no-op — the array is
unchanged and there is nothing for any check to catch. The mutation that carries
the brief's intent is the neighbouring one: swap the two priors with the SMALLEST
NON-ZERO gap in the corpus. `swap-closest` does that, and
`test_there_are_no_prior_ordering_inversions` catches it. The brief's criterion
is met.

WHAT THE DRILL FOUND WHILE MEETING IT
=====================================
`swap-closest` was written expecting to be INVISIBLE to the magnitude half of
Gate 2 — an inversion far inside the 1e-6 tolerance, proving the ordering
criterion is load-bearing rather than implied. It is not invisible. **The
smallest non-zero gap between two priors anywhere in the 500-position corpus is
1.93e-6, which is above the 1e-6 bound**, so swapping the closest pair moves the
data by nearly twice the tolerance and both halves of the gate fire.

That is a fact about the corpus, not about the checks: bf16 logits quantise
coarsely (8 mantissa bits), and a coarse input makes a coarse output, so on this
corpus every pair the ordering check can catch is far enough apart that the
magnitude check catches it too. It is reported, not worked around — the drill
prints the minimum gap and treats the magnitude result on `swap-closest` as an
observation rather than a requirement.

The independence of the two criteria then has to be shown by construction, which
is what `invert-inside-tolerance` is for. It collapses the closest pair onto its
own midpoint and separates it there by one float32 ulp, the wrong way round:
each prior moves 9.6e-7, under the bound, and the pair comes out reversed. The
magnitude check MUST pass and the ordering check MUST fail, and the drill asserts
both. Without that mutation, "zero prior-ordering inversions" would be a
criterion this corpus cannot distinguish from the one beside it.

THE OTHER THREE
===============
  swap-adjacent   two priors adjacent in sorted order at a typical position,
                  rather than the corpus minimum, and chosen from the pairs that
                  are actually SEPARATED. Confirms the ordering check is not
                  sensitive only at one hand-picked place.
  nudge-over      one prior moved by 2e-6, just over the bound and in a direction
                  that does NOT reorder anything. The mirror of
                  invert-inside-tolerance: the magnitude check must fail and the
                  ordering check must pass, so neither half stands in for the
                  other in either direction.
  drop-a-move     one position's move list shortened by one. The move-list check
                  must fail — a gather compared against a different move list
                  would otherwise produce a plausible number about nothing.

`golden/` IS NEVER TOUCHED (Amendment B). Everything runs against copies in a
scratch directory, reached through GUOFISH_GOLDEN_C10_GATE2 and
GUOFISH_GOLDEN_C10_GATE2_MANIFEST, and the real files' SHA-256 is recorded before
and after so the report can say so rather than claim it. Copies and writes are
binary (Amendment E).

Usage:
    python tools/drill_c10_gate2.py
    python tools/drill_c10_gate2.py --only invert-inside-tolerance --keep
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GOLDEN = REPO_ROOT / "golden" / "c10_gate2.npz"
MANIFEST = REPO_ROOT / "golden" / "c10_gate2_manifest.json"

COLUMNS = ("priors_cpu_pychess", "priors_gpu_pychess", "priors_cpu_libchess")

ORDER_TEST = "test_there_are_no_prior_ordering_inversions"
DELTA_TEST = "test_the_max_absolute_delta_is_within_the_gate"
MOVES_TEST = "test_the_move_lists_agree"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as handle:
        return {k: handle[k] for k in handle.files}


def save(path: Path, arrays: dict) -> None:
    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

def _smallest_gap_pair(arrays: dict) -> tuple[int, int, int, float]:
    """(position, index a, index b, gap) for the closest non-equal prior pair.

    Searched over the reference column the gate's first parametrisation uses;
    the swap is then applied to all three so no column can quietly still agree.
    """
    priors = arrays["priors_cpu_pychess"]
    offsets = arrays["move_offset"]
    best = None
    for i in range(len(offsets) - 1):
        begin, end = int(offsets[i]), int(offsets[i + 1])
        chunk = priors[begin:end].astype(np.float64)
        order = np.argsort(chunk, kind="stable")
        gaps = np.diff(chunk[order])
        nonzero = np.nonzero(gaps > 0)[0]
        if nonzero.size == 0:
            continue
        k = nonzero[np.argmin(gaps[nonzero])]
        gap = float(gaps[k])
        if best is None or gap < best[3]:
            best = (i, int(order[k]), int(order[k + 1]), gap)
    assert best is not None, "no position has two distinct priors"
    return best


def mutate_swap_closest(arrays: dict) -> str:
    position, a, b, gap = _smallest_gap_pair(arrays)
    begin = int(arrays["move_offset"][position])
    for column in COLUMNS:
        col = arrays[column]
        col[begin + a], col[begin + b] = col[begin + b], col[begin + a]
    return (f"swapped the corpus' closest-scored prior pair: position {position}, "
            f"moves {a}/{b}, gap {gap:.3e}")


def mutate_swap_adjacent(arrays: dict) -> str:
    """An ordinary adjacent pair, away from the extreme swap-closest picks.

    Selected from the pairs that are actually SEPARATED. bf16 logits are coarse
    enough that exact ties are common — the four promotions from one square
    share a policy index by construction, and coincidences add more — and
    swapping two equal values changes nothing, which is a property of the
    mutation rather than of the check.
    """
    offsets = arrays["move_offset"]
    position = len(offsets) // 3
    begin, end = int(offsets[position]), int(offsets[position + 1])
    chunk = arrays["priors_cpu_pychess"][begin:end].astype(np.float64)
    order = np.argsort(chunk, kind="stable")
    separated = np.nonzero(np.diff(chunk[order]) > 0)[0]
    assert separated.size > 0, f"position {position} has no two distinct priors"
    k = int(separated[separated.size // 2])
    a, b = int(order[k]), int(order[k + 1])
    for column in COLUMNS:
        col = arrays[column]
        col[begin + a], col[begin + b] = col[begin + b], col[begin + a]
    return (f"swapped an ordinary adjacent pair at position {position}: moves {a}/{b}, "
            f"gap {abs(chunk[a] - chunk[b]):.3e}")


def mutate_invert_inside_tolerance(arrays: dict, measured=None) -> str:
    """Reverse one pair's order WITHOUT moving any prior more than the bound allows.

    THIS IS THE ONE THAT PROVES THE TWO CRITERIA ARE INDEPENDENT, and it exists
    because the corpus refused to prove it the easy way. `swap-closest` was meant
    to be invisible to the magnitude check; it is not, because the smallest
    non-zero gap between two priors anywhere in the corpus is ~1.9e-6, which is
    ABOVE the 1e-6 bound. bf16 logits quantise coarsely, and a coarse input makes
    a coarse output: on this corpus, every pair the ordering check can catch is
    far enough apart that the magnitude check catches it too.

    So the independence has to be constructed, and the cheapest construction is
    to collapse the pair onto its own MIDPOINT and separate it there by a single
    float32 ulp, the wrong way round:

        mid = (cpp[lo] + cpp[hi]) / 2
        ref[lo] = mid + ulp            ref[hi] = mid - ulp

    Each prior then moves by g/2 + ulp. With g = 1.93e-6 that is ~9.6e-7 — under
    the bound, so the magnitude check must PASS — while the pair comes out
    reversed, so the ordering check must FAIL. That is the whole argument for
    keeping an ordering criterion at all, and it now rests on a demonstration
    rather than on an assertion about a policy head this port does not have.

    If the corpus' closest gap were >= 2e-6 the construction would be impossible,
    and that would itself be the finding — reported, not worked around.
    """
    assert measured is not None
    position, a, b, gap = _smallest_gap_pair(arrays)
    begin = int(arrays["move_offset"][position])
    lo, hi = (a, b) if measured[begin + a] < measured[begin + b] else (b, a)

    mid = 0.5 * (measured[begin + lo] + measured[begin + hi])
    ulp = float(np.spacing(np.float32(mid)))
    moved = 0.5 * gap + ulp
    if moved >= 1e-6:
        raise AssertionError(
            f"reversing the corpus' closest pair (gap {gap:.3e}) moves each prior by "
            f"{moved:.3e}, which is at or over the 1e-6 bound. No pair can then be "
            f"reordered while staying inside the tolerance, and the ordering criterion "
            f"is strictly implied by the magnitude one on this corpus. That is the "
            f"finding; it is not worked around.")
    for column in COLUMNS:
        arrays[column][begin + lo] = np.float32(mid + ulp)
        arrays[column][begin + hi] = np.float32(mid - ulp)
    return (f"reversed position {position}'s closest pair (moves {lo}/{hi}, gap "
            f"{gap:.3e}) about its midpoint; each prior moved {moved:.3e}, inside "
            f"the 1e-6 bound")


def mutate_nudge_over(arrays: dict) -> str:
    """Move one prior just over the bound WITHOUT reordering anything.

    The nudge is applied to the position's largest prior and is upward, so the
    move that was top stays top and no pair changes places. Only the magnitude
    check can see it.
    """
    offsets = arrays["move_offset"]
    position = 7
    begin, end = int(offsets[position]), int(offsets[position + 1])
    top = begin + int(np.argmax(arrays["priors_cpu_pychess"][begin:end]))
    for column in COLUMNS:
        arrays[column][top] = np.float32(float(arrays[column][top]) + 2e-6)
    return (f"raised position {position}'s largest prior by 2e-6 (over the 1e-6 bound, "
            f"and upward so nothing changes places)")


def mutate_drop_a_move(arrays: dict) -> str:
    """Delete one move and its prior, shifting the CSR offsets after it."""
    offsets = arrays["move_offset"].copy()
    position = 3
    victim = int(offsets[position]) + 1
    arrays["moves"] = np.delete(arrays["moves"], victim)
    for column in COLUMNS:
        arrays[column] = np.delete(arrays[column], victim)
    offsets[position + 1:] -= 1
    arrays["move_offset"] = offsets
    return f"deleted move index 1 from position {position} and closed the CSR gap"


# name -> (mutate, the check that MUST fail, the complementary check, what the
# complement is expected to do). "pass" makes the complement a requirement — the
# two halves of the gate are being shown to be independent. "observe" only
# reports it, for the mutation where the corpus itself decides the answer.
MUTATIONS = {
    "swap-closest": (mutate_swap_closest, ORDER_TEST, DELTA_TEST, "observe"),
    "swap-adjacent": (mutate_swap_adjacent, ORDER_TEST, None, None),
    "invert-inside-tolerance": (mutate_invert_inside_tolerance, ORDER_TEST, DELTA_TEST, "pass"),
    "nudge-over": (mutate_nudge_over, DELTA_TEST, ORDER_TEST, "pass"),
    "drop-a-move": (mutate_drop_a_move, MOVES_TEST, None, None),
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_gate(scratch_npz: Path, scratch_manifest: Path, selector: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["GUOFISH_GOLDEN_C10_GATE2"] = str(scratch_npz)
    env["GUOFISH_GOLDEN_C10_GATE2_MANIFEST"] = str(scratch_manifest)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_c10_gate2.py", "-k", selector, "-q",
         "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma-separated mutation names")
    parser.add_argument("--keep", action="store_true", help="leave the scratch directory")
    args = parser.parse_args()

    if not GOLDEN.exists():
        print(f"missing {GOLDEN}; run tools/gen_c10_gate2_golden.py first", file=sys.stderr)
        return 2

    wanted = [n.strip() for n in args.only.split(",") if n.strip()] or list(MUTATIONS)
    unknown = [n for n in wanted if n not in MUTATIONS]
    if unknown:
        print(f"unknown mutation(s): {unknown}. Known: {list(MUTATIONS)}", file=sys.stderr)
        return 2

    before = {"npz": sha256_of(GOLDEN), "manifest": sha256_of(MANIFEST)}
    print("golden/ digests BEFORE the drill")
    for name, digest in before.items():
        print(f"  {name:9s} {digest}")

    # The C++ priors, for the one mutation that has to be positioned relative to
    # them, and for the gap statistic that explains why it exists.
    import guofish_core  # noqa: PLC0415 - kept out of module scope; the drill is not a test
    pristine = load(GOLDEN)
    measured = np.zeros(int(pristine["move_offset"][-1]), dtype=np.float64)
    for i, fen in enumerate(pristine["fens"]):
        begin, end = int(pristine["move_offset"][i]), int(pristine["move_offset"][i + 1])
        measured[begin:end] = guofish_core.gather_softmax(str(fen), pristine["logits"][i])[1]
    _, _, _, min_gap = _smallest_gap_pair(pristine)
    print(f"\ncorpus' smallest non-zero gap between two priors: {min_gap:.3e}")
    print(f"  the Gate 2 magnitude bound is 1.0e-06, so a swap of the closest pair is "
          f"{'NOT ' if min_gap > 1e-6 else ''}invisible to it -- see invert-inside-tolerance")

    scratch = Path(tempfile.mkdtemp(prefix="guofish-c10-drill-"))
    print(f"\nscratch: {scratch}\n")

    results = []
    try:
        for name in wanted:
            mutate, must_fail, complement, complement_expectation = MUTATIONS[name]
            work_npz = scratch / f"{name}.npz"
            work_manifest = scratch / f"{name}.manifest.json"
            # Binary copy, Amendment E. The manifest travels unchanged so the
            # provenance test is exercised against the same metadata.
            shutil.copyfile(MANIFEST, work_manifest)

            arrays = load(GOLDEN)
            if name == "invert-inside-tolerance":
                description = mutate(arrays, measured)
            else:
                description = mutate(arrays)
            save(work_npz, arrays)

            code, output = run_gate(work_npz, work_manifest, must_fail)
            caught = code != 0
            line = f"{name:14s} {description}"
            print(line)
            print(f"  {'CAUGHT  ' if caught else 'MISSED  '} {must_fail} "
                  f"-> pytest exit {code}")
            if not caught:
                print("  ---- pytest output ----")
                print("\n".join(output.splitlines()[-25:]))

            still_ok = None
            if complement is not None:
                other_code, _ = run_gate(work_npz, work_manifest, complement)
                passed = other_code == 0
                if complement_expectation == "pass":
                    still_ok = passed
                    print(f"  {'as required ' if passed else 'UNEXPECTED  '} {complement} "
                          f"exit {other_code}: this mutation must be invisible to it, "
                          f"which is what makes the two criteria independent")
                else:
                    print(f"  observed     {complement} exit {other_code}: "
                          f"{'it also fires' if not passed else 'blind to this one'}; "
                          f"a property of the corpus, not a requirement")

            results.append((name, caught, still_ok))
            print()
    finally:
        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)

    after = {"npz": sha256_of(GOLDEN), "manifest": sha256_of(MANIFEST)}
    print("golden/ digests AFTER the drill")
    for name, digest in after.items():
        mark = "unchanged" if digest == before[name] else "*** CHANGED ***"
        print(f"  {name:9s} {digest}  {mark}")

    print()
    failures = [n for n, caught, _ in results if not caught]
    unexpected = [n for n, _, still in results if still is False]
    for name, caught, still in results:
        note = "" if still is None else (
            "; complement blind, as required" if still else "; COMPLEMENT ALSO FIRED")
        print(f"  {name:24s} {'caught' if caught else 'MISSED'}{note}")

    if after != before:
        print("\nFAILED: the drill modified golden/. That is an Amendment B violation.")
        return 1
    if failures:
        print(f"\nFAILED: {failures} were not caught. The gate is not testing what it claims.")
        return 1
    if unexpected:
        print(f"\nFAILED: {unexpected} failed a check that should have been blind to them; "
              f"the two halves of Gate 2 are not independent.")
        return 1
    print("\nAll mutations caught, by the intended check and only by it. golden/ unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
