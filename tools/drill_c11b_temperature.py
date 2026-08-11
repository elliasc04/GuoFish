#!/usr/bin/env python
"""Amendment B mutation drill for the C11b Gate 2 temperature extension.

WHAT THIS DRILL SETTLES THAT C10'S COULD NOT
============================================
`tools/drill_c10_gate2.py` set out to show that Gate 2's two criteria are
INDEPENDENT — that "zero prior-ordering inversions" catches something the
1e-6 magnitude bound does not — and found it could not do so naturally at
T = 1.0. The corpus' smallest non-zero gap between two priors is 1.927e-06,
almost twice the bound, so any swap the ordering check can catch also moves the
data far enough for the magnitude check to catch it. C10 had to CONSTRUCT the
independence with `invert-inside-tolerance`, which collapses the closest pair
onto its midpoint and re-separates it by one ulp the wrong way round.

**At T = 0.7 the construction is unnecessary, because the corpus supplies it.**
Sharpening drives the near-zero tail of each position's prior distribution toward
zero and collapses the absolute gaps between adjacent tiny priors: the corpus
minimum falls from 1.927e-06 at T = 1.0 to 5.884e-08 at T = 0.7, a factor of 33
and well under the bound. So `swap-closest` at T = 0.7 is exactly the mutation
C10 wanted and could not have — a genuine ordering inversion, in real data, that
the magnitude check cannot see.

That is the drill's headline, and it is an argument for the ordering criterion
being load-bearing rather than decorative.

(The direction is worth stating plainly because the brief predicted the
opposite: it expected flattening to compress the gaps. It does not. The
statistic is a MINIMUM over the corpus, so it lives in the near-zero tail rather
than among the top moves, and flattening lifts that tail toward 1/n and spreads
it out. See tests/test_c11b_temperature.py.)

THE MUTATIONS
=============
  swap-closest-t070   swap the closest prior pair in the T = 0.7 columns. The
                      ordering check MUST fail and the magnitude check MUST
                      PASS. See above: this is the whole point.
  swap-closest-t100   the same at T = 1.0, where the corpus does not cooperate.
                      Reported as an observation, matching C10's finding.
  break-the-identity  perturb one T = 1.0 prior by 1e-3. The identity self-check
                      MUST fail — that column is what certifies the other six,
                      so a golden generated from a different reference than
                      Gate 2's must not survive.
  swap-the-temperatures  exchange the T = 0.7 and T = 1.5 columns wholesale. The
                      magnitude check MUST fail, which is what proves the tests
                      read the column belonging to the temperature they gathered
                      at rather than any column that happens to be present.
  nudge-over-t150     one T = 1.5 prior moved by 2e-6, over the bound, in a
                      direction that reorders nothing. The magnitude check MUST
                      fail and the ordering check MUST pass — the mirror of
                      swap-closest, so neither half stands in for the other in
                      either direction.

`golden/` IS NEVER TOUCHED (Amendment B). Every mutation runs against a copy in
a scratch directory, reached through GUOFISH_GOLDEN_C11B_GATE2_TEMP and
GUOFISH_GOLDEN_C11B_GATE2_TEMP_MANIFEST, and the real files' SHA-256 is recorded
before and after so the report can state it rather than claim it. Copies and
writes are binary (Amendment E).

Usage:
    python tools/drill_c11b_temperature.py
    python tools/drill_c11b_temperature.py --only swap-closest-t070 --keep
"""

from __future__ import annotations

import argparse
import hashlib
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

GOLDEN = REPO_ROOT / "golden" / "c11b_gate2_temp.npz"
MANIFEST = REPO_ROOT / "golden" / "c11b_gate2_temp_manifest.json"

COLUMNS = ("priors_cpu_pychess", "priors_gpu_pychess", "priors_cpu_libchess")
TEMPERATURES = (0.7, 1.0, 1.5)

ORDER_TEST = "test_there_are_no_prior_ordering_inversions"
DELTA_TEST = "test_the_max_absolute_delta_is_within_the_gate"
IDENTITY_TEST = "test_the_identity_temperature_reproduces_gate_2s_own_golden"

MAX_ABS_DELTA = 1e-6


def suffix(temperature: float) -> str:
    return f"t{int(round(temperature * 100)):03d}"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as handle:
        return {k: handle[k] for k in handle.files}


def save(path: Path, arrays: dict) -> None:
    np.savez_compressed(path, **arrays)


def _smallest_gap_pair(arrays: dict, temperature: float):
    """(position, index a, index b, gap) for the closest non-equal prior pair.

    Searched over the column the gate's first parametrisation uses; the swap is
    then applied to all three, so no column can quietly still agree.
    """
    priors = arrays[f"priors_cpu_pychess_{suffix(temperature)}"]
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


def mutate_swap_closest(arrays: dict, temperature: float) -> str:
    position, a, b, gap = _smallest_gap_pair(arrays, temperature)
    begin = int(arrays["move_offset"][position])
    for column in COLUMNS:
        col = arrays[f"{column}_{suffix(temperature)}"]
        col[begin + a], col[begin + b] = col[begin + b], col[begin + a]
    visible = "VISIBLE to" if gap > MAX_ABS_DELTA else "INVISIBLE to"
    return (f"swapped the closest prior pair at T={temperature:g}: position "
            f"{position}, moves {a}/{b}, gap {gap:.3e} — {visible} the "
            f"{MAX_ABS_DELTA:.0e} magnitude bound")


def mutate_break_the_identity(arrays: dict) -> str:
    """Move one T = 1.0 prior far enough that it is no longer Gate 2's own.

    1e-3, not 1e-9: the identity check compares BIT PATTERNS against
    golden/c10_gate2.npz, so any change at all is caught. A large one is used
    anyway, because a drill that only proves a bit-comparison notices a bit
    is proving less than it looks.
    """
    key = f"priors_cpu_pychess_{suffix(1.0)}"
    arrays[key][0] = np.float32(float(arrays[key][0]) + 1e-3)
    return ("moved one T=1.0 prior by 1e-3, so the identity columns are no "
            "longer bit-identical to golden/c10_gate2.npz")


def mutate_swap_the_temperatures(arrays: dict) -> str:
    for column in COLUMNS:
        cold = f"{column}_{suffix(0.7)}"
        hot = f"{column}_{suffix(1.5)}"
        arrays[cold], arrays[hot] = arrays[hot].copy(), arrays[cold].copy()
    return ("exchanged the T=0.7 and T=1.5 reference columns, so every "
            "comparison is made against the wrong temperature's ATen answer")


def mutate_nudge_over(arrays: dict) -> str:
    """One prior over the bound, in a direction that reorders nothing.

    Moved AWAY from its neighbours — toward the position's maximum — so the
    sorted order of that position is unchanged and only the magnitude check can
    see it.
    """
    offsets = arrays["move_offset"]
    key = f"priors_cpu_pychess_{suffix(1.5)}"
    priors = arrays[key]
    # The largest prior of the first position: nothing sits above it, so raising
    # it further cannot cross anything.
    begin, end = int(offsets[0]), int(offsets[1])
    top = begin + int(np.argmax(priors[begin:end]))
    moved = 2e-6
    for column in COLUMNS:
        col = arrays[f"{column}_{suffix(1.5)}"]
        col[top] = np.float32(float(col[top]) + moved)
    return (f"raised position 0's LARGEST T=1.5 prior by {moved:.0e} — over the "
            f"{MAX_ABS_DELTA:.0e} bound, and above every sibling, so no pair is "
            f"reordered")


# name -> (mutate, must-fail selector, complementary selector, expectation)
# "pass" makes the complement a REQUIREMENT: the two halves of the gate are
# being shown to be independent. "observe" only reports it, for the mutation
# where the corpus itself decides the answer.
MUTATIONS = {
    "swap-closest-t070": (lambda a: mutate_swap_closest(a, 0.7),
                          f"{ORDER_TEST} and 0.7", f"{DELTA_TEST} and 0.7", "pass"),
    "swap-closest-t100": (lambda a: mutate_swap_closest(a, 1.0),
                          f"{ORDER_TEST} and 1.0", f"{DELTA_TEST} and 1.0", "observe"),
    "break-the-identity": (mutate_break_the_identity, IDENTITY_TEST, None, None),
    "swap-the-temperatures": (mutate_swap_the_temperatures, DELTA_TEST, None, None),
    "nudge-over-t150": (mutate_nudge_over,
                        f"{DELTA_TEST} and 1.5", f"{ORDER_TEST} and 1.5", "pass"),
}


def run_gate(scratch_npz: Path, scratch_manifest: Path, selector: str):
    env = dict(os.environ)
    env["GUOFISH_GOLDEN_C11B_GATE2_TEMP"] = str(scratch_npz)
    env["GUOFISH_GOLDEN_C11B_GATE2_TEMP_MANIFEST"] = str(scratch_manifest)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_c11b_temperature.py",
         "-k", selector, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma-separated mutation names")
    parser.add_argument("--keep", action="store_true",
                        help="leave the scratch directory in place")
    args = parser.parse_args()

    if not GOLDEN.exists():
        print(f"missing {GOLDEN}; run tools/gen_c11b_gate2_temp_golden.py first",
              file=sys.stderr)
        return 2

    wanted = [n.strip() for n in args.only.split(",") if n.strip()] or list(MUTATIONS)
    unknown = [n for n in wanted if n not in MUTATIONS]
    if unknown:
        print(f"unknown mutation(s): {unknown}. Known: {list(MUTATIONS)}",
              file=sys.stderr)
        return 2

    before = {"npz": sha256_of(GOLDEN), "manifest": sha256_of(MANIFEST)}
    print("golden/ digests BEFORE the drill")
    for name, digest in before.items():
        print(f"  {name:9s} {digest}")

    pristine = load(GOLDEN)
    print(f"\nsmallest non-zero inter-prior gap, per temperature "
          f"(the magnitude bound is {MAX_ABS_DELTA:.0e})")
    for temperature in TEMPERATURES:
        _, _, _, gap = _smallest_gap_pair(pristine, temperature)
        verdict = ("INVISIBLE to the magnitude check" if gap <= MAX_ABS_DELTA
                   else "visible to the magnitude check too")
        print(f"  T={temperature:<4g} {gap:.3e}  ({gap / MAX_ABS_DELTA:5.2f}x the "
              f"bound)  a swap of this pair is {verdict}")

    scratch = Path(tempfile.mkdtemp(prefix="guofish-c11b-drill-"))
    print(f"\nscratch: {scratch}\n")

    results = []
    try:
        for name in wanted:
            mutate, must_fail, complement, expectation = MUTATIONS[name]
            work_npz = scratch / f"{name}.npz"
            work_manifest = scratch / f"{name}.manifest.json"
            # Binary copy, Amendment E. The manifest travels unchanged so the
            # provenance test is exercised against the same metadata.
            shutil.copyfile(MANIFEST, work_manifest)

            arrays = load(GOLDEN)
            description = mutate(arrays)
            save(work_npz, arrays)

            code, output = run_gate(work_npz, work_manifest, must_fail)
            caught = code != 0
            print(f"{name:22s} {description}")
            print(f"  {'CAUGHT  ' if caught else 'MISSED  '} {must_fail!r} "
                  f"-> pytest exit {code}")
            if not caught:
                print("\n".join(output.splitlines()[-12:]))

            complement_ok = None
            if complement is not None:
                comp_code, comp_output = run_gate(work_npz, work_manifest, complement)
                complement_ok = comp_code == 0
                if expectation == "pass":
                    verdict = "PASSED (independent)" if complement_ok else \
                        "ALSO FAILED — the two criteria are not independent here"
                else:
                    verdict = "passed" if complement_ok else "also failed"
                print(f"  complement {complement!r}: {verdict}")
                if expectation == "pass" and not complement_ok:
                    print("\n".join(comp_output.splitlines()[-12:]))

            results.append({
                "mutation": name,
                "description": description,
                "must_fail": must_fail,
                "caught": caught,
                "complement": complement,
                "complement_expectation": expectation,
                "complement_passed": complement_ok,
            })
    finally:
        if args.keep:
            print(f"\nscratch kept: {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)

    after = {"npz": sha256_of(GOLDEN), "manifest": sha256_of(MANIFEST)}
    print("\ngolden/ digests AFTER the drill")
    unchanged = True
    for name, digest in after.items():
        same = digest == before[name]
        unchanged &= same
        print(f"  {name:9s} {digest}  {'UNCHANGED' if same else 'CHANGED — Amendment B VIOLATED'}")

    missed = [r for r in results if not r["caught"]]
    broken = [r for r in results
              if r["complement_expectation"] == "pass" and r["complement_passed"] is False]

    print()
    print(f"{len(results) - len(missed)}/{len(results)} mutations caught")
    for row in missed:
        print(f"  MISSED {row['mutation']}: {row['must_fail']} did not fail")
    for row in broken:
        print(f"  NOT INDEPENDENT {row['mutation']}: {row['complement']} was "
              f"required to pass and did not")
    if not unchanged:
        print("  AMENDMENT B VIOLATED: a file under golden/ changed during the drill")
    return 0 if (not missed and not broken and unchanged) else 1


if __name__ == "__main__":
    raise SystemExit(main())
