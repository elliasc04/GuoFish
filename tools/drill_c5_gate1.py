"""Amendment B mutation drill for the C5 Gate 1 suite.

A test that cannot fail is not a test. This corrupts the Gate 1 golden data and
checks that `tests/test_c5_gate1_quiet.py` notices — loudly, naming the divergent
node's path from the root rather than reporting a bare "trees differ".

**`golden/` IS NEVER WRITTEN TO.** Amendment B is explicit about this and it is
the standing resolution of the drill-versus-Rule-1/2 conflict: the corrupted
copies live in a scratch directory, the suite is pointed at them through the
`GUOFISH_GOLDEN_GATE1_*` environment overrides, and the real files' SHA-256
digests are recorded before and after so the run itself proves they were not
touched.

Usage:
    python tools/drill_c5_gate1.py
    python tools/drill_c5_gate1.py --scratch /path/to/scratch --keep

Each drill must FAIL the suite. A drill that leaves the suite green is the
finding: it means the gate is not comparing what it claims to.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = REPO_ROOT / "golden"
FILES = {
    "GUOFISH_GOLDEN_GATE1_DUMP": GOLDEN / "gate1_dump.npz",
    "GUOFISH_GOLDEN_GATE1_TREES": GOLDEN / "gate1_trees.npz",
    "GUOFISH_GOLDEN_GATE1_MANIFEST": GOLDEN / "gate1_manifest.json",
}
TEST = "tests/test_c5_gate1_quiet.py"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rewrite_npz(source: Path, target: Path, mutate) -> str:
    """Copy an npz, applying `mutate` to its arrays. Returns a description."""
    with np.load(source) as data:
        arrays = {name: data[name] for name in data.files}
    note = mutate(arrays)
    np.savez_compressed(target, **arrays)
    return note


def mutate_tree_value_sum(arrays) -> str:
    """One ulp on one node's value_sum, deep enough in the DFS to have a path.

    Node 1 is the root's first visited child in the first recorded run, so the
    failure report has to print a real move rather than "(root)".
    """
    index = 1
    before = arrays["value_sum"][index]
    arrays["value_sum"][index] = np.nextafter(before, before + 1.0)
    return (f"gate1_trees.npz: value_sum[{index}] {before!r} -> "
            f"{arrays['value_sum'][index]!r} (one ulp)")


def mutate_tree_visits(arrays) -> str:
    index = 1
    before = int(arrays["visits"][index])
    arrays["visits"][index] = before + 1
    return f"gate1_trees.npz: visits[{index}] {before} -> {before + 1}"


def mutate_dump_root_priors(arrays) -> str:
    """One ulp on the FIRST prior of every root entry.

    This drill perturbs the SEARCH rather than the comparison: the C++ tree
    really diverges, instead of a correct tree being compared against a doctored
    reference. That is the harder of the two things to detect and the one that
    matters, since it is the shape a real implementation bug takes.

    Why every root entry rather than one. The dump is sorted by (nn_key,
    is_root), so an entry's INDEX says nothing about which position it belongs
    to — and the dump also carries entries from the seven candidates the
    generator rejected, which no accepted run ever looks up. The first version
    of this drill mutated priors[0] and the suite stayed green for exactly that
    reason: it had corrupted a position nothing reads. There are only 26 root
    entries, so touching the first prior of each is still a 26-float change out
    of ~4 million, and it is guaranteed to include the position under test.
    """
    roots = np.flatnonzero(arrays["is_root"] != 0)
    for slot in roots:
        begin = int(arrays["move_offset"][slot])
        before = arrays["priors"][begin]
        arrays["priors"][begin] = np.nextafter(before, np.float32(1.0),
                                               dtype=np.float32)
    return (f"gate1_dump.npz: one ulp on the first prior of each of "
            f"{len(roots)} root entries")


def mutate_dump_interior_values(arrays) -> str:
    """1e-9 on every interior entry's value.

    A value reaches the comparison only as an accumulated `value_sum`, through
    backpropagate(), so this exercises the arithmetic path rather than the
    storage path the prior drill covers. 1e-9 rather than one ulp because a
    one-ulp seed difference is genuinely absorbed once the running sum passes
    ~1 — see DECISIONS.md.
    """
    interior = arrays["is_root"] == 0
    arrays["values"][interior] = arrays["values"][interior] + 1e-9
    return (f"gate1_dump.npz: +1e-9 on {int(interior.sum())} interior values")


DRILLS = (
    ("tree value_sum, one ulp", "GUOFISH_GOLDEN_GATE1_TREES", mutate_tree_value_sum),
    ("tree visit count, +1", "GUOFISH_GOLDEN_GATE1_TREES", mutate_tree_visits),
    ("dump root priors, one ulp", "GUOFISH_GOLDEN_GATE1_DUMP", mutate_dump_root_priors),
    ("dump interior values, 1e-9", "GUOFISH_GOLDEN_GATE1_DUMP",
     mutate_dump_interior_values),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scratch", type=Path, default=None)
    parser.add_argument("--keep", action="store_true",
                        help="leave the scratch directory in place afterwards")
    args = parser.parse_args()

    for path in FILES.values():
        if not path.exists():
            print(f"missing golden file: {path}", file=sys.stderr)
            return 2

    before = {path.name: sha256_of(path) for path in FILES.values()}
    print("golden SHA-256 BEFORE")
    for name, digest in sorted(before.items()):
        print(f"  {name:<22} {digest}")
    print()

    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="guofish-c5-drill-"))
    scratch.mkdir(parents=True, exist_ok=True)

    failures = []
    try:
        for label, env_var, mutate in DRILLS:
            print(f"--- drill: {label}")
            env = dict(os.environ)
            for var, path in FILES.items():
                copy = scratch / f"{var.lower()}{path.suffix}"
                if var == env_var and path.suffix == ".npz":
                    note = rewrite_npz(path, copy, mutate)
                    print(f"    {note}")
                else:
                    shutil.copyfile(path, copy)
                env[var] = str(copy)

            result = subprocess.run(
                [sys.executable, "-m", "pytest", TEST, "-q", "--no-header",
                 "-x", "--tb=long"],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                failures.append(label)
                print("    *** THE SUITE STILL PASSED. The gate is not comparing "
                      "what it claims to. ***")
            else:
                tail = [line for line in result.stdout.splitlines()
                        if "path from root" in line or "first divergence" in line
                        or "structural difference" in line]
                print("    suite FAILED as required")
                for line in tail[:4]:
                    print(f"      {line.strip()}")
                if not tail:
                    failures.append(f"{label} (failed, but printed no DFS path)")
                    print("    *** it failed without naming the divergent node's "
                          "path — a bare 'trees differ' is not acceptable ***")
            print()
    finally:
        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)
        else:
            print(f"scratch kept at {scratch}")

    after = {path.name: sha256_of(path) for path in FILES.values()}
    print("golden SHA-256 AFTER")
    for name, digest in sorted(after.items()):
        mark = "unchanged" if after[name] == before[name] else "*** CHANGED ***"
        print(f"  {name:<22} {digest}  {mark}")

    if after != before:
        print("\nGLOBAL RULE 1 VIOLATION: a golden file changed during the drill.",
              file=sys.stderr)
        return 1

    if failures:
        print(f"\n{len(failures)} drill(s) did not produce the required failure:",
              file=sys.stderr)
        for label in failures:
            print(f"  {label}", file=sys.stderr)
        return 1

    print("\nall drills produced a loud, path-naming failure; golden/ untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
