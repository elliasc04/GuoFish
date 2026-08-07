"""Amendment B mutation drill for the C6 terminal suite.

A test that cannot fail is not a test. This corrupts the C6 terminal golden data
and checks that `tests/test_c6_gate1_full.py` and
`tests/test_c6_terminal_invariants.py` notice — loudly, naming the divergent
node's path from the root rather than reporting a bare "trees differ".

**`golden/` IS NEVER WRITTEN TO.** Amendment B is explicit about this and it is
the standing resolution of the drill-versus-Rule-1/2 conflict: the corrupted
copies live in a scratch directory, the suites are pointed at them through the
`GUOFISH_GOLDEN_C6_*` environment overrides, and the real files' SHA-256 digests
are recorded before and after so the run itself proves they were not touched.

WHAT THIS DRILLS THAT THE C5 DRILL DID NOT
------------------------------------------
C5's four drills perturb priors, values, visit counts and value sums — the
numeric surface. C6 adds three fields whose whole job is to be discrete, and a
numeric drill says nothing about any of them:

    terminal            the bit. Clearing it on a node the reference marked, or
                        setting it on one it did not, must fail.
    terminal_value      the cached game result. 1.0 -> 0.0 turns a checkmate into
                        a draw and changes every backup above it.
    max_tree_depth      the per-run cap. Off by one and the depth-cap frontier
                        moves, which is the specific thing the depth-cap specs
                        exist to pin.

The brief's own instruction is "alter a terminal value or visit count on a node
that should hit the depth cap, and verify the test fails loudly with the DFS path
and exact node diff". `depthcap terminal bit` and `depthcap visits` below are
that drill literally; the rest generalise it.

Usage:
    python tools/drill_c6_gate1.py
    python tools/drill_c6_gate1.py --scratch /path/to/scratch --keep

Each drill must FAIL its suite. A drill that leaves the suite green is the
finding: it means the gate is not comparing what it claims to.
"""

import argparse
import hashlib
import json
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
    "GUOFISH_GOLDEN_C6_DUMP": GOLDEN / "gate1_terminal_dump.npz",
    "GUOFISH_GOLDEN_C6_TREES": GOLDEN / "gate1_terminal_trees.npz",
    "GUOFISH_GOLDEN_C6_MANIFEST": GOLDEN / "gate1_terminal_manifest.json",
}
GATE = "tests/test_c6_gate1_full.py"
INVARIANTS = "tests/test_c6_terminal_invariants.py"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rewrite_npz(source: Path, target: Path, mutate) -> str:
    with np.load(source) as data:
        arrays = {name: data[name] for name in data.files}
    note = mutate(arrays)
    np.savez_compressed(target, **arrays)
    return note


def rewrite_json(source: Path, target: Path, mutate) -> str:
    with open(source, encoding="utf-8") as handle:
        payload = json.load(handle)
    note = mutate(payload)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return note


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------


def _first_terminal(arrays) -> int:
    hits = np.flatnonzero(arrays["terminal"] != 0)
    if hits.size == 0:
        raise SystemExit("the terminal golden trees contain no terminal node; "
                         "there is nothing for this drill to corrupt")
    return int(hits[0])


def mutate_clear_terminal_bit(arrays) -> str:
    """Un-mark a node the reference marked terminal.

    The most surgical C6-specific change available: one byte, from 1 to 0, on a
    node whose visit count, value_sum and priors are all left alone. Nothing in
    C5's comparison would see it.
    """
    index = _first_terminal(arrays)
    arrays["terminal"][index] = 0
    return (f"gate1_terminal_trees.npz: terminal[{index}] 1 -> 0 "
            f"(terminal_value stays {arrays['terminal_value'][index]!r})")


def mutate_set_terminal_bit(arrays) -> str:
    """The other direction: mark a node the reference did not.

    Node 1 is the root's first visited child in the first recorded run, so the
    failure has a real move to name. Chosen only if it is not already terminal.
    """
    index = next(i for i in range(1, len(arrays["terminal"]))
                 if arrays["terminal"][i] == 0)
    arrays["terminal"][index] = 1
    return f"gate1_terminal_trees.npz: terminal[{index}] 0 -> 1"


def mutate_terminal_value(arrays) -> str:
    """Turn a checkmate into a draw, or a draw into a checkmate.

    `terminal_value` is what the fast path backs up on every later visit, so this
    is the field a wrong perspective conversion would land in — the risk the brief
    calls out by name.
    """
    index = _first_terminal(arrays)
    before = float(arrays["terminal_value"][index])
    arrays["terminal_value"][index] = np.float32(0.0 if before == 1.0 else 1.0)
    return (f"gate1_terminal_trees.npz: terminal_value[{index}] {before!r} -> "
            f"{float(arrays['terminal_value'][index])!r}")


def mutate_depthcap_visits(arrays) -> str:
    """THE BRIEF'S OWN DRILL: a visit count on a node at the depth-cap frontier.

    The frontier is where the cap fired, so this is a node the reference reached
    and backed 0.0 up from without expanding. Its visit count is pure
    cap-accounting; if the comparison did not cover it, a C++ that capped one ply
    early or late would go unnoticed.
    """
    # The deepest visited node in the file is on some run's frontier. Taking the
    # deepest rather than a named run keeps this working if the corpus is
    # regenerated with different specs.
    index = int(np.argmax(arrays["depth"]))
    before = int(arrays["visits"][index])
    arrays["visits"][index] = before + 1
    return (f"gate1_terminal_trees.npz: visits[{index}] {before} -> {before + 1} "
            f"at depth {int(arrays['depth'][index])} (the deepest node recorded, "
            f"i.e. on a cap or terminal frontier)")


def mutate_manifest_all_depth_caps(payload) -> str:
    """Move every depth-cap run's cap by one.

    Not a corruption of the tree at all — a corruption of the CONFIGURATION the
    C++ side is told to run at. The tree stays the reference's; C++ is asked to
    reproduce it while capping one ply deeper, which it cannot. This is the drill
    that proves the per-run `max_tree_depth` is actually read rather than
    defaulted to 80.
    """
    note = None
    for run in payload["runs"]:
        if "depthcap" in run.get("categories", []):
            if note is None:
                note = (f"gate1_terminal_manifest.json: every depthcap run's "
                        f"max_tree_depth +1 (first was {run['max_tree_depth']})")
            run["max_tree_depth"] += 1
    if note is None:
        raise SystemExit("no depthcap run in the manifest to corrupt")
    return note


def mutate_manifest_history(payload) -> str:
    """Drop the pre-root repetition history from the threefold specs.

    The history is the only input that makes a threefold reachable inside search
    range. Without it the C++ search finds fewer draws, its tree diverges, and
    the invariants suite's history test loses its premise. A `set_position` that
    ignored the argument would pass every other test in the repository, because
    every C5 position has an empty history.
    """
    touched = 0
    for position in payload["positions"]:
        if position["history"]:
            position["history"] = []
            touched += 1
    if touched == 0:
        raise SystemExit("no position in the manifest carries a history")
    return (f"gate1_terminal_manifest.json: cleared the recorded history of "
            f"{touched} position(s)")


def mutate_dump_root_priors(arrays) -> str:
    """One ulp on the first prior of every root entry.

    Carried over from the C5 drill because it perturbs the SEARCH rather than the
    comparison — the C++ tree really diverges, which is the shape a real
    implementation bug takes. Every root entry rather than one, because the dump
    is sorted by (nn_key, is_root) and an entry's index says nothing about which
    position it belongs to.
    """
    roots = np.flatnonzero(arrays["is_root"] != 0)
    for slot in roots:
        begin = int(arrays["move_offset"][slot])
        before = arrays["priors"][begin]
        arrays["priors"][begin] = np.nextafter(before, np.float32(1.0),
                                               dtype=np.float32)
    return (f"gate1_terminal_dump.npz: one ulp on the first prior of each of "
            f"{len(roots)} root entries")


DRILLS = (
    ("terminal bit cleared", "GUOFISH_GOLDEN_C6_TREES", mutate_clear_terminal_bit, GATE),
    ("terminal bit set", "GUOFISH_GOLDEN_C6_TREES", mutate_set_terminal_bit, GATE),
    ("terminal value flipped", "GUOFISH_GOLDEN_C6_TREES", mutate_terminal_value, GATE),
    ("depth-cap frontier visits, +1", "GUOFISH_GOLDEN_C6_TREES",
     mutate_depthcap_visits, GATE),
    ("depth cap moved by one", "GUOFISH_GOLDEN_C6_MANIFEST",
     mutate_manifest_all_depth_caps, GATE),
    ("repetition history dropped", "GUOFISH_GOLDEN_C6_MANIFEST",
     mutate_manifest_history, INVARIANTS),
    ("dump root priors, one ulp", "GUOFISH_GOLDEN_C6_DUMP",
     mutate_dump_root_priors, GATE),
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
            print(f"missing golden file: {path}\n"
                  f"Generate it with "
                  f"`python tools/gen_gate1_golden.py --corpus terminal`.",
                  file=sys.stderr)
            return 2

    before = {path.name: sha256_of(path) for path in FILES.values()}
    print("golden SHA-256 BEFORE")
    for name, digest in sorted(before.items()):
        print(f"  {name:<32} {digest}")
    print()

    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="guofish-c6-drill-"))
    scratch.mkdir(parents=True, exist_ok=True)

    failures = []
    try:
        for label, env_var, mutate, test in DRILLS:
            print(f"--- drill: {label}  ({test})")
            env = dict(os.environ)
            for var, path in FILES.items():
                copy = scratch / f"{var.lower()}{path.suffix}"
                if var != env_var:
                    shutil.copyfile(path, copy)
                elif path.suffix == ".npz":
                    print(f"    {rewrite_npz(path, copy, mutate)}")
                else:
                    print(f"    {rewrite_json(path, copy, mutate)}")
                env[var] = str(copy)

            result = subprocess.run(
                [sys.executable, "-m", "pytest", test, "-q", "--no-header",
                 "-x", "--tb=long"],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                failures.append(label)
                print("    *** THE SUITE STILL PASSED. The gate is not comparing "
                      "what it claims to. ***")
            else:
                # Two shapes of path-naming failure count, and both are required
                # to name the node:
                #
                #   _first_divergence   "path from root : <moves>" — the tree
                #                       comparison found a differing node.
                #   ReplayMiss          "path   : <moves>" — the search walked
                #                       into a position the reference never
                #                       evaluated, which is what a corrupted
                #                       depth cap produces before any comparison
                #                       runs. It is the LOUDER of the two and it
                #                       carries the FEN and the key as well.
                tail = [line for line in result.stdout.splitlines()
                        if "path from root" in line or "first divergence" in line
                        or "structural difference" in line
                        or "DIFFERS" in line
                        or "replay dump miss" in line or "path   :" in line]
                print("    suite FAILED as required")
                for line in tail[:5]:
                    print(f"      {line.strip()}")
                if not tail and test == GATE:
                    failures.append(f"{label} (failed, but printed no DFS path)")
                    print("    *** it failed without naming the divergent node's "
                          "path — a bare 'trees differ' is not acceptable ***")
                elif not tail:
                    # The invariants suite is structural rather than a tree
                    # comparison, so a DFS path is not the right evidence there.
                    for line in result.stdout.splitlines():
                        if line.startswith("E ") and line.strip() != "E":
                            print(f"      {line.strip()}")
                            break
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
        print(f"  {name:<32} {digest}  {mark}")

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

    print("\nall drills produced a loud failure; golden/ untouched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
