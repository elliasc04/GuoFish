"""Amendment B mutation drill for C8 — tree reuse and the ping-pong arenas.

A test that cannot fail is not a test. This breaks C8 in eight specific ways and
requires each break to be caught, loudly, by a named check.

TWO KINDS OF MUTATION, AND THE FIRST ONE IS THE POINT
=====================================================
The C5 and C6 drills corrupt GOLDEN DATA: they rewrite a copy of an .npz and ask
whether the suite notices. That is the right shape when the thing under test is
a comparison. C8's brief asks for something else:

    "In your scratch directory, manually corrupt a `children_offset` index in
     the alternate arena post-compaction. Verify that the structural diff catches
     the corrupted tree layout and fails loudly."

The subject is the C++ compaction, not a golden file, so corrupting a golden file
would drill the wrong thing entirely. **Part A therefore mutates the SOURCE.** It
copies `cpp/` and `CMakeLists.txt` into a scratch directory, applies a one-line
change to the fixup arithmetic, builds a separate module there, and runs the C8
comparison against it. `golden/` is not touched at all, and neither is the
repository's own module — the scratch build writes to `<scratch>/module` and the
driver runs with that directory ahead of everything on `sys.path`.

Part B is the classic form, on corrupted copies of the C8 golden files, because
the acceptance suite also has to be able to see a reference that has moved.

WHAT EACH MUTATION IS FOR
=========================
Part A (source):

  offset-off-by-one        A remapped `children_offset` one slot low, on the
                           second block the compaction allocates. THE BRIEF'S
                           MUTATION. The engine's own structural diff
                           (`verify_compaction`, scope §7) must throw
                           TreeCorruption naming the path.
  offset-no-verify         The SAME mutation with the engine's diff switched
                           off, so the question becomes whether the acceptance
                           suite catches it unaided. It must: the full-tree shape
                           digest is there for exactly this.
  terminal-not-copied      The compaction drops the terminal bit. Fails the
                           equivalence comparison, and would have silently
                           un-drawn every fifty-move node in the corpus.
  history-not-advanced     `apply_move` forgets to put the old root into the
                           repetition history. This is implementation-scope
                           item 2 — the path/history repartition — and the
                           rep_history check is what stands under it.
  expand-root-accumulates  `_expand_root`'s `visit_count = 1` written as `+= 1`.
                           Invisible before tree reuse existed and wrong the
                           moment a promoted root arrives with visits.

Part B (golden copies):

  golden-visits            one visit count moved by 1
  golden-value-1ulp        one value_sum moved by ONE ULP — the mutation a
                           tolerance-based comparison would pass
  golden-children          one children_count moved by 1 (a shape change the
                           visited records alone can still agree on)

Each drill must FAIL. A drill that leaves the check green is the finding: it
means the check is not looking at what it claims to.

Usage:
    python tools/drill_c8_reuse.py
    python tools/drill_c8_reuse.py --only offset-off-by-one
    python tools/drill_c8_reuse.py --part b --keep
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

C8_FILES = {
    "GUOFISH_GOLDEN_C8_TREES": GOLDEN / "c8_reuse_trees.npz",
    "GUOFISH_GOLDEN_C8_DUMP": GOLDEN / "c8_reuse_dump.npz",
    "GUOFISH_GOLDEN_C8_MANIFEST": GOLDEN / "c8_reuse_manifest.json",
}
SUITE = "tests/test_c8_reuse.py"

# The already-fetched dependency sources, so a scratch configure needs no network
# and re-clones nothing. Names are FetchContent's, upper-cased.
PREBUILT_DEPS = REPO_ROOT / "build" / "msvc-release" / "_deps"
DEP_DIRS = {
    "FETCHCONTENT_SOURCE_DIR_PYBIND11": PREBUILT_DEPS / "pybind11-src",
    "FETCHCONTENT_SOURCE_DIR_CHESS_LIBRARY": PREBUILT_DEPS / "chess_library-src",
    "FETCHCONTENT_SOURCE_DIR_FATHOM": PREBUILT_DEPS / "fathom-src",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Part A — source mutations
# ---------------------------------------------------------------------------

# (name, file, find, replace, game, verify_compaction, detectors that count)
#
# `find` must occur EXACTLY ONCE in the file. A mutation that matched twice, or
# not at all, would silently drill nothing, and a drill that silently does
# nothing reports a pass — the one outcome this file exists to prevent.
SOURCE_MUTATIONS = [
    {
        "name": "offset-off-by-one",
        "file": "cpp/search.hpp",
        "find": "            dst.set_children(d, offset, count);",
        "replace": (
            "            { static int drill_hits = 0; const bool bite = (++drill_hits == 2);\n"
            "              dst.set_children(d, bite ? offset - 1 : offset, count); }"),
        "game": "fifty-walk",
        "plies": 4,
        "verify": True,
        "expect": {"engine-diff"},
        "why": "the brief's mutation: a remapped children_offset one slot low",
    },
    {
        "name": "offset-no-verify",
        "file": "cpp/search.hpp",
        "find": "            dst.set_children(d, offset, count);",
        "replace": (
            "            { static int drill_hits = 0; const bool bite = (++drill_hits == 2);\n"
            "              dst.set_children(d, bite ? offset - 1 : offset, count); }"),
        "game": "fifty-walk",
        "plies": 4,
        "verify": False,
        "expect": {"shape-digest", "record-diff", "node-count"},
        "why": "the same corruption with the engine's own diff disabled",
    },
    {
        "name": "terminal-not-copied",
        "file": "cpp/search.hpp",
        "find": "        if (arena_.is_terminal(src)) {",
        # Never true — terminal_value is 0.0, 1.0 or -1.0 — but not constant-
        # folded, so the scratch build does not warn about unreachable code.
        "replace": "        if (arena_.terminal_value(src) > 2.0f && arena_.is_terminal(src)) {",
        "game": "fifty-walk",
        "plies": 4,
        "verify": True,
        "expect": {"engine-diff"},
        "why": "the compaction drops the terminal bit",
    },
    {
        "name": "history-not-advanced",
        "file": "cpp/search.hpp",
        "find": "        history_keys_.insert(history_keys_.begin(), root_rep_key_);",
        "replace": "        // drill: the old root never joins the history",
        "game": "fifty-walk",
        "plies": 6,
        "verify": True,
        "expect": {"rep-history", "record-diff", "shape-digest", "node-count"},
        "why": "implementation scope item 2: the path/history repartition",
    },
    {
        "name": "expand-root-accumulates",
        "file": "cpp/search.hpp",
        "find": "        arena_.set_visits(root_, 1);",
        "replace": "        arena_.add_visits(root_, 1);",
        "game": "fifty-walk",
        "plies": 6,
        "verify": True,
        "expect": {"record-diff"},
        "why": "_expand_root's `visit_count = 1` written as `+= 1`",
    },
]


DRIVER = '''\
"""Replay a few plies of one C8 game against the module on sys.path.

Written into the scratch directory by tools/drill_c8_reuse.py and run there, so
`import guofish_core` finds the MUTATED module rather than the repository's.

The comparison itself is imported from tests/test_c8_reuse.py rather than
reimplemented: the question the drill asks is whether THAT code catches the
mutation, and a second copy of it would answer a different question.
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1] if False else Path(sys.argv[1])
GAME_NAME = sys.argv[2]
PLIES = int(sys.argv[3])
VERIFY = sys.argv[4] == "1"

import numpy as np
import guofish_core

spec = importlib.util.spec_from_file_location(
    "c8_suite", REPO_ROOT / "tests" / "test_c8_reuse.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)

assert Path(guofish_core.__file__).parent != REPO_ROOT, (
    "the driver imported the repository's module, not the scratch one: "
    + guofish_core.__file__)

manifest = json.loads((REPO_ROOT / "golden" / "c8_reuse_manifest.json").read_text())
with np.load(REPO_ROOT / "golden" / "c8_reuse_trees.npz") as data:
    trees = {k: data[k] for k in data.files}
with np.load(REPO_ROOT / "golden" / "c8_reuse_dump.npz") as data:
    dump = {k: data[k] for k in data.files}

game = next(g for g in manifest["games"] if g["name"] == GAME_NAME)
detectors = []

try:
    config = guofish_core.SearchConfig(
        virtual_loss=game["virtual_loss"], max_tree_depth=game["max_tree_depth"],
        arena_capacity=1 << 19, cache_slots=0, verify_compaction=VERIFY)
    search = guofish_core.ReplaySearchDouble(config)
    search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                     dump["moves"], dump["priors"], dump["values"])
    search.set_position(game["root_fen"], game["history"])

    trail = list(game["history"])
    previous_fen = game["root_fen"]
    for offset, audit in enumerate(game["snapshots"][: 2 * PLIES]):
        index = game["snapshot_index"][offset]
        if audit["kind"] == "search":
            search.search(game["sims"])
        else:
            search.apply_move(audit["move"])
            trail.insert(0, previous_fen)
            previous_fen = audit["fen"]
            window = min(audit["halfmove_clock"], len(trail))
            expected = {guofish_core.rep_key(audit["fen"]): 1}
            for fen in trail[:window]:
                key = guofish_core.rep_key(fen)
                expected[key] = expected.get(key, 0) + 1
            if dict(search.rep_history()) != expected:
                detectors.append("rep-history")

        golden = suite._golden_snapshot(trees, index)
        if suite._first_difference(golden, search.dump_tree_arrays(1)) is not None:
            detectors.append("record-diff")
        if int(search.nodes) != audit["full_nodes"]:
            detectors.append("node-count")
        if suite._shape_digest(search.dump_tree_arrays(0)) != audit["full_shape_sha256"]:
            detectors.append("shape-digest")
        if detectors:
            break
except RuntimeError as exc:
    text = str(exc)
    if "compacted tree" in text:
        detectors.append("engine-diff")
    else:
        detectors.append("runtime-error")
    # The WHOLE message, not its first line. "Fails loudly" is the requirement,
    # and a drill that only proved something was raised would not have checked
    # whether the raise names the node.
    print(text, file=sys.stderr)
except Exception:
    traceback.print_exc()
    detectors.append("exception")

print(json.dumps({"detectors": sorted(set(detectors))}))
'''


def write_scratch_build_script(scratch: Path) -> Path:
    """A vcvars-first build script for the scratch tree.

    A copy of build/win.bat's mechanism rather than a call to it: win.bat hard-
    codes its own repository as the source directory, which is exactly the thing
    a source mutation must not build.
    """
    deps = " ".join(f'-D{name}="{path}"' for name, path in DEP_DIRS.items())
    script = scratch / "build_scratch.bat"
    script.write_text(
        "@echo off\r\n"
        'set "VS=C:\\Program Files\\Microsoft Visual Studio\\18\\Community"\r\n'
        'call "%VS%\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul\r\n'
        'set "PATH=%VS%\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin;'
        '%VS%\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\Ninja;%PATH%"\r\n'
        f'cmake -S "{scratch / "src"}" -B "{scratch / "build"}" -G Ninja '
        f'-DCMAKE_BUILD_TYPE=Release -DGUOFISH_VALUE_SUM=q32 '
        f'-DGUOFISH_MODULE_OUTPUT_DIR="{scratch / "module"}" {deps} || exit /b 1\r\n'
        f'cmake --build "{scratch / "build"}" || exit /b 1\r\n'
        "echo SCRATCH_BUILD_OK\r\n",
        encoding="utf-8")
    return script


def run_source_drill(spec: dict, scratch_root: Path, verbose: bool) -> tuple[bool, str]:
    scratch = scratch_root / spec["name"]
    src = scratch / "src"
    src.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT / "cpp", src / "cpp")
    shutil.copy2(REPO_ROOT / "CMakeLists.txt", src / "CMakeLists.txt")
    (scratch / "module").mkdir(parents=True, exist_ok=True)

    target = src / spec["file"]
    text = target.read_text(encoding="utf-8")
    occurrences = text.count(spec["find"])
    if occurrences != 1:
        return False, (f"the mutation site occurs {occurrences} times in "
                       f"{spec['file']}; it must occur exactly once or the drill "
                       f"is not drilling what it says")
    target.write_text(text.replace(spec["find"], spec["replace"]), encoding="utf-8")

    script = write_scratch_build_script(scratch)
    build = subprocess.run(["cmd", "/c", str(script)], capture_output=True, text=True)
    if "SCRATCH_BUILD_OK" not in build.stdout:
        tail = "\n".join((build.stdout + build.stderr).splitlines()[-25:])
        return False, f"the mutated source did not build:\n{tail}"

    driver = scratch / "drill_driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(scratch / "module")
    result = subprocess.run(
        [sys.executable, str(driver), str(REPO_ROOT), spec["game"],
         str(spec["plies"]), "1" if spec["verify"] else "0"],
        capture_output=True, text=True, cwd=str(scratch), env=env)
    if verbose or result.returncode != 0:
        print(result.stdout, result.stderr, sep="\n")
    if result.returncode != 0:
        return False, f"the driver crashed:\n{result.stderr[-2000:]}"

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    fired = set(payload["detectors"])
    if not fired:
        return False, ("NOTHING NOTICED. The mutated engine produced the "
                       "reference's tree, which means the check this drill "
                       "targets is not looking at what it claims to.")
    wanted = set(spec["expect"])
    if not (fired & wanted):
        return False, (f"caught by {sorted(fired)}, but this drill is for "
                       f"{sorted(wanted)} — something noticed, and not the thing "
                       f"that was supposed to")
    return True, f"caught by {sorted(fired)}"


# ---------------------------------------------------------------------------
# Part B — golden-copy mutations
# ---------------------------------------------------------------------------

def _first_apply_snapshot(manifest_path: Path) -> int:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    game = payload["games"][0]
    for offset, snap in enumerate(game["snapshots"]):
        if snap["kind"] == "apply":
            return game["snapshot_index"][offset]
    raise AssertionError("no apply snapshot in the manifest")


def mutate_visits(arrays, index) -> str:
    begin = int(arrays["run_offset"][index])
    arrays["visits"][begin + 1] += 1
    return f"visits[{begin + 1}] += 1 (second node of the first post-apply tree)"


def mutate_value_1ulp(arrays, index) -> str:
    begin = int(arrays["run_offset"][index])
    old = float(arrays["value_sum"][begin + 1])
    arrays["value_sum"][begin + 1] = np.nextafter(old, np.inf)
    return (f"value_sum[{begin + 1}] moved ONE ULP, {old!r} -> "
            f"{float(arrays['value_sum'][begin + 1])!r}")


def mutate_children(arrays, index) -> str:
    begin = int(arrays["run_offset"][index])
    arrays["children"][begin] += 1
    return f"children[{begin}] += 1 (the promoted root's child count)"


GOLDEN_MUTATIONS = [
    ("golden-visits", mutate_visits),
    ("golden-value-1ulp", mutate_value_1ulp),
    ("golden-children", mutate_children),
]


def run_golden_drill(name, mutate, scratch_root: Path, verbose: bool) -> tuple[bool, str]:
    scratch = scratch_root / name
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)

    index = _first_apply_snapshot(C8_FILES["GUOFISH_GOLDEN_C8_MANIFEST"])
    note = ""
    for var, path in C8_FILES.items():
        copy = scratch / path.name
        if var == "GUOFISH_GOLDEN_C8_TREES":
            with np.load(path) as data:
                arrays = {k: data[k].copy() for k in data.files}
            note = mutate(arrays, index)
            np.savez_compressed(copy, **arrays)
            # np.savez_compressed appends .npz if absent; it is present here.
        else:
            shutil.copy2(path, copy)
        env[var] = str(copy)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", SUITE, "-x", "-q",
         "-k", "every_snapshot"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env)
    if verbose:
        print(result.stdout[-4000:])
    if result.returncode == 0:
        return False, f"{note}: the suite PASSED against corrupted golden data"
    reported = [line for line in result.stdout.splitlines()
                if "differs" in line or "reference" in line]
    return True, f"{note}; suite failed" + (f" — {reported[0].strip()[:140]}" if reported else "")


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scratch", type=Path, default=None)
    parser.add_argument("--part", choices=("a", "b", "both"), default="both")
    parser.add_argument("--only", default="", help="comma-separated drill names")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    missing = [str(p) for p in C8_FILES.values() if not p.exists()]
    if missing:
        print("C8 golden data missing; run tools/gen_c8_reuse_golden.py first:",
              file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    # Amendment B: record the real files' digests before and after, so the run
    # itself is the evidence that golden/ was not written to.
    before = {path.name: sha256_of(path) for path in C8_FILES.values()}

    scratch = args.scratch or Path(tempfile.mkdtemp(prefix="guofish-c8-drill-"))
    scratch.mkdir(parents=True, exist_ok=True)
    wanted = {n.strip() for n in args.only.split(",") if n.strip()}

    drills = []
    if args.part in ("a", "both"):
        drills += [("A", spec["name"], spec["why"],
                    lambda s=spec: run_source_drill(s, scratch, args.verbose))
                   for spec in SOURCE_MUTATIONS]
    if args.part in ("b", "both"):
        drills += [("B", name, "the acceptance suite against a moved reference",
                    lambda n=name, m=mutate: run_golden_drill(n, m, scratch, args.verbose))
                   for name, mutate in GOLDEN_MUTATIONS]
    if wanted:
        drills = [d for d in drills if d[1] in wanted]
    if not drills:
        print(f"no drill matched {sorted(wanted)}", file=sys.stderr)
        return 2

    print(f"scratch: {scratch}\n")
    results = []
    for part, name, why, run in drills:
        print(f"[{part}] {name:<24} {why}", flush=True)
        ok, detail = run()
        results.append((part, name, ok, detail))
        print(f"    {'CAUGHT ' if ok else 'MISSED '} {detail}\n", flush=True)

    after = {path.name: sha256_of(path) for path in C8_FILES.values()}
    untouched = before == after

    print("=" * 78)
    for part, name, ok, detail in results:
        print(f"  [{part}] {name:<24} {'caught' if ok else 'MISSED'}")
    print()
    print("golden/ digests, before and after:")
    for name, digest in before.items():
        mark = "unchanged" if after[name] == digest else "*** CHANGED ***"
        print(f"  {digest}  {name}  ({mark})")
    if not untouched:
        print("\nAMENDMENT B VIOLATED: a golden file changed during this run.",
              file=sys.stderr)

    if not args.keep:
        shutil.rmtree(scratch, ignore_errors=True)
    else:
        print(f"\nscratch kept at {scratch}")

    missed = [name for _, name, ok, _ in results if not ok]
    if missed or not untouched:
        print(f"\nFAILED: {len(missed)} drill(s) not caught: {missed}", file=sys.stderr)
        return 1
    print(f"\nAll {len(results)} drills caught.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
