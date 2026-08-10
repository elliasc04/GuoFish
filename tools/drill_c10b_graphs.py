#!/usr/bin/env python
"""C10b mutation drill — can tests/test_c10b_graphs.py actually fail?

    python tools/drill_c10b_graphs.py
    python tools/drill_c10b_graphs.py --only pad-leak --keep

WHY THIS DRILL HAS A DIFFERENT SHAPE FROM THE C5/C6/C10 ONES
============================================================
Those gates compare against recorded golden data, so corrupting a COPY of the
golden file is the natural mutation and Amendment B's rule ("drills never touch
`golden/`") is satisfied by pointing the suite at a scratch copy.

C10b's properties are not golden comparisons. "The padded rows' outputs never
reach an expansion", "the callback allocates nothing", "capture did not change
the forward" are claims about CODE, so the mutations are applied to the code —
in a scratch pytest plugin that monkeypatches the evaluator before any fixture
builds one. `golden/` is never opened for writing by anything here, and the
digests are recorded before and after anyway, because a drill that asserts its
own good behaviour is worth more than one that assumes it (Amendment B, and the
C8 drill's precedent of mutating a copy of the source).

WHAT EACH MUTATION IS FOR
=========================
Each one breaks exactly one property and names the test that must catch it. A
mutation caught by *no* test is the finding; a mutation caught by *every* test
is a sign the mutation was too broad to be diagnostic.

  pad-leak          hand leaf i the answer to leaf i+1, so the last leaf of every
                    batch gets a PADDING row's priors. This is the failure the
                    C10b brief names — "a test must fail if a padded row's prior
                    leaks into an expansion" — and it is constructed rather than
                    imagined.
  no-pad-refill     stop restoring the pad tail, so a small batch runs on the
                    tokens a larger previous batch left behind.
  round-down        let `pad_to` round DOWN, so a 21-row batch is evaluated at
                    shape 16 and three leaves get no answer at all.
  stale-replay      skip the graph launch and re-read the previous batch's
                    outputs. The shape is right, the buffers are right, and the
                    numbers are one batch old.
  allocating-call   allocate one fresh device tensor per batch — the discipline
                    graph capture requires and the one a careless edit breaks.

A CRASH COUNTS AS CAUGHT, and one mutation reliably produces one: a padded row's
priors are a real position's priors, but `pad-leak` combined with the poison the
leak test writes puts bf16 NaN into a softmax, and the Release build has no
guard for a non-finite prior. That is recorded in DECISIONS.md as an observation
about the search rather than fixed here — the network cannot emit NaN, and C10b
is not the chunk that adds a guard to the hot path.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = "tests/test_c10b_graphs.py"

GOLDEN_WATCHED = ("c10_gate2.npz", "c10_gate2_manifest.json", "c10_corpus.json")

# Each mutation is the body of a pytest plugin. It runs at plugin-import time,
# which is before collection and therefore before any fixture builds an
# evaluator.
MUTATIONS: dict[str, tuple[str, str]] = {
    "pad-leak": (
        "test_a_padded_rows_prior_cannot_reach_an_expansion",
        """
from playing.v6 import graphs

_original = graphs.GraphedForward.outputs


def outputs(self, count, padded):
    policy, value = self._policy[padded], self._value[padded]
    if padded == count:
        return policy[:count], value[:count]
    # Off by one, but ONLY where there is padding to leak: leaf i is handed leaf
    # i+1's row and the last leaf of the batch is handed a PADDING row. Exactly
    # the leak the brief names, and narrow enough to be diagnostic — an
    # exactly-captured shape still behaves, so the capture-fidelity and Gate 2
    # tests (which run at exact shapes) stay green and the leak test is on its
    # own.
    return policy[1:count + 1], value[1:count + 1]


graphs.GraphedForward.outputs = outputs
""",
    ),
    "no-pad-refill": (
        "test_the_padding_rows_hold_the_pad_position_and_not_the_previous_batch",
        """
from playing.v6 import graphs


def stage(self, count, source):
    padded = self.pad_to(count)
    self.tokens[:count].copy_(source, non_blocking=True)
    # The refill is gone: the tail keeps whatever a larger previous batch wrote.
    self._dirty = count
    return padded


graphs.GraphedForward.stage = stage
""",
    ),
    "round-down": (
        "test_the_captured_shapes_cover_every_batch_the_dispatcher_can_hand_over",
        """
from playing.v6 import graphs


def pad_to(self, count):
    if count < 1 or count > self.max_batch:
        raise ValueError(count)
    below = [s for s in self.sizes if s <= count]
    return below[-1] if below else self.sizes[0]


graphs.GraphedForward.pad_to = pad_to
""",
    ),
    "stale-replay": (
        "test_the_graphed_forward_is_bit_identical_to_the_eager_one",
        """
from playing.v6 import graphs


def replay(self, padded):
    # The launch is skipped. Every buffer is the right shape and the right
    # address, and every answer is one batch old.
    self.replays += 1


graphs.GraphedForward.replay = replay
""",
    ),
    "allocating-call": (
        "test_the_callback_allocates_no_device_memory",
        """
import torch

from playing.v6 import graphs

_original = graphs.GraphedForward.run


def run(self, count, source, poison=False):
    policy, value = _original(self, count, source, poison)
    # One fresh device tensor per batch, kept alive on the instance so the
    # allocator cannot hand the block straight back.
    self._scratch = getattr(self, "_scratch", [])
    self._scratch.append(torch.empty((count, 64), device=self.device))
    return policy, value


graphs.GraphedForward.run = run
""",
    ),
}


def digests() -> dict[str, str]:
    out = {}
    for name in GOLDEN_WATCHED:
        path = REPO_ROOT / "golden" / name
        if path.exists():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def run_mutation(name: str, source: str, scratch: Path,
                 only: str | None = None) -> tuple[int, str]:
    plugin = f"gfmut_{name.replace('-', '_')}"
    (scratch / f"{plugin}.py").write_text(source, encoding="utf-8", newline="\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(scratch) + os.pathsep + env.get("PYTHONPATH", "")
    # No `-x`: the question is WHICH tests catch a mutation, and stopping at the
    # first failure answers a different one. A mutation caught only by a test
    # that was not aimed at it is a finding about the suite's coverage.
    command = [sys.executable, "-m", "pytest", TARGET, "-p", plugin, "-q", "--no-header"]
    if only is not None:
        command += ["-k", only]
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def failing_tests(output: str) -> list[str]:
    names = []
    for line in output.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            names.append(line.split(" ", 1)[1].split(" ")[0])
        elif "::" in line and line.strip().endswith("Failed"):
            names.append(line.strip())
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", default=list(MUTATIONS), choices=list(MUTATIONS))
    parser.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = parser.parse_args()

    before = digests()
    print(f"golden/ digests before: {len(before)} files")
    for name, digest in before.items():
        print(f"  {name:28s} {digest}")

    scratch = Path(tempfile.mkdtemp(prefix="guofish-c10b-drill-"))
    print(f"\nscratch: {scratch}")

    print("\nbaseline (no mutation)")
    baseline = subprocess.run(
        [sys.executable, "-m", "pytest", TARGET, "-q", "--no-header"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    print(f"  exit={baseline.returncode}  {baseline.stdout.strip().splitlines()[-1]}")
    if baseline.returncode != 0:
        print("REFUSING TO DRILL: the unmutated suite does not pass, so a mutation that "
              "'fails' would prove nothing.", file=sys.stderr)
        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)
        return 2

    caught = 0
    print()
    for name in args.only:
        expected, source = MUTATIONS[name]
        code, output = run_mutation(name, source, scratch)
        failures = failing_tests(output)
        crashed = code not in (0, 1)
        hit = any(expected in f for f in failures)
        status = ("CAUGHT" if (hit or crashed) else "MISSED")
        # A CRASH HAS TO BE ATTRIBUTED, or "caught" means "the process died
        # somewhere". Re-run the expected test alone: if it still dies, the
        # mutation is caught BY THAT TEST rather than merely near it.
        attribution = ""
        if crashed:
            solo, _ = run_mutation(name, source, scratch, only=expected)
            attribution = (f", and by {expected} alone (exit={solo})"
                           if solo not in (0,) else
                           f", but {expected} alone PASSED (exit={solo}) — not attributable")
            hit = solo != 0
            status = "CAUGHT" if hit else "MISSED"
        detail = (f"crash, exit={code}{attribution}" if crashed
                  else ", ".join(f.split("::")[-1] for f in failures) or "no failure")
        print(f"  {name:<17} {status:<7} expected {expected}")
        print(f"  {'':<17} exit={code}  {detail}")
        if not hit and not crashed:
            print(f"  {'':<17} FINDING: this mutation survived the suite.")
            print("\n".join("      " + line for line in output.splitlines()[-15:]))
        caught += bool(hit)

    after = digests()
    print(f"\ngolden/ digests after:")
    for name, digest in after.items():
        mark = "unchanged" if before.get(name) == digest else "CHANGED"
        print(f"  {name:28s} {digest}  {mark}")
    unchanged = before == after
    print(f"\n{caught}/{len(args.only)} mutations caught; "
          f"golden/ {'unchanged' if unchanged else 'MODIFIED — this is a drill defect'}")

    if not args.keep:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0 if (caught == len(args.only) and unchanged) else 1


if __name__ == "__main__":
    raise SystemExit(main())
