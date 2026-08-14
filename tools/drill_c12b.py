#!/usr/bin/env python
"""C12b mutation drill — can tests/test_c12b_gate2prime.py actually fail?

    python tools/drill_c12b.py
    python tools/drill_c12b.py --only no-inductor --keep

WHY THE DRILL EXISTS AND WHAT IT IS AIMED AT
============================================
Gate 2' is a differential between two forwards, and the failure mode a
differential has that a golden comparison does not is **self-comparison**: if
`compile=True` quietly produced the eager forward, every test in the file would
pass — capture fidelity would compare eager against eager, determinism would
agree trivially, and the move-agreement criterion would report 100% against a
baseline it had merely reproduced. The C12b brief calls that the worst outcome
available to this chunk. `no-inductor` below constructs exactly it.

AMENDMENT B, AND WHERE C12b's DATA ACTUALLY LIVES
=================================================
The rule is that drills never touch `golden/`. C12b writes nothing there — its
artifacts are in `baseline/`, because they are C++ and torch output rather than
Python-reference output (Global Rule 2) — so this drill watches **both**
directories and prints their digests before and after. `corrupt-baseline`
mutates a **copy** in a scratch directory and points the suite at it through the
`GUOFISH_C12B_FORWARD` override, exactly as Amendment B requires of `golden/`;
the copy is written in binary mode (Amendment E).

Code mutations are applied as a pytest plugin that monkeypatches at plugin-import
time, which is before collection and therefore before any fixture builds an
evaluator — the same mechanism `tools/drill_c10b_graphs.py` uses.

WHAT EACH MUTATION IS FOR
=========================
  no-inductor       make `compile=True` produce the UNFUSED forward while still
                    reporting `compiled=True`. The self-comparison trap, with the
                    constructor's own guard disabled so the drill measures the
                    SUITE rather than the guard.
  autotune-on       restore `triton.autotune_pointwise`, the setting that makes
                    Inductor benchmark Triton configs and cache a winner chosen
                    from timings — the thing that made 17 of 28 `.best_config`
                    files differ between two cold compiles.
  recompile-limit   put dynamo's recompile limit below the captured ladder, which
                    is the shipped-config bug this chunk found: the ninth shape
                    silently falls back to eager and is then captured that way.
  stale-replay      skip the graph launch and re-read the previous batch's
                    outputs. Right shapes, right addresses, answers one batch old.
  corrupt-baseline  flip bits in a COPY of the frozen baseline, so the anchor the
                    whole chain hangs from no longer describes the eager engine.

A mutation caught by *no* test is the finding. A mutation caught by *every* test
is a sign it was too broad to be diagnostic.

THE TWO CRITERION TESTS ARE DESELECTED FROM THE BASELINE RUN, AND THAT IS NOT
HIDING THEM
============================================================================
`test_the_inductor_engine_agrees_with_the_baseline_on_99_percent_of_moves` and
`test_every_disagreement_is_a_near_tie` FAIL at the time of writing — Gate 2' does
not pass, which is reported in BENCH.md and DECISIONS.md rather than worked
around. A drill refuses to run when its target does not pass, because a mutation
that "fails" a suite that was already failing proves nothing; so those two are
excluded from the baseline and from the mutation runs, and every other property
in the file is drilled normally. **When Gate 2' passes, delete `BASELINE_DESELECT`
and re-run** — the drill is not correct while it carries an exclusion list.
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
TARGET = "tests/test_c12b_gate2prime.py"

# THE FOUR TESTS THAT SHARE THE `differential` FIXTURE, deselected for two
# separate reasons that happen to name the same set.
#
# COST. The fixture is a 520-position sweep at 1,600 simulations — about 4.5
# minutes — and pytest builds it once per process, so leaving it in would put it
# in the baseline run AND in every mutation run: six sweeps, half an hour, to
# drill five properties that have nothing to do with search output. None of the
# mutations below is expected to be caught by a move-agreement test. (This is the
# same reasoning C12 applies to Gate 2b's own four differential tests, and by
# coincidence of design it is the same four names.)
#
# **THE EXCLUSION IS NOW PURELY A COST ONE, AND THAT IS THE REVISIT THIS COMMENT
# ASKED FOR.** It used to carry a second, correctness reason: two of these four
# were RED, and a drill refuses to run when its target does not pass because a
# mutation that "fails" an already-failing suite proves nothing. After the owner's
# ruling on Gate 2' (see `MIN_AGREEMENT` in the test file) all four pass, so only
# the cost argument survives — and it survives intact, since no mutation below is
# aimed at search output.
BASELINE_DESELECT = (
    "not test_every_search_delivered_the_budget "
    "and not test_move_agreement_against_the_baseline_meets_the_ruled_floor "
    "and not test_the_decisive_disagreements_are_listed_for_adjudication "
    "and not test_the_margin_distribution_explains_the_agreement_rate")

WATCHED = (
    ("golden", "c10_corpus.json"),
    ("golden", "gate1_manifest.json"),
    ("baseline", "c12b_eager_forward.npz"),
    ("baseline", "c12b_eager_search.json"),
    ("baseline", "c12b_eager_manifest.json"),
)

MUTATIONS: dict[str, tuple[str, str]] = {
    "no-inductor": (
        "test_the_prior_shift_is_reported",
        """
import torch

from playing.v6 import graphs

# The constructor's own guard would catch this before a single test ran, and then
# the drill would be measuring the guard instead of the suite. It is disabled so
# the question stays "does the SUITE notice that Inductor never ran?".
graphs.InductorGraphedForward.assert_every_shape_is_fused = lambda self: None


def _eager(self, tokens_int32):
    # Still an InductorGraphedForward, still reporting compiled=True, still
    # capturing and replaying — but the module it captures is the ORIGINAL,
    # unfused one. Gate 2' is now the frozen baseline compared against itself.
    with torch.no_grad():
        with torch.amp.autocast_mode.autocast(device_type="cuda",
                                              dtype=graphs.AUTOCAST_DTYPE, enabled=True):
            return self.eager_model(tokens_int32.to(torch.int64))


graphs.InductorGraphedForward._eager = _eager
""",
    ),
    "autotune-on": (
        "test_autotuning_is_off_because_that_is_what_pins_the_kernels",
        """
from playing.v6 import graphs

_original = graphs.configure_inductor


def configure_inductor():
    settings = _original()
    import torch._inductor.config as inductor_config

    # Back on: Inductor benchmarks several Triton configs per pointwise and
    # reduction kernel and caches the winner it timed fastest. A reduction
    # kernel's block size changes the accumulation order, so the priors stop
    # being a function of the code alone.
    inductor_config.triton.autotune_pointwise = True
    settings["triton.autotune_pointwise"] = True
    return settings


graphs.configure_inductor = configure_inductor
""",
    ),
    "recompile-limit": (
        "test_warmup_converged_before_anything_was_captured",
        """
from playing.v6 import graphs


def _raise_recompile_limit(self):
    import torch._dynamo.config as dynamo_config

    # Below the captured ladder rather than above it. This is the shipped-config
    # bug in miniature: dynamo stops compiling once the limit is hit and runs the
    # remaining shapes EAGER, silently, and the capture then records them that way.
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(dynamo_config, name):
            setattr(dynamo_config, name, 2)
    self.recompile_limit = 2


graphs.InductorGraphedForward._raise_recompile_limit = _raise_recompile_limit
""",
    ),
    "stale-replay": (
        "test_the_manual_capture_did_not_change_the_inductor_forward",
        """
from playing.v6 import graphs


def replay(self, padded):
    # The launch is skipped. Every buffer is the right shape at the right
    # address, and every answer is one batch old.
    self.replays += 1


graphs.InductorGraphedForward.replay = replay
""",
    ),
}

# Mutations that corrupt a FILE rather than the code. Amendment B: the original is
# never opened for writing; a copy is made in a scratch directory, corrupted
# there, and the suite is pointed at it through the documented env override.
FILE_MUTATIONS: dict[str, tuple[str, str, str]] = {
    "corrupt-baseline": (
        "test_compile_false_reproduces_the_frozen_baseline_bit_exactly",
        "GUOFISH_C12B_FORWARD",
        "baseline/c12b_eager_forward.npz",
    ),
}


def digests() -> dict[str, str]:
    out = {}
    for directory, name in WATCHED:
        path = REPO_ROOT / directory / name
        if path.exists():
            out[f"{directory}/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def corrupt_copy(source: Path, destination: Path) -> None:
    """Copy the baseline and change ONE policy word in the copy. Never touches the original.

    A BYTE FLIP AT A FIXED OFFSET WAS TRIED FIRST AND THE DRILL CAUGHT IT AS A
    MISS, which is the drill working. `np.savez_compressed` writes members in the
    order they were passed, so an `.npz` tail holds `fens` and `source` — the two
    arrays `test_compile_false_reproduces_the_frozen_baseline_bit_exactly` never
    reads. Flipping bits there corrupted the file and changed nothing the gate
    compares, so the mutation "survived" a suite that was in fact working
    correctly. A mutation has to land on the quantity under test to say anything
    about it.

    So the corruption is semantic and aimed: re-save every array unchanged except
    one bf16 word of `policy_1`, flipped to a different bit pattern. That is the
    smallest possible lie about what the frozen eager engine computed, and it is
    exactly the class of drift the gate exists to notice.

    Amendment B: the original is opened READ-ONLY and the corrupted copy is
    written into a scratch directory. Amendment E: `np.load`/`np.savez_compressed`
    are binary throughout; no text mode touches either file.
    """
    import numpy as np

    with np.load(source, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    policy = arrays["policy_1"].copy()
    # One word, in the first row, guaranteed to differ from whatever is there.
    policy[0, 0] = np.uint16(policy[0, 0] ^ np.uint16(0x0001))
    arrays["policy_1"] = policy
    np.savez_compressed(destination, **arrays)


def run_pytest(scratch: Path, plugin: str | None, env_extra: dict | None,
               only: str | None) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(scratch) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(env_extra or {})
    command = [sys.executable, "-m", "pytest", TARGET, "-q", "--no-header"]
    if plugin:
        command += ["-p", plugin]
    # No `-x`: the question is WHICH tests catch a mutation, and stopping at the
    # first failure answers a different one.
    command += ["-k", only if only else BASELINE_DESELECT]
    result = subprocess.run(command, cwd=REPO_ROOT, env=env,
                            capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def failing_tests(output: str) -> list[str]:
    names = []
    for line in output.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            names.append(line.split(" ", 1)[1].split(" ")[0])
    return names


def main() -> int:
    everything = list(MUTATIONS) + list(FILE_MUTATIONS)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", default=everything, choices=everything)
    parser.add_argument("--keep", action="store_true", help="keep the scratch directory")
    args = parser.parse_args()

    before = digests()
    print(f"watched digests before: {len(before)} files")
    for name, digest in before.items():
        print(f"  {name:38s} {digest}")

    scratch = Path(tempfile.mkdtemp(prefix="guofish-c12b-drill-"))
    print(f"\nscratch: {scratch}")

    print("\nbaseline (no mutation, criterion tests deselected — see the docstring)")
    code, output = run_pytest(scratch, None, None, None)
    last = output.strip().splitlines()[-1] if output.strip() else "(no output)"
    print(f"  exit={code}  {last}")
    if code != 0:
        print("REFUSING TO DRILL: the unmutated target does not pass, so a mutation "
              "that 'fails' would prove nothing.", file=sys.stderr)
        print("\n".join("    " + line for line in output.splitlines()[-25:]))
        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)
        return 2

    caught = 0
    print()
    for name in args.only:
        if name in MUTATIONS:
            expected, source = MUTATIONS[name]
            plugin = f"gfmut_{name.replace('-', '_')}"
            (scratch / f"{plugin}.py").write_text(source, encoding="utf-8", newline="\n")
            code, output = run_pytest(scratch, plugin, None, None)
        else:
            expected, variable, relative = FILE_MUTATIONS[name]
            source_path = REPO_ROOT / relative
            copy = scratch / Path(relative).name
            corrupt_copy(source_path, copy)
            code, output = run_pytest(scratch, None, {variable: str(copy)}, None)

        failures = failing_tests(output)
        crashed = code not in (0, 1)
        hit = any(expected in f for f in failures)
        status = "CAUGHT" if (hit or crashed) else "MISSED"

        # A CRASH HAS TO BE ATTRIBUTED, or "caught" means "the process died
        # somewhere near here". Re-run the expected test alone.
        attribution = ""
        if crashed:
            if name in MUTATIONS:
                solo, _ = run_pytest(scratch, f"gfmut_{name.replace('-', '_')}",
                                     None, expected)
            else:
                expected_, variable, relative = FILE_MUTATIONS[name]
                solo, _ = run_pytest(scratch, None,
                                     {variable: str(scratch / Path(relative).name)},
                                     expected_)
            hit = solo != 0
            attribution = (f", and by {expected} alone (exit={solo})" if hit else
                           f", but {expected} alone PASSED (exit={solo}) — not attributable")
            status = "CAUGHT" if hit else "MISSED"

        detail = (f"crash, exit={code}{attribution}" if crashed
                  else ", ".join(f.split("::")[-1] for f in failures) or "no failure")
        print(f"  {name:<18} {status:<7} expected {expected}")
        print(f"  {'':<18} exit={code}  {detail}")
        if status == "MISSED":
            print(f"  {'':<18} FINDING: this mutation survived the suite.")
            print("\n".join("      " + line for line in output.splitlines()[-15:]))
        caught += bool(hit)

    after = digests()
    print("\nwatched digests after:")
    for name, digest in after.items():
        mark = "unchanged" if before.get(name) == digest else "CHANGED"
        print(f"  {name:38s} {digest}  {mark}")
    unchanged = before == after
    print(f"\n{caught}/{len(args.only)} mutations caught; golden/ and baseline/ "
          f"{'unchanged' if unchanged else 'MODIFIED — this is a drill defect'}")

    if not args.keep:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0 if (caught == len(args.only) and unchanged) else 1


if __name__ == "__main__":
    raise SystemExit(main())
