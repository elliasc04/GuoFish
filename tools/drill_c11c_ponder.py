#!/usr/bin/env python
"""Amendment B mutation drill for C11c — do the ponder tests actually bite?

    python tools/drill_c11c_ponder.py
    python tools/drill_c11c_ponder.py --only deadline-never-fires

WHAT THIS DRILL IS FOR, AND WHY IT IS NOT A GOLDEN-DATA DRILL
=============================================================
C11c ADDS NO GOLDEN DATA. Every previous drill in `tools/` corrupts a copy of a
`golden/` file and asserts that the tests notice. There is nothing here to
corrupt: the mutable deadline and the arena backstop are MECHANISMS, and their
acceptance is a set of behavioural assertions rather than a comparison against a
recorded reference.

So the mutation target is the ENGINE, not the data — each drill below disables
one mechanism at the Python boundary and asserts that a named test goes red.
That is the same question Amendment B exists to answer ("a test that cannot fail
is not a test"), asked of the only artifact this chunk produces.

Amendment B's letter is still honoured and is worth stating explicitly:

  * **`golden/` IS NEVER TOUCHED.** Nothing here opens a file under `golden/`
    for writing, and the SHA-256 of every file in it is recorded before and
    after the run and compared. A drill that silently altered reference data
    would be worse than no drill.
  * **Nothing is patched on disk.** The mutations are monkeypatches inside this
    process, reverted in a `finally`. `git status` is unchanged by a run.

THE MUTATIONS
=============
  deadline-never-fires   `set_deadline_in` becomes a no-op, so the C++ deadline
                         is never armed. The cross-thread conversion test MUST
                         fail — a `ponderhit` would then run to the node ceiling
                         instead of to the clock, which is the whole of
                         requirement 1.

  exhaustion-silent      `parallel_stats()['arena_exhausted']` is forced False,
                         so a degraded search reports as a sound one. The
                         harness-rejection test MUST fail. This is the exact
                         defect requirement 3 names: "a silent degradation that
                         only shows up as unexplained weakness is worse than a
                         crash".

  merge-the-sim-counts   `SearchOutcome.search_sims` returns `total_sims`, i.e.
                         the ponder's work folded into the post-hit figure. The
                         separation test MUST fail. This is scope E6's
                         over-commit defect wearing a reporting hat: the engine
                         looks 2-3x faster on a hit than it is.

  arena-formula-halved   `arena_nodes` returns half its value, as it would if
                         the observed ~32 nodes/sim had been fitted instead of
                         the conservative 60. The formula test MUST fail.

  coupling-unchecked     `coupling_holds` returns True unconditionally. The
                         pinned-cap test MUST fail — which is what makes the
                         startup line's VIOLATED warning load-bearing rather
                         than decorative.

Each mutation names the test it must break. A mutation that breaks NOTHING is
the finding — it means the mechanism is untested — and is reported as a FAIL of
the drill rather than as a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

import guofish_core  # noqa: E402
from playing.v6 import playv6  # noqa: E402

GOLDEN = REPO_ROOT / "golden"
TESTS = REPO_ROOT / "tests" / "test_c11c_ponder.py"


def golden_digests() -> dict:
    """SHA-256 of every file under `golden/`. Amendment B's before/after record."""
    out = {}
    for path in sorted(GOLDEN.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(GOLDEN))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    return out


def run_test(node_id: str) -> tuple[bool, str]:
    """Run one test by node id in this process. (passed, captured output)."""
    buffer = io.StringIO()
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buffer
    try:
        code = pytest.main(["-q", "-p", "no:cacheprovider", "--no-header",
                            f"{TESTS}::{node_id}"])
    finally:
        sys.stdout, sys.stderr = stdout, stderr
    return code == 0, buffer.getvalue()


# --- the mutations ---------------------------------------------------------
#
# Each is a (name, apply, revert, expected-to-break) tuple. `apply` returns
# whatever `revert` needs to put things back; nothing touches disk.


def _mutations() -> dict:
    search_cls = guofish_core.ReplaySearchQ32
    outcome_cls = playv6.SearchOutcome
    config_cls = playv6.EngineConfig

    def make_property_mutation(cls, name, replacement):
        original = getattr(cls, name)

        def apply():
            setattr(cls, name, replacement)

        def revert():
            setattr(cls, name, original)

        return apply, revert

    mutations = {}

    # 1. The deadline never arms.
    #
    # SUBSTITUTED RATHER THAN PATCHED. `ReplaySearchQ32` is a pybind11 type and
    # its methods are read-only slots, so `setattr` on the class raises. It IS
    # subclassable, though, so the mutation swaps the name in `guofish_core` for
    # a subclass whose `set_deadline_in` does nothing — which is what a
    # never-armed deadline looks like from every caller's side, in the tests and
    # in the host alike.
    #
    # `clear_deadline` is deliberately left working: the mutation is "the
    # deadline is never SET", not "the deadline machinery is absent", and
    # leaving the disarm intact keeps the tests failing for the reason the
    # mutation names rather than for a second one it introduced.
    class DeadlineDeaf(search_cls):                     # noqa: N801
        def set_deadline_in(self, seconds):             # noqa: D102
            return None

    mutations["deadline-never-fires"] = {
        "apply": lambda: setattr(guofish_core, "ReplaySearchQ32", DeadlineDeaf),
        "revert": lambda: setattr(guofish_core, "ReplaySearchQ32", search_cls),
        "must_break": [
            "test_a_deadline_armed_before_the_search_ends_it_on_time",
            "test_the_deadline_is_settable_from_another_thread_mid_search",
            "test_clearing_the_deadline_returns_the_search_to_its_node_budget",
        ],
    }

    # 2. Exhaustion is reported as sound.
    apply2, revert2 = make_property_mutation(
        outcome_cls, "arena_exhausted", property(lambda self: False))
    mutations["exhaustion-silent"] = {
        "apply": apply2, "revert": revert2,
        "must_break": [
            "test_a_degraded_move_says_so_loudly_in_its_own_telemetry",
        ],
    }

    # 3. The two sim counts are merged.
    apply3, revert3 = make_property_mutation(
        outcome_cls, "search_sims",
        property(lambda self: self.ponder_sims + self.delivered))
    mutations["merge-the-sim-counts"] = {
        "apply": apply3, "revert": revert3,
        "must_break": [
            "test_ponder_and_search_simulations_are_reported_separately",
        ],
    }

    # 4. The arena formula fitted the observed ratio instead of the safe one.
    original_arena = config_cls.arena_nodes
    apply4, revert4 = make_property_mutation(
        config_cls, "arena_nodes",
        property(lambda self: original_arena.fget(self) // 2))
    mutations["arena-formula-halved"] = {
        "apply": apply4, "revert": revert4,
        "must_break": [
            "test_the_arena_default_is_sixty_nodes_per_simulation_of_both_budgets",
        ],
    }

    # 5. The coupling is never checked.
    apply5, revert5 = make_property_mutation(
        config_cls, "coupling_holds", property(lambda self: True))
    mutations["coupling-unchecked"] = {
        "apply": apply5, "revert": revert5,
        "must_break": [
            "test_a_pinned_ponder_cap_that_breaks_the_coupling_is_reported_not_refused",
        ],
    }

    return mutations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    before = golden_digests()
    print(f"golden/: {len(before)} files, SHA-256 recorded (Amendment B)",
          flush=True)

    mutations = _mutations()
    names = args.only or list(mutations)
    rows: list[dict] = []
    ok = True

    for name in names:
        if name not in mutations:
            raise SystemExit(f"unknown mutation {name!r}; "
                             f"known: {', '.join(mutations)}")
        spec = mutations[name]
        print(f"\n=== {name} ===", flush=True)

        # THE CONTROL FIRST. A test that was already failing would make the
        # mutation look effective when it changed nothing, which is the one way
        # a mutation drill can lie in the reassuring direction.
        baseline = {}
        for node in spec["must_break"]:
            passed, _ = run_test(node)
            baseline[node] = passed
            print(f"  baseline {node}: {'PASS' if passed else 'FAIL'}",
                  flush=True)
            if not passed:
                ok = False
                print(f"  !! the test is red BEFORE the mutation; this drill "
                      f"proves nothing about it", flush=True)

        spec["apply"]()
        try:
            for node in spec["must_break"]:
                passed, output = run_test(node)
                broke = (not passed) and baseline[node]
                print(f"  mutated  {node}: {'PASS' if passed else 'FAIL'}"
                      f"  -> {'CAUGHT' if broke else 'NOT CAUGHT'}", flush=True)
                rows.append({"mutation": name, "test": node,
                             "baseline_passed": baseline[node],
                             "mutated_passed": passed, "caught": broke})
                if not broke:
                    ok = False
                    print("  !! THE MUTATION SURVIVED. The mechanism is not "
                          "covered by the test named for it.", flush=True)
                    print("  " + "\n  ".join(output.strip().splitlines()[-4:]))
        finally:
            spec["revert"]()

    after = golden_digests()
    unchanged = before == after
    print(f"\ngolden/ unchanged: {unchanged}")
    if not unchanged:
        ok = False
        changed = [k for k in before if before.get(k) != after.get(k)]
        print(f"  !! AMENDMENT B VIOLATION: {changed}")

    caught = sum(1 for r in rows if r["caught"])
    print(f"\n{caught}/{len(rows)} mutations caught")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"rows": rows, "golden_unchanged": unchanged}, indent=1),
            encoding="utf-8", newline="\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
