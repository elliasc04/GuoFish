"""Populate the C0b tables in BENCH.md — contended GIL acquisition.

NOTE: this is a benchmark, not a golden-data generator. Nothing here writes to
golden/ and nothing here is a reference implementation (Global Rule 2).

Usage:
    python tools/bench_c0b.py [--scale N]

The experiment itself lives in ``tests/test_c0b_contention.py`` and is imported
from there rather than reimplemented. That is deliberate: the numbers published
in BENCH.md and the numbers the acceptance test asserts on must come from the
same harness, or a future edit to one can silently invalidate the other.

``--scale`` multiplies the per-config iteration counts. The test uses scale 1 to
stay fast; the published table is generated at a higher scale so the p99 and max
columns are backed by more samples. Config B costs a full switch interval (5 ms)
per iteration, so scale it with that in mind.
"""

import argparse
import platform
import statistics
import sys
from pathlib import Path

# pytest.ini's `pythonpath = .` only applies under pytest, so make both the
# built extension and the tests package importable however this is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import guofish_core  # noqa: E402

from tests.test_c0b_contention import (  # noqa: E402
    BUILD,
    DISPATCH_GAP_US,
    FAST_SWITCH_INTERVAL,
    GATE_MAX_US,
    GATE_P99_US,
    ITERS,
    UciFormatterLoad,
    _measure,
    print_report,
    run_configurations,
)


def stability(repeats, iters):
    """Re-run the gate configuration `repeats` times.

    ``max`` is one sample out of thousands, so a single run says very little
    about it. This is the evidence the BENCH.md verdict rests on: how often the
    tail crosses 2 ms, on this build, over a large number of samples.
    """
    print(f"platform : {platform.system()} {platform.machine()}")
    print(f"build    : {BUILD['compiler']}, asan={BUILD['asan']} ubsan={BUILD['ubsan']}")
    print(f"config C, batch 256, {repeats} runs x {iters} iterations")
    print()
    print("| run | p50 | p95 | p99 | max |")
    print("|---:|---:|---:|---:|---:|")

    _measure(32, 200)
    original = sys.getswitchinterval()
    p99s, maxes = [], []
    try:
        sys.setswitchinterval(FAST_SWITCH_INTERVAL)
        with UciFormatterLoad():
            for i in range(repeats):
                a = _measure(256, iters, DISPATCH_GAP_US)["acquire_wait_us"]
                p99s.append(a["p99"])
                maxes.append(a["max"])
                print(
                    f"| {i} | {a['p50']:.1f} | {a['p95']:.1f} | {a['p99']:.1f} | {a['max']:.1f} |"
                )
    finally:
        sys.setswitchinterval(original)

    print()
    print(f"samples          : {repeats * iters:,}")
    print(
        f"p99 across runs  : min {min(p99s):.1f} median {statistics.median(p99s):.1f} "
        f"max {max(p99s):.1f} us  (gate {GATE_P99_US:.0f})"
    )
    print(
        f"max across runs  : min {min(maxes):.1f} median {statistics.median(maxes):.1f} "
        f"max {max(maxes):.1f} us  (gate {GATE_MAX_US:.0f})"
    )
    print(f"runs over p99 gate: {sum(p >= GATE_P99_US for p in p99s)}/{repeats}")
    print(f"runs over max gate: {sum(m >= GATE_MAX_US for m in maxes)}/{repeats}")


SWEEP_INTERVALS = (0.0001, 0.0005, 0.0009, 0.0011, 0.0015, 0.002, 0.0025, 0.003, 0.005, 0.010)


def sweep():
    """Acquire wait as a function of sys.setswitchinterval.

    This is what shows that the mitigation is a cliff rather than a slope, and
    where the cliff is. The callback is deliberately trivial (no numpy): a
    callback that releases the GIL internally would blur the boundary being
    located.
    """
    print(f"platform : {platform.system()} {platform.machine()}")
    print(f"build    : {BUILD['compiler']}, asan={BUILD['asan']}")
    print(f"batch 256, gap {DISPATCH_GAP_US} us, trivial callback")
    print()
    print("| switchinterval (s) | -> ms timeout | p50 | p95 | p99 | max | iters |")
    print("|---|---:|---:|---:|---:|---:|---:|")

    def trivial(arr):
        pass

    guofish_core.contention_bench(32, 200, trivial, 0.0)
    original = sys.getswitchinterval()
    try:
        for interval in SWEEP_INTERVALS:
            sys.setswitchinterval(interval)
            # Above the cliff each iteration costs a full handoff, so fewer.
            iters = 200 if interval >= 0.002 else 600
            with UciFormatterLoad():
                r = guofish_core.contention_bench(256, iters, trivial, DISPATCH_GAP_US)
            a = r["acquire_wait_us"]
            print(
                f"| {interval} | {int(interval * 1e6) // 1000} | {a['p50']:.1f} | "
                f"{a['p95']:.1f} | {a['p99']:.1f} | {a['max']:.1f} | {iters} |"
            )
    finally:
        sys.setswitchinterval(original)

    print()
    print("The '-> ms timeout' column is (interval_us // 1000): what CPython's Windows")
    print("condition variable receives, since SleepConditionVariableSRW takes a DWORD")
    print("of milliseconds.")
    if platform.system() == "Windows":
        print("Expect a step function: every interval below 1 ms truncates to a 0 ms wait")
        print("and the drop request fires immediately, so p50 collapses to a few us. At")
        print("1 ms and above the wait quantises to the ~15.6 ms system timer tick.")
    else:
        print("Expect p50 to track the interval continuously: pthread_cond_timedwait takes")
        print("a nanosecond-resolution deadline, so there is no cliff and no quantisation.")
        print("Bounding the acquire wait here means genuinely choosing a smaller interval.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        type=float,
        default=4.0,
        help="multiplier on the per-config iteration counts (default 4)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="instead of the A/B/C tables, re-run the gate configuration N times "
             "and report how stable p99 and max are",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="instead of the A/B/C tables, sweep sys.setswitchinterval and show "
             "where the acquire wait changes regime",
    )
    args = parser.parse_args()

    assert guofish_core.ping() == "pong"

    if args.sweep:
        sweep()
        return

    if args.repeat:
        stability(args.repeat, int(ITERS["C"] * args.scale))
        return

    iters = {k: max(1, int(v * args.scale)) for k, v in ITERS.items()}
    est_s = iters["B"] * 0.005 + iters["C"] * 0.0005
    print(f"scale {args.scale} -> iters {iters} (config B alone will take ~{est_s:.0f} s)")
    print()

    measurements = run_configurations(iters)
    print_report(measurements)

    print()
    gate = measurements[("C", 256)]["acquire_wait_us"]
    p99_ok = gate["p99"] < GATE_P99_US
    max_ok = gate["max"] < GATE_MAX_US
    verdict = "PASS" if (p99_ok and max_ok) else "FAIL"
    print(
        f"gate (config C, batch 256): p99 {gate['p99']:.1f} us vs {GATE_P99_US:.0f} us "
        f"[{'ok' if p99_ok else 'FAIL'}], max {gate['max']:.1f} us vs {GATE_MAX_US:.0f} us "
        f"[{'ok' if max_ok else 'FAIL'}] -> {verdict}"
    )

    a = measurements[("A", 256)]["acquire_wait_us"]["p50"]
    b = measurements[("B", 256)]["acquire_wait_us"]["p50"]
    print(
        f"validity (batch 256): p50 acquire wait A={a:.3f} us -> B={b:.1f} us "
        f"({b / max(a, 1e-9):.0f}x). B must visibly degrade or the experiment is broken."
    )


if __name__ == "__main__":
    main()
