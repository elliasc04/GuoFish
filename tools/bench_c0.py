"""Populate the C0 table in BENCH.md.

NOTE: this is a benchmark, not a golden-data generator. Nothing here writes to
golden/ and nothing here is a reference implementation (Global Rule 2).

Usage:
    python tools/bench_c0.py [--iters N]

Reports the GIL acquire -> Python callback -> return round-trip latency at the
batch sizes the C0 brief names. The number that matters is p99 at 256 rows: if
it exceeds ~2 ms the dispatcher design in scope 2.1 needs revisiting before C5.

C0b: the callback now receives the live zero-copy view and sums it, so this
table includes the cost of crossing the boundary *with data*. The GIL is still
uncontended here — see tools/bench_c0b.py for the contended numbers.
"""

import argparse
import platform
import sys
from pathlib import Path

# pytest.ini's `pythonpath = .` only applies under pytest, so make the built
# extension importable however this script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import guofish_core  # noqa: E402

BATCH_SIZES = (32, 64, 128, 256)


def run(rows, iters):
    calls = 0
    total = 0

    def callback(arr):
        nonlocal calls, total
        calls += 1
        # Touch the payload: without a read, nothing forces the data to actually
        # cross the boundary and the table would understate the real cost.
        total += int(arr.sum())

    res = guofish_core.roundtrip_bench(rows, iters, callback)
    assert calls == iters, f"callback ran {calls} times, expected {iters}"
    assert total != 0, "callback summed an all-zero buffer; payload is not live"
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20000)
    args = parser.parse_args()

    # Warm up: first call pays interpreter/branch-predictor cold-start costs.
    run(32, 2000)

    clock = guofish_core.clock_info(200000)

    print(f"platform : {platform.system()} {platform.machine()}")
    print(f"python   : {platform.python_version()} ({sys.implementation.name})")
    print(f"iters    : {args.iters} per batch size")
    print(
        f"clock    : steady_clock nominal tick {clock['nominal_tick_ns']:.1f} ns, "
        f"measured {clock['measured_tick_ns']:.1f} ns"
    )
    print()
    print("| rows | median (us) | p99 (us) | mean (us) | max (us) | p99 (ms) |")
    print("|-----:|------------:|---------:|----------:|---------:|---------:|")

    worst = 0.0
    for rows in BATCH_SIZES:
        r = run(rows, args.iters)
        worst = max(worst, r["p99_ms"])
        print(
            f"| {rows} | {r['median_us']:.3f} | {r['p99_us']:.3f} | "
            f"{r['mean_us']:.3f} | {r['max_us']:.3f} | {r['p99_ms']:.4f} |"
        )

    print()
    verdict = "PASS" if worst < 2.0 else "FAIL"
    print(f"gate: worst p99 = {worst:.4f} ms vs 2 ms budget -> {verdict}")


if __name__ == "__main__":
    main()
