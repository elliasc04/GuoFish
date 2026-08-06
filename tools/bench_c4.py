"""Populate the C4 section of BENCH.md — the sibling scan, under both accumulators.

NOTE: this is a benchmark, not a golden-data generator. Nothing here writes to
golden/ and nothing here is a reference implementation (Global Rule 2).

Usage:
    python tools/bench_c4.py [--trials N] [--sweep] [--markdown]

What this exists to decide
--------------------------
C4's microbenchmark is a design decision, not a measurement. `value_sum` is a
Q32 fixed-point `atomic<int64>` in production, and every read of it in PUCT
selection has to be converted back to a double. That conversion sits in the
hottest loop in the engine — a scan of all 32-ish siblings, once per node on
every simulation's path. If it does not disappear into the loop, the layout has
to change *now*: the fallback is a separate `float` array of Q values updated on
backup, which is a second array to keep coherent and a second thing to get
wrong. Changing this after C9 is expensive.

So the loop below reads exactly what selection reads — `visit_count`,
`value_sum` and `prior` for a contiguous block of siblings — and reduces them to
an argmax so the reads are observable. It is deliberately *not* the PUCT
formula: no cpuct, no sqrt(parent visits), no virtual loss. That makes the
per-child arithmetic cheaper than the real thing, so the conversion's share of
the loop is overstated here rather than flattered. The conservative direction
for a decision about affordability.

Four working sets, because the answer differs by regime and only one of the four
is the engine's:

    32 nodes      one block, everything in L1. Pure ALU cost of the conversion.
    2,048         still L1 (26 KB of hot fields).
    131,072       L2/L3.
    2,097,152     the scope's 2-3M node budget. Memory-bound; this is the row
                  the decision is made on.

Each configuration is run `--trials` times and the median reported, because a
single trial on a desktop measures whatever else the machine was doing.
"""

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

# pytest.ini's `pythonpath = .` only applies under pytest, so make the built
# extension importable however this script is invoked.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

BLOCK_SIZE = 32

# (label, blocks, repeats). `repeats` is chosen so every configuration runs for
# roughly the same wall time regardless of how much memory it touches.
CONFIGURATIONS = [
    ("L1 - one block", 1, 2_000_000),
    ("L1 - 2,048 nodes", 64, 100_000),
    ("L2/L3 - 131k nodes", 4_096, 2_000),
    ("RAM - 2.1M nodes", 65_536, 100),
]


def run_configuration(blocks, repeats, trials):
    q32 = []
    dbl = []
    checksum = None
    for _ in range(trials):
        result = guofish_core.sibling_scan_bench(
            blocks=blocks, block_size=BLOCK_SIZE, repeats=repeats
        )
        q32.append(result["q32_ns_per_scan"])
        dbl.append(result["double_ns_per_scan"])
        if checksum is None:
            checksum = result["checksum"]
        elif checksum != result["checksum"]:
            sys.exit(
                "sibling_scan_bench is not deterministic across trials — the filler or the "
                "scan changed behaviour between runs, and neither should"
            )
    if not checksum:
        sys.exit("sibling_scan_bench checksum is zero: the scan's reads were optimised away")
    return statistics.median(q32), statistics.median(dbl)


def human_bytes(n):
    for unit, scale in (("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:,.0f} {unit}"
    return f"{n:,.0f} B"


def hot_bytes_per_node():
    probe = guofish_core.sibling_scan_bench(blocks=1, block_size=BLOCK_SIZE, repeats=1)
    return probe["hot_bytes_per_node_q32"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5, help="runs per configuration; median reported")
    parser.add_argument("--sweep", action="store_true", help="also time the exhaustive Q32 sweep")
    parser.add_argument("--markdown", action="store_true", help="emit the BENCH.md table only")
    args = parser.parse_args()

    info = guofish_core.build_info()
    layout = guofish_core.arena_layout()
    hot = hot_bytes_per_node()

    if not args.markdown:
        print(f"platform      : {platform.platform()}")
        print(f"python        : {platform.python_version()}")
        print(f"compiler      : {info['compiler']}")
        print(f"asan / ubsan  : {info['asan']} / {info['ubsan']}   asserts: {info['asserts']}")
        print(f"default accum : {layout['default_accumulator']}")
        print(f"bytes/node    : {layout['bytes_per_node_q32']} (all nine fields)")
        print(f"hot bytes/node: {hot} (visit_count + value_sum + prior)")
        print()
        if info["asan"] or info["asserts"]:
            print(
                "WARNING: this is not a production build. ASan and live asserts both\n"
                "         distort the loop this benchmark exists to measure; the numbers\n"
                "         in BENCH.md come from a Release build.\n"
            )

    rows = []
    for label, blocks, repeats in CONFIGURATIONS:
        q32_ns, dbl_ns = run_configuration(blocks, repeats, args.trials)
        nodes = blocks * BLOCK_SIZE
        rows.append(
            {
                "label": label,
                "nodes": nodes,
                "hot_bytes": nodes * hot,
                "q32_scan": q32_ns,
                "dbl_scan": dbl_ns,
                "q32_child": q32_ns / BLOCK_SIZE,
                "dbl_child": dbl_ns / BLOCK_SIZE,
                "ratio": q32_ns / dbl_ns,
            }
        )

    print(
        "| working set | nodes | hot set | Q32 ns/scan | double ns/scan | Q32 ns/child | "
        "double ns/child | Q32 / double |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['label']} | {row['nodes']:,} | {human_bytes(row['hot_bytes'])} | "
            f"{row['q32_scan']:.2f} | {row['dbl_scan']:.2f} | "
            f"{row['q32_child']:.3f} | {row['dbl_child']:.3f} | {row['ratio']:.3f}x |"
        )

    if args.sweep:
        print()
        start = time.perf_counter()
        result = guofish_core.q32_roundtrip_sweep(1)
        elapsed = time.perf_counter() - start
        print(f"exhaustive Q32 float sweep: {result['floats_examined']:,} floats in {elapsed:.1f}s")
        print(f"  max abs error : {result['max_abs_error']:.6e}  (half resolution "
              f"{result['half_resolution']:.6e})")
        print(f"  inexact       : {result['inexact']:,}, largest {result['largest_inexact']:.10g} "
              f"(2^-9 = {2.0**-9})")
        print(f"  code mismatch : {result['code_mismatches']}   asymmetric: {result['asymmetric']}")


if __name__ == "__main__":
    main()
