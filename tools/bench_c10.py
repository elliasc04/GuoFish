#!/usr/bin/env python
"""Populate the C10 tables in BENCH.md — the live evaluation boundary.

NOTE: this is a benchmark, not a golden-data generator. Nothing here writes to
golden/ and nothing here is a reference implementation (Global Rule 2).

THE TWO QUESTIONS
=================
1. WHAT DOES THE SWITCH INTERVAL COST? C0b established what the default interval
   costs when the dispatcher is CONTENDED (median 15.25 ms acquire wait on
   Windows, a ceiling of ~60 batches/s) and made `sys.setswitchinterval(0.0005)`
   mandatory. It could not establish the other half: the shorter interval raises
   context-switch frequency interpreter-wide, and C0b flagged that overhead as
   unmeasured. So this runs a REAL SEARCH at 0.005 and at 0.0005, with and
   without a competing Python thread, and reports delivered simulations per
   second. Four cells, one of which is the production configuration and one of
   which is the disaster the setting exists to prevent.

2. IS C++-SIDE `info` EMISSION REQUIRED? The C10 brief makes it contingent on a
   measurement rather than mandatory: implement it if the dispatcher's
   acquire-wait p99 exceeds 200 us, or if the total acquire wait exceeds 1% of
   the move's wall time. Both are properties of a distribution, so the histogram
   is collected per batch and reported as percentiles, never as a mean — a mean
   hides the tail behind the many fast acquisitions and the tail is the entire
   cost.

WHY THE COMPETING THREAD IS THE INTERESTING ROW
===============================================
Under scope §2.1's discipline the only Python-relevant threads during a search
are the dispatcher and a main thread blocked on stdin, which releases the GIL
while blocked — so the quiet rows measure the intended state and should show
almost no wait. The competing-thread rows measure the state the trigger was
written for: a pure-Python UCI formatter emitting `info` lines while the search
runs, which is exactly what the contingent C++ emission would remove. Running
only the quiet rows would answer "is there a problem right now" and not "does
the mitigation hold", and the second is what the threshold is about.

`UciFormatterLoad` is imported from tests/test_c0b_contention.py rather than
reimplemented: the numbers published in BENCH.md and the ones the acceptance
tests assert on have to come from the same competitor, or an edit to one
silently invalidates the other.

Usage:
    python tools/bench_c10.py
    python tools/bench_c10.py --sims 4000 --repeats 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import guofish_core  # noqa: E402
from tests.test_c0b_contention import UciFormatterLoad  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# A quiet midgame, deep enough that the tree is real and the cache is working.
BENCH_FEN = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"
# The C10 smoke position, so the throughput table and the acceptance run are
# talking about the same search.
SMOKE_FEN = "8/2R5/4R2p/5pp1/2N1k3/6Pb/r4r1P/6K1 b - - 11 46"

DEFAULT_INTERVAL = 0.0005
SLOW_INTERVAL = 0.005

CACHE_SLOTS = 400_000

# The C10 brief's thresholds.
TRIGGER_P99_US = 200.0
TRIGGER_WALL_SHARE = 0.01


def percentiles(samples_ns):
    """Nearest-rank, no interpolation: every number is a wait that occurred."""
    ordered = sorted(int(s) for s in samples_ns)
    if not ordered:
        return {}
    def at(q):
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))] / 1000.0
    return {
        "n": len(ordered),
        "p50": at(0.50),
        "p90": at(0.90),
        "p99": at(0.99),
        "max": ordered[-1] / 1000.0,
        "mean": statistics.fmean(ordered) / 1000.0,
    }


def one_run(model, device, *, interval, workers, in_flight, max_batch, sims, fen,
            competing: bool):
    """One search, on a freshly built evaluator at `interval`.

    A fresh evaluator per cell because the switch interval is set in its
    constructor — that is where scope §2.1's "before any search thread" lives —
    and because a fresh cache makes the cells comparable.
    """
    from playing.v6 import evaluator as live_evaluator

    torch_evaluator = live_evaluator.TorchEvaluator(model, device, max_batch,
                                                    switch_interval=interval)
    config = guofish_core.SearchConfig()
    config.cache_slots = CACHE_SLOTS
    search = guofish_core.ReplaySearchDouble(config)
    search.set_evaluator(torch_evaluator.core)
    parallel = guofish_core.ParallelConfig(workers=workers, in_flight=in_flight,
                                           max_batch=max_batch)
    try:
        # Warm the kernels and the cache so the timed run measures a steady
        # state rather than cuDNN's first-call autotuning.
        search.set_position(fen)
        search.search_parallel(max(64, sims // 8), parallel)

        search.set_position(fen)
        if competing:
            with UciFormatterLoad() as load:
                started = time.perf_counter()
                stats = search.search_parallel(sims, parallel)
                wall = time.perf_counter() - started
            competitor_iterations = load.iterations
        else:
            started = time.perf_counter()
            stats = search.search_parallel(sims, parallel)
            wall = time.perf_counter() - started
            competitor_iterations = 0

        parallel_stats = search.parallel_stats()
        eval_stats = search.eval_stats()
        acquire = percentiles(eval_stats["acquire_wait_ns_samples"])
        call = percentiles(eval_stats["call_ns_samples"])
        return {
            "interval": interval,
            "competing": competing,
            "workers": workers,
            "in_flight": in_flight,
            "max_batch": max_batch,
            "sims": sims,
            "delivered": int(parallel_stats["delivered"]),
            "wall_s": wall,
            "delivered_per_s": parallel_stats["delivered"] / wall,
            "batches": int(eval_stats["batches"]),
            "rows": int(eval_stats["rows"]),
            "cache_skipped": int(eval_stats["cache_skipped"]),
            "mean_rows": eval_stats["mean_rows"],
            "acquire_us": acquire,
            "call_us": call,
            "acquire_total_ms": eval_stats["acquire_wait_ns"] / 1e6,
            "acquire_wall_share": eval_stats["acquire_wait_ns"] / (wall * 1e9),
            "competitor_iterations": competitor_iterations,
            "best_move": stats["best_move"],
        }
    finally:
        search.set_evaluator(None)
        torch_evaluator.close()


def print_throughput(rows):
    print()
    print("### C10 — switch interval vs real search throughput")
    print()
    print("| interval | competing Python thread | W | K | delivered | wall (s) | "
          "delivered sims/s | batches | rows/batch |")
    print("|---:|:--|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['interval']} | {'yes' if r['competing'] else 'no'} | {r['workers']} | "
              f"{r['in_flight']} | {r['delivered']} | {r['wall_s']:.2f} | "
              f"{r['delivered_per_s']:.0f} | {r['batches']} | {r['mean_rows']:.1f} |")


def print_histogram(rows):
    print()
    print("### C10 — dispatcher GIL acquire wait, per batch (microseconds)")
    print()
    print("| interval | competing | batches | p50 | p90 | p99 | max | mean | "
          "total (ms) | share of wall | trigger |")
    print("|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|:--|")
    for r in rows:
        a = r["acquire_us"]
        fired = (a["p99"] > TRIGGER_P99_US) or (r["acquire_wall_share"] > TRIGGER_WALL_SHARE)
        print(f"| {r['interval']} | {'yes' if r['competing'] else 'no'} | {a['n']} | "
              f"{a['p50']:.2f} | {a['p90']:.2f} | {a['p99']:.2f} | {a['max']:.2f} | "
              f"{a['mean']:.2f} | {r['acquire_total_ms']:.2f} | "
              f"{r['acquire_wall_share']:.4%} | {'FIRED' if fired else 'clear'} |")
    print()
    print(f"Trigger: p99 > {TRIGGER_P99_US:.0f} us, or acquire wait > "
          f"{TRIGGER_WALL_SHARE:.0%} of the move's wall time. 'FIRED' in any row that "
          f"represents a configuration the engine actually ships in makes C++-side "
          f"`info` emission mandatory for C10.")


def print_call(rows):
    print()
    print("### C10 — callback duration, per batch (microseconds)")
    print()
    print("NOT a contention metric. Torch re-acquires the GIL inside the call, so a "
          "handoff wait lands in the middle of it; the ratio to `acquire` is what says "
          "whether a slow batch was contention or was just the model.")
    print()
    print("| interval | competing | rows/batch | p50 | p90 | p99 | max |")
    print("|---:|:--|---:|---:|---:|---:|---:|")
    for r in rows:
        c = r["call_us"]
        print(f"| {r['interval']} | {'yes' if r['competing'] else 'no'} | "
              f"{r['mean_rows']:.1f} | {c['p50']:.1f} | {c['p90']:.1f} | {c['p99']:.1f} | "
              f"{c['max']:.1f} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sims", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--in-flight", type=int, default=32)
    parser.add_argument("--max-batch", type=int, default=256)
    parser.add_argument("--fen", default=BENCH_FEN)
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the raw rows here")
    parser.add_argument("--allow-instrumented", action="store_true",
                        help="publish tables from a sanitizer build anyway")
    args = parser.parse_args()

    import torch
    from playing.v6 import evaluator as live_evaluator

    if not torch.cuda.is_available():
        print("ERROR: this benchmark measures the CUDA path; refusing to publish "
              "numbers from a CPU run.", file=sys.stderr)
        return 1

    build = guofish_core.build_info()
    # REFUSED RATHER THAN WARNED, because this already happened once. Every build
    # directory stages the module into the repo root, so a `Release` build whose
    # inputs are older than a sanitizer build's staged output relinks nothing and
    # reports BUILD_OK; a three-and-a-half-hour run then produced instrumented
    # throughput numbers that looked publishable. `build_info()` is the check.
    # Delete the staged `.pyd` before switching configurations.
    if (build["asan"] or build["ubsan"]) and not args.allow_instrumented:
        print(f"ERROR: this is a sanitizer build ({build}). Throughput measured under "
              f"instrumentation is not comparable to anything in BENCH.md — see "
              f"README_BUILD.md, 'Benchmark on the Release build'. Delete the staged "
              f"guofish_core*.pyd, rebuild Release, and check build_info() again. "
              f"Pass --allow-instrumented to override.", file=sys.stderr)
        return 1

    model, device = live_evaluator.load_default_model()

    print(f"platform : {platform.system()} {platform.machine()}")
    print(f"build    : {build['compiler']}, asan={build['asan']} ubsan={build['ubsan']} "
          f"asserts={build['asserts']}")
    print(f"device   : {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    print(f"position : {args.fen}")
    print(f"budget   : {args.sims} simulations, W={args.workers} K={args.in_flight} "
          f"max_batch={args.max_batch}")

    rows = []
    for interval in (SLOW_INTERVAL, DEFAULT_INTERVAL):
        for competing in (False, True):
            row = one_run(model, device, interval=interval, workers=args.workers,
                          in_flight=args.in_flight, max_batch=args.max_batch,
                          sims=args.sims, fen=args.fen, competing=competing)
            rows.append(row)
            print(f"  ran interval={interval} competing={competing}: "
                  f"{row['delivered_per_s']:.0f} sims/s, acquire p99 "
                  f"{row['acquire_us']['p99']:.2f} us", flush=True)

    print_throughput(rows)
    print_histogram(rows)
    print_call(rows)

    quiet = {r["interval"]: r for r in rows if not r["competing"]}
    loud = {r["interval"]: r for r in rows if r["competing"]}
    print()
    print("### C10 — what the shorter interval bought and what it cost")
    print()
    cost = (quiet[DEFAULT_INTERVAL]["delivered_per_s"] /
            quiet[SLOW_INTERVAL]["delivered_per_s"] - 1.0)
    gain = (loud[DEFAULT_INTERVAL]["delivered_per_s"] /
            loud[SLOW_INTERVAL]["delivered_per_s"] - 1.0)
    print(f"- Cost, quiet interpreter (0.005 -> 0.0005): {cost:+.2%} delivered sims/s. "
          f"This is the context-switch overhead C0b flagged as unmeasured.")
    print(f"- Gain, competing Python thread: {gain:+.2%} delivered sims/s.")
    print(f"- Acquire-wait p99 under contention: "
          f"{loud[SLOW_INTERVAL]['acquire_us']['p99']:.1f} us at 0.005, "
          f"{loud[DEFAULT_INTERVAL]['acquire_us']['p99']:.1f} us at 0.0005.")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8",
                             newline="\n")
        print(f"\nraw rows -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
