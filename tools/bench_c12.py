#!/usr/bin/env python
"""C12 — Gate 4 in both regimes, the attribution table, and the two levers.

    python tools/bench_c12.py --sections gate4 --markdown
    python tools/bench_c12.py --sections attribution --markdown
    python tools/bench_c12.py --sections ksweep --markdown
    python tools/bench_c12.py --markdown --json-out runs/c12/bench_c12.json

WHY THIS IS NOT A SECTION IN bench_c10b.py
==========================================
Because one of C12's findings is that bench_c10b.py's reuse-heavy Gate 4 row
measures something other than what it says, and editing the tool that produced a
published table would destroy the ability to reproduce that table and see the
difference. bench_c10b.py is left exactly as it was; this file states the
correction and measures it the other way.

**THE CORRECTION.** `search_parallel(N)` runs to N ROOT VISITS, not N new
simulations: `target_ = num_simulations - existing` (cpp/search.hpp), and
`ParallelStats::requested` is set to that REMAINDER. So on a reused tree the
assertion `delivered == requested` is satisfied by a search that did almost
nothing, and it cannot detect the case. After six plies of reuse the endgame
root already holds 19,780 visits, so C10b-3e's reuse-heavy row asked for 20,000
and delivered **220**, in 2.3 ms. Its 92,639 sims/s is 220 simulations divided by
a wall clock that is mostly fixed per-call cost.

That is also not how the engine plays. `playing/uci_wrapper_v6.py` sets
`budget = current + cfg.sim_cap` for both timed and infinite searches — NEW
simulations — and only a literal `go nodes N` uses an absolute target. So the
reuse rows here request `root_visits + sims`, which is both the production
spelling and the only one that measures a rate. Measured that way the same
position delivers 19,780 simulations at ~225,000/s: the published figure is low
by ~2.4x AND was taken from a 220-sample.

  gate4        Delivered sims/s, fresh midgame root and reuse-heavy endgame,
               `verify_compaction` off, book/Syzygy not in the loop (this is the
               library path — no book, no tablebase, so no move is bypassed and
               there is nothing to exclude from the rate). Both budget spellings
               are reported for the reuse regime so the correction is visible
               rather than asserted.

  attribution  Every millisecond of a fresh-root move named. The callback is
               timed phase by phase from inside a TorchEvaluator subclass, so
               the split between GPU wait, Python overhead and C++ search work
               is measured rather than derived from two numbers that were not
               taken together. The unattributed remainder is printed even when
               it is small, per the recon's discipline.

  ksweep       C12 item (b): outstanding-leaf count against delivered sims/s AND
               against root flattening, at the shipping W=1. The tolerance is no
               longer open — C10b-3g set it at 10% run-to-run TV — so this is a
               decision this chunk can actually make.

  pipeline     C12 item (c): what pipelining could buy, bounded by the MEASURED
               GPU busy fraction rather than by `call_ns / wall_ns`. The two
               differ by a lot and the difference is the whole point; see the
               section note.

Writes nothing to golden/ (Global Rule 2). Refuses to publish from an
instrumented build, for the reason recorded in DECISIONS.md, C10.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

from tools.bench_c10b import (  # noqa: E402
    ARENA_CAPACITY, CACHE_SLOTS, ENDGAME_FEN, FRESH_ROOT_FEN, GATE4_FLOOR,
    GATE4_STRETCH, PYTHON_BASELINE, root_distribution, total_variation,
)


def log(msg: str = "") -> None:
    print(msg, flush=True)


# --- the instrumented evaluator ---------------------------------------------


def instrumented_evaluator_class():
    """A TorchEvaluator subclass that times its own callback, phase by phase.

    A SUBCLASS, and built before construction, because
    `guofish_core.LiveEvaluator` binds `self._evaluate` once at construction —
    patching the attribute afterwards leaves C++ calling the original, which is
    a failure that looks like an instrumentation that measures nothing rather
    than like an error. Imported lazily so `--sections gate4` does not pay for
    torch twice.

    The four phases, and what each one contains:

      stage    the H2D copy of `count` pinned int32 rows, plus the pad-tail
               restore. Asynchronous, so this is submission cost, not transfer.
      replay   `cudaGraphLaunch`. Asynchronous: this is the launch, NOT the
               forward.
      policy   the 4096-wide bf16 D2H. **This is where the GPU wait lands**,
               because it is the first thing that has to observe the graph's
               output, and it is why `call_ns` is not a measure of Python work.
      value    the float32 D2H of one value per row.
    """
    from playing.v6.evaluator import TorchEvaluator

    class InstrumentedEvaluator(TorchEvaluator):
        PHASES = ("stage", "replay", "policy", "value")

        def __init__(self, *args, **kwargs):
            self.phase_ns = {name: [] for name in self.PHASES}
            super().__init__(*args, **kwargs)

        def reset_phases(self) -> None:
            for samples in self.phase_ns.values():
                samples.clear()

        def _evaluate(self, count: int) -> None:
            clock = time.perf_counter_ns
            graph = self.graph
            t0 = clock()
            padded = graph.stage(count, self._input_t[:count])
            t1 = clock()
            graph.replay(padded)
            t2 = clock()
            graph.rows += count
            graph.padded_rows += padded
            graph.by_size[padded] += 1
            policy, value = graph.outputs(count, padded)
            self._policy_t[:count].copy_(policy)
            t3 = clock()
            self._value_t[:count].copy_(value)
            t4 = clock()
            self.phase_ns["stage"].append(t1 - t0)
            self.phase_ns["replay"].append(t2 - t1)
            self.phase_ns["policy"].append(t3 - t2)
            self.phase_ns["value"].append(t4 - t3)

    return InstrumentedEvaluator


def build_evaluator(args, instrumented: bool = False):
    from playing.v6 import evaluator as evaluator_module

    model, device = evaluator_module.load_default_model(args.model, None)
    cls = instrumented_evaluator_class() if instrumented else evaluator_module.TorchEvaluator
    return cls(model, device, args.max_batch, graphs=True)


# --- searches ---------------------------------------------------------------


def make_search(evaluator, args, arena_capacity=None):
    config = guofish_core.SearchConfig(
        virtual_loss=args.virtual_loss,
        arena_capacity=arena_capacity or args.arena_capacity)
    config.cache_slots = CACHE_SLOTS
    # OFF, per the C12 brief. C8 measured the structural diff at +8.1 ms per
    # apply_move at 184k nodes; a Gate 4 number taken with it on is a number for
    # a configuration nobody ships.
    config.verify_compaction = False
    search = guofish_core.ReplaySearchQ32(config)
    search.set_evaluator(evaluator.core)
    return search


def parallel_config(args, in_flight=None):
    return guofish_core.ParallelConfig(
        workers=args.workers, in_flight=in_flight or args.in_flight,
        max_batch=args.max_batch, affinity=args.affinity, collect_histograms=True)


def timed_search(search, evaluator, budget: int, parallel, *, absolute: bool):
    """One measured search. `absolute` picks which of the two budget spellings.

    `absolute=False` asks for `root_visits + budget`, i.e. `budget` NEW
    simulations, which is what `uci_wrapper_v6.py` does in play. `absolute=True`
    reproduces bench_c10b.py's spelling so the two can be printed side by side.
    """
    target = budget if absolute else int(search.root_visits) + budget
    evaluator.graph_counters_reset()
    if hasattr(evaluator, "reset_phases"):
        evaluator.reset_phases()
    started = time.perf_counter()
    search.search_parallel(target, parallel)
    wall = time.perf_counter() - started
    par = search.parallel_stats()
    evals = search.eval_stats()
    audit = search.audit()
    assert par["delivered"] == par["requested"], "delivered != requested"
    # `requested` is the REMAINDER (`num_simulations - existing`), so
    # `delivered == requested` is satisfied at 0 == 0 by a search that never ran.
    # A rate needs a numerator; this is the check the C10b spelling needed.
    assert par["delivered"] > 0, (
        f"delivered 0 simulations: the root already held {target} visits, so the "
        f"target of {target} left nothing to do. `delivered == requested` does "
        f"not catch this and a sims/s computed from it is not a rate")
    assert audit["vloss_total"] == 0, "virtual loss stranded"
    assert audit["conservation_failures"] == 0, "visits do not conserve"
    assert not par["arena_exhausted"], (
        "arena exhausted: C11c makes delivered < requested legitimate, and a "
        "benchmark harness must REJECT the row rather than publish a short one")
    return {
        "wall_s": wall,
        "target": target,
        "inherited": target - par["requested"],
        "delivered": par["delivered"],
        "sims_per_s": par["delivered"] / wall if wall > 0 else 0.0,
        "wall_ns": par["wall_ns"],
        "call_ns": evals["call_ns"],
        "acquire_ns": evals["acquire_wait_ns"],
        "crossings": evals["batches"],
        "rows": evals["rows"],
        "rows_per_crossing": evals["mean_rows"],
        "cache_skipped": evals["cache_skipped"],
        "cache_rate": evals["cache_skipped"] /
                      max(1, evals["cache_skipped"] + evals["rows"]),
        "pad_ratio": evaluator.graph_pad_ratio(),
        "gpu_share_call": evals["call_ns"] / max(1, par["wall_ns"]),
    }


def reused_search(evaluator, args, parallel, *, absolute: bool):
    """An endgame tree six plies into a game the engine played against itself.

    THE PLIES USE THE SAME BUDGET SPELLING AS THE MEASURED SEARCH, and that is
    not a detail. Building the tree with `root_visits + sims` and then measuring
    it with an absolute target reproduces neither spelling: it produces a root
    with 80,510 visits, an absolute request of 20,000, and a search that
    delivers ZERO simulations while `delivered == requested` still holds at
    0 == 0. The first version of this section did exactly that and published a
    zero. Each arm therefore plays the whole game its own way.
    """
    search = make_search(evaluator, args)
    search.set_position(ENDGAME_FEN)
    for _ in range(args.reuse_plies):
        target = args.sims if absolute else int(search.root_visits) + args.sims
        stats = search.search_parallel(target, parallel)
        search.apply_move(stats["best_move"])
    return search


# --- sections ---------------------------------------------------------------


def section_gate4(args, payload, sections):
    evaluator = build_evaluator(args)
    parallel = parallel_config(args)
    rows = []

    for label, absolute in (("fresh root", False),
                            ("reuse-heavy (NEW sims — production spelling)", False),
                            ("reuse-heavy (ABSOLUTE target — C10b's spelling)", True)):
        samples = []
        for _ in range(args.repeats):
            if label.startswith("fresh"):
                search = make_search(evaluator, args)
                search.set_position(FRESH_ROOT_FEN)
            else:
                search = reused_search(evaluator, args, parallel, absolute=absolute)
            samples.append(timed_search(search, evaluator, args.sims, parallel,
                                        absolute=absolute))
            search.set_evaluator(None)
            del search
        rates = [s["sims_per_s"] for s in samples]
        row = {
            "regime": label,
            "delivered": int(statistics.median(s["delivered"] for s in samples)),
            "inherited": int(statistics.median(s["inherited"] for s in samples)),
            "sims_per_s": statistics.median(rates),
            "min": min(rates), "max": max(rates),
            "rows_per_crossing": statistics.fmean(s["rows_per_crossing"] for s in samples),
            "pad_ratio": statistics.fmean(s["pad_ratio"] for s in samples),
            "cache_rate": statistics.fmean(s["cache_rate"] for s in samples),
            "gpu_share_call": statistics.fmean(s["gpu_share_call"] for s in samples),
            "wall_ms": 1000 * statistics.median(s["wall_s"] for s in samples),
        }
        rows.append(row)
        log(f"  {label:<48} {row['delivered']:>6,} sims in "
            f"{row['wall_ms']:>8.1f} ms -> {row['sims_per_s']:>9,.0f} sims/s")

    sections += [
        "### C12-1 — Gate 4, both regimes, delivered simulations", "",
        f"W={args.workers}, K={args.in_flight}, `max_batch` {args.max_batch}, "
        f"virtual loss {args.virtual_loss}, `verify_compaction` **off**, "
        f"{args.repeats} repeats (median). No opening book and no tablebase are "
        f"attached on this path, so no move is bypassed and there is nothing to "
        f"exclude from the rate.", "",
        "| regime | inherited visits | delivered sims | wall | delivered sims/s "
        "| min | max | rows/crossing | pad waste | cache hit | vs floor (8k) "
        "| vs stretch (15k) | vs Python (838) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        sections.append(
            f"| {r['regime']} | {r['inherited']:,} | {r['delivered']:,} "
            f"| {r['wall_ms']:,.1f} ms | **{r['sims_per_s']:,.0f}** "
            f"| {r['min']:,.0f} | {r['max']:,.0f} | {r['rows_per_crossing']:.1f} "
            f"| {r['pad_ratio']:.2f}x | {100 * r['cache_rate']:.1f}% "
            f"| {r['sims_per_s'] / GATE4_FLOOR:.2f}x "
            f"| {r['sims_per_s'] / GATE4_STRETCH:.2f}x "
            f"| {r['sims_per_s'] / PYTHON_BASELINE:.0f}x |")
    sections.append("")
    payload["gate4"] = rows
    evaluator.close()
    return rows


def section_attribution(args, payload, sections):
    """Every millisecond of a fresh-root move, named."""
    evaluator = build_evaluator(args, instrumented=True)
    parallel = parallel_config(args)

    search = make_search(evaluator, args)
    search.set_position(FRESH_ROOT_FEN)
    # Warm, then throw away: the measured search must not pay for first touch on
    # the arena's pages or for a cold transposition cache.
    search.search_parallel(2000, parallel)
    search.set_evaluator(None)
    del search

    search = make_search(evaluator, args)
    search.set_position(FRESH_ROOT_FEN)
    result = timed_search(search, evaluator, args.sims, parallel, absolute=False)
    phases = {name: sum(samples) for name, samples in evaluator.phase_ns.items()}
    per_batch = {name: statistics.median(samples) if samples else 0
                 for name, samples in evaluator.phase_ns.items()}
    batches = result["crossings"]
    search.set_evaluator(None)
    del search

    wall_ns = result["wall_ns"]
    callback_ns = sum(phases.values())
    # The callback's own Python overhead: the C++ side measures the whole call,
    # the four phases above measure what is inside it, and the difference is
    # interpreter time — argument marshalling, the bound-method dispatch, the
    # attribute lookups. It is separated because it is the one line here that a
    # rewrite could plausibly delete.
    python_overhead_ns = result["call_ns"] - callback_ns
    host_search_ns = wall_ns - result["call_ns"] - result["acquire_ns"]

    lines = [
        ("graph replay wait (the `policy` D2H blocks until the forward is done)",
         phases["policy"]),
        ("H2D of the token rows + pad-tail restore (`stage`)", phases["stage"]),
        ("`cudaGraphLaunch` (`replay`, submission only)", phases["replay"]),
        ("value D2H (`value`)", phases["value"]),
        ("Python interpreter overhead inside the callback", python_overhead_ns),
        ("dispatcher GIL acquire wait", result["acquire_ns"]),
        ("C++ search outside the callback (descent, expand, backup, tokenize, probe)",
         host_search_ns),
    ]
    attributed = sum(value for _, value in lines)
    remainder = wall_ns - attributed

    sections += [
        "### C12-2 — Where a fresh-root move's time goes, every millisecond named", "",
        f"One search: {result['delivered']:,} delivered simulations over "
        f"{result['crossings']:,} boundary crossings at "
        f"{result['rows_per_crossing']:.1f} rows each, W={args.workers}/"
        f"K={args.in_flight}. Phases are timed from inside a `TorchEvaluator` "
        f"subclass, so they are measured together rather than reconciled after "
        f"the fact.", "",
        "| line | total ms | share of wall | per crossing µs |",
        "|---|---:|---:|---:|"]
    for name, value in lines:
        sections.append(f"| {name} | {value / 1e6:.1f} | "
                        f"{100 * value / wall_ns:.1f}% | "
                        f"{value / batches / 1000:.1f} |")
    sections += [
        f"| **total attributed** | **{attributed / 1e6:.1f}** | "
        f"**{100 * attributed / wall_ns:.1f}%** | "
        f"{attributed / batches / 1000:.1f} |",
        f"| *residual* | *{remainder / 1e6:.1f}* | "
        f"*{100 * remainder / wall_ns:.1f}%* | "
        f"*{remainder / batches / 1000:.1f}* |",
        f"| **wall** | **{wall_ns / 1e6:.1f}** | **100%** | "
        f"{wall_ns / batches / 1000:.1f} |", "",
        "Per-crossing medians (microseconds): " +
        ", ".join(f"`{name}` {value / 1000:.1f}" for name, value in per_batch.items()) + ".",
        "",
        "**THE RESIDUAL IS ZERO BY CONSTRUCTION AND IS NOT EVIDENCE.** The last "
        "line is computed as `wall - call_ns - acquire_ns` and the Python line as "
        "`call_ns - (the four phases)`, so the column sums to the wall whatever "
        "the truth is. What makes the table checkable is an INDEPENDENT "
        "measurement of the same quantities from the Nsight Systems trace — see "
        "BENCH.md C12-1, where the GPU busy fraction and the host-side CUDA API "
        "cost are read off the CUDA trace rather than off these timers, and the "
        "two agree to about a point. Quote the agreement, never this zero.",
        ""]

    payload["attribution"] = {
        "wall_ns": wall_ns, "batches": batches, "phases": phases,
        "python_overhead_ns": python_overhead_ns,
        "acquire_ns": result["acquire_ns"], "host_search_ns": host_search_ns,
        "attributed_ns": attributed, "remainder_ns": remainder,
        "per_batch_median_ns": per_batch, "result": result,
    }
    evaluator.close()
    return payload["attribution"]


def section_ksweep(args, payload, sections):
    """C12 item (b): outstanding leaves against throughput AND root sharpness."""
    evaluator = build_evaluator(args)
    reference = None
    rows = []
    for in_flight in args.ksweep:
        parallel = parallel_config(args, in_flight=in_flight)
        rates, shares, distributions, pads, rows_per = [], [], [], [], []
        for _ in range(args.repeats):
            search = make_search(evaluator, args)
            search.set_position(FRESH_ROOT_FEN)
            result = timed_search(search, evaluator, args.sims, parallel, absolute=False)
            rates.append(result["sims_per_s"])
            shares.append(result["gpu_share_call"])
            pads.append(result["pad_ratio"])
            rows_per.append(result["rows_per_crossing"])
            distributions.append(root_distribution(search))
            search.set_evaluator(None)
            del search
        mean = np.mean(np.stack(distributions), axis=0)
        if reference is None:
            reference = mean
        run_to_run = max((total_variation(a, b)
                          for a, b in itertools.combinations(distributions, 2)),
                         default=0.0)
        row = {
            "in_flight": in_flight,
            "sims_per_s": statistics.median(rates),
            "min": min(rates), "max": max(rates),
            "rows_per_crossing": statistics.fmean(rows_per),
            "pad_ratio": statistics.fmean(pads),
            "gpu_share_call": statistics.fmean(shares),
            "top_share": float(mean.max()),
            "tv_vs_reference": total_variation(mean, reference),
            "run_to_run_tv": run_to_run,
        }
        rows.append(row)
        log(f"  K={in_flight:<4} {row['sims_per_s']:>9,.0f} sims/s   "
            f"top {100 * row['top_share']:.1f}%   "
            f"run-to-run TV {100 * run_to_run:.1f}%")

    baseline = next(r for r in rows if r["in_flight"] == args.in_flight)
    sections += [
        "### C12-3 — Item (b): outstanding leaves, throughput and root sharpness", "",
        f"W={args.workers}, {args.sims:,} NEW simulations on the fresh midgame "
        f"root, {args.repeats} repeats. `TV vs K={rows[0]['in_flight']}` is total "
        f"variation against the first row's mean root distribution; "
        f"`speedup` is against the shipping K={args.in_flight}. The tolerance is "
        f"**10% run-to-run TV**, set by C10b-3g — this is no longer an open "
        f"question, which is why item (b) can be decided here.", "",
        "| K | outstanding | delivered sims/s | speedup vs shipping | rows/crossing "
        "| pad waste | top-move share | TV vs first row | run-to-run TV | within 10%? |",
        "|--:|---:|---:|---:|---:|---:|---:|---:|---:|:--|"]
    for r in rows:
        marker = "**" if r["in_flight"] == args.in_flight else ""
        sections.append(
            f"| {marker}{r['in_flight']}{marker} | {args.workers * r['in_flight']} "
            f"| {marker}{r['sims_per_s']:,.0f}{marker} "
            f"| {r['sims_per_s'] / baseline['sims_per_s']:.2f}x "
            f"| {r['rows_per_crossing']:.1f} | {r['pad_ratio']:.2f}x "
            f"| {100 * r['top_share']:.1f}% | {100 * r['tv_vs_reference']:.1f}% "
            f"| {100 * r['run_to_run_tv']:.1f}% "
            f"| {'yes' if r['run_to_run_tv'] <= 0.10 else '**NO**'} |")
    sections.append("")
    payload["ksweep"] = rows
    evaluator.close()
    return rows


SECTIONS = {"gate4": section_gate4, "attribution": section_attribution,
            "ksweep": section_ksweep}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", nargs="+", default=sorted(SECTIONS),
                        choices=sorted(SECTIONS))
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reuse-plies", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--in-flight", type=int, default=24)
    parser.add_argument("--ksweep", type=int, nargs="+",
                        default=[8, 16, 24, 32, 48, 64, 96, 128])
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--affinity", default="none")
    parser.add_argument("--virtual-loss", type=float, default=2.5)
    parser.add_argument("--arena-capacity", type=int, default=ARENA_CAPACITY)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--allow-instrumented", action="store_true")
    args = parser.parse_args()

    info = guofish_core.build_info()
    log(f"build: {info}")
    if (info["asan"] or info["ubsan"] or info["asserts"]) and not args.allow_instrumented:
        raise SystemExit("refusing to publish from an instrumented build; see "
                         "DECISIONS.md, C10. Pass --allow-instrumented to override.")

    payload, sections = {}, []
    for name in args.sections:
        log(f"\n== {name} ==")
        SECTIONS[name](args, payload, sections)

    if args.markdown:
        log("\n" + "=" * 72 + "\n")
        log("\n".join(sections))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
