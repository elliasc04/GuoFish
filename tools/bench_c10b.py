#!/usr/bin/env python
"""C10b Stage 3 — the decision surface, re-measured live and graphed.

    python tools/bench_c10b.py --markdown
    python tools/bench_c10b.py --sections grid --markdown
    python tools/bench_c10b.py --sections gate4 padding histogram --markdown

WHY EVERY PRIOR ABSOLUTE FIGURE IS VOID HERE
============================================
C9b's W x K grid ran on the REPLAY evaluator with a stand-in filling 24-37% of
expansions, so its `TV vs serial` column is an upper bound and its `run-to-run
TV` column tracks the stand-in's share rather than the engine's own variation.
C9a's throughput ran on an UNGRAPHED forward, i.e. on ~4 ms of host dispatch per
batch. Both confounds are gone: the real evaluator answers every leaf and the
forward is one graph launch. So the grid is re-run rather than adjusted, and
`stand-in%` is not a column any more — `run-to-run TV` is now the engine's own
number and closes C9's deferred absolute root-stability tolerance.

THE SECTIONS, AND WHAT EACH ONE DECIDES
=======================================
  grid      W x K on the live evaluator: sims/s, mean batch, TV vs serial,
            run-to-run TV, top-move share. Fixes W x K on throughput and
            sharpness, which is the whole of C10b Stage 3.2.
  control   W=1 at each cell's outstanding count. C9c's control, re-run live:
            it is what separates virtual-loss exposure from a concurrency tax
            of the parallelism model's own, and the finding it produced is the
            reason `max_outstanding` is the governing knob.
  gate4     delivered sims/s at the chosen configuration, on a fresh midgame
            root AND in the reuse-heavy regime a real game spends most of its
            moves in. Both, because C10h measured 3.2 rows per crossing under
            deep reuse and a fresh-root-only number would flatter the engine.
  padding   crossings per move, rows per crossing, and the shape histogram —
            so the cost of padding a 3-row batch up to a captured size is
            visible rather than assumed small.
  histogram C10d re-run: the dispatcher's GIL acquire wait against the 200 us /
            1%-of-wall trigger, with the adversarial `info` formatter running.
            The prediction under test is that graph capture shrank the held
            window from ~4 ms of kernel launching to a copy plus one launch.

Writes nothing to golden/. This is a benchmark, not a reference implementation
(Global Rule 2). Paste the tables into BENCH.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

QUIET_MANIFEST = REPO_ROOT / "golden" / "gate1_manifest.json"

# The C10 bench position and the C10 smoke position, so the throughput tables
# here and the acceptance runs there are talking about the same searches.
FRESH_ROOT_FEN = "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9"
ENDGAME_FEN = "8/2R5/4R2p/5pp1/2N1k3/6Pb/r4r1P/6K1 b - - 11 46"

CACHE_SLOTS = 400_000
# Sized from the budget rather than from optimism: an expansion adds ~30
# children, so 20,000 simulations reach ~600k nodes and this is 2x that. It is
# not free — construction costs 22 ms against 13 ms at 200k and 44 ms at 4M, and
# the grid constructs a few hundred searches — so it is a measured default and
# `--arena-capacity` overrides it.
ARENA_CAPACITY = 1_200_000

# The C10 brief's contingency thresholds, restated once.
TRIGGER_P99_US = 200.0
TRIGGER_WALL_SHARE = 0.01

# Gate 4 (chunk table, C12): >= 8,000 delivered sims/s, stretch 15,000,
# against the Python baseline of 838.
GATE4_FLOOR = 8_000
GATE4_STRETCH = 15_000
PYTHON_BASELINE = 838


def log(msg: str = "") -> None:
    print(msg, flush=True)


# --- corpus -----------------------------------------------------------------


def quiet_positions(limit: int, virtual_loss: float):
    """The C5/C9 quiet corpus, by FEN only.

    The same positions C9b's grid ran on, so the live grid can be read against
    it position for position. Only the FEN and the pre-root history are taken —
    the recorded trees belong to the replay evaluator and have no place here.
    """
    manifest = json.loads(QUIET_MANIFEST.read_text(encoding="utf-8"))
    runs = [r for r in manifest["runs"]
            if r["virtual_loss"] == virtual_loss and not r.get("full_tree")]
    return [(r["fen"], r.get("history", [])) for r in runs[:limit]]


def affinity_for(workers: int) -> str:
    """C9d's pinning policy, made a function of W rather than a constant.

    C9d measured one thread per P-core worth +14.9% at W=4 and +24.5% at W=6:
    a worker parked on an E-core holds the root's contended atomics longer, and
    the root is on every descent path. At W=1 there is no contention to
    mitigate and pinning measures -9% (BENCH.md C10b-3f) — the single worker
    and the dispatcher end up competing for one core instead of being placed by
    the scheduler. So `none` at W=1 is a measurement, not an omission.
    """
    return "none" if workers == 1 else "pcore_physical"


def make_search(virtual_loss: float, evaluator, arena_capacity=ARENA_CAPACITY):
    config = guofish_core.SearchConfig(virtual_loss=virtual_loss,
                                       arena_capacity=arena_capacity)
    config.cache_slots = CACHE_SLOTS
    search = guofish_core.ReplaySearchQ32(config)
    search.set_evaluator(evaluator.core)
    return search


# --- distributions ----------------------------------------------------------


def root_distribution(search):
    """Root children's visit shares, in canonical move order. C9's function.

    Canonical order rather than sorted-by-visits, so two runs are compared move
    against move; sorting first would report two different move sets as
    identical whenever they happened to have the same shape.
    """
    arrays = search.dump_tree_arrays(0)
    at_depth_one = arrays["depth"] == 1
    visits = arrays["visits"][at_depth_one].astype(np.float64)
    moves = arrays["move"][at_depth_one]
    order = np.argsort(moves)
    visits = visits[order]
    total = visits.sum()
    return (visits / total) if total > 0 else visits


def total_variation(p, q):
    n = min(len(p), len(q))
    return 0.5 * float(np.abs(p[:n] - q[:n]).sum())


# --- one cell ---------------------------------------------------------------


def run_cell(evaluator, positions, workers, in_flight, *, sims, repeats, max_batch,
             affinity, virtual_loss, reference=None):
    """One W x K cell: `repeats` searches of every position, all metrics.

    A cell whose numbers are not conserved is not a measurement — delivered must
    equal requested, no virtual loss may be stranded and visits must conserve —
    and that is asserted before anything is returned, exactly as C9's grid did.
    """
    per_position = []
    sims_per_s, collisions, mean_batches, largest = [], [], [], []
    rows_per_crossing, pad_ratio, cache_rate, crossings = [], [], [], []
    gpu_share = []
    for fen, history in positions:
        distributions = []
        for _ in range(repeats):
            search = make_search(virtual_loss, evaluator)
            search.set_position(fen, history)
            evaluator.graph_counters_reset()
            stats = search.search_parallel(
                sims, guofish_core.ParallelConfig(workers=workers, in_flight=in_flight,
                                                  max_batch=max_batch, affinity=affinity,
                                                  collect_histograms=True))
            par = search.parallel_stats()
            audit = search.audit()
            assert par["delivered"] == par["requested"], "delivered != requested"
            assert audit["vloss_total"] == 0, "virtual loss stranded"
            assert audit["conservation_failures"] == 0, "visits do not conserve"
            assert stats["synthetic_evaluations"] == 0, (
                "the stand-in evaluator answered a leaf on the LIVE path; that is the "
                "confound C10b exists to remove and its presence invalidates the cell")

            evals = search.eval_stats()
            distributions.append(root_distribution(search))
            sims_per_s.append(par["sims_per_s"])
            collisions.append(par["select_collisions"] / max(1, par["delivered"]))
            mean_batches.append(par["mean_batch"])
            largest.append(par["largest_batch"])
            crossings.append(evals["batches"])
            rows_per_crossing.append(evals["mean_rows"])
            cache_rate.append(evals["cache_skipped"] /
                              max(1, evals["cache_skipped"] + evals["rows"]))
            pad_ratio.append(evaluator.graph_pad_ratio())
            # THE NUMBER STAGE 2 IS DECIDED ON. The callback is the whole of the
            # GPU wait — H2D, replay, D2H — and the dispatcher is blocked inside
            # it, so `call_ns / wall_ns` is the fraction of the cycle already
            # spent on the device. Pipelining can hide at most the REMAINDER, so
            # its throughput ceiling is 1/gpu_share and this measures it rather
            # than modelling it.
            gpu_share.append(evals["call_ns"] / max(1, par["wall_ns"]))
            search.set_evaluator(None)
        per_position.append(distributions)

    run_to_run = [max((total_variation(a, b)
                       for a, b in itertools.combinations(d, 2)), default=0.0)
                  for d in per_position]
    means = [np.mean(np.stack(d), axis=0) for d in per_position]
    row = {
        "workers": workers,
        "in_flight": in_flight,
        "outstanding": workers * in_flight,
        "sims_per_s": statistics.fmean(sims_per_s),
        "mean_batch": statistics.fmean(mean_batches),
        "largest_batch": max(largest),
        "collisions_per_sim": statistics.fmean(collisions),
        "crossings": statistics.fmean(crossings),
        "rows_per_crossing": statistics.fmean(rows_per_crossing),
        "cache_rate": statistics.fmean(cache_rate),
        "pad_ratio": statistics.fmean(pad_ratio),
        "gpu_share": statistics.fmean(gpu_share),
        "run_to_run_tv": statistics.fmean(run_to_run),
        "run_to_run_tv_worst": max(run_to_run) if run_to_run else 0.0,
        "top_share": statistics.fmean(float(m.max()) for m in means),
        "distributions": means,
    }
    row["tv_vs_serial"] = (statistics.fmean(total_variation(a, b)
                                            for a, b in zip(means, reference))
                           if reference is not None else 0.0)
    return row


def format_grid(rows, title, note=""):
    out = [f"### {title}", ""]
    if note:
        out += [note, ""]
    out.append("| W | K | outstanding | sims/s | mean batch | rows/crossing | pad waste "
               "| GPU share | pipeline ceiling | collisions/sim | TV vs serial "
               "| run-to-run TV (mean / worst) | top share |")
    out.append("|--:|--:|------------:|-------:|-----------:|--------------:|----------:"
               "|----------:|-----------------:|---------------:|-------------:"
               "|----------------------------:|----------:|")
    for r in rows:
        out.append(
            f"| {r['workers']} | {r['in_flight']} | {r['outstanding']} "
            f"| {r['sims_per_s']:,.0f} | {r['mean_batch']:.1f} "
            f"| {r['rows_per_crossing']:.1f} | {r['pad_ratio']:.2f}x "
            f"| {100 * r['gpu_share']:.0f}% | {1 / max(r['gpu_share'], 1e-9):.2f}x "
            f"| {r['collisions_per_sim']:.4f} "
            f"| {100 * r['tv_vs_serial']:.1f}% "
            f"| {100 * r['run_to_run_tv']:.1f}% / {100 * r['run_to_run_tv_worst']:.1f}% "
            f"| {100 * r['top_share']:.1f}% |")
    return out + [""]


# --- sections ---------------------------------------------------------------


def section_grid(evaluator, args, payload, sections):
    positions = quiet_positions(args.positions, args.virtual_loss)
    log(f"grid: {len(positions)} quiet positions x {args.repeats} repeats, "
        f"{args.sims} sims, VL {args.virtual_loss}, max_batch {args.max_batch}")

    log("  reference row: W=1 K=1 (one leaf in flight, deterministic descent)")
    serial = run_cell(evaluator, positions, 1, 1, sims=args.sims, repeats=args.serial_repeats,
                      max_batch=args.max_batch, affinity="none",
                      virtual_loss=args.virtual_loss)
    reference = serial["distributions"]
    serial["tv_vs_serial"] = 0.0
    rows = [serial]
    for in_flight in args.in_flight:
        for workers in args.workers:
            log(f"  W={workers} K={in_flight}")
            rows.append(run_cell(evaluator, positions, workers, in_flight, sims=args.sims,
                                 repeats=args.repeats, max_batch=args.max_batch,
                                 affinity=affinity_for(workers),
                                 virtual_loss=args.virtual_loss,
                                 reference=reference))
    rows.sort(key=lambda r: (r["in_flight"], r["workers"]))
    sections += format_grid(
        rows, "C10b-3b — W x K on the live graphed evaluator",
        "`stand-in%` is gone: every leaf was answered by the network (asserted per cell), "
        "so `run-to-run TV` is the engine's own reproducibility rather than an upper bound.")
    payload["grid"] = [{k: v for k, v in r.items() if k != "distributions"} for r in rows]

    if args.control:
        control = [serial]
        for outstanding in sorted({w * k for w in args.workers for k in args.in_flight}):
            if outstanding > args.max_batch:
                continue
            log(f"  control W=1 K={outstanding}")
            control.append(run_cell(evaluator, positions, 1, outstanding, sims=args.sims,
                                    repeats=2, max_batch=args.max_batch, affinity="none",
                                    virtual_loss=args.virtual_loss, reference=reference))
        sections += format_grid(
            control, "C10b-3c — control: W=1 at the same outstanding-leaf counts",
            "Same virtual-loss exposure, no concurrency. C9c found the concurrent rows "
            "consistently LESS flattened than these; this is that finding re-run with the "
            "stand-in removed.")
        payload["control"] = [{k: v for k, v in r.items() if k != "distributions"}
                              for r in control]
    return rows


def section_gate4(evaluator, args, payload, sections):
    """Delivered sims/s at the chosen config, fresh root and reuse-heavy.

    The reuse-heavy regime is produced the way a game produces it: search, play
    the engine's own move, search again, for `--reuse-plies` plies. That leaves
    the tree, the arena and the transposition cache in the state C8 and C10h
    measured, which is where 92% of leaves never reach the network.
    """
    log(f"gate4: {args.gate4_configs}, max_batch={args.max_batch}, "
        f"{args.gate4_sims} sims x {args.gate4_repeats}")
    out = []
    for spec in args.gate4_configs:
        workers, in_flight = (int(part) for part in spec.lower().split("x"))
        parallel = guofish_core.ParallelConfig(
            workers=workers, in_flight=in_flight, max_batch=args.max_batch,
            affinity=affinity_for(workers), collect_histograms=True)
        for label, fen, plies in (("fresh root", FRESH_ROOT_FEN, 0),
                                  ("reuse-heavy", ENDGAME_FEN, args.reuse_plies)):
            rates, rows_per, pads, caches, shares = [], [], [], [], []
            for _ in range(args.gate4_repeats):
                search = make_search(args.virtual_loss, evaluator)
                search.set_position(fen)
                # The reuse regime is produced the way a game produces it:
                # search, play the engine's own move, search again. That leaves
                # the tree, the arena and the transposition cache in the state
                # C8 and C10h measured, where most leaves never reach the
                # network.
                for _ in range(plies):
                    stats = search.search_parallel(args.gate4_sims, parallel)
                    search.apply_move(stats["best_move"])
                evaluator.graph_counters_reset()
                started = time.perf_counter()
                search.search_parallel(args.gate4_sims, parallel)
                wall = time.perf_counter() - started
                par = search.parallel_stats()
                evals = search.eval_stats()
                assert par["delivered"] == par["requested"]
                rates.append(par["delivered"] / wall)
                rows_per.append(evals["mean_rows"])
                pads.append(evaluator.graph_pad_ratio())
                caches.append(evals["cache_skipped"] /
                              max(1, evals["cache_skipped"] + evals["rows"]))
                shares.append(evals["call_ns"] / max(1, par["wall_ns"]))
                search.set_evaluator(None)
            row = {
                "config": f"W={workers}, K={in_flight}",
                "workers": workers,
                "in_flight": in_flight,
                "regime": label,
                "sims_per_s": statistics.median(rates),
                "sims_per_s_min": min(rates),
                "sims_per_s_max": max(rates),
                "rows_per_crossing": statistics.fmean(rows_per),
                "pad_ratio": statistics.fmean(pads),
                "cache_rate": statistics.fmean(caches),
                "gpu_share": statistics.fmean(shares),
            }
            out.append(row)
            log(f"  {row['config']:<12} {label:<12} {row['sims_per_s']:>9,.0f} "
                f"delivered sims/s  (GPU {100 * row['gpu_share']:.0f}% of wall)")

    sections += ["### C10b-3e — Gate 4 margin, both regimes", "",
                 f"`max_batch`={args.max_batch}, {args.gate4_sims} simulations, "
                 f"{args.gate4_repeats} repeats (median). **Delivered** simulations, "
                 f"asserted equal to the budget. The reuse-heavy rows played "
                 f"{args.reuse_plies} plies of the engine's own moves first, so the tree, "
                 f"the arena and the cache are in the state a real game leaves them in.", "",
                 "| config | regime | delivered sims/s | min | max | rows/crossing "
                 "| pad waste | cache hit | GPU share | vs floor (8k) | vs stretch (15k) "
                 "| vs Python (838) |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in out:
        sections.append(
            f"| {r['config']} | {r['regime']} | {r['sims_per_s']:,.0f} "
            f"| {r['sims_per_s_min']:,.0f} | {r['sims_per_s_max']:,.0f} "
            f"| {r['rows_per_crossing']:.1f} | {r['pad_ratio']:.2f}x "
            f"| {100 * r['cache_rate']:.1f}% | {100 * r['gpu_share']:.0f}% "
            f"| {r['sims_per_s'] / GATE4_FLOOR:.2f}x "
            f"| {r['sims_per_s'] / GATE4_STRETCH:.2f}x "
            f"| {r['sims_per_s'] / PYTHON_BASELINE:.0f}x |")
    sections.append("")
    payload["gate4"] = out
    return out


def section_padding(evaluator, args, payload, sections):
    """The shape histogram in both regimes — where padding actually lands."""
    parallel = guofish_core.ParallelConfig(workers=args.gate4_workers,
                                           in_flight=args.gate4_in_flight,
                                           max_batch=args.max_batch,
                                           affinity=affinity_for(args.gate4_workers))
    out = []
    # Three regimes, because the reuse axis has two distinct ends and only the
    # middle one is representative. The endgame at a FRESH root is the C10h
    # measurement (3.2 rows per crossing, 92% cache-answered) and is where
    # padding is most visible; the deeply reused one is where the network is
    # barely used at all and the histogram has too few crossings to read.
    for label, fen, plies, sims in (
            ("fresh midgame root", FRESH_ROOT_FEN, 0, args.gate4_sims),
            ("endgame root (the C10h regime)", ENDGAME_FEN, 0, args.smoke_sims),
            ("endgame after "
             f"{args.reuse_plies} plies of reuse", ENDGAME_FEN, args.reuse_plies,
             args.gate4_sims)):
        search = make_search(args.virtual_loss, evaluator)
        search.set_position(fen)
        for _ in range(plies):
            stats = search.search_parallel(sims, parallel)
            search.apply_move(stats["best_move"])
        evaluator.graph_counters_reset()
        search.search_parallel(sims, parallel)
        evals = search.eval_stats()
        graph = evaluator.graph
        out.append({
            "regime": label,
            "crossings": int(evals["batches"]),
            "rows": int(evals["rows"]),
            "rows_per_crossing": evals["mean_rows"],
            "cache_skipped": int(evals["cache_skipped"]),
            "padded_rows": graph.padded_rows,
            "pad_ratio": evaluator.graph_pad_ratio(),
            "by_size": dict(graph.by_size),
        })
        search.set_evaluator(None)
        log(f"  {label:<34} {out[-1]['crossings']:>5} crossings, "
            f"{out[-1]['rows_per_crossing']:.2f} rows/crossing, "
            f"{out[-1]['pad_ratio']:.2f}x padded")

    sizes = evaluator.graph_sizes
    sections += ["### C10b-3d — crossings, rows per crossing, and what padding costs", "",
                 f"One move at W={args.gate4_workers}/K={args.gate4_in_flight}. "
                 f"`pad waste` is padded rows divided by real rows — the GPU work spent "
                 f"on rows nobody asked about.", "",
                 "| regime | crossings | rows | rows/crossing | cache-answered | pad waste | "
                 + " | ".join(f"shape {s}" for s in sizes) + " |",
                 "|---|---:|---:|---:|---:|---:|" + "".join("---:|" for _ in sizes)]
    for r in out:
        sections.append(
            f"| {r['regime']} | {r['crossings']:,} | {r['rows']:,} "
            f"| {r['rows_per_crossing']:.2f} | {r['cache_skipped']:,} "
            f"| {r['pad_ratio']:.2f}x | "
            + " | ".join(f"{r['by_size'].get(s, 0):,}" for s in sizes) + " |")
    sections.append("")
    payload["padding"] = out
    return out


def section_histogram(args, payload, sections, model, device):
    """C10d re-run: the acquire-wait histogram, with and without the competitor.

    A FRESH EVALUATOR PER CELL, because the switch interval is set in its
    constructor — that is where scope §2.1's "before any search thread" lives —
    and because capture must not be the first thing the timed run does.
    """
    from playing.v6 import evaluator as live_evaluator
    from tests.test_c0b_contention import UciFormatterLoad

    rows = []
    for graphs in ([True, False] if args.histogram_both else [True]):
        for competing in (False, True):
            evaluator = live_evaluator.TorchEvaluator(model, device, args.max_batch,
                                                      graphs=graphs)
            search = make_search(args.virtual_loss, evaluator)
            parallel = guofish_core.ParallelConfig(workers=args.gate4_workers,
                                                   in_flight=args.gate4_in_flight,
                                                   max_batch=args.max_batch,
                                                   affinity=affinity_for(args.gate4_workers),
                                                   collect_histograms=True)
            try:
                search.set_position(FRESH_ROOT_FEN)
                search.search_parallel(max(64, args.hist_sims // 8), parallel)
                search.set_position(FRESH_ROOT_FEN)
                if competing:
                    with UciFormatterLoad():
                        started = time.perf_counter()
                        search.search_parallel(args.hist_sims, parallel)
                        wall = time.perf_counter() - started
                else:
                    started = time.perf_counter()
                    search.search_parallel(args.hist_sims, parallel)
                    wall = time.perf_counter() - started
                evals = search.eval_stats()
                par = search.parallel_stats()
                rows.append({
                    "graphs": graphs,
                    "competing": competing,
                    "batches": int(evals["batches"]),
                    "rows_per_crossing": evals["mean_rows"],
                    "delivered_per_s": par["delivered"] / wall,
                    "acquire_us": percentiles(evals["acquire_wait_ns_samples"]),
                    "call_us": percentiles(evals["call_ns_samples"]),
                    "acquire_total_ms": evals["acquire_wait_ns"] / 1e6,
                    "acquire_wall_share": evals["acquire_wait_ns"] / (wall * 1e9),
                })
                log(f"  graphs={graphs} competing={competing}: "
                    f"{rows[-1]['delivered_per_s']:,.0f} sims/s, acquire p99 "
                    f"{rows[-1]['acquire_us']['p99']:.2f} us, call p50 "
                    f"{rows[-1]['call_us']['p50']:.0f} us")
            finally:
                search.set_evaluator(None)
                evaluator.close()

    sections += ["### C10b-4 — dispatcher GIL acquire wait, per batch (microseconds)", "",
                 "Sampled from OUTSIDE the `gil_scoped_acquire` scope, so it contains "
                 "waiting and nothing else. The competitor is `UciFormatterLoad` from "
                 "tests/test_c0b_contention.py, imported rather than reimplemented.", "",
                 "| forward | competing | batches | rows/batch | sims/s | p50 | p90 | p99 "
                 "| max | total (ms) | share of wall | C10 trigger |",
                 "|:--|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|"]
    for r in rows:
        a = r["acquire_us"]
        fired = (a["p99"] > TRIGGER_P99_US) or (r["acquire_wall_share"] > TRIGGER_WALL_SHARE)
        sections.append(
            f"| {'graphed' if r['graphs'] else 'eager'} "
            f"| {'yes' if r['competing'] else 'no'} | {a['n']} "
            f"| {r['rows_per_crossing']:.1f} | {r['delivered_per_s']:,.0f} "
            f"| {a['p50']:.2f} | {a['p90']:.2f} | {a['p99']:.2f} | {a['max']:.2f} "
            f"| {r['acquire_total_ms']:.2f} | {r['acquire_wall_share']:.4%} "
            f"| {'**FIRED**' if fired else 'clear'} |")
    sections += ["",
                 "### C10b-4b — callback duration, per batch (microseconds)", "",
                 "NOT a contention metric (C10e). It is here because it is where the "
                 "GIL-held window shows up: the callback IS the held window, and Stage 1 "
                 "predicted it would fall from ~4 ms of kernel launching to a copy plus "
                 "one launch.", "",
                 "| forward | competing | p50 | p90 | p99 | max |",
                 "|:--|:--|---:|---:|---:|---:|"]
    for r in rows:
        c = r["call_us"]
        sections.append(f"| {'graphed' if r['graphs'] else 'eager'} "
                        f"| {'yes' if r['competing'] else 'no'} | {c['p50']:.1f} "
                        f"| {c['p90']:.1f} | {c['p99']:.1f} | {c['max']:.1f} |")
    sections.append("")
    payload["histogram"] = rows
    return rows


def percentiles(samples_ns):
    """Nearest-rank, no interpolation: every number is a wait that occurred."""
    ordered = sorted(int(s) for s in samples_ns)
    if not ordered:
        return {"n": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}

    def at(q):
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))] / 1000.0

    return {"n": len(ordered), "p50": at(0.50), "p90": at(0.90), "p99": at(0.99),
            "max": ordered[-1] / 1000.0, "mean": statistics.fmean(ordered) / 1000.0}


# --- entry point ------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", nargs="+",
                        default=["grid", "padding", "gate4", "histogram"],
                        choices=["grid", "padding", "gate4", "histogram"])
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--serial-repeats", type=int, default=1)
    # WHICH CHECKPOINT THE TABLE DESCRIBES. Defaulted to None, i.e. the shipped
    # v5 student, so every figure already in BENCH.md reproduces unchanged.
    #
    # It is a flag at all because the C10b tables are architecture-specific and
    # were not labelled as such: the knee, the W x K selection and the capture
    # ladder were all measured on the 10.9M student, and none of them transfers
    # to the 25.6M legacy net (8 layers x 512 against 6 x 384) whose forward has
    # a different fixed-to-per-row cost ratio. A grid run against another
    # checkpoint has to be able to say so, which is why the resolved path is
    # logged with the platform and build lines rather than left implicit.
    parser.add_argument("--model", type=Path, default=None,
                        help="checkpoint to measure (default: the shipped v5 "
                             "student, which is what every C10b table in "
                             "BENCH.md was measured on)")
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--virtual-loss", type=float, default=2.5)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--in-flight", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--control", action="store_true", default=True)
    parser.add_argument("--no-control", dest="control", action="store_false")
    parser.add_argument("--gate4-configs", nargs="+",
                        default=["1x24", "1x32", "1x64", "4x8", "8x4", "8x16"],
                        help="WxK candidates, e.g. 1x32 4x8")
    parser.add_argument("--gate4-workers", type=int, default=1,
                        help="the config the padding section and the histogram run at")
    parser.add_argument("--gate4-in-flight", type=int, default=24)
    parser.add_argument("--gate4-sims", type=int, default=20000)
    parser.add_argument("--gate4-repeats", type=int, default=5)
    parser.add_argument("--reuse-plies", type=int, default=6)
    parser.add_argument("--smoke-sims", type=int, default=7000,
                        help="budget for the C10h endgame regime in the padding table")
    parser.add_argument("--hist-sims", type=int, default=20000)
    parser.add_argument("--histogram-both", action="store_true", default=True,
                        help="run the histogram on the eager callback too, for contrast")
    parser.add_argument("--no-histogram-both", dest="histogram_both", action="store_false")
    parser.add_argument("--graph-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--no-graphs", dest="graphs", action="store_false", default=True)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--allow-instrumented", action="store_true")
    args = parser.parse_args()

    import torch
    from playing.v6 import evaluator as live_evaluator

    if not torch.cuda.is_available():
        log("ERROR: this benchmark measures the CUDA path; refusing to publish numbers "
            "from a CPU run.")
        return 1

    build = guofish_core.build_info()
    # Refused rather than warned: C10 published one set of instrumented tables
    # before this check existed (DECISIONS.md, C10).
    if (build["asan"] or build["ubsan"]) and not args.allow_instrumented:
        log(f"ERROR: this is a sanitizer build ({build}). Delete the staged "
            f"guofish_core*.pyd, rebuild Release, and check build_info() again. "
            f"Pass --allow-instrumented to override.")
        return 1

    model, device = live_evaluator.load_default_model(args.model)
    # C11b. See tools/bench_provenance.py: every artifact records the resolved
    # book/Syzygy state, and "not applicable" is a resolved state. This harness
    # drives guofish_core directly, so no number below can contain a
    # zero-simulation bypassed move.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import bench_provenance
    feature_state = bench_provenance.not_applicable(
        "this harness drives guofish_core directly; no opening book or "
        "tablebase is ever opened and no move can bypass MCTS")

    log(f"platform : {platform.system()} {platform.machine()}  {platform.platform()}")
    log(f"build    : {build['compiler']}, asan={build['asan']} asserts={build['asserts']}")
    for line in bench_provenance.require_recorded_state(feature_state):
        log(f"bypass   : {line}")
    log(f"device   : {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    # Provenance, on the same footing as the platform and build lines: a W x K
    # table is only meaningful for the forward it was measured against, and the
    # C10b tables predate this line by being implicitly about one checkpoint.
    log(f"model    : {args.model or live_evaluator.DEFAULT_MODEL} "
        f"({type(model).__name__}, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    evaluator = live_evaluator.TorchEvaluator(model, device, args.max_batch,
                                              graphs=args.graphs,
                                              graph_sizes=args.graph_sizes)
    if evaluator.graph_report is not None:
        log(f"capture  : {evaluator.graph_report.describe()}")
    log(f"topology : {guofish_core.ReplaySearchQ32.topology()['source']}")
    log("")

    # Paths are stringified rather than left to `json.dumps(default=float)`,
    # which raises on a WindowsPath and takes the whole run's results with it
    # AFTER every cell has been measured. Only reachable once `--model` is
    # passed — the default is None — so it stayed latent until a harness
    # measured a second checkpoint.
    payload = {"args": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()} | {"json_out": None},
               "build": dict(build),
               "device": torch.cuda.get_device_name(0),
               "capture": (evaluator.graph_report.describe()
                           if evaluator.graph_report else "eager")}
    sections: list[str] = []
    try:
        if "grid" in args.sections:
            section_grid(evaluator, args, payload, sections)
        if "padding" in args.sections:
            section_padding(evaluator, args, payload, sections)
        if "gate4" in args.sections:
            section_gate4(evaluator, args, payload, sections)
    finally:
        evaluator.close()

    # The histogram builds its own evaluators — one per cell, and one of them is
    # deliberately ungraphed — so it runs after the shared one is gone.
    if "histogram" in args.sections:
        section_histogram(args, payload, sections, model, device)

    if args.markdown:
        log("")
        for line in sections:
            log(line)

    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=1, default=float),
                                 encoding="utf-8", newline="\n")
        log(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
