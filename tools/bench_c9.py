#!/usr/bin/env python
"""C9 — the W x K sweep, the root visit distribution, and the affinity question.

    python tools/bench_c9.py --markdown
    python tools/bench_c9.py --markdown --positions 8 --repeats 6
    python tools/bench_c9.py --affinity-only

WHAT THIS MEASURES AND WHY IT IS TWO AXES
-----------------------------------------
Throughput alone is the wrong criterion and the C9 brief says so: a configuration
10% faster and visibly flatter at the root is the wrong trade. Every in-flight
path holds virtual loss on 10-30 nodes, so raising W or K buys batch size by
spending search quality, and the spend is only visible in the ROOT VISIT
DISTRIBUTION.

So every cell reports both:

  sims/s           delivered simulations per second (delivered, never requested)
  TV vs serial     total variation distance between this cell's root visit
                   distribution and the W=1/K=1 row's, on the same position.
                   W=1/K=1 is this engine's own serial ground truth and the only
                   fully valid comparison; Python's serial 58/28/11/3 to
                   32-outstanding 37/29/23/11 is a useful external landmark for
                   the SIZE of the effect (21 points of TV) but was measured on a
                   different parallelism model.
  run-to-run TV    worst pairwise distance between repeats of the SAME cell.
                   Nonzero above W=1 by construction: scheduling decides where
                   virtual loss sits.

THE CONTROL ROW THAT MAKES THE GRID INTERPRETABLE
-------------------------------------------------
`--control` adds W=1 rows at K equal to each cell's W*K. Those have the same
number of outstanding leaves and therefore the same virtual-loss exposure, but no
concurrency at all — they are deterministic. Comparing W=4/K=8 against W=1/K=32
separates the two effects the grid otherwise conflates: how much of the root
flattening is virtual loss (present in both) and how much is a concurrency tax of
the new model's own (present only in the first).

THE STAND-IN EVALUATOR IS REPORTED IN EVERY ROW, AND IT MATTERS
---------------------------------------------------------------
C9 runs on the replay evaluator, whose dump holds exactly the positions the
SERIAL Python reference evaluated. A parallel search leaves that set almost
immediately — that is leaf parallelism working — so the holes are filled by
cpp/search.hpp's stand-in evaluator. `stand-in%` is the share of expansions it
answered. Rows above ~10% are measuring a search that disagrees with the network
on a large fraction of its leaves, and their root-distribution columns should be
read as an upper bound on the engine's own variation rather than as a
measurement of it. The full-budget version of that question is C10's, where the
real evaluator makes it answerable.

Writes nothing. Paste the tables into BENCH.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

QUIET_MANIFEST = REPO_ROOT / "golden" / "gate1_manifest.json"
QUIET_TREES = REPO_ROOT / "golden" / "gate1_trees.npz"
QUIET_DUMP = REPO_ROOT / "golden" / "gate1_dump.npz"


def log(msg: str = "") -> None:
    print(msg, flush=True)


# --- corpus -----------------------------------------------------------------


def load_corpus():
    for path in (QUIET_MANIFEST, QUIET_TREES, QUIET_DUMP):
        if not path.exists():
            raise SystemExit(
                f"missing golden file: {path}\n"
                "Global Rule 2: regenerate with `python tools/gen_gate1_golden.py`.")
    with open(QUIET_MANIFEST, encoding="utf-8") as handle:
        manifest = json.load(handle)
    with np.load(QUIET_TREES) as data:
        trees = {k: data[k] for k in data.files}
    with np.load(QUIET_DUMP) as data:
        dump = {k: data[k] for k in data.files}

    capacity = 0
    for i in range(len(manifest["runs"])):
        begin, end = int(trees["run_offset"][i]), int(trees["run_offset"][i + 1])
        capacity = max(capacity, 1 + int(trees["children"][begin:end].sum()))
    return manifest, dump, capacity


def build(dump, virtual_loss, capacity, accumulator="q32", synthetic=True):
    config = guofish_core.SearchConfig(virtual_loss=virtual_loss, arena_capacity=capacity)
    cls = (guofish_core.ReplaySearchQ32 if accumulator == "q32"
           else guofish_core.ReplaySearchDouble)
    search = cls(config)
    search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                     dump["moves"], dump["priors"], dump["values"])
    search.synthetic_fallback = synthetic
    return search


# --- distributions ----------------------------------------------------------


def root_distribution(search):
    """Root children's visit shares, in canonical move order.

    Canonical order rather than sorted-by-visits, so two runs are compared move
    against move. Sorting first would report two different move sets as identical
    whenever they happened to have the same shape.
    """
    arrays = search.dump_tree_arrays(0)
    at_depth_one = arrays["depth"] == 1
    visits = arrays["visits"][at_depth_one].astype(np.float64)
    moves = arrays["move"][at_depth_one]
    order = np.argsort(moves)
    visits = visits[order]
    total = visits.sum()
    return (visits / total) if total > 0 else visits, moves[order]


def total_variation(p, q):
    n = min(len(p), len(q))
    return 0.5 * float(np.abs(p[:n] - q[:n]).sum())


# --- one cell ---------------------------------------------------------------


def run_cell(manifest, dump, capacity, workers, in_flight, *, sims, positions, repeats,
             max_batch, affinity, virtual_loss, reference=None):
    cases = [r for r in manifest["runs"]
             if r["virtual_loss"] == virtual_loss and not r.get("full_tree")][:positions]

    per_position = []
    sims_per_s, collisions, mean_batches, largest, stand_in, waits = [], [], [], [], [], []
    for case in cases:
        distributions = []
        for _ in range(repeats):
            search = build(dump, virtual_loss, capacity)
            search.set_position(case["fen"], case.get("history", []))
            stats = search.search_parallel(
                sims,
                guofish_core.ParallelConfig(workers=workers, in_flight=in_flight,
                                            max_batch=max_batch, affinity=affinity))
            par = search.parallel_stats()
            audit = search.audit()
            # A cell whose numbers are not conserved is not a measurement.
            assert par["delivered"] == par["requested"], "delivered != requested"
            assert audit["vloss_total"] == 0, "virtual loss stranded"
            assert audit["conservation_failures"] == 0, "visits do not conserve"

            distribution, _ = root_distribution(search)
            distributions.append(distribution)
            sims_per_s.append(par["sims_per_s"])
            collisions.append(par["select_collisions"] / max(1, par["delivered"]))
            mean_batches.append(par["mean_batch"])
            largest.append(par["largest_batch"])
            waits.append(par["worker_waits"] / max(1, par["delivered"]))
            stand_in.append(stats["synthetic_evaluations"] / max(1, stats["expansions"]))
        per_position.append(distributions)

    run_to_run = []
    for distributions in per_position:
        if len(distributions) > 1:
            run_to_run.append(max(total_variation(a, b)
                                  for a, b in itertools.combinations(distributions, 2)))
        else:
            run_to_run.append(0.0)

    row = {
        "workers": workers,
        "in_flight": in_flight,
        "outstanding": workers * in_flight,
        "sims_per_s": statistics.fmean(sims_per_s),
        "mean_batch": statistics.fmean(mean_batches),
        "largest_batch": max(largest),
        "collisions_per_sim": statistics.fmean(collisions),
        "waits_per_sim": statistics.fmean(waits),
        "stand_in": statistics.fmean(stand_in),
        "run_to_run_tv": statistics.fmean(run_to_run),
        "run_to_run_tv_worst": max(run_to_run) if run_to_run else 0.0,
        # The mean of each position's distribution over its repeats, kept so a
        # later cell can be compared against this one.
        "distributions": [np.mean(np.stack(d), axis=0) for d in per_position],
        "top_share": statistics.fmean(
            float(np.mean(np.stack(d), axis=0).max()) for d in per_position),
    }
    if reference is not None:
        row["tv_vs_serial"] = statistics.fmean(
            total_variation(a, b) for a, b in zip(row["distributions"], reference))
    else:
        row["tv_vs_serial"] = 0.0
    return row


def format_grid(rows, title):
    out = [f"### {title}", ""]
    out.append("| W | K | outstanding | sims/s | mean batch | max batch | collisions/sim "
               "| TV vs serial | run-to-run TV (mean / worst) | top share | stand-in% |")
    out.append("|--:|--:|------------:|-------:|-----------:|----------:|---------------:"
               "|-------------:|----------------------------:|----------:|----------:|")
    for r in rows:
        out.append(
            f"| {r['workers']} | {r['in_flight']} | {r['outstanding']} "
            f"| {r['sims_per_s']:,.0f} | {r['mean_batch']:.1f} | {r['largest_batch']} "
            f"| {r['collisions_per_sim']:.4f} | {100 * r['tv_vs_serial']:.1f}% "
            f"| {100 * r['run_to_run_tv']:.1f}% / {100 * r['run_to_run_tv_worst']:.1f}% "
            f"| {100 * r['top_share']:.1f}% | {100 * r['stand_in']:.1f}% |")
    out.append("")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--virtual-loss", type=float, default=2.5)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--in-flight", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--control", action="store_true", default=True,
                        help="add W=1 rows at matching outstanding counts")
    parser.add_argument("--no-control", dest="control", action="store_false")
    parser.add_argument("--affinity-only", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    manifest, dump, capacity = load_corpus()
    # Parallel runs open branches the serial reference never did, so they
    # allocate more nodes than the golden tree's exact figure.
    capacity *= 4

    topology = guofish_core.ReplaySearchQ32.topology()
    log(f"machine: {platform.platform()}")
    log(f"topology: {topology['source']}  hybrid={topology['hybrid']}  "
        f"P-cores={topology['pcore_physical']}  P-logical={topology['pcore_all']}  "
        f"E-cores={topology['ecore_all']}")
    log(f"corpus: {args.positions} quiet positions x {args.repeats} repeats, "
        f"{args.sims} sims, virtual loss {args.virtual_loss}, max_batch {args.max_batch}")
    log("")

    payload = {"topology": topology, "args": vars(args) | {"json_out": None}}
    sections = []

    if not args.affinity_only:
        log("reference row: W=1 K=1 (deterministic)")
        serial = run_cell(manifest, dump, capacity, 1, 1, sims=args.sims,
                          positions=args.positions, repeats=1, max_batch=args.max_batch,
                          affinity="none",
                          virtual_loss=args.virtual_loss)
        reference = serial["distributions"]
        serial["tv_vs_serial"] = 0.0
        rows = [serial]

        for in_flight in args.in_flight:
            for workers in args.workers:
                log(f"  W={workers} K={in_flight}")
                rows.append(run_cell(manifest, dump, capacity, workers, in_flight,
                                     sims=args.sims, positions=args.positions,
                                     repeats=args.repeats, max_batch=args.max_batch,
                                     affinity="pcore_physical",
                                     virtual_loss=args.virtual_loss, reference=reference))
        rows.sort(key=lambda r: (r["in_flight"], r["workers"]))
        sections += format_grid(rows, "W x K grid (W=1/K=1 is the reference row)")
        payload["grid"] = [{k: v for k, v in r.items() if k != "distributions"} for r in rows]

        if args.control:
            control_rows = [serial]
            for outstanding in sorted({w * k for w in args.workers for k in args.in_flight}):
                if outstanding > 128:
                    continue
                log(f"  control W=1 K={outstanding}")
                control_rows.append(run_cell(manifest, dump, capacity, 1, outstanding,
                                             sims=args.sims, positions=args.positions,
                                             repeats=2, max_batch=args.max_batch,
                                             affinity="none",
                                             virtual_loss=args.virtual_loss,
                                             reference=reference))
            sections += format_grid(
                control_rows,
                "Control: W=1 at the same outstanding-leaf counts (deterministic, so "
                "run-to-run TV is 0 by construction)")
            payload["control"] = [{k: v for k, v in r.items() if k != "distributions"}
                                  for r in control_rows]

    # --- affinity ----------------------------------------------------------
    log("affinity: W=6 physical vs W=12 SMT vs unpinned")
    affinity_rows = []
    cases = [
        ("W=6, one thread per P-core", 6, "pcore_physical"),
        ("W=6, unpinned", 6, "none"),
        ("W=12, both SMT siblings", 12, "pcore_smt"),
        ("W=12, unpinned", 12, "none"),
        ("W=12, all logical (E-cores included)", 12, "all_logical"),
        ("W=4, one thread per P-core", 4, "pcore_physical"),
        ("W=4, unpinned", 4, "none"),
    ]
    for label, workers, policy in cases:
        row = run_cell(manifest, dump, capacity, workers, 8, sims=args.sims,
                       positions=args.positions, repeats=args.repeats,
                       max_batch=args.max_batch, affinity=policy,
                       virtual_loss=args.virtual_loss)
        row["label"] = label
        affinity_rows.append(row)
        log(f"  {label:<40} {row['sims_per_s']:>10,.0f} sims/s")

    sections += ["### Affinity (K=8 throughout)", "",
                 "| configuration | outstanding | sims/s | mean batch | collisions/sim "
                 "| waits/sim |",
                 "|---------------|------------:|-------:|-----------:|---------------:"
                 "|----------:|"]
    for r in affinity_rows:
        sections.append(f"| {r['label']} | {r['outstanding']} | {r['sims_per_s']:,.0f} "
                        f"| {r['mean_batch']:.1f} | {r['collisions_per_sim']:.4f} "
                        f"| {r['waits_per_sim']:.3f} |")
    sections.append("")
    payload["affinity"] = [{k: v for k, v in r.items() if k != "distributions"}
                           for r in affinity_rows]

    # --- GIL ---------------------------------------------------------------
    log("GIL acquisition on the dispatcher")
    case = next(r for r in manifest["runs"]
                if r["virtual_loss"] == args.virtual_loss and not r.get("full_tree"))
    search = build(dump, args.virtual_loss, capacity)
    search.set_position(case["fen"], case.get("history", []))
    probe = guofish_core.GilProbe()
    search.set_batch_hook(probe)
    search.search_parallel(args.sims, guofish_core.ParallelConfig(workers=4, in_flight=8,
                                                                  max_batch=args.max_batch))
    search.set_batch_hook(None)
    par = search.parallel_stats()
    samples = np.sort(par["hook_wait_ns_samples"])

    def pct(p):
        return float(samples[min(len(samples) - 1, int(p * len(samples)))])

    gil = {
        "batches": int(par["batches"]),
        "median_ns": pct(0.50),
        "p90_ns": pct(0.90),
        "p99_ns": pct(0.99),
        "max_ns": float(samples[-1]),
        "total_ms": par["hook_wait_ns"] / 1e6,
        "wall_ms": par["wall_ns"] / 1e6,
    }
    payload["gil"] = gil
    sections += [
        "### Dispatcher GIL acquisition", "",
        "| batches | median | p90 | p99 | max | total | share of wall |",
        "|--------:|-------:|----:|----:|----:|------:|--------------:|",
        f"| {gil['batches']} | {gil['median_ns'] / 1000:.2f} us | {gil['p90_ns'] / 1000:.2f} us "
        f"| {gil['p99_ns'] / 1000:.2f} us | {gil['max_ns'] / 1000:.2f} us "
        f"| {gil['total_ms']:.3f} ms | {100 * gil['total_ms'] / gil['wall_ms']:.2f}% |", ""]

    if args.markdown:
        log("")
        for line in sections:
            log(line)

    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        log(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
