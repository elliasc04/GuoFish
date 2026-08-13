#!/usr/bin/env python
"""C12 Step 1 — turn an nsys trace into the four numbers the brief asks for.

    nsys export --type sqlite -o runs/c12/x.sqlite runs/c12/x.nsys-rep
    python tools/nsys_report_c12.py runs/c12/x.sqlite --markdown

WHAT IT COMPUTES, AND WHY NOT `nsys stats`
==========================================
`nsys stats --report cuda_gpu_kern_sum` sums kernel durations over the WHOLE
process, which for this workload is dominated by graph capture and warmup — ~0.7
s of capture against a 1.4 s search. Every number below is restricted to the
NVTX region the workload pushed around the measured search, so a busy fraction
here is the busy fraction of the search and not of the program.

  GPU busy fraction   the UNION of GPU activity intervals divided by the NVTX
                      region's wall time. A union, not a sum: kernels and copies
                      overlap across streams, and summing would let a 90%-busy
                      device report 130%. This is the number the brief says
                      "disambiguates the pipelining payoff by an order of
                      magnitude", so it is computed the conservative way.

  per-batch gap       the idle interval between one graph execution ending and
                      the next beginning. This is what the dispatcher costs the
                      GPU: descent, backup, tokenization, the cache probe, the
                      boundary crossing and the two copies. Reported as a
                      histogram, never a mean — a mean of a distribution with a
                      long tail is the statistic C9's brief specifically banned.

  CPU/GPU overlap     the share of each gap that is NOT explained by the H2D/D2H
                      copies bracketing it, i.e. the part that is genuinely host
                      work with an idle device.

  kernel breakdown    from a `--cuda-graph-trace=node` capture, where the graph's
                      nodes appear individually in CUPTI_ACTIVITY_KIND_KERNEL.
                      A `graph`-granularity capture has one row per graph launch
                      and no breakdown; the tool says so rather than printing an
                      empty table.

`--cuda-graph-trace=node` perturbs what it measures (each node is instrumented
separately), so the timing numbers should be read off a `graph` capture and the
breakdown off a `node` one. `--label` carries which is which into the output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import statistics
import sys


def union_duration(intervals) -> int:
    """Total time covered by at least one interval. Inputs need not be sorted."""
    ordered = sorted(intervals)
    total = 0
    current_start, current_end = None, None
    for start, end in ordered:
        if current_end is None:
            current_start, current_end = start, end
        elif start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        elif end > current_end:
            current_end = end
    if current_end is not None:
        total += current_end - current_start
    return total


def percentiles(values, quantiles=(50, 90, 99)) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    out = {"min": ordered[0], "max": ordered[-1], "mean": statistics.fmean(ordered)}
    for q in quantiles:
        index = min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))
        out[f"p{q}"] = ordered[index]
    return out


def region(connection, name: str):
    """The [start, end] of the NVTX range whose text starts with `name`."""
    rows = list(connection.execute(
        "select e.start, e.end, coalesce(e.text, s.value) as label "
        "from NVTX_EVENTS e left join StringIds s on s.id = e.textId "
        "where label like ?", (name + "%",)))
    if not rows:
        available = [r[0] for r in connection.execute(
            "select coalesce(e.text, s.value) from NVTX_EVENTS e "
            "left join StringIds s on s.id = e.textId")]
        raise SystemExit(f"no NVTX region matching {name!r}; found {available}")
    if len(rows) > 1:
        raise SystemExit(f"{len(rows)} NVTX regions match {name!r}: "
                         f"{[r[2] for r in rows]}")
    return rows[0][0], rows[0][1], rows[0][2]


def within(connection, table, start, end, columns="start, end"):
    return list(connection.execute(
        f"select {columns} from {table} where start >= ? and end <= ? order by start",
        (start, end)))


def table_exists(connection, name: str) -> bool:
    return bool(list(connection.execute(
        "select 1 from sqlite_master where type='table' and name=?", (name,))))


def analyse(path: Path, region_name: str) -> dict:
    connection = sqlite3.connect(str(path))
    start, end, label = region(connection, region_name)
    wall_ns = end - start

    graphs = (within(connection, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", start, end)
              if table_exists(connection, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE") else [])
    kernels = within(connection, "CUPTI_ACTIVITY_KIND_KERNEL", start, end)
    copies = within(connection, "CUPTI_ACTIVITY_KIND_MEMCPY", start, end,
                    "start, end, bytes, copyKind")

    # A graph-granularity capture reports the graph launch and NOT its nodes; a
    # node-granularity one reports the nodes and no graph row. Taking the union
    # of both is correct in either case and double-counts in neither, because
    # exactly one of the two is populated for any given launch.
    activity = [(s, e) for s, e in graphs] + [(s, e) for s, e in kernels] + \
               [(row[0], row[1]) for row in copies]
    busy_ns = union_duration(activity)

    # The device-side execution unit. On a graph capture this is one row per
    # boundary crossing; on a node capture the launches have to be rebuilt from
    # the node timestamps, so gaps are reported only from a graph capture.
    granularity = "graph" if graphs else "node"
    if graphs:
        launches = [(s, e) for s, e in graphs]
    else:
        # A node capture has no launch row, so the launches are rebuilt from the
        # nodes: a graph replays the SAME node ids in the same order every time,
        # so a repeat of an id already seen in the current cluster is the first
        # node of the next replay. That is exact — it needs no gap threshold, and
        # a threshold would have been a guess about how long the host takes.
        launches = []
        seen, cluster_start, cluster_end = set(), None, None
        for kernel_start, kernel_end, node_id in connection.execute(
                "select start, end, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL "
                "where start >= ? and end <= ? order by start", (start, end)):
            if node_id is None:
                continue
            if node_id in seen:
                launches.append((cluster_start, cluster_end))
                seen, cluster_start = set(), None
            if cluster_start is None:
                cluster_start = kernel_start
            cluster_end = max(cluster_end or 0, kernel_end)
            seen.add(node_id)
        if cluster_start is not None:
            launches.append((cluster_start, cluster_end))
    gaps = [next_start - previous_end
            for (_, previous_end), (next_start, _) in zip(launches, launches[1:])]

    kernel_rows = []
    if kernels and table_exists(connection, "StringIds"):
        by_name = {}
        for kernel_start, kernel_end, name in connection.execute(
                "select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                "join StringIds s on s.id = k.demangledName "
                "where k.start >= ? and k.end <= ?", (start, end)):
            entry = by_name.setdefault(name, {"count": 0, "ns": 0})
            entry["count"] += 1
            entry["ns"] += kernel_end - kernel_start
        total_kernel_ns = sum(e["ns"] for e in by_name.values()) or 1
        kernel_rows = sorted(
            ({"name": name, "count": e["count"], "total_ns": e["ns"],
              "mean_us": e["ns"] / e["count"] / 1000,
              "share": e["ns"] / total_kernel_ns}
             for name, e in by_name.items()),
            key=lambda r: -r["total_ns"])

    copy_rows = {}
    for copy_start, copy_end, nbytes, kind in copies:
        entry = copy_rows.setdefault(int(kind), {"count": 0, "ns": 0, "bytes": 0})
        entry["count"] += 1
        entry["ns"] += copy_end - copy_start
        entry["bytes"] += int(nbytes or 0)

    return {
        "file": str(path),
        "region": label,
        "granularity": granularity,
        "wall_ms": wall_ns / 1e6,
        "gpu_busy_ms": busy_ns / 1e6,
        "gpu_busy_fraction": busy_ns / wall_ns,
        "gpu_idle_ms": (wall_ns - busy_ns) / 1e6,
        "launches": len(launches),
        "kernels": len(kernels),
        "copies": len(copies),
        "launch_us": {k: v / 1000 for k, v in
                      percentiles([e - s for s, e in launches]).items()},
        "gap_us": {k: v / 1000 for k, v in percentiles(gaps).items()},
        "gap_total_ms": sum(gaps) / 1e6,
        "gap_share_of_wall": sum(gaps) / wall_ns if wall_ns else 0.0,
        "kernel_breakdown": kernel_rows,
        "copy_breakdown": {str(k): v for k, v in copy_rows.items()},
    }


def markdown(result: dict) -> str:
    lines = [
        f"NVTX region `{result['region']}`, capture granularity "
        f"`{result['granularity']}`.", "",
        "| quantity | value |", "|---|---:|",
        f"| region wall | {result['wall_ms']:.1f} ms |",
        f"| **GPU busy (union of all device activity)** | "
        f"**{result['gpu_busy_ms']:.1f} ms** |",
        f"| **GPU busy fraction** | **{100 * result['gpu_busy_fraction']:.1f}%** |",
        f"| GPU idle | {result['gpu_idle_ms']:.1f} ms |",
        f"| graph launches | {result['launches']:,} |",
        f"| individual kernels traced | {result['kernels']:,} |",
        f"| memcpys | {result['copies']:,} |",
    ]
    if result["launch_us"]:
        launch = result["launch_us"]
        gap = result["gap_us"]
        lines += ["", "Per-launch execution and the gap between launches "
                      "(microseconds, histogram not mean):", "",
                  "| | min | p50 | p90 | p99 | max | mean |",
                  "|---|---:|---:|---:|---:|---:|---:|",
                  f"| graph execution | {launch['min']:.1f} | {launch['p50']:.1f} "
                  f"| {launch['p90']:.1f} | {launch['p99']:.1f} | {launch['max']:.1f} "
                  f"| {launch['mean']:.1f} |"]
        if gap:
            lines.append(
                f"| gap between launches | {gap['min']:.1f} | {gap['p50']:.1f} "
                f"| {gap['p90']:.1f} | {gap['p99']:.1f} | {gap['max']:.1f} "
                f"| {gap['mean']:.1f} |")
            lines += ["",
                      f"Gaps total {result['gap_total_ms']:.1f} ms = "
                      f"{100 * result['gap_share_of_wall']:.1f}% of the region."]
    if result["kernel_breakdown"]:
        lines += ["", "Kernel breakdown (top 20 by total device time):", "",
                  "| kernel | calls | total ms | mean µs | share |",
                  "|---|---:|---:|---:|---:|"]
        for row in result["kernel_breakdown"][:20]:
            name = row["name"].replace("|", "\\|")
            if len(name) > 78:
                name = name[:75] + "..."
            lines.append(f"| `{name}` | {row['count']:,} | "
                         f"{row['total_ns'] / 1e6:.2f} | {row['mean_us']:.1f} | "
                         f"{100 * row['share']:.1f}% |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sqlite", type=Path, nargs="+")
    parser.add_argument("--region", default="SEARCH",
                        help="NVTX region prefix to restrict every figure to")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for path in args.sqlite:
        result = analyse(path, args.region)
        results.append(result)
        if args.markdown:
            print(f"\n#### {path.name}\n")
            print(markdown(result))
        else:
            print(json.dumps(result, indent=2)[:4000])
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
