#!/usr/bin/env python
"""C12 Step 1b — turn an Nsight Compute report into the kernel-by-kernel table.

    python tools/ncu_report_c12.py runs/c12/ncu_shape24.ncu-rep --markdown

Reads the report through `ncu --import ... --csv --page details`, which needs no
privileges — only the COLLECTION does (see tools/run_ncu_c12.ps1). One row per
kernel invocation per metric comes back; this rolls them up two ways.

  by kernel   every distinct kernel, its total device time inside the profiled
              forward, its achieved occupancy and its compute/memory throughput.
              Sorted by total time, because that is the order in which fixing
              them matters.

  by class    the same kernels bucketed into GEMM / attention / normalisation /
              elementwise-and-copy / other. This is the table the roofline
              argument is actually made on: the C12 brief's claim is that the
              forward is bound by activation materialisation rather than by
              compute, and the share of device time that is NOT a GEMM is the
              direct measurement of it.

The classifier matches on the kernel's mangled name and is deliberately explicit
rather than clever: an unmatched kernel lands in `other` and `other` is printed
with its members, so a misclassification is visible instead of being absorbed
into a bucket that happened to look right.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import sys

NCU = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0"
           r"\target\windows-desktop-win7-x64\ncu.exe")

# The metrics the C12 acceptance criteria name, by the label ncu prints them
# under on the details page.
WANTED = {
    "Duration": "duration",
    "Compute (SM) Throughput": "compute_pct",
    "Memory Throughput": "memory_pct",
    "DRAM Throughput": "dram_pct",
    "Achieved Occupancy": "achieved_occupancy",
    "Theoretical Occupancy": "theoretical_occupancy",
    "Waves Per SM": "waves_per_sm",
    "Registers Per Thread": "registers",
    "Achieved Active Warps Per SM": "active_warps",
}

# Ordered: the first pattern that matches wins, so the GEMM test has to come
# before the generic cutlass one and the attention test before both.
CLASSES = (
    ("attention", re.compile(r"fmha|AttentionKernel|flash", re.I)),
    ("GEMM", re.compile(r"gemm|cutlass|gemv", re.I)),
    ("normalisation", re.compile(r"layer_norm|LayerNorm|softmax", re.I)),
    ("elementwise/copy", re.compile(
        r"elementwise_kernel|direct_copy|vectorized_gather|CatArrayBatched|"
        r"index_elementwise|fill_|copy_device_to_device", re.I)),
)


def classify(name: str) -> str:
    for label, pattern in CLASSES:
        if pattern.search(name):
            return label
    return "other"


def short(name: str, width: int = 70) -> str:
    """The kernel's identity without its template argument list.

    Two different template instantiations of `elementwise_kernel` are two
    different kernels and are kept apart by the full name; this is only what
    gets PRINTED, and the full name stays in the JSON.
    """
    trimmed = re.sub(r"^void\s+", "", name)
    trimmed = trimmed.split("<")[0].split("(")[0]
    return trimmed if len(trimmed) <= width else trimmed[:width - 3] + "..."


def load(report: Path) -> list[dict]:
    """One record per kernel invocation, metrics flattened onto it."""
    if not NCU.exists():
        raise SystemExit(f"ncu.exe not found at {NCU}")
    completed = subprocess.run(
        [str(NCU), "--import", str(report), "--csv", "--page", "details"],
        capture_output=True, text=True, check=True)
    reader = csv.DictReader(io.StringIO(completed.stdout))

    invocations: dict[tuple, dict] = {}
    for row in reader:
        metric = row.get("Metric Name", "")
        if metric not in WANTED:
            continue
        key = (row["ID"], row["Kernel Name"])
        record = invocations.setdefault(key, {"name": row["Kernel Name"],
                                              "id": row["ID"],
                                              "grid": row.get("Grid Size", ""),
                                              "block": row.get("Block Size", "")})
        raw = (row.get("Metric Value") or "").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        # ncu reports Duration in whatever unit fits; normalise to microseconds
        # so a nanosecond row and a microsecond row are not summed as if equal.
        if metric == "Duration":
            unit = row.get("Metric Unit", "us")
            value *= {"ns": 1e-3, "us": 1.0, "ms": 1e3, "second": 1e6}.get(unit, 1.0)
        record[WANTED[metric]] = value
    return list(invocations.values())


def aggregate(records: list[dict]) -> dict:
    by_kernel: dict[str, dict] = {}
    for record in records:
        entry = by_kernel.setdefault(record["name"], {
            "name": record["name"], "class": classify(record["name"]),
            "calls": 0, "duration_us": 0.0, "grid": record["grid"],
            "block": record["block"], "weighted": {}})
        entry["calls"] += 1
        duration = record.get("duration", 0.0)
        entry["duration_us"] += duration
        # Every percentage is weighted by the kernel's own duration: a 0.9 µs
        # kernel at 3% occupancy and a 40 µs kernel at 30% do not average to
        # 16.5% of anything anybody cares about.
        for field in ("compute_pct", "memory_pct", "dram_pct",
                      "achieved_occupancy", "theoretical_occupancy",
                      "waves_per_sm", "registers"):
            if field in record:
                slot = entry["weighted"].setdefault(field, [0.0, 0.0])
                slot[0] += record[field] * duration
                slot[1] += duration

    for entry in by_kernel.values():
        for field, (total, weight) in entry["weighted"].items():
            entry[field] = total / weight if weight else 0.0
        del entry["weighted"]

    kernels = sorted(by_kernel.values(), key=lambda k: -k["duration_us"])
    total_us = sum(k["duration_us"] for k in kernels) or 1.0

    classes: dict[str, dict] = {}
    for kernel in kernels:
        entry = classes.setdefault(kernel["class"], {
            "class": kernel["class"], "calls": 0, "duration_us": 0.0,
            "distinct": 0, "members": []})
        entry["calls"] += kernel["calls"]
        entry["duration_us"] += kernel["duration_us"]
        entry["distinct"] += 1
        entry["members"].append(short(kernel["name"]))
    for entry in classes.values():
        entry["share"] = entry["duration_us"] / total_us
        # Duration-weighted occupancy of the class, from its members.
        weight = sum(k["duration_us"] for k in kernels if k["class"] == entry["class"])
        entry["achieved_occupancy"] = (
            sum(k["duration_us"] * k.get("achieved_occupancy", 0.0)
                for k in kernels if k["class"] == entry["class"]) / weight
            if weight else 0.0)
        entry["compute_pct"] = (
            sum(k["duration_us"] * k.get("compute_pct", 0.0)
                for k in kernels if k["class"] == entry["class"]) / weight
            if weight else 0.0)
        entry["dram_pct"] = (
            sum(k["duration_us"] * k.get("dram_pct", 0.0)
                for k in kernels if k["class"] == entry["class"]) / weight
            if weight else 0.0)

    for kernel in kernels:
        kernel["share"] = kernel["duration_us"] / total_us

    return {
        "total_us": total_us,
        "invocations": sum(k["calls"] for k in kernels),
        "distinct_kernels": len(kernels),
        "kernels": kernels,
        "classes": sorted(classes.values(), key=lambda c: -c["duration_us"]),
    }


def markdown(result: dict, label: str, top: int) -> str:
    lines = [
        f"One forward. **{result['invocations']} kernel launches**, "
        f"{result['distinct_kernels']} distinct kernels, "
        f"{result['total_us']:.0f} µs of device time.", "",
        "| class | launches | device µs | share | achieved occupancy "
        "| compute SoL | DRAM SoL |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for entry in result["classes"]:
        lines.append(
            f"| **{entry['class']}** | {entry['calls']} | {entry['duration_us']:.1f} "
            f"| {100 * entry['share']:.1f}% | {entry['achieved_occupancy']:.1f}% "
            f"| {entry['compute_pct']:.1f}% | {entry['dram_pct']:.1f}% |")
    lines += ["", f"Top {top} kernels by device time:", "",
              "| kernel | class | calls | device µs | share | grid | block "
              "| occupancy (achieved / theoretical) | compute SoL | DRAM SoL |",
              "|---|---|---:|---:|---:|---|---|---:|---:|---:|"]
    for kernel in result["kernels"][:top]:
        lines.append(
            f"| `{short(kernel['name'], 58)}` | {kernel['class']} | {kernel['calls']} "
            f"| {kernel['duration_us']:.1f} | {100 * kernel['share']:.1f}% "
            f"| {kernel['grid']} | {kernel['block']} "
            f"| {kernel.get('achieved_occupancy', 0):.1f}% / "
            f"{kernel.get('theoretical_occupancy', 0):.1f}% "
            f"| {kernel.get('compute_pct', 0):.1f}% "
            f"| {kernel.get('dram_pct', 0):.1f}% |")
    other = [c for c in result["classes"] if c["class"] == "other"]
    if other:
        lines += ["", "Unclassified (`other`), listed so a misclassification is "
                      "visible rather than absorbed: " +
                  ", ".join(f"`{m}`" for m in sorted(set(other[0]["members"]))) + "."]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--top", type=int, default=18)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    payload = {}
    for report in args.reports:
        result = aggregate(load(report))
        payload[report.stem] = result
        if args.markdown:
            print(f"\n#### {report.stem}\n")
            print(markdown(result, report.stem, args.top))
        else:
            print(report.stem, json.dumps(
                {k: v for k, v in result.items() if k != "kernels"}, indent=2)[:2000])
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
