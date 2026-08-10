#!/usr/bin/env python
"""C10b Stage 3.1 — the GPU knee, re-measured on the graphed forward.

    python tools/bench_c10b_knee.py --markdown
    python tools/bench_c10b_knee.py --batch 8 16 32 64 128 256 --iters 300

WHY EVERY C9a NUMBER IS VOID AND THIS FILE EXISTS
=================================================
C9a measured the curve on an UNGRAPHED forward and found the small-batch cost on
the host: ~110 ATen submissions per forward, `launch ~ total`, `drain ~ 0`, and a
flat ~4 ms per batch from 8 rows to 64. That is a measurement of dispatch, not of
the GPU, so every absolute figure it produced belongs to a configuration C10b no
longer runs. The knee has to be found again.

THE METHODOLOGY IS C9a's, IMPORTED RATHER THAN RESTATED
=======================================================
`time_path`, `time_saturated`, `sm_clock_mhz` and `build_tokens` come from
tools/bench_c9_knee.py. Two tables that differ in their timing discipline are not
comparable, and the whole point of this file is a comparison — so the discipline
is shared code and only the PATHS are new. In particular the synchronize around
every timed iteration stays, and so does the SM-clock column: an RTX 5070 drops
to ~300 MHz inside a synchronized gap and a curve measured through that ramp is
the artefact C9a documented.

THE PATHS, AND WHY THERE ARE THREE GRAPHED ONES
===============================================
  graph-forward     static device input, replay only. The pure GPU curve.
  graph-gathered    pinned int32 H2D -> replay -> 64-wide GPU gather -> narrow
                    D2H. This is C9a's `gathered` column with the forward
                    graphed, and it is the row C10b's acceptance criterion 3 is
                    denominated in ("batch-32 gathered >= 2x the C9 figure").
  graph-production  pinned int32 H2D -> replay -> FULL 4096-wide bf16 D2H +
                    value D2H. This is what playing/v6/evaluator.py actually
                    does, because C10 put the gather in C++ (cpp/evaluator.hpp,
                    `gather_softmax_canonical`) rather than on the GPU.

The last two are separated deliberately. `gathered` is the comparable row and
`production` is the true one, and they are not the same path: scope §2.5's
64-wide D2H was never built, so quoting only `gathered` would be quoting a
throughput the engine does not have. At batch 256 the difference is a 2 MiB
copy against a 32 KiB one.

The ungraphed paths are re-run in the same session so the speedup column is a
comparison rather than a subtraction across two days of driver state.

Writes nothing. Paste the tables into BENCH.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.bench_c9_knee import (  # noqa: E402
    GATHER_WIDTH,
    PATHS as UNGRAPHED_PATHS,
    POLICY_SIZE,
    SEQ_LENGTH,
    build_tokens,
    log,
    make_forward,
    time_path,
    time_saturated,
)

DEFAULT_MODEL = _PROJECT_ROOT / "models" / "guofish5_20M" / "v5_10.9M_best.pt"
DEFAULT_SHARDS = _PROJECT_ROOT / "data" / "processed" / "multipv"

# C9a's batch-32 `gathered` figure, and the acceptance bar built on it.
C9_BATCH32_POS_PER_S = 8374
ACCEPTANCE_MULTIPLE = 2.0


def graphed_paths(graph, pinned_host, gather_index, policy_host, value_host):
    """The three graphed paths, closed over one captured `GraphedForward`.

    Every one of them is allocation-free per call, which is the property the
    callback has to have; a path that allocated would measure the allocator.
    """

    def forward(_f, _d, _h, _p, _i):
        graph.replay(graph.pad_to(pinned_host.shape[0]))

    def gathered(_f, _d, _h, _p, _i):
        count = pinned_host.shape[0]
        padded = graph.stage(count, pinned_host)
        graph.replay(padded)
        policy, values = graph.outputs(count, padded)
        torch.gather(policy, 1, gather_index).to("cpu", non_blocking=False)
        values.to("cpu", non_blocking=False)

    def production(_f, _d, _h, _p, _i):
        count = pinned_host.shape[0]
        padded = graph.stage(count, pinned_host)
        graph.replay(padded)
        policy, values = graph.outputs(count, padded)
        policy_host.copy_(policy)
        value_host.copy_(values)

    return {
        "graph-forward": forward,
        "graph-gathered": gathered,
        "graph-production": production,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--capture", type=int, nargs="+", default=None,
                        help="shapes to capture (default: every --batch value, so the "
                             "table has no padding in it)")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--ungraphed", action="store_true", default=True,
                        help="also re-run C9a's paths in this session")
    parser.add_argument("--no-ungraphed", dest="ungraphed", action="store_false")
    parser.add_argument("--source", choices=("auto", "data", "synthetic"), default="auto")
    parser.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        log("ERROR: the knee is a GPU property; refusing to publish a CPU curve.")
        return 1
    device = torch.device("cuda")

    from playing.v5.playv5 import load_model
    from playing.v6 import graphs as graph_module

    log(f"torch {torch.__version__}  device {device}  {torch.cuda.get_device_name(0)}")
    log(f"python {platform.python_version()}  {platform.platform()}")
    model = load_model(args.model, device)
    model.eval()
    forward = make_forward(model, device)

    biggest = max(args.batch)
    tokens, origin = build_tokens(args, biggest)
    log(f"positions: {origin}")

    # EVERY SWEPT BATCH IS CAPTURED by default. This table is about where the GPU
    # knees, and a padded row would report the cost of the shape above it —
    # which is a real production effect but a different question, measured in
    # C10b-3d. `--capture` narrows the set when the padded curve IS the question.
    capture = sorted(set(args.capture if args.capture else args.batch))
    graph = graph_module.GraphedForward(model, device, biggest, sizes=capture)
    log(f"captured: {graph.report.describe()}")
    log("")

    rows = []
    for batch in args.batch:
        host = tokens[:batch].contiguous()
        device_tokens = host.to(device)
        pinned = torch.empty((batch, SEQ_LENGTH), dtype=torch.int32, pin_memory=True)
        pinned.copy_(host.to(torch.int32))
        index = torch.randint(0, POLICY_SIZE, (batch, GATHER_WIDTH), device=device,
                              dtype=torch.int64)
        # Pinned host landing buffers, exactly as LiveEvaluator's are page-locked.
        policy_host = torch.empty((batch, POLICY_SIZE), dtype=torch.bfloat16, pin_memory=True)
        value_host = torch.empty((batch,), dtype=torch.bfloat16, pin_memory=True)

        paths = dict(graphed_paths(graph, pinned, index, policy_host, value_host))
        if args.ungraphed:
            paths.update(UNGRAPHED_PATHS)

        for name, fn in paths.items():
            row = time_path(name, fn, forward, device_tokens, host, pinned, index,
                            args.iters, args.warmup, True)
            row.update(time_saturated(fn, forward, device_tokens, host, pinned, index,
                                      max(8, args.iters // 8), args.depth, True))
            row["batch"] = batch
            row["ms_per_batch"] = row["median_s"] * 1e3
            row["pos_per_s"] = batch / row["median_s"]
            row["sat_ms_per_batch"] = row["sat_median_s"] * 1e3
            row["sat_us_per_pos"] = row["sat_median_s"] * 1e6 / batch
            row["sat_pos_per_s"] = batch / row["sat_median_s"]
            row["best_ms"] = row["sat_min_s"] * 1e3
            row["best_pos_per_s"] = batch / row["sat_min_s"]
            rows.append(row)
            log(f"  batch {batch:>4}  {name:<17} isolated {row['ms_per_batch']:8.3f} ms   "
                f"best {row['best_ms']:8.3f} ms ({row['best_pos_per_s']:7.0f} pos/s, "
                f"{row['sat_sm_clock_mhz'] or 0:.0f} MHz)")
        del device_tokens, pinned, index, policy_host, value_host

    names = list(dict.fromkeys(r["path"] for r in rows))
    if args.markdown:
        log("")
        for name in names:
            log(f"### {name}")
            log("")
            log("| batch | isolated ms | saturated ms | best ms | best pos/s | SM MHz |")
            log("|------:|------------:|-------------:|--------:|-----------:|-------:|")
            for row in rows:
                if row["path"] != name:
                    continue
                log(f"| {row['batch']} | {row['ms_per_batch']:.3f} "
                    f"| {row['sat_ms_per_batch']:.3f} | {row['best_ms']:.3f} "
                    f"| {row['best_pos_per_s']:,.0f} | {row['sat_sm_clock_mhz'] or 0:.0f} |")
            log("")

    # --- acceptance criterion 3, stated as a verdict rather than left to the reader
    gathered32 = next((r for r in rows
                       if r["path"] == "graph-gathered" and r["batch"] == 32), None)
    if gathered32 is not None:
        target = C9_BATCH32_POS_PER_S * ACCEPTANCE_MULTIPLE
        got = gathered32["best_pos_per_s"]
        log(f"C10b acceptance 3 — batch-32 `gathered`: {got:,.0f} pos/s against C9a's "
            f"{C9_BATCH32_POS_PER_S:,} ({got / C9_BATCH32_POS_PER_S:.2f}x), "
            f"target {target:,.0f}: {'MET' if got >= target else 'NOT MET'}")
        production32 = next((r for r in rows
                             if r["path"] == "graph-production" and r["batch"] == 32), None)
        if production32 is not None:
            log(f"  the shipped path (full 4096-wide D2H) at batch 32: "
                f"{production32['best_pos_per_s']:,.0f} pos/s "
                f"({production32['best_pos_per_s'] / C9_BATCH32_POS_PER_S:.2f}x C9a)")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "model": str(args.model),
            "positions": origin,
            "captured": capture,
            "capture_seconds": graph.report.seconds,
            "capture_bytes": graph.report.reserved_delta,
            "rows": rows,
        }, indent=2), encoding="utf-8")
        log(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
