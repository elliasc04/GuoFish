#!/usr/bin/env python
"""Pick the micro-batch empirically on THIS GPU.

    python training/v5_multiPV/bench_batch.py
    python training/v5_multiPV/bench_batch.py --micro-batch 512 1024 2048 --compile

Every configuration exercises the REAL step - real shards through
MultiPVDataset with the colour mirror on, forward, dense-4096 masked KL plus
value MSE, backward, optimizer step. A forward-only probe underestimates badly:
the 4096-wide logits and the float32 log_softmax over them are a first-class
memory consumer, and the backward through log_softmax holds another copy.

WINDOWS/WDDM WARNING, which is why this file exists at all: on WDDM, exceeding
VRAM does NOT raise OOM. The driver silently spills allocations to shared system
memory over PCIe and throughput collapses instead. So the stopping signal here
is the THROUGHPUT KNEE - the first config whose samples/s falls below the
previous config's - not an exception. Reserved memory (not allocated) is what
pressures the driver, so the ceiling is checked against reserved.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# MUST run before `import torch`: Inductor resolves its cache root during
# import. See _shorten_inductor_cache_dir in train_v5.py for why.
if sys.platform == "win32" and not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
        os.environ.get("USERPROFILE") or tempfile.gettempdir(), "ti")

import torch
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_DATA_MULTIPV = _PROJECT_ROOT / "data" / "multiPV"
for _p in (str(_HERE), str(_DATA_MULTIPV), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from dataset import MultiPVCollate, MultiPVDataset      # noqa: E402
from losses import compute_losses, policy_denominator   # noqa: E402
from model_v5 import ModelConfig, build_model           # noqa: E402

DEFAULT_SHARDS = _PROJECT_ROOT / "data" / "processed" / "multipv"
DEFAULT_MANIFEST = _DATA_MULTIPV / "manifests" / "dataset_manifest.json"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def gib(n: int) -> float:
    return n / 1024 ** 3


def make_loader(ds, micro_batch, workers, prefetch, mirror_prob):
    return DataLoader(
        ds, batch_size=micro_batch, shuffle=True, num_workers=workers,
        drop_last=True, collate_fn=MultiPVCollate(mirror_prob=mirror_prob),
        pin_memory=True, persistent_workers=bool(workers),
        prefetch_factor=prefetch if workers else None)


def to_device(batch, device):
    keys = ("tokens", "policy", "legal_mask", "value", "value_cp", "has_policy")
    return {k: (v.to(device, non_blocking=True) if k in keys else v)
            for k, v in batch.items()}


def bench_one(ds, micro_batch, args, device, amp_ctx, coverage) -> dict:
    """One micro-batch size, real step, warmup then timed."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build_model(ModelConfig(
        d_model=args.d_model, num_layers=args.num_layers, nhead=args.nhead,
        dim_feedforward=args.dim_feedforward, head_dim=args.head_dim,
        dropout=args.dropout)).to(device)
    if args.compile:
        model = torch.compile(model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    model.train()

    loader = make_loader(ds, micro_batch, args.workers, args.prefetch_factor,
                         args.mirror_prob)
    it = iter(loader)
    C_pol = policy_denominator(micro_batch, coverage)
    C_val = float(micro_batch)

    def one_step():
        nonlocal it
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        b = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx():
            logits, value = model(b["tokens"])
        parts = compute_losses(logits, value, b)
        loss = parts.policy_kl_sum / C_pol + parts.value_se_sum / C_val
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return b["tokens"].shape[0]

    result = {"micro_batch": micro_batch}
    try:
        for _ in range(args.warmup):
            one_step()
        torch.cuda.synchronize()
        # Peak from the warmup is the steady state; reset so the timed window
        # measures the same thing without allocator start-up noise.
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        n = 0
        for _ in range(args.steps):
            n += one_step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        result.update({
            "ok": True,
            "seconds": dt,
            "samples_per_s": n / dt,
            "steps_per_s": args.steps / dt,
            "tokens_per_s": n / dt * 68,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
        })
    except torch.cuda.OutOfMemoryError as e:
        result.update({"ok": False, "error": "OOM", "detail": str(e)[:200]})
    except RuntimeError as e:
        result.update({"ok": False, "error": "RuntimeError", "detail": str(e)[:200]})
    finally:
        try:
            it = None
            del loader
        except Exception:
            pass
        del model, optimizer
        torch.cuda.empty_cache()
    return result


def bench_loader(ds, micro_batch, args) -> float:
    """Loader-only throughput: does the dataloader still outrun the GPU?"""
    loader = make_loader(ds, micro_batch, args.workers, args.prefetch_factor,
                         args.mirror_prob)
    it = iter(loader)
    for _ in range(max(3, args.warmup // 2)):
        next(it)
    t0 = time.perf_counter()
    n = 0
    for _ in range(args.loader_steps):
        n += next(it)["tokens"].shape[0]
    dt = time.perf_counter() - t0
    del it, loader
    return n / dt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="VRAM / throughput sweep for train_v5.py micro-batch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--subset", type=int, default=600_000,
                   help="train records to sweep over (cycled)")
    p.add_argument("--micro-batch", type=int, nargs="+",
                   default=[128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
                            3072, 4096],
                   help="68 tokens per record means even 256 is a large batch "
                        "in token terms, so the low end is sampled finely")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--loader-steps", type=int, default=40)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--mirror-prob", type=float, default=0.5)
    p.add_argument("--vram-ceiling", type=float, default=0.90,
                   help="fraction of total VRAM that reserved must stay under")
    p.add_argument("--improve-tol", type=float, default=0.02,
                   help="samples/s must beat the best so far by this fraction "
                        "to keep sweeping; absorbs run-to-run scatter")
    p.add_argument("--no-early-stop", action="store_true", default=False,
                   help="measure every config even past the knee (diagnostics)")
    p.add_argument("--plateau-tol", type=float, default=0.03,
                   help="a samples/s drop within this fraction of the previous "
                        "config is a PLATEAU (saturated), not a spill")
    p.add_argument("--spill-vram", type=float, default=0.60,
                   help="reserved fraction above which a COLLAPSE may be "
                        "attributed to spilling rather than saturation")
    p.add_argument("--spill-drop", type=float, default=0.25,
                   help="minimum throughput collapse to call it a spill; a "
                        "PCIe spill is catastrophic, not a few percent")
    p.add_argument("--plateau", type=float, default=0.95,
                   help="recommend the smallest config reaching this fraction "
                        "of peak samples/s")
    p.add_argument("--accum-target", type=int, default=0,
                   help="if set, suggest accum-steps to reach this effective batch")
    p.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    p.add_argument("--tf32", dest="tf32", action="store_true", default=True)
    p.add_argument("--no-tf32", dest="tf32", action="store_false")
    p.add_argument("--matmul-precision", default="high",
                   choices=("highest", "high", "medium"))
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--nhead", type=int, default=6)
    p.add_argument("--dim-feedforward", type=int, default=1536)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        log("CUDA not available; this benchmark measures VRAM and is CUDA-only.")
        return 1

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_float32_matmul_precision(args.matmul_precision)

    device = torch.device("cuda")
    if args.amp == "off":
        amp_ctx = contextlib.nullcontext
    else:
        dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        def amp_ctx():
            return autocast(device_type="cuda", dtype=dtype)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    hist = manifest["post_selection_piece_histogram"]
    coverage = (sum(v["with_policy"] for v in hist.values())
                / sum(v["count"] for v in hist.values()))

    props = torch.cuda.get_device_properties(0)
    total = props.total_memory
    ds = MultiPVDataset(args.shards, split="train", subset=args.subset)

    model_probe = build_model(ModelConfig(
        d_model=args.d_model, num_layers=args.num_layers, nhead=args.nhead,
        dim_feedforward=args.dim_feedforward, head_dim=args.head_dim,
        dropout=args.dropout))

    log("=" * 100)
    log(f"bench_batch | {props.name} | {gib(total):.2f} GiB total | "
        f"driver ceiling {args.vram_ceiling * 100:.0f}% reserved")
    log(f"  model    {model_probe.num_parameters():,} params "
        f"(d_model={args.d_model}, layers={args.num_layers}, ff={args.dim_feedforward})")
    log(f"  data     {len(ds):,} records, mirror {args.mirror_prob}, "
        f"{args.workers} workers (persistent, pinned), prefetch {args.prefetch_factor}")
    log(f"  step     real: forward + masked KL(4096) + value MSE + backward + AdamW")
    log(f"  amp      {args.amp} | tf32 {args.tf32} | matmul {args.matmul_precision} "
        f"| compile {args.compile}")
    log(f"  timing   {args.warmup} warmup + {args.steps} timed per config")
    log("=" * 100)
    del model_probe

    header = (f"{'micro':>7} {'peak_alloc':>11} {'peak_resv':>11} {'%VRAM':>7} "
              f"{'samples/s':>11} {'steps/s':>9} {'tok/s':>10}  note")
    log(header)
    log("-" * len(header))

    rows = []
    prev_sps = 0.0
    knee_at = None
    for mb in sorted(args.micro_batch):
        r = bench_one(ds, mb, args, device, amp_ctx, coverage)
        rows.append(r)
        if not r["ok"]:
            log(f"{mb:>7} {'-':>11} {'-':>11} {'-':>7} {'-':>11} {'-':>9} "
                f"{'-':>10}  {r['error']} - stopping sweep")
            log(f"         {r['detail']}")
            knee_at = (mb, r["error"])
            break
        pct = r["peak_reserved"] / total
        r["pct_vram"] = pct
        notes = []
        if pct > args.vram_ceiling:
            notes.append(f"OVER {args.vram_ceiling * 100:.0f}% RESERVED")

        # Saturation and spilling look alike in a throughput column and must
        # not be conflated. Once the GPU is saturated a larger batch buys
        # nothing and usually gives back a few percent to worse tiling -
        # nothing is wrong. A WDDM spill differs in KIND: the driver pages to
        # shared system memory over PCIe rather than raising OOM, and
        # throughput falls by MULTIPLES. Calling a 5% wobble a spill because
        # reserved crossed some fraction is crying wolf, so a spill needs both
        # real memory pressure AND a real collapse.
        #
        # `stalled` compares against the BEST so far with a noise band:
        # run-to-run scatter here is 1-2%, and a bare "below the previous one"
        # rule ends the sweep on noise.
        drop = (1 - r["samples_per_s"] / prev_sps) if prev_sps > 0 else 0.0
        stalled = prev_sps > 0 and r["samples_per_s"] <= prev_sps * (1 + args.improve_tol)
        collapsed = drop > args.plateau_tol
        spill = drop > args.spill_drop and pct >= args.spill_vram
        plateau = stalled and not collapsed
        if spill:
            notes.append(f"THROUGHPUT KNEE - collapsed {drop * 100:.0f}% at "
                         f"{pct * 100:.0f}% reserved: spilling to shared memory "
                         f"(WDDM does not OOM); stopping sweep")
        elif collapsed:
            notes.append(f"throughput dropped {drop * 100:.0f}% at "
                         f"{pct * 100:.0f}% reserved - saturated, not spilling; "
                         f"stopping sweep")
        elif plateau:
            notes.append("PLATEAU - GPU already saturated at the previous "
                         "micro-batch; stopping sweep")
        r["over_ceiling"] = pct > args.vram_ceiling
        r["knee"] = bool(stalled)
        r["spill"] = bool(spill)
        r["plateau"] = bool(plateau)
        log(f"{mb:>7} {gib(r['peak_allocated']):>9.2f}Gi "
            f"{gib(r['peak_reserved']):>9.2f}Gi {pct * 100:>6.1f}% "
            f"{r['samples_per_s']:>11,.0f} {r['steps_per_s']:>9.2f} "
            f"{r['tokens_per_s'] / 1e6:>9.2f}M  {'; '.join(notes)}")
        if stalled and not args.no_early_stop:
            knee_at = (mb, "spill" if spill else
                       ("saturation" if collapsed else "plateau"))
            break
        prev_sps = max(prev_sps, r["samples_per_s"])

    ok_rows = [r for r in rows if r.get("ok")]
    if not ok_rows:
        log("\nNo configuration completed. Lower --micro-batch and retry.")
        return 1

    log("-" * len(header))

    # (a) largest config that stays under the reserved ceiling. A spilled
    #     config's memory numbers are meaningless, so it is excluded; a merely
    #     saturated one is not.
    under = [r for r in ok_rows if not r["over_ceiling"] and not r["spill"]]
    largest_safe = max(under, key=lambda r: r["micro_batch"]) if under else None

    # (b) smallest config reaching >= plateau of the peak observed throughput
    peak_sps = max(r["samples_per_s"] for r in ok_rows)
    plateau_pool = [r for r in ok_rows
                    if r["samples_per_s"] >= args.plateau * peak_sps
                    and not r["over_ceiling"] and not r["spill"]]
    smallest_fast = (min(plateau_pool, key=lambda r: r["micro_batch"])
                     if plateau_pool else largest_safe)

    log("\nRECOMMENDATION")
    if largest_safe:
        log(f"  (a) largest under {args.vram_ceiling * 100:.0f}% reserved : "
            f"micro-batch {largest_safe['micro_batch']} "
            f"({largest_safe['pct_vram'] * 100:.1f}% reserved, "
            f"{largest_safe['samples_per_s']:,.0f} samples/s)")
    else:
        log(f"  (a) nothing stayed under {args.vram_ceiling * 100:.0f}% reserved")
    log(f"  (b) smallest at >={args.plateau * 100:.0f}% of peak throughput : "
        f"micro-batch {smallest_fast['micro_batch']} "
        f"({smallest_fast['pct_vram'] * 100:.1f}% reserved, "
        f"{smallest_fast['samples_per_s']:,.0f} samples/s, "
        f"{smallest_fast['samples_per_s'] / peak_sps * 100:.1f}% of peak "
        f"{peak_sps:,.0f})")
    plateau_members = sorted(r["micro_batch"] for r in plateau_pool)
    log(f"\n  --> USE (b): micro-batch {smallest_fast['micro_batch']}")
    if args.accum_target and len(plateau_members) > 1:
        # The (b) rule's usual justification - "a bigger micro-batch costs
        # optimizer steps" - only holds when micro-batch IS the effective
        # batch. Once accumulation is targeting a fixed effective batch, every
        # micro-batch on the plateau yields the same throughput AND the same
        # number of optimizer steps, so the choice is decided by second-order
        # things instead: log volume, and distance from the spill cliff.
        log(f"      NOTE: --accum-target {args.accum_target} is set, so the "
            f"optimizer-step count is fixed by the")
        log(f"      effective batch, not by the micro-batch. Every config in "
            f"{plateau_members} is")
        log(f"      throughput-equivalent here; any of them trains identically. "
            f"Prefer a larger one")
        log(f"      to cut per-micro-batch log volume and accumulation "
            f"iterations, while staying well")
        log(f"      below the observed spill point.")
    log("      Accumulation only raises the EFFECTIVE batch, so a micro-batch "
        "past the throughput")
    log("      plateau buys no samples/s and costs optimizer steps per epoch. "
        "The leftover VRAM")
    log("      is not waste - it is headroom against allocator fragmentation "
        "and against the")
    log("      driver quietly paging to shared system memory, which on WDDM "
        "shows up as a")
    log("      throughput collapse rather than an OOM you could catch.")
    if knee_at:
        why = {"spill": "throughput collapsed under memory pressure (WDDM spill)",
               "saturation": "throughput fell with no memory pressure - saturated",
               "plateau": "throughput stopped rising - GPU already saturated",
               }.get(knee_at[1], knee_at[1])
        log(f"      Sweep stopped at micro-batch {knee_at[0]}: {why}.")

    rec = smallest_fast["micro_batch"]
    log(f"\nLoader-only throughput at micro-batch {rec} "
        f"({args.workers} workers, mirror {args.mirror_prob}):")
    loader_sps = bench_loader(ds, rec, args)
    gpu_sps = smallest_fast["samples_per_s"]
    log(f"  loader {loader_sps:,.0f} samples/s vs GPU {gpu_sps:,.0f} samples/s "
        f"-> {loader_sps / gpu_sps:.2f}x headroom "
        f"({'loader keeps up' if loader_sps > gpu_sps * 1.1 else 'LOADER IS THE BOTTLENECK'})")

    accum = 1
    if args.accum_target:
        accum = max(1, round(args.accum_target / rec))
    log(f"\nSuggested command line:")
    log(f"  python training/v5_multiPV/train_v5.py \\")
    log(f"      --config training/v5_multiPV/configs/base.yaml \\")
    log(f"      --micro-batch {rec} --accum-steps {accum} \\")
    log(f"      --workers {args.workers} --prefetch-factor {args.prefetch_factor} \\")
    log(f"      --amp {args.amp} {'--tf32' if args.tf32 else '--no-tf32'} "
        f"--matmul-precision {args.matmul_precision}"
        + (" --compile" if args.compile else "") + " \\")
    log(f"      # effective batch {rec * accum}; re-run --lr-range-test at this "
        f"effective batch before committing")

    if args.json_out:
        payload = {
            "device": props.name, "total_vram": total,
            "coverage": coverage, "args": {k: str(v) for k, v in vars(args).items()},
            "rows": rows, "recommended_micro_batch": rec,
            "recommended_accum": accum,
            "loader_samples_per_s": loader_sps,
            "peak_samples_per_s": peak_sps,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        log(f"\njson -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
