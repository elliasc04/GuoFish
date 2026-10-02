"""Inference cost of an architecture arm (H5), relative to plain d384x10.

    python -m training.v6.tools.fwd_cost --arm A6=<config.yaml> [--ref A1=<config.yaml>]

The capacity campaign's A1 method (tools/bench_c12b.py --sections forward): the
engine's forward, Inductor-compiled in default mode and captured as one CUDA
graph per batch size, bf16 autocast, 20 warm-up then 200 timed replays, CUDA
events, device time per replay. Timed at batch 24 (the engine's operating batch)
and 128, on real frozen-val positions in each model's token scheme, random
weights (cost is shape-only). Plain d384x10 is always measured alongside; all
models are re-timed over --rounds interleaved rounds and the median is kept.
Every model is reported against plain; the arm also against --ref (default
plain), which is the marginal cost its screening verdict uses.

Required KL gain, the campaign's A4 exchange (CAMPAIGN_RECORD §7): sims at equal
time scale as 1 / (f*r + 1 - f), f = 0.908 the GPU-bound share, r the batch-24
forward ratio; doublings lost = log2(f*r + 1 - f); required gain = doublings x
6.50% KL per doubling. Prints one JSON object; the GPU must be otherwise idle.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics

import numpy as np
import torch

from core.guofish_net import build_model
from training.v6.config import load_config
from training.v6.data.batch import BatchBuilder
from training.v6.data.formats import REPO
from training.v6.data.reader import ShardSet

F_GPU = 0.908            # GPU-bound share of search time (campaign T1, fresh root, K=24)
KL_PER_DOUBLING = 6.50   # % relative policy KL per doubling of search (campaign §7.1)
BATCHES = (24, 128)
PLAIN = REPO / "training/v6/config/configs/shapes/d384x10.yaml"
FROZEN = REPO / "data/processed/val_frozen_90m_v2"


def required_gain(ratio: float) -> float:
    """% relative KL an arm costing `ratio` x its reference must gain to break even."""
    return math.log2(1.0 + F_GPU * (ratio - 1.0)) * KL_PER_DOUBLING


class Timed:
    """One model: compiled once, one captured graph per batch size."""

    def __init__(self, model_cfg, dev):
        ss = ShardSet(FROZEN, "val")
        idx = np.sort(np.random.default_rng(20260926).choice(len(ss), max(BATCHES), replace=False))
        tokens = BatchBuilder(model_cfg.token_scheme, 0.0, 0, 0.05, None)(ss.read(idx), idx)["tokens"]
        ss.close()
        model = build_model(model_cfg).to(dev).eval()
        fn = torch.compile(model, dynamic=False)
        self.graphs = {}
        for b in BATCHES:
            static = tokens[:b].to(dev).clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(3):
                    fn(static)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                fn(static)
            self.graphs[b] = (g, static)

    def time_us(self, b: int, warm: int = 20, timed: int = 200) -> float:
        g = self.graphs[b][0]
        for _ in range(warm):
            g.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(timed):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / timed * 1000.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, metavar="NAME=CONFIG")
    ap.add_argument("--ref", default=None, metavar="NAME=CONFIG")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args(argv)
    # Engine settings (playing/v6/graphs.configure_inductor): the static launcher
    # fails on this machine; pointwise autotuning makes kernels clock-dependent.
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config
    inductor_config.use_static_cuda_launcher = False
    inductor_config.triton.autotune_pointwise = False
    inductor_config.triton.descriptive_names = False     # MAX_PATH, as in train.py
    dynamo_config.cache_size_limit = dynamo_config.accumulated_cache_size_limit = 256
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")

    specs = {"plain_d384x10": PLAIN}
    for m in filter(None, (args.arm, args.ref)):
        name, _, path = m.partition("=")
        if not name or not path or name in specs:
            raise SystemExit(f"{m!r}: expected NAME=CONFIG with a new name")
        specs[name] = path
    arm = args.arm.partition("=")[0]
    ref = args.ref.partition("=")[0] if args.ref else "plain_d384x10"
    timed = {n: Timed(load_config(p).model, dev) for n, p in specs.items()}
    raw = {n: {b: [] for b in BATCHES} for n in specs}
    for _ in range(args.rounds):
        for n, t in timed.items():
            for b in BATCHES:
                raw[n][b].append(t.time_us(b))
    us = {n: {b: statistics.median(v) for b, v in per.items()} for n, per in raw.items()}
    plain = us["plain_d384x10"]
    out = {"method": "compiled (default) + CUDA graph, bf16, 20 warm + 200 timed replays, "
                     f"median of {args.rounds} interleaved rounds", "f_gpu": F_GPU,
           "kl_per_doubling": KL_PER_DOUBLING, "gpu": torch.cuda.get_device_name(0), "models": {}}
    for n in specs:
        r = {b: us[n][b] / plain[b] for b in BATCHES}
        out["models"][n] = {"config": str(specs[n]), **{f"us_b{b}": us[n][b] for b in BATCHES},
                            **{f"ratio_b{b}": r[b] for b in BATCHES},
                            "required_kl_gain_pct": required_gain(r[24]),
                            "spread_b24_pct": 100 * (max(raw[n][24]) - min(raw[n][24])) / us[n][24]}
    r = {b: us[arm][b] / us[ref][b] for b in BATCHES}
    out["arm_vs_ref"] = {"arm": arm, "ref": ref, **{f"ratio_b{b}": r[b] for b in BATCHES},
                         "required_kl_gain_pct": required_gain(r[24])}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
