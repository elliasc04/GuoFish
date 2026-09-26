"""Micro-batch sweep: training throughput and peak VRAM per micro-batch (§8.4).

    python -m training.v6.tools.bench --config <cfg> [--set ...] --micro 256 512 1024

Times the real step (forward_train under the config's precision/compile,
the v6 loss, backward, clip, optimizer) on one real batch of each size read
from the configured corpus and kept on the device, so the number is model
throughput, not loader throughput. An OOM ends the sweep. Prints one JSON line
per micro-batch; pick the fastest micro x accum that gives the effective batch.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time

import numpy as np
import torch

from core.guofish_net import build_model
from training.v6.config import load_config
from training.v6.data.batch import BatchBuilder
from training.v6.data.reader import ShardSet
from training.v6.losses import LossFn
from training.v6.optim import build_optimizer
from training.v6.train import resolve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--micro", type=int, nargs="+", default=[256, 512, 768, 1024])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    dev = torch.device(cfg.system.device)
    torch.backends.cuda.matmul.allow_tf32 = cfg.system.tf32
    ss = ShardSet(resolve(cfg.data.corpus), cfg.data.split, resolve(cfg.data.manifest))
    t = cfg.targets
    builder = BatchBuilder(cfg.model.token_scheme, t.mirror_prob, cfg.run.data_seed,
                           t.policy_soft.epsilon, t.policy_soft.temperature)
    amp = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16))
           if cfg.system.precision == "bf16" else contextlib.nullcontext)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    results = []
    for mb in args.micro:
        idx = np.random.default_rng(0).choice(len(ss), mb, replace=False)
        b = {k: v.to(dev) for k, v in builder(ss.read(idx), idx).items()}
        torch.manual_seed(cfg.run.init_seed)
        model = build_model(cfg.model, cfg.run.init_seed).to(dev)
        fn = (torch.compile(model.forward_train, mode=cfg.system.compile_mode)
              if cfg.system.compile else model.forward_train)
        opt = build_optimizer(model, cfg.optim, cfg.optim.lr)
        loss_fn = LossFn(cfg, {"soft": mb * 0.6, "hard": mb * 0.25, "value": float(mb)}, dev)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            for i in range(args.warmup + args.steps):
                if i == args.warmup:
                    sync()
                    t0 = time.perf_counter()
                with amp():
                    out = fn(b["tokens"])
                loss, _ = loss_fn(out, b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            sync()
            el = time.perf_counter() - t0
        except torch.OutOfMemoryError:
            results.append({"micro_batch": mb, "oom": True})
            print(json.dumps(results[-1]), flush=True)
            break
        r = {"micro_batch": mb, "samples_per_s": mb * args.steps / el,
             "step_ms": 1000 * el / args.steps,
             "peak_vram_mib": torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else None,
             "compile": cfg.system.compile, "precision": cfg.system.precision,
             "shape": f"d{cfg.model.d_model}x{cfg.model.n_layers}", "device": dev.type}
        results.append(r)
        print(json.dumps(r), flush=True)
        del model, opt, fn, b
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    ss.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
