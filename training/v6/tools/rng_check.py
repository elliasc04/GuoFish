"""G2 follow-up: does a fresh process that restores the checkpointed RNG state
reproduce the compiled, dropout-on forward bit for bit? (The resume path's only
GPU-specific state; S3 on CPU already covers the rest.)

    python -m training.v6.tools.rng_check --config training/v6/config/configs/v5_compat.yaml --out <dir>

Process 1 (this one) builds the model, compiles forward_train, runs one
warm-up call, then saves weights + torch and CUDA RNG state the way the
checkpoint does and runs the forward twice (train mode, bf16): A1, then A2
without restoring (positive control: dropout must change the output).
Process 2 (a subprocess) loads the weights, restores the RNG state, compiles
and runs the same forward once: B. Pass: B == A1 exactly and A2 != A1.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from core.guofish_net import build_model
from training.v6.config import load_config
from training.v6.data.batch import BatchBuilder
from training.v6.data.reader import ShardSet
from training.v6.train import resolve  # also sets the Triton kernel-name fix


def forward(cfg, model, tokens):
    fn = torch.compile(model.forward_train, mode=cfg.system.compile_mode)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return fn, lambda: fn(tokens)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--child", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, ["system.allow_dirty=true"])
    torch.backends.cuda.matmul.allow_tf32 = cfg.system.tf32
    dev = torch.device("cuda")
    ss = ShardSet(resolve(cfg.eval.frozen_dir), "val")
    idx = np.arange(512)
    tokens = BatchBuilder("v5_68", 0.0, 0, 0.05, None)(ss.read(idx), idx)["tokens"].to(dev)
    model = build_model(cfg.model).to(dev).train()
    fn = torch.compile(model.forward_train, mode=cfg.system.compile_mode)

    def run():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fn(tokens)
        return torch.cat([out["policy_logits"].float().flatten(), out["value"].float().flatten()]).cpu()

    if args.child:
        st = torch.load(args.out / "state.pt", weights_only=True)
        model.load_state_dict(st["model"])
        torch.set_rng_state(st["rng"]["torch"])
        torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        torch.save(run(), args.out / "B.pt")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.run.seed)
    run()                                                      # compile + warm-up
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}},
               args.out / "state.pt")
    a1, a2 = run(), run()
    subprocess.run([sys.executable, "-m", "training.v6.tools.rng_check", "--config", args.config,
                    "--out", str(args.out), "--child"], check=True)
    b = torch.load(args.out / "B.pt", weights_only=True)
    res = {"b_equals_a1": bool(torch.equal(b, a1)), "max_abs_b_minus_a1": float((b - a1).abs().max()),
           "a2_differs_from_a1": not torch.equal(a2, a1), "max_abs_a2_minus_a1": float((a2 - a1).abs().max()),
           "n": int(a1.numel())}
    res["passed"] = res["b_equals_a1"] and res["a2_differs_from_a1"]
    print(json.dumps(res, indent=1))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
