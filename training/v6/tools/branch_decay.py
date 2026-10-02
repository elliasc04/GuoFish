"""Run only the decay from a stable checkpoint (§8.2).

    python -m training.v6.tools.branch_decay models/v6/<run>/stable/s<N>.pt [--set KEY=VALUE ...]
        [--resume] [--seen-eval GROUP:N]

Resumes model, optimizer, EMA and sampler at the checkpoint's sample index N
with schedule.total_samples = ceil(N / (1 - decay_frac)), so the decay starts at N
and lasts decay_frac/(1 - decay_frac) x N samples (a branch at 240M with
decay_frac 0.2 decays for 60M). Output: <run>/branches/s<N>_d<decay>/final.pt,
with full evals of raw and EMA weights at the end. Overrides are limited to the
resume whitelist (the trainer enforces it). --resume continues an interrupted
branch from its own latest rolling checkpoint (<branch>/ckpt/s*.pt).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from training.v6.ckpt import latest_checkpoint
from training.v6.config import build_config, to_plain
from training.v6.config.loader import apply_override, merge
from training.v6.train import parse_seen, resolve, run


def branch_config(ck: dict, overrides=()):
    cfg = build_config(ck["config"])
    if cfg.schedule.kind != "wsd":
        raise SystemExit("decay branches need schedule.kind=wsd")
    s_b = int(ck["samples"])
    total = math.ceil(s_b / (1.0 - cfg.schedule.decay_frac))
    if total <= s_b:
        raise SystemExit("decay_frac gives no decay")
    d = merge(to_plain(cfg), {"schedule": {"total_samples": total}})
    for spec in overrides:
        d = apply_override(d, spec)
    return build_config(d), s_b, total - s_b


def branch_dir(cfg, s_b: int, decay: int) -> Path:
    return resolve(cfg.run.out_root) / cfg.run.name / "branches" / f"s{s_b}_d{decay}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", action="store_true", help="continue from the branch's latest rolling checkpoint")
    ap.add_argument("--seen-eval", default=None, type=parse_seen, metavar="GROUP:N")
    args = ap.parse_args(argv)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if ck["reason"] != "stable":
        raise SystemExit(f"{args.ckpt} is a {ck['reason']!r} checkpoint, not a stable one")
    cfg, s_b, decay = branch_config(ck, args.set)
    out = branch_dir(cfg, s_b, decay)
    if args.resume:
        resume = latest_checkpoint(out)
    elif out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    else:
        resume = args.ckpt
    run(cfg, run_dir=out, resume=resume, branch=True, seen_eval=args.seen_eval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
