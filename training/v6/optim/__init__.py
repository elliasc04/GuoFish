"""Optimizer construction (§8.1)."""
from __future__ import annotations

import re

import torch

from training.v6.config.schema import OptimConfig
from training.v6.optim.ema import EMA  # noqa: F401
from training.v6.optim.muon import Muon, MuonAdamW
from training.v6.optim.schedule import Schedule  # noqa: F401

MUON_PARAM = re.compile(r"blocks\.\d+\.(qkv|out|ff1|ff2|smolgen\.(compress|fc1|fc2))\.weight")


def no_decay(name: str, p: torch.Tensor, decay_embedding: bool) -> bool:
    """Biases, norms, the positional embedding and the static bias never decay;
    the token embedding decays only when `decay_embedding` (v5 compat)."""
    if p.ndim <= 1 or name in ("pos_embedding", "static_bias"):
        return True
    return name == "embedding.weight" and not decay_embedding


def param_groups(model, cfg: OptimConfig, exclude=()) -> list[dict]:
    decay, keep = [], []
    for name, p in model.named_parameters():
        if name in exclude or not p.requires_grad:
            continue
        (keep if no_decay(name, p, cfg.decay_embedding) else decay).append(p)
    return [{"params": decay, "weight_decay": cfg.weight_decay, "name": "decay"},
            {"params": keep, "weight_decay": 0.0, "name": "no_decay"}]


def build_optimizer(model, cfg: OptimConfig, initial_lr: float, muon_impl: str = "batched",
                    compile: bool = False):
    """muon_impl / compile: how the Muon update is computed (batched, compiled on CUDA, in
    production; `reference` for the parity gates). Not config: same update, same state."""
    adamw_kw = dict(lr=initial_lr, betas=tuple(cfg.betas), eps=cfg.eps, fused=True)
    if cfg.name == "adamw":
        return torch.optim.AdamW(param_groups(model, cfg), **adamw_kw)
    muon_names = {n for n, _ in model.named_parameters() if MUON_PARAM.fullmatch(n)}
    if not muon_names:
        raise ValueError("muon_adamw: no block matrices matched")
    muon = Muon([p for n, p in model.named_parameters() if n in muon_names], lr=initial_lr,
                momentum=cfg.muon.momentum, nesterov=cfg.muon.nesterov,
                ns_steps=cfg.muon.ns_steps, weight_decay=cfg.weight_decay,
                impl=muon_impl, compile=compile)
    adamw = torch.optim.AdamW(param_groups(model, cfg, exclude=muon_names), **adamw_kw)
    return MuonAdamW(muon, adamw)
