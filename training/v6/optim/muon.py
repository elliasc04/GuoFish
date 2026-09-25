"""Muon (§8.1, experimental; `optim.name: muon_adamw`).

Momentum 0.95 with Nesterov, orthogonalised by 5 quintic Newton-Schulz
steps (Jordan's coefficients 3.4445, -4.7750, 2.0315). The update is scaled
by 0.2*sqrt(max(rows, cols)) so its RMS matches AdamW's (Liu et al. 2025,
"Muon is Scalable for LLM Training"), which lets the AdamW LR and decoupled
weight decay carry over unchanged.
"""
from __future__ import annotations

import torch

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(g: torch.Tensor, steps: int) -> torch.Tensor:
    a, b, c = NS_COEFFS
    x = g.bfloat16() if g.is_cuda else g.float()
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    if tall:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))
        for g in self.param_groups:
            for p in g["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon takes 2-D weights only, got shape {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Muon does not take a closure")
        for g in self.param_groups:
            mu, lr, wd = g["momentum"], g["lr"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("momentum_buffer", torch.zeros_like(p))
                buf.mul_(mu).add_(p.grad)
                u = p.grad.add(buf, alpha=mu) if g["nesterov"] else buf
                u = newton_schulz(u, g["ns_steps"]) * (0.2 * max(p.shape) ** 0.5)
                p.mul_(1.0 - lr * wd)
                p.add_(u, alpha=-lr)


class MuonAdamW:
    """Muon on the block matrices, AdamW on the rest; one step, one LR."""

    def __init__(self, muon: Muon, adamw: torch.optim.Optimizer):
        self.muon, self.adamw = muon, adamw

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def step(self):
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.muon.load_state_dict(sd["muon"])
        self.adamw.load_state_dict(sd["adamw"])
