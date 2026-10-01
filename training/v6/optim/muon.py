"""Muon (§8.1, experimental; `optim.name: muon_adamw`).

Momentum 0.95 with Nesterov, orthogonalised by 5 quintic Newton-Schulz
steps (Jordan's coefficients 3.4445, -4.7750, 2.0315). The update is scaled
by 0.2*sqrt(max(rows, cols)) so its RMS matches AdamW's (Liu et al. 2025,
"Muon is Scalable for LLM Training"), which lets the AdamW LR and decoupled
weight decay carry over unchanged.

Two implementations of the same update (VM harness follow-ups §1):
  reference  one matrix at a time (the screening runs' code; kept for the gates)
  batched    matrices grouped by shape after orienting tall ones wide (as the
             reference does), Newton-Schulz once per group as batched matmuls,
             momentum and the parameter update as foreach ops; on CUDA the
             whole step is one torch.compile graph. Same dtype, iterations,
             coefficients, momentum, Nesterov and scale rule; per-parameter
             `momentum_buffer` state, so checkpoints are interchangeable.
"""
from __future__ import annotations

import functools

import torch

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def ns_dtype_for(t: torch.Tensor, ns_dtype=None) -> torch.dtype:
    return ns_dtype or (torch.bfloat16 if t.is_cuda else torch.float32)


def newton_schulz(g: torch.Tensor, steps: int, ns_dtype=None) -> torch.Tensor:
    a, b, c = NS_COEFFS
    x = g.to(ns_dtype_for(g, ns_dtype))
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


def newton_schulz_batched(x: torch.Tensor, steps: int) -> torch.Tensor:
    """x: (B, r, c) with r <= c, already in the Newton-Schulz dtype; per-matrix norms."""
    a, b, c = NS_COEFFS
    x = x / (torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * s @ s) @ x
    return x


def plan_groups(params) -> tuple:
    """((indices, tall flags, scale), ...) grouping matrices by oriented (wide) shape."""
    groups: dict[tuple, list] = {}
    for i, p in enumerate(params):
        r, c = p.shape
        groups.setdefault((min(r, c), max(r, c)), []).append((i, r > c))
    return tuple((tuple(i for i, _ in m), tuple(t for _, t in m), 0.2 * max(shape) ** 0.5)
                 for shape, m in sorted(groups.items()))


def batched_step(params, grads, bufs, lr, mult, skip, *, mu, nesterov, steps, ns_dtype, plan):
    """One Muon step for one param group. lr and mult (= 1 - lr*wd, computed in double on
    the host, as the reference's Python floats are): floats (eager) or 0-d tensors in the
    params' dtype (compiled). skip: 0-d tensor, nonzero = leave params and momentum untouched."""
    new_bufs = torch._foreach_add(torch._foreach_mul(bufs, mu), grads)
    us = torch._foreach_add(grads, torch._foreach_mul(new_bufs, mu)) if nesterov else new_bufs
    upd = [None] * len(params)
    for idx, tall, scale in plan:
        x = torch.stack([us[i].mT if t else us[i] for i, t in zip(idx, tall)]).to(ns_dtype)
        x = newton_schulz_batched(x, steps)
        for j, (i, t) in enumerate(zip(idx, tall)):
            upd[i] = (x[j].mT if t else x[j]).to(params[i].dtype) * scale
    new_ps = torch._foreach_mul(params, mult)
    new_ps = torch._foreach_sub(new_ps, torch._foreach_mul(upd, lr))
    keep = skip > 0
    for p, b, np_, nb in zip(params, bufs, new_ps, new_bufs):
        p.copy_(torch.where(keep, p, np_))
        b.copy_(torch.where(keep, b, nb))


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0, impl: str = "batched",
                 compile: bool = False, ns_dtype=None):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))
        if impl not in ("batched", "reference"):
            raise ValueError(f"Muon impl {impl!r}; expected batched | reference")
        for g in self.param_groups:
            for p in g["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon takes 2-D weights only, got shape {tuple(p.shape)}")
        self.impl, self.compile, self.ns_dtype = impl, compile, ns_dtype
        self._fns: dict[int, object] = {}      # param group index -> (compiled) step

    @torch.no_grad()
    def step(self, closure=None):
        """Honours `self.found_inf` (a 0/1 device tensor, the GradScaler
        protocol fused AdamW uses): when 1, params and momentum are left
        untouched, decided on device with no host sync."""
        if closure is not None:
            raise ValueError("Muon does not take a closure")
        skip = getattr(self, "found_inf", None)
        (self._step_batched if self.impl == "batched" else self._step_reference)(skip)

    def _step_reference(self, skip):
        for g in self.param_groups:
            mu, lr, wd = g["momentum"], g["lr"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("momentum_buffer", torch.zeros_like(p))
                new_buf = buf * mu + p.grad
                u = p.grad + mu * new_buf if g["nesterov"] else new_buf
                u = newton_schulz(u, g["ns_steps"], self.ns_dtype) * (0.2 * max(p.shape) ** 0.5)
                new_p = p * (1.0 - lr * wd) - lr * u
                if skip is None:
                    buf.copy_(new_buf)
                    p.copy_(new_p)
                else:
                    buf.copy_(torch.where(skip > 0, buf, new_buf))
                    p.copy_(torch.where(skip > 0, p, new_p))

    def _step_batched(self, skip):
        for gi, g in enumerate(self.param_groups):
            params = [p for p in g["params"] if p.grad is not None]
            if not params:
                continue
            if len(params) != len(g["params"]):
                raise ValueError("batched Muon needs a gradient for every matrix of a group")
            bufs = [self.state[p].setdefault("momentum_buffer", torch.zeros_like(p)) for p in params]
            dev = params[0].device
            fn = self._fns.get(gi)
            if fn is None:
                fn = functools.partial(batched_step, mu=g["momentum"], nesterov=g["nesterov"],
                                       steps=g["ns_steps"],
                                       ns_dtype=ns_dtype_for(params[0], self.ns_dtype),
                                       plan=plan_groups(params))
                if self.compile:
                    fn = torch.compile(fn, fullgraph=True, dynamic=False)
                    self._lr = torch.zeros((), device=dev, dtype=params[0].dtype)
                    self._mult = torch.zeros((), device=dev, dtype=params[0].dtype)
                    self._zero = torch.zeros((), device=dev)
                self._fns[gi] = fn
            sk = skip if skip is not None else (self._zero if self.compile else torch.zeros((), device=dev))
            lr, mult = g["lr"], 1.0 - g["lr"] * g["weight_decay"]
            if self.compile:                   # tensors: a new LR must not recompile
                self._lr.fill_(lr)
                self._mult.fill_(mult)
                lr, mult = self._lr, self._mult
            fn(params, [p.grad for p in params], bufs, lr, mult, sk)


class MuonAdamW:
    """Muon on the block matrices, AdamW on the rest; one step, one LR."""

    def __init__(self, muon: Muon, adamw: torch.optim.Optimizer):
        self.muon, self.adamw = muon, adamw

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    @property
    def found_inf(self):
        return self.adamw.found_inf

    @found_inf.setter
    def found_inf(self, flag):
        self.muon.found_inf = flag
        self.adamw.found_inf = flag
        self.adamw.grad_scale = None

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
