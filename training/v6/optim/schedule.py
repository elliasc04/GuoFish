"""LR schedules in samples (§8.2). `s` is the sample index at the START of an
optimizer step (a multiple of the effective batch); lr(s) is that step's LR.

wsd       lr = peak * s/W during warmup (0 at step 0), peak while stable, then
          over the last decay_frac of total_samples decays to final_lr_frac*peak
          with f(p) = 1-sqrt(p) | 1-p | (1+cos(pi p))/2, p in [0, 1].
onecycle  torch.optim.lr_scheduler.OneCycleLR (two-phase, cos) evaluated at
          step k = s / eff with total_steps = total_samples / eff, including
          its beta1 cycling. Closed-form, so resume needs no scheduler state.
"""
from __future__ import annotations

import math

from training.v6.config.schema import ScheduleConfig


def _cos(start, end, pct):
    return end + (start - end) / 2.0 * (math.cos(math.pi * pct) + 1)


class Schedule:
    def __init__(self, cfg: ScheduleConfig, peak_lr: float, effective_batch: int):
        self.cfg, self.peak, self.eff = cfg, float(peak_lr), int(effective_batch)
        self.total = int(cfg.total_samples)
        self.total_steps = -(-self.total // self.eff)          # ceil
        if cfg.kind == "wsd":
            self.decay_start = self.total - cfg.decay_frac * self.total
            self.decay_len = cfg.decay_frac * self.total
        else:
            oc = cfg.onecycle
            if self.total % self.eff:
                raise ValueError("onecycle total_samples must be a multiple of the effective batch")
            self.initial_lr = self.peak / oc.div_factor
            self.min_lr = self.initial_lr / oc.final_div_factor
            T = self.total // self.eff
            self.phases = [
                (float(oc.pct_start * T) - 1, self.initial_lr, self.peak, oc.max_momentum, oc.base_momentum),
                (T - 1, self.peak, self.min_lr, oc.base_momentum, oc.max_momentum),
            ]

    def describe(self) -> dict:
        d = {"kind": self.cfg.kind, "total_samples": self.total, "effective_batch": self.eff,
             "total_steps": self.total_steps}
        if self.cfg.kind == "wsd":
            d.update(warmup_samples=self.cfg.warmup_samples,
                     warmup_steps=-(-self.cfg.warmup_samples // self.eff),
                     decay_start_samples=self.decay_start, decay_samples=self.decay_len,
                     stable_end_step=int(self.decay_start // self.eff))
        else:
            d.update(initial_lr=self.initial_lr, min_lr=self.min_lr,
                     phase1_end_step=self.phases[0][0])
        return d

    def _onecycle(self, s: int):
        k = s // self.eff
        if k > self.total // self.eff:
            raise ValueError(f"step {k} past the OneCycle length")
        start = 0.0
        for i, (end, lr0, lr1, m0, m1) in enumerate(self.phases):
            if k <= end or i == len(self.phases) - 1:
                pct = (k - start) / (end - start)
                return _cos(lr0, lr1, pct), _cos(m0, m1, pct)
            start = end

    def lr(self, s: int) -> float:
        if s % self.eff:
            raise ValueError(f"sample index {s} is not on a step boundary")
        if self.cfg.kind == "onecycle":
            return self._onecycle(s)[0]
        c = self.cfg
        if s < c.warmup_samples:
            return self.peak * s / c.warmup_samples
        if s < self.decay_start or self.decay_len == 0:
            return self.peak
        p = min(1.0, (s - self.decay_start) / self.decay_len)
        f = {"one_minus_sqrt": 1.0 - math.sqrt(p), "linear": 1.0 - p,
             "cosine": 0.5 * (1.0 + math.cos(math.pi * p))}[c.decay_shape]
        return self.peak * (c.final_lr_frac + (1.0 - c.final_lr_frac) * f)

    def beta1(self, s: int):
        """OneCycle momentum cycling; None when beta1 is held constant (H4)."""
        if self.cfg.kind == "onecycle" and self.cfg.onecycle.cycle_momentum:
            return self._onecycle(s)[1]
        return None

    def phase(self, s: int) -> str:
        if self.cfg.kind == "onecycle":
            return "onecycle"
        if s < self.cfg.warmup_samples:
            return "warmup"
        return "stable" if s < self.decay_start else "decay"
