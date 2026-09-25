"""EMA of the weights (§8.3): fp32, updated every optimizer step.

decay per step = 0.5 ** (effective_batch / half_life_samples), so the EMA
forgets half its past after half_life_samples samples whatever the batch.
"""
from __future__ import annotations

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, half_life_samples: float, effective_batch: int):
        self.decay = 0.5 ** (effective_batch / half_life_samples)
        self.names = [n for n, _ in model.named_parameters()]
        self.shadow = [p.detach().float().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        params = [p.detach().float() for p in model.parameters()]
        torch._foreach_lerp_(self.shadow, params, 1.0 - self.decay)

    def state_dict(self) -> dict:
        return {n: t for n, t in zip(self.names, self.shadow)}

    def load_state_dict(self, sd: dict) -> None:
        if list(sd) != self.names:
            raise KeyError("EMA state does not match the model's parameters")
        for t, n in zip(self.shadow, self.names):
            t.copy_(sd[n])

    def weights_for(self, model: torch.nn.Module) -> dict:
        """A full state dict for `model` with EMA parameters (buffers from model)."""
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        for n, t in zip(self.names, self.shadow):
            sd[n] = t.to(sd[n].dtype)
        return sd
