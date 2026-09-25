"""v6 config schema (§4): one frozen dataclass per section, validated on build.

Every field has a default. Errors name the full key path (the loader builds
through core.guofish_net.strict.from_dict_strict). Cross-section rules live in
Config.__post_init__.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from core.guofish_net.model import ModelConfig
from core.guofish_net.strict import from_dict_strict

STRATA_FIELDS = {
    "bucket": ("le5", "6_14", "15_27", "ge28"),
    "label": ("multipv", "hard_only", "value_only"),
    "value": ("exact_zero", "mate", "middle"),
    "material": ("level", "ahead", "compensated"),
    "origin": ("root", "derived"),
}


def _enum(path, v, allowed):
    if v not in allowed:
        raise ValueError(f"{path}={v!r}; expected one of {sorted(allowed)}")


def _pos(path, v):
    if v <= 0:
        raise ValueError(f"{path}={v!r}; must be > 0")


def _nonneg(path, v):
    if v < 0:
        raise ValueError(f"{path}={v!r}; must be >= 0")


def _unit(path, v, lo_open=False):
    if not ((0.0 < v if lo_open else 0.0 <= v) and v <= 1.0):
        raise ValueError(f"{path}={v!r}; must be in {'(' if lo_open else '['}0, 1]")


@dataclass(frozen=True)
class RunConfig:
    name: str = "v6-base"
    seed: int = 20260924
    out_root: str = "models/v6"

    def __post_init__(self):
        if not self.name or any(c in self.name for c in '/\\:*?"<>| '):
            raise ValueError(f"run.name={self.name!r} must be a plain directory name")


@dataclass(frozen=True)
class DataConfig:
    corpus: str = "data/processed/multipv_90m"
    manifest: str = "data/multiPV/manifests/dataset_manifest_90m.json"
    split: str = "train"
    strata: str = "data/processed/multipv_90m/strata_train_v1.npy"
    workers: int = 8
    prefetch_factor: int = 2

    def __post_init__(self):
        _nonneg("data.workers", self.workers)
        _pos("data.prefetch_factor", self.prefetch_factor)


@dataclass(frozen=True)
class GroupConfig:
    name: str
    where: dict[str, Any]
    share: float

    def __post_init__(self):
        if not self.where:
            raise ValueError("where must name at least one strata field")
        for k, v in self.where.items():
            if k not in STRATA_FIELDS:
                raise KeyError(f"unknown strata field where.{k}; expected one of {sorted(STRATA_FIELDS)}")
            vals = v if isinstance(v, (list, tuple)) else [v]
            for x in vals:
                _enum(f"where.{k}", x, STRATA_FIELDS[k])
        _unit("share", self.share, lo_open=True)
        if self.name == "rest":
            raise ValueError("'rest' is reserved for unmatched records")


@dataclass(frozen=True)
class MixtureConfig:
    groups: Any = "natural"      # "natural" | list of {name, where, share}

    def __post_init__(self):
        if self.groups == "natural":
            return
        if not isinstance(self.groups, (list, tuple)) or not self.groups:
            raise ValueError("mixture.groups must be 'natural' or a non-empty list")
        built = tuple(from_dict_strict(GroupConfig, g, f"mixture.groups[{i}]")
                      for i, g in enumerate(self.groups))
        names = [g.name for g in built]
        if len(set(names)) != len(names):
            raise ValueError(f"mixture.groups: duplicate names {names}")
        total = math.fsum(g.share for g in built)
        if total > 1.0 + 1e-12:
            raise ValueError(f"mixture.groups: shares sum to {total} > 1")
        object.__setattr__(self, "groups", built)

    @property
    def rest_share(self) -> float:
        if self.groups == "natural":
            return 1.0
        return max(0.0, 1.0 - math.fsum(g.share for g in self.groups))


@dataclass(frozen=True)
class PolicySoftConfig:
    weight: float = 1.0
    source: str = "stored"               # stored pv_prob | rebuilt from pv_score
    temperature: Optional[float] = None
    epsilon: float = 0.05

    def __post_init__(self):
        _nonneg("weight", self.weight)
        _enum("source", self.source, {"stored", "pv_score"})
        if (self.source == "pv_score") != (self.temperature is not None):
            raise ValueError("temperature is required with source=pv_score and "
                             "must be null with source=stored")
        if self.temperature is not None:
            _pos("temperature", self.temperature)
        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError(f"epsilon={self.epsilon}; must be in [0, 1)")


@dataclass(frozen=True)
class PolicyHardConfig:
    weight: float = 0.0
    epsilon: float = 0.1
    head: str = "main"                   # main | aux

    def __post_init__(self):
        _nonneg("weight", self.weight)
        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError(f"epsilon={self.epsilon}; must be in [0, 1)")
        _enum("head", self.head, {"main", "aux"})


@dataclass(frozen=True)
class ValueTargetConfig:
    weight: float = 1.0
    stratum_weights: dict[str, float] = field(
        default_factory=lambda: {"exact_zero": 1.0, "mate": 1.0, "middle": 1.0})

    def __post_init__(self):
        _nonneg("weight", self.weight)
        if set(self.stratum_weights) != set(STRATA_FIELDS["value"]):
            raise ValueError(f"stratum_weights must name exactly {STRATA_FIELDS['value']}")
        for k, v in self.stratum_weights.items():
            _nonneg(f"stratum_weights.{k}", v)


@dataclass(frozen=True)
class TargetsConfig:
    policy_soft: PolicySoftConfig = field(default_factory=PolicySoftConfig)
    policy_hard: PolicyHardConfig = field(default_factory=PolicyHardConfig)
    value: ValueTargetConfig = field(default_factory=ValueTargetConfig)
    mirror_prob: float = 0.5

    def __post_init__(self):
        _unit("targets.mirror_prob", self.mirror_prob)


@dataclass(frozen=True)
class MuonConfig:
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5

    def __post_init__(self):
        _unit("momentum", self.momentum)
        _pos("ns_steps", self.ns_steps)


@dataclass(frozen=True)
class OptimConfig:
    name: str = "adamw"                  # adamw | muon_adamw
    lr: float = 3.5e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    decay_embedding: bool = False        # v5 decayed embedding.weight
    grad_clip: float = 1.0
    micro_batch: int = 512
    accum: int = 2
    muon: MuonConfig = field(default_factory=MuonConfig)

    def __post_init__(self):
        _enum("optim.name", self.name, {"adamw", "muon_adamw"})
        _pos("optim.lr", self.lr)
        for i, b in enumerate(self.betas):
            if not 0.0 <= b < 1.0:
                raise ValueError(f"optim.betas[{i}]={b}; must be in [0, 1)")
        _pos("optim.eps", self.eps)
        _nonneg("optim.weight_decay", self.weight_decay)
        _pos("optim.grad_clip", self.grad_clip)
        _pos("optim.micro_batch", self.micro_batch)
        _pos("optim.accum", self.accum)

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.accum


@dataclass(frozen=True)
class OneCycleConfig:
    pct_start: float = 0.1
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    cycle_momentum: bool = True
    base_momentum: float = 0.85
    max_momentum: float = 0.95

    def __post_init__(self):
        _unit("pct_start", self.pct_start, lo_open=True)
        _pos("div_factor", self.div_factor)
        _pos("final_div_factor", self.final_div_factor)


@dataclass(frozen=True)
class ScheduleConfig:
    kind: str = "wsd"                    # wsd | onecycle
    total_samples: int = 360_000_000
    warmup_samples: int = 7_200_000
    decay_frac: float = 0.2
    decay_shape: str = "one_minus_sqrt"  # one_minus_sqrt | linear | cosine
    final_lr_frac: float = 0.0
    onecycle: OneCycleConfig = field(default_factory=OneCycleConfig)

    def __post_init__(self):
        _enum("schedule.kind", self.kind, {"wsd", "onecycle"})
        _pos("schedule.total_samples", self.total_samples)
        _nonneg("schedule.warmup_samples", self.warmup_samples)
        _unit("schedule.decay_frac", self.decay_frac)
        _enum("schedule.decay_shape", self.decay_shape, {"one_minus_sqrt", "linear", "cosine"})
        _unit("schedule.final_lr_frac", self.final_lr_frac)
        if self.kind == "wsd" and (self.warmup_samples
                                   + self.decay_frac * self.total_samples > self.total_samples):
            raise ValueError("schedule: warmup_samples + decay_frac*total_samples > total_samples")


@dataclass(frozen=True)
class EmaConfig:
    enabled: bool = True
    half_life_samples: float = 10e6

    def __post_init__(self):
        _pos("ema.half_life_samples", self.half_life_samples)


@dataclass(frozen=True)
class EvalConfig:
    frozen_dir: str = "data/processed/val_frozen_90m_v1"
    frozen_strata: str = "data/processed/val_frozen_90m_v1/strata_val_v1.npy"
    quick_every_samples: int = 2_048_000
    quick_size: int = 32_768
    quick_seed: int = 20260924
    full_every_samples: int = 0          # 0: stable checkpoints and run end only
    batch: int = 1024
    workers: int = 4
    best_metric: str = "total"           # total | policy_kl | value_mse
    mirror_n: int = 10_000

    def __post_init__(self):
        _nonneg("eval.workers", self.workers)
        _nonneg("eval.quick_every_samples", self.quick_every_samples)
        _pos("eval.quick_size", self.quick_size)
        _nonneg("eval.full_every_samples", self.full_every_samples)
        _pos("eval.batch", self.batch)
        _enum("eval.best_metric", self.best_metric, {"total", "policy_kl", "value_mse"})
        _nonneg("eval.mirror_n", self.mirror_n)


@dataclass(frozen=True)
class CkptConfig:
    every_samples: int = 10_240_000
    keep_last: int = 3
    stable_every_samples: int = 0        # 0: no stable checkpoints
    export_weights: str = "ema"          # ema | raw | best

    def __post_init__(self):
        _pos("ckpt.every_samples", self.every_samples)
        _pos("ckpt.keep_last", self.keep_last)
        _nonneg("ckpt.stable_every_samples", self.stable_every_samples)
        _enum("ckpt.export_weights", self.export_weights, {"ema", "raw", "best"})


@dataclass(frozen=True)
class SystemConfig:
    device: str = "cuda"
    precision: str = "bf16"              # bf16 | fp32
    compile: bool = True
    compile_mode: str = "default"        # default | max-autotune-no-cudagraphs
    tf32: bool = True
    allow_dirty: bool = False
    log_every: int = 50                  # optimizer steps

    def __post_init__(self):
        _enum("system.device", self.device, {"cuda", "cpu"})
        _enum("system.precision", self.precision, {"bf16", "fp32"})
        _enum("system.compile_mode", self.compile_mode, {"default", "max-autotune-no-cudagraphs"})
        _pos("system.log_every", self.log_every)


@dataclass(frozen=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mixture: MixtureConfig = field(default_factory=MixtureConfig)
    targets: TargetsConfig = field(default_factory=TargetsConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ckpt: CkptConfig = field(default_factory=CkptConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def __post_init__(self):
        if self.model.token_scheme == "canonical_65" and self.targets.mirror_prob != 0.0:
            raise ValueError("targets.mirror_prob must be 0 under model.token_scheme="
                             "canonical_65 (mirroring is a no-op there)")
        if (self.targets.policy_hard.head == "aux") != self.model.aux_policy_head:
            raise ValueError("targets.policy_hard.head=aux requires model.aux_policy_head=true "
                             "and vice versa")
        eff = self.optim.effective_batch
        if self.schedule.kind == "onecycle" and self.schedule.total_samples % eff:
            raise ValueError(f"schedule.total_samples={self.schedule.total_samples} must be a "
                             f"multiple of the effective batch {eff} for onecycle")
        if self.mixture.groups != "natural":
            for g in self.mixture.groups:
                if g.share * self.optim.micro_batch < 1.0:
                    raise ValueError(f"mixture group {g.name}: share*micro_batch < 1 sample")
        for name in ("every_samples", "stable_every_samples"):
            v = getattr(self.ckpt, name)
            if v % eff:
                raise ValueError(f"ckpt.{name}={v} must be a multiple of the effective batch {eff}")
        for name in ("quick_every_samples", "full_every_samples"):
            v = getattr(self.eval, name)
            if v % eff:
                raise ValueError(f"eval.{name}={v} must be a multiple of the effective batch {eff}")
        if self.system.device == "cpu" and self.system.precision == "bf16":
            raise ValueError("system.precision=bf16 needs system.device=cuda")


SECTIONS = tuple(f.name for f in fields(Config))
