"""M4: schedules, EMA, param groups, losses, Muon."""
from __future__ import annotations

import math
import sys

import numpy as np
import pytest
import torch

from core.guofish_net import ModelConfig, build_model
from training.v6.config import load_config
from training.v6.config.schema import ScheduleConfig
from training.v6.data.batch import BatchBuilder
from training.v6.data.formats import REPO
from training.v6.data.mixture import Mixture
from training.v6.data.reader import ShardSet
from training.v6.data.strata import load_strata
from training.v6.losses import LossFn, hard_ce, loss_normalizers, soft_kl, value_loss
from training.v6.optim import EMA, Schedule, build_optimizer, param_groups
from training.v6.optim.muon import newton_schulz

CFG = REPO / "training/v6/config/configs"
FROZEN = REPO / "data/processed/val_frozen_90m_v1"


# ---------------------------------------------------------------- schedule

def test_wsd_hand_computed_points():
    eff, peak = 1024, 1e-3
    cfg = ScheduleConfig(kind="wsd", total_samples=1000 * eff, warmup_samples=100 * eff,
                         decay_frac=0.2, decay_shape="one_minus_sqrt", final_lr_frac=0.0)
    s = Schedule(cfg, peak, eff)
    pts = {0: 0.0, 50: 0.5e-3, 99: 0.99e-3, 100: 1e-3, 500: 1e-3, 799: 1e-3, 800: 1e-3,
           900: 1e-3 * (1 - math.sqrt(0.5)), 950: 1e-3 * (1 - math.sqrt(0.75)), 1000: 0.0}
    for step, want in pts.items():
        assert s.lr(step * eff) == pytest.approx(want, rel=1e-12, abs=1e-18), step
    assert [s.phase(k * eff) for k in (0, 100, 800)] == ["warmup", "stable", "decay"]
    assert s.beta1(0) is None                                       # beta1 held constant
    lin = Schedule(ScheduleConfig(total_samples=1000 * eff, warmup_samples=0, decay_frac=0.2,
                                  decay_shape="linear", final_lr_frac=0.1), peak, eff)
    assert lin.lr(0) == peak and lin.lr(900 * eff) == pytest.approx(peak * (0.1 + 0.9 * 0.5))
    assert lin.lr(1000 * eff) == pytest.approx(0.1 * peak)
    cos = Schedule(ScheduleConfig(total_samples=1000 * eff, warmup_samples=0, decay_frac=0.2,
                                  decay_shape="cosine"), peak, eff)
    assert cos.lr(850 * eff) == pytest.approx(peak * 0.5 * (1 + math.cos(math.pi * 0.25)))
    # doc example: a branch at 240M with decay_frac 0.2 decays for 60M
    br = Schedule(ScheduleConfig(total_samples=300_000_000, warmup_samples=0, decay_frac=0.2),
                  peak, eff)
    assert br.decay_start == 240_000_000 and br.decay_len == 60_000_000


def test_onecycle_matches_torch_including_momentum():
    """The compat schedule, every one of its 351,858 steps, vs torch's OneCycleLR."""
    c = load_config(CFG / "v5_compat.yaml")
    eff = c.optim.effective_batch
    ours = Schedule(c.schedule, c.optim.lr, eff)
    T = c.schedule.total_samples // eff
    oc = c.schedule.onecycle
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=c.optim.lr / oc.div_factor, betas=c.optim.betas)
    ref = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=c.optim.lr, total_steps=T, pct_start=oc.pct_start, div_factor=oc.div_factor,
        final_div_factor=oc.final_div_factor, cycle_momentum=oc.cycle_momentum,
        base_momentum=oc.base_momentum, max_momentum=oc.max_momentum)
    lr_err = b_err = 0.0
    for k in range(T):
        g = opt.param_groups[0]
        lr_err = max(lr_err, abs(ours.lr(k * eff) - g["lr"]))
        b_err = max(b_err, abs(ours.beta1(k * eff) - g["betas"][0]))
        opt.step()
        ref.step()
    print(f"\nOneCycle vs torch over {T:,} steps: max |d lr| {lr_err:.3e}, max |d beta1| {b_err:.3e}")
    assert lr_err == 0.0 and b_err == 0.0
    assert ours.lr(0) == pytest.approx(c.optim.lr / 25, rel=1e-12) and ours.beta1(0) == 0.95


# --------------------------------------------------------------------- EMA

def test_ema_half_life():
    m = torch.nn.Linear(4, 4, bias=False)
    ema = EMA(m, half_life_samples=10e6, effective_batch=1024)
    print(f"\nEMA decay per step at 10M half-life, batch 1024: {ema.decay:.8f}")
    assert round(ema.decay, 5) == 0.99993
    ema2 = EMA(m, half_life_samples=1024 * 100, effective_batch=1024)   # 100-step half-life
    with torch.no_grad():
        start = m.weight.clone()
        m.weight.add_(1.0)
    for _ in range(100):
        ema2.update(m)
    moved = (ema2.shadow[0] - start).mean().item()
    assert moved == pytest.approx(0.5, abs=1e-5)


# ------------------------------------------------------------ param groups

def test_param_groups_reproduce_v5_split():
    c = load_config(CFG / "v5_compat.yaml")
    model = build_model(c.model)
    dec, keep = param_groups(model, c.optim)
    assert (len(dec["params"]), sum(p.numel() for p in dec["params"])) == (29, 10_830_336)
    assert (len(keep["params"]), sum(p.numel() for p in keep["params"])) == (55, 57_345)
    base = load_config(CFG / "base.yaml")
    model = build_model(ModelConfig.from_dict({**base.model.to_dict(), "attn_bias": "static"}))
    dec, keep = param_groups(model, base.optim)
    ids = {id(p) for p in keep["params"]}
    assert id(model.embedding.weight) in ids and id(model.static_bias) in ids
    assert id(model.pos_embedding) in ids and id(model.blocks[0].qkv.weight) not in ids


def test_fused_adamw_on_cpu_and_muon_split():
    c = load_config(CFG / "base.yaml", ["optim.name=muon_adamw", "model.attn_bias=smolgen"])
    model = build_model(c.model)
    opt = build_optimizer(model, c.optim, 1e-3)
    muon_ids = {id(p) for g in opt.muon.param_groups for p in g["params"]}
    assert id(model.blocks[0].qkv.weight) in muon_ids and id(model.blocks[5].smolgen.fc2.weight) in muon_ids
    assert id(model.smolgen_shared.weight) not in muon_ids and id(model.from_proj.weight) not in muon_ids
    assert len(muon_ids) == 6 * 7
    adam = build_optimizer(model, load_config(CFG / "base.yaml").optim, 1e-3)
    assert adam.defaults["fused"] is True


# ------------------------------------------------------------------ losses

@pytest.fixture(scope="module")
def val_batch():
    ss = ShardSet(FROZEN, "val")
    idx = np.random.default_rng(3).choice(len(ss), 512, replace=False)
    b = BatchBuilder("v5_68", 0.0, 0, 0.05, None)(ss.read(idx), idx)
    ss.close()
    return b


def test_soft_kl_matches_v5(val_batch):
    sys.path.insert(0, str(REPO / "training/v5_multiPV"))
    from training.v5_multiPV.losses import policy_kl_per_sample
    torch.manual_seed(0)
    logits = torch.randn(512, 4096) * 3
    ours, _ = soft_kl(logits, val_batch["policy"], val_batch["legal"])
    ours = ours * val_batch["has_policy"]
    ref, _ = policy_kl_per_sample(logits, val_batch["policy"], val_batch["legal"].float(),
                                  has_policy=val_batch["has_policy"].float())
    assert torch.equal(ours, ref)


def test_hard_ce_properties(val_batch):
    torch.manual_seed(1)
    logits = torch.randn(512, 4096, requires_grad=True)
    legal = val_batch["legal"]
    hm = torch.where(legal.any(1), legal.float().argmax(1), torch.full((512,), -1))
    hm[::3] = -1
    ce, nll = hard_ce(logits, hm, legal, 0.1)
    assert torch.isfinite(ce).all() and (ce[hm < 0] == 0).all() and (nll[hm < 0] == 0).all()
    ce.sum().backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad[hm < 0] == 0).all()
    # a hard move outside the stored legal set (truncation) stays finite
    bad = hm.clone()
    bad[1] = int((~legal[1]).float().argmax())
    assert torch.isfinite(hard_ce(logits.detach(), bad, legal, 0.1)[0]).all()
    # the minimiser of CE is the target itself: logits = log(target) gives CE = H(target)
    row = 2
    n = legal[row].sum()
    target = legal[row].float() * (0.1 / n)
    target[hm[row]] += 0.9
    best = torch.where(target > 0, target.log(), torch.full_like(target, -1e9)).unsqueeze(0)
    got = hard_ce(best, hm[row:row + 1], legal[row:row + 1], 0.1)[0]
    ent = -(target[target > 0] * target[target > 0].log()).sum()
    assert got.item() == pytest.approx(ent.item(), rel=1e-5)


def test_value_losses():
    y = torch.tensor([0.0, 0.5, -0.99])
    assert torch.equal(value_loss({"value": y}, y, "scalar"), torch.zeros(3))
    from core.guofish_net import hlgauss_target
    logits = hlgauss_target(y).clamp_min(1e-30).log()                # finite, like a model's
    ce = value_loss({"value_logits": logits}, y, "hlgauss")
    assert torch.isfinite(ce).all() and (ce > 0).all()               # = target entropy
    assert (value_loss({"value_logits": logits + 0.1 * torch.randn(3, 101)}, y, "hlgauss") > ce).all()


def test_normalizers_and_accumulation_exact(val_batch):
    c = load_config(CFG / "base.yaml", ["system.device=cpu", "system.precision=fp32"])
    strata = np.asarray(load_strata(FROZEN / "strata_val_v1.npy", 452_405))
    mix = Mixture(c.mixture, 452_405, c.optim.micro_batch, 0, strata=strata)
    norms = loss_normalizers(c, mix, strata)
    assert norms["soft"] == pytest.approx(1024 * 271_876 / 452_405, rel=1e-12)
    assert norms["value"] == 1024 and norms["hard"] == 0
    grouped = load_config(CFG / "base.yaml", [
        "mixture.groups=[{name: p, where: {label: multipv}, share: 0.75}]"])
    gm = Mixture(grouped.mixture, 452_405, 512, 0, strata=strata)
    assert loss_normalizers(grouped, gm, strata)["soft"] == pytest.approx(768.0)

    # two micro-batches of 256 accumulate to the same gradient as one of 512
    torch.manual_seed(0)
    model = build_model(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128))
    loss_fn = LossFn(c, norms, "cpu")

    def grads(chunks):
        model.zero_grad()
        for sl in chunks:
            b = {k: v[sl] for k, v in val_batch.items()}
            loss_fn(model.forward_train(b["tokens"]), b)[0].backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()])

    one = grads([slice(0, 512)])
    two = grads([slice(0, 256), slice(256, 512)])
    rel = ((one - two).abs().max() / one.abs().max()).item()
    print(f"\naccumulation 1x512 vs 2x256: max rel grad diff {rel:.2e}")
    assert rel < 1e-5


# -------------------------------------------------------------------- Muon

def test_newton_schulz_and_update_scale():
    torch.manual_seed(0)
    g = torch.randn(384, 1536)
    o = newton_schulz(g, 5)
    sv = torch.linalg.svdvals(o)
    assert 0.5 < sv.min() and sv.max() < 1.3                  # quintic NS: roughly orthogonal
    rms = (o * 0.2 * max(g.shape) ** 0.5).pow(2).mean().sqrt().item()
    assert 0.15 < rms < 0.25                                   # AdamW-like update RMS


def test_muon_adamw_trains_and_round_trips(val_batch):
    c = load_config(CFG / "base.yaml", ["optim.name=muon_adamw", "system.device=cpu",
                                        "system.precision=fp32"])
    torch.manual_seed(0)
    model = build_model(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128))
    opt = build_optimizer(model, c.optim, 3e-3)
    loss_fn = LossFn(c, {"soft": 300.0, "hard": 1.0, "value": 512.0}, "cpu")
    b = val_batch
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss, _ = loss_fn(model.forward_train(b["tokens"]), b)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0]
    sd = opt.state_dict()
    opt2 = build_optimizer(model, c.optim, 3e-3)
    opt2.load_state_dict(sd)
    assert set(sd) == {"muon", "adamw"}
