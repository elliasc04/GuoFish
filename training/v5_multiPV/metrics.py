"""Evaluation metrics for the v5 multi-PV student.

Aggregate value MSE is misleading on this corpus and is never headlined here.
21.0% of the targets are exactly zero and 20.2% are mates pinned at +-0.995, so
a head that collapses toward zero scores well on a fifth of the data by doing
nothing at all - and that is completely invisible in a single number. Every
value metric is therefore reported per stratum, alongside the prediction std
(Phase 2 tracked 0.352 -> 0.501; the middle-only label std is 0.510).

The strata are defined exactly as pass_b_convert.py counted them, off the RAW
`value_cp` rather than the squashed target, so they survive a change of value
scale and are invariant under the colour mirror (which only negates):

    mate        |value_cp| >= 29000        (VALUE_MATE_MIN)
    exact-zero  value_cp == 0
    middle      everything else
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_MULTIPV = _PROJECT_ROOT / "data" / "multiPV"
if str(_DATA_MULTIPV) not in sys.path:
    sys.path.insert(0, str(_DATA_MULTIPV))

from labels import VALUE_MATE_MIN                      # noqa: E402
from dataset import color_mirror                       # noqa: E402
from mirror import POLICY_PERM                         # noqa: E402

from losses import masked_log_softmax, policy_kl_per_sample  # noqa: E402

STRATA = ("exact_zero", "mate", "middle")
_POLICY_PERM_T = torch.from_numpy(POLICY_PERM)


# ---------------------------------------------------------------------------
# small statistics helpers
# ---------------------------------------------------------------------------
def pearson_r(a: torch.Tensor, b: torch.Tensor) -> float:
    """Sample Pearson correlation. NaN when either side is constant, which is
    the honest answer for a collapsed head rather than a spurious 0.0."""
    if a.numel() < 2:
        return float("nan")
    a = a.double() - a.double().mean()
    b = b.double() - b.double().mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return float("nan")
    return float((a @ b) / denom)


def value_strata(value_cp: torch.Tensor) -> dict:
    """value_cp (B,) -> {stratum: bool mask}. Partition, not overlapping sets."""
    cp = value_cp.round().abs()
    mate = cp >= VALUE_MATE_MIN
    zero = (value_cp == 0) & ~mate
    return {"exact_zero": zero, "mate": mate, "middle": ~(zero | mate)}


# ---------------------------------------------------------------------------
# policy top-k
# ---------------------------------------------------------------------------
def policy_topk_correct(logits: torch.Tensor, target: torch.Tensor,
                        legal_mask: torch.Tensor, k: int) -> torch.Tensor:
    """(B,) bool. Logits are masked first, exactly as the engine masks them, so
    top-k can never be won by an illegal move."""
    log_q, _ = masked_log_softmax(logits, legal_mask)
    gold = target.argmax(dim=-1)
    if k == 1:
        return log_q.argmax(dim=-1) == gold
    topk = log_q.topk(min(k, log_q.shape[-1]), dim=-1).indices
    return (topk == gold.unsqueeze(-1)).any(dim=-1)


# ---------------------------------------------------------------------------
# streaming accumulators
# ---------------------------------------------------------------------------
class ValueAccumulator:
    """Collects (pred, target, value_cp) on CPU and reports stratified metrics.

    The full val split is 147,963 records, so holding three float32 vectors is
    ~1.8 MB. Streaming sums would buy nothing and would make Pearson r awkward.
    """

    def __init__(self):
        self._pred = []
        self._target = []
        self._cp = []

    @torch.no_grad()
    def add(self, pred: torch.Tensor, target: torch.Tensor,
            value_cp: torch.Tensor) -> None:
        self._pred.append(pred.detach().float().cpu())
        self._target.append(target.detach().float().cpu())
        self._cp.append(value_cp.detach().float().cpu())

    def __len__(self) -> int:
        return sum(t.numel() for t in self._pred)

    def result(self) -> dict:
        if not self._pred:
            return {}
        pred = torch.cat(self._pred)
        target = torch.cat(self._target)
        cp = torch.cat(self._cp)

        out = {
            "value_mse": float((pred - target).pow(2).mean()),
            "value_pearson_r": pearson_r(pred, target),
            "value_pred_std": float(pred.std(unbiased=False)),
            "value_label_std": float(target.std(unbiased=False)),
            "value_pred_mean": float(pred.mean()),
            "value_label_mean": float(target.mean()),
            "n": int(pred.numel()),
        }
        for name, mask in value_strata(cp).items():
            n = int(mask.sum())
            key = f"value_{name}"
            if n == 0:
                out.update({f"{key}_n": 0, f"{key}_mse": float("nan"),
                            f"{key}_r": float("nan"),
                            f"{key}_pred_std": float("nan"),
                            f"{key}_label_std": float("nan"),
                            f"{key}_pred_mean": float("nan")})
                continue
            p, t = pred[mask], target[mask]
            out.update({
                f"{key}_n": n,
                f"{key}_share": n / pred.numel(),
                f"{key}_mse": float((p - t).pow(2).mean()),
                f"{key}_r": pearson_r(p, t),
                f"{key}_pred_std": float(p.std(unbiased=False)) if n > 1 else float("nan"),
                f"{key}_label_std": float(t.std(unbiased=False)) if n > 1 else float("nan"),
                f"{key}_pred_mean": float(p.mean()),
            })
        return out


class PolicyAccumulator:
    """Policy KL / top-1 / top-5 over has_policy records only."""

    def __init__(self, topk=(1, 5)):
        self.topk = tuple(topk)
        self.kl_sum = 0.0
        self.n_policy = 0
        self.n_samples = 0
        self.correct = {k: 0 for k in self.topk}

    @torch.no_grad()
    def add(self, logits: torch.Tensor, batch: dict) -> None:
        has = batch["has_policy"] > 0
        self.n_samples += int(has.numel())
        n = int(has.sum())
        if n == 0:
            return
        self.n_policy += n

        logits_p = logits[has]
        target_p = batch["policy"][has]
        mask_p = batch["legal_mask"][has]

        kl, _ = policy_kl_per_sample(logits_p, target_p, mask_p)
        self.kl_sum += float(kl.sum())
        for k in self.topk:
            self.correct[k] += int(
                policy_topk_correct(logits_p, target_p, mask_p, k).sum())

    def result(self) -> dict:
        out = {
            "policy_kl": self.kl_sum / self.n_policy if self.n_policy else float("nan"),
            "policy_n": self.n_policy,
            "policy_coverage": (self.n_policy / self.n_samples
                                if self.n_samples else float("nan")),
        }
        for k in self.topk:
            out[f"policy_top{k}"] = (self.correct[k] / self.n_policy
                                     if self.n_policy else float("nan"))
        return out


# ---------------------------------------------------------------------------
# mirror consistency
# ---------------------------------------------------------------------------
@torch.no_grad()
def mirror_consistency(model, loader, device, max_records: int = 10_000,
                       autocast_ctx=None) -> dict:
    """The colour mirror is an exact involution, so this diagnostic is free.

    Runs every position both ways and checks the two things the augmentation is
    supposed to buy:

        value    v(mirror(x)) == -v(x)
        policy   argmax over mirror(x) == POLICY_PERM[argmax over x]

    Drift means the model is leaning on absolute board orientation - exactly the
    capacity waste the 50% mirror exists to prevent. Also reports the
    mirror-BALANCED prediction mean (v(x) + v(mirror(x))) / 2, which is the
    deterministic version of "predictions should sit near 0 with mirroring on":
    it is identically 0 for a perfectly equivariant head no matter how skewed
    the underlying corpus is.
    """
    import contextlib
    if autocast_ctx is None:
        autocast_ctx = contextlib.nullcontext

    was_training = model.training
    model.eval()

    perm = _POLICY_PERM_T.to(device)
    v_sum_abs = 0.0
    v_signed_sum = 0.0
    v_max_abs = 0.0
    top1_agree = 0
    kl_sym_sum = 0.0
    n = 0
    n_policy = 0

    for batch in loader:
        if n >= max_records:
            break
        take = min(batch["tokens"].shape[0], max_records - n)
        batch = {k: v[:take] for k, v in batch.items()}

        all_true = torch.ones(take, dtype=torch.bool)
        mirrored = color_mirror(batch, all_true)

        tok = batch["tokens"].to(device, non_blocking=True)
        tok_m = mirrored["tokens"].to(device, non_blocking=True)
        mask = batch["legal_mask"].to(device, non_blocking=True)
        mask_m = mirrored["legal_mask"].to(device, non_blocking=True)

        with autocast_ctx():
            logits, value = model(tok)
            logits_m, value_m = model(tok_m)
        logits, logits_m = logits.float(), logits_m.float()
        value, value_m = value.float(), value_m.float()

        resid = value + value_m                      # should be 0
        v_sum_abs += float(resid.abs().sum())
        v_signed_sum += float(resid.sum())
        v_max_abs = max(v_max_abs, float(resid.abs().max()))

        lq, _ = masked_log_softmax(logits, mask)
        lq_m, _ = masked_log_softmax(logits_m, mask_m)
        top1_agree += int((lq_m.argmax(-1) == perm[lq.argmax(-1)]).sum())

        # Symmetrised KL between the original policy and the un-mirrored image
        # of the mirrored policy: a scale-free companion to top-1 agreement.
        q = lq.exp()
        q_back = lq_m.exp()[:, perm]
        kl_sym_sum += float(
            (torch.xlogy(q, q.clamp_min(1e-12) / q_back.clamp_min(1e-12))
             ).sum(-1).sum())

        n += take
        n_policy += int((batch["has_policy"] > 0).sum())

    if was_training:
        model.train()

    if n == 0:
        return {}
    return {
        "mirror_n": n,
        "mirror_value_abs_resid": v_sum_abs / n,       # mean |v(x) + v(mx)|
        "mirror_value_max_resid": v_max_abs,
        "mirror_pred_mean_balanced": v_signed_sum / (2 * n),
        "mirror_policy_top1_agreement": top1_agree / n,
        "mirror_policy_kl": kl_sym_sum / n,
        "mirror_has_policy_n": n_policy,
    }


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def format_value_report(m: dict) -> str:
    lines = [
        f"  value  aggregate     MSE {m['value_mse']:.5f}  "
        f"r {m['value_pearson_r']:+.4f}  "
        f"pred_std {m['value_pred_std']:.4f}  label_std {m['value_label_std']:.4f}",
        f"         mean          pred {m['value_pred_mean']:+.4f}  "
        f"label {m['value_label_mean']:+.4f}",
    ]
    for name in STRATA:
        k = f"value_{name}"
        if m.get(f"{k}_n", 0) == 0:
            continue
        lines.append(
            f"         {name:<12}  MSE {m[f'{k}_mse']:.5f}  "
            f"r {m[f'{k}_r']:+.4f}  "
            f"pred_std {m[f'{k}_pred_std']:.4f}  "
            f"label_std {m[f'{k}_label_std']:.4f}  "
            f"n {m[f'{k}_n']:,} ({m[f'{k}_share'] * 100:.1f}%)")
    return "\n".join(lines)


def format_policy_report(m: dict) -> str:
    return (f"  policy KL {m['policy_kl']:.5f}  "
            f"top1 {m['policy_top1'] * 100:.2f}%  "
            f"top5 {m.get('policy_top5', float('nan')) * 100:.2f}%  "
            f"n {m['policy_n']:,} (coverage {m['policy_coverage'] * 100:.2f}%)")


def format_mirror_report(m: dict) -> str:
    if not m:
        return "  mirror: (no records)"
    return (f"  mirror mean|v(x)+v(mx)| {m['mirror_value_abs_resid']:.5f}  "
            f"max {m['mirror_value_max_resid']:.5f}  "
            f"balanced_pred_mean {m['mirror_pred_mean_balanced']:+.5f}  "
            f"policy top1 agree {m['mirror_policy_top1_agreement'] * 100:.2f}%  "
            f"n {m['mirror_n']:,}")
