"""Soft policy, hard policy and value losses (§7), each normalised per term.

    L = w_soft * sum_soft KL(t||q) / N_soft
      + w_hard * sum_hard CE(h, q)  / N_hard
      + w_v    * sum_all  lambda_s * l_v / N

N_* are constants per optimizer window, computed once from the config and the
strata sidecar (`loss_normalizers`): the expected row count of each term in a
window. With label-aligned mixture groups that is the exact count; under
`natural` it is window x corpus coverage, v5's constant denominator with the
coverage measured exactly rather than estimated. Constant denominators keep
gradient accumulation exact.

No host syncs: every metric is a detached device tensor, summed by the
caller and read once per log interval (H19). v5's masked_log_softmax has a
`bool(empty.any())` sync, so the mask repair is re-written here without it;
`test_m4` checks the KL against v5's function.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core.guofish_net.model import POLICY_SIZE, hlgauss_target
from training.v6.config.schema import STRATA_FIELDS, Config
from training.v6.data.strata import field_of


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """fp32 log-softmax over `mask`; a row with nothing allowed falls back to
    unmasked (such rows carry no target, so they contribute nothing)."""
    mask = mask | ~mask.any(-1, keepdim=True)
    return torch.log_softmax(logits.float().masked_fill(~mask, float("-inf")), dim=-1)


def soft_kl(logits, target, legal):
    """KL(target || q) per row, target's support unioned into the legal mask
    (the truncated-legal repair). All-zero target rows give exactly 0."""
    support = target > 0
    log_q = masked_log_softmax(logits, legal | support)
    ce = -(target * log_q.masked_fill(~support, 0.0)).sum(-1)
    with torch.no_grad():
        neg_ent = torch.xlogy(target, target).sum(-1)
    return ce + neg_ent, log_q


def hard_ce(logits, hard_move, legal, epsilon: float):
    """CE against (1-eps) on hard_move + eps spread over the legal moves.
    Rows with hard_move < 0 give 0. Returns (ce, nll of hard_move)."""
    valid = hard_move >= 0
    hm = hard_move.clamp_min(0)
    onehot = F.one_hot(hm, POLICY_SIZE).bool() & valid.unsqueeze(1)
    support = legal | onehot
    n = legal.sum(-1).clamp_min(1).float()
    target = legal.float() * (epsilon / n).unsqueeze(1) + onehot.float() * (1.0 - epsilon)
    log_q = masked_log_softmax(logits, support)
    ce = -(target * log_q.masked_fill(~support, 0.0)).sum(-1) * valid
    nll = -log_q.gather(1, hm.unsqueeze(1)).squeeze(1).masked_fill(~valid, 0.0)
    return ce, nll


def value_loss(out: dict, y: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "scalar":
        return (out["value"] - y).pow(2)
    return -(hlgauss_target(y) * torch.log_softmax(out["value_logits"], -1)).sum(-1)


def loss_normalizers(cfg: Config, mixture, strata: np.ndarray) -> dict:
    """Expected per-window row counts of each term, from config + strata."""
    eff = cfg.optim.effective_batch
    label = field_of(np.asarray(strata), "label")
    soft = hard = 0.0
    for g in range(len(mixture.names)):
        mem = mixture._member(g)
        lab = label if mem is None else label[np.asarray(mem)]
        soft += mixture.shares[g] * float((lab == 0).mean())
        hard += mixture.shares[g] * float((lab == 1).mean())
    out = {"soft": eff * soft, "hard": eff * hard, "value": float(eff)}
    t = cfg.targets
    for term, w in (("soft", t.policy_soft.weight), ("hard", t.policy_hard.weight)):
        if w > 0 and out[term] <= 0:
            raise ValueError(f"targets.policy_{term}.weight={w} but the mixture has no {term} rows")
    return out


class LossFn:
    def __init__(self, cfg: Config, norms: dict, device):
        t = cfg.targets
        self.w_soft, self.w_hard, self.w_v = t.policy_soft.weight, t.policy_hard.weight, t.value.weight
        self.eps_hard = t.policy_hard.epsilon
        self.hard_head = "aux_policy_logits" if t.policy_hard.head == "aux" else "policy_logits"
        self.kind = cfg.model.value_repr.kind
        self.norms = norms
        self.lam = torch.tensor([t.value.stratum_weights[k] for k in STRATA_FIELDS["value"]],
                                dtype=torch.float32, device=device)

    def __call__(self, out: dict, b: dict):
        """-> (loss, metrics). metrics are detached device-tensor SUMS."""
        has_pol = b["has_policy"]
        kl, _ = soft_kl(out["policy_logits"], b["policy"], b["legal"])
        kl = kl * has_pol
        hard_rows = ~has_pol & (b["hard_move"] >= 0)
        hm = torch.where(hard_rows, b["hard_move"], torch.full_like(b["hard_move"], -1))
        hce, hnll = hard_ce(out[self.hard_head], hm, b["legal"], self.eps_hard)
        lv = value_loss(out, b["value"], self.kind)
        lam = self.lam[b["value_stratum"]]

        loss = self.w_v * (lam * lv).sum() / self.norms["value"]
        if self.w_soft:
            loss = loss + self.w_soft * kl.sum() / self.norms["soft"]
        if self.w_hard:
            loss = loss + self.w_hard * hce.sum() / self.norms["hard"]
        se = (out["value"] - b["value"]).pow(2)
        m = {"soft_kl": kl.sum(), "n_soft": has_pol.sum(),
             "hard_ce": hce.sum(), "hard_nll": hnll.sum(), "n_hard": hard_rows.sum(),
             "value_loss": (lam * lv).sum(), "value_se": se.sum(), "loss": loss}
        return loss, {k: v.detach().float() for k, v in m.items()}
