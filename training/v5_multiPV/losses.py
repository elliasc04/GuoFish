"""Loss functions for the v5 multi-PV student.

Everything here returns SUMS, never means. The caller divides by a constant it
chose, which is the only way gradient accumulation and the policy-coverage
normalisation can both be correct at once - see `policy_denominator`.

Three properties are load-bearing and are pinned by tests/test_losses.py:

  1. The legal mask is applied to the logits BEFORE log_softmax, with -inf, so
     an illegal move receives EXACTLY zero probability. This is the same
     operation the engine performs at inference (ChessTransformerV5.forward's
     `legal_move_mask` branch); a train/inference mismatch here costs real
     strength and never shows up in a loss curve.
  2. Records with has_policy=0 contribute exactly zero policy gradient, and no
     NaN. Their dense target is all-zero, so the naive `target * log q` term is
     `0 * -inf` -> NaN in the forward AND poisons the backward through any
     torch.where written after the fact. The masking is done on log q itself,
     before the multiply, so the -inf never meets a zero.
  3. The KL is a real KL: the target entropy term is included, so a perfectly
     fitted policy reports 0.0 rather than the target's entropy. It is computed
     under no_grad (the target is a constant) so it changes nothing downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor,
                       allow_empty: bool = True,
                       extra_legal: Optional[torch.Tensor] = None) -> tuple:
    """log-softmax restricted to the legal moves.

    Returns (log_probs, mask_bool). Illegal entries of `log_probs` are exactly
    -inf, so `exp(log_probs)` is exactly 0 there and the legal entries sum to 1.

    `allow_empty` guards a row with no legal moves at all (checkmate/stalemate;
    none are expected in this corpus but a single one turns the whole batch into
    NaN). Such a row falls back to unmasked softmax; it is a has_policy=0 record
    by construction, so it contributes nothing either way.

    `extra_legal` is a second bool mask UNIONED into the legal set, and exists
    for one specific corpus defect. `legal_idx` is a fixed-width field of
    MAX_LEGAL=128 entries and Pass B TRUNCATES it, so a position with more than
    128 legal moves stores an incomplete legal set. When one of the PV moves
    falls outside the 128 that were kept, the target carries mass at an index
    this mask calls illegal - log q is -inf there, and the record's KL is +inf.

    That is a storage artifact, not a claim about chess: the target came from a
    multi-PV search that only ever proposes legal moves, so any index with
    target mass IS legal and the stored mask is simply missing it. Unioning the
    target's support back in repairs the lossy field from the reliable one.

    Measured incidence in data/processed/multipv_90m: 1 record in 452,405 on
    the val split and ~1 per million policy records on train - every one of them
    with n_legal == 128 exactly. Rare enough to have gone unnoticed, and fatal
    anyway, because a single +inf makes the MEAN KL over a whole split +inf.
    The 20M run shows it in 45 separate log windows.

    For every record whose target support already sits inside the legal set -
    which is all but ~1e-6 of them - `mask | support` IS `mask`, so this is
    bit-identical to not passing it. It cannot quietly change a healthy record.
    """
    mask = legal_mask if legal_mask.dtype == torch.bool else legal_mask > 0
    if extra_legal is not None:
        mask = mask | (extra_legal if extra_legal.dtype == torch.bool
                       else extra_legal > 0)
    if allow_empty:
        empty = ~mask.any(dim=-1, keepdim=True)
        if bool(empty.any()):
            mask = mask | empty
    # float32 regardless of autocast dtype: log_softmax over 4096 entries in
    # bf16 has ~3 decimal digits, which is coarser than the KL we are measuring.
    z = logits.float().masked_fill(~mask, float("-inf"))
    return torch.log_softmax(z, dim=-1), mask


def policy_kl_per_sample(logits: torch.Tensor, target: torch.Tensor,
                         legal_mask: torch.Tensor,
                         has_policy: Optional[torch.Tensor] = None,
                         check_support: bool = False,
                         repair_truncated_legal: bool = True) -> tuple:
    """KL(target || model) per sample, (B,).

    `target` is the dense 4096 distribution (~0.95 over PV moves, 0.05 spread
    over legal moves) and is all-zero where has_policy=0. No argmax is taken
    anywhere: the whole distribution is the label.

    `repair_truncated_legal` unions the target's support into the legal mask,
    which makes the KL finite for the ~1e-6 of records whose 128-entry
    `legal_idx` was truncated past one of their own PV moves. See
    masked_log_softmax. Default ON: without it a single such record makes the
    mean KL over its whole split +inf, and this run's val KL is one half of a
    pre-registered threshold. Pass False to reproduce the pre-repair numbers.
    """
    support = target > 0
    log_q, mask = masked_log_softmax(
        logits, legal_mask, extra_legal=support if repair_truncated_legal else None)

    if check_support:
        # Deliberately against the RAW legal mask, not the repaired one. The two
        # are different questions: the repair asks "can this record produce a
        # finite KL", check_support asks "did the converter emit mass outside
        # the legal set at all" - and the answer to the second must not become
        # 'no' just because the first was switched on.
        raw = legal_mask if legal_mask.dtype == torch.bool else legal_mask > 0
        leaked = (support & ~raw)
        if bool(leaked.any()):
            rows = torch.nonzero(leaked.any(dim=-1)).flatten().tolist()
            raise AssertionError(
                f"target has mass on illegal indices in rows {rows[:8]} "
                f"({int(leaked.sum())} entries); KL would be +inf")

    # Zero log q OUTSIDE the target's support before multiplying. Both halves
    # matter: forward avoids 0 * -inf = NaN, and backward gets an exactly-zero
    # incoming gradient at those positions instead of a NaN one.
    log_q_safe = log_q.masked_fill(~support, 0.0)
    cross_entropy = -(target * log_q_safe).sum(dim=-1)

    with torch.no_grad():
        # xlogy(0, 0) == 0. Target is a constant, so no gradient path exists.
        neg_entropy = torch.xlogy(target, target).sum(dim=-1)

    kl = cross_entropy + neg_entropy

    if has_policy is not None:
        # Redundant with the all-zero target, and deliberately so: it is the
        # cheap assertion that survives a future change to the label format.
        kl = kl * (has_policy > 0).to(kl.dtype)
    return kl, log_q


def value_se_per_sample(pred: torch.Tensor, target: torch.Tensor
                        ) -> torch.Tensor:
    """Squared error per sample, (B,). No clamping, no rescaling - the stored
    target at the manifest scale is authoritative for this run."""
    return (pred.float() - target.float()).pow(2)


def policy_denominator(effective_batch_samples: int, coverage: float) -> float:
    """C = effective_batch * coverage.

    THE point of this function is that it does not look at the batch. Under
    accumulation, dividing each micro-batch's KL sum by that micro-batch's own
    has_policy count produces a mean-of-means: records that land in a sparse
    micro-batch get up-weighted, and the effective policy LR jitters with
    per-batch coverage. Because backward() accumulates linearly, dividing every
    micro-batch by the SAME constant makes the accumulated gradient exactly

        (sum of KL over the whole effective batch) / C

    which is the per-policy-record mean at the expected coverage, and is
    identical whether the effective batch was fed in 1 chunk or 8.
    """
    if effective_batch_samples <= 0:
        raise ValueError("effective_batch_samples must be positive")
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    return float(effective_batch_samples) * float(coverage)


@dataclass
class LossParts:
    """Raw, separately auditable components.

    `*_sum` are graph-attached and are what backward sees (before division).
    Everything else is a plain float, detached at construction - reading a
    running total off the graph tensor would both warn and pin the whole
    micro-batch's autograd graph alive for the length of the accumulation
    window.
    """

    policy_kl_sum: torch.Tensor
    value_se_sum: torch.Tensor
    n_policy: int                   # realised has_policy count
    n_samples: int
    policy_kl_total: float          # detached sum, for logging
    value_se_total: float           # detached sum, for logging
    policy_kl_mean: float           # per has_policy record
    value_mse_mean: float           # per record


def compute_losses(policy_logits: torch.Tensor, value_pred: torch.Tensor,
                   batch: dict, check_support: bool = False) -> LossParts:
    """One micro-batch -> summed components. No normalisation applied."""
    target = batch["policy"]
    legal_mask = batch["legal_mask"]
    has_policy = batch["has_policy"]

    kl, _ = policy_kl_per_sample(policy_logits, target, legal_mask,
                                 has_policy=has_policy,
                                 check_support=check_support)
    se = value_se_per_sample(value_pred, batch["value"])

    kl_sum = kl.sum()
    se_sum = se.sum()
    n_policy = int((has_policy > 0).sum().item())
    n_samples = int(has_policy.shape[0])
    kl_total = float(kl_sum.detach())
    se_total = float(se_sum.detach())

    return LossParts(
        policy_kl_sum=kl_sum,
        value_se_sum=se_sum,
        n_policy=n_policy,
        n_samples=n_samples,
        policy_kl_total=kl_total,
        value_se_total=se_total,
        policy_kl_mean=(kl_total / n_policy) if n_policy else 0.0,
        value_mse_mean=se_total / max(1, n_samples),
    )
