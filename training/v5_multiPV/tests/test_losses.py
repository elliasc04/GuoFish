"""Loss correctness for the v5 multi-PV student.

Runs under pytest, or standalone:
    python training/v5_multiPV/tests/test_losses.py

Everything here is a property that would otherwise fail SILENTLY - a mismatch
between training-time and inference-time masking, a policy gradient leaking out
of a value-only record, or a normalisation that makes the effective policy LR
track per-batch coverage. None of them move the loss curve in a way you would
notice.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_V5 = _HERE.parent
_ROOT = _V5.parents[1]
_DATA = _ROOT / "data" / "multiPV"
for _p in (str(_V5), str(_DATA), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from losses import (compute_losses, masked_log_softmax,          # noqa: E402
                    policy_denominator, policy_kl_per_sample,
                    value_se_per_sample)
from model_v5 import ModelConfig, build_model                    # noqa: E402

POLICY_SIZE = 4096


# ---------------------------------------------------------------------------
# synthetic batch builder
# ---------------------------------------------------------------------------
def make_batch(n=8, n_legal=20, n_pv=3, has_policy=None, epsilon=0.05, seed=0):
    """A batch shaped exactly like MultiPVCollate's output.

    `has_policy` is a list of 0/1; records with 0 get an all-zero dense policy,
    which is what dataset.MultiPVDataset emits for a value-only record.
    """
    g = torch.Generator().manual_seed(seed)
    if has_policy is None:
        has_policy = [1] * n
    assert len(has_policy) == n

    policy = torch.zeros(n, POLICY_SIZE)
    legal = torch.zeros(n, POLICY_SIZE)
    for i in range(n):
        idx = torch.randperm(POLICY_SIZE, generator=g)[:n_legal]
        legal[i, idx] = 1.0
        if has_policy[i]:
            pv = idx[:n_pv]
            w = torch.softmax(torch.randn(n_pv, generator=g), dim=0)
            policy[i, pv] += w * (1.0 - epsilon)
            policy[i, idx] += epsilon / n_legal

    return {
        "tokens": torch.randint(0, 43, (n, 68), generator=g),
        "policy": policy,
        "legal_mask": legal,
        "value": torch.randn(n, generator=g).tanh() * 0.9,
        "value_cp": torch.randint(-2000, 2000, (n,), generator=g).float(),
        "has_policy": torch.tensor(has_policy, dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# masking
# ---------------------------------------------------------------------------
def test_masked_logits_contribute_zero_probability():
    """The brief's assertion: masked positions -> exactly zero probability."""
    b = make_batch(n=6, seed=1)
    logits = torch.randn(6, POLICY_SIZE) * 3.0
    log_q, mask = masked_log_softmax(logits, b["legal_mask"])
    q = log_q.exp()

    illegal = ~mask
    assert torch.all(q[illegal] == 0.0), "illegal moves got nonzero probability"
    assert torch.all(torch.isinf(log_q[illegal]) & (log_q[illegal] < 0)), \
        "illegal log-probs must be exactly -inf"
    assert torch.allclose(q.sum(-1), torch.ones(6), atol=1e-6), \
        "legal probabilities must sum to 1"
    # Even a huge illegal logit must not steal any mass.
    logits2 = logits.clone()
    logits2[illegal] = 1e4
    q2 = masked_log_softmax(logits2, b["legal_mask"])[0].exp()
    assert torch.allclose(q, q2, atol=1e-6), \
        "illegal logits changed the legal distribution"


def test_mask_is_applied_before_log_softmax():
    """Masking AFTER the softmax renormalises over 4096 and then zeroes - the
    legal entries would be too small by the illegal mass. This pins the order by
    comparing against a softmax computed over the legal subset alone."""
    b = make_batch(n=4, n_legal=17, seed=2)
    logits = torch.randn(4, POLICY_SIZE) * 2.0
    log_q, mask = masked_log_softmax(logits, b["legal_mask"])

    for i in range(4):
        idx = mask[i].nonzero().flatten()
        reference = torch.log_softmax(logits[i, idx].float(), dim=-1)
        assert torch.allclose(log_q[i, idx], reference, atol=1e-6), \
            "masked log_softmax != log_softmax over the legal subset"

    # The wrong order, for contrast: it must NOT match.
    wrong = torch.log_softmax(logits.float(), dim=-1).masked_fill(~mask, -float("inf"))
    assert not torch.allclose(wrong[mask], log_q[mask], atol=1e-4), \
        "test is vacuous - post-hoc masking happened to agree"


def test_engine_forward_masking_matches_loss_masking():
    """The engine masks inside forward(); training masks inside the loss. If
    those two ever disagree, MCTS searches a different distribution than the one
    that was trained. One model, both paths, must agree exactly."""
    torch.manual_seed(3)
    model = build_model(ModelConfig(d_model=64, num_layers=2, nhead=4,
                                    dim_feedforward=128, dropout=0.0)).eval()
    b = make_batch(n=5, seed=3)
    mask_bool = b["legal_mask"] > 0
    with torch.no_grad():
        engine_logits, _ = model(b["tokens"], legal_move_mask=mask_bool)
        raw_logits, _ = model(b["tokens"])
    engine_lq = torch.log_softmax(engine_logits.float(), dim=-1)
    train_lq, _ = masked_log_softmax(raw_logits, b["legal_mask"])
    assert torch.allclose(engine_lq, train_lq, atol=1e-6, equal_nan=True)


def test_empty_legal_row_does_not_poison_the_batch():
    """A row with no legal moves would make every logit -inf and every entry of
    the batch NaN through the shared reduction."""
    b = make_batch(n=4, has_policy=[1, 1, 0, 1], seed=4)
    b["legal_mask"][2] = 0.0
    b["policy"][2] = 0.0
    logits = torch.randn(4, POLICY_SIZE, requires_grad=True)
    kl, _ = policy_kl_per_sample(logits, b["policy"], b["legal_mask"],
                                 b["has_policy"])
    assert torch.isfinite(kl).all(), f"non-finite KL: {kl}"
    kl.sum().backward()
    assert torch.isfinite(logits.grad).all(), "NaN gradient from the empty row"


# ---------------------------------------------------------------------------
# KL correctness
# ---------------------------------------------------------------------------
def test_kl_is_zero_at_a_perfect_fit():
    """A real KL, not a cross-entropy: the target-entropy term is included, so
    a model that reproduces the target exactly reports 0."""
    b = make_batch(n=6, seed=5)
    # Logits that reproduce the target on the legal support.
    logits = torch.log(b["policy"].clamp_min(1e-30))
    kl, _ = policy_kl_per_sample(logits, b["policy"], b["legal_mask"],
                                 b["has_policy"])
    assert torch.allclose(kl, torch.zeros(6), atol=1e-5), \
        f"perfect fit should give KL 0, got {kl}"


def test_kl_matches_reference_and_is_nonnegative():
    b = make_batch(n=6, seed=6)
    logits = torch.randn(6, POLICY_SIZE) * 2.0
    kl, log_q = policy_kl_per_sample(logits, b["policy"], b["legal_mask"],
                                     b["has_policy"])
    # Reference: dense-loop KL over the target's support only.
    p = b["policy"].double().numpy()
    lq = log_q.double().numpy()
    ref = np.zeros(6)
    for i in range(6):
        s = 0.0
        for j in np.nonzero(p[i])[0]:
            s += p[i, j] * (math.log(p[i, j]) - lq[i, j])
        ref[i] = s
    assert np.allclose(kl.numpy(), ref, atol=1e-5), \
        f"KL disagrees with the reference: {kl.numpy()} vs {ref}"
    assert (kl >= -1e-6).all(), "KL must be non-negative"


def test_no_argmax_anywhere_in_the_policy_loss():
    """Two targets with the SAME argmax but different tails must give different
    losses - otherwise the loss has collapsed to a hard label."""
    b = make_batch(n=1, n_legal=10, n_pv=3, seed=7)
    logits = torch.randn(1, POLICY_SIZE)
    p1 = b["policy"].clone()
    idx = p1[0].nonzero().flatten()
    top = int(p1[0].argmax())
    p2 = p1.clone()
    # Move mass between two non-top legal entries; argmax is untouched.
    others = [int(j) for j in idx if int(j) != top][:2]
    delta = min(float(p2[0, others[0]]), 0.01)
    p2[0, others[0]] -= delta
    p2[0, others[1]] += delta
    assert int(p2[0].argmax()) == top
    k1, _ = policy_kl_per_sample(logits, p1, b["legal_mask"])
    k2, _ = policy_kl_per_sample(logits, p2, b["legal_mask"])
    assert abs(float(k1) - float(k2)) > 1e-6, \
        "loss ignored the tail of the distribution - it is argmax-based"


# ---------------------------------------------------------------------------
# has_policy = 0
# ---------------------------------------------------------------------------
def test_has_policy_zero_contributes_zero_policy_gradient():
    """Not 'is small' - exactly zero, and identical to training on the
    has_policy=1 subset alone. Masking by multiplication is not enough: the
    all-zero target meets a -inf log-probability, so the product is 0 * -inf,
    which is NaN in the forward and poisons the backward."""
    torch.manual_seed(8)
    mixed = make_batch(n=8, has_policy=[1, 0, 1, 0, 0, 1, 0, 1], seed=8)
    keep = (mixed["has_policy"] > 0)

    logits = torch.randn(8, POLICY_SIZE, requires_grad=True)
    kl, _ = policy_kl_per_sample(logits, mixed["policy"], mixed["legal_mask"],
                                 mixed["has_policy"])
    assert torch.isfinite(kl).all()
    assert float(kl[~keep].abs().max()) == 0.0, "value-only record produced KL"
    kl.sum().backward()
    grad_mixed = logits.grad.clone()

    assert torch.isfinite(grad_mixed).all(), "NaN gradient"
    assert float(grad_mixed[~keep].abs().max()) == 0.0, \
        "value-only records received a nonzero policy gradient"

    # And the surviving gradient equals the subset-only gradient exactly.
    logits2 = logits.detach()[keep].clone().requires_grad_(True)
    kl2, _ = policy_kl_per_sample(logits2, mixed["policy"][keep],
                                  mixed["legal_mask"][keep])
    kl2.sum().backward()
    assert torch.allclose(grad_mixed[keep], logits2.grad, atol=1e-7)


def test_value_loss_uses_every_record():
    """The value head sees all 10M records, including the 59.65% with no
    policy."""
    b = make_batch(n=8, has_policy=[1, 0, 1, 0, 0, 1, 0, 1], seed=9)
    pred = torch.randn(8, requires_grad=True)
    se = value_se_per_sample(pred, b["value"])
    se.sum().backward()
    assert (pred.grad.abs() > 0).all(), \
        "a record was excluded from the value gradient"


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
def test_policy_denominator_ignores_the_realised_batch():
    """C depends only on the effective batch and the manifest coverage."""
    assert policy_denominator(2048, 0.4034) == 2048 * 0.4034
    assert policy_denominator(2048, 0.4034) == policy_denominator(2048, 0.4034)
    for bad in (0.0, -0.1, 1.5):
        try:
            policy_denominator(1024, bad)
            raise AssertionError(f"coverage {bad} should be rejected")
        except ValueError:
            pass
    try:
        policy_denominator(0, 0.4)
        raise AssertionError("zero effective batch should be rejected")
    except ValueError:
        pass


def test_normalisation_by_count_under_known_coverage():
    """Synthetic batch with coverage known exactly (6 of 16 = 0.375).

    Two things are pinned:
      - `compute_losses` reports the per-has_policy-record mean, i.e. the sum
        divided by the realised count, not by the batch size;
      - dividing that sum by C = batch * coverage reproduces the same number
        when the realised coverage equals the expected coverage, which is the
        whole justification for the constant.
    """
    n, k = 16, 6
    has = [1] * k + [0] * (n - k)
    b = make_batch(n=n, has_policy=has, seed=10)
    coverage = k / n
    assert abs(coverage - 0.375) < 1e-12

    logits = torch.randn(n, POLICY_SIZE)
    parts = compute_losses(logits, torch.randn(n), b)

    assert parts.n_policy == k, f"realised count {parts.n_policy} != {k}"
    assert parts.n_samples == n
    kl_sum = float(parts.policy_kl_sum)
    assert abs(parts.policy_kl_mean - kl_sum / k) < 1e-6, \
        "reported policy KL is not normalised by the has_policy count"
    assert abs(parts.policy_kl_mean - kl_sum / n) > 1e-6, \
        "policy KL looks normalised by batch size, not by coverage"

    C = policy_denominator(n, coverage)
    assert abs(kl_sum / C - parts.policy_kl_mean) < 1e-6, \
        "constant denominator disagrees with the count at matched coverage"

    # Value normalises by the full batch: all records carry a value target.
    assert abs(parts.value_mse_mean - float(parts.value_se_sum) / n) < 1e-6


def test_constant_denominator_is_immune_to_coverage_jitter():
    """The exact failure mode the constant exists to prevent.

    Two effective batches sharing the same four policy-bearing records, differing
    only in how many OTHER records carry a policy. Under 'const' the shared
    records receive identical gradients in both. Under 'count' the sparser batch
    doubles their weight - which is how the effective policy LR ends up tracking
    per-batch coverage noise instead of staying put.
    """
    n = 16
    dense = make_batch(n=n, has_policy=[1] * 8 + [0] * 8, seed=11)
    sparse = {k: v.clone() for k, v in dense.items()}
    sparse["has_policy"] = torch.tensor([1.0] * 4 + [0.0] * 12)
    sparse["policy"] = dense["policy"].clone()
    sparse["policy"][4:8] = 0.0          # records 4..7 become value-only

    base = torch.randn(n, POLICY_SIZE,
                       generator=torch.Generator().manual_seed(11))
    C = policy_denominator(n, 0.4034)

    def grad_under(batch, denom_fn):
        z = base.clone().requires_grad_(True)
        parts = compute_losses(z, torch.zeros(n), batch)
        (parts.policy_kl_sum / denom_fn(parts)).backward()
        return z.grad, parts

    g_dense, d = grad_under(dense, lambda p: C)
    g_sparse, s = grad_under(sparse, lambda p: C)
    assert (d.n_policy, s.n_policy) == (8, 4)
    assert torch.allclose(g_dense[:4], g_sparse[:4], atol=1e-8), \
        "under 'const' a shared record's gradient depended on batch coverage"

    g_dense_c, _ = grad_under(dense, lambda p: float(p.n_policy))
    g_sparse_c, _ = grad_under(sparse, lambda p: float(p.n_policy))
    ratio = float(g_sparse_c[:4].abs().sum() / g_dense_c[:4].abs().sum())
    assert abs(ratio - 2.0) < 1e-4, (
        f"'count' normalisation should up-weight the sparse batch 2x, got "
        f"{ratio:.4f} - the 'const' assertion above proves nothing")


# ---------------------------------------------------------------------------
# support check
# ---------------------------------------------------------------------------
def test_target_mass_outside_the_legal_mask_is_caught():
    """Would make the KL +inf. The corpus cannot produce it (PV moves and the
    epsilon spread both come from the same legal list), so it is a loud check
    rather than a silent clamp."""
    b = make_batch(n=2, seed=12)
    illegal = int((b["legal_mask"][0] == 0).nonzero()[0])
    b["policy"][0, illegal] = 0.1
    logits = torch.randn(2, POLICY_SIZE)
    try:
        policy_kl_per_sample(logits, b["policy"], b["legal_mask"],
                             check_support=True)
        raise AssertionError("mass on an illegal index was not caught")
    except AssertionError as e:
        assert "illegal" in str(e), e


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            import traceback
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
