"""The MAX_LEGAL truncation defect, and the repair that makes KL finite.

Runs under pytest, or standalone:
    python training/v5_multiPV/tests/test_truncated_legal.py

THE DEFECT
==========
`legal_idx` is a fixed-width field of MAX_LEGAL = 128 entries and Pass B
truncates it. A position with more than 128 legal moves therefore stores an
INCOMPLETE legal set, and if one of its PV moves is among the ones dropped, the
policy target carries mass at an index the stored mask calls illegal. log q is
-inf there, so that record's KL is +inf - and one +inf makes the MEAN KL over
its entire split +inf.

Measured on data/processed/multipv_90m:

    val    1 record  of 271,876 policy records   (val_0008.bin row 31344)
    train  6 records of 5.77M policy records sampled (~1e-6)

every one of them with n_legal == 128 exactly. It is not new - the 20M run's log
carries `KL inf` in 45 separate windows.

WHY IT WAS SURVIVABLE BEFORE AND IS NOT NOW
===========================================
The gradient is clean: `loss.backward()` seeds grad_output = 1.0 on the scalar,
so the +inf VALUE never enters the backward and the weights are unharmed. That
is pinned below, because it is the reason nine epochs of the 20M run completed
with healthy grad norms while printing an infinite loss.

What it does destroy is every mean that contains it, and this run cannot absorb
that: an infinite epoch `total_loss` means `best_val` never improves, so
`_best.pt` is never written and the head-to-head gate has no candidate to play.
The pre-registered "5% KL reduction" is likewise unmeasurable against an
infinite KL.

THE REPAIR
==========
The target came from a multi-PV search, which only ever proposes legal moves, so
an index carrying target mass IS legal and the stored mask is merely missing it.
Unioning the target's support into the mask repairs the lossy field from the
reliable one. For every record whose support already sits inside the legal set -
all but ~1e-6 - the union is the identity, which is the property that makes this
safe and is tested first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
_V5 = _HERE.parent
_ROOT = _V5.parents[1]
_DATA = _ROOT / "data" / "multiPV"
for _p in (str(_V5), str(_DATA), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from losses import (compute_losses, masked_log_softmax,           # noqa: E402
                    policy_kl_per_sample)
from tests.test_losses import make_batch                          # noqa: E402

POLICY_SIZE = 4096
MAX_LEGAL_FIELD = 128


def truncated_batch(n_extra_records: int = 3, seed: int = 0) -> dict:
    """A batch shaped like the shards: healthy records plus one whose legal
    list was truncated at 128 past one of its own PV moves."""
    b = make_batch(n=n_extra_records + 1, n_legal=20, n_pv=3, seed=seed)
    bad = n_extra_records

    b["policy"][bad] = 0.0
    b["legal_mask"][bad] = 0.0
    legal_idx = torch.arange(500, 500 + MAX_LEGAL_FIELD)
    b["legal_mask"][bad, legal_idx] = 1.0
    b["policy"][bad, legal_idx[:3]] = 0.3
    b["policy"][bad, 1373] = 0.1          # the PV move truncation dropped
    return b, bad


# ---------------------------------------------------------------------------
# the safety property: healthy records are untouched, bit for bit
# ---------------------------------------------------------------------------
def test_repair_is_the_identity_when_support_is_inside_the_legal_set():
    """All but ~1e-6 of the corpus takes this path, so it must be EXACT."""
    b = make_batch(n=8, n_legal=20, n_pv=3, seed=3)
    logits = torch.randn(8, POLICY_SIZE, generator=torch.Generator().manual_seed(1))

    on, _ = policy_kl_per_sample(logits, b["policy"], b["legal_mask"],
                                 repair_truncated_legal=True)
    off, _ = policy_kl_per_sample(logits, b["policy"], b["legal_mask"],
                                 repair_truncated_legal=False)
    assert torch.equal(on, off), "the repair perturbed a healthy record"


def test_extra_legal_union_is_the_identity_on_a_subset_mask():
    b = make_batch(n=4, n_legal=20, seed=5)
    logits = torch.randn(4, POLICY_SIZE)
    plain, m0 = masked_log_softmax(logits, b["legal_mask"])
    unioned, m1 = masked_log_softmax(logits, b["legal_mask"],
                                     extra_legal=b["policy"] > 0)
    assert torch.equal(m0, m1)
    assert torch.equal(plain, unioned)


# ---------------------------------------------------------------------------
# the defect, and that the repair removes it
# ---------------------------------------------------------------------------
def test_truncated_record_is_infinite_without_the_repair():
    b, bad = truncated_batch()
    logits = torch.randn(b["policy"].shape[0], POLICY_SIZE)
    kl, _ = policy_kl_per_sample(logits, b["policy"], b["legal_mask"],
                                 repair_truncated_legal=False)
    assert torch.isinf(kl[bad]), "the defect did not reproduce"
    assert torch.isfinite(kl[:bad]).all()
    assert torch.isinf(kl.mean()), "one +inf must poison the whole mean"


def test_repair_makes_it_finite_and_leaves_its_neighbours_alone():
    b, bad = truncated_batch()
    logits = torch.randn(b["policy"].shape[0], POLICY_SIZE)
    fixed, _ = policy_kl_per_sample(logits, b["policy"], b["legal_mask"],
                                    repair_truncated_legal=True)
    broken, _ = policy_kl_per_sample(logits, b["policy"], b["legal_mask"],
                                     repair_truncated_legal=False)
    assert torch.isfinite(fixed).all()
    assert torch.isfinite(fixed.mean())
    # The healthy records in the SAME batch are unchanged.
    assert torch.equal(fixed[:bad], broken[:bad])


def test_repaired_kl_is_a_real_kl_not_a_shrug():
    """The repaired record must behave like a normal KL: non-negative, and
    exactly zero when the model reproduces the target."""
    b, bad = truncated_batch()
    target = b["policy"][bad]
    # Logits that reproduce the target exactly over its support.
    logits = torch.full((1, POLICY_SIZE), -30.0)
    logits[0, target > 0] = torch.log(target[target > 0])
    kl, _ = policy_kl_per_sample(logits, target.unsqueeze(0),
                                 b["legal_mask"][bad].unsqueeze(0))
    assert float(kl[0]) >= -1e-5
    assert float(kl[0]) == pytest.approx(0.0, abs=2e-3)


def test_the_repaired_index_receives_real_probability():
    """Not merely finite - the truncated-away move must be a live option the
    model can be graded on, otherwise the repair is just hiding the record."""
    b, bad = truncated_batch()
    logits = torch.zeros(1, POLICY_SIZE)
    log_q, mask = masked_log_softmax(
        logits, b["legal_mask"][bad].unsqueeze(0),
        extra_legal=(b["policy"][bad] > 0).unsqueeze(0))
    assert bool(mask[0, 1373]), "the truncated PV index is still masked out"
    assert torch.isfinite(log_q[0, 1373])
    assert float(log_q.exp().sum()) == pytest.approx(1.0, abs=1e-5)
    # 128 stored legal moves plus the one recovered from the target.
    assert int(mask.sum()) == MAX_LEGAL_FIELD + 1


# ---------------------------------------------------------------------------
# why nine epochs of the 20M run survived printing an infinite loss
# ---------------------------------------------------------------------------
def test_the_infinite_forward_does_not_poison_the_backward():
    """Pinned because it is the difference between 'the metrics are wrong' and
    'the 17-hour run is destroyed', and the answer is not obvious."""
    b, bad = truncated_batch()
    n = b["policy"].shape[0]
    logits = torch.randn(n, POLICY_SIZE, requires_grad=True)
    value = torch.zeros(n, requires_grad=True)

    parts = compute_losses(logits, value, b)
    loss = parts.policy_kl_sum / float(n)
    assert torch.isinf(loss) or torch.isfinite(loss)

    loss.backward()
    assert torch.isfinite(logits.grad).all(), \
        "backward is poisoned - the defect would destroy the run, not just the log"
    assert int(torch.isnan(logits.grad).sum()) == 0


def test_check_support_still_reports_the_defect_with_the_repair_on():
    """The repair must not silence the diagnostic. They answer different
    questions and both have to keep working."""
    b, bad = truncated_batch()
    logits = torch.randn(b["policy"].shape[0], POLICY_SIZE)
    with pytest.raises(AssertionError, match="illegal"):
        policy_kl_per_sample(logits, b["policy"], b["legal_mask"],
                             check_support=True, repair_truncated_legal=True)


# ---------------------------------------------------------------------------
# against the real record that broke the 90M val split
# ---------------------------------------------------------------------------
def test_the_actual_offending_record_from_the_90m_val_split():
    """val_0008.bin row 31344: 5 PV entries, 128 stored legal moves, PV index
    1373 among the truncated. Skipped where the shards are not present."""
    shard = _ROOT / "data" / "processed" / "multipv_90m" / "val_0008.bin"
    if not shard.exists():
        pytest.skip("90M shards not on this box")

    import numpy as np
    from record_format import RECORD_DTYPE

    m = np.memmap(shard, dtype=RECORD_DTYPE, mode="r")
    try:
        rec = m[31344]
        n_pv, n_legal = int(rec["n_pv"]), int(rec["n_legal"])
        assert int(rec["has_policy"]) == 1
        assert n_legal == MAX_LEGAL_FIELD, \
            "the record no longer looks truncated; was the corpus rebuilt?"
        pv = set(rec["pv_idx"][:n_pv].tolist())
        legal = set(rec["legal_idx"][:n_legal].tolist())
        outside = pv - legal
        assert outside, "the defect is gone from the shard - update this test"

        policy = torch.zeros(1, POLICY_SIZE)
        mask = torch.zeros(1, POLICY_SIZE)
        mask[0, list(legal)] = 1.0
        probs = rec["pv_prob"][:n_pv].astype("float32")
        for i, p in zip(rec["pv_idx"][:n_pv].tolist(), probs.tolist()):
            policy[0, i] += p
    finally:
        m._mmap.close()

    logits = torch.randn(1, POLICY_SIZE)
    broken, _ = policy_kl_per_sample(logits, policy, mask,
                                     repair_truncated_legal=False)
    fixed, _ = policy_kl_per_sample(logits, policy, mask,
                                    repair_truncated_legal=True)
    assert torch.isinf(broken[0]), "the real record no longer reproduces +inf"
    assert torch.isfinite(fixed[0]), "the repair does not fix the real record"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
