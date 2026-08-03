"""Gradient accumulation invariance.

Runs under pytest, or standalone:
    python training/v5_multiPV/tests/test_accum.py

The contract from the addendum: an effective batch fed as 1 chunk and as 4
chunks must produce the SAME gradient. That is only true if every micro-batch
divides by the same constant. Divide by the micro-batch's own has_policy count
and you get a mean-of-means, which silently up-weights records that happened to
land in a sparse micro-batch - and the effective policy LR then tracks
per-batch coverage noise instead of staying put.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_V5 = _HERE.parent
_ROOT = _V5.parents[1]
_DATA = _ROOT / "data" / "multiPV"
for _p in (str(_V5), str(_DATA), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from losses import compute_losses, policy_denominator      # noqa: E402
from model_v5 import ModelConfig, build_model              # noqa: E402
from train_v5 import plan_epoch                            # noqa: E402
from tests.test_losses import make_batch                   # noqa: E402

COVERAGE = 0.4034


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return build_model(ModelConfig(d_model=64, num_layers=2, nhead=4,
                                   dim_feedforward=128, head_dim=16,
                                   dropout=0.0))


def _run(model, batch, chunks, effective, coverage=COVERAGE, by_count=False):
    """Accumulate over `chunks` slices of one effective batch; return grads."""
    model.zero_grad(set_to_none=True)
    C_pol = policy_denominator(effective, coverage)
    C_val = float(effective)
    n = batch["has_policy"].shape[0]
    size = n // chunks
    for c in range(chunks):
        sl = slice(c * size, (c + 1) * size)
        sub = {k: v[sl] for k, v in batch.items()}
        logits, value = model(sub["tokens"])
        parts = compute_losses(logits, value, sub)
        if by_count:
            denom = max(1.0, float(parts.n_policy))
            loss = parts.policy_kl_sum / denom + parts.value_se_sum / float(size)
        else:
            loss = parts.policy_kl_sum / C_pol + parts.value_se_sum / C_val
        loss.backward()
    return {k: p.grad.detach().clone() for k, p in model.named_parameters()
            if p.grad is not None}


def test_accum_1_matches_accum_4_exactly():
    model = _tiny_model(1)
    # Deliberately uneven coverage across the four quarters: 4 / 1 / 3 / 0.
    has = ([1, 1, 1, 1, 0, 0, 0, 0] +
           [1, 0, 0, 0, 0, 0, 0, 0] +
           [1, 1, 1, 0, 0, 0, 0, 0] +
           [0, 0, 0, 0, 0, 0, 0, 0])
    batch = make_batch(n=32, has_policy=has, seed=21)

    g1 = _run(model, batch, chunks=1, effective=32)
    g4 = _run(model, batch, chunks=4, effective=32)

    worst = 0.0
    for k in g1:
        d = float((g1[k] - g4[k]).abs().max())
        worst = max(worst, d)
    assert worst < 1e-6, f"accum=1 vs accum=4 gradients differ by {worst:.3e}"


def test_accum_8_also_matches():
    model = _tiny_model(2)
    has = [1, 0, 0, 1, 1, 0, 0, 0] * 4
    batch = make_batch(n=32, has_policy=has, seed=22)
    g1 = _run(model, batch, chunks=1, effective=32)
    g8 = _run(model, batch, chunks=8, effective=32)
    worst = max(float((g1[k] - g8[k]).abs().max()) for k in g1)
    assert worst < 1e-6, f"accum=1 vs accum=8 gradients differ by {worst:.3e}"


def test_per_microbatch_count_normalisation_is_NOT_invariant():
    """The negative control. If this ever passes, the test above is vacuous."""
    model = _tiny_model(3)
    has = ([1, 1, 1, 1, 0, 0, 0, 0] +
           [1, 0, 0, 0, 0, 0, 0, 0] +
           [1, 1, 1, 0, 0, 0, 0, 0] +
           [1, 0, 0, 0, 0, 0, 0, 0])
    batch = make_batch(n=32, has_policy=has, seed=23)
    g1 = _run(model, batch, chunks=1, effective=32, by_count=True)
    g4 = _run(model, batch, chunks=4, effective=32, by_count=True)
    worst = max(float((g1[k] - g4[k]).abs().max()) for k in g1)
    assert worst > 1e-5, (
        "per-micro-batch count normalisation appears accumulation-invariant; "
        f"max diff {worst:.3e} - the invariance test proves nothing")


def test_value_gradient_is_also_invariant():
    """Value divides by micro_batch * accum_steps, a constant, so the same
    argument applies - and the short-final-micro-batch case is what breaks it
    if the constant is assumed rather than computed."""
    model = _tiny_model(4)
    batch = make_batch(n=24, has_policy=[0] * 24, seed=24)   # value-only
    g1 = _run(model, batch, chunks=1, effective=24)
    g3 = _run(model, batch, chunks=3, effective=24)
    worst = max(float((g1[k] - g3[k]).abs().max()) for k in g1)
    assert worst < 1e-6, f"value-only gradients differ by {worst:.3e}"
    # And the policy head got nothing at all from a value-only effective batch.
    for name in ("from_proj.weight", "to_proj.weight"):
        assert float(g1[name].abs().max()) == 0.0, \
            f"{name} received gradient from value-only records"


# ---------------------------------------------------------------------------
# epoch planning: the short final window must be sized, not assumed
# ---------------------------------------------------------------------------
def test_plan_epoch_full_windows():
    sizes, windows = plan_epoch(1000, micro_batch=100, accum=2, drop_last=True)
    assert sizes == [100] * 10
    assert windows == [(2, 200)] * 5


def test_plan_epoch_short_final_window():
    """9 micro-batches at accum 4 -> windows of 4, 4, 1. The last window must
    divide by 1 micro-batch worth of samples, not 4."""
    sizes, windows = plan_epoch(900, micro_batch=100, accum=4, drop_last=True)
    assert len(sizes) == 9
    assert windows == [(4, 400), (4, 400), (1, 100)]
    assert sum(w[1] for w in windows) == 900


def test_plan_epoch_short_final_micro_batch():
    """drop_last=False leaves a ragged tail; the denominator follows it."""
    sizes, windows = plan_epoch(950, micro_batch=100, accum=4, drop_last=False)
    assert sizes == [100] * 9 + [50]
    assert windows == [(4, 400), (4, 400), (2, 150)]
    assert sum(sizes) == 950
    assert sum(w[1] for w in windows) == 950


def test_plan_epoch_drop_last_discards_the_tail():
    sizes, windows = plan_epoch(950, micro_batch=100, accum=4, drop_last=True)
    assert sum(sizes) == 900
    assert all(s == 100 for s in sizes)


def test_short_window_denominator_matches_its_actual_samples():
    """End to end: a 3-micro-batch window normalised by its own 300 samples
    equals the same 300 samples run in one shot."""
    model = _tiny_model(5)
    batch = make_batch(n=30, has_policy=[1, 0, 1] * 10, seed=25)
    g1 = _run(model, batch, chunks=1, effective=30)
    g3 = _run(model, batch, chunks=3, effective=30)
    worst = max(float((g1[k] - g3[k]).abs().max()) for k in g1)
    assert worst < 1e-6, f"short window differs by {worst:.3e}"

    # Using the FULL window size (as if the tail were not short) rescales
    # everything - which is exactly the silent reweighting to avoid.
    g_wrong = _run(model, batch, chunks=3, effective=40)
    ratio = float(g_wrong["from_proj.weight"].abs().sum()
                  / g1["from_proj.weight"].abs().sum())
    assert abs(ratio - 30 / 40) < 1e-4, \
        f"assuming a full final window should rescale by 0.75, got {ratio:.4f}"


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
