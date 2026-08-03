"""Architecture and metric checks for the v5 student.

Runs under pytest, or standalone:
    python training/v5_multiPV/tests/test_model.py

The head math is pinned against the engine's ChessTransformerV2 because MCTS
loads these checkpoints: the policy head must stay a from/to outer product
scaled 1/sqrt(head_dim), and the value head must stay a tanh MLP reading CLS at
position 67. The final-LayerNorm check is here because train.py's model does NOT
have one - Pre-LN without it hands both heads an unnormalised residual stream.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_V5 = _HERE.parent
_ROOT = _V5.parents[1]
_DATA = _ROOT / "data" / "multiPV"
for _p in (str(_V5), str(_DATA), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metrics import (PolicyAccumulator, ValueAccumulator,        # noqa: E402
                     pearson_r, value_strata)
from model_v5 import (ModelConfig, build_model,                  # noqa: E402
                      config_from_checkpoint, load_from_checkpoint)
from labels import VALUE_MATE_MIN                                # noqa: E402
from tests.test_losses import make_batch                         # noqa: E402


def test_default_config_matches_the_brief():
    c = ModelConfig()
    assert (c.d_model, c.num_layers, c.nhead, c.head_dim,
            c.dim_feedforward) == (384, 6, 6, 64, 1536)
    assert (c.vocab_size, c.seq_len, c.dropout) == (43, 68, 0.1)
    assert c.activation == "gelu" and c.norm_first is True
    assert c.smolgen is False


def test_final_layernorm_exists_before_both_heads():
    m = build_model()
    assert isinstance(m.final_norm, nn.LayerNorm), \
        "Pre-LN without a final LayerNorm leaves the output stream unnormalised"
    assert m.final_norm.normalized_shape == (m.config.d_model,)
    assert "final_norm.weight" in dict(m.named_parameters())

    # And it is genuinely in the path, not a decoration: scaling its weight
    # must change both heads' outputs.
    m.eval()
    x = torch.randint(0, 43, (3, 68))
    with torch.no_grad():
        p0, v0 = m(x)
        m.final_norm.weight.mul_(2.0)
        p1, v1 = m(x)
    assert not torch.allclose(p0, p1), "final_norm bypassed by the policy head"
    assert not torch.allclose(v0, v1), "final_norm bypassed by the value head"


def test_final_norm_can_be_disabled_and_the_flag_is_honest():
    m = build_model(ModelConfig(final_norm=False))
    assert isinstance(m.final_norm, nn.Identity)
    assert not any(k.startswith("final_norm.") for k in m.state_dict())


def test_ffn_activation_is_gelu_not_the_torch_default_relu():
    """nn.TransformerEncoderLayer defaults to ReLU and train.py never overrides
    it, so every earlier GuoFish generation has a ReLU FFN. v5 specifies GELU."""
    m = build_model()
    act = m.transformer.layers[0].activation
    assert act is torch.nn.functional.gelu or isinstance(act, nn.GELU), act


def test_parameter_count_is_in_the_expected_band():
    m = build_model()
    n = m.num_parameters()
    assert 10_000_000 < n < 12_000_000, f"{n:,} params, expected ~10.9M"
    b = m.param_breakdown()
    assert b["total"] == n
    # ~10.6M of it is the encoder stack; the heads are small on purpose.
    assert b["encoder"] > 10_000_000
    assert b["policy_head"] == 2 * (384 * 64 + 64)
    assert b["value_head"] == (384 * 384 + 384) + (384 + 1)


def test_policy_head_is_a_scaled_from_to_outer_product():
    """Reproduce the head by hand from the encoder output. This is the exact
    contract the MCTS engine relies on."""
    torch.manual_seed(0)
    m = build_model(ModelConfig(d_model=64, num_layers=2, nhead=4,
                                dim_feedforward=128, dropout=0.0)).eval()
    x = torch.randint(0, 43, (2, 68))
    with torch.no_grad():
        h = m.embedding(x) + m.pos_encoder
        h = m.transformer(h)
        h = m.final_norm(h)
        f = m.from_proj(h[:, :64, :])
        t = m.to_proj(h[:, :64, :])
        ref = torch.bmm(f, t.transpose(1, 2)).reshape(2, 4096) / math.sqrt(m.head_dim)
        got, _ = m(x)
    assert torch.allclose(got, ref, atol=1e-5)
    assert abs(m.logit_scale - 1.0 / math.sqrt(m.head_dim)) < 1e-12


def test_value_head_reads_cls_at_67_and_is_bounded():
    torch.manual_seed(0)
    m = build_model(ModelConfig(d_model=64, num_layers=2, nhead=4,
                                dim_feedforward=128, dropout=0.0)).eval()
    x = torch.randint(0, 43, (4, 68))
    with torch.no_grad():
        h = m.final_norm(m.transformer(m.embedding(x) + m.pos_encoder))
        ref = m.value_head(h[:, 67, :]).squeeze(-1)
        _, got = m(x)
    assert torch.allclose(got, ref, atol=1e-6)
    assert m.cls_index == 67
    assert got.abs().max() < 1.0, "tanh output must stay inside (-1, 1)"


def test_forward_shapes_and_mask_semantics():
    m = build_model(ModelConfig(d_model=64, num_layers=2, nhead=4,
                                dim_feedforward=128, dropout=0.0)).eval()
    b = make_batch(n=3, seed=40)
    with torch.no_grad():
        logits, value = m(b["tokens"], legal_move_mask=b["legal_mask"] > 0)
    assert logits.shape == (3, 4096) and value.shape == (3,)
    illegal = b["legal_mask"] == 0
    assert torch.all(torch.isinf(logits[illegal]) & (logits[illegal] < 0))
    # A float mask must be accepted too - the loss path passes floats around.
    with torch.no_grad():
        logits2, _ = m(b["tokens"], legal_move_mask=b["legal_mask"])
    assert torch.equal(logits, logits2)


def test_checkpoint_round_trip_carries_the_architecture():
    m = build_model(ModelConfig(d_model=128, num_layers=3, nhead=4,
                                dim_feedforward=256, head_dim=32))
    ckpt = {"model_state_dict": m.state_dict(), "config": m.config.to_dict()}
    cfg = config_from_checkpoint(ckpt)
    assert cfg.d_model == 128 and cfg.num_layers == 3 and cfg.head_dim == 32
    m2 = load_from_checkpoint(ckpt)
    for (k1, v1), (k2, v2) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)


def test_config_inferred_when_the_checkpoint_predates_the_config_key():
    m = build_model(ModelConfig(d_model=128, num_layers=3, nhead=4,
                                dim_feedforward=256, head_dim=32))
    cfg = config_from_checkpoint({"model_state_dict": m.state_dict()})
    assert (cfg.d_model, cfg.num_layers, cfg.dim_feedforward,
            cfg.head_dim, cfg.final_norm) == (128, 3, 256, 32, True)


def test_weight_decay_group_split():
    """Weight decay constrains the function a weight MATRIX represents. Applied
    to LayerNorm gains it shrinks the activation scale feeding the next block -
    compounding through 13 Pre-LN norms - and applied to pos_encoder it erodes
    positional information, including position 67 that the value head reads."""
    from train_v5 import param_groups

    m = build_model()
    groups = param_groups(m, 0.01)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.01
    assert groups[1]["weight_decay"] == 0.0

    decayed = {id(p) for p in groups[0]["params"]}
    undecayed = {id(p) for p in groups[1]["params"]}

    for name, p in m.named_parameters():
        if "norm" in name or name.endswith(".bias") or "in_proj_bias" in name:
            assert id(p) in undecayed, f"{name} must not be decayed"
        elif name == "pos_encoder":
            # ndim == 3, so a bare `ndim >= 2` rule would wrongly decay it.
            assert p.ndim == 3
            assert id(p) in undecayed, "pos_encoder must not be decayed"
        elif name == "embedding.weight":
            assert id(p) in decayed, "token embedding is decayed by convention"
        else:
            assert p.ndim >= 2 and id(p) in decayed, f"{name} should be decayed"


def test_weight_decay_groups_partition_every_parameter():
    """The dangerous failure is silent: a parameter dropped from both groups
    gets NO optimizer update at all, and nothing would report it."""
    from train_v5 import param_groups

    m = build_model()
    groups = param_groups(m, 0.01)
    seen = [id(p) for g in groups for p in g["params"]]
    assert len(seen) == len(set(seen)), "a parameter landed in both groups"
    assert set(seen) == {id(p) for p in m.parameters() if p.requires_grad}, \
        "param_groups dropped or invented parameters"

    n_split = sum(p.numel() for g in groups for p in g["params"])
    assert n_split == m.num_parameters(), \
        f"{n_split:,} params in groups vs {m.num_parameters():,} in the model"

    # And the undecayed set is the small one it should be: ~0.3% of the model.
    n_no = sum(p.numel() for p in groups[1]["params"])
    assert 20_000 < n_no < 60_000, f"undecayed set is {n_no:,}, expected ~31k"


def test_smolgen_is_refused_rather_than_silently_ignored():
    try:
        ModelConfig(smolgen=True)
        raise AssertionError("smolgen=True should not be silently accepted")
    except NotImplementedError:
        pass


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_value_strata_partition_matches_pass_b():
    cp = torch.tensor([0.0, 1.0, -1.0, 2000.0, -2000.0,
                       float(VALUE_MATE_MIN), -float(VALUE_MATE_MIN),
                       29999.0, -30000.0])
    s = value_strata(cp)
    assert s["exact_zero"].tolist() == [True] + [False] * 8
    assert s["mate"].tolist() == [False] * 5 + [True] * 4
    assert s["middle"].tolist() == [False, True, True, True, True] + [False] * 4
    # A partition: every record in exactly one stratum.
    total = s["exact_zero"].int() + s["mate"].int() + s["middle"].int()
    assert torch.equal(total, torch.ones_like(total))


def test_stratified_report_catches_a_collapsed_head():
    """A head that always outputs 0 looks fine on the exact-zero fifth of the
    corpus and terrible everywhere else. The aggregate hides it; the strata
    do not."""
    n = 900
    cp = torch.cat([torch.zeros(300),
                    torch.full((300,), 30000.0),
                    torch.linspace(-1500, 1500, 300)])
    target = torch.cat([torch.zeros(300),
                        torch.full((300,), 0.995),
                        torch.tanh(torch.linspace(-1500, 1500, 300) / 290.6806)])
    collapsed = torch.zeros(n)
    acc = ValueAccumulator()
    acc.add(collapsed, target, cp)
    m = acc.result()

    assert m["value_exact_zero_mse"] == 0.0, "collapsed head should ace the zeros"
    assert m["value_mate_mse"] > 0.9, "collapsed head should fail the mates"
    assert m["value_pred_std"] == 0.0, "prediction std must expose the collapse"
    assert math.isnan(m["value_pearson_r"]), \
        "a constant prediction has undefined correlation, not 0"
    assert m["value_exact_zero_n"] == 300 and m["value_mate_n"] == 300
    assert m["value_middle_n"] == 300


def test_pearson_r_matches_a_known_value():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert abs(pearson_r(a, 2 * a + 1) - 1.0) < 1e-9
    assert abs(pearson_r(a, -a) + 1.0) < 1e-9
    assert math.isnan(pearson_r(a, torch.ones(4)))


def test_policy_accumulator_counts_only_has_policy_records():
    b = make_batch(n=10, has_policy=[1, 0] * 5, seed=41)
    acc = PolicyAccumulator(topk=(1, 5))
    # Perfect logits on the has_policy records.
    logits = torch.log(b["policy"].clamp_min(1e-30))
    logits[b["has_policy"] == 0] = torch.randn(4096)
    acc.add(logits, b)
    r = acc.result()
    assert r["policy_n"] == 5
    assert abs(r["policy_coverage"] - 0.5) < 1e-9
    assert r["policy_top1"] == 1.0 and r["policy_top5"] == 1.0
    assert r["policy_kl"] < 1e-5


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
