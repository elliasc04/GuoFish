"""M1 gates: v5-compatible shape, v5 parity through the converter, S8."""
from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from core.guofish_net import ModelConfig, build_model, hlgauss_centers, hlgauss_target
from core.guofish_net.tokenizers import (
    board_from_v5_tokens, tokens_canonical_65, tokens_v5_68,
)
from core.guofish_net.v5_compat import load_v5_checkpoint
from training.v5_multiPV.model_v5 import load_from_checkpoint as v5_load
from training.v6.data.formats import REPO
from training.v6.data.reader import ShardSet

FROZEN = REPO / "data/processed/val_frozen_90m_v1"
V5_CKPT = REPO / "models/guofish5_90M/v5_10.9M_best.pt"


@pytest.fixture(scope="module")
def val_tokens():
    ss = ShardSet(FROZEN, "val")
    idx = np.sort(np.random.default_rng(20260924).choice(len(ss), 1000, replace=False))
    tok = torch.from_numpy(ss.read(idx)["tokens"].astype(np.int64))
    ss.close()
    return tok


@pytest.fixture(scope="module")
def v6_from_v5():
    return load_v5_checkpoint(V5_CKPT).eval()


def _run(model, tok, grad: bool):
    with torch.set_grad_enabled(grad):
        outs = [model(t) for t in tok.split(250)]
    return torch.cat([o[0] for o in outs]).detach(), torch.cat([o[1] for o in outs]).detach()


def test_v5_compat_param_count(v6_from_v5):
    assert sum(p.numel() for p in v6_from_v5.parameters()) == 10_887_681


@pytest.mark.parametrize("grad", [False, True], ids=["v5_fastpath", "v5_slowpath"])
def test_v5_parity_on_frozen_val(val_tokens, v6_from_v5, grad):
    """fp32, CPU, eval mode, 1,000 seeded frozen-val records, within 1e-5.

    Under no_grad v5's nn.TransformerEncoder takes the fused fast path;
    with grad on it runs the plain modules. Both are checked."""
    v5 = v5_load(torch.load(V5_CKPT, map_location="cpu", weights_only=True)).eval()
    p5, v5v = _run(v5, val_tokens, grad)
    p6, v6v = _run(v6_from_v5, val_tokens, False)
    dp = (p5 - p6).abs().max().item()
    dv = (v5v - v6v).abs().max().item()
    print(f"\nPARITY[{'slow' if grad else 'fast'}] policy max|d| {dp:.3e} "
          f"(|logit| max {p5.abs().max().item():.2f}) value max|d| {dv:.3e}")
    assert dp <= 1e-5 and dv <= 1e-5


def test_v5_parity_float64(val_tokens, v6_from_v5):
    """The fp32 residual (~1e-5) is kernel rounding, not a mapping error: run
    in float64, v6's output (cast to fp32 by contract, §7) is v5's float64
    output rounded to fp32, to within one fp32 ulp."""
    v5 = v5_load(torch.load(V5_CKPT, map_location="cpu", weights_only=True)).double().eval()
    v6 = load_v5_checkpoint(V5_CKPT).double().eval()
    p5, v5v = _run(v5, val_tokens, True)
    p6, v6v = _run(v6, val_tokens, False)
    assert p6.dtype == torch.float32 and v6v.dtype == torch.float32
    ulp = 2.0 ** -23
    rp = ((p5 - p6.double()).abs() / p5.abs().clamp_min(1e-30)).max().item()
    rv = ((v5v - v6v.double()).abs() / v5v.abs().clamp_min(1e-30)).max().item()
    print(f"\nPARITY[float64] policy max rel|d| {rp:.3e} value max rel|d| {rv:.3e} "
          f"(fp32 ulp {ulp:.3e})")
    assert rp <= ulp and rv <= ulp


@pytest.mark.parametrize("bias", ["static", "smolgen"])
def test_s8_zeroed_bias_reproduces_plain(val_tokens, v6_from_v5, bias):
    """S8: a bias model loaded with a plain model's weights and a zeroed bias
    output reproduces the plain model; a non-zero bias then changes it."""
    torch.manual_seed(8)
    cfg = ModelConfig.from_dict({**v6_from_v5.cfg.to_dict(), "attn_bias": bias})
    biased = build_model(cfg).eval()
    res = biased.load_state_dict(v6_from_v5.state_dict(), strict=False)
    assert not res.unexpected_keys
    if bias == "static":
        assert res.missing_keys == ["static_bias"]
        zero_out = biased.static_bias
    else:
        assert all(k.startswith("smolgen_shared") or ".smolgen." in k for k in res.missing_keys)
        assert "smolgen_shared.weight" in res.missing_keys
        zero_out = biased.smolgen_shared.weight
        # the per-layer generator is live (random), only its output is zero
        assert biased.blocks[0].smolgen.fc1.weight.abs().sum() > 0
    assert torch.count_nonzero(zero_out) == 0

    tok = val_tokens[:256]
    p0, v0 = _run(v6_from_v5, tok, False)
    p1, v1 = _run(biased, tok, False)
    dp, dv = (p0 - p1).abs().max().item(), (v0 - v1).abs().max().item()
    print(f"\nS8[{bias}] zeroed: policy max|d| {dp:.3e} value max|d| {dv:.3e}")
    assert torch.equal(p0, p1) and torch.equal(v0, v1)

    with torch.no_grad():
        zero_out.normal_(std=0.5)
    p2, _ = _run(biased, tok, False)
    moved = (p0 - p2).abs().max().item()
    print(f"S8[{bias}] live bias moves policy by {moved:.3e}")
    assert moved > 1e-3


def test_init_modes():
    torch.manual_seed(0)
    dc = build_model(ModelConfig(init="v5_deepcopy", dropout=0.1))
    sd = dc.state_dict()
    for k in [k for k in sd if k.startswith("blocks.1.")]:
        assert torch.equal(sd[k], sd[k.replace("blocks.1.", "blocks.0.")]), k
    ind = build_model(ModelConfig(init="independent"))
    assert not torch.equal(ind.blocks[0].qkv.weight, ind.blocks[1].qkv.weight)
    assert abs(ind.blocks[0].qkv.weight.std().item() - 0.02) < 1e-3
    assert abs(ind.blocks[0].ff2.weight.std().item() - 0.02 / 12 ** 0.5) < 5e-4
    assert torch.count_nonzero(ind.blocks[0].qkv.bias) == 0


def test_model_config_strict():
    with pytest.raises(KeyError, match=r"model\.smolgen\.bogus"):
        ModelConfig.from_dict({"smolgen": {"bogus": 1}})
    with pytest.raises(ValueError, match="attn_bias"):
        ModelConfig.from_dict({"attn_bias": "alibi"})
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig.from_dict({"d_model": 100, "n_heads": 6})
    assert ModelConfig.from_dict(ModelConfig().to_dict()) == ModelConfig()


def test_hlgauss():
    y = torch.tensor([-0.995, -0.5, 0.0, 0.3317, 0.9951])
    t = hlgauss_target(y)
    assert torch.allclose(t.sum(1), torch.ones(5), atol=1e-6)
    c = hlgauss_centers()
    assert abs(c.max().item() - 0.990099) < 1e-5
    mean = t @ c
    assert (mean[1:4] - y[1:4]).abs().max() < 1e-4       # interior: unbiased


def test_v5_tokens_decode_roundtrip(val_tokens):
    for row in val_tokens[:1000].numpy():
        assert np.array_equal(tokens_v5_68(board_from_v5_tokens(row)), row.astype(np.int8))


def test_canonical_tokenizer_hand_positions():
    # Black to move, legal ep capture d4xe3, full castling rights.
    b = chess.Board("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3")
    assert b.has_legal_en_passant()
    t = tokens_canonical_65(b)
    assert t[chess.E3 ^ 56] == 15                     # ep target, flipped to e6
    assert t[0] == 13 and t[7] == 13                  # our (black) castling rooks
    assert t[56] == 14 and t[63] == 14                # their castling rooks
    assert t[chess.E8 ^ 56] == 6 and t[chess.E1 ^ 56] == 12   # kings: ours/theirs
    assert t[chess.D4 ^ 56] == 1                      # our pawn
    assert t[64] == 16

    # Partial rights: Black keeps q only, White keeps K only.
    t = tokens_canonical_65(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1"))
    assert (t[0], t[7], t[63], t[56]) == (13, 4, 14, 10)

    # ep square set but no legal capture: no ep token.
    t = tokens_canonical_65(chess.Board("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1"))
    assert 15 not in t

    for fen in ["r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",
                "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
                "8/P5k1/8/8/8/8/6K1/8 w - - 0 1"]:
        b = chess.Board(fen)
        assert np.array_equal(tokens_canonical_65(b), tokens_canonical_65(b.mirror())), fen
