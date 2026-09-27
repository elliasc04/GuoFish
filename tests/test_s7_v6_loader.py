"""S7 — the engine loads v6 exports through `core.guofish_net.load_for_inference`.

CPU only; the captured / compiled path and the search are certified on the GPU
by `tools/s7_check.py`. What is pinned here:

  1. A contract-A export loads through the core package, the policy leaves the
     module as bf16 (the policy buffer's dtype) and both outputs equal the
     training-side forward exactly.
  2. Storing the Linear layers in bf16 changes no output bit under bf16 autocast:
     that cast is the one autocast applies anyway.
  3. Contract-B exports and exports without a value_scale are refused.
  4. v5 and legacy checkpoints are not mistaken for v6 exports; they keep the
     loader every golden is anchored to.

Per Amendment D there are no module-scope skips; tests that need a checkpoint on
disk are marked individually with the reason.
"""
import json
from pathlib import Path

import pytest
import torch

import guofish_core
from core.guofish_net import ModelConfig, build_model, load_for_inference
from playing.v6 import evaluator as ev
from training.v6.ckpt import atomic_save, export_blob

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "golden" / "c10_corpus.json"
VALUE_SCALE = 290.6806


def _export(tmp_path: Path, name: str = "net", value_scale=VALUE_SCALE, drop_key=False,
            **cfg) -> Path:
    mc = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, **cfg)
    net = build_model(mc, init_seed=7)
    with torch.no_grad():          # a zero-initialised bias would hide a wiring fault
        for p in net.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=torch.Generator().manual_seed(p.numel())))
    blob = export_blob({k: v.float() for k, v in net.state_dict().items()}, mc, weights="raw",
                       source_run=name, source_ckpt="test", cfg_hash="0" * 64, samples=0,
                       value_scale=value_scale)
    if drop_key:                   # an export written before value_scale was recorded
        del blob["value_scale"]
    path = tmp_path / f"{name}.pt"
    atomic_save(blob, path)
    return path


def _tokens(n: int = 16) -> torch.Tensor:
    """Rows as the C++ tokenizer writes them, int64 as the graph widens them."""
    fens = [p["fen"] for p in json.loads(CORPUS.read_text(encoding="utf-8"))["positions"][:n]]
    return torch.stack([torch.from_numpy(guofish_core.tokens(f)) for f in fens]).long()


@pytest.mark.parametrize("attn_bias", ["none", "static", "smolgen"])
def test_a_v6_export_loads_through_the_core_package(tmp_path, attn_bias):
    path = _export(tmp_path, attn_bias=attn_bias)
    assert ev.is_v6_export(path)
    model, device = ev.load_default_model(path, torch.device("cpu"))
    assert isinstance(model, ev.ExportedNet) and device.type == "cpu"
    assert model.seq_length == guofish_core.SEQ_LENGTH and model.value_scale == VALUE_SCALE
    assert ev.require_engine_contract(model) == guofish_core.SEQ_LENGTH

    reference, contract = load_for_inference(path)
    assert contract == "A"
    tokens = _tokens()
    with torch.no_grad():
        policy, value = model(tokens)
        ref_policy, ref_value = reference(tokens)
    assert policy.dtype == torch.bfloat16 and policy.shape == (len(tokens), guofish_core.POLICY_SIZE)
    assert value.dtype == torch.float32 and value.shape == (len(tokens),)
    assert torch.equal(policy.float(), ref_policy.to(torch.bfloat16).float())
    assert torch.equal(value, ref_value)
    # the narrowing is exact wherever the logits were bf16 to begin with (the engine's autocast)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        p_amp, _ = model(tokens)
        r_amp, _ = reference(tokens)
    assert torch.equal(p_amp.float(), r_amp)


@pytest.mark.parametrize("attn_bias", ["none", "static", "smolgen"])
def test_bf16_linear_storage_changes_no_output_bit_under_autocast(tmp_path, attn_bias):
    as_trained, _ = load_for_inference(_export(tmp_path, attn_bias=attn_bias))
    stored, _ = load_for_inference(_export(tmp_path, attn_bias=attn_bias))
    n = ev.cast_linears(stored, torch.bfloat16)
    assert n >= 2 * 4 + 3                                   # qkv/out/ff1/ff2 per block + heads
    assert stored.embedding.weight.dtype == stored.pos_embedding.dtype == torch.float32
    assert all(m.weight.dtype == torch.float32 for m in stored.modules()
               if isinstance(m, torch.nn.LayerNorm))
    tokens = _tokens()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        a = as_trained(tokens)
        b = stored(tokens)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_contract_b_exports_are_refused(tmp_path):
    path = _export(tmp_path, token_scheme="canonical_65")
    with pytest.raises(ValueError, match="contract-B"):
        ev.load_default_model(path, torch.device("cpu"))


@pytest.mark.parametrize("drop_key", [False, True])
def test_an_export_without_value_scale_is_refused(tmp_path, drop_key):
    path = _export(tmp_path, value_scale=None, drop_key=drop_key)
    with pytest.raises(ValueError, match="value_scale"):
        ev.load_default_model(path, torch.device("cpu"))


@pytest.mark.skipif(not (ev.DEFAULT_MODEL.exists() and ev.SHIPPING_MODEL.exists()),
                    reason="the v5 20M / 90M checkpoints are not on disk")
def test_v5_checkpoints_keep_the_loader_the_goldens_are_anchored_to():
    for path in (ev.DEFAULT_MODEL, ev.SHIPPING_MODEL):
        assert not ev.is_v6_export(path)
    model, _ = ev.load_default_model(ev.SHIPPING_MODEL, torch.device("cpu"))
    assert type(model).__name__ == "ChessTransformerV5"


@pytest.mark.skipif(not list((REPO_ROOT / "models").glob("guofish[24]/*.pt")),
                    reason="no legacy guofish2 / guofish4 checkpoint on disk")
def test_legacy_checkpoints_are_not_v6_exports():
    for path in sorted((REPO_ROOT / "models").glob("guofish[24]/*.pt"))[:3]:
        assert not ev.is_v6_export(path), path
