"""Load a v5 (training/v5_multiPV) checkpoint into the v6 module.

v5's nn.TransformerEncoderLayer keeps q/k/v packed in `in_proj_weight` in the
same [q; k; v] row order as Block.qkv, so conversion is a key map. Every v5 key
must map and every v6 key must be filled; anything else is an error.
"""
from __future__ import annotations

import torch

from core.guofish_net.model import GuoFishNet, ModelConfig, build_model

_LAYER = {
    "self_attn.in_proj_weight": "qkv.weight",
    "self_attn.in_proj_bias": "qkv.bias",
    "self_attn.out_proj.weight": "out.weight",
    "self_attn.out_proj.bias": "out.bias",
    "linear1.weight": "ff1.weight",
    "linear1.bias": "ff1.bias",
    "linear2.weight": "ff2.weight",
    "linear2.bias": "ff2.bias",
    "norm1.weight": "ln1.weight",
    "norm1.bias": "ln1.bias",
    "norm2.weight": "ln2.weight",
    "norm2.bias": "ln2.bias",
}
_TOP = {
    "pos_encoder": "pos_embedding",
    "embedding.weight": "embedding.weight",
    "final_norm.weight": "final_norm.weight",
    "final_norm.bias": "final_norm.bias",
    "value_head.0.weight": "value_fc1.weight",
    "value_head.0.bias": "value_fc1.bias",
    "value_head.3.weight": "value_fc2.weight",
    "value_head.3.bias": "value_fc2.bias",
    "from_proj.weight": "from_proj.weight",
    "from_proj.bias": "from_proj.bias",
    "to_proj.weight": "to_proj.weight",
    "to_proj.bias": "to_proj.bias",
}
_FIXED = {"vocab_size": 43, "seq_len": 68, "cls_index": 67, "policy_size": 4096,
          "activation": "gelu", "norm_first": True, "smolgen": False}


def v5_model_config(v5: dict, init: str = "v5_deepcopy") -> ModelConfig:
    for k, want in _FIXED.items():
        if v5.get(k) != want:
            raise ValueError(f"v5 config {k}={v5.get(k)!r}; converter supports only {want!r}")
    return ModelConfig(d_model=v5["d_model"], n_layers=v5["num_layers"], n_heads=v5["nhead"],
                       d_ff=v5["dim_feedforward"], head_dim=v5["head_dim"],
                       dropout=float(v5["dropout"]), final_norm=bool(v5["final_norm"]),
                       init=init, token_scheme="v5_68")


def convert_v5_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k in _TOP:
            out[_TOP[k]] = v
            continue
        parts = k.split(".", 3)
        if len(parts) == 4 and parts[:2] == ["transformer", "layers"] and parts[3] in _LAYER:
            out[f"blocks.{int(parts[2])}.{_LAYER[parts[3]]}"] = v
            continue
        raise KeyError(f"unmapped v5 key {k!r}")
    return out


def load_v5_checkpoint(path, map_location="cpu") -> GuoFishNet:
    ck = torch.load(path, map_location=map_location, weights_only=True)
    model = build_model(v5_model_config(ck["config"]))
    model.load_state_dict(convert_v5_state_dict(ck["model_state_dict"]), strict=True)
    return model
