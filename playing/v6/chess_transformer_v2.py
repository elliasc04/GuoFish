"""The legacy ChessTransformerV2 architecture (guofish2 .. guofish4), and the
checkpoint-loading family that decides which generation a `.pt` file holds.

WHAT THIS FILE IS FOR
=====================
v6 is meant to stand on its own. Before this file existed, the only way to get a
model into `playing/v6/evaluator.py` was `playing.v5.playv5.load_model`, and
importing that module executes `import core.mctsv3` and `import core.mctsv4` at
its top (playv5.py:48-49) purely so that `build_mcts` can route between the two
Python MCTS implementations. v6 does not use either — it drives the C++ core —
so the whole legacy search tree was being dragged in as a side effect of asking
for a loader. `core.mctsv4` is on its way out; this file is what lets v6 stop
depending on it without also losing the ability to load a checkpoint.

WHY THE ARCHITECTURE IS MOVED HERE RATHER THAN COPIED HERE
==========================================================
There were already TWO copies of `ChessTransformerV2` in the tree, and they had
ALREADY DRIFTED. `playing/v4/playv4.py:84` hardcodes `dim_feedforward=d_model*4`
in the encoder layer; `playing/v5/playv5.py:79` takes it as a constructor
argument. Neither difference is visible at load time for a net whose FFN happens
to be 4x wide — `load_state_dict` checks shapes, and both spellings produce the
same shapes for the checkpoints that exist today — so the drift is silent until
somebody trains a net with a different FFN ratio and the older copy quietly
refuses it. A third copy inside `playing/v6/` would have been a third chance for
that. This file carries the SUPERSET (playv5's), and the intent is that
playv4/playv5 eventually import it rather than restate it.

The same argument is why `ChessTransformerV5` is NOT redefined here. It already
has exactly one definition, in `training/v5_multiPV/model_v5.py`, written from
the same `ModelConfig` that gets serialized into the checkpoint's `config` dict,
which is what makes width, depth, FFN activation and the final LayerNorm
impossible to drift between training and play. That module imports nothing but
`torch` and the standard library — no `core`, no `training.train` — so importing
it costs v6 nothing it was trying to avoid. This file imports it and dispatches
to it; it does not restate it.

THE TWO GENERATIONS, AND WHAT ACTUALLY SEPARATES THEM
=====================================================
Both emit the SAME board contract, and that is the point worth being explicit
about, because it is what makes a v6 engine able to run either one:

    68 tokens in  (64 squares + side-to-move + castling + en-passant + CLS@67)
    4096 policy logits out, indexed from_square * 64 + to_square
    1 value out, tanh-bounded, pooled from the CLS slot

That contract is hardcoded on the C++ side as `guofish::kSeqLength` (68) and
`guofish::kPolicySize` (4096) — see `cpp/tokens.hpp` and `cpp/evaluator.hpp`,
which re-implement the tokenizer and the policy gather in C++ — and both
generations satisfy it. What separates them is entirely INTERNAL: V5 uses a GELU
FFN and a final LayerNorm, V2 inherits `nn.TransformerEncoderLayer`'s RELU
default and `nn.TransformerEncoder`'s `norm=None`, which under `norm_first`
(Pre-LN) leaves the residual stream unnormalised at the output. Those are the
two things a state dict cannot tell you, which is why they are pinned in the
class rather than inferred, and why V5 and V2 cannot share one class.

So the legacy nets are NOT a different input/output format that the C++ core
would have to be taught. They are a different set of layers behind an identical
boundary.

STDOUT
======
`load_model` reports what it detected, and it reports it through the `log`
callable rather than through a bare `print`, because in v6 stdout IS the UCI
stream and a stray line on it is a protocol violation. The default is `print`,
so `playing/v5/playv5.py`'s behaviour is unchanged if it adopts this; the v6
caller is expected to pass `playv6.err` (or keep the existing
`contextlib.redirect_stdout` in `Engine.ensure_ready`, which also works).

WHAT IS DELIBERATELY NOT HERE
=============================
`build_mcts` / `is_v5_model`'s ROUTING ROLE. Deciding which Python MCTS drives a
model is the exact dependency this file exists to break, and v6 has no such
decision to make: there is one search, it is in C++, and it takes whichever
module this loader returns. `is_v5_model` is still here as a PREDICATE — callers
have a legitimate need to ask which generation they got, e.g. to decide whether
`value_scale` should have been present — but nothing here selects a search.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

# `training.v5_multiPV.model_v5` lives at the project root, which is not on the
# path when this file is reached as a script rather than as a package module.
# Same guard, for the same reason, as playing/v6/playv6.py's.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.v5_multiPV.model_v5 import (  # noqa: E402
    ChessTransformerV5, ModelConfig, config_from_checkpoint,
)

# The token count both generations emit, restated here as the number this file
# REFUSES anything else at (see build_model_for_checkpoint). It is deliberately a
# literal rather than an import from `core.mctsv4.SEQ_LENGTH`: that module is
# being retired, and `guofish_core.SEQ_LENGTH` — the C++ constant, which is the
# one that actually governs at run time — is not importable here without making
# the compiled extension a hard dependency of a file that only defines layers.
# The v6 evaluator is the right place to assert the two agree, and it can, since
# every module this returns advertises `seq_length`.
SEQ_LENGTH = 68

# The CLS slot the value head pools from, and the policy width. Same reasoning.
CLS_INDEX = 67
POLICY_SIZE = 4096


# ---------------------------------------------------------------------------
# The architecture
# ---------------------------------------------------------------------------


class ChessTransformerV2(nn.Module):
    """Legacy architecture: 68 tokens (64 squares + side + castling + ep + CLS), CLS pooling.

    Everything through guofish4 is this. Two of its properties are inherited
    defaults rather than choices, and both are unrecoverable from a state dict,
    so they are pinned here: nn.TransformerEncoderLayer defaults to a RELU FFN,
    and nn.TransformerEncoder defaults to norm=None, which under norm_first
    (Pre-LN) leaves the residual stream unnormalised at the output. v5 fixes
    both -- see ChessTransformerV5 -- which is why the two cannot share a class.
    """

    def __init__(self, vocab_size=43, d_model=512, nhead=8, num_layers=8, dropout=0.1,
                 head_dim=64, dim_feedforward=None):
        super().__init__()
        self.seq_length = 68
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.pos_encoder = nn.Parameter(torch.randn(1, 68, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward if dim_feedforward else d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        # `enable_nested_tensor=False` is the ONE deviation from the copy in
        # playing/v5/playv5.py, and it is a no-op that removes a warning rather
        # than a behaviour change. torch forces `use_nested_tensor` to False
        # whenever `norm_first=True` -- which it is, here and in V5 -- and warns
        # on every construction that it has done so. V5 passes False explicitly
        # for the same reason (model_v5.py). Saying it here keeps a UserWarning
        # off the engine's stderr at every startup, where it reads like a fault.
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                 enable_nested_tensor=False)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1), nn.Tanh(),
        )
        self.head_dim = head_dim
        self.from_proj = nn.Linear(d_model, head_dim)
        self.to_proj = nn.Linear(d_model, head_dim)
        self.logit_scale = 1.0 / math.sqrt(head_dim)

    def forward(self, x, legal_move_mask=None):
        x = self.embedding(x) + self.pos_encoder
        x = self.emb_dropout(x)
        x = self.transformer(x)
        cls_state = x[:, 67, :]  # CLS token at position 67
        value = self.value_head(cls_state).squeeze(-1)
        x_squares = x[:, :64, :]
        from_feats = self.from_proj(x_squares)
        to_feats = self.to_proj(x_squares)
        policy_logits = torch.bmm(from_feats, to_feats.transpose(1, 2)) * self.logit_scale
        policy_logits = policy_logits.view(x.size(0), 4096)
        if legal_move_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_move_mask, float('-inf'))
        return policy_logits, value


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------

# Attributes load_model attaches to the returned module and must survive the
# CPU quantization rebuild below. `seq_length` is the one a caller reads to
# validate the tokenization; the other two are metadata for callers.
_CARRIED_ATTRS = ("seq_length", "config", "value_scale")


def inference_dtype(device: torch.device) -> torch.dtype:
    """The dtype weights should live in for this device.

    BF16 on CUDA, because that is what the engine actually computes in: every
    forward pass runs under bf16 autocast (playing/v6/evaluator.py and
    playing/v6/graphs.py both pin `AUTOCAST_DTYPE`), and v5 was trained in bf16
    too (training/v5_multiPV/configs/base.yaml pins `amp: bf16  # matches
    inference precision`). Storing bf16 makes training, storage and inference one
    format end to end.

    The previous .half() was a pure round trip -- fp32 checkpoint -> fp16
    (10-bit mantissa) -> bf16 (7-bit) at the first autocast matmul -- so it
    rounded twice and landed in bf16 anyway. It costs nothing on a 25.6M net
    but the 10.9M student has less redundancy to absorb it.

    Pre-Ampere cards have no bf16, so they keep fp16. CPU stays fp32 and is
    then dynamically quantized to INT8 by load_model.
    """
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _is_v5_model_config(cfg: object) -> bool:
    """True if a checkpoint's `config` entry is a v5 ModelConfig.

    The key alone is not enough: guofish3/guofish4 also write `config`, but it
    holds the TRAINING hyperparameters (BATCH_SIZE, EPOCHS, ...). ModelConfig
    .from_dict silently drops unknown keys, so handing it one of those would
    quietly yield a default-shaped v5 model instead of failing. Require the two
    architectural fields only a real ModelConfig carries.
    """
    return isinstance(cfg, dict) and "d_model" in cfg and "seq_len" in cfg


def _v2_kwargs_from_state_dict(state_dict: dict) -> dict:
    """Infer ChessTransformerV2's constructor args from a legacy state dict.

    Everything but `nhead` is recoverable. Attention packs Q/K/V into a single
    (3*d_model, d_model) in_proj_weight regardless of the head count, so the
    split leaves no trace in the weights; fall back to the 64-dim-per-head
    convention every GuoFish generation has used (d_model=512 -> 8 heads).
    """
    d_model = state_dict["pos_encoder"].shape[2]
    return dict(
        vocab_size=state_dict["embedding.weight"].shape[0],
        d_model=d_model,
        nhead=max(1, d_model // 64),
        num_layers=1 + max(int(k.split(".")[2]) for k in state_dict
                           if k.startswith("transformer.layers.")),
        dim_feedforward=state_dict["transformer.layers.0.linear1.weight"].shape[0],
        head_dim=state_dict["from_proj.weight"].shape[0],
    )


def build_model_for_checkpoint(ckpt: dict, state_dict: dict) -> tuple[nn.Module, str]:
    """Construct the architecture `ckpt` was trained with. Weights NOT loaded yet.

    Dispatch, in order:
      1. a real ModelConfig in ckpt["config"] -> ChessTransformerV5 rebuilt
         exactly as trained. The only path that recovers the parameterless
         choices (FFN activation, head count), so it is the reliable one.
      2. `final_norm.*` keys in the state dict -> ChessTransformerV5 with the
         shape read off the weights, for a v5 checkpoint written before
         `config` existed. Activation falls back to the v5 default (GELU).
      3. otherwise -> legacy ChessTransformerV2, shape from the weights.

    Returns (model, description) so the caller can report what it picked.
    """
    cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    from_config = _is_v5_model_config(cfg)

    if from_config or any(k.startswith("final_norm.") for k in state_dict):
        if from_config:
            config = ModelConfig.from_dict(cfg)  # type: ignore[arg-type]
        else:
            # Hand config_from_checkpoint a bare state dict so it takes its
            # shape-inference path instead of reading a `config` key back out.
            config = config_from_checkpoint({"model_state_dict": state_dict})
        return ChessTransformerV5(config), (
            f"V5 architecture ({'from config' if from_config else 'inferred from weights'}): "
            f"d_model={config.d_model} x{config.num_layers} layers, "
            f"{config.nhead} heads, ffn={config.dim_feedforward}, "
            f"{config.activation.upper()} FFN, final_norm={config.final_norm}")

    seq_length = state_dict["pos_encoder"].shape[1]
    if seq_length != SEQ_LENGTH:
        # Notably the V1 65-token scheme. The tokenizer emits 68 tokens and the
        # C++ core's `kSeqLength` is 68, so such a model would be fed a
        # mis-shaped board and silently evaluate garbage. Refuse it here rather
        # than at the first forward pass.
        raise ValueError(f"Unknown architecture: pos_encoder has {seq_length} "
                         f"positions; only the {SEQ_LENGTH}-token scheme is supported")

    kwargs = _v2_kwargs_from_state_dict(state_dict)
    return ChessTransformerV2(**kwargs), (
        f"V2 architecture: d_model={kwargs['d_model']} x{kwargs['num_layers']} layers, "
        f"{kwargs['nhead']} heads, ffn={kwargs['dim_feedforward']}, "
        "RELU FFN, no final_norm")


def describe_checkpoint(ckpt: dict) -> list[str]:
    """Training provenance worth echoing at load time. Empty for a bare state dict."""
    if not isinstance(ckpt, dict):
        return []
    lines = []
    if "val_acc" in ckpt:                       # v2-era policy accuracy
        lines.append(f"Model accuracy: {ckpt['val_acc']:.1f}%")
    if "best_val" in ckpt:                      # v5 composite val loss
        lines.append(f"Checkpoint: epoch {ckpt.get('epoch')} step {ckpt.get('step')} "
                     f"({ckpt.get('reason', '?')}), best_val {ckpt['best_val']:.4f}")
    if "value_scale" in ckpt:
        # The value head was trained on tanh(cp / value_scale), so this is the
        # constant that inverts its output back to centipawns.
        lines.append(f"Value scale: {ckpt['value_scale']:.4f} cp "
                     "(target was tanh(cp / scale))")
    return lines


def load_model(checkpoint_path: Path, device: torch.device,
               log: Callable[[str], None] = print) -> nn.Module:
    """Load a checkpoint, selecting the architecture from its own metadata.

    Handles both generations -- v5 (training/v5_multiPV) and legacy v2
    (guofish2..guofish4) -- with no flag to get wrong; see
    build_model_for_checkpoint for the dispatch. The returned module always
    advertises `seq_length`, plus `config`/`value_scale` when the checkpoint
    carried them.

    NOTE ON `value_scale`: it is attached only when the checkpoint has it, and
    the legacy v2 checkpoints predate it. A caller that inverts the value head
    to centipawns therefore cannot assume it is present -- see
    playing/v6/playv6.py, which currently REQUIRES it. That is a decision for the
    caller, not a default this loader should invent, so nothing is fabricated
    here.

    `log` is where the detection report goes. It defaults to `print`, which is
    the historical behaviour; a UCI caller must pass something that writes to
    stderr, because stdout is the protocol stream.
    """
    log(f"Loading {checkpoint_path} on {device}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract state dict. A bare state dict is also accepted (fp16 exports).
    state_dict = (ckpt["model_state_dict"]
                  if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)

    model, description = build_model_for_checkpoint(ckpt, state_dict)
    log(f"Detected {description}")
    for line in describe_checkpoint(ckpt):
        log(line)

    dtype = inference_dtype(device)
    log(f"Precision: {str(dtype).removeprefix('torch.')}")
    model = model.to(device=device, dtype=dtype)
    state_dict = {k: (v.to(dtype) if v.is_floating_point() else v)
                  for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    if isinstance(ckpt, dict) and "value_scale" in ckpt:
        setattr(model, "value_scale", float(ckpt["value_scale"]))

    if device.type == "cpu":
        # Dynamic INT8 quantization on all Linear layers (FFN, attention out_proj,
        # value/policy heads). Weights are stored as INT8 with per-batch activation
        # quantization. Typically 2-4x faster on CPU with negligible accuracy loss.
        carried = {name: getattr(model, name) for name in _CARRIED_ATTRS
                   if hasattr(model, name)}
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        for name, value in carried.items():
            setattr(model, name, value)

        # After quantization, Linear.weight becomes a method (returns dequantized weight
        # on demand) instead of a tensor attribute. nn.TransformerEncoderLayer's fast-path
        # eligibility check iterates `tensor_args` and reads `.device.type`, which crashes
        # on the methodified weights. Trip an earlier short-circuit in the fast-path check
        # so the device probe never runs -- the slow path uses self.activation (still the
        # real activation), so this only affects the fast-path decision, not the math.
        for layer in model.transformer.layers:
            layer.activation_relu_or_gelu = False

    return model


def is_v5_model(model: nn.Module) -> bool:
    """True if `model` is the v5 student rather than a legacy v2 net.

    Tested on the ModelConfig the module carries rather than isinstance, so a
    torch.ao dynamically-quantized copy (the CPU path in load_model, which
    returns a rebuilt module) still answers True. Legacy v2 nets have no
    `config` at all.

    A PREDICATE, not a router. v6 has one search and it takes either generation;
    what a caller legitimately wants this for is deciding whether a missing
    `value_scale` is expected (it is, on v2) or a broken checkpoint (it would be,
    on v5 -- train_v5.save_checkpoint writes one unconditionally).
    """
    return hasattr(getattr(model, "config", None), "seq_len")


__all__ = [
    "CLS_INDEX",
    "POLICY_SIZE",
    "SEQ_LENGTH",
    "ChessTransformerV2",
    "ChessTransformerV5",
    "ModelConfig",
    "build_model_for_checkpoint",
    "config_from_checkpoint",
    "describe_checkpoint",
    "inference_dtype",
    "is_v5_model",
    "load_model",
]
