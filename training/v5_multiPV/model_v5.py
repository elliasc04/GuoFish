"""GuoFish v5 - the ~10.9M multi-PV student architecture.

Deliberately NOT a subclass of training.train.ChessTransformer. Two things in
that class are wrong for this run and one of them is invisible:

  final LayerNorm   `nn.TransformerEncoder(layer, n)` defaults to `norm=None`.
                    With norm_first=True (Pre-LN) that leaves the residual
                    stream unnormalised at the output, so both heads read a
                    stream whose scale grows with depth. Added here as
                    `final_norm`, controllable via ModelConfig.final_norm.
  FFN activation    `nn.TransformerEncoderLayer` defaults to RELU, and train.py
                    never overrides it - so every previous GuoFish generation
                    has a ReLU FFN despite a GELU value head. The v5 brief
                    specifies GELU; this file passes it explicitly.

Both are reported in the run report rather than silently absorbed.

Everything else is byte-compatible with the engine's ChessTransformerV2
(playing/v5/playv5.py) on purpose - same parameter NAMES, same forward math,
same masking semantics - so an MCTS checkpoint loader only has to learn the two
new things above plus the width/depth, both of which travel in the checkpoint's
`config` dict.

  Policy  factored from/to projections to head_dim, outer product over the 64
          square tokens -> 4096 logits, scaled 1/sqrt(head_dim).
  Value   2-layer MLP (d_model -> d_model -> 1), GELU, dropout, tanh, reading
          the CLS token at position 67. White-POV.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Every architectural knob. Serialised into every checkpoint."""

    vocab_size: int = 43
    seq_len: int = 68
    d_model: int = 384
    nhead: int = 6
    num_layers: int = 6
    dim_feedforward: int = 1536
    head_dim: int = 64
    dropout: float = 0.1
    activation: str = "gelu"
    norm_first: bool = True         # Pre-LN
    final_norm: bool = True         # see module docstring
    policy_size: int = 4096
    cls_index: int = 67
    smolgen: bool = False           # placeholder; separate flagged A/B later

    def __post_init__(self):
        if self.d_model % self.nhead:
            raise ValueError(
                f"d_model={self.d_model} not divisible by nhead={self.nhead}")
        if self.cls_index >= self.seq_len:
            raise ValueError(
                f"cls_index={self.cls_index} outside seq_len={self.seq_len}")
        if self.policy_size != 4096:
            raise ValueError("policy head is a 64x64 outer product; "
                             "policy_size must be 4096")
        if self.smolgen:
            raise NotImplementedError(
                "Smolgen is a separate flagged A/B and is not implemented here")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class ChessTransformerV5(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None, **overrides):
        super().__init__()
        if config is None:
            config = ModelConfig(**overrides)
        elif overrides:
            merged = config.to_dict()
            merged.update(overrides)
            config = ModelConfig.from_dict(merged)
        self.config = config

        # `seq_length` (not seq_len): the engine reads this attribute off the
        # loaded module, so the spelling is load-bearing.
        self.seq_length = config.seq_len
        self.head_dim = config.head_dim
        self.cls_index = config.cls_index
        self.policy_size = config.policy_size

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.pos_encoder = nn.Parameter(
            torch.randn(1, config.seq_len, config.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=config.norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers,
            enable_nested_tensor=False)

        # Pre-LN without this leaves the output stream unnormalised.
        self.final_norm = (nn.LayerNorm(config.d_model)
                           if config.final_norm else nn.Identity())

        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
            nn.Tanh(),
        )

        self.from_proj = nn.Linear(config.d_model, config.head_dim)
        self.to_proj = nn.Linear(config.d_model, config.head_dim)
        self.logit_scale = 1.0 / math.sqrt(config.head_dim)

    # -- introspection -----------------------------------------------------
    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)

    def param_breakdown(self) -> dict:
        groups = {"embedding": 0, "pos_encoder": 0, "encoder": 0,
                  "final_norm": 0, "value_head": 0, "policy_head": 0}
        for name, p in self.named_parameters():
            if name.startswith("embedding"):
                groups["embedding"] += p.numel()
            elif name.startswith("pos_encoder"):
                groups["pos_encoder"] += p.numel()
            elif name.startswith("transformer"):
                groups["encoder"] += p.numel()
            elif name.startswith("final_norm"):
                groups["final_norm"] += p.numel()
            elif name.startswith("value_head"):
                groups["value_head"] += p.numel()
            else:
                groups["policy_head"] += p.numel()
        groups["total"] = sum(v for k, v in groups.items() if k != "total")
        return groups

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor,
                legal_move_mask: Optional[torch.Tensor] = None):
        """x: (B, seq_len) int64 tokens.

        `legal_move_mask` is the engine's inference-time contract: a BOOL
        (B, 4096) where True == legal, applied to the logits with -inf. Training
        does its own masking inside the loss (which must match this exactly),
        so train_v5.py calls forward WITHOUT a mask and masks in losses.py -
        one implementation, verified equal by tests/test_losses.py.
        """
        h = self.embedding(x) + self.pos_encoder
        h = self.emb_dropout(h)
        h = self.transformer(h)
        h = self.final_norm(h)

        cls_state = h[:, self.cls_index, :]
        value = self.value_head(cls_state).squeeze(-1)

        squares = h[:, :64, :]
        from_feats = self.from_proj(squares)
        to_feats = self.to_proj(squares)
        policy_logits = torch.bmm(
            from_feats, to_feats.transpose(1, 2)) * self.logit_scale
        policy_logits = policy_logits.reshape(x.size(0), self.policy_size)

        if legal_move_mask is not None:
            if legal_move_mask.dtype != torch.bool:
                legal_move_mask = legal_move_mask > 0
            policy_logits = policy_logits.masked_fill(
                ~legal_move_mask, float("-inf"))
        return policy_logits, value


def build_model(config: Optional[ModelConfig] = None, **overrides
                ) -> ChessTransformerV5:
    return ChessTransformerV5(config, **overrides)


def config_from_checkpoint(ckpt: dict) -> ModelConfig:
    """Reconstruct the architecture from a checkpoint written by train_v5.py.

    Falls back to inferring the shape from the state dict for checkpoints that
    predate the `config` key, so the engine has one code path.
    """
    if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
        return ModelConfig.from_dict(ckpt["config"])

    sd = ckpt.get("model_state_dict", ckpt)
    d_model = sd["pos_encoder"].shape[2]
    seq_len = sd["pos_encoder"].shape[1]
    vocab = sd["embedding.weight"].shape[0]
    layers = 1 + max(int(k.split(".")[2]) for k in sd
                     if k.startswith("transformer.layers."))
    ff = sd["transformer.layers.0.linear1.weight"].shape[0]
    head_dim = sd["from_proj.weight"].shape[0]
    return ModelConfig(
        vocab_size=vocab, seq_len=seq_len, d_model=d_model,
        num_layers=layers, dim_feedforward=ff, head_dim=head_dim,
        nhead=max(1, d_model // 64),
        final_norm=any(k.startswith("final_norm.") for k in sd),
    )


def load_from_checkpoint(ckpt: dict, strict: bool = True) -> ChessTransformerV5:
    model = build_model(config_from_checkpoint(ckpt))
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=strict)
    return model


if __name__ == "__main__":
    m = build_model()
    print(m.config)
    for k, v in m.param_breakdown().items():
        print(f"  {k:<14} {v:>12,}")
    x = torch.randint(0, 43, (4, m.seq_length))
    pl, v = m(x)
    print(f"policy {tuple(pl.shape)}  value {tuple(v.shape)}  "
          f"|v|max {v.abs().max().item():.4f}")
