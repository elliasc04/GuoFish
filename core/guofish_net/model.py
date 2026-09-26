"""The v6 network: one definition shared by the trainer and the engine (§3, §5).

A hand-written Pre-LN block so attention can take a per-sample additive bias.
With `attn_bias: none`, `init: v5_deepcopy` and `dropout: 0.1` the math is
v5's `nn.TransformerEncoderLayer(norm_first=True, activation=gelu)` exactly:
the fused QKV uses the same packed [q; k; v] layout as `in_proj_weight`, so a
v5 checkpoint converts by key map alone (v5_compat.py).

A new variant is a new config value plus a branch here, never a new file.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.guofish_net.strict import from_dict_strict

ARCH_VERSION = 1
POLICY_SIZE = 4096
HLGAUSS_BINS = 101
HLGAUSS_SIGMA_FRAC = 0.75       # sigma = 0.75 x bin width

TOKEN_SCHEMES = {
    # name: (seq_len, vocab, cls_index, contract)
    "v5_68": (68, 43, 67, "A"),
    "canonical_65": (65, 17, 64, "B"),
}


def _enum(name: str, value, allowed) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r}; expected one of {sorted(allowed)}")


def _positive(name: str, value) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name}={value!r}; expected a positive int")


@dataclass(frozen=True)
class SmolgenConfig:
    compress: int = 16
    hidden: int = 128
    gen: int = 128

    def __post_init__(self):
        for f in fields(self):
            _positive(f"smolgen.{f.name}", getattr(self, f.name))


@dataclass(frozen=True)
class ValueHeadConfig:
    pool: str = "cls_mlp"

    def __post_init__(self):
        _enum("value_head.pool", self.pool, {"cls_mlp", "flatten"})


@dataclass(frozen=True)
class ValueReprConfig:
    kind: str = "scalar"

    def __post_init__(self):
        _enum("value_repr.kind", self.kind, {"scalar", "hlgauss"})


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    head_dim: int = 64                  # policy from/to projection width
    dropout: float = 0.0
    final_norm: bool = True
    init: str = "independent"           # independent | v5_deepcopy
    token_scheme: str = "v5_68"         # v5_68 | canonical_65
    attn_bias: str = "none"             # none | static | smolgen
    smolgen: SmolgenConfig = field(default_factory=SmolgenConfig)
    value_head: ValueHeadConfig = field(default_factory=ValueHeadConfig)
    value_repr: ValueReprConfig = field(default_factory=ValueReprConfig)
    aux_policy_head: bool = False       # second policy head the engine never reads

    def __post_init__(self):
        for name in ("d_model", "n_layers", "n_heads", "d_ff", "head_dim"):
            _positive(name, getattr(self, name))
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout={self.dropout}; expected [0, 1)")
        _enum("init", self.init, {"independent", "v5_deepcopy"})
        _enum("token_scheme", self.token_scheme, set(TOKEN_SCHEMES))
        _enum("attn_bias", self.attn_bias, {"none", "static", "smolgen"})
        for name in ("final_norm", "aux_policy_head"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")

    @property
    def seq_len(self) -> int:
        return TOKEN_SCHEMES[self.token_scheme][0]

    @property
    def vocab_size(self) -> int:
        return TOKEN_SCHEMES[self.token_scheme][1]

    @property
    def cls_index(self) -> int:
        return TOKEN_SCHEMES[self.token_scheme][2]

    @property
    def contract(self) -> str:
        return TOKEN_SCHEMES[self.token_scheme][3]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return from_dict_strict(cls, d, "model")


# --- value representation ------------------------------------------------

def hlgauss_edges(device=None) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, HLGAUSS_BINS + 1, dtype=torch.float32, device=device)


def hlgauss_centers(device=None) -> torch.Tensor:
    e = hlgauss_edges(device)
    return 0.5 * (e[1:] + e[:-1])


def hlgauss_target(y: torch.Tensor) -> torch.Tensor:
    """(B,) labels in [-1, 1] -> (B, 101) bin probabilities.

    A Gaussian centred on the label, sigma = 0.75 x bin width, integrated per
    bin and renormalised over [-1, 1]."""
    e = hlgauss_edges(y.device)
    sigma = HLGAUSS_SIGMA_FRAC * float(e[1] - e[0])
    cdf = torch.special.ndtr((e.unsqueeze(0) - y.float().unsqueeze(1)) / sigma)
    mass = cdf[:, 1:] - cdf[:, :-1]
    return mass / (cdf[:, -1:] - cdf[:, :1])


# --- modules -------------------------------------------------------------

class Smolgen(nn.Module):
    """Per-layer half of smolgen; the 128 -> 64x64 projection is shared."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        s = cfg.smolgen
        self.n_heads, self.gen = cfg.n_heads, s.gen
        self.compress = nn.Linear(cfg.d_model, s.compress, bias=False)
        self.fc1 = nn.Linear(64 * s.compress, s.hidden)
        self.ln1 = nn.LayerNorm(s.hidden)
        self.fc2 = nn.Linear(s.hidden, cfg.n_heads * s.gen)
        self.ln2 = nn.LayerNorm(cfg.n_heads * s.gen)

    def forward(self, h: torch.Tensor, shared: nn.Linear) -> torch.Tensor:
        B, T = h.shape[0], h.shape[1]
        z = self.compress(h[:, :64]).reshape(B, -1)
        z = self.ln1(F.silu(self.fc1(z)))
        z = self.ln2(F.silu(self.fc2(z))).view(B, self.n_heads, self.gen)
        bias = shared(z).view(B, self.n_heads, 64, 64)
        return F.pad(bias, (0, T - 64, 0, T - 64))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads, self.dropout = cfg.n_heads, cfg.dropout
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, cfg.d_ff)
        self.ff2 = nn.Linear(cfg.d_ff, d)
        self.smolgen = Smolgen(cfg) if cfg.attn_bias == "smolgen" else None

    def forward(self, x, bias=None, shared=None):
        B, T, d = x.shape
        p = self.dropout if self.training else 0.0
        h = self.ln1(x)
        if self.smolgen is not None:
            bias = self.smolgen(h, shared)
        q, k, v = self.qkv(h).view(B, T, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=p)
        a = a.transpose(1, 2).reshape(B, T, d)
        x = x + F.dropout(self.out(a), p, self.training)
        h = F.dropout(F.gelu(self.ff1(self.ln2(x))), p, self.training)
        return x + F.dropout(self.ff2(h), p, self.training)


class GuoFishNet(nn.Module):
    def __init__(self, cfg: ModelConfig, init_seed: int = 0):
        super().__init__()
        self.cfg = cfg
        d, T = cfg.d_model, cfg.seq_len
        self.seq_length = T             # attribute name the v5 engine reads
        self.embedding = nn.Embedding(cfg.vocab_size, d)
        self.pos_embedding = nn.Parameter(torch.zeros(1, T, d))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(d) if cfg.final_norm else nn.Identity()

        self.static_bias = (nn.Parameter(torch.zeros(cfg.n_layers, cfg.n_heads, T, T))
                            if cfg.attn_bias == "static" else None)
        self.smolgen_shared = (nn.Linear(cfg.smolgen.gen, 64 * 64, bias=False)
                               if cfg.attn_bias == "smolgen" else None)

        self.from_proj = nn.Linear(d, cfg.head_dim)
        self.to_proj = nn.Linear(d, cfg.head_dim)
        if cfg.aux_policy_head:
            self.aux_from_proj = nn.Linear(d, cfg.head_dim)
            self.aux_to_proj = nn.Linear(d, cfg.head_dim)
        self.logit_scale = 1.0 / math.sqrt(cfg.head_dim)

        v_out = 1 if cfg.value_repr.kind == "scalar" else HLGAUSS_BINS
        if cfg.value_head.pool == "cls_mlp":
            self.value_fc1 = nn.Linear(d, d)
            self.value_fc2 = nn.Linear(d, v_out)
        else:
            self.value_sq = nn.Linear(d, 32)
            self.value_fc1 = nn.Linear(64 * 32, 128)
            self.value_fc2 = nn.Linear(128, v_out)
        self.register_buffer("hl_centers", hlgauss_centers(), persistent=False)
        _init_weights(self, cfg, init_seed)

    def _policy(self, sq, from_proj, to_proj):
        logits = torch.bmm(from_proj(sq), to_proj(sq).transpose(1, 2)) * self.logit_scale
        return logits.reshape(sq.shape[0], POLICY_SIZE)

    def forward_train(self, tokens: torch.Tensor) -> dict:
        cfg, p = self.cfg, (self.cfg.dropout if self.training else 0.0)
        x = F.dropout(self.embedding(tokens.long()) + self.pos_embedding, p, self.training)
        for i, blk in enumerate(self.blocks):
            # SDPA picks its CPU kernel from the mask's shape and requires_grad
            # (even under no_grad): a 3-D mask or a grad-requiring one leaves the
            # fused kernel, so a zero static bias would stop reproducing the
            # plain model at inference (S8). 4-D, and detached when grad is off.
            bias = None
            if self.static_bias is not None:
                bias = self.static_bias[i:i + 1]
                if not torch.is_grad_enabled():
                    bias = bias.detach()
            x = blk(x, bias, self.smolgen_shared)
        x = self.final_norm(x)
        sq = x[:, :64]
        out = {"policy_logits": self._policy(sq, self.from_proj, self.to_proj).float()}
        if cfg.aux_policy_head:
            out["aux_policy_logits"] = self._policy(sq, self.aux_from_proj, self.aux_to_proj).float()

        if cfg.value_head.pool == "cls_mlp":
            h = x[:, cfg.cls_index]
        else:
            h = self.value_sq(sq).flatten(1)
        v = self.value_fc2(F.dropout(F.gelu(self.value_fc1(h)), p, self.training)).float()
        if cfg.value_repr.kind == "scalar":
            out["value"] = torch.tanh(v.squeeze(-1))
        else:
            probs = torch.softmax(v, dim=-1)
            mean = probs @ self.hl_centers
            out["value_logits"] = v
            out["value"] = mean
            out["value_spread"] = torch.sqrt(
                (probs * (self.hl_centers - mean.unsqueeze(1)).pow(2)).sum(-1).clamp_min(0))
        return out

    def forward(self, tokens: torch.Tensor):
        """The engine's only call: (policy_logits[B, 4096], value[B]), fp32."""
        out = self.forward_train(tokens)
        return out["policy_logits"], out["value"]


def _keyed_normal_(p: torch.Tensor, init_seed: int, name: str, std: float = 0.02) -> None:
    """N(0, std) from a generator keyed on (init_seed, parameter name) alone."""
    key = int.from_bytes(hashlib.sha256(f"{init_seed}:{name}".encode()).digest()[:8], "little")
    g = torch.Generator().manual_seed(key >> 1)
    with torch.no_grad():
        p.copy_(torch.randn(p.shape, generator=g) * std)


def _init_weights(model: GuoFishNet, cfg: ModelConfig, init_seed: int) -> None:
    if model.static_bias is not None:
        nn.init.zeros_(model.static_bias)
    if model.smolgen_shared is not None:
        nn.init.zeros_(model.smolgen_shared.weight)      # starts equal to plain

    if cfg.init == "v5_deepcopy":
        # PyTorch defaults (what v5 got) plus MultiheadAttention's own reset,
        # with every block a copy of block 0 (H20, kept on purpose here).
        nn.init.normal_(model.pos_embedding, std=0.02)
        b0 = model.blocks[0]
        nn.init.xavier_uniform_(b0.qkv.weight)
        nn.init.zeros_(b0.qkv.bias)
        nn.init.zeros_(b0.out.bias)
        for blk in model.blocks[1:]:
            blk.load_state_dict(b0.state_dict())
        return

    # independent: every tensor drawn separately, each from its own generator
    # keyed on (init_seed, parameter name), so adding a module never changes
    # another parameter's starting weights (paired screening arms).
    for name, mod in model.named_modules():
        if name == "smolgen_shared":
            continue
        if isinstance(mod, (nn.Linear, nn.Embedding)):
            _keyed_normal_(mod.weight, init_seed, f"{name}.weight")
            if getattr(mod, "bias", None) is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.LayerNorm):
            nn.init.ones_(mod.weight)
            nn.init.zeros_(mod.bias)
    _keyed_normal_(model.pos_embedding, init_seed, "pos_embedding")
    scale = 1.0 / math.sqrt(2 * cfg.n_layers)
    with torch.no_grad():
        for blk in model.blocks:
            blk.out.weight.mul_(scale)
            blk.ff2.weight.mul_(scale)


def build_model(cfg: ModelConfig, init_seed: int = 0) -> GuoFishNet:
    """The only constructor. `init: independent` is keyed on (init_seed, name)
    and leaves the global RNG where it was; `v5_deepcopy` draws from the global
    RNG, as v5 did, and ignores init_seed."""
    if not isinstance(cfg, ModelConfig):
        raise TypeError(f"build_model takes a ModelConfig, got {type(cfg).__name__}")
    if cfg.init == "v5_deepcopy":
        return GuoFishNet(cfg)
    with torch.random.fork_rng(devices=[]):     # module constructors draw default inits
        return GuoFishNet(cfg, init_seed)


def load_for_inference(path, map_location="cpu"):
    """Export file -> (eval-mode module, contract 'A' | 'B')."""
    blob = torch.load(path, map_location=map_location, weights_only=True)
    for key in ("arch_version", "model_config", "state_dict", "contract", "token_scheme"):
        if key not in blob:
            raise KeyError(f"{path}: not a v6 export file (missing {key!r})")
    if blob["arch_version"] != ARCH_VERSION:
        raise ValueError(f"{path}: arch_version {blob['arch_version']} != {ARCH_VERSION}")
    cfg = ModelConfig.from_dict(blob["model_config"])
    if cfg.token_scheme != blob["token_scheme"] or cfg.contract != blob["contract"]:
        raise ValueError(f"{path}: contract/token_scheme disagree with model_config")
    model = build_model(cfg)
    model.load_state_dict(blob["state_dict"], strict=True)
    return model.eval(), cfg.contract
