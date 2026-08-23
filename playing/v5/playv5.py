"""Play chess against a ChessTransformer model.

Loads either engine-compatible generation, picking the architecture from the
checkpoint's own metadata rather than from a flag (see load_model):

    v5   training/v5_multiPV -- ~10.9M student, every architectural knob
         carried in the checkpoint's `config` dict (GELU FFN, final LayerNorm).
    v2   guofish2 .. guofish4 -- predates `config`; shape is read off the
         weights and the parameterless choices are the legacy convention
         (ReLU FFN, no final LayerNorm).

Both consume the same 68-token board encoding, but they do not share a search:
build_mcts routes v5 students to core.mctsv4 (specialized to them, and it
rejects anything else) and legacy v2 nets to core.mctsv3.

Supports both regular and FP16-compressed checkpoints.
Optionally uses MCTS for stronger play.

Usage:
    python play.py                                           # uses default checkpoint
    python play.py models/guofish5_10M/v5_10.9M_best.pt
    python play.py --mcts --simulations 800                  # use MCTS search
"""

import argparse
import io
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
import chess.polyglot
import chess.syzygy
import torch
import torch.nn as nn

# Make the project root importable when this file is run as a script
# (python playing/v4/playv4.py) rather than as a module. Must happen before
# importing `core`, which lives at the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import core.mctsv3 as mctsv3
import core.mctsv4 as mctsv4
from core.mctsv3 import count_pieces  # architecture-independent board helper
# The v5 student's definition lives with its trainer so there is exactly one
# copy of the architecture: the checkpoint's `config` dict is written from the
# same ModelConfig we rebuild it with, so width, depth, FFN activation and the
# final LayerNorm can never drift between training and play.
from training.v5_multiPV.model_v5 import (ChessTransformerV5, ModelConfig,
                                          config_from_checkpoint)

OPENING_BOOK_PATH = _PROJECT_ROOT / "assets" / "gm2001.bin"

# Path for syzygy tablebase
SYZYGY_PATH = _PROJECT_ROOT / "assets" / "syzygy"
# Total piece count (both kings included) at or below which we probe the
# tablebase instead of searching. The downloaded set covers up to 5 pieces.
TABLEBASE_MAX_PIECES = 5

# --- Turbo MCTS evaluator tuning (--turbo) ---
# Hand-tuned overrides for the BatchedEvaluator, aimed at raising GPU
# utilization above the auto-tuned defaults (which leave the GPU idle when
# few workers feed small batches). More workers keep more leaves in flight,
# and a larger min batch forces the evaluator to coalesce them into fuller
# forward passes before firing. Wired in only when --turbo is passed; edit
# these freely to tune. See ParallelMCTS / BatchedEvaluator in core/mctsv3.py.
TURBO_NUM_WORKERS = 128
TURBO_MIN_BATCH_SIZE = 128
TURBO_MAX_BATCH_SIZE = 1024

# --- Model definition ---

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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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


# --- Loading utilities ---

# Attributes load_model attaches to the returned module and must survive the
# CPU quantization rebuild below. `seq_length` is the one core.mctsv3 reads to
# validate the tokenization; the other two are metadata for callers.
_CARRIED_ATTRS = ("seq_length", "config", "value_scale")


def inference_dtype(device: torch.device) -> torch.dtype:
    """The dtype weights should live in for this device.

    BF16 on CUDA, because that is what the engine actually computes in:
    core.mctsv3's evaluator runs every forward pass under bf16 autocast, and v5
    was trained in bf16 too (training/v5_multiPV/configs/base.yaml pins
    `amp: bf16  # matches inference precision`). Storing bf16 makes training,
    storage and inference one format end to end.

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
    if seq_length != 68:
        # Notably the V1 65-token scheme: core.mctsv3 only emits 68 tokens, so
        # such a model would be fed a mis-shaped board and silently evaluate
        # garbage. Refuse it here rather than at the first forward pass.
        raise ValueError(f"Unknown architecture: pos_encoder has {seq_length} "
                         "positions; only the 68-token scheme is supported")

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


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load a checkpoint, selecting the architecture from its own metadata.

    Handles both generations ParallelMCTS can drive -- v5 (training/v5_multiPV)
    and legacy v2 (guofish2..guofish4) -- with no flag to get wrong; see
    build_model_for_checkpoint for the dispatch. The returned module always
    advertises `seq_length`, plus `config`/`value_scale` when the checkpoint
    carried them.
    """
    print(f"Loading {checkpoint_path} on {device}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract state dict. A bare state dict is also accepted (fp16 exports).
    state_dict = (ckpt["model_state_dict"]
                  if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt)

    model, description = build_model_for_checkpoint(ckpt, state_dict)
    print(f"Detected {description}")
    for line in describe_checkpoint(ckpt):
        print(line)

    dtype = inference_dtype(device)
    print(f"Precision: {str(dtype).removeprefix('torch.')}")
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
        # so the device probe never runs — the slow path uses self.activation (still the
        # real activation), so this only affects the fast-path decision, not the math.
        for layer in model.transformer.layers:
            layer.activation_relu_or_gelu = False

    return model


# --- Search engine selection ---

# Either search implementation. They expose the same ParallelMCTS surface
# (search/get_policy/apply_move/reset/ponder_*/evaluator), so callers only need
# the union for typing; build_mcts decides which one they actually get.
MCTSEngine = mctsv3.ParallelMCTS | mctsv4.ParallelMCTS


def is_v5_model(model: nn.Module) -> bool:
    """True if `model` is the v5 student, i.e. mctsv4's exclusive target.

    Tested on the ModelConfig the module carries rather than isinstance, so a
    torch.ao dynamically-quantized copy (the CPU path in load_model, which
    returns a rebuilt module) still answers True. Legacy v2 nets have no
    `config` at all -- see core.mctsv4.require_v5_config, which re-checks this
    and the format contracts behind it.
    """
    return hasattr(getattr(model, "config", None), "seq_len")


def build_mcts(model: nn.Module, device: torch.device, **kwargs) -> MCTSEngine:
    """Construct the search engine matching the model's architecture.

    v5 students get core.mctsv4, which is specialized to them and validates the
    encoding/policy/CLS contracts against the model's own ModelConfig. Legacy v2
    nets keep core.mctsv3, which is unchanged and still accepts them; mctsv4
    would refuse them at construction.

    Routing lives here, next to load_model, so the architecture is decided once
    from the checkpoint. `kwargs` passes straight through to the chosen
    ParallelMCTS.

    Used by this file's interactive play, which accepts either generation. The
    UCI wrappers do NOT route: playing/uci_wrapper.py is v2-only and
    playing/uci_wrapper_v5.py is v5-only, each constructing its one search
    directly, so a tournament engine can never silently run the other
    architecture's search.
    """
    module = mctsv4 if is_v5_model(model) else mctsv3
    return module.ParallelMCTS(model, device, **kwargs)


# --- ANSI color codes for console output ---

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


# --- Token constants for V2 architecture ---

TOKEN_WHITE_TO_MOVE = 13
TOKEN_BLACK_TO_MOVE = 14
TOKEN_CASTLING_BASE = 15  # castling = base + (K*8 + Q*4 + k*2 + q*1)
TOKEN_EP_NONE = 31
TOKEN_EP_BASE = 32  # ep file a-h = base + file (0-7)
TOKEN_CLS = 40


# --- Encoding helpers ---

def board_to_tokens_v2(board: chess.Board) -> torch.Tensor:
    """V2: 68 tokens (64 squares + side + castling + ep + CLS)."""
    tokens = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            tokens.append(0)
        else:
            offset = 0 if piece.color else 6
            tokens.append(piece.piece_type + offset)

    # Position 64: side to move
    tokens.append(TOKEN_WHITE_TO_MOVE if board.turn else TOKEN_BLACK_TO_MOVE)

    # Position 65: castling rights (4-bit encoded)
    castling_bits = (
        (8 if board.has_kingside_castling_rights(chess.WHITE) else 0) |
        (4 if board.has_queenside_castling_rights(chess.WHITE) else 0) |
        (2 if board.has_kingside_castling_rights(chess.BLACK) else 0) |
        (1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
    )
    tokens.append(TOKEN_CASTLING_BASE + castling_bits)

    # Position 66: en passant target file
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        tokens.append(TOKEN_EP_BASE + ep_file)
    else:
        tokens.append(TOKEN_EP_NONE)

    # Position 67: CLS token
    tokens.append(TOKEN_CLS)

    return torch.tensor(tokens, dtype=torch.long)


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(4096, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move.from_square * 64 + move.to_square] = True
    return mask


def decode_move(index: int, board: chess.Board) -> chess.Move:
    from_sq = index // 64
    to_sq = index % 64
    piece = board.piece_at(from_sq)
    promotion = None
    if piece is not None and piece.piece_type == chess.PAWN:
        rank = chess.square_rank(to_sq)
        if (piece.color == chess.WHITE and rank == 7) or (piece.color == chess.BLACK and rank == 0):
            promotion = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion=promotion)


def pick_engine_move(model: nn.Module, board: chess.Board, device: torch.device,
                     mcts_engine=None, num_simulations: int = 800,
                     temperature: float = 0.0, avoid_repetition: bool = True) -> tuple[chess.Move | None, dict]:
    """Pick engine move using either raw policy or MCTS.

    Args:
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        avoid_repetition: If True and winning, avoid moves that repeat positions

    Returns:
        tuple: (move, stats_dict) where stats_dict contains timing and search info
    """
    import numpy as np
    start_time = time.time()
    stats = {}

    if mcts_engine is not None:
        # Use MCTS search - get policy distribution
        policy_dict = mcts_engine.get_policy(board, num_simulations=num_simulations)
        elapsed = time.time() - start_time

        if not policy_dict:
            return None, stats

        moves = list(policy_dict.keys())
        visit_probs = np.array([policy_dict[m] for m in moves])

        # Get eval from root Q-value.
        # last_root_q is from engine's (side-to-move) perspective.
        # Convert to absolute: positive = White winning, negative = Black winning.
        root_q = mcts_engine.last_root_q
        stats['eval'] = root_q if board.turn == chess.WHITE else -root_q

        # Anti-repetition: only trigger when we're winning AND opponent has already
        # caused a 2-fold repetition (one more repeat by us = 3-fold draw claim).
        # Among moves MCTS considers near-co-best (>= 90% of best visits), avoid
        # the ones that would complete the 3-fold repetition.
        if avoid_repetition and root_q > 0.15 and board.is_repetition(2):
            best_prob = np.max(visit_probs)
            threshold = best_prob * 0.9  # Only moves MCTS considers near-equal to best
            good_moves_mask = visit_probs >= threshold

            # Check which moves would trigger a 3-fold repetition draw
            repeats = []
            for move in moves:
                board.push(move)
                repeats.append(board.is_repetition(3))
                board.pop()

            # Only penalize if there's at least one near-co-best move that doesn't draw
            has_good_non_repeat = any(good_moves_mask[i] and not repeats[i]
                                       for i in range(len(moves)))

            if has_good_non_repeat:
                for i in range(len(moves)):
                    if good_moves_mask[i] and repeats[i]:
                        visit_probs[i] *= 0.01

                if visit_probs.sum() > 0:
                    visit_probs = visit_probs / visit_probs.sum()
                else:
                    visit_probs = np.array([policy_dict[m] for m in moves])

        # Apply temperature sampling
        if temperature > 0.001:
            # Transform visit probabilities with temperature
            visit_counts = visit_probs * num_simulations  # Approximate counts
            visit_counts = np.power(visit_counts, 1.0 / temperature)
            visit_probs = visit_counts / visit_counts.sum()
            move = np.random.choice(moves, p=visit_probs)
            stats['sampled'] = True
        else:
            # Deterministic: pick most visited
            move = moves[np.argmax(visit_probs)]
            stats['sampled'] = False

        # Capture the raw root search distribution (visit_count per legal move)
        # straight from the persistent tree. This is the TRUE allocation of
        # simulations, read before the temperature/anti-repetition reshaping
        # above could influence selection. `board` is still the pre-move (root)
        # position here, so board.san() is valid for each child move. q_value is
        # from the side-to-move (engine) perspective; virtual losses are already
        # reverted post-search, so it is just value_sum / visit_count.
        root_node = getattr(mcts_engine, 'root', None)
        if root_node is not None and root_node.children:
            total_visits = sum(c.visit_count for c in root_node.children.values())
            root_dist = [
                {
                    'move': m,
                    'san': board.san(m),
                    'visits': c.visit_count,
                    'share': (c.visit_count / total_visits) if total_visits else 0.0,
                    'prior': c.prior,
                    'q': c.q_value,
                }
                for m, c in root_node.children.items()
            ]
            root_dist.sort(key=lambda d: d['visits'], reverse=True)
            stats['root_dist'] = root_dist
            stats['root_total_visits'] = total_visits

            # Capture the principal variation: from the root, repeatedly descend
            # into the most-visited child (the same criterion search uses to pick
            # the move), recording each ply's visits, prior, and Q. This is the
            # line MCTS actually believes will be played out. `q` is from the
            # mover's perspective (who moved TO that node); a separate board copy
            # generates SAN since we push moves past the root position.
            pv = []
            pv_board = board.copy(stack=False)
            node = root_node
            while node.children:
                child_move, child = max(node.children.items(),
                                        key=lambda kv: kv[1].visit_count)
                if child.visit_count == 0:
                    break
                pv.append({
                    'san': pv_board.san(child_move),
                    'visits': child.visit_count,
                    'prior': child.prior,
                    'q': child.q_value,
                })
                pv_board.push(child_move)
                node = child
            stats['pv'] = pv

        # Get stats
        stats['time'] = elapsed
        stats['simulations'] = num_simulations
        stats['sims_per_sec'] = num_simulations / elapsed if elapsed > 0 else 0
        stats['batches'] = mcts_engine.evaluator.total_batches
        stats['avg_batch'] = (mcts_engine.evaluator.total_evals /
                              max(1, mcts_engine.evaluator.total_batches))

        # Reset evaluator stats for next move
        mcts_engine.evaluator.total_batches = 0
        mcts_engine.evaluator.total_evals = 0
    else:
        # Use raw policy (single forward pass)
        tokens = board_to_tokens_v2(board).unsqueeze(0).to(device)
        mask = legal_move_mask(board).unsqueeze(0).to(device)

        with torch.no_grad():
            # Low-precision weights (bf16 on Ampere+, fp16 otherwise) need the
            # matching autocast so activations are computed in the same format
            # the MCTS evaluator uses. FP32/INT8 CPU models fall through.
            param_dtype = next(model.parameters()).dtype
            if param_dtype in (torch.float16, torch.bfloat16):
                with torch.autocast(device_type=device.type, dtype=param_dtype):
                    policy_logits, value = model(tokens, legal_move_mask=mask)
            else:
                policy_logits, value = model(tokens, legal_move_mask=mask)

        best_index = int(torch.argmax(policy_logits, dim=1).item())
        move = decode_move(best_index, board)
        elapsed = time.time() - start_time

        # NN outputs absolute value already (White winning = +, Black winning = -).
        stats['time'] = elapsed
        stats['eval'] = value.item()
        stats['simulations'] = None
        stats['sampled'] = False

    return move, stats


def probe_opening_book(book_reader: Optional[chess.polyglot.MemoryMappedReader],
                       board: chess.Board,
                       rng: Optional[random.Random] = None) -> Optional[chess.Move]:
    """Look up current position in opening book. Returns weighted-random book move or None.

    `rng` makes the weighted choice reproducible when seeded (--book-seed);
    pass None for nondeterministic selection.
    """
    if book_reader is None:
        return None
    try:
        entry = book_reader.weighted_choice(board, random=rng)
        return entry.move
    except IndexError:
        return None


def probe_tablebase_move(tablebase: Optional[chess.syzygy.Tablebase],
                         board: chess.Board) -> Optional[chess.Move]:
    """Mode 1 root bypass: return the tablebase-optimal move for `board`, or None.

    Mirrors the UCI wrapper's bypass. Among legal moves, picks the best WDL
    outcome from our perspective (win > draw > loss), then within that outcome
    the move that makes fastest progress toward a zeroing move (smallest DTZ
    when winning, largest when losing, so the 50-move counter resets before it
    can force a draw). python-chess's Tablebase has no probe_root, so we do the
    manual root probe: probe_wdl/probe_dtz report from the side-to-move's
    perspective, and after we push our move it is the opponent to move, so we
    negate to get our perspective.

    Returns None (caller falls back to search) if the tablebase is closed or any
    required table is missing.
    """
    if tablebase is None:
        return None

    best_move: Optional[chess.Move] = None
    best_key: Optional[tuple] = None
    try:
        for move in board.legal_moves:
            zeroing = board.is_zeroing(move)
            board.push(move)
            try:
                if board.is_checkmate():
                    # Immediate mate: strictly better than any tablebase win.
                    outcome, distance = 3, 0
                elif board.is_stalemate() or board.is_insufficient_material():
                    # Drawn terminal position regardless of the tablebase.
                    outcome, distance = 0, 0
                else:
                    wdl_child = tablebase.probe_wdl(board)
                    dtz_child = tablebase.probe_dtz(board)
                    our_wdl = -wdl_child  # negate: opponent is to move at child
                    if our_wdl > 0:
                        # Winning: a zeroing move that keeps the win is the best
                        # kind of progress; else minimize plies-to-zero.
                        outcome = 1
                        distance = 0 if zeroing else -dtz_child
                    elif our_wdl < 0:
                        # Losing: stall — maximize the opponent's plies-to-zero.
                        outcome = -1
                        distance = -dtz_child
                    else:
                        outcome, distance = 0, 0
            finally:
                board.pop()

            # Maximize outcome, then minimize distance.
            key = (outcome, -distance)
            if best_key is None or key > best_key:
                best_key = key
                best_move = move
    except (chess.syzygy.MissingTableError, KeyError, ValueError):
        # MissingTableError subclasses KeyError; any miss => fall back to search.
        return None

    return best_move


def format_engine_stats(stats: dict) -> str:
    """Format engine stats for display."""
    parts = []

    if stats.get('eval') is not None:
        parts.append(f"eval: {stats['eval']:+.3f}")

    if stats.get('simulations') is not None:
        parts.append(f"sims: {stats['simulations']}")
        parts.append(f"{stats['sims_per_sec']:.0f} sims/s")
        if stats.get('avg_batch'):
            parts.append(f"batch: {stats['avg_batch']:.0f}")

    parts.append(f"time: {stats['time']*1000:.0f}ms")

    return " | ".join(parts)


def print_batch_summary(mcts_engine: Optional[MCTSEngine]) -> None:
    """Print the end-of-game batch-size diagnostic (gated by --stats at the call
    site). The detail — average size, histogram, and min/max saturation — lives
    on the evaluator, which holds the configured min/max batch parameters. No-op
    for raw-policy play (no evaluator batching happened)."""
    if mcts_engine is None:
        return
    print()
    print(mcts_engine.evaluator.format_batch_summary())
    print()


def format_move_history(board: chess.Board, last_n: int = 8) -> str:
    """Return the last last_n plies of the game in PGN-style notation."""
    moves = list(board.move_stack)
    if not moves:
        return "(start of game)"

    # Replay from scratch to get SAN for each ply
    tmp = chess.Board()
    san_list = []
    for move in moves:
        san_list.append(tmp.san(move))
        tmp.push(move)

    start_ply = max(0, len(san_list) - last_n)
    recent = san_list[start_ply:]

    parts = []
    move_num = (start_ply // 2) + 1
    i = 0

    if start_ply % 2 == 1:
        # First shown ply is Black's move
        parts.append(f"{move_num}...{recent[0]}")
        i, move_num = 1, move_num + 1

    while i < len(recent):
        white = recent[i]
        if i + 1 < len(recent):
            parts.append(f"{move_num}.{white} {recent[i + 1]}")
            i += 2
        else:
            parts.append(f"{move_num}.{white}")
            i += 1
        move_num += 1

    return " ".join(parts)


def format_search_distribution(stats: dict, chosen_move: Optional[chess.Move] = None,
                               top_n: int = 16) -> str:
    """Render the root search distribution gathered by pick_engine_move.

    Shows, per root move, how MCTS actually allocated its simulations: visit
    count and share (the real search distribution), the network's policy prior,
    and the backed-up Q from the side-to-move's perspective (+ = good for the
    engine). Sorted by visits; the move the engine ultimately played is marked
    (it can differ from the most-visited move under temperature sampling or the
    anti-repetition reshaping in pick_engine_move).
    """
    dist = stats.get('root_dist')
    if not dist:
        return "(no search distribution -- book/tablebase/raw-policy move)"
    total = stats.get('root_total_visits') or sum(d['visits'] for d in dist) or 1

    lines = []
    head = f"Search distribution: {total} root visits across {len(dist)} moves"
    ev = stats.get('eval')
    if ev is not None:
        head += f" | root eval {ev:+.3f} (absolute, + = White)"
    lines.append(head)
    lines.append(f"  {'move':<8}{'visits':>9}{'share':>8}{'prior':>8}{'Q(stm)':>9}")
    for d in dist[:top_n]:
        mark = "  <-- played" if (chosen_move is not None and d['move'] == chosen_move) else ""
        lines.append(f"  {d['san']:<8}{d['visits']:>9}{d['share'] * 100:>7.1f}%"
                     f"{d['prior'] * 100:>7.1f}%{d['q']:>+9.3f}{mark}")
    rest = dist[top_n:]
    if rest:
        rv = sum(d['visits'] for d in rest)
        lines.append(f"  {('(+%d more)' % len(rest)):<8}{rv:>9}{rv / total * 100:>7.1f}%")
    return "\n".join(lines)


def format_pv(stats: dict, max_plies: int = 12) -> str:
    """Render the principal variation captured by pick_engine_move.

    The PV is the line MCTS most believes will be played: from the root, descend
    into the most-visited child at each ply. For each ply we show its SAN, the
    visit count at that node (how much search backed this continuation), the
    network's policy prior, and the backed-up Q from the mover's perspective
    (+ = good for whoever moved into that node, so the sign alternates down the
    line). Truncated to `max_plies`.
    """
    pv = stats.get('pv')
    if not pv:
        return "(no PV -- book/tablebase/raw-policy move)"

    # One-line SAN summary with move numbers, then a per-ply detail table.
    san_line = format_san_line([p['san'] for p in pv])
    lines = [f"Principal variation ({len(pv)} plies): {san_line}"]
    lines.append(f"  {'ply':<5}{'move':<8}{'visits':>9}{'prior':>8}{'Q(mover)':>10}")
    for i, p in enumerate(pv[:max_plies], start=1):
        lines.append(f"  {i:<5}{p['san']:<8}{p['visits']:>9}"
                     f"{p['prior'] * 100:>7.1f}%{p['q']:>+10.3f}")
    if len(pv) > max_plies:
        lines.append(f"  (+{len(pv) - max_plies} more plies)")
    return "\n".join(lines)


def format_san_line(sans: list[str], start_move_no: int = 1,
                    white_to_move: bool = True) -> str:
    """Join a list of SAN strings into PGN-style numbered notation.

    Defaults assume the line starts with the side to move at the search root;
    the engine root display always starts on the moving side, so move numbering
    is purely cosmetic here (always begins '1.').
    """
    parts = []
    move_no = start_move_no
    i = 0
    if not white_to_move and sans:
        parts.append(f"{move_no}...{sans[0]}")
        i, move_no = 1, move_no + 1
    while i < len(sans):
        if i + 1 < len(sans):
            parts.append(f"{move_no}.{sans[i]} {sans[i + 1]}")
            i += 2
        else:
            parts.append(f"{move_no}.{sans[i]}")
            i += 1
        move_no += 1
    return " ".join(parts)


def load_pgn_mainline(pgn_path: Path, max_ply: Optional[int] = None) -> list[chess.Move]:
    """Read the first game from a PGN file and return its mainline moves.

    Used by --pgn for walk-back analysis. `max_ply` (from --pgn-ply) truncates
    the line so the caller can stop at the exact position to analyze.
    """
    with open(pgn_path, encoding="utf-8") as f:
        game = chess.pgn.read_game(f)
    if game is None:
        raise ValueError(f"No game found in PGN file: {pgn_path}")
    moves = list(game.mainline_moves())
    if max_ply is not None:
        moves = moves[:max_ply]
    return moves


class RestartGame(Exception):
    """Raised from any input prompt to abort the current game and start a new one."""


_RESTART_WORDS = {"new", "new game", "restart", "again", "play again"}


def prompt(message: str) -> str:
    """input() wrapper that raises RestartGame when the user types a restart keyword."""
    value = input(message).strip()
    if value.lower() in _RESTART_WORDS:
        raise RestartGame()
    return value


def ask_play_again() -> bool:
    while True:
        ans = input("Play again? [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", "quit", "exit"):
            return False
        print("Please enter 'y' or 'n'.")

# "guofish2_25.6M_54.8p.pt"
def main():
    parser = argparse.ArgumentParser(description="Play chess against ChessTransformer")
    parser.add_argument("checkpoint", type=Path, nargs="?",
                        default=_PROJECT_ROOT / "models" / "guofish5_20M" / "v5_10.9M_best.pt",
                        help="Path to model checkpoint (v5 or legacy v2; the "
                             "architecture is read from the checkpoint)")
    parser.add_argument("--mcts", action="store_true",
                        help="Use MCTS search instead of raw policy")
    parser.add_argument("--simulations", type=int, default=800,
                        help="Number of MCTS simulations per move (default: 800)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of MCTS worker threads (default: auto)")
    parser.add_argument("--turbo", action="store_true",
                        help=f"Override the MCTS evaluator's auto-tuning with the "
                             f"hardcoded TURBO_* constants ({TURBO_NUM_WORKERS} workers, "
                             f"batch {TURBO_MIN_BATCH_SIZE}-{TURBO_MAX_BATCH_SIZE}) to "
                             "push GPU utilization higher. --workers still overrides "
                             "the worker count if given.")
    parser.add_argument("--stats", action="store_true",
                        help="Verbose search diagnostics (requires --mcts). After every "
                             "engine search, print the root search distribution: visits, "
                             "share, policy prior, and Q per candidate move, followed by "
                             "the principal variation (most-visited line from the root) "
                             "with per-ply visits and priors. At the end of each game "
                             "(checkmate or 'quit'), also print a batch-size diagnostic "
                             "(average size, histogram, min/max saturation) useful for "
                             "tuning --turbo / GPU utilization.")
    parser.add_argument("--pv", action="store_true",
                        help="Print the principal variation after every engine move "
                             "(requires --mcts). This is the PV half of --stats without "
                             "the root search distribution or the end-of-game batch "
                             "diagnostic; --stats already implies it, so passing both "
                             "prints the PV once.")
    parser.add_argument("--pgn", type=Path, default=None,
                        help="Walk-back analysis: load this PGN file and replay its "
                             "mainline before interactive play starts, so the first game "
                             "begins at the loaded position with the engine to move. "
                             "Combine with --pgn-ply to stop at a specific point.")
    parser.add_argument("--pgn-ply", type=int, default=None,
                        help="With --pgn, replay only the first N plies (half-moves) of "
                             "the mainline. Set N so the engine is the side to move at "
                             "the position you want to analyze (default: whole game).")
    parser.add_argument("--book", type=str, default=str(OPENING_BOOK_PATH),
                        help=f"Polyglot opening book path (default: {OPENING_BOOK_PATH}). "
                             "On a book hit the engine returns the book move and skips MCTS.")
    parser.add_argument("--no-book", dest="use_book", action="store_false",
                        help="Disable the opening book; play every move with MCTS.")
    parser.add_argument("--book-seed", type=int, default=None,
                        help="Seed for the book's weighted-random move selection "
                             "(default: nondeterministic). Pin this for reproducible "
                             "book lines in tournaments.")
    parser.add_argument("--syzygy", type=str, default=str(SYZYGY_PATH),
                        help=f"Syzygy tablebase directory (default: {SYZYGY_PATH}). "
                             f"Positions with <= {TABLEBASE_MAX_PIECES} pieces skip "
                             "MCTS and play the tablebase-optimal move.")
    parser.add_argument("--no-syzygy", dest="use_syzygy", action="store_false",
                        help="Disable the endgame tablebase bypass; play every "
                             "position with MCTS.")
    parser.add_argument("--no-ponder", dest="ponder", action="store_false",
                        help="Disable background pondering while waiting for the human's move.")
    parser.set_defaults(use_book=True, use_syzygy=True, ponder=True)
    args = parser.parse_args()

    book_path = Path(args.book) if args.use_book else None
    tablebase_path = Path(args.syzygy) if args.use_syzygy else None

    if not args.checkpoint.exists():
        print(f"Error: {args.checkpoint} not found")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    # Initialize MCTS if requested
    mcts_engine: Optional[MCTSEngine] = None
    if args.mcts:
        if args.turbo:
            # Turbo: feed the hardcoded TURBO_* constants straight into the
            # BatchedEvaluator (via ParallelMCTS) instead of its hardware
            # auto-tuning. An explicit --workers still wins over the worker count.
            mcts_engine = build_mcts(
                model, device,
                num_workers=args.workers if args.workers is not None else TURBO_NUM_WORKERS,
                min_batch_size=TURBO_MIN_BATCH_SIZE,
                max_batch_size=TURBO_MAX_BATCH_SIZE,
            )
        else:
            # Let ParallelMCTS auto-tune workers unless explicitly specified
            mcts_engine = build_mcts(model, device, num_workers=args.workers)
        print(f"MCTS enabled: {type(mcts_engine).__module__} | {args.simulations} simulations, "
              f"{mcts_engine.num_workers} workers, batch size "
              f"{mcts_engine.evaluator.min_batch_size}-{mcts_engine.evaluator.max_batch_size}"
              f"{' [turbo]' if args.turbo else ''}")

    # Open the Polyglot book once at init. Keep the reader open for the
    # process lifetime; chess.polyglot uses an mmap so the per-lookup
    # cost is just a binary search on the move-key table.
    book_reader: Optional[chess.polyglot.MemoryMappedReader] = None
    if book_path is not None:
        try:
            book_reader = chess.polyglot.open_reader(str(book_path))
        except Exception as e:
            book_reader = None
    # Seedable RNG for the book's weighted choice (--book-seed); None => random.
    book_rng = random.Random(args.book_seed)

    # Open the Syzygy tablebase once. Like the book, this is a one-time
    # cost; chess.syzygy mmaps the table files so per-probe overhead is
    # just a lookup into already-mapped (and, after warmup, cached) pages.
    tablebase: Optional[chess.syzygy.Tablebase] = None
    if tablebase_path is not None:
        if tablebase_path.is_dir():
            try:
                tablebase = chess.syzygy.open_tablebase(str(tablebase_path))
            except Exception as e:
                tablebase = None

    # Hand the same tablebase handle to MCTS for Mode 2 (leaf evaluation:
    # interior leaves with <= 5 pieces get the exact WDL value instead of
    # the neural value head's estimate). Shared read-only across workers;
    # chess.syzygy probing is thread-safe with per-worker board copies.
    if mcts_engine:
        mcts_engine.tablebase = tablebase

    # Optional PGN walk-back: preload the mainline so the FIRST game starts at
    # the loaded position with the engine to move (for analyzing a real game).
    # Consumed once; restarts ('new') fall back to the normal side prompt.
    pgn_preload: Optional[list[chess.Move]] = None
    if args.pgn is not None:
        pgn_preload = load_pgn_mainline(args.pgn, args.pgn_ply)

    # Outer restart loop — any input prompt can raise RestartGame to land back here.
    while True:
        try:
            if pgn_preload is not None:
                # Walk-back mode: replay the loaded PGN line and let the engine
                # take whichever side is to move at the final position.
                board = chess.Board()
                for mv in pgn_preload:
                    board.push(mv)
                human_side = chess.BLACK if board.turn == chess.WHITE else chess.WHITE
                if mcts_engine is not None:
                    mcts_engine.reset()  # search builds a fresh tree from the loaded position
                print(f"\nLoaded {len(pgn_preload)} plies from {args.pgn.name}. "
                      f"Engine ({'White' if board.turn == chess.WHITE else 'Black'}) to move.")
            else:
                # Ask the user which side to play
                while True:
                    side_input = prompt("Play as white or black? [w/b]: ").lower()
                    if side_input in ("w", "white"):
                        human_side = chess.WHITE
                        break
                    if side_input in ("b", "black"):
                        human_side = chess.BLACK
                        break
                    print("Please enter 'w' or 'b'.")

                board = chess.Board()

                # Optional: paste a full PGN move sequence to start from a specific position
                pgn_inject = prompt("Paste PGN moves to start from a position (or press Enter for new game): ").strip()
                if pgn_inject:
                    try:
                        game = chess.pgn.read_game(io.StringIO(pgn_inject))
                        if game is not None:
                            pgn_moves = list(game.mainline_moves())
                            for mv in pgn_moves:
                                board.push(mv)
                            print(f"Loaded {len(pgn_moves)} moves.")
                            print(board)
                            print(f"History: {format_move_history(board, last_n=8)}")
                            print()
                        else:
                            print("Could not parse PGN. Starting from the beginning.")
                    except Exception as e:
                        print(f"Could not parse PGN ({e}). Starting from the beginning.")

            # Fresh batch-size log per game so the --stats end-of-game summary
            # only reflects this game's searches.
            if args.stats and mcts_engine is not None:
                mcts_engine.evaluator.reset_batch_history()
            print("\nStarting game. Enter moves in SAN (e.g. e4, Nf3, O-O).")
            print("To inject an engine move, enter two moves: 'e4 e5' (your move, then engine's).")
            print("Type 'undo' to rewind one full move, 'new' to start a new game, 'quit' to stop.\n")

            def mcts_apply(move: chess.Move) -> None:
                """Advance the MCTS tree to match the board. No-op if MCTS not in use."""
                if mcts_engine is not None:
                    mcts_engine.apply_move(move)

            def start_ponder() -> None:
                """Start background MCTS on the predicted user reply, if possible."""
                if mcts_engine is None or board.is_game_over() or not args.ponder:
                    return
                # ParallelMCTS.ponder_start auto-selects top-1 or top-K
                # branches based on root-visit confidence.
                mcts_engine.ponder_start(board)

            def stop_ponder() -> None:
                if mcts_engine is not None:
                    mcts_engine.ponder_stop()

            def play_engine_move() -> bool:
                """Probe book, then tablebase, else run engine search.

                Returns False if there are no legal moves. Move-selection order:
                opening book (openings) -> Syzygy bypass (<= 5-piece endgames,
                Mode 1) -> engine search. The tablebase bypass fires in BOTH raw-
                policy and MCTS modes, so endgames are played perfectly even when
                MCTS is off (Mode 2 leaf eval only helps when MCTS is running).
                """
                book_move = probe_opening_book(book_reader, board, book_rng)
                if book_move is not None:
                    print(f"{RED}Engine plays: {board.san(book_move)}{RESET}  [book]\n")
                    board.push(book_move)
                    mcts_apply(book_move)
                    start_ponder()
                    return True

                # Mode 1: <= 5-piece positions skip search and play the
                # tablebase-optimal move directly. probe_tablebase_move returns
                # None on any miss, so we fall through to search.
                if tablebase is not None and count_pieces(board) <= TABLEBASE_MAX_PIECES:
                    tb_move = probe_tablebase_move(tablebase, board)
                    if tb_move is not None:
                        print(f"{RED}Engine plays: {board.san(tb_move)}{RESET}  [syzygy]\n")
                        board.push(tb_move)
                        mcts_apply(tb_move)
                        start_ponder()
                        return True

                move, stats = pick_engine_move(model, board, device, mcts_engine, args.simulations)
                if move is None:
                    print("Engine has no legal moves!")
                    return False
                print(f"{RED}Engine plays: {board.san(move)}{RESET}  [{format_engine_stats(stats)}]\n")
                # Both blocks hang off root_dist because that is what says MCTS
                # actually ran: a raw-policy move (no --mcts) has no tree, so
                # there is no distribution to show and no line to descend.
                if stats.get('root_dist'):
                    if args.stats:
                        print(format_search_distribution(stats, move))
                        print()
                    if args.stats or args.pv:
                        print(format_pv(stats))
                        print()
                board.push(move)
                mcts_apply(move)
                start_ponder()
                return True

            def end_game() -> None:
                """Ask whether to play again. Returns via RestartGame if yes, else exits main()."""
                # Game-level batch diagnostic before the next game's reset wipes
                # the stats (or the process exits).
                if args.stats:
                    print_batch_summary(mcts_engine)
                if not ask_play_again():
                    raise SystemExit(0)
                raise RestartGame()

            # PGN walk-back: the engine is to move at the loaded position, so
            # search it immediately (this prints the --stats distribution), then
            # hand control to the user to inject the game's continuation.
            if pgn_preload is not None:
                pgn_preload = None  # only the first game uses the preload
                print(format_move_history(board, last_n=12))
                print(board)
                print()
                if not play_engine_move():
                    end_game()
            # Engine goes first: either fresh game with human=black, or PGN-loaded position
            elif board.turn != human_side:
                if len(board.move_stack) == 0:
                    # Fresh game, human is black: allow injecting engine's opening move
                    first_move_input = prompt("Engine's first move (press Enter for engine choice): ")
                    if first_move_input:
                        try:
                            move = board.parse_san(first_move_input)
                            print(f"{RED}Engine plays: {board.san(move)}{RESET}  [injected]\n")
                            board.push(move)
                            mcts_apply(move)
                            start_ponder()
                        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError) as e:
                            print(f"Could not parse '{first_move_input}' ({e.__class__.__name__}). Engine will choose.")
                            if not play_engine_move():
                                end_game()
                    else:
                        if not play_engine_move():
                            end_game()
                else:
                    # Loaded position with engine to move: search immediately
                    if not play_engine_move():
                        end_game()

            while True:
                raw = prompt(f"{GREEN}Your move: {RESET}")
                # Stop any pondering before mutating board/tree in response to input.
                # Ponder was running in the background while prompt() blocked.
                stop_ponder()
                if raw.lower() in ("quit", "exit"):
                    print("Quitting.")
                    if args.stats:
                        print_batch_summary(mcts_engine)
                    return

                # Handle undo command
                if raw.lower() in ("undo", "back"):
                    if len(board.move_stack) >= 2:
                        # Undo both player and engine moves
                        undone_engine = board.pop()
                        undone_player = board.pop()
                        # Tree has no inverse of apply_move; rebuild from scratch next search.
                        if mcts_engine is not None:
                            mcts_engine.reset()
                        print(f"Undid: {undone_player} (you), {undone_engine} (engine)")
                        print(board)
                        print(f"History: {format_move_history(board)}")
                        print()
                    elif len(board.move_stack) == 1 and human_side == chess.BLACK:
                        # Undo just the engine's first move (playing as black)
                        undone_engine = board.pop()
                        if mcts_engine is not None:
                            mcts_engine.reset()
                        print(f"Undid engine's first move: {undone_engine}")
                        print(board)
                        print(f"History: {format_move_history(board)}")
                        print()
                        # Re-prompt for engine's first move
                        first_move_input = prompt("Engine's first move (press Enter for engine choice): ")
                        if first_move_input:
                            try:
                                move = board.parse_san(first_move_input)
                                print(f"{RED}Engine plays: {board.san(move)}{RESET}  [injected]\n")
                                board.push(move)
                                mcts_apply(move)
                                start_ponder()
                            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError) as e:
                                print(f"Could not parse '{first_move_input}' ({e.__class__.__name__}). Engine will choose.")
                                if not play_engine_move():
                                    end_game()
                        else:
                            if not play_engine_move():
                                end_game()
                    else:
                        print("Nothing to undo.")
                    continue

                # Split input to check for injected engine move
                parts = raw.split()
                if not parts:
                    continue

                # Parse user's move (first part)
                try:
                    human_move = board.parse_san(parts[0])
                except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError) as e:
                    print(f"Could not parse '{parts[0]}' as a legal move ({e.__class__.__name__}). Try again.")
                    continue

                board.push(human_move)
                mcts_apply(human_move)

                if board.is_game_over():
                    outcome = board.outcome()
                    print(f"Game over: {outcome.result() if outcome else 'unknown'}")
                    if board.is_checkmate():
                        print("You win by checkmate!")
                    end_game()

                # Check for injected engine move (second part)
                injected_move = None
                if len(parts) >= 2:
                    try:
                        injected_move = board.parse_san(parts[1])
                    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError) as e:
                        print(f"Could not parse injected move '{parts[1]}' ({e.__class__.__name__}). Engine will play normally.")
                        injected_move = None

                if injected_move is not None:
                    # Use injected move instead of engine search
                    print(f"{RED}Engine plays: {board.san(injected_move)}{RESET}  [injected]\n")
                    board.push(injected_move)
                    mcts_apply(injected_move)
                    start_ponder()
                else:
                    # Normal engine move (book first, then search)
                    if not play_engine_move():
                        end_game()

                if board.is_game_over():
                    outcome = board.outcome()
                    print(f"Game over: {outcome.result() if outcome else 'unknown'}")
                    if board.is_checkmate():
                        print("Engine wins by checkmate!")
                    end_game()
        except RestartGame:
            # Stop any in-flight pondering and drop the old game's tree before
            # starting fresh. RestartGame can be raised mid-prompt so ponder
            # may still be running at this point.
            if mcts_engine is not None:
                mcts_engine.reset()
            print("\n--- Starting new game ---\n")
            continue


if __name__ == "__main__":
    main()
