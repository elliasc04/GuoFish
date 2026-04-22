"""Play chess against a ChessTransformer model.

Supports both regular and FP16-compressed checkpoints.
Optionally uses MCTS for stronger play.

Usage:
    python play.py                                           # uses default checkpoint
    python play.py models/chess_transformer_25.8M_50.5pct_fp16.pt
    python play.py --mcts --simulations 800                  # use MCTS search
"""

import argparse
import math
import time
from pathlib import Path

import chess
import torch
import torch.nn as nn


# --- Model definitions ---

class ChessTransformerV1(nn.Module):
    """Original architecture: 65 tokens (64 squares + side-to-move), mean pooling."""

    def __init__(self, vocab_size=15, d_model=512, nhead=8, num_layers=8, dropout=0.1, head_dim=256):
        super().__init__()
        self.seq_length = 65
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.pos_encoder = nn.Parameter(torch.randn(1, 65, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
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
        pooled_state = x.mean(dim=1)
        value = self.value_head(pooled_state).squeeze(-1)
        x_squares = x[:, :64, :]
        from_feats = self.from_proj(x_squares)
        to_feats = self.to_proj(x_squares)
        policy_logits = torch.bmm(from_feats, to_feats.transpose(1, 2)) * self.logit_scale
        policy_logits = policy_logits.view(x.size(0), 4096)
        if legal_move_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_move_mask, float('-inf'))
        return policy_logits, value


class ChessTransformerV2(nn.Module):
    """New architecture: 68 tokens (64 squares + side + castling + ep + CLS), CLS pooling."""

    def __init__(self, vocab_size=43, d_model=512, nhead=8, num_layers=8, dropout=0.1, head_dim=64):
        super().__init__()
        self.seq_length = 68
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.pos_encoder = nn.Parameter(torch.randn(1, 68, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
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

def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model from checkpoint, auto-detecting architecture version."""
    print(f"Loading {checkpoint_path} on {device}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        if "val_acc" in ckpt:
            print(f"Model accuracy: {ckpt['val_acc']:.1f}%")
    else:
        state_dict = ckpt

    # Auto-detect architecture from pos_encoder shape
    pos_encoder_shape = state_dict["pos_encoder"].shape
    seq_length = pos_encoder_shape[1]

    if seq_length == 65:
        print("Detected V1 architecture (65 tokens, mean pooling)")
        ModelClass = ChessTransformerV1
    elif seq_length == 68:
        print("Detected V2 architecture (68 tokens, CLS pooling)")
        ModelClass = ChessTransformerV2
    else:
        raise ValueError(f"Unknown architecture: pos_encoder has {seq_length} positions")

    # Convert FP16 weights back to FP32 for CPU, keep FP16 for CUDA
    if device.type == "cuda":
        model = ModelClass().half().to(device)
        state_dict = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
    else:
        model = ModelClass().to(device)
        state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


# --- Token constants for V2 architecture ---

TOKEN_WHITE_TO_MOVE = 13
TOKEN_BLACK_TO_MOVE = 14
TOKEN_CASTLING_BASE = 15  # castling = base + (K*8 + Q*4 + k*2 + q*1)
TOKEN_EP_NONE = 31
TOKEN_EP_BASE = 32  # ep file a-h = base + file (0-7)
TOKEN_CLS = 40


# --- Encoding helpers ---

def board_to_tokens_v1(board: chess.Board) -> torch.Tensor:
    """V1: 65 tokens (64 squares + side-to-move)."""
    tokens = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            tokens.append(0)
        else:
            offset = 0 if piece.color else 6
            tokens.append(piece.piece_type + offset)
    tokens.append(13 if board.turn else 14)
    return torch.tensor(tokens, dtype=torch.long)


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


def board_to_tokens(board: chess.Board, seq_length: int = 65) -> torch.Tensor:
    """Convert board to tokens using appropriate scheme based on seq_length."""
    if seq_length == 65:
        return board_to_tokens_v1(board)
    else:
        return board_to_tokens_v2(board)


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
                     mcts_engine=None, num_simulations: int = 800) -> tuple[chess.Move, dict]:
    """Pick engine move using either raw policy or MCTS.

    Returns:
        tuple: (move, stats_dict) where stats_dict contains timing and search info
    """
    start_time = time.time()
    stats = {}

    if mcts_engine is not None:
        # Use MCTS search
        move = mcts_engine.search(board, num_simulations=num_simulations)
        elapsed = time.time() - start_time

        # Get root node stats for display
        stats['time'] = elapsed
        stats['simulations'] = num_simulations
        stats['sims_per_sec'] = num_simulations / elapsed if elapsed > 0 else 0
        stats['batches'] = mcts_engine.evaluator.total_batches
        stats['avg_batch'] = (mcts_engine.evaluator.total_evals /
                              max(1, mcts_engine.evaluator.total_batches))
        stats['eval'] = mcts_engine.last_best_child_q  # Q-value of chosen move

        # Reset evaluator stats for next move
        mcts_engine.evaluator.total_batches = 0
        mcts_engine.evaluator.total_evals = 0
    else:
        # Use raw policy (single forward pass)
        seq_length: int = model.seq_length  # type: ignore[assignment]
        tokens = board_to_tokens(board, seq_length).unsqueeze(0).to(device)
        mask = legal_move_mask(board).unsqueeze(0).to(device)

        with torch.no_grad():
            # Handle FP16 model on GPU
            if next(model.parameters()).dtype == torch.float16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    policy_logits, value = model(tokens, legal_move_mask=mask)
            else:
                policy_logits, value = model(tokens, legal_move_mask=mask)

        best_index = int(torch.argmax(policy_logits, dim=1).item())
        move = decode_move(best_index, board)
        elapsed = time.time() - start_time

        stats['time'] = elapsed
        stats['eval'] = value.item()
        stats['simulations'] = None

    return move, stats


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


def main():
    parser = argparse.ArgumentParser(description="Play chess against ChessTransformer")
    parser.add_argument("checkpoint", type=Path, nargs="?",
                        default=Path("models/guofish_25.8M_51.9p.pt"),
                        help="Path to model checkpoint")
    parser.add_argument("--mcts", action="store_true",
                        help="Use MCTS search instead of raw policy")
    parser.add_argument("--simulations", type=int, default=800,
                        help="Number of MCTS simulations per move (default: 800)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of MCTS worker threads (default: auto)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: {args.checkpoint} not found")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    # Initialize MCTS if requested
    mcts_engine = None
    if args.mcts:
        from mcts import ParallelMCTS
        # Let ParallelMCTS auto-tune workers unless explicitly specified
        mcts_engine = ParallelMCTS(model, device, num_workers=args.workers)
        print(f"MCTS enabled: {args.simulations} simulations, {mcts_engine.num_workers} workers, "
              f"batch size {mcts_engine.evaluator.min_batch_size}-{mcts_engine.evaluator.max_batch_size}")

    # Ask the user which side to play
    while True:
        side_input = input("Play as white or black? [w/b]: ").strip().lower()
        if side_input in ("w", "white"):
            human_side = chess.WHITE
            break
        if side_input in ("b", "black"):
            human_side = chess.BLACK
            break
        print("Please enter 'w' or 'b'.")

    board = chess.Board()
    print("\nStarting game. Enter moves in SAN (e.g. e4, Nf3, O-O). Type 'quit' to stop.\n")

    # If the engine plays white, make its opening move first
    if human_side == chess.BLACK:
        move, stats = pick_engine_move(model, board, device, mcts_engine, args.simulations)
        print(f"Engine plays: {board.san(move)}  [{format_engine_stats(stats)}]\n")
        board.push(move)

    while True:
        raw = input("Your move: ").strip()
        if raw.lower() in ("quit", "exit"):
            print("Quitting.")
            return

        try:
            human_move = board.parse_san(raw)
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError) as e:
            print(f"Could not parse '{raw}' as a legal move ({type(e).__name__}). Try again.")
            continue

        board.push(human_move)

        if board.is_game_over():
            outcome = board.outcome()
            print(f"Game over: {outcome.result() if outcome else 'unknown'}")
            if board.is_checkmate():
                print("You win by checkmate!")
            return

        engine_move, stats = pick_engine_move(model, board, device, mcts_engine, args.simulations)
        print(f"Engine plays: {board.san(engine_move)}  [{format_engine_stats(stats)}]\n")
        board.push(engine_move)

        if board.is_game_over():
            outcome = board.outcome()
            print(f"Game over: {outcome.result() if outcome else 'unknown'}")
            if board.is_checkmate():
                print("Engine wins by checkmate!")
            return


if __name__ == "__main__":
    main()
