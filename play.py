"""Play chess against a ChessTransformer model.

Supports both regular and FP16-compressed checkpoints.

Usage:
    python play.py                                           # uses default checkpoint
    python play.py models/chess_transformer_25.8M_50.5pct_fp16.pt
"""

import argparse
import math
from pathlib import Path

import chess
import torch
import torch.nn as nn


# --- Model definition ---

class ChessTransformer(nn.Module):
    def __init__(self, vocab_size=15, d_model=512, nhead=8, num_layers=8, dropout=0.1, head_dim=256):
        super().__init__()
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


# --- Loading utilities ---

def load_model(checkpoint_path: Path, device: torch.device) -> ChessTransformer:
    """Load model from checkpoint, handling both FP32 and FP16 weights."""
    print(f"Loading {checkpoint_path} on {device}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        if "val_acc" in ckpt:
            print(f"Model accuracy: {ckpt['val_acc']:.1f}%")
    else:
        state_dict = ckpt

    # Convert FP16 weights back to FP32 for CPU, keep FP16 for CUDA
    if device.type == "cuda":
        # Keep as FP16 on GPU (faster inference, less memory)
        model = ChessTransformer().half().to(device)
        # Convert state dict to FP16 if not already
        state_dict = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
    else:
        # Convert to FP32 on CPU
        model = ChessTransformer().to(device)
        state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


# --- Encoding helpers ---

def board_to_tokens(board: chess.Board) -> torch.Tensor:
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


def pick_engine_move(model: ChessTransformer, board: chess.Board, device: torch.device) -> chess.Move:
    tokens = board_to_tokens(board).unsqueeze(0).to(device)
    mask = legal_move_mask(board).unsqueeze(0).to(device)

    with torch.no_grad():
        # Handle FP16 model on GPU
        if next(model.parameters()).dtype == torch.float16:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                policy_logits, value = model(tokens, legal_move_mask=mask)
        else:
            policy_logits, value = model(tokens, legal_move_mask=mask)

    best_index = int(torch.argmax(policy_logits, dim=1).item())
    print(f"[engine eval: {value.item():+.3f}]")
    return decode_move(best_index, board)


def main():
    parser = argparse.ArgumentParser(description="Play chess against ChessTransformer")
    parser.add_argument("checkpoint", type=Path, nargs="?",
                        default=Path("models/chess_transformer_25.8M_50.5pct.pt"),
                        help="Path to model checkpoint")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: {args.checkpoint} not found")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

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
        move = pick_engine_move(model, board, device)
        print(f"Engine plays: {board.san(move)}\n")
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

        engine_move = pick_engine_move(model, board, device)
        print(f"Engine plays: {board.san(engine_move)}\n")
        board.push(engine_move)

        if board.is_game_over():
            outcome = board.outcome()
            print(f"Game over: {outcome.result() if outcome else 'unknown'}")
            if board.is_checkmate():
                print("Engine wins by checkmate!")
            return


if __name__ == "__main__":
    main()
