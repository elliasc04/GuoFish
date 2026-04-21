"""Interactive play script against a trained ChessTransformer.

Just hit Run. You'll be prompted for your side, then alternate moves in
Standard Algebraic Notation (e.g. e4, Nf3, O-O, exd5, e8=Q). Type 'quit' to stop.
"""

import math
from pathlib import Path

import chess
import torch
import torch.nn as nn


CHECKPOINT_PATH = Path("models/chess_transformer_25.8M_50.5pct.pt")


# --- Model definition (must match training) ---

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
        # Value head: mean pool all 65 tokens
        pooled_state = x.mean(dim=1)
        value = self.value_head(pooled_state).squeeze(-1)
        # Policy head: only use first 64 tokens (board squares)
        x_squares = x[:, :64, :]
        from_feats = self.from_proj(x_squares)
        to_feats = self.to_proj(x_squares)
        policy_logits = torch.bmm(from_feats, to_feats.transpose(1, 2)) * self.logit_scale
        policy_logits = policy_logits.view(x.size(0), 4096)
        if legal_move_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_move_mask, float('-inf'))
        return policy_logits, value


# --- Encoding helpers (must match preprocessing) ---

def board_to_tokens(board: chess.Board) -> torch.Tensor:
    tokens = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            tokens.append(0)
        else:
            offset = 0 if piece.color else 6
            tokens.append(piece.piece_type + offset)
    # 65th token: 13 = White to move, 14 = Black to move
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
        policy_logits, value = model(tokens, legal_move_mask=mask)
    best_index = int(torch.argmax(policy_logits, dim=1).item())
    print(f"[engine eval: {value.item():+.3f}]")
    return decode_move(best_index, board)


def announce_engine_move(move: chess.Move, board_before_push: chess.Board) -> None:
    # Report the engine's move in SAN so it's readable alongside the user's input.
    san = board_before_push.san(move)
    print(f"Engine plays: {san}\n")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {CHECKPOINT_PATH} on {device}")
    model = ChessTransformer().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Ask the user which side to play.
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
    print("\nStarting game. Enter moves in Standard Algebraic Notation (e.g. e4, Nf3, O-O). Type 'quit' to stop.\n")

    # If the engine plays white, make its opening move first.
    if human_side == chess.BLACK:
        move = pick_engine_move(model, board, device)
        announce_engine_move(move, board)
        board.push(move)

    while True:
        raw = input("Your move: ").strip()
        if raw.lower() in ("quit", "exit"):
            print("Quitting.")
            return
        try:
            # parse_san accepts SAN like e4, Nf3, O-O, exd5, e8=Q, Qxh7#.
            # It validates legality against the current position.
            human_move = board.parse_san(raw)
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError) as e:
            print(f"Could not parse '{raw}' as a legal move ({type(e).__name__}). Try again.")
            continue
        board.push(human_move)

        # Check if human's move ended the game
        if board.is_game_over():
            outcome = board.outcome()
            print(f"Game over: {outcome.result() if outcome else 'unknown'}")
            if board.is_checkmate():
                print("You win by checkmate!")
            return

        engine_move = pick_engine_move(model, board, device)
        announce_engine_move(engine_move, board)
        board.push(engine_move)

        # Check if engine's move ended the game
        if board.is_game_over():
            outcome = board.outcome()
            print(f"Game over: {outcome.result() if outcome else 'unknown'}")
            if board.is_checkmate():
                print("Engine wins by checkmate!")
            return


if __name__ == "__main__":
    main()
