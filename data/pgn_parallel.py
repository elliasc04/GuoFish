"""Worker-side parsing for parallel PGN extraction.

Lives in a .py module because Windows multiprocessing uses spawn semantics:
spawned workers re-import the main module to find target functions, and
notebook-defined functions aren't importable that way. Anything referenced
by `Pool.imap*` must live in a real module.
"""
import io

import chess
import chess.pgn
import torch


# =============================================================================
# Token ID Ranges (vocab_size = 43, sequence length = 68)
# =============================================================================
# Positions 0-63: board squares
#   0      : empty square
#   1-6    : white pieces (P=1, N=2, B=3, R=4, Q=5, K=6)
#   7-12   : black pieces (p=7, n=8, b=9, r=10, q=11, k=12)
#
# Position 64: side to move
#   13     : white to move
#   14     : black to move
#
# Position 65: castling rights (16 combinations, 4-bit encoding)
#   15-30  : K=8, Q=4, k=2, q=1 -> value + 15
#            e.g., KQkq=15 -> token 30, none -> token 15
#
# Position 66: en passant target file
#   31     : no en passant
#   32-39  : files a-h (32=a, 33=b, ..., 39=h)
#
# Position 67: CLS token (state summary for value head)
#   40     : CLS token
#
# Reserved: 41-42
# =============================================================================

VOCAB_SIZE = 43
SEQ_LENGTH = 68

# Token ID constants
TOKEN_WHITE_TO_MOVE = 13
TOKEN_BLACK_TO_MOVE = 14
TOKEN_CASTLING_BASE = 15  # castling = base + (K*8 + Q*4 + k*2 + q*1)
TOKEN_EP_NONE = 31
TOKEN_EP_BASE = 32  # ep file a-h = base + file (0-7)
TOKEN_CLS = 40


def _board_to_tokens(board: chess.Board) -> list[int]:
    """Convert board state to 68 tokens."""
    # Positions 0-63: piece placement
    tokens = [0] * 64
    for square, piece in board.piece_map().items():
        offset = 0 if piece.color else 6
        tokens[square] = piece.piece_type + offset

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

    return tokens


def parse_game_block(pgn_text: str):
    """Parse one game's raw PGN text.

    Returns (tokens, moves, values) tensors where:
      tokens: int8,  shape (N, 68)  — 64 squares + side + castling + ep + CLS
      moves:  long,  shape (N,)     — flat (from*64 + to) index
      values: int8,  shape (N,)     — -1 / 0 / +1 game result
    Returns None if the game has no mainline moves (empty or malformed).
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return None

    result = game.headers.get("Result", "*")
    if result == "1-0":
        value = 1
    elif result == "0-1":
        value = -1
    else:
        value = 0

    board = game.board()
    all_tokens: list[int] = []
    move_indices: list[int] = []
    for move in game.mainline_moves():
        all_tokens.extend(_board_to_tokens(board))
        move_indices.append(move.from_square * 64 + move.to_square)
        board.push(move)

    n = len(move_indices)
    if n == 0:
        return None

    tokens = torch.tensor(all_tokens, dtype=torch.int8).view(n, SEQ_LENGTH)
    moves = torch.tensor(move_indices, dtype=torch.long)
    values = torch.full((n,), value, dtype=torch.int8)
    return tokens, moves, values
