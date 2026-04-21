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


def _board_to_tokens(board: chess.Board) -> list[int]:
    # piece_map() only iterates occupied squares (16-32 items) vs. iterating
    # all 64 via piece_at — faster for a typical midgame position.
    tokens = [0] * 64
    for square, piece in board.piece_map().items():
        offset = 0 if piece.color else 6
        tokens[square] = piece.piece_type + offset
    # 65th token: 13 = White to move, 14 = Black to move
    tokens.append(13 if board.turn else 14)
    return tokens


def parse_game_block(pgn_text: str):
    """Parse one game's raw PGN text.

    Returns (tokens, moves, values) tensors where:
      tokens: int8,  shape (N, 65)  — 64 piece tokens + 1 turn token per position
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

    tokens = torch.tensor(all_tokens, dtype=torch.int8).view(n, 65)
    moves = torch.tensor(move_indices, dtype=torch.long)
    values = torch.full((n,), value, dtype=torch.int8)
    return tokens, moves, values
