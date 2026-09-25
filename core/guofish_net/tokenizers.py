"""Reference tokenizers from a chess.Board, one per input scheme (§5.4).

`v5_68` is data/pgn_parallel._board_to_tokens itself - reused, not copied.
`canonical_65` is written here from python-chess directly, independent of the
worker-side token transform it is the reference for.
"""
from __future__ import annotations

import chess
import numpy as np

from data.pgn_parallel import _board_to_tokens

# canonical_65 vocabulary
C_EMPTY = 0
C_OURS = 0          # + piece_type (1..6)
C_THEIRS = 6        # + piece_type (7..12)
C_OUR_CASTLE_ROOK = 13
C_THEIR_CASTLE_ROOK = 14
C_EP_TARGET = 15
C_CLS = 16


def tokens_v5_68(board: chess.Board) -> np.ndarray:
    return np.asarray(_board_to_tokens(board), dtype=np.int8)


def tokens_canonical_65(board: chess.Board) -> np.ndarray:
    """Side-to-move canonical: when Black moves, square s -> s^56 and colours
    swap. Castling rights ride on the rook token; the ep target square is
    marked only when an en-passant capture is legal."""
    flip = 0 if board.turn == chess.WHITE else 56
    castle = board.clean_castling_rights()
    out = np.zeros(65, dtype=np.int8)
    for sq, pc in board.piece_map().items():
        ours = pc.color == board.turn
        if pc.piece_type == chess.ROOK and castle & chess.BB_SQUARES[sq]:
            tok = C_OUR_CASTLE_ROOK if ours else C_THEIR_CASTLE_ROOK
        else:
            tok = pc.piece_type + (C_OURS if ours else C_THEIRS)
        out[sq ^ flip] = tok
    if board.has_legal_en_passant():
        out[board.ep_square ^ flip] = C_EP_TARGET
    out[64] = C_CLS
    return out


def board_from_v5_tokens(tokens) -> chess.Board:
    """Decode 68 v5 tokens to a Board (move counters are not encoded)."""
    t = [int(x) for x in tokens]
    if len(t) != 68 or t[67] != 40 or t[64] not in (13, 14):
        raise ValueError(f"not a v5_68 token row: {t}")
    b = chess.Board(None)
    for sq in range(64):
        tok = t[sq]
        if tok:
            if not 1 <= tok <= 12:
                raise ValueError(f"square {sq} holds non-piece token {tok}")
            b.set_piece_at(sq, chess.Piece(tok - 6 if tok > 6 else tok,
                                           chess.BLACK if tok > 6 else chess.WHITE))
    b.turn = t[64] == 13
    bits = t[65] - 15
    if not 0 <= bits < 16:
        raise ValueError(f"castling token {t[65]} out of range")
    b.set_castling_fen("".join(c for c, m in zip("KQkq", (8, 4, 2, 1)) if bits & m) or "-")
    if t[66] != 31:
        f = t[66] - 32
        if not 0 <= f < 8:
            raise ValueError(f"ep token {t[66]} out of range")
        b.ep_square = chess.square(f, 5 if b.turn else 2)
    return b
