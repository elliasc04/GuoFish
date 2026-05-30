"""Worker-side parsing for parallel Stockfish-eval CSV extraction.

Mirrors the role of `pgn_parallel` for the (FEN, eval) dataset in
`chessData.csv`. Lives in a .py module so Windows spawn workers can
re-import it (notebook-defined functions aren't importable that way).

Input rows: (fen_str, eval_str)
  fen_str  — standard FEN
  eval_str — signed integer centipawns ("+56", "-10") OR mate score
             ("#+6", "#-19") indicating forced mate in N for/against
             the side whose perspective the eval is taken from.

Convention: evaluations in this dataset are from WHITE's perspective
(positive = white better), which matches the value labels used in the
original game-result training (+1 white-win / -1 black-win / 0 draw).
No side-to-move flip is applied.
"""
import math

import chess
import torch

from data import pgn_parallel


# Centipawn -> value scaling. tanh(cp / CP_SCALE) maps cp to [-1, 1].
# 290.6806 is the empirical scale used by the Lc0 team to convert
# centipawns to expected score (matches their WDL calibration).
CP_SCALE = 290.6806

# Clip raw centipawns before tanh so extreme engine values (e.g. +9000)
# don't all collapse to exactly 1.0 and lose what little gradient remains.
CP_CLIP = 2000

# Forced mate -> saturated value.
MATE_VALUE = 1.0

# Sentinel for the `moves` column on positions with no move label.
# Set CrossEntropyLoss(ignore_index=MOVE_IGNORE) in train.py to skip
# policy loss on these positions while still training the value head.
MOVE_IGNORE = -100


def _parse_eval(eval_str: str) -> float:
    """Convert a chessData.csv evaluation string to a value in [-1, 1]."""
    s = eval_str.strip()
    if not s:
        raise ValueError("empty eval")

    if s[0] == "#":
        # Mate score: "#+N" (mate for white in N) or "#-N" (mate against white).
        # The sign carries the only information we need for the value target.
        sign_char = s[1] if len(s) > 1 else ""
        if sign_char == "-":
            return -MATE_VALUE
        return MATE_VALUE

    cp = int(s)  # handles leading '+' or '-'
    if cp > CP_CLIP:
        cp = CP_CLIP
    elif cp < -CP_CLIP:
        cp = -CP_CLIP
    return math.tanh(cp / CP_SCALE)


def parse_csv_chunk(rows):
    """Parse a chunk of (fen, eval_str) rows.

    Returns (tokens, values) tensors:
      tokens: int8, shape (n, 68)  — same layout as pgn_parallel
      values: float32, shape (n,)  — eval mapped to [-1, 1]
    Returns None if every row in the chunk failed to parse.
    """
    tokens_buf: list[int] = []
    values_buf: list[float] = []

    for row in rows:
        if len(row) < 2:
            continue
        fen, eval_str = row[0], row[1]
        try:
            board = chess.Board(fen)
        except (ValueError, IndexError):
            continue
        try:
            value = _parse_eval(eval_str)
        except (ValueError, TypeError):
            continue
        tokens_buf.extend(pgn_parallel._board_to_tokens(board))
        values_buf.append(value)

    n = len(values_buf)
    if n == 0:
        return None

    tokens = torch.tensor(tokens_buf, dtype=torch.int8).view(n, pgn_parallel.SEQ_LENGTH)
    values = torch.tensor(values_buf, dtype=torch.float32)
    return tokens, values
