"""Exact colour-mirror transform for the 68-token encoding and the 4096 policy.

The model uses ABSOLUTE board orientation with a White-POV value head, so it
does not get colour symmetry for free the way a side-to-move-relative encoding
does - it has to spend capacity learning that a position and its mirror are the
same position. This transform hands it that instead, and simultaneously
symmetrises a corpus that is genuinely skewed (mates in the selected subset run
59.9% positive).

The mapping is exact, not approximate:

    squares    sq ^ 56                 (rank flip; file preserved)
    pieces     white <-> black         (token +-6)
    stm        white <-> black
    castling   KQ <-> kq               (bit permutation, 16-entry LUT)
    en passant unchanged               (the token stores the FILE only)
    CLS        unchanged
    value      negated
    policy     (from^56)*64 + (to^56)

Every piece of it is an involution, so `mirror(mirror(x)) == x` exactly - which
is what the unit test pins, and what makes the 50%-of-samples application at
collate time unbiased.

Castling caveat: the transform maps rights to rights, which is correct for
standard chess where the rook/king files are fixed. Chess960 positions are
skipped in Pass B, so no position reaching here has castling rights that the
rank flip would misplace.
"""
from __future__ import annotations

import numpy as np

VOCAB_SIZE = 43
TOKEN_WHITE_TO_MOVE = 13
TOKEN_BLACK_TO_MOVE = 14
TOKEN_CASTLING_BASE = 15        # base + (K*8 + Q*4 + k*2 + q*1)
TOKEN_EP_NONE = 31
TOKEN_EP_BASE = 32
TOKEN_CLS = 40

SEQ_LENGTH = 68
POLICY_SIZE = 4096


def _castling_mirror(bits: int) -> int:
    """K,Q,k,q = 8,4,2,1  ->  swap the white pair with the black pair."""
    return ((bits & 0b0011) << 2) | ((bits & 0b1100) >> 2)


def _build_vocab_mirror() -> np.ndarray:
    """token -> colour-swapped token. Identity outside the affected ranges."""
    lut = np.arange(VOCAB_SIZE, dtype=np.int64)
    for pt in range(1, 7):              # 1-6 white pieces, 7-12 black
        lut[pt] = pt + 6
        lut[pt + 6] = pt
    lut[TOKEN_WHITE_TO_MOVE] = TOKEN_BLACK_TO_MOVE
    lut[TOKEN_BLACK_TO_MOVE] = TOKEN_WHITE_TO_MOVE
    for b in range(16):
        lut[TOKEN_CASTLING_BASE + b] = TOKEN_CASTLING_BASE + _castling_mirror(b)
    # TOKEN_EP_NONE, the eight ep-file tokens and CLS are all fixed points.
    return lut


def _build_token_perm() -> np.ndarray:
    """Position permutation over the 68 slots: squares flip rank, tail is fixed."""
    perm = np.arange(SEQ_LENGTH, dtype=np.int64)
    perm[:64] = np.arange(64, dtype=np.int64) ^ 56
    return perm


def _build_policy_perm() -> np.ndarray:
    """from*64 + to  ->  (from^56)*64 + (to^56)."""
    i = np.arange(POLICY_SIZE, dtype=np.int64)
    return (((i >> 6) ^ 56) << 6) | ((i & 63) ^ 56)


VOCAB_MIRROR = _build_vocab_mirror()
TOKEN_PERM = _build_token_perm()
POLICY_PERM = _build_policy_perm()


def mirror_tokens_np(tokens: np.ndarray) -> np.ndarray:
    """(..., 68) -> mirrored. Reads the source at the flipped square, then
    recolours - doing it in the other order gives the same answer, but this way
    round-trips through a single gather."""
    return VOCAB_MIRROR[tokens[..., TOKEN_PERM]]


def mirror_policy_index_np(idx: np.ndarray) -> np.ndarray:
    return POLICY_PERM[idx]


def mirror_dense_np(dense: np.ndarray) -> np.ndarray:
    """POLICY_PERM is an involution, so gathering with it both applies and
    inverts the permutation - `out[perm[i]] = x[i]` is the same array as
    `out = x[perm]`."""
    return dense[..., POLICY_PERM]
