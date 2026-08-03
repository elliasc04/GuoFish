"""The on-disk record contract, shared by Pass B, the Dataset, and the tests.

Fixed-width so a shard is a flat numpy memmap: __getitem__ is an index, not a
parse. 377 bytes/record; 30M records ~= 10.5 GiB across all shards.
"""
from __future__ import annotations

import numpy as np

from labels import MAX_LEGAL, MAX_PV, VALUE_MATE_MIN  # noqa: F401

POLICY_SIZE = 4096

RECORD_DTYPE = np.dtype([
    ("tokens", np.int8, 68),         # 64 squares + stm + castling + ep + CLS
    ("value", np.float16),           # White-POV, tanh-squashed, |v| <= 0.995
    ("value_cp", np.int16),          # raw White-POV cp, clipped +-2000;
                                     # |v| >= VALUE_MATE_MIN is a mate, encoded
                                     # sign*(30000 - |mate|) so distance survives
    ("has_policy", np.uint8),        # 0 => value-only; policy loss MUST be masked
    ("n_pv", np.uint8),              # entries used in pv_* (<= MAX_PV)
    ("pv_idx", np.int16, MAX_PV),    # from*64 + to; MAY repeat (underpromotions)
    ("pv_prob", np.float16, MAX_PV), # sums to (1 - epsilon)
    ("pv_score", np.float16, MAX_PV),# raw clamped stm-relative scores, pre-softmax
    ("n_legal", np.uint8),           # legal moves stored (truncated at MAX_LEGAL)
    ("legal_idx", np.int16, MAX_LEGAL),
], align=False)

RECORD_SIZE = RECORD_DTYPE.itemsize


def shard_name(split: str, i: int) -> str:
    return f"{split}_{i:04d}.bin"


def empty_record() -> np.ndarray:
    """A single zeroed record, ready to fill."""
    return np.zeros(1, dtype=RECORD_DTYPE)
