"""On-disk record contracts: v1 (379 B, the 30M/90M corpora) and v2 (387 B, §6.1).

v1 is imported from data/multiPV/record_format.py, not copied. v2 is defined
here. A shard directory's manifest names its dtype the way Pass B writes it
(`record_dtype` = [[name, str(dtype)], ...]); `dtype_from_manifest` maps that
back to one of these two and refuses anything else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
_MPV = REPO / "data" / "multiPV"
if str(_MPV) not in sys.path:
    sys.path.insert(0, str(_MPV))

from labels import MAX_LEGAL, MAX_PV  # noqa: E402
from record_format import RECORD_DTYPE as V1_DTYPE  # noqa: E402

V2_DTYPE = np.dtype([
    ("tokens", np.int8, 68),          # raw orientation
    ("value", np.float16),            # White-POV target
    ("value_cp", np.int16),
    ("value_depth", np.uint8),        # depth of the value block
    ("has_policy", np.uint8),
    ("n_pv", np.uint8),
    ("pv_idx", np.int16, MAX_PV),
    ("pv_prob", np.float16, MAX_PV),
    ("pv_score", np.int16, MAX_PV),   # exact stm-relative policy score (H5)
    ("hard_move", np.int16),          # SF best move from*64+to; -1 if absent
    ("origin", np.uint8),             # 0 root, k = k plies along a PV
    ("src_line", np.uint32),          # dump line number
    ("n_legal", np.uint8),
    ("legal_idx", np.int16, MAX_LEGAL),
], align=False)

if V1_DTYPE.itemsize != 379 or V2_DTYPE.itemsize != 387:
    raise RuntimeError(f"record sizes drifted: v1 {V1_DTYPE.itemsize}, "
                       f"v2 {V2_DTYPE.itemsize}")

FORMATS = {"v1": V1_DTYPE, "v2": V2_DTYPE}


def dtype_descr(dt: np.dtype) -> list:
    """The manifest spelling Pass B uses: [[name, str(field dtype)], ...]."""
    return [[n, str(dt.fields[n][0])] for n in dt.names]


def format_of(dt_descr: list) -> str:
    for name, dt in FORMATS.items():
        if [list(x) for x in dt_descr] == dtype_descr(dt):
            return name
    raise ValueError(f"manifest record_dtype matches neither v1 nor v2: {dt_descr}")


def dtype_from_manifest(manifest: dict) -> tuple[str, np.dtype]:
    if "record_dtype" not in manifest:
        raise KeyError("manifest has no record_dtype")
    name = format_of(manifest["record_dtype"])
    dt = FORMATS[name]
    if int(manifest["record_size_bytes"]) != dt.itemsize:
        raise ValueError(f"manifest record_size_bytes {manifest['record_size_bytes']}"
                         f" != {name} itemsize {dt.itemsize}")
    return name, dt
