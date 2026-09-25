"""Stratum codes, one uint16 per record (§6.6).

    bits 0-1  bucket    le5 | 6_14 | 15_27 | ge28       (pieces on the board)
    bits 2-3  label     multipv | hard_only | value_only
    bits 4-5  value     exact_zero | mate | middle      (value_cp == 0, |value_cp| >= 29000)
    bits 6-7  material  level | ahead | compensated
    bit  8    origin    root | derived

Material M = White - Black with P=1, N=B=3, R=5, Q=9. `level` is |M| <= 1;
otherwise `ahead` when the eval favours the side up material, else
`compensated` (eval equal or favouring the side down material).

Every code is invariant under the colour mirror (all five fields depend on
|M|, |value_cp| or sign(M)*sign(value_cp)), so strata computed on raw records
hold for mirrored samples too. The sidecar JSON carries DEFINITION_HASH.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from training.v6.config.schema import STRATA_FIELDS
from training.v6.data.formats import _MPV  # noqa: F401  (puts data/multiPV on sys.path)
from labels import VALUE_MATE_MIN  # noqa: E402

LAYOUT = {"bucket": (0, 2), "label": (2, 2), "value": (4, 2), "material": (6, 2), "origin": (8, 1)}
BUCKET_LOWER_EDGES = (6, 15, 28)          # le5 < 6 <= 6_14 < 15 <= 15_27 < 28 <= ge28
PIECE_VALUES = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}

DEFINITION = {
    "version": 1,
    "layout": LAYOUT,
    "names": STRATA_FIELDS,
    "bucket_lower_edges": BUCKET_LOWER_EDGES,
    "piece_values": PIECE_VALUES,
    "value_mate_min": VALUE_MATE_MIN,
    "material_level_max": 1,
    "label": "has_policy -> multipv; else hard_move >= 0 -> hard_only; else value_only",
    "material": "|M|<=1 level; value_cp*sign(M) > 0 ahead; else compensated",
    "origin": "origin > 0 -> derived",
}
DEFINITION_HASH = hashlib.sha256(json.dumps(DEFINITION, sort_keys=True).encode()).hexdigest()

# token -> signed material (tokens 1-6 White PNBRQK, 7-12 Black)
_MAT = np.zeros(43, dtype=np.int64)
for _i, _p in enumerate("PNBRQK"):
    _MAT[1 + _i] = PIECE_VALUES[_p]
    _MAT[7 + _i] = -PIECE_VALUES[_p]


def compute_strata(rec: np.ndarray) -> np.ndarray:
    """v2-dtype records (raw orientation, v5_68 tokens) -> uint16 codes."""
    sq = rec["tokens"][:, :64].astype(np.int64)
    bucket = np.digitize((sq != 0).sum(1), BUCKET_LOWER_EDGES)
    has_pol = rec["has_policy"] > 0
    label = np.where(has_pol, 0, np.where(rec["hard_move"] >= 0, 1, 2))
    cp = rec["value_cp"].astype(np.int64)
    value = np.where(cp == 0, 0, np.where(np.abs(cp) >= VALUE_MATE_MIN, 1, 2))
    m = _MAT[sq].sum(1)
    material = np.where(np.abs(m) <= 1, 0, np.where(cp * np.sign(m) > 0, 1, 2))
    origin = (rec["origin"] > 0).astype(np.int64)
    code = bucket | (label << 2) | (value << 4) | (material << 6) | (origin << 8)
    return code.astype(np.uint16)


def field_of(codes: np.ndarray, name: str) -> np.ndarray:
    """uint16 in, uint16 out: no upcast of a 150M-record sidecar."""
    shift, width = LAYOUT[name]
    return (np.asarray(codes, dtype=np.uint16) >> np.uint16(shift)) & np.uint16((1 << width) - 1)


def match(codes: np.ndarray, where: dict) -> np.ndarray:
    """Bool mask of records matching every clause; a list value means any-of."""
    mask = np.ones(len(codes), dtype=bool)
    for name, want in where.items():
        vals = want if isinstance(want, (list, tuple)) else [want]
        ids = [STRATA_FIELDS[name].index(v) for v in vals]
        mask &= np.isin(field_of(codes, name), ids)
    return mask


def counts(codes: np.ndarray) -> dict:
    return {name: {v: int((field_of(codes, name) == i).sum())
                   for i, v in enumerate(STRATA_FIELDS[name])}
            for name in LAYOUT}


def load_strata(path, n_records: int, source_manifest_sha256: str | None = None) -> np.ndarray:
    """Load a sidecar and refuse a stale definition, wrong length or wrong source."""
    from pathlib import Path
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["definition_hash"] != DEFINITION_HASH:
        raise ValueError(f"{path}: strata definition {meta['definition_hash'][:12]} "
                         f"!= current {DEFINITION_HASH[:12]}; rebuild it")
    codes = np.load(path, mmap_mode="r")
    if len(codes) != n_records or meta["n_records"] != n_records:
        raise ValueError(f"{path}: {len(codes):,} codes for {n_records:,} records")
    if source_manifest_sha256 is not None and meta["source_manifest_sha256"] != source_manifest_sha256:
        raise ValueError(f"{path}: built from a different manifest")
    return codes
