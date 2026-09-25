"""Fixed-width shard reader for v1 and v2 records (§6.1).

`ShardSet.read(idx)` returns records in the v2 dtype whatever the on-disk
format. v1 records report hard_move = -1 and origin = 0, as the doc specifies,
plus value_depth = 0 and src_line = UNKNOWN_SRC_LINE, because v1 stores
neither (a real value depth is >= 20, so 0 cannot be confused with one).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.v6.data.formats import V1_DTYPE, V2_DTYPE, dtype_from_manifest

UNKNOWN_SRC_LINE = np.uint32(0xFFFFFFFF)


def upcast_v1(raw: np.ndarray) -> np.ndarray:
    out = np.zeros(len(raw), dtype=V2_DTYPE)
    for name in V1_DTYPE.names:
        if name == "pv_score":
            # v1 stored an int score as float16, so every value is an integer
            # (float16 rounds large ints to multiples of 2/4/8, never to a
            # fraction). The int16 is therefore the stored value exactly.
            s = raw[name].astype(np.int16)
            if not np.array_equal(s.astype(np.float16), raw[name]):
                raise ValueError("v1 pv_score holds a non-integral value")
            out[name] = s
        else:
            out[name] = raw[name]
    out["hard_move"] = -1
    out["origin"] = 0
    out["value_depth"] = 0
    out["src_line"] = UNKNOWN_SRC_LINE
    return out


class ShardSet:
    """One split's `{split}_*.bin` shards, concatenated in name order.

    Memmaps open lazily and never cross a process boundary (Windows spawn
    pickles a memmap by materialising it)."""

    def __init__(self, shard_dir, split: str, manifest=None):
        self.shard_dir = Path(shard_dir)
        self.split = split
        self.manifest_path = Path(manifest) if manifest else self.shard_dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        self.format, self.dtype = dtype_from_manifest(self.manifest)
        self.paths = sorted(self.shard_dir.glob(f"{split}_*.bin"))
        if not self.paths:
            raise FileNotFoundError(f"no {split}_*.bin shards in {self.shard_dir}")
        counts = []
        for p in self.paths:
            size = p.stat().st_size
            if size % self.dtype.itemsize:
                raise ValueError(f"{p} is not a whole number of {self.format} records")
            counts.append(size // self.dtype.itemsize)
        self.offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        np.cumsum(counts, out=self.offsets[1:])
        self._maps = [None] * len(self.paths)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"] = [None] * len(self.paths)
        return state

    def close(self) -> None:
        for i, m in enumerate(self._maps):
            if m is not None:
                m._mmap.close()
                self._maps[i] = None

    def _map(self, i: int) -> np.memmap:
        if self._maps[i] is None:
            self._maps[i] = np.memmap(self.paths[i], dtype=self.dtype, mode="r")
        return self._maps[i]

    def read_raw(self, idx) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError(f"record index out of [0, {len(self)})")
        shard = np.searchsorted(self.offsets, idx, side="right") - 1
        out = np.empty(len(idx), dtype=self.dtype)
        for s in np.unique(shard):
            sel = shard == s
            out[sel] = self._map(int(s))[idx[sel] - self.offsets[s]]
        return out

    def read(self, idx) -> np.ndarray:
        raw = self.read_raw(idx)
        return raw if self.format == "v2" else upcast_v1(raw)
