"""Mixture sampler (§6.7): stream state is (seed, global sample index).

Micro-batch k holds samples [k*B, (k+1)*B). Its per-group counts are
    c_g(k) = A_g(B*(k+1)) - A_g(B*k),
where A(N) apportions N samples over the group shares by largest remainder
(ties by group order). Every micro-batch therefore has exact counts that
sum to B, and the cumulative count of every group stays within one sample of
share*N ("remainders rotated deterministically"). Group g's j-th sample is
record members_g[perm_{g,p}(j mod n_g)], with pass p = j // n_g and perm a
seeded Feistel permutation of [0, n_g). A new pass gets a fresh permutation.
Everything is closed-form in k: resume needs no stored cursor.

Groups must be disjoint; unmatched records form `rest` with the leftover share.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from training.v6.config.schema import MixtureConfig
from training.v6.data.strata import match

_U = np.uint64
_M64 = (1 << 64) - 1


def splitmix64(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = x + _U(0x9E3779B97F4A7C15)
        z = (z ^ (z >> _U(30))) * _U(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> _U(27))) * _U(0x94D049BB133111EB)
        return z ^ (z >> _U(31))


def mix_key(*parts: int) -> int:
    """Deterministic 64-bit key from integers (seed, stream id, group, pass...)."""
    h = 0x2545F4914F6CDD1D
    for p in parts:
        h = int(splitmix64(np.uint64((h ^ (int(p) & _M64)) & _M64)))
    return h


def feistel_perm(x: np.ndarray, n: int, key: int, rounds: int = 6) -> np.ndarray:
    """Pseudo-random permutation of [0, n) applied to x (int array), via a
    balanced Feistel network on 2h bits (4^h >= n) plus cycle walking."""
    if n <= 0:
        raise ValueError("empty permutation domain")
    if n == 1:
        return np.zeros_like(np.asarray(x, dtype=np.int64))
    h = max(1, math.ceil(math.log2(n) / 2))
    mask = _U((1 << h) - 1)
    keys = [_U(mix_key(key, r)) for r in range(rounds)]

    def enc(v):
        left, right = v >> _U(h), v & mask
        for k in keys:
            left, right = right, left ^ (splitmix64(right ^ k) & mask)
        return (left << _U(h)) | right

    y = enc(np.asarray(x, dtype=np.uint64))
    bad = y >= _U(n)
    while bad.any():
        y[bad] = enc(y[bad])
        bad = y >= _U(n)
    return y.astype(np.int64)


def uniform01(seed_key: int, idx: np.ndarray) -> np.ndarray:
    """Hash (key, index) -> float64 in [0, 1). Used for mirror decisions."""
    u = splitmix64(np.asarray(idx, dtype=np.uint64) ^ _U(seed_key))
    return (u >> _U(11)).astype(np.float64) * (1.0 / (1 << 53))


def apportion(shares: np.ndarray, total: int) -> np.ndarray:
    q = shares * float(total)
    base = np.floor(q).astype(np.int64)
    rem = int(total - base.sum())
    if rem < 0 or rem > len(shares):
        raise ArithmeticError(f"apportion: remainder {rem} out of range")
    order = np.lexsort((np.arange(len(shares)), -(q - base)))
    base[order[:rem]] += 1
    return base


class Mixture:
    """Group definitions resolved against a strata sidecar (or natural)."""

    SAMPLER_STREAM = 1

    def __init__(self, cfg: MixtureConfig, n_records: int, micro_batch: int, seed: int,
                 strata: np.ndarray | None = None, members_dir: Path | None = None):
        self.micro_batch = int(micro_batch)
        self.seed = int(seed)
        self.n_records = int(n_records)
        self.names: list[str] = []
        shares: list[float] = []
        self._members: list[np.ndarray | None] = []     # None = identity
        self._member_paths: list[Path | None] = []
        self.sizes: list[int] = []
        self.rest_excluded = 0          # records in no group when rest has no share

        if cfg.groups == "natural":
            self.names, shares, self.sizes = ["all"], [1.0], [self.n_records]
            self._members, self._member_paths = [None], [None]
        else:
            if strata is None:
                raise ValueError("a grouped mixture needs a strata sidecar")
            if len(strata) != self.n_records:
                raise ValueError(f"strata has {len(strata):,} codes for {self.n_records:,} records")
            masks = [match(strata, g.where) for g in cfg.groups]
            cover = np.sum(masks, axis=0)
            if (cover > 1).any():
                over = {cfg.groups[i].name: int((masks[i] & (cover > 1)).sum())
                        for i in range(len(masks))}
                raise ValueError(f"mixture groups overlap on {int((cover > 1).sum()):,} records "
                                 f"{over}; make the where-clauses disjoint")
            rest = cover == 0
            rest_share = cfg.rest_share
            if rest_share < 1e-12:
                rest_share = 0.0
            items = [(g.name, g.share, masks[i]) for i, g in enumerate(cfg.groups)]
            if rest_share > 0:
                items.append(("rest", rest_share, rest))
            for name, share, mask in items:
                idx = np.flatnonzero(mask)
                if len(idx) == 0:
                    raise ValueError(f"mixture group {name!r} has share {share} but no records")
                self.names.append(name)
                shares.append(share)
                self.sizes.append(len(idx))
                if members_dir is not None:
                    # ponytail: members as uint32 memmaps so workers share page
                    # cache instead of each holding a pickled copy.
                    members_dir.mkdir(parents=True, exist_ok=True)
                    p = members_dir / f"{name}.npy"
                    np.save(p, idx.astype(np.uint32))
                    self._members.append(None)
                    self._member_paths.append(p)
                else:
                    self._members.append(idx.astype(np.int64))
                    self._member_paths.append(None)
            self.rest_excluded = int(rest.sum()) if rest_share == 0 else 0
        s = np.asarray(shares, dtype=np.float64)
        self.shares = s / s.sum()
        self.shares.setflags(write=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_members"] = [None if p is not None else m
                             for m, p in zip(self._members, self._member_paths)]
        return state

    def _member(self, g: int):
        if self._member_paths[g] is not None and self._members[g] is None:
            self._members[g] = np.load(self._member_paths[g], mmap_mode="r")
        return self._members[g]

    def consumed(self, sample_index: int) -> np.ndarray:
        """Per-group samples drawn before `sample_index` (a micro-batch boundary)."""
        if sample_index % self.micro_batch:
            raise ValueError(f"sample index {sample_index} is not on a micro-batch boundary")
        return apportion(self.shares, sample_index)

    def counts(self, k: int) -> np.ndarray:
        B = self.micro_batch
        return apportion(self.shares, B * (k + 1)) - apportion(self.shares, B * k)

    def passes(self, sample_index: int) -> dict:
        c = self.consumed(sample_index)
        return {n: int(c[g] // self.sizes[g]) for g, n in enumerate(self.names)}

    def batch(self, k: int):
        """-> (record_idx[B], sample_idx[B], group_id[B]) for micro-batch k."""
        B = self.micro_batch
        start = apportion(self.shares, B * k)
        cnt = apportion(self.shares, B * (k + 1)) - start
        if (cnt < 0).any():
            raise ArithmeticError(f"negative group count at micro-batch {k}: {cnt}")
        recs, gids = [], []
        for g, c in enumerate(cnt):
            if c == 0:
                continue
            j = start[g] + np.arange(c, dtype=np.int64)
            n = self.sizes[g]
            p, off = j // n, j % n
            pos = np.empty(c, dtype=np.int64)
            for pv in np.unique(p):
                sel = p == pv
                pos[sel] = feistel_perm(off[sel], n,
                                        mix_key(self.seed, self.SAMPLER_STREAM, g, int(pv)))
            mem = self._member(g)
            recs.append(pos if mem is None else np.asarray(mem[pos], dtype=np.int64))
            gids.append(np.full(c, g, dtype=np.int64))
        return (np.concatenate(recs), k * B + np.arange(B, dtype=np.int64),
                np.concatenate(gids))
