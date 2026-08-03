"""PHASE 3 - map-style Dataset over the Pass B shards.

Map-style, NOT IterableDataset: shards are fixed-width memmaps, so a global
index is just (shard, row) arithmetic and DataLoader can shuffle freely.

`__getitem__` does no JSON parsing and no python-chess calls - it indexes,
unpacks, and returns. Everything expensive was done in Pass B.

Three things happen at COLLATE time rather than conversion time, so none of
them require re-running Pass B:

  temperature   `pv_score` holds the raw clamped stm-relative scores, so a
                different T re-derives the policy target. Because pv_idx is
                stored per-ENTRY (not per-4096-index), re-softmaxing and
                accumulating with += reproduces the converter exactly,
                including underpromotion merges.
  value scale   `value_cp` holds the raw White-POV score; revalue() re-squashes.
  colour mirror exact involution on tokens/value/policy, 50% of samples.

Usage:
    ds = MultiPVDataset("data/processed/multipv", split="train")
    ds = MultiPVDataset(..., subset=5_000_000)      # first N, deterministic
    dl = DataLoader(ds, collate_fn=MultiPVCollate(mirror_prob=0.5), ...)
    python data/multiPV/dataset.py --benchmark
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from labels import (  # noqa: E402
    DEFAULT_EPSILON, VALUE_CLAMP, VALUE_CP_CLIP, VALUE_MATE, VALUE_MATE_MIN,
    VALUE_SCALE,
)
from mirror import POLICY_PERM, TOKEN_PERM, VOCAB_MIRROR  # noqa: E402
from record_format import POLICY_SIZE, RECORD_DTYPE  # noqa: E402

_VOCAB_MIRROR_T = torch.from_numpy(VOCAB_MIRROR)
_TOKEN_PERM_T = torch.from_numpy(TOKEN_PERM)
_POLICY_PERM_T = torch.from_numpy(POLICY_PERM)


class MultiPVDataset(Dataset):
    """Fixed-width shards -> tensors. One memmap per shard, opened lazily.

    Lazy opening matters: DataLoader workers are forked/spawned after __init__,
    and a memmap created in the parent does not survive cleanly into workers on
    Windows spawn. Each worker opens its own handle on first touch.
    """

    def __init__(self, shard_dir, split: str = "train",
                 epsilon: float = DEFAULT_EPSILON,
                 subset: int | None = None,
                 policy_size: int = POLICY_SIZE):
        self.shard_dir = Path(shard_dir)
        self.split = split
        self.epsilon = epsilon
        self.policy_size = policy_size

        self.paths = sorted(self.shard_dir.glob(f"{split}_*.bin"))
        if not self.paths:
            raise FileNotFoundError(
                f"no {split}_*.bin shards in {self.shard_dir}")

        itemsize = RECORD_DTYPE.itemsize
        self.counts = []
        for p in self.paths:
            size = p.stat().st_size
            if size % itemsize:
                raise ValueError(f"{p} is not a whole number of records "
                                 f"({size} % {itemsize} != 0)")
            self.counts.append(size // itemsize)

        self.offsets = np.zeros(len(self.counts) + 1, dtype=np.int64)
        np.cumsum(self.counts, out=self.offsets[1:])
        self._total = int(self.offsets[-1])

        self._length = self._total if subset is None else min(subset, self._total)
        self._maps: list[np.memmap | None] = [None] * len(self.paths)

    def __len__(self) -> int:
        return self._length

    def __getstate__(self):
        """Never let a memmap cross a process boundary.

        Windows uses spawn, so DataLoader pickles the Dataset into each worker.
        numpy pickles a memmap by MATERIALISING it, so a parent that has already
        touched every shard would try to push the whole dataset through the
        spawn pipe - 11 GB, which dies as `OSError: [Errno 22]` partway through.
        Dropping the handles here means each worker re-opens its own lazily,
        which is what _map() was designed for.
        """
        state = self.__dict__.copy()
        state["_maps"] = [None] * len(self.paths)
        return state

    def close(self) -> None:
        """Release the memmaps. On Windows an open memmap blocks deletion of the
        underlying file, so anything that removes shards (tests, tmpdirs) must
        call this first."""
        for i, m in enumerate(self._maps):
            if m is not None:
                m._mmap.close()
                self._maps[i] = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _map(self, i: int) -> np.memmap:
        m = self._maps[i]
        if m is None:
            m = np.memmap(self.paths[i], dtype=RECORD_DTYPE, mode="r")
            self._maps[i] = m
        return m

    def __getitem__(self, i: int):
        if i < 0:
            i += self._length
        if not 0 <= i < self._length:
            raise IndexError(i)
        s = int(np.searchsorted(self.offsets, i, side="right") - 1)
        rec = self._map(s)[i - int(self.offsets[s])]

        n_pv = int(rec["n_pv"])
        n_legal = int(rec["n_legal"])

        policy = np.zeros(self.policy_size, dtype=np.float32)
        legal = np.zeros(self.policy_size, dtype=np.float32)

        if n_legal:
            li = rec["legal_idx"][:n_legal].astype(np.int64)
            legal[li] = 1.0
        has_policy = float(rec["has_policy"])
        if has_policy and n_pv:
            pi = rec["pv_idx"][:n_pv].astype(np.int64)
            pp = rec["pv_prob"][:n_pv].astype(np.float32)
            # += not assignment: two entries may share an index (underpromotion)
            np.add.at(policy, pi, pp)
            if n_legal:
                np.add.at(policy, li, self.epsilon / n_legal)

        return {
            "tokens": torch.from_numpy(rec["tokens"].astype(np.int64)),
            "value": torch.tensor(float(rec["value"]), dtype=torch.float32),
            "value_cp": torch.tensor(int(rec["value_cp"]), dtype=torch.float32),
            "has_policy": torch.tensor(has_policy, dtype=torch.float32),
            "policy": torch.from_numpy(policy),
            "legal_mask": torch.from_numpy(legal),
            "pv_idx": torch.from_numpy(rec["pv_idx"].astype(np.int64)),
            "pv_score": torch.from_numpy(rec["pv_score"].astype(np.float32)),
            "n_pv": torch.tensor(n_pv, dtype=torch.int64),
            "n_legal": torch.tensor(n_legal, dtype=torch.int64),
        }


def retemperature(batch, temperature: float, epsilon: float = DEFAULT_EPSILON,
                  policy_size: int = POLICY_SIZE) -> torch.Tensor:
    """Rebuild the dense policy target at a new T from the stored raw scores.

    Reproduces build_policy_target exactly: softmax over the stored entries,
    scale to (1-eps), accumulate with += (so repeated indices merge), then
    spread eps over the legal moves.
    """
    pv_idx, pv_score = batch["pv_idx"], batch["pv_score"]
    n_pv, has_policy = batch["n_pv"], batch["has_policy"]
    legal_mask = batch["legal_mask"]
    B = pv_idx.shape[0]

    ar = torch.arange(pv_idx.shape[1], device=pv_idx.device).unsqueeze(0)
    valid = ar < n_pv.unsqueeze(1)

    logits = pv_score / temperature
    logits = logits.masked_fill(~valid, float("-inf"))
    w = torch.softmax(logits, dim=1)
    w = torch.nan_to_num(w, nan=0.0) * (1.0 - epsilon)

    out = torch.zeros(B, policy_size, dtype=torch.float32, device=pv_idx.device)
    out.scatter_add_(1, pv_idx.clamp(0, policy_size - 1), w)
    out *= has_policy.unsqueeze(1)

    n_legal = legal_mask.sum(dim=1, keepdim=True).clamp(min=1)
    out = out + legal_mask * (epsilon / n_legal) * has_policy.unsqueeze(1)
    return out


def revalue(batch, scale: float = VALUE_SCALE) -> torch.Tensor:
    """Rebuild the value target at a different tanh scale, from the stored raw
    cp. The value-scale analogue of `retemperature`: the scale can be swept
    without re-running Pass B.

    The shipped scale (290.6806) is a calibration and is not expected to move,
    but ~20% of the selected subset is mates pinned at +-0.995, so anyone
    tempted to refit should be able to see the consequences without a 4-hour
    reconversion. Mates are identified by |raw| >= VALUE_MATE_MIN and stay
    pinned whatever the scale; their distance is recoverable but unused.
    """
    raw = batch["value_cp"]
    is_mate = raw.abs() >= VALUE_MATE_MIN
    v = torch.tanh(raw.clamp(-VALUE_CP_CLIP, VALUE_CP_CLIP) / scale)
    v = torch.where(is_mate, torch.sign(raw) * VALUE_MATE, v)
    return v.clamp(-VALUE_CLAMP, VALUE_CLAMP)


def color_mirror(batch, mask: torch.Tensor):
    """Apply the exact colour mirror to the rows selected by `mask` (B,) bool.

    Returns a NEW dict; the underlying memmap-backed tensors are never written.
    Every field that carries orientation is transformed together - tokens,
    value, value_cp, the dense policy and legal mask, and the sparse pv_idx -
    so a mirrored batch stays self-consistent and `retemperature` still
    reproduces the dense target from the sparse entries.
    """
    if not bool(mask.any()):
        return batch
    out = dict(batch)
    m = mask.to(batch["tokens"].device)
    mf = m.to(torch.float32)

    perm = _TOKEN_PERM_T.to(batch["tokens"].device)
    lut = _VOCAB_MIRROR_T.to(batch["tokens"].device)
    ppm = _POLICY_PERM_T.to(batch["tokens"].device)

    out["tokens"] = torch.where(
        m.unsqueeze(1), lut[batch["tokens"][:, perm]], batch["tokens"])

    sign = 1.0 - 2.0 * mf
    for k in ("value", "value_cp"):
        if k in batch:
            out[k] = batch[k] * sign

    for k in ("policy", "legal_mask"):
        if k in batch:
            out[k] = torch.where(m.unsqueeze(1), batch[k][:, ppm], batch[k])

    if "pv_idx" in batch:
        idx = batch["pv_idx"]
        # Padded slots hold 0, which is a real index under the permutation.
        # Keep them at 0 so a mirrored batch is byte-comparable with an
        # unmirrored one and the involution test is exact.
        valid = (torch.arange(idx.shape[1], device=idx.device).unsqueeze(0)
                 < batch["n_pv"].unsqueeze(1))
        mirrored = torch.where(valid, ppm[idx.clamp(0, POLICY_SIZE - 1)], idx)
        out["pv_idx"] = torch.where(m.unsqueeze(1), mirrored, idx)

    return out


class MultiPVCollate:
    """Default-collate plus the optional collate-time transforms.

    Order is deliberate: temperature rebuilds the dense policy from the stored
    raw scores, so it must run BEFORE the mirror permutes both. The value scale
    runs after the mirror, reading the already-negated value_cp - tanh is odd,
    so the two compose correctly in either order, but this way `value` is
    written exactly once.
    """

    def __init__(self, mirror_prob: float = 0.5,
                 temperature: float | None = None,
                 value_scale: float | None = None,
                 epsilon: float = DEFAULT_EPSILON,
                 seed: int | None = None):
        self.mirror_prob = mirror_prob
        self.temperature = temperature
        self.value_scale = value_scale
        self.epsilon = epsilon
        # Per-worker generator: DataLoader workers must not draw the same
        # mirror pattern, and the default global RNG is re-seeded per worker
        # by torch anyway. An explicit generator makes a run reproducible.
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, samples):
        from torch.utils.data._utils.collate import default_collate
        batch = default_collate(samples)
        if self.temperature is not None:
            batch["policy"] = retemperature(
                batch, self.temperature, self.epsilon)
        if self.mirror_prob > 0:
            mask = torch.rand(batch["tokens"].shape[0],
                              generator=self.generator) < self.mirror_prob
            batch = color_mirror(batch, mask)
        if self.value_scale is not None:
            batch["value"] = revalue(batch, self.value_scale)
        return batch


def _benchmark(args) -> int:
    ds = MultiPVDataset(args.shard_dir, split=args.split, subset=args.subset)
    collate = MultiPVCollate(mirror_prob=args.mirror_prob)
    print(f"dataset: {len(ds):,} records from {len(ds.paths)} {args.split} shards")
    print(f"collate: mirror_prob={args.mirror_prob}")

    print(f"\n{'workers':>8} {'batches':>9} {'samples':>11} {'sec':>8} {'samples/s':>12}")
    print("-" * 54)
    results = {}
    for w in args.workers:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=w, drop_last=True, collate_fn=collate,
                        persistent_workers=bool(w), pin_memory=False,
                        prefetch_factor=4 if w else None)
        it = iter(dl)
        for _ in range(args.warmup):        # exclude worker spin-up
            next(it)
        t0 = time.time()
        n = 0
        for _ in range(args.batches):
            b = next(it)
            n += b["tokens"].shape[0]
        el = time.time() - t0
        rate = n / el
        results[w] = rate
        print(f"{w:>8} {args.batches:>9} {n:>11,} {el:>8.2f} {rate:>12,.0f}")
        del it, dl

    best = max(results.values())
    print("-" * 54)
    print(f"best: {best:,.0f} samples/s")
    print(f"target >20,000/s : {'PASS' if best > 20000 else 'FAIL'}")
    print(f"model needs ~6,000/s: {'PASS' if best > 6000 else 'FAIL'} "
          f"({best/6000:.1f}x headroom)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path,
                    default=_HERE.parents[1] / "data" / "processed" / "multipv")
    ap.add_argument("--split", default="train")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 6, 8])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--mirror-prob", type=float, default=0.5,
                    help="colour-mirror augmentation rate; 0 disables")
    args = ap.parse_args()

    if args.benchmark:
        return _benchmark(args)

    ds = MultiPVDataset(args.shard_dir, split=args.split, subset=args.subset)
    print(f"{len(ds):,} records across {len(ds.paths)} shards")
    s = ds[0]
    for k, v in s.items():
        print(f"  {k:<12} {tuple(v.shape)} {v.dtype}")
    print(f"  policy sums to {float(s['policy'].sum()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
