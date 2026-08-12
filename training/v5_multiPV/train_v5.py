#!/usr/bin/env python
"""GuoFish v5 - train the ~10.9M multi-PV student on the Pass B shards.

    python training/v5_multiPV/train_v5.py --config training/v5_multiPV/configs/base.yaml

The value scale, the strata shares and the corpus-wide coverage come from
data/multiPV/manifests/dataset_manifest.json - nothing the manifest states is
hardcoded here, and all of it is carried into every checkpoint. The coverage
actually USED in the loss is measured on the selected records rather than taken
from the manifest; see below.

The four things most likely to silently cost ELO, and what this script does
about them:

  masking      The legal mask is applied to the logits BEFORE log_softmax, with
               -inf, exactly as the engine masks at inference. losses.py owns
               the single implementation; tests/test_losses.py pins it.
  normalisation Every micro-batch divides its policy KL by the SAME constant,
               C = effective_batch * coverage, never by its own realised
               has_policy count. Under accumulation a per-micro-batch count
               gives a mean-of-means and makes the effective policy LR jitter
               with per-batch coverage. Realised vs expected coverage is logged
               per micro-batch and per effective batch, and warns on drift.
  coverage     C scales the policy gradient directly, so coverage is MEASURED
               on the records this run selects, never assumed. The corpus-wide
               figure is only a reference point for the self-check.
  subset       Source order is not exchangeable - every shard sweeps openings to
               endgames - so --subset takes a prefix of a seeded global
               PERMUTATION (select_subset). That makes any subset size
               representative by construction and makes 5M/10M/20M nested.

Validation is DETERMINISTIC: the colour mirror is OFF on the val split
(--val-mirror-prob 0.0). The mirrored half of the world is covered instead by
the mirror-consistency diagnostic, which is exact rather than sampled.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def _shorten_inductor_cache_dir() -> None:
    """Keep torch.compile's cache path under the Windows 260-char MAX_PATH.

    Inductor writes each kernel to

        <root>/triton/0/<52-char hash>/tmp.pid_<50 chars>/<kernel name>.source

    and this model's fused kernel names run to ~90 characters
    (triton_per_fused__to_copy_add_clone_embedding_native_dropout_native_layer_norm_0),
    so ~190 characters are spoken for before the root even starts. The default
    root, %TEMP%/torchinductor_<username>, already sits ON the limit for a
    username with a space in it, and any deeper root sails past it.

    The failure mode is what makes this worth a helper rather than a README
    line: it surfaces as `FileNotFoundError` on a temp file inside the cache,
    not as anything mentioning path length, and it only bites once a graph is
    big enough to generate long kernel names - a two-layer smoke model compiles
    fine. Enabling LongPathsEnabled would also fix it, but that is a
    machine-wide registry change and needs admin.

    Must run BEFORE `import torch`; Inductor resolves the root at import time.
    Respects an existing TORCHINDUCTOR_CACHE_DIR.
    """
    if sys.platform != "win32" or os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        return
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
        os.environ.get("USERPROFILE") or tempfile.gettempdir(), "ti")


_shorten_inductor_cache_dir()

import numpy as np
import torch
# `from torch import _dynamo`, not `import torch._dynamo`: the latter binds the
# name `torch` in whatever scope it runs, so doing it inside a function shadows
# the module-level import and every earlier `torch.` use in that function
# becomes an UnboundLocalError.
from torch import _dynamo
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Sampler

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_DATA_MULTIPV = _PROJECT_ROOT / "data" / "multiPV"
for _p in (str(_HERE), str(_DATA_MULTIPV), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import MultiPVCollate, MultiPVDataset          # noqa: E402
from gates import (SeenUnseenProbe, evaluate_thresholds,     # noqa: E402
                   format_gap_report, format_verdict,
                   head2head_preflight, probe_metrics,
                   relative_gap, run_head2head, score_baseline,
                   write_gate_report)
from losses import (compute_losses, policy_denominator,      # noqa: E402
                    policy_kl_per_sample)
from metrics import (PolicyAccumulator, ValueAccumulator,    # noqa: E402
                     format_mirror_report, format_policy_report,
                     format_value_report, mirror_consistency,
                     policy_topk_correct)
from model_v5 import ModelConfig, build_model                # noqa: E402

DEFAULT_SHARDS = _PROJECT_ROOT / "data" / "processed" / "multipv"
DEFAULT_MANIFEST = _DATA_MULTIPV / "manifests" / "dataset_manifest.json"
DEFAULT_OUT = _PROJECT_ROOT / "models" / "guofish5"


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def log(msg: str = "") -> None:
    print(msg, flush=True)


def fmt_hms(seconds: float) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "n/a"          # int(nan) raises; an unknown ETA is not a crash
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


class JsonlLogger:
    """One line per event, flushed immediately so `tail -f` is live."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", buffering=1, encoding="utf-8")

    def write(self, event: str, **fields) -> None:
        rec = {"event": event, "t": time.time(), **fields}
        self._f.write(json.dumps(rec, default=_jsonable) + "\n")

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def _jsonable(o):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


def sanitize_config(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x if isinstance(x, (int, float, str, bool)) or x is None
                      else str(x) for x in v]
        else:
            out[k] = str(v)
    return out


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_PROJECT_ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def manifest_coverage(m: dict) -> float:
    """Overall has_policy share, summed from the per-bucket histogram."""
    h = m["post_selection_piece_histogram"]
    total = sum(v["count"] for v in h.values())
    with_policy = sum(v["with_policy"] for v in h.values())
    return with_policy / total


def select_subset(total: int, subset, seed: int) -> "np.ndarray":
    """Choose which records a run trains on: a prefix of ONE seeded global
    permutation of the corpus.

    Source order is NOT exchangeable. Every shard is a full sweep from
    opening-like positions to endgames, so where a record sits inside its shard
    predicts almost everything about it - measured across within-shard offsets,
    mean piece count runs 23.6 -> 16.2, has_policy 0.75 -> 0.19, and the mate
    share nearly triples. A prefix of raw source order is therefore only
    representative when it happens to land on a whole number of shards: the
    first 10M records (21.6 shards) match the corpus to 2.5% on every statistic
    measured, while the first 50k (a fragment of one shard) carry 0.380 of their
    mass in >=28-piece positions against a corpus figure of 0.236, and 57.6%
    policy coverage against 40.2%.

    Permuting first makes representativeness hold BY CONSTRUCTION instead of by
    an undocumented accident of shard-boundary alignment, and makes subsets
    NESTED for a fixed seed: 5M is a subset of 10M is a subset of 20M, so a
    scaling study varies quantity with composition held fixed rather than
    confounding the two.

    `seed` is deliberately separate from the training seed: which records a run
    sees should not change because the weight init changed.

    Returned as int32. At 90M records the int64 array this used to return is
    720 MB, and the sampler holds a second one - 1.4 GB of index taken straight
    out of the page cache that the 32 GiB corpus is competing for on a 31.8 GiB
    box. int32 tops out at 2.1e9 records, ~24x the current corpus, and halves
    both. The permutation itself is still drawn at int64 so the draw is
    bit-identical to previous runs at the same seed.
    """
    perm = np.random.default_rng(seed).permutation(total)
    if subset is not None and subset < total:
        perm = perm[:int(subset)]
    if total > np.iinfo(np.int32).max:
        return perm
    return perm.astype(np.int32, copy=False)


def estimate_coverage(dataset, indices, n_samples: int = 200_000,
                      seed: int = 0):
    """Realised has_policy share of the records this run will actually train on.

    Kept even though the permutation should make coverage land on the corpus
    figure every time: that is exactly what makes it a useful free self-check.
    If a subset's measured coverage stops landing near the corpus value, the
    permutation is not being applied.

    Indices are sampled then SORTED, turning scattered reads into a forward
    scan that the OS readahead can follow.
    """
    from record_format import RECORD_DTYPE as _DT

    rng = np.random.default_rng(seed)
    k = min(int(n_samples), len(indices))
    pick = np.sort(indices[rng.choice(len(indices), size=k, replace=False)]
                   if k < len(indices) else indices)

    shard_of = np.searchsorted(dataset.offsets, pick, side="right") - 1
    total = hits = 0
    for sh in np.unique(shard_of):
        sel = pick[shard_of == sh] - int(dataset.offsets[sh])
        m = np.memmap(dataset.paths[sh], dtype=_DT, mode="r")
        try:
            hp = m["has_policy"][sel]
            total += len(sel)
            hits += int(np.count_nonzero(hp))
        finally:
            m._mmap.close()
    return (hits / total if total else float("nan")), total


NO_DECAY_NAMES = ("pos_encoder",)


def param_groups(model, weight_decay: float, verbose: bool = False):
    """Split parameters into decayed and undecayed groups.

    Weight decay is a constraint on the FUNCTION a weight matrix represents.
    Applied to the other parameters it does something else entirely:

      LayerNorm gains  decaying a gain pulls it toward 0, shrinking the
                       activation scale handed to the next block. In a Pre-LN
                       stack that compounds multiplicatively through depth (13
                       LayerNorms here: 2 per layer plus final_norm), so it
                       fights the residual stream rather than regularising it.
      biases           a shift has no capacity to constrain; decaying one just
                       biases it toward zero for no benefit.
      pos_encoder      an embedding-like TABLE, not a linear map. Decaying it
                       erodes positional information - including position 67,
                       the CLS token the entire value head reads.

    `pos_encoder` is excluded BY NAME rather than by shape: it is (1, 68, 384),
    so `ndim >= 2` sweeps it into the decay group. A dimensionality heuristic
    alone gets this one wrong.

    `embedding.weight` IS decayed, following the usual convention for token
    embeddings. With a 43-token vocabulary it is 16,512 params either way.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or any(k in name for k in NO_DECAY_NAMES):
            no_decay.append((name, p))
        else:
            decay.append((name, p))
    if verbose:
        log(f"  weight decay {weight_decay}: "
            f"{len(decay)} tensors / {sum(p.numel() for _, p in decay):,} params")
        log(f"  no decay:     "
            f"{len(no_decay)} tensors / {sum(p.numel() for _, p in no_decay):,} "
            f"params (LayerNorm gains+biases, all biases, pos_encoder)")
    return [
        {"params": [p for _, p in decay], "weight_decay": weight_decay},
        {"params": [p for _, p in no_decay], "weight_decay": 0.0},
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_GPU_UTIL_OK = True


def gpu_utilization() -> float:
    """Percent, or NaN when pynvml is unavailable (it usually is not, on
    Windows, without nvidia-ml-py installed)."""
    global _GPU_UTIL_OK
    if not _GPU_UTIL_OK or not torch.cuda.is_available():
        return float("nan")
    try:
        return float(torch.cuda.utilization())
    except Exception:
        _GPU_UTIL_OK = False
        return float("nan")


# ---------------------------------------------------------------------------
# sampler: reproducible per-epoch permutation with mid-epoch skip
# ---------------------------------------------------------------------------
class ResumableRandomSampler(Sampler):
    """Deterministic per-epoch shuffle that can start partway through.

    Resuming mid-epoch by iterating and discarding batches would re-read every
    skipped record; skipping in the index space costs nothing. `epoch` and
    `skip` are mutated from the main process between epochs - which is safe
    even with persistent_workers, because the sampler is iterated in the main
    process and only the resulting indices are shipped to workers.

    The gather is CHUNKED rather than done in one shot. `self.indices[order]`
    at 90M records materialises a third full-length array on top of `indices`
    and `order`, and at the 20M scale that was 480 MB of transient nobody had
    to think about. Here it is 1.1 GB, competing for the page cache the 32 GiB
    corpus is streaming through on a 31.8 GiB box. Yielding through a 1M-element
    window caps the transient at ~4 MB and produces exactly the same sequence.
    """

    GATHER_CHUNK = 1 << 20

    def __init__(self, indices, seed: int, epoch: int = 0, skip: int = 0):
        self.indices = np.asarray(indices)
        self.n = int(self.indices.shape[0])
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.skip = int(skip)

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        self.epoch = int(epoch)
        self.skip = int(skip)

    def __iter__(self):
        g = np.random.default_rng(self.seed * 1_000_003 + self.epoch)
        order = g.permutation(self.n)
        for lo in range(self.skip, self.n, self.GATHER_CHUNK):
            block = self.indices[order[lo:lo + self.GATHER_CHUNK]]
            yield from block.tolist()

    def __len__(self) -> int:
        return max(0, self.n - self.skip)

    def epoch_head_tail(self, epoch: int, k: int):
        """The first k and last k record indices of `epoch`'s order.

        What makes the seen/unseen probe free: these are two uniform draws from
        the same corpus that differ only in WHEN training reaches them, so no
        record has to be withheld to get an unseen slice. Drawn from the same
        generator __iter__ uses, so they describe the order training will
        actually follow rather than a parallel one.
        """
        k = int(min(k, self.n // 2))
        g = np.random.default_rng(self.seed * 1_000_003 + int(epoch))
        order = g.permutation(self.n)
        head = self.indices[order[:k]].copy()
        tail = self.indices[order[self.n - k:]].copy()
        return head, tail


# ---------------------------------------------------------------------------
# epoch plan: exact denominators, including a short final window
# ---------------------------------------------------------------------------
def plan_epoch(n_records: int, micro_batch: int, accum: int, drop_last: bool):
    """-> (micro_sizes, windows) where windows[i] = (n_micro, n_samples).

    The short final micro-batch (and hence the short final accumulation window)
    is sized explicitly here rather than being discovered at runtime, so no
    window can silently reweight itself by dividing by a full-batch constant it
    did not actually see.
    """
    if drop_last:
        n_micro = n_records // micro_batch
        sizes = [micro_batch] * n_micro
    else:
        n_micro = math.ceil(n_records / micro_batch)
        sizes = [micro_batch] * n_micro
        rem = n_records % micro_batch
        if rem:
            sizes[-1] = rem
    windows = []
    for i in range(0, n_micro, accum):
        chunk = sizes[i:i + accum]
        windows.append((len(chunk), sum(chunk)))
    return sizes, windows


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_loaders(args, value_scale: float):
    # Full corpus; --subset selects a prefix of a seeded global PERMUTATION of
    # it (see select_subset), not a prefix of source order.
    train_ds = MultiPVDataset(args.shards, split="train", subset=None)
    train_idx = select_subset(len(train_ds), args.subset, args.subset_seed)
    collate_train = MultiPVCollate(
        mirror_prob=args.mirror_prob,
        temperature=args.temperature,
        value_scale=value_scale if args.revalue else None)
    sampler = ResumableRandomSampler(train_idx, seed=args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.micro_batch, sampler=sampler,
        num_workers=args.workers, drop_last=args.drop_last,
        collate_fn=collate_train, pin_memory=True,
        persistent_workers=bool(args.workers),
        prefetch_factor=args.prefetch_factor if args.workers else None)

    # Validation is deterministic: mirror OFF, no shuffle, fixed record order.
    collate_val = MultiPVCollate(
        mirror_prob=args.val_mirror_prob,
        temperature=args.temperature,
        value_scale=value_scale if args.revalue else None)

    val_fixed_ds = MultiPVDataset(args.shards, split="val",
                                  subset=args.val_subset)
    val_fixed_loader = DataLoader(
        val_fixed_ds, batch_size=args.val_batch, shuffle=False,
        num_workers=args.val_workers, drop_last=False, collate_fn=collate_val,
        pin_memory=True, persistent_workers=bool(args.val_workers),
        prefetch_factor=args.prefetch_factor if args.val_workers else None)

    val_full_ds = MultiPVDataset(args.shards, split="val")
    return (train_ds, train_idx, train_loader, sampler,
            val_fixed_ds, val_fixed_loader, val_full_ds, collate_val)


def to_device(batch: dict, device, keys=("tokens", "policy", "legal_mask",
                                         "value", "value_cp", "has_policy")):
    return {k: (batch[k].to(device, non_blocking=True) if k in keys else batch[k])
            for k in batch}


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def quick_validate(model, loader, device, amp_ctx, max_records: int) -> dict:
    """Cheap, fixed-subsample validation for mid-epoch curves. Must stay <~5s,
    so it reports only policy KL / top-1 / value MSE / value pred std and never
    touches the stratified or mirror diagnostics."""
    was_training = model.training
    model.eval()
    kl_sum = 0.0
    n_pol = 0
    top1 = 0
    se_sum = 0.0
    n = 0
    preds = []
    for batch in loader:
        if n >= max_records:
            break
        b = to_device(batch, device)
        with amp_ctx():
            logits, value = model(b["tokens"])
        logits = logits.float()
        value = value.float()
        se_sum += float((value - b["value"]).pow(2).sum())
        preds.append(value.cpu())
        n += int(b["tokens"].shape[0])

        has = b["has_policy"] > 0
        k = int(has.sum())
        if k:
            kl, _ = policy_kl_per_sample(logits[has], b["policy"][has],
                                         b["legal_mask"][has])
            kl_sum += float(kl.sum())
            top1 += int(policy_topk_correct(
                logits[has], b["policy"][has], b["legal_mask"][has], 1).sum())
            n_pol += k
    if was_training:
        model.train()
    pred = torch.cat(preds) if preds else torch.zeros(0)
    return {
        "val_policy_kl": kl_sum / n_pol if n_pol else float("nan"),
        "val_policy_top1": top1 / n_pol if n_pol else float("nan"),
        "val_value_mse": se_sum / max(1, n),
        "val_value_pred_std": float(pred.std(unbiased=False)) if pred.numel() > 1 else float("nan"),
        "val_value_pred_mean": float(pred.mean()) if pred.numel() else float("nan"),
        "val_n": n,
        "val_policy_n": n_pol,
    }


@torch.no_grad()
def full_validate(model, loader, device, amp_ctx) -> dict:
    """Epoch-boundary validation: stratified value metrics + policy top-1/5/KL."""
    was_training = model.training
    model.eval()
    va = ValueAccumulator()
    pa = PolicyAccumulator(topk=(1, 5))
    for batch in loader:
        b = to_device(batch, device)
        with amp_ctx():
            logits, value = model(b["tokens"])
        va.add(value.float(), b["value"], b["value_cp"])
        pa.add(logits.float(), b)
    if was_training:
        model.train()
    out = va.result()
    out.update(pa.result())
    return out


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------
def rng_snapshot() -> dict:
    np_state = np.random.get_state()
    py_state = random.getstate()
    return {
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else []),
        "numpy_keys": torch.from_numpy(np_state[1].astype(np.int64)),
        "numpy_pos": int(np_state[2]),
        "numpy_has_gauss": int(np_state[3]),
        "numpy_cached_gauss": float(np_state[4]),
        "python_keys": list(py_state[1]),
        "python_gauss": py_state[2],
    }


def rng_restore(snap: dict) -> None:
    if not snap:
        return
    try:
        torch.set_rng_state(snap["torch"].to(torch.uint8))
        if torch.cuda.is_available() and snap.get("cuda"):
            torch.cuda.set_rng_state_all([s.to(torch.uint8) for s in snap["cuda"]])
        np.random.set_state((
            "MT19937",
            np.asarray(snap["numpy_keys"], dtype=np.uint32),
            snap["numpy_pos"], snap["numpy_has_gauss"],
            snap["numpy_cached_gauss"]))
        random.setstate((3, tuple(int(x) for x in snap["python_keys"]),
                         snap["python_gauss"]))
    except Exception as e:                                   # pragma: no cover
        log(f"[warn] could not restore RNG state ({e}); continuing")


def save_checkpoint(path: Path, *, model, optimizer, scheduler, args,
                    model_config: ModelConfig, epoch: int, step: int,
                    micro_in_epoch: int, samples_seen: int,
                    best_val: float, metrics: dict, run_info: dict,
                    reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # --- what the MCTS engine and the eval harness need ---
        "model_state_dict": model.state_dict(),
        "config": model_config.to_dict(),
        "value_scale": run_info["value_scale"],
        "manifest_hash": run_info["manifest_hash"],
        "manifest_path": run_info["manifest_path"],
        "git_sha": run_info["git_sha"],
        # --- what a resume needs ---
        "epoch": epoch,
        "step": step,
        "micro_in_epoch": micro_in_epoch,
        "samples_seen": samples_seen,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": rng_snapshot(),
        # --- what makes runs comparable ---
        "effective_batch": run_info["effective_batch"],
        "micro_batch": args.micro_batch,
        "accum_steps": args.accum_steps,
        "max_lr": args.max_lr,
        "lr": optimizer.param_groups[0]["lr"],
        "coverage": run_info["coverage"],
        "coverage_source": run_info["coverage_source"],
        "corpus_coverage": run_info["corpus_coverage"],
        "subset_seed": run_info["subset_seed"],
        "subset_size": run_info["subset_size"],
        "corpus_size": run_info["corpus_size"],
        "policy_weight": args.policy_weight,
        "value_weight": args.value_weight,
        "train_config": sanitize_config(vars(args)),
        "metrics": sanitize_config(metrics),
        "best_val": best_val,
        "num_params": model.num_parameters(),
        "reason": reason,
        "created_utc": utcnow(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def find_latest_checkpoint(out_dir: Path):
    """Newest resumable checkpoint in `out_dir`, by recorded step then mtime.

    Which file is newest depends on how the run stopped - a clean epoch writes
    `_last.pt`, Ctrl-C writes `_interrupt_step<N>.pt`, a crash writes
    `_emergency_step<N>.pt`, and a hard kill leaves only the periodic
    `_step<N>.pt`. Choosing by hand means reading `ls` output correctly under
    exactly the circumstances where you are least inclined to; `--resume latest`
    does it by the step count stored INSIDE each file, which is authoritative
    where a filename is not.

    `_best.pt` is deliberately excluded: it is the best-scoring epoch, which may
    be several epochs behind the newest state.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return None
    best = None
    for p in out_dir.glob("*.pt"):
        if p.name.endswith("_best.pt") or p.name.endswith(".tmp"):
            continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=True)
            key = (int(ck.get("step", -1)), p.stat().st_mtime)
        except Exception:
            continue
        if best is None or key > best[0]:
            best = (key, p)
    return best[1] if best else None


def prune_step_checkpoints(out_dir: Path, tag: str, keep: int) -> None:
    if keep <= 0:
        return
    ckpts = sorted(out_dir.glob(f"{tag}_step*.pt"),
                   key=lambda p: p.stat().st_mtime)
    for p in ckpts[:-keep]:
        try:
            p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# LR range test
# ---------------------------------------------------------------------------
def run_lr_range_test(model, loader, device, amp_ctx, args, run_info,
                      jsonl: JsonlLogger) -> dict:
    """Exponential LR sweep on a fresh net. Destroys the weights, so this is a
    terminal mode - the script exits afterwards.

    max_lr 5e-4 was validated in Phase 1 for the 25.6M model. A 10.9M net at a
    different effective batch is a different optimisation problem, so the value
    is re-derived here rather than inherited.
    """
    n = args.lr_range_steps
    lo, hi = args.lr_range_min, args.lr_range_max
    gamma = (hi / lo) ** (1.0 / max(1, n - 1))

    optimizer = optim.AdamW(param_groups(model, args.weight_decay), lr=lo,
                            betas=(args.beta1, args.beta2), eps=args.adam_eps)
    C_pol = policy_denominator(run_info["effective_batch"], run_info["coverage"])
    C_val = float(run_info["effective_batch"])

    rows = []
    smoothed = None
    best = (float("inf"), lo)
    diverge_at = None
    it = iter(loader)
    model.train()

    log(f"\nLR range test: {n} steps, {lo:.2e} -> {hi:.2e} "
        f"(x{gamma:.4f}/step), effective batch {run_info['effective_batch']}")
    log(f"{'step':>6} {'lr':>11} {'total':>10} {'policyKL':>10} {'valueMSE':>10} {'smoothed':>10}")
    log("-" * 62)

    for i in range(n):
        lr = lo * (gamma ** i)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        acc_p = acc_v = 0.0
        acc_np = acc_ns = 0
        for _ in range(args.accum_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            b = to_device(batch, device)
            with amp_ctx():
                logits, value = model(b["tokens"])
            parts = compute_losses(logits, value, b)
            loss = (args.policy_weight * parts.policy_kl_sum / C_pol
                    + args.value_weight * parts.value_se_sum / C_val)
            loss.backward()
            acc_p += parts.policy_kl_total
            acc_v += parts.value_se_total
            acc_np += parts.n_policy
            acc_ns += parts.n_samples

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        pkl = acc_p / acc_np if acc_np else float("nan")
        vmse = acc_v / max(1, acc_ns)
        total = args.policy_weight * pkl + args.value_weight * vmse
        smoothed = total if smoothed is None else (
            args.lr_range_smooth * smoothed + (1 - args.lr_range_smooth) * total)
        rows.append({"step": i, "lr": lr, "total": total,
                     "policy_kl": pkl, "value_mse": vmse, "smoothed": smoothed})
        jsonl.write("lr_range", **rows[-1])

        if smoothed < best[0]:
            best = (smoothed, lr)
        if i % max(1, n // 25) == 0 or i == n - 1:
            log(f"{i:>6} {lr:>11.3e} {total:>10.4f} {pkl:>10.4f} "
                f"{vmse:>10.5f} {smoothed:>10.4f}")
        if not math.isfinite(smoothed) or (
                i > 10 and smoothed > args.lr_range_diverge * best[0]):
            diverge_at = lr
            log(f"  diverged at lr {lr:.3e} "
                f"(smoothed {smoothed:.4f} > {args.lr_range_diverge}x best)")
            break

    # Steepest descent: most negative d(smoothed)/d(log lr), measured over a
    # window and restricted to the region AT OR BEFORE the minimum. Taking a
    # single-step difference over the whole sweep instead picks the largest
    # one-step drop, which past the minimum is just the recovery after a noise
    # spike - it reliably returns a wildly-too-high LR.
    i_min = min(range(len(rows)), key=lambda i: rows[i]["smoothed"])
    steepest_lr = best[1]
    win = max(2, args.lr_range_slope_window)
    if i_min >= win:
        slopes = []
        for i in range(win, i_min + 1):
            dloss = rows[i]["smoothed"] - rows[i - win]["smoothed"]
            dlog = math.log(rows[i]["lr"] / rows[i - win]["lr"])
            slopes.append((dloss / dlog, rows[i - win // 2]["lr"]))
        steepest_lr = min(slopes, key=lambda s: s[0])[1]

    # Flatness guard. A range test is only informative if the loss actually
    # moves; a fresh net given a few hundred steps often barely descends, and
    # then the "minimum" is noise dressed up as a measurement.
    span = max(r["smoothed"] for r in rows) - best[0]
    flat = span < args.lr_range_min_span * best[0]

    suggestion = best[1] / args.lr_range_div
    # Usable band for a OneCycle max_lr: from an order of magnitude below the
    # loss minimum up to the minimum itself. Above it the sweep is already
    # getting worse.
    band = (best[1] / 10.0, best[1])
    result = {
        "best_smoothed_loss": best[0],
        "smoothed_span": span,
        "flat_curve": bool(flat),
        "lr_at_min_loss": best[1],
        "lr_steepest_descent": steepest_lr,
        "lr_usable_band": list(band),
        "lr_diverged_at": diverge_at,
        "suggested_max_lr": suggestion,
        "phase1_max_lr": args.max_lr,
        "effective_batch": run_info["effective_batch"],
        "n_steps": len(rows),
    }
    log("-" * 62)
    log(f"  smoothed loss span  {span:.4f} over the whole sweep "
        f"({span / best[0] * 100:.1f}% of the minimum)")
    log(f"  min smoothed loss   {best[0]:.4f} at lr {best[1]:.3e}")
    log(f"  steepest descent    lr {steepest_lr:.3e} "
        f"(windowed, restricted to <= the minimum)")
    log(f"  diverged at         {('%.3e' % diverge_at) if diverge_at else 'not reached'}")
    log(f"  usable max_lr band  {band[0]:.2e} .. {band[1]:.2e}")
    log(f"  conservative pick   {suggestion:.3e} (min-loss lr / {args.lr_range_div:g})")
    if flat:
        log(f"  [WARN] the curve is FLAT (span < {args.lr_range_min_span * 100:.0f}% "
            f"of the loss). {len(rows)} steps is not enough for this net to "
            f"separate learning rates; treat the numbers below as weak evidence "
            f"and prefer the incumbent unless it sits outside the band.")

    lo, hi = band
    if lo <= args.max_lr <= hi:
        verdict = "HOLDS - inside the usable band"
    elif args.max_lr > hi:
        verdict = (f"TOO HIGH - above the loss minimum ({best[1]:.2e}), where "
                   f"the sweep is already degrading")
    else:
        verdict = (f"conservative - below the band, would train slower than "
                   f"necessary")
    log(f"  VERDICT: phase-1 max_lr {args.max_lr:.1e} at effective batch "
        f"{run_info['effective_batch']}: {verdict}")
    jsonl.write("lr_range_result", **result)

    if args.lr_range_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot([r["lr"] for r in rows], [r["smoothed"] for r in rows],
                    label="smoothed total")
            ax.plot([r["lr"] for r in rows], [r["total"] for r in rows],
                    alpha=0.3, linewidth=0.8, label="raw total")
            ax.axvline(suggestion, color="green", linestyle="--",
                       label=f"suggested {suggestion:.2e}")
            ax.axvline(args.max_lr, color="gray", linestyle=":",
                       label=f"phase1 {args.max_lr:.0e}")
            ax.set_xscale("log")
            ax.set_xlabel("learning rate")
            ax.set_ylabel("loss")
            ax.set_title(f"LR range test - effective batch {run_info['effective_batch']}")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            plt.tight_layout()
            p = Path(args.lr_range_plot)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=120)
            log(f"  plot -> {p}")
        except Exception as e:
            log(f"[warn] LR range plot failed: {e}")
    return result


# ---------------------------------------------------------------------------
# the automated gate: KL against the baseline, then the match, then the bar
# ---------------------------------------------------------------------------
def run_gate(args, out_dir: Path, tag: str, val_full_ds, collate_val, device,
             amp_ctx, last_metrics: dict, jsonl) -> dict:
    """Everything that happens after the last optimizer step, unattended.

    Never raises. This fires at the end of a ~17-hour run, so a gate that threw
    would be indistinguishable from the training having failed, and would do it
    AFTER every checkpoint was already safely on disk. Failures are logged,
    recorded in the report, and returned.
    """
    log("\n" + "=" * 96)
    log("AUTOMATED GATE")
    log("=" * 96)

    payload: dict = {"requested": True}
    candidate = out_dir / f"{tag}_{args.h2h_candidate}.pt"
    baseline = args.h2h_baseline

    if not candidate.exists():
        log(f"  [gate] candidate checkpoint missing: {candidate}")
        payload["error"] = f"candidate missing: {candidate}"
        return payload
    if baseline is None:
        log("  [gate] --h2h-baseline not set; nothing to compare against.")
        payload["error"] = "no baseline configured"
        return payload
    if not Path(baseline).exists():
        log(f"  [gate] baseline checkpoint missing: {baseline}")
        payload["error"] = f"baseline missing: {baseline}"
        return payload

    payload["candidate"] = str(candidate)
    payload["baseline"] = str(baseline)
    log(f"  candidate {candidate.name}  ({args.h2h_candidate})")
    log(f"  baseline  {Path(baseline).name}")

    # --- criterion 1: KL, both terms measured the same way -----------------
    candidate_kl = float(last_metrics.get("policy_kl", float("nan")))
    baseline_metrics = {}
    try:
        t0 = time.time()
        baseline_metrics = score_baseline(
            Path(baseline), val_full_ds, collate_val, device, amp_ctx,
            batch_size=args.val_batch, workers=args.val_workers,
            prefetch=args.prefetch_factor, policy_weight=args.policy_weight,
            value_weight=args.value_weight)
        log(f"\n  baseline scored on THIS run's val split "
            f"({baseline_metrics['n']:,} records, {time.time() - t0:.1f}s): "
            f"KL {baseline_metrics['policy_kl']:.5f} "
            f"MSE {baseline_metrics['value_mse']:.5f}")
        log(f"  candidate on the same split: "
            f"KL {candidate_kl:.5f} "
            f"MSE {last_metrics.get('value_mse', float('nan')):.5f}")
    except Exception as exc:                            # noqa: BLE001
        log(f"  [gate] baseline scoring failed: {type(exc).__name__}: {exc}")
        payload["baseline_error"] = f"{type(exc).__name__}: {exc}"
    payload["baseline_metrics"] = baseline_metrics
    payload["candidate_metrics"] = {
        "policy_kl": candidate_kl,
        "value_mse": last_metrics.get("value_mse", float("nan")),
        "total_loss": last_metrics.get("total_loss", float("nan")),
    }

    # The engine processes each capture their own CUDA graph ladder (~730 MiB
    # at 10.9M) and concurrency multiplies that, so hand the VRAM back before
    # any of them start rather than competing with a dead training graph.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- criterion 2: the match --------------------------------------------
    sims_label = (f"{args.h2h_sims // 1000}k" if args.h2h_sims >= 1000
                  else str(args.h2h_sims))
    event = f"{tag}_vs_{Path(baseline).parent.name}_{sims_label}"
    match_dir = (args.h2h_out if args.h2h_out is not None
                 else _PROJECT_ROOT / "benchmarking" / "engine" / "games" /
                 "v6" / "head2head" / event)
    match = run_head2head(
        _PROJECT_ROOT, candidate, Path(baseline), Path(match_dir),
        games=args.h2h_games, nodes=args.h2h_sims,
        concurrency=args.h2h_concurrency,
        candidate_name=f"cand-{tag}", baseline_name="base-20M-ep9",
        event=event, log=log)
    payload["match"] = match.to_dict()
    # Nested, not splatted: MatchResult carries its own `event` field and
    # JsonlLogger.write's first positional is also called `event`.
    jsonl.write("head2head", match=match.to_dict())

    # --- the pre-registered bar --------------------------------------------
    verdict = evaluate_thresholds(
        candidate_kl, baseline_metrics.get("policy_kl", float("nan")), match)
    payload["verdict"] = verdict.to_dict()
    log(format_verdict(verdict, match))
    jsonl.write("gate_verdict", verdict=verdict.to_dict())

    report = out_dir / f"{tag}_gate_report.json"
    write_gate_report(report, payload)
    log(f"\n  gate report -> {report}")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GuoFish v5 multi-PV student training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--config", type=Path, default=None,
                   help="YAML of defaults; explicit CLI flags still win")

    g = p.add_argument_group("data")
    g.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    g.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    g.add_argument("--subset", type=int, default=10_000_000,
                   help="N train records, drawn as a prefix of a seeded global "
                        "permutation of the corpus; 0 = all")
    g.add_argument("--subset-seed", type=int, default=20260802,
                   help="seeds WHICH records --subset selects. Separate from "
                        "--seed on purpose: subsets stay nested and identical "
                        "across runs even when the training seed changes, so a "
                        "5M/10M/20M scaling study varies quantity only")
    g.add_argument("--mirror-prob", type=float, default=0.5,
                   help="train-time colour mirror rate")
    g.add_argument("--val-mirror-prob", type=float, default=0.0,
                   help="MUST stay fixed across runs; 0.0 = deterministic val")
    g.add_argument("--temperature", type=float, default=None,
                   help="re-derive the policy target at a new T (None = stored)")
    g.add_argument("--revalue", action="store_true",
                   help="re-derive value from value_cp at the manifest scale "
                        "instead of using the stored float16 target")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--val-workers", type=int, default=4)
    g.add_argument("--prefetch-factor", type=int, default=2)
    g.add_argument("--drop-last", dest="drop_last", action="store_true", default=True)
    g.add_argument("--no-drop-last", dest="drop_last", action="store_false")

    g = p.add_argument_group("model")
    g.add_argument("--d-model", type=int, default=384)
    g.add_argument("--num-layers", type=int, default=6)
    g.add_argument("--nhead", type=int, default=6)
    g.add_argument("--head-dim", type=int, default=64)
    g.add_argument("--dim-feedforward", type=int, default=1536)
    g.add_argument("--vocab-size", type=int, default=43)
    g.add_argument("--seq-len", type=int, default=68)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--activation", type=str, default="gelu")
    g.add_argument("--final-norm", dest="final_norm", action="store_true", default=True)
    g.add_argument("--no-final-norm", dest="final_norm", action="store_false")
    g.add_argument("--smolgen", action="store_true", default=False,
                   help="not implemented; separate flagged A/B")

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=9)
    g.add_argument("--micro-batch", type=int, default=512,
                   help="from bench_batch.py on an RTX 5070; throughput is "
                        "flat 128-1024 and collapses at 1536 (WDDM spill)")
    g.add_argument("--accum-steps", type=int, default=2,
                   help="micro_batch x accum_steps = effective batch 1024, "
                        "which is what --max-lr was tuned at")
    g.add_argument("--max-lr", type=float, default=3.5e-4,
                   help="LR range test minimum for this net at effective "
                        "batch 1024; Phase 1's 5e-4 was for the 25.6M model "
                        "and sits above the loss minimum here")
    g.add_argument("--pct-start", type=float, default=0.1)
    g.add_argument("--div-factor", type=float, default=25.0)
    g.add_argument("--final-div-factor", type=float, default=1e4)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--beta1", type=float, default=0.9)
    g.add_argument("--beta2", type=float, default=0.999)
    g.add_argument("--adam-eps", type=float, default=1e-8)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--policy-weight", type=float, default=1.0)
    g.add_argument("--value-weight", type=float, default=1.0)
    g.add_argument("--coverage", type=float, default=None,
                   help="pin the has_policy share used in the policy "
                        "denominator; default is to MEASURE it on the selected "
                        "records, which is always at least as good")
    g.add_argument("--coverage-samples", type=int, default=200_000,
                   help="records sampled to measure coverage (+-0.11%% at 200k)")
    g.add_argument("--policy-norm", choices=("const", "count"), default="const",
                   help="'const' divides every micro-batch by "
                        "effective_batch*coverage (accumulation-invariant). "
                        "'count' divides by the realised per-window count and "
                        "is NOT invariant under accumulation.")
    g.add_argument("--max-steps", type=int, default=0,
                   help="stop after N optimizer steps (0 = no limit)")
    g.add_argument("--seed", type=int, default=20260802)

    g = p.add_argument_group("precision")
    g.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16",
                   help="bf16 matches inference precision")
    g.add_argument("--tf32", dest="tf32", action="store_true", default=True)
    g.add_argument("--no-tf32", dest="tf32", action="store_false")
    g.add_argument("--matmul-precision", default="high",
                   choices=("highest", "high", "medium"))
    # Default ON: measured numerically faithful (eager-vs-compiled gradients
    # agree to 2.5e-6 in fp32) and ~10% faster end to end. Needs a Triton
    # matching the torch version - torch 2.8 -> triton 3.4.x.
    g.add_argument("--compile", dest="compile", action="store_true",
                   help="torch.compile the model (default: on)")
    g.add_argument("--no-compile", dest="compile", action="store_false",
                   help="run eager; use if Triton is missing or mismatched")
    g.set_defaults(compile=True)

    g = p.add_argument_group("logging / checkpointing")
    g.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    g.add_argument("--run-name", type=str, default=None)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--val-every", type=int, default=2000)
    g.add_argument("--val-subset", type=int, default=20_000)
    g.add_argument("--val-batch", type=int, default=1024)
    g.add_argument("--ckpt-every", type=int, default=5000)
    g.add_argument("--keep-step-ckpts", type=int, default=3)
    g.add_argument("--mirror-subset", type=int, default=10_000)
    g.add_argument("--drift-window", type=int, default=200)
    g.add_argument("--drift-tol", type=float, default=0.05)
    g.add_argument("--resume", type=str, default=None,
                   help="checkpoint to resume from, or the literal "
                        "'latest' to pick the newest resumable one in "
                        "--out-dir by its recorded step (never _best.pt, "
                        "which may be several epochs behind)")

    g = p.add_argument_group("evaluation gates")
    g.add_argument("--gap-probe", type=int, default=0,
                   help="records per side of the seen/unseen probe (0 = off). "
                        "Costs no training data: the two slices are the head "
                        "and tail of epoch 0's sampler order, which differ "
                        "only in whether training has reached them yet")
    g.add_argument("--gap-at-frac", type=float, default=0.25,
                   help="point in epoch 1 at which the seen/unseen gap is "
                        "measured; must clear the probe size at both ends")
    g.add_argument("--gap-every-epoch", dest="gap_every_epoch",
                   action="store_true", default=True,
                   help="also report the train-probe-vs-val gap at every epoch "
                        "boundary - the version that stays meaningful once "
                        "every record has been seen (default: on)")
    g.add_argument("--no-gap-every-epoch", dest="gap_every_epoch",
                   action="store_false")
    g.add_argument("--h2h-gate", dest="h2h_gate", action="store_true",
                   default=False,
                   help="on run completion, play an automated match against "
                        "--h2h-baseline and evaluate the pre-registered "
                        "thresholds. OFF by default: this launches a "
                        "multi-hour external process")
    g.add_argument("--no-h2h-gate", dest="h2h_gate", action="store_false")
    g.add_argument("--h2h-baseline", type=Path, default=None,
                   help="checkpoint to beat; also the KL reference, re-scored "
                        "on THIS run's val split rather than read off an old log")
    g.add_argument("--h2h-candidate", type=str, default="best",
                   choices=("best", "last"),
                   help="which of this run's checkpoints enters the match")
    g.add_argument("--h2h-games", type=int, default=200,
                   help="paired: each opening is played twice with colours "
                        "reversed, so this is rounded up to an even number")
    g.add_argument("--h2h-sims", type=int, default=12000)
    g.add_argument("--h2h-concurrency", type=int, default=2)
    g.add_argument("--h2h-out", type=Path, default=None,
                   help="match artifacts directory (default: under "
                        "benchmarking/engine/games/v6/head2head)")

    g = p.add_argument_group("lr range test")
    g.add_argument("--lr-range-test", action="store_true", default=False)
    g.add_argument("--lr-range-steps", type=int, default=500,
                   help="250 was too few to separate LRs on this net")
    g.add_argument("--lr-range-min", type=float, default=1e-5)
    g.add_argument("--lr-range-max", type=float, default=5e-3,
                   help="1e-6..1e-2 spends most of the sweep where nothing "
                        "happens; 1e-5..5e-3 is where this net responds")
    g.add_argument("--lr-range-smooth", type=float, default=0.9)
    g.add_argument("--lr-range-div", type=float, default=4.0)
    g.add_argument("--lr-range-slope-window", type=int, default=10,
                   help="steps over which the steepest-descent slope is fitted")
    g.add_argument("--lr-range-min-span", type=float, default=0.05,
                   help="smoothed-loss span below this fraction of the minimum "
                        "means the sweep did not separate learning rates")
    g.add_argument("--lr-range-diverge", type=float, default=4.0)
    g.add_argument("--lr-range-plot", type=Path, default=None)
    return p


def parse_args(argv=None) -> argparse.Namespace:
    p = build_parser()
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        import yaml
        with open(pre.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        flat = {}
        for k, v in cfg.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
        known = {a.dest for a in p._actions}
        unknown = set(flat) - known
        if unknown:
            raise SystemExit(f"unknown keys in {pre.config}: {sorted(unknown)}")
        p.set_defaults(**flat)
    args = p.parse_args(argv)
    if args.subset == 0:
        args.subset = None
    # YAML carries repo-relative paths so a config is portable; resolve them
    # against the project root rather than the CWD, so the script works from
    # anywhere instead of only from the repo root.
    for name in ("shards", "manifest", "out_dir", "resume", "config",
                 "lr_range_plot", "h2h_baseline", "h2h_out"):
        v = getattr(args, name, None)
        if v is None or (name == "resume" and str(v) == "latest"):
            continue          # 'latest' is a sentinel, resolved after out_dir
        pth = Path(v)
        setattr(args, name, pth if pth.is_absolute() else _PROJECT_ROOT / pth)
    return args


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)

    run_name = args.run_name or f"v5_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    console_log = logs_dir / f"{run_name}.log"
    sys.stdout = Tee(sys.__stdout__, open(console_log, "a", buffering=1, encoding="utf-8"))
    sys.stderr = sys.stdout
    jsonl = JsonlLogger(logs_dir / f"{run_name}.jsonl")

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_float32_matmul_precision(args.matmul_precision)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp == "off" or device.type != "cuda":
        amp_ctx = contextlib.nullcontext
    else:
        dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        def amp_ctx():
            return autocast(device_type="cuda", dtype=dtype)

    # --- manifest ---------------------------------------------------------
    manifest = load_manifest(args.manifest)
    m_hash = file_sha256(args.manifest)
    value_scale = float(manifest["value_scale"])
    corpus_coverage = manifest_coverage(manifest)
    sha = git_sha()

    log("=" * 96)
    log(f"GuoFish v5 multi-PV student | run {run_name}")
    log(f"  device        {device} "
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    log(f"  git sha       {sha}")
    log(f"  manifest      {args.manifest}")
    log(f"  manifest sha  {m_hash[:16]}...")
    log(f"  value scale   {value_scale}  (from manifest, not hardcoded)")
    log(f"  coverage      {corpus_coverage:.6f}  (corpus-wide has_policy share)")
    log(f"  strata        exact-zero {manifest['value_strata']['value_exact_zero']['share'] * 100:.1f}%  "
        f"mate {manifest['value_strata']['value_mate']['share'] * 100:.1f}%  "
        f"middle {manifest['value_strata']['value_middle']['share'] * 100:.1f}%")
    log(f"  console log   {console_log}")
    log(f"  jsonl log     {logs_dir / (run_name + '.jsonl')}")

    # --- model ------------------------------------------------------------
    model_config = ModelConfig(
        vocab_size=args.vocab_size, seq_len=args.seq_len, d_model=args.d_model,
        nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, head_dim=args.head_dim,
        dropout=args.dropout, activation=args.activation,
        final_norm=args.final_norm, smolgen=args.smolgen)
    model = build_model(model_config).to(device)
    breakdown = model.param_breakdown()
    log(f"\nModel: d_model={args.d_model} layers={args.num_layers} "
        f"heads={args.nhead} ff={args.dim_feedforward} head_dim={args.head_dim} "
        f"dropout={args.dropout} act={args.activation} pre-LN final_norm={args.final_norm}")
    for k in ("embedding", "pos_encoder", "encoder", "final_norm",
              "value_head", "policy_head", "total"):
        log(f"    {k:<14} {breakdown[k]:>12,}")
    if not args.final_norm:
        log("  [WARN] final LayerNorm DISABLED - Pre-LN leaves the residual "
            "stream unnormalised at the output.")

    if args.compile:
        try:
            # One compiled module serves several batch shapes: the training
            # micro-batch, --val-batch, and the ragged final batch of each of
            # the quick-val / full-val / mirror passes - about five distinct
            # (shape, train-vs-eval) graphs. Dynamo's default cache_size_limit
            # is 8, and exceeding it does not error: it gives up on the frame
            # and silently runs EAGER for the rest of the run, quietly costing
            # the speedup this flag exists for. Raising it is free (each entry
            # is just another cached graph).
            #
            # Measured, and deliberately NOT worked around: running the val
            # shapes does not degrade the training kernel afterwards
            # (throughput retention 99.6%), so dynamic=False is unnecessary.
            _dynamo.config.cache_size_limit = max(
                32, _dynamo.config.cache_size_limit)
            model = torch.compile(model)
            log(f"  torch.compile: enabled "
                f"(dynamo cache_size_limit={torch._dynamo.config.cache_size_limit})")
        except Exception as e:
            log(f"  [warn] torch.compile failed ({e}); continuing eager. "
                f"If this is an ImportError about triton, the Triton version "
                f"must match torch (torch 2.8 -> triton 3.4.x).")

    # --- data -------------------------------------------------------------
    (train_ds, train_idx, train_loader, sampler, val_fixed_ds,
     val_fixed_loader, val_full_ds, collate_val) = build_loaders(args, value_scale)
    n_train = len(train_idx)

    # --- the policy denominator's coverage --------------------------------
    # Measured on the records --subset actually selects, not inherited from the
    # manifest: the corpus is not stationary, so a small --subset can be wildly
    # off the corpus-wide figure (the first 50k records are 57% policy-bearing
    # against a global 40.3%) and C would be wrong by that whole ratio.
    coverage_source = "explicit --coverage"
    if args.coverage is not None:
        coverage = float(args.coverage)
        measured = None
    else:
        t0 = time.time()
        measured, n_sampled = estimate_coverage(
            train_ds, train_idx, n_samples=args.coverage_samples,
            seed=args.subset_seed)
        coverage = measured
        coverage_source = (f"measured on {n_sampled:,} of the selected records "
                           f"({time.time() - t0:.1f}s)")

    micro_sizes, windows = plan_epoch(
        n_train, args.micro_batch, args.accum_steps, args.drop_last)
    n_micro = len(micro_sizes)
    steps_per_epoch = len(windows)
    effective_batch = args.micro_batch * args.accum_steps
    total_steps_full = steps_per_epoch * args.epochs
    total_steps = (min(args.max_steps, total_steps_full)
                   if args.max_steps else total_steps_full)

    run_info = {
        "value_scale": value_scale, "coverage": coverage,
        "coverage_source": coverage_source,
        "coverage_measured": measured,
        "corpus_coverage": corpus_coverage,
        "subset_seed": args.subset_seed,
        "subset_size": n_train,
        "corpus_size": len(train_ds),
        "manifest_hash": m_hash, "manifest_path": str(args.manifest),
        "git_sha": sha, "effective_batch": effective_batch,
    }

    log(f"\nData: train {n_train:,} records selected from {len(train_ds):,} "
        f"({len(train_ds.paths)} shards) - prefix of a seeded global "
        f"permutation, subset_seed={args.subset_seed}")
    log(f"      val   {len(val_full_ds):,} records | fixed mid-epoch subset "
        f"{len(val_fixed_ds):,} | mirror slice {min(args.mirror_subset, len(val_fixed_ds)):,}")
    log(f"      train mirror {args.mirror_prob:.2f} | "
        f"VAL MIRROR {args.val_mirror_prob:.2f} "
        f"({'OFF - deterministic' if args.val_mirror_prob == 0 else 'STOCHASTIC - not comparable across runs'})")
    log(f"\nBatching: micro {args.micro_batch} x accum {args.accum_steps} "
        f"= EFFECTIVE {effective_batch}")
    log(f"      micro-batches/epoch {n_micro:,} | optimizer steps/epoch "
        f"{steps_per_epoch:,} | total {total_steps:,}"
        + (f" (capped from {total_steps_full:,} by --max-steps)"
           if total_steps < total_steps_full else ""))
    if windows and windows[-1][0] != args.accum_steps:
        log(f"      final window is SHORT: {windows[-1][0]} micro-batches, "
            f"{windows[-1][1]:,} samples - denominators sized explicitly")
    C_pol_full = policy_denominator(effective_batch, coverage)
    log(f"      coverage {coverage:.6f} - {coverage_source}")
    if measured is not None:
        ratio = measured / corpus_coverage
        log(f"      subset coverage / corpus coverage = {ratio:.4f} "
            f"({measured:.6f} vs {corpus_coverage:.6f})")
        if abs(ratio - 1.0) > args.drift_tol:
            # Under a global permutation a subset's coverage should land on the
            # corpus figure at any size. If it does not, the permutation is not
            # reaching the sampler - which would mean the run is training on a
            # source-order prefix, where composition (piece mix, value strata)
            # is skewed far beyond coverage alone.
            log(f"      [WARN] selected coverage is {ratio:.3f}x the corpus "
                f"figure. A permuted subset should match it at ANY size, so "
                f"this points at the permutation not being applied - not "
                f"merely at a mis-set C. Check select_subset / --subset-seed "
                f"before trusting this run.")
    log(f"      policy denominator C = {effective_batch} x {coverage:.6f} "
        f"= {C_pol_full:.2f}  (mode={args.policy_norm})")
    log(f"      value  denominator   = {effective_batch} (samples/effective batch)")
    log(f"      loss weights: policy {args.policy_weight} / value {args.value_weight}")
    if args.policy_norm == "count" and args.accum_steps > 1:
        log("      [WARN] --policy-norm count with accumulation is a "
            "mean-of-means; gradients depend on how the effective batch was split.")

    # --- LR range test is a terminal mode ---------------------------------
    if args.lr_range_test:
        run_lr_range_test(model, train_loader, device, amp_ctx, args,
                          run_info, jsonl)
        jsonl.close()
        return 0

    # --- optimizer / schedule --------------------------------------------
    optimizer = optim.AdamW(param_groups(model, args.weight_decay, verbose=True),
                            lr=args.max_lr / args.div_factor,
                            betas=(args.beta1, args.beta2), eps=args.adam_eps)

    # --- resume (weights, optimizer, counters, RNG) ------------------------
    start_epoch = 0
    step = 0
    micro_in_epoch = 0
    samples_seen = 0
    best_val = float("inf")
    resumed_sched_total = None
    if args.resume:
        if str(args.resume) == "latest":
            rp = find_latest_checkpoint(out_dir)
            if rp is None:
                raise SystemExit(
                    f"--resume latest: no resumable checkpoint in {out_dir}")
            log(f"\n--resume latest -> {rp.name}")
        else:
            rp = Path(args.resume)
        if not rp.exists():
            raise SystemExit(f"resume checkpoint not found: {rp}")
        log(f"\nResuming from {rp}")
        ck = torch.load(rp, map_location="cpu", weights_only=True)
        (model._orig_mod if hasattr(model, "_orig_mod") else model
         ).load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        rng_restore(ck.get("rng_state", {}))
        start_epoch = int(ck["epoch"])
        step = int(ck["step"])
        micro_in_epoch = int(ck.get("micro_in_epoch", 0))
        samples_seen = int(ck.get("samples_seen", 0))
        best_val = float(ck.get("best_val", float("inf")))
        resumed_sched_total = (ck.get("scheduler_state_dict") or {}).get("total_steps")
        if ck.get("effective_batch") != effective_batch:
            log(f"  [WARN] checkpoint effective batch {ck.get('effective_batch')} "
                f"!= current {effective_batch}; the LR schedule is not comparable.")
        log(f"  epoch {start_epoch} step {step} micro_in_epoch {micro_in_epoch} "
            f"samples {samples_seen:,} best_val {best_val:.5f}")

    # --- schedule, built AFTER resume -------------------------------------
    # Deliberately NOT scheduler.load_state_dict(): OneCycleLR bakes total_steps
    # and its phase boundaries into its state, so loading them silently
    # overrides the CURRENT run's --epochs/--max-steps with the ones the
    # checkpoint was written under - which either follows the wrong curve or
    # hard-fails with "Tried to step N times". OneCycleLR's LR is a closed-form
    # function of the step count, so replaying it from last_epoch is exact.
    if step > 0:
        if step >= total_steps:
            raise SystemExit(
                f"resumed at step {step} but this run only has {total_steps} "
                f"steps (epochs={args.epochs}, max_steps={args.max_steps}); "
                f"raise --epochs or --max-steps")
        for gp in optimizer.param_groups:
            gp.setdefault("initial_lr", args.max_lr / args.div_factor)
        if resumed_sched_total not in (None, total_steps):
            log(f"  [WARN] checkpoint scheduled {resumed_sched_total} total steps, "
                f"this run schedules {total_steps}; rebuilding the cycle at the "
                f"current setting - the LR curve will not match the original run.")
    scheduler = OneCycleLR(
        optimizer, max_lr=args.max_lr, total_steps=total_steps,
        pct_start=args.pct_start, div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
        last_epoch=step - 1 if step > 0 else -1)
    log(f"      OneCycleLR max_lr {args.max_lr:.2e} pct_start {args.pct_start} "
        f"div {args.div_factor} final_div {args.final_div_factor:g} | "
        f"AdamW wd {args.weight_decay} clip {args.grad_clip} | amp {args.amp}")

    tag = f"v5_{model.num_parameters() / 1e6:.1f}M"
    jsonl.write("run_start", run=run_name, tag=tag, git_sha=sha,
                manifest_hash=m_hash, value_scale=value_scale,
                coverage=coverage, coverage_source=coverage_source,
                subset_seed=args.subset_seed, subset_size=n_train,
                corpus_size=len(train_ds), effective_batch=effective_batch,
                micro_batch=args.micro_batch, accum_steps=args.accum_steps,
                steps_per_epoch=steps_per_epoch, total_steps=total_steps,
                num_params=model.num_parameters(),
                model_config=model_config.to_dict(),
                train_config=sanitize_config(vars(args)))

    # --- seen/unseen probe -------------------------------------------------
    # Built from epoch 0's order, so it describes the order training will
    # actually follow. Costs no training data - see gates.SeenUnseenProbe.
    gap_probe = None
    if args.gap_probe:
        gap_probe = SeenUnseenProbe(sampler, args.gap_probe, args.gap_at_frac)
        if gap_probe.valid:
            log(f"\nSeen/unseen probe: {gap_probe.n_probe:,} records/side, "
                f"measured at {args.gap_at_frac * 100:.0f}% of epoch 1")
            log(f"      seen = head of epoch-0 order (trained by then), "
                f"unseen = tail (not yet reached)")
        else:
            log(f"\n[WARN] --gap-probe {args.gap_probe:,} does not fit around "
                f"--gap-at-frac {args.gap_at_frac}: each side is "
                f"{gap_probe.n_probe / gap_probe.n_total:.3f} of the epoch, so "
                f"the slices would overlap the measurement point. Probe "
                f"DISABLED - lower --gap-probe or move --gap-at-frac to the "
                f"middle of the epoch.")
            gap_probe = None
    # A fixed train slice for the epoch-boundary gap, which stays meaningful
    # after epoch 1 when 'unseen training record' no longer exists.
    train_probe_idx = (train_idx[:args.gap_probe] if args.gap_every_epoch
                       and args.gap_probe else None)

    # --- gate preflight, at MINUTE ZERO ------------------------------------
    # run_head2head preflights itself, but that happens ~17 hours in. A typo'd
    # baseline path or a missing opening book is worth discovering now, while
    # it costs nothing to fix. Deliberately a WARNING and not fatal: a run whose
    # gate cannot fire still produces the checkpoints, and killing it here would
    # throw away the expensive half over the cheap one.
    if args.h2h_gate:
        problems = list(head2head_preflight(
            _PROJECT_ROOT,
            out_dir / f"{tag}_{args.h2h_candidate}.pt",   # written later, skipped
            Path(args.h2h_baseline) if args.h2h_baseline else Path("<unset>")))
        problems = [p for p in problems if "candidate" not in p]
        if args.h2h_baseline is None:
            problems.append("--h2h-gate is on but --h2h-baseline is not set")
        if problems:
            log(f"\n[WARN] head-to-head gate preflight found "
                f"{len(problems)} problem(s) - the gate will be SKIPPED at the "
                f"end of this run unless they are fixed before then:")
            for p in problems:
                log(f"        {p}")
        else:
            log(f"\nGate preflight OK: cutechess, opening book, uci wrapper and "
                f"baseline all resolve")
            log(f"      on completion: {args.h2h_games} paired games @ "
                f"{args.h2h_sims:,} sims vs {Path(args.h2h_baseline).name}, "
                f"then the >=5% KL / >=100 ELO bar")

    # --- interrupt handling ----------------------------------------------
    interrupted = {"flag": False}

    def _sigint(signum, frame):
        if interrupted["flag"]:
            log("\n[sigint] second interrupt - exiting now.")
            sys.exit(1)
        interrupted["flag"] = True
        log("\n[sigint] will checkpoint and stop at the next optimizer step.")

    signal.signal(signal.SIGINT, _sigint)

    # --- training ---------------------------------------------------------
    log("\n" + "=" * 96)
    training_start = time.time()
    stop = False
    last_metrics: dict = {}
    drift_hist: list = []
    last_drift_warn = -10 ** 9

    def _save(reason: str, path: Path, metrics: dict):
        save_checkpoint(path, model=(model._orig_mod
                                     if hasattr(model, "_orig_mod") else model),
                        optimizer=optimizer, scheduler=scheduler, args=args,
                        model_config=model_config, epoch=epoch,
                        step=step, micro_in_epoch=micro_in_epoch,
                        samples_seen=samples_seen, best_val=best_val,
                        metrics=metrics, run_info=run_info, reason=reason)
        log(f"  [ckpt] {reason} -> {path.name}")
        jsonl.write("checkpoint", reason=reason, path=str(path), step=step,
                    epoch=epoch)

    epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs):
            skip_micro = micro_in_epoch if epoch == start_epoch else 0
            sampler.set_epoch(epoch, skip=skip_micro * args.micro_batch)
            window_start = skip_micro // args.accum_steps
            model.train()

            epoch_start = time.time()
            win_kl = win_se = 0.0
            win_np = win_ns = 0
            log_kl = log_se = 0.0
            log_np = log_ns = 0
            log_t0 = time.time()
            log_steps = 0
            log_gn = 0.0
            epoch_np = epoch_ns = 0

            log(f"\n{'=' * 96}\nEpoch {epoch + 1}/{args.epochs}"
                + (f" (resuming at micro-batch {skip_micro:,}/{n_micro:,})"
                   if skip_micro else "")
                + f" | {steps_per_epoch - window_start:,} optimizer steps to go")
            log("=" * 96)

            optimizer.zero_grad(set_to_none=True)
            micro_idx = skip_micro

            for batch in train_loader:
                w = micro_idx // args.accum_steps
                if w >= len(windows):
                    break
                win_micro, win_total_samples = windows[w]

                b = to_device(batch, device)
                with amp_ctx():
                    logits, value = model(b["tokens"])
                parts = compute_losses(logits, value, b)

                if args.policy_norm == "const":
                    C_pol = policy_denominator(win_total_samples, coverage)
                else:
                    C_pol = max(1.0, float(parts.n_policy))
                C_val = float(win_total_samples)

                loss = (args.policy_weight * parts.policy_kl_sum / C_pol
                        + args.value_weight * parts.value_se_sum / C_val)
                loss.backward()

                win_kl += parts.policy_kl_total
                win_se += parts.value_se_total
                win_np += parts.n_policy
                win_ns += parts.n_samples
                epoch_np += parts.n_policy
                epoch_ns += parts.n_samples
                samples_seen += parts.n_samples
                micro_idx += 1
                micro_in_epoch = micro_idx

                jsonl.write("micro", step=step, epoch=epoch,
                            micro=micro_idx, n=parts.n_samples,
                            has_policy=parts.n_policy,
                            policy_kl=parts.policy_kl_mean,
                            value_mse=parts.value_mse_mean)

                at_boundary = (micro_idx - w * args.accum_steps) >= win_micro
                if not at_boundary:
                    continue

                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                # coverage drift over the effective batch
                expected = win_total_samples * coverage
                drift_hist.append((win_np, expected))
                if len(drift_hist) > args.drift_window:
                    drift_hist.pop(0)

                log_kl += win_kl
                log_se += win_se
                log_np += win_np
                log_ns += win_ns
                log_gn += grad_norm
                log_steps += 1

                if step % args.log_every == 0 or step == 1:
                    dt = max(1e-9, time.time() - log_t0)
                    sps = log_ns / dt
                    # The very first log window starts at the epoch's t0, so it
                    # includes DataLoader worker spawn and, with --compile, the
                    # one-off graph compilation. Extrapolating a multi-hour ETA
                    # from that is alarming and wrong, so the rate-derived
                    # numbers are withheld until there is a clean window.
                    warmup_window = (step == 1)
                    epoch_frac = (w + 1) / steps_per_epoch
                    steps_left_epoch = steps_per_epoch - (w + 1)
                    steps_left_run = max(0, total_steps - step)
                    spstep = dt / max(1, log_steps)
                    row = {
                        "step": step, "epoch": epoch,
                        "epoch_frac": round(epoch_frac, 5),
                        "lr": scheduler.get_last_lr()[0],
                        "policy_kl": log_kl / log_np if log_np else float("nan"),
                        "value_mse": log_se / max(1, log_ns),
                        "has_policy": log_np,
                        "has_policy_frac": log_np / max(1, log_ns),
                        "grad_norm": log_gn / max(1, log_steps),
                        "samples_per_s": sps,
                        "tokens_per_s": sps * args.seq_len,
                        "gpu_util": gpu_utilization(),
                        "samples_seen": samples_seen,
                        "eta_epoch_s": (float("nan") if warmup_window
                                        else spstep * steps_left_epoch),
                        "eta_run_s": (float("nan") if warmup_window
                                      else spstep * steps_left_run),
                        "warmup_window": warmup_window,
                    }
                    jsonl.write("step", **row)
                    log(f"E{epoch + 1} step {step:>6}/{total_steps} "
                        f"({epoch_frac * 100:5.1f}% ep) "
                        f"lr {row['lr']:.3e} | "
                        f"KL {row['policy_kl']:.4f} MSE {row['value_mse']:.5f} | "
                        f"hp {log_np:>6,} ({row['has_policy_frac'] * 100:4.1f}%) | "
                        f"gn {row['grad_norm']:.3f} | "
                        + (f"{'warmup':>6} smp/s (spawn+compile) | ETA pending"
                           if warmup_window else
                           f"{sps:>6,.0f} smp/s "
                           f"{row['tokens_per_s'] / 1e6:.2f}M tok/s | "
                           f"ETA ep {fmt_hms(row['eta_epoch_s'])} "
                           f"run {fmt_hms(row['eta_run_s'])}"))
                    log_kl = log_se = 0.0
                    log_np = log_ns = 0
                    log_gn = 0.0
                    log_steps = 0
                    log_t0 = time.time()

                # coverage drift warning
                if len(drift_hist) >= args.drift_window and step - last_drift_warn > args.drift_window:
                    got = sum(a for a, _ in drift_hist)
                    exp = sum(b for _, b in drift_hist)
                    ratio = got / exp if exp else float("nan")
                    if abs(ratio - 1.0) > args.drift_tol:
                        last_drift_warn = step
                        log(f"  [WARN] has_policy realised/expected = {ratio:.4f} "
                            f"over the last {args.drift_window} effective batches "
                            f"({got:,} vs {exp:,.0f}); the policy denominator "
                            f"C = effective_batch x {coverage:.4f} may be wrong "
                            f"for this subset.")
                        jsonl.write("coverage_drift", step=step, ratio=ratio,
                                    realised=got, expected=exp)

                win_kl = win_se = 0.0
                win_np = win_ns = 0

                # mid-epoch validation (fixed subsample, cheap)
                if args.val_every and step % args.val_every == 0:
                    t0 = time.time()
                    vm = quick_validate(model, val_fixed_loader, device,
                                        amp_ctx, args.val_subset)
                    vm["val_seconds"] = time.time() - t0
                    jsonl.write("val_quick", step=step, epoch=epoch, **vm)
                    log(f"  [val] step {step} KL {vm['val_policy_kl']:.4f} "
                        f"top1 {vm['val_policy_top1'] * 100:.2f}% "
                        f"MSE {vm['val_value_mse']:.5f} "
                        f"pred_std {vm['val_value_pred_std']:.4f} "
                        f"({vm['val_seconds']:.1f}s, n={vm['val_n']:,})")

                # seen/unseen gap - once, partway through epoch 1, while the
                # tail of this epoch's order is still genuinely untouched.
                if gap_probe is not None and gap_probe.due(
                        epoch, (w + 1) / steps_per_epoch):
                    gap = gap_probe.measure(
                        model, train_ds, collate_val, device, amp_ctx,
                        batch_size=args.val_batch, workers=args.val_workers,
                        prefetch=args.prefetch_factor,
                        policy_weight=args.policy_weight,
                        value_weight=args.value_weight)
                    gap["step"] = step
                    gap["epoch"] = epoch
                    log(format_gap_report(
                        gap, f"Seen/unseen gap at step {step:,} "
                             f"({gap['n_probe']:,}/side, {gap['seconds']:.1f}s)"))
                    log(f"  reference: the 20M ep9 run's gap is ~2%; with 4.5x "
                        f"the unique data this is expected to shrink toward 0")
                    jsonl.write("seen_unseen_gap", **sanitize_config(gap))

                if args.ckpt_every and step % args.ckpt_every == 0:
                    _save("step", out_dir / f"{tag}_step{step}.pt", last_metrics)
                    # Keep _last.pt meaning "the newest resumable state". If it
                    # were only written at epoch boundaries it could be ~28
                    # minutes staler than the _step file sitting next to it,
                    # and resuming from it would silently redo that work.
                    _save("last", out_dir / f"{tag}_last.pt", last_metrics)
                    prune_step_checkpoints(out_dir, tag, args.keep_step_ckpts)

                if interrupted["flag"]:
                    _save("sigint", out_dir / f"{tag}_interrupt_step{step}.pt",
                          last_metrics)
                    stop = True
                    break
                if args.max_steps and step >= args.max_steps:
                    log(f"\n--max-steps {args.max_steps} reached.")
                    stop = True
                    break

            # ---- epoch boundary: the expensive diagnostics ----------------
            train_time = time.time() - epoch_start
            log(f"\n--- epoch {epoch + 1} train done in {fmt_hms(train_time)} "
                f"({samples_seen:,} samples cumulative) ---")

            t0 = time.time()
            val_full_loader = DataLoader(
                val_full_ds, batch_size=args.val_batch, shuffle=False,
                num_workers=args.val_workers, drop_last=False,
                collate_fn=collate_val, pin_memory=True,
                persistent_workers=False,
                prefetch_factor=args.prefetch_factor if args.val_workers else None)
            metrics = full_validate(model, val_full_loader, device, amp_ctx)
            del val_full_loader
            metrics.update(mirror_consistency(
                model, val_fixed_loader, device,
                max_records=args.mirror_subset, autocast_ctx=amp_ctx))
            metrics["val_seconds"] = time.time() - t0
            metrics["epoch"] = epoch + 1
            metrics["step"] = step
            metrics["train_seconds"] = train_time
            metrics["train_coverage_realised"] = (epoch_np / epoch_ns
                                                  if epoch_ns else float("nan"))
            metrics["train_coverage_expected"] = coverage
            metrics["total_loss"] = (args.policy_weight * metrics["policy_kl"]
                                     + args.value_weight * metrics["value_mse"])
            last_metrics = metrics

            log(f"\nEpoch {epoch + 1} validation "
                f"({metrics['n']:,} records, mirror {args.val_mirror_prob:.2f}, "
                f"{metrics['val_seconds']:.1f}s)")
            log(format_value_report(metrics))
            log(format_policy_report(metrics))
            log(format_mirror_report(metrics))
            log(f"  total (policy {args.policy_weight} + value {args.value_weight}) "
                f"= {metrics['total_loss']:.5f}")
            cov_r = metrics["train_coverage_realised"]
            cov_ratio = cov_r / coverage if coverage else float("nan")
            log(f"  train coverage realised {cov_r:.5f} vs expected "
                f"{coverage:.5f} (ratio {cov_ratio:.4f}) over {epoch_ns:,} samples")
            if abs(cov_ratio - 1.0) > args.drift_tol:
                log(f"  [WARN] the policy denominator C assumed {coverage:.4f} "
                    f"coverage but this epoch realised {cov_r:.4f}; the policy "
                    f"loss was scaled by {cov_ratio:.3f}x. Re-run with "
                    f"--coverage {cov_r:.6f}, or check that select_subset is "
                    f"being applied.")
            # The durable generalisation gap: a fixed slice of TRAIN against
            # the val split, both scored deterministically. Unlike the epoch-1
            # head/tail construction this stays meaningful for all 4 epochs,
            # and it is the quantity the ~2% quoted for 20M ep9 refers to.
            if train_probe_idx is not None:
                t0 = time.time()
                seen = probe_metrics(
                    model, train_ds, train_probe_idx, collate_val, device,
                    amp_ctx, batch_size=args.val_batch,
                    workers=args.val_workers, prefetch=args.prefetch_factor,
                    policy_weight=args.policy_weight,
                    value_weight=args.value_weight)
                held = {"policy_kl": metrics["policy_kl"],
                        "value_mse": metrics["value_mse"],
                        "total_loss": metrics["total_loss"]}
                egap = relative_gap(held, seen)
                egap.update({"epoch": epoch + 1, "step": step,
                             "n_probe": len(train_probe_idx),
                             "seconds": time.time() - t0,
                             "construction": "fixed train slice vs val split"})
                log(format_gap_report(
                    egap, f"Epoch {epoch + 1} train/val gap "
                          f"({len(train_probe_idx):,} train records vs "
                          f"{metrics['n']:,} val, {egap['seconds']:.1f}s)"))
                jsonl.write("train_val_gap", **sanitize_config(egap))
                metrics["train_val_gap_rel"] = egap["total_loss_gap_rel"]

            jsonl.write("epoch", **sanitize_config(metrics))

            _save("epoch", out_dir / f"{tag}_ep{epoch + 1}.pt", metrics)
            _save("last", out_dir / f"{tag}_last.pt", metrics)
            if metrics["total_loss"] < best_val:
                best_val = metrics["total_loss"]
                _save("best", out_dir / f"{tag}_best.pt", metrics)
                log(f"  >>> new best val total {best_val:.5f}")
            else:
                log(f"  no improvement (best {best_val:.5f})")

            micro_in_epoch = 0
            if stop:
                break

    except KeyboardInterrupt:
        _save("keyboard_interrupt",
              out_dir / f"{tag}_interrupt_step{step}.pt", last_metrics)
        raise
    except Exception:
        import traceback
        log("\n[error] unhandled exception - saving emergency checkpoint:")
        traceback.print_exc()
        _save("exception", out_dir / f"{tag}_emergency_step{step}.pt",
              last_metrics)
        raise

    # --- final report -----------------------------------------------------
    total_time = time.time() - training_start
    if not last_metrics:
        # smoke / max-steps runs stop before an epoch boundary; still validate.
        val_full_loader = DataLoader(
            val_full_ds, batch_size=args.val_batch, shuffle=False,
            num_workers=args.val_workers, drop_last=False,
            collate_fn=collate_val, pin_memory=True, persistent_workers=False,
            prefetch_factor=args.prefetch_factor if args.val_workers else None)
        last_metrics = full_validate(model, val_full_loader, device, amp_ctx)
        del val_full_loader
        last_metrics.update(mirror_consistency(
            model, val_fixed_loader, device, max_records=args.mirror_subset,
            autocast_ctx=amp_ctx))
        last_metrics["total_loss"] = (
            args.policy_weight * last_metrics["policy_kl"]
            + args.value_weight * last_metrics["value_mse"])
        log(f"\nFinal validation ({last_metrics['n']:,} records)")
        log(format_value_report(last_metrics))
        log(format_policy_report(last_metrics))
        log(format_mirror_report(last_metrics))
        _save("final", out_dir / f"{tag}_final_step{step}.pt", last_metrics)

    log("\n" + "=" * 96)
    log(f"Done in {fmt_hms(total_time)} | {step:,} optimizer steps | "
        f"{samples_seen:,} samples | best val total {best_val:.5f}")
    log(f"  samples/s overall {samples_seen / max(1e-9, total_time):,.0f}  "
        f"tokens/s {samples_seen * args.seq_len / max(1e-9, total_time) / 1e6:.2f}M")
    log("=" * 96)

    # --- automated gate ----------------------------------------------------
    # Runs on completion, with no human in the loop. Deliberately AFTER the
    # run_end accounting above is computed but before it is written, so the
    # verdict lands in the same jsonl record as the run it judges.
    gate_payload = None
    if args.h2h_gate:
        gate_payload = run_gate(args, out_dir, tag, val_full_ds, collate_val,
                                device, amp_ctx, last_metrics, jsonl)

    jsonl.write("run_end", **{**sanitize_config(last_metrics),
                              "step": step, "samples": samples_seen,
                              "seconds": total_time, "best_val": best_val,
                              "gate": gate_payload or {}})
    jsonl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
