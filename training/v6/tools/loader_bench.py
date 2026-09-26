"""Loader throughput with no model: the trainer's DataLoader (mixture sampler,
shard reader, target builder and collate) over a config's corpus.

    CUDA_VISIBLE_DEVICES=-1 python -m training.v6.tools.loader_bench \
        --config training/v6/config/configs/mix_v2.yaml --members-dir <scratch> [--seconds 120]

Built the way train.run builds it (same Mixture, BatchBuilder, StreamDataset,
DataLoader arguments; pin_memory off since no GPU is used). The first
--warmup micro-batches (worker spawn, first reads) are not timed. Prints one
JSON line: samples/s over the timed window and the realised group shares.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from training.v6.config import load_config
from training.v6.data.batch import BatchBuilder, StreamDataset
from training.v6.data.mixture import Mixture
from training.v6.data.reader import ShardSet
from training.v6.data.strata import load_strata
from training.v6.train import resolve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--members-dir", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    shards = ShardSet(resolve(cfg.data.corpus), cfg.data.split, resolve(cfg.data.manifest))
    strata = np.asarray(load_strata(resolve(cfg.data.strata), len(shards)))
    mixture = Mixture(cfg.mixture, len(shards), cfg.optim.micro_batch, cfg.run.data_seed,
                      strata=strata, members_dir=args.members_dir)
    t = cfg.targets
    builder = BatchBuilder(cfg.model.token_scheme, t.mirror_prob, cfg.run.data_seed,
                           t.policy_soft.epsilon, t.policy_soft.temperature)
    n_micro = 10 ** 7
    loader = torch.utils.data.DataLoader(
        StreamDataset(shards, mixture, builder, n_micro), batch_size=None, sampler=range(n_micro),
        num_workers=cfg.data.workers, prefetch_factor=cfg.data.prefetch_factor,
        pin_memory=False, persistent_workers=False, generator=torch.Generator())
    groups = np.zeros(len(mixture.names), dtype=np.int64)
    n = 0
    t_start = time.perf_counter()
    for k, b in enumerate(loader):
        if k == args.warmup:
            t0 = time.perf_counter()
        if k >= args.warmup:
            n += len(b["record_index"])
            groups += np.bincount(b["group"].numpy(), minlength=len(groups))
            if time.perf_counter() - t0 >= args.seconds:
                break
    el = time.perf_counter() - t0
    out = {"samples_per_s": n / el, "samples": n, "seconds": round(el, 1),
           "warmup_s": round(t0 - t_start, 1), "workers": cfg.data.workers,
           "micro_batch": cfg.optim.micro_batch, "records": len(shards),
           "group_shares": dict(zip(mixture.names, (groups / groups.sum()).round(4).tolist())),
           "group_sizes": mixture.sizes}
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
