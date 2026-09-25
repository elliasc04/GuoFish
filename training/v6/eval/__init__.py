"""Eval sets and metrics (§9).

Per-row results stay on the device and are concatenated once at the end of a
pass, then reduced on the CPU. Metric groups:

1. v5 continuity   policy KL / top-1 / top-5 on multi-PV rows (v5 definitions:
                   legal-masked argmax vs the dense target's argmax); value MSE,
                   Pearson r, pred std per value stratum; total = KL + MSE;
                   mirror consistency under v5_68.
2. target-invariant  top-1 vs SF's best move (pv_idx[0] on multi-PV rows,
                   hard_move where present); PV-restricted KL (model and target
                   renormalised over the record's PV moves); hard-move NLL;
                   mean policy entropy.
3. slices          piece bucket x value stratum; material class with MSE and
                   mean signed error (pred - label).
4. hlgauss         mean predicted spread per stratum; corr(spread, |error|).
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from training.v6.config.schema import STRATA_FIELDS
from training.v6.data.batch import BatchBuilder, EvalDataset, mirror_records
from training.v6.data.mixture import apportion
from training.v6.data.reader import ShardSet
from training.v6.data.strata import field_of, load_strata
from training.v6.losses import hard_ce, masked_log_softmax, soft_kl

from training.v6.data.formats import _MPV  # noqa: F401
from mirror import POLICY_PERM  # noqa: E402

_PERM_T = torch.from_numpy(POLICY_PERM)


def quick_subset(codes: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Seeded random subset stratified over the full strata code (H3 fix):
    each cell gets its largest-remainder share of `size`, drawn uniformly."""
    codes = np.asarray(codes)
    cells, inv = np.unique(codes, return_inverse=True)
    counts = np.bincount(inv)
    if size > len(codes):
        raise ValueError(f"quick subset {size} > {len(codes)} records")
    take = apportion(counts / counts.sum(), size)
    out = []
    for c, (cell, n) in enumerate(zip(cells, take)):
        if n:
            members = np.flatnonzero(inv == c)
            rng = np.random.default_rng([seed, int(cell)])
            out.append(rng.choice(members, int(n), replace=False))
    return np.sort(np.concatenate(out))


class EvalSet:
    def __init__(self, name: str, shard_dir, split: str, strata_path, token_scheme: str,
                 batch: int, workers: int, indices=None, manifest=None):
        self.name = name
        self.shards = ShardSet(shard_dir, split, manifest)
        codes = np.asarray(load_strata(strata_path, len(self.shards)))
        self.indices = np.arange(len(self.shards)) if indices is None else np.asarray(indices)
        self.codes = codes[self.indices]
        self.builder = BatchBuilder(token_scheme, 0.0, seed=0, epsilon=0.05, temperature=None)
        self.batch, self.workers = batch, workers

    def loader(self, indices=None):
        ds = EvalDataset(self.shards, self.indices if indices is None else indices,
                         self.batch, self.builder)
        # own generator: a loader without one draws from the global torch RNG,
        # which would shift dropout masks and break exact resume
        return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=self.workers,
                                           persistent_workers=False, generator=torch.Generator())


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return float("nan")
    a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else float("nan")


def _mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    n = int(mask.sum())
    return float(x[mask].double().sum() / n) if n else float("nan")


@torch.no_grad()
def _rows(model, loader, device, amp) -> dict:
    cols: dict[str, list] = {}
    for b in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with amp():
            out = model.forward_train(b["tokens"])
        logits = out["policy_logits"].float()
        has, legal = b["has_policy"], b["legal"]
        kl, _ = soft_kl(logits, b["policy"], legal)
        lq = masked_log_softmax(logits, legal)
        gold = b["policy"].argmax(-1)
        top5 = lq.topk(5, dim=-1).indices
        am = lq.argmax(-1)
        p = lq.exp()
        ent = -(torch.where(legal, p * lq, torch.zeros_like(p))).sum(-1)

        # PV-restricted KL: model and target both renormalised over the PV set
        pv_ok = torch.arange(b["pv_idx"].shape[1], device=device)[None] < b["n_pv"][:, None]
        pv_ok &= has[:, None]
        support = torch.zeros(legal.shape, dtype=torch.int32, device=device).scatter_add_(
            1, b["pv_idx"].clamp(0, 4095), pv_ok.int()) > 0
        t = b["policy"] * support
        t = t / t.sum(-1, keepdim=True).clamp_min(1e-30)
        lq_pv = masked_log_softmax(logits, support)
        pv_kl = (torch.xlogy(t, t) - t * lq_pv.masked_fill(~support, 0.0)).sum(-1)

        _, hnll = hard_ce(logits, b["hard_move"], legal, 0.0)
        row = {
            "kl": kl, "has": has, "top1": am == gold, "top5": (top5 == gold[:, None]).any(-1),
            "sf_pv0": am == b["pv_idx"][:, 0], "sf_hard": am == b["hard_move"],
            "has_hard": b["hard_move"] >= 0, "hard_nll": hnll, "pv_kl": pv_kl,
            "entropy": ent, "has_legal": legal.any(-1),
            "pred": out["value"].float(), "label": b["value"],
            "record_index": b["record_index"],
        }
        if "value_spread" in out:
            row["spread"] = out["value_spread"].float()
        for k, v in row.items():
            cols.setdefault(k, []).append(v)
    return {k: torch.cat(v).cpu() for k, v in cols.items()}


def evaluate(model, es: EvalSet, device, amp=contextlib.nullcontext, indices=None,
             mirror_n: int = 0, mirror_seed: int = 0) -> dict:
    was = model.training
    model.eval()
    r = _rows(model, es.loader(indices), device, amp)
    pos = np.searchsorted(es.indices, r["record_index"].numpy())
    codes = es.codes[pos]
    has, hh = r["has"], r["has_hard"]
    err = r["pred"] - r["label"]
    kl_mean = _mean(r["kl"], has)
    mse = float(err.double().pow(2).mean())
    m = {"n": int(len(err)), "policy_n": int(has.sum()),
         "policy_kl": kl_mean, "policy_top1": _mean(r["top1"].float(), has),
         "policy_top5": _mean(r["top5"].float(), has),
         "value_mse": mse, "value_pearson_r": _pearson(r["pred"], r["label"]),
         "value_pred_std": float(r["pred"].std(unbiased=False)),
         "total": kl_mean + mse,
         "sf_top1_pv0": _mean(r["sf_pv0"].float(), has),
         "sf_top1_hard": _mean(r["sf_hard"].float(), hh), "hard_n": int(hh.sum()),
         "hard_nll": _mean(r["hard_nll"], hh), "pv_kl": _mean(r["pv_kl"], has),
         "policy_entropy": _mean(r["entropy"], r["has_legal"])}

    vs = torch.from_numpy(field_of(codes, "value").astype(np.int64))
    bucket = torch.from_numpy(field_of(codes, "bucket").astype(np.int64))
    material = torch.from_numpy(field_of(codes, "material").astype(np.int64))
    for i, s in enumerate(STRATA_FIELDS["value"]):
        sel = vs == i
        m[f"value/{s}/n"] = int(sel.sum())
        m[f"value/{s}/mse"] = _mean(err.pow(2), sel)
        m[f"value/{s}/r"] = _pearson(r["pred"][sel], r["label"][sel])
        m[f"value/{s}/pred_std"] = float(r["pred"][sel].std(unbiased=False)) if sel.sum() > 1 else float("nan")
        for j, bk in enumerate(STRATA_FIELDS["bucket"]):
            cell = sel & (bucket == j)
            m[f"slice/{bk}/{s}/n"] = int(cell.sum())
            m[f"slice/{bk}/{s}/value_mse"] = _mean(err.pow(2), cell)
            m[f"slice/{bk}/{s}/policy_kl"] = _mean(r["kl"], cell & has)
        if "spread" in r:
            m[f"hlgauss/{s}/spread"] = _mean(r["spread"], sel)
            m[f"hlgauss/{s}/spread_err_r"] = _pearson(r["spread"][sel], err[sel].abs())
    if "spread" in r:
        m["hlgauss/spread_err_r"] = _pearson(r["spread"], err.abs())
    for i, c in enumerate(STRATA_FIELDS["material"]):
        sel = material == i
        m[f"material/{c}/n"] = int(sel.sum())
        m[f"material/{c}/value_mse"] = _mean(err.pow(2), sel)
        m[f"material/{c}/value_bias"] = _mean(err, sel)

    if mirror_n and es.builder.token_scheme == "v5_68":
        m.update(mirror_consistency(model, es, device, amp, mirror_n, mirror_seed))
    model.train(was)
    return m


@torch.no_grad()
def mirror_consistency(model, es: EvalSet, device, amp, n: int, seed: int) -> dict:
    """v5's diagnostic on a seeded random subset (not a source-order prefix)."""
    idx = np.sort(np.random.default_rng(seed).choice(es.indices, min(n, len(es.indices)),
                                                     replace=False))
    perm = _PERM_T.to(device)
    abs_sum = signed = kl_sym = 0.0
    vmax, agree = 0.0, 0
    for s in range(0, len(idx), es.batch):
        chunk = idx[s:s + es.batch]
        rec = es.shards.read(chunk)
        a = es.builder(rec, chunk)
        b = es.builder(mirror_records(rec, np.ones(len(rec), dtype=bool)), chunk)
        with amp():
            la, va = model(a["tokens"].to(device))
            lb, vb = model(b["tokens"].to(device))
        resid = va.float() + vb.float()
        abs_sum += float(resid.abs().sum())
        signed += float(resid.sum())
        vmax = max(vmax, float(resid.abs().max()))
        qa = masked_log_softmax(la.float(), a["legal"].to(device))
        qb = masked_log_softmax(lb.float(), b["legal"].to(device))
        agree += int((qb.argmax(-1) == perm[qa.argmax(-1)]).sum())
        pa, pb = qa.exp(), qb.exp()[:, perm]
        kl_sym += float(torch.xlogy(pa, pa.clamp_min(1e-12) / pb.clamp_min(1e-12)).sum())
    k = len(idx)
    return {"mirror_n": k, "mirror_value_abs_resid": abs_sum / k, "mirror_value_max_resid": vmax,
            "mirror_pred_mean_balanced": signed / (2 * k), "mirror_policy_top1_agreement": agree / k,
            "mirror_policy_kl": kl_sym / k}

