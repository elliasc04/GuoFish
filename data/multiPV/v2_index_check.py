"""Corpus v2 pre-build check (design doc §6.4): what does value_min_depth 24 add?

    CUDA_VISIBLE_DEVICES= python data/multiPV/v2_index_check.py --out <dir>

One sequential pass over the Pass A index (5.5 GB, ~1 min measured), one
process, BELOW_NORMAL priority, numpy only. It reproduces the 90M selection
exactly as pass_b_convert.build_selection does, and SELF-CHECKS the counts
against dataset_manifest_90m.json before reporting anything (the recon's
headroom_scan.py method). Then it reports the bucket x source-decile
composition of the NEW policy-eligible rows at depth 24, meaning rows with
24 <= max_depth < 26 and policy_depth >= 20, plus the new value-only rows
for context.

Writes <out>/v2_index_check.json and <out>/v2_index_check.csv. Exits 1 if the
self-check fails.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

if sys.platform == "win32":                                   # BELOW_NORMAL_PRIORITY_CLASS
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from feasibility_scan import BUCKETS, MAX_PIECES  # noqa: E402
from pass_a_index import INDEX_DTYPE  # noqa: E402

CHUNK = 5_000_000
NEW_VMD = 24


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index", type=Path, default=HERE / "index" / "pass_a_index.bin")
    ap.add_argument("--manifest", type=Path, default=HERE / "manifests" / "dataset_manifest_90m.json")
    args = ap.parse_args()
    m90 = json.loads(args.manifest.read_text())
    vmd, pmd = m90["value_min_depth"], m90["policy_min_depth"]
    r_pol, r_val = m90["bucket_sampling_rates_policy"], m90["bucket_sampling_rates_value_only"]

    ix = np.memmap(args.index, dtype=INDEX_DTYPE, mode="r")
    n = len(ix)
    rng = np.random.default_rng(m90["seed"])
    nb = len(BUCKETS)
    keys = ("new_policy", "new_value_only", "policy_eligible_24", "policy_eligible_26")
    tab = {k: np.zeros((10, nb), dtype=np.int64) for k in keys}
    sel90 = sel90_pol = sel90_val = 0
    t0 = time.time()
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        pc = np.asarray(ix["piece_count"][start:stop])
        md = np.asarray(ix["max_depth"][start:stop])
        pdp = np.asarray(ix["policy_depth"][start:stop])
        u = rng.random(stop - start)                         # the 90M selection's u stream
        legal = pc <= MAX_PIECES
        has_pol = pdp >= pmd
        e26 = legal & (md >= vmd)
        e24 = legal & (md >= NEW_VMD)
        keep = np.zeros(stop - start, dtype=bool)
        bk = np.full(stop - start, -1, dtype=np.int64)
        for j, (name, lo, hi) in enumerate(BUCKETS):
            b = (pc >= lo) & (pc <= hi)
            bk[b] = j
            keep |= (e26 & b & has_pol) & (u < r_pol[name])
            keep |= (e26 & b & ~has_pol) & (u < r_val[name])
        sel90 += int(keep.sum())
        sel90_pol += int((keep & has_pol).sum())
        sel90_val += int((keep & ~has_pol).sum())
        dec = (np.arange(start, stop, dtype=np.int64) * 10) // n
        masks = {"new_policy": e24 & ~e26 & has_pol, "new_value_only": e24 & ~e26 & ~has_pol,
                 "policy_eligible_24": e24 & has_pol, "policy_eligible_26": e26 & has_pol}
        ok = bk >= 0
        cell = dec * nb + np.where(ok, bk, 0)
        for k, msk in masks.items():
            tab[k] += np.bincount(cell[msk & ok], minlength=10 * nb).reshape(10, nb)
        print(f"  {stop:,}/{n:,} rows  {time.time() - t0:.1f}s", flush=True)
    del ix

    sc = m90["source_order_quantiles"]["counts"]
    checks = {"sel90_vs_manifest": [sel90, m90["selected_lines"]],
              "sel90_policy_vs_manifest": [sel90_pol, sc["selected_policy"]],
              "sel90_value_only_vs_manifest": [sel90_val, sc["selected_value_only"]],
              "policy_eligible_26_vs_manifest": [int(tab["policy_eligible_26"].sum()),
                                                 sc["eligible_policy"]]}
    passed = all(a == b for a, b in checks.values())
    names = [b[0] for b in BUCKETS]
    tot = {k: int(v.sum()) for k, v in tab.items()}
    summary = {
        "elapsed_s": round(time.time() - t0, 1), "index_rows": n,
        "value_min_depth_new": NEW_VMD, "value_min_depth_90m": vmd, "policy_min_depth": pmd,
        "self_check_passed": passed, "self_check": checks, "totals": tot,
        "new_policy_by_bucket": {nm: int(tab["new_policy"][:, j].sum()) for j, nm in enumerate(names)},
        "new_policy_bucket_share": {nm: float(tab["new_policy"][:, j].sum() / max(1, tot["new_policy"]))
                                    for j, nm in enumerate(names)},
        "new_policy_by_decile": [int(x) for x in tab["new_policy"].sum(1)],
        "new_value_only_by_bucket": {nm: int(tab["new_value_only"][:, j].sum())
                                     for j, nm in enumerate(names)},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "v2_index_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["decile", "bucket", *keys])
        for d in range(10):
            for j, nm in enumerate(names):
                w.writerow([d, nm, *(int(tab[k][d, j]) for k in keys)])
    (args.out / "v2_index_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        print("SELF-CHECK FAILED: the reproduced 90M selection does not match the manifest",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
