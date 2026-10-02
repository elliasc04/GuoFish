"""Corpus v3 brief §1: Pass A index rows per max_depth tier x label x piece bucket.

    python data/multiPV/v3_index_scan.py [--index I] [--m90 M] [--out data/multiPV/manifests/v3_index_scan.json]

Tiers by max_depth: old >= 26, new 24-25, t20 20-23. Label policy = policy_depth >= 20,
else value_only. Rows with > 32 pieces are excluded (every builder excludes them).
Self-check: the same pass replays the 90M selection (its seed, u stream and rates) and
must reproduce its manifest's selected_lines, or nothing is written.
Decision: t20 policy rows are included iff they number >= 10M outside the <=5 bucket.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from feasibility_scan import BUCKETS, MAX_PIECES  # noqa: E402
from pass_a_index import INDEX_DTYPE  # noqa: E402

TIER_EDGES = {"old": (26, 1 << 15), "new": (24, 26), "t20": (20, 24)}
T20_MIN_ROWS = 10_000_000


def scan(index: Path, m90: dict) -> dict:
    ix = np.memmap(index, dtype=INDEX_DTYPE, mode="r")
    rng = np.random.default_rng(m90["seed"])
    counts = {t: {lab: {b: 0 for b, _, _ in BUCKETS} for lab in ("policy", "value_only")}
              for t in TIER_EDGES}
    n90 = 0
    for start in range(0, len(ix), 20_000_000):
        stop = min(start + 20_000_000, len(ix))
        pc = np.asarray(ix["piece_count"][start:stop])
        md = np.asarray(ix["max_depth"][start:stop])
        pol = np.asarray(ix["policy_depth"][start:stop]) >= m90["policy_min_depth"]
        u = rng.random(stop - start)
        for b, lo, hi in BUCKETS:
            in_b = (pc >= lo) & (pc <= hi) & (pc <= MAX_PIECES)
            for lab, lm in (("policy", pol), ("value_only", ~pol)):
                cell = in_b & lm
                for t, (dlo, dhi) in TIER_EDGES.items():
                    counts[t][lab][b] += int((cell & (md >= dlo) & (md < dhi)).sum())
                n90 += int((cell & (md >= m90["value_min_depth"])
                            & (u < m90[f"bucket_sampling_rates_{lab}"][b])).sum())
        print(f"  {stop:,}/{len(ix):,} rows", file=sys.stderr, flush=True)
    return {"index_rows": len(ix), "counts": counts, "replay_90m_selected": n90}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=_HERE / "index" / "pass_a_index.bin")
    ap.add_argument("--m90", type=Path, default=_HERE / "manifests" / "dataset_manifest_90m.json")
    ap.add_argument("--out", type=Path, default=_HERE / "manifests" / "v3_index_scan.json")
    args = ap.parse_args(argv)
    m90 = json.loads(args.m90.read_text())
    t0 = time.time()
    r = scan(args.index, m90)
    if r["replay_90m_selected"] != m90["selected_lines"]:
        raise SystemExit(f"self-check failed: 90M replay {r['replay_90m_selected']:,} != "
                         f"{m90['selected_lines']:,}; the scan is not trustworthy")
    c = r["counts"]
    t20_pol_gt5 = sum(v for b, v in c["t20"]["policy"].items() if b != "<=5")
    r.update({
        "tiers_max_depth": {t: [lo, hi - 1] for t, (lo, hi) in TIER_EDGES.items()},
        "policy_min_depth": m90["policy_min_depth"], "max_pieces": MAX_PIECES,
        "totals": {t: {lab: sum(v.values()) for lab, v in c[t].items()} for t in c},
        "value_only_le5_split": {t: {"<=5": c[t]["value_only"]["<=5"],
                                     "6+": sum(v for b, v in c[t]["value_only"].items() if b != "<=5")}
                                 for t in c},
        "t20_policy_outside_le5": t20_pol_gt5,
        "t20_threshold": T20_MIN_ROWS,
        "t20_included": t20_pol_gt5 >= T20_MIN_ROWS,
        "seconds": round(time.time() - t0, 1),
    })
    args.out.write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
