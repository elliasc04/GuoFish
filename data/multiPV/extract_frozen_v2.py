"""Re-materialise the frozen 90M val set in v2 format (v6 design doc §6.5 steps 2-3; gate S6).

    python data/multiPV/extract_frozen_v2.py --corpus data/processed/multipv_v2 \
        --out data/processed/val_frozen_90m_v2

Replays the 90M selection (the floor manifest's seed and rates, value_min_depth
26, same Pass A index) and keeps the corpus-v2 val records whose src_line it
selects. Those are routed exactly as the 90M build routed its val records
(shard = splitmix64(line ^ seed) % n_val, source order within a shard), so the
output lines up record for record with val_frozen_90m_v1. Then it verifies:
the count equals the frozen v1 count (452,405), per-shard counts match, and
every field shared with v1 is byte-identical, pv_score compared after
conversion to float16. Any mismatch deletes nothing but exits 1 with the
per-field counts. --out must be new or empty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pass_b_convert import _splitmix64, build_selection  # noqa: E402
from record_format import shard_name  # noqa: E402
from training.v6.data.formats import V1_DTYPE, V2_DTYPE, dtype_descr  # noqa: E402
from training.v6.data.reader import ShardSet  # noqa: E402


def replay_90m(index: Path, m90: dict) -> np.ndarray:
    cfg = {"value_min_depth": m90["value_min_depth"], "policy_min_depth": m90["policy_min_depth"],
           "bucket_rates_policy": m90["bucket_sampling_rates_policy"],
           "bucket_rates_value_only": m90["bucket_sampling_rates_value_only"]}
    sel = build_selection(index, cfg, m90["seed"])
    if len(sel) != m90["selected_lines"]:
        raise SystemExit(f"replayed {len(sel):,} lines, manifest says {m90['selected_lines']:,}")
    return sel


def extract(corpus: Path, sel90: np.ndarray, seed: int, n_val: int) -> list[np.ndarray]:
    ss = ShardSet(corpus, "val")
    if ss.format != "v2":
        raise SystemExit(f"{corpus} is {ss.format}, expected v2")
    kept = []
    for s in range(0, len(ss), 1 << 18):
        rec = ss.read_raw(np.arange(s, min(len(ss), s + (1 << 18))))
        kept.append(rec[np.isin(rec["src_line"].astype(np.int64), sel90)])
    ss.close()
    rec = np.concatenate(kept) if kept else np.zeros(0, V2_DTYPE)
    if (rec["origin"] != 0).any():
        raise SystemExit("derived records found in val shards")
    route = np.array([_splitmix64(int(ln) ^ seed) % n_val for ln in rec["src_line"]], dtype=np.int64)
    out = []
    for j in range(n_val):
        part = rec[route == j]
        out.append(part[np.argsort(part["src_line"], kind="stable")])
    return out


def verify(shards_v2: list[np.ndarray], frozen_v1: Path) -> dict:
    v1 = ShardSet(frozen_v1, "val")
    counts_v1 = np.diff(v1.offsets).tolist()
    counts_v2 = [len(s) for s in shards_v2]
    report = {"count_v1": len(v1), "count_v2": sum(counts_v2),
              "shard_counts_match": counts_v1 == counts_v2, "field_mismatches": {}}
    if report["shard_counts_match"]:
        a = v1.read_raw(np.arange(len(v1)))
        b = np.concatenate(shards_v2)
        def raw(x):                          # per-row bytes: -0.0 != 0.0 here, as it should be
            return np.ascontiguousarray(x).view(np.uint8).reshape(len(x), -1)
        for name in V1_DTYPE.names:
            y = b[name].astype(np.float16) if name == "pv_score" else b[name]
            bad = ~np.all(raw(a[name]) == raw(y), axis=1)
            report["field_mismatches"][name] = int(bad.sum())
    v1.close()
    report["passed"] = (report["shard_counts_match"] and report["count_v1"] == report["count_v2"]
                        and not any(report["field_mismatches"].values()))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frozen-v1", type=Path, default=_ROOT / "data/processed/val_frozen_90m_v1")
    ap.add_argument("--index", type=Path, default=_HERE / "index" / "pass_a_index.bin")
    ap.add_argument("--m90", type=Path, default=_HERE / "manifests" / "dataset_manifest_90m.json")
    ap.add_argument("--expect", type=int, default=452_405)
    args = ap.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is not empty")
    m90 = json.loads(args.m90.read_text())
    sel90 = replay_90m(args.index, m90)
    shards = extract(args.corpus, sel90, m90["seed"], m90["n_val_shards"])
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for j, part in enumerate(shards):
        p = args.out / shard_name("val", j)
        part.tofile(p)
        entries.append({"name": p.name, "records": len(part),
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    report = verify(shards, args.frozen_v1)
    report["expected"] = args.expect
    report["passed"] = report["passed"] and report["count_v2"] == args.expect
    (args.out / "manifest.json").write_text(json.dumps({
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "name": args.out.name, "source_corpus": str(args.corpus), "record_format": "v2",
        "record_dtype": dtype_descr(V2_DTYPE), "record_size_bytes": V2_DTYPE.itemsize,
        "records_total": report["count_v2"], "shards": entries, "s6": report}, indent=2) + "\n")
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
