"""Write stratum sidecars (§6.6, definition v3) for corpus splits or an eval set.

    python -m training.v6.tools.build_strata --shards data/processed/multipv_v2 \
        --split train val valderived [--manifest M] [--out-dir data/processed/strata]

For each split writes <out-dir>/<shards dir name>_<split>.strata2.npy (uint16
per record, in ShardSet order) and its .strata2.json (definition and hash,
source manifest sha256, per-field counts). One sequential pass per split.
v2 shards also need the Pass A index: depth_tier is its max_depth at
src_line, in_90m replays the 90M selection (--index, --m90). Both are loaded
once and shared by every split. Refuses to overwrite an existing sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from training.v6.data.formats import REPO
from training.v6.data.reader import ShardSet
from training.v6.data.strata import DEFINITION, DEFINITION_HASH, compute_strata, counts

CHUNK = 1 << 18
MPV = REPO / "data" / "multiPV"


def index_context(index: Path, m90: Path) -> tuple[np.ndarray, np.ndarray]:
    """(sorted 90M line numbers, max_depth per index row); ~1.5 GB, two index passes."""
    from extract_frozen_v2 import replay_90m
    from pass_a_index import INDEX_DTYPE
    sel90 = replay_90m(index, json.loads(m90.read_text()))
    ix = np.memmap(index, dtype=INDEX_DTYPE, mode="r")
    md = np.empty(len(ix), dtype=np.int16)
    for s in range(0, len(ix), 20_000_000):
        md[s:s + 20_000_000] = ix["max_depth"][s:s + 20_000_000]
    del ix
    return sel90, md


def build(ss: ShardSet, ctx) -> np.ndarray:
    n = len(ss)
    codes = np.empty(n, dtype=np.uint16)
    for start in range(0, n, CHUNK):
        rec = ss.read(np.arange(start, min(n, start + CHUNK)))
        if ss.format == "v2":
            sel90, md = ctx
            ln = rec["src_line"].astype(np.int64)
            pos = np.minimum(np.searchsorted(sel90, ln), len(sel90) - 1)
            codes[start:start + len(rec)] = compute_strata(rec, md[ln], sel90[pos] == ln)
        else:
            codes[start:start + len(rec)] = compute_strata(rec)
    return codes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--split", nargs="+", required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "processed" / "strata")
    ap.add_argument("--index", type=Path, default=MPV / "index" / "pass_a_index.bin")
    ap.add_argument("--m90", type=Path, default=MPV / "manifests" / "dataset_manifest_90m.json")
    args = ap.parse_args()
    outs = {s: args.out_dir / f"{args.shards.name}_{s}.strata2.npy" for s in args.split}
    for out in outs.values():
        if out.exists() or out.with_suffix(".json").exists():
            raise SystemExit(f"{out} (or its .json) exists; refusing to overwrite")
    sets = {s: ShardSet(args.shards, s, args.manifest) for s in args.split}
    t0 = time.time()
    ctx = (index_context(args.index, args.m90) if any(ss.format == "v2" for ss in sets.values())
           else None)
    t_index = time.time() - t0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, ss in sets.items():
        t0 = time.time()
        codes = build(ss, ctx)
        ss.close()
        np.save(outs[split], codes)
        meta = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "definition_hash": DEFINITION_HASH,
            "definition": DEFINITION,
            "source_dir": str(args.shards).replace("\\", "/"),
            "split": split,
            "source_manifest": str(ss.manifest_path).replace("\\", "/"),
            "source_manifest_sha256": hashlib.sha256(ss.manifest_path.read_bytes()).hexdigest(),
            "record_format": ss.format,
            "index": str(args.index).replace("\\", "/") if ss.format == "v2" else None,
            "m90": str(args.m90).replace("\\", "/") if ss.format == "v2" else None,
            "n_records": len(codes),
            "codes_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
            "counts": counts(codes),
            "seconds": round(time.time() - t0, 2),
            "index_seconds": round(t_index, 2),
        }
        outs[split].with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps({k: meta[k] for k in ("split", "n_records", "record_format", "counts",
                                               "seconds")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
