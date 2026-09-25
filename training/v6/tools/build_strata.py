"""Write the stratum sidecar for one corpus split or eval set (§6.6).

    python -m training.v6.tools.build_strata --shards data/processed/val_frozen_90m_v1 \
        --split val --out data/processed/val_frozen_90m_v1/strata_val_v1.npy

Writes <out> (uint16 per record, in ShardSet order) and <out>.json (definition
and its hash, source manifest sha256, per-field counts). One sequential pass.
Refuses to overwrite an existing sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from training.v6.data.reader import ShardSet
from training.v6.data.strata import DEFINITION, DEFINITION_HASH, compute_strata, counts

CHUNK = 1 << 18


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() or args.out.with_suffix(".json").exists():
        raise SystemExit(f"{args.out} (or its .json) exists; refusing to overwrite")

    ss = ShardSet(args.shards, args.split, args.manifest)
    n = len(ss)
    codes = np.empty(n, dtype=np.uint16)
    t0 = time.time()
    for start in range(0, n, CHUNK):
        stop = min(n, start + CHUNK)
        codes[start:stop] = compute_strata(ss.read(np.arange(start, stop)))
    ss.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, codes)
    meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "definition_hash": DEFINITION_HASH,
        "definition": DEFINITION,
        "source_dir": str(args.shards).replace("\\", "/"),
        "split": args.split,
        "source_manifest": str(ss.manifest_path).replace("\\", "/"),
        "source_manifest_sha256": hashlib.sha256(ss.manifest_path.read_bytes()).hexdigest(),
        "record_format": ss.format,
        "n_records": n,
        "codes_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
        "counts": counts(codes),
        "seconds": round(time.time() - t0, 2),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("n_records", "record_format", "counts", "seconds")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
