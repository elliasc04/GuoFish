"""§6.5 step 1: copy the 90M val shards out of the live corpus directory.

    python -m training.v6.tools.freeze_val

Copies data/processed/multipv_90m/val_*.bin (12 shards, 452,405 records) to
data/processed/val_frozen_90m_v1/ and writes manifest.json with each file's
sha256. The source and the copy are hashed independently; any mismatch, a
wrong shard or record count, or a non-empty destination is a hard error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from training.v6.data.formats import REPO, V1_DTYPE, dtype_descr

N_SHARDS = 12
N_RECORDS = 452_405


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=REPO / "data/processed/multipv_90m")
    ap.add_argument("--dst", type=Path, default=REPO / "data/processed/val_frozen_90m_v1")
    args = ap.parse_args()

    shards = sorted(args.src.glob("val_*.bin"))
    if len(shards) != N_SHARDS:
        raise SystemExit(f"expected {N_SHARDS} val shards in {args.src}, found {len(shards)}")
    sizes = [p.stat().st_size for p in shards]
    bad = [p.name for p, s in zip(shards, sizes) if s % V1_DTYPE.itemsize]
    if bad:
        raise SystemExit(f"not whole v1 records: {bad}")
    n = sum(sizes) // V1_DTYPE.itemsize
    if n != N_RECORDS:
        raise SystemExit(f"expected {N_RECORDS:,} records, found {n:,}")

    if args.dst.exists() and any(args.dst.iterdir()):
        raise SystemExit(f"{args.dst} exists and is not empty; refusing to overwrite")
    args.dst.mkdir(parents=True, exist_ok=True)

    entries = []
    for p, size in zip(shards, sizes):
        src_hash = sha256_file(p)
        out = args.dst / p.name
        shutil.copyfile(p, out)
        dst_hash = sha256_file(out)
        if dst_hash != src_hash:
            raise SystemExit(f"sha256 mismatch for {p.name}: {src_hash} vs {dst_hash}")
        entries.append({"name": p.name, "bytes": size,
                        "records": size // V1_DTYPE.itemsize, "sha256": dst_hash})
        print(f"{p.name}  {size // V1_DTYPE.itemsize:>7,} records  {dst_hash}")

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "name": "val_frozen_90m_v1",
        "source_dir": str(args.src.relative_to(REPO)).replace("\\", "/"),
        "record_format": "v1",
        "record_dtype": dtype_descr(V1_DTYPE),
        "record_size_bytes": V1_DTYPE.itemsize,
        "records_total": n,
        "shards": entries,
    }
    (args.dst / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{n:,} records in {len(entries)} shards -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
