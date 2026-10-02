"""Synthetic shards for the CPU dry run (VM harness brief §8), carved from real
records: frozen90 v2 rows [0, n_train) become a `train` split and the next n_val
rows a frozen `val` split, each with a manifest and a strata sidecar (the source
codes, sliced), plus a sha256.txt so `setup.sh --data-prefix` can pull the lot.
It also writes synth_v6_flip: the same train rows with value and value_cp
negated. Training on it (value loss up-weighted) makes frozen-val MSE worse from
branch to branch, which drives the stop rule's "worse total" path (its strata
keep the source codes, so the material field is stale there; nothing reads it).

    python -m training.v6.tools.make_synth <out> [--n-train 16384] [--n-val 2048]

<out> is shaped like data/: <out>/processed/{synth_v6,synth_v6_flip,synth_v6_val,strata}/
and <out>/sha256.txt (`sha256  size  relative_path`, relative to <out>).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from training.v6.ckpt import sha256_file, utc
from training.v6.data.formats import REPO
from training.v6.data.reader import ShardSet
from training.v6.data.strata import DEFINITION, DEFINITION_HASH, counts

SRC = REPO / "data/processed/val_frozen_90m_v2"
SRC_STRATA = REPO / "data/processed/strata/val_frozen_90m_v2_val.strata2.npy"


def write_split(root: Path, name: str, split: str, rec: np.ndarray, codes: np.ndarray, src_man: dict) -> None:
    d = root / "processed" / name
    d.mkdir(parents=True)
    shard = d / f"{split}_0000.bin"
    rec.tofile(shard)
    man = {"created_utc": utc(), "name": name, "synthetic_from": "val_frozen_90m_v2",
           **{k: src_man[k] for k in ("record_format", "record_dtype", "record_size_bytes")},
           "records_total": len(rec),
           "shards": [{"name": shard.name, "records": len(rec), "sha256": sha256_file(shard)}]}
    (d / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    sp = root / "processed" / "strata" / f"{name}_{split}.strata2.npy"
    sp.parent.mkdir(parents=True, exist_ok=True)
    codes = np.ascontiguousarray(codes)
    np.save(sp, codes)
    meta = {"created_utc": utc(), "definition_hash": DEFINITION_HASH, "definition": DEFINITION,
            "source_dir": f"data/processed/{name}", "split": split,
            "source_manifest_sha256": sha256_file(d / "manifest.json"), "record_format": man["record_format"],
            "n_records": len(rec), "codes_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
            "counts": counts(codes)}
    sp.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def build(out: Path, n_train: int, n_val: int) -> None:
    if (out / "processed").exists():
        raise SystemExit(f"{out / 'processed'} exists; refusing to overwrite")
    ss = ShardSet(SRC, "val")
    rec = ss.read_raw(np.arange(n_train + n_val))
    ss.close()
    codes = np.asarray(np.load(SRC_STRATA, mmap_mode="r")[:n_train + n_val])
    man = json.loads((SRC / "manifest.json").read_text())
    write_split(out, "synth_v6", "train", rec[:n_train], codes[:n_train], man)
    flip = rec[:n_train].copy()
    flip["value"], flip["value_cp"] = -flip["value"], -flip["value_cp"]
    write_split(out, "synth_v6_flip", "train", flip, codes[:n_train], man)
    write_split(out, "synth_v6_val", "val", rec[n_train:], codes[n_train:], man)
    files = sorted(p for p in (out / "processed").rglob("*") if p.is_file())
    (out / "sha256.txt").write_text("".join(
        f"{sha256_file(p)}  {p.stat().st_size}  {p.relative_to(out).as_posix()}\n" for p in files))
    print(f"wrote {len(files)} files under {out} ({n_train:,} train, {n_val:,} val records)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n-train", type=int, default=16384)
    ap.add_argument("--n-val", type=int, default=2048)
    a = ap.parse_args(argv)
    build(a.out, a.n_train, a.n_val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
