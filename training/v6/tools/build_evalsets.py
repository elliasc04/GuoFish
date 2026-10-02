"""Named eval-set sidecars (H3): data/processed/evalsets/<name>.json.

    CUDA_VISIBLE_DEVICES=-1 python -m training.v6.tools.build_evalsets

    frozen90       val_frozen_90m_v2 (452,405; the v1 records plus hard_move, S6)
    v2val_roots    corpus v2 `val` (571,232 roots: frozen90 + 118,827 new-tier and 15-27 additions)
    v2val_derived  corpus v2 `valderived` (161,340 PV-unrolled records)

The records stay where they are. A sidecar pins every shard's sha256, the
manifest's sha256 and the strata sidecar (checked against that manifest), so
`eval.load_evalset` refuses a changed shard. An existing sidecar is never
overwritten.
"""
from __future__ import annotations

import json

import numpy as np

from training.v6.ckpt import git_state, sha256_file, utc
from training.v6.data.formats import REPO
from training.v6.data.reader import ShardSet
from training.v6.data.strata import counts, load_strata
from training.v6.eval import EVALSETS

SETS = {  # name: (shard dir, split, strata sidecar, manifest key with the expected count)
    "frozen90": ("data/processed/val_frozen_90m_v2", "val",
                 "data/processed/strata/val_frozen_90m_v2_val.strata2.npy", "records_total"),
    "v2val_roots": ("data/processed/multipv_v2", "val",
                    "data/processed/strata/multipv_v2_val.strata2.npy", "records_val"),
    "v2val_derived": ("data/processed/multipv_v2", "valderived",
                      "data/processed/strata/multipv_v2_valderived.strata2.npy", "records_valderived"),
}


def build(name: str) -> dict:
    shard_dir, split, strata, count_key = SETS[name]
    manifest = f"{shard_dir}/manifest.json"
    ss = ShardSet(REPO / shard_dir, split, REPO / manifest)
    if len(ss) != int(ss.manifest[count_key]):
        raise SystemExit(f"{name}: {len(ss):,} records, manifest {count_key} says {ss.manifest[count_key]:,}")
    msha = sha256_file(REPO / manifest)
    codes = np.asarray(load_strata(REPO / strata, len(ss), msha))
    meta = json.loads((REPO / strata).with_suffix(".json").read_text())
    return {"name": name, "created_utc": utc(), "git_sha": git_state()["git_sha"],
            "shard_dir": shard_dir, "split": split, "manifest": manifest, "manifest_sha256": msha,
            "record_format": ss.format, "n_records": len(ss),
            "shards": [{"name": p.name, "records": int(n), "sha256": sha256_file(p)}
                       for p, n in zip(ss.paths, np.diff(ss.offsets))],
            "strata": strata, "strata_codes_sha256": meta["codes_sha256"], "counts": counts(codes)}


def main() -> int:
    EVALSETS.mkdir(parents=True, exist_ok=True)
    for name in SETS:
        out = EVALSETS / f"{name}.json"
        if out.exists():
            raise SystemExit(f"{out} exists; refusing to overwrite")
    for name in SETS:
        spec = build(name)
        (EVALSETS / f"{name}.json").write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
        print(f"{name}: {spec['n_records']:,} records, {len(spec['shards'])} shards, "
              f"label {spec['counts']['label']}, tier {spec['counts']['depth_tier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
