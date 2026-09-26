"""Post-build gates and counts for corpus v2 (brief steps 6-7; gate S9 in full).

    python data/multiPV/corpus_v2_gates.py s9 [--corpus data/processed/multipv_v2]
    python data/multiPV/corpus_v2_gates.py counts [--corpus ...] [--frozen-v2 ...]

s9      reads the manifest and checks: the manifest copy is byte-identical;
        nesting was checked over the whole index with 0 of the 91,350,634
        replayed 90M lines dropped; hard_move on >= 99% of roots; rejection
        and dedup counts present; shard counts add up; then re-runs the builder
        into the same directory and requires the H1 refusal. Exit 1 on any fail.
counts  from the strata2 sidecars: roots by tier x label x bucket against the
        manifest's selected counts, the realised policy share, depth_tier x
        label, material classes and in_90m per split. Prints JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from training.v6.config.schema import STRATA_FIELDS  # noqa: E402
from training.v6.data.strata import field_of, load_strata  # noqa: E402

STRATA = REPO / "data/processed/strata"
BUCKET_NAMES = {"le5": "<=5", "6_14": "6-14", "15_27": "15-27", "ge28": ">=28"}


def s9(corpus: Path, copy: Path) -> int:
    man_path = corpus / "manifest.json"
    m = json.loads(man_path.read_text())
    c = m["counters"]
    shard_total = sum(sum(v) for v in m["shard_counts"].values())
    checks = {
        "manifest_copy_identical": copy.read_bytes() == man_path.read_bytes(),
        "nesting_whole_index": m["nesting"]["scope"] == "whole index",
        "nesting_90m_replayed_91350634": m["nesting"]["floor_rows_replayed"] == 91_350_634,
        "nesting_0_dropped": m["nesting"]["floor_rows_dropped"] == 0,
        "hard_move_coverage_ge_99pct": m["hard_move_coverage"] >= 0.99,
        "rejection_histogram_present": bool(m["rejection_histogram"]),
        "dedup_counts_present": all(k in c for k in ("derived_dropped_root_duplicate",
                                                    "derived_dropped_derived_duplicate")),
        "invariant_violation_rate_le_0.1pct": m["invariant_violation_rate"] <= 0.001,
        "records_add_up": shard_total == m["records_train"] + m["records_val"] + m["records_valderived"]
                          == m["roots"] + c["derived_written_train"] + c["derived_written_valderived"],
        "roots_plus_rejections_eq_selected":
            m["roots"] + sum(m["rejection_histogram"].values()) == m["selected_lines"],
    }
    r = subprocess.run([sys.executable, str(HERE / "pass_b_v2.py"), "--rate-plan", m["rate_plan_file"],
                        "--out-dir", str(corpus)], cwd=REPO, capture_output=True, text=True)
    checks["second_run_refused_H1"] = r.returncode != 0 and "refusing (H1)" in r.stderr
    out = {"checks": checks, "passed": all(checks.values()),
           "manifest_sha256": hashlib.sha256(man_path.read_bytes()).hexdigest(),
           "hard_move_coverage": m["hard_move_coverage"], "hard_move_failures": m["hard_move_failures"],
           "rejection_histogram": m["rejection_histogram"],
           "second_run_stderr": r.stderr.strip()[-200:]}
    print(json.dumps(out, indent=1))
    return 0 if out["passed"] else 1


def _codes(strata_dir: Path, corpus_name: str, split: str, n: int, manifest: Path):
    path = strata_dir / f"{corpus_name}_{split}.strata2.npy"
    return np.asarray(load_strata(path, n, hashlib.sha256(manifest.read_bytes()).hexdigest()))


def _tab(codes, a: str, b: str) -> dict:
    fa, fb = field_of(codes, a), field_of(codes, b)
    return {str(x): {str(y): int(((fa == i) & (fb == j)).sum()) for j, y in enumerate(STRATA_FIELDS[b])}
            for i, x in enumerate(STRATA_FIELDS[a])}


def counts(corpus: Path, frozen_v2: Path, strata_dir: Path) -> int:
    m = json.loads((corpus / "manifest.json").read_text())
    n = {"train": m["records_train"], "val": m["records_val"], "valderived": m["records_valderived"]}
    codes = {s: _codes(strata_dir, corpus.name, s, k, corpus / "manifest.json") for s, k in n.items()}
    fz = json.loads((frozen_v2 / "manifest.json").read_text())
    codes["frozen_v2"] = _codes(strata_dir, frozen_v2.name, "val", fz["records_total"], frozen_v2 / "manifest.json")
    roots = np.concatenate([codes["train"][field_of(codes["train"], "origin") == 0], codes["val"]])
    tier, lab, bk = field_of(roots, "depth_tier"), field_of(roots, "label"), field_of(roots, "bucket")
    actual = {t: {k: {} for k in ("policy", "value_only")} for t in ("old", "new")}
    for ti, t in enumerate(("old", "new")):
        for bi, b in enumerate(STRATA_FIELDS["bucket"]):
            cell = (tier == ti) & (bk == bi)
            actual[t]["policy"][BUCKET_NAMES[b]] = int((cell & (lab == 0)).sum())
            actual[t]["value_only"][BUCKET_NAMES[b]] = int((cell & (lab > 0)).sum())
    sel = m["selected_counts"]
    out = {
        "roots": int(len(roots)), "roots_tier_v1": int((tier == 2).sum()),
        "selected_vs_actual_roots": {t: {k: {b: [sel[t][k][b], actual[t][k][b]] for b in sel[t][k]}
                                         for k in sel[t]} for t in sel},
        "realised_policy_share_roots": float((lab == 0).mean()),
        "per_split": {s: {"n": int(len(c)), "depth_tier_x_label": _tab(c, "depth_tier", "label"),
                          "origin": {o: int((field_of(c, "origin") == i).sum())
                                     for i, o in enumerate(STRATA_FIELDS["origin"])},
                          "material": {x: int((field_of(c, "material") == i).sum())
                                       for i, x in enumerate(STRATA_FIELDS["material"])},
                          "material_x_label": _tab(c, "material", "label"),
                          "in_90m_x_origin": _tab(c, "in_90m", "origin"),
                          "in_90m_x_depth_tier": _tab(c, "in_90m", "depth_tier")}
                      for s, c in codes.items()},
    }
    print(json.dumps(out, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("s9", "counts"))
    ap.add_argument("--corpus", type=Path, default=REPO / "data/processed/multipv_v2")
    ap.add_argument("--manifest-copy", type=Path, default=HERE / "manifests/dataset_manifest_v2.json")
    ap.add_argument("--frozen-v2", type=Path, default=REPO / "data/processed/val_frozen_90m_v2")
    ap.add_argument("--strata-dir", type=Path, default=STRATA)
    args = ap.parse_args(argv)
    return s9(args.corpus, args.manifest_copy) if args.what == "s9" else counts(args.corpus, args.frozen_v2, args.strata_dir)


if __name__ == "__main__":
    raise SystemExit(main())
