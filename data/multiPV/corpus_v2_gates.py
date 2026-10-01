"""Post-build gates and counts for a pass_b_v2 corpus (v2, and v3 per the corpus v3 brief).

    python data/multiPV/corpus_v2_gates.py s9 [--corpus data/processed/multipv_v2]
        [--manifest-copy M | --manifest-copy none] [--expect-counts C] [--smoke] [--out J]
    python data/multiPV/corpus_v2_gates.py counts [--corpus ...] [--frozen-v2 ... | none]

s9      reads <corpus>/manifest.json and checks: the manifest copy (if any) is
        byte-identical; nesting was checked over the whole index with 0 of the
        91,350,634 replayed 90M lines dropped (and 0 of the nest manifest's, if
        the build had one); the selection equals --expect-counts; hard_move on
        >= 99% of roots; rejection and dedup counts present; shard counts add up;
        roots + rejections = selected lines; then re-runs the builder into the
        same directory (with the manifest's own source, index and rate plan)
        and requires the H1 refusal. --smoke (a --limit build): the whole-index
        checks are reported as null and only "0 dropped" is required. Exit 1 on
        any fail; --out also writes the JSON result.
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


def s9(corpus: Path, copy: Path | None, expect_counts: Path | None = None, smoke: bool = False,
       out: Path | None = None, floor_replayed: int = 91_350_634) -> int:
    man_path = corpus / "manifest.json"
    m = json.loads(man_path.read_text())
    c, ns = m["counters"], m["nesting"]
    shard_total = sum(sum(v) for v in m["shard_counts"].values())
    want = json.loads(expect_counts.read_text()) if expect_counts is not None else None
    checks = {
        "manifest_copy_identical": None if copy is None else copy.read_bytes() == man_path.read_bytes(),
        "nesting_whole_index": None if smoke else ns["scope"] == "whole index",
        "nesting_90m_replayed_all": None if smoke else ns["floor_rows_replayed"] == floor_replayed,
        "nesting_0_dropped": ns["floor_rows_dropped"] == 0,
        "nesting_nest_0_dropped": ns["nest_rows_dropped"] == 0 if "nest_rows_dropped" in ns else None,
        "selection_equals_expected_counts": None if want is None else (
            [want["selected_lines"], want["selected_counts"]] == [m["selected_lines"], m["selected_counts"]]),
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
                        "--source", m["source"], "--index", m["index"], "--out-dir", str(corpus)],
                       cwd=REPO, capture_output=True, text=True)
    checks["second_run_refused_H1"] = r.returncode != 0 and "refusing (H1)" in r.stderr
    res = {"checks": checks, "passed": all(v is not False for v in checks.values()),
           "not_applicable": sorted(k for k, v in checks.items() if v is None),
           "manifest_sha256": hashlib.sha256(man_path.read_bytes()).hexdigest(),
           "hard_move_coverage": m["hard_move_coverage"], "hard_move_failures": m["hard_move_failures"],
           "rejection_histogram": m["rejection_histogram"],
           "second_run_stderr": r.stderr.strip()[-200:]}
    print(json.dumps(res, indent=1))
    if out is not None:
        out.write_text(json.dumps(res, indent=1) + "\n")
    return 0 if res["passed"] else 1


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
    codes = {s: _codes(strata_dir, corpus.name, s, k, corpus / "manifest.json")
             for s, k in n.items() if k}                 # v3 has no valderived (--derived-rate 0)
    if "train" not in codes or "val" not in codes:
        raise SystemExit(f"{corpus}: empty train or val split")
    if frozen_v2 is not None:
        fz = json.loads((frozen_v2 / "manifest.json").read_text())
        codes["frozen_v2"] = _codes(strata_dir, frozen_v2.name, "val", fz["records_total"],
                                    frozen_v2 / "manifest.json")
    roots = np.concatenate([codes["train"][field_of(codes["train"], "origin") == 0], codes["val"]])
    tier, lab, bk = field_of(roots, "depth_tier"), field_of(roots, "label"), field_of(roots, "bucket")
    tiers = list(m["selected_counts"])
    actual = {t: {k: {} for k in ("policy", "value_only")} for t in tiers}
    for t in tiers:
        ti = STRATA_FIELDS["depth_tier"].index(t)
        for bi, b in enumerate(STRATA_FIELDS["bucket"]):
            cell = (tier == ti) & (bk == bi)
            actual[t]["policy"][BUCKET_NAMES[b]] = int((cell & (lab == 0)).sum())
            actual[t]["value_only"][BUCKET_NAMES[b]] = int((cell & (lab > 0)).sum())
    sel = m["selected_counts"]
    out = {
        "roots": int(len(roots)),
        "roots_tier_v1": int((tier == STRATA_FIELDS["depth_tier"].index("v1")).sum()),
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
    ap.add_argument("--manifest-copy", default=str(HERE / "manifests/dataset_manifest_v2.json"),
                    help="'none' skips the copy check (a VM build writes no copy into the checkout)")
    ap.add_argument("--frozen-v2", default=str(REPO / "data/processed/val_frozen_90m_v2"), help="or 'none'")
    ap.add_argument("--strata-dir", type=Path, default=STRATA)
    ap.add_argument("--expect-counts", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true", help="a --limit build: whole-index checks n/a")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--floor-replayed", type=int, default=91_350_634, help="the 90M selected_lines")
    args = ap.parse_args(argv)
    opt = lambda s: None if s == "none" else Path(s)  # noqa: E731
    if args.what == "s9":
        return s9(args.corpus, opt(args.manifest_copy), args.expect_counts, args.smoke, args.out,
                  args.floor_replayed)
    return counts(args.corpus, opt(args.frozen_v2), args.strata_dir)


if __name__ == "__main__":
    raise SystemExit(main())
