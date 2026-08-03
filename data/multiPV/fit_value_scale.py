"""Refit the tanh value scale on the SELECTED subset - and show the trade-off.

DECIDED, 2026-08-02: the scale stays at 290.6806 (labels.VALUE_SCALE). This
script is retained as the analysis that produced that decision and as the tool
for revisiting it, NOT as a step in the pipeline - Pass B no longer requires its
output. See value_scale_report.md for the table it emitted.

The short version of why the refit was rejected: the "overall std 0.5" fit
returns 849.70, which maps a +100cp edge to v=0.117, i.e. a 55.9% expected
score for being a clean pawn up. That is miscalibrated, not merely flat. At
290.6806 the same edge reads 66.6%, which is about right at strong play.
Because `value_cp` stores the raw score, the scale is a collate-time transform
(dataset.revalue) and the decision remains reversible without reconversion.

On 290.6806: it is NOT a fit on our own data. csv_parallel.py documents it as
the Lc0 WDL calibration constant, chosen so that tanh(cp/290.6806) approximates
EXPECTED SCORE. That is a semantic anchor, not a spread target, and the rest of
the engine relies on it: acpl_elo_estimator inverts with 290.6806*atanh(q), and
C_PUCT / value-weight tuning was done against Q-as-expected-score. Rescaling the
value target silently invalidates all of that.

The complication this script exists to expose: a large fraction of the selected
subset has a forced mate in the value block, and those targets are pinned at
+-0.995 regardless of scale. They therefore contribute FIXED variance. Fitting
"overall std = 0.5" with a big mate mass does not spread the centipawn values -
it SHRINKS them until the mates stop overshooting the target. The two
populations must be reported separately or the fitted number is uninterpretable.

So this reports a decision table rather than one number:
  - the empirical mate fraction AND sign split (not assumed 50/50)
  - the variance each population contributes
  - candidate scales (legacy, with-mates fit, cp-only fits at several targets)
  - for each: overall std, cp-only std, saturation share, and what +100cp maps to

Saturation is the metric that actually matters for training: a target at |v|>0.9
sits where tanh' < 0.19, and at +-0.995 where tanh' ~ 0.01, so those samples
contribute almost no gradient however wrong the model is.

Streams the dump once (~14 min, decompression-bound) and JSON-parses only a
random subsample of the lines Pass B would select, so the cp distribution is the
one the dataset will actually contain - not a head-of-file slice (coverage in
this corpus drifts from ~70% in the first 2M lines to ~40% globally).

The raw cp sample is written to disk so re-analysis never needs another pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from labels import VALUE_CLAMP, VALUE_CP_CLIP, VALUE_MATE  # noqa: E402
from pass_a_index import INDEX_DTYPE, iter_lines  # noqa: E402

BUCKETS = [("<=5", 0, 5), ("6-14", 6, 14), ("15-27", 15, 27), (">=28", 28, 32)]
LEGACY_SCALE = 290.6806
SATURATION = 0.9


def cp_values(cps: np.ndarray, scale: float) -> np.ndarray:
    v = np.tanh(np.clip(cps, -VALUE_CP_CLIP, VALUE_CP_CLIP) / scale)
    return np.clip(v, -VALUE_CLAMP, VALUE_CLAMP)


def mate_values(n_pos: int, n_neg: int) -> np.ndarray:
    """Mate targets using the EMPIRICAL sign split, not an assumed 50/50.
    Mates carry most of the variance, so this assumption is load-bearing."""
    return np.concatenate([np.full(n_pos, VALUE_MATE), np.full(n_neg, -VALUE_MATE)])


def combined_std(cps, n_pos, n_neg, scale, include_mates: bool) -> float:
    v = cp_values(cps, scale)
    if include_mates and (n_pos + n_neg):
        v = np.concatenate([v, mate_values(n_pos, n_neg)])
    return float(v.std())


def solve_scale(cps, n_pos, n_neg, target_std, include_mates,
                lo=5.0, hi=20000.0) -> float | None:
    """Bisect for the scale hitting target_std. std is monotonically decreasing
    in scale, so this is exact. Returns None if the target is unreachable."""
    if combined_std(cps, n_pos, n_neg, hi, include_mates) > target_std:
        return None      # even an infinite scale cannot get std this low
    if combined_std(cps, n_pos, n_neg, lo, include_mates) < target_std:
        return None      # even a tiny scale cannot get std this high
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if combined_std(cps, n_pos, n_neg, mid, include_mates) > target_std:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=_HERE / "lichess_db_eval.jsonl.zst")
    ap.add_argument("--index", type=Path, default=_HERE / "index" / "pass_a_index.bin")
    ap.add_argument("--value-min-depth", type=int, default=26)
    ap.add_argument("--target", type=int, default=30_000_000)
    ap.add_argument("--shares", type=str,
                    default='{"<=5":0.01,"6-14":0.35,"15-27":0.40,">=28":0.24}')
    ap.add_argument("--sample", type=int, default=2_000_000)
    ap.add_argument("--target-std", type=float, default=0.5,
                    help="overall std target for the with-mates fit")
    ap.add_argument("--cp-target-stds", type=float, nargs="+",
                    default=[0.30, 0.35, 0.40, 0.50],
                    help="cp-only std targets to tabulate")
    ap.add_argument("--exclude-mates-from-fit", action="store_true",
                    help="emit the cp-only fit as THE answer instead of with-mates")
    ap.add_argument("--cp-target-std", type=float, default=0.35,
                    help="which cp-only target to emit when --exclude-mates-from-fit")
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--cp-dump", type=Path, default=_HERE / "index" / "value_cp_sample.npy",
                    help="raw cp sample, so re-analysis needs no second pass")
    ap.add_argument("--reuse-cp-dump", action="store_true",
                    help="skip the stream and re-analyse a previous --cp-dump")
    ap.add_argument("--report", type=Path, default=_HERE / "value_scale_report.md")
    ap.add_argument("--manifest", type=Path,
                    default=_HERE / "manifests" / "value_scale.json")
    args = ap.parse_args()

    meta_path = args.cp_dump.with_suffix(".meta.json")

    if args.reuse_cp_dump and args.cp_dump.exists():
        arr = np.load(args.cp_dump).astype(np.float64)
        meta = json.loads(meta_path.read_text())
        n_pos, n_neg, n_missing = meta["mate_pos"], meta["mate_neg"], meta["unusable"]
        print(f"reusing {args.cp_dump} ({len(arr):,} cp values)", file=sys.stderr)
    else:
        shares = json.loads(args.shares)
        ix = np.memmap(args.index, dtype=INDEX_DTYPE, mode="r")
        pc = np.asarray(ix["piece_count"])
        md = np.asarray(ix["max_depth"])
        eligible = (pc <= 32) & (md >= args.value_min_depth)

        rates = {}
        for name, lo, hi in BUCKETS:
            a = int((eligible & (pc >= lo) & (pc <= hi)).sum())
            rates[name] = min(args.target * shares.get(name, 0.0) / a, 1.0) if a else 0.0

        rng = np.random.default_rng(args.seed)
        u = rng.random(len(pc))
        keep = np.zeros(len(pc), dtype=bool)
        for name, lo, hi in BUCKETS:
            if rates[name] > 0:
                keep |= eligible & (pc >= lo) & (pc <= hi) & (u < rates[name])
        sel = np.nonzero(keep)[0]
        if len(sel) > args.sample:
            sel = np.sort(rng.choice(sel, size=args.sample, replace=False))
        del pc, md, eligible, keep, u
        print(f"parsing {len(sel):,} selected lines", file=sys.stderr)

        want = set(int(x) for x in sel)
        cps: list[int] = []
        n_pos = n_neg = n_missing = 0
        ln = 0
        remaining = len(want)
        t0 = time.time()
        for _off, raw in iter_lines(args.source):
            cur = ln
            ln += 1
            if cur not in want:
                continue
            remaining -= 1
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            best, bd = None, -1
            for ev in rec.get("evals") or []:
                d = int(ev.get("depth", -1))
                if d >= args.value_min_depth and d > bd and (ev.get("pvs") or []):
                    best, bd = ev, d
            if best is None:
                continue
            pv0 = best["pvs"][0]
            if pv0.get("cp") is not None:
                cps.append(int(pv0["cp"]))
            elif pv0.get("mate") is not None and int(pv0["mate"]) != 0:
                if int(pv0["mate"]) > 0:
                    n_pos += 1
                else:
                    n_neg += 1
            else:
                n_missing += 1
            if remaining <= 0:
                break
            if ln % 50_000_000 == 0:
                print(f"  src {ln:,} | cp {len(cps):,} | {time.time()-t0:.0f}s",
                      file=sys.stderr, flush=True)

        arr = np.asarray(cps, dtype=np.float64)
        args.cp_dump.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.cp_dump, np.asarray(cps, dtype=np.int32))
        meta_path.write_text(json.dumps({
            "mate_pos": n_pos, "mate_neg": n_neg, "unusable": n_missing,
            "value_min_depth": args.value_min_depth, "seed": args.seed}, indent=2))

    n_mate = n_pos + n_neg
    n_total = len(arr) + n_mate
    f_mate = n_mate / n_total
    split = n_pos / n_mate if n_mate else 0.0
    print(f"collected {len(arr):,} cp, {n_mate:,} mates "
          f"({n_pos:,}+ / {n_neg:,}-), {n_missing:,} unusable", file=sys.stderr)

    # --- variance decomposition at the with-mates fit ---
    s_mates = solve_scale(arr, n_pos, n_neg, args.target_std, True)
    floor_std = combined_std(arr, n_pos, n_neg, 1e9, True)

    cp_only_scales = {}
    for t in args.cp_target_stds:
        cp_only_scales[t] = solve_scale(arr, n_pos, n_neg, t, False)

    candidates = [("legacy (Lc0 WDL calibration)", LEGACY_SCALE)]
    if s_mates:
        candidates.append((f"fit incl. mates -> overall std {args.target_std}", s_mates))
    for t, s in cp_only_scales.items():
        if s:
            candidates.append((f"fit cp-only -> cp std {t:g}", s))

    def row(scale):
        v_cp = cp_values(arr, scale)
        v_all = np.concatenate([v_cp, mate_values(n_pos, n_neg)]) if n_mate else v_cp
        sat_cp = float((np.abs(v_cp) > SATURATION).mean())
        sat_all = float((np.abs(v_all) > SATURATION).mean())
        return dict(scale=scale, overall_std=float(v_all.std()),
                    cp_std=float(v_cp.std()), sat_cp=sat_cp, sat_all=sat_all,
                    pawn=float(np.tanh(100 / scale)),
                    p95=float(np.tanh(496 / scale)))

    rows = [(label, row(s)) for label, s in candidates]

    n_zero = int((arr == 0).sum())
    chosen = (cp_only_scales.get(args.cp_target_std) if args.exclude_mates_from_fit
              else s_mates) or LEGACY_SCALE

    L = [
        "# Value scale refit - decision table",
        "",
        f"- sample: **{n_total:,}** selected positions "
        f"({len(arr):,} cp / {n_mate:,} mate / {n_missing:,} unusable)",
        f"- value_min_depth: {args.value_min_depth}",
        "",
        "## The complication: mates dominate the variance",
        "",
        f"- mate fraction: **{100*f_mate:.2f}%** of value targets",
        f"- empirical mate sign split: **{100*split:.2f}% positive** "
        f"({n_pos:,}+ / {n_neg:,}-) - measured, not assumed 50/50",
        f"- mates are pinned at +-{VALUE_MATE} regardless of scale, so they "
        f"contribute FIXED variance {f_mate*VALUE_MATE**2:.4f} "
        f"= **{100*f_mate*VALUE_MATE**2/args.target_std**2:.1f}% of the "
        f"{args.target_std} target's budget** from {100*f_mate:.1f}% of the data",
        f"- floor: as scale -> infinity the overall std cannot fall below "
        f"**{floor_std:.4f}** (mates alone)",
        "",
        "Fitting *overall* std therefore does not spread the centipawn values, it "
        "shrinks them until the mates stop overshooting. Read the `cp std` column, "
        "not just `overall std`.",
        "",
        "## Decision table",
        "",
        "| fit | scale | overall std | cp-only std | +100cp | +496cp (p95) | "
        "sat. cp | sat. all |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in rows:
        L.append(f"| {label} | **{r['scale']:.2f}** | {r['overall_std']:.3f} | "
                 f"{r['cp_std']:.3f} | {r['pawn']:+.3f} | {r['p95']:+.3f} | "
                 f"{100*r['sat_cp']:.1f}% | {100*r['sat_all']:.1f}% |")
    L += [
        "",
        f"`sat.` = share of targets with |v| > {SATURATION}, where tanh' < 0.19 and "
        "the sample contributes almost no gradient. `sat. all` includes the mate "
        f"mass, which is {100*f_mate:.1f}% and saturated at ANY scale.",
        "",
        "## Downstream consequences of changing the scale",
        "",
        f"`{LEGACY_SCALE}` is the Lc0 WDL calibration - it makes v ~ expected score. "
        "It was NOT fitted on our data. Changing it means:",
        "",
        "- `benchmarking/player/acpl_elo_estimator.py` inverts with "
        f"`{LEGACY_SCALE}*atanh(q)`; it would need the new constant or every ACPL "
        "Elo estimate is wrong by the ratio.",
        "- C_PUCT (2.0) and the value weight (2.5x) were tuned against "
        "Q-as-expected-score; rescaling Q changes what those mean.",
        "- `playing/uci_wrapper.py` reports `int(q*1000)`, a third mapping already "
        "inconsistent with both. Worth unifying whatever is chosen.",
        "",
        "## Raw cp distribution (pre-tanh)",
        "",
        "| quantile | cp |", "|---|---:|",
    ]
    for q in (1, 5, 25, 50, 75, 95, 99):
        L.append(f"| p{q} | {np.percentile(arr, q):,.0f} |")
    L += [
        "",
        f"- exactly 0: **{n_zero:,}** ({100*n_zero/len(arr):.2f}%) - dead-drawn "
        "positions. This atom lands in the `[0.00,+0.10)` histogram bin, which is "
        "why that bin looks lopsided; it is mostly binning, not a White bias.",
        f"- mean cp {arr.mean():+.1f}, median {np.median(arr):+.1f} "
        f"(a mild genuine White skew remains on top of the zero atom)",
        f"- |cp| >= {VALUE_CP_CLIP} (clipped): {int((np.abs(arr) >= VALUE_CP_CLIP).sum()):,} "
        f"({100*float((np.abs(arr) >= VALUE_CP_CLIP).mean()):.3f}%)",
        "",
        f"## Emitted scale: **{chosen:.4f}**",
        "",
        ("Selected with --exclude-mates-from-fit "
         f"(cp-only, target cp std {args.cp_target_std:g})."
         if args.exclude_mates_from_fit else
         f"Selected as the with-mates fit to overall std {args.target_std:g}. "
         "Pass --exclude-mates-from-fit to emit the cp-only fit instead."),
        "",
        "## Value target histogram at the emitted scale (cp only)",
        "",
        "| bin | count |", "|---|---:|",
    ]
    hist, edges = np.histogram(cp_values(arr, chosen), bins=20, range=(-1, 1))
    for i, c in enumerate(hist):
        L.append(f"| [{edges[i]:+.2f}, {edges[i+1]:+.2f}) | {int(c):,} |")

    args.report.write_text("\n".join(L), encoding="utf-8")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({
        "emitted_scale": chosen,
        "fit_mode": "cp_only" if args.exclude_mates_from_fit else "with_mates",
        "legacy_scale": LEGACY_SCALE,
        "legacy_provenance": "Lc0 WDL calibration constant (expected score); "
                             "NOT fitted on GuoFish data",
        "sample_total": n_total, "sample_cp": len(arr), "sample_mate": n_mate,
        "mate_fraction": f_mate, "mate_sign_split_positive": split,
        "mate_variance_share_of_target": f_mate * VALUE_MATE ** 2 / args.target_std ** 2,
        "overall_std_floor": floor_std,
        "cp_exactly_zero": n_zero,
        "candidates": {label: r for label, r in rows},
        "value_min_depth": args.value_min_depth,
        "cp_sample_dump": str(args.cp_dump),
        "note": "mates are pinned at +-0.995 and carry fixed variance; fitting "
                "overall std shrinks cp rather than spreading it. See report.",
    }, indent=2, default=float), encoding="utf-8")

    print("\n".join(L))
    print(f"\nEMITTED SCALE = {chosen:.4f}", file=sys.stderr)
    print(f"pass to Pass B as: --value-scale {chosen:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
