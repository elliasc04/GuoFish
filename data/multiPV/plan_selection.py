"""Choose depth thresholds and per-bucket sampling rates from the Pass A index.

Consumes the Pass A index (no re-scan of the dump) and answers three questions
the Phase 2 brief leaves to measurement:

  1. Which value_min_depth leaves ~4x the Pass B target in the eligible pool?
     Too low and we keep opening-skewed junk; too high and we select the biased
     "someone parked an engine here" subset, which reintroduces the very skew
     stratification exists to remove.
  2. What per-bucket sampling RATES realise the requested piece-count
     composition, and are they feasible (rate <= 1.0 means the bucket actually
     contains enough positions)?
  3. What does the post-selection piece-count histogram look like?

Outputs a markdown report. Selects nothing and writes no shards - Pass B
consumes the rates recorded here.

Usage:
    python data/multiPV/plan_selection.py --index data/multiPV/index/pass_a_index.bin
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pass_a_index import INDEX_DTYPE  # noqa: F401  (dtype shared with Pass A)

# Piece-count buckets. Edges are inclusive-lower, inclusive-upper.
BUCKETS = [
    ("<=5", 0, 5),
    ("6-14", 6, 14),
    ("15-27", 15, 27),
    (">=28", 28, 32),
]

# Requested composition of the Pass B selection, as shares of the total.
# <=5 is a token sanity set only: Syzygy Mode 1 bypasses MCTS below 6 pieces and
# Mode 2 overrides the value head at leaves, so these labels train an output that
# inference discards.
DEFAULT_SHARES = {"<=5": 0.01, "6-14": 0.35, "15-27": 0.40, ">=28": 0.24}


def load_index(path: Path) -> np.ndarray:
    return np.memmap(path, dtype=INDEX_DTYPE, mode="r")


def _pct(a, b):
    return f"{100.0 * a / b:.3f}%" if b else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path,
                    default=Path(__file__).resolve().parent / "index" / "pass_a_index.bin")
    ap.add_argument("--target", type=int, default=30_000_000,
                    help="Pass B conversion target (oversampled vs the final 15-20M)")
    ap.add_argument("--value-min-depths", type=int, nargs="+",
                    default=[20, 22, 24, 26, 28, 30])
    ap.add_argument("--policy-min-depth", type=int, default=20)
    ap.add_argument("--chosen-value-depth", type=int, default=None,
                    help="if given, emit rates for this threshold")
    ap.add_argument("--shares", type=str, default=None,
                    help='JSON override, e.g. \'{"6-14":0.4,"15-27":0.35}\'')
    ap.add_argument("--report", type=Path,
                    default=Path(__file__).resolve().parent / "selection_plan.md")
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).resolve().parent / "manifests" / "selection_plan.json")
    args = ap.parse_args()

    shares = dict(DEFAULT_SHARES)
    if args.shares:
        shares.update(json.loads(args.shares))

    ix = load_index(args.index)
    n = len(ix)
    pc = np.asarray(ix["piece_count"])
    md = np.asarray(ix["max_depth"])
    pdp = np.asarray(ix["policy_depth"])

    # Positions with >32 pieces are user-constructed illegal setups on the
    # analysis board. They would be rejected by board.is_valid() in Pass B
    # anyway; excluding them here keeps the pool estimate honest.
    legal_mass = pc <= 32
    n_illegal = int((~legal_mass).sum())

    lines = [
        "# Selection plan (from Pass A index)",
        "",
        f"- index: `{args.index}`",
        f"- indexed positions: **{n:,}**",
        f"- excluded as illegal (piece_count > 32): {n_illegal:,} ({_pct(n_illegal, n)})",
        f"- Pass B target: **{args.target:,}**",
        f"- policy_min_depth: {args.policy_min_depth}",
        "",
        "## 1. Depth threshold choice",
        "",
        "`pool` = positions with max_depth >= threshold (and <=32 pieces). "
        "`pool/target` is the headroom stratification has to work with; the brief "
        "asks for ~4x.",
        "",
        "| value_min_depth | eligible pool | share of corpus | pool/target | with policy block |",
        "|---:|---:|---:|---:|---:|",
    ]
    pool_by_depth = {}
    for v in args.value_min_depths:
        m = legal_mass & (md >= v)
        c = int(m.sum())
        wp = int((m & (pdp >= args.policy_min_depth)).sum())
        pool_by_depth[v] = c
        lines.append(f"| {v} | {c:,} | {_pct(c, n)} | {c / args.target:.2f}x | "
                     f"{wp:,} ({_pct(wp, c)}) |")

    # Pick the threshold whose pool/target is closest to 4x from above.
    chosen = args.chosen_value_depth
    if chosen is None:
        viable = [(v, c) for v, c in pool_by_depth.items() if c >= 4 * args.target]
        chosen = max(viable)[0] if viable else max(pool_by_depth, key=pool_by_depth.get)
    lines += ["", f"**Chosen value_min_depth = {chosen}** "
                  f"(pool {pool_by_depth[chosen]:,} = "
                  f"{pool_by_depth[chosen] / args.target:.2f}x target)", ""]

    # --- bucket analysis at the chosen threshold ---
    eligible = legal_mass & (md >= chosen)
    has_policy = eligible & (pdp >= args.policy_min_depth)

    lines += [
        "## 2. Per-bucket availability and sampling rates",
        "",
        "| bucket | available | natural share | target share | target count | "
        "**rate** | feasible | w/ policy |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    avail = {}
    for name, lo, hi in BUCKETS:
        m = eligible & (pc >= lo) & (pc <= hi)
        avail[name] = int(m.sum())
    total_avail = sum(avail.values()) or 1

    rates = {}
    infeasible = []
    for name, lo, hi in BUCKETS:
        a = avail[name]
        want = int(round(args.target * shares.get(name, 0.0)))
        rate = (want / a) if a else 0.0
        rates[name] = min(rate, 1.0)
        ok = rate <= 1.0
        if not ok:
            infeasible.append((name, want, a))
        m = eligible & (pc >= lo) & (pc <= hi) & (pdp >= args.policy_min_depth)
        wp = int(m.sum())
        lines.append(
            f"| {name} | {a:,} | {_pct(a, total_avail)} | "
            f"{100*shares.get(name,0.0):.1f}% | {want:,} | **{rate:.4f}** | "
            f"{'yes' if ok else '**NO**'} | {wp:,} ({_pct(wp, a)}) |")

    if infeasible:
        lines += ["", "> **INFEASIBLE**: " + "; ".join(
            f"{nm} wants {w:,} but only {a:,} available" for nm, w, a in infeasible)
            + ". Lower the share or the depth threshold."]

    # --- resulting composition ---
    lines += ["", "## 3. Post-selection composition (expected)", "",
              "| bucket | selected | share |", "|---|---:|---:|"]
    sel_total = sum(int(avail[n_] * rates[n_]) for n_, _, _ in BUCKETS)
    for name, _, _ in BUCKETS:
        s = int(avail[name] * rates[name])
        lines.append(f"| {name} | {s:,} | {_pct(s, sel_total)} |")
    lines.append(f"| **total** | **{sel_total:,}** | |")

    # natural vs selected, for the record
    lines += ["", "### Natural corpus composition, for contrast", "",
              "| bucket | corpus | share |", "|---|---:|---:|"]
    for name, lo, hi in BUCKETS:
        c = int((legal_mass & (pc >= lo) & (pc <= hi)).sum())
        lines.append(f"| {name} | {c:,} | {_pct(c, n)} |")

    # --- fine-grained piece histogram of the selection ---
    lines += ["", "## 4. Piece-count detail (eligible pool at chosen depth)", "",
              "| pieces | eligible | with policy block |", "|---:|---:|---:|"]
    for p in range(3, 33):
        m = eligible & (pc == p)
        c = int(m.sum())
        if c == 0:
            continue
        wp = int((m & (pdp >= args.policy_min_depth)).sum())
        lines.append(f"| {p} | {c:,} | {wp:,} |")

    args.report.write_text("\n".join(lines), encoding="utf-8")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({
        "index": str(args.index),
        "indexed_positions": int(n),
        "illegal_excluded": n_illegal,
        "pass_b_target": args.target,
        "value_min_depth": int(chosen),
        "policy_min_depth": args.policy_min_depth,
        "bucket_edges": {nm: [lo, hi] for nm, lo, hi in BUCKETS},
        "target_shares": shares,
        "bucket_available": avail,
        "bucket_sampling_rates": rates,
        "expected_selected": sel_total,
    }, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[report -> {args.report}]\n[manifest -> {args.manifest}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
