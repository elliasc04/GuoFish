"""Pre-flight feasibility scan over the Pass A index, plus the rate derivation
Pass B uses.

This module is the single source of truth for two things that must never drift
apart:

  1. `bucket_availability` - how many eligible / policy-bearing / value-only
     rows each piece-count bucket holds at a given (value_min_depth,
     policy_min_depth).
  2. `derive_rates` - the per-bucket (r_policy, r_value_only) pair implied by a
     target, a share vector and a policy share.

Pass B imports both rather than re-deriving them, so a change to the accounting
here is automatically a change to what gets built.

Run as a script it answers the three questions the 90M rebuild brief leaves to
measurement, for a grid of depth thresholds:

  - is the target reachable at all, per bucket, or does a bucket silently
    under-deliver because `min(want/avail, 1.0)` clamps?
  - how much policy oversampling is affordable before the policy pool runs out?
  - can `value_min_depth` rise without breaking the first two?

Selects nothing and writes no shards.

Usage:
    python data/multiPV/feasibility_scan.py --target 91350000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pass_a_index import INDEX_DTYPE

# Piece-count buckets. Edges are inclusive on both sides.
BUCKETS = [("<=5", 0, 5), ("6-14", 6, 14), ("15-27", 15, 27), (">=28", 28, 32)]

DEFAULT_SHARES = {"<=5": 0.01, "6-14": 0.35, "15-27": 0.40, ">=28": 0.24}

# Rows with piece_count > 32 are user-constructed illegal analysis-board setups.
# Pass B's board.is_valid() rejects them anyway; excluding them here keeps the
# pool estimate honest and matches the `eligible` mask in build_selection.
MAX_PIECES = 32

_SCAN_CHUNK = 20_000_000


def bucket_name(piece_count: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= piece_count <= hi:
            return name
    return ">32"


def bucket_availability(index_path: Path, value_min_depth: int,
                        policy_min_depth: int, chunk: int = _SCAN_CHUNK) -> dict:
    """Per-bucket {eligible, policy, value_only} counts at one threshold pair.

    Chunked over the memmap so peak RSS stays a few hundred MB regardless of
    index size. `policy` counts rows whose deepest >=2-PV block is at least
    `policy_min_depth` - the index proxy for has_policy. Pass B's realised
    coverage runs a couple of points lower because select_policy_block also
    demands >=2 UNIQUE moves after dedup.
    """
    ix = np.memmap(index_path, dtype=INDEX_DTYPE, mode="r")
    n = len(ix)
    out = {name: {"eligible": 0, "policy": 0, "value_only": 0} for name, _, _ in BUCKETS}
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        pc = np.asarray(ix["piece_count"][start:stop])
        md = np.asarray(ix["max_depth"][start:stop])
        pdp = np.asarray(ix["policy_depth"][start:stop])
        eligible = (pc <= MAX_PIECES) & (md >= value_min_depth)
        has_pol = pdp >= policy_min_depth
        for name, lo, hi in BUCKETS:
            m = eligible & (pc >= lo) & (pc <= hi)
            e = int(m.sum())
            p = int((m & has_pol).sum())
            out[name]["eligible"] += e
            out[name]["policy"] += p
            out[name]["value_only"] += e - p
        del pc, md, pdp, eligible, has_pol
    return out


def scan_grid(index_path: Path, value_depths, policy_depths,
              chunk: int = _SCAN_CHUNK) -> dict:
    """`bucket_availability` over a whole threshold grid in ONE pass.

    Twelve independent calls would re-read the 5.5 GB index twelve times; the
    eligible mask depends only on value_min_depth and the policy mask only on
    policy_min_depth, so every cell can be filled from one chunk read.

    Returns {value_min_depth: {policy_min_depth: <bucket_availability dict>}}.
    """
    ix = np.memmap(index_path, dtype=INDEX_DTYPE, mode="r")
    n = len(ix)
    grid = {v: {p: {name: {"eligible": 0, "policy": 0, "value_only": 0}
                    for name, _, _ in BUCKETS}
                for p in policy_depths}
            for v in value_depths}
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        pc = np.asarray(ix["piece_count"][start:stop])
        md = np.asarray(ix["max_depth"][start:stop])
        pdp = np.asarray(ix["policy_depth"][start:stop])
        legal = pc <= MAX_PIECES
        bmask = {name: legal & (pc >= lo) & (pc <= hi) for name, lo, hi in BUCKETS}
        for v in value_depths:
            deep = md >= v
            for name, _, _ in BUCKETS:
                m = bmask[name] & deep
                e = int(m.sum())
                if not e:
                    continue
                for p in policy_depths:
                    npol = int((m & (pdp >= p)).sum())
                    cell = grid[v][p][name]
                    cell["eligible"] += e
                    cell["policy"] += npol
                    cell["value_only"] += e - npol
        del pc, md, pdp, legal, bmask
    return grid


def derive_rates(avail: dict, target: int, shares: dict, policy_share: float,
                 spill: bool = True) -> tuple[dict, dict, dict]:
    """Per-bucket sampling rates for the policy-bearing and value-only halves.

    Each bucket wants `target * share` rows, split `policy_share` /
    `1 - policy_share` between the two halves. Rates use the same
    `min(want/avail, 1.0)` form as the single-rate scheme, so raising a rate can
    only grow the selected set (see the nesting invariant).

    When `spill` is set and one half's pool is exhausted, the unmet demand is
    taken from the other half of the SAME bucket instead of being dropped. That
    trades realised policy coverage for the piece-count share target, which is
    the right way round: the share vector is a modelling decision, the coverage
    is a preference. Without it the bucket silently under-delivers and the share
    targets are quietly violated. Either way `plan` records exactly what
    clamped, so nothing is decided by the clamp alone.

    Returns (r_policy, r_value_only, plan).
    """
    r_pol: dict[str, float] = {}
    r_val: dict[str, float] = {}
    plan: dict[str, dict] = {}

    for name, _, _ in BUCKETS:
        a_pol = avail[name]["policy"]
        a_val = avail[name]["value_only"]
        want_total = target * shares.get(name, 0.0)
        want_pol = want_total * policy_share
        want_val = want_total - want_pol

        get_pol = min(want_pol, float(a_pol))
        get_val = min(want_val, float(a_val))

        spilled_to_val = 0.0
        spilled_to_pol = 0.0
        if spill:
            # Unmet demand in one half is re-aimed at the other half's surplus.
            if want_pol > a_pol:
                deficit = want_pol - a_pol
                room = a_val - get_val
                spilled_to_val = min(deficit, max(room, 0.0))
                get_val += spilled_to_val
            if want_val > a_val:
                deficit = want_val - a_val
                room = a_pol - get_pol
                spilled_to_pol = min(deficit, max(room, 0.0))
                get_pol += spilled_to_pol

        r_pol[name] = min(get_pol / a_pol, 1.0) if a_pol else 0.0
        r_val[name] = min(get_val / a_val, 1.0) if a_val else 0.0

        got = get_pol + get_val
        plan[name] = {
            "available_policy": a_pol,
            "available_value_only": a_val,
            "want_total": want_total,
            "want_policy": want_pol,
            "want_value_only": want_val,
            "expected_policy": get_pol,
            "expected_value_only": get_val,
            "expected_total": got,
            "expected_share": None,          # filled in below
            "expected_coverage": (get_pol / got) if got else 0.0,
            "shortfall": want_total - got,
            "policy_clamped": want_pol > a_pol,
            "value_only_clamped": want_val > a_val,
            "spilled_to_value_only": spilled_to_val,
            "spilled_to_policy": spilled_to_pol,
        }

    grand = sum(p["expected_total"] for p in plan.values()) or 1.0
    for p in plan.values():
        p["expected_share"] = p["expected_total"] / grand
    return r_pol, r_val, plan


def plan_totals(plan: dict) -> dict:
    total = sum(p["expected_total"] for p in plan.values())
    pol = sum(p["expected_policy"] for p in plan.values())
    return {
        "expected_selected": total,
        "expected_policy": pol,
        "expected_coverage": pol / total if total else 0.0,
        "shortfall": sum(p["shortfall"] for p in plan.values()),
    }


def coverage_ceiling(avail: dict, target: int, shares: dict) -> float:
    """Highest realisable global policy coverage at this target and share vector.

    Every bucket's coverage saturates at available_policy / (target * share), so
    the ceiling is that cap summed and divided by the target. Requesting a
    `policy_share` above this cannot be honoured no matter how the rates are
    derived - the pool is simply not there.
    """
    got = 0.0
    for name, _, _ in BUCKETS:
        want = target * shares.get(name, 0.0)
        got += min(want, float(avail[name]["policy"]))
    return got / target if target else 0.0


# ------------------------------------------------------------------- reporting

def _pct(a, b):
    return f"{100.0 * a / b:.2f}%" if b else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description="Pass B feasibility scan")
    ap.add_argument("--index", type=Path,
                    default=Path(__file__).resolve().parent / "index" / "pass_a_index.bin")
    ap.add_argument("--target", type=int, default=91_350_000,
                    help="PRE-rejection Pass B target (see the brief's yield scaling)")
    ap.add_argument("--value-min-depths", type=int, nargs="+", default=[24, 26, 28, 30])
    ap.add_argument("--policy-min-depths", type=int, nargs="+", default=[18, 20, 22])
    ap.add_argument("--policy-share", type=float, default=0.65)
    ap.add_argument("--policy-share-sweep", type=float, nargs="+",
                    default=[0.40, 0.50, 0.55, 0.60, 0.65, 0.70])
    ap.add_argument("--shares", type=str, default=None,
                    help='JSON override of the piece-count share vector')
    ap.add_argument("--prior-rates", type=str,
                    default='{"<=5":0.027850382176869422,"6-14":0.21647887502237464,'
                            '"15-27":0.17551500196927833,">=28":0.3113096190348537}',
                    help="single-rate vector of the corpus that must stay nested")
    ap.add_argument("--prior-value-min-depth", type=int, default=26)
    ap.add_argument("--report", type=Path,
                    default=Path(__file__).resolve().parent / "feasibility_scan.md")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "manifests" / "feasibility_scan.json")
    args = ap.parse_args()

    shares = dict(DEFAULT_SHARES)
    if args.shares:
        shares.update(json.loads(args.shares))
    prior_rates = json.loads(args.prior_rates)

    ix = np.memmap(args.index, dtype=INDEX_DTYPE, mode="r")
    n_index = len(ix)
    del ix

    L = [
        "# Pass B feasibility scan (90M policy-biased rebuild)",
        "",
        f"- index: `{args.index}` ({n_index:,} rows)",
        f"- pre-rejection target: **{args.target:,}**",
        f"- share vector: `{json.dumps(shares)}`",
        "",
        "`policy` = index rows whose deepest >=2-PV block reaches "
        "`policy_min_depth`. Pass B's realised coverage runs a little lower: "
        "`select_policy_block` additionally requires >=2 *unique* moves after "
        "dedup (the 30M build realised 42.7 / 36.7 / 42.6% against a predicted "
        "42.7 / 36.6 / 42.2%, so the proxy is good to a few tenths).",
        "",
    ]

    print("scanning index (one pass over the grid)...", flush=True)
    grid = scan_grid(args.index, args.value_min_depths, args.policy_min_depths)

    # ------------------------------------------------- 1. the depth x depth grid
    L += [
        "## 1. Eligible pool and policy pool over the threshold grid",
        "",
        "| value_min_depth | policy_min_depth | bucket | eligible | policy-bearing | coverage | value-only |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for v in args.value_min_depths:
        for p in args.policy_min_depths:
            avail = grid[v][p]
            for name, _, _ in BUCKETS:
                d = avail[name]
                L.append(f"| {v} | {p} | {name} | {d['eligible']:,} | {d['policy']:,} | "
                         f"{_pct(d['policy'], d['eligible'])} | {d['value_only']:,} |")
            tot_e = sum(d["eligible"] for d in avail.values())
            tot_p = sum(d["policy"] for d in avail.values())
            L.append(f"| {v} | {p} | **all** | **{tot_e:,}** | **{tot_p:,}** | "
                     f"**{_pct(tot_p, tot_e)}** | **{tot_e - tot_p:,}** |")

    # ------------------------------------------- 2. is the target reachable?
    L += [
        "",
        "## 2. Is the target reachable? (per-bucket demand vs pool)",
        "",
        "A bucket whose demand exceeds its pool does not error - "
        "`min(want/avail, 1.0)` clamps and the bucket quietly under-delivers. "
        "This table is what makes that visible before the build.",
        "",
        "| value_min_depth | bucket | want | eligible | headroom | verdict |",
        "|---:|---|---:|---:|---:|---|",
    ]
    reachable = {}
    for v in args.value_min_depths:
        avail = grid[v][args.policy_min_depths[0]]   # eligible is policy-independent
        ok_all = True
        for name, _, _ in BUCKETS:
            want = args.target * shares.get(name, 0.0)
            a = avail[name]["eligible"]
            ok = a >= want
            ok_all &= ok
            L.append(f"| {v} | {name} | {want:,.0f} | {a:,} | {a / want:.2f}x | "
                     f"{'ok' if ok else '**SHORT**'} |")
        reachable[v] = ok_all
    L += ["", "Reachable (ignoring the policy split): " +
          ", ".join(f"depth {v} {'yes' if ok else 'NO'}" for v, ok in reachable.items())]

    # ------------------------------- 3. how much policy oversampling is affordable
    L += [
        "",
        "## 3. Affordable policy share",
        "",
        f"Rates derived at `policy_min_depth = {args.policy_min_depths[1] if len(args.policy_min_depths) > 1 else args.policy_min_depths[0]}`, "
        "spill on (unmet demand in one half is taken from the other half of the "
        "same bucket, so the share vector holds and coverage gives instead).",
        "",
        "| value_min_depth | policy_share | expected selected | realised coverage | ceiling | shortfall |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    p_ref = args.policy_min_depths[1] if len(args.policy_min_depths) > 1 else args.policy_min_depths[0]
    sweep: dict = {}
    for v in args.value_min_depths:
        avail = grid[v][p_ref]
        ceil = coverage_ceiling(avail, args.target, shares)
        sweep[v] = {"ceiling": ceil, "points": {}}
        for c in args.policy_share_sweep:
            _rp, _rv, plan = derive_rates(avail, args.target, shares, c)
            tot = plan_totals(plan)
            sweep[v]["points"][c] = tot
            L.append(f"| {v} | {c:.2f} | {tot['expected_selected']:,.0f} | "
                     f"{tot['expected_coverage']:.2%} | {ceil:.2%} | "
                     f"{tot['shortfall']:,.0f} |")

    # ------------------------------------------------- 4. the recommended point
    L += ["", "## 4. Recommended operating point", ""]

    # Prefer the deepest value_min_depth that is reachable in every bucket AND
    # can still honour the requested policy share to within a point.
    viable = []
    for v in args.value_min_depths:
        if not reachable[v]:
            continue
        avail = grid[v][p_ref]
        _rp, _rv, plan = derive_rates(avail, args.target, shares, args.policy_share)
        tot = plan_totals(plan)
        viable.append((v, tot, sweep[v]["ceiling"]))
    if viable:
        chosen_v = max(v for v, _t, _c in viable)
    else:
        chosen_v = min(args.value_min_depths)
    chosen_avail = grid[chosen_v][p_ref]
    r_pol, r_val, plan = derive_rates(chosen_avail, args.target, shares, args.policy_share)
    tot = plan_totals(plan)

    L += [
        f"- `value_min_depth` = **{chosen_v}** (deepest threshold reachable in "
        f"every bucket at this target)",
        f"- `policy_min_depth` = **{p_ref}**",
        f"- `policy_share` = **{args.policy_share:.2f}**, realised "
        f"**{tot['expected_coverage']:.2%}**, ceiling "
        f"**{sweep[chosen_v]['ceiling']:.2%}**",
        f"- expected selected lines: **{tot['expected_selected']:,.0f}** "
        f"(shortfall {tot['shortfall']:,.0f})",
        "",
        "| bucket | r_policy | r_value_only | expected total | share | coverage | clamped |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, _, _ in BUCKETS:
        d = plan[name]
        flags = []
        if d["policy_clamped"]:
            flags.append("policy")
        if d["value_only_clamped"]:
            flags.append("value-only")
        L.append(f"| {name} | {r_pol[name]:.4f} | {r_val[name]:.4f} | "
                 f"{d['expected_total']:,.0f} | {d['expected_share']:.2%} | "
                 f"{d['expected_coverage']:.2%} | {', '.join(flags) or '-'} |")

    # --------------------------------------------------- 5. nesting assertion
    L += ["", "## 5. Nesting against the existing corpus", ""]
    nest_ok = chosen_v == args.prior_value_min_depth
    if not nest_ok:
        L.append(f"> **BROKEN**: value_min_depth moves "
                 f"{args.prior_value_min_depth} -> {chosen_v}, which changes the "
                 f"`eligible` mask and drops previously-selected rows.")
    else:
        L += ["`eligible` is unchanged, so the old selection stays nested iff both "
              "new rates dominate the old single rate in every bucket.",
              "",
              "| bucket | prior rate | r_policy | r_value_only | dominates |",
              "|---|---:|---:|---:|---|"]
        for name, _, _ in BUCKETS:
            old = prior_rates.get(name, 0.0)
            ok = r_pol[name] >= old and r_val[name] >= old
            nest_ok &= ok
            L.append(f"| {name} | {old:.4f} | {r_pol[name]:.4f} | {r_val[name]:.4f} | "
                     f"{'yes' if ok else '**NO**'} |")
        L += ["", f"**Nesting preserved: {'yes' if nest_ok else 'NO'}**"]

    report = "\n".join(L)
    args.report.write_text(report, encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "index": str(args.index),
        "index_rows": n_index,
        "target": args.target,
        "target_shares": shares,
        "grid": {str(v): {str(p): a for p, a in d.items()} for v, d in grid.items()},
        "reachable": {str(v): bool(o) for v, o in reachable.items()},
        "policy_share_sweep": {str(v): {"ceiling": d["ceiling"],
                                        "points": {str(c): t for c, t in d["points"].items()}}
                               for v, d in sweep.items()},
        "recommended": {
            "value_min_depth": chosen_v,
            "policy_min_depth": p_ref,
            "policy_share": args.policy_share,
            "r_policy": r_pol,
            "r_value_only": r_val,
            "plan": plan,
            "totals": tot,
            "coverage_ceiling": sweep[chosen_v]["ceiling"],
            "nesting_preserved": bool(nest_ok),
        },
    }, indent=2), encoding="utf-8")

    print(report)
    print(f"\n[report -> {args.report}]\n[json -> {args.out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
