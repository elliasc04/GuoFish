"""Post-build verification of a rebuilt corpus against the one it must nest under.

Three checks the manifest cannot make on its own, because two of them need the
source FENs and one is worth re-deriving independently of the build:

  1. NESTING - every line the prior selection kept is kept by the new one.
     Recomputed from scratch here rather than trusted from the build's own
     diagnostic, chunk by chunk, so a bug in build_selection cannot certify
     itself.
  2. VAL SPLIT CONTINUITY - the split is sha1(fen) % 1000 < N, keyed on the FEN
     and not the line number, so it must be stable across rebuilds. Sampled on
     real FENs from the dump: every line that was val in the prior corpus must
     still be val, and must still be selected.
  3. VAL LEAKAGE - no FEN may appear on both sides of the boundary. This is
     automatic given a FEN-keyed split, and is checked on the same sample.

Usage:
    python data/multiPV/verify_corpus_90m.py \
        --new manifests/dataset_manifest_90m.json \
        --prior manifests/dataset_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from feasibility_scan import BUCKETS, MAX_PIECES  # noqa: E402
from pass_a_index import INDEX_DTYPE, iter_lines  # noqa: E402

_STEP = 20_000_000


def _rates(m: dict) -> tuple[dict, dict]:
    pol = m.get("bucket_sampling_rates_policy") or m["bucket_sampling_rates"]
    val = m.get("bucket_sampling_rates_value_only") or m["bucket_sampling_rates"]
    return pol, val


def _keep_masks(pc, md, pdp, u, m: dict):
    """Reproduce one manifest's selection mask for a chunk."""
    r_pol, r_val = _rates(m)
    elig = (pc <= MAX_PIECES) & (md >= m["value_min_depth"])
    has_pol = pdp >= m["policy_min_depth"]
    keep = np.zeros(len(pc), dtype=bool)
    for name, lo, hi in BUCKETS:
        b = elig & (pc >= lo) & (pc <= hi)
        rp = r_pol.get(name, 0.0)
        rv = r_val.get(name, 0.0)
        if rp > 0:
            keep |= (b & has_pol) & (u < rp)
        if rv > 0:
            keep |= (b & ~has_pol) & (u < rv)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path,
                    default=_HERE / "manifests" / "dataset_manifest_90m.json")
    ap.add_argument("--prior", type=Path,
                    default=_HERE / "manifests" / "dataset_manifest.json")
    ap.add_argument("--index", type=Path, default=_HERE / "index" / "pass_a_index.bin")
    ap.add_argument("--source", type=Path, default=_HERE / "lichess_db_eval.jsonl.zst")
    ap.add_argument("--sample-lines", type=int, default=8_000_000,
                    help="source lines to stream for the FEN-level val check")
    args = ap.parse_args()

    new = json.loads(args.new.read_text())
    prior = json.loads(args.prior.read_text())
    if new["seed"] != prior["seed"]:
        print(f"seed differs ({prior['seed']} -> {new['seed']}); nesting cannot hold",
              file=sys.stderr)
        return 1

    # --- 1. nesting, recomputed independently -------------------------------
    ix = np.memmap(args.index, dtype=INDEX_DTYPE, mode="r")
    n = len(ix)
    rng = np.random.default_rng(new["seed"])
    dropped = 0
    n_prior = n_new = 0
    for start in range(0, n, _STEP):
        stop = min(start + _STEP, n)
        pc = np.asarray(ix["piece_count"][start:stop])
        md = np.asarray(ix["max_depth"][start:stop])
        pdp = np.asarray(ix["policy_depth"][start:stop])
        u = rng.random(stop - start)
        k_new = _keep_masks(pc, md, pdp, u, new)
        k_old = _keep_masks(pc, md, pdp, u, prior)
        dropped += int((k_old & ~k_new).sum())
        n_prior += int(k_old.sum())
        n_new += int(k_new.sum())
        del pc, md, pdp, u, k_new, k_old
    nest_ok = dropped == 0
    print(f"[1] nesting: prior selected {n_prior:,}, new selected {n_new:,}, "
          f"prior rows dropped {dropped:,} -> "
          f"{'STRICT SUPERSET' if nest_ok else 'BROKEN'}")

    # --- 2/3. val split continuity and leakage, on real FENs -----------------
    permille_new = int(new["val_split"].rsplit("<", 1)[1])
    permille_old = int(prior["val_split"].rsplit("<", 1)[1])
    if permille_new != permille_old:
        print(f"[2] val_permille changed {permille_old} -> {permille_new}; "
              f"the old val set is NOT a subset", file=sys.stderr)

    # Rebuild both masks for just the sampled prefix, using the same seeded
    # stream (chunk boundaries must match, hence the same _STEP walk).
    rng = np.random.default_rng(new["seed"])
    limit = min(args.sample_lines, n)
    sel_new = np.zeros(limit, dtype=bool)
    sel_old = np.zeros(limit, dtype=bool)
    for start in range(0, n, _STEP):
        stop = min(start + _STEP, n)
        u = rng.random(stop - start)
        if start >= limit:
            break
        pc = np.asarray(ix["piece_count"][start:stop])
        md = np.asarray(ix["max_depth"][start:stop])
        pdp = np.asarray(ix["policy_depth"][start:stop])
        hi = min(stop, limit) - start
        sel_new[start:start + hi] = _keep_masks(pc, md, pdp, u, new)[:hi]
        sel_old[start:start + hi] = _keep_masks(pc, md, pdp, u, prior)[:hi]
        del pc, md, pdp, u

    def is_val(fen: str, permille: int) -> bool:
        return (int.from_bytes(hashlib.sha1(fen.encode()).digest()[:8], "big")
                % 1000) < permille

    old_val = old_train = 0
    still_val = still_selected = 0
    seen: dict[str, bool] = {}
    leaks = 0
    for ln, (_off, raw) in enumerate(iter_lines(args.source)):
        if ln >= limit:
            break
        if not sel_old[ln]:
            continue
        try:
            fen = json.loads(raw).get("fen")
        except Exception:
            continue
        if not fen:
            continue
        v_old = is_val(fen, permille_old)
        v_new = is_val(fen, permille_new)
        if v_old:
            old_val += 1
            still_val += int(v_new)
            still_selected += int(bool(sel_new[ln]))
        else:
            old_train += 1
        # A FEN that lands on both sides across duplicate source lines is a leak.
        if fen in seen and seen[fen] != v_new:
            leaks += 1
        else:
            seen[fen] = v_new

    val_ok = old_val > 0 and still_val == old_val and still_selected == old_val
    print(f"[2] val continuity over {limit:,} sampled source lines: "
          f"{old_val:,} prior-val records, {still_val:,} still hash to val, "
          f"{still_selected:,} still selected -> {'OK' if val_ok else 'FAIL'}")
    print(f"[3] val leakage: {len(seen):,} distinct FENs, {leaks:,} FENs on both "
          f"sides -> {'OK' if leaks == 0 else 'FAIL'}")
    print(f"    (prior-train in sample: {old_train:,}; "
          f"observed val share {old_val / max(old_val + old_train, 1):.4%}, "
          f"expected {permille_old / 1000:.4%})")

    ok = nest_ok and val_ok and leaks == 0
    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
