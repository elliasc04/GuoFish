"""PASS B - convert the selected lines into fixed-width binary shards.

Reads the Pass A index to decide WHICH lines to convert (stratified on piece
count), then streams the dump once and does the expensive work - board
construction, legal-move generation, tokenization, label construction - only on
those lines.

Why one reader and a worker pool, rather than workers seeking to their own line
ranges: a zstd stream cannot be randomly seeked, so a worker assigned a late
range would have to decompress everything before it. With W workers that is
~W/2 full decompressions. Instead the main process decompresses once (measured:
~420k lines/s, 13.6 min for the whole file) and ships only the selected raw
lines to the pool, which is where the 147 us/position of chess work happens.
Wall clock is therefore bounded by the single decompression pass.

Selection, val split, and shard routing are all deterministic functions of the
line number and a seed, so a resumed or re-run job reproduces the same dataset.

Usage:
    python data/multiPV/pass_b_convert.py

Every constant has a flag, but the defaults are the decided values - the value
scale in particular is locked (see labels.VALUE_SCALE) and should not be swept
here; it is a collate-time knob in Phase 3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import chess
import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data.pgn_parallel import _board_to_tokens  # noqa: E402
from labels import (  # noqa: E402
    CP_CLAMP, DEFAULT_EPSILON, DEFAULT_TEMPERATURE, MAX_LEGAL, MAX_PV,
    VALUE_SCALE, build_policy_target, select_policy_block, select_value_block,
    value_cp_is_mate, value_from_raw_cp, value_raw_cp,
)
from feasibility_scan import (  # noqa: E402
    BUCKETS, MAX_PIECES, bucket_availability, bucket_name, coverage_ceiling,
    derive_rates, plan_totals,
)
from pass_a_index import INDEX_DTYPE, iter_lines  # noqa: E402
from record_format import RECORD_DTYPE, shard_name  # noqa: E402

# ---------------------------------------------------------------- worker side

_CFG: dict = {}


def _init_worker(cfg: dict):
    global _CFG
    _CFG = cfg


def _splitmix64(x: int) -> int:
    """Deterministic scalar hash - shard routing must not depend on RNG state."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def convert_one(raw: bytes, cfg: dict, stats: Counter):
    """Convert one JSONL line. Returns (record_array, is_val) or (None, reason)."""
    try:
        rec = json.loads(raw)
    except Exception:
        return None, "json_error"
    fen = rec.get("fen")
    evals = rec.get("evals") or []
    if not fen or not evals:
        return None, "no_evals"

    # --- board, with explicit Chess960 / invalid separation (never a bare except)
    try:
        board = chess.Board(fen)
    except Exception:
        try:
            b960 = chess.Board(fen, chess960=True)
        except Exception:
            return None, "unparseable"
        return None, "chess960_skipped" if b960.is_valid() else "invalid_fen"
    if not board.is_valid():
        try:
            if chess.Board(fen, chess960=True).is_valid():
                return None, "chess960_skipped"
        except Exception:
            pass
        return None, "invalid_fen"

    legal = list(board.legal_moves)
    if not legal:
        return None, "no_legal_moves"

    # --- value (White-POV, no conversion) ---
    vblock = select_value_block(evals, cfg["value_min_depth"])
    if vblock is None:
        return None, "below_value_depth"
    raw_cp = value_raw_cp(vblock["pvs"][0])
    if raw_cp is None:
        stats["value_pv_missing_score"] += 1
        return None, "value_pv_missing_score"
    value = value_from_raw_cp(raw_cp, cfg["value_scale"])

    try:
        tokens = _board_to_tokens(board)
    except Exception:
        return None, "tokenize_failed"
    if len(tokens) != 68:
        return None, "tokenize_failed"

    out = np.zeros(1, dtype=RECORD_DTYPE)
    out["tokens"][0] = np.asarray(tokens, dtype=np.int8)
    out["value"][0] = np.float16(value)
    out["value_cp"][0] = np.int16(raw_cp)

    n_legal = len(legal)
    if n_legal > MAX_LEGAL:
        stats["legal_truncated"] += 1
        legal = legal[:MAX_LEGAL]
        n_legal = MAX_LEGAL
    out["n_legal"][0] = n_legal
    out["legal_idx"][0, :n_legal] = np.asarray(
        [m.from_square * 64 + m.to_square for m in legal], dtype=np.int16)

    # --- policy (stm-relative; separate path from value) ---
    sel = select_policy_block(board, evals, cfg["policy_min_depth"], cfg["cp_clamp"])
    if sel is None:
        stats["value_only"] += 1
    else:
        entries, _depth, dropped, removed = sel
        if dropped:
            stats["pv_dropped_missing_score"] += dropped
        if removed:
            stats["duplicate_pvs_removed"] += removed
            stats["positions_with_duplicate_pvs"] += 1
        stats[f"unique_pv_count_{min(len(entries), 16)}"] += 1

        idxs, probs, scores, ok = build_policy_target(
            entries, board.turn == chess.WHITE,
            cfg["temperature"], cfg["epsilon"], MAX_PV)
        if not ok:
            return None, "invariant_violation"

        if len(entries) > MAX_PV:
            stats["policy_truncated"] += 1

        n = len(idxs)
        out["has_policy"][0] = 1
        out["n_pv"][0] = n
        out["pv_idx"][0, :n] = np.asarray(idxs, dtype=np.int16)
        out["pv_prob"][0, :n] = np.asarray(probs, dtype=np.float16)
        out["pv_score"][0, :n] = np.asarray(
            [max(-32000, min(32000, s)) for s in scores], dtype=np.float16)

        mass = float(np.sum(np.asarray(probs, dtype=np.float64)))
        if abs(mass - (1.0 - cfg["epsilon"])) > 1e-4:
            return None, "mass_check_failed"

    # --- reporting: policy coverage by piece count, value mass by stratum ---
    # Aggregate value MSE is misleading on this corpus - roughly a quarter of
    # targets are exactly 0 and a fifth are pinned mates, so a head that always
    # outputs ~0 scores well on the zero mass by doing nothing. Count the three
    # strata at conversion time so training can report against them.
    b = bucket_name(chess.popcount(board.occupied))
    stats[f"bucket_{b}"] += 1
    if out["has_policy"][0]:
        stats[f"bucket_{b}_policy"] += 1
    if value_cp_is_mate(raw_cp):
        stats["value_mate"] += 1
    elif raw_cp == 0:
        stats["value_exact_zero"] += 1
    else:
        stats["value_middle"] += 1

    is_val = (int.from_bytes(hashlib.sha1(fen.encode()).digest()[:8], "big")
              % 1000) < cfg["val_permille"]
    return out, is_val


def convert_batch(task):
    """Worker entry: (list[(line_no, raw)]) -> (payloads, stats Counter)."""
    lines, = task
    cfg = _CFG
    stats: Counter = Counter()
    payloads: list[tuple[int, bytes]] = []
    for line_no, raw in lines:
        out, info = convert_one(raw, cfg, stats)
        if out is None:
            stats[f"reject_{info}"] += 1
            continue
        is_val = info
        h = _splitmix64(line_no ^ cfg["shard_seed"])
        if is_val:
            shard = cfg["n_train_shards"] + (h % cfg["n_val_shards"])
            stats["accepted_val"] += 1
        else:
            shard = h % cfg["n_train_shards"]
            stats["accepted_train"] += 1
        payloads.append((shard, out.tobytes()))
    return payloads, stats


# ------------------------------------------------------------------ main side

# Source-order diagnostic resolution. 1024 bins over ~395M index rows is ~386k
# rows per bin, i.e. quantiles good to ~0.1% of the file - far finer than the
# effect being looked for.
_ORDER_BINS = 1024


def _quantiles_from_hist(hist: np.ndarray, n_rows: int, qs) -> list[float]:
    """Quantiles of a line-number distribution held as a fixed-width histogram.

    Returns positions as FRACTIONS of the file, linearly interpolated inside the
    containing bin. Exact enough at 1024 bins and costs no extra pass.
    """
    total = hist.sum()
    if total == 0:
        return [float("nan")] * len(qs)
    cum = np.cumsum(hist)
    width = n_rows / len(hist)
    out = []
    for q in qs:
        want = q * total
        b = int(np.searchsorted(cum, want, side="left"))
        b = min(b, len(hist) - 1)
        before = cum[b - 1] if b else 0
        frac_in_bin = (want - before) / hist[b] if hist[b] else 0.0
        out.append(((b + frac_in_bin) * width) / n_rows)
    return out


def build_selection(index_path: Path, cfg: dict, seed: int,
                    diag: dict | None = None) -> np.ndarray:
    """Sorted array of line numbers to convert.

    Stratified on piece count, and within each bucket split again on whether the
    row is policy-bearing (`policy_depth >= policy_min_depth`), so the two halves
    can be sampled at different rates.

    The SAME uniform draw `u` feeds both branches. Drawing twice would break both
    reproducibility and the nesting property, because nesting rests on
    `keep = eligible & bucket & (u < r)` growing monotonically in r against a
    fixed u.

    `diag`, if given, is filled with the nesting check against `cfg["prior"]`
    and the source-order histograms behind the §7 quantile report - both are
    free here because the masks already exist.
    """
    ix = np.memmap(index_path, dtype=INDEX_DTYPE, mode="r")
    n = len(ix)
    r_pol = cfg["bucket_rates_policy"]
    r_val = cfg["bucket_rates_value_only"]
    prior = cfg.get("prior")
    rng = np.random.default_rng(seed)
    chunks = []
    step = 20_000_000

    order_hist = {k: np.zeros(_ORDER_BINS, dtype=np.int64) for k in
                  ("eligible_policy", "eligible_value_only",
                   "selected_policy", "selected_value_only")}
    bin_width = n / _ORDER_BINS
    prior_lost = 0

    for start in range(0, n, step):
        stop = min(start + step, n)
        pc = np.asarray(ix["piece_count"][start:stop])
        md = np.asarray(ix["max_depth"][start:stop])
        pdp = np.asarray(ix["policy_depth"][start:stop])
        eligible = (pc <= MAX_PIECES) & (md >= cfg["value_min_depth"])
        has_pol = pdp >= cfg["policy_min_depth"]
        keep = np.zeros(stop - start, dtype=bool)
        u = rng.random(stop - start)
        for name, lo, hi in BUCKETS:
            in_bucket = eligible & (pc >= lo) & (pc <= hi)
            rp = r_pol.get(name, 0.0)
            rv = r_val.get(name, 0.0)
            if rp > 0:
                keep |= (in_bucket & has_pol) & (u < rp)
            if rv > 0:
                keep |= (in_bucket & ~has_pol) & (u < rv)

        if diag is not None:
            # Nesting: the prior corpus used ONE rate per bucket over the same u
            # and the same eligible mask, so any row it kept must survive here.
            if prior is not None:
                old_keep = np.zeros(stop - start, dtype=bool)
                old_elig = (pc <= MAX_PIECES) & (md >= prior["value_min_depth"])
                old_pol = pdp >= prior["policy_min_depth"]
                for name, lo, hi in BUCKETS:
                    ob = old_elig & (pc >= lo) & (pc <= hi)
                    rp = prior["bucket_rates_policy"].get(name, 0.0)
                    rv = prior["bucket_rates_value_only"].get(name, 0.0)
                    if rp > 0:
                        old_keep |= (ob & old_pol) & (u < rp)
                    if rv > 0:
                        old_keep |= (ob & ~old_pol) & (u < rv)
                prior_lost += int((old_keep & ~keep).sum())
                del old_keep, old_elig, old_pol

            bins = np.minimum(((np.arange(start, stop) // bin_width)
                               ).astype(np.int64), _ORDER_BINS - 1)
            for key, m in (("eligible_policy", eligible & has_pol),
                           ("eligible_value_only", eligible & ~has_pol),
                           ("selected_policy", keep & has_pol),
                           ("selected_value_only", keep & ~has_pol)):
                order_hist[key] += np.bincount(bins[m], minlength=_ORDER_BINS)
            del bins

        idx = np.nonzero(keep)[0]
        if len(idx):
            chunks.append(idx.astype(np.int64) + start)
        del pc, md, pdp, eligible, has_pol, keep, u

    if diag is not None:
        qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        diag["source_order_quantiles"] = {
            "quantiles": qs,
            "unit": "fraction of source file by line number",
            "populations": {k: _quantiles_from_hist(v, n, qs)
                            for k, v in order_hist.items()},
            "counts": {k: int(v.sum()) for k, v in order_hist.items()},
        }
        if prior is not None:
            diag["nesting"] = {
                "prior_value_min_depth": prior["value_min_depth"],
                "prior_manifest": prior["manifest"],
                "prior_bucket_rates_policy": prior["bucket_rates_policy"],
                "prior_bucket_rates_value_only": prior["bucket_rates_value_only"],
                "prior_rows_dropped": prior_lost,
                "is_strict_superset": prior_lost == 0,
            }

    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_PROJECT_ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def file_hash(path: Path, nbytes: int = 1 << 26) -> str:
    """Hash of the first/last 64MiB plus size - full hashing of 21GB is wasteful
    and this is enough to detect a different dump."""
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
        if size > nbytes:
            fh.seek(max(0, size - nbytes))
            h.update(fh.read(nbytes))
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Pass B: convert selected lines to shards")
    ap.add_argument("--source", type=Path,
                    default=_HERE / "lichess_db_eval.jsonl.zst")
    ap.add_argument("--index", type=Path, default=_HERE / "index" / "pass_a_index.bin")
    ap.add_argument("--out-dir", type=Path,
                    default=_PROJECT_ROOT / "data" / "processed" / "multipv_90m")
    ap.add_argument("--manifest", type=Path,
                    default=_HERE / "manifests" / "dataset_manifest_90m.json")

    ap.add_argument("--value-scale", type=float, default=VALUE_SCALE,
                    help="tanh calibration constant; LOCKED at the default. "
                         "value_cp stores the raw score, so this is also a "
                         "collate-time knob via dataset.revalue().")
    ap.add_argument("--value-min-depth", type=int, default=26)
    ap.add_argument("--policy-min-depth", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    ap.add_argument("--cp-clamp", type=int, default=CP_CLAMP)
    ap.add_argument("--val-permille", type=int, default=5,
                    help="sha1(fen) %% 1000 < N goes to val")

    ap.add_argument("--target", type=int, default=91_355_000,
                    help="PRE-rejection selection target. Scale it by the yield "
                         "and val share of a previous manifest to land a given "
                         "number of TRAIN records.")
    ap.add_argument("--shares", type=str,
                    default='{"<=5":0.01,"6-14":0.35,"15-27":0.40,">=28":0.24}')
    ap.add_argument("--policy-share", type=float, default=0.65,
                    help="fraction of each bucket drawn from policy-bearing "
                         "rows. Clamps at the pool ceiling; see feasibility_scan.")
    ap.add_argument("--no-spill", dest="spill", action="store_false",
                    help="when one half's pool is exhausted, do NOT make it up "
                         "from the other half - let the bucket under-deliver "
                         "and the piece-count share targets slip instead.")
    ap.add_argument("--n-train-shards", type=int, default=192)
    ap.add_argument("--n-val-shards", type=int, default=12)
    ap.add_argument("--nest-under", type=Path,
                    default=_HERE / "manifests" / "dataset_manifest.json",
                    help="manifest of a corpus this build must remain a strict "
                         "superset of. Empty path disables the check.")

    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--batch-lines", type=int, default=2000)
    ap.add_argument("--flush-records", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--limit-lines", type=int, default=0,
                    help="stop after N source lines (smoke runs)")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--max-violation-rate", type=float, default=0.001,
                    help="hard-fail if invariant violations exceed this share")
    args = ap.parse_args()

    for p in (args.source, args.index):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    shares = json.loads(args.shares)

    # --- derive per-bucket rates from the index -----------------------------
    # Availability and rate derivation both come from feasibility_scan, which is
    # also what the pre-flight report runs, so the two cannot disagree.
    print("scanning index for per-bucket availability...", file=sys.stderr)
    avail_full = bucket_availability(args.index, args.value_min_depth,
                                     args.policy_min_depth)
    avail = {k: v["eligible"] for k, v in avail_full.items()}
    r_pol, r_val, plan = derive_rates(avail_full, args.target, shares,
                                      args.policy_share, spill=args.spill)
    plan_tot = plan_totals(plan)
    ceiling = coverage_ceiling(avail_full, args.target, shares)

    prior = None
    if str(args.nest_under) and args.nest_under.exists():
        pm = json.loads(args.nest_under.read_text())
        # A single-rate manifest is just the two-rate form with both equal.
        p_pol = pm.get("bucket_sampling_rates_policy") or pm["bucket_sampling_rates"]
        p_val = pm.get("bucket_sampling_rates_value_only") or pm["bucket_sampling_rates"]
        prior = {"manifest": str(args.nest_under),
                 "value_min_depth": pm["value_min_depth"],
                 "policy_min_depth": pm["policy_min_depth"],
                 "bucket_rates_policy": p_pol,
                 "bucket_rates_value_only": p_val}

    cfg = dict(
        value_min_depth=args.value_min_depth,
        policy_min_depth=args.policy_min_depth,
        temperature=args.temperature,
        epsilon=args.epsilon,
        cp_clamp=args.cp_clamp,
        value_scale=args.value_scale,
        val_permille=args.val_permille,
        n_train_shards=args.n_train_shards,
        n_val_shards=args.n_val_shards,
        shard_seed=args.seed,
        bucket_rates_policy=r_pol,
        bucket_rates_value_only=r_val,
        prior=prior,
    )

    print(f"\npolicy_share {args.policy_share:.2f} -> expected coverage "
          f"{plan_tot['expected_coverage']:.2%} (pool ceiling {ceiling:.2%})",
          file=sys.stderr)
    print(f"  {'bucket':<8} {'r_policy':>10} {'r_value':>10} {'expected':>12} "
          f"{'share':>8} {'cov':>8}  clamped", file=sys.stderr)
    for name, _, _ in BUCKETS:
        d = plan[name]
        cl = ",".join(k for k, f in (("policy", d["policy_clamped"]),
                                     ("value-only", d["value_only_clamped"])) if f)
        print(f"  {name:<8} {r_pol[name]:>10.4f} {r_val[name]:>10.4f} "
              f"{d['expected_total']:>12,.0f} {d['expected_share']:>7.2%} "
              f"{d['expected_coverage']:>7.2%}  {cl or '-'}", file=sys.stderr)
    if plan_tot["shortfall"] > 0:
        print(f"  WARNING: expected shortfall {plan_tot['shortfall']:,.0f} rows",
              file=sys.stderr)

    print("\nbuilding selection from index...", file=sys.stderr)
    t0 = time.time()
    diag: dict = {}
    selected = build_selection(args.index, cfg, args.seed, diag=diag)
    print(f"selected {len(selected):,} lines in {time.time()-t0:.1f}s", file=sys.stderr)
    nest = diag.get("nesting")
    if nest is not None:
        print(f"nesting vs {prior['manifest']}: "
              f"{'STRICT SUPERSET' if nest['is_strict_superset'] else 'BROKEN'} "
              f"({nest['prior_rows_dropped']:,} prior rows dropped)", file=sys.stderr)

    n_shards = args.n_train_shards + args.n_val_shards
    ckpt_path = args.out_dir / "pass_b_checkpoint.json"
    shard_counts = [0] * n_shards
    lines_done = 0
    sel_pos = 0
    stats: Counter = Counter()

    if ckpt_path.exists() and not args.restart:
        ck = json.loads(ckpt_path.read_text())
        shard_counts = ck["shard_counts"]
        lines_done = ck["lines_done"]
        sel_pos = ck["sel_pos"]
        stats = Counter(ck["stats"])
        print(f"resuming at source line {lines_done:,} "
              f"({sum(shard_counts):,} records written)", file=sys.stderr)

    # Truncate shards to their checkpointed lengths (a kill can leave a partial tail).
    handles = []
    for i in range(n_shards):
        split = "train" if i < args.n_train_shards else "val"
        j = i if i < args.n_train_shards else i - args.n_train_shards
        p = args.out_dir / shard_name(split, j)
        want = shard_counts[i] * RECORD_DTYPE.itemsize
        if args.restart and p.exists():
            p.unlink()
        if p.exists():
            with open(p, "r+b") as fh:
                fh.truncate(want)
            handles.append(open(p, "ab"))
        else:
            handles.append(open(p, "wb"))

    buffers: list[list[bytes]] = [[] for _ in range(n_shards)]
    buffered = 0

    def flush():
        nonlocal buffered
        for i, buf in enumerate(buffers):
            if buf:
                handles[i].write(b"".join(buf))
                handles[i].flush()
                shard_counts[i] += len(buf)
                buf.clear()
        buffered = 0
        ckpt_path.write_text(json.dumps({
            "lines_done": lines_done, "sel_pos": sel_pos,
            "shard_counts": shard_counts, "stats": dict(stats),
        }))

    def tasks():
        """Yield batches of selected raw lines, in source order."""
        nonlocal lines_done, sel_pos
        batch: list[tuple[int, bytes]] = []
        for _offset, raw in iter_lines(args.source, skip_lines=lines_done):
            ln = lines_done
            lines_done += 1
            if args.limit_lines and ln >= args.limit_lines:
                break
            while sel_pos < len(selected) and selected[sel_pos] < ln:
                sel_pos += 1
            if sel_pos < len(selected) and selected[sel_pos] == ln:
                sel_pos += 1
                batch.append((ln, raw))
                if len(batch) >= args.batch_lines:
                    yield (batch,)
                    batch = []
            if sel_pos >= len(selected):
                break
        if batch:
            yield (batch,)

    pool = None
    t0 = time.time()
    try:
        if args.workers > 1:
            import multiprocessing as mp
            pool = mp.Pool(args.workers, initializer=_init_worker, initargs=(cfg,))
            results = pool.imap(convert_batch, tasks(), chunksize=1)
        else:
            _init_worker(cfg)
            results = (convert_batch(t) for t in tasks())

        for payloads, st in results:
            stats.update(st)
            for shard, blob in payloads:
                buffers[shard].append(blob)
                buffered += 1
            if buffered >= args.flush_records:
                flush()
                done = sum(shard_counts)
                el = time.time() - t0
                print(f"  src {lines_done:,} | written {done:,} | "
                      f"{done/max(el,1e-9):,.0f} rec/s", file=sys.stderr, flush=True)
        flush()
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        for h in handles:
            h.close()

    total = sum(shard_counts)
    accepted = stats["accepted_train"] + stats["accepted_val"]
    violations = stats["reject_invariant_violation"]
    considered = accepted + sum(v for k, v in stats.items() if k.startswith("reject_"))
    viol_rate = violations / considered if considered else 0.0

    # --- derived reports -----------------------------------------------------
    post_hist = {}
    for name, _, _ in BUCKETS:
        n = stats.get(f"bucket_{name}", 0)
        npol = stats.get(f"bucket_{name}_policy", 0)
        post_hist[name] = {
            "count": n,
            "share": n / accepted if accepted else 0.0,
            "with_policy": npol,
            "policy_coverage": npol / n if n else 0.0,
        }
    value_strata = {}
    for k in ("value_exact_zero", "value_mate", "value_middle"):
        v = stats.get(k, 0)
        value_strata[k] = {"count": v, "share": v / accepted if accepted else 0.0}

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": git_sha(),
        "source": str(args.source),
        "source_hash": file_hash(args.source),
        "index": str(args.index),
        "pov_convention": "WHITE_RELATIVE (hardcoded; Phase 1)",
        "value_path": "no conversion; cp clipped +-2000 then tanh(cp/scale); "
                      "mate -> +-0.995; clamp +-0.995",
        "value_cp_encoding": "cp clipped +-2000; mate -> sign*(30000 - "
                             "min(|mate|,1000)); |value_cp| >= 29000 == mate",
        "policy_path": "score * stm, then softmax(rel/T)",
        "mate_mapping": "sign*(11020 - 20*min(|mate|,50)); cp clamped +-10000; "
                        "bands disjoint by construction",
        "value_scale": args.value_scale,
        "value_min_depth": args.value_min_depth,
        "policy_min_depth": args.policy_min_depth,
        "temperature": args.temperature,
        "epsilon": args.epsilon,
        "cp_clamp": args.cp_clamp,
        "val_split": f"sha1(fen) % 1000 < {args.val_permille}",
        "seed": args.seed,
        "target": args.target,
        "target_shares": shares,
        "policy_share": args.policy_share,
        "policy_share_spill": bool(args.spill),
        "policy_coverage_ceiling": ceiling,
        "bucket_available": avail,
        "bucket_available_split": avail_full,
        "bucket_sampling_rates_policy": r_pol,
        "bucket_sampling_rates_value_only": r_val,
        "selection_plan": plan,
        "selection_plan_totals": plan_tot,
        "realised_policy_coverage": (
            sum(stats.get(f"bucket_{nm}_policy", 0) for nm, _, _ in BUCKETS)
            / accepted if accepted else 0.0),
        "nesting": diag.get("nesting", {
            "is_strict_superset": None,
            "reason": "no --nest-under manifest given",
        }),
        "source_order_quantiles": diag.get("source_order_quantiles", {}),
        "selected_lines": int(len(selected)),
        "record_dtype": [[n, str(RECORD_DTYPE.fields[n][0])] for n in RECORD_DTYPE.names],
        "record_size_bytes": RECORD_DTYPE.itemsize,
        "n_train_shards": args.n_train_shards,
        "n_val_shards": args.n_val_shards,
        "shard_counts": shard_counts,
        "records_total": total,
        "records_train": sum(shard_counts[:args.n_train_shards]),
        "records_val": sum(shard_counts[args.n_train_shards:]),
        "invariant_violation_rate": viol_rate,
        "post_selection_piece_histogram": post_hist,
        "value_strata": value_strata,
        "rejection_histogram": {k: v for k, v in sorted(stats.items())
                                if k.startswith("reject_")},
        "counters": {k: v for k, v in sorted(stats.items())
                     if not k.startswith("reject_")},
        "notes": [
            "duplicate_pvs_removed: repeated PVs dropped BEFORE the softmax; "
            "keying on full Move so underpromotions survive and merge at 4096 via +=.",
            "unique_pv_count_N: distribution of unique moves per selected block "
            "(N capped at 16).",
            "value_cp holds the RAW White-POV score: cp clipped +-2000, or "
            "sign*(30000 - min(|mate|,1000)) for a forced mate, so mate DISTANCE "
            "survives. |value_cp| >= 29000 identifies a mate. value_scale is "
            "therefore a collate-time transform via dataset.revalue() - changing "
            "it does NOT require re-running Pass B. `value` is the target at the "
            "scale recorded here, with every mate pinned to +-0.995 regardless "
            "of distance.",
            "value_scale 290.6806 is the Lc0 WDL calibration constant (v ~ "
            "expected score), shared with benchmarking/player/acpl_elo_estimator.py. "
            "It is NOT fitted to this corpus and deliberately so: MCTS Q backup "
            "and the tuned C_PUCT / VALUE_LOSS are denominated in these units. "
            "~11.5% of cp values saturate above |v|=0.99; accepted.",
            "post_selection_piece_histogram.policy_coverage: NATURAL policy "
            "coverage is roughly flat across piece-count buckets (Pass A "
            "predicts 43/37/42% for 6-14 / 15-27 / >=28), so policy coverage "
            "and piece count are close to independent. This build DOES "
            "preferentially sample policy-bearing rows - see policy_share - but "
            "because the two are independent, that oversampling needs no "
            "piece-count-stratified correction: the split is applied WITHIN "
            "each bucket, so target_shares are unaffected. The coverage numbers "
            "here are therefore post-oversampling and are NOT the natural rate.",
            "policy_share is a request, not a guarantee: "
            "policy_coverage_ceiling is the highest coverage the pool can give "
            "at this target and share vector, and a bucket whose policy pool "
            "runs out has its deficit made up from its value-only pool (see "
            "policy_share_spill and selection_plan[*].spilled_to_value_only). "
            "That keeps target_shares exact and lets coverage give instead.",
            "source_order_quantiles: policy coverage varies ~4x along SOURCE "
            "ORDER (dead end D5 - source order is not exchangeable), so heavy "
            "policy oversampling can pull the selected set toward whichever "
            "region of the dump is policy-dense, which correlates with a "
            "different Stockfish-version mix. Compare selected_policy against "
            "eligible_policy: if the selected quantiles are tighter, the policy "
            "labels are drawn from a narrower slice of the file than the value "
            "labels and are correspondingly more homogeneous in provenance.",
            "value_strata: report value metrics against these three groups, not "
            "in aggregate. A head collapsing toward 0 scores well on the "
            "exact-zero mass by doing nothing; also track prediction std.",
            "LABEL NOISE: the dump carries no engine version field and was "
            "produced by many browser Stockfish builds, so cp is heterogeneous "
            "in scale across positions. Irreducible here; not corrected for.",
            "Colour-mirror augmentation is applied at collate time "
            "(dataset.MultiPVCollate, 50% default), NOT baked into the shards.",
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nPass B complete: {total:,} records "
          f"({manifest['records_train']:,} train / {manifest['records_val']:,} val) "
          f"in {(time.time()-t0)/60:.1f} min", file=sys.stderr)

    print("\npost-selection piece histogram (policy coverage conditioned on it):",
          file=sys.stderr)
    print(f"  {'bucket':<8} {'count':>12} {'share':>8} {'w/ policy':>12} {'cov':>8}",
          file=sys.stderr)
    for name, d in post_hist.items():
        print(f"  {name:<8} {d['count']:>12,} {d['share']:>7.2%} "
              f"{d['with_policy']:>12,} {d['policy_coverage']:>7.2%}",
          file=sys.stderr)

    realised_cov = manifest["realised_policy_coverage"]
    print(f"\npolicy coverage: requested {args.policy_share:.2%}, "
          f"realised {realised_cov:.2%}, pool ceiling {ceiling:.2%}",
          file=sys.stderr)

    print("\nvalue strata (report metrics against these, not in aggregate):",
          file=sys.stderr)
    for k, d in value_strata.items():
        print(f"  {k:<18} {d['count']:>12,} {d['share']:>7.2%}", file=sys.stderr)

    soq = diag.get("source_order_quantiles")
    if soq:
        print("\nsource-order quantiles (fraction of file; selected vs eligible).",
              file=sys.stderr)
        print("A tighter selected spread than eligible means those labels come "
              "from a narrower\nslice of the dump - i.e. a narrower SF-version "
              "mix. Watch the policy rows.", file=sys.stderr)
        hdr = "  ".join(f"q{int(q*100):02d}" for q in soq["quantiles"])
        print(f"  {'population':<22} {'count':>12}  {hdr}", file=sys.stderr)
        for k, vals in soq["populations"].items():
            cells = "  ".join(f"{v:.3f}" for v in vals)
            print(f"  {k:<22} {soq['counts'][k]:>12,}  {cells}", file=sys.stderr)

    print("\nrejections:", file=sys.stderr)
    for k, v in sorted(stats.items()):
        if k.startswith("reject_"):
            print(f"  {k[7:]:<26} {v:>12,}", file=sys.stderr)
    print(f"\ndedup: {stats['duplicate_pvs_removed']:,} duplicate PVs removed "
          f"across {stats['positions_with_duplicate_pvs']:,} positions",
          file=sys.stderr)

    print(f"\nmanifest -> {args.manifest}", file=sys.stderr)

    if viol_rate > args.max_violation_rate:
        print(f"\nHARD FAIL: invariant violation rate {viol_rate:.4%} exceeds "
              f"{args.max_violation_rate:.4%}", file=sys.stderr)
        return 1
    print(f"invariant violation rate {viol_rate:.4%} "
          f"(gate {args.max_violation_rate:.4%}) OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
