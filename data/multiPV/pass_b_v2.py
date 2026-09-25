"""Corpus v2 builder (v6 design doc §6.1-6.4). pass_b_convert.py (v1) is left as is.

    python data/multiPV/pass_b_v2.py --out-dir data/processed/multipv_v2 [--limit 200000]

What changes from v1 (everything else - parsing, filters, labels, dedup of
repeated PVs, val split, shard routing - is v1's code, imported, not copied):

  record    v2, 387 B (training/v6/data/formats.py): + value_depth, hard_move,
            origin, src_line; pv_score stored exactly as int16.
  roots     value_min_depth 24. Per-(bucket, policy|value-only) rates are
            derived for --target as in v1, then FLOORED at the --floor-manifest
            (90M) rates, so every 90M line is selected again. The nesting check
            runs before any conversion and a dropped 90M line is a hard fail.
  hard_move first move of the value block's pvs[0].line (the block the value
            label comes from), from*64+to; -1 when missing / unparseable /
            illegal, counted by reason. Kept on policy-bearing rows too.
  derived   for k = 1..--max-ply: the position after k moves of that PV, if all
            k moves are legal, the position is not terminal, a legal (k+1)-th
            move exists and value_depth - k >= --derived-min-depth. Kept with a
            seeded probability --derived-rate. Labels: the root's value (mate
            distance shortened by the mating side's moves played), hard_move =
            PV move k+1, has_policy 0, origin k, value_depth = root depth - k.
            A derived position inherits its root's split; val ones go to
            valderived_* shards, never into val_*.
  dedup     post-pass: a derived position whose 64-bit position hash matches
            any root (either split) or an earlier derived position is dropped.
            Survivors are appended to the train shards / written to valderived.
  H1        --out-dir is required and must be new or empty; a directory holding
            a manifest is refused. There is no resume: rebuild into a new dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from array import array
from collections import Counter
from pathlib import Path

import chess
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (_ROOT, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.pgn_parallel import _board_to_tokens  # noqa: E402
from feasibility_scan import (  # noqa: E402
    BUCKETS, bucket_availability, bucket_name, coverage_ceiling, derive_rates, plan_totals,
)
from labels import (  # noqa: E402
    CP_CLAMP, DEFAULT_EPSILON, DEFAULT_TEMPERATURE, MAX_LEGAL, MAX_PV, VALUE_MATE_BASE,
    VALUE_MATE_MAX_DISTANCE, VALUE_SCALE, build_policy_target, move_index, select_policy_block,
    select_value_block, value_cp_is_mate, value_from_raw_cp, value_raw_cp,
)
from pass_a_index import INDEX_DTYPE, iter_lines  # noqa: E402
from pass_b_convert import _splitmix64, build_selection, file_hash  # noqa: E402
from record_format import shard_name  # noqa: E402
from training.v6.ckpt import git_state  # noqa: E402
from training.v6.data.formats import V2_DTYPE, dtype_descr  # noqa: E402

DERIVED_STREAM = 0x5D1E_7E0D


def position_key(board: chess.Board) -> int:
    """64-bit hash of placement, side, castling and legal-ep (no move counters)."""
    fen = " ".join(board.fen(en_passant="legal").split(" ")[:4])
    return int.from_bytes(hashlib.blake2b(fen.encode(), digest_size=8).digest(), "little")


def _parse_board(fen: str):
    """v1's board rules: (board, None) or (None, reject reason)."""
    try:
        board = chess.Board(fen)
    except ValueError:
        try:
            b960 = chess.Board(fen, chess960=True)
        except ValueError:
            return None, "unparseable"
        return None, "chess960_skipped" if b960.is_valid() else "invalid_fen"
    if not board.is_valid():
        try:
            if chess.Board(fen, chess960=True).is_valid():
                return None, "chess960_skipped"
        except ValueError:
            pass
        return None, "invalid_fen"
    return board, None


def _legal_fill(out, board: chess.Board, stats: Counter) -> None:
    legal = list(board.legal_moves)
    if len(legal) > MAX_LEGAL:
        stats["legal_truncated"] += 1
        legal = legal[:MAX_LEGAL]
    out["n_legal"][0] = len(legal)
    out["legal_idx"][0, :len(legal)] = np.asarray([move_index(m) for m in legal], dtype=np.int16)


def _parse_move(board: chess.Board, uci: str):
    """-> (move, None) or (None, reason); python-chess normalises king-takes-rook
    castling on a standard board, as v1's parse did."""
    try:
        return board.parse_uci(uci), None
    except chess.IllegalMoveError:
        return None, "illegal"
    except chess.InvalidMoveError:
        return None, "unparseable"


def _shorten_mate(raw_cp: int, root_white: bool, k: int):
    """Mate distance after k PV plies: the mating side has played some of its
    moves. Returns the new raw value_cp, or None when the mate is used up."""
    if not value_cp_is_mate(raw_cp):
        return raw_cp
    mate = VALUE_MATE_BASE - abs(raw_cp)
    if mate >= VALUE_MATE_MAX_DISTANCE:
        return raw_cp                                   # distance saturated; keep as stored
    white_mates = raw_cp > 0
    played = sum(1 for i in range(1, k + 1) if (root_white if i % 2 else not root_white) == white_mates)
    left = mate - played
    if left <= 0:
        return None
    return (1 if white_mates else -1) * (VALUE_MATE_BASE - left)


def convert_v2(line_no: int, raw: bytes, cfg: dict, stats: Counter):
    """One dump line -> (root_bytes, root_key, is_val, [(k, bytes, key)]) or (None, reason)."""
    try:
        rec = json.loads(raw)
    except ValueError:                      # JSONDecodeError and UnicodeDecodeError
        return None, "json_error"
    fen, evals = rec.get("fen"), rec.get("evals") or []
    if not fen or not evals:
        return None, "no_evals"
    board, why = _parse_board(fen)
    if board is None:
        return None, why
    if not any(True for _ in board.legal_moves):
        return None, "no_legal_moves"

    vblock = select_value_block(evals, cfg["value_min_depth"])
    if vblock is None:
        return None, "below_value_depth"
    raw_cp = value_raw_cp(vblock["pvs"][0])
    if raw_cp is None:
        return None, "value_pv_missing_score"
    vdepth = int(vblock["depth"])
    if not 0 <= vdepth <= 255:
        return None, "value_depth_out_of_range"

    out = np.zeros(1, dtype=V2_DTYPE)
    out["tokens"][0] = np.asarray(_board_to_tokens(board), dtype=np.int8)
    out["value"][0] = np.float16(value_from_raw_cp(raw_cp, cfg["value_scale"]))
    out["value_cp"][0] = np.int16(raw_cp)
    out["value_depth"][0] = vdepth
    out["origin"][0] = 0
    out["src_line"][0] = line_no
    _legal_fill(out, board, stats)

    sel = select_policy_block(board, evals, cfg["policy_min_depth"], cfg["cp_clamp"])
    if sel is None:
        stats["value_only"] += 1
    else:
        entries, _d, dropped, removed = sel
        stats["pv_dropped_missing_score"] += dropped
        if removed:
            stats["duplicate_pvs_removed"] += removed
            stats["positions_with_duplicate_pvs"] += 1
        idxs, probs, scores, ok = build_policy_target(
            entries, board.turn == chess.WHITE, cfg["temperature"], cfg["epsilon"], MAX_PV)
        if not ok:
            return None, "invariant_violation"
        if len(entries) > MAX_PV:
            stats["policy_truncated"] += 1
        n = len(idxs)
        out["has_policy"][0] = 1
        out["n_pv"][0] = n
        out["pv_idx"][0, :n] = np.asarray(idxs, dtype=np.int16)
        out["pv_prob"][0, :n] = np.asarray(probs, dtype=np.float16)
        out["pv_score"][0, :n] = np.asarray([max(-32000, min(32000, s)) for s in scores],
                                            dtype=np.int16)
        if abs(float(np.sum(np.asarray(probs, dtype=np.float64))) - (1.0 - cfg["epsilon"])) > 1e-4:
            return None, "mass_check_failed"

    # hard move: first move of the value block's principal variation
    line = (vblock["pvs"][0].get("line") or "").split()
    first, why = (None, "missing_line") if not line else _parse_move(board, line[0])
    out["hard_move"][0] = -1 if first is None else move_index(first)
    stats[f"hard_move_{'ok' if first is not None else why}"] += 1

    b = bucket_name(chess.popcount(board.occupied))
    stats[f"bucket_{b}"] += 1
    stats[f"bucket_{b}_policy"] += int(out["has_policy"][0])

    derived = []
    if first is not None and cfg["max_ply"] > 0:
        derived = _unroll(board, line, first, raw_cp, vdepth, line_no, cfg, stats)
    is_val = (int.from_bytes(hashlib.sha1(fen.encode()).digest()[:8], "big")
              % 1000) < cfg["val_permille"]
    return (out.tobytes(), position_key(board), is_val, derived), None


def _unroll(root: chess.Board, line: list, first: chess.Move, raw_cp: int, vdepth: int,
            line_no: int, cfg: dict, stats: Counter) -> list:
    out = []
    b = root.copy(stack=False)
    move = first
    for k in range(1, cfg["max_ply"] + 1):
        b.push(move)
        if vdepth - k < cfg["derived_min_depth"]:
            stats["derived_stop_depth"] += 1
            break
        if b.is_game_over():
            stats["derived_stop_terminal"] += 1
            break
        if len(line) <= k:
            stats["derived_stop_line_end"] += 1
            break
        nxt, why = _parse_move(b, line[k])
        if nxt is None:
            stats[f"derived_stop_next_{why}"] += 1
            break
        stats[f"derived_candidates_k{k}"] += 1
        u = _splitmix64(((line_no << 3) | k) ^ cfg["derived_seed"]) / 2.0 ** 64
        if u < cfg["derived_rate"]:
            new_cp = _shorten_mate(raw_cp, root.turn == chess.WHITE, k)
            if new_cp is None:
                stats["derived_stop_mate_used_up"] += 1
                break
            rec = np.zeros(1, dtype=V2_DTYPE)
            rec["tokens"][0] = np.asarray(_board_to_tokens(b), dtype=np.int8)
            rec["value"][0] = np.float16(value_from_raw_cp(new_cp, cfg["value_scale"]))
            rec["value_cp"][0] = np.int16(new_cp)
            rec["value_depth"][0] = vdepth - k
            rec["hard_move"][0] = move_index(nxt)
            rec["origin"][0] = k
            rec["src_line"][0] = line_no
            _legal_fill(rec, b, stats)
            out.append((k, rec.tobytes(), position_key(b)))
            stats[f"derived_sampled_k{k}"] += 1
        move = nxt
    return out


def convert_batch(task):
    lines, cfg = task
    stats: Counter = Counter()
    results = []
    for line_no, raw in lines:
        res, why = convert_v2(line_no, raw, cfg, stats)
        if res is None:
            stats[f"reject_{why}"] += 1
        results.append((line_no, res))
    return results, stats


def _require_new_dir(out_dir: Path) -> None:
    if (out_dir / "manifest.json").exists():
        raise SystemExit(f"{out_dir} already holds a manifest; refusing (H1)")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} exists and is not empty; --out-dir must be new or empty (H1)")
    out_dir.mkdir(parents=True, exist_ok=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=_HERE / "lichess_db_eval.jsonl.zst")
    ap.add_argument("--index", type=Path, default=_HERE / "index" / "pass_a_index.bin")
    ap.add_argument("--floor-manifest", type=Path, default=_HERE / "manifests" / "dataset_manifest_90m.json",
                    help="rates are floored at this build's rates and it must nest")
    ap.add_argument("--value-scale", type=float, default=VALUE_SCALE)
    ap.add_argument("--value-min-depth", type=int, default=24)
    ap.add_argument("--policy-min-depth", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    ap.add_argument("--cp-clamp", type=int, default=CP_CLAMP)
    ap.add_argument("--val-permille", type=int, default=5)
    ap.add_argument("--target", type=int, default=121_100_000,
                    help="pre-rejection root selection target (~120M roots at the 90M yield)")
    ap.add_argument("--shares", type=str, default='{"<=5":0.01,"6-14":0.35,"15-27":0.40,">=28":0.24}')
    ap.add_argument("--policy-share", type=float, default=0.65)
    ap.add_argument("--no-spill", dest="spill", action="store_false")
    ap.add_argument("--max-ply", type=int, default=2, choices=range(0, 5))
    ap.add_argument("--derived-rate", type=float, default=0.125)
    ap.add_argument("--derived-min-depth", type=int, default=20)
    ap.add_argument("--n-train-shards", type=int, default=256)
    ap.add_argument("--n-val-shards", type=int, default=16)
    ap.add_argument("--n-valderived-shards", type=int, default=4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--batch-lines", type=int, default=2000)
    ap.add_argument("--flush-records", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--limit", type=int, default=0, help="stop after N source lines (smoke runs)")
    ap.add_argument("--max-violation-rate", type=float, default=0.001)
    args = ap.parse_args(argv)
    if not 0.0 <= args.derived_rate <= 1.0:
        raise SystemExit("--derived-rate must be in [0, 1]")
    for p in (args.source, args.index, args.floor_manifest):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    _require_new_dir(args.out_dir)
    t_start = time.time()

    # ---- selection: derive, floor, nest --------------------------------
    shares = json.loads(args.shares)
    floor = json.loads(args.floor_manifest.read_text())
    avail = bucket_availability(args.index, args.value_min_depth, args.policy_min_depth)
    r_pol_d, r_val_d, plan = derive_rates(avail, args.target, shares, args.policy_share, spill=args.spill)
    r_pol = {b: max(r_pol_d[b], floor["bucket_sampling_rates_policy"][b]) for b, _, _ in BUCKETS}
    r_val = {b: max(r_val_d[b], floor["bucket_sampling_rates_value_only"][b]) for b, _, _ in BUCKETS}
    if floor["seed"] != args.seed or floor["value_min_depth"] < args.value_min_depth \
            or floor["policy_min_depth"] != args.policy_min_depth:
        raise SystemExit("floor manifest seed/depths are incompatible with nesting")
    prior = {"manifest": str(args.floor_manifest), "value_min_depth": floor["value_min_depth"],
             "policy_min_depth": floor["policy_min_depth"],
             "bucket_rates_policy": floor["bucket_sampling_rates_policy"],
             "bucket_rates_value_only": floor["bucket_sampling_rates_value_only"]}
    sel_cfg = {"value_min_depth": args.value_min_depth, "policy_min_depth": args.policy_min_depth,
               "bucket_rates_policy": r_pol, "bucket_rates_value_only": r_val, "prior": prior}
    diag: dict = {}
    sel_index = args.index
    if args.limit:
        # Smoke runs select over the first N index rows only: the u stream is
        # positional (seeded from row 0), so this is exactly the full build's
        # selection of those rows, without holding ~120M line numbers in RAM.
        sel_index = args.out_dir / "_index_head.bin"
        np.asarray(np.memmap(args.index, dtype=INDEX_DTYPE, mode="r")[:args.limit]).tofile(sel_index)
    selected = build_selection(sel_index, sel_cfg, args.seed, diag=diag)
    if args.limit:
        sel_index.unlink()
    print(f"selected {len(selected):,} lines; nesting vs {args.floor_manifest.name}: "
          f"{diag['nesting']['prior_rows_dropped']:,} prior rows dropped", file=sys.stderr, flush=True)
    if diag["nesting"]["prior_rows_dropped"]:
        raise SystemExit("nesting broken: the floored rates drop 90M lines")

    cfg = dict(value_min_depth=args.value_min_depth, policy_min_depth=args.policy_min_depth,
               temperature=args.temperature, epsilon=args.epsilon, cp_clamp=args.cp_clamp,
               value_scale=args.value_scale, val_permille=args.val_permille, max_ply=args.max_ply,
               derived_rate=args.derived_rate, derived_min_depth=args.derived_min_depth,
               derived_seed=args.seed ^ DERIVED_STREAM)

    # ---- conversion ----------------------------------------------------
    nt, nv, nvd = args.n_train_shards, args.n_val_shards, args.n_valderived_shards
    paths = ([args.out_dir / shard_name("train", i) for i in range(nt)]
             + [args.out_dir / shard_name("val", i) for i in range(nv)])
    handles = [open(p, "wb") for p in paths]
    counts = [0] * len(paths)
    bufs: list[list[bytes]] = [[] for _ in paths]
    staging_path = args.out_dir / "_derived_staging.bin"
    staging = open(staging_path, "wb")
    # 8 bytes per key, not a Python int each: ~1 GB at 120M roots
    root_keys, d_keys = array("Q"), array("Q")
    d_val, d_k, d_line = array("B"), array("B"), array("q")
    stats: Counter = Counter()
    buffered = 0

    def flush():
        nonlocal buffered
        for i, bf in enumerate(bufs):
            if bf:
                handles[i].write(b"".join(bf))
                counts[i] += len(bf)
                bf.clear()
        buffered = 0

    def tasks():
        pos, batch = 0, []
        for ln, (_off, raw) in enumerate(iter_lines(args.source)):
            if args.limit and ln >= args.limit:
                break
            while pos < len(selected) and selected[pos] < ln:
                pos += 1
            if pos >= len(selected):
                break
            if selected[pos] == ln:
                batch.append((ln, raw))
                if len(batch) >= args.batch_lines:
                    yield batch, cfg
                    batch = []
        if batch:
            yield batch, cfg

    pool = None
    t0 = time.time()
    try:
        if args.workers > 1:
            import multiprocessing as mp
            pool = mp.Pool(args.workers)
            results = pool.imap(convert_batch, tasks(), chunksize=1)
        else:
            results = (convert_batch(t) for t in tasks())
        for batch_results, st in results:
            stats.update(st)
            for line_no, res in batch_results:
                if res is None:
                    continue
                root, key, is_val, derived = res
                h = _splitmix64(line_no ^ args.seed)
                shard = nt + h % nv if is_val else h % nt
                bufs[shard].append(root)
                buffered += 1
                stats["roots_val" if is_val else "roots_train"] += 1
                root_keys.append(key)
                for k, blob, dkey in derived:
                    staging.write(blob)
                    d_keys.append(dkey)
                    d_val.append(is_val)
                    d_line.append(line_no)
                    d_k.append(k)
            if buffered >= args.flush_records:
                flush()
                print(f"  roots {sum(counts):,} | derived staged {len(d_keys):,} | "
                      f"{sum(counts) / max(time.time() - t0, 1e-9):,.0f} roots/s", file=sys.stderr, flush=True)
        flush()
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        staging.close()

    # ---- derived dedup post-pass ---------------------------------------
    rk = np.unique(np.frombuffer(root_keys, dtype=np.uint64))
    dk = np.frombuffer(d_keys, dtype=np.uint64)
    dup_root = np.isin(dk, rk)
    first = np.zeros(len(dk), dtype=bool)
    first[np.unique(dk, return_index=True)[1]] = True
    keep = ~dup_root & first
    stats["derived_dropped_root_duplicate"] = int(dup_root.sum())
    stats["derived_dropped_derived_duplicate"] = int((~dup_root & ~first).sum())
    d_val_a = np.frombuffer(d_val, dtype=np.uint8).astype(bool)
    d_route = np.array([_splitmix64(((ln << 3) | k) ^ cfg["derived_seed"])
                        for ln, k in zip(d_line, d_k)], dtype=np.uint64)
    shard_of = np.where(d_val_a, d_route % np.uint64(nvd), d_route % np.uint64(nt)).astype(np.int64)
    vd_paths = [args.out_dir / shard_name("valderived", i) for i in range(nvd)]
    vd_handles = [open(p, "wb") for p in vd_paths]
    vd_counts = [0] * nvd
    if len(dk):
        staged = np.memmap(staging_path, dtype=V2_DTYPE, mode="r")
        for s in range(0, len(dk), 1 << 20):              # stream: never all staged rows in RAM
            sl = slice(s, min(len(dk), s + (1 << 20)))
            rows = np.asarray(staged[sl])
            kp, val, sh = keep[sl], d_val_a[sl], shard_of[sl]
            for i in np.unique(sh[kp]):
                for is_val, hs, cnt in ((True, vd_handles, vd_counts), (False, handles, counts)):
                    m = kp & (val == is_val) & (sh == i)
                    if m.any():
                        hs[i].write(rows[m].tobytes())
                        cnt[i] += int(m.sum())
        staged._mmap.close()
        del staged
    stats["derived_written_train"] = int((keep & ~d_val_a).sum())
    stats["derived_written_valderived"] = int((keep & d_val_a).sum())
    for h_ in handles + vd_handles:
        h_.close()
    staging_path.unlink()

    # ---- manifest ------------------------------------------------------
    roots = stats["roots_train"] + stats["roots_val"]
    hard_ok = stats["hard_move_ok"]
    considered = roots + sum(v for k, v in stats.items() if k.startswith("reject_"))
    viol = stats["reject_invariant_violation"] / considered if considered else 0.0
    git = git_state()
    git.pop("_patch")
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "data/multiPV/pass_b_v2.py", **git,
        "source": str(args.source), "source_hash": file_hash(args.source), "index": str(args.index),
        "limit_lines": args.limit or None,
        "record_format": "v2", "record_dtype": dtype_descr(V2_DTYPE),
        "record_size_bytes": V2_DTYPE.itemsize,
        "value_scale": args.value_scale, "value_min_depth": args.value_min_depth,
        "policy_min_depth": args.policy_min_depth, "temperature": args.temperature,
        "epsilon": args.epsilon, "cp_clamp": args.cp_clamp,
        "val_split": f"sha1(fen) % 1000 < {args.val_permille}", "seed": args.seed,
        "target": args.target, "target_shares": shares, "policy_share": args.policy_share,
        "policy_share_spill": bool(args.spill), "bucket_available_split": avail,
        "selection_plan": plan, "selection_plan_totals": plan_totals(plan),
        "policy_coverage_ceiling": coverage_ceiling(avail, args.target, shares),
        "bucket_rates_policy_derived": r_pol_d, "bucket_rates_value_only_derived": r_val_d,
        "floor_manifest": str(args.floor_manifest),
        "bucket_sampling_rates_policy": r_pol, "bucket_sampling_rates_value_only": r_val,
        "nesting": diag["nesting"], "source_order_quantiles": diag["source_order_quantiles"],
        "selection_scope": (f"first {args.limit:,} index rows (smoke)" if args.limit
                            else "whole index"),
        "selected_lines": int(len(selected)),
        "derived": {"max_ply": args.max_ply, "rate": args.derived_rate,
                    "min_remaining_depth": args.derived_min_depth},
        "n_train_shards": nt, "n_val_shards": nv, "n_valderived_shards": nvd,
        "shard_counts": {"train": counts[:nt], "val": counts[nt:], "valderived": vd_counts},
        "records_train": sum(counts[:nt]), "records_val": sum(counts[nt:]),
        "records_valderived": sum(vd_counts), "roots": roots,
        "hard_move_coverage": hard_ok / roots if roots else None,
        "hard_move_failures": {k[10:]: v for k, v in stats.items()
                               if k.startswith("hard_move_") and k != "hard_move_ok"},
        "invariant_violation_rate": viol,
        "rejection_histogram": {k: v for k, v in sorted(stats.items()) if k.startswith("reject_")},
        "counters": {k: v for k, v in sorted(stats.items()) if not k.startswith("reject_")},
        "seconds": round(time.time() - t_start, 1),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("selected_lines", "roots", "records_train",
                      "records_val", "records_valderived", "hard_move_coverage",
                      "hard_move_failures", "invariant_violation_rate", "seconds")}, indent=1))
    if viol > args.max_violation_rate:
        print(f"HARD FAIL: invariant violation rate {viol:.4%}", file=sys.stderr)
        return 1
    if roots and hard_ok / roots < 0.99:
        print(f"S9: hard_move on {hard_ok / roots:.2%} of roots (< 99%); investigate before training",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
