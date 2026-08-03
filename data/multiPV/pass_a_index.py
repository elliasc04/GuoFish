"""PASS A - cheap index over every line of the Lichess eval dump.

JSON parsing ONLY. No python-chess, no board construction, no legal-move
generation: those cost ~100x more per line and belong in Pass B, which touches
only the ~25-30M selected lines.

Per line we record:
    offset        decompressed byte offset of the line start (Pass B seeks here)
    piece_count   character count of the FEN board field (endgame stratification)
    max_depth     deepest eval block
    max_pvs       widest eval block
    policy_depth  deepest block that has >=2 PVs, or -1

`policy_depth` is stored instead of the brief's "whether any block meets
policy_min_depth with >=2 PVs" boolean deliberately: it is strictly more
informative and lets the policy threshold be re-chosen from the index without
re-running a 400M-line pass.

Streams the compressed dump; never decompresses to disk. Resumable: progress is
checkpointed after every flush, and a resumed run re-streams (decompression only,
no parsing) to the checkpoint before appending.

Usage:
    python data/multiPV/pass_a_index.py --source data/multiPV/lichess_db_eval.jsonl.zst
    python data/multiPV/pass_a_index.py --source ... --limit 2000000   # smoke run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Packed (align=False) => exactly 14 bytes/row. 400M lines ~= 5.6 GB.
INDEX_DTYPE = np.dtype([
    ("offset", np.int64),
    ("piece_count", np.uint8),
    ("max_depth", np.int16),
    ("max_pvs", np.uint8),
    ("policy_depth", np.int16),
], align=False)

# Board-field characters that are NOT pieces.
_NON_PIECE = set("012345678/")


def count_pieces(board_field: str) -> int:
    """Piece count from the FEN board field by character counting."""
    n = 0
    for ch in board_field:
        if ch not in _NON_PIECE:
            n += 1
    return n


def scan_line(offset: int, raw: bytes):
    """Extract one index row from a raw JSONL line. Returns a tuple or None."""
    try:
        rec = json.loads(raw)
    except Exception:
        return None
    fen = rec.get("fen")
    evals = rec.get("evals")
    if not fen or not evals:
        return None

    sp = fen.find(" ")
    board_field = fen if sp < 0 else fen[:sp]
    pc = count_pieces(board_field)

    max_depth = -1
    max_pvs = 0
    policy_depth = -1
    for ev in evals:
        d = ev.get("depth")
        d = -1 if d is None else int(d)
        pvs = ev.get("pvs")
        npv = 0 if not pvs else len(pvs)
        if d > max_depth:
            max_depth = d
        if npv > max_pvs:
            max_pvs = npv
        if npv >= 2 and d > policy_depth:
            policy_depth = d

    return (offset,
            min(pc, 255),
            max(-1, min(max_depth, 32767)),
            min(max_pvs, 255),
            max(-1, min(policy_depth, 32767)))


def scan_batch(batch):
    """Worker entry point: list[(offset, raw)] -> packed index array."""
    rows = []
    for offset, raw in batch:
        row = scan_line(offset, raw)
        if row is not None:
            rows.append(row)
    return np.array(rows, dtype=INDEX_DTYPE) if rows else np.empty(0, dtype=INDEX_DTYPE)


def iter_lines(source: Path, skip_lines: int = 0, read_chunk: int = 1 << 22):
    """Yield (decompressed_offset, raw_line_bytes).

    Works on bytes throughout - no TextIOWrapper decode - because json.loads
    accepts bytes and byte offsets must be exact for Pass B to seek by them.
    `skip_lines` fast-forwards without yielding (decompression only), which is
    how resume works: a zstd stream cannot be randomly seeked.
    """
    import zstandard

    with open(source, "rb") as fh:
        with zstandard.ZstdDecompressor().stream_reader(fh) as reader:
            buf = b""
            offset = 0
            seen = 0
            while True:
                data = reader.read(read_chunk)
                if not data:
                    break
                if buf:
                    buf += data
                else:
                    buf = data
                parts = buf.split(b"\n")
                buf = parts.pop()
                for p in parts:
                    if seen >= skip_lines:
                        yield offset, p
                    offset += len(p) + 1
                    seen += 1
            if buf:
                if seen >= skip_lines:
                    yield offset, buf


def main() -> int:
    ap = argparse.ArgumentParser(description="Pass A: index the eval dump")
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "index")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N lines (0 = whole file). For smoke runs.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="parser processes. 1 = in-process, no pool.")
    ap.add_argument("--batch-lines", type=int, default=20_000,
                    help="lines per work unit dispatched to a worker")
    ap.add_argument("--flush-batches", type=int, default=20,
                    help="completed batches between index flush + checkpoint")
    ap.add_argument("--restart", action="store_true",
                    help="ignore any checkpoint and start over")
    args = ap.parse_args()

    if not args.source.exists():
        print(f"ERROR: source not found: {args.source}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "pass_a_index.bin"
    ckpt_path = args.out_dir / "pass_a_checkpoint.json"

    lines_done = 0
    rows_done = 0
    if ckpt_path.exists() and not args.restart:
        ck = json.loads(ckpt_path.read_text())
        lines_done = int(ck["lines_done"])
        rows_done = int(ck["rows_done"])
        # Truncate any bytes written past the checkpoint (partial flush after a kill).
        want = rows_done * INDEX_DTYPE.itemsize
        if index_path.exists() and index_path.stat().st_size > want:
            with open(index_path, "r+b") as fh:
                fh.truncate(want)
        print(f"resuming at line {lines_done:,} ({rows_done:,} rows indexed)",
              file=sys.stderr)
    elif args.restart and index_path.exists():
        index_path.unlink()

    mode = "ab" if lines_done else "wb"
    t0 = time.time()
    n_lines = lines_done
    n_rows = rows_done
    pending: list = []
    pool = None
    if args.workers > 1:
        import multiprocessing as mp
        pool = mp.Pool(args.workers)

    def batches():
        nonlocal n_lines
        cur: list = []
        for offset, raw in iter_lines(args.source, skip_lines=lines_done):
            if args.limit and n_lines >= args.limit:
                break
            cur.append((offset, raw))
            n_lines += 1
            if len(cur) >= args.batch_lines:
                yield cur
                cur = []
        if cur:
            yield cur

    try:
        with open(index_path, mode) as out:
            def flush(arrays):
                nonlocal n_rows
                if not arrays:
                    return
                merged = np.concatenate(arrays)
                out.write(merged.tobytes())
                out.flush()
                n_rows += len(merged)
                ckpt_path.write_text(json.dumps({
                    "lines_done": n_lines, "rows_done": n_rows,
                    "source": str(args.source),
                    "dtype": [[k, str(INDEX_DTYPE.fields[k][0])] for k in INDEX_DTYPE.names],
                }, indent=2))

            if pool is not None:
                it = pool.imap(scan_batch, batches(), chunksize=1)
            else:
                it = (scan_batch(b) for b in batches())

            for arr in it:
                pending.append(arr)
                if len(pending) >= args.flush_batches:
                    flush(pending)
                    pending = []
                    el = time.time() - t0
                    done = n_lines - lines_done
                    print(f"  {n_lines:,} lines  {n_rows:,} rows  "
                          f"{done / max(el, 1e-9):,.0f} lines/s", file=sys.stderr, flush=True)
            flush(pending)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    elapsed = time.time() - t0
    print(f"\nPass A complete: {n_lines:,} lines -> {n_rows:,} index rows "
          f"in {elapsed / 60:.1f} min", file=sys.stderr)
    print(f"index: {index_path} ({index_path.stat().st_size / 2**30:.2f} GiB)",
          file=sys.stderr)
    return 0


def load_index(path: Path) -> np.ndarray:
    """Memmap a Pass A index written by this script."""
    return np.memmap(path, dtype=INDEX_DTYPE, mode="r")


if __name__ == "__main__":
    raise SystemExit(main())
