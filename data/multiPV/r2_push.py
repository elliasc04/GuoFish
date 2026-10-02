"""Upload the files of a sha256 list to the object store, resumably, then publish the list.

    python data/multiPV/r2_push.py --root data --list L --prefix data/ \
        [--workers 4] [--part-concurrency 8] [--no-publish] [--marker upload_ok.json]

L holds `sha256  size  relative_path` lines (tools/make_sha256_list.py), paths relative to
--root. Each file goes to <prefix><relative_path>; one already in the store at the same size
is skipped (sync semantics: an interrupted push resumes; the pull verifies sha256). Then the
store is listed and every size compared with L; any mismatch fails before anything is
published. Then, unless --no-publish, L is uploaded as <prefix>sha256.txt, LAST, so a reader
never sees a list naming missing files. --marker NAME then writes <prefix>NAME (JSON stats).
Credentials: training/v6/r2.py (environment or .env; PROD_STORE=file://... for tests).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from training.v6.r2 import open_store, parse_sha_list, put_json  # noqa: E402


def push(store, root: Path, list_text: str, prefix: str, workers: int, publish: bool,
         marker: str | None, log=print) -> dict:
    entries = parse_sha_list(list_text)
    for _sha, size, rel in entries:
        p = root / rel
        if not p.is_file() or p.stat().st_size != size:
            raise SystemExit(f"{p}: missing or not {size:,} bytes; the list is stale")
    total = sum(e[1] for e in entries)
    have = store.list(prefix)
    todo = [e for e in entries if have.get(prefix + e[2]) != e[1]]
    todo_b = sum(e[1] for e in todo)
    log(f"push {len(entries)} files ({total / 1e9:.2f} GB) to {store.where}/{prefix}: "
        f"{len(entries) - len(todo)} already there, {len(todo)} to send ({todo_b / 1e9:.2f} GB)")
    done = [0]
    lock = threading.Lock()
    t0 = time.monotonic()

    def one(e):
        _sha, size, rel = e
        store.put_file(root / rel, prefix + rel)
        with lock:
            done[0] += size
            dt = time.monotonic() - t0
            log(f"  {rel}  {size / 1e9:.2f} GB | {done[0] / 1e9:.2f}/{todo_b / 1e9:.2f} GB "
                f"at {done[0] / 1e6 / max(dt, 1e-9):,.1f} MB/s")

    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(one, todo))
    dt = time.monotonic() - t0
    remote = store.list(prefix)
    bad = [f"{rel}: remote {remote.get(prefix + rel)} != {size}" for _s, size, rel in entries
           if remote.get(prefix + rel) != size]
    if bad:
        raise SystemExit(f"{len(bad)} remote size mismatch(es):\n  " + "\n  ".join(bad[:50]))
    stats = {"files": len(entries), "bytes": total, "sent_files": len(todo), "sent_bytes": todo_b,
             "seconds": round(dt, 1), "mb_per_s": round(todo_b / 1e6 / max(dt, 1e-9), 2),
             "remote_sizes_verified": len(entries)}
    if publish:
        store.put_bytes(list_text.encode(), prefix + "sha256.txt")
        if store.get_bytes(prefix + "sha256.txt") != list_text.encode():
            raise SystemExit(f"{prefix}sha256.txt did not read back identical")
        stats["published"] = prefix + "sha256.txt"
    if marker:
        put_json(store, prefix + marker, {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                          **stats})
    log(json.dumps(stats))
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--workers", type=int, default=4, help="files in flight")
    ap.add_argument("--part-concurrency", type=int, default=8, help="multipart parts in flight per file")
    ap.add_argument("--no-publish", action="store_true", help="upload the files, not the list")
    ap.add_argument("--marker", default=None, help="write <prefix><marker> after the list")
    a = ap.parse_args(argv)
    if not a.prefix.endswith("/"):
        raise SystemExit("--prefix must end with '/'")
    store = open_store()
    if hasattr(store, "tx"):                         # R2Store: boto3 multipart concurrency
        from boto3.s3.transfer import TransferConfig
        store.tx = TransferConfig(multipart_chunksize=64 << 20, max_concurrency=a.part_concurrency)
    push(store, a.root, a.list.read_text(), a.prefix, a.workers, not a.no_publish, a.marker,
         log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
