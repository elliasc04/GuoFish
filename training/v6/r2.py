"""Object store for the production run and setup.sh (VM harness brief §3, §6).

Credentials come from the environment or an untracked `.env` at the repo root
(KEY=VALUE lines; the environment wins):
    R2_ACCOUNT_ID (or R2_ENDPOINT), R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
PROD_STORE=file:///some/dir swaps R2 for a local directory (tests, offline dry runs).

    python -m training.v6.r2 pull data/ [--workers 16]
    python -m training.v6.r2 cat runs/prod_v6/status.json          # print an object
    python -m training.v6.r2 get runs/prod_v6/export/<file> <dest>  # download one
    python -m training.v6.r2 put runs/prod_v6/control.json <file|->  # upload (- = stdin)
    python -m training.v6.r2 ls runs/prod_v6/evals/

`pull` fetches every file named in <prefix>sha256.txt (lines `sha256  size
relative_path`) from <prefix><relative_path> to data/<relative_path>, and
verifies each file's size and sha256; any mismatch fails. Files already present
are verified, not fetched again.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from training.v6.ckpt import sha256_file
from training.v6.data.formats import REPO


def load_env(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            k, sep, v = ln.partition("=")
            if not sep:
                raise SystemExit(f"{path}: line without '=': {k[:20]!r}")
            os.environ.setdefault(k.strip(), v.strip())


class DirStore:
    """A directory with the object-store interface; keys are relative paths."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.where = f"file://{self.root.as_posix()}"

    def _p(self, key: str) -> Path:
        return self.root / key

    def put_file(self, path: Path, key: str) -> None:
        dst = self._p(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        shutil.copyfile(path, tmp)
        os.replace(tmp, dst)

    def put_bytes(self, data: bytes, key: str) -> None:
        dst = self._p(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dst)

    def get_bytes(self, key: str) -> bytes | None:
        p = self._p(key)
        return p.read_bytes() if p.is_file() else None

    def get_file(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._p(key), path)

    def list(self, prefix: str) -> dict[str, int]:
        base = self._p(prefix) if prefix.endswith("/") else self._p(prefix).parent
        if not base.exists():
            return {}
        out = {}
        for p in base.rglob("*"):
            k = p.relative_to(self.root).as_posix()
            if p.is_file() and k.startswith(prefix) and not k.endswith(".part"):
                out[k] = p.stat().st_size
        return out

    def delete(self, key: str) -> None:
        self._p(key).unlink(missing_ok=True)


class R2Store:
    def __init__(self):
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config
        env = os.environ
        need = ["R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]
        missing = [k for k in need if not env.get(k)]
        if not env.get("R2_ENDPOINT") and not env.get("R2_ACCOUNT_ID"):
            missing.append("R2_ACCOUNT_ID or R2_ENDPOINT")
        if missing:
            raise SystemExit(f"R2 credentials missing from the environment / .env: {', '.join(missing)}")
        endpoint = env.get("R2_ENDPOINT") or f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
        self.bucket = env["R2_BUCKET"]
        self.where = f"r2://{self.bucket}"
        self.s3 = boto3.client("s3", endpoint_url=endpoint, region_name="auto",
                               aws_access_key_id=env["R2_ACCESS_KEY_ID"],
                               aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
                               config=Config(retries={"max_attempts": 10, "mode": "standard"}))
        self.tx = TransferConfig(multipart_chunksize=64 << 20, max_concurrency=8)

    def put_file(self, path: Path, key: str) -> None:
        self.s3.upload_file(str(path), self.bucket, key, Config=self.tx)

    def put_bytes(self, data: bytes, key: str) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get_bytes(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def get_file(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, key, str(path), Config=self.tx)

    def list(self, prefix: str) -> dict[str, int]:
        out = {}
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            out.update({o["Key"]: o["Size"] for o in page.get("Contents", [])})
        return out

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=key)


def open_store():
    load_env()
    url = os.environ.get("PROD_STORE", "")
    if url.startswith("file://"):
        return DirStore(Path(url[len("file://"):]))
    if url:
        raise SystemExit(f"PROD_STORE={url!r}: expected file:///<dir> or unset (R2)")
    return R2Store()


def put_json(store, key: str, obj) -> None:
    store.put_bytes((json.dumps(obj, indent=1, allow_nan=False) + "\n").encode(), key)


def get_json(store, key: str):
    b = store.get_bytes(key)
    return None if b is None else json.loads(b)


def parse_sha_list(text: str) -> list[tuple[str, int, str]]:
    out = []
    for n, ln in enumerate(text.splitlines(), 1):
        if not ln.strip():
            continue
        parts = ln.split(None, 2)
        if len(parts) != 3 or len(parts[0]) != 64 or not parts[1].isdigit():
            raise SystemExit(f"sha256.txt line {n}: expected 'sha256  size  relative_path', got {ln[:80]!r}")
        rel = PurePosixPath(parts[2])
        if "\\" in parts[2] or rel.is_absolute() or ".." in rel.parts:
            raise SystemExit(f"sha256.txt line {n}: {parts[2]!r} is not a plain relative posix path")
        out.append((parts[0], int(parts[1]), str(rel)))
    return out


def pull(store, prefix: str, dest: Path, workers: int) -> None:
    raw = store.get_bytes(prefix + "sha256.txt")
    if raw is None:
        raise SystemExit(f"{store.where}/{prefix}sha256.txt does not exist")
    entries = parse_sha_list(raw.decode())
    total = sum(e[1] for e in entries)
    print(f"pull {store.where}/{prefix} -> {dest}: {len(entries)} files, {total / 1e9:.2f} GB", flush=True)
    t0 = time.monotonic()

    def one(e):
        sha, size, rel = e
        path = dest / rel
        fetched = not (path.is_file() and path.stat().st_size == size)
        if fetched:
            store.get_file(prefix + rel, path)
        if path.stat().st_size != size:
            return f"{rel}: size {path.stat().st_size} != {size}"
        got = sha256_file(path)
        return None if got == sha else f"{rel}: sha256 {got[:12]} != {sha[:12]}"

    with ThreadPoolExecutor(workers) as ex:
        bad = [r for r in ex.map(one, entries) if r]
    dt = time.monotonic() - t0
    if bad:
        raise SystemExit(f"{len(bad)} file(s) failed verification:\n  " + "\n  ".join(bad[:50]))
    (dest / "sha256.txt").write_bytes(raw)
    print(f"verified {len(entries)} files against sha256.txt in {dt:.0f} s "
          f"({total / 1e6 / max(dt, 1e-9):,.0f} MB/s)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    p.add_argument("prefix", help="store prefix holding sha256.txt, e.g. data/")
    p.add_argument("--dest", type=Path, default=REPO / "data")
    p.add_argument("--workers", type=int, default=16)
    sub.add_parser("cat").add_argument("key")
    sub.add_parser("ls").add_argument("prefix")
    p = sub.add_parser("get")
    p.add_argument("key")
    p.add_argument("dest", type=Path)
    p = sub.add_parser("put")
    p.add_argument("key")
    p.add_argument("src", help="a file, or - for stdin")
    args = ap.parse_args(argv)
    store = open_store()
    if args.cmd == "pull":
        if not args.prefix.endswith("/"):
            raise SystemExit("prefix must end with '/'")
        pull(store, args.prefix, args.dest, args.workers)
    elif args.cmd == "cat":
        b = store.get_bytes(args.key)
        if b is None:
            raise SystemExit(f"{store.where}/{args.key} does not exist")
        sys.stdout.write(b.decode())
    elif args.cmd == "ls":
        for k, size in sorted(store.list(args.prefix).items()):
            print(f"{size:>14,}  {k}")
    elif args.cmd == "get":
        store.get_file(args.key, args.dest)
    else:
        data = sys.stdin.buffer.read() if args.src == "-" else Path(args.src).read_bytes()
        if args.key.endswith(".json"):
            json.loads(data)                  # refuse to upload malformed JSON (control.json)
        store.put_bytes(data, args.key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
