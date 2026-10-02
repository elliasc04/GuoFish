"""Write `sha256  size  relative_path` for every file under a set of directories.

    python tools/make_sha256_list.py --root data processed/multipv_v3 processed/strata/x.npy \
        [--out list.txt] [--workers 8]

Paths in the list are posix and relative to --root, the convention of training/v6/r2.py's
`pull` (key = <prefix><relative_path>, local file = <dest>/<relative_path>). A path argument
may be a directory (every file under it, recursively) or a single file; each must lie under
--root. Lines are sorted by path. Refuses an empty selection and refuses to overwrite --out.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def files_under(root: Path, paths: list[str]) -> list[Path]:
    out = set()
    for a in paths:
        p = (root / a).resolve()
        if not p.is_relative_to(root):
            raise SystemExit(f"{a} is not under --root {root}")
        if p.is_file():
            out.add(p)
        elif p.is_dir():
            out.update(q for q in p.rglob("*") if q.is_file())
        else:
            raise SystemExit(f"{p} does not exist")
    if not out:
        raise SystemExit("no files selected")
    return sorted(out, key=lambda q: q.relative_to(root).as_posix())


def make_list(root: Path, paths: list[str], workers: int = 8) -> str:
    root = root.resolve()
    files = files_under(root, paths)
    with ThreadPoolExecutor(workers) as ex:          # hashlib releases the GIL
        shas = list(ex.map(sha256_file, files))
    return "".join(f"{s}  {f.stat().st_size}  {f.relative_to(root).as_posix()}\n"
                   for s, f in zip(shas, files))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("paths", nargs="+", help="directories or files, relative to --root")
    ap.add_argument("--out", type=Path, default=None, help="default: stdout")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.out is not None and a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    text = make_list(a.root, a.paths, a.workers)
    if a.out is None:
        sys.stdout.write(text)
    else:
        a.out.write_text(text, newline="\n")
        print(f"{a.out}: {text.count(chr(10))} files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
