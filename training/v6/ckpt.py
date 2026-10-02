"""Run directory, provenance, checkpoints and export (§10).

    models/v6/<run>/  config.resolved.yaml  provenance.json  code.patch
                      ckpt/s<samples>.pt (last N)  stable/s<samples>.pt  best.pt
                      branches/s<from>_d<decay>/...  logs/train.jsonl  logs/console.log
                      export/<run>_<shape>_<cfghash8>_s<samples>_<weights>.pt
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from core.guofish_net.model import ARCH_VERSION
from training.v6.data.formats import REPO


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args) -> str:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def git_state() -> dict:
    """SHA, dirty files (tracked changes and untracked, ignored excluded) and the
    working-tree diff against HEAD. Untracked files are listed, not diffed."""
    sha = _git("rev-parse", "HEAD").strip()
    porcelain = [ln for ln in _git("status", "--porcelain", "--untracked-files=all").splitlines() if ln]
    patch = _git("diff", "HEAD", "--binary")
    return {"git_sha": sha, "dirty_files": porcelain,
            "diff_sha256": hashlib.sha256(patch.encode()).hexdigest(), "_patch": patch}


def versions(device: str) -> dict:
    import chess
    # str(): torch.__version__ is a TorchVersion, which weights_only loading refuses
    out = {"python": sys.version.split()[0], "torch": str(torch.__version__),
           "numpy": str(np.__version__), "python_chess": str(chess.__version__),
           "hostname": socket.gethostname(),
           "platform": platform.platform()}
    try:
        import triton
        out["triton"] = str(triton.__version__)
    except ImportError:
        out["triton"] = "not installed"
    if device == "cuda":
        out["gpu"] = torch.cuda.get_device_name(0)
        out["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
    else:
        out["gpu"] = "none (cpu run)"
    return out


def provenance(cfg, cfg_hash: str, hashes: dict) -> tuple[dict, str]:
    g = git_state()
    patch = g.pop("_patch")
    p = {"created_utc": utc(), **g, "config_hash": cfg_hash, **hashes,
         "versions": versions(cfg.system.device), "argv": sys.argv}
    return p, patch


def check_dirty(prov: dict, allow_dirty: bool) -> None:
    if prov["dirty_files"] and not allow_dirty:
        raise SystemExit("working tree is dirty (system.allow_dirty=false):\n  "
                         + "\n  ".join(prov["dirty_files"][:20]))


def prepare_run_dir(run_dir: Path, resume: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise SystemExit(f"{run_dir} is not empty; pass --resume to continue it")
    for sub in ("ckpt", "stable", "logs", "export"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def checkpoint_blob(*, model, ema, optimizer, samples, step, cfg_plain, cfg_hash, model_cfg,
                    prov, metrics, reason, best, device: str) -> dict:
    rng = {"torch": torch.get_rng_state()}
    if device == "cuda":
        # only on a CUDA run: get_rng_state_all() initialises CUDA, i.e. creates
        # a context on every visible GPU, which a CPU run must never do
        rng["cuda"] = torch.cuda.get_rng_state_all()
    return {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "ema": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(), "samples": int(samples), "step": int(step),
            "config": cfg_plain, "config_hash": cfg_hash, "model_config": model_cfg.to_dict(),
            "arch_version": ARCH_VERSION, "token_scheme": model_cfg.token_scheme,
            "value_repr": model_cfg.value_repr.kind, "provenance": prov, "metrics": metrics,
            "reason": reason, "best": best, "rng": rng, "created_utc": utc()}


def rotate(ckpt_dir: Path, keep: int) -> None:
    files = sorted(ckpt_dir.glob("s*.pt"), key=lambda p: int(p.stem[1:]))
    for p in files[:-keep]:
        p.unlink()


def latest_checkpoint(run_dir: Path) -> Path:
    files = sorted((run_dir / "ckpt").glob("s*.pt"), key=lambda p: int(p.stem[1:]))
    if not files:
        raise SystemExit(f"--resume latest: no checkpoint in {run_dir / 'ckpt'}")
    return files[-1]


def weights_sha256(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def export_blob(state_dict, model_cfg, *, weights: str, source_run: str, source_ckpt: str,
                cfg_hash: str, samples: int, value_scale: float) -> dict:
    return {"arch_version": ARCH_VERSION, "model_config": model_cfg.to_dict(), "value_scale": value_scale,
            "token_scheme": model_cfg.token_scheme, "value_repr": model_cfg.value_repr.kind,
            "contract": model_cfg.contract, "state_dict": state_dict,
            "weights_sha256": weights_sha256(state_dict), "weights": weights,
            "source_run": source_run, "source_ckpt": source_ckpt, "config_hash": cfg_hash,
            "samples": int(samples), "created_utc": utc()}


def export_name(run: str, model_cfg, cfg_hash: str, samples: int, weights: str) -> str:
    return (f"{run}_d{model_cfg.d_model}x{model_cfg.n_layers}_{cfg_hash[:8]}"
            f"_s{samples}_{weights}.pt")


def json_safe(x):
    """Non-finite floats -> None, recursively (a NaN loss must still be loggable)."""
    if isinstance(x, float):
        return x if x == x and abs(x) != float("inf") else None
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


class JsonlLog:
    def __init__(self, path: Path, console: Path):
        self.f = open(path, "a", encoding="utf-8")
        self.c = open(console, "a", encoding="utf-8")

    def event(self, kind: str, **fields) -> None:
        rec = json_safe({"event": kind, "utc": utc(), **fields})
        self.f.write(json.dumps(rec, allow_nan=False) + "\n")
        self.f.flush()

    def say(self, msg: str) -> None:
        print(msg, flush=True)
        self.c.write(msg + "\n")
        self.c.flush()

    def close(self) -> None:
        self.f.close()
        self.c.close()
