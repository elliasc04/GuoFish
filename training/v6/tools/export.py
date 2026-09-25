"""Export engine-ready weights from a checkpoint (§10.5).

    python -m training.v6.tools.export models/v6/<run>/ckpt/s<N>.pt [--weights ema|raw|best]

Writes <run_dir>/export/<run>_d<D>x<L>_<cfghash8>_s<samples>_<weights>.pt with
ModelConfig, arch_version, token_scheme, value_repr, contract, the weights'
sha256 and the source run/config hash, then smoke-tests it: loaded through
load_for_inference, a fixed 64-position frozen-val batch must give outputs
identical to the training-side module holding the same weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from core.guofish_net import build_model, load_for_inference
from training.v6.ckpt import atomic_save, export_blob, export_name
from training.v6.config import build_config
from training.v6.data.batch import BatchBuilder
from training.v6.data.formats import REPO
from training.v6.data.reader import ShardSet


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def smoke_batch(cfg, n: int = 64) -> torch.Tensor:
    ss = ShardSet(resolve(cfg.eval.frozen_dir), "val")
    idx = np.sort(np.random.default_rng(20260924).choice(len(ss), n, replace=False))
    tok = BatchBuilder(cfg.model.token_scheme, 0.0, 0, 0.05, None)(ss.read(idx), idx)["tokens"]
    ss.close()
    return tok


def export(ckpt_path: Path, weights: str | None = None) -> Path:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = build_config(ck["config"])
    weights = weights or cfg.ckpt.export_weights
    run_dir = resolve(cfg.run.out_root) / cfg.run.name
    samples = int(ck["samples"])
    if weights == "raw":
        sd = ck["model"]
    elif weights == "ema":
        if ck["ema"] is None:
            raise SystemExit("checkpoint has no EMA weights (ema.enabled=false)")
        sd = {**ck["model"], **ck["ema"]}
    elif weights == "best":
        best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        if best["config_hash"] != ck["config_hash"]:
            raise SystemExit("best.pt belongs to a different config")
        sd, samples = best["state_dict"], int(best["samples"])
        weights = f"best-{best['weights']}"
    else:
        raise SystemExit(f"--weights {weights!r}; expected ema | raw | best")

    sd = {k: v.float().contiguous() for k, v in sd.items()}
    out = run_dir / "export" / export_name(cfg.run.name, cfg.model, ck["config_hash"], samples, weights)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(export_blob(sd, cfg.model, weights=weights, source_run=cfg.run.name,
                            source_ckpt=str(ckpt_path), cfg_hash=ck["config_hash"],
                            samples=samples), out)

    train_side = build_model(cfg.model).eval()
    train_side.load_state_dict(sd)
    engine_side, contract = load_for_inference(out)
    tok = smoke_batch(cfg)
    with torch.no_grad():
        p0, v0 = train_side(tok)
        p1, v1 = engine_side(tok)
    if contract != cfg.model.contract or not (torch.equal(p0, p1) and torch.equal(v0, v1)):
        out.unlink()
        raise SystemExit("export smoke test failed: load_for_inference output differs")
    print(f"exported {out}\n  contract {contract}, weights {weights}, {samples:,} samples; "
          f"smoke: 64 positions identical (policy and value)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--weights", default=None, choices=["ema", "raw", "best"])
    args = ap.parse_args(argv)
    export(args.ckpt, args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
