"""Score a checkpoint's raw or EMA weights with the v6 evaluator (S2, screening).

    python -m training.v6.tools.score <ckpt> [--weights raw|ema]
        [--sets frozen90 v2val_roots v2val_derived] [--v5-crosscheck] [--precision bf16|fp32]

<ckpt> is a v6 training checkpoint or a train_v5.py checkpoint (raw weights
only, loaded through the M1 converter). Set names are the
data/processed/evalsets/ sidecars (H3); `frozen90` is val_frozen_90m_v2, the
v1 records plus hard_move, every shared field byte-identical (S6). Settings are
the trainer's full eval: tf32, bf16 autocast, batch 1024, no mirroring, the
model's own token scheme. --v5-crosscheck (v5 checkpoints only) also scores the
v1 frozen copy with v5's own gates.score_baseline and requires policy KL and
value MSE to agree with frozen90 within --tol relative.

Prints one JSON object: checkpoint, kind, step, weights, precision, seconds,
`sets` {name: every metric}, and the first set's headline keys at the top
level (what s2.py reads). Exit 1 if the cross-check fails.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

from core.guofish_net import ModelConfig, build_model
from core.guofish_net.v5_compat import load_v5_checkpoint
from training.v6.ckpt import json_safe
from training.v6.data.formats import REPO
from training.v6.eval import evaluate, load_evalset

FROZEN_V1 = REPO / "data/processed/val_frozen_90m_v1"
KEYS = ("n", "policy_n", "policy_kl", "value_mse", "policy_top1", "total")


def load(path: Path, weights: str):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in ck:
        if weights != "raw":
            raise SystemExit("a train_v5.py checkpoint has raw weights only")
        return load_v5_checkpoint(path), "v5", int(ck.get("step", -1))
    # the model section only: older checkpoints' run sections predate data/init seeds
    model = build_model(ModelConfig.from_dict(ck["config"]["model"]))
    sd = ck["model"]
    if weights == "ema":
        if ck["ema"] is None:
            raise SystemExit("checkpoint has no EMA weights (ema.enabled=false)")
        sd = {**sd, **ck["ema"]}
    model.load_state_dict(sd)
    return model, "v6", int(ck["step"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--weights", choices=("raw", "ema"), default="raw")
    ap.add_argument("--sets", nargs="+", default=["frozen90"])
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--v5-crosscheck", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args(argv)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = args.precision == "bf16"
    amp = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16))
           if args.precision == "bf16" else contextlib.nullcontext)
    model, kind, step = load(args.ckpt, args.weights)
    if args.v5_crosscheck and (kind != "v5" or "frozen90" not in args.sets):
        raise SystemExit("--v5-crosscheck needs a train_v5.py checkpoint and the frozen90 set")
    model = model.to(dev)
    t0 = time.time()
    sets = {name: evaluate(model, load_evalset(name, model.cfg.token_scheme, 1024, 4), dev, amp)
            for name in args.sets}
    head = sets[args.sets[0]]
    out = {"checkpoint": str(args.ckpt), "kind": kind, "step": step, "weights": args.weights,
           "precision": args.precision, **{k: head[k] for k in KEYS},
           "seconds": round(time.time() - t0, 1), "sets": sets}
    ok = True
    if args.v5_crosscheck:
        for p in (REPO / "training/v5_multiPV", REPO / "data/multiPV"):
            sys.path.insert(0, str(p))
        from dataset import MultiPVCollate, MultiPVDataset  # noqa: E402
        from gates import score_baseline  # noqa: E402
        torch.set_float32_matmul_precision("high" if args.precision == "bf16" else "highest")
        b = score_baseline(args.ckpt, MultiPVDataset(FROZEN_V1, split="val"),
                           MultiPVCollate(mirror_prob=0.0, temperature=None, value_scale=None),
                           dev, amp, batch_size=1024, workers=4, prefetch=2)
        m = sets["frozen90"]
        rel = {k: abs(m[k] - b[k]) / abs(b[k]) for k in ("policy_kl", "value_mse")}
        ok = b["n"] == m["n"] and b["n_policy"] == m["policy_n"] and max(rel.values()) <= args.tol
        out["v5_score_baseline"] = {k: b[k] for k in ("n", "n_policy", "policy_kl", "value_mse")}
        out["crosscheck_rel"] = rel
        out["crosscheck_passed"] = ok
    print(json.dumps(json_safe(out), indent=1, allow_nan=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
