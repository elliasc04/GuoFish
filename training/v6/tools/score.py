"""Score one checkpoint's raw weights on frozen90 with the v6 evaluator (gate S2).

    python -m training.v6.tools.score <ckpt> [--v5-crosscheck] [--precision bf16|fp32]

<ckpt> is a v6 training checkpoint (its raw `model` weights) or a train_v5.py
checkpoint, loaded through the M1 converter. Settings are the trainer's
end-of-run full eval: tf32, bf16 autocast, batch 1024, no mirroring.
--v5-crosscheck (v5 checkpoints only) also scores the same file with v5's own
gates.score_baseline on the same records and requires policy KL and value MSE
to agree within --tol relative. Prints one JSON object; exit 1 if the
cross-check fails.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

from core.guofish_net import build_model
from core.guofish_net.v5_compat import load_v5_checkpoint
from training.v6.config import build_config
from training.v6.data.formats import REPO
from training.v6.eval import EvalSet, evaluate

FROZEN = REPO / "data/processed/val_frozen_90m_v1"
FROZEN_STRATA = REPO / "data/processed/strata/val_frozen_90m_v1_val.strata2.npy"
KEYS = ("n", "policy_n", "policy_kl", "value_mse", "policy_top1", "total")


def load(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in ck:
        return load_v5_checkpoint(path), "v5", int(ck.get("step", -1))
    model = build_model(build_config(ck["config"]).model)
    model.load_state_dict(ck["model"])
    return model, "v6", int(ck["step"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--v5-crosscheck", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args(argv)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = args.precision == "bf16"
    amp = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16))
           if args.precision == "bf16" else contextlib.nullcontext)
    model, kind, step = load(args.ckpt)
    if args.v5_crosscheck and kind != "v5":
        raise SystemExit("--v5-crosscheck needs a train_v5.py checkpoint")
    t0 = time.time()
    es = EvalSet("frozen90", FROZEN, "val", FROZEN_STRATA, "v5_68", 1024, 4)
    m = evaluate(model.to(dev), es, dev, amp)
    out = {"checkpoint": str(args.ckpt), "kind": kind, "step": step, "precision": args.precision,
           **{k: m[k] for k in KEYS}, "seconds": round(time.time() - t0, 1)}
    ok = True
    if args.v5_crosscheck:
        for p in (REPO / "training/v5_multiPV", REPO / "data/multiPV"):
            sys.path.insert(0, str(p))
        from dataset import MultiPVCollate, MultiPVDataset  # noqa: E402
        from gates import score_baseline  # noqa: E402
        torch.set_float32_matmul_precision("high" if args.precision == "bf16" else "highest")
        b = score_baseline(args.ckpt, MultiPVDataset(FROZEN, split="val"),
                           MultiPVCollate(mirror_prob=0.0, temperature=None, value_scale=None),
                           dev, amp, batch_size=1024, workers=4, prefetch=2)
        rel = {k: abs(m[k] - b[k]) / abs(b[k]) for k in ("policy_kl", "value_mse")}
        ok = b["n"] == m["n"] and b["n_policy"] == m["policy_n"] and max(rel.values()) <= args.tol
        out["v5_score_baseline"] = {k: b[k] for k in ("n", "n_policy", "policy_kl", "value_mse")}
        out["crosscheck_rel"] = rel
        out["crosscheck_passed"] = ok
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
