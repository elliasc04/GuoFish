"""Measure a legacy (pre-v5) checkpoint's value-head spread on the v5 val split.

Why this exists: v5's value head has a visibly compressed dynamic range, and the
compression RATIO is only meaningful if the numerator and denominator come from
the same records, the same strata definition and the same std estimator. So this
script does not re-implement any of that - it drives the real
`metrics.ValueAccumulator` over the real `MultiPVDataset`, exactly as
train_v5.full_validate does, and only swaps in the older model.

The comparison is legitimate on three counts, all verified rather than assumed:

  tokens      data/multiPV/pass_b_convert.py imports `_board_to_tokens` from
              data/pgn_parallel.py - literally the same function that produced
              guofish3/guofish4's training tokens. Same 68-token scheme, same
              vocab of 43, same CLS at index 67.
  value POV   both are WHITE-relative with no side-to-move flip
              (data/csv_parallel.py's header; labels.py's "no stm conversion").
  value scale both squash with tanh(cp / 290.6806) after clipping cp to +-2000.
              They differ only in the mate pin (v4 +-1.0, v5 +-0.995), which
              cannot touch the middle stratum - mates are a separate stratum by
              construction.

Architecture selection is delegated to playv5.build_model_for_checkpoint, the
same dispatcher the engine uses, so a guofish4 checkpoint lands on the legacy
ChessTransformerV2 (RELU FFN, no final norm) and a v5 one on ChessTransformerV5.
Nothing in core/ is touched.

Usage (from the repo root):
    python benchmarking/engine/eval_value_spread.py
    python benchmarking/engine/eval_value_spread.py --amp off
    python benchmarking/engine/eval_value_spread.py \
        --checkpoint models/guofish5/<v5>.pt --compare-std 0.31 --json out.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_V5_DIR = _PROJECT_ROOT / "training" / "v5_multiPV"
_DATA_MULTIPV = _PROJECT_ROOT / "data" / "multiPV"
# Same three entries train_v5.py adds, in the same order: metrics.py and
# dataset.py import their siblings by bare name.
for _p in (str(_V5_DIR), str(_DATA_MULTIPV), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import MultiPVCollate, MultiPVDataset            # noqa: E402
from metrics import (PolicyAccumulator, ValueAccumulator,     # noqa: E402
                     format_policy_report, format_value_report)
from playing.v5.playv5 import build_model_for_checkpoint      # noqa: E402

DEFAULT_CHECKPOINT = (_PROJECT_ROOT / "models" / "guofish4"
                      / "guofish4_25.6M_policy_final.pt")
DEFAULT_SHARDS = _PROJECT_ROOT / "data" / "processed" / "multipv"


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stratified value metrics for a checkpoint on the v5 val split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    p.add_argument("--split", default="val")
    p.add_argument("--subset", type=int, default=None,
                   help="first N records only; default is the whole split")
    p.add_argument("--batch-size", type=int, default=1024,
                   help="matches train_v5's --val-batch")
    p.add_argument("--workers", type=int, default=4,
                   help="matches train_v5's --val-workers")
    p.add_argument("--mirror-prob", type=float, default=0.0,
                   help="train_v5 validates with the mirror OFF; keep it there "
                        "or the run stops being deterministic")
    p.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16",
                   help="bf16 reproduces the precision v5's own val numbers "
                        "were measured in; off gives the fp32 truth")
    p.add_argument("--device", default=None, help="default: cuda if available")
    p.add_argument("--no-policy", action="store_true",
                   help="skip policy KL/top-k (kept by default as a "
                        "tokenization sanity check)")
    p.add_argument("--compare-std", type=float, default=None,
                   help="another model's middle-stratum pred_std; prints the "
                        "compression ratio against this run")
    p.add_argument("--json", type=Path, default=None,
                   help="also write the full metrics dict here")
    return p.parse_args()


def load_model(path: Path, device: torch.device):
    """Rebuild the architecture from the checkpoint's own metadata, fp32.

    Deliberately not playv5.load_model: that one casts weights to the engine's
    inference dtype and INT8-quantizes on CPU, both of which would perturb the
    very statistic being measured. Precision is controlled by --amp instead.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model, description = build_model_for_checkpoint(ckpt, state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    return model, description, n_params


def amp_context(mode: str, device: torch.device):
    if mode == "off" or device.type != "cuda":
        return contextlib.nullcontext
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    return lambda: autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate(model, loader, device, amp_ctx, with_policy: bool) -> dict:
    """train_v5.full_validate, minus the training-loop plumbing."""
    va = ValueAccumulator()
    pa = PolicyAccumulator(topk=(1, 5)) if with_policy else None
    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        with amp_ctx():
            logits, value = model(tokens)
        # .float() before the accumulator: std of a bf16 tensor is computed in
        # bf16 and would quantize the number we came here for.
        va.add(value.float(), batch["value"], batch["value_cp"])
        if pa is not None:
            pa.add(logits.float().cpu(), batch)
    out = va.result()
    if pa is not None:
        out.update(pa.result())
    return out


def main() -> int:
    args = build_args()
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    model, description, n_params = load_model(args.checkpoint, device)
    print(f"checkpoint  {args.checkpoint}")
    print(f"            {description}")
    print(f"            {n_params:,} parameters")

    ds = MultiPVDataset(args.shards, split=args.split, subset=args.subset)
    # temperature/value_scale left at None: train_v5 only overrides them behind
    # --temperature/--revalue, so the stored targets are what v5 was scored on.
    collate = MultiPVCollate(mirror_prob=args.mirror_prob, temperature=None,
                             value_scale=None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, drop_last=False,
                        collate_fn=collate, pin_memory=(device.type == "cuda"),
                        persistent_workers=bool(args.workers))
    print(f"dataset     {len(ds):,} {args.split} records from "
          f"{len(ds.paths)} shards, mirror {args.mirror_prob:.2f}")
    print(f"precision   {args.amp} on {device}")

    t0 = time.time()
    try:
        metrics = evaluate(model, loader, device,
                           amp_context(args.amp, device), not args.no_policy)
    finally:
        del loader
        ds.close()
    elapsed = time.time() - t0

    print(f"\n{metrics['n']:,} records in {elapsed:.1f}s")
    print(format_value_report(metrics))
    if not args.no_policy:
        print(format_policy_report(metrics))

    middle = metrics.get("value_middle_pred_std", float("nan"))
    print(f"\n=== middle-stratum pred_std: {middle:.4f} ===")
    print(f"    (label_std {metrics['value_middle_label_std']:.4f} over "
          f"{metrics['value_middle_n']:,} records)")
    if args.compare_std is not None:
        print(f"    vs {args.compare_std:.4f}: ratio "
              f"{args.compare_std / middle:.4f} "
              f"({(1 - args.compare_std / middle) * 100:+.1f}% spread)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(args.checkpoint),
            "architecture": description,
            "n_params": n_params,
            "shards": str(args.shards),
            "split": args.split,
            "subset": args.subset,
            "mirror_prob": args.mirror_prob,
            "amp": args.amp,
            "device": str(device),
            "elapsed_sec": elapsed,
            "metrics": metrics,
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
