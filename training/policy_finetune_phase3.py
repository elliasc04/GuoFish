"""Policy distillation Phase 3 — standalone script.

Warm-starts from the Phase 2 final checkpoint (value head already tuned against
Stockfish targets) and fine-tunes the policy head on Stockfish best-move
labels, while freezing the value head and anchoring it via MSE against a
frozen reference model. Mirrors value_finetune_phase2.py's structure.

Run from the repo root:
    python training/policy_finetune_phase3.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset


def fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Paths and hyperparameters
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train import ChessTransformer  # noqa: E402

PHASE2_CKPT_GLOB = "guofish3_25.6M_final_*.pt"
PHASE2_CKPT_DIR  = PROJECT_ROOT / "models" / "guofish3"
DATA_PATH        = PROJECT_ROOT / "data" / "processed" / "policy_singlepv_3M.pt"
MODELS_DIR       = PROJECT_ROOT / "models"
CKPT_DIR         = MODELS_DIR / "guofish4"
LOGS_DIR         = PROJECT_ROOT / "training" / "logs"
MODEL_TAG        = "guofish4_25.6M"

VAL_FRACTION         = 0.05
BATCH_SIZE           = 640
ACCUM_STEPS          = 3
EPOCHS               = 4
POLICY_HEAD_LR       = 1e-4
BACKBONE_LR          = 1e-5
WEIGHT_DECAY         = 1e-4
WARMUP_STEPS         = 200
GRAD_CLIP            = 1.0
LABEL_SMOOTHING      = 0.1
VALUE_ANCHOR_WEIGHT  = 1.0

CHECKPOINTS_PER_EPOCH = 3
LOG_INTERVAL          = 200

SEED               = 42
DATALOADER_WORKERS = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

POLICY_HEAD_KEYS = ("from_proj", "to_proj")


# ---------------------------------------------------------------------------
# Logging tee
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"phase3_{run_id}.log"
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    return log_path


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def find_latest_phase2_checkpoint() -> Path:
    matches = sorted(PHASE2_CKPT_DIR.glob(PHASE2_CKPT_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No Phase 2 final checkpoint matching {PHASE2_CKPT_GLOB} in {PHASE2_CKPT_DIR}"
        )
    return matches[-1]


# ---------------------------------------------------------------------------
# Dataset format handling
#
# Two possible layouts for the policy targets:
#   (a) Class indices: shape [N], int dtype, values in [0, 4095].
#       Cross entropy takes the indices directly.
#   (b) Distributions: shape [N, 4096], float32. Cross entropy accepts
#       float targets natively (PyTorch >= 1.10).
# ---------------------------------------------------------------------------
def detect_target_format(targets: torch.Tensor) -> str:
    if targets.ndim == 1:
        return "indices"
    if targets.ndim == 2 and targets.size(1) == 4096:
        return "distribution"
    raise ValueError(f"Unrecognized target shape {tuple(targets.shape)}; expected [N] or [N, 4096]")


def policy_top1_correct(logits: torch.Tensor, targets: torch.Tensor, fmt: str) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    if fmt == "indices":
        return preds == targets
    return preds == targets.argmax(dim=-1)


def policy_topk_correct(logits: torch.Tensor, targets: torch.Tensor, fmt: str, k: int) -> torch.Tensor:
    topk = logits.topk(k, dim=-1).indices
    if fmt == "indices":
        return (topk == targets.unsqueeze(-1)).any(dim=-1)
    return (topk == targets.argmax(dim=-1, keepdim=True)).any(dim=-1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(
    model: nn.Module,
    ref_model: nn.Module,
    val_tokens: torch.Tensor,
    val_targets: torch.Tensor,
    target_fmt: str,
    policy_criterion: nn.Module,
    batch_size: int = BATCH_SIZE,
    label: str = "",
):
    model.eval()
    total_loss = 0.0
    total_n = 0
    top1 = 0
    top5 = 0
    cur_v_chunks: list[torch.Tensor] = []
    ref_v_chunks: list[torch.Tensor] = []

    for i in range(0, val_tokens.size(0), batch_size):
        x = val_tokens[i:i + batch_size].to(DEVICE, dtype=torch.long, non_blocking=True)
        y = val_targets[i:i + batch_size]
        y_dev = y.to(DEVICE, dtype=torch.long if target_fmt == "indices" else torch.float32,
                     non_blocking=True)
        with autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            logits, cur_v = model(x)
            _, ref_v = ref_model(x)
        logits_f = logits.float()
        loss = policy_criterion(logits_f, y_dev)
        total_loss += loss.item() * x.size(0)
        top1 += policy_top1_correct(logits_f, y_dev, target_fmt).sum().item()
        top5 += policy_topk_correct(logits_f, y_dev, target_fmt, 5).sum().item()
        total_n += x.size(0)
        cur_v_chunks.append(cur_v.float().cpu())
        ref_v_chunks.append(ref_v.float().cpu())

    cur_v = torch.cat(cur_v_chunks)
    ref_v = torch.cat(ref_v_chunks)
    value_drift = (cur_v - ref_v).abs().mean().item()
    if cur_v.std() > 0 and ref_v.std() > 0:
        value_corr = torch.corrcoef(torch.stack([cur_v, ref_v]))[0, 1].item()
    else:
        value_corr = float("nan")

    metrics = dict(
        val_policy_loss=total_loss / total_n,
        val_top1=top1 / total_n,
        val_top5=top5 / total_n,
        val_value_drift=value_drift,
        val_value_correlation=value_corr,
    )

    header = f"  {label}  " if label else ""
    print("=" * 60)
    print(header)
    print(f"  Val policy loss:        {metrics['val_policy_loss']:.4f}")
    print(f"  Val Top-1 acc:          {metrics['val_top1'] * 100:.2f}%")
    print(f"  Val Top-5 acc:          {metrics['val_top5'] * 100:.2f}%")
    print(f"  Val value drift:        {metrics['val_value_drift']:.4f}   (target ~0)")
    print(f"  Val value correlation:  {metrics['val_value_correlation']:.4f}   (target ~1.0)")
    print("=" * 60)

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Config + checkpoint helpers
# ---------------------------------------------------------------------------
def build_config(n_train: int, n_val: int, target_fmt: str, checkpoint_path: Path) -> dict:
    return {
        "VAL_FRACTION":         VAL_FRACTION,
        "BATCH_SIZE":           BATCH_SIZE,
        "ACCUM_STEPS":          ACCUM_STEPS,
        "EPOCHS":                EPOCHS,
        "POLICY_HEAD_LR":       POLICY_HEAD_LR,
        "BACKBONE_LR":          BACKBONE_LR,
        "WEIGHT_DECAY":         WEIGHT_DECAY,
        "WARMUP_STEPS":         WARMUP_STEPS,
        "GRAD_CLIP":            GRAD_CLIP,
        "LABEL_SMOOTHING":      LABEL_SMOOTHING,
        "VALUE_ANCHOR_WEIGHT":  VALUE_ANCHOR_WEIGHT,
        "CHECKPOINTS_PER_EPOCH": CHECKPOINTS_PER_EPOCH,
        "LOG_INTERVAL":          LOG_INTERVAL,
        "SEED":                 SEED,
        "target_format":        target_fmt,
        "warm_started_from":    str(checkpoint_path),
        "n_train":              n_train,
        "n_val":                n_val,
    }


def save_emergency_checkpoint(model, optimizer, scheduler, positions_processed,
                              epoch, step, n_train, n_val, target_fmt, checkpoint_path, reason):
    path = CKPT_DIR / f"{MODEL_TAG}_emergency_step{step}.pt"
    torch.save({
        "reason":                reason,
        "positions_processed":   positions_processed,
        "epoch":                 epoch,
        "step":                  step,
        "model_state_dict":      model.state_dict(),
        "optimizer_state_dict":  optimizer.state_dict(),
        "scheduler_state_dict":  scheduler.state_dict(),
        "warm_started_from":     str(checkpoint_path),
        "config":                build_config(n_train, n_val, target_fmt, checkpoint_path),
    }, path)
    print(f"\n[emergency] checkpoint saved -> {path.name} ({reason})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 policy distillation")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help=f"Phase 2 checkpoint (default: latest {PHASE2_CKPT_GLOB})")
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--accum-steps", type=int, default=ACCUM_STEPS)
    p.add_argument("--policy-lr", type=float, default=POLICY_HEAD_LR)
    p.add_argument("--backbone-lr", type=float, default=BACKBONE_LR)
    p.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    p.add_argument("--value-anchor-weight", type=float, default=VALUE_ANCHOR_WEIGHT)
    p.add_argument("--output-dir", type=Path, default=CKPT_DIR)
    p.add_argument("--resume", type=Path, default=None,
                   help="Phase 3 checkpoint to resume from")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    global BATCH_SIZE, ACCUM_STEPS, EPOCHS, POLICY_HEAD_LR, BACKBONE_LR
    global LABEL_SMOOTHING, VALUE_ANCHOR_WEIGHT, CKPT_DIR
    BATCH_SIZE          = args.batch_size
    ACCUM_STEPS         = args.accum_steps
    EPOCHS              = args.epochs
    POLICY_HEAD_LR      = args.policy_lr
    BACKBONE_LR         = args.backbone_lr
    LABEL_SMOOTHING     = args.label_smoothing
    VALUE_ANCHOR_WEIGHT = args.value_anchor_weight
    CKPT_DIR            = args.output_dir

    log_path = setup_logging()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Phase 3 (policy distillation) | log: {log_path}")
    print(f"Checkpoint dir: {CKPT_DIR}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Resolve Phase 2 checkpoint ---
    checkpoint_path = args.checkpoint or find_latest_phase2_checkpoint()
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f"Phase 2 checkpoint not found: {checkpoint_path}"
    print(f"\nLoading Phase 2 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "final_metrics" in ckpt:
        fm = ckpt["final_metrics"]
        print(f"Phase 2 final MSE: {fm.get('mse', float('nan')):.4f}  "
              f"r: {fm.get('pearson_r', float('nan')):.4f}  "
              f"pred_std: {fm.get('pred_std', float('nan')):.4f}")

    # --- Build model + load weights ---
    model = ChessTransformer()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)

    assert model.embedding.num_embeddings == 43
    assert model.embedding.embedding_dim == 512
    assert len(model.transformer.layers) == 8
    assert model.head_dim == 64

    # --- Freeze value head ---
    for p in model.value_head.parameters():
        p.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {frozen:,}")
    assert frozen > 250_000, "Value head freeze didn't take"

    # --- Frozen reference model (for value anchoring) ---
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model = ref_model.to(DEVICE)
    if DEVICE.type == "cuda":
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:,.0f} MB")

    # --- Dataset: inspect + load ---
    data_path = Path(args.data)
    assert data_path.exists(), f"dataset not found: {data_path}"
    print(f"\nLoading dataset: {data_path}")
    print(f"  size on disk: {data_path.stat().st_size / 1024**3:.2f} GB")
    data = torch.load(data_path, weights_only=True)
    print(f"  keys: {list(data.keys())}")

    tokens = data["tokens"]
    if "moves" in data:
        targets = data["moves"]
        target_key = "moves"
    elif "targets" in data:
        targets = data["targets"]
        target_key = "targets"
    else:
        raise KeyError(f"Expected 'moves' or 'targets' key, got: {list(data.keys())}")

    print(f"  tokens : shape={tuple(tokens.shape)}  dtype={tokens.dtype}")
    print(f"  {target_key}: shape={tuple(targets.shape)}  dtype={targets.dtype}")
    print(f"  {target_key}[0]: {targets[0]}")

    target_fmt = detect_target_format(targets)
    print(f"  detected target format: {target_fmt}")
    if target_fmt == "indices":
        targets = targets.to(torch.long)
        assert targets.min().item() >= 0 and targets.max().item() < 4096, \
            f"target indices out of range: [{targets.min().item()}, {targets.max().item()}]"
    else:
        targets = targets.to(torch.float32)

    n_total = tokens.size(0)
    print(f"Dataset: {n_total:,} positions")

    # --- Train/val split ---
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n_total, generator=g)
    n_val   = int(n_total * VAL_FRACTION)
    n_train = n_total - n_val

    train_tokens  = tokens[perm[:n_train]]
    train_targets = targets[perm[:n_train]]
    val_tokens    = tokens[perm[n_train:]]
    val_targets   = targets[perm[n_train:]]

    print(f"Train: {n_train:,} | Val: {n_val:,}")
    total_positions_to_process = n_train * EPOCHS
    print(f"Total positions across {EPOCHS} epochs: {total_positions_to_process / 1e6:.1f}M")

    checkpoint_interval = n_train // CHECKPOINTS_PER_EPOCH
    print(f"Checkpoints/epoch: {CHECKPOINTS_PER_EPOCH} (every {checkpoint_interval / 1e6:.2f}M positions)")

    # --- Loss functions ---
    policy_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    value_anchor_criterion = nn.MSELoss()

    # --- Baseline ---
    print("\nMeasuring Phase 3 baseline (loaded Phase 2 checkpoint)...")
    baseline = validate(model, ref_model, val_tokens, val_targets, target_fmt,
                        policy_criterion, label="=== Phase 3 Baseline ===")

    if abs(baseline["val_value_drift"]) > 1e-6 or abs(baseline["val_value_correlation"] - 1.0) > 1e-4:
        print(f"\n[ABORT] reference model misconfigured: "
              f"drift={baseline['val_value_drift']:.6f}, "
              f"corr={baseline['val_value_correlation']:.6f}")
        sys.exit(1)

    # --- DataLoader ---
    train_ds = TensorDataset(train_tokens, train_targets)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=DATALOADER_WORKERS,
        persistent_workers=True, prefetch_factor=4, drop_last=False,
    )

    # --- Parameter groups ---
    def in_policy_head(name: str) -> bool:
        return any(k in name for k in POLICY_HEAD_KEYS)

    def in_value_head(name: str) -> bool:
        return "value_head" in name

    policy_head_params = [p for n, p in model.named_parameters()
                          if in_policy_head(n) and p.requires_grad]
    backbone_params    = [p for n, p in model.named_parameters()
                          if not in_policy_head(n) and not in_value_head(n) and p.requires_grad]
    print(f"Policy-head params: {sum(p.numel() for p in policy_head_params):,}")
    print(f"Backbone params:    {sum(p.numel() for p in backbone_params):,}")

    optimizer = optim.AdamW(
        [
            {"params": policy_head_params, "lr": POLICY_HEAD_LR},
            {"params": backbone_params,    "lr": BACKBONE_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    num_batches = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(num_batches / ACCUM_STEPS)
    total_optimizer_steps     = optimizer_steps_per_epoch * EPOCHS

    def warmup_then_constant(step):
        if step < WARMUP_STEPS:
            return 0.1 + 0.9 * (step / max(1, WARMUP_STEPS))
        return 1.0
    scheduler = LambdaLR(optimizer, warmup_then_constant)

    print(f"Train batches: {num_batches:,} | Optimizer steps/epoch: {optimizer_steps_per_epoch:,}")
    print(f"Total optimizer steps: {total_optimizer_steps:,}")
    print(f"Effective batch: {BATCH_SIZE * ACCUM_STEPS} | Warmup: {WARMUP_STEPS} steps")

    # --- Resume ---
    start_epoch = 0
    positions_processed = 0
    step = 0
    next_checkpoint_at = checkpoint_interval
    ckpt_count = 0
    if args.resume is not None:
        resume_path = Path(args.resume)
        assert resume_path.exists(), f"resume checkpoint not found: {resume_path}"
        print(f"\nResuming from {resume_path}")
        rckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        model.load_state_dict(rckpt["model_state_dict"])
        optimizer.load_state_dict(rckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in rckpt:
            scheduler.load_state_dict(rckpt["scheduler_state_dict"])
        start_epoch = int(rckpt.get("epoch", 1))
        positions_processed = int(rckpt.get("positions_processed", 0))
        step = int(rckpt.get("step", 0))
        next_checkpoint_at = ((positions_processed // checkpoint_interval) + 1) * checkpoint_interval
        match = re.search(r"ep(\d+)_(\d+)M", resume_path.name)
        if match:
            ckpt_count = int(match.group(2))
        print(f"Resumed at epoch {start_epoch}, step {step}, positions {positions_processed:,}")

    # --- Training state ---
    log_steps:      list[int]   = []
    log_policy:     list[float] = []
    log_value:      list[float] = []
    log_total:      list[float] = []
    log_top1:       list[float] = []
    ckpt_log:       list[dict]  = []
    ckpt_log_path = LOGS_DIR / f"phase3_ckpt_log_{log_path.stem.split('_', 1)[1]}.json"

    # --- Early stopping bookkeeping ---
    best_val_top1_overall = baseline["val_top1"]
    best_epoch = 0  # epoch number where best was set (0 = baseline)
    EARLY_STOP_PATIENCE_EPOCHS = 1

    optimizer.zero_grad(set_to_none=True)
    model.train()
    ref_model.eval()

    # --- Ctrl-C handler ---
    interrupted = {"flag": False}

    def _sigint(signum, frame):
        if interrupted["flag"]:
            print("\n[sigint] second interrupt — exiting immediately.")
            sys.exit(1)
        interrupted["flag"] = True
        print("\n[sigint] interrupt received — will save emergency checkpoint at next step boundary.")

    signal.signal(signal.SIGINT, _sigint)

    training_start = time.time()
    accum_policy_sum = accum_value_sum = accum_total_sum = 0.0
    accum_correct = accum_top5 = accum_seen = 0
    accum_count = 0
    epoch = start_epoch

    try:
        for epoch in range(start_epoch, EPOCHS):
            epoch_start = time.time()
            print("=" * 80)
            print(f"Epoch {epoch + 1}/{EPOCHS} starting "
                  f"(batches={num_batches:,}, optimizer steps={optimizer_steps_per_epoch:,})")
            print("=" * 80)
            for batch_idx, (tok, tgt) in enumerate(train_loader):
                tok = tok.to(DEVICE, dtype=torch.long, non_blocking=True)
                if target_fmt == "indices":
                    tgt = tgt.to(DEVICE, dtype=torch.long, non_blocking=True)
                else:
                    tgt = tgt.to(DEVICE, dtype=torch.float32, non_blocking=True)

                with autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                    cur_logits, cur_v = model(tok)
                    with torch.no_grad():
                        _, ref_v = ref_model(tok)

                policy_loss = policy_criterion(cur_logits.float(), tgt)
                value_anchor_loss = value_anchor_criterion(cur_v.float(), ref_v.float())
                total_loss = policy_loss + VALUE_ANCHOR_WEIGHT * value_anchor_loss

                (total_loss / ACCUM_STEPS).backward()

                with torch.no_grad():
                    correct = policy_top1_correct(cur_logits.float(), tgt, target_fmt).sum().item()
                    top5    = policy_topk_correct(cur_logits.float(), tgt, target_fmt, 5).sum().item()
                accum_correct += correct
                accum_top5    += top5
                accum_seen    += tok.size(0)

                positions_processed += tok.size(0)
                accum_policy_sum += policy_loss.item()
                accum_value_sum  += value_anchor_loss.item()
                accum_total_sum  += total_loss.item()
                accum_count      += 1

                is_accum_boundary = (batch_idx + 1) % ACCUM_STEPS == 0
                is_last_batch     = (batch_idx + 1) == num_batches

                if is_accum_boundary or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1

                    if step % LOG_INTERVAL == 0 or step == 1:
                        avg_p = accum_policy_sum / accum_count
                        avg_v = accum_value_sum  / accum_count
                        avg_t = accum_total_sum  / accum_count
                        top1  = accum_correct / max(1, accum_seen)
                        top5r = accum_top5    / max(1, accum_seen)
                        lr_h  = optimizer.param_groups[0]["lr"]
                        lr_b  = optimizer.param_groups[1]["lr"]
                        log_steps.append(step)
                        log_policy.append(avg_p)
                        log_value.append(avg_v)
                        log_total.append(avg_t)
                        log_top1.append(top1)

                        elapsed_total = time.time() - training_start
                        elapsed_epoch = time.time() - epoch_start
                        samples_per_sec = (positions_processed / elapsed_total) if elapsed_total > 0 else 0.0
                        remaining_steps = max(0, total_optimizer_steps - step)
                        eta_sec = (elapsed_total / step) * remaining_steps if step > 0 else 0.0
                        pos_M = positions_processed / 1e6

                        print(
                            f"E{epoch + 1}/{EPOCHS} "
                            f"step {step:>5}/{total_optimizer_steps} "
                            f"({100.0 * step / total_optimizer_steps:5.1f}%) | "
                            f"pos {pos_M:6.2f}M | "
                            f"P {avg_p:.4f}  Vanc {avg_v:.4f}  tot {avg_t:.4f} | "
                            f"top1 {top1 * 100:5.2f}%  top5 {top5r * 100:5.2f}% | "
                            f"lr_h {lr_h:.1e} lr_b {lr_b:.1e} | "
                            f"{samples_per_sec:>5.0f} pos/s | "
                            f"epoch {fmt_hms(elapsed_epoch)}  "
                            f"total {fmt_hms(elapsed_total)}  "
                            f"ETA {fmt_hms(eta_sec)}"
                        )

                    accum_policy_sum = accum_value_sum = accum_total_sum = 0.0
                    accum_correct = accum_top5 = accum_seen = 0
                    accum_count = 0

                    if interrupted["flag"]:
                        save_emergency_checkpoint(
                            model, optimizer, scheduler, positions_processed,
                            epoch + 1, step, n_train, n_val, target_fmt, checkpoint_path,
                            reason="SIGINT",
                        )
                        return

                    while positions_processed >= next_checkpoint_at:
                        ckpt_count += 1
                        pos_M = positions_processed / 1e6
                        elapsed = time.time() - training_start

                        print(f"\n{'=' * 60}")
                        print(f"Checkpoint {ckpt_count:2d}  |  {pos_M:.1f}M positions  "
                              f"|  epoch {epoch + 1}/{EPOCHS}  |  step {step:,}  "
                              f"|  elapsed {elapsed / 3600:.2f}h")
                        metrics = validate(
                            model, ref_model, val_tokens, val_targets, target_fmt,
                            policy_criterion,
                            label=f"Checkpoint {ckpt_count:2d} - {pos_M:.1f}M pos",
                        )

                        save_name = (
                            f"policy_finetune_phase3_ep{epoch + 1}"
                            f"_{int(round(pos_M)):d}M_top1_{metrics['val_top1']:.3f}.pt"
                        )
                        save_path = CKPT_DIR / save_name
                        torch.save({
                            "checkpoint_num":        ckpt_count,
                            "positions_processed":   positions_processed,
                            "epoch":                 epoch + 1,
                            "step":                  step,
                            "model_state_dict":      model.state_dict(),
                            "optimizer_state_dict":  optimizer.state_dict(),
                            "scheduler_state_dict":  scheduler.state_dict(),
                            "metrics":               metrics,
                            "warm_started_from":     str(checkpoint_path),
                            "config":                build_config(n_train, n_val, target_fmt, checkpoint_path),
                        }, save_path)
                        print(f"Saved checkpoint -> {save_path.name}")

                        ckpt_log.append({
                            "ckpt": ckpt_count, "pos_M": pos_M,
                            "epoch": epoch + 1, "step": step, **metrics,
                        })
                        with open(ckpt_log_path, "w", encoding="utf-8") as f:
                            json.dump(ckpt_log, f, indent=2)

                        if metrics["val_top1"] > best_val_top1_overall:
                            best_val_top1_overall = metrics["val_top1"]
                            best_epoch = epoch + 1

                        model.train()
                        next_checkpoint_at += checkpoint_interval

            # --- End-of-epoch validation ---
            print(f"\n--- End of epoch {epoch + 1} ---")
            epoch_metrics = validate(
                model, ref_model, val_tokens, val_targets, target_fmt,
                policy_criterion, label=f"End of epoch {epoch + 1}",
            )
            ckpt_log.append({
                "ckpt": f"epoch{epoch + 1}_end", "pos_M": positions_processed / 1e6,
                "epoch": epoch + 1, "step": step, **epoch_metrics,
            })
            with open(ckpt_log_path, "w", encoding="utf-8") as f:
                json.dump(ckpt_log, f, indent=2)

            if epoch_metrics["val_top1"] > best_val_top1_overall:
                best_val_top1_overall = epoch_metrics["val_top1"]
                best_epoch = epoch + 1

            epochs_without_improvement = (epoch + 1) - best_epoch
            if epochs_without_improvement >= EARLY_STOP_PATIENCE_EPOCHS:
                print(f"\n[early stop] no val Top-1 improvement for "
                      f"{epochs_without_improvement} epoch(s) "
                      f"(best {best_val_top1_overall * 100:.2f}% at epoch {best_epoch}). "
                      f"Stopping.")
                break

        train_time = time.time() - training_start
        print(f"\nTraining complete in {train_time / 3600:.2f}h ({step:,} optimizer steps)")

        # --- Final eval + save ---
        print("\nMeasuring final metrics on full val split...")
        final_metrics = validate(
            model, ref_model, val_tokens, val_targets, target_fmt,
            policy_criterion, label="FINAL - end of Phase 3",
        )

        print("\nPhase 3 summary:")
        print(f"  Baseline policy loss:   {baseline['val_policy_loss']:.4f}")
        print(f"  Final policy loss:      {final_metrics['val_policy_loss']:.4f}")
        print(f"  Baseline Top-1:         {baseline['val_top1'] * 100:.2f}%")
        print(f"  Final Top-1:            {final_metrics['val_top1'] * 100:.2f}%")
        print(f"  Baseline Top-5:         {baseline['val_top5'] * 100:.2f}%")
        print(f"  Final Top-5:            {final_metrics['val_top5'] * 100:.2f}%")
        print(f"  Final value drift:      {final_metrics['val_value_drift']:.4f}")
        print(f"  Final value corr:       {final_metrics['val_value_correlation']:.4f}")

        final_save_path = CKPT_DIR / (
            f"policy_finetune_phase3_final_top1_{final_metrics['val_top1']:.3f}.pt"
        )
        torch.save({
            "epoch":                 epoch + 1,
            "positions_processed":   positions_processed,
            "step":                  step,
            "model_state_dict":      model.state_dict(),
            "optimizer_state_dict":  optimizer.state_dict(),
            "baseline_metrics":      baseline,
            "final_metrics":         final_metrics,
            "warm_started_from":     str(checkpoint_path),
            "config":                build_config(n_train, n_val, target_fmt, checkpoint_path),
        }, final_save_path)
        print(f"\nFinal model saved -> {final_save_path}")

        # --- Plot to PNG ---
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(18, 4))

            axes[0].plot(log_steps, log_policy, label="train policy loss", linewidth=0.8)
            axes[0].axhline(baseline["val_policy_loss"], color="gray", linestyle="--",
                            label=f"baseline ({baseline['val_policy_loss']:.3f})")
            axes[0].axhline(final_metrics["val_policy_loss"], color="green", linestyle=":",
                            label=f"final val ({final_metrics['val_policy_loss']:.3f})")
            for i, row in enumerate(ckpt_log):
                if isinstance(row["ckpt"], int):
                    axes[0].axvline(row["step"], color="blue", alpha=0.2, linewidth=0.7,
                                    label="checkpoint" if i == 0 else None)
            axes[0].set_xlabel("optimizer step"); axes[0].set_ylabel("CE loss")
            axes[0].set_title("Policy loss (Stockfish best-move)")
            axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

            axes[1].plot(log_steps, log_value, color="tab:orange", linewidth=0.8,
                         label="train value anchor MSE")
            axes[1].set_xlabel("optimizer step"); axes[1].set_ylabel("MSE")
            axes[1].set_title("Value anchor drift (current vs frozen Phase 2)")
            axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

            axes[2].plot(log_steps, [t * 100 for t in log_top1], color="tab:green", linewidth=0.8,
                         label="train Top-1")
            axes[2].axhline(baseline["val_top1"] * 100, color="gray", linestyle="--",
                            label=f"baseline ({baseline['val_top1'] * 100:.1f}%)")
            axes[2].axhline(final_metrics["val_top1"] * 100, color="green", linestyle=":",
                            label=f"final val ({final_metrics['val_top1'] * 100:.1f}%)")
            axes[2].set_xlabel("optimizer step"); axes[2].set_ylabel("Top-1 (%)")
            axes[2].set_title("Top-1 accuracy")
            axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = LOGS_DIR / f"phase3_trajectory_{log_path.stem.split('_', 1)[1]}.png"
            plt.savefig(plot_path, dpi=120)
            print(f"Trajectory plot saved -> {plot_path}")
        except Exception as e:
            print(f"[warn] plot generation failed: {e}")

        ckpt_log.append({
            "ckpt": "final", "pos_M": positions_processed / 1e6,
            "epoch": epoch + 1, "step": step, **final_metrics,
        })
        with open(ckpt_log_path, "w", encoding="utf-8") as f:
            json.dump(ckpt_log, f, indent=2)
        print(f"Checkpoint log saved -> {ckpt_log_path}")

    except KeyboardInterrupt:
        save_emergency_checkpoint(
            model, optimizer, scheduler, positions_processed,
            epoch + 1, step, n_train, n_val, target_fmt, checkpoint_path,
            reason="KeyboardInterrupt",
        )
        raise
    except Exception:
        print("\n[error] unhandled exception — saving emergency checkpoint before re-raising:")
        traceback.print_exc()
        save_emergency_checkpoint(
            model, optimizer, scheduler, positions_processed,
            epoch + 1, step, n_train, n_val, target_fmt, checkpoint_path,
            reason="unhandled exception",
        )
        raise


if __name__ == "__main__":
    main()
