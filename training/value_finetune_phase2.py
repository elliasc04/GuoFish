"""Value-head fine-tune Phase 2 — standalone script.

Equivalent to value_finetune_phase2.ipynb but designed for long unattended runs:
  - plain tqdm (no widget/kernel dependency)
  - stdout tee'd to a log file so progress survives a terminal crash
  - ckpt_log written to JSON after every checkpoint, so we never lose progression
  - matplotlib output saved to PNG instead of shown
  - Ctrl-C triggers a final emergency checkpoint before exit
  - guarded under `if __name__ == "__main__"` so Windows DataLoader workers spawn cleanly

Run from the repo root:
    python training/value_finetune_phase2.py
"""

from __future__ import annotations

import copy
import json
import math
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths and hyperparameters (mirror the notebook)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train import ChessTransformer  # noqa: E402

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "value_finetune_phase1_0.076.pt"
DATA_PATH       = PROJECT_ROOT / "data" / "processed" / "stockfish_eval_dataset_full.pt"
MODELS_DIR      = PROJECT_ROOT / "models"
CKPT_DIR        = MODELS_DIR / "guofish3"
LOGS_DIR        = PROJECT_ROOT / "training" / "logs"
MODEL_TAG       = "guofish3_25.6M"

VAL_FRACTION  = 0.05
BATCH_SIZE    = 640
ACCUM_STEPS   = 3
EPOCHS        = 3
VALUE_LR      = 1e-4
BACKBONE_LR   = 1e-5
WEIGHT_DECAY  = 1e-4
WARMUP_STEPS  = 200
GRAD_CLIP     = 1.0

USE_KL_REG = True
KL_WEIGHT  = 1.0

CHECKPOINT_INTERVAL = 4_000_000
DRIFT_BATCH_SIZE    = 5000

SEED               = 42
DATALOADER_WORKERS = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Logging: tee stdout/stderr to a file so we can review after the fact
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
    log_path = LOGS_DIR / f"phase2_{run_id}.log"
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    return log_path


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_policy_kl(m, ref, tokens_t, batch_size=BATCH_SIZE):
    m.eval()
    kl_sum = 0.0
    total = 0
    for i in range(0, tokens_t.size(0), batch_size):
        x = tokens_t[i:i + batch_size]
        with autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            cur_logits, _ = m(x)
            ref_logits, _ = ref(x)
        cur_log_probs = F.log_softmax(cur_logits.float(), dim=-1)
        ref_probs     = F.softmax(ref_logits.float(),     dim=-1)
        kl_sum += F.kl_div(cur_log_probs, ref_probs, reduction="sum").item()
        total  += x.size(0)
    return kl_sum / total


@torch.no_grad()
def full_diagnostic(m, ref, val_tok, val_val, drift_tok, label=""):
    m.eval()
    preds_buf, targets_buf = [], []
    for i in range(0, val_tok.size(0), BATCH_SIZE):
        x = val_tok[i:i + BATCH_SIZE].to(DEVICE, dtype=torch.long, non_blocking=True)
        y = val_val[i:i + BATCH_SIZE]
        with autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            _, pred = m(x)
        preds_buf.append(pred.float().cpu())
        targets_buf.append(y)

    preds   = torch.cat(preds_buf)
    targets = torch.cat(targets_buf)

    mse       = F.mse_loss(preds, targets).item()
    pearson_r = torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()
    pred_mean = preds.mean().item()
    pred_std  = preds.std().item()
    kl_val    = measure_policy_kl(m, ref, drift_tok)

    header = f"  {label}  " if label else ""
    print("=" * 60)
    print(header)
    print(f"  MSE:              {mse:.4f}   (target: 0.04-0.05)")
    print(f"  Pearson r:        {pearson_r:.4f}   (target: 0.90-0.93)")
    print(f"  Prediction mean:  {pred_mean:+.4f}")
    print(f"  Prediction std:   {pred_std:.4f}   (target: 0.47-0.49)")
    print(f"  Policy KL:        {kl_val:.6e}   (should stay near 0)")
    print("=" * 60)

    m.train()
    return dict(mse=mse, pearson_r=pearson_r,
                pred_mean=pred_mean, pred_std=pred_std, policy_kl=kl_val)


# ---------------------------------------------------------------------------
# Checkpoint serialization
# ---------------------------------------------------------------------------
def build_config(n_train: int, n_val: int) -> dict:
    return {
        "VAL_FRACTION":        VAL_FRACTION,
        "BATCH_SIZE":          BATCH_SIZE,
        "ACCUM_STEPS":         ACCUM_STEPS,
        "EPOCHS":              EPOCHS,
        "VALUE_LR":            VALUE_LR,
        "BACKBONE_LR":         BACKBONE_LR,
        "WEIGHT_DECAY":        WEIGHT_DECAY,
        "USE_KL_REG":          USE_KL_REG,
        "KL_WEIGHT":           KL_WEIGHT,
        "WARMUP_STEPS":        WARMUP_STEPS,
        "CHECKPOINT_INTERVAL": CHECKPOINT_INTERVAL,
        "SEED":                SEED,
        "n_train":             n_train,
        "n_val":               n_val,
    }


def save_emergency_checkpoint(model, optimizer, scheduler, positions_processed,
                              epoch, step, n_train, n_val, reason):
    path = CKPT_DIR / f"{MODEL_TAG}_emergency_step{step}.pt"
    torch.save({
        "reason":               reason,
        "positions_processed":  positions_processed,
        "epoch":                epoch,
        "step":                 step,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "phase1_checkpoint":    str(CHECKPOINT_PATH),
        "config":               build_config(n_train, n_val),
    }, path)
    print(f"\n[emergency] checkpoint saved -> {path.name} ({reason})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log_path = setup_logging()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Phase 2 standalone run | log: {log_path}")
    print(f"Checkpoint dir: {CKPT_DIR}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Load Phase 1 checkpoint ---
    print(f"\nLoading Phase 1 checkpoint: {CHECKPOINT_PATH}")
    assert CHECKPOINT_PATH.exists(), f"checkpoint not found: {CHECKPOINT_PATH}"
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    print(f"Phase 1 final_value_loss: {ckpt.get('final_value_loss', float('nan')):.4f}")
    print(f"Phase 1 final_kl:         {ckpt.get('final_kl', float('nan')):.6e}")

    model = ChessTransformer()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)

    assert model.embedding.num_embeddings == 43
    assert model.embedding.embedding_dim == 512
    assert len(model.transformer.layers) == 8
    assert model.head_dim == 64

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")

    # --- Frozen reference for KL ---
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model = ref_model.to(DEVICE)
    if DEVICE.type == "cuda":
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:,.0f} MB")

    # --- Dataset ---
    assert DATA_PATH.exists(), (
        f"dataset not found: {DATA_PATH}\n"
        f"Build it with:  python data/eval_data_processing.py --full"
    )
    print(f"\nLoading full dataset: {DATA_PATH}")
    print(f"  size on disk: {DATA_PATH.stat().st_size / 1024**3:.2f} GB")
    data = torch.load(DATA_PATH, weights_only=True)
    tokens, values = data["tokens"], data["values"]
    n_total = tokens.size(0)
    print(f"Dataset: {n_total:,} positions")
    assert values.min() >= -1.0 and values.max() <= 1.0

    v_np = values.numpy()
    p05, p50, p95 = np.percentile(v_np, [5, 50, 95])
    print(f"Value distribution: mean={v_np.mean():+.4f} std={v_np.std():.4f} "
          f"p05/p50/p95={p05:+.4f}/{p50:+.4f}/{p95:+.4f}")

    # --- Train/val split ---
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n_total, generator=g)
    n_val   = int(n_total * VAL_FRACTION)
    n_train = n_total - n_val

    train_tokens = tokens[perm[:n_train]]
    train_values = values[perm[:n_train]]
    val_tokens   = tokens[perm[n_train:]]
    val_values   = values[perm[n_train:]]

    print(f"Train: {n_train:,} | Val: {n_val:,}")
    total_positions_to_process = n_train * EPOCHS
    print(f"Total positions across {EPOCHS} epochs: {total_positions_to_process / 1e6:.1f}M")

    drift_n      = min(DRIFT_BATCH_SIZE, n_val)
    drift_tokens = val_tokens[:drift_n].to(DEVICE, dtype=torch.long, non_blocking=True)

    # --- Baseline ---
    print("\nMeasuring Phase 2 baseline (= Phase 1 final state)...")
    baseline_metrics = full_diagnostic(
        model, ref_model, val_tokens, val_values, drift_tokens,
        label="BASELINE (Phase 1 final)",
    )

    # --- DataLoaders ---
    train_ds = TensorDataset(train_tokens, train_values)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=DATALOADER_WORKERS,
        persistent_workers=True, prefetch_factor=4, drop_last=False,
    )

    # --- Optimizer / scheduler ---
    value_head_params = list(model.value_head.parameters())
    value_head_ids    = {id(p) for p in value_head_params}
    backbone_params   = [p for p in model.parameters() if id(p) not in value_head_ids]
    print(f"Value head params: {sum(p.numel() for p in value_head_params):,}")
    print(f"Backbone params:   {sum(p.numel() for p in backbone_params):,}")

    optimizer = optim.AdamW(
        [
            {"params": value_head_params, "lr": VALUE_LR},
            {"params": backbone_params,   "lr": BACKBONE_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    num_batches               = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(num_batches / ACCUM_STEPS)
    total_optimizer_steps     = optimizer_steps_per_epoch * EPOCHS

    def warmup_const_then_decay(step):
        # Warmup -> full LR through epochs 1-2 -> cosine decay to 10% across epoch 3.
        if step < WARMUP_STEPS:
            return 0.1 + 0.9 * (step / max(1, WARMUP_STEPS))
        decay_start = 2 * optimizer_steps_per_epoch
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(1, total_optimizer_steps - decay_start)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, warmup_const_then_decay)

    print(f"Train batches: {num_batches:,} | Optimizer steps/epoch: {optimizer_steps_per_epoch:,}")
    print(f"Total optimizer steps: {total_optimizer_steps:,}")
    print(f"LR schedule: warmup {WARMUP_STEPS} -> constant through step "
          f"{2 * optimizer_steps_per_epoch:,} -> cosine decay to 10% by step "
          f"{total_optimizer_steps:,}")

    # --- Training state ---
    log_steps:      list[int]   = []
    log_value_loss: list[float] = []
    log_kl_loss:    list[float] = []
    log_total_loss: list[float] = []
    ckpt_log:       list[dict]  = []

    ckpt_log_path = LOGS_DIR / f"phase2_ckpt_log_{log_path.stem.split('_', 1)[1]}.json"

    positions_processed = 0
    next_checkpoint_at  = CHECKPOINT_INTERVAL
    ckpt_count          = 0

    step = 0
    epoch = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    ref_model.eval()

    # --- Ctrl-C handler: trigger emergency save on next batch boundary ---
    interrupted = {"flag": False}

    def _sigint(signum, frame):
        if interrupted["flag"]:
            print("\n[sigint] second interrupt — exiting immediately.")
            sys.exit(1)
        interrupted["flag"] = True
        print("\n[sigint] interrupt received — will save emergency checkpoint at next step boundary.")

    signal.signal(signal.SIGINT, _sigint)

    training_start  = time.time()
    accum_value_sum = accum_kl_sum = accum_total_sum = 0.0
    accum_count = 0

    try:
        for epoch in range(EPOCHS):
            pbar = tqdm(
                enumerate(train_loader),
                total=num_batches,
                desc=f"Epoch {epoch + 1}/{EPOCHS}",
                mininterval=2.0,
            )
            for batch_idx, (tok, tgt_v) in pbar:
                tok   = tok.to(DEVICE,   dtype=torch.long,    non_blocking=True)
                tgt_v = tgt_v.to(DEVICE, dtype=torch.float32, non_blocking=True)

                with autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                    cur_logits, pred_v = model(tok)
                    if USE_KL_REG:
                        with torch.no_grad():
                            ref_logits, _ = ref_model(tok)

                value_loss = F.mse_loss(pred_v.float(), tgt_v)

                if USE_KL_REG:
                    cur_log_probs = F.log_softmax(cur_logits.float(), dim=-1)
                    ref_probs     = F.softmax(ref_logits.float(),     dim=-1)
                    kl_loss = F.kl_div(cur_log_probs, ref_probs, reduction="batchmean")
                    total_loss = value_loss + KL_WEIGHT * kl_loss
                else:
                    kl_loss    = torch.zeros((), device=DEVICE)
                    total_loss = value_loss

                (total_loss / ACCUM_STEPS).backward()

                positions_processed += tok.size(0)
                accum_value_sum += value_loss.item()
                accum_kl_sum    += kl_loss.item()
                accum_total_sum += total_loss.item()
                accum_count     += 1

                is_accum_boundary = (batch_idx + 1) % ACCUM_STEPS == 0
                is_last_batch     = (batch_idx + 1) == num_batches

                if is_accum_boundary or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1

                    if step % 200 == 0 or step == 1:
                        avg_v = accum_value_sum / accum_count
                        avg_k = accum_kl_sum    / accum_count
                        avg_t = accum_total_sum / accum_count
                        lr_v  = optimizer.param_groups[0]["lr"]
                        lr_b  = optimizer.param_groups[1]["lr"]
                        log_steps.append(step)
                        log_value_loss.append(avg_v)
                        log_kl_loss.append(avg_k)
                        log_total_loss.append(avg_t)
                        pbar.set_postfix(
                            step=step, V=f"{avg_v:.4f}", KL=f"{avg_k:.4f}",
                            lr_v=f"{lr_v:.1e}", lr_b=f"{lr_b:.1e}",
                            pos_M=f"{positions_processed / 1e6:.1f}",
                        )

                    accum_value_sum = accum_kl_sum = accum_total_sum = 0.0
                    accum_count = 0

                    if interrupted["flag"]:
                        save_emergency_checkpoint(
                            model, optimizer, scheduler, positions_processed,
                            epoch + 1, step, n_train, n_val, reason="SIGINT",
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
                        metrics = full_diagnostic(
                            model, ref_model, val_tokens, val_values, drift_tokens,
                            label=f"Checkpoint {ckpt_count:2d} - {pos_M:.1f}M pos processed",
                        )

                        save_name = f"{MODEL_TAG}_ckpt{ckpt_count:02d}_{metrics['mse']:.4f}.pt"
                        save_path = CKPT_DIR / save_name
                        torch.save({
                            "checkpoint_num":       ckpt_count,
                            "positions_processed":  positions_processed,
                            "epoch":                epoch + 1,
                            "step":                 step,
                            "model_state_dict":     model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "metrics":              metrics,
                            "phase1_checkpoint":    str(CHECKPOINT_PATH),
                            "config":               build_config(n_train, n_val),
                        }, save_path)
                        print(f"Saved checkpoint -> {save_path.name}")

                        ckpt_log.append({
                            "ckpt": ckpt_count, "pos_M": pos_M,
                            "epoch": epoch + 1, "step": step, **metrics,
                        })
                        with open(ckpt_log_path, "w", encoding="utf-8") as f:
                            json.dump(ckpt_log, f, indent=2)

                        model.train()
                        next_checkpoint_at += CHECKPOINT_INTERVAL

        train_time = time.time() - training_start
        print(f"\nTraining complete in {train_time / 3600:.2f}h ({step:,} optimizer steps)")

        # --- Final eval + save ---
        print("\nMeasuring final metrics on full val split...")
        final_metrics = full_diagnostic(
            model, ref_model, val_tokens, val_values, drift_tokens,
            label="FINAL - end of Phase 2",
        )

        print("\nPhase 2 summary:")
        print(f"  Baseline MSE:     {baseline_metrics['mse']:.4f}")
        print(f"  Final MSE:        {final_metrics['mse']:.4f}  (target: 0.04-0.05)")
        print(f"  Baseline r:       {baseline_metrics['pearson_r']:.4f}")
        print(f"  Final r:          {final_metrics['pearson_r']:.4f}  (target: 0.90-0.93)")
        print(f"  Final pred std:   {final_metrics['pred_std']:.4f}  (target: 0.47-0.49)")
        print(f"  Final policy KL:  {final_metrics['policy_kl']:.6e}")

        final_save_path = CKPT_DIR / f"{MODEL_TAG}_final_{final_metrics['mse']:.4f}.pt"
        torch.save({
            "epoch":                EPOCHS,
            "positions_processed":  positions_processed,
            "step":                 step,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "baseline_metrics":     baseline_metrics,
            "final_metrics":        final_metrics,
            "phase1_checkpoint":    str(CHECKPOINT_PATH),
            "config":               build_config(n_train, n_val),
        }, final_save_path)
        print(f"\nFinal model saved -> {final_save_path}")

        # --- Plot to PNG (no plt.show) ---
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(13, 4))

            axes[0].plot(log_steps, log_value_loss, label="train value loss (MSE)", linewidth=0.8)
            axes[0].axhline(baseline_metrics["mse"], color="gray", linestyle="--",
                            label=f"baseline ({baseline_metrics['mse']:.3f})")
            axes[0].axhline(final_metrics["mse"], color="green", linestyle=":",
                            label=f"final val ({final_metrics['mse']:.3f})")
            axes[0].axhline(0.045, color="orange", linestyle=":", alpha=0.6, label="target mid (0.045)")
            for i, row in enumerate(ckpt_log):
                axes[0].axvline(row["step"], color="blue", alpha=0.2, linewidth=0.7,
                                label="checkpoint" if i == 0 else None)
            axes[0].set_xlabel("optimizer step"); axes[0].set_ylabel("MSE")
            axes[0].set_title("Value loss vs Stockfish targets")
            axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

            axes[1].plot(log_steps, log_kl_loss, color="tab:orange", linewidth=0.8,
                         label="train KL (per batch)")
            axes[1].axhline(final_metrics["policy_kl"], color="green", linestyle=":",
                            label=f"final val KL ({final_metrics['policy_kl']:.4f})")
            for i, row in enumerate(ckpt_log):
                axes[1].axvline(row["step"], color="blue", alpha=0.2, linewidth=0.7,
                                label="checkpoint" if i == 0 else None)
            axes[1].set_xlabel("optimizer step"); axes[1].set_ylabel("KL divergence")
            axes[1].set_title("Policy drift (KL vs frozen Phase 1 reference)")
            axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = LOGS_DIR / f"phase2_trajectory_{log_path.stem.split('_', 1)[1]}.png"
            plt.savefig(plot_path, dpi=120)
            print(f"Trajectory plot saved -> {plot_path}")
        except Exception as e:
            print(f"[warn] plot generation failed: {e}")

        # Final ckpt_log dump
        ckpt_log.append({
            "ckpt": "final", "pos_M": positions_processed / 1e6,
            "epoch": EPOCHS, "step": step, **final_metrics,
        })
        with open(ckpt_log_path, "w", encoding="utf-8") as f:
            json.dump(ckpt_log, f, indent=2)
        print(f"Checkpoint log saved -> {ckpt_log_path}")

    except KeyboardInterrupt:
        save_emergency_checkpoint(
            model, optimizer, scheduler, positions_processed,
            epoch + 1, step, n_train, n_val, reason="KeyboardInterrupt",
        )
        raise
    except Exception:
        print("\n[error] unhandled exception — saving emergency checkpoint before re-raising:")
        traceback.print_exc()
        save_emergency_checkpoint(
            model, optimizer, scheduler, positions_processed,
            epoch + 1, step, n_train, n_val, reason="unhandled exception",
        )
        raise


if __name__ == "__main__":
    main()
