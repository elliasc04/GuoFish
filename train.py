"""Standalone training script for ChessTransformer.

Usage:
    python train.py
    python train.py --epochs 20 --batch-size 1024
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset


# --- Model Definition ---

class ChessTransformer(nn.Module):
    def __init__(self, vocab_size=15, d_model=512, nhead=8, num_layers=8, dropout=0.1, head_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.pos_encoder = nn.Parameter(torch.randn(1, 65, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Tanh()
        )
        self.head_dim = head_dim
        self.from_proj = nn.Linear(d_model, head_dim)
        self.to_proj = nn.Linear(d_model, head_dim)
        self.logit_scale = 1.0 / math.sqrt(head_dim)

    def forward(self, x, legal_move_mask=None):
        x = self.embedding(x) + self.pos_encoder
        x = self.emb_dropout(x)
        x = self.transformer(x)
        pooled_state = x.mean(dim=1)
        value = self.value_head(pooled_state).squeeze(-1)
        x_squares = x[:, :64, :]
        from_feats = self.from_proj(x_squares)
        to_feats = self.to_proj(x_squares)
        policy_logits = torch.bmm(from_feats, to_feats.transpose(1, 2)) * self.logit_scale
        policy_logits = policy_logits.view(x.size(0), 4096)
        if legal_move_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_move_mask, float('-inf'))
        return policy_logits, value


def log(msg: str):
    """Print with flush for real-time output."""
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train ChessTransformer")
    parser.add_argument("--data", type=str, default="data/lichess_processed_dataset.pt",
                        help="Path to processed dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=870, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Max learning rate for OneCycleLR")
    parser.add_argument("--workers", type=int, default=6, help="DataLoader workers")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Enable cuDNN autotuning
    torch.backends.cudnn.benchmark = True

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # --- Load Data ---
    log(f"Loading data from {args.data}...")
    if not os.path.exists(args.data):
        log(f"ERROR: {args.data} not found. Run data_processing.py first.")
        sys.exit(1)

    data = torch.load(args.data, weights_only=True)
    tokens_tensor = data['tokens']
    moves_tensor = data['moves']
    values_tensor = data['values']
    log(f"Loaded {tokens_tensor.size(0):,} positions")
    log(f"Tokens: {tokens_tensor.shape} | Moves: {moves_tensor.shape} | Values: {values_tensor.shape}")

    # --- Create DataLoaders ---
    dataset = TensorDataset(tokens_tensor, moves_tensor, values_tensor)
    val_fraction = 0.1
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    split_generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=split_generator
    )
    log(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.workers, drop_last=False,
        persistent_workers=True, prefetch_factor=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, num_workers=args.workers, drop_last=False,
        persistent_workers=True, prefetch_factor=4
    )

    # --- Initialize Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"\nTraining on: {device}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    model = ChessTransformer().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr / 10, weight_decay=1e-4)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    VALUE_LOSS_WEIGHT = 0.5

    num_batches = len(train_loader)

    start_epoch = 0
    best_val_loss = float('inf')

    # --- Resume from checkpoint ---
    if args.resume:
        if os.path.exists(args.resume):
            log(f"Resuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0)
            best_val_loss = ckpt.get('val_loss', float('inf'))
            log(f"Resumed at epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        else:
            log(f"WARNING: {args.resume} not found, starting fresh")

    # Initialize scheduler AFTER resume so we can set last_epoch correctly
    steps_completed = start_epoch * num_batches
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=num_batches,
        last_epoch=steps_completed - 1 if steps_completed > 0 else -1
    )

    log(f"\nTrain batches: {num_batches:,} | Val batches: {len(val_loader):,} | Epochs: {args.epochs}")
    log(f"Batch size: {args.batch_size} | Max LR: {args.lr}")
    log("=" * 80)

    # --- Training Loop ---
    training_start = time.time()

    try:
        for epoch in range(start_epoch, args.epochs):
            # --- Train ---
            model.train()
            epoch_start = time.time()
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            correct_moves = 0
            total_positions = 0

            for batch_idx, (tokens, target_moves, target_values) in enumerate(train_loader):
                tokens = tokens.to(device, non_blocking=True)
                target_moves = target_moves.to(device, non_blocking=True)
                target_values = target_values.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    policy_logits, predicted_values = model(tokens)
                    policy_loss = policy_criterion(policy_logits, target_moves)
                    value_loss = value_criterion(predicted_values, target_values)
                    batch_loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += batch_loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

                predictions = torch.argmax(policy_logits, dim=1)
                correct_moves += (predictions == target_moves).sum().item()
                total_positions += tokens.size(0)

                if batch_idx % 100 == 0:
                    current_acc = (correct_moves / total_positions) * 100
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - epoch_start
                    samples_per_sec = total_positions / elapsed if elapsed > 0 else 0
                    log(
                        f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{num_batches} | "
                        f"Loss: {batch_loss.item():.4f} (P: {policy_loss.item():.4f}, V: {value_loss.item():.4f}) | "
                        f"Acc: {current_acc:.2f}% | LR: {current_lr:.2e} | "
                        f"Time: {elapsed:.1f}s | {samples_per_sec:.0f} samples/s"
                    )

            train_time = time.time() - epoch_start
            train_acc = (correct_moves / total_positions) * 100
            train_avg_loss = total_loss / num_batches
            train_avg_policy = total_policy_loss / num_batches
            train_avg_value = total_value_loss / num_batches

            # --- Validate ---
            model.eval()
            val_start = time.time()
            val_total_loss = 0.0
            val_total_policy_loss = 0.0
            val_total_value_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for tokens, target_moves, target_values in val_loader:
                    tokens = tokens.to(device, non_blocking=True)
                    target_moves = target_moves.to(device, non_blocking=True)
                    target_values = target_values.to(device, non_blocking=True)

                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        policy_logits, predicted_values = model(tokens)
                        policy_loss = policy_criterion(policy_logits, target_moves)
                        value_loss = value_criterion(predicted_values, target_values)
                        batch_loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

                    val_total_loss += batch_loss.item()
                    val_total_policy_loss += policy_loss.item()
                    val_total_value_loss += value_loss.item()
                    val_correct += (torch.argmax(policy_logits, dim=1) == target_moves).sum().item()
                    val_total += tokens.size(0)

            val_time = time.time() - val_start
            val_acc = (val_correct / val_total) * 100
            val_avg_loss = val_total_loss / len(val_loader)
            val_avg_policy = val_total_policy_loss / len(val_loader)
            val_avg_value = val_total_value_loss / len(val_loader)

            log("=" * 80)
            log(
                f"Epoch {epoch+1}/{args.epochs} Complete | "
                f"Train Loss: {train_avg_loss:.4f} (P: {train_avg_policy:.4f}, V: {train_avg_value:.4f}) Acc: {train_acc:.2f}%"
            )
            log(
                f"Val Loss: {val_avg_loss:.4f} (P: {val_avg_policy:.4f}, V: {val_avg_value:.4f}) Acc: {val_acc:.2f}% | "
                f"Time: {train_time:.1f}s train + {val_time:.1f}s val"
            )

            # --- Save best model ---
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                params_str = f"{num_params // 1000}k" if num_params < 1_000_000 else f"{num_params / 1_000_000:.1f}M"
                checkpoint_path = f"models/guofish_{params_str}_{val_acc:.1f}p.pt"

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_avg_loss,
                    'val_acc': val_acc,
                    'num_params': num_params,
                }, checkpoint_path)
                log(f">>> New best! Saved to {checkpoint_path}")
            else:
                log(f"No improvement (best: {best_val_loss:.4f})")
            log("=" * 80 + "\n")

    except KeyboardInterrupt:
        log("\n\nTraining interrupted by user.")

    # --- Summary ---
    total_time = time.time() - training_start
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    log(f"\nTraining complete! Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    log(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
