"""Compress a ChessTransformer checkpoint for deployment.

Compression techniques applied:
1. Strip optimizer state and metadata (keep only model weights)
2. Convert to float16 (half precision) - 50% size reduction
3. Optional: int8 dynamic quantization for even smaller size

Usage:
    python compress_model.py models/chess_transformer_25.8M_50.5pct.pt
    python compress_model.py models/chess_transformer_25.8M_50.5pct.pt --quantize
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn


class ChessTransformer(nn.Module):
    """Model definition needed for quantization."""
    def __init__(self, vocab_size=15, d_model=512, nhead=8, num_layers=8, dropout=0.1, head_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.pos_encoder = nn.Parameter(torch.randn(1, 65, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1), nn.Tanh(),
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


def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def compress_to_fp16(state_dict: dict) -> dict:
    """Convert all float32 tensors to float16."""
    compressed = {}
    for key, tensor in state_dict.items():
        if tensor.dtype == torch.float32:
            compressed[key] = tensor.half()
        else:
            compressed[key] = tensor
    return compressed


def main():
    parser = argparse.ArgumentParser(description="Compress a ChessTransformer checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint file")
    parser.add_argument("--quantize", action="store_true", help="Apply int8 dynamic quantization")
    parser.add_argument("--output", type=Path, help="Output path (default: adds _compressed suffix)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: {args.checkpoint} not found")
        return

    original_size = get_file_size_mb(args.checkpoint)
    print(f"Loading {args.checkpoint} ({original_size:.1f} MB)")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # Extract model state dict only
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        val_acc = ckpt.get("val_acc", None)
        print(f"Extracted model_state_dict (val_acc: {val_acc:.1f}%)" if val_acc else "Extracted model_state_dict")
    else:
        state_dict = ckpt
        val_acc = None

    # Count parameters
    num_params = sum(t.numel() for t in state_dict.values())
    print(f"Parameters: {num_params:,}")

    if args.quantize:
        # Dynamic quantization: quantizes Linear layers to int8
        print("\nApplying int8 dynamic quantization...")
        model = ChessTransformer()
        model.load_state_dict(state_dict)
        model.eval()

        # Quantize linear layers (transformer attention, FFN, heads)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = args.checkpoint.with_stem(args.checkpoint.stem + "_quantized")

        # Save quantized model (entire model, not just state_dict)
        torch.save(quantized_model, output_path)

    else:
        # FP16 compression
        print("\nConverting to float16...")
        compressed_state_dict = compress_to_fp16(state_dict)

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = args.checkpoint.with_stem(args.checkpoint.stem + "_fp16")

        # Save compressed checkpoint (state_dict only, no optimizer)
        save_dict = {"model_state_dict": compressed_state_dict}
        if val_acc is not None:
            save_dict["val_acc"] = val_acc
        torch.save(save_dict, output_path)

    compressed_size = get_file_size_mb(output_path)
    reduction = (1 - compressed_size / original_size) * 100

    print(f"\nSaved to {output_path}")
    print(f"Original:   {original_size:.1f} MB")
    print(f"Compressed: {compressed_size:.1f} MB")
    print(f"Reduction:  {reduction:.1f}%")


if __name__ == "__main__":
    main()
