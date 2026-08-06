"""Generate golden tokenization reference data from FENs in golden/movegen.jsonl.

Outputs:
    golden/tokens.npz containing:
    - fens: 1D array of string FENs
    - tokens: 2D int32 array of shape (N, 68)
"""

import json
import os
import sys
from pathlib import Path
import chess
import numpy as np

# Ensure project root is in sys.path to import core.mctsv4
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.mctsv4 import board_to_tokens


def main():
    jsonl_path = project_root / "golden" / "movegen.jsonl"
    out_path = project_root / "golden" / "tokens.npz"

    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading FENs from {jsonl_path}...")
    fens = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                fens.append(data["fen"])

    print(f"Generating 68-token encodings for {len(fens)} positions...")
    token_list = []
    for fen in fens:
        board = chess.Board(fen)
        # board_to_tokens returns a torch.Tensor, convert to numpy int32
        t = board_to_tokens(board).numpy().astype(np.int32)
        token_list.append(t)

    tokens_arr = np.array(token_list, dtype=np.int32)
    fens_arr = np.array(fens, dtype=object)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, fens=fens_arr, tokens=tokens_arr)
    print(f"Successfully saved {len(fens)} entries to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()