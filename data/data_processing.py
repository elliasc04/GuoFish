from pathlib import Path
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

import chess
import chess.pgn

# Make the project root importable so spawned workers can resolve
# data.pgn_parallel on Windows (which uses spawn-style multiprocessing).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data import pgn_parallel  # worker lives in a .py module (spawn semantics on Windows)

PGN_PATH = _PROJECT_ROOT / 'data' / 'processed' / 'lichess_elite_2020-08.pgn'
OUT_DIR  = _PROJECT_ROOT / 'data' / 'processed'

NUM_GAMES = 150000


def iter_pgn_blocks(path, max_games):
    with open(path, encoding="utf-8") as f:
        buffer = []
        count = 0
        saw_moves = False
        for line in f:
            stripped = line.strip()
            if not stripped and saw_moves and buffer:
                buffer.append(line)
                yield "".join(buffer)
                count += 1
                if count >= max_games:
                    return
                buffer = []
                saw_moves = False
                continue
            buffer.append(line)
            if stripped and not stripped.startswith("["):
                saw_moves = True
        if buffer and saw_moves:
            yield "".join(buffer)


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("Splitting PGN into game blocks...")
    game_blocks = list(iter_pgn_blocks(PGN_PATH, NUM_GAMES))
    print(f"Split {len(game_blocks)} games in {time.time()-t0:.1f}s")

    num_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Parsing with {num_workers} worker processes...")
    t0 = time.time()

    tokens_chunks, moves_chunks, values_chunks = [], [], []
    processed = 0

    with Pool(num_workers) as pool:
        for result in pool.imap_unordered(pgn_parallel.parse_game_block, game_blocks, chunksize=32):
            if result is None:
                continue
            t, m, v = result
            tokens_chunks.append(t)
            moves_chunks.append(m)
            values_chunks.append(v)
            processed += 1
            if processed % 1000 == 0:
                print(f"Processed {processed} games")

    tokens_tensor = torch.cat(tokens_chunks, dim=0)
    moves_tensor  = torch.cat(moves_chunks,  dim=0)
    values_tensor = torch.cat(values_chunks, dim=0)
    print(f"Parsed {processed} games in {time.time()-t0:.1f}s")
    print(f"Total positions: {tokens_tensor.size(0)}")

    tokens_tensor = tokens_tensor.to(torch.long)
    moves_tensor  = moves_tensor.to(torch.long)
    values_tensor = values_tensor.to(torch.float32)

    print(f"Tokens shape: {tokens_tensor.shape}")
    print(f"Moves shape:  {moves_tensor.shape}")
    print(f"Values shape: {values_tensor.shape}")

    # Sanity check: verify vocab_size and sequence length match pgn_parallel constants
    print(f"\nSanity check:")
    print(f"  Sequence length: {tokens_tensor.shape[1]} (expected: {pgn_parallel.SEQ_LENGTH})")
    print(f"  Vocab size: {pgn_parallel.VOCAB_SIZE}")
    print(f"  Token range in data: [{tokens_tensor.min().item()}, {tokens_tensor.max().item()}]")

    out_path = _PROJECT_ROOT / 'data' / 'processed' / 'lichess_processed_dataset.pt'
    torch.save({
        'tokens': tokens_tensor,
        'moves':  moves_tensor,
        'values': values_tensor,
    }, out_path)
    print(f"\nSaved tensors to {out_path}")
