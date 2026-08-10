#!/usr/bin/env python
"""C10 Gate 2 — record the model's logits and ATen's answer over them.

WHAT THIS FILE PRODUCES AND WHY IT HAS THREE REFERENCE COLUMNS
==============================================================
For every position in `golden/c10_corpus.json`:

  * `logits`   the model's full 4096-wide bf16 policy row, as raw uint16 bit
               patterns. These are the INPUT to both sides of Gate 2. They are
               produced once, by the reference's own forward pass under the
               reference's autocast dtype, so the gate compares two gathers of
               one set of numbers rather than two forward passes.

  * `priors_cpu_pychess`  ATen's softmax the way the reference computes it for
               an INTERIOR node: `BatchedEvaluator` does a bulk `.cpu()`, then
               `MCTSNode.expand` gathers `policy[legal_indices]` in python-chess's
               generation order, `.float()`s, and softmaxes on the CPU.

  * `priors_gpu_pychess`  the same, on the GPU, which is what the reference does
               for a ROOT: `_expand_root` hands `expand()` a CUDA tensor. These
               two columns are the reference defect C5 found -- one position,
               two answers, max delta 1.9e-9 across 6 of 37 priors -- and having
               both in the file is what lets the test state the size of the
               divergence C10 accepts rather than assert it away.

  * `priors_cpu_libchess`  ATen's CPU softmax over CHESS-LIBRARY's generation
               order, obtained from `guofish_core.generation_order`. Same
               implementation as the first column, different reduction order.

The last column is the one that makes the gate diagnostic instead of merely
pass/fail. `torch.softmax` is not permutation-invariant (scope §2.6: 109 of 200
random permutations reproduce bit-identical priors, max delta 3e-7), so a C++/
ATen difference has two independent causes:

    C++ vs priors_cpu_libchess   -> the softmax IMPLEMENTATION alone
                                    (same order, hand-rolled vs SLEEF/AVX)
    priors_cpu_libchess vs priors_cpu_pychess
                                 -> the PERMUTATION alone (same ATen, two
                                    generation orders)

Only the first is this port's to answer for, and reporting them separately is
what stops a permutation artifact from being read as a porting bug -- or the
reverse. It is also why Gate 2's bound is 1e-6 and not the <= 4 ULP originally
written: permutation alone puts 3e-7 on the table for a correct implementation.

EVERY COLUMN IS PERMUTED INTO CANONICAL ORDER BEFORE IT IS WRITTEN.
The softmax reduces in generation order; only the storage is canonical. That is
the discipline (scope §2.6, verified 300/300 bit-identical) and it is applied
identically to all three columns, so the columns differ only in what they were
computed from -- never in how they were laid out.

Global Rule 2: this reads the Python reference and nothing else. No column here
has ever seen C++ output.

Usage:
    python tools/gen_c10_gate2_golden.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import chess
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402
from core import mctsv4  # noqa: E402
from playing.v5.playv5 import load_model  # noqa: E402

DEFAULT_CORPUS = REPO_ROOT / "golden" / "c10_corpus.json"
DEFAULT_MODEL = REPO_ROOT / "models" / "guofish5_20M" / "v5_10.9M_best.pt"
DEFAULT_OUT = REPO_ROOT / "golden" / "c10_gate2.npz"
DEFAULT_MANIFEST = REPO_ROOT / "golden" / "c10_gate2_manifest.json"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_key(move: chess.Move, board: chess.Board) -> str:
    """The UCI string the port sorts children by.

    python-chess's `Move.uci()` already emits standard UCI for castling (`e1g1`),
    which is the normalisation C1 has to perform on chess-library's
    king-takes-rook encoding. So the sort key is just the string, and canonical
    order is byte-wise lexicographic order over it -- exactly what
    `guofish::canonical_move_key` produces (scope §2.6).
    """
    del board
    return move.uci()


def reference_priors(logits_row: torch.Tensor, order: list[chess.Move],
                     device: torch.device) -> np.ndarray:
    """ATen's priors for `order`, reduced in `order`, returned in `order`.

    Transcribed from `MCTSNode.expand`: gather the legal logits by
    `from_square * 64 + to_square`, `.float()` if not already, softmax over
    dim 0. No policy temperature -- the default is 1.0 and the reference guards
    the divide so that the default path is bit-identical to no divide at all.
    """
    index = torch.tensor([m.from_square * 64 + m.to_square for m in order],
                         dtype=torch.long, device=device)
    legal = logits_row.to(device)[index]
    if legal.dtype != torch.float32:
        legal = legal.float()
    return torch.softmax(legal, dim=0).float().cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for path in (args.out, args.manifest):
        if path.exists() and not args.force:
            print(f"refusing to overwrite {path} without --force", file=sys.stderr)
            return 2

    if not torch.cuda.is_available():
        print("ERROR: Gate 2's golden data records the GPU softmax path, which is "
              "half of the reference defect it exists to measure. Refusing to "
              "generate it on CPU.", file=sys.stderr)
        return 1

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    positions = corpus["positions"]
    device = torch.device("cuda")
    model = load_model(args.model, device)
    mctsv4.require_v5_config(model)

    fens = [p["fen"] for p in positions]
    boards = [chess.Board(f) for f in fens]

    # --- one forward pass per position, batched -------------------------
    logits = np.zeros((len(fens), mctsv4.POLICY_SIZE), dtype=np.uint16)
    values = np.zeros(len(fens), dtype=np.float32)
    for start in range(0, len(fens), args.batch):
        chunk = boards[start:start + args.batch]
        tokens = torch.stack([mctsv4.board_to_tokens(b) for b in chunk]).to(device)
        with torch.no_grad():
            with torch.amp.autocast_mode.autocast(device_type="cuda",
                                                  dtype=mctsv4.AUTOCAST_DTYPE, enabled=True):
                policy, value = model(tokens)
        if policy.dtype != torch.bfloat16:
            raise AssertionError(
                f"the policy head returned {policy.dtype}, not bfloat16. "
                f"policy_buffer is bf16 because the reference's is; a different "
                f"dtype here means Gate 2 would be comparing something the "
                f"production gather never sees.")
        # bf16 bit patterns, unchanged. `.view(torch.uint16)` reinterprets; it
        # does not convert, so the file holds exactly what the network produced.
        logits[start:start + len(chunk)] = policy.cpu().view(torch.uint16).numpy()
        values[start:start + len(chunk)] = value.float().cpu().numpy()
        print(f"  forward {start + len(chunk)}/{len(fens)}", flush=True)

    # --- the three reference columns, all in canonical order ------------
    move_offset = np.zeros(len(fens) + 1, dtype=np.uint64)
    moves_flat: list[str] = []
    cpu_pychess: list[np.ndarray] = []
    gpu_pychess: list[np.ndarray] = []
    cpu_libchess: list[np.ndarray] = []

    for i, board in enumerate(boards):
        # python-chess's GENERATION order -- the order the reference reduces in.
        pychess_order = list(board.legal_moves)
        canonical = sorted(pychess_order, key=lambda m: canonical_key(m, board))
        canonical_uci = [m.uci() for m in canonical]

        # chess-library's GENERATION order, from the port itself. Cross-checked
        # against python-chess as a set: if the two movegens ever disagreed
        # about WHICH moves are legal, that is a C1 regression and it must not
        # be discovered as a mysterious Gate 2 delta.
        lib_order_uci = guofish_core.generation_order(fens[i])
        if sorted(lib_order_uci) != sorted(canonical_uci):
            raise AssertionError(
                f"movegen disagreement at {fens[i]}\n"
                f"  python-chess  : {sorted(canonical_uci)}\n"
                f"  chess-library : {sorted(lib_order_uci)}")
        by_uci = {m.uci(): m for m in pychess_order}
        lib_order = [by_uci[u] for u in lib_order_uci]

        row = torch.from_numpy(logits[i].copy()).view(torch.bfloat16)

        # Reduced in python-chess order, then permuted to canonical.
        cpu_p = reference_priors(row, pychess_order, torch.device("cpu"))
        gpu_p = reference_priors(row, pychess_order, device)
        # Reduced in chess-library order, then permuted to canonical.
        cpu_l = reference_priors(row, lib_order, torch.device("cpu"))

        pychess_pos = {m.uci(): k for k, m in enumerate(pychess_order)}
        lib_pos = {u: k for k, u in enumerate(lib_order_uci)}
        perm_p = [pychess_pos[u] for u in canonical_uci]
        perm_l = [lib_pos[u] for u in canonical_uci]

        cpu_pychess.append(cpu_p[perm_p])
        gpu_pychess.append(gpu_p[perm_p])
        cpu_libchess.append(cpu_l[perm_l])
        moves_flat.extend(canonical_uci)
        move_offset[i + 1] = len(moves_flat)
        if (i + 1) % 100 == 0 or i + 1 == len(fens):
            print(f"  softmax {i + 1}/{len(fens)}", flush=True)

    np.savez_compressed(
        args.out,
        fens=np.array(fens, dtype=object),
        logits=logits,
        values=values,
        move_offset=move_offset,
        moves=np.array(moves_flat, dtype=object),
        priors_cpu_pychess=np.concatenate(cpu_pychess).astype(np.float32),
        priors_gpu_pychess=np.concatenate(gpu_pychess).astype(np.float32),
        priors_cpu_libchess=np.concatenate(cpu_libchess).astype(np.float32),
        allow_pickle=True,
    )

    manifest = {
        "generator": "tools/gen_c10_gate2_golden.py",
        "python": platform.python_version(),
        "python_chess": chess.__version__,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "model": {"path": str(args.model.relative_to(REPO_ROOT)),
                  "sha256": sha256_of(args.model)},
        "corpus": {"path": str(args.corpus.relative_to(REPO_ROOT)),
                   "sha256": sha256_of(args.corpus),
                   "positions": len(fens)},
        "autocast_dtype": str(mctsv4.AUTOCAST_DTYPE),
        "policy_size": int(mctsv4.POLICY_SIZE),
        "columns": {
            "priors_cpu_pychess": "reference INTERIOR path: ATen CPU softmax, "
                                  "python-chess generation order",
            "priors_gpu_pychess": "reference ROOT path: ATen CUDA softmax, "
                                  "python-chess generation order",
            "priors_cpu_libchess": "ATen CPU softmax, chess-library generation order "
                                   "(isolates permutation from implementation)",
        },
        "dump_sha256": sha256_of(args.out),
    }
    args.manifest.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8",
                             newline="\n")

    # --- what the reference disagrees with itself about -----------------
    cpu_all = np.concatenate(cpu_pychess)
    gpu_all = np.concatenate(gpu_pychess)
    lib_all = np.concatenate(cpu_libchess)
    split = np.abs(cpu_all - gpu_all)
    perm = np.abs(cpu_all - lib_all)
    print(f"wrote {args.out}")
    print(f"wrote {args.manifest}")
    print(f"  positions {len(fens)}  priors {cpu_all.size}")
    print(f"  reference root/interior device split : max {split.max():.3e}  "
          f"differing priors {int((split > 0).sum())}/{split.size}")
    print(f"  permutation alone (same ATen)        : max {perm.max():.3e}  "
          f"differing priors {int((perm > 0).sum())}/{perm.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
