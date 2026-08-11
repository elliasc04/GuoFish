#!/usr/bin/env python
"""C11b Gate 2 extension — ATen's answer over the recorded logits at each T.

    python tools/gen_c11b_gate2_temp_golden.py [--force]

WHY THIS IS A SECOND FILE AND NOT A REGENERATION OF THE FIRST
=============================================================
`golden/c10_gate2.npz` is Gate 2's input and its reference at T = 1.0. It cost a
CUDA forward pass over 500 positions and it is pinned by digest in two manifests
and by Amendment A. Regenerating it to add temperature columns would invalidate
every figure C10 and C10b recorded against it for no gain, because THE INPUT DOES
NOT CHANGE: temperature divides the logits, and the logits are already in that
file, bit-for-bit as the network produced them.

So this generator reads that file and runs ATen over it again at each T. No
model is loaded, no forward pass happens, and the network never runs twice. What
comes out is a statement about `torch.softmax` at three temperatures over one
fixed set of logits — which is exactly what the Gate 2 extension compares the
C++ gather against.

THE DIVIDE IS PLACED WHERE THE REFERENCE PLACES IT
==================================================
`core.mctsv4.MCTSNode.expand` gathers the legal logits, `.float()`s them, and
only then divides:

    legal_logits = policy[legal_indices]
    if legal_logits.dtype != torch.float32:
        legal_logits = legal_logits.float()
    if policy_temperature != 1.0:
        legal_logits = legal_logits / policy_temperature

Three details of that are reproduced verbatim below and none is cosmetic:

  * The divide is AFTER the widening, so it happens in float32 over a value that
    came from bf16 — not in bf16, and not in double.
  * `policy_temperature` is a PYTHON FLOAT divided into a float32 tensor. Torch's
    weak-scalar rule converts the scalar to the tensor's dtype, so the division
    is float32 / float32. This is why `SearchConfig::policy_temperature` is a
    `float` on the C++ side; a double there would compute a different number and
    the gate would be measuring the wrong pair of functions.
  * The `!= 1.0` guard is kept. Division by 1.0f is exact in IEEE-754 so it
    changes nothing, and that is the point — carrying the guard here means the
    T = 1.0 columns this file writes are produced by literally the same
    statements that produced `golden/c10_gate2.npz`, which `--verify` then
    checks bit-for-bit.

THE T = 1.0 COLUMNS ARE A SELF-CHECK, NOT PADDING
=================================================
They must come out bit-identical to `golden/c10_gate2.npz`'s three columns. This
generator asserts that before it writes anything. If they do not match, the
transcription above is wrong somewhere and every T != 1.0 column it would have
written is wrong in the same way — so the identity column is what makes the other
two trustworthy, and it fails loudly rather than being reported as a curiosity.

Global Rule 2: this reads the Python reference and `golden/c10_gate2.npz`, and
nothing else. `guofish_core.generation_order` is consulted for the third column,
exactly as `tools/gen_c10_gate2_golden.py` consults it, and for the same reason:
it supplies a REDUCTION ORDER, never a value. No column here has ever seen a
prior computed by C++.
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

DEFAULT_SOURCE = REPO_ROOT / "golden" / "c10_gate2.npz"
DEFAULT_SOURCE_MANIFEST = REPO_ROOT / "golden" / "c10_gate2_manifest.json"
DEFAULT_CORPUS = REPO_ROOT / "golden" / "c10_corpus.json"
DEFAULT_OUT = REPO_ROOT / "golden" / "c11b_gate2_temp.npz"
DEFAULT_MANIFEST = REPO_ROOT / "golden" / "c11b_gate2_temp_manifest.json"

# The three the brief names. 0.7 sharpens, 1.5 flattens, 1.0 is the identity and
# is present as the self-check described in the module docstring.
TEMPERATURES = (0.7, 1.0, 1.5)

COLUMNS = ("priors_cpu_pychess", "priors_gpu_pychess", "priors_cpu_libchess")


def suffix(temperature: float) -> str:
    """`0.7` -> `t070`. npz keys go through `np.savez`'s kwargs, so no dots."""
    return f"t{int(round(temperature * 100)):03d}"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_priors(logits_row: torch.Tensor, order: list[chess.Move],
                     device: torch.device, temperature: float) -> np.ndarray:
    """ATen's priors for `order`, reduced in `order`, returned in `order`.

    Transcribed from `MCTSNode.expand`, temperature included. See the module
    docstring for why each of the three lines below is where it is.
    """
    index = torch.tensor([m.from_square * 64 + m.to_square for m in order],
                         dtype=torch.long, device=device)
    legal = logits_row.to(device)[index]
    if legal.dtype != torch.float32:
        legal = legal.float()
    if temperature != 1.0:
        legal = legal / temperature
    return torch.softmax(legal, dim=0).float().cpu().numpy()


def smallest_nonzero_gap(priors: np.ndarray, offsets: np.ndarray) -> tuple[float, int]:
    """(smallest non-zero within-position gap, the position it is at).

    THE STATISTIC THE BRIEF ASKS TO BE RE-MEASURED, and it is measured exactly as
    `tools/drill_c10_gate2.py::_smallest_gap_pair` measures it: sort each
    position's priors, take adjacent differences, ignore the exact ties that
    promotion collisions produce, and keep the smallest.

    It matters because Gate 2's ordering criterion is only INDEPENDENTLY
    checkable while the corpus' closest pair is separated by more than the 1e-6
    magnitude bound. C10b recorded 1.927e-06 at T = 1.0 — 1.9x the bound, which
    is not much headroom. Flattening compresses priors toward uniform, so T > 1
    shrinks this number, and if it crosses 1e-6 the drill's
    `invert-inside-tolerance` construction stops being possible at that
    temperature. That is a finding to report, not a bound to quietly loosen.
    """
    best = float("inf")
    where = -1
    for i in range(len(offsets) - 1):
        begin, end = int(offsets[i]), int(offsets[i + 1])
        chunk = np.sort(priors[begin:end].astype(np.float64))
        gaps = np.diff(chunk)
        nonzero = gaps[gaps > 0]
        if nonzero.size == 0:
            continue
        smallest = float(nonzero.min())
        if smallest < best:
            best, where = smallest, i
    return best, where


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Gate 2's recorded logits and T=1.0 reference columns")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for path in (args.out, args.manifest):
        if path.exists() and not args.force:
            print(f"refusing to overwrite {path} without --force", file=sys.stderr)
            return 2

    if not torch.cuda.is_available():
        print("ERROR: the `priors_gpu_pychess` column records the CUDA softmax path, "
              "which is half of the reference's root/interior split. Refusing to "
              "generate it on CPU.", file=sys.stderr)
        return 1

    source = np.load(args.source, allow_pickle=True)
    fens = [str(f) for f in source["fens"]]
    logits = source["logits"]
    offsets = source["move_offset"]
    moves = [str(m) for m in source["moves"]]
    boards = [chess.Board(f) for f in fens]
    device = torch.device("cuda")

    print(f"source {args.source}  positions {len(fens)}  priors {len(moves)}")

    # --- the per-position orders, resolved once and reused at every T -------
    orders = []
    for i, board in enumerate(boards):
        pychess_order = list(board.legal_moves)
        canonical_uci = sorted(m.uci() for m in pychess_order)
        lib_order_uci = guofish_core.generation_order(fens[i])
        if sorted(lib_order_uci) != canonical_uci:
            raise AssertionError(
                f"movegen disagreement at {fens[i]}\n"
                f"  python-chess  : {canonical_uci}\n"
                f"  chess-library : {sorted(lib_order_uci)}")
        recorded = moves[int(offsets[i]):int(offsets[i + 1])]
        if recorded != canonical_uci:
            raise AssertionError(
                f"the recorded canonical move list for {fens[i]} is not the one "
                f"python-chess produces now:\n  golden: {recorded}\n  now   : {canonical_uci}")
        by_uci = {m.uci(): m for m in pychess_order}
        lib_order = [by_uci[u] for u in lib_order_uci]
        pychess_pos = {m.uci(): k for k, m in enumerate(pychess_order)}
        lib_pos = {u: k for k, u in enumerate(lib_order_uci)}
        orders.append((pychess_order, lib_order,
                       [pychess_pos[u] for u in canonical_uci],
                       [lib_pos[u] for u in canonical_uci]))

    # --- one pass per temperature -------------------------------------------
    out: dict = {
        "fens": np.array(fens, dtype=object),
        "move_offset": offsets,
        "moves": np.array(moves, dtype=object),
        "temperatures": np.array(TEMPERATURES, dtype=np.float64),
    }
    gaps: dict = {}

    for temperature in TEMPERATURES:
        cpu_p, gpu_p, cpu_l = [], [], []
        for i, board in enumerate(boards):
            pychess_order, lib_order, perm_p, perm_l = orders[i]
            row = torch.from_numpy(logits[i].copy()).view(torch.bfloat16)
            cpu_p.append(reference_priors(row, pychess_order, torch.device("cpu"),
                                          temperature)[perm_p])
            gpu_p.append(reference_priors(row, pychess_order, device, temperature)[perm_p])
            cpu_l.append(reference_priors(row, lib_order, torch.device("cpu"),
                                          temperature)[perm_l])
            if (i + 1) % 100 == 0 or i + 1 == len(fens):
                print(f"  T={temperature:g}  softmax {i + 1}/{len(fens)}", flush=True)

        columns = {
            "priors_cpu_pychess": np.concatenate(cpu_p).astype(np.float32),
            "priors_gpu_pychess": np.concatenate(gpu_p).astype(np.float32),
            "priors_cpu_libchess": np.concatenate(cpu_l).astype(np.float32),
        }
        for name, values in columns.items():
            out[f"{name}_{suffix(temperature)}"] = values

        # THE SELF-CHECK. See the module docstring: if the identity columns are
        # not bit-identical to Gate 2's, the transcription is wrong and so is
        # everything else this file would write.
        if temperature == 1.0:
            for name, values in columns.items():
                recorded = source[name]
                if not np.array_equal(values.view(np.uint32), recorded.view(np.uint32)):
                    differing = int((values != recorded).sum())
                    print(f"ERROR: the T=1.0 column `{name}` is not bit-identical to "
                          f"{args.source}'s ({differing}/{values.size} priors differ, "
                          f"max |delta| {np.abs(values.astype(np.float64) - recorded.astype(np.float64)).max():.3e}).\n"
                          f"  The temperature transcription in reference_priors() does not "
                          f"reproduce the reference at the identity, so no column here can be "
                          f"trusted. Nothing was written.", file=sys.stderr)
                    return 1
            print(f"  T=1.0 self-check: all three columns bit-identical to "
                  f"{args.source.name}")

        gap, position = smallest_nonzero_gap(columns["priors_cpu_pychess"], offsets)
        gaps[f"{temperature:g}"] = {"min_nonzero_gap": gap,
                                    "position": position,
                                    "fen": fens[position] if position >= 0 else None}

    np.savez_compressed(args.out, allow_pickle=True, **out)

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    manifest = {
        "generator": "tools/gen_c11b_gate2_temp_golden.py",
        "python": platform.python_version(),
        "python_chess": chess.__version__,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "temperatures": list(TEMPERATURES),
        # The logits are not re-derived here, so the provenance that matters is
        # the file they came from — and through it, the checkpoint and the
        # autocast dtype the C10 manifest already pins (Amendment A).
        "source": {"path": str(args.source.relative_to(REPO_ROOT)),
                   "sha256": sha256_of(args.source)},
        "source_model": source_manifest["model"],
        "autocast_dtype": source_manifest["autocast_dtype"],
        "corpus": {"path": str(args.corpus.relative_to(REPO_ROOT)),
                   "sha256": sha256_of(args.corpus),
                   "positions": len(fens)},
        "policy_size": int(mctsv4.POLICY_SIZE),
        "identity_self_check": "the T=1.0 columns were verified bit-identical to "
                               "golden/c10_gate2.npz before this file was written",
        "min_nonzero_prior_gap": gaps,
        "dump_sha256": None,
    }
    manifest["dump_sha256"] = sha256_of(args.out)
    args.manifest.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8",
                             newline="\n")

    print(f"wrote {args.out}")
    print(f"wrote {args.manifest}")
    print()
    print("smallest non-zero inter-prior gap, per temperature "
          "(Gate 2's magnitude bound is 1.0e-06)")
    for temperature in TEMPERATURES:
        record = gaps[f"{temperature:g}"]
        verdict = "OK" if record["min_nonzero_gap"] > 1e-6 else "BELOW THE BOUND"
        print(f"  T={temperature:<4g} {record['min_nonzero_gap']:.3e}  "
              f"({record['min_nonzero_gap'] / 1e-6:.2f}x the bound)  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
