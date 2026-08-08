#!/usr/bin/env python
"""Average forward-pass latency of a GuoFish checkpoint, per batch and per position.

    python benchmarking/bench_inference.py models/guofish5_10M/v5_10.9M_best.pt
    python benchmarking/bench_inference.py models/guofish4/guofish4_25.6M_policy_final.pt
    python benchmarking/bench_inference.py <ckpt> --batch-size 128 256 --json-out out.json
    python benchmarking/bench_inference.py --check-only        # compatibility check alone

What is measured is the ONE call MCTS actually makes per batch: a no-grad
forward under the engine's autocast, on a (B, 68) int64 token batch already
resident on the device, with NO legal_move_mask -- core.mctsv4._evaluate_batch
masks inside MCTSNode.expand, not in the model. `--mask` measures the masked
variant (the raw-policy path in playv5.pick_engine_move) instead.

MODEL LOADING is delegated to playing/v5/playv5.load_model, which is the only
loader in the tree that takes both generations interchangeably: it dispatches on
the checkpoint's own metadata, so the 10.9M v5 student (d_model=384 x6,
GELU FFN, final LayerNorm) and the 25.6M legacy v2 net (d_model=512 x8, ReLU
FFN, no final norm) both load through one code path with no flag to get wrong.
playv4.load_model does NOT: it is bound to core.mctsv2 and the v1/v2 classes and
has no ChessTransformerV5 to build, so a v5 checkpoint is refused there with a
pointer to playv5 rather than a wall of missing-key errors. --check-only (run by
default before every benchmark) proves the interchange on this machine by
loading each reference checkpoint and shape-checking a real forward pass.

INPUTS are real board positions by default: the `tokens` field of the Pass B
shards under data/processed/multipv, sampled at random. That is the same 68-token
encoding the engine emits, so the piece-density and token distribution are those
of positions the net was trained and plays on. `--source synthetic` falls back to
random legal playouts encoded through playv5.board_to_tokens_v2 -- identical
encoding, no data dependency -- and is selected automatically if the shards are
absent.

TIMING uses torch.cuda.Event pairs recorded around each iteration with no
in-loop synchronize (one sync at the end), so the reported figure is GPU time in
the steady state rather than launch latency. CPU runs fall back to
time.perf_counter. Warmup iterations are untimed and discarded.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from playing.v5.playv5 import (board_to_tokens_v2, inference_dtype,  # noqa: E402
                               legal_move_mask, load_model)

DEFAULT_SHARDS = _PROJECT_ROOT / "data" / "processed" / "multipv"

# Checkpoints the compatibility gate loads. The point of the pair is that they
# are DIFFERENT architectures behind the same loader, so keep one of each.
REFERENCE_CHECKPOINTS = [
    ("10.9M v5 student", _PROJECT_ROOT / "models" / "guofish5_10M" / "v5_10.9M_best.pt"),
    ("25.6M legacy v2", _PROJECT_ROOT / "models" / "guofish4" / "guofish4_25.6M_policy_final.pt"),
    ("25.6M legacy v2 (alt)", _PROJECT_ROOT / "models" / "guofish2" / "guofish2_25.6M_54.8p.pt"),
]

SEQ_LENGTH = 68
POLICY_SIZE = 4096


def log(msg: str = "") -> None:
    print(msg, flush=True)


# --- input data -------------------------------------------------------------

def tokens_from_shards(shard_dir: Path, n: int, split: str,
                       rng: np.random.Generator) -> torch.Tensor:
    """Sample `n` real positions' token rows out of the Pass B shards.

    The shards are fixed-width memmaps, so this is index arithmetic, not a
    parse; see data/multiPV/record_format.py for the dtype. Only the `tokens`
    field is touched -- the policy/value targets are irrelevant to a forward
    pass -- and the sample is drawn across ALL shards rather than off the head
    of one, since shards are written in game order and the head is opening-heavy.
    """
    sys.path.insert(0, str(_PROJECT_ROOT / "data" / "multiPV"))
    from record_format import RECORD_DTYPE  # noqa: E402

    paths = sorted(shard_dir.glob(f"{split}_*.bin"))
    if not paths:
        raise FileNotFoundError(f"no {split}_*.bin shards in {shard_dir}")

    itemsize = RECORD_DTYPE.itemsize
    counts = [p.stat().st_size // itemsize for p in paths]
    total = sum(counts)
    if total == 0:
        raise ValueError(f"shards in {shard_dir} hold no records")

    # Draw global indices, then group by shard so each file is opened once.
    picks = rng.integers(0, total, size=n, endpoint=False)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    out = np.empty((n, SEQ_LENGTH), dtype=np.int64)
    shard_of = np.searchsorted(offsets, picks, side="right") - 1
    for s in np.unique(shard_of):
        where = np.flatnonzero(shard_of == s)
        rows = picks[where] - offsets[s]
        mm = np.memmap(paths[s], dtype=RECORD_DTYPE, mode="r")
        out[where] = mm["tokens"][rows].astype(np.int64)
        del mm
    return torch.from_numpy(out)


def tokens_synthetic(n: int, seed: int) -> torch.Tensor:
    """`n` positions from random legal playouts, encoded exactly as the engine does.

    Random self-play walks the same distribution of legal board states the
    encoder sees at play time (castling rights decaying, ep files appearing,
    material coming off), which a uniform random token tensor would not: it
    would produce boards with no kings and impossible piece counts, and the
    attention pattern -- hence the runtime -- is not input-independent enough to
    make that a safe substitute.
    """
    import random

    import chess

    rng = random.Random(seed)
    rows = []
    board = chess.Board()
    while len(rows) < n:
        if board.is_game_over(claim_draw=False) or board.fullmove_number > 120:
            board = chess.Board()
            continue
        rows.append(board_to_tokens_v2(board))
        moves = list(board.legal_moves)
        board.push(rng.choice(moves))
    return torch.stack(rows).to(torch.int64)


def build_input_pool(args, batch_size: int, device: torch.device,
                     model: torch.nn.Module) -> tuple[list[torch.Tensor], str]:
    """`--pool` distinct device-resident batches to rotate through while timing.

    One batch reused 1000 times would be a fair measure of the forward pass but
    an unfalsifiable one -- nothing in the loop could tell a real forward from a
    cached result. Rotating a handful of distinct batches costs a few MB and
    removes the doubt. All of them are uploaded before timing starts, so the
    measured window contains no host-to-device copy.
    """
    n = batch_size * args.pool
    rng = np.random.default_rng(args.seed)
    source = args.source
    if source == "auto":
        source = "data" if DEFAULT_SHARDS.exists() or args.shards.exists() else "synthetic"

    if source == "data":
        try:
            tokens = tokens_from_shards(args.shards, n, args.split, rng)
            origin = f"data ({args.shards})"
        except (FileNotFoundError, ValueError) as e:
            if args.source == "data":
                raise
            log(f"  shards unavailable ({e}); falling back to synthetic positions")
            tokens, origin = tokens_synthetic(n, args.seed), "synthetic (fallback)"
    else:
        tokens, origin = tokens_synthetic(n, args.seed), "synthetic"

    validate_tokens(tokens, model)
    pool = [tokens[i * batch_size:(i + 1) * batch_size].to(device, non_blocking=False)
            for i in range(args.pool)]
    return pool, origin


def validate_tokens(tokens: torch.Tensor, model: torch.nn.Module) -> None:
    """Fail loudly on a shape/vocab mismatch rather than at the embedding lookup.

    An out-of-range token id is a CUDA-side device assert: asynchronous, fatal to
    the whole context, and reported far from its cause. Checking on the host
    first costs one pass over a few thousand ints.
    """
    if tokens.shape[1] != SEQ_LENGTH:
        raise ValueError(f"expected {SEQ_LENGTH}-token rows, got {tokens.shape[1]}")
    seq_length = getattr(model, "seq_length", SEQ_LENGTH)
    if seq_length != SEQ_LENGTH:
        raise ValueError(f"model expects {seq_length} tokens; this benchmark only "
                         "emits the 68-token scheme")
    vocab = model.embedding.num_embeddings
    lo, hi = int(tokens.min()), int(tokens.max())
    if lo < 0 or hi >= vocab:
        raise ValueError(f"token ids [{lo}, {hi}] outside the model's vocab of {vocab}")


def build_mask(pool: list[torch.Tensor], seed: int) -> list[torch.Tensor]:
    """Per-batch legal-move masks, for --mask.

    The masks cannot be recovered from the token rows without replaying the
    position, so they are generated from independent random playouts. The mask is
    a masked_fill over the 4096 logits: its cost depends on how many entries are
    True, and a realistic ~30-legal-moves mask is the right density for that.
    """
    import random

    import chess

    rng = random.Random(seed + 1)
    masks = []
    board = chess.Board()
    for batch in pool:
        rows = []
        while len(rows) < batch.shape[0]:
            if board.is_game_over(claim_draw=False) or board.fullmove_number > 120:
                board = chess.Board()
                continue
            rows.append(legal_move_mask(board))
            board.push(rng.choice(list(board.legal_moves)))
        masks.append(torch.stack(rows).to(batch.device))
    return masks


# --- compatibility gate -----------------------------------------------------

def check_compatibility(device: torch.device, checkpoints=None) -> list[dict]:
    """Load each reference checkpoint through the shared loader and forward it.

    "Loads without raising" is a weak claim -- build_model_for_checkpoint's whole
    job is to pick the right architecture, and picking the WRONG one still loads
    cleanly whenever the shapes happen to line up. So each model is also run on a
    real batch and checked for the (B, 4096) / (B,) output contract with finite
    values, which is what the engine consumes.
    """
    results = []
    for label, path in (checkpoints or REFERENCE_CHECKPOINTS):
        row = {"label": label, "path": str(path)}
        if not path.exists():
            row.update(ok=None, note="not present")
            results.append(row)
            continue
        try:
            model = load_model(path, device)
            params = sum(p.numel() for p in model.parameters())
            tokens = tokens_synthetic(4, seed=0).to(device)
            param_dtype = next(model.parameters()).dtype
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=param_dtype,
                                    enabled=param_dtype in (torch.float16, torch.bfloat16)):
                    policy, value = model(tokens)
            assert policy.shape == (4, POLICY_SIZE), f"policy {tuple(policy.shape)}"
            assert value.shape == (4,), f"value {tuple(value.shape)}"
            assert torch.isfinite(policy.float()).all(), "non-finite policy logits"
            assert torch.isfinite(value.float()).all(), "non-finite value"
            row.update(ok=True, arch=type(model).__name__, params=params,
                       dtype=str(param_dtype).removeprefix("torch."),
                       note=f"forward OK, |v|max {value.abs().max().item():.3f}")
            del model
        except Exception as e:                     # noqa: BLE001 - reported, not swallowed
            row.update(ok=False, note=f"{type(e).__name__}: {str(e).splitlines()[0][:120]}")
        results.append(row)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def report_compatibility(rows: list[dict]) -> bool:
    log("MODEL COMPATIBILITY (playing/v5/playv5.load_model)")
    for r in rows:
        if r["ok"] is None:
            log(f"  [skip] {r['label']:<22} {r['note']} -- {r['path']}")
        elif r["ok"]:
            log(f"  [ ok ] {r['label']:<22} {r['arch']} {r['params'] / 1e6:.2f}M params, "
                f"{r['dtype']} -- {r['note']}")
        else:
            log(f"  [FAIL] {r['label']:<22} {r['note']}")
    loaded = [r for r in rows if r["ok"]]
    sizes = {round(r["params"] / 1e6, 1) for r in loaded}
    if any(r["ok"] is False for r in rows):
        log("  => a reference checkpoint FAILED to load; the loader is not "
            "interchangeable on this tree")
        return False
    if len(sizes) >= 2:
        log(f"  => one loader, {len(loaded)} checkpoints, "
            f"{len(sizes)} distinct architectures ({', '.join(f'{s}M' for s in sorted(sizes))}) "
            "-- interchangeable")
    elif loaded:
        log(f"  => only {len(sizes)} architecture present; interchange not exercised")
    else:
        log("  => no reference checkpoints present; interchange not exercised")
    return True


# --- benchmark --------------------------------------------------------------

def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def bench_one(model, pool, masks, args, device) -> dict:
    """Warmup, then `--iters` timed forward passes over the rotating pool."""
    param_dtype = next(model.parameters()).dtype
    use_autocast = param_dtype in (torch.float16, torch.bfloat16)
    n_pool = len(pool)

    def forward(i: int):
        batch = pool[i % n_pool]
        mask = masks[i % n_pool] if masks else None
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=param_dtype,
                                enabled=use_autocast):
                return model(batch, legal_move_mask=mask)

    for i in range(args.warmup):
        forward(i)
    if device.type == "cuda":
        torch.cuda.synchronize()

    if device.type == "cuda":
        # Events are recorded around every iteration but synchronized ONCE at
        # the end. An in-loop synchronize would drain the queue each time and
        # fold launch latency into every sample, which measures the harness
        # rather than the model.
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        wall0 = time.perf_counter()
        for i in range(args.iters):
            starts[i].record()
            forward(i)
            ends[i].record()
        torch.cuda.synchronize()
        wall = time.perf_counter() - wall0
        times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    else:
        times_ms = []
        wall0 = time.perf_counter()
        for i in range(args.iters):
            t0 = time.perf_counter()
            forward(i)
            times_ms.append((time.perf_counter() - t0) * 1e3)
        wall = time.perf_counter() - wall0

    batch_size = pool[0].shape[0]
    mean_ms = statistics.fmean(times_ms)
    return {
        "batch_size": batch_size,
        "iters": args.iters,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "p95_ms": percentile(times_ms, 95),
        "p99_ms": percentile(times_ms, 99),
        "per_position_us": mean_ms * 1e3 / batch_size,
        "positions_per_s": batch_size / (mean_ms / 1e3),
        "wall_s": wall,
        # Timed GPU work vs. wall clock. Well under 1.0 means the CPU could not
        # keep the queue full and the batch is too small to saturate the device.
        "gpu_busy_fraction": (sum(times_ms) / 1e3 / wall) if device.type == "cuda" else 1.0,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Average forward-pass inference time for a GuoFish checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint", type=Path, nargs="?",
                   help="Path to the model checkpoint (v5 or legacy v2; the "
                        "architecture is read from the checkpoint itself)")
    p.add_argument("--batch-size", type=int, nargs="+", default=[256],
                   help="Positions per forward pass; several values sweep")
    p.add_argument("--warmup", type=int, default=50,
                   help="Untimed passes to build CUDA kernels and autotune")
    p.add_argument("--iters", type=int, default=1000, help="Timed passes")
    p.add_argument("--pool", type=int, default=8,
                   help="Distinct input batches held on the device and rotated")
    p.add_argument("--source", choices=("auto", "data", "synthetic"), default="auto",
                   help="Where the positions come from")
    p.add_argument("--shards", type=Path, default=DEFAULT_SHARDS,
                   help="Pass B shard directory for --source data")
    p.add_argument("--split", default="train", help="Shard split to sample")
    p.add_argument("--mask", action="store_true",
                   help="Also pass a legal_move_mask. OFF by default: the MCTS "
                        "evaluator masks in MCTSNode.expand, not in the model")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check-only", action="store_true",
                   help="Run the compatibility check and exit")
    p.add_argument("--no-check", action="store_true",
                   help="Skip the compatibility check before benchmarking")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        log("CUDA requested but not available.")
        return 1

    torch.manual_seed(args.seed)
    dev_name = (torch.cuda.get_device_name(0) if device.type == "cuda"
                else "CPU")
    log("=" * 92)
    log(f"bench_inference | {dev_name} | torch {torch.__version__}")
    log("=" * 92)

    if not args.no_check or args.check_only:
        if not report_compatibility(check_compatibility(device)):
            return 1
        log("")
    if args.check_only:
        return 0

    if args.checkpoint is None:
        p.error("a checkpoint path is required unless --check-only is given")
    if not args.checkpoint.exists():
        log(f"Error: {args.checkpoint} not found")
        return 1

    log("MODEL UNDER TEST")
    model = load_model(args.checkpoint, device)
    params = sum(q.numel() for q in model.parameters())
    param_dtype = next(model.parameters()).dtype
    # load_model dynamically quantizes every Linear on CPU, and a quantized
    # Linear holds its weight as a packed buffer rather than a Parameter -- so
    # the count below legitimately excludes most of the net there. Say so
    # instead of printing a number that looks like the wrong checkpoint.
    quantized = any("quantized" in type(m).__module__ for m in model.modules())
    log(f"  {type(model).__name__}, {params:,} params "
        f"({params / 1e6:.2f}M), weights {str(param_dtype).removeprefix('torch.')}"
        + ("  [INT8-quantized Linears excluded from the count]" if quantized else ""))
    log(f"  autocast {str(inference_dtype(device)).removeprefix('torch.')} "
        f"on {device.type} | legal_move_mask {'ON' if args.mask else 'OFF (engine default)'}")
    log("")

    rows = []
    for batch_size in args.batch_size:
        pool, origin = build_input_pool(args, batch_size, device, model)
        masks = build_mask(pool, args.seed) if args.mask else None
        log(f"BATCH {batch_size} | inputs: {origin}, {len(pool)} batches "
            f"({batch_size * len(pool):,} positions) resident on {device.type}")
        log(f"  {args.warmup} warmup + {args.iters} timed passes, "
            f"{'torch.cuda.Event' if device.type == 'cuda' else 'time.perf_counter'} timing")
        r = bench_one(model, pool, masks, args, device)
        r["source"] = origin
        rows.append(r)

        log(f"  per batch     mean {r['mean_ms']:.3f} ms   "
            f"median {r['median_ms']:.3f}   sd {r['stdev_ms']:.3f}   "
            f"min {r['min_ms']:.3f}   p95 {r['p95_ms']:.3f}   "
            f"p99 {r['p99_ms']:.3f}   max {r['max_ms']:.3f}")
        log(f"  per position  {r['per_position_us']:.2f} us          "
            f"throughput {r['positions_per_s']:,.0f} positions/s")
        if device.type == "cuda":
            log(f"  gpu busy      {r['gpu_busy_fraction'] * 100:.1f}% of the "
                f"{r['wall_s']:.2f}s wall window")
        log("")
        del pool, masks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    log("-" * 92)
    log(f"{'batch':>7} {'ms/batch':>10} {'us/position':>13} {'positions/s':>14}")
    for r in rows:
        log(f"{r['batch_size']:>7} {r['mean_ms']:>10.3f} "
            f"{r['per_position_us']:>13.2f} {r['positions_per_s']:>14,.0f}")
    log("-" * 92)

    if args.json_out:
        payload = {
            "device": dev_name,
            "torch": torch.__version__,
            "checkpoint": str(args.checkpoint),
            "arch": type(model).__name__,
            "params": params,
            "weight_dtype": str(param_dtype).removeprefix("torch."),
            "masked": args.mask,
            "args": {k: str(v) for k, v in vars(args).items()},
            "rows": rows,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log(f"json -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
