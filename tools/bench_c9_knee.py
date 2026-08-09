#!/usr/bin/env python
"""C9 prerequisite — where the GPU evaluation curve actually knees.

    python tools/bench_c9_knee.py
    python tools/bench_c9_knee.py --markdown
    python tools/bench_c9_knee.py --batch 8 16 32 64 128 256 --iters 300

WHY THIS EXISTS AND WHY IT DOES NOT REUSE benchmarking/bench_inference.py
------------------------------------------------------------------------
C9 sizes `W x K` — search threads times in-flight paths per thread — from the
batch size at which the GPU stops getting faster per position. Two prior
measurements disagree about where that is:

  recon era   B = 4.86 ms fixed launch cost, I = 4 us/item, "flat to batch 64"
  2026-08     batch 128 -> 6.755 ms, batch 256 -> 13.857 ms, i.e. B ~ 0, I ~ 54 us

Those give OPPOSITE W x K advice. Under the first you need ~128 outstanding
leaves to reach the target throughput; under the second any batch past the knee
delivers the same pos/s and the outstanding count should be minimised outright,
because every in-flight path holds virtual loss on 10-30 nodes and flattens the
root visit distribution.

"Flat to batch 64" has the signature of un-synchronized async launch timing: a
CUDA kernel launch returns to the host immediately, so a timer that does not
synchronize measures the launch, not the work, and the launch cost is genuinely
flat in batch size. `benchmarking/bench_inference.py` times with cuda.Event
pairs and ONE synchronize at the end, which is the right way to measure steady
-state GPU occupancy and the wrong way to measure what a dispatcher thread
waits for. This script therefore calls `torch.cuda.synchronize()` around EVERY
timed iteration, which is the number C9 needs: the wall-clock latency of one
dispatcher round trip.

The three paths, measured separately because they are three different claims
-----------------------------------------------------------------------------
  forward   tokens already resident on the device, forward only, sync after.
            The pure GPU compute curve; comparable to the prior tables.
  reference core.mctsv4._evaluate_batch verbatim: host int64 tokens -> .to(cuda)
            -> forward -> policy_logits.cpu() + values.cpu(). The full 4096-wide
            policy row per item crosses the bus. This is the curve the Python
            engine actually ran on.
  gathered  scope 2.5's production path: H2D from a PINNED int32 buffer (what
            C++ writes), forward, then gather 64 padded policy logits per row on
            the GPU and copy only those back, plus the value. This is what C10
            will do, and it is the curve C9's max_batch should be sized from.

`--source` picks the positions. Real board tokens come from the Pass B shards if
they are present; synthetic ones are random legal playouts through the same
encoder. The token distribution matters — attention cost is not input
-independent — so the source is reported with every table.

Writes nothing. Paste the tables into BENCH.md.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from playing.v5.playv5 import load_model  # noqa: E402

DEFAULT_MODEL = _PROJECT_ROOT / "models" / "guofish5_20M" / "v5_10.9M_best.pt"
DEFAULT_SHARDS = _PROJECT_ROOT / "data" / "processed" / "multipv"
SEQ_LENGTH = 68
POLICY_SIZE = 4096

# scope 2.5: the production gather pads the legal-move set to 64 and masks the
# rest with -inf, so the D2H is 64 wide per row rather than 4096.
GATHER_WIDTH = 64

# core.mctsv4.AUTOCAST_DTYPE
AUTOCAST_DTYPE = torch.bfloat16


def log(msg: str = "") -> None:
    print(msg, flush=True)


# --- inputs -----------------------------------------------------------------


def tokens_synthetic(n: int, seed: int) -> torch.Tensor:
    """`n` positions from random legal playouts, through the engine's encoder."""
    import random

    import chess

    from playing.v5.playv5 import board_to_tokens_v2

    rng = random.Random(seed)
    rows = []
    board = chess.Board()
    while len(rows) < n:
        if board.is_game_over(claim_draw=False) or board.fullmove_number > 120:
            board = chess.Board()
            continue
        rows.append(board_to_tokens_v2(board))
        board.push(rng.choice(list(board.legal_moves)))
    return torch.stack(rows).to(torch.int64)


def tokens_from_shards(shard_dir: Path, n: int, split: str,
                       rng: np.random.Generator) -> torch.Tensor:
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


def build_tokens(args, n: int) -> tuple[torch.Tensor, str]:
    source = args.source
    if source == "auto":
        source = "data" if args.shards.exists() else "synthetic"
    if source == "data":
        try:
            rng = np.random.default_rng(args.seed)
            return tokens_from_shards(args.shards, n, args.split, rng), f"data ({args.shards})"
        except (FileNotFoundError, ValueError) as exc:
            if args.source == "data":
                raise
            log(f"  shards unavailable ({exc}); falling back to synthetic positions")
    return tokens_synthetic(n, args.seed), "synthetic"


# --- the three paths --------------------------------------------------------


def make_forward(model, device):
    def forward(tokens):
        with torch.no_grad():
            with torch.amp.autocast_mode.autocast(device_type=device.type,
                                                  dtype=AUTOCAST_DTYPE,
                                                  enabled=device.type == "cuda"):
                return model(tokens)

    return forward


def path_forward(forward, device_tokens, _host, _pinned, _index):
    """Device-resident tokens; forward only."""
    forward(device_tokens)


def path_reference(forward, _device_tokens, host_tokens, _pinned, _index):
    """core.mctsv4._evaluate_batch: H2D int64, forward, full 4096-wide D2H."""
    batch = host_tokens.to("cuda")
    policy, values = forward(batch)
    policy.cpu()
    values.cpu()


def path_gathered(forward, _device_tokens, _host, pinned, index):
    """scope 2.1/2.5: pinned int32 H2D, forward, 64-wide gather, narrow D2H."""
    batch = pinned.to("cuda", non_blocking=True).to(torch.int64)
    policy, values = forward(batch)
    gathered = torch.gather(policy, 1, index)
    gathered.to("cpu", non_blocking=False)
    values.to("cpu", non_blocking=False)


PATHS = {
    "forward": path_forward,
    "reference": path_reference,
    "gathered": path_gathered,
}


# --- timing -----------------------------------------------------------------


def sm_clock_mhz() -> float | None:
    """Current SM clock, so a table can say whether the GPU was ramped.

    This is not decoration. At batch 8 the device is busy for well under a
    millisecond and then idles inside `torch.cuda.synchronize()`, and an RTX
    5070 drops to ~300 MHz of its 3090 MHz within that gap. A synchronized
    per-call measurement at small batch therefore measures the DVFS ramp, not
    the forward pass, and the resulting curve looks like a large fixed launch
    cost. That is almost certainly what the recon-era "B = 4.86 ms fixed" model
    was: the artefact is stable and reproducible, which is what makes it
    convincing and wrong.
    """
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, check=True)
    except (subprocess.SubprocessError, OSError):
        return None
    first = out.stdout.strip().splitlines()[0].strip()
    try:
        return float(first)
    except ValueError:
        return None


def time_path(name, fn, forward, device_tokens, host_tokens, pinned, index,
              iters: int, warmup: int, cuda: bool) -> dict:
    """Wall-clock per call, with a full device synchronize around EVERY sample.

    The synchronize is the whole point: it is what a dispatcher thread does when
    it waits for a batch. An event-pair measurement without it reports GPU busy
    time and hides launch latency, which is exactly how "flat to batch 64" was
    produced.

    Read this column together with `sat_ms`: this one is the LATENCY a lone
    dispatcher round trip costs from a cold-ish device, and it carries the DVFS
    ramp described in sm_clock_mhz.
    """
    for _ in range(warmup):
        fn(forward, device_tokens, host_tokens, pinned, index)
    if cuda:
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(forward, device_tokens, host_tokens, pinned, index)
        if cuda:
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    clock = sm_clock_mhz()

    samples.sort()
    n = len(samples)
    return {
        "path": name,
        "iters": n,
        "median_s": samples[n // 2],
        "mean_s": statistics.fmean(samples),
        "p10_s": samples[max(0, int(0.10 * n) - 1)],
        "p90_s": samples[min(n - 1, int(0.90 * n))],
        "min_s": samples[0],
        "max_s": samples[-1],
        "sm_clock_mhz": clock,
    }


def time_saturated(fn, forward, device_tokens, host_tokens, pinned, index,
                   iters: int, depth: int, cuda: bool) -> dict:
    """Marginal cost per batch with the device kept busy.

    `depth` calls are enqueued back to back and ONE synchronize closes the
    window, so the GPU never idles long enough to downclock and the reported
    figure is the steady-state cost of an additional batch rather than the cost
    of waking the device up. A search running at any useful rate keeps the GPU
    in this regime — the dispatcher has ~180 us of descent to do between
    batches at 32 outstanding, not seconds — so this is the number the
    throughput ceiling should be computed from, while `median_s` above is the
    number a single isolated round trip costs.
    """
    for _ in range(2):
        for _ in range(depth):
            fn(forward, device_tokens, host_tokens, pinned, index)
        if cuda:
            torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        for _ in range(depth):
            fn(forward, device_tokens, host_tokens, pinned, index)
        if cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) / depth)
    clock = sm_clock_mhz()

    samples.sort()
    n = len(samples)
    return {
        "sat_median_s": samples[n // 2],
        "sat_min_s": samples[0],
        "sat_depth": depth,
        "sat_sm_clock_mhz": clock,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch", type=int, nargs="+",
                        default=[8, 16, 32, 64, 128, 256],
                        help="batch sizes to sweep (C9 brief: 8..256)")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--depth", type=int, default=16,
                        help="batches enqueued back to back per saturated sample")
    parser.add_argument("--paths", nargs="+", default=list(PATHS),
                        choices=list(PATHS))
    parser.add_argument("--source", choices=("auto", "data", "synthetic"), default="auto")
    parser.add_argument("--shards", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    if not cuda:
        log("WARNING: no CUDA device. The knee is a GPU property; this run measures the CPU "
            "path and must not be recorded as the knee.")

    log(f"torch {torch.__version__}  device {device}"
        f"{'  ' + torch.cuda.get_device_name(0) if cuda else ''}")
    log(f"python {platform.python_version()}  {platform.platform()}")
    model = load_model(args.model, device)
    model.eval()
    forward = make_forward(model, device)

    biggest = max(args.batch)
    tokens, origin = build_tokens(args, biggest)
    log(f"positions: {origin}")
    log("")

    rows = []
    for batch in args.batch:
        host = tokens[:batch].contiguous()
        device_tokens = host.to(device)
        # The C++ side writes int32 into a pinned buffer (scope 2.1).
        pinned = torch.empty((batch, SEQ_LENGTH), dtype=torch.int32,
                             pin_memory=cuda)
        pinned.copy_(host.to(torch.int32))
        # A plausible legal-move gather: 64 padded indices per row.
        index = torch.randint(0, POLICY_SIZE, (batch, GATHER_WIDTH),
                              device=device, dtype=torch.int64)

        for name in args.paths:
            row = time_path(name, PATHS[name], forward, device_tokens, host, pinned,
                            index, args.iters, args.warmup, cuda)
            row.update(time_saturated(PATHS[name], forward, device_tokens, host, pinned,
                                      index, max(8, args.iters // 8), args.depth, cuda))
            row["batch"] = batch
            row["ms_per_batch"] = row["median_s"] * 1e3
            row["us_per_pos"] = row["median_s"] * 1e6 / batch
            row["pos_per_s"] = batch / row["median_s"]
            row["sat_ms_per_batch"] = row["sat_median_s"] * 1e3
            row["sat_us_per_pos"] = row["sat_median_s"] * 1e6 / batch
            row["sat_pos_per_s"] = batch / row["sat_median_s"]
            rows.append(row)
            log(f"  batch {batch:>4}  {name:<9}  isolated {row['ms_per_batch']:8.3f} ms "
                f"({row['pos_per_s']:7.0f} pos/s, {row['sm_clock_mhz'] or 0:.0f} MHz)   "
                f"saturated {row['sat_ms_per_batch']:8.3f} ms "
                f"({row['sat_pos_per_s']:7.0f} pos/s, {row['sat_sm_clock_mhz'] or 0:.0f} MHz)")
        del device_tokens, pinned, index
        if cuda:
            torch.cuda.empty_cache()

    if args.markdown:
        log("")
        for name in args.paths:
            log(f"### {name}")
            log("")
            log("| batch | isolated ms | isolated pos/s | saturated ms | saturated us/pos "
                "| saturated pos/s | best ms | best pos/s | SM MHz |")
            log("|------:|------------:|---------------:|-------------:|-----------------:"
                "|----------------:|--------:|-----------:|-------:|")
            for row in rows:
                if row["path"] != name:
                    continue
                best = row["sat_min_s"]
                log(f"| {row['batch']} | {row['ms_per_batch']:.3f} | {row['pos_per_s']:.0f} "
                    f"| {row['sat_ms_per_batch']:.3f} | {row['sat_us_per_pos']:.2f} "
                    f"| {row['sat_pos_per_s']:.0f} | {best * 1e3:.3f} "
                    f"| {row['batch'] / best:.0f} | {row['sat_sm_clock_mhz'] or 0:.0f} |")
            log("")

    if args.json_out:
        payload = {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if cuda else "cpu",
            "model": str(args.model),
            "positions": origin,
            "iters": args.iters,
            "rows": rows,
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
