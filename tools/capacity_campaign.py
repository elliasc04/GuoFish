#!/usr/bin/env python
"""Capacity campaign, Stage 0 — the analysis stages of docs/capacity/ANALYSIS_PLAN.md.

    python tools/capacity_campaign.py --self-check                  # CPU asserts, no GPU
    python tools/capacity_campaign.py --stages a1 a2 a3 --dry-run   # what would run
    python tools/capacity_campaign.py --stages a2 a5                # CPU only, seconds
    python tools/capacity_campaign.py --stages a5-val               # ~2 min GPU
    python tools/capacity_campaign.py --stages a1                   # ~1.5 h GPU
    python tools/capacity_campaign.py --stages a2-replay            # ~1.6 h GPU (reference)
    python tools/capacity_campaign.py --stages a3                   # ~16-25 h GPU

STAGES
  a1         Cost grid. Per measured shape, random weights: the Inductor +
             graph-captured forward's device time at batch 24 and 128
             (tools/bench_c12b.py), and a 200-step training smoke at micro-batch
             512 (samples/s, device-wide peak VRAM). Then cost = a(d)*L + b(d)
             over the 35-shape grid, the 5% depth-check rule, the 12 GB filter.
  a2         Log parsing, CPU only. Frozen copies of the post-2026-08-24 Lichess
             logs, the t distribution, N_ref(t) with delivered / inherited /
             ponder sims kept apart, and the 6 replay games.
  a2-replay  The replay bench: those 6 games re-run through a local engine with
             the deployed flags. `--replay-model/--replay-tag` replay a brief-pass
             arm later; each replay is compared move-for-move against the log
             (reference) or against the reference's replay (an arm).
  a3         The strength surface. New fixed-node cells in opening blocks with the
             ordo precision stop, the reused cells (verified, relabelled by the
             net they actually ran), the conditional rule, and one ordo fit per
             connected group of players.
  a5-val     The three endpoints scored on the 90M val set (the 10M re-evaluation).
  a5         Learning-curve arithmetic off the training logs, CPU only.

The match harness, logging, resumable state and telemetry parsing are
tools/capacity_suite.py's, imported rather than restated. Results land in
runs/capacity_campaign/results.json after every cell; re-running skips completed
cells, `--force-cell NAME` re-runs one. No stage plays Lichess games.

WHERE THIS DEPARTS FROM THE PLAN'S TEXT, because the text is wrong on disk:
  * T4's rungs were played by 20M ep9, not 90M ep4 (runs/capacity_suite/
    results.json, both commands). They enter the fit as 20M ep9's ladder; the
    90M ep4 ladder is only the two new rungs.
  * The reused 10M->20M 12k cell is 100 games (+155.5 +-54.4), not 200.
  * With 50-opening blocks a cell can only stop at 100 or 200 games, so §3's
    "~150 games" expectation needs `--a3-block-openings 25`. The default follows
    the plan's stated block size.
  * The cells form separate groups of players (no rung joins 2k, 12k and 50k),
    so ordo is run per group rather than as one scale.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capacity_suite as cs  # noqa: E402  (also puts the repo root on sys.path)

from telemetry.move_stats import git_facts, sha256_of  # noqa: E402

log = cs.log
REPO = cs.REPO_ROOT
OUT = REPO / "runs" / "capacity_campaign"
GAMES = REPO / "benchmarking" / "engine" / "games"
MODELS = REPO / "models"

# The three quality levels (§3 A3). Canonical files are the epoch-end
# checkpoints. The reused cells ran `best.pt` twins; a3 checks those carry the
# same weights, because the basename is shared across three directories (D-7).
NETS = {
    "10Mep9": MODELS / "guofish5_10M" / "v5_10.9M_ep9.pt",
    "20Mep9": MODELS / "guofish5_20M" / "v5_10.9M_ep9.pt",
    "90Mep4": MODELS / "guofish5_90M" / "v5_10.9M_ep4.pt",
}
REFERENCE_DEPLOYED = MODELS / "guofish5_90M" / "v5_10.9M_best.pt"   # what lichess-bot loads
SHARDS_90M = REPO / "data" / "processed" / "multipv_90m"

# ---------------------------------------------------------------------------
# A1 constants
# ---------------------------------------------------------------------------
A1_WIDTHS = (256, 320, 384, 448, 512, 576, 640)
A1_DEPTHS = (4, 6, 8, 10, 12)
A1_CHECK_WIDTHS = (256, 448, 640)
A1_CHECK_DEPTHS = (4, 8, 12)
A1_CONTROL = (384, 6)
A1_FIT_TOLERANCE = 0.05      # a depth-check shape further off the fit -> measure the grid
A1_VRAM_MARGIN = 0.10        # a FITTED shape within 10% of the device -> measure it
A1_DRIFT_TOLERANCE = 0.03
# WDDM pages to shared memory before `used` reaches `total`: d640x12 plateaued at
# 11,908 of 12,227 MiB (97%) and trained at 171 smp/s against a ~1,450 trend.
A1_SPILL_FRACTION = 0.95
A1_SMOKE_STEPS = 200
A1_RATE_AFTER_STEP = 50      # samples/s from steps past compile and loader spin-up
CORPUS90M_RECORDS = 90_076_117
CORPUS90M_COVERAGE = 0.60312  # measured at the 90M run's start; passing it skips a 20 s re-measure
TRAINER = REPO / "training" / "v5_multiPV" / "train_v5.py"
TRAIN_CONFIG = REPO / "training" / "v5_multiPV" / "configs" / "corpus90m.yaml"
FORWARD_BENCH = REPO / "tools" / "bench_c12b.py"

# ---------------------------------------------------------------------------
# A2 constants
# ---------------------------------------------------------------------------
LICHESS_LOG_DIR = REPO.parent / "lichess-bot" / "lichess_bot_auto_logs"
A2_LOGS = ("lichess-bot.log.2026-08-24", "lichess-bot.log.2026-08-25", "lichess-bot.log")
# config.yml's mtime (TimeManager on, SimCap 500k, PonderhitFloor 0.6), and an
# end bound so a bot that plays again later cannot grow the frozen set.
A2_WINDOW = ("2026-08-24 15:43", "2026-08-27")
A2_PLAN_COUNTS = (76, 3528)              # [measured] games, searched moves, in the plan
A2_REPLAY_GAMES = 6
# Pinned to the config the frozen games were played under, not read from
# config.yml at run time: the data is frozen, so the flags are too. The replay
# tool's own defaults are the pre-TM 200k config.
DEPLOYED_FLAGS = ("--sim-cap", "500000", "--ponder-max-sims", "500000",
                  "--ponder-decay", "1.0", "--ponderhit-floor", "0.6", "--time-manager")
REF_TAG = "ref"
PARSER = REPO / "telemetry" / "parse_lichess_log.py"
REPLAY_TOOL = REPO / "telemetry" / "replay_lichess_game.py"
A3_CEILING_ROOT = 50_000                 # 50k nominal ~ root visits at decision
A3_CEILING_DELIVERED = 28_000            # §3's 25-28k delivered, upper end

# ---------------------------------------------------------------------------
# A3 constants
# ---------------------------------------------------------------------------
A3_PRECISION = 40.0          # stop once the cell's ordo error is at or below this
A3_MAX_GAMES = 200
A3_CONCURRENCY = 2           # 2 GvG matches = the measured sweet spot (record §8.4)
A3_HOURS_PER_KSIMS = 0.085   # §3 A3's fit, per 200 games, both sides' nominal summed
# The PCIe fault killed a block's engines at startup on 2026-09-23 and cutechess
# waited 12 h. At 50k the longest gap between results was 7.9 min, so 30 min of
# silence is a hang; a block that stalls or plays nothing is retried.
A3_STALL_SECONDS = 1800
A3_BLOCK_ATTEMPTS = 3
ORDO = REPO / "ordo-win64.exe"


@dataclass(frozen=True)
class Cell:
    name: str
    a: tuple[str, int]        # (net, nominal sims): the lower side of the gap
    b: tuple[str, int]        # the higher side
    conditional: bool = False


# §3 A3, in the plan's cell order: the 2k cells first (cheap, and they feed the
# conditional), then the upper link at 50k, then 25k->50k, conditional last.
NEW_CELLS = (
    Cell("link_lo_2k", ("10Mep9", 2000), ("20Mep9", 2000)),
    Cell("link_up_2k", ("20Mep9", 2000), ("90Mep4", 2000)),
    Cell("rung_90M_2k_4k", ("90Mep4", 2000), ("90Mep4", 4000)),
    Cell("link_up_50k", ("20Mep9", 50000), ("90Mep4", 50000)),
    Cell("rung_90M_25k_50k", ("90Mep4", 25000), ("90Mep4", 50000)),
    Cell("link_lo_50k", ("10Mep9", 50000), ("20Mep9", 50000), conditional=True),
)


@dataclass(frozen=True)
class Reused:
    name: str
    pgn: Path
    a: tuple[str, str, int, Path]   # (pgn name, net, nominal, file the match loaded)
    b: tuple[str, str, int, Path]
    flags: tuple[str, ...]


_best = {k: MODELS / f"guofish5_{k}" / "v5_10.9M_best.pt" for k in ("10M", "20M", "90M")}
REUSED = (
    Reused("link_lo_12k", GAMES / "v6/head2head/10M_vs_20M/10M_vs_20M_12k.pgn",
           ("v5-10M", "10Mep9", 12000, _best["10M"]), ("v5-20M", "20Mep9", 12000, _best["20M"]),
           ("adjudication on", "pre-Inductor (2026-08-11)",
            "100 games: below the +-40 precision stop")),
    Reused("link_up_12k",
           GAMES / "v6/head2head/v5_10.9M_vs_guofish5_20M_12k/v5_10.9M_vs_guofish5_20M_12k.pgn",
           ("base-20M-ep9", "20Mep9", 12000, NETS["20Mep9"]),
           ("cand-v5_10.9M", "90Mep4", 12000, _best["90M"]),
           ("adjudication on", "pre-Inductor (2026-08-13)")),
    Reused("t4_20M_8k_16k", GAMES / "t4/8k_vs_16k/8k_vs_16k.pgn",
           ("v5-8k", "20Mep9", 8000, _best["20M"]), ("v5-16k", "20Mep9", 16000, _best["20M"]),
           ("adjudication on (same net both sides)", "pre-Inductor (2026-08-11)",
            "20M ep9's ladder, not the reference's")),
    Reused("t4_20M_16k_32k", GAMES / "t4/16k_vs_32k/16k_vs_32k.pgn",
           ("v5-16k", "20Mep9", 16000, _best["20M"]), ("v5-32k", "20Mep9", 32000, _best["20M"]),
           ("adjudication on (same net both sides)", "pre-Inductor (2026-08-11)",
            "20M ep9's ladder, not the reference's")),
)

# ---------------------------------------------------------------------------
# A5 constants
# ---------------------------------------------------------------------------
TRAIN_LOGS = {
    "10M": [MODELS / "guofish5_10M/logs/v5_20260803_022226.jsonl",
            MODELS / "guofish5_10M/logs/v5_20260803_045310.jsonl"],
    "20M": [MODELS / "guofish5_20M/logs/v5_20M.jsonl"],
    "90M": [MODELS / "guofish5_90M/logs/corpus90m_stitched.jsonl"],
}
ENDPOINT_STEPS = {"10Mep9": 87_894, "20Mep9": 175_779, "90Mep4": 351_860}
# The gate report's re-scores on the 90M val set; a5-val should reproduce them.
RECORDED_90M_VAL_KL = {"20Mep9": 0.8676954127780075, "90Mep4": 0.7901253986138356}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def quantiles(xs, ps=(0.1, 0.25, 0.5, 0.75, 0.9)) -> Optional[dict]:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return {f"p{round(p * 100)}": xs[min(len(xs) - 1, int(p * len(xs)))] for p in ps}


def player(net: str, nominal: int) -> str:
    return f"{net}_{cs._budget(nominal)}"


def player_nominal(pid: str) -> int:
    budget = pid.rsplit("_", 1)[1]
    return int(budget[:-1]) * 1000 if budget.endswith("k") else int(budget)


def weights_digest(path: Path) -> str:
    """sha256 over the state dict's tensors, so two files that differ only in
    metadata (`best.pt` vs `ep9.pt`) are recognisably the same net."""
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("model_state_dict", sd)
    h = hashlib.sha256()
    for key in sorted(sd):
        t = sd[key]
        if torch.is_tensor(t):
            h.update(key.encode())
            h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def provenance(state: cs.SuiteState, path: Path, *, weights: bool = False) -> dict:
    """Full path + sha256 (D-7), cached in results.json; the weights digest on request."""
    cache = state.data.setdefault("provenance", {})
    entry = cache.setdefault(str(path), {})
    sha = sha256_of(path)
    if entry.get("sha256") != sha:
        entry.clear()
        entry["sha256"] = sha
    if weights and "weights" not in entry:
        entry["weights"] = weights_digest(path)
    state.flush()
    return {"path": str(path), **entry}


def gpu_memory() -> tuple[int, int]:
    """(used, total) MiB on GPU 0, device-wide."""
    out = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used,memory.total",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=15).stdout
    used, total = (int(x) for x in out.strip().splitlines()[0].split(","))
    return used, total


class PeakVram:
    """Polls nvidia-smi while a child trains. Device-wide on purpose: the CUDA
    context and the allocator's cache are part of what "fits in 12 GB" means,
    and a WDDM spill to shared memory shows up as `used` pinned at `total`."""

    def __init__(self, every: float = 0.5):
        self.every = every
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            try:
                self.peak = max(self.peak, gpu_memory()[0])
            except Exception:                       # noqa: BLE001 - a missed sample is not a failure
                pass

    def __enter__(self) -> "PeakVram":
        self.baseline, self.total = gpu_memory()
        self.peak = self.baseline
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        self._thread.join(timeout=5)
        return False


def require_idle_gpu(allow: bool) -> None:
    """Refuse a GPU stage while another engine or trainer holds the GPU (§6).

    WDDM lists every desktop app as a compute app, so only python, cutechess and
    Stockfish entries count. The Lichess bot is a python process and shows up.
    """
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                          "--format=csv,noheader"], capture_output=True, text=True,
                         timeout=15).stdout
    busy = []
    for line in out.splitlines():
        pid, _, name = line.partition(",")
        pid = pid.strip()
        if re.search(r"python|cutechess|stockfish", name, re.I) and pid.isdigit() \
                and int(pid) != os.getpid():
            cmd = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                capture_output=True, text=True, timeout=30).stdout.strip()
            busy.append(f"pid {pid}: {cmd or name.strip()}")
    if busy:
        msg = "the GPU is not idle:\n    " + "\n    ".join(busy)
        if not allow:
            raise cs.Blocked(msg + "\n  Stop them (the Lichess bot included) or pass "
                                   "--allow-busy-gpu, and every timing is then suspect.")
        log(f"WARNING: {msg}")


# ---------------------------------------------------------------------------
# A1 — cost grid
# ---------------------------------------------------------------------------


def a1_name(d: int, layers: int) -> str:
    return f"d{d}x{layers}"


def a1_measured_set(full_grid: bool) -> list[tuple[int, int]]:
    """§3 A1: all 7 widths at 6 layers plus the 9-shape depth check, control first."""
    if full_grid:
        shapes = [(d, L) for d in A1_WIDTHS for L in A1_DEPTHS]
    else:
        shapes = [(d, 6) for d in A1_WIDTHS]
        shapes += [(d, L) for d in A1_CHECK_WIDTHS for L in A1_CHECK_DEPTHS]
    shapes.remove(A1_CONTROL)
    return [A1_CONTROL] + shapes


def a1_forward(checkpoint: Path, cell_dir: Path) -> dict:
    """Device time of the shipped forward (Inductor, graph-captured) per row."""
    out = cell_dir / "forward.json"
    code = cs.run_subprocess([sys.executable, FORWARD_BENCH, "--sections", "forward",
                              "--model", checkpoint, "--json-out", out],
                             cell_dir / "forward.log")
    if code or not out.exists():
        raise cs.Blocked(f"bench_c12b.py exited {code}; see {cell_dir / 'forward.log'}")
    rows = {r["shape"]: r for r in json.loads(out.read_text(encoding="utf-8"))["forward"]}
    return {f"us_per_row_{b}": rows[b]["inductor_us"] / b for b in (24, 128)}


def a1_training_smoke(d: int, layers: int, run_dir: Path) -> dict:
    """200 steps of the real trainer at micro-batch 512, torch.compile on.

    The trainer is stopped at its `--max-steps ... reached.` line: everything
    measured is on disk by then (the JSONL is line-buffered), and what follows
    is a full 452k-record validation plus three model+optimizer checkpoints.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", TRAINER, "--config", TRAIN_CONFIG,
           "--d-model", d, "--num-layers", layers, "--nhead", d // 64,
           "--dim-feedforward", 4 * d, "--max-steps", A1_SMOKE_STEPS,
           "--coverage", CORPUS90M_COVERAGE, "--out-dir", run_dir, "--run-name", "smoke",
           "--log-every", 10, "--val-every", 10 ** 9, "--ckpt-every", 10 ** 9,
           "--gap-probe", 0, "--no-gap-every-epoch", "--no-h2h-gate"]
    log_path = run_dir / "smoke.stdout.log"
    stopped = False
    log(f"    $ train_v5.py d{d}x{layers} --max-steps {A1_SMOKE_STEPS} -> {log_path.name}")
    with PeakVram() as vram, log_path.open("w", encoding="utf-8", newline="\n") as sink, \
            cs.Heartbeat(f"smoke d{d}x{layers}"):
        proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", cwd=str(REPO), bufsize=1)
        for line in proc.stdout:
            sink.write(line)
            if "--max-steps" in line and "reached" in line:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                               capture_output=True)
                stopped = True
                break
        code = proc.wait()
    for stray in run_dir.rglob("*.pt"):             # random-init weights, 200 steps in
        stray.unlink()

    oom = "out of memory" in log_path.read_text(encoding="utf-8", errors="replace").lower()
    if not stopped and not oom:
        raise cs.Blocked(f"training smoke d{d}x{layers} exited {code} before --max-steps; "
                         f"see {log_path}")
    steps = []
    jsonl = run_dir / "logs" / "smoke.jsonl"
    if jsonl.exists():
        with jsonl.open(encoding="utf-8") as fh:
            steps = [json.loads(l) for l in fh if '"event": "step"' in l]
    rates = [r["samples_per_s"] for r in steps
             if r["step"] > A1_RATE_AFTER_STEP and not r.get("warmup_window")]
    sps = statistics.median(rates) if rates else None
    return {"oom": oom, "exit_code": code, "stopped_at_max_steps": stopped,
            "samples_per_s": sps, "rate_points": len(rates),
            "hours_per_90M_epoch": CORPUS90M_RECORDS / sps / 3600 if sps else None,
            # The trainer's own footprint: the desktop's share (the baseline) is
            # the box's, not the shape's. Peak at the device total means WDDM
            # spilled to shared memory, which craters throughput rather than OOMs.
            "vram_footprint_mib": vram.peak - vram.baseline, "vram_peak_mib": vram.peak,
            "vram_baseline_mib": vram.baseline, "vram_total_mib": vram.total,
            "command": [str(c) for c in cmd]}


def a1_cell(d: int, layers: int, name: str, args, state: cs.SuiteState) -> dict:
    log(f"  {name}: d_model={d} x{layers} layers, {d // 64} heads, ff={4 * d}")
    started = time.monotonic()
    cand = cs.Candidate(a1_name(d, layers), d, layers, d // 64, 0.0, "a1")
    checkpoint = cs.write_random_checkpoint(cand, OUT / "checkpoints", args.seed)
    cell_dir = OUT / "a1" / name
    clock = cs.sm_clock_mhz()
    cell = {"status": "ok", "d": d, "L": layers, "checkpoint": str(checkpoint),
            "forward": a1_forward(checkpoint, cell_dir),
            "train": a1_training_smoke(d, layers, cell_dir / "train"),
            "sm_clock_mhz_before": clock, "sm_clock_mhz_after": cs.sm_clock_mhz()}
    cell["seconds"] = round(time.monotonic() - started, 1)
    f, t = cell["forward"], cell["train"]
    log(f"    forward {f['us_per_row_24']:.2f} us/row @24, {f['us_per_row_128']:.2f} @128 | "
        + ("train OOM at micro-batch 512" if t["oom"] else
           f"train {t['samples_per_s']:,.0f} smp/s ({t['hours_per_90M_epoch']:.2f} h/epoch), "
           f"footprint {t['vram_footprint_mib']:,} of {t['vram_total_mib']:,} MiB"
           + (" ** SPILLED: does not fit **" if spilled(t) else "")))
    state.put_cell("a1", name, cell)
    return cell


def spilled(train: dict) -> bool:
    """Did not train inside the device: OOM, or peak at the WDDM spill line."""
    return bool(train.get("oom")) or train["vram_peak_mib"] >= A1_SPILL_FRACTION * train["vram_total_mib"]


A1_METRICS = {
    "fwd24_us_per_row": lambda c: c["forward"]["us_per_row_24"],
    "fwd128_us_per_row": lambda c: c["forward"]["us_per_row_128"],
    # seconds per sample, not samples/s: time is what adds up layer by layer
    # A spilled shape's rate is the spill's, not the shape's: kept out of the fit.
    "train_s_per_sample": lambda c: (None if spilled(c["train"]) or not c["train"].get("samples_per_s")
                                     else 1.0 / c["train"]["samples_per_s"]),
    "vram_footprint_mib": lambda c: None if spilled(c["train"]) else c["train"]["vram_footprint_mib"],
}
A1_COST_METRICS = ("fwd24_us_per_row", "fwd128_us_per_row", "train_s_per_sample")


def fit_depth(points: dict) -> tuple[dict, float]:
    """cost = a(d)*L + b(d) (§3 A1). Least squares at the depth-check widths;
    b(d) interpolated between them, a(d) elsewhere from the L=6 point. Returns
    the prediction for every grid shape and the worst relative residual at the
    depth-check shapes."""
    import numpy as np
    ab = {}
    for d in A1_CHECK_WIDTHS:
        pts = sorted((L, v) for (dd, L), v in points.items() if dd == d and v is not None)
        if len(pts) >= 2:
            a, b = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
            ab[d] = (float(a), float(b))
    if len(ab) < 2:
        return {}, math.inf
    xs = sorted(ab)
    bs = [ab[d][1] for d in xs]
    pred = {}
    for d in A1_WIDTHS:
        if d in ab:
            a, b = ab[d]
        elif points.get((d, 6)) is not None:
            b = float(np.interp(d, xs, bs))
            a = (points[(d, 6)] - b) / 6
        else:
            continue
        for L in A1_DEPTHS:
            pred[(d, L)] = a * L + b
    resid = [abs(pred[k] / v - 1) for k, v in points.items()
             if k[0] in ab and v and k in pred]
    return pred, max(resid, default=math.inf)


def a1_fit(state: cs.SuiteState) -> dict:
    cells = {(c["d"], c["L"]): c for n, c in state.cells("a1").items()
             if c.get("status") == "ok" and not n.endswith("_final")}
    if A1_CONTROL not in cells:
        raise cs.Blocked("A1 fit needs the control cell d384x6")
    fits, preds = {}, {}
    for metric, read in A1_METRICS.items():
        points = {k: read(c) for k, c in cells.items()}
        preds[metric], worst = fit_depth(points)
        fits[metric] = {"worst_depth_check_residual": worst,
                        "ok": worst <= A1_FIT_TOLERANCE}
    control = {m: A1_METRICS[m](cells[A1_CONTROL]) for m in A1_METRICS}
    total = cells[A1_CONTROL]["train"]["vram_total_mib"]

    table = {}
    for d in A1_WIDTHS:
        for L in A1_DEPTHS:
            row = {"d": d, "L": L}
            for m, read in A1_METRICS.items():
                measured = read(cells[(d, L)]) if (d, L) in cells else None
                value = measured if measured is not None else preds[m].get((d, L))
                row[m] = value
                row[m + "_source"] = ("measured" if measured is not None else
                                      "fitted" if value is not None else "none")
            train = cells[(d, L)]["train"] if (d, L) in cells else {}
            if train and spilled(train):
                row["vram_footprint_mib_source"] = "measured: " + ("OOM" if train.get("oom") else "spill")
            for b in (24, 128):
                v = row[f"fwd{b}_us_per_row"]
                row[f"forward_cost_ratio_{b}"] = v / control[f"fwd{b}_us_per_row"] if v else None
            s = row["train_s_per_sample"]
            row["train_hours_per_90M_epoch"] = s * CORPUS90M_RECORDS / 3600 if s else None
            v = row["vram_footprint_mib"]
            # A fitted shape near the line is unknown, not "fits": the spill line
            # sits below the total, by however much the desktop is holding.
            near = row["vram_footprint_mib_source"] == "fitted" and v is not None \
                and v >= (1 - A1_VRAM_MARGIN) * total
            row["fits_12gb"] = (False if row["vram_footprint_mib_source"].startswith("measured: ")
                                else None if v is None or near else v < total)
            table[a1_name(d, L)] = row

    near_limit = [n for n, r in table.items() if r["vram_footprint_mib_source"] == "fitted"
                  and r["vram_footprint_mib"] is not None
                  and r["vram_footprint_mib"] >= (1 - A1_VRAM_MARGIN) * total]
    drift = None
    final = state.get_cell("a1", a1_name(*A1_CONTROL) + "_final")
    if final and final.get("status") == "ok":
        drift = {m: A1_METRICS[m](final) / control[m] - 1
                 for m in ("fwd24_us_per_row", "train_s_per_sample") if control[m]}
    fit_ok = all(fits[m]["ok"] for m in A1_COST_METRICS)
    return {"fits": fits, "fit_ok": fit_ok, "table": table, "near_vram_limit": near_limit,
            "device_total_mib": total, "drift": drift,
            "drift_flag": bool(drift) and any(abs(v) > A1_DRIFT_TOLERANCE for v in drift.values())}


def a1_log_table(result: dict) -> None:
    log("")
    log("| shape | fwd ratio @24 | @128 | train h/epoch | VRAM footprint MiB | fits | source |")
    log("|---|---:|---:|---:|---:|:-:|---|")
    for name, r in result["table"].items():
        def fmt(v, spec):
            return format(v, spec) if v is not None else "-"
        src = "measured" if r["fwd24_us_per_row_source"] == "measured" else "fitted"
        log(f"| {name} | {fmt(r['forward_cost_ratio_24'], '.2f')} "
            f"| {fmt(r['forward_cost_ratio_128'], '.2f')} "
            f"| {fmt(r['train_hours_per_90M_epoch'], '.2f')} "
            f"| {fmt(r['vram_footprint_mib'], ',.0f')} | {r['fits_12gb']} | {src} |")


def run_a1(args, state: cs.SuiteState) -> None:
    log("")
    log("=" * 78)
    log("A1 — cost grid (random weights; shape only)")
    log("=" * 78)
    shapes = a1_measured_set(args.a1_full_grid)
    queue = [(d, L, a1_name(d, L)) for d, L in shapes]
    queue.append((*A1_CONTROL, a1_name(*A1_CONTROL) + "_final"))     # drift: control last

    def measure(items):
        for i, (d, L, name) in enumerate(items, 1):
            if state.done("a1", name) and name not in args.force_cell:
                log(f"  [{i}/{len(items)}] {name}: cached")
                continue
            log(f"  [{i}/{len(items)}] {name}: starting")
            a1_cell(d, L, name, args, state)

    # The final control cell is measured after the whole set, including any
    # shapes the VRAM rule adds, so it brackets everything.
    measure(queue[:-1])
    result = a1_fit(state)
    if result["near_vram_limit"] and not args.a1_no_extend:
        log(f"  fitted within {A1_VRAM_MARGIN:.0%} of {result['device_total_mib']:,} MiB, "
            f"measuring: {', '.join(result['near_vram_limit'])}")
        extra = [(int(n[1:].split("x")[0]), int(n.split("x")[1]), n)
                 for n in result["near_vram_limit"]]
        measure(extra)
    measure(queue[-1:])
    result = a1_fit(state)
    state.data["a1"]["result"] = result
    state.flush()
    a1_log_table(result)
    for m, f in result["fits"].items():
        log(f"  fit {m}: worst depth-check residual {f['worst_depth_check_residual']:.1%}"
            + ("" if f["ok"] or m not in A1_COST_METRICS else "  ** OVER 5% **"))
    if not result["fit_ok"] and not args.a1_full_grid:
        log("  RULE: a depth-check shape misses the fit by more than 5%. Measure the full "
            "grid: --stages a1 --a1-full-grid (+~2.5 h).")
    if result["drift_flag"]:
        log(f"  DRIFT: control moved {result['drift']} between first and last cell "
            f"(>{A1_DRIFT_TOLERANCE:.0%}); flag the batch.")


# ---------------------------------------------------------------------------
# A2 — deployment workload, N_ref(t), replay bench
# ---------------------------------------------------------------------------


def epoch_utc(ts: str) -> float:
    """The parser's `go_ts`, read the way the replay tool reads a log stamp
    (local wall time taken as UTC), so the two meet on the same key."""
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()


def in_window(r: dict) -> bool:
    return A2_WINDOW[0] <= r["go_ts"] < A2_WINDOW[1]


def regime(r: dict) -> str:
    kind = r.get("go_kind") or r.get("resolved_by")
    return "ponderhit" if kind == "ponderhit" else "fresh"


def root_visits(r: dict) -> int:
    return (r.get("delivered") or 0) + (r.get("inherited") or 0)


def pick_replay_games(records: list[dict], k: int = A2_REPLAY_GAMES) -> list[dict]:
    """k whole games stratified by length: equal-count strata, the middle game
    of each. A game must sit in one log file and start at our first move."""
    by_game: dict[str, list[dict]] = {}
    for r in records:
        by_game.setdefault(r["game_id"], []).append(r)
    eligible = []
    for gid, rs in by_game.items():
        files = {r["log_file"] for r in rs}
        if len(files) == 1 and min(r["move_number"] for r in rs) == 1:
            eligible.append({"game_id": gid, "log_file": files.pop(), "our_moves": len(rs),
                             "searched": sum(r["source"] == "search" for r in rs)})
    eligible.sort(key=lambda g: (g["our_moves"], g["game_id"]))
    n = len(eligible)
    if n < k:
        raise cs.Blocked(f"only {n} whole games in the window; need {k}")
    return [eligible[(2 * i + 1) * n // (2 * k)] for i in range(k)]


def a2_summarize(window: list[dict]) -> dict:
    post = [r for r in window if r["source"] == "search"]
    tc = {}
    for gid in {r["game_id"] for r in window}:
        rs = sorted((r for r in window if r["game_id"] == gid), key=lambda r: r["move_number"])
        base = max((r["clock_before_ms"] or 0) for r in rs[:2])
        inc = rs[0]["go_winc_ms" if rs[0]["our_color"] == "white" else "go_binc_ms"] or 0
        tc[gid] = f"{round(base / 60000)}+{int(inc) // 1000}"

    def sims(rows):
        return {"delivered": quantiles(r["delivered"] for r in rows),
                "inherited": quantiles(r["inherited"] for r in rows),
                "ponder_sims": quantiles(r["ponder_sims"] for r in rows),
                "root_visits": quantiles(root_visits(r) for r in rows)}

    def t_s(rows):
        return quantiles(r["search_wall_ms"] / 1000 for r in rows if r["search_wall_ms"] is not None)

    groups = {"all": post, "ponderhit": [r for r in post if regime(r) == "ponderhit"],
              "fresh": [r for r in post if regime(r) == "fresh"]}
    edges = [0, 0.5, 1, 2, 4, 8, 16, 32, math.inf]
    n_ref = []
    for lo, hi in zip(edges, edges[1:]):
        rows = [r for r in post if r["search_wall_ms"] is not None
                and lo <= r["search_wall_ms"] / 1000 < hi]
        if rows:
            n_ref.append({"t_lo": lo, "t_hi": hi, "n": len(rows),
                          "ponderhit_share": sum(regime(r) == "ponderhit" for r in rows) / len(rows),
                          **{k: quantiles(v, (0.1, 0.5, 0.9)) for k, v in (
                              ("delivered", [r["delivered"] for r in rows]),
                              ("root_visits", [root_visits(r) for r in rows]))}})
    return {
        "games": len(tc), "searched_moves": len(post),
        "games_by_tc": dict(Counter(tc.values())),
        "moves_by_regime": {k: len(v) for k, v in groups.items()},
        "t_s": {"by_regime": {k: t_s(v) for k, v in groups.items()},
                "by_tc": {c: t_s([r for r in post if tc[r["game_id"]] == c])
                          for c in set(tc.values())}},
        "sims": {k: sims(v) for k, v in groups.items()},
        "n_ref_by_t": n_ref,
        "vs_a3_ceiling": {
            "share_root_visits_le_50k": sum(root_visits(r) <= A3_CEILING_ROOT for r in post) / len(post),
            "share_delivered_le_28k": sum((r["delivered"] or 0) <= A3_CEILING_DELIVERED
                                          for r in post) / len(post),
            "note": "S's x-axis unit (delivered vs root visits) is not yet pinned; both are reported"},
    }


def a2_records() -> list[dict]:
    path = OUT / "a2" / "moves_all.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def run_a2(args, state: cs.SuiteState) -> None:
    log("")
    log("=" * 78)
    log("A2 — deployment workload (frozen logs, CPU only)")
    log("=" * 78)
    a2 = OUT / "a2"
    frozen = a2 / "logs"
    frozen.mkdir(parents=True, exist_ok=True)
    sources = []
    for name in A2_LOGS:
        dst = frozen / name
        if not dst.exists():          # first copy wins: later bot sessions cannot move the data
            shutil.copy2(args.lichess_logs / name, dst)
        sources.append({"file": name, "sha256": sha256_of(dst), "bytes": dst.stat().st_size})
    code = cs.run_subprocess([sys.executable, PARSER, *[frozen / n for n in A2_LOGS],
                              "--out", a2 / "moves_all.jsonl",
                              "--games-out", a2 / "games_all.jsonl", "--quiet"],
                             a2 / "parse.log")
    if code:
        raise cs.Blocked(f"parse_lichess_log.py exited {code}; see {a2 / 'parse.log'}")
    window = [r for r in a2_records() if in_window(r)]
    summary = a2_summarize(window)
    games = pick_replay_games(window)
    state.data["a2"].update({"sources": sources, "window": A2_WINDOW,
                             "summary": summary, "replay_games": games})
    state.flush()
    (a2 / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    log(f"  {summary['games']} games, {summary['searched_moves']:,} searched moves "
        f"(plan: {A2_PLAN_COUNTS[0]} / {A2_PLAN_COUNTS[1]:,})"
        + ("" if (summary["games"], summary["searched_moves"]) == A2_PLAN_COUNTS
           else "  ** DIFFERS FROM THE PLAN **"))
    log(f"  time controls: {summary['games_by_tc']}; regimes: {summary['moves_by_regime']}")
    for k, q in summary["t_s"]["by_regime"].items():
        log(f"  t (s) {k:<9} {q}")
    for k, q in summary["sims"].items():
        log(f"  {k:<9} delivered p50 {q['delivered']['p50']:,}  root p50 {q['root_visits']['p50']:,}")
    c = summary["vs_a3_ceiling"]
    log(f"  inside A3's range: {c['share_root_visits_le_50k']:.1%} of moves by root visits, "
        f"{c['share_delivered_le_28k']:.1%} by delivered")
    log(f"  replay games: {', '.join(g['game_id'] + ' (' + str(g['our_moves']) + ')' for g in games)}")


def replay_rows(tag: str) -> dict[float, dict]:
    rows = {}
    for path in sorted((OUT / "a2" / "replay" / tag).glob("*.json")):
        for r in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            if r["kind"] == "move":
                rows[round(r["recorded_at"], 3)] = r
    return rows


def logged_rows(searched_only: bool = True) -> dict[float, dict]:
    return {round(epoch_utc(r["go_ts"]), 3): r for r in a2_records()
            if in_window(r) and (r["source"] == "search" or not searched_only)}


def a2_compare(tag: str, against: str) -> dict:
    """Per-move ratios of `tag`'s replay to `against` (the log, or another
    replay), paired on the recorded instant. Searched moves only."""
    a = replay_rows(tag)
    logged = logged_rows()
    b = logged if against == "log" else replay_rows(against)
    keys = sorted(a.keys() & b.keys() & logged.keys())

    def ratios(fn, rows):
        return quantiles([fn(x) / fn(y) for x, y in rows if fn(y)], (0.1, 0.5, 0.9))

    out = {"against": against, "paired": len(keys),
           "replay_moves_not_in_log": len(a.keys() - logged_rows(searched_only=False).keys())}
    for reg in ("all", "fresh", "ponderhit"):
        pairs = [(a[k], b[k]) for k in keys if reg == "all" or regime(logged[k]) == reg]
        out[reg] = {"n": len(pairs),
                    "delivered_ratio": ratios(lambda r: r["delivered"] or 0, pairs),
                    "root_visits_ratio": ratios(root_visits, pairs),
                    "search_wall_ratio": ratios(lambda r: r["search_wall_ms"] or 0, pairs)}
    return out


def run_a2_replay(args, state: cs.SuiteState) -> None:
    log("")
    log("=" * 78)
    log(f"A2 — replay bench, tag '{args.replay_tag}'")
    log("=" * 78)
    games = state.data["a2"].get("replay_games")
    if not games:
        raise cs.Blocked("no replay games selected; run --stages a2 first")
    rdir = OUT / "a2" / "replay" / args.replay_tag
    for i, g in enumerate(games, 1):
        out = rdir / f"{g['game_id']}.json"
        if out.exists() and json.loads(out.read_text(encoding="utf-8")).get("moves_replayed"):
            log(f"  [{i}/{len(games)}] {g['game_id']}: cached")
            continue
        log(f"  [{i}/{len(games)}] {g['game_id']}: {g['our_moves']} moves")
        code = cs.run_subprocess(
            [sys.executable, "-u", REPLAY_TOOL, "--log", OUT / "a2" / "logs" / g["log_file"],
             "--game", g["game_id"], "--last-moves", "0", "--model", args.replay_model,
             *DEPLOYED_FLAGS, "--out", out], rdir / f"{g['game_id']}.log")
        if code or not out.exists():
            raise cs.Blocked(f"replay of {g['game_id']} exited {code}; see {rdir}")
    against = "log" if args.replay_tag == REF_TAG else REF_TAG
    if against != "log" and not replay_rows(REF_TAG):
        raise cs.Blocked("replay the reference first (--replay-tag ref)")
    comparison = a2_compare(args.replay_tag, against)
    state.data["a2"].setdefault("replays", {})[args.replay_tag] = {
        "model": provenance(state, args.replay_model), "flags": list(DEPLOYED_FLAGS),
        "games": [g["game_id"] for g in games], "comparison": comparison}
    state.flush()
    log(f"  {comparison['paired']} moves paired against {against}")
    for reg in ("fresh", "ponderhit"):
        c = comparison[reg]
        log(f"  {reg:<9} n={c['n']:<4} delivered x{(c['delivered_ratio'] or {}).get('p50')} "
            f"root x{(c['root_visits_ratio'] or {}).get('p50')} "
            f"wall x{(c['search_wall_ratio'] or {}).get('p50')}")


# ---------------------------------------------------------------------------
# A3 — strength surface
# ---------------------------------------------------------------------------


def rename_players(text: str, names: dict[str, str]) -> str:
    """Rewrite the two player tags only; `WhiteElo` and friends are untouched."""
    return re.sub(r'^\[(White|Black) "([^"]*)"\]',
                  lambda m: f'[{m.group(1)} "{names.get(m.group(2), m.group(2))}"]',
                  text, flags=re.M)


def ordo_ratings(pgn: Path, anchor: str) -> dict[str, dict]:
    """ordo with the gate report's flags, `anchor` pinned at 0 (one estimator,
    record §7.1). ordo refuses some inputs (a sweep); that returns {}."""
    csv_out = pgn.with_suffix(".ordo.csv")
    csv_out.unlink(missing_ok=True)
    subprocess.run([str(ORDO), "-q", "-a", "0", "-A", anchor, "-D", "-W", "-s", "1000", "-J",
                    "-p", str(pgn), "-c", str(csv_out)], cwd=str(REPO),
                   capture_output=True, text=True, timeout=900)
    if not csv_out.exists() or not csv_out.stat().st_size:
        return {}

    def num(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return None

    out = {}
    with csv_out.open(encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            out[row["player"]] = {"elo": num(row.get("rating")), "error": num(row.get("error"))}
    return out


def components(edges: list[tuple[str, str]]) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[str, set[str]] = {}
    for x in list(parent):
        groups.setdefault(find(x), set()).add(x)
    return sorted(groups.values(), key=lambda g: sorted(g))


def conditional_fires(lo_2k, up_2k, lo_12k, up_12k) -> dict:
    """§3's rule for the lower link at 50k: run it only if the ratio of the two
    links' gaps at 2k and at 12k differ by more than their combined 95% CI.
    Each argument is (elo, ordo 95% error); delta method on each ratio."""
    def ratio(up, lo):
        (gu, eu), (gl, el) = up, lo
        if not gu or not gl:
            return None
        r = gu / gl
        return r, abs(r) * math.hypot(eu / 1.96 / gu, el / 1.96 / gl)

    r2, r12 = ratio(up_2k, lo_2k), ratio(up_12k, lo_12k)
    if r2 is None or r12 is None:
        return {"fires": False, "reason": "a link gap is zero; the ratio is undefined"}
    diff, ci = r2[0] - r12[0], 1.96 * math.hypot(r2[1], r12[1])
    return {"ratio_2k": r2[0], "ratio_12k": r12[0], "difference": diff, "ci95": ci,
            "fires": abs(diff) > ci}


def a3_block_command_args(args) -> SimpleNamespace:
    # §6: adjudication off (Q-denominated), a Q-independent -maxmoves cap instead.
    return SimpleNamespace(maxmoves=args.maxmoves, opening_plies=16, timemargin=300_000)


def a3_arms(cell: Cell) -> tuple[cs.EngineArm, cs.EngineArm]:
    # Arena at 75 nodes per sim of budget (§6; T4's first 32k run died short of it).
    return tuple(cs.EngineArm(player(net, n), NETS[net], nodes=n, arena_capacity=cs.arena_for(n))
                 for net, n in (cell.a, cell.b))


def a3_block_hours(cell: Cell, openings: int) -> float:
    per_200 = max(0.5, A3_HOURS_PER_KSIMS * (cell.a[1] + cell.b[1]) / 1000)
    return per_200 * (2 * openings) / 200


def run_a3_cell(cell: Cell, args, state: cs.SuiteState) -> dict:
    rec = state.get_cell("a3", cell.name) or {}
    if rec.get("status") == "ok" and cell.name not in args.force_cell:
        log(f"  {cell.name}: cached ({rec['ordo']['games']} games, "
            f"{rec['ordo']['elo']:+.1f} +-{rec['ordo']['error']})")
        return rec
    if cell.name in args.force_cell and rec.get("status") == "ok":
        rec = {}
    pa, pb = player(*cell.a), player(*cell.b)
    # Block k plays openings from (k-1)*size+1, so a size change mid-cell would
    # replay openings the cell already has.
    first = (rec.get("blocks") or {}).get("1")
    size = rec.setdefault("block_openings", int(first["command"][first["command"].index("-rounds") + 1])
                          if first else args.a3_block_openings)
    if rec.get("blocks") and size != args.a3_block_openings:
        raise cs.Blocked(f"{cell.name} was started with {size}-opening blocks; re-run it with "
                         f"--a3-block-openings {size}")
    blocks = rec.setdefault("blocks", {})
    cell_dir = GAMES / "capacity" / "a3" / cell.name
    max_blocks = math.ceil(A3_MAX_GAMES / (2 * args.a3_block_openings))
    rec.update(status="running", a=pa, b=pb,
               checkpoints={p: provenance(state, NETS[net], weights=True)
                            for p, net in ((pa, cell.a[0]), (pb, cell.b[0]))})
    for block in range(1, max_blocks + 1):
        if blocks.get(str(block), {}).get("status") != "ok":
            start = (block - 1) * args.a3_block_openings + 1
            log(f"  {cell.name} block {block}/{max_blocks}: openings {start}.."
                f"{start + args.a3_block_openings - 1}, ~{a3_block_hours(cell, args.a3_block_openings):.1f} h")
            for attempt in range(1, A3_BLOCK_ATTEMPTS + 1):
                res = cs.play(a3_arms(cell), cell_dir / f"b{block}", event=f"{cell.name}_b{block}",
                              rounds=args.a3_block_openings, concurrency=A3_CONCURRENCY,
                              adjudicate=False, sprt=None, args=a3_block_command_args(args),
                              opening_start=start, stall_seconds=A3_STALL_SECONDS)
                res["attempt"] = attempt
                blocks[str(block)] = res
                state.put_cell("a3", cell.name, rec)
                # Stalled or empty: engines lost, not a wrong result, so replay it.
                # "corrupted" (fallback moves) would repeat, so it stops the run.
                if res["status"] not in ("stalled", "failed") or attempt == A3_BLOCK_ATTEMPTS:
                    break
                log(f"    block {block} {res['status']} (attempt {attempt}/{A3_BLOCK_ATTEMPTS}); "
                    f"replaying it")
            if res["status"] != "ok":
                raise cs.Blocked(
                    f"A3 {cell.name} block {block} is '{res['status']}' "
                    f"(fallback moves: {res['fallback']['by_arm']}). Re-run to replay the "
                    f"block; its artifacts are moved aside, not deleted.")
        pgn = cell_dir / f"{cell.name}.pgn"
        pgn.write_text("\n".join(Path(blocks[str(i)]["pgn"]).read_text(encoding="utf-8")
                                 for i in range(1, block + 1)), encoding="utf-8")
        games = sum(blocks[str(i)]["result"]["games"] for i in range(1, block + 1))
        rating = ordo_ratings(pgn, pa).get(pb, {})
        rec["ordo"] = {"elo": rating.get("elo"), "error": rating.get("error"), "games": games}
        rec["pgn"] = str(pgn)
        log(f"    after {games} games: {pb} over {pa} {rating.get('elo')} +-{rating.get('error')}")
        if rating.get("error") is not None and rating["error"] <= A3_PRECISION:
            break      # reads only the error bar, never the point estimate (§3)

    delivered = {}
    for arm in (pa, pb):
        tel = [b["telemetry"][arm] for b in blocks.values()]
        moves = sum(t["moves"] for t in tel)
        delivered[arm] = {"moves": moves,
                          "delivered_mean": sum(t["delivered_total"] for t in tel) / moves if moves else None,
                          "inherited_mean": sum(t["inherited_mean"] * t["moves"] for t in tel) / moves
                          if moves else None,
                          "malformed_lines": sum(t["malformed"] for t in tel)}
    rec.update(status="ok", delivered=delivered)
    state.put_cell("a3", cell.name, rec)
    return rec


def a3_reused(state: cs.SuiteState) -> None:
    """Verify each reused cell ran the weights its label claims, then rate it."""
    for r in REUSED:
        if state.done("a3", r.name):
            continue
        command = r.pgn.with_name(r.pgn.stem + ".command.txt").read_text(encoding="utf-8")
        checks = {}
        for pgn_name, net, n, path in (r.a, r.b):
            if str(path) not in command:
                raise cs.Blocked(f"{r.name}: {path} is not in its command line")
            loaded = provenance(state, path, weights=True)
            canonical = provenance(state, NETS[net], weights=True)
            if loaded["weights"] != canonical["weights"]:
                raise cs.Blocked(f"{r.name}: {path} is not the same net as {NETS[net]}")
            checks[player(net, n)] = {"loaded": loaded, "same_weights_as": str(NETS[net])}
        names = {r.a[0]: player(r.a[1], r.a[2]), r.b[0]: player(r.b[1], r.b[2])}
        pgn = OUT / "a3" / f"{r.name}.pgn"
        pgn.parent.mkdir(parents=True, exist_ok=True)
        pgn.write_text(rename_players(r.pgn.read_text(encoding="utf-8"), names), encoding="utf-8")
        pa, pb = names[r.a[0]], names[r.b[0]]
        rating = ordo_ratings(pgn, pa).get(pb, {})
        games = pgn.read_text(encoding="utf-8").count("[Result ")
        state.put_cell("a3", r.name, {
            "status": "ok", "reused": True, "a": pa, "b": pb, "source_pgn": str(r.pgn),
            "pgn": str(pgn), "flags": list(r.flags), "checkpoints": checks,
            "ordo": {"elo": rating.get("elo"), "error": rating.get("error"), "games": games}})
        log(f"  {r.name} (reused): {pb} over {pa} {rating.get('elo')} +-{rating.get('error')} "
            f"over {games} games; {'; '.join(r.flags)}")


def a3_gap(state: cs.SuiteState, name: str) -> Optional[tuple[float, float]]:
    o = (state.get_cell("a3", name) or {}).get("ordo") or {}
    return (o["elo"], o["error"]) if o.get("elo") is not None and o.get("error") is not None else None


def a3_joint(state: cs.SuiteState) -> None:
    """One ordo fit per connected group of players. Groups share no games, so
    their ratings have no common zero and are not put on one scale."""
    cells = [(n, c) for n, c in state.cells("a3").items() if c.get("status") == "ok"]
    groups = components([(c["a"], c["b"]) for _, c in cells])
    out = []
    for i, group in enumerate(groups, 1):
        members = [(n, c) for n, c in cells if c["a"] in group]
        pgn = OUT / "a3" / f"group{i}.pgn"
        pgn.write_text("\n".join(Path(c["pgn"]).read_text(encoding="utf-8") for _, c in members),
                       encoding="utf-8")
        refs = [p for p in group if p.startswith("90Mep4")]
        anchor = max(refs, key=player_nominal) if refs else sorted(group)[0]
        out.append({"players": sorted(group, key=lambda p: (p.split("_")[0], player_nominal(p))),
                    "cells": [n for n, _ in members], "anchor": anchor,
                    "ratings": ordo_ratings(pgn, anchor), "pgn": str(pgn)})
        log(f"  group {i} (anchor {anchor} = 0): "
            + ", ".join(f"{p} {out[-1]['ratings'].get(p, {}).get('elo')}" for p in out[-1]["players"]))
    state.data["a3"]["joint"] = {"groups": out,
                                 "note": "separate groups share no games and no common zero"}
    state.flush()


def run_a3(args, state: cs.SuiteState) -> None:
    log("")
    log("=" * 78)
    log("A3 — strength surface")
    log("=" * 78)
    log("  T4's rungs are 20M ep9's ladder (see results.json of the capacity suite);")
    log("  they enter under that label. The 90M ep4 ladder is the two new rungs only.")
    a3_reused(state)
    wanted = set(args.a3_cells or [c.name for c in NEW_CELLS])
    for cell in NEW_CELLS:
        if not cell.conditional and cell.name in wanted:
            run_a3_cell(cell, args, state)

    gaps = [a3_gap(state, n) for n in ("link_lo_2k", "link_up_2k", "link_lo_12k", "link_up_12k")]
    if all(gaps) and state.done("a3", "link_up_50k"):
        decision = conditional_fires(*gaps)
        state.data["a3"]["conditional"] = decision
        state.flush()
        log(f"  conditional (lower link @50k): {decision}")
        cond = next(c for c in NEW_CELLS if c.conditional)
        if decision["fires"] and not args.a3_no_conditional and cond.name in wanted:
            run_a3_cell(cond, args, state)
    else:
        log("  conditional not decided: needs both 2k links and the upper link at 50k")
    a3_joint(state)


# ---------------------------------------------------------------------------
# A5 — learning curves
# ---------------------------------------------------------------------------


def run_a5_val(args, state: cs.SuiteState) -> None:
    log("")
    log("=" * 78)
    log("A5 — the three endpoints on the 90M val set")
    log("=" * 78)
    import torch
    from torch.amp.autocast_mode import autocast
    for p in (REPO / "training" / "v5_multiPV", REPO / "data" / "multiPV"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from dataset import MultiPVCollate, MultiPVDataset       # noqa: E402
    from gates import score_baseline                          # noqa: E402

    # As train_v5 was set when its gate re-scored 20M ep9 at 0.86770.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    val = MultiPVDataset(SHARDS_90M, split="val")
    collate = MultiPVCollate(mirror_prob=0.0, temperature=None, value_scale=None)

    def amp():
        return autocast(device_type="cuda", dtype=torch.bfloat16)

    for net, path in NETS.items():
        if state.done("a5", net) and net not in args.force_cell:
            log(f"  {net}: cached")
            continue
        started = time.monotonic()
        m = score_baseline(path, val, collate, torch.device("cuda"), amp,
                           batch_size=1024, workers=4, prefetch=2)
        cell = {"status": "ok", "checkpoint": provenance(state, path),
                "policy_kl": m["policy_kl"], "value_mse": m["value_mse"], "n": m["n"],
                "metrics": m, "seconds": round(time.monotonic() - started, 1)}
        if net in RECORDED_90M_VAL_KL:
            cell["delta_vs_record"] = m["policy_kl"] - RECORDED_90M_VAL_KL[net]
        state.put_cell("a5", net, cell)
        log(f"  {net}: KL {m['policy_kl']:.5f}  MSE {m['value_mse']:.5f}  "
            f"({m['n']:,} records, {cell['seconds']:.0f} s)"
            + (f"  vs record {cell['delta_vs_record']:+.5f}" if "delta_vs_record" in cell else ""))


def read_events(paths: list[Path], event: str) -> list[dict]:
    needle = f'"event": "{event}"'
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if needle in l]
    return rows


def run_a5(args, state: cs.SuiteState) -> None:
    import numpy as np
    log("")
    log("=" * 78)
    log("A5 — learning-curve arithmetic (CPU only)")
    log("=" * 78)
    out: dict = {}

    # 1. Endpoints on one val set. Confounded: steps, unique data and (for 90M)
    #    corpus and policy coverage all move together.
    ends = {n: state.get_cell("a5", n) for n in NETS}
    if all(c and c.get("status") == "ok" for c in ends.values()):
        ref = ends["90Mep4"]["policy_kl"]
        out["endpoints"] = {n: {"steps": ENDPOINT_STEPS[n], "policy_kl": c["policy_kl"],
                                "value_mse": c["value_mse"], "kl_over_90Mep4": c["policy_kl"] / ref}
                            for n, c in ends.items()}
        for n, e in out["endpoints"].items():
            log(f"  {n}: {e['steps']:,} steps  KL {e['policy_kl']:.5f}  ({e['kl_over_90Mep4']:.4f}x 90Mep4)")
    else:
        log("  endpoints: run --stages a5-val first")

    # 2. Epochs 5-6 of the 90M run, if each epoch's gain keeps shrinking by the
    #    last observed ratio. [estimated]: a longer run re-shapes OneCycle.
    kl = [e["policy_kl"] for e in sorted(read_events(TRAIN_LOGS["90M"], "epoch"),
                                         key=lambda e: e["t"])]
    d = [a - b for a, b in zip(kl, kl[1:])]
    r = d[-1] / d[-2]
    ep5 = kl[-1] - d[-1] * r
    ep6 = ep5 - d[-1] * r * r
    out["epochs_5_6_estimated"] = {"epoch_kl": kl, "shrink_ratio": r, "ep5": ep5, "ep6": ep6,
                                   "gain_rel_ep5": 1 - ep5 / kl[-1], "gain_rel_ep6": 1 - ep6 / kl[-1]}
    log(f"  90M epochs {[round(k, 5) for k in kl]} -> ep5 ~{ep5:.5f} ({1 - ep5 / kl[-1]:.2%}), "
        f"ep6 ~{ep6:.5f} ({1 - ep6 / kl[-1]:.2%}) [estimated]")

    # 3. Late-schedule val_quick jitter. The 20k subset is FIXED, so this is the
    #    checkpoint-to-checkpoint scatter about the trend, not the subset's
    #    sampling error against the full val set.
    out["val_quick_jitter"] = {}
    for run, paths in TRAIN_LOGS.items():
        pts = {e["step"]: e["val_policy_kl"] for e in read_events(paths, "val_quick")}
        steps = sorted(pts)
        late = [s for s in steps if s >= 0.75 * steps[-1]]
        y = np.array([pts[s] for s in late])
        x = np.array(late, dtype=float)
        resid = y - np.polyval(np.polyfit(x, y, 2), x)
        out["val_quick_jitter"][run] = {"points": len(late), "resid_sd": float(resid.std(ddof=3)),
                                        "resid_sd_rel": float(resid.std(ddof=3) / y.mean())}
        log(f"  val_quick {run}: {len(late)} late points, scatter about a quadratic "
            f"trend {resid.std(ddof=3):.5f} ({resid.std(ddof=3) / y.mean():.3%})")
    state.data["a5"]["curves"] = out
    state.flush()


# ---------------------------------------------------------------------------
# A4 — break-even frontier and candidate set (arithmetic on A1-A3, A5)
# ---------------------------------------------------------------------------
# Every modelling choice below is stated in results.json next to the numbers it
# produced; none is in the plan's text, and the operator confirms the list.
A4_GPU_SHARE = 0.908         # [assumed] T1 gate4, d384x6 control, fresh root, K=24
T3_FLOOR_KL = 0.32395        # T3 pairwise label disagreement (upper bound on the floor)
A4_COST_BINS = (0.0, 0.85, 1.15, 1.6, 2.3, math.inf)
A4_MAX_ARMS = 3              # §7's "control + 3 arms"
A4_PAIR_TOLERANCE = 0.05     # "two shapes at equal cost with different width/depth"
REFERENCE_SHAPE = (384, 6)
A4_ASSUMPTIONS = [
    "x-axis = root visits at decision (nominal in a fixed-node match; delivered + "
    "inherited in deployment). Delivered-sims mapping reported as a sensitivity.",
    "Quality exchange k(N) = upper-link gap (20M ep9 -> 90M ep4) / its relative KL "
    "drop, at 2k/12k/50k nominal; the reference slope s(N) = 90M ep4's rung Elo per "
    "nominal doubling at each rung's geometric midpoint. Both piecewise-linear in "
    "log2 N and FLAT beyond the measured ends [assumed]: ~98% of deployment moves "
    "sit above 50k root visits, so the break-even is mostly the top points.",
    f"A shape's sims at equal time scale by 1 / (f*r + 1 - f), r = A1 forward cost "
    f"ratio at batch 24, f = {A4_GPU_SHARE} GPU-bound share [assumed, T1 random "
    f"weights]; applied to delivered and ponder-inherited sims alike.",
    "Required quality per deployment move = doublings lost x s(midpoint) / k(shape's N); "
    "the shape's figure is the median over A2's 3,528 moves, IQR alongside.",
    "Ranking: required relative KL per training-hour (A1 hours per 90M epoch), "
    "ascending; best per cost bin; only shapes that fit 12 GB.",
]


def a4_curves(state: cs.SuiteState) -> dict:
    """k(N) and s(N) from A3, KL from A5. Blocked if any input is missing."""
    missing = [n for n in ("link_up_2k", "link_up_12k", "link_up_50k",
                           "rung_90M_2k_4k", "rung_90M_25k_50k") if not a3_gap(state, n)]
    missing += [f"a5:{n}" for n in NETS if not state.done("a5", n)]
    if missing:
        raise cs.Blocked(f"A4 needs completed inputs; missing: {', '.join(missing)}")
    kl = {n: state.get_cell("a5", n)["policy_kl"] for n in NETS}
    rel_up = 100 * (1 - kl["90Mep4"] / kl["20Mep9"])          # % relative KL
    k = [(n, *(v / rel_up for v in a3_gap(state, name)))
         for name, n in (("link_up_2k", 2000), ("link_up_12k", 12000), ("link_up_50k", 50000))]
    s = [(math.sqrt(lo * hi), *a3_gap(state, name))
         for name, lo, hi in (("rung_90M_2k_4k", 2000, 4000), ("rung_90M_25k_50k", 25000, 50000))]
    top = state.get_cell("a3", "link_up_50k")["delivered"]
    rho = statistics.fmean(v["delivered_mean"] for v in top.values()) / 50000
    return {"kl": kl, "rel_kl_up_pct": rel_up, "k": k, "s": s, "delivered_over_nominal_50k": rho}


def interp_log2(points: list, x: float) -> float:
    """Piecewise-linear in log2(x) through (x, y, ...) points, flat beyond the ends."""
    import numpy as np
    pts = sorted(points)
    return float(np.interp(math.log2(x), [math.log2(p[0]) for p in pts], [p[1] for p in pts]))


def required_gain(cost_ratio: float, xs: list[float], k: list, s: list) -> list[float]:
    """% relative KL a shape must gain to match the reference, per deployment move."""
    c = A4_GPU_SHARE * cost_ratio + (1 - A4_GPU_SHARE)
    lost = math.log2(c)                      # doublings of search; negative = gained
    return [lost * interp_log2(s, x / math.sqrt(c)) / interp_log2(k, x / c) for x in xs]


def run_a4(args, state: cs.SuiteState) -> None:
    log("")
    log("=" * 78)
    log("A4 — break-even frontier and candidate set (CPU only)")
    log("=" * 78)
    cur = a4_curves(state)
    a1 = state.data["a1"].get("result")
    if not a1:
        raise cs.Blocked("A4 needs the A1 table; run --stages a1")
    moves = [r for r in a2_records() if in_window(r) and r["source"] == "search"]
    x_root = [root_visits(r) for r in moves]
    x_deliv = [max(1, r["delivered"] or 0) / cur["delivered_over_nominal_50k"] for r in moves]
    ponder = [regime(r) == "ponderhit" for r in moves]
    (s_n, s_top, s_err), (k_n, k_top, k_err) = cur["s"][-1], cur["k"][-1]
    log(f"  k(N) Elo per 1% KL: " + ", ".join(f"{n // 1000}k {v:.1f}+-{e:.1f}" for n, v, e in cur["k"]))
    log(f"  s(N) Elo per doubling: " + ", ".join(f"~{n / 1000:.1f}k {v:.1f}+-{e:.1f}" for n, v, e in cur["s"]))

    rows = {}
    for name, r in a1["table"].items():
        cost = r["forward_cost_ratio_24"]
        if cost is None:
            continue
        q = required_gain(cost, x_root, cur["k"], cur["s"])
        qs = quantiles(q, (0.25, 0.5, 0.75))
        hours = r["train_hours_per_90M_epoch"]
        rel_se = math.hypot(s_err / 1.96 / s_top, k_err / 1.96 / k_top)
        rows[name] = {
            "d": r["d"], "L": r["L"], "forward_cost_ratio": cost,
            "doublings_lost": math.log2(A4_GPU_SHARE * cost + 1 - A4_GPU_SHARE),
            "required_rel_kl_pct": qs, "required_ci95_pct": 1.96 * abs(qs["p50"]) * rel_se,
            "required_by_regime_p50": {
                "ponderhit": statistics.median(v for v, p in zip(q, ponder) if p),
                "fresh": statistics.median(v for v, p in zip(q, ponder) if not p)},
            "required_p50_delivered_mapping": statistics.median(
                required_gain(cost, x_deliv, cur["k"], cur["s"])),
            "required_kl_vs_4ep_reference": cur["kl"]["90Mep4"] * (1 - qs["p50"] / 100),
            "hours_per_epoch": hours, "fits_12gb": r["fits_12gb"],
            "score_pct_per_hour": qs["p50"] / hours if hours else None}
        rows[name]["achievable"] = rows[name]["required_kl_vs_4ep_reference"] > T3_FLOOR_KL

    eligible = [n for n, r in rows.items() if r["fits_12gb"] is True and r["achievable"]
                and (r["d"], r["L"]) != REFERENCE_SHAPE and r["score_pct_per_hour"] is not None]
    bins = []
    for lo, hi in zip(A4_COST_BINS, A4_COST_BINS[1:]):
        members = sorted((n for n in eligible if lo <= rows[n]["forward_cost_ratio"] < hi),
                         key=lambda n: rows[n]["score_pct_per_hour"])
        if not members:
            continue
        best = rows[members[0]]
        pairs = [n for n in members[1:]
                 if abs(rows[n]["forward_cost_ratio"] / best["forward_cost_ratio"] - 1) <= A4_PAIR_TOLERANCE
                 and rows[n]["L"] != best["L"]]
        bins.append({"cost_bin": [lo, hi], "best": members[0], "equal_cost_pairs": pairs,
                     "ranked": members})
    proposed = sorted(bins, key=lambda b: rows[b["best"]]["score_pct_per_hour"])[:A4_MAX_ARMS]
    state.data["a4"] = {"assumptions": A4_ASSUMPTIONS, "curves": cur, "shapes": rows,
                        "bins": bins, "status": "proposed — the operator confirms (§6 roles)",
                        "proposed": [{"name": b["best"], "pairs": b["equal_cost_pairs"]}
                                     for b in proposed]}
    state.flush()

    log("")
    log("| shape | cost | doublings | required %KL p50 [IQR] | +-95% | delivered-map | h/epoch | %KL/h | fits |")
    log("|---|---:|---:|---|---:|---:|---:|---:|:-:|")
    for n, r in sorted(rows.items(), key=lambda kv: kv[1]["forward_cost_ratio"]):
        q = r["required_rel_kl_pct"]
        log(f"| {n} | {r['forward_cost_ratio']:.2f} | {r['doublings_lost']:+.2f} "
            f"| {q['p50']:+.2f} [{q['p25']:+.2f}, {q['p75']:+.2f}] | {r['required_ci95_pct']:.2f} "
            f"| {r['required_p50_delivered_mapping']:+.2f} | {r['hours_per_epoch'] or 0:.2f} "
            f"| {r['score_pct_per_hour'] if r['score_pct_per_hour'] is None else round(r['score_pct_per_hour'], 3)} "
            f"| {r['fits_12gb']} |")
    for b in bins:
        log(f"  cost {b['cost_bin'][0]:.2f}-{b['cost_bin'][1]:.2f}: best {b['best']}"
            + (f", equal-cost pair {', '.join(b['equal_cost_pairs'])}" if b["equal_cost_pairs"] else ""))
    log(f"  PROPOSED ARMS (operator confirms): "
        + ", ".join(p["name"] + (f" (+{', '.join(p['pairs'])})" if p["pairs"] else "")
                    for p in state.data["a4"]["proposed"]))


# ---------------------------------------------------------------------------
# Brief pass (§4): LR range tests, then 1-epoch OneCycle arms on the 90M corpus
# ---------------------------------------------------------------------------
BRIEF_MODELS = MODELS / "capacity_brief"
# The operator's pre-authorized rule (2026-09-23, before A4 had run): train the
# top-ranked shape in each of these A4 cost bins, cheapest first.
AUTHORIZED_BANDS = ((0.85, 1.15), (1.15, 1.6), (1.6, 2.3))


def resolve_arms(args, state: cs.SuiteState) -> list[str]:
    """`--arms`, plus A4's per-band picks when `--arms-from-a4 bands` was given."""
    arms = list(args.arms)
    if args.arms_from_a4 == "bands":
        a4 = state.data.get("a4") or {}
        if not a4.get("bins"):
            raise cs.Blocked("--arms-from-a4 needs A4's result; run --stages a4 first")
        picks = [b["best"] for b in a4["bins"] if tuple(b["cost_bin"]) in AUTHORIZED_BANDS]
        a4["authorized"] = {"rule": "top-ranked shape per cost band " + str(AUTHORIZED_BANDS),
                            "arms": picks, "by": "operator, 2026-09-23, before A4 ran"}
        state.flush()
        arms += [a for a in picks if a not in arms]
    return arms


def arm_shape(arm: str) -> tuple[int, int]:
    if arm == "control":
        return REFERENCE_SHAPE
    m = re.fullmatch(r"d(\d+)x(\d+)", arm)
    if not m:
        raise cs.Blocked(f"arm {arm!r} is neither 'control' nor dNxL")
    return int(m.group(1)), int(m.group(2))


def trainer_cmd(arm: str, extra: list) -> list:
    """Every arm: the corpus90m config (same permutation, seed, effective batch
    1024, dropout, schedule shape), its own shape, and the reference run's
    measured coverage so no arm re-measures it (§6: only max_lr varies)."""
    d, layers = arm_shape(arm)
    return [sys.executable, "-u", TRAINER, "--config", TRAIN_CONFIG,
            "--d-model", d, "--num-layers", layers, "--nhead", d // 64,
            "--dim-feedforward", 4 * d, "--coverage", CORPUS90M_COVERAGE,
            "--no-h2h-gate", *extra]


# 2026-09-24: on the 90M corpus every arm's 500-step range test is flat to within
# 1% from ~1e-4 to ~2e-3, so its minimum is batch noise (the reference shape's
# own test returned 1.6e-3, 4.6x its validated 3.5e-4). 3.5e-4 sits inside every
# arm's within-1% band, so `--lr-rule common` gives every arm that value.
COMMON_MAX_LR = 3.5e-4
BRIEF_TRAIN_ATTEMPTS = 3


def floor_2sf(x: float) -> float:
    """The reference's rule: 3.567e-4 at the range test's loss minimum -> 3.5e-4."""
    unit = 10 ** (math.floor(math.log10(x)) - 1)
    return math.floor(x / unit + 1e-9) * unit


def last_event(path: Path, event: str) -> Optional[dict]:
    rows = read_events([path], event) if path.exists() else []
    return rows[-1] if rows else None


def run_brief_lr(args, state: cs.SuiteState) -> None:
    log("")
    log("=" * 78)
    log("Brief pass — LR range tests (control + A4's proposed arms)")
    log("=" * 78)
    arms = ["control"] + [p["name"] for p in state.data.get("a4", {}).get("proposed", [])]
    arms += [a for a in resolve_arms(args, state) if a not in arms]
    for arm in arms:
        name = f"lr_{arm}"
        if state.done("brief", name) and name not in args.force_cell:
            log(f"  {arm}: cached, max_lr {state.get_cell('brief', name)['max_lr']:.2e}")
            continue
        run_dir = OUT / "brief" / arm / "lr"
        code = cs.run_subprocess(trainer_cmd(arm, ["--lr-range-test", "--out-dir", run_dir,
                                                   "--run-name", "lr"]), run_dir / "lr.stdout.log")
        res = last_event(run_dir / "logs" / "lr.jsonl", "lr_range_result")
        if code or not res:
            raise cs.Blocked(f"LR range test for {arm} exited {code}; see {run_dir}")
        cell = {"status": "ok", "arm": arm, "result": res, "max_lr": floor_2sf(res["lr_at_min_loss"]),
                "rule": "floor to 2 s.f. of the smoothed-loss minimum (reproduces 3.5e-4)"}
        if res.get("flat_curve"):
            cell["status"] = "flagged"
            cell["reason"] = "flat curve: the sweep did not separate learning rates"
        state.put_cell("brief", name, cell)
        log(f"  {arm}: loss minimum at {res['lr_at_min_loss']:.3e} -> max_lr {cell['max_lr']:.2e}"
            + (f"  ** {cell['reason']} **" if cell["status"] != "ok" else ""))


def run_brief_train(args, state: cs.SuiteState) -> None:
    log("")
    log("=" * 78)
    arms = resolve_arms(args, state)
    log(f"Brief pass — 1-epoch OneCycle arms: {', '.join(arms)}")
    log("=" * 78)
    for arm in arms:
        if state.done("brief", arm) and arm not in args.force_cell:
            log(f"  {arm}: cached (KL {state.get_cell('brief', arm)['policy_kl']:.5f})")
            continue
        lr_cell = state.get_cell("brief", f"lr_{arm}")
        if args.lr_rule == "common":
            max_lr = COMMON_MAX_LR
        elif lr_cell and lr_cell.get("status") == "ok":
            max_lr = lr_cell["max_lr"]
        else:
            raise cs.Blocked(f"{arm} has no usable LR range test; run --stages brief-lr --arms {arm}")
        out = BRIEF_MODELS / arm
        extra = ["--epochs", 1, "--max-lr", max_lr, "--out-dir", out, "--run-name", arm]
        # A crash costs at most the steps since the last checkpoint: d384x10 hit a
        # transient CUDA OOM at step 24,850 on 2026-09-24 and, with no retry, the
        # queue sat idle for ~10 h. Each attempt's output goes to its own file.
        for attempt in range(1, BRIEF_TRAIN_ATTEMPTS + 1):
            if list(out.glob("*_ep1.pt")):
                break
            resume = ["--resume", "latest"] if (list(out.glob("*_step*.pt"))
                                                or list(out.glob("*_last.pt"))) else []
            code = cs.run_subprocess(trainer_cmd(arm, extra + resume), out / f"train.stdout.{attempt}.log",
                                     label=f"{arm} (attempt {attempt})")
            if code and attempt == BRIEF_TRAIN_ATTEMPTS:
                raise cs.Blocked(f"training {arm} failed {attempt} times; see {out}/train.stdout.*.log")
            if code:
                log(f"    {arm} exited {code}; resuming from its latest checkpoint")
        ep1 = next(iter(out.glob("*_ep1.pt")), None)
        metrics = last_event(out / "logs" / f"{arm}.jsonl", "epoch")
        if not ep1 or not metrics:
            raise cs.Blocked(f"{arm}: no epoch-1 checkpoint or evaluation in {out}")
        state.put_cell("brief", arm, {
            "status": "ok", "shape": arm_shape(arm), "max_lr": max_lr, "lr_rule": args.lr_rule,
            "checkpoint": provenance(state, ep1), "policy_kl": metrics["policy_kl"],
            "value_mse": metrics["value_mse"], "command": [str(c) for c in trainer_cmd(arm, extra)]})
        log(f"  {arm}: 1 epoch -> KL {metrics['policy_kl']:.5f}, MSE {metrics['value_mse']:.5f}")


def run_brief_replay(args, state: cs.SuiteState) -> None:
    """§4's measured N(t): each trained candidate replays A2's games, paired
    against the reference's replay. The control shares the reference's shape."""
    for arm in resolve_arms(args, state):
        cell = state.get_cell("brief", arm)
        if arm == "control" or not cell or cell.get("status") != "ok":
            continue
        if arm in state.data["a2"].get("replays", {}) and arm not in args.force_cell:
            log(f"  replay {arm}: cached")
            continue
        run_a2_replay(SimpleNamespace(**{**vars(args), "replay_model": Path(cell["checkpoint"]["path"]),
                                         "replay_tag": arm}), state)


def wait_for_pid(pid: int) -> None:
    """Block until another process exits (Windows), without polling or signals."""
    import ctypes
    k32 = ctypes.windll.kernel32
    handle = k32.OpenProcess(0x00100000, False, pid)         # SYNCHRONIZE
    if handle:
        k32.WaitForSingleObject(handle, 0xFFFFFFFF)
        k32.CloseHandle(handle)


# ---------------------------------------------------------------------------
# Preflight, dry run, self-check
# ---------------------------------------------------------------------------


def stage_inputs(stage: str, args) -> list[Path]:
    if stage == "a1":
        return [TRAINER, TRAIN_CONFIG, FORWARD_BENCH, SHARDS_90M]
    if stage == "a2":
        return [PARSER, *[args.lichess_logs / n for n in A2_LOGS
                          if not (OUT / "a2" / "logs" / n).exists()]]
    if stage == "a2-replay":
        return [REPLAY_TOOL, args.replay_model]
    if stage == "a3":
        return [cs.CUTECHESS, ORDO, cs.OPENINGS, *NETS.values(),
                *[p for r in REUSED for p in (r.pgn, r.a[3], r.b[3])]]
    if stage == "a5-val":
        return [SHARDS_90M, *NETS.values()]
    if stage in ("brief-lr", "brief-train", "brief-replay"):
        return [TRAINER, TRAIN_CONFIG, SHARDS_90M]
    if stage == "a4":
        return []
    return [p for paths in TRAIN_LOGS.values() for p in paths]


def dry_run(args, state: cs.SuiteState) -> None:
    for stage in args.stages:
        log("")
        log(f"--- {stage} (dry run) ---")
        if stage == "a1":
            shapes = a1_measured_set(args.a1_full_grid)
            log(f"  {len(shapes)} shapes, control first and again last: "
                + ", ".join(a1_name(*s) for s in shapes))
            log(f"  per shape: bench_c12b.py --sections forward (batch 24, 128), then "
                f"train_v5.py --max-steps {A1_SMOKE_STEPS} with nvidia-smi peak VRAM")
            log(f"  ~{(len(shapes) + 1) * 5 / 60:.1f} h at ~5 min per shape")
        elif stage == "a2":
            for n in A2_LOGS:
                frozen, p = OUT / "a2" / "logs" / n, args.lichess_logs / n
                log(f"  already frozen: {frozen}" if frozen.exists() else
                    f"  freeze {p} ({p.stat().st_size / 2**20:.1f} MiB) -> {frozen.parent}")
            log(f"  window {A2_WINDOW}; expect {A2_PLAN_COUNTS[0]} games / "
                f"{A2_PLAN_COUNTS[1]:,} searched moves; pick {A2_REPLAY_GAMES} replay games")
        elif stage == "a2-replay":
            games = state.data.get("a2", {}).get("replay_games")
            log(f"  model {args.replay_model} as tag '{args.replay_tag}', flags {' '.join(DEPLOYED_FLAGS)}")
            log(f"  games: {[g['game_id'] for g in games] if games else 'not selected yet (run a2)'}")
            log(f"  ~{A2_REPLAY_GAMES * 16 / 60:.1f} h (median game 15.8 min wall)")
        elif stage == "a3":
            for r in REUSED:
                log(f"  reuse {r.name}: {r.pgn.relative_to(REPO)} [{'; '.join(r.flags)}]")
            total = 0.0
            for c in NEW_CELLS:
                blocks = math.ceil(A3_MAX_GAMES / (2 * args.a3_block_openings))
                one = a3_block_hours(c, args.a3_block_openings)
                total += 0 if c.conditional else one * blocks
                log(f"  {'(conditional) ' if c.conditional else ''}{c.name}: "
                    f"{player(*c.a)} vs {player(*c.b)}, {one:.1f} h/block, up to {blocks} blocks")
            log(f"  up to ~{total:.1f} h before the conditional cell")
            first = NEW_CELLS[0]
            cmd = cs.build_match_command(
                a3_arms(first), GAMES / "capacity" / "a3" / first.name / "b1",
                rounds=args.a3_block_openings, concurrency=A3_CONCURRENCY,
                event=f"{first.name}_b1", adjudicate=False, sprt=None,
                maxmoves=args.maxmoves, opening_plies=16, timemargin=300_000)
            log(f"  first block: {' '.join(cmd)}")
        elif stage == "a5-val":
            log(f"  score {', '.join(str(p.relative_to(REPO)) for p in NETS.values())} "
                f"on {SHARDS_90M.relative_to(REPO)} val (452,405 records)")
        elif stage == "a5":
            log("  read " + ", ".join(p.name for ps in TRAIN_LOGS.values() for p in ps))
        elif stage == "a4":
            log("  arithmetic on A1-A3 + A5; assumptions recorded with the numbers:")
            for a in A4_ASSUMPTIONS:
                log(f"   - {a}")
        elif stage == "brief-lr":
            proposed = [p["name"] for p in state.data.get("a4", {}).get("proposed", [])]
            log(f"  500-step range test for control + {proposed or 'A4 proposals (after a4)'}"
                f" + {args.arms}; max_lr = floor to 2 s.f. of the loss minimum; ~5 min each")
        elif stage == "brief-replay":
            log(f"  each trained candidate replays the {A2_REPLAY_GAMES} A2 games with its epoch-1 "
                f"checkpoint (~1.6 h each), paired against the reference's replay")
        elif stage == "brief-train":
            table = state.data.get("a1", {}).get("result", {}).get("table", {})
            for arm in args.arms:
                d, L = arm_shape(arm)
                h = table.get(a1_name(d, L), {}).get("train_hours_per_90M_epoch")
                log(f"  {arm} ({a1_name(d, L)}): 1 epoch of 90M, OneCycle, -> "
                    f"{BRIEF_MODELS / arm}, ~{h or 0:.1f} h")


def self_check() -> int:
    """CPU-only asserts on the logic that decides things. No GPU, no games."""
    # A1: the fit is exact on data of its own form, and the 5% rule fires on a bend.
    lin = {(d, L): 0.5 * d * L + 3 * d for d in A1_WIDTHS for L in A1_DEPTHS}
    measured = {k: lin[k] for k in a1_measured_set(False)}
    pred, worst = fit_depth(measured)
    assert worst < 1e-9 and all(abs(pred[k] - lin[k]) < 1e-6 for k in lin), "A1 fit"
    bent = dict(measured)
    bent[(448, 12)] *= 1.2
    assert fit_depth(bent)[1] > A1_FIT_TOLERANCE, "A1 5% rule"
    assert len(a1_measured_set(False)) == 16 and a1_measured_set(False)[0] == A1_CONTROL

    # A3: the conditional rule, and the cell graph's groups.
    assert not conditional_fires((100, 40), (120, 40), (150, 50), (180, 40))["fires"]
    assert conditional_fires((50, 5), (200, 5), (150, 5), (150, 5))["fires"]
    edges = [(player(*c.a), player(*c.b)) for c in NEW_CELLS]
    edges += [(player(r.a[1], r.a[2]), player(r.b[1], r.b[2])) for r in REUSED]
    assert len(components(edges)) == 4, components(edges)
    assert player("90Mep4", 25000) == "90Mep4_25k" and player_nominal("90Mep4_25k") == 25000
    txt = '[White "v5-10M"]\n[Black "v5-20M"]\n[WhiteElo "v5-10M"]\n'
    assert rename_players(txt, {"v5-10M": "10Mep9_12k"}) == \
        '[White "10Mep9_12k"]\n[Black "v5-20M"]\n[WhiteElo "v5-10M"]\n'
    arms = a3_arms(NEW_CELLS[0])
    cmd = cs.build_match_command(arms, Path("x"), rounds=50, concurrency=2, event="e",
                                 adjudicate=False, sprt=None, maxmoves=200,
                                 opening_plies=16, timemargin=300_000, opening_start=51)
    assert cmd[cmd.index("plies=16") + 1] == "start=51" and "-resign" not in cmd
    assert cmd.count(f"tscale={cs.ENGINE_TSCALE}") == 2, "D-5: every engine gets tscale"
    assert "start=1" not in cs.build_match_command(
        arms, Path("x"), rounds=50, concurrency=2, event="e", adjudicate=False, sprt=None,
        maxmoves=200, opening_plies=16, timemargin=300_000)
    assert arms[0].arena_capacity == 75 * 2000

    # A2: stratified picks, whole-game replay, and the pairing key.
    recs = [{"game_id": f"g{i:02d}", "log_file": "x", "move_number": m, "source": "search"}
            for i in range(30) for m in range(1, 10 + i)]
    picks = pick_replay_games(recs)
    assert len({p["game_id"] for p in picks}) == 6
    assert [p["our_moves"] for p in picks] == sorted(p["our_moves"] for p in picks)
    from telemetry.replay_lichess_game import _stamp, plan_sends
    transcript = []
    for i in range(3):
        transcript += [{"t": i, "dir": "<<", "text": ("position startpos moves " + "e2e4 " * i).strip()},
                       {"t": i + .1, "dir": "<<", "text": "go wtime 1000 btime 1000"},
                       {"t": i + .5, "dir": ">>", "text": "bestmove e2e4"}]
    sends, _ = plan_sends(transcript, 0)
    assert sum(s["text"].startswith("go") for s in sends) == 3, "whole-game replay"
    assert epoch_utc("2026-08-24 15:43:09.199000") == _stamp("2026-08-24 15:43:09,199")
    assert regime({"go_kind": "ponderhit"}) == regime({"resolved_by": "ponderhit"}) == "ponderhit"

    # The stall watchdog kills a silent match and says so.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        code, _ = cs.run_match([sys.executable, "-c", "import time; time.sleep(60)"],
                               Path(tmp), "stall", stall_seconds=2)
        assert code == cs.STALLED, code
    # A4: the reference needs nothing; costlier shapes need more, cheaper less;
    # interpolation is flat past the ends; the LR rule reproduces the reference.
    k = [(2000, 20.0), (50000, 24.0)]
    s = [(2828, 180.0), (35355, 150.0)]
    assert required_gain(1.0, [1e5], k, s) == [0.0]
    assert required_gain(2.0, [1e5], k, s)[0] > 0 > required_gain(0.7, [1e5], k, s)[0]
    assert interp_log2(k, 1e6) == 24.0 and interp_log2(k, 100) == 20.0
    assert abs(interp_log2(k, 10000) - (20 + 4 * math.log2(5) / math.log2(25))) < 1e-9
    assert abs(floor_2sf(3.5670845979562155e-04) - 3.5e-4) < 1e-12
    assert arm_shape("control") == REFERENCE_SHAPE and arm_shape("d448x8") == (448, 8)
    print("self-check PASS")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STAGES = {"a1": run_a1, "a2": run_a2, "a2-replay": run_a2_replay, "a3": run_a3,
          "a5-val": run_a5_val, "a5": run_a5, "a4": run_a4,
          "brief-lr": run_brief_lr, "brief-train": run_brief_train,
          "brief-replay": run_brief_replay}
GPU_STAGES = {"a1", "a2-replay", "a3", "a5-val", "brief-lr", "brief-train", "brief-replay"}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", nargs="+", choices=list(STAGES), default=[],
                   help="run in the order given; GPU stages never overlap")
    p.add_argument("--dry-run", action="store_true", help="say what would run; touch nothing")
    p.add_argument("--self-check", action="store_true", help="CPU asserts, then exit")
    p.add_argument("--force-cell", nargs="*", default=[], help="re-run these cells")
    p.add_argument("--allow-busy-gpu", action="store_true")
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--after-pid", type=int, default=None,
                   help="wait for this process (a running campaign) to exit first; "
                        "results.json is read only after it has")
    g = p.add_argument_group("A1")
    g.add_argument("--a1-full-grid", action="store_true",
                   help="measure all 35 shapes (the plan's response to a failed fit)")
    g.add_argument("--a1-no-extend", action="store_true",
                   help="do not measure fitted shapes near the VRAM limit")
    g = p.add_argument_group("A2")
    g.add_argument("--lichess-logs", type=Path, default=LICHESS_LOG_DIR)
    g.add_argument("--replay-model", type=Path, default=REFERENCE_DEPLOYED)
    g.add_argument("--replay-tag", default=REF_TAG,
                   help="'ref' is compared against the log; any other tag against 'ref'")
    g = p.add_argument_group("A3")
    g.add_argument("--a3-cells", nargs="*", choices=[c.name for c in NEW_CELLS],
                   help="only these new cells (default: all, conditional by its rule)")
    g.add_argument("--a3-block-openings", type=int, default=50,
                   help="openings per block; 25 lets a cell stop at 150 games")
    g.add_argument("--a3-no-conditional", action="store_true",
                   help="decide the conditional cell but do not play it")
    g.add_argument("--maxmoves", type=int, default=200)
    g = p.add_argument_group("brief pass")
    g.add_argument("--arms", nargs="+", default=["control"],
                   help="arms to train ('control' or dNxL). Candidates only after "
                        "the operator confirms A4's list (§6 roles)")
    g.add_argument("--lr-rule", choices=["range-test", "common"], default="range-test",
                   help=f"'common' trains every arm at {COMMON_MAX_LR:g} (see COMMON_MAX_LR)")
    g.add_argument("--arms-from-a4", choices=["bands"], default=None,
                   help="add A4's top shape per cost band (operator pre-authorized "
                        "2026-09-23)")
    args = p.parse_args(argv)
    if args.after_pid:
        log(f"waiting for pid {args.after_pid} to exit before {' '.join(args.stages)}")
        wait_for_pid(args.after_pid)
        log(f"pid {args.after_pid} has exited")

    if args.self_check:
        return self_check()
    if not args.stages:
        p.print_help()
        return 2
    missing = [str(x) for s in args.stages for x in stage_inputs(s, args) if not Path(x).exists()]
    if missing:
        log("missing inputs:\n  " + "\n  ".join(missing))
        return 1

    state = cs.SuiteState(OUT / "results.json")
    for key in ("a1", "a2", "a3", "a4", "a5", "brief"):
        state.data.setdefault(key, {})
    if args.dry_run:
        dry_run(args, state)
        return 0

    cs.open_run_log(OUT / "run.log")
    try:
        if GPU_STAGES & set(args.stages):
            build = cs.require_release_build()
            require_idle_gpu(args.allow_busy_gpu)
            state.data["meta"]["build"] = build
        state.data["meta"].update({
            "last_session_utc": cs.utcnow(), "stages": args.stages, "git": git_facts(REPO),
            "cutechess_sha256": sha256_of(cs.CUTECHESS), "ordo_sha256": sha256_of(ORDO),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
        state.flush()
        for stage in args.stages:
            started = time.monotonic()
            STAGES[stage](args, state)
            log(f"  {stage} done in {cs._hms(time.monotonic() - started)}")
        log(f"results: {state.path}")
        return 0
    except cs.Blocked as exc:
        log(f"BLOCKED: {exc}")
        return 1
    except KeyboardInterrupt:
        log("INTERRUPTED. Completed cells are in results.json and are skipped on re-run.")
        return 130
    finally:
        cs.close_run_log()


if __name__ == "__main__":
    raise SystemExit(main())
