#!/usr/bin/env python
"""T1-T4 — the capacity decision suite. One resumable driver, four tests.

    python tools/capacity_suite.py                       # the whole suite, in order
    python tools/capacity_suite.py --tests t3            # one test
    python tools/capacity_suite.py --tests t1 --force t1 # re-run a completed test
    python tools/capacity_suite.py --tests summary       # re-emit the summary only

WHAT THIS DECIDES
=================
The 20M ep9 seen/unseen gap is ~2%, so the model fits fresh data as well as its
own training set: it is capacity-limited, not data-limited. The C++ port then
made the GPU the bottleneck (80-96% GPU share across the C10b grid, 87% at the
shipping point), so capacity now costs throughput and throughput is ELO. This
prices that trade without training a throwaway model:

    cost_elo(candidate) = doublings_lost(T1) x elo_per_doubling(T4)
    benefit_elo         = T2a equal-sims margin        # a LOWER bound; see T2
    verdict(candidate)  = benefit_elo > cost_elo(candidate)

with T3 as a precondition: if the corpus noise floor already sits at the model's
own loss, no capacity candidate is worth training and the lever becomes label
quality instead.

IT TRAINS NOTHING. It writes random-weight checkpoints (T1 measures shape, not
weights), PGNs, logs, one results JSON and one markdown summary. Nothing under
golden/ or tests/ is touched (Global Rule 1 and 2), and cpp/ is not modified.

WHAT THE BRIEF GOT WRONG, VERIFIED AGAINST THE TREE AT C11b
===========================================================
Three of the brief's environment facts predate C11b (commit fad5956) and are
corrected here rather than carried forward, because two of them would have
changed the harness:

  1. "Policy temperature cannot be varied. SearchConfig has no temperature
     field." FALSE since C11b. `EngineConfig.policy_temperature` is a real
     SearchConfig field, `--policy-temperature` is a real flag, and the C++
     `gather_softmax_canonical` divides logits by it at the root and at every
     interior node. What IS still refused is Dirichlet noise
     (`playv6.UNSUPPORTED_IN_CORE`). This suite sweeps NEITHER — that is a scope
     decision (non-goal 3), not an engine limitation, and the summary says so
     rather than repeating the wrong reason.

  2. "uci_wrapper_v6.py is mid-update for book, Syzygy and temperature." It has
     landed. The flags this harness uses are read off the current signature:
     `--model`, `--threads`, `--max-outstanding`, `--max-batch`, `--affinity`,
     `--sim-cap`, `--c-puct-init`, `--fpu-tree`, `--no-book`, `--no-syzygy`.
     The shipping config (W=1, K=24, max_batch 128, affinity none) IS the
     dataclass default, so an arm that wants it passes nothing.

  3. "Verify the capture ladder is built from the loaded model's shape rather
     than hardcoded to the 10.9M one." It never was hardcoded: the ladder is a
     set of BATCH SIZES (`graphs.DEFAULT_CAPTURE_SIZES`, resolved against
     `max_batch`), and `GraphedForward` captures whatever module it is handed.
     Measured on both nets: 730 MiB reserved at 10.9M, 816 MiB at 25.6M, same
     nine shapes. The VRAM scales with d_model; the ladder does not need to.

  T2 IS NOT BLOCKED. v4 (`models/guofish4/guofish4_25.6M_policy_final.pt`)
  loads on the v6 engine as ChessTransformerV2 — 25.60M params, RELU FFN, no
  final_norm, all three reconstructed from the checkpoint rather than guessed —
  captures its own ladder and searches through the C++ core at 7,203 delivered
  sims/s against v5's 13,900 on the same box. `--check` re-runs that
  verification. v4 carries no `value_scale`, so it resolves to
  LEGACY_VALUE_SCALE 290.6806, which is the same constant v5's checkpoint
  carries — the two report `score cp` on one scale.

WHAT IS BLOCKED, AND STAYS BLOCKED
==================================
The T1 "Smolgen variant" row cannot be measured. There is no Smolgen in this
tree: `ModelConfig.__post_init__` raises NotImplementedError on `smolgen=True`
and `training/v5_multiPV/model_v5.py` carries the flag as a placeholder only.
Measuring its latency would mean implementing an architecture, which is a
training-side change this suite is explicitly not allowed to make. The cell is
recorded as `blocked` with that reason and the summary says so, rather than
being quietly dropped from the candidate table.

THE ENVIRONMENT CONSTRAINTS THAT SHAPE THE SCHEDULE
===================================================
SERIAL IS MANDATORY. GPU share is 80-96% across the C10b grid, so any two GPU
tests running together corrupt both. T3 is CPU/IO-bound and still does not
overlap T1, because T1's dispatcher timing is sensitive to CPU contention
(C10b-3f: pinning is a 9% LOSS at W=1 precisely because the dispatcher competes
for a core). The driver runs one thing at a time and never forks a second
GPU job.

CONCURRENCY NO LONGER BUYS THROUGHPUT. The pipeline ceiling in C10b-3b is
1.05-1.21x. `-concurrency 2` for fixed-node arms, `-concurrency 1` for anything
timed, and each concurrent process captures its OWN ladder (730 MiB at 10.9M,
816 MiB at 25.6M), so concurrency multiplies VRAM.

`--workers 32` IS DEAD. C10b-3g ships W=1, K=24, max_batch 128, affinity none.
W=1 is also deterministic (0.0% run-to-run TV), which is what makes every match
here reproducible from the same openings; the T1 cells assert it rather than
hoping.

DELIVERED != NOMINAL. Tree reuse makes `search_parallel` target absolute root
visits, so `go nodes N` can deliver fewer than N. T4's x-axis is DELIVERED sims
parsed from the engine's own `[search] ... delivered=` stderr line, never the
`nodes=` value. Timed searches report `nominal=n/a` by design and none is
synthesised.

BOOK AND SYZYGY ARE OFF FOR ALL FOUR TESTS, and that is recorded in the results
file per arm. Both default ON since C11b and both BYPASS MCTS: a book move would
nullify `-openings`, and Syzygy changes endgame cache-hit rates, which are what
drive T1's reuse-regime numbers. Every match arm passes `--no-book --no-syzygy`
and every arm's verdict is read back out of the ENGINE'S OWN stderr through
`bench_provenance`, not out of this harness's arguments.

THERMAL DRIFT over a ~24 h serial run is a real confound (C10b logged 2857-2917
MHz). SM clock is recorded on every cell and the T1 control candidate is
re-measured as the LAST action of the suite; a shift of more than
`--thermal-tolerance` flags every T1 number as suspect instead of reporting
drift as capacity cost.

RESUME
======
Everything is checkpointed to `<out-dir>/results.json` after every cell. A run
that dies in T2 leaves T1, T3 and T4 complete and readable. Re-running skips
completed cells; `--force t1 t4` re-runs whole tests, `--force-cell d512x6`
re-runs one.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# `tools/` is not a package; the siblings are imported by path, which is the
# convention tools/smoke_c11.py already established.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_provenance  # noqa: E402

# ---------------------------------------------------------------------------
# Paths and fixed quantities
# ---------------------------------------------------------------------------

CUTECHESS = REPO_ROOT / "cutechess-1.4.0-win64" / "cutechess-cli.exe"
WRAPPER = "playing/uci_wrapper_v6.py"          # repo-relative; cutechess sets dir=
OPENINGS = REPO_ROOT / "assets" / "8moves_v3.pgn"
EVAL_DUMP = REPO_ROOT / "data" / "multiPV" / "lichess_db_eval.jsonl.zst"
CORPUS_MANIFEST = REPO_ROOT / "data" / "multiPV" / "manifests" / "dataset_manifest_90m.json"

V5_MODEL = REPO_ROOT / "models" / "guofish5_20M" / "v5_10.9M_best.pt"
V4_MODEL = REPO_ROOT / "models" / "guofish4" / "guofish4_25.6M_policy_final.pt"

DEFAULT_OUT = REPO_ROOT / "runs" / "capacity_suite"
T2_PGN_DIR = REPO_ROOT / "benchmarking" / "engine" / "games" / "t2"
T4_PGN_DIR = REPO_ROOT / "benchmarking" / "engine" / "games" / "t4"

KNEE_BENCH = REPO_ROOT / "tools" / "bench_c10b_knee.py"
GRID_BENCH = REPO_ROOT / "tools" / "bench_c10b.py"

# C10b-3g's selection. Restated as constants because every arm in this suite is
# denominated in it and a drifting default would silently re-price the whole
# comparison. `EngineConfig` carries the same numbers as its defaults; these are
# what the harness ASSERTS the arms ran at.
SHIP_WORKERS = 1
SHIP_IN_FLIGHT = 24
SHIP_MAX_BATCH = 128
SHIP_AFFINITY = "none"

# Arena nodes to provision per simulation of BUDGET. Set manually per arm until
# the pondering work makes this scale off `--sims` on the engine side.
#
# WHY THIS EXISTS. The first T4 run sized every arm at the 1,200,000 default.
# That is 37.5 nodes per sim at a 32,000 budget, and the tree needs about 50 —
# so the 32k arm exhausted its arena mid-search, `go` raised, and the wrapper's
# exception handler answered with the first legal move in the position. 647
# moves across 133 of 150 games were played that way, which inverted the sign of
# `elo_per_doubling` (-462.7) and with it every candidate verdict.
#
# WHERE 75 COMES FROM, AND WHY NOT MORE. The arena cost of a search is
# `root_visits x mean legal moves per expanded node`, and the honest denominator
# is root visits, not `delivered` — `nodes` counts the inherited subtree too, so
# dividing by the new sims alone reports 45,508 nodes/sim on a root that
# inherited 31,977 of its 32,000 visits. Measured per root visit, the three arms
# that never hit their cap peak at 48.05, 48.68 and 50.53. The 32k arm's 37.50 is
# NOT a fourth measurement: it aborted at the cap, so its distribution is
# right-censored at exactly 1,200,000/32,000 and cannot report a ceiling.
#
# 75 is the headroom the 16k arm demonstrably ran on (1,200,000/16,000) across
# 300 games, which makes this "give 32k what 16k already had" rather than a new
# guess. It is deliberately not larger: `compact_and_promote` does two
# `assign(arena_capacity, ...)` passes on EVERY `apply_move`, so capacity is an
# O(capacity) per-move cost and not just a ceiling — 4M would memset ~32 MB per
# move to buy margin over a ceiling that three uncensored arms put near 50.
ARENA_NODES_PER_SIM = 75


def arena_for(budget: int) -> int:
    """The arena a fixed-node arm at `budget` sims is given.

    A function rather than a literal per arm so the two T4 rungs cannot drift
    apart: an arena that differed between rungs for any reason other than their
    budgets would be a difference between the things being compared.
    """
    return budget * ARENA_NODES_PER_SIM

# Gate 4 (chunk table, C12), restated from tools/bench_c10b.py so the T1 verdict
# and the C10b tables read against the same bar.
GATE4_FLOOR = 8_000
GATE4_STRETCH = 15_000

# T2 fairness, §7. v5's middle-stratum output is ~7% narrower than v4's (ratio
# 1.073), so v4 is given proportionally wider Q-denominated constants. Both nets
# remain untuned on this engine, but at least symmetrically so. PROVENANCE: both
# measured middle-stratum on the same val split; the all-record figures are NOT
# interchangeable with these and must not be substituted.
Q_RATIO_V4_OVER_V5 = 1.073
Q_RATIO_PROVENANCE = ("middle-stratum output width, both nets on the same val "
                      "split; all-record figures are not interchangeable")
V5_C_PUCT_INIT, V5_FPU_TREE = 1.43, 0.30
V4_C_PUCT_INIT = round(V5_C_PUCT_INIT * Q_RATIO_V4_OVER_V5, 3)      # 1.534
V4_FPU_TREE = round(V5_FPU_TREE * Q_RATIO_V4_OVER_V5, 3)            # 0.322

# T3's thresholds. `model_kl / pairwise_kl` above GATE means real reducible
# signal remains and capacity is live; below NEAR means the model is already at
# the corpus ceiling. Between the two is neither, and is reported as neither.
T3_LIVE_RATIO = 2.0
T3_CEILING_RATIO = 1.25

# The run-to-run total-variation tolerance C9 deferred and C10b closed. W=1 is
# deterministic, so this is a guard against a cell that was not the cell it
# claimed to be, not a real tolerance.
TV_TOLERANCE = 0.10


@dataclass(frozen=True)
class Candidate:
    """One T1 shape. `macs_ratio_brief` is what §5's table asserts; the harness
    computes its own from the shape and prints both, so a table that disagrees
    with arithmetic says so instead of being believed."""

    name: str
    d_model: int
    layers: int
    nhead: int
    macs_ratio_brief: float
    axis: str
    blocked: str = ""

    @property
    def dim_feedforward(self) -> int:
        return 4 * self.d_model


# §5's table. head_dim is 64 throughout, so nhead = d_model / 64; ff = 4x.
CANDIDATES: tuple[Candidate, ...] = (
    Candidate("d384x6_control", 384, 6, 6, 1.00, "control"),
    Candidate("d448x6", 448, 6, 7, 1.36, "width"),
    Candidate("d512x6", 512, 6, 8, 1.78, "width"),
    Candidate("d576x6", 576, 6, 9, 2.25, "width"),
    Candidate("d384x8", 384, 8, 6, 1.33, "depth"),
    Candidate("d512x8_v4shape", 512, 8, 8, 2.37, "v4 shape"),
    Candidate("smolgen", 0, 0, 0, 0.0, "smolgen",
              blocked="Smolgen is not implemented in this tree. "
                      "training/v5_multiPV/model_v5.py carries `smolgen` as a "
                      "placeholder and ModelConfig.__post_init__ raises "
                      "NotImplementedError on smolgen=True. Measuring its "
                      "latency would require implementing the architecture, "
                      "which is a training-side change outside this suite's "
                      "scope (non-goal 1)."),
)

CONTROL = CANDIDATES[0]

SEQ_LENGTH = 68


def macs_per_position(d_model: int, layers: int, seq: int = SEQ_LENGTH) -> float:
    """A shape-only MAC estimate for one forward, used to check §5's ratios.

    Per layer, per token: attention projections 4*d^2, attention scores and
    values 2*d*seq, FFN 2*d*ff with ff = 4d. Embedding, heads and norms are
    left out — they are the same across candidates and would only dilute a
    ratio. This is a RATIO instrument, not an absolute FLOP count.
    """
    ff = 4 * d_model
    per_token = 4 * d_model ** 2 + 2 * d_model * seq + 2 * d_model * ff
    return float(layers * seq * per_token)


# ---------------------------------------------------------------------------
# Logging and state
# ---------------------------------------------------------------------------

_LOG_SINK: list[str] = []
_LOG_FILE = None                  # set by open_run_log; None until out-dir exists
# How many `_LOG_SINK` entries have already reached the file. `open_run_log`
# backfills from HERE rather than from zero, so reopening streams only what the
# file has not seen; backfilling the whole sink would write every earlier line a
# second time.
_LOG_STREAMED = 0
_STARTED = time.monotonic()

# How often a Heartbeat says a child process is still alive. Set from
# --heartbeat in main(); 0 disables. A module global rather than a parameter
# threaded through play() -> run_match -> Heartbeat, because it is a property of
# the RUN, not of any one cell, and the plumbing would touch four signatures to
# say one thing.
#
# The default is a SEPARATE constant so that `main()` can name it in the
# argparse default without that counting as a use of the mutable global before
# its `global` declaration, which Python rejects at compile time.
DEFAULT_HEARTBEAT_SECONDS = 300.0
HEARTBEAT_SECONDS = DEFAULT_HEARTBEAT_SECONDS


def _hms(seconds: float) -> str:
    """H:MM:SS at a FIXED WIDTH, so the log's columns do not shift at 10 hours.

    `{h:2d}` rather than `{h:d}`: every line carries this in its prefix, and a
    prefix that grows by one character part-way through the night would misalign
    every table printed after it against every table printed before.
    """
    hours, rest = divmod(int(max(0.0, seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:2d}:{minutes:02d}:{secs:02d}"


def log(msg: str = "") -> None:
    """One line to stdout and, once `open_run_log` has run, to disk as well.

    STREAMED AND FLUSHED PER LINE. The previous implementation accumulated
    `_LOG_SINK` and wrote `run.log` once, at the end of `main()` — so the file a
    reader would want to `tail` did not exist until the thing they wanted to
    watch had already finished, and a run killed by Ctrl-C, a reboot or an
    unhandled exception left no log at all. A 24 h serial suite is precisely the
    case where that matters, so the file is opened early and every line hits it
    immediately.

    Each line carries wall-clock time and elapsed time: `[02:14:07  6:41:22]`.
    Wall clock answers "when did this happen", elapsed answers "how long has it
    been going", and overnight you need both. Blank lines stay blank so the
    existing section spacing still reads.
    """
    global _LOG_STREAMED
    line = (f"[{datetime.now().strftime('%H:%M:%S')} "
            f"{_hms(time.monotonic() - _STARTED)}] {msg}") if msg else ""
    print(line, flush=True)
    _LOG_SINK.append(line)
    if _LOG_FILE is not None:
        _LOG_FILE.write(line + "\n")
        _LOG_FILE.flush()
        _LOG_STREAMED = len(_LOG_SINK)


def open_run_log(path: Path) -> None:
    """Begin streaming to `path`, backfilling whatever was logged before it.

    APPEND, not truncate. Resuming is this suite's normal mode — a run that dies
    in T2 is re-invoked and skips the completed cells — and truncating would
    destroy the record of the very run whose failure the reader is trying to
    understand. Each session writes a banner so the boundaries stay visible.
    """
    global _LOG_FILE, _LOG_STREAMED
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = path.open("a", encoding="utf-8", newline="\n")
    _LOG_FILE.write(f"\n{'=' * 78}\n=== session {utcnow()}\n{'=' * 78}\n")
    backlog = _LOG_SINK[_LOG_STREAMED:]
    if backlog:
        _LOG_FILE.write("\n".join(backlog) + "\n")
    _LOG_STREAMED = len(_LOG_SINK)
    _LOG_FILE.flush()


def close_run_log() -> None:
    global _LOG_FILE
    if _LOG_FILE is not None:
        _LOG_FILE.flush()
        _LOG_FILE.close()
        _LOG_FILE = None


class Heartbeat:
    """A background "still alive" line while a child process runs.

    A THREAD RATHER THAN A TICK ON EACH LINE OF CHILD OUTPUT, because the case
    that needs it most is the one with no output. A T1 knee cell can sit for
    minutes inside a capture with nothing on stdout, and a heartbeat driven by
    the child's own lines falls silent exactly when a reader most needs to know
    the run has not hung. Ticking on output would report liveness only for the
    processes that were already reporting it.

    The thread runs in the PARENT, which is not the process under measurement —
    the child owns the GPU and the timings — and it wakes once per interval to
    print one line. `Event.wait` returns True the moment it is set, so __exit__
    is not delayed by the interval.
    """

    def __init__(self, label: str, every: Optional[float] = None):
        self.label = label
        self.every = HEARTBEAT_SECONDS if every is None else every
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            log(f"    ... {self.label}: still running, "
                f"{_hms(time.monotonic() - self.started)} elapsed")

    def __enter__(self) -> "Heartbeat":
        if self.every and self.every > 0:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return False


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SuiteState:
    """The resumable results file, flushed after every cell.

    Cell-level rather than test-level, because the expensive things here are
    cells: one T1 candidate is minutes, one T2 arm is hours, and a run that dies
    part-way through T1 must not re-measure the candidates it already has. Every
    cell carries its own `status`, so a resumed run can tell "not started" from
    "ran and failed" from "ran and was blocked".
    """

    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {"meta": {}, "t3": {}, "t1": {}, "t4": {}, "t2": {},
                         "thermal": {}, "verdicts": {}}
        for key in ("meta", "t3", "t1", "t4", "t2", "thermal", "verdicts"):
            self.data.setdefault(key, {})

    def cells(self, test: str) -> dict:
        return self.data[test].setdefault("cells", {})

    def get_cell(self, test: str, name: str) -> Optional[dict]:
        return self.cells(test).get(name)

    def done(self, test: str, name: str) -> bool:
        cell = self.get_cell(test, name)
        return bool(cell) and cell.get("status") in ("ok", "blocked")

    def put_cell(self, test: str, name: str, cell: dict) -> None:
        cell.setdefault("recorded_utc", utcnow())
        self.cells(test)[name] = cell
        self.flush()

    def drop(self, test: str, name: Optional[str] = None) -> None:
        if name is None:
            self.data[test] = {}
        else:
            self.cells(test).pop(name, None)
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str),
                       encoding="utf-8", newline="\n")
        tmp.replace(self.path)


# ---------------------------------------------------------------------------
# Environment guards
# ---------------------------------------------------------------------------


class Blocked(RuntimeError):
    """A precondition this suite will not publish around."""


def sm_clock_mhz() -> Optional[float]:
    """Imported rather than restated — tools/bench_c9_knee.py owns this."""
    try:
        from tools.bench_c9_knee import sm_clock_mhz as _clock
        return _clock()
    except Exception:
        return None


def require_release_build() -> dict:
    """Refuse to publish from a sanitizer build, as bench_c10b.py already does."""
    import guofish_core
    build = dict(guofish_core.build_info())
    if build["asan"] or build["ubsan"]:
        raise Blocked(
            f"this is an instrumented build ({build}). Every timing this suite "
            f"would report is a sanitizer's timing, not the engine's. Delete the "
            f"staged guofish_core*.pyd, rebuild Release, and check build_info().")
    return build


def assert_core_cell(par: dict, audit: dict, stats: dict, *, fixed_budget: bool,
                     where: str) -> None:
    """The per-cell guards, asserted loudly, on a cell driven through the core.

    `fixed_budget` is False only where tree reuse legitimately delivers less
    than it was asked for — which is T4's whole subject. Applying the
    delivered==requested guard there would fail the measurement for exhibiting
    the effect being measured.
    """
    if stats.get("synthetic_evaluations", 0) != 0:
        raise Blocked(f"{where}: the stand-in evaluator answered a leaf on the "
                      f"LIVE path ({stats['synthetic_evaluations']} times). That "
                      f"is the confound C9b carried and C10b removed; its "
                      f"presence invalidates the cell.")
    if audit.get("vloss_total", 0) != 0:
        raise Blocked(f"{where}: virtual loss stranded (vloss_total="
                      f"{audit['vloss_total']})")
    if audit.get("conservation_failures", 0) != 0:
        raise Blocked(f"{where}: visits do not conserve "
                      f"({audit['conservation_failures']} failures)")
    if fixed_budget and par["delivered"] != par["requested"]:
        raise Blocked(f"{where}: delivered {par['delivered']} != requested "
                      f"{par['requested']} under a fixed budget")


# ---------------------------------------------------------------------------
# Random-weight checkpoints for T1
# ---------------------------------------------------------------------------


def write_random_checkpoint(candidate: Candidate, out_dir: Path, seed: int) -> Path:
    """A random-weight checkpoint carrying correct config metadata.

    THROUGHPUT DEPENDS ON SHAPE, NOT WEIGHTS — that is what makes T1 possible
    without training anything. What the checkpoint has to get right is the
    METADATA, because `build_model_for_checkpoint` dispatches on it: a `config`
    dict that is a real ModelConfig routes to ChessTransformerV5 rebuilt exactly
    as trained (activation and final LayerNorm included, neither of which is
    recoverable from a state dict). `value_scale` is written too, so the loader
    reports `source=checkpoint` and not the legacy fallback — these files must
    not be distinguishable from a trained one by anything that affects the
    forward's cost.
    """
    import torch
    from training.v5_multiPV.model_v5 import ChessTransformerV5, ModelConfig
    from playing.v6.playv6 import LEGACY_VALUE_SCALE

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{candidate.name}_random.pt"
    if path.exists():
        return path

    torch.manual_seed(seed)
    config = ModelConfig(
        d_model=candidate.d_model,
        nhead=candidate.nhead,
        num_layers=candidate.layers,
        dim_feedforward=candidate.dim_feedforward,
        head_dim=64,
        activation="gelu",
        norm_first=True,
        final_norm=True,
    )
    model = ChessTransformerV5(config)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "value_scale": float(LEGACY_VALUE_SCALE),
        "epoch": 0,
        "step": 0,
        "reason": "random weights — tools/capacity_suite.py T1, shape only",
        "best_val": float("nan"),
    }, path)
    return path


# ---------------------------------------------------------------------------
# T3 — the label-noise floor
# ---------------------------------------------------------------------------


def t3_collect_pairs(args) -> tuple[list[dict], dict]:
    """Scan the dump for positions carrying two policy-eligible blocks at
    matching depth, and build both targets through the full Pass B path.

    MATCHING DEPTH IS THE WHOLE CONSTRUCTION. Lichess lines carry several eval
    blocks per FEN from independent analyses; two blocks at very different
    depths disagree partly because one of them is simply better, and that
    disagreement is genuine depth improvement rather than noise. Requiring the
    two to sit within +-`--t3-depth-window` plies of each other is what keeps
    improvement out of the measurement.

    The label functions are IMPORTED from data/multiPV/labels.py, never
    reimplemented, so the floor is measured through the exact path that built
    the training targets. A second implementation that agreed with the first
    would prove nothing, and one that disagreed would be measuring itself.
    """
    import chess
    from data.multiPV.labels import (
        DEFAULT_EPSILON, build_policy_target, dense_policy, select_policy_block,
        value_from_raw_cp, value_raw_cp,
    )
    from data.multiPV.pass_a_index import iter_lines

    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    policy_min_depth = int(manifest["policy_min_depth"])
    cp_clamp = int(manifest["cp_clamp"])
    temperature = float(manifest["temperature"])
    epsilon = float(manifest.get("epsilon", DEFAULT_EPSILON))
    value_scale = float(manifest["value_scale"])

    log(f"  corpus thresholds from {CORPUS_MANIFEST.name}: "
        f"policy_min_depth={policy_min_depth} value_min_depth="
        f"{manifest['value_min_depth']} T={temperature} eps={epsilon} "
        f"value_scale={value_scale}")

    pairs: list[dict] = []
    counters = {"lines": 0, "parsed": 0, "no_second_block": 0,
                "depth_window_miss": 0, "invariant_violation": 0,
                "value_missing": 0, "usable": 0}
    started = time.perf_counter()
    # PROGRESS IS TIME-DRIVEN, NOT LINE-DRIVEN. This scan runs IN-PROCESS, so
    # the Heartbeat that covers the subprocess cells cannot see it, and the
    # original 500,000-line trigger meant the only sign of life during a slow
    # stretch was one line every few minutes — measured at over 30 s of total
    # silence at the start of a scan. The modulo is checked first because it is
    # cheap and this loop runs millions of times; the clock is only consulted on
    # that boundary.
    report_every = HEARTBEAT_SECONDS if HEARTBEAT_SECONDS > 0 else float("inf")
    last_report = started

    for _offset, raw in iter_lines(EVAL_DUMP):
        counters["lines"] += 1
        if counters["lines"] > args.t3_max_lines or len(pairs) >= args.t3_max_pairs:
            break
        if counters["lines"] % 20_000 == 0:
            now = time.perf_counter()
            if counters["lines"] % 500_000 == 0 or now - last_report >= report_every:
                last_report = now
                log(f"    {counters['lines']:,} lines, {len(pairs):,} usable "
                    f"pairs, {_hms(now - started)} elapsed")
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        fen, evals = rec.get("fen"), rec.get("evals") or []
        if not fen or len(evals) < 2:
            continue
        try:
            board = chess.Board(fen)
        except Exception:
            continue
        if not board.is_valid() or not board.legal_moves:
            continue
        counters["parsed"] += 1

        # Block 1 is what Pass B would have chosen: the deepest block yielding
        # >= 2 unique moves. Block 2 is the deepest OTHER block that also
        # qualifies and sits inside the depth window.
        first = select_policy_block(board, evals, policy_min_depth, cp_clamp)
        if first is None:
            continue
        entries1, depth1, _d, _r = first
        chosen1 = None
        for ev in evals:
            if int(ev.get("depth", -1)) == depth1 and (ev.get("pvs") or []):
                chosen1 = ev
                break

        second = None
        for ev in evals:
            if ev is chosen1:
                continue
            depth2 = int(ev.get("depth", -1))
            if depth2 < policy_min_depth:
                continue
            if abs(depth2 - depth1) > args.t3_depth_window:
                counters["depth_window_miss"] += 1
                continue
            got = select_policy_block(board, [ev], policy_min_depth, cp_clamp)
            if got is not None:
                second = (got, depth2, ev)
                break
        if second is None:
            counters["no_second_block"] += 1
            continue
        (entries2, _d2, _dd, _rr), depth2, block2 = second

        stm_white = board.turn == chess.WHITE
        i1, p1, _s1, ok1 = build_policy_target(entries1, stm_white, temperature,
                                               epsilon)
        i2, p2, _s2, ok2 = build_policy_target(entries2, stm_white, temperature,
                                               epsilon)
        if not (ok1 and ok2):
            counters["invariant_violation"] += 1
            continue

        legal_idxs = [m.from_square * 64 + m.to_square for m in board.legal_moves]
        dense1 = dense_policy(i1, p1, legal_idxs, epsilon)
        dense2 = dense_policy(i2, p2, legal_idxs, epsilon)

        raw1 = value_raw_cp((chosen1 or {}).get("pvs", [{}])[0]) if chosen1 else None
        raw2 = value_raw_cp((block2.get("pvs") or [{}])[0])
        if raw1 is None or raw2 is None:
            counters["value_missing"] += 1
            v1 = v2 = None
        else:
            v1 = value_from_raw_cp(raw1, value_scale)
            v2 = value_from_raw_cp(raw2, value_scale)

        pairs.append({
            "fen": fen, "depth1": depth1, "depth2": depth2,
            "dense1": dense1, "dense2": dense2,
            "legal_idxs": legal_idxs, "value1": v1, "value2": v2,
        })
        counters["usable"] += 1

    counters["seconds"] = round(time.perf_counter() - started, 1)
    counters["config"] = {"policy_min_depth": policy_min_depth,
                          "temperature": temperature, "epsilon": epsilon,
                          "value_scale": value_scale, "cp_clamp": cp_clamp,
                          "depth_window": args.t3_depth_window}
    return pairs, counters


def _kl(p, q, floor: float = 1e-12) -> float:
    """KL(p || q), summed over p's support only. Both arguments carry the same
    epsilon smoothing over the same legal-move set, so q is strictly positive
    wherever p is and the floor never fires in practice; it is there so a
    malformed row produces a large number rather than an inf that poisons a mean.
    """
    import numpy as np
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], floor))))


def t3_model_side(pairs: list[dict], model_path: Path, epsilon: float,
                  batch: int = 256) -> tuple[list[float], list[float]]:
    """The 10.9M model over the same subset: its policy KL and value error.

    THIS HALF IS NOT OPTIONAL. Pairwise disagreement between two noisy labels is
    an UPPER bound on the irreducible floor, not the floor itself: a model
    trained on many such labels averages the noise and can legitimately beat it.
    The pairwise number alone therefore cannot distinguish "the corpus is the
    ceiling" from "the model is nowhere near it", which is the entire question.

    The model's distribution is put on the labels' footing before comparing:
    softmax over LEGAL indices, then the same (1-eps) rescale plus a uniform
    eps/n_legal, so both sides carry identical support and identical smoothing
    and the two KLs are the same measurement of two different things.
    """
    import numpy as np
    import torch
    import guofish_core
    from playing.v6 import evaluator as live

    model, device = live.load_default_model(model_path)
    policy_kls: list[float] = []
    value_errs: list[float] = []

    with torch.inference_mode():
        for start in range(0, len(pairs), batch):
            chunk = pairs[start:start + batch]
            tokens = np.stack([guofish_core.tokens(p["fen"]) for p in chunk])
            x = torch.from_numpy(tokens).to(device=device, dtype=torch.int64)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                logits, values = model(x)
            logits = logits.float().cpu().numpy()
            values = values.float().cpu().numpy()

            for row, pair, value in zip(logits, chunk, values):
                legal = np.asarray(pair["legal_idxs"], dtype=np.int64)
                sel = row[legal]
                sel = sel - sel.max()
                probs = np.exp(sel)
                probs /= probs.sum()
                dense = np.zeros(4096, dtype=np.float64)
                # `+=` in both loops, exactly as labels.dense_policy does: a
                # promotion pair maps four legal moves onto one 4096 index and
                # assignment would drop three of them.
                for idx, prob in zip(legal, probs * (1.0 - epsilon)):
                    dense[idx] += prob
                share = epsilon / len(legal)
                for idx in legal:
                    dense[idx] += share
                policy_kls.append(_kl(pair["dense1"], dense))
                if pair["value1"] is not None:
                    value_errs.append(float((pair["value1"] - value) ** 2))

    del model
    torch.cuda.empty_cache()
    return policy_kls, value_errs


def run_t3(args, state: SuiteState) -> dict:
    log("")
    log("=" * 78)
    log("T3 — the label-noise floor of the corpus")
    log("=" * 78)
    if state.done("t3", "floor") and not args.force_all and "t3" not in args.force:
        log("  cached; skipping")
        return state.get_cell("t3", "floor")

    if not EVAL_DUMP.exists():
        raise Blocked(f"the Lichess eval dump is missing: {EVAL_DUMP}")

    log(f"  scanning {EVAL_DUMP.name} for positions with two policy-eligible "
        f"blocks within +-{args.t3_depth_window} depth")
    pairs, counters = t3_collect_pairs(args)
    log(f"  {counters['usable']:,} usable pairs from {counters['lines']:,} lines "
        f"in {counters['seconds']}s")
    if not pairs:
        raise Blocked("T3 found no depth-matched block pairs; widen "
                      "--t3-depth-window or raise --t3-max-lines")

    pairwise_kl = [_kl(p["dense1"], p["dense2"]) for p in pairs]
    pairwise_val = [float((p["value1"] - p["value2"]) ** 2) for p in pairs
                    if p["value1"] is not None and p["value2"] is not None]

    log(f"  model side: {V5_MODEL.name} over the same {len(pairs):,} positions")
    epsilon = counters["config"]["epsilon"]
    model_kl, model_val = t3_model_side(pairs, V5_MODEL, epsilon)

    pk, mk = statistics.fmean(pairwise_kl), statistics.fmean(model_kl)
    ratio = mk / pk if pk > 0 else float("inf")
    if ratio >= T3_LIVE_RATIO:
        verdict, gate = "capacity is live", True
    elif ratio <= T3_CEILING_RATIO:
        verdict, gate = "corpus-limited", False
    else:
        verdict, gate = "ambiguous", False

    cell = {
        "status": "ok",
        "pairs": len(pairs),
        "scan": {k: v for k, v in counters.items() if k != "config"},
        "label_config": counters["config"],
        "model": str(V5_MODEL),
        "policy": {
            "pairwise_kl_mean": pk,
            "pairwise_kl_median": statistics.median(pairwise_kl),
            "model_kl_mean": mk,
            "model_kl_median": statistics.median(model_kl),
            "ratio_model_over_pairwise": ratio,
        },
        "value": {
            "pairwise_mse": statistics.fmean(pairwise_val) if pairwise_val else None,
            "model_mse": statistics.fmean(model_val) if model_val else None,
            "n": len(pairwise_val),
        },
        "verdict": verdict,
        "gate_passed": gate,
        "gate_thresholds": {"live_at_or_above": T3_LIVE_RATIO,
                            "ceiling_at_or_below": T3_CEILING_RATIO},
        "sm_clock_mhz": sm_clock_mhz(),
    }
    log(f"  pairwise KL {pk:.5f}   model KL {mk:.5f}   ratio {ratio:.2f}x  "
        f"-> {verdict}")
    if cell["value"]["pairwise_mse"] is not None:
        log(f"  value MSE: pairwise {cell['value']['pairwise_mse']:.6f}  "
            f"model {cell['value']['model_mse']:.6f}")
    state.put_cell("t3", "floor", cell)
    return cell


# ---------------------------------------------------------------------------
# T1 — the throughput ladder
# ---------------------------------------------------------------------------


def run_subprocess(command: list[str], log_path: Path, *, label: str = "") -> int:
    """Run a child, tee its output to `log_path`, and report how long it took.

    The child's stdout goes to the file and NOT to this log — a knee bench emits
    hundreds of rows and interleaving them would bury the suite's own progress.
    What reaches the reader is the start line, a heartbeat while it runs, and
    the exit line with a duration; the detail is one `tail` away in `log_path`.
    """
    label = label or log_path.stem
    log(f"    $ {' '.join(str(c) for c in command[:5])} ... "
        f"({len(command)} tokens) -> {log_path.name}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace", newline="\n") as sink:
        with Heartbeat(label):
            proc = subprocess.Popen([str(c) for c in command], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace",
                                    cwd=str(REPO_ROOT), bufsize=1)
            for line in proc.stdout:
                sink.write(line)
            code = proc.wait()
    log(f"    {label} exited {code} after {_hms(time.monotonic() - started)}")
    return code


def fit_forward_terms(knee_rows: list[dict], path: str = "graph-production") -> dict:
    """Least squares on (batch, saturated seconds) -> fixed ms + per-row us.

    Saturated rather than isolated, and `graph-production` rather than
    `graph-gathered`: the saturated column is the back-to-back rate the
    dispatcher actually sees, and production is the path the engine runs — the
    full 4096-wide bf16 D2H, because C10 put the gather in C++ rather than on
    the GPU. Quoting `gathered` here would fit a throughput the engine does not
    have.
    """
    rows = [r for r in knee_rows if r.get("path") == path]
    if len(rows) < 2:
        return {"path": path, "fixed_ms": None, "per_row_us": None, "n": len(rows)}
    xs = [float(r["batch"]) for r in rows]
    ys = [float(r["sat_median_s"]) for r in rows]
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    return {
        "path": path,
        "fixed_ms": intercept * 1e3,
        "per_row_us": slope * 1e6,
        "n": n,
        "rmse_ms": (statistics.fmean(r ** 2 for r in residuals) ** 0.5) * 1e3,
        "batches": [int(x) for x in xs],
    }


def t1_candidate(candidate: Candidate, args, state: SuiteState, out_dir: Path,
                 *, label: Optional[str] = None) -> dict:
    """One candidate: knee curve, fitted terms, live Gate 4, and a K sweep."""
    name = label or candidate.name
    if candidate.blocked:
        cell = {"status": "blocked", "candidate": candidate.name,
                "reason": candidate.blocked}
        log(f"  {name}: BLOCKED — {candidate.blocked.splitlines()[0]}")
        state.put_cell("t1", name, cell)
        return cell

    log(f"  {name}: d_model={candidate.d_model} x{candidate.layers} layers, "
        f"{candidate.nhead} heads, ff={candidate.dim_feedforward}")
    cell_started = time.monotonic()
    checkpoint = write_random_checkpoint(candidate, out_dir / "checkpoints",
                                         args.seed)
    cell_dir = out_dir / "t1" / name
    cell_dir.mkdir(parents=True, exist_ok=True)

    clock_before = sm_clock_mhz()

    knee_json = cell_dir / "knee.json"
    code = run_subprocess([
        sys.executable, str(KNEE_BENCH), "--model", str(checkpoint),
        "--batch", *[str(b) for b in args.t1_batches],
        "--iters", str(args.t1_iters), "--no-ungraphed",
        "--source", "synthetic", "--json-out", str(knee_json),
    ], cell_dir / "knee.log")
    if code != 0 or not knee_json.exists():
        raise Blocked(f"T1 {name}: bench_c10b_knee.py exited {code}; see "
                      f"{cell_dir / 'knee.log'}")
    knee = json.loads(knee_json.read_text(encoding="utf-8"))

    grid_json = cell_dir / "gate4.json"
    k_configs = [f"1x{k}" for k in args.t1_k_sweep]
    code = run_subprocess([
        sys.executable, str(GRID_BENCH), "--sections", "gate4",
        "--model", str(checkpoint), "--max-batch", str(SHIP_MAX_BATCH),
        "--gate4-configs", *k_configs,
        "--gate4-sims", str(args.t1_gate4_sims),
        "--gate4-repeats", str(args.t1_gate4_repeats),
        "--reuse-plies", str(args.t1_reuse_plies),
        "--json-out", str(grid_json),
    ], cell_dir / "gate4.log")
    if code != 0 or not grid_json.exists():
        raise Blocked(f"T1 {name}: bench_c10b.py exited {code}; see "
                      f"{cell_dir / 'gate4.log'}")
    grid = json.loads(grid_json.read_text(encoding="utf-8"))

    gate4 = grid.get("gate4", [])
    fresh = {int(r["in_flight"]): r for r in gate4 if r["regime"] == "fresh root"}
    reuse = {int(r["in_flight"]): r for r in gate4 if r["regime"] == "reuse-heavy"}
    ship = fresh.get(SHIP_IN_FLIGHT)

    # The smallest K in the sweep that clears the floor at a fresh root. This is
    # what surfaces the "pays for capacity twice" effect: a candidate that needs
    # K=48-64 to clear Gate 4 buys the throughput back with ~10 points of
    # top-move share (C10b-3e), so its real cost is not the ladder alone.
    clearing_k = sorted(k for k, r in fresh.items()
                        if r["sims_per_s"] >= GATE4_FLOOR)

    cell = {
        "status": "ok",
        "candidate": candidate.name,
        "label": name,
        "shape": {"d_model": candidate.d_model, "layers": candidate.layers,
                  "nhead": candidate.nhead, "head_dim": 64,
                  "dim_feedforward": candidate.dim_feedforward,
                  "activation": "gelu", "final_norm": True},
        "axis": candidate.axis,
        "checkpoint": str(checkpoint),
        "macs_ratio_brief": candidate.macs_ratio_brief,
        "macs_ratio_computed": (macs_per_position(candidate.d_model, candidate.layers)
                                / macs_per_position(CONTROL.d_model, CONTROL.layers)),
        "knee_rows": knee.get("rows", []),
        "forward_fit": fit_forward_terms(knee.get("rows", [])),
        "gate4": gate4,
        "shipping_cell": ship,
        "gate4_fresh_by_k": {str(k): r["sims_per_s"] for k, r in sorted(fresh.items())},
        "gate4_reuse_by_k": {str(k): r["sims_per_s"] for k, r in sorted(reuse.items())},
        "clears_gate4_at_k": clearing_k,
        "min_k_clearing_gate4": clearing_k[0] if clearing_k else None,
        "sm_clock_mhz_before": clock_before,
        "sm_clock_mhz_after": sm_clock_mhz(),
        "capture": grid.get("capture"),
        "build": grid.get("build"),
    }
    if ship:
        log(f"    Gate 4 @ K={SHIP_IN_FLIGHT}: {ship['sims_per_s']:,.0f} sims/s "
            f"({ship['sims_per_s'] / GATE4_FLOOR:.2f}x floor), "
            f"reuse-heavy {reuse.get(SHIP_IN_FLIGHT, {}).get('sims_per_s', 0):,.0f}")
    fit = cell["forward_fit"]
    if fit.get("fixed_ms") is not None:
        log(f"    forward: {fit['fixed_ms']:.3f} ms fixed + {fit['per_row_us']:.1f} "
            f"us/row (rmse {fit['rmse_ms']:.3f} ms)")
    # Recorded as well as logged: a resumed run's cached cells still carry what
    # they cost, which is what lets the next run estimate its own duration.
    cell["seconds"] = round(time.monotonic() - cell_started, 1)
    log(f"    {name} complete in {_hms(cell['seconds'])}")
    state.put_cell("t1", name, cell)
    return cell


def run_t1(args, state: SuiteState, out_dir: Path) -> dict:
    log("")
    log("=" * 78)
    log("T1 — throughput ladder on random weights")
    log("=" * 78)
    log("  Throughput depends on SHAPE, not weights, which is what makes this")
    log("  measurable without training anything. Control runs first and again")
    log("  last (thermal).")

    forced = args.force_all or "t1" in args.force
    # Selected up front so the progress counter has a denominator. Without it a
    # reader watching overnight can see which cell is running but not how much
    # of T1 is left, which is the question they actually have.
    selected = [c for c in CANDIDATES
                if not args.t1_candidates or c.name in args.t1_candidates]
    log(f"  {len(selected)} candidates: {', '.join(c.name for c in selected)}")
    for index, candidate in enumerate(selected, start=1):
        name = candidate.name
        if state.done("t1", name) and not forced and name not in args.force_cell:
            log(f"  [{index}/{len(selected)}] {name}: cached; skipping")
            continue
        log(f"  [{index}/{len(selected)}] {name}: starting")
        t1_candidate(candidate, args, state, out_dir)
    return state.data["t1"]


def _cell_clocks(cell: dict) -> list[float]:
    """One T1 cell's under-load SM-clock samples."""
    return [float(r["sat_sm_clock_mhz"]) for r in (cell.get("knee_rows") or [])
            if r.get("sat_sm_clock_mhz")]


def _loaded_clocks(state: SuiteState) -> list[float]:
    """Every SM-clock sample taken while the GPU was actually working.

    `bench_c9_knee.time_saturated` records `sat_sm_clock_mhz` per row from
    inside the back-to-back loop, which is the only place the reading means
    anything: between cells the device is idle and reports its floor.
    """
    out: list[float] = []
    for cell in state.cells("t1").values():
        for row in cell.get("knee_rows") or []:
            clock = row.get("sat_sm_clock_mhz")
            if clock:
                out.append(float(clock))
    return out


def run_thermal_control(args, state: SuiteState, out_dir: Path) -> dict:
    """The control candidate, re-measured as the LAST action of the suite.

    C10b logged 2857-2917 MHz across its grid, so a ~24 h serial run can drift
    a couple of percent without anything being wrong with the engine. The point
    of re-measuring the control cell is that drift and capacity cost are
    indistinguishable in a single pass: if this cell has moved, every T1 number
    is suspect and the summary says so rather than reporting the drift as a
    property of the candidates.
    """
    log("")
    log("=" * 78)
    log("THERMAL — the T1 control cell, re-measured last")
    log("=" * 78)
    if state.done("t1", "d384x6_control_final") and not args.force_all:
        log("  cached; skipping")
    else:
        t1_candidate(CONTROL, args, state, out_dir, label="d384x6_control_final")

    first = state.get_cell("t1", CONTROL.name) or {}
    last = state.get_cell("t1", "d384x6_control_final") or {}
    a = (first.get("shipping_cell") or {}).get("sims_per_s")
    b = (last.get("shipping_cell") or {}).get("sims_per_s")

    # UNDER LOAD, from the knee bench's own per-row samples, NOT from a sample
    # taken between subprocesses. An RTX 5070 drops to ~750 MHz the moment it
    # goes idle (bench_c9_knee.sm_clock_mhz documents the same DVFS ramp as the
    # artefact that produced the recon-era "4.86 ms fixed" model), so a harness
    # that sampled in the gaps would report a ~292% "spread" that is the GPU
    # going to sleep between cells and nothing to do with thermal drift.
    clocks = [c for c in _loaded_clocks(state) if c]
    drift = abs(b - a) / a if (a and b) else None

    # LIKE FOR LIKE. The spread ACROSS batch sizes is the DVFS ramp, not drift:
    # at batch 8 the device is busy for well under a millisecond and then idles
    # inside `torch.cuda.synchronize()`, so the small-batch rows legitimately
    # read low. Comparing the same cell's fully-ramped clock at the start of the
    # suite against the same cell's at the end is the only comparison in which a
    # difference can only be thermal.
    peak_first = max(_cell_clocks(first), default=None)
    peak_last = max(_cell_clocks(last), default=None)
    clock_drift = (abs(peak_last - peak_first) / peak_first
                   if (peak_first and peak_last) else None)
    verdict = {
        "status": "ok",
        "control_first_sims_per_s": a,
        "control_last_sims_per_s": b,
        "throughput_drift": drift,
        "sm_clocks_mhz": clocks,
        "control_peak_clock_first": peak_first,
        "control_peak_clock_last": peak_last,
        "sm_clock_drift": clock_drift,
        "tolerance": args.thermal_tolerance,
        "t1_suspect": bool(drift is not None and drift > args.thermal_tolerance),
    }
    if drift is not None:
        log(f"  control {a:,.0f} -> {b:,.0f} sims/s ({100 * drift:.1f}% drift, "
            f"tolerance {100 * args.thermal_tolerance:.0f}%) — "
            f"{'T1 SUSPECT' if verdict['t1_suspect'] else 'within tolerance'}")
    if clock_drift is not None:
        log(f"  control peak SM clock {peak_first:.0f} -> {peak_last:.0f} MHz "
            f"({100 * clock_drift:.1f}% drift); all T1 rows under load spanned "
            f"{min(clocks):.0f}-{max(clocks):.0f} MHz (the ramp, not drift)")
    state.data["thermal"] = verdict
    state.flush()
    return verdict


# ---------------------------------------------------------------------------
# Match machinery — shared by T4 and T2
# ---------------------------------------------------------------------------

# The engine's own per-move telemetry, from playv6.SearchOutcome.telemetry().
# `delivered` is what the core backed up into the root; `nominal` is what the
# caller asked for, or `n/a` for a timed search — which is a real distinction
# and never synthesised into a number.
SEARCH_LINE = re.compile(
    r"\[search\] source=(?P<source>\w+) delivered=(?P<delivered>\d+) "
    r"nominal=(?P<nominal>n/a|\d+) inherited=(?P<inherited>\d+)")
SCORE_LINE = re.compile(
    r"Score of (?P<a>.+?) vs (?P<b>.+?): (?P<w>\d+) - (?P<l>\d+) - (?P<d>\d+)")
SPRT_LINE = re.compile(r"SPRT:.*?llr (?P<llr>[-\d.]+).*?\[(?P<bounds>[^\]]*)\]", re.I)
SPRT_VERDICT = re.compile(r"(H0|H1) was accepted", re.I)


@dataclass
class EngineArm:
    """One side of a match: a checkpoint, its Q-unit constants, its clock."""

    name: str
    model: Path
    nodes: Optional[int] = None
    tc: Optional[str] = None
    c_puct_init: float = V5_C_PUCT_INIT
    fpu_tree: float = V5_FPU_TREE
    # None leaves `EngineConfig.arena_capacity` (1,200,000) alone, which is
    # correct for every arm whose budget it actually covers — see
    # ARENA_NODES_PER_SIM for what "covers" means and what happened when it did
    # not. Fixed-node arms set this from `arena_for(nodes)`.
    arena_capacity: Optional[int] = None
    extra: tuple[str, ...] = field(default_factory=tuple)

    def args(self) -> list[str]:
        """The wrapper's flags. THE SHIPPING CONFIG IS THE DATACLASS DEFAULT, so
        W/K/max_batch/affinity are passed explicitly only to make the artifact
        self-describing — an arm that silently inherited a changed default would
        be unreadable a month from now.

        `--no-book --no-syzygy` on every arm: both default ON since C11b and
        both BYPASS MCTS, which would nullify `-openings` and shift the endgame
        cache-hit rate that T1's reuse-regime numbers are built on.
        """
        out = [
            "-u", WRAPPER,
            "--model", str(self.model),
            "--threads", str(SHIP_WORKERS),
            "--max-outstanding", str(SHIP_IN_FLIGHT),
            "--max-batch", str(SHIP_MAX_BATCH),
            "--affinity", SHIP_AFFINITY,
            "--c-puct-init", str(self.c_puct_init),
            "--fpu-tree", str(self.fpu_tree),
            "--no-book", "--no-syzygy",
        ]
        if self.arena_capacity is not None:
            out += ["--arena-capacity", str(self.arena_capacity)]
        return out + list(self.extra)


def build_match_command(arms: tuple[EngineArm, EngineArm], out_dir: Path, *,
                        rounds: int, concurrency: int, event: str,
                        adjudicate: bool, sprt: Optional[str],
                        maxmoves: int, opening_plies: int,
                        timemargin: int) -> list[str]:
    """The cutechess-cli invocation, as a list — no shell quoting to get wrong.

    THE HOUSE SYNTAX, from tools/smoke_c11.py and the v5 A/B command line:
    `cmd=python` with a repo-relative script path and `dir=` set to the repo
    root, one `arg=` per argument, `-rounds R -games 2 -repeat` over sequential
    openings. Deviating would make these PGNs incomparable with everything else
    under benchmarking/engine/games/.

    NO `-debug`: this build of cutechess-cli 1.4 rejects it in any form. The
    evidence it would have given comes from the engine instead — each engine
    writes its own stderr to a file via `stderr=`, and that is where the
    delivered-sims telemetry and the resolved book/Syzygy state are read from.

    ADJUDICATION IS A PARAMETER because it is not symmetric across all arms.
    `-resign score=600` and `-draw score=10` fire on `score cp = value_scale *
    atanh(q)`, which is derived from the value head — so two nets with different
    Q distributions get adjudicated differently and identical play can score
    differently. T4 runs the same net on both sides and keeps the house
    adjudication; T2 does not, and turns it off in favour of a Q-independent
    `-maxmoves` cap.
    """
    command: list[str] = [str(CUTECHESS)]
    for arm in arms:
        command += ["-engine", f"name={arm.name}", "cmd=python"]
        command += [f"arg={a}" for a in arm.args()]
        command += [f"dir={REPO_ROOT}", "proto=uci"]
        if arm.nodes is not None:
            command += ["tc=inf", f"nodes={arm.nodes}", f"timemargin={timemargin}"]
        else:
            command += [f"tc={arm.tc}"]
        command += [f"stderr={out_dir / (arm.name + '.stderr.log')}"]

    command += ["-openings", f"file={OPENINGS}", "format=pgn",
                "order=sequential", f"plies={opening_plies}"]
    if adjudicate:
        command += ["-resign", "movecount=3", "score=600",
                    "-draw", "movenumber=40", "movecount=8", "score=10"]
    else:
        # Q-INDEPENDENT TERMINATION. With the score-based adjudicators off, a
        # drawn game has nothing to end it, so the cap is what keeps the arm
        # finite. It is a ply count and is identical for both engines, which is
        # exactly the property `-resign score=` does not have across two nets.
        command += ["-maxmoves", str(maxmoves)]
    if sprt:
        command += ["-sprt", *sprt.split()]
    command += ["-recover", "-concurrency", str(concurrency),
                "-rounds", str(rounds), "-games", "2", "-repeat",
                "-event", event,
                "-pgnout", str(out_dir / f"{event}.pgn")]
    return command


def supersede_previous(out_dir: Path, event: str) -> Optional[Path]:
    """Move a previous run of this event aside, and say where it went.

    REQUIRED FOR CORRECTNESS ON A RE-RUN, not merely tidy. Two of the four
    artifacts here are written by Cutechess rather than by this harness, and
    neither is truncated: `-pgnout` APPENDS, so re-running a 150-game rung in
    place leaves a 300-game PGN whose halves came from different configurations,
    and the per-arm `stderr=` files accumulate the same way — which would let a
    previous run's `[error] command=go` lines fire `arm_fallback_moves` on a
    clean re-run and condemn it for its predecessor's failure. (The cutechess
    log alone is safe: `run_match` opens it "w".)

    MOVED, NEVER DELETED. The superseded run is the evidence for whatever the
    re-run is fixing, and the first T4 run's stderr is precisely how the
    first-legal-move fallback was diagnosed at all.
    """
    stale = [p for p in (list(out_dir.glob(f"{event}.*")) + list(out_dir.glob("*.stderr.log")))
             if p.is_file()]
    if not stale:
        return None
    attic = out_dir / f"_superseded_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    attic.mkdir(parents=True, exist_ok=True)
    for p in stale:
        p.replace(attic / p.name)
    log(f"    superseded {len(stale)} artifact(s) from a previous run -> "
        f"{attic.name}/")
    return attic


def run_match(command: list[str], out_dir: Path, event: str) -> tuple[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    supersede_previous(out_dir, event)
    log_path = out_dir / f"{event}.cutechess.log"
    (out_dir / f"{event}.command.txt").write_text(
        " ".join(str(c) for c in command), encoding="utf-8", newline="\n")
    log(f"    running {event}: {len(command)} tokens -> {log_path.name}")
    started = time.perf_counter()
    # `Score of ...` arrives once per game pair, so a match reports its own
    # progress; the heartbeat covers the gaps between games and, more usefully,
    # the case where cutechess is stuck starting an engine and emits nothing at
    # all. `games` counts the score lines so the heartbeat can say how far in.
    games = 0
    with log_path.open("w", encoding="utf-8", errors="replace", newline="\n") as sink:
        with Heartbeat(event):
            proc = subprocess.Popen([str(c) for c in command], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace",
                                    cwd=str(REPO_ROOT), bufsize=1)
            for line in proc.stdout:
                sink.write(line)
                if line.startswith("Score of") or "SPRT" in line:
                    if line.startswith("Score of"):
                        games += 1
                    log(f"      [{games}] {line.rstrip()}")
            code = proc.wait()
    log(f"    {event} exited {code} after {_hms(time.perf_counter() - started)} "
        f"({games} score lines)")
    return code, log_path


def parse_delivered(stderr_path: Path) -> dict:
    """Delivered simulations per move, from the engine's own telemetry.

    NOT the `nodes=` value. Tree reuse makes `search_parallel` target absolute
    root visits, so a `go nodes N` on a reused root delivers fewer than N — a
    20-game fixed-node run aggregated to 1.16x inflation and a fully-reused root
    delivers zero and reports `inf`. An ELO-per-doubling denominator built on
    the nominal figure is wrong by exactly that factor.

    A CAVEAT THIS FUNCTION CANNOT REMOVE: at `-concurrency 2` two processes of
    the same engine config write to ONE `stderr=` file, so a line can be torn.
    The regex requires a complete well-formed line and `malformed` counts what
    it rejected, so a corrupted arm is visible rather than silently short.
    """
    if not stderr_path.exists():
        return {"moves": 0, "delivered_total": 0, "malformed": 0}
    text = stderr_path.read_text(encoding="utf-8", errors="replace")
    delivered, nominal, inherited, bypassed = [], [], [], 0
    candidates = 0
    for line in text.splitlines():
        if "[search]" not in line:
            continue
        candidates += 1
        m = SEARCH_LINE.search(line)
        if not m:
            continue
        if m.group("source") != "search":
            bypassed += 1
            continue
        delivered.append(int(m.group("delivered")))
        inherited.append(int(m.group("inherited")))
        if m.group("nominal") != "n/a":
            nominal.append(int(m.group("nominal")))
    return {
        "moves": len(delivered),
        "malformed": candidates - len(delivered) - bypassed,
        "bypassed_moves": bypassed,
        "delivered_total": sum(delivered),
        "delivered_mean": statistics.fmean(delivered) if delivered else 0.0,
        "delivered_median": statistics.median(delivered) if delivered else 0.0,
        "nominal_total": sum(nominal),
        "nominal_mean": statistics.fmean(nominal) if nominal else None,
        "inherited_mean": statistics.fmean(inherited) if inherited else 0.0,
        # Aggregate over the WHOLE arm rather than a mean of per-move ratios: a
        # fully-reused root contributes delivered=0 and a per-move ratio of inf,
        # which no mean survives.
        "inflation": (sum(nominal) / sum(delivered)) if (nominal and sum(delivered)) else None,
    }


def parse_result(log_path: Path, sprt_requested: bool = False) -> dict:
    """The final score line, plus the SPRT verdict when one was requested."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    scores = SCORE_LINE.findall(text)
    out: dict = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    if scores:
        a, b, w, l, d = scores[-1]
        out = {"engine_a": a.strip(), "engine_b": b.strip(),
               "wins": int(w), "losses": int(l), "draws": int(d),
               "games": int(w) + int(l) + int(d)}
    # ONLY when an SPRT was actually requested. cutechess-cli 1.4 prints a
    # `SPRT: llr 0 ... lbound -inf, ubound inf` line on every match whether or
    # not `-sprt` was passed, and recording that would put a hypothesis test in
    # the artifact of an arm that ran none.
    if sprt_requested:
        verdict = SPRT_VERDICT.search(text)
        if verdict:
            out["sprt_verdict"] = verdict.group(0)
        llrs = SPRT_LINE.findall(text)
        if llrs:
            out["sprt_last_llr"] = float(llrs[-1][0])
    out["finished_games"] = len([l for l in text.splitlines()
                                 if l.startswith("Finished game")])
    return out


def elo_with_ci(wins: int, losses: int, draws: int, z: float = 1.96) -> dict:
    """ELO of A over B with a normal CI on the score, propagated through the
    logistic. The per-game variance is computed from the actual W/D/L split
    rather than assumed, which matters here because these arms draw heavily and
    a draw contributes no variance at all around a 50% score.
    """
    n = wins + losses + draws
    if n == 0:
        return {"games": 0, "elo": None, "ci95": None, "score": None}
    score = (wins + 0.5 * draws) / n
    var = (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2
           + losses * score ** 2) / n
    stderr_ = math.sqrt(var / n) if var > 0 else 0.0

    def to_elo(s: float) -> Optional[float]:
        s = min(max(s, 1e-9), 1 - 1e-9)
        return -400.0 * math.log10(1.0 / s - 1.0)

    lo, hi = to_elo(max(1e-9, score - z * stderr_)), to_elo(min(1 - 1e-9, score + z * stderr_))
    point = to_elo(score)
    return {"games": n, "wins": wins, "losses": losses, "draws": draws,
            "score": score, "elo": point, "elo_lo": lo, "elo_hi": hi,
            "ci95": (hi - lo) / 2 if (lo is not None and hi is not None) else None,
            "draw_rate": draws / n,
            # A clean sweep has zero sample variance, so the CI collapses to
            # +-0.0 and the point estimate is whatever the score clamp allows.
            # Both are artefacts of n, not measurements, and saying so here is
            # what keeps a short arm from reading as an infinitely precise one.
            "degenerate": var == 0.0 or n < 20}


def arm_feature_state(out_dir: Path, arms: tuple[EngineArm, EngineArm]) -> dict:
    """The resolved book/Syzygy state, read from each ENGINE'S OWN stderr.

    Read from the engine and not from `EngineArm.args()` on purpose: the config
    says what was ASKED for, the engine says what actually opened. An arm whose
    book was on by accident says so in its own artifact, which is the discipline
    C11b's brief attached to defaulting both features ON.
    """
    states = {}
    for arm in arms:
        path = out_dir / f"{arm.name}.stderr.log"
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        try:
            state = bench_provenance.from_engine_log(text)
            state["header"] = bench_provenance.require_recorded_state(state)
            state["recorded"] = True
        except Exception as exc:
            state = {"recorded": False, "error": str(exc).splitlines()[0]}
        states[arm.name] = state
    return states


# One `[error] command=go` line per failed search. The wrapper prints the
# exception message twice — once here, once as the traceback's last line — so
# counting the message would double every total; this anchors on the marker.
GO_ERROR = re.compile(r"^\[error\] command=go (?P<exc>\w+):", re.M)


def arm_fallback_moves(out_dir: Path, arms: tuple[EngineArm, EngineArm]) -> dict:
    """Moves each arm answered from its exception handler rather than its search.

    THE MATCH CANNOT BE READ WITHOUT THIS. When `go` raises, the wrapper still
    owes Cutechess a `bestmove` and answers with the first legal move in the
    position — a real move, in a legal game, recorded in the PGN with a plausible
    score. Nothing in the result, the ELO or the PGN distinguishes it from a
    searched move, so a corrupted arm publishes as a clean one: the first T4 run
    scored 647 such moves and reported -462.7 ELO/doubling.

    Read from the engine's own stderr for the same reason `arm_feature_state` is:
    the harness's arguments say what was asked for, and only the engine can say
    what happened. `-recover` does not help and its absence is not evidence —
    the process never dies, it catches and carries on.
    """
    out = {"total": 0, "by_arm": {}, "by_exception": {}}
    for arm in arms:
        path = out_dir / f"{arm.name}.stderr.log"
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        hits = GO_ERROR.findall(text)
        out["by_arm"][arm.name] = len(hits)
        out["total"] += len(hits)
        for exc in hits:
            out["by_exception"][exc] = out["by_exception"].get(exc, 0) + 1
    return out


def play(arms: tuple[EngineArm, EngineArm], out_dir: Path, *, event: str,
         rounds: int, concurrency: int, adjudicate: bool, sprt: Optional[str],
         args) -> dict:
    command = build_match_command(
        arms, out_dir, rounds=rounds, concurrency=concurrency, event=event,
        adjudicate=adjudicate, sprt=sprt, maxmoves=args.maxmoves,
        opening_plies=args.opening_plies, timemargin=args.timemargin)
    clock_before = sm_clock_mhz()
    code, log_path = run_match(command, out_dir, event)
    result = parse_result(log_path, sprt_requested=bool(sprt))
    telemetry = {arm.name: parse_delivered(out_dir / f"{arm.name}.stderr.log")
                 for arm in arms}
    elo = elo_with_ci(result.get("wins", 0), result.get("losses", 0),
                      result.get("draws", 0))
    fallback = arm_fallback_moves(out_dir, arms)
    if fallback["total"]:
        log(f"    !! {fallback['total']} move(s) answered by the wrapper's "
            f"exception handler, not by search: {fallback['by_arm']} "
            f"({fallback['by_exception']})")
    return {
        # "corrupted" is deliberately NOT one of SuiteState.done()'s terminal
        # statuses, so the cell is written for diagnosis but re-runs rather than
        # being treated as a measurement.
        "status": ("corrupted" if fallback["total"]
                   else "ok" if result.get("games") else "failed"),
        "event": event,
        "arena_capacity": {arm.name: arm.arena_capacity for arm in arms},
        "fallback": fallback,
        "exit_code": code,
        "command": [str(c) for c in command],
        "log": str(log_path),
        "pgn": str(out_dir / f"{event}.pgn"),
        "concurrency": concurrency,
        "adjudication": ("resign score=600 / draw score=10 (house)" if adjudicate
                         else f"DISABLED — Q-denominated; -maxmoves {args.maxmoves} instead"),
        "sprt": sprt,
        "result": result,
        "elo": elo,
        "telemetry": telemetry,
        "feature_state": arm_feature_state(out_dir, arms),
        "sm_clock_mhz_before": clock_before,
        "sm_clock_mhz_after": sm_clock_mhz(),
    }


# ---------------------------------------------------------------------------
# T4 — ELO per sim-doubling at deployment sims
# ---------------------------------------------------------------------------


def _budget(n: int) -> str:
    """A node budget as a cell name. Integer-divided thousands collapsed every
    sub-1000 budget to `0k`, which is only reachable at smoke sizes but made two
    distinct rungs share a prefix and read as one."""
    return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)


def run_t4(args, state: SuiteState, out_dir: Path) -> dict:
    log("")
    log("=" * 78)
    log("T4 — ELO per sim-doubling at deployment sims")
    log("=" * 78)
    log("  The 91 ELO/doubling figure was measured at 800->2000 sims in Python.")
    log("  The rate that prices T1's cost is the MARGINAL rate at deployment")
    log("  sims, and it decays — so both rungs are reported, never averaged.")

    forced = args.force_all or "t4" in args.force
    rungs = [(args.t4_low, args.t4_mid), (args.t4_mid, args.t4_high)]
    for low, high in rungs:
        name = f"{_budget(low)}_vs_{_budget(high)}"
        if state.done("t4", name) and not forced and name not in args.force_cell:
            log(f"  {name}: cached; skipping")
            continue
        log(f"  rung {name}: {args.t4_games} games, paired openings, fixed nodes")
        arms = (EngineArm(f"v5-{_budget(low)}", V5_MODEL, nodes=low,
                          arena_capacity=arena_for(low)),
                EngineArm(f"v5-{_budget(high)}", V5_MODEL, nodes=high,
                          arena_capacity=arena_for(high)))
        log(f"    arena: {arms[0].name} {arena_for(low):,} nodes, "
            f"{arms[1].name} {arena_for(high):,} nodes "
            f"({ARENA_NODES_PER_SIM}/sim)")
        cell = play(arms, T4_PGN_DIR / name, event=name,
                    rounds=max(1, args.t4_games // 2), concurrency=2,
                    adjudicate=True, sprt=None, args=args)

        # BEFORE the rung math, because the rung math would happily turn
        # fallback moves into an ELO-per-doubling figure — which is exactly how
        # -462.7 got published. The cell is recorded first so the evidence
        # survives the abort.
        if cell.get("fallback", {}).get("total"):
            state.put_cell("t4", name, cell)
            raise Blocked(
                f"T4 {name}: {cell['fallback']['total']} move(s) came from the "
                f"wrapper's exception handler instead of its search "
                f"({cell['fallback']['by_arm']}, {cell['fallback']['by_exception']}). "
                f"Those are first-legal-move answers, so every number this rung "
                f"would report is contaminated.\n  Arena was "
                f"{cell.get('arena_capacity')}; if the exception is a RuntimeError "
                f"the arena is still short — raise ARENA_NODES_PER_SIM (currently "
                f"{ARENA_NODES_PER_SIM}) and re-run this cell. The per-arm stderr "
                f"logs under {T4_PGN_DIR / name} carry the tracebacks.")

        # THE DENOMINATOR. A nominal 2x that reuse compresses into a delivered
        # 1.7x makes every ELO-per-doubling figure built on it wrong by that
        # factor, so the realised ratio is measured and the rate is quoted
        # against it — not against the `nodes=` values.
        lo_t = cell["telemetry"][arms[0].name]
        hi_t = cell["telemetry"][arms[1].name]
        delivered_ratio = ((hi_t["delivered_mean"] / lo_t["delivered_mean"])
                           if lo_t["delivered_mean"] else None)
        doublings = math.log2(delivered_ratio) if delivered_ratio and delivered_ratio > 0 else None
        elo = cell["elo"]
        # `elo` is A-over-B and A is the LOW arm, so the high arm's advantage is
        # its negation.
        margin = -elo["elo"] if elo["elo"] is not None else None
        cell["rung"] = {
            "nominal_low": low, "nominal_high": high,
            "nominal_ratio": high / low,
            "delivered_mean_low": lo_t["delivered_mean"],
            "delivered_mean_high": hi_t["delivered_mean"],
            "delivered_ratio": delivered_ratio,
            "doublings_realised": doublings,
            "elo_high_over_low": margin,
            "elo_ci95": elo["ci95"],
            "elo_per_doubling": (margin / doublings)
            if (margin is not None and doublings) else None,
            "elo_per_doubling_ci95": (elo["ci95"] / doublings)
            if (elo["ci95"] is not None and doublings) else None,
            "inflation_low": lo_t["inflation"],
            "inflation_high": hi_t["inflation"],
        }
        r = cell["rung"]
        if r["elo_per_doubling"] is not None:
            log(f"    delivered {r['delivered_mean_low']:,.0f} -> "
                f"{r['delivered_mean_high']:,.0f} "
                f"({r['delivered_ratio']:.2f}x against a nominal "
                f"{r['nominal_ratio']:.0f}x)")
            log(f"    {r['elo_high_over_low']:+.1f} +-{r['elo_ci95']:.1f} ELO "
                f"= {r['elo_per_doubling']:+.1f} ELO/doubling")
        state.put_cell("t4", name, cell)
    return state.data["t4"]


def t4_marginal_rate(state: SuiteState) -> Optional[dict]:
    """The rate that prices T1's cost: the HIGHEST rung measured, not a mean.

    Averaging the two rungs would price a capacity loss at deployment sims using
    a rate partly measured below them, and the decay between the rungs is the
    thing being reported. The lower rung stays in the summary as the decay
    evidence.
    """
    cells = [c for c in state.cells("t4").values()
             if c.get("status") == "ok" and c.get("rung", {}).get("elo_per_doubling")]
    if not cells:
        return None
    top = max(cells, key=lambda c: c["rung"]["nominal_high"])
    return top["rung"]


# ---------------------------------------------------------------------------
# T2 — v4 vs v5 on the same engine
# ---------------------------------------------------------------------------


def run_t2(args, state: SuiteState, out_dir: Path) -> dict:
    log("")
    log("=" * 78)
    log("T2 — v4 vs v5 on the same engine (the most informative test)")
    log("=" * 78)
    log("  v4 is a 2.37x-MACs net that is strictly worse trained: RELU defect,")
    log("  no final LayerNorm, mismatched teachers, human-imitation policy. So")
    log("  v4 WINNING is unambiguous evidence that capacity dominates and")
    log("  lower-bounds a well-trained net at that size. v5 winning is")
    log("  genuinely ambiguous and is reported as such.")
    log(f"  Q-unit constants: v4 c_init={V4_C_PUCT_INIT} fpu_tree={V4_FPU_TREE} "
        f"against v5 {V5_C_PUCT_INIT} / {V5_FPU_TREE} (ratio "
        f"{Q_RATIO_V4_OVER_V5})")
    log("  Adjudication is DISABLED for both arms: -resign/-draw fire on a")
    log("  Q-derived score and the two nets have different Q distributions.")

    if not V4_MODEL.exists():
        raise Blocked(f"T2 is blocked: {V4_MODEL} is missing. There is no "
                      f"fallback — a v4-on-Python vs v5-on-C++ comparison "
                      f"measures the port, not capacity.")

    forced = args.force_all or "t2" in args.force

    def v4_arm(name: str, **clock) -> EngineArm:
        return EngineArm(name, V4_MODEL, c_puct_init=V4_C_PUCT_INIT,
                         fpu_tree=V4_FPU_TREE, **clock)

    def v5_arm(name: str, **clock) -> EngineArm:
        return EngineArm(name, V5_MODEL, c_puct_init=V5_C_PUCT_INIT,
                         fpu_tree=V5_FPU_TREE, **clock)

    # --- T2a: equal sims ---------------------------------------------------
    if state.done("t2", "t2a_equal_sims") and not forced \
            and "t2a_equal_sims" not in args.force_cell:
        log("  t2a_equal_sims: cached; skipping")
    else:
        log(f"  T2a — equal sims: {args.t2a_nodes} nodes, {args.t2a_games} paired "
            f"games, concurrency 2")
        arms = (v4_arm("v4-25.6M", nodes=args.t2a_nodes),
                v5_arm("v5-10.9M", nodes=args.t2a_nodes))
        cell = play(arms, T2_PGN_DIR / "t2a", event="t2a_equal_sims",
                    rounds=max(1, args.t2a_games // 2), concurrency=2,
                    adjudicate=False, sprt=None, args=args)
        cell["arm"] = "T2a — equal sims"
        cell["q_constants"] = {"v4": [V4_C_PUCT_INIT, V4_FPU_TREE],
                               "v5": [V5_C_PUCT_INIT, V5_FPU_TREE],
                               "ratio": Q_RATIO_V4_OVER_V5,
                               "provenance": Q_RATIO_PROVENANCE}
        # `elo` is A-over-B and A is v4, so this is v4's margin over v5 — the
        # capacity lower bound, positive when the bigger net wins.
        cell["v4_over_v5_elo"] = cell["elo"]["elo"]
        cell["interpretation"] = (
            "v4 ahead: capacity dominates; this is a LOWER bound on what a "
            "well-trained net at 25.6M would do"
            if (cell["elo"]["elo"] or 0) > 0 else
            "v5 ahead: AMBIGUOUS — training quality is confounded with "
            "capacity here and this is not evidence that capacity does not matter")
        state.put_cell("t2", "t2a_equal_sims", cell)

    # --- T2b: equal wall clock --------------------------------------------
    if state.done("t2", "t2b_equal_clock") and not forced \
            and "t2b_equal_clock" not in args.force_cell:
        log("  t2b_equal_clock: cached; skipping")
    else:
        log(f"  T2b — equal wall clock at {args.t2b_tc}, concurrency 1 "
            f"(MANDATORY: this arm is timed, so any GPU contention distorts the "
            f"result itself and not merely the schedule)")
        arms = (v4_arm("v4-25.6M", tc=args.t2b_tc),
                v5_arm("v5-10.9M", tc=args.t2b_tc))
        cell = play(arms, T2_PGN_DIR / "t2b", event="t2b_equal_clock",
                    rounds=max(1, args.t2b_games // 2), concurrency=1,
                    adjudicate=False, sprt=args.t2b_sprt, args=args)
        cell["arm"] = "T2b — equal wall clock"
        cell["q_constants"] = {"v4": [V4_C_PUCT_INIT, V4_FPU_TREE],
                               "v5": [V5_C_PUCT_INIT, V5_FPU_TREE],
                               "ratio": Q_RATIO_V4_OVER_V5,
                               "provenance": Q_RATIO_PROVENANCE}
        cell["v4_over_v5_elo"] = cell["elo"]["elo"]
        cell["note"] = ("integrates fresh-root cost, the reuse-regime discount "
                        "and Gate 4 automatically; this is the deployment "
                        "question answered without a training step")
        state.put_cell("t2", "t2b_equal_clock", cell)

    return state.data["t2"]


# ---------------------------------------------------------------------------
# Verdicts and the summary
# ---------------------------------------------------------------------------


def compute_verdicts(state: SuiteState, args) -> dict:
    """cost_elo, benefit_elo and the verdict per candidate.

        cost_elo(candidate) = doublings_lost(T1) x elo_per_doubling(T4)
        benefit_elo         = T2a equal-sims margin (a LOWER bound)
        verdict             = benefit_elo > cost_elo

    `doublings_lost` is log2(control throughput / candidate throughput) at the
    shipping configuration, computed on BOTH the fresh-root and reuse-heavy
    Gate 4 rows. The fresh-root number alone over-states the penalty: at 84.6%
    cache hit and 48% GPU share a bigger net barely slows endgames, so the
    game-average cost is milder. Both are carried and the fresh-root one is the
    conservative headline.
    """
    control = state.get_cell("t1", CONTROL.name) or {}
    base_fresh = (control.get("shipping_cell") or {}).get("sims_per_s")
    base_reuse = control.get("gate4_reuse_by_k", {}).get(str(SHIP_IN_FLIGHT))

    rung = t4_marginal_rate(state)
    rate = rung["elo_per_doubling"] if rung else None

    t2a = state.get_cell("t2", "t2a_equal_sims") or {}
    benefit = t2a.get("v4_over_v5_elo")
    benefit_ci = (t2a.get("elo") or {}).get("ci95")

    out: dict = {
        "elo_per_doubling": rate,
        "elo_per_doubling_source": (f"T4 rung {rung['nominal_low']}->"
                                    f"{rung['nominal_high']} nominal, "
                                    f"{rung['delivered_ratio']:.2f}x delivered"
                                    if rung else None),
        "benefit_elo": benefit,
        "benefit_elo_ci95": benefit_ci,
        "benefit_is_lower_bound": True,
        "control_fresh_sims_per_s": base_fresh,
        "control_reuse_sims_per_s": base_reuse,
        "candidates": {},
    }

    for candidate in CANDIDATES:
        cell = state.get_cell("t1", candidate.name) or {}
        if cell.get("status") == "blocked":
            out["candidates"][candidate.name] = {"status": "blocked",
                                                 "reason": cell.get("reason")}
            continue
        fresh = (cell.get("shipping_cell") or {}).get("sims_per_s")
        reuse = cell.get("gate4_reuse_by_k", {}).get(str(SHIP_IN_FLIGHT))
        entry: dict = {"status": cell.get("status", "missing")}
        if fresh and base_fresh:
            lost_fresh = math.log2(base_fresh / fresh)
            entry["doublings_lost_fresh"] = lost_fresh
            entry["cost_elo_fresh"] = lost_fresh * rate if rate else None
        if reuse and base_reuse:
            lost_reuse = math.log2(base_reuse / reuse)
            entry["doublings_lost_reuse"] = lost_reuse
            entry["cost_elo_reuse"] = lost_reuse * rate if rate else None
        entry["min_k_clearing_gate4"] = cell.get("min_k_clearing_gate4")
        entry["clears_gate4_at_shipping_k"] = bool(
            fresh and fresh >= GATE4_FLOOR)
        entry["pays_twice"] = bool(
            entry.get("min_k_clearing_gate4") and
            entry["min_k_clearing_gate4"] > SHIP_IN_FLIGHT)
        cost = entry.get("cost_elo_fresh")
        if cost is not None and benefit is not None:
            entry["verdict"] = "justified" if benefit > cost else "not-yet-justified"
            entry["margin_elo"] = benefit - cost
        else:
            entry["verdict"] = "incomplete"
        out["candidates"][candidate.name] = entry

    state.data["verdicts"] = out
    state.flush()
    return out


def write_summary(state: SuiteState, out_dir: Path) -> Path:
    d = state.data
    lines: list[str] = []
    add = lines.append

    add("# T1-T4 — Capacity Decision Suite")
    add("")
    add(f"Generated {utcnow()} · `tools/capacity_suite.py`")
    add("")
    meta = d.get("meta", {})
    add(f"- platform: {meta.get('platform', '?')}")
    add(f"- device: {meta.get('device', '?')}, torch {meta.get('torch', '?')}")
    add(f"- build: {meta.get('build', '?')}")
    add(f"- shipping config: W={SHIP_WORKERS}, K={SHIP_IN_FLIGHT}, "
        f"max_batch={SHIP_MAX_BATCH}, affinity={SHIP_AFFINITY} (C10b-3g)")
    add(f"- book and Syzygy: **OFF on every arm** (`--no-book --no-syzygy`); "
        f"both default ON since C11b and both bypass MCTS")
    add(f"- policy temperature: **not swept** (T=1.0 throughout). It IS a real "
        f"knob since C11b; not sweeping it is a scope decision (non-goal 3), "
        f"not an engine limitation. Dirichlet noise remains refused by the core.")
    add("")

    # --- verdicts ---------------------------------------------------------
    v = d.get("verdicts", {})
    add("## Verdict")
    add("")
    if v.get("elo_per_doubling") is None or v.get("benefit_elo") is None:
        add("> Incomplete — the verdict needs T1, T4 and T2a. See the per-test "
            "sections for what ran.")
        add("")
    add(f"- `elo_per_doubling` = **{_fmt(v.get('elo_per_doubling'), '+.1f')}** "
        f"({v.get('elo_per_doubling_source') or 'T4 not run'})")
    add(f"- `benefit_elo` = **{_fmt(v.get('benefit_elo'), '+.1f')}** "
        f"± {_fmt(v.get('benefit_elo_ci95'), '.1f')} (T2a equal sims, v4 over v5)")
    add("")
    degenerate = [name for name, cell in
                  list(state.cells("t2").items()) + list(state.cells("t4").items())
                  if (cell.get("elo") or {}).get("degenerate")]
    if degenerate:
        add(f"> **The verdict below is not usable.** These arms ran too few "
            f"games for their ELO to be a measurement rather than an artefact "
            f"of the sample: `{'`, `'.join(degenerate)}`. A clean sweep has "
            f"zero sample variance, so its CI collapses to ±0.0 and its point "
            f"estimate is whatever the score clamp allows — neither is a "
            f"number to act on.")
        add("")
    add("| candidate | MACs (brief / computed) | fresh sims/s | doublings lost "
        "| cost ELO | benefit ELO | margin | min K clearing Gate 4 | verdict |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for candidate in CANDIDATES:
        e = (v.get("candidates") or {}).get(candidate.name, {})
        t1 = state.get_cell("t1", candidate.name) or {}
        if e.get("status") == "blocked":
            add(f"| `{candidate.name}` | — | — | — | — | — | — | — | "
                f"**BLOCKED** |")
            continue
        fresh = (t1.get("shipping_cell") or {}).get("sims_per_s")
        computed = t1.get("macs_ratio_computed")
        add(f"| `{candidate.name}` | {candidate.macs_ratio_brief:.2f}x / "
            f"{(format(computed, '.2f') + 'x') if computed else '—'} "
            f"| {_fmt(fresh, ',.0f')} "
            f"| {_fmt(e.get('doublings_lost_fresh'), '.3f')} "
            f"| {_fmt(e.get('cost_elo_fresh'), '.1f')} "
            f"| {_fmt(v.get('benefit_elo'), '+.1f')} "
            f"| {_fmt(e.get('margin_elo'), '+.1f')} "
            f"| {e.get('min_k_clearing_gate4') or '—'} "
            f"| {e.get('verdict', '—')} |")
    add("")
    add("> **The interpretive asymmetry, stated next to the verdict.** T2 "
        "LOWER-BOUNDS the capacity benefit. v4 carries three training deficits "
        "that work against it — a RELU FFN, no final LayerNorm, and a "
        "human-imitation policy from mismatched teachers — so a well-trained "
        "net at that size would do better than v4 does. A candidate that fails "
        "the bar narrowly has therefore **not been ruled out; it has been ruled "
        "not-yet-justified.**")
    add("")
    add("> A candidate whose `min K clearing Gate 4` exceeds "
        f"{SHIP_IN_FLIGHT} **pays for capacity twice**: recovering the floor "
        "needs a larger K, and C10b-3e prices K=48-64 at about 10 points of "
        "top-move share. That second cost is not in the `cost ELO` column.")
    add("")

    # --- T3 ---------------------------------------------------------------
    add("## T3 — the label-noise floor")
    add("")
    t3 = state.get_cell("t3", "floor")
    if not t3:
        add("_not run_")
    else:
        p, val = t3["policy"], t3["value"]
        add(f"{t3['pairs']:,} positions carrying two policy-eligible blocks "
            f"within ±{t3['label_config']['depth_window']} depth, both targets "
            f"built through the full Pass B path "
            f"(`select_policy_block` → `build_policy_target`, T="
            f"{t3['label_config']['temperature']}, ε="
            f"{t3['label_config']['epsilon']}).")
        add("")
        add("| quantity | pairwise (two labels) | model (10.9M) | ratio |")
        add("|---|---:|---:|---:|")
        add(f"| policy KL (mean) | {p['pairwise_kl_mean']:.5f} | "
            f"{p['model_kl_mean']:.5f} | "
            f"{p['ratio_model_over_pairwise']:.2f}x |")
        add(f"| policy KL (median) | {p['pairwise_kl_median']:.5f} | "
            f"{p['model_kl_median']:.5f} | |")
        if val["pairwise_mse"] is not None:
            add(f"| value MSE | {val['pairwise_mse']:.6f} | "
                f"{val['model_mse']:.6f} | "
                f"{val['model_mse'] / val['pairwise_mse']:.2f}x |")
        add("")
        add(f"**{t3['verdict']}** — model KL / pairwise KL = "
            f"{p['ratio_model_over_pairwise']:.2f}x "
            f"(live at ≥{T3_LIVE_RATIO}x, corpus-limited at ≤{T3_CEILING_RATIO}x).")
        add("")
        add("Pairwise disagreement is an **upper** bound on the irreducible "
            "floor, not the floor: a model trained on many noisy labels averages "
            "them and can legitimately beat the pairwise number. The model-side "
            "half is what separates *the corpus is the ceiling* from *the model "
            "is nowhere near it*.")
        if t3["verdict"] == "corpus-limited":
            add("")
            add("> Capacity buys little here. The lever is **label quality**: "
                "raise `value_min_depth` / `policy_min_depth` in a corpus "
                "re-run and re-price against the feasibility scan's yield.")
    add("")

    # --- T1 ---------------------------------------------------------------
    add("## T1 — throughput ladder (random weights, shape only)")
    add("")
    add("| candidate | shape | MACs | fixed ms | µs/row | Gate 4 fresh @K=24 "
        "| ×floor | reuse-heavy @K=24 | K=24 | K=32 | K=48 | K=64 |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for candidate in CANDIDATES:
        cell = state.get_cell("t1", candidate.name)
        if not cell:
            continue
        if cell.get("status") == "blocked":
            add(f"| `{candidate.name}` | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        fit = cell.get("forward_fit", {})
        ship = cell.get("shipping_cell") or {}
        fresh_by_k = cell.get("gate4_fresh_by_k", {})
        reuse = cell.get("gate4_reuse_by_k", {}).get(str(SHIP_IN_FLIGHT))
        row = (f"| `{candidate.name}` | d{candidate.d_model}×{candidate.layers} "
               f"| {candidate.macs_ratio_brief:.2f}x "
               f"| {_fmt(fit.get('fixed_ms'), '.3f')} "
               f"| {_fmt(fit.get('per_row_us'), '.1f')} "
               f"| {_fmt(ship.get('sims_per_s'), ',.0f')} "
               f"| {_fmt(ship.get('sims_per_s', 0) / GATE4_FLOOR if ship else None, '.2f')} "
               f"| {_fmt(reuse, ',.0f')} ")
        for k in (24, 32, 48, 64):
            row += f"| {_fmt(fresh_by_k.get(str(k)), ',.0f')} "
        add(row + "|")
    add("")
    add(f"Gate 4 floor is {GATE4_FLOOR:,} delivered sims/s at a fresh root; "
        f"stretch {GATE4_STRETCH:,}. The **reuse-heavy** column is the one a "
        "game average is built from — at 84.6% cache hit and 48% GPU share a "
        "bigger net barely slows endgames, so the fresh-root figure alone "
        "over-states the capacity penalty.")
    add("")
    blocked = [c for c in CANDIDATES if c.blocked]
    if blocked:
        add("### Blocked candidates")
        add("")
        for candidate in blocked:
            add(f"- **`{candidate.name}`** — {candidate.blocked}")
        add("")

    # --- T4 ---------------------------------------------------------------
    add("## T4 — ELO per sim-doubling at deployment sims")
    add("")
    add("| rung | nominal | delivered (low → high) | delivered ratio | doublings "
        "| ELO (high over low) | ±95% | ELO/doubling | games | draw rate | "
        "searched | arena |")
    add("|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|")
    for name, cell in sorted(state.cells("t4").items()):
        r = cell.get("rung") or {}
        e = cell.get("elo") or {}
        fb = cell.get("fallback") or {}
        # A cell recorded before this column existed carries no `fallback` key,
        # and absence is not a clean bill of health — say so rather than
        # printing a tick nobody measured.
        searched = ("—" if not fb else
                    "yes" if not fb.get("total") else
                    f"**NO ({fb['total']})**")
        arenas = cell.get("arena_capacity") or {}
        arena = (" → ".join(f"{v:,}" if v else "default"
                            for v in arenas.values()) if arenas else "—")
        add(f"| `{name}` | {r.get('nominal_low')}→{r.get('nominal_high')} "
            f"| {_fmt(r.get('delivered_mean_low'), ',.0f')} → "
            f"{_fmt(r.get('delivered_mean_high'), ',.0f')} "
            f"| {_fmt(r.get('delivered_ratio'), '.2f')}x "
            f"| {_fmt(r.get('doublings_realised'), '.3f')} "
            f"| {_fmt(r.get('elo_high_over_low'), '+.1f')} "
            f"| {_fmt(r.get('elo_ci95'), '.1f')} "
            f"| {_fmt(r.get('elo_per_doubling'), '+.1f')} "
            f"| {e.get('games', '—')} "
            f"| {_fmt(e.get('draw_rate'), '.1%')} "
            f"| {searched} | {arena} |")
    add("")
    add(f"**searched** is the integrity column: every move came from MCTS and "
        f"none from the wrapper's exception handler. When `go` raises, the "
        f"wrapper still owes Cutechess a move and answers with the first legal "
        f"one — indistinguishable from a real move in the PGN, the result and the "
        f"ELO. A rung with any such move is recorded `corrupted` and re-runs "
        f"rather than being reported. Arenas are provisioned at "
        f"{ARENA_NODES_PER_SIM} nodes per sim of budget; the tree needs ~50 per "
        f"root visit, and the 1,200,000 default supplies only 37.5 at a 32,000 "
        f"budget, which is what corrupted the first run of this test.")
    add("")
    add("The x-axis is **delivered** simulations parsed from the engine's own "
        "`[search] … delivered=` telemetry, never the `nodes=` value: tree reuse "
        "makes `search_parallel` target absolute root visits, so a nominal 2× "
        "compresses into whatever the delivered ratio says. The decay between "
        "the two rungs is part of the answer and they are not averaged — the "
        "**higher** rung is what prices T1's cost.")
    add("")

    # --- T2 ---------------------------------------------------------------
    add("## T2 — v4 vs v5 on the same engine")
    add("")
    add(f"Q-unit constants: v4 `c_init={V4_C_PUCT_INIT}` / "
        f"`FPU_TREE={V4_FPU_TREE}` against v5 `{V5_C_PUCT_INIT}` / "
        f"`{V5_FPU_TREE}`, from the {Q_RATIO_V4_OVER_V5} width ratio "
        f"({Q_RATIO_PROVENANCE}). Both nets remain untuned on this engine, but "
        f"symmetrically so.")
    add("")
    add("Adjudication is **disabled on both arms**. `-resign score=600` and "
        "`-draw score=10` fire on `score cp = value_scale · atanh(q)`, which is "
        "derived from the value head; two nets with different Q distributions "
        "get adjudicated differently, so identical play would score differently. "
        "A Q-independent `-maxmoves` cap replaces them.")
    add("")
    add("| arm | regime | games | W-L-D | ELO (v4 over v5) | ±95% | SPRT | "
        "concurrency |")
    add("|---|---|---:|---|---:|---:|---|---:|")
    for name in ("t2a_equal_sims", "t2b_equal_clock"):
        cell = state.get_cell("t2", name)
        if not cell:
            continue
        e, r = cell.get("elo") or {}, cell.get("result") or {}
        add(f"| `{name}` | {cell.get('arm', '')} | {e.get('games', '—')} "
            f"| {r.get('wins', 0)}-{r.get('losses', 0)}-{r.get('draws', 0)} "
            f"| {_fmt(e.get('elo'), '+.1f')} | {_fmt(e.get('ci95'), '.1f')} "
            f"| {r.get('sprt_verdict', '—')} | {cell.get('concurrency')} |")
    add("")
    t2a = state.get_cell("t2", "t2a_equal_sims")
    if t2a:
        add(f"**T2a interpretation.** {t2a.get('interpretation', '')}")
        add("")

    # --- thermal ----------------------------------------------------------
    add("## Thermal drift")
    add("")
    th = d.get("thermal") or {}
    if not th:
        add("_the control cell was not re-measured; T1's numbers carry no drift "
            "check_")
    else:
        add(f"- control cell (d384×6 at the shipping config): "
            f"{_fmt(th.get('control_first_sims_per_s'), ',.0f')} → "
            f"{_fmt(th.get('control_last_sims_per_s'), ',.0f')} sims/s "
            f"({_fmt(th.get('throughput_drift'), '.1%')} drift, tolerance "
            f"{th.get('tolerance', 0):.0%})")
        clocks = th.get("sm_clocks_mhz") or []
        if th.get("control_peak_clock_first"):
            add(f"- control cell peak SM clock under load: "
                f"{th['control_peak_clock_first']:.0f} → "
                f"{th['control_peak_clock_last']:.0f} MHz "
                f"({_fmt(th.get('sm_clock_drift'), '.1%')} drift). C10b logged "
                f"2857–2917 MHz.")
        if clocks:
            add(f"- all T1 rows under load spanned {min(clocks):.0f}–"
                f"{max(clocks):.0f} MHz across {len(clocks)} samples. That "
                f"spread is the **DVFS ramp across batch sizes**, not drift: at "
                f"batch 8 the device is busy for well under a millisecond and "
                f"then idles inside `torch.cuda.synchronize()`. Samples taken "
                f"between cells are excluded entirely — an idle RTX 5070 reads "
                f"~750 MHz.")
        if th.get("t1_suspect"):
            add("")
            add("> **Every T1 number in this report is suspect.** The control "
                "cell moved by more than the tolerance between the first and "
                "last measurement, and drift is indistinguishable from capacity "
                "cost in a single pass. Re-run T1 on a cold box before acting "
                "on the verdict table.")
    add("")

    path = out_dir / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _fmt(value, spec: str) -> str:
    if value is None:
        return "—"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# --check: the §2 inspection, re-runnable
# ---------------------------------------------------------------------------


def run_check() -> int:
    """Re-run the pre-flight inspection §2 asks for, and report each answer.

    Every one of these was verified before the harness was written; this exists
    so the answers are re-checkable on another box rather than trusted from a
    docstring.
    """
    import torch
    from playing.v6 import evaluator as live
    from playing.v6.playv6 import EngineConfig
    import guofish_core

    ok = True
    build = require_release_build()
    log(f"build: {build}")
    log(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'}")

    for label, path in (("v5 (10.9M)", V5_MODEL), ("v4 (25.6M)", V4_MODEL)):
        log("")
        log(f"--- {label}: {path}")
        if not path.exists():
            log("  MISSING")
            ok = False
            continue
        model, device = live.load_default_model(path)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        cfg = getattr(model, "config", None)
        log(f"  architecture={type(model).__name__} params={params:.2f}M "
            f"seq_length={model.seq_length}")
        if cfg is not None:
            log(f"  config: d_model={cfg.d_model} x{cfg.num_layers} nhead={cfg.nhead} "
                f"activation={cfg.activation} final_norm={cfg.final_norm}")
        else:
            log("  config: none — legacy V2, RELU FFN and no final_norm pinned "
                "by the class (both unrecoverable from a state dict)")
        log(f"  value_scale carried by the checkpoint: "
            f"{getattr(model, 'value_scale', None)}")
        evaluator = live.TorchEvaluator(model, device, SHIP_MAX_BATCH, graphs=True)
        log(f"  capture: {evaluator.graph_report.describe()}")
        log(f"  ladder:  {evaluator.graph_sizes}  (batch shapes, resolved against "
            f"max_batch — NOT a function of d_model)")
        search = guofish_core.ReplaySearchQ32(EngineConfig().to_search_config())
        search.set_evaluator(evaluator.core)
        search.set_position("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/"
                            "PP1N1PPP/R2Q1RK1 w - - 0 9")
        pc = guofish_core.ParallelConfig(workers=SHIP_WORKERS,
                                         in_flight=SHIP_IN_FLIGHT,
                                         max_batch=SHIP_MAX_BATCH,
                                         affinity=SHIP_AFFINITY)
        started = time.perf_counter()
        stats = search.search_parallel(4000, pc)
        wall = time.perf_counter() - started
        par, audit = search.parallel_stats(), search.audit()
        try:
            assert_core_cell(par, audit, stats, fixed_budget=True,
                             where=f"--check {label}")
            log(f"  search: best={stats['best_move']} "
                f"{par['delivered'] / wall:,.0f} delivered sims/s, guards clean")
        except Blocked as exc:
            log(f"  GUARD FAILED: {exc}")
            ok = False
        search.set_evaluator(None)
        evaluator.close()
        del model, evaluator, search
        torch.cuda.empty_cache()

    log("")
    log("wrapper flags derived from the current signature "
        "(playing/v6/playv6.add_config_arguments):")
    log("  --model --threads --max-outstanding --max-batch --affinity "
        "--sim-cap --fixed-sims")
    log("  --c-puct-init --fpu-tree --policy-temperature --value-scale")
    log("  --no-book --book-path --book-seed --no-syzygy --syzygy-path")
    log("  book and Syzygy DEFAULT ON; the shipping W/K/max_batch/affinity IS "
        "the default")
    log("")
    log(f"Smolgen: NOT implementable here — ModelConfig(smolgen=True) raises "
        f"NotImplementedError; the T1 row is recorded as blocked.")
    log("")
    log("PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


ORDER = ("t3", "t1", "t4", "t2")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tests", nargs="+", default=list(ORDER) + ["summary"],
                        choices=list(ORDER) + ["summary", "thermal"],
                        help="T3 first because it is 20 minutes and can "
                             "invalidate everything after it; T1 before the "
                             "matches; T4 before T2 because T2's interpretation "
                             "needs the exchange rate")
    parser.add_argument("--check", action="store_true",
                        help="re-run the §2 pre-flight inspection and exit")
    parser.add_argument("--force", nargs="*", default=[], choices=list(ORDER),
                        help="re-run these tests even if cached")
    parser.add_argument("--force-cell", nargs="*", default=[],
                        help="re-run these named cells even if cached")
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument("--skip-gates", action="store_true",
                        help="do not stop on a failed inter-test gate (§8). The "
                             "gate verdicts are still recorded either way.")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--thermal-tolerance", type=float, default=0.03)
    parser.add_argument("--heartbeat", type=float, default=DEFAULT_HEARTBEAT_SECONDS,
                        metavar="SECONDS",
                        help="how often a running child process reports that it "
                             "is still alive (default: %(default)gs, 0 to "
                             "disable). Every line of this suite's log is "
                             "streamed to <out-dir>/run.log as it is produced, "
                             "so the run can be followed with `tail -f`")

    g = parser.add_argument_group("T3 — label-noise floor")
    g.add_argument("--t3-max-pairs", type=int, default=20_000)
    g.add_argument("--t3-max-lines", type=int, default=3_000_000)
    g.add_argument("--t3-depth-window", type=int, default=2,
                   help="max |depth1 - depth2|; matching depth is what keeps "
                        "genuine depth improvement out of the measurement")

    g = parser.add_argument_group("T1 — throughput ladder")
    g.add_argument("--t1-candidates", nargs="*", default=[],
                   choices=[c.name for c in CANDIDATES],
                   help="measure only these shapes (default: all of §5's table)")
    g.add_argument("--t1-batches", type=int, nargs="+",
                   default=[8, 16, 32, 64, 128, 256])
    g.add_argument("--t1-iters", type=int, default=200)
    g.add_argument("--t1-k-sweep", type=int, nargs="+", default=[24, 32, 48, 64])
    g.add_argument("--t1-gate4-sims", type=int, default=20_000)
    g.add_argument("--t1-gate4-repeats", type=int, default=5)
    g.add_argument("--t1-reuse-plies", type=int, default=6)

    g = parser.add_argument_group("T4 — ELO per sim-doubling")
    g.add_argument("--t4-low", type=int, default=8_000)
    g.add_argument("--t4-mid", type=int, default=16_000)
    g.add_argument("--t4-high", type=int, default=32_000)
    g.add_argument("--t4-games", type=int, default=150)

    g = parser.add_argument_group("T2 — v4 vs v5")
    g.add_argument("--t2a-nodes", type=int, default=4_000)
    g.add_argument("--t2a-games", type=int, default=200)
    g.add_argument("--t2b-tc", type=str, default="10+0.1")
    g.add_argument("--t2b-games", type=int, default=200)
    g.add_argument("--t2b-sprt", type=str,
                   default="elo0=0 elo1=30 alpha=0.05 beta=0.05",
                   help="SPRT is a DECISION and belongs only on the directional "
                        "arm; T4 needs a point estimate with a CI and gets none")

    g = parser.add_argument_group("match shape")
    g.add_argument("--opening-plies", type=int, default=16)
    g.add_argument("--maxmoves", type=int, default=200,
                   help="Q-independent termination for the arms that run with "
                        "score adjudication disabled")
    g.add_argument("--timemargin", type=int, default=300_000)

    args = parser.parse_args(argv)
    args.force = list(args.force)
    args.force_cell = list(args.force_cell)

    if args.check:
        return run_check()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # BEFORE anything that can fail or block. The release-build check and the
    # CUDA check below both return non-zero without raising, and their reasons
    # belong in the file a reader will look at rather than only on a stdout
    # nobody was watching at 3 a.m.
    global HEARTBEAT_SECONDS
    HEARTBEAT_SECONDS = args.heartbeat
    open_run_log(out_dir / "run.log")
    log(f"log      : {out_dir / 'run.log'} (streamed; tail -f to follow)")
    log(f"heartbeat: {args.heartbeat:g}s"
        if args.heartbeat > 0 else "heartbeat: disabled")

    state = SuiteState(out_dir / "results.json")

    try:
        build = require_release_build()
    except Blocked as exc:
        log(f"BLOCKED: {exc}")
        return 1

    import torch
    if not torch.cuda.is_available():
        log("BLOCKED: CUDA is not available. Every figure this suite produces "
            "is a GPU figure; there is no CPU path that has been measured.")
        return 1

    state.data["meta"] = {
        "started_utc": utcnow(),
        "platform": f"{platform.system()} {platform.machine()} {platform.platform()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "build": build,
        "shipping_config": {"workers": SHIP_WORKERS, "in_flight": SHIP_IN_FLIGHT,
                            "max_batch": SHIP_MAX_BATCH,
                            "affinity": SHIP_AFFINITY},
        "book_syzygy": "OFF on every arm (--no-book --no-syzygy); both default "
                       "ON since C11b and both bypass MCTS",
        "policy_temperature": "1.0 throughout; not swept (scope, not a core "
                              "limitation — C11b made it a real knob)",
        "dirichlet": "refused by the core (playv6.UNSUPPORTED_IN_CORE)",
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    state.flush()

    log(f"out-dir  : {out_dir}")
    log(f"platform : {state.data['meta']['platform']}")
    log(f"device   : {state.data['meta']['device']}, torch {torch.__version__}")
    log(f"build    : {build['compiler']}, asan={build['asan']}")
    log(f"order    : {' -> '.join(t for t in ORDER if t in args.tests)}"
        + (" -> thermal -> summary" if "summary" in args.tests else ""))

    try:
        if "t3" in args.tests:
            started = time.monotonic()
            t3 = run_t3(args, state)
            log(f"  T3 complete in {_hms(time.monotonic() - started)}")
            if not t3.get("gate_passed"):
                msg = (f"GATE: T3 says '{t3['verdict']}' (model KL / pairwise KL "
                       f"= {t3['policy']['ratio_model_over_pairwise']:.2f}x). "
                       f"The corpus, not capacity, is the binding constraint; "
                       f"the lever is label quality.")
                log("")
                log(msg)
                if not args.skip_gates:
                    log("Stopping per §8. Pass --skip-gates to run T1-T2 anyway.")
                    write_summary(state, out_dir)
                    return 2

        if "t1" in args.tests:
            started = time.monotonic()
            run_t1(args, state, out_dir)
            log(f"  T1 complete in {_hms(time.monotonic() - started)}")
            cleared = [c for c in state.cells("t1").values()
                       if c.get("status") == "ok"
                       and c.get("min_k_clearing_gate4") is not None
                       and c["min_k_clearing_gate4"] <= 32
                       and c.get("candidate") != CONTROL.name]
            if not cleared:
                log("")
                log("GATE: no capacity candidate clears Gate 4 at K<=32. Every "
                    "candidate would have to buy its throughput back with a "
                    "larger K, which C10b-3e prices at ~10 points of top-move "
                    "share — paying for capacity twice.")
                if not args.skip_gates:
                    log("Stopping per §8. Pass --skip-gates to continue.")
                    write_summary(state, out_dir)
                    return 2

        if "t4" in args.tests:
            started = time.monotonic()
            run_t4(args, state, out_dir)
            log(f"  T4 complete in {_hms(time.monotonic() - started)}")

        if "t2" in args.tests:
            started = time.monotonic()
            run_t2(args, state, out_dir)
            log(f"  T2 complete in {_hms(time.monotonic() - started)}")

        if "thermal" in args.tests or "summary" in args.tests:
            if any(state.cells("t1")):
                run_thermal_control(args, state, out_dir)

        # The success tail lives INSIDE the try so that `finally` closes the log
        # after these lines rather than before them.
        compute_verdicts(state, args)
        path = write_summary(state, out_dir)
        log("")
        log(f"results: {state.path}")
        log(f"summary: {path}")
        return 0

    except Blocked as exc:
        log("")
        log(f"BLOCKED: {exc}")
        compute_verdicts(state, args)
        path = write_summary(state, out_dir)
        log(f"partial results: {state.path}")
        log(f"partial summary: {path}")
        return 1
    except KeyboardInterrupt:
        # Overnight this is the likely ending, and it used to leave no log at
        # all. Every completed cell is already flushed to results.json by
        # `put_cell`, so saying so is the useful thing to record.
        log("")
        log("INTERRUPTED (Ctrl-C). Every completed cell is already in "
            "results.json and will be skipped when this is re-run.")
        return 130
    finally:
        log(f"total wall time {_hms(time.monotonic() - _STARTED)}")
        close_run_log()


if __name__ == "__main__":
    raise SystemExit(main())
