"""Automated evaluation gates for the 90M x 4 regime.

Three things the brief asks the pipeline to decide for itself, with no human in
the loop:

  seen/unseen gap   Does the model still overfit its own training records? At
                    4.5x the unique data this should collapse toward zero.
  head-to-head      200 paired games at 12k sims against the legacy 20M ep9
                    checkpoint, launched on run completion.
  the bar           >=5% KL reduction AND >=100 ELO gain, pre-registered here in
                    code so the threshold cannot be adjusted after seeing the
                    result.

WHAT "SEEN/UNSEEN" MEANS HERE, AND WHY IT IS MEASURED TWICE
===========================================================
The number wanted is the cost of exposure: how much better the model scores a
record because it has already trained on it. That demands two slices which
differ in exposure and in NOTHING ELSE - same distribution, same collate, same
determinism.

During epoch 1 that is exactly available for free, and does not cost a single
withheld record. The epoch's sampler order is a permutation fixed before the
first step, so at any point mid-epoch the records at the FRONT of that order
have been trained on and the ones at the BACK have not. Both are uniform draws
from the same corpus. `SeenUnseenProbe` takes a slice of each and measures at
`--gap-at-frac` through epoch 1 (default 0.25, comfortably clear of both ends).

That construction dies at the epoch-1 boundary - from epoch 2 every record is
seen - so it is NOT the only measurement. The durable one is the fixed train
probe against the val split, reported at every epoch boundary. That is the
generalisation gap in the usual sense and is what the ~2% quoted for 20M ep9 in
tools/capacity_suite.py refers to, so it is the number that stays comparable
across runs and across the whole 4 epochs.

Both run with dropout off and the colour mirror off. A stochastic probe would
put augmentation noise straight into a difference of two small numbers.

WHY THE KL THRESHOLD RE-EVALUATES THE BASELINE
==============================================
"5 percent reduction in KL" is a comparison, so it needs both terms measured
the same way. Reading the new model's KL off this run and the old model's KL
off a months-old log would compare two different val splits (the 90M rebuild
re-drew them: 452,405 val records against the 20M corpus's 147,963) through two
different code paths. So the baseline checkpoint is loaded and scored on THIS
run's val loader, and both figures are reported side by side.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from losses import compute_losses

# The pre-registered bar. Named constants, not literals at the comparison site,
# so `git log -p gates.py` shows any move.
KL_REDUCTION_REQUIRED = 0.05        # >= 5% lower KL than the baseline
ELO_GAIN_REQUIRED = 100.0           # >= 100 ELO, on the point estimate


# ---------------------------------------------------------------------------
# deterministic loss on an explicit set of records
# ---------------------------------------------------------------------------
@torch.no_grad()
def probe_metrics(model, dataset, indices, collate, device, amp_ctx, *,
                  batch_size: int = 1024, workers: int = 4,
                  prefetch: int = 2, policy_weight: float = 1.0,
                  value_weight: float = 1.0) -> dict:
    """Policy KL / value MSE over exactly the records in `indices`.

    `indices` is handed to DataLoader as the sampler, so the records are read in
    the order given and nothing else is touched. Eval mode throughout: dropout
    off, and `collate` is expected to be the val collate (mirror 0.0).
    """
    idx = [int(i) for i in indices]
    if not idx:
        return {"n": 0, "n_policy": 0, "policy_kl": float("nan"),
                "value_mse": float("nan"), "total_loss": float("nan")}

    was_training = model.training
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, sampler=idx,
                        num_workers=workers, drop_last=False,
                        collate_fn=collate, pin_memory=True,
                        persistent_workers=False,
                        prefetch_factor=prefetch if workers else None)
    kl_sum = se_sum = 0.0
    n = n_pol = 0
    try:
        for batch in loader:
            b = {k: (v.to(device, non_blocking=True)
                     if torch.is_tensor(v) else v) for k, v in batch.items()}
            with amp_ctx():
                logits, value = model(b["tokens"])
            parts = compute_losses(logits.float(), value.float(), b)
            kl_sum += parts.policy_kl_total
            se_sum += parts.value_se_total
            n += parts.n_samples
            n_pol += parts.n_policy
    finally:
        del loader
        if was_training:
            model.train()

    policy_kl = kl_sum / n_pol if n_pol else float("nan")
    value_mse = se_sum / max(1, n)
    return {
        "n": n,
        "n_policy": n_pol,
        "policy_kl": policy_kl,
        "value_mse": value_mse,
        "total_loss": policy_weight * policy_kl + value_weight * value_mse,
    }


def relative_gap(unseen: dict, seen: dict) -> dict:
    """(unseen - seen) / seen, per component. Positive == the model is better on
    what it has already trained on, which is the overfitting direction."""
    out = {}
    for key in ("policy_kl", "value_mse", "total_loss"):
        s, u = seen.get(key, float("nan")), unseen.get(key, float("nan"))
        out[f"{key}_seen"] = s
        out[f"{key}_unseen"] = u
        out[f"{key}_gap_abs"] = u - s
        out[f"{key}_gap_rel"] = ((u - s) / s
                                 if s and math.isfinite(s) and s != 0
                                 else float("nan"))
    return out


# ---------------------------------------------------------------------------
# the epoch-1 seen/unseen probe
# ---------------------------------------------------------------------------
class SeenUnseenProbe:
    """Two same-distribution slices of epoch 0's sampler order.

    `seen` is the head of the order and `unseen` is the tail, so the only thing
    separating them at measurement time is whether training has reached them
    yet. Sizes are equal, so the two estimates carry the same standard error and
    their difference is not dominated by one side being noisier.
    """

    def __init__(self, sampler, n_probe: int, measure_at_frac: float = 0.25):
        self.n_total = int(sampler.n)
        self.n_probe = int(min(n_probe, self.n_total // 4))
        self.measure_at_frac = float(measure_at_frac)
        head, tail = sampler.epoch_head_tail(0, self.n_probe)
        self.seen_idx = head
        self.unseen_idx = tail
        self.done = False

    @property
    def valid(self) -> bool:
        """The construction only holds when both slices sit strictly inside the
        untouched/touched split at measurement time."""
        if self.n_probe <= 0:
            return False
        frac = self.n_probe / self.n_total
        return frac < self.measure_at_frac < 1.0 - frac

    def due(self, epoch: int, epoch_frac: float) -> bool:
        return (not self.done and self.valid and epoch == 0
                and epoch_frac >= self.measure_at_frac)

    def measure(self, model, dataset, collate, device, amp_ctx, **kw) -> dict:
        """-> the gap dict. Marks itself done; only ever runs once."""
        self.done = True
        t0 = time.time()
        seen = probe_metrics(model, dataset, self.seen_idx, collate, device,
                             amp_ctx, **kw)
        unseen = probe_metrics(model, dataset, self.unseen_idx, collate, device,
                               amp_ctx, **kw)
        out = relative_gap(unseen, seen)
        out.update({
            "n_probe": self.n_probe,
            "measured_at_frac": self.measure_at_frac,
            "seconds": time.time() - t0,
            "construction": "epoch-0 order: head=trained, tail=not yet reached",
        })
        return out


def format_gap_report(gap: dict, title: str) -> str:
    """Both components and the total, with the direction spelled out - a bare
    signed number here is easy to read backwards."""
    lines = [f"\n{title}"]
    lines.append(f"  {'':<12} {'seen':>12} {'unseen':>12} {'gap':>12} {'rel':>9}")
    for key, label in (("policy_kl", "policy KL"),
                       ("value_mse", "value MSE"),
                       ("total_loss", "total")):
        s = gap.get(f"{key}_seen", float("nan"))
        u = gap.get(f"{key}_unseen", float("nan"))
        d = gap.get(f"{key}_gap_abs", float("nan"))
        r = gap.get(f"{key}_gap_rel", float("nan"))
        lines.append(f"  {label:<12} {s:>12.5f} {u:>12.5f} {d:>+12.5f} "
                     f"{r * 100:>+8.2f}%")
    rel = gap.get("total_loss_gap_rel", float("nan"))
    if math.isfinite(rel):
        if rel > 0.01:
            verdict = (f"unseen records cost {rel * 100:.2f}% more - the model "
                       f"still fits its own training set better")
        elif rel < -0.01:
            verdict = (f"unseen records score BETTER by {-rel * 100:.2f}%; at "
                       f"this size that is sampling noise, not generalisation")
        else:
            verdict = ("gap is within +-1% - no measurable advantage from "
                       "having seen a record")
        lines.append(f"  -> {verdict}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the baseline checkpoint, scored on this run's val loader
# ---------------------------------------------------------------------------
def score_baseline(ckpt_path: Path, val_ds, collate, device, amp_ctx, *,
                   batch_size: int = 1024, workers: int = 4,
                   prefetch: int = 2, policy_weight: float = 1.0,
                   value_weight: float = 1.0) -> dict:
    """Load the legacy checkpoint and measure its KL on THIS run's val split.

    Architecture comes out of the checkpoint (`config_from_checkpoint` falls
    back to inferring it from the state dict), so a baseline of a different
    shape still scores correctly.
    """
    from model_v5 import load_from_checkpoint

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = load_from_checkpoint(ckpt).to(device)
    model.eval()
    try:
        out = probe_metrics(model, val_ds, range(len(val_ds)), collate, device,
                            amp_ctx, batch_size=batch_size, workers=workers,
                            prefetch=prefetch, policy_weight=policy_weight,
                            value_weight=value_weight)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out["checkpoint"] = str(ckpt_path)
    out["num_params"] = sum(
        v.numel() for k, v in ckpt.get("model_state_dict", ckpt).items()
        if torch.is_tensor(v))
    return out


# ---------------------------------------------------------------------------
# head to head
# ---------------------------------------------------------------------------
# From tools/run_head2head.ps1, which these mirror deliberately: an arm that
# silently inherited a changed default would be unreadable a month later, so the
# shipping configuration is passed explicitly on every engine block.
SHIP_WORKERS = 1
SHIP_OUTSTANDING = 24
SHIP_MAX_BATCH = 128
SHIP_AFFINITY = "none"
CPUCT_INIT = 1.43           # v5-architecture Q denomination, both arms
FPU_TREE = 0.3
ARENA_NODES_PER_SIM = 75    # capacity_suite.py: 37.5/sim exhausted the arena
                            # mid-search and 647 moves came back as first-legal
                            # fallbacks, which inverted the sign of an ELO figure


@dataclass
class MatchResult:
    event: str
    exit_code: int
    seconds: float
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games: int = 0
    score: float = float("nan")

    # The headline pair the verdict is judged on, and where they came from.
    elo: float = float("nan")
    elo_lo: float = float("nan")
    elo_hi: float = float("nan")
    elo_source: str = ""

    # ordo (-a 0 -A <baseline>): the house authority for ratings and CIs.
    ordo_elo: float = float("nan")
    ordo_error: float = float("nan")
    ordo_cfs: float = float("nan")          # confidence for superiority, %
    ordo_white_advantage: float = float("nan")
    ordo_draw_rate: float = float("nan")
    ordo_error_note: str = ""

    # cutechess's own figure, free with the match.
    cc_elo: float = float("nan")
    cc_margin: float = float("nan")
    cc_los: float = float("nan")
    cc_draw_ratio: float = float("nan")

    # Closed form on the trinomial score - a consistency check, not the answer.
    closed_elo: float = float("nan")
    closed_lo: float = float("nan")
    closed_hi: float = float("nan")
    estimator_disagreement: float = float("nan")

    pgn: str = ""
    log: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def elo_from_score(score: float) -> float:
    """Logistic ELO. +-inf at a clean sweep, which is honest: 200-0 does not
    bound the difference from above."""
    if not 0.0 < score < 1.0:
        return math.inf if score >= 1.0 else -math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def elo_with_ci(wins: int, draws: int, losses: int, z: float = 1.96):
    """Point estimate and a normal-approximation interval on the score.

    Trinomial: the per-game scores are treated as iid draws from {1, 0.5, 0}.
    Paired openings make the true variance LOWER than this, so the interval is
    conservative - which is the right direction for a threshold that has to be
    cleared rather than merely touched.
    """
    n = wins + draws + losses
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    score = (wins + 0.5 * draws) / n
    var = (wins * (1.0 - score) ** 2
           + draws * (0.5 - score) ** 2
           + losses * (0.0 - score) ** 2) / n
    se = math.sqrt(var / n) if var > 0 else 0.0
    return (elo_from_score(score),
            elo_from_score(min(1.0, max(0.0, score - z * se))),
            elo_from_score(min(1.0, max(0.0, score + z * se))))


_SCORE_RE = re.compile(
    r"^Score of (.+?) vs (.+?): (\d+) - (\d+) - (\d+)\s+\[([\d.]+)\]\s+(\d+)")

# cutechess-cli 1.4.0, immediately after the colour-split lines:
#   Elo difference: 636.4 +/- nan, LOS: 100.0 %, DrawRatio: 5.0 %
# The margin is legitimately `nan` on a degenerate result (a sweep), so the
# number and its error are parsed separately and either may be absent.
_ELO_RE = re.compile(
    r"^Elo difference:\s*(-?[\d.]+|-?inf|nan)\s*\+/-\s*(-?[\d.]+|-?inf|nan)"
    r"(?:,\s*LOS:\s*([\d.]+)\s*%)?(?:,\s*DrawRatio:\s*([\d.]+)\s*%)?")


def _to_float(text) -> float:
    if text is None:
        return float("nan")
    t = str(text).strip().strip('"')
    if t in ("", "-", "----", "nan"):
        return float("nan")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def parse_cutechess_elo(log_text: str, invert: bool = False) -> dict:
    """cutechess's own Elo difference / LOS / draw ratio, from the LAST such line.

    Oriented like the score line it follows: from the FIRST-named engine's point
    of view. `invert` flips it for the case where the candidate was named second.
    """
    last = None
    for line in log_text.splitlines():
        m = _ELO_RE.match(line.strip())
        if m:
            last = m
    if last is None:
        return {}
    elo = _to_float(last.group(1))
    los = _to_float(last.group(3))
    if invert:
        elo = -elo
        if math.isfinite(los):
            los = 100.0 - los
    return {
        "cc_elo": elo,
        "cc_margin": _to_float(last.group(2)),
        "cc_los": los,
        "cc_draw_ratio": _to_float(last.group(4)),
    }


# ---------------------------------------------------------------------------
# ordo - the house authority for ratings and confidence intervals
# ---------------------------------------------------------------------------
# Flags match benchmarking/engine/util/parse_cpuct_results.py:run_ordo, minus
# the Stockfish anchor, which does not apply to a two-engine match:
#   -D  fit the draw rate      -W  fit the white advantage
#   -s  simulations for the error bars     -J  confidence-for-superiority column
ORDO_SIMULATIONS = 1000
ORDO_DISAGREEMENT_WARN = 25.0       # ELO; see run_head2head


def find_ordo(repo: Path) -> "Path | None":
    for candidate in (repo / "ordo-win64.exe",
                      repo / "benchmarking" / "engine" / "games" / "ordo-win64.exe"):
        if candidate.exists():
            return candidate
    return None


def run_ordo_diff(repo: Path, pgn: Path, candidate_name: str,
                  baseline_name: str, *, simulations: int = ORDO_SIMULATIONS,
                  timeout: float = 900.0, log=print) -> dict:
    """Rate a two-engine PGN with the BASELINE anchored at 0.

    With only two players the anchor is a pure additive offset on the whole
    scale, so pinning the baseline at 0 makes the candidate's rating the ELO
    DIFFERENCE directly - sign included - rather than something to subtract
    afterwards. Verified against benchmarking/engine/games/v5:
    `-a 0 -A Guofish5-10.9M-BIG` puts SMALL at -102.0 +/- 50.5 over 100 games.

    Returns {} and logs the reason on any failure. ordo legitimately REFUSES
    some inputs: a clean sweep leaves it with all-wins/all-losses players, it
    purges them, the database stops being connected, and it writes no CSV at
    all. That is a real outcome of a real match, not a bug, so it must not take
    the run down - the caller falls back to cutechess's own figure.
    """
    ordo = find_ordo(repo)
    if ordo is None:
        return {"ordo_error_note": "ordo binary not found"}
    if not pgn.exists():
        return {"ordo_error_note": f"PGN missing: {pgn}"}

    csv_out = pgn.with_suffix(".ordo.csv")
    cmd = [str(ordo), "-q", "-a", "0", "-A", baseline_name, "-D", "-W",
           "-s", str(simulations), "-J", "-p", str(pgn), "-c", str(csv_out)]
    try:
        proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout)
    except Exception as exc:                            # noqa: BLE001
        return {"ordo_error_note": f"{type(exc).__name__}: {exc}"}

    stdout = proc.stdout or ""
    out: dict = {}
    # White advantage / draw rate come off stdout; they are diagnostics, not
    # ratings, and a match whose draw rate is ~100% has told you nothing.
    m = re.search(r"White advantage\s*=\s*(-?[\d.]+)", stdout)
    if m:
        out["ordo_white_advantage"] = _to_float(m.group(1))
    m = re.search(r"Draw rate \(equal opponents\)\s*=\s*([\d.]+)", stdout)
    if m:
        out["ordo_draw_rate"] = _to_float(m.group(1))

    if not csv_out.exists() or csv_out.stat().st_size == 0:
        note = "ordo produced no CSV"
        if "not well connected" in stdout:
            note += " (database not well connected - a sweep or too few games)"
        out["ordo_error_note"] = note
        log(f"  [ordo] {note}")
        return out

    import csv as _csv

    try:
        with open(csv_out, "r", encoding="utf-8", errors="replace") as fh:
            rows = list(_csv.DictReader(fh))
    except Exception as exc:                            # noqa: BLE001
        out["ordo_error_note"] = f"CSV unreadable: {exc}"
        return out

    def key(fields, want):
        return next((k for k in (fields or []) if k.strip().lower() == want), None)

    fields = rows[0].keys() if rows else []
    k_name = key(fields, "player")
    k_rating = key(fields, "rating")
    k_error = key(fields, "error")
    k_cfs = next((k for k in (fields or []) if k.strip().lower().startswith("cfs")), None)
    if not (k_name and k_rating):
        out["ordo_error_note"] = f"unexpected ordo CSV header: {list(fields)}"
        return out

    for row in rows:
        if (row.get(k_name) or "").strip() != candidate_name:
            continue
        out["ordo_elo"] = _to_float(row.get(k_rating))
        # The ANCHORED player's error is a literal '-'; the candidate's is a
        # number. Both go through _to_float, which maps '-' to nan.
        out["ordo_error"] = _to_float(row.get(k_error)) if k_error else float("nan")
        out["ordo_cfs"] = _to_float(row.get(k_cfs)) if k_cfs else float("nan")
        return out

    out["ordo_error_note"] = (
        f"candidate {candidate_name!r} not in ordo output "
        f"(players: {[ (r.get(k_name) or '').strip() for r in rows ]})")
    return out


def parse_cutechess_score(log_text: str, candidate_name: str):
    """Last 'Score of A vs B: W - L - D [s] N' line, oriented to the candidate.

    Orientation is not cosmetic: cutechess reports from the FIRST-named engine's
    point of view, and reading a 60-140 loss as a win is the single easiest way
    to ship a worse net. If the candidate is named second, the counts are
    swapped here.
    """
    last = None
    for line in log_text.splitlines():
        m = _SCORE_RE.match(line.strip())
        if m:
            last = m
    if last is None:
        return None
    a, b, w, l, d, _s, n = (last.group(1).strip(), last.group(2).strip(),
                            int(last.group(3)), int(last.group(4)),
                            int(last.group(5)), last.group(6),
                            int(last.group(7)))
    if candidate_name == a:
        return w, d, l, n
    if candidate_name == b:
        return l, d, w, n
    return None


def _guofish_arm(repo: Path, name: str, model: Path, nodes: int, arena: int,
                 stderr_path: Path) -> list:
    return [
        "-engine", f"name={name}", "cmd=python",
        "arg=-u", "arg=playing/uci_wrapper_v6.py",
        "arg=--model", f"arg={model}",
        "arg=--threads", f"arg={SHIP_WORKERS}",
        "arg=--max-outstanding", f"arg={SHIP_OUTSTANDING}",
        "arg=--max-batch", f"arg={SHIP_MAX_BATCH}",
        "arg=--affinity", f"arg={SHIP_AFFINITY}",
        "arg=--c-puct-init", f"arg={CPUCT_INIT}",
        "arg=--fpu-tree", f"arg={FPU_TREE}",
        # Both default ON since C11b and both BYPASS MCTS, which would nullify
        # -openings and put zero-simulation moves into the telemetry.
        "arg=--no-book", "arg=--no-syzygy",
        "arg=--arena-capacity", f"arg={arena}",
        f"dir={repo}", "proto=uci",
        "tc=inf", f"nodes={nodes}", "timemargin=300000",
        f"stderr={stderr_path}",
    ]


def head2head_preflight(repo: Path, candidate: Path, baseline: Path) -> list:
    """Every path the match needs, checked before the first game."""
    problems = []
    required = {
        "cutechess-cli": repo / "cutechess-1.4.0-win64" / "cutechess-cli.exe",
        "openings book": repo / "assets" / "8moves_v3.pgn",
        "uci wrapper": repo / "playing" / "uci_wrapper_v6.py",
        "candidate checkpoint": candidate,
        "baseline checkpoint": baseline,
    }
    for label, path in required.items():
        if not Path(path).exists():
            problems.append(f"MISSING {label}: {path}")
    return problems


def supersede_previous(directory: Path, event: str) -> int:
    """Move a previous run of this event aside. Required for CORRECTNESS, not
    tidiness: -pgnout appends and stderr= accumulates, so re-running in place
    blends two runs into one artifact. Moved, never deleted."""
    if not directory.exists():
        return 0
    stale = [p for p in directory.iterdir()
             if p.is_file() and (p.name.startswith(event + ".")
                                 or p.name.endswith(".stderr.log"))]
    if not stale:
        return 0
    attic = directory / ("_superseded_" + time.strftime("%Y%m%dT%H%M%S"))
    attic.mkdir(parents=True, exist_ok=True)
    for path in stale:
        path.rename(attic / path.name)
    return len(stale)


def run_head2head(repo: Path, candidate: Path, baseline: Path, out_dir: Path, *,
                  games: int = 200, nodes: int = 12000, concurrency: int = 2,
                  candidate_name: str = "v5-90M", baseline_name: str = "v5-20M-ep9",
                  event: str = "90Mx4_vs_20Mep9_12k", log=print) -> MatchResult:
    """One paired match, run to completion. Returns a MatchResult either way.

    Never raises: this fires at the end of a 17-hour training run, and an
    exception here would be indistinguishable from the training itself having
    failed. Every failure path returns a result carrying `error` instead.
    """
    started = time.time()
    result = MatchResult(event=event, exit_code=-1, seconds=0.0)

    problems = head2head_preflight(repo, candidate, baseline)
    if problems:
        result.error = "; ".join(problems)
        log(f"  [h2h] preflight FAILED, no games played:")
        for p in problems:
            log(f"        {p}")
        return result

    cutechess = repo / "cutechess-1.4.0-win64" / "cutechess-cli.exe"
    openings = repo / "assets" / "8moves_v3.pgn"
    out_dir.mkdir(parents=True, exist_ok=True)
    moved = supersede_previous(out_dir, event)
    if moved:
        log(f"  [h2h] superseded {moved} artifact(s) from a previous run")

    # -games 2 -repeat -rounds N/2 is what makes them PAIRED: each opening is
    # played twice with the colours reversed, so an opening that simply favours
    # White cancels instead of landing in the margin.
    if games % 2:
        games += 1
        log(f"  [h2h] rounded up to {games} games (pairing needs an even count)")
    rounds = games // 2
    arena = int(nodes * ARENA_NODES_PER_SIM)
    pgn = out_dir / f"{event}.pgn"

    args = []
    args += _guofish_arm(repo, candidate_name, candidate, nodes, arena,
                         out_dir / f"{candidate_name}.stderr.log")
    args += _guofish_arm(repo, baseline_name, baseline, nodes, arena,
                         out_dir / f"{baseline_name}.stderr.log")
    args += ["-openings", f"file={openings}", "format=pgn",
             "order=sequential", "plies=16"]
    # Valid because both checkpoints carry the same value_scale, so
    # `score cp = value_scale * atanh(q)` puts both arms on ONE scale and
    # -resign fires symmetrically.
    args += ["-resign", "movecount=3", "score=600"]
    args += ["-draw", "movenumber=40", "movecount=8", "score=10"]
    args += ["-recover", "-concurrency", str(concurrency),
             "-rounds", str(rounds), "-games", "2", "-repeat",
             "-event", event, "-pgnout", str(pgn)]

    command_path = out_dir / f"{event}.command.txt"
    rendered = " ".join(f'"{a}"' if " " in a else a for a in args)
    command_path.write_text(f'"{cutechess}" {rendered}\n', encoding="utf-8")

    log_path = out_dir / f"{event}.cutechess.log"
    log(f"  [h2h] {games} paired games ({rounds} openings x2), {nodes:,} sims, "
        f"concurrency {concurrency}")
    log(f"        {candidate_name} vs {baseline_name}")
    log(f"        -> {log_path.name}")

    chunks = []
    try:
        with open(log_path, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen(
                [str(cutechess)] + args, cwd=str(repo),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
                chunks.append(line)
                if line.startswith("Score of"):
                    log(f"        {line.rstrip()}")
            proc.wait()
        result.exit_code = proc.returncode
    except Exception as exc:                        # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        log(f"  [h2h] launch failed: {result.error}")

    result.seconds = time.time() - started
    result.pgn = str(pgn)
    result.log = str(log_path)

    parsed = parse_cutechess_score("".join(chunks), candidate_name)
    if parsed is None:
        if not result.error:
            result.error = "no parsable 'Score of' line in the cutechess output"
        log(f"  [h2h] {result.error}")
        return result
    w, d, l, n = parsed
    result.wins, result.draws, result.losses, result.games = w, d, l, n
    result.score = (w + 0.5 * d) / n if n else float("nan")
    log(f"  [h2h] {candidate_name} {w}W-{l}L-{d}D of {n} "
        f"({result.score * 100:.1f}%) in {result.seconds / 3600:.2f} h")

    # --- three estimators, in order of authority --------------------------
    text = "".join(chunks)
    # cutechess orients its Elo line the same way as the score line above it.
    first_named = None
    for line in text.splitlines():
        m = _SCORE_RE.match(line.strip())
        if m:
            first_named = m.group(1).strip()
    for k, v in parse_cutechess_elo(
            text, invert=(first_named is not None
                          and first_named != candidate_name)).items():
        setattr(result, k, v)

    result.closed_elo, result.closed_lo, result.closed_hi = elo_with_ci(w, d, l)

    for k, v in run_ordo_diff(repo, pgn, candidate_name, baseline_name,
                              log=log).items():
        setattr(result, k, v)

    # ordo is the house authority (parse_cpuct_results.py, run_15k_ab.ps1): it
    # FITS the white advantage and the draw rate rather than assuming them, and
    # its interval comes from simulation. cutechess's own figure is the fallback
    # because it is always present even when ordo refuses the input, and the
    # closed form is the last resort.
    if math.isfinite(result.ordo_elo):
        result.elo, result.elo_source = result.ordo_elo, "ordo"
        half = result.ordo_error
    elif math.isfinite(result.cc_elo):
        result.elo, result.elo_source = result.cc_elo, "cutechess"
        half = result.cc_margin
    else:
        result.elo, result.elo_source = result.closed_elo, "closed-form"
        half = float("nan")
    if result.elo_source == "closed-form":
        result.elo_lo, result.elo_hi = result.closed_lo, result.closed_hi
    else:
        result.elo_lo = result.elo - half
        result.elo_hi = result.elo + half

    # A disagreement between independent estimators on the SAME games is almost
    # never a modelling difference at this size - it is a name or orientation
    # bug, which is the failure that silently promotes a worse net. On the real
    # 100-game v5 PGN ordo and the closed form agree to 2.0 ELO.
    finite = [x for x in (result.ordo_elo, result.cc_elo, result.closed_elo)
              if math.isfinite(x)]
    if len(finite) >= 2:
        result.estimator_disagreement = max(finite) - min(finite)

    # ordo prints CFS "relative to the player in the next row", so the bottom
    # row of a two-player table legitimately has none.
    cfs = (f" (CFS {result.ordo_cfs:.0f}%)"
           if math.isfinite(result.ordo_cfs) else "")
    log(f"        ordo      {result.ordo_elo:+.1f} +/- {result.ordo_error:.1f}{cfs}"
        + (f"  [{result.ordo_error_note}]" if result.ordo_error_note else ""))
    log(f"        cutechess {result.cc_elo:+.1f} +/- {result.cc_margin:.1f} "
        f"(LOS {result.cc_los:.1f}%, draws {result.cc_draw_ratio:.1f}%)")
    log(f"        closed    {result.closed_elo:+.1f} "
        f"[{result.closed_lo:+.1f}, {result.closed_hi:+.1f}]")
    if math.isfinite(result.ordo_white_advantage):
        log(f"        white advantage {result.ordo_white_advantage:+.1f}, "
            f"draw rate {result.ordo_draw_rate:.1f}%")
    if (math.isfinite(result.estimator_disagreement)
            and result.estimator_disagreement > ORDO_DISAGREEMENT_WARN):
        log(f"  [WARN] estimators disagree by "
            f"{result.estimator_disagreement:.1f} ELO. At this sample size that "
            f"points at an engine-name or orientation problem rather than a "
            f"modelling difference - check the score line's orientation before "
            f"trusting the verdict.")
    log(f"  [h2h] ELO {result.elo:+.1f} [{result.elo_lo:+.1f}, "
        f"{result.elo_hi:+.1f}] (source: {result.elo_source})")
    return result


# ---------------------------------------------------------------------------
# the pre-registered bar
# ---------------------------------------------------------------------------
@dataclass
class GateVerdict:
    kl_candidate: float = float("nan")
    kl_baseline: float = float("nan")
    kl_reduction: float = float("nan")
    kl_required: float = KL_REDUCTION_REQUIRED
    kl_pass: bool = False
    elo: float = float("nan")
    elo_lo: float = float("nan")
    elo_hi: float = float("nan")
    elo_source: str = ""
    elo_required: float = ELO_GAIN_REQUIRED
    elo_pass: bool = False
    elo_ci_clears: bool = False
    overall_pass: bool = False
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_thresholds(candidate_kl: float, baseline_kl: float,
                        match: "MatchResult | None") -> GateVerdict:
    """Both pre-registered criteria, evaluated together.

    `overall_pass` requires BOTH. The ELO criterion is judged on the point
    estimate, because that is what was pre-registered; whether the confidence
    interval also clears 100 is reported separately as `elo_ci_clears` rather
    than quietly substituted for the stated bar.
    """
    v = GateVerdict(kl_candidate=candidate_kl, kl_baseline=baseline_kl)

    if (math.isfinite(candidate_kl) and math.isfinite(baseline_kl)
            and baseline_kl > 0):
        v.kl_reduction = (baseline_kl - candidate_kl) / baseline_kl
        v.kl_pass = v.kl_reduction >= KL_REDUCTION_REQUIRED
    else:
        v.notes.append("KL criterion not evaluable: a KL is missing or "
                       "non-finite")

    if match is None:
        v.notes.append("ELO criterion not evaluable: no match was run")
    elif match.error and not match.games:
        v.notes.append(f"ELO criterion not evaluable: {match.error}")
    else:
        v.elo, v.elo_lo, v.elo_hi = match.elo, match.elo_lo, match.elo_hi
        v.elo_source = match.elo_source
        v.elo_pass = math.isfinite(v.elo) and v.elo >= ELO_GAIN_REQUIRED
        v.elo_ci_clears = math.isfinite(v.elo_lo) and v.elo_lo >= ELO_GAIN_REQUIRED
        if match.exit_code != 0:
            v.notes.append(f"cutechess exited {match.exit_code}; the score line "
                           f"parsed but the match may be short")
        if match.elo_source != "ordo":
            v.notes.append(
                f"ELO came from {match.elo_source or 'nothing'}, not ordo"
                + (f" ({match.ordo_error_note})" if match.ordo_error_note else ""))
        if (math.isfinite(match.estimator_disagreement)
                and match.estimator_disagreement > ORDO_DISAGREEMENT_WARN):
            v.notes.append(
                f"estimators disagree by {match.estimator_disagreement:.1f} ELO "
                f"- suspect engine naming/orientation before believing this")
        if math.isfinite(match.ordo_draw_rate) and match.ordo_draw_rate > 85.0:
            v.notes.append(
                f"draw rate {match.ordo_draw_rate:.1f}% - the match carries "
                f"little information at this draw level")

    v.overall_pass = bool(v.kl_pass and v.elo_pass)
    return v


def format_verdict(v: GateVerdict, match: "MatchResult | None" = None) -> str:
    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    lines = ["\n" + "=" * 96,
             "PRE-REGISTERED SUCCESS THRESHOLDS",
             "=" * 96]

    # All three estimators, side by side, in the block someone actually reads.
    # The one the verdict used is starred. Printing only the winner invites the
    # question "what did the others say" at exactly the moment the logs have
    # scrolled away.
    if match is not None and match.games:
        lines.append(f"  {match.wins}W-{match.losses}L-{match.draws}D of "
                     f"{match.games} ({match.score * 100:.1f}%)")
        rows = [
            ("ordo", match.ordo_elo, match.ordo_error,
             (f"CFS {match.ordo_cfs:.0f}%"
              if math.isfinite(match.ordo_cfs) else match.ordo_error_note)),
            ("cutechess", match.cc_elo, match.cc_margin,
             (f"LOS {match.cc_los:.1f}%, draws {match.cc_draw_ratio:.1f}%"
              if math.isfinite(match.cc_los) else "")),
            ("closed-form", match.closed_elo,
             (match.closed_hi - match.closed_elo
              if math.isfinite(match.closed_hi) else float("nan")), ""),
        ]
        for name, elo, err, extra in rows:
            star = "*" if name == v.elo_source else " "
            err_s = f"+/- {err:.1f}" if math.isfinite(err) else "+/-  n/a"
            lines.append(f"   {star} {name:<12} {elo:+8.1f} {err_s:<10} {extra}")
        if math.isfinite(match.ordo_white_advantage):
            lines.append(f"     white advantage {match.ordo_white_advantage:+.1f}"
                         f" | draw rate {match.ordo_draw_rate:.1f}%")
        if math.isfinite(match.estimator_disagreement):
            lines.append(f"     spread across estimators "
                         f"{match.estimator_disagreement:.1f} ELO")
        lines.append("")
    # `kl_reduction` is (baseline - candidate)/baseline, so POSITIVE means the
    # candidate improved. The printed comparison has to read the same way round
    # as the test that produced the verdict.
    lines.append(
        f"  KL   candidate {v.kl_candidate:.5f} vs baseline {v.kl_baseline:.5f} "
        f"-> {v.kl_reduction * 100:+.2f}% reduction "
        f"(need >= +{v.kl_required * 100:.0f}%)   [{mark(v.kl_pass)}]")
    lines.append(
        f"  ELO  {v.elo:+.1f} [{v.elo_lo:+.1f}, {v.elo_hi:+.1f}] "
        f"via {v.elo_source or 'n/a'} "
        f"(need >= +{v.elo_required:.0f})   [{mark(v.elo_pass)}]")
    if math.isfinite(v.elo_lo):
        lines.append(
            f"       95% lower bound {'clears' if v.elo_ci_clears else 'does NOT clear'} "
            f"+{v.elo_required:.0f} - the bar is on the point estimate, this is "
            f"reported alongside it")
    for note in v.notes:
        lines.append(f"  [note] {note}")
    lines.append(f"  OVERALL: {mark(v.overall_pass)} "
                 f"(both criteria required)")
    lines.append("=" * 96)
    return "\n".join(lines)


def write_gate_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str),
                    encoding="utf-8")
