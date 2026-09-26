"""Gate S2 driver: train_v5.py (`ref`) and two v5_compat seeds (`c1`, `c2`), back to back.

    python -m training.v6.tools.s2 [ref c1 c2]

Each run: d384x6 on the 90M corpus, exactly 58,594 optimizer steps
(60,000,256 samples at 1,024) with OneCycle compressed to that length, v5
hyperparameters, under models/v6/s2/<run>. A run that exits non-zero is
resumed from its latest checkpoint (`--resume latest`), at most MAX_RESTARTS
times; with no checkpoint yet, its directory is moved aside to
<run>_failed<k> and the run starts fresh. Every start, restart and exit goes
to models/v6/s2/driver.jsonl, and each attempt's console output to
models/v6/s2/<run>_attempt<k>.log. Before starting a run the driver waits
while models/v6/s2/HOLD exists, so a throughput-sensitive job can use the gap.
Children always see the GPU (CUDA_VISIBLE_DEVICES is removed from their env).

    python -m training.v6.tools.s2 analyze [--busy <utc start>/<utc end> ...]

reads the three runs' logs and models/v6/s2/score_<run>.json (tools/score.py
output; a missing one is produced first by scoring the run's final checkpoint
on the GPU, ref with --v5-crosscheck) and prints the S2 table: frozen90 policy KL / value MSE / top-1,
d = |c1 - c2| per metric (also relative to their mean), the pass rule (ref
within [min(c1, c2) - d, max(c1, c2) + d] on KL and MSE), training KL + MSE
over the final 1,069,056 samples (last 1,044 steps = 21 v6 log rows; count-weighted),
median samples/s split by overlap with the --busy windows (Track C running),
and the LR / beta1 traces: every logged value of all three runs against one
torch OneCycleLR replay (v5 logs LR only).
"""
from __future__ import annotations

import calendar
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "models" / "v6" / "s2"
MAX_RESTARTS = 2
STEPS, SAMPLES = 58_594, 60_000_256
PY = [sys.executable, "-u"]
V6 = PY + ["-m", "training.v6.train", "--config", "training/v6/config/configs/v5_compat.yaml", "--set",
           "run.out_root=models/v6/s2", f"schedule.total_samples={SAMPLES}", "ema.enabled=false",
           "system.allow_dirty=true"]
RUNS = {
    "ref": PY + ["training/v5_multiPV/train_v5.py", "--config", "training/v5_multiPV/configs/corpus90m.yaml",
                 "--epochs", "1", "--max-steps", str(STEPS), "--no-h2h-gate", "--gap-probe", "0",
                 "--seed", "20260802", "--out-dir", "models/v6/s2/ref", "--run-name", "ref"],
    "c1": V6 + ["run.name=c1", "run.data_seed=20260802", "run.init_seed=20260802"],
    "c2": V6 + ["run.name=c2", "run.data_seed=20260925", "run.init_seed=20260925"],
}
assert SAMPLES == STEPS * 1024
# Training-loss window: the last 21 v6 log rows (log_every 50; 58,594 = 1,171 x 50 + 44),
# i.e. the last 1,044 steps = 1,069,056 samples, the same steps for v5.
WINDOW_STEPS = 1_044


def has_checkpoint(name: str) -> bool:
    d = OUT / name
    return any((d / "ckpt").glob("s*.pt")) if name != "ref" else any(d.glob("*_step*.pt"))


def event(**kw) -> None:
    kw = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}
    with open(OUT / "driver.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def run(name: str) -> int:
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    resume = False
    for attempt in range(MAX_RESTARTS + 1):
        cmd = RUNS[name] + (["--resume", "latest"] if resume else [])
        event(event="start", run=name, attempt=attempt, cmd=cmd[1:])
        t0 = time.time()
        with open(OUT / f"{name}_attempt{attempt}.log", "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        event(event="exit", run=name, attempt=attempt, returncode=rc, seconds=round(time.time() - t0))
        if rc == 0:
            return 0
        if attempt == MAX_RESTARTS:
            break
        resume = has_checkpoint(name)
        if not resume and (OUT / name).exists():
            (OUT / name).rename(OUT / f"{name}_failed{attempt}")
        event(event="restart", run=name, next_attempt=attempt + 1,
              mode="resume latest" if resume else "fresh (no checkpoint; old dir moved aside)")
    event(event="gave_up", run=name)
    return rc


def _utc(x) -> float:
    return calendar.timegm(time.strptime(x, "%Y-%m-%dT%H:%M:%SZ"))


def _v5(run_dir: Path) -> dict:
    rows = [json.loads(ln) for ln in (run_dir / "logs" / "ref.jsonl").read_text().splitlines()]
    micro = [r for r in rows if r["event"] == "micro"]
    last = micro[-2 * WINDOW_STEPS:]          # 2 micro rows per step, logged 0-based
    assert last[-1]["step"] == STEPS - 1 and last[0]["step"] == STEPS - WINDOW_STEPS
    kl = sum(r["policy_kl"] * r["has_policy"] for r in last) / sum(r["has_policy"] for r in last)
    mse = sum(r["value_mse"] * r["n"] for r in last) / sum(r["n"] for r in last)
    steps = [r for r in rows if r["event"] == "step" and not r.get("warmup_window")]
    return {"train_kl": kl, "train_mse": mse, "window_samples": sum(r["n"] for r in last),
            "rate": [(r["t"], r["samples_per_s"]) for r in steps],
            "lr": {r["step"]: r["lr"] for r in steps}}


def _v6(run_dir: Path) -> dict:
    rows = [json.loads(ln) for ln in (run_dir / "logs" / "train.jsonl").read_text().splitlines()]
    steps = [r for r in rows if r["event"] == "step"]
    last = [r for r in steps if r["step"] > STEPS - WINDOW_STEPS]
    assert last[-1]["step"] == STEPS and sum(len(r["step_losses"]) for r in last) == WINDOW_STEPS
    n_rows = WINDOW_STEPS * 1024
    kl = sum(r["soft_kl"] * r["n_soft"] for r in last) / sum(r["n_soft"] for r in last)
    mse = sum(r["value_mse"] * len(r["step_losses"]) * 1024 for r in last) / n_rows
    first = {r["step"] for r in rows if r["event"] == "run_start"}
    rate = [(_utc(r["utc"]), r["samples_per_s"]) for r in steps
            if r["step"] - 50 not in {s for s in first} and r["step"] > 100]
    return {"train_kl": kl, "train_mse": mse, "train_objective": sum(
        sum(r["step_losses"]) for r in last) / WINDOW_STEPS, "window_samples": n_rows, "rate": rate,
        "lr": {r["step"]: r["lr"] for r in steps}, "beta1": {r["step"]: r["beta1"] for r in steps}}


def final_checkpoint(name: str) -> Path:
    if name != "ref":
        return OUT / name / "ckpt" / f"s{SAMPLES}.pt"
    import torch                     # v5 saves _ep1.pt at the epoch boundary it stops at
    hits = [p for p in (OUT / "ref").glob("*_ep1.pt")
            if int(torch.load(p, map_location="cpu", weights_only=True)["step"]) == STEPS]
    if len(hits) != 1:
        raise SystemExit(f"ref: expected one _ep1.pt at step {STEPS}, found {hits}")
    return hits[0]


def score(name: str) -> dict:
    """models/v6/s2/score_<run>.json, scoring the final checkpoint first if it is missing."""
    path = OUT / f"score_{name}.json"
    if not path.exists():
        cmd = PY + ["-m", "training.v6.tools.score", str(final_checkpoint(name))]
        cmd += ["--v5-crosscheck"] if name == "ref" else []
        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        if not r.stdout.strip().startswith("{"):
            raise SystemExit(f"scoring {name} failed (exit {r.returncode}): {r.stderr[-3000:]}")
        path.write_text(r.stdout)
    return json.loads(path.read_text())


def analyze(busy: list) -> dict:
    import statistics
    import torch
    windows = [tuple(_utc(x) for x in w.split("/")) for w in busy]
    runs = {"ref": _v5(OUT / "ref"), "c1": _v6(OUT / "c1"), "c2": _v6(OUT / "c2")}
    out = {"runs": {}, "d": {}, "pass": {}}
    for name, r in runs.items():
        sc = score(name)
        on = [v for t, v in r["rate"] if any(a <= t <= b for a, b in windows)]
        off = [v for t, v in r["rate"] if not any(a <= t <= b for a, b in windows)]
        out["runs"][name] = {
            "frozen90": {k: sc[k] for k in ("policy_kl", "value_mse", "policy_top1", "n", "policy_n")},
            "step": sc["step"], "train_window_samples": r["window_samples"],
            "train_kl_final_window": r["train_kl"], "train_mse_final_window": r["train_mse"],
            "train_kl_plus_mse": r["train_kl"] + r["train_mse"],
            "samples_per_s_median_trackC_running": statistics.median(on) if on else None,
            "samples_per_s_median_trackC_idle": statistics.median(off) if off else None,
            "rate_intervals": [len(on), len(off)]}
        if name == "ref":
            out["ref_crosscheck"] = {k: sc.get(k) for k in ("v5_score_baseline", "crosscheck_rel",
                                                            "crosscheck_passed")}
    for k in ("policy_kl", "value_mse", "policy_top1"):
        a, b = (out["runs"][c]["frozen90"][k] for c in ("c1", "c2"))
        d = abs(a - b)                                  # the brief's rule: [min - d, max + d]
        ref = out["runs"]["ref"]["frozen90"][k]
        out["d"][k] = {"abs": d, "rel": d / ((a + b) / 2)}
        out["pass"][k] = {"band": [min(a, b) - d, max(a, b) + d], "ref": ref,
                          "inside": min(a, b) - d <= ref <= max(a, b) + d}
    out["s2_passed"] = out["pass"]["policy_kl"]["inside"] and out["pass"]["value_mse"]["inside"]
    # Schedule traces against one OneCycleLR replay. lr_after[k] / b1_after[k]: the
    # values set after k scheduler steps, i.e. used by 0-based step k. v5 logs
    # get_last_lr() after its step S -> lr_after[S]; v6 logs the values used by its
    # last step, step S (1-based) -> lr_after[S - 1].
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=3.5e-4, betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3.5e-4, total_steps=STEPS, pct_start=0.1,
                                              div_factor=25.0, final_div_factor=1e4)
    lr_after, b1_after = [opt.param_groups[0]["lr"]], [opt.param_groups[0]["betas"][0]]
    for _ in range(STEPS - 1):
        opt.step()
        sch.step()
        lr_after.append(opt.param_groups[0]["lr"])
        b1_after.append(opt.param_groups[0]["betas"][0])

    def rel(x, y):
        return abs(x - y) / abs(y)
    tr = {"ref": {"max_rel_lr_vs_replay": max(rel(v, lr_after[s]) for s, v in runs["ref"]["lr"].items()
                                              if s < STEPS), "points": len(runs["ref"]["lr"])}}
    for c in ("c1", "c2"):
        tr[c] = {"max_rel_lr_vs_replay": max(rel(v, lr_after[s - 1]) for s, v in runs[c]["lr"].items()),
                 "max_abs_beta1_vs_replay": max(abs(v - b1_after[s - 1]) for s, v in runs[c]["beta1"].items()),
                 "points": len(runs[c]["lr"])}
    out["schedule_traces"] = tr
    return out


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv[:1] == ["analyze"]:
        busy = argv[argv.index("--busy") + 1:] if "--busy" in argv else []
        print(json.dumps(analyze(busy), indent=1))
        return 0
    names = argv or list(RUNS)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        while (OUT / "HOLD").exists():
            time.sleep(30)
        if run(name):
            return 1
    event(event="all_done", runs=names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
