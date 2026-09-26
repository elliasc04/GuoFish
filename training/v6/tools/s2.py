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
"""
from __future__ import annotations

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
    "c1": V6 + ["run.name=c1", "run.seed=20260802"],
    "c2": V6 + ["run.name=c2", "run.seed=20260925"],
}
assert SAMPLES == STEPS * 1024


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


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(RUNS)
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
