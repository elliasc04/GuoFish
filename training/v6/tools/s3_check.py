"""Gate S3 for any config: kill-and-resume must reproduce the run.

    python -m training.v6.tools.s3_check --config <cfg> --steps 300 --kill-after 157 \
        --ckpt-every 25 --tol 1e-3 [--set KEY=VALUE ...]

Run A trains `--steps` optimizer steps uninterrupted. Run B is killed hard
(os._exit, no checkpoint) after `--kill-after` steps and resumed from its latest
rolling checkpoint. Both log every step. Pass: every step's sample-stream digest
(record indices + mirror flags) is identical, and every step's loss agrees with
A's within `--tol` relative (0 = bit-identical, the CPU fp32 criterion).
Prints a JSON verdict; exit 0 on pass, 1 on fail.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from training.v6.config import load_config
from training.v6.data.formats import REPO
from training.v6.train import CRASH_EXIT, resolve


def _train(args, name: str, extra: list, expect: int):
    cmd = [sys.executable, "-m", "training.v6.train", "--config", args.config, "--set",
           *args.set, f"run.name={name}", *extra]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=os.environ.copy())
    if r.returncode != expect:
        raise SystemExit(f"{name}: exit {r.returncode}, expected {expect}\n{r.stderr[-3000:]}")


def _steps(run_dir: Path) -> list:
    rows = [json.loads(ln) for ln in (run_dir / "logs/train.jsonl").read_text().splitlines()]
    return [r for r in rows if r["event"] == "step"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--kill-after", type=int, default=157)
    ap.add_argument("--ckpt-every", type=int, default=25, help="rolling checkpoint cadence, steps")
    ap.add_argument("--tol", type=float, default=0.0)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    eff = cfg.optim.effective_batch
    if not 0 < args.kill_after < args.steps or args.kill_after % args.ckpt_every == 0:
        raise SystemExit("--kill-after must fall strictly between checkpoints and before --steps")
    args.set = [*args.set, f"schedule.total_samples={args.steps * eff}",
                f"ckpt.every_samples={args.ckpt_every * eff}", "system.log_every=1"]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    a, b = f"s3a_{stamp}", f"s3b_{stamp}"
    _train(args, a, [], 0)
    _train(args, b, ["--crash-after-steps", str(args.kill_after)], CRASH_EXIT)
    _train(args, b, ["--resume", "latest"], 0)
    root = resolve(load_config(args.config, args.set).run.out_root)
    sa = {s["step"]: s for s in _steps(root / a)}
    sb = _steps(root / b)
    resume_at = (args.kill_after // args.ckpt_every) * args.ckpt_every
    worst, bad_stream = 0.0, []
    for s in sb:
        ref = sa[s["step"]]
        if s["stream_sha"] != ref["stream_sha"]:
            bad_stream.append(s["step"])
        for x, y in zip(s["step_losses"], ref["step_losses"]):
            if x is None or y is None:
                raise SystemExit(f"non-finite loss at step {s['step']}")
            worst = max(worst, abs(x - y) / max(abs(y), 1e-12))
    verdict = {"steps": args.steps, "killed_after": args.kill_after, "resumed_at": resume_at,
               "steps_compared": len(sb), "stream_mismatches": bad_stream,
               "max_rel_loss_diff": worst, "tol": args.tol,
               "passed": not bad_stream and worst <= args.tol, "runs": [str(root / a), str(root / b)]}
    print(json.dumps(verdict, indent=1))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
