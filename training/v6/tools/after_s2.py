"""One-off: wait for the S2 driver (training.v6.tools.s2) to finish, run its
gate analyzer, then hand the result to a headless Claude Code review — read
only, proposes the next step, does not launch anything.

    python training/v6/tools/after_s2.py <s2 driver pid>

Find the driver's pid:
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
      Where-Object { $_.CommandLine -match 'training\.v6\.tools\.s2' } |
      Select-Object ProcessId, CommandLine

Reuses tools/capacity_campaign.py's wait_for_pid (WaitForSingleObject on the
process handle - no polling), same as runs/capacity_campaign/followup_common_lr.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import capacity_campaign as cc  # noqa: E402

if len(sys.argv) != 2:
    sys.exit("usage: after_s2.py <s2 driver pid>")
pid = int(sys.argv[1])

print(f"waiting for S2 driver pid {pid}", flush=True)
cc.wait_for_pid(pid)
print("S2 driver exited; running the gate analyzer", flush=True)

analyze = subprocess.run(
    [sys.executable, "-u", "-m", "training.v6.tools.s2", "analyze"],
    cwd=str(REPO), capture_output=True, text=True)
report_path = REPO / "models" / "v6" / "s2" / "analyze_output.txt"
report_path.write_text(analyze.stdout + "\n" + analyze.stderr, encoding="utf-8")
print(f"analyzer exit {analyze.returncode}; wrote {report_path}", flush=True)

prompt = (
    "S2 (the v6 harness's stack-reproduction gate) just finished. Read "
    f"{report_path.as_posix()}, models/v6/s2/driver.jsonl, and the pass rule "
    "in training/v6/tools/s2.py's docstring (ref must land within "
    "[min(c1,c2)-d, max(c1,c2)+d] on KL and MSE). Tell me clearly: did it "
    "pass? If yes, read docs/capacity/training_stack.md section 13.2 and "
    "propose (do not launch) the exact command for the first Phase 2 "
    "screening arm, A0. If it failed, diagnose why from the logs and "
    "propose a fix. Write a short summary to models/v6/s2/REVIEW.md. Do not "
    "start, kill, or modify any training job."
)
proc = subprocess.Popen(["claude", "-p", prompt, "--model", "claude-opus-5-5"], cwd=str(REPO))
print(f"launched claude review as pid {proc.pid}", flush=True)
