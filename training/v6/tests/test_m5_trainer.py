"""M5: tiny CPU training run, S3 (kill + resume is bit-identical), export
round trip, branch decay, and the refusal paths. Each test launches the real
trainer as a subprocess. CPU, fp32, no compile. Several minutes in total,
mostly full evals of the 452k-record frozen set.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from core.guofish_net import build_model, load_for_inference
from training.v6.config import build_config
from training.v6.data.formats import REPO
from training.v6.train import CRASH_EXIT

TINY = "training/v6/config/configs/tiny_cpu.yaml"


def train(out_root: Path, name: str, *sets, resume=None, crash=None, check_exit=0):
    cmd = [sys.executable, "-m", "training.v6.train", "--config", TINY, "--set",
           f"run.out_root={out_root.as_posix()}", f"run.name={name}", "system.allow_dirty=true", *sets]
    if resume:
        cmd += ["--resume", resume]
    if crash:
        cmd += ["--crash-after-steps", str(crash)]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if check_exit is not None and r.returncode != check_exit:
        raise AssertionError(f"exit {r.returncode} (want {check_exit})\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r


def events(run_dir: Path, kind=None):
    rows = [json.loads(ln) for ln in (run_dir / "logs/train.jsonl").read_text().splitlines()]
    return [r for r in rows if kind is None or r["event"] == kind]


# ------------------------------------------------------- CUDA hygiene

def test_cpu_checkpoint_never_initialises_cuda(monkeypatch):
    """Even with a GPU visible, a CPU run's checkpoint must not touch CUDA:
    get_rng_state_all() creates a context on every visible device. Simulated
    without a GPU: availability forced True, CUDA init replaced by a recorder."""
    import torch.cuda.random
    from core.guofish_net import ModelConfig
    from training.v6.ckpt import checkpoint_blob
    calls = []

    def recorder(*a, **k):
        calls.append(1)
        raise RuntimeError("CUDA init attempted")

    monkeypatch.setattr(torch._C, "_cuda_init", recorder)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda.random, "device_count", lambda: 1)
    m = torch.nn.Linear(2, 2)
    kw = dict(model=m, ema=None, optimizer=torch.optim.AdamW(m.parameters()), samples=0, step=0,
              cfg_plain={}, cfg_hash="", model_cfg=ModelConfig(), prov={}, metrics={},
              reason="t", best=0.0)
    blob = checkpoint_blob(**kw, device="cpu")
    assert not calls and "cuda" not in blob["rng"]
    with pytest.raises(RuntimeError, match="CUDA init attempted"):      # positive control
        checkpoint_blob(**kw, device="cuda")
    assert calls


# --------------------------------------------------------- quick subset

def test_quick_subset_is_stratified_and_seeded():
    """H3 fix: a seeded random 32,768-record subset of frozen90, every strata
    cell at its proportional share (within one record), not a source prefix."""
    import numpy as np
    from training.v6.data.strata import field_of, load_strata
    from training.v6.eval import quick_subset
    codes = np.asarray(load_strata(REPO / "data/processed/val_frozen_90m_v1/strata_val_v1.npy", 452_405))
    a = quick_subset(codes, 32_768, 20260924)
    assert np.array_equal(a, quick_subset(codes, 32_768, 20260924))
    assert not np.array_equal(a, quick_subset(codes, 32_768, 1))
    assert len(a) == len(np.unique(a)) == 32_768 and np.all(np.diff(a) > 0)
    cells, full = np.unique(codes, return_counts=True)
    sub = dict(zip(*np.unique(codes[a], return_counts=True)))
    assert max(abs(sub.get(c, 0) - n * 32_768 / len(codes)) for c, n in zip(cells, full)) < 1.0
    share = float((field_of(codes[a], "label") == 0).mean())
    print(f"\nquick subset: {len(cells)} cells, policy share {share:.4f} (full 0.6010, H3 prefix 0.772)")
    assert abs(share - 271_876 / 452_405) < 1e-3


# ------------------------------------------------------------ tiny run

@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    train(root, "tiny", "ckpt.stable_every_samples=19200")
    return root / "tiny"


def test_tiny_300_steps(tiny_run):
    steps = events(tiny_run, "step")
    assert steps[-1]["step"] == 300 and steps[-1]["samples"] == 38_400
    losses = [l for s in steps for l in s["step_losses"]]
    assert len(losses) == 300 and all(map(lambda x: x == x and abs(x) < 1e6, losses))
    first, last = sum(losses[:30]) / 30, sum(losses[-30:]) / 30
    print(f"\ntiny d64x2: 300 steps, mean loss first 30 {first:.4f} -> last 30 {last:.4f}; "
          f"{steps[-1]['samples_per_s']:,.0f} samples/s")
    assert last < first
    for key in ("lr", "soft_kl", "value_mse", "grad_norm", "samples_per_s", "loader_wait_frac",
                "passes", "stream_sha", "nonfinite_steps"):
        assert key in steps[-1]
    assert (tiny_run / "config.resolved.yaml").exists() and (tiny_run / "provenance.json").exists()
    assert (tiny_run / "code.patch").exists() and (tiny_run / "best.pt").exists()
    assert sorted(p.name for p in (tiny_run / "ckpt").glob("s*.pt")) == [
        "s25600.pt", "s32000.pt", "s38400.pt"]                               # last 3 kept
    assert [p.name for p in (tiny_run / "stable").glob("*.pt")] == ["s19200.pt"]
    prov = json.loads((tiny_run / "provenance.json").read_text())
    for k in ("git_sha", "dirty_files", "diff_sha256", "config_hash", "corpus_manifest_sha256",
              "strata_definition_hash", "frozen_val_manifest_sha256", "versions"):
        assert k in prov
    full = events(tiny_run, "full_eval")
    assert {(e["samples"], e["weights"]) for e in full} == {
        (19200, "raw"), (19200, "ema"), (38400, "raw"), (38400, "ema")}
    for k in ("policy_kl", "policy_top1", "policy_top5", "value_mse", "total", "sf_top1_pv0",
              "pv_kl", "policy_entropy", "value/mate/mse", "slice/le5/middle/value_mse",
              "material/compensated/value_bias", "mirror_policy_top1_agreement"):
        assert full[-1][k] is not None, k
    assert full[-1]["hard_n"] == 0 and full[-1]["hard_nll"] is None       # v1 data: no hard moves
    assert len(events(tiny_run, "quick_eval")) == 3


def test_export_round_trip(tiny_run):
    ckpt = tiny_run / "ckpt/s38400.pt"
    for weights in ("ema", "raw", "best"):
        r = subprocess.run([sys.executable, "-m", "training.v6.tools.export", str(ckpt),
                            "--weights", weights], cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        assert "identical" in r.stdout
    exports = sorted((tiny_run / "export").glob("*.pt"))
    assert len(exports) == 3 and len({p.name for p in exports}) == 3
    # independent check: the ema export equals the checkpoint's EMA weights
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    ema_file = next(p for p in exports if p.name.endswith("_ema.pt"))
    model, contract = load_for_inference(ema_file)
    assert contract == "A" and not model.training
    for k, v in ck["ema"].items():
        assert torch.equal(model.state_dict()[k], v), k
    again = subprocess.run([sys.executable, "-m", "training.v6.tools.export", str(ckpt),
                            "--weights", "ema"], cwd=REPO, capture_output=True, text=True)
    assert again.returncode != 0 and "refusing to overwrite" in again.stderr


def test_branch_decay(tiny_run):
    stable = tiny_run / "stable/s19200.pt"
    r = subprocess.run([sys.executable, "-m", "training.v6.tools.branch_decay", str(stable)],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    br = tiny_run / "branches/s19200_d4800"
    assert (br / "final.pt").exists()
    steps = events(br, "step")
    start = events(br, "run_start")[0]
    assert start["branch"] and start["step"] == 150 and start["schedule"]["total_samples"] == 24_000
    assert steps[-1]["samples"] == 188 * 128                        # ceil(24000/128) steps
    lrs = [s["lr"] for s in steps]
    assert lrs[0] <= 1e-3 and lrs[-1] < 0.1 * 1e-3 and lrs == sorted(lrs, reverse=True)
    assert {e["weights"] for e in events(br, "full_eval")} == {"raw", "ema"}


# ------------------------------------------------------------------ S3

S3 = ["model.dropout=0.1", "optim.micro_batch=78", "optim.accum=2",
      "mixture.groups=[{name: endgame, where: {bucket: le5}, share: 0.25}]",
      "schedule.total_samples=31200", "schedule.warmup_samples=3120",
      "ema.half_life_samples=3120", "ckpt.every_samples=780", "ckpt.stable_every_samples=0",
      "eval.quick_every_samples=7800", "eval.quick_size=2048", "system.log_every=1"]


def test_s3_kill_and_resume_is_bit_identical(tmp_path):
    """Run A uninterrupted for 200 steps. Run B is killed (os._exit, no
    checkpoint) after step 57 and after step 117, resuming each time from the
    latest rolling checkpoint: step 55 (mid-pass for both groups) and step 115,
    where the 4,485-record `endgame` group has consumed exactly one full pass
    (115 x 39 = 4,485) - the pass boundary. Dropout 0.1 exercises the RNG
    restore; quick evals at steps 50/100/150/200 sit between the kill points."""
    train(tmp_path, "a", *S3)
    train(tmp_path, "b", *S3, crash=57, check_exit=CRASH_EXIT)
    assert sorted(p.name for p in (tmp_path / "b/ckpt").glob("s*.pt"))[-1] == "s8580.pt"
    train(tmp_path, "b", *S3, resume="latest", crash=117, check_exit=CRASH_EXIT)
    ck = torch.load(tmp_path / "b/ckpt/s17940.pt", map_location="cpu", weights_only=True)
    assert ck["samples"] == 17_940
    train(tmp_path, "b", *S3, resume="latest")

    a = {s["step"]: s for s in events(tmp_path / "a", "step")}
    b_steps = events(tmp_path / "b", "step")
    assert len(a) == 200 and len(b_steps) == 57 + (117 - 55) + (200 - 115)
    assert a[115]["passes"]["endgame"] == 1 and a[114]["passes"]["endgame"] == 0
    starts = [s["step"] for s in events(tmp_path / "b", "run_start")]
    assert starts == [0, 55, 115]
    worst = 0.0
    for s in b_steps:
        ref = a[s["step"]]
        assert s["stream_sha"] == ref["stream_sha"], s["step"]          # indices + mirror flags
        assert s["step_losses"] == ref["step_losses"], (s["step"], s["step_losses"], ref["step_losses"])
        worst = max(worst, abs(s["loss"] - ref["loss"]))
    ca = torch.load(tmp_path / "a/ckpt/s31200.pt", map_location="cpu", weights_only=True)
    cb = torch.load(tmp_path / "b/ckpt/s31200.pt", map_location="cpu", weights_only=True)
    for part in ("model", "ema"):
        for k in ca[part]:
            assert torch.equal(ca[part][k], cb[part][k]), (part, k)
    fa = [e for e in events(tmp_path / "a", "full_eval")]
    fb = [e for e in events(tmp_path / "b", "full_eval")]
    strip = lambda e: {k: v for k, v in e.items() if k != "utc"}  # noqa: E731
    assert [strip(e) for e in fa] == [strip(e) for e in fb]
    print(f"\nS3: {len(b_steps)} resumed-run steps (kills after 57 and 117, resumes at 55 and "
          f"115 = endgame pass boundary) vs uninterrupted: max |d loss| {worst:.1e}; "
          f"stream digests, final weights, EMA and full-eval metrics identical")
    assert worst == 0.0


# ----------------------------------------------------------- refusals

def test_refusals(tmp_path):
    r = subprocess.run([sys.executable, "-m", "training.v6.train", "--config", TINY, "--set",
                        f"run.out_root={tmp_path.as_posix()}", "run.name=dirty"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0 and "working tree is dirty" in r.stderr     # allow_dirty=false

    short = ["schedule.total_samples=2560", "schedule.warmup_samples=0", "ckpt.every_samples=1280",
             "ckpt.stable_every_samples=0", "eval.quick_every_samples=0", "ema.enabled=false"]
    train(tmp_path, "x", *short, crash=12, check_exit=CRASH_EXIT)
    r = train(tmp_path, "x", *short, check_exit=None)
    assert r.returncode != 0 and "not empty" in r.stderr                  # no --resume
    ck = str(tmp_path / "x/ckpt/s1280.pt")
    r = train(tmp_path, "x", *short, "optim.lr=2e-3", resume=ck, check_exit=None)
    assert r.returncode != 0 and "optim.lr" in r.stderr                   # config drift
    r = train(tmp_path, "x", *short, "schedule.total_samples=3840", resume=ck, check_exit=None)
    assert r.returncode != 0 and "schedule.total_samples" in r.stderr     # only when branching
    train(tmp_path, "x", *short, "system.log_every=5", resume=ck)         # whitelisted


def test_nonfinite_guard(tmp_path):
    r = train(tmp_path, "nan", "optim.lr=1e30", "schedule.warmup_samples=0",
              "schedule.total_samples=6400", "eval.quick_every_samples=0",
              "ckpt.stable_every_samples=0", "system.log_every=5", check_exit=None)
    assert r.returncode != 0 and "non-finite" in r.stderr
    run = tmp_path / "nan"
    anomaly = events(run, "anomaly")
    assert anomaly and anomaly[0]["nonfinite_steps"] >= 1
    em = list((run / "ckpt").glob("emergency_s*.pt"))
    assert len(em) == 1
    ck = torch.load(em[0], map_location="cpu", weights_only=True)
    assert all(torch.isfinite(v).all() for v in ck["model"].values())   # skipped steps kept weights finite
    model = build_model(build_config(ck["config"]).model)
    model.load_state_dict(ck["model"])
