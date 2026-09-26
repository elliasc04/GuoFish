"""Screening-pass harness (brief of 2026-09-26): H1 keyed init, H2 loader-worker
BLAS threads, H3 v2 eval sets, H5 forward-cost conversion, H4 queue decisions.
CPU only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from core.guofish_net import ModelConfig, build_model
from training.v6.config import load_config
from training.v6.data.formats import REPO

CFG = REPO / "training/v6/config/configs"


# ---------------------------------------------------------------- H1

def _d384x10(**kw) -> ModelConfig:
    return ModelConfig.from_dict({**load_config(CFG / "shapes/d384x10.yaml").model.to_dict(), **kw})


def test_keyed_init_shares_every_common_parameter():
    """Plain, static-bias, smolgen (and flatten-head) d384x10 models built with
    one init_seed hold identical weights under every shared name and shape."""
    plain = build_model(_d384x10(), init_seed=20260802).state_dict()
    assert sum(v.numel() for v in plain.values()) > 17_000_000          # it is the d384x10
    for variant in ({"attn_bias": "static"}, {"attn_bias": "smolgen"},
                    {"value_head": {"pool": "flatten"}}):
        other = build_model(_d384x10(**variant), init_seed=20260802).state_dict()
        shared = [k for k in plain if k in other and plain[k].shape == other[k].shape]
        assert len(shared) >= len(plain) - 3, variant        # flatten reshapes value_fc1/fc2
        for k in shared:
            assert torch.equal(plain[k], other[k]), (variant, k)
        assert set(other) - set(plain), variant              # the variant did add something

    again = build_model(_d384x10(), init_seed=20260802).state_dict()
    assert all(torch.equal(plain[k], again[k]) for k in plain)
    reseeded = build_model(_d384x10(), init_seed=20260925).state_dict()
    assert not torch.equal(plain["blocks.3.qkv.weight"], reseeded["blocks.3.qkv.weight"])
    assert not torch.equal(plain["blocks.3.qkv.weight"], plain["blocks.4.qkv.weight"])


def test_keyed_init_leaves_the_global_rng_alone():
    """The dropout stream starts where torch.manual_seed put it, whatever the
    architecture; v5_deepcopy still draws its init from the global RNG."""
    torch.manual_seed(5)
    want = torch.rand(4)
    for kw in ({}, {"attn_bias": "smolgen"}):
        torch.manual_seed(5)
        build_model(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, **kw), init_seed=1)
        assert torch.equal(torch.rand(4), want)
    torch.manual_seed(5)
    build_model(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, init="v5_deepcopy"))
    assert not torch.equal(torch.rand(4), want)


def test_seed_split_reaches_sampler_and_init():
    c = load_config(CFG / "base.yaml", ["run.data_seed=1", "run.init_seed=2"])
    assert (c.run.data_seed, c.run.init_seed) == (1, 2)
    with pytest.raises(Exception, match="seed"):
        load_config(CFG / "base.yaml", ["run.seed=1"])


# ---------------------------------------------------------------- H2

_WORKER_PROBE = r'''
import hashlib, json, os, sys
os.environ.pop("OPENBLAS_NUM_THREADS", None)
if sys.argv[1] == "trainer":
    import training.v6.train  # noqa: F401  (the trainer's import sets the worker environment)
import psutil, torch
from training.v6.config import load_config
from training.v6.data.batch import BatchBuilder, StreamDataset
from training.v6.data.formats import REPO
from training.v6.data.mixture import Mixture
from training.v6.data.reader import ShardSet


class Probe(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, k):
        return self.ds[k], psutil.Process().memory_info().private


if __name__ == "__main__":
    cfg = load_config(REPO / "training/v6/config/configs/tiny_cpu.yaml")
    ss = ShardSet(REPO / cfg.data.corpus, cfg.data.split, REPO / cfg.data.manifest)
    mix = Mixture(cfg.mixture, len(ss), cfg.optim.micro_batch, cfg.run.data_seed)
    bb = BatchBuilder(cfg.model.token_scheme, 0.5, cfg.run.data_seed, 0.05, None)
    dl = torch.utils.data.DataLoader(Probe(StreamDataset(ss, mix, bb, 3)), batch_size=None,
                                     num_workers=1, sampler=range(3))
    h, private = hashlib.sha256(), 0
    for b, private in dl:
        for k in sorted(b):
            h.update(k.encode() + b[k].numpy().tobytes())
    print(json.dumps({"private": private, "digest": h.hexdigest()}))
'''


def test_trainer_workers_run_one_blas_thread(tmp_path):
    """H2: a spawned loader worker under the trainer's environment holds less
    private memory (OpenBLAS's per-core buffers) and builds identical batches."""
    script = tmp_path / "probe.py"
    script.write_text(_WORKER_PROBE)
    out = {}
    for mode in ("default", "trainer"):
        r = subprocess.run([sys.executable, str(script), mode], cwd=REPO, capture_output=True,
                           text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
        assert r.returncode == 0, r.stderr[-3000:]
        out[mode] = json.loads(r.stdout.strip().splitlines()[-1])
    drop = out["default"]["private"] - out["trainer"]["private"]
    print(f"worker private: default {out['default']['private'] / 2**20:.0f} MiB, "
          f"trainer {out['trainer']['private'] / 2**20:.0f} MiB, drop {drop / 2**20:.0f} MiB")
    assert drop > 100 * 2**20
    assert out["default"]["digest"] == out["trainer"]["digest"]


# ---------------------------------------------------------------- H3

def test_evalset_sidecars_pin_their_shards(tmp_path, monkeypatch):
    """v2val_derived loads through its sidecar (all PV-unrolled, every tier
    reported); a sidecar whose sha256 no longer matches a shard is refused."""
    import numpy as np
    from training.v6 import eval as ev
    from training.v6.data.strata import field_of
    es = ev.load_evalset("v2val_derived", "v5_68", 1024, 0)
    assert len(es.shards) == 161_340
    assert set(np.unique(field_of(es.codes, "origin")).tolist()) == {1, 2}      # ply1, ply2
    m = ev.evaluate(build_model(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128)), es,
                    torch.device("cpu"), indices=es.indices[::400])
    assert m["policy_n"] == 0 and m["hard_n"] == m["n"] and m["tier/new/n"] > 0
    assert m["tier/old/n"] + m["tier/new/n"] == m["n"] and m["tier/v1/n"] == 0

    spec = json.loads((ev.EVALSETS / "v2val_derived.json").read_text())
    spec["shards"][1]["sha256"] = "0" * 64
    (tmp_path / "v2val_derived.json").write_text(json.dumps(spec))
    monkeypatch.setattr(ev, "EVALSETS", tmp_path)
    with pytest.raises(SystemExit, match="no longer matches"):
        ev.load_evalset("v2val_derived", "v5_68", 1024, 0)


# ---------------------------------------------------------------- H5

def test_required_gain_is_the_campaign_exchange():
    from training.v6.tools.fwd_cost import required_gain
    assert required_gain(1.0) == 0.0
    assert required_gain(1.624) == pytest.approx(4.21, abs=0.01)   # CAMPAIGN_RECORD A4, d384x10
    assert required_gain(0.93) < 0 < required_gain(1.03)


# ---------------------------------------------------------------- H4

def _row(arm, kl, mse, *, pv=None, top1=0.40, ema_total=None, **kw):
    f90 = {"policy_kl": kl, "value_mse": mse, "total": kl + mse, "pv_kl": pv if pv else kl / 2,
           "policy_top1": top1, "sf_top1_hard": top1}
    ema = dict(f90, total=ema_total if ema_total is not None else f90["total"] + 1)
    return {"arm": arm, "scores": {"raw": {"frozen90": f90}, "ema": {"frozen90": ema}},
            "adopted": False, "overrides": [], "best_after": arm, **kw}


def test_calibration_picks_weights_floors_and_stop():
    from training.v6.tools import screen
    a0 = _row("A0", 0.80, 0.060, ema_total=0.80)          # EMA better on total
    cal = screen.calibrate(a0, _row("A0r", 0.80 * 1.001, 0.060 * 1.02, ema_total=0.80))
    assert cal["weights"] == "ema" and not cal["stop"]
    assert cal["floors"]["policy_kl"] == 0.0025                          # r 0.1% < 0.25%
    assert cal["floors"]["value_mse"] == pytest.approx(0.02)             # r 2% > 1.7%
    assert screen.calibrate(a0, _row("A0r", 0.80 * 1.006, 0.060))["stop"]   # r_KL 0.6% > 0.5%


def test_decision_rules():
    from training.v6.tools import screen
    cal = {"weights": "raw", "floors": {"policy_kl": 0.0025, "pv_kl": 0.003, "value_mse": 0.017}}
    ref = _row("A0", 0.800, 0.0600)

    def j(kl, mse, required=None):
        return screen.judge(_row("X", kl, mse), ref, cal, "policy", required)["adopt"]
    assert j(0.800 * (1 - 0.0051), 0.0600 * 1.016)           # KL >= 2f, MSE within -f
    assert not j(0.800 * (1 - 0.0049), 0.0600)               # KL short of 2f
    assert not j(0.800 * (1 - 0.0060), 0.0600 * 1.018)       # MSE regresses past f
    assert j(0.800 * 1.0024, 0.0600 * (1 - 0.0341))          # MSE >= 2f, KL within -f
    assert not j(0.800 * 1.0026, 0.0600 * (1 - 0.0341))
    assert not j(0.800 * (1 - 0.006), 0.06, required=0.7)    # inference cost: 0.6% < 0.7% needed
    assert j(0.800 * (1 - 0.006), 0.06, required=-1.0)       # a cheaper arm
    pv = screen.judge(_row("B", 0.9, 0.06, pv=0.4 * (1 - 0.0061)), _row("A", 0.8, 0.06, pv=0.4),
                      cal, "pv", None)
    assert pv["adopt"] and pv["keys"][0] == "pv_kl"          # KL got worse; PV-KL is what is judged
    c = screen.confirm({"kl": 0.010, "mse": -0.002}, {"kl": 0.006, "mse": -0.03})
    assert c["confirmed"] and list(c["per_metric"]) == ["kl"]
    assert not screen.confirm({"kl": 0.010, "mse": 0.0}, {"kl": 0.004, "mse": 0.0})["confirmed"]
    assert not screen.confirm({"kl": 0.010, "mse": 0.0}, {"kl": -0.02, "mse": 0.0})["confirmed"]


def test_queue_walk_and_composition():
    """The real queue: calibration first, B0 picks its variant from A8's verdict,
    the pause holds until released, and arms build on the best's overrides."""
    from training.v6.tools import screen
    q = screen.load_queue(screen.QUEUE)
    led = []
    for name in ("A0", "A0r", "A1", "A4", "A5", "A6", "A7", "A9", "A8", "A3"):
        e = screen.next_entry(q, led, set())
        assert e["name"] == name
        base, ref, ov = screen.compose(e, led)
        adopted = name in ("A1", "A6", "A8")
        best = name if adopted or name == "A0" else screen.best(led)
        led.append({"arm": name, "overrides": ov, "adopted": adopted, "best_after": best})
    assert screen.best(led) == "A8"
    a8 = screen.row_of(led, "A8")["overrides"]
    assert a8[:2] == ["model.dropout=0.0", "model.attn_bias=static"] and a8[2].startswith("mixture.groups=")
    e = screen.next_entry(q, led, set())
    assert e["name"] == "B0" and e["only_if"] == {"adopted": "A8"}
    base, ref, ov = screen.compose(e, led)
    assert (base, ref) == ("A8", "A8") and ov[-1].startswith("mixture.groups=[{name: policy, where: {origin: root")
    for name in ("B0", "B1a", "B1b"):
        led.append({"arm": name, "overrides": [], "adopted": False, "best_after": "A8"})
    assert screen.next_entry(q, led, set())["id"] == "phase3_review"
    e = screen.next_entry(q, led, {"phase3_review"})
    assert e["name"] == "A0t" and screen.compose(e, led) == ("A0", None, e["set"])
    led.append({"arm": "A0t", "overrides": e["set"], "adopted": False, "best_after": "A8"})
    e = screen.next_entry(q, led, {"phase3_review"})
    assert e["name"] == "Ct" and screen.compose(e, led)[:2] == ("A8", "A0t")


def test_every_queue_arm_resolves_even_with_everything_adopted():
    from training.v6.tools import screen
    q = screen.load_queue(screen.QUEUE)
    acc = []
    for e in q["arms"]:
        if "name" in e:
            acc += e.get("set", [])
            load_config(REPO / q["config"], [f"run.name={e['name']}", *e.get("set", [])])
            load_config(REPO / q["config"], [f"run.name={e['name']}", *acc])


def test_report_renders(tmp_path, monkeypatch):
    from training.v6.tools import screen
    monkeypatch.setattr(screen, "OUT", tmp_path)
    monkeypatch.setattr(screen, "REPORT", tmp_path / "REPORT.md")
    cfg = tmp_path / "A1.yaml"
    cfg.write_text("run: {name: A1}\n")
    rows = []
    for name, verdict, best in (("A0", "reference", "A0"), ("A0r", "replicate", "A0"), ("A1", "ADOPT", "A1")):
        r = _row(name, 0.8, 0.06, ref="A0" if name != "A0" else None, change=["model.dropout=0.0"],
                 metrics="policy", verdict=verdict, best_after=best, config=cfg.as_posix(),
                 config_hash="ab" * 32, samples_per_s_median=4100.0, peak_vram_mib=6000.0, attempts=1,
                 starts=1, utc_start="t0", utc_end="t1", fwd_cost=None, also={})
        for w in ("raw", "ema"):
            f90 = {k: 0.1 for k in screen.HEAD} | r["scores"][w]["frozen90"]
            r["scores"][w] = {s: f90 for s in screen.SETS}
        rows.append(r)
    rows[2]["judgement"] = {"keys": ["policy_kl", "value_mse", "policy_top1"], "why": "passes",
                            "delta": {"kl": 0.006, "mse": 0.0, "top1": 0.001}}
    (tmp_path / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    screen.report()
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert "| A1 | `model.dropout=0.0` | A0 |" in text and "**ADOPT**" in text
    assert "Best arm **A1**" in text and "run: {name: A1}" in text


def test_trainer_full_eval_reports_extra_sets(tmp_path):
    """H3 wiring: a tiny CPU run with eval.extra_sets logs each set for raw and
    EMA at its full eval; frozen90 alone decides best.pt."""
    from training.v6.tests.test_m5_trainer import events, train
    train(tmp_path, "x", "schedule.total_samples=6400", "schedule.warmup_samples=640",
          "ckpt.every_samples=6400", "ckpt.stable_every_samples=0", "eval.quick_every_samples=6400",
          "eval.extra_sets=[v2val_derived]")
    full = events(tmp_path / "x", "full_eval")
    assert sorted((e["set"], e["weights"]) for e in full) == [
        ("frozen90", "ema"), ("frozen90", "raw"), ("v2val_derived", "ema"), ("v2val_derived", "raw")]
    d = [e for e in full if e["set"] == "v2val_derived"][0]
    assert d["n"] == 161_340 and d["policy_n"] == 0 and d["hard_n"] == 161_340
    best = torch.load(tmp_path / "x/best.pt", weights_only=True)
    assert best["value"] == min(e["total"] for e in full if e["set"] == "frozen90")
