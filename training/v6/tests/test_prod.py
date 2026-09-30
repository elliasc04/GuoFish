"""VM harness (brief §2-§4, §8): prod.yaml against A3, the stop rule, the store, and the
CPU dry run of tools/prod.py end to end: every stop path, the control actions, kill +
wipe + resume from the store (bit-identical on CPU) and checksum-sidecar refusal.

The dry run uses a directory store and synthetic shards built from frozen90 v2 unless:
    PROD_DRYRUN_STORE=r2      R2 (credentials from .env), under runs/_test/<unique>/
    PROD_DRYRUN_DATA=pulled   data/processed/synth_* as pulled by `setup.sh --data-prefix`
The dry-run tests take ~7 minutes on CPU; `-k "not dryrun"` skips them.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
import yaml

from training.v6.config import build_config, diff_paths, load_config, to_plain
from training.v6.data.formats import REPO
from training.v6.r2 import DirStore, get_json, open_store, parse_sha_list, put_json
from training.v6.tools import prod
from training.v6.tools.branch_decay import branch_config

CONFIGS = REPO / "training/v6/config/configs"
DRY = "training/v6/config/configs/dryrun_cpu.yaml"
# every key prod.yaml changes relative to A3 (VM harness brief §2), and nothing else
OVERRIDES = {"run.name", "run.out_root", "data.corpus", "data.manifest", "data.strata", "data.workers",
             "mixture.groups", "schedule.total_samples", "schedule.warmup_samples", "schedule.decay_frac",
             "eval.quick_every_samples", "eval.extra_sets", "ckpt.keep_last", "ckpt.stable_every_samples",
             "system.allow_dirty"}


# ------------------------------------------------------------- configs

def test_a3_resolved_is_the_screening_config():
    src = REPO / "runs/screening/configs/A3.yaml"
    if not src.exists():
        pytest.skip("runs/screening is local to the screening box")
    assert (CONFIGS / "a3_resolved.yaml").read_bytes() == src.read_bytes()


def test_prod_equals_a3_except_the_overrides():
    a3 = build_config(yaml.safe_load((CONFIGS / "a3_resolved.yaml").read_text()))
    p = load_config(CONFIGS / "prod.yaml")
    assert diff_paths(to_plain(a3), to_plain(p)) == OVERRIDES
    assert (p.run.data_seed, p.run.init_seed) == (20260802, 20260802)
    assert p.optim.effective_batch == 1024 and p.model.contract == "B"
    for mb, acc in ((512, 2), (1024, 1)):     # both smoke splits are valid configs
        load_config(CONFIGS / "prod.yaml", [f"optim.micro_batch={mb}", f"optim.accum={acc}"])


def test_prod_grid_and_branch_geometry():
    cfg, p = prod.load("training/v6/config/configs/prod.yaml", [])
    bps = p.branch_points
    assert [round(b / 1e8) for b in bps] == [3, 6, 12, 24]
    assert cfg.schedule.decay_frac == 0.15 and cfg.schedule.warmup_samples == 6_000_000
    assert cfg.ema.half_life_samples == 2e6 and cfg.eval.quick_size == 32_768
    total = bps[-1]
    for bp in bps:
        name, end = prod.branch_of(cfg, bp)
        _, s_b, decay = branch_config({"config": to_plain(cfg), "samples": bp})
        assert name == f"s{s_b}_d{decay}"                     # the same name branch_decay writes
        assert abs((end - bp) / bp - 0.15 / 0.85) < 1e-5 and end % 1024 == 0
        total += end - bp
    assert 3.19e9 < total < 3.2e9
    assert p.sanity.samples == 40_960_000 and p.sanity.samples % 2_048_000 == 0   # on C0's quick grid too
    assert p.seen_eval.group == "policy" and p.seen_eval.n == 200_000


@pytest.mark.parametrize("sets, match", [
    (["prod.branch_points=[300001280,600000000]"], "multiples"),
    (["schedule.total_samples=2823541460"], "total_samples"),
    (["prod.sanity.samples=40000000"], "quick-val grid"),
])
def test_load_refuses_a_broken_grid(sets, match):
    with pytest.raises(SystemExit, match=match):
        prod.load("training/v6/config/configs/prod.yaml", sets)


# ----------------------------------------------------------- stop rule

def m(kl, mse):
    return {"policy_kl": kl, "value_mse": mse, "total": kl + mse}


def test_stop_rule():
    prev = {"raw": m(1.00, 0.10), "ema": m(0.99, 0.10)}
    first = prod.decide(0, 4, prev, None, 0.0075, [])
    assert first["go"] and "always" in first["reason"]
    bad = prod.decide(0, 4, prev, None, 0.0075, [{"kind": "nonfinite", "detail": "x"}])
    assert not bad["go"] and "anomaly" in bad["reason"]
    lag = prod.decide(0, 4, prev, None, 0.0075, [{"kind": "upload_lag", "detail": "x"}])
    assert lag["go"]                                   # not a run anomaly
    ok = prod.decide(1, 4, {"raw": m(0.98, 0.10), "ema": m(0.97, 0.12)}, prev, 0.0075, [])
    assert ok["go"] and abs(ok["kl_gain"] - 0.02 / 0.99) < 1e-12   # better of raw and EMA each side
    small = prod.decide(1, 4, {"raw": m(0.985, 0.10), "ema": m(0.99, 0.09)}, prev, 0.0075, [])
    assert not small["go"] and "insufficient gain" in small["reason"] and small["total_ok"]
    worse = prod.decide(2, 4, {"raw": m(0.95, 0.20), "ema": m(0.96, 0.20)}, prev, 0.0075, [])
    assert not worse["go"] and worse["reason"] == "worse total" and worse["kl_ok"]
    last = prod.decide(3, 4, {"raw": m(0.90, 0.05), "ema": m(0.90, 0.05)}, prev, 0.0075, [])
    assert not last["go"] and "last branch" in last["reason"]


def test_ship_picks_best_total_over_branches_and_weights():
    br = [{"name": "a", "frozen90": {"raw": m(1.0, 0.1), "ema": m(0.9, 0.1)}},
          {"name": "b", "frozen90": {"raw": m(0.95, 0.1), "ema": m(0.96, 0.1)}}]
    assert prod.pick_ship(br) == {"branch": "a", "weights": "ema", "frozen90_total": 1.0}
    assert prod.pick_ship([]) is None


# ---------------------------------------------------------------- store

def test_sha_list_and_local_paths(tmp_path):
    good = f"{'a' * 64}  12  processed/x/train_0000.bin\n"
    assert parse_sha_list(good) == [("a" * 64, 12, "processed/x/train_0000.bin")]
    for bad in (f"{'a' * 64}  12  ../x", f"{'a' * 64}  12  /abs", f"{'a' * 64}  12  a\\b", "short 12 x"):
        with pytest.raises(SystemExit):
            parse_sha_list(bad)
    r = tmp_path
    assert prod.local_path(r, "ckpt/rolling/s10.pt") == r / "ckpt/s10.pt"
    assert prod.local_path(r, "ckpt/stable/s10.pt") == r / "stable/s10.pt"
    assert prod.local_path(r, "branches/s1_d2/rolling/s10.pt") == r / "branches/s1_d2/ckpt/s10.pt"
    assert prod.local_path(r, "branches/s1_d2/final.pt") == r / "branches/s1_d2/final.pt"
    s = DirStore(tmp_path / "store")
    put_json(s, "a/b.json", {"x": 1})
    assert get_json(s, "a/b.json") == {"x": 1} and get_json(s, "a/none.json") is None
    assert s.list("a/") == {"a/b.json": len(s.get_bytes("a/b.json"))}


# ------------------------------------------------------------- dry run

@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    root = tmp_path_factory.mktemp("dry")
    if os.environ.get("PROD_DRYRUN_DATA") == "pulled":
        data = REPO / "data"
    else:
        from training.v6.tools.make_synth import build
        data = root / "data"
        build(data, 16384, 2048)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    prefix = "runs/_test/"
    if os.environ.get("PROD_DRYRUN_STORE") == "r2":
        env.pop("PROD_STORE", None)
        prefix = f"runs/_test/{uuid.uuid4().hex[:8]}/"
    else:
        env["PROD_STORE"] = f"file://{(root / 'store').as_posix()}"
    old = os.environ.get("PROD_STORE")
    os.environ.update({k: env[k] for k in ("PROD_STORE",) if k in env})
    if "PROD_STORE" not in env:
        os.environ.pop("PROD_STORE", None)
    d = (data / "processed").as_posix()
    sets = [f"run.out_root={(root / 'models').as_posix()}", "system.allow_dirty=true",
            f"prod.store_prefix={prefix}", f"data.corpus={d}/synth_v6",
            f"data.manifest={d}/synth_v6/manifest.json", f"data.strata={d}/strata/synth_v6_train.strata2.npy",
            f"eval.frozen_dir={d}/synth_v6_val", f"eval.frozen_strata={d}/strata/synth_v6_val_val.strata2.npy"]
    yield {"root": root, "env": env, "sets": sets, "prefix": prefix, "store": open_store(), "data": d, "ref": {}}
    if old is None:
        os.environ.pop("PROD_STORE", None)
    else:
        os.environ["PROD_STORE"] = old


def launch(dry, name, *sets, resume=False):
    args = ["--resume", name, "--config", DRY, "--set", *dry["sets"], *sets] if resume else \
        ["--config", DRY, "--set", *dry["sets"], f"run.name={name}", *sets]
    log = open(dry["root"] / f"{name}.log", "a")
    kw = {"start_new_session": True} if os.name == "posix" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return subprocess.Popen([sys.executable, "-u", "-m", "training.v6.tools.prod", *args], cwd=REPO,
                            env=dry["env"], stdout=log, stderr=subprocess.STDOUT, **kw)


def finish(dry, proc, name, want=0, timeout=900):
    rc = proc.wait(timeout=timeout)
    text = (dry["root"] / f"{name}.log").read_text()
    assert rc == want, f"prod.py exit {rc} (want {want})\n{text[-4000:]}"
    return text


def kill_tree(proc):
    if os.name == "posix":
        os.killpg(proc.pid, signal.SIGKILL)
    else:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    proc.wait()


def sj(dry, name, key):
    return get_json(dry["store"], f"{dry['prefix']}{name}/{key}")


def until(pred, what, timeout=600):
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        v = pred()
        if v:
            return v
        time.sleep(0.5)
    raise AssertionError(f"timed out waiting for {what}")


def control(dry, name, action):
    put_json(dry["store"], f"{dry['prefix']}{name}/control.json", {"action": action, "by": "test", "utc": time.time()})


def steps_of(dry, name) -> dict:
    """(segment, step) -> step event, from the store's metrics segments; a replayed step
    (after a resume) keeps its latest copy."""
    out = {}
    keys = sorted(k for k in dry["store"].list(f"{dry['prefix']}{name}/metrics/") if k.endswith(".jsonl"))
    for k in keys:
        for ln in dry["store"].get_bytes(k).decode().splitlines():
            ev = json.loads(ln)
            if ev["event"] == "step":
                out[(ev["segment"], ev["step"])] = ev
    return out


def test_dryrun_reference_run(dry):
    """Uninterrupted run: continues past the first branches, stops by the rule, ships
    and exports the best branch; every store object the runbook documents exists."""
    name = "ref"
    text = finish(dry, launch(dry, name, "prod.sanity.samples=0", "prod.min_kl_gain=-1"), name)
    st, summ = sj(dry, name, "state.json"), sj(dry, name, "summary.json")
    ds = st["decisions"]
    dry["ref"].update(kl5120=sj(dry, name, "evals/main_quick_s5120.json")["policy_kl"],
                      steps=steps_of(dry, name), decisions=ds)
    print(f"\nreference: {len(ds)} branches; " + "; ".join(f"{d['branch']}: {d['reason']}" for d in ds))
    assert st["phase"] == "done" and ds[0]["go"] and len(ds) >= 2 and not ds[-1]["go"]
    assert all(d["go"] for d in ds[:-1]) and st["stopped"] == ds[-1]["reason"]
    ship = st["shipped"]
    assert ship == summ["shipped"] and ship["contract"] == "B"
    best = min((b["frozen90"][w]["total"], b["name"], w) for b in st["branches"] for w in ("raw", "ema"))
    assert (ship["frozen90_total"], ship["branch"], ship["weights"]) == best
    keys = dry["store"].list(f"{dry['prefix']}{name}/")
    rel = {k[len(dry["prefix"]) + len(name) + 1:] for k in keys}
    assert {"config.resolved.yaml", "provenance.json", "code.patch", "state.json", "status.json",
            "summary.json", ship["export_key"], ship["export_key"] + ".sha256"} <= rel
    rolling = [k for k in rel if k.startswith("ckpt/rolling/") and k.endswith(".pt")]
    assert len(rolling) == 2 and all(k + ".sha256" in rel for k in rolling)
    stable = sorted(int(k[13:-3]) for k in rel if k.startswith("ckpt/stable/") and k.endswith(".pt"))
    assert stable == list(range(3840, stable[-1] + 1, 3840))            # all kept
    for d, b in zip(ds, st["branches"]):
        assert f"evals/decision_{d['branch']}.json" in rel and f"branches/{d['branch']}/final.pt.sha256" in rel
        g = b["memgap"]["seen_policy"]["ema"]
        assert f"evals/{b['name']}_memgap_s{b['end']}.json" in rel and g["valid"] == (g["group_passes"] >= 1)
    assert st["branches"][-1]["memgap"]["seen_policy"]["ema"]["valid"]      # a full pass by then
    status = sj(dry, name, "status.json")
    for k in ("phase", "segment", "samples", "step", "samples_per_s", "lr", "step_time_s", "step_time_frac",
              "gpu", "passes", "eta_segment_end_h", "eta_next_branch_end_h", "cap_remaining_h", "updated_utc"):
        assert k in status
    assert set(status["step_time_s"]) == {"interval", "data_wait", "h2d", "fwd_bwd", "optim", "eval"}
    assert set(status["passes"]) == {"policy", "value"}
    assert "Traceback" not in text


def test_dryrun_insufficient_gain(dry):
    name = "gain"
    finish(dry, launch(dry, name, "prod.sanity.samples=0", "prod.min_kl_gain=0.99"), name)
    st = sj(dry, name, "state.json")
    assert len(st["decisions"]) == 2 and st["decisions"][0]["go"]
    assert "insufficient gain" in st["stopped"] and st["shipped"]["export_key"]


def test_dryrun_worse_total(dry):
    """Value labels negated in training, value loss x10, policy loss x0.1: frozen-val MSE,
    and so total, gets worse from one branch to the next while the KL rule (min gain -100%) holds."""
    name = "worse"
    d = dry["data"]
    finish(dry, launch(dry, name, "prod.sanity.samples=0", "prod.min_kl_gain=-1",
                       f"data.corpus={d}/synth_v6_flip", f"data.manifest={d}/synth_v6_flip/manifest.json",
                       f"data.strata={d}/strata/synth_v6_flip_train.strata2.npy",
                       "targets.value.weight=10", "targets.policy_soft.weight=0.1"), name)
    st = sj(dry, name, "state.json")
    assert len(st["decisions"]) == 2 and st["stopped"] == "worse total" and st["decisions"][1]["kl_ok"]
    assert st["shipped"]["branch"] == st["decisions"][0]["branch"]


def test_dryrun_time_cap(dry):
    name = "cap"
    finish(dry, launch(dry, name, "prod.sanity.samples=0", "prod.time_cap_h=0.000001"), name)
    st = sj(dry, name, "state.json")
    assert st["stopped"] == "time cap" and len(st["decisions"]) == 1 and st["decisions"][0]["go"]
    cfg, p = prod.load(DRY, dry["sets"])
    tc = sj(dry, name, f"evals/timecap_{prod.branch_of(cfg, p.branch_points[1])[0]}.json")
    assert tc["ok"] is False and tc["projected_h"] > tc["time_cap_h"]


def test_dryrun_sanity_pause_then_stop_now(dry):
    """Quick-val KL off C0's by more than 3% at the check: the run pauses itself via
    control.json and records why; stop_now then ends it with nothing to ship."""
    name = "sanity"
    kl = dry["ref"].get("kl5120")
    if kl is None:
        pytest.skip("needs the reference run's quick-val KL")
    proc = launch(dry, name, f"prod.sanity.ref_kl={kl * 1.5}")
    until(lambda: (sj(dry, name, "state.json") or {}).get("paused"), "the sanity pause")
    ctl, st = sj(dry, name, "control.json"), sj(dry, name, "state.json")
    assert ctl["action"] == "pause" and ctl["by"] == "prod.py" and "sanity" in ctl["reason"]
    assert st["sanity"]["pass"] is False and st["main"]["samples"] == 5120      # halted on that checkpoint
    assert any(a["kind"] == "sanity" for a in st["anomalies"])
    control(dry, name, "stop_now")
    finish(dry, proc, name)
    st = sj(dry, name, "state.json")
    assert st["stopped"] == "operator: stop_now" and st["shipped"] is None and st["phase"] == "done"


def test_dryrun_pause_continue_stop_after_branch(dry):
    name = "ctl"
    kl = dry["ref"].get("kl5120")
    if kl is None:
        pytest.skip("needs the reference run's quick-val KL")
    proc = launch(dry, name, f"prod.sanity.ref_kl={kl}")
    until(lambda: ((sj(dry, name, "state.json") or {}).get("main") or {}).get("samples", 0) >= 2560, "a checkpoint")
    control(dry, name, "pause")
    st = until(lambda: (lambda s: s if s and s.get("paused") else None)(sj(dry, name, "state.json")), "pause")
    paused_at = st["main"]["samples"]
    time.sleep(3)
    assert sj(dry, name, "state.json")["main"]["samples"] == paused_at        # trainer is stopped
    assert st["sanity"] is None or st["sanity"]["pass"]
    control(dry, name, "continue")
    until(lambda: (sj(dry, name, "state.json") or {}).get("paused") is False, "continue")
    control(dry, name, "stop_after_branch")
    finish(dry, proc, name)
    st = sj(dry, name, "state.json")
    assert st["stopped"] == "operator: stop_after_branch" and len(st["decisions"]) == 1
    assert st["decisions"][0]["go"] and st["shipped"]["branch"] == st["decisions"][0]["branch"]
    if dry["ref"].get("steps"):          # a pause + resume replays nothing differently
        mine = steps_of(dry, name)
        for key, ev in mine.items():
            ref = dry["ref"]["steps"][key]
            assert ev["step_losses"] == ref["step_losses"] and ev["stream_sha"] == ref["stream_sha"], key


def test_dryrun_kill_wipe_resume_bit_identical(dry):
    """Kill prod.py and its trainer mid-segment, wipe the local run, refuse a tampered or
    missing sidecar, then --resume from the store alone: every step's losses and stream
    digest equal the uninterrupted reference run's, and so do the decisions."""
    name = "kill"
    ref = dry["ref"]
    if not ref.get("steps"):
        pytest.skip("needs the reference run")
    proc = launch(dry, name, f"prod.sanity.ref_kl={ref['kl5120']}", "prod.min_kl_gain=-1")
    until(lambda: (lambda s: s and len(s["branches"]) == 1 and s["phase"] == "stable"
                   and (s["main"] or {}).get("samples", 0) >= 10240)(sj(dry, name, "state.json")), "mid segment 1")
    kill_tree(proc)
    shutil.rmtree(dry["root"] / "models" / name)
    shutil.rmtree(dry["root"] / "models" / f".{name}_upload", ignore_errors=True)
    main = sj(dry, name, "state.json")["main"]
    side = f"{dry['prefix']}{name}/{main['key']}.sha256"
    good = dry["store"].get_bytes(side)
    dry["store"].put_bytes(("0" * 64).encode() + good[64:], side)
    text = finish(dry, launch(dry, name, resume=True), name, want=1)
    assert "refusing" in text and "sidecar" in text
    dry["store"].delete(side)
    text = finish(dry, launch(dry, name, resume=True), name, want=1)
    assert "sidecar is missing" in text
    dry["store"].put_bytes(good, side)
    finish(dry, launch(dry, name, resume=True), name)
    st = sj(dry, name, "state.json")
    assert [d["reason"] for d in st["decisions"]] == [d["reason"] for d in ref["decisions"]]
    assert [d["total"] for d in st["decisions"]] == [d["total"] for d in ref["decisions"]]
    mine = steps_of(dry, name)
    assert set(mine) == set(ref["steps"])
    for key, ev in mine.items():
        r = ref["steps"][key]
        assert ev["step_losses"] == r["step_losses"] and ev["stream_sha"] == r["stream_sha"], key
    print(f"\nkill/resume: {len(mine)} logged steps across {len({k[0] for k in mine})} segments bit-identical")
