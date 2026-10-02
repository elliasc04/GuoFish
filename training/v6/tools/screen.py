"""Screening queue driver (H4, brief of 2026-09-26).

    python -m training.v6.tools.screen [--queue training/v6/screen_queue.yaml] [--out runs/screening]
    python -m training.v6.tools.screen report          # regenerate the report only

--out elsewhere (a smoke test) also moves the report into that directory.

Launch it detached so it outlives the agent session (PowerShell, repo root):

    $env:CUDA_VISIBLE_DEVICES = "-1"
    Start-Process python -ArgumentList "-u","-m","training.v6.tools.screen" -WindowStyle Hidden `
        -RedirectStandardOutput runs/screening/driver.out -RedirectStandardError runs/screening/driver.err

runs/screening/ledger.jsonl and training/v6/SCREENING_REPORT.md are the source of
truth for progress. The queue is re-read before every arm, so entries can be
appended or edited while it runs; an arm is done once it has a ledger row, so a
relaunch continues where the last one stopped (a finished run is scored, a
killed one resumed).

Queue entries (YAML list `arms`):
    {name, set: [key=value, ...], base, ref, also, arch, adopt, confirms, metrics, only_if}
    {pause: <message>, id: <id>}
  base      whose overrides the arm builds on: `best` (default), an arm, or null (the queue's config)
  ref       what the verdict compares against: an arm, `best`, or null (default: whatever `base` resolved to)
  also      further arms to report deltas against (not used for the verdict)
  arch      measure inference cost first (tools/fwd_cost.py, arm vs ref) and gate on it
  adopt     false: never becomes the best (calibration and confirmation runs)
  confirms  `best` or an arm: confirmation of that arm's seed-s delta against the reference arm
  metrics   policy (frozen90 KL, MSE) | pv (PV-restricted KL, MSE; top-1 vs SF's move reported);
            default pv when the arm trains hard labels (targets.policy_hard.weight > 0)
  only_if   {adopted: <arm>} or {not_adopted: <arm>}
A pause entry creates runs/screening/HOLD with its message and waits for the
operator to delete it; HOLD is honoured between any two runs.

Per arm: [fwd_cost] -> train under auto-resume (at most MAX_RESTARTS restarts) ->
score the final raw and EMA weights on frozen90, v2val_roots, v2val_derived
(tools/score.py) -> delete rolling checkpoints and best.pt -> ledger row with
deltas and the verdict (section 4 of the brief) -> report. Children get the GPU
(CUDA_VISIBLE_DEVICES removed); this process imports no torch.

Stops (writes runs/screening/STOP with the reason and exits 1; delete STOP to
relaunch): an anomaly event or non-finite step, an arm failing after its
restarts, a failed score or fwd_cost, C: below MIN_FREE_GB, calibration
r_KL above R_KL_MAX, or an arm judged before calibration exists.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "runs" / "screening"
QUEUE = REPO / "training" / "v6" / "screen_queue.yaml"
REPORT = REPO / "training" / "v6" / "SCREENING_REPORT.md"
PY = [sys.executable, "-u"]
MAX_RESTARTS = 2
MIN_FREE_GB = 25
R_KL_MAX = 0.005                                  # calibration: stop above 0.5%
FLOOR = {"kl": 0.0025, "mse": 0.017}              # capacity campaign's paired LR-contrast gaps
SETS = ("frozen90", "v2val_roots", "v2val_derived")
HEAD = ("n", "policy_n", "hard_n", "policy_kl", "value_mse", "total", "policy_top1", "pv_kl",
        "sf_top1_pv0", "sf_top1_hard", "hard_nll", "policy_entropy", "value_pearson_r",
        "tier/old/policy_kl", "tier/old/value_mse", "tier/new/policy_kl", "tier/new/value_mse",
        "material/compensated/value_bias")
ENTRY_KEYS = {"name", "set", "base", "ref", "also", "arch", "adopt", "confirms", "metrics", "only_if",
              "pause", "id"}
LOWER_BETTER = {"policy_kl", "pv_kl", "value_mse", "total", "hard_nll"}


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jsonl(path: Path) -> list:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()] \
        if path.exists() else []


def event(**kw) -> None:
    kw = {"utc": utc(), **kw}
    with open(OUT / "driver.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


class Stop(Exception):
    """Stop the queue and report."""


# --------------------------------------------------------------- decisions

def keys(metrics: str) -> tuple[str, str, str]:
    """(KL, MSE, top-1) keys of the primary metrics."""
    return (("pv_kl", "value_mse", "sf_top1_hard") if metrics == "pv"
            else ("policy_kl", "value_mse", "policy_top1"))


def rel_delta(arm: dict, ref: dict, key: str) -> float | None:
    """Relative improvement of arm over ref; positive is better."""
    a, r = arm.get(key), ref.get(key)
    if a is None or r is None or r == 0:
        return None
    return (r - a) / r if key in LOWER_BETTER else (a - r) / r


def calibrate(a0: dict, a0r: dict, r_kl_max: float = R_KL_MAX) -> dict:
    """r per metric (|A0r - A0| / A0 on frozen90) for both weight sets, the
    primary weights (whichever A0 scores better on frozen90 total, fixed from
    then on), the floors, and whether pairing is too noisy to go on."""
    f90 = lambda row, w: row["scores"][w]["frozen90"]  # noqa: E731
    w = min(("raw", "ema"), key=lambda x: f90(a0, x)["total"])
    r = {x: {k: abs(rel_delta(f90(a0r, x), f90(a0, x), k)) for k in
             ("policy_kl", "value_mse", "pv_kl", "total", "policy_top1", "sf_top1_hard")}
         for x in ("raw", "ema")}
    f = {"policy_kl": max(r[w]["policy_kl"], FLOOR["kl"]), "pv_kl": max(r[w]["pv_kl"], FLOOR["kl"]),
         "value_mse": max(r[w]["value_mse"], FLOOR["mse"])}
    return {"weights": w, "r": r, "floors": f, "r_kl_max": r_kl_max,
            "stop": r[w]["policy_kl"] > r_kl_max, "reference": a0["arm"], "replicate": a0r["arm"]}


def judge(arm: dict, ref: dict, cal: dict, metrics: str, required: float | None) -> dict:
    """Section 4: adopt if dKL >= 2 f_KL and dMSE >= -f_MSE, or dMSE >= 2 f_MSE
    and dKL >= -f_KL; an architecture arm's KL gain must also exceed its
    required gain. Anything else keeps the reference (ties: the cheaper option,
    which here is always the reference)."""
    w, (kk, mk, tk) = cal["weights"], keys(metrics)
    a, r = arm["scores"][w]["frozen90"], ref["scores"][w]["frozen90"]
    d = {"kl": rel_delta(a, r, kk), "mse": rel_delta(a, r, mk), "top1": rel_delta(a, r, tk)}
    fk, fm = cal["floors"][kk], cal["floors"][mk]
    adopt = ((d["kl"] >= 2 * fk and d["mse"] >= -fm) or (d["mse"] >= 2 * fm and d["kl"] >= -fk))
    why = "passes the noise rule" if adopt else "inside the floors or a regression"
    if adopt and required is not None and not 100 * d["kl"] > required:
        adopt, why = False, f"KL gain {100 * d['kl']:.2f}% does not exceed the required {required:.2f}%"
    return {"metrics": metrics, "keys": [kk, mk, tk], "delta": d, "floors": {"kl": fk, "mse": fm},
            "required_kl_gain_pct": required, "adopt": adopt, "why": why}


def confirm(delta_s: dict, delta_t: dict) -> dict:
    """Section 3.4: on every primary metric the recipe gained on at seed s, its
    seed-t delta has the same sign and at least half the size."""
    per = {k: {"s": delta_s[k], "t": delta_t[k],
               "ok": delta_t[k] is not None and delta_t[k] > 0 and delta_t[k] >= delta_s[k] / 2}
           for k in ("kl", "mse") if delta_s[k] is not None and delta_s[k] > 0}
    return {"per_metric": per, "confirmed": bool(per) and all(v["ok"] for v in per.values())}


# ------------------------------------------------------------------ queue

def load_queue(path: Path) -> dict:
    q = yaml.safe_load(path.read_text(encoding="utf-8"))
    for e in q["arms"]:
        if set(e) - ENTRY_KEYS:          # also catches YAML 1.1's `on:` / `yes:` read as booleans
            raise SystemExit(f"queue entry {e}: unknown keys {set(e) - ENTRY_KEYS}")
        if ("name" in e) == ("pause" in e):
            raise SystemExit(f"queue entry {e}: exactly one of name / pause")
    ids = [e.get("id") for e in q["arms"] if "pause" in e]
    if None in ids or len(set(ids)) != len(ids):
        raise SystemExit("queue: every pause needs a unique id")
    return q


def best(ledger: list) -> str | None:
    return ledger[-1]["best_after"] if ledger else None


def row_of(ledger: list, name: str) -> dict:
    hits = [r for r in ledger if r["arm"] == name]
    if not hits:
        raise Stop(f"arm {name!r} has no ledger row")
    return hits[-1]


def condition(entry: dict, ledger: list) -> bool:
    c = entry.get("only_if")
    if not c:
        return True
    (kind, arm), = c.items()
    done = {r["arm"]: r for r in ledger}
    if arm not in done:
        return False
    return done[arm]["adopted"] == (kind == "adopted")


def next_entry(q: dict, ledger: list, released: set) -> dict | None:
    done = {r["arm"] for r in ledger}
    for e in q["arms"]:
        if "pause" in e:
            if e["id"] not in released:
                return e
        elif e["name"] not in done and condition(e, ledger):
            return e
    return None


def compose(entry: dict, ledger: list) -> tuple[str | None, str | None, list]:
    """(base arm, ref arm, overrides): the base arm's overrides plus the entry's."""
    base = entry.get("base", "best")
    base = best(ledger) if base == "best" else base
    inherited = row_of(ledger, base)["overrides"] if base else []
    ref = entry.get("ref", base)
    ref = best(ledger) if ref == "best" else ref
    return base, ref, [*inherited, *entry.get("set", [])]


# -------------------------------------------------------------- the work

def child_env() -> dict:
    return {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}


def resolve(config: str, name: str, overrides: list) -> dict:
    """Resolve and validate an arm's config in a subprocess (keeps torch out of
    this long-lived process); writes OUT/configs/<name>.yaml."""
    out = OUT / "configs" / f"{name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(PY + ["-m", "training.v6.tools.screen", "resolve", str(out), config,
                             f"run.name={name}", *overrides],
                       cwd=REPO, capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    if r.returncode:
        raise Stop(f"{name}: config does not resolve: {r.stderr[-2000:]}")
    return {"path": out, **json.loads(r.stdout)}


def _resolve_main(out: str, config: str, *overrides: str) -> int:
    from training.v6.config import config_hash, dump_yaml, load_config
    cfg = load_config(REPO / config, list(overrides))
    Path(out).write_text(dump_yaml(cfg), encoding="utf-8")
    print(json.dumps({"config_hash": config_hash(cfg), "w_hard": cfg.targets.policy_hard.weight,
                      "run_dir": str(Path(cfg.run.out_root) / cfg.run.name),
                      "model": cfg.model.to_dict()}))
    return 0


def disk_free_gb() -> float:
    return shutil.disk_usage(REPO).free / 1e9


def log_events(run_dir: Path, kind: str | None = None) -> list:
    rows = jsonl(run_dir / "logs" / "train.jsonl")
    return [r for r in rows if kind is None or r["event"] == kind]


def check_healthy(name: str, run_dir: Path) -> None:
    bad = [r for r in log_events(run_dir) if r["event"] == "anomaly"
           or (r["event"] == "step" and r.get("nonfinite_steps"))]
    if bad:
        raise Stop(f"{name}: anomaly / non-finite step at samples {bad[0].get('samples')}")


def train(name: str, cfg_path: Path, run_dir: Path) -> int:
    """Run to completion under auto-resume; returns the number of attempts."""
    if log_events(run_dir, "run_end"):
        event(event="already_trained", arm=name)
        return 0
    cmd = PY + ["-m", "training.v6.train", "--config", str(cfg_path)]
    for attempt in range(MAX_RESTARTS + 1):
        resume = any((run_dir / "ckpt").glob("s*.pt"))
        if not resume and run_dir.exists():
            aside = run_dir.with_name(f"{run_dir.name}_failed{attempt}_{int(time.time())}")
            run_dir.rename(aside)
            event(event="moved_aside", arm=name, to=str(aside.relative_to(REPO)))
        full = cmd + (["--resume", "latest"] if resume else [])
        event(event="start", arm=name, attempt=attempt, resume=resume)
        t0 = time.time()
        with open(OUT / "logs" / f"{name}_attempt{attempt}.log", "w", encoding="utf-8") as f:
            rc = subprocess.run(full, cwd=REPO, env=child_env(), stdout=f, stderr=subprocess.STDOUT).returncode
        event(event="exit", arm=name, attempt=attempt, returncode=rc, seconds=round(time.time() - t0))
        check_healthy(name, run_dir)
        if rc == 0:
            return attempt + 1
        if disk_free_gb() < MIN_FREE_GB:
            raise Stop(f"{name}: C: has {disk_free_gb():.1f} GB free (< {MIN_FREE_GB})")
    raise Stop(f"{name}: failed {MAX_RESTARTS + 1} times (exit {rc}); see runs/screening/logs")


def score(name: str, ckpt: Path) -> dict:
    out = {}
    for w in ("raw", "ema"):
        path = OUT / "scores" / f"{name}_{w}.json"
        if not path.exists():
            event(event="score", arm=name, weights=w)
            r = subprocess.run(PY + ["-m", "training.v6.tools.score", str(ckpt), "--weights", w,
                                     "--sets", *SETS], cwd=REPO, env=child_env(), capture_output=True, text=True)
            if r.returncode or not r.stdout.strip().startswith("{"):
                raise Stop(f"{name}: scoring {w} failed (exit {r.returncode}): {r.stderr[-2000:]}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r.stdout, encoding="utf-8")
        sc = json.loads(path.read_text(encoding="utf-8"))
        out[w] = {s: {k: sc["sets"][s].get(k) for k in HEAD} for s in SETS}
    return out


def fwd_cost(name: str, cfg: dict, ref: str | None) -> dict:
    path = OUT / "fwd_cost" / f"{name}.json"
    if not path.exists():
        cmd = PY + ["-m", "training.v6.tools.fwd_cost", "--arm", f"{name}={cfg['path']}"]
        if ref:
            cmd += ["--ref", f"{ref}={OUT / 'configs' / f'{ref}.yaml'}"]
        event(event="fwd_cost", arm=name, ref=ref)
        r = subprocess.run(cmd, cwd=REPO, env=child_env(), capture_output=True, text=True)
        if r.returncode or not r.stdout.strip().startswith("{"):
            raise Stop(f"{name}: fwd_cost failed (exit {r.returncode}): {r.stderr[-2000:]}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.stdout, encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def cleanup(run_dir: Path, final: Path) -> list:
    gone = [p for p in (run_dir / "ckpt").glob("s*.pt") if p != final] + \
           [p for p in [run_dir / "best.pt"] if p.exists()]
    for p in gone:
        p.unlink()
    return [p.relative_to(REPO).as_posix() for p in gone]


def throughput(run_dir: Path) -> dict:
    rows = log_events(run_dir)
    starts, rate, vram = set(), [], 0.0
    for r in rows:
        if r["event"] == "run_start":
            starts.add(r["step"])
        elif r["event"] == "step":
            vram = max(vram, r.get("peak_vram_mib") or 0.0)
            if not any(s < r["step"] <= s + 100 for s in starts) and r.get("samples_per_s"):
                rate.append(r["samples_per_s"])      # skip each start's compile interval
    return {"samples_per_s_median": statistics.median(rate) if rate else None,
            "peak_vram_mib": vram, "starts": len(starts)}


def run_arm(entry: dict, q: dict, ledger: list, cal: dict | None) -> dict:
    name = entry["name"]
    cal_names = q.get("calibration", {})
    base, ref, overrides = compose(entry, ledger)
    judged = name not in cal_names.values() and ref is not None
    if judged and cal is None:
        raise Stop(f"{name}: no calibration yet; the queue must run the reference and replicate first")
    cfg = resolve(q["config"], name, overrides)
    run_dir = REPO / cfg["run_dir"]
    metrics = entry.get("metrics", "pv" if cfg["w_hard"] > 0 else "policy")
    cost = fwd_cost(name, cfg, ref) if entry.get("arch") else None
    t0 = utc()
    attempts = train(name, cfg["path"], run_dir)
    end = log_events(run_dir, "run_end")[-1]["samples"]
    final = run_dir / "ckpt" / f"s{end}.pt"
    scores = score(name, final)
    deleted = cleanup(run_dir, final)
    row = {"arm": name, "utc_start": t0, "utc_end": utc(), "base": base, "ref": ref,
           "change": entry.get("set", []), "overrides": overrides, "config_hash": cfg["config_hash"],
           "config": cfg["path"].relative_to(REPO).as_posix(), "run_dir": Path(cfg["run_dir"]).as_posix(),
           "final_ckpt": final.relative_to(REPO).as_posix(), "deleted": deleted, "attempts": attempts,
           "metrics": metrics, "scores": scores, "fwd_cost": cost, **throughput(run_dir),
           "also": {}, "adopted": False}
    if name == cal_names.get("reference"):
        row.update(verdict="reference", best_after=name)
        return row
    if not judged:
        row.update(verdict="replicate" if name == cal_names.get("replicate") else "no reference",
                   best_after=best(ledger))
        return row
    required = cost["arm_vs_ref"]["required_kl_gain_pct"] if cost else None
    j = judge(row, row_of(ledger, ref), cal, metrics, required)
    for other in entry.get("also", []):
        row["also"][other] = judge(row, row_of(ledger, other), cal, metrics, None)["delta"]
    if entry.get("confirms"):
        target = best(ledger) if entry["confirms"] == "best" else entry["confirms"]
        s = judge(row_of(ledger, target), row_of(ledger, cal_names["reference"]), cal, metrics, None)
        row["confirmation"] = {"of": target, **confirm(s["delta"], j["delta"])}
    adopt = j["adopt"] and entry.get("adopt", True)
    row.update(judgement=j, adopted=adopt, best_after=name if adopt else best(ledger),
               verdict="ADOPT" if adopt else ("not a candidate" if not entry.get("adopt", True) else "reject"))
    return row


# ----------------------------------------------------------------- report

def _pct(x, nd=2):
    return "–" if x is None else f"{100 * x:+.{nd}f}%"


def _f(x, nd=5):
    return "–" if x is None else f"{x:.{nd}f}"


def report() -> None:
    ledger = jsonl(OUT / "ledger.jsonl")
    cal = json.loads((OUT / "calibration.json").read_text()) if (OUT / "calibration.json").exists() else None
    ev = jsonl(OUT / "driver.jsonl")
    w = cal["weights"] if cal else "raw"
    L = ["# v6 screening pass — report", "",
         "Regenerated by `training/v6/tools/screen.py` after every run; finalised by hand. Brief:",
         "`docs/capacity/screening_pass.md`. Decisions: `training/v6/DECISIONS.md`, \"Brief of 2026-09-26\".",
         "Every number below is **[measured]** (ledger `runs/screening/ledger.jsonl`, scores",
         "`runs/screening/scores/`) unless tagged otherwise.", "", "## Status", ""]
    last = ev[-1] if ev else {}
    L += [f"- Updated {utc()}. Last driver event: `{json.dumps(last)[:300]}`",
          f"- Arms done: {len(ledger)}. Current best: **{best(ledger) or '–'}**.",
          f"- HOLD: {'yes — ' + (OUT / 'HOLD').read_text(encoding='utf-8').strip()[:300] if (OUT / 'HOLD').exists() else 'no'}.",
          f"- STOP: {'yes — ' + (OUT / 'STOP').read_text(encoding='utf-8').strip()[:500] if (OUT / 'STOP').exists() else 'no'}.", ""]

    L += ["## Calibration", ""]
    if cal:
        L += [f"- Primary weights: **{cal['weights']}** (A0's better frozen90 total; fixed from here on).",
              f"- r (|{cal['replicate']} − {cal['reference']}| / {cal['reference']}, frozen90):", "",
              "| weights | KL | MSE | PV-KL | total | top-1 | top-1 vs SF |", "|---|---:|---:|---:|---:|---:|---:|"]
        for x in ("raw", "ema"):
            r = cal["r"][x]
            L.append(f"| {x} | {_pct(r['policy_kl'], 3)} | {_pct(r['value_mse'], 3)} | {_pct(r['pv_kl'], 3)} | "
                     f"{_pct(r['total'], 3)} | {_pct(r['policy_top1'], 3)} | {_pct(r['sf_top1_hard'], 3)} |")
        fl = cal["floors"]
        L += ["", f"- Floors: f_KL {_pct(fl['policy_kl'], 3)}, f_PV-KL {_pct(fl['pv_kl'], 3)}, "
              f"f_MSE {_pct(fl['value_mse'], 3)}. Stop rule r_KL > {100 * cal['r_kl_max']:.1f}%: "
              f"**{'FIRED' if cal['stop'] else 'not fired'}**.", ""]
    else:
        L += ["Not yet (needs the reference and its replicate).", ""]

    L += [f"## Decisions ({w} weights, frozen90 v2)", "",
          "Δ is the relative improvement over `ref` (positive is better). Cost: forward time at batch 24 "
          "vs `ref` (vs plain d384×10 in brackets) and the KL gain it requires.", "",
          "| arm | change | ref | KL | MSE | ΔKL | ΔMSE | Δtop-1 | cost | verdict | best after |",
          "|---|---|---|---:|---:|---:|---:|---:|---|---|---|"]
    for r in ledger:
        f = r["scores"][w]["frozen90"]
        j = r.get("judgement") or {}
        kk, mk = (j.get("keys") or keys(r["metrics"]))[:2]
        d = j.get("delta", {})
        c = r.get("fwd_cost")
        cost = "–" if not c else (f"{c['arm_vs_ref']['ratio_b24']:.3f}× "
                                  f"({c['models'][r['arm']]['ratio_b24']:.3f}×), needs "
                                  f"{c['arm_vs_ref']['required_kl_gain_pct']:+.2f}%")
        change = "; ".join(r["change"]) or "–"
        tag = " (PV)" if kk == "pv_kl" else ""
        L.append(f"| {r['arm']} | `{change[:90]}` | {r['ref'] or '–'} | {_f(f[kk])}{tag} | {_f(f[mk])} | "
                 f"{_pct(d.get('kl'))} | {_pct(d.get('mse'))} | {_pct(d.get('top1'))} | {cost} | "
                 f"**{r['verdict']}**{': ' + j['why'] if j.get('why') else ''} | {r['best_after']} |")
    also = [(r["arm"], o, d) for r in ledger for o, d in r.get("also", {}).items()]
    if also:
        L += ["", "Also compared: " + "; ".join(f"{a} vs {o}: ΔKL {_pct(d['kl'])}, ΔMSE {_pct(d['mse'])}"
                                                 for a, o, d in also) + "."]

    L += ["", f"## Every eval set ({w} weights)", "",
          "| arm | f90 KL | f90 MSE | f90 PV-KL | f90 top-1 | f90 top-1 vs SF | v2val_roots KL | MSE | "
          "new-tier KL | new-tier MSE | v2val_derived top-1 vs SF | hard NLL | MSE |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in ledger:
        f, v, dv = (r["scores"][w][s] for s in SETS)
        L.append(f"| {r['arm']} | {_f(f['policy_kl'])} | {_f(f['value_mse'])} | {_f(f['pv_kl'])} | "
                 f"{_f(f['policy_top1'], 4)} | {_f(f['sf_top1_hard'], 4)} | {_f(v['policy_kl'])} | "
                 f"{_f(v['value_mse'])} | {_f(v['tier/new/policy_kl'])} | {_f(v['tier/new/value_mse'])} | "
                 f"{_f(dv['sf_top1_hard'], 4)} | {_f(dv['hard_nll'], 4)} | {_f(dv['value_mse'])} |")
    L += ["", "Raw vs EMA, frozen90 total: " + "; ".join(
        f"{r['arm']} {_f(r['scores']['raw']['frozen90']['total'])} / {_f(r['scores']['ema']['frozen90']['total'])}"
        for r in ledger) + "." if ledger else ""]

    L += ["", "## Adopted recipe", ""]
    b = best(ledger)
    if b:
        br = row_of(ledger, b)
        L += [f"Best arm **{b}**: `proxy.yaml` plus `{'; '.join(br['overrides']) or 'nothing'}`. Resolved "
              f"config (`{br['config']}`, hash `{br['config_hash'][:12]}`):", "", "```yaml",
              (REPO / br["config"]).read_text(encoding="utf-8").rstrip(), "```"]

    conf = [r for r in ledger if r.get("confirmation")]
    L += ["", "## Confirmation (seed t)", ""]
    for r in conf:
        c = r["confirmation"]
        L.append(f"- {r['arm']} confirms {c['of']}: **{'CONFIRMED' if c['confirmed'] else 'NOT confirmed'}** — " +
                 "; ".join(f"Δ{k} s {_pct(v['s'])} → t {_pct(v['t'])}" for k, v in c["per_metric"].items()))
    if not conf:
        L.append("Not yet. The 200-game head-to-head at 12k sims waits on the engine loader switch (S7), "
                 "which is not made here.")

    L += ["", "## Throughput, VRAM, restarts", "",
          "| arm | samples/s (median) | peak VRAM MiB | attempts | starts | wall |", "|---|---:|---:|---:|---:|---|"]
    for r in ledger:
        L.append(f"| {r['arm']} | {r['samples_per_s_median'] or 0:,.0f} | {r['peak_vram_mib']:,.0f} | "
                 f"{r['attempts']} | {r['starts']} | {r['utc_start']} → {r['utc_end']} |")
    restarts = [e for e in ev if e.get("event") == "exit" and e.get("returncode")]
    L += ["", f"Non-zero trainer exits: {len(restarts)}" + (": " + "; ".join(
        f"{e['arm']} attempt {e['attempt']} exit {e['returncode']}" for e in restarts) if restarts else "") + ".",
          "", "## Deviations, and anything that should change the production plan", "",
          "_Finalised by hand; see DECISIONS.md._", ""]
    REPORT.write_text("\n".join(L), encoding="utf-8", newline="\n")


# ------------------------------------------------------------------- loop

def stop(reason: str) -> int:
    (OUT / "STOP").write_text(f"{utc()} {reason}\n", encoding="utf-8")
    event(event="stopped", reason=reason)
    report()
    return 1


def main(argv=None) -> int:
    global OUT, REPORT
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--out" in argv:
        OUT = REPO / argv[argv.index("--out") + 1]
        REPORT = OUT / "SCREENING_REPORT.md"
    if argv[:1] == ["resolve"]:
        return _resolve_main(*argv[1:])
    if argv[:1] == ["report"]:
        report()
        return 0
    queue = Path(argv[argv.index("--queue") + 1]) if "--queue" in argv else QUEUE
    for d in ("logs", "scores", "configs", "fwd_cost"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    if (OUT / "STOP").exists():
        print(f"refusing to start: {OUT / 'STOP'} says: {(OUT / 'STOP').read_text().strip()}")
        return 1
    event(event="driver_start", pid=os.getpid(), queue=str(queue))
    while True:
        if (OUT / "HOLD").exists():
            event(event="hold", message=(OUT / "HOLD").read_text(encoding="utf-8").strip()[:300])
            while (OUT / "HOLD").exists():
                time.sleep(30)
            event(event="hold_released")
        q = load_queue(queue)
        ledger = jsonl(OUT / "ledger.jsonl")
        released = {e["id"] for e in jsonl(OUT / "driver.jsonl") if e.get("event") == "pause_released"}
        entry = next_entry(q, ledger, released)
        if entry is None:
            event(event="queue_done", arms=len(ledger))
            report()
            return 0
        if "pause" in entry:
            (OUT / "HOLD").write_text(entry["pause"].strip() + "\n", encoding="utf-8")
            event(event="pause", id=entry["id"])
            report()
            while (OUT / "HOLD").exists():
                time.sleep(30)
            event(event="pause_released", id=entry["id"])
            continue
        if disk_free_gb() < MIN_FREE_GB:
            return stop(f"C: has {disk_free_gb():.1f} GB free (< {MIN_FREE_GB}) before {entry['name']}")
        cal_path = OUT / "calibration.json"
        cal = json.loads(cal_path.read_text()) if cal_path.exists() else None
        try:
            row = run_arm(entry, q, ledger, cal)
        except Stop as e:
            return stop(str(e))
        with open(OUT / "ledger.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        event(event="arm_done", arm=row["arm"], verdict=row["verdict"], best_after=row["best_after"])
        if row["verdict"] == "replicate":
            cal = calibrate(row_of(ledger, q["calibration"]["reference"]), row,
                            q["calibration"].get("r_kl_max", R_KL_MAX))
            cal_path.write_text(json.dumps(cal, indent=1) + "\n", encoding="utf-8")
            event(event="calibrated", weights=cal["weights"], r_kl=cal["r"][cal["weights"]]["policy_kl"])
            if cal["stop"]:
                return stop(f"calibration: r_KL {100 * cal['r'][cal['weights']]['policy_kl']:.3f}% > "
                            f"{100 * cal['r_kl_max']:.1f}%; pairing does not beat seed noise by enough")
        report()


if __name__ == "__main__":
    raise SystemExit(main())
