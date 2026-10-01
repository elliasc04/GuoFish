"""Production driver (VM harness brief §3-§5).

    python -m training.v6.tools.prod [--config C] [--set KEY=VALUE ...]        # fresh run
    python -m training.v6.tools.prod --resume <run> [--set KEY=VALUE ...]      # any VM, empty disk ok
    python -m training.v6.tools.prod --smoke [--mb 512x2 1024x1] [--steps 300] [--config C]

The loop: a stable segment to the next branch point (train.py --stop-at-samples),
that branch's decay (branch_decay.py), the pre-registered stop rule on frozen90,
the decision to the store, then the next segment, or export and exit. Every
trainer is a subprocess; this process tails its JSONL and streams checkpoints
(file, then .sha256, then state.json), evals, metrics and status to the store
from one background upload thread, and polls control.json. --smoke runs N steps
per micro-batch split and reports throughput; it never touches the store.
Store layout and every schema: training/v6/PROD_RUNBOOK.md.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")      # H2: before numpy
import argparse
import json
import math
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Optional

import yaml

from core.guofish_net.strict import from_dict_strict
from training.v6.ckpt import check_dirty, git_state, json_safe, sha256_file, utc
from training.v6.config import build_config, to_plain
from training.v6.config.loader import _read_yaml, apply_override
from training.v6.data.formats import REPO
from training.v6.r2 import get_json, open_store, put_json

try:
    from botocore.exceptions import BotoCoreError, ClientError
    STORE_ERRORS: tuple = (OSError, BotoCoreError, ClientError)
except ImportError:
    STORE_ERRORS = (OSError,)

PY = [sys.executable, "-u"]
DEFAULT_CONFIG = "training/v6/config/configs/prod.yaml"
A3_LOCAL_SAMPLES_PER_S = 3346     # SCREENING_REPORT.md: A3's median on the local 5070 [read]
ACTIONS = ("continue", "pause", "stop_after_branch", "stop_now")
STOP_ANOMALIES = ("nonfinite", "trainer_exit")     # the kinds that end the run after the first branch


@dataclass(frozen=True)
class Sanity:
    samples: int = 0                    # 0: no check
    ref_kl: Optional[float] = None      # primary reference: A3's quick-val policy KL at `samples`
    ref_run: str = ""                   # where ref_kl was read (run, config hash, log line)
    tol: float = 0.03


@dataclass(frozen=True)
class Pins:
    strata_definition_hash: str = ""    # must equal training.v6.data.strata.DEFINITION_HASH
    build_ok: str = ""                  # build_ok.json: the corpus manifest must hash to its manifest_sha256
    corpus_manifest_sha256: str = ""    # optional literal pin of the same hash


@dataclass(frozen=True)
class SeenEval:
    group: str = ""
    n: int = 0                          # 0: no memorization-gap eval


@dataclass(frozen=True)
class Settings:
    store_prefix: str = "runs/"
    branch_points: tuple[int, ...] = ()
    time_cap_h: float = 120.0
    min_kl_gain: float = 0.0075
    sanity: Sanity = field(default_factory=Sanity)
    seen_eval: SeenEval = field(default_factory=SeenEval)
    pins: Pins = field(default_factory=Pins)
    status_every_s: float = 60.0
    metrics_every_s: float = 300.0
    control_every_s: float = 300.0
    rolling_keep: int = 2
    max_upload_lag: int = 3


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def load(config: str, sets) -> tuple:
    d = _read_yaml(resolve(config), ())
    for spec in sets:
        d = apply_override(d, spec)
    if "prod" not in d:
        raise SystemExit(f"{config} has no prod: section")
    p = from_dict_strict(Settings, d.pop("prod"), "prod")
    cfg = build_config(d)
    bps, every = p.branch_points, cfg.ckpt.stable_every_samples
    if not bps or list(bps) != sorted(set(bps)):
        raise SystemExit("prod.branch_points must be non-empty and strictly increasing")
    if every <= 0 or any(b % every for b in bps):
        raise SystemExit(f"prod.branch_points must be multiples of ckpt.stable_every_samples ({every})")
    want = math.ceil(bps[-1] / (1.0 - cfg.schedule.decay_frac))
    if cfg.schedule.kind != "wsd" or cfg.schedule.total_samples != want:
        raise SystemExit(f"schedule must be wsd with total_samples={want}: the main line stays at "
                         f"peak LR through the last branch point, whose branch is the final decay")
    q = cfg.eval.quick_every_samples
    if p.sanity.samples and (not q or p.sanity.samples % q):
        raise SystemExit(f"prod.sanity.samples={p.sanity.samples} is not on the quick-val grid ({q})")
    return cfg, p


def check_pins(cfg, p: Settings) -> dict:
    """Refuse a run whose inputs drifted from the config's pins (follow-ups §4): the strata
    definition, the corpus manifest (against build_ok.json, and a literal pin if set) and the
    quick-val subset file. -> what was checked, for state.json."""
    from training.v6.data.strata import DEFINITION_HASH
    pin, got = p.pins, {"strata_definition_hash": DEFINITION_HASH}
    if pin.strata_definition_hash and pin.strata_definition_hash != DEFINITION_HASH:
        raise SystemExit(f"strata definition {DEFINITION_HASH[:12]} is not the pinned "
                         f"{pin.strata_definition_hash[:12]} (prod.pins.strata_definition_hash)")
    if pin.build_ok or pin.corpus_manifest_sha256:
        man = sha256_file(resolve(cfg.data.manifest))
        got["corpus_manifest_sha256"] = man
        if pin.corpus_manifest_sha256 and man != pin.corpus_manifest_sha256:
            raise SystemExit(f"{cfg.data.manifest} hashes to {man[:12]}, not the pinned {pin.corpus_manifest_sha256[:12]}")
        if pin.build_ok:
            path = resolve(pin.build_ok)
            if not path.exists():
                raise SystemExit(f"{pin.build_ok} is missing: build corpus v3 (setup.sh --build-corpus) "
                                 f"or pull it (setup.sh --pull-corpus) first")
            ok = json.loads(path.read_text())
            if ok.get("smoke_limit_lines"):
                raise SystemExit(f"{pin.build_ok} is from a --smoke build ({ok['smoke_limit_lines']:,} dump "
                                 f"lines), not a production corpus")
            if ok["manifest_sha256"] != man:
                raise SystemExit(f"{cfg.data.manifest} hashes to {man[:12]}, not build_ok.json's "
                                 f"{ok['manifest_sha256'][:12]}")
            if ok["strata_definition_hash"] != DEFINITION_HASH:
                raise SystemExit(f"build_ok.json's strata definition {ok['strata_definition_hash'][:12]} "
                                 f"!= {DEFINITION_HASH[:12]}")
            got["build_ok"] = {k: ok.get(k) for k in ("created_utc", "git_sha", "manifest_sha256")}
    if cfg.eval.quick_indices:
        sha = sha256_file(resolve(cfg.eval.quick_indices))
        if sha != cfg.eval.quick_indices_sha256:
            raise SystemExit(f"{cfg.eval.quick_indices} hashes to {sha[:12]}, not the pinned "
                             f"{cfg.eval.quick_indices_sha256[:12]}")
        got["quick_indices_sha256"] = sha
    print(f"[prod] pins verified: { {k: (v[:12] if isinstance(v, str) else v) for k, v in got.items()} }",
          flush=True)
    return got


def branch_of(cfg, bp: int) -> tuple[str, int]:
    """(branch name as branch_decay.py names it, its last sample)."""
    total = math.ceil(bp / (1.0 - cfg.schedule.decay_frac))
    eff = cfg.optim.effective_batch
    return f"s{bp}_d{total - bp}", -(-total // eff) * eff


def local_path(run_dir: Path, key: str) -> Path:
    """Store key (relative to the run) -> the trainer's local path."""
    k = key.split("/")
    if k[:2] == ["ckpt", "rolling"]:
        return run_dir / "ckpt" / k[2]
    if k[:2] == ["ckpt", "stable"]:
        return run_dir / "stable" / k[2]
    if k[0] == "branches" and len(k) == 4 and k[2] == "rolling":
        return run_dir / "branches" / k[1] / "ckpt" / k[3]
    if k[0] == "branches" and len(k) == 3:
        return run_dir / "branches" / k[1] / k[2]
    raise ValueError(f"no local path for store key {key!r}")


# ------------------------------------------------------------ stop rule

def better(m: dict, key: str) -> float:
    """The better of raw and EMA on frozen90."""
    return min(w[key] for w in m.values())


def decide(i: int, n: int, cur: dict, prev: dict | None, min_gain: float, anomalies: list) -> dict:
    """Pre-registered (brief §4). i: index of the branch just finished; cur/prev:
    {"raw": frozen90 metrics, "ema": ...}. -> {"go": continue?, "reason": ...}."""
    d = {"branch_index": i, "policy_kl": better(cur, "policy_kl"), "total": better(cur, "total")}
    if i == 0:
        bad = [a for a in anomalies if a["kind"] in STOP_ANOMALIES]
        d.update(go=not bad, reason=f"anomaly: {bad[0]['detail']}" if bad else "first branch: always continue")
    else:
        kl0, t0 = better(prev, "policy_kl"), better(prev, "total")
        gain = (kl0 - d["policy_kl"]) / kl0
        kl_ok, tot_ok = gain >= min_gain, d["total"] < t0
        why = [f"insufficient gain ({gain:+.3%} < {min_gain:.2%})"] * (not kl_ok) + ["worse total"] * (not tot_ok)
        d.update(kl_gain=gain, kl_ok=kl_ok, total_ok=tot_ok, prev_total=t0, go=kl_ok and tot_ok,
                 reason="; ".join(why) or f"KL gain {gain:+.3%} and total improved")
    if d["go"] and i == n - 1:
        d.update(go=False, reason=d["reason"] + "; that was the last branch (the final decay)")
    return d


def pick_ship(branches: list) -> dict | None:
    c = [(b["frozen90"][w]["total"], b["name"], w) for b in branches for w in sorted(b["frozen90"])]
    if not c:
        return None
    t, name, w = min(c)
    return {"branch": name, "weights": w, "frozen90_total": t}


def gpu_query() -> dict | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode or not r.stdout.strip():
        return None
    u, used, total = (float(x) for x in r.stdout.splitlines()[0].split(","))
    return {"util_pct": u, "mem_used_mib": used, "mem_total_mib": total}


def _metrics_only(ev: dict) -> dict:
    return {k: v for k, v in ev.items()
            if k not in ("event", "utc", "segment", "samples", "weights", "set", "reason")}


# --------------------------------------------------------------- driver

class Driver:
    def __init__(self, cfg, p: Settings, store, state: dict):
        self.cfg, self.p, self.store, self.state = cfg, p, store, state
        self.bps = list(p.branch_points)
        self.run_dir = resolve(cfg.run.out_root) / cfg.run.name
        self.stage = self.run_dir.parent / f".{cfg.run.name}_upload"     # hardlinks survive rotation
        self.key = f"{p.store_prefix}{cfg.run.name}/"
        self.lock = threading.Lock()
        self.q: queue.Queue = queue.Queue()
        self.ckpts_queued, self.lagging = 0, False
        self.t_session, self.elapsed0 = time.monotonic(), float(state["elapsed_s"])
        self.offsets: dict[Path, int] = {}
        self.seen: dict[Path, int] = {}                 # file -> mtime_ns already handled
        self.evals: dict[str, dict] = {}
        self.metrics: list[str] = []
        self.last_step: dict = {}
        self.control_seen = self.reference_seen = None
        self.action, self.pause_req, self.stop_after = "continue", bool(state.get("paused")), False
        self.proc, self.killed, self.seg, self.seg_end = None, None, None, None
        self.next = {"status": 0.0, "metrics": 0.0, "control": 0.0}
        threading.Thread(target=self._uploader, daemon=True).start()

    # ---- state and uploads ---------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            self.state["elapsed_s"] = self.elapsed0 + time.monotonic() - self.t_session
            self.state["updated_utc"] = utc()
            return json.loads(json.dumps(self.state))

    def set(self, **kw) -> None:
        with self.lock:
            self.state.update(kw)
        self.q.put(("state",))

    def put(self, rel: str, obj) -> None:
        self.q.put(("json", rel, json_safe(json.loads(json.dumps(obj)))))

    def anomaly(self, kind: str, detail: str) -> None:
        print(f"[prod] ANOMALY {kind}: {detail}", flush=True)
        with self.lock:
            self.state["anomalies"].append({"kind": kind, "detail": detail, "utc": utc(),
                                            "segment": self.seg})
        self.q.put(("state",))

    def drain(self) -> None:
        self.q.join()

    def _uploader(self) -> None:
        while True:
            item = self.q.get()
            delay = 5
            while True:
                try:
                    self._upload(item)
                    if item[0] == "ckpt":   # counted once, after the whole item (retries re-run it)
                        with self.lock:
                            self.ckpts_queued -= 1
                            self.lagging = self.lagging and self.ckpts_queued > self.p.max_upload_lag
                    break
                except STORE_ERRORS as e:   # a store outage must not end the run: retried, loudly
                    print(f"[prod] upload {item[0]} {item[1] if len(item) > 1 else ''} failed "
                          f"({type(e).__name__}: {e}); retry in {delay} s", flush=True)
                    time.sleep(delay)
                    delay = min(2 * delay, 300)
                except Exception:           # a bug: this thread must survive it, and say so
                    traceback.print_exc()
                    self.anomaly("upload_error", f"dropped upload item {item[:2]!r}: {traceback.format_exc(limit=1)}")
                    break
            self.q.task_done()

    def _upload(self, item) -> None:
        kind = item[0]
        if kind == "state":
            put_json(self.store, self.key + "state.json", self.snapshot())
        elif kind == "json":
            put_json(self.store, self.key + item[1], item[2])
        elif kind == "bytes":
            self.store.put_bytes(item[2], self.key + item[1])
        elif kind == "file":
            self.store.put_file(item[2], self.key + item[1])
        elif kind == "ckpt":                # file, then its .sha256, then state.json
            _, staged, rel, meta = item
            sha, size = sha256_file(staged), staged.stat().st_size
            self.store.put_file(staged, self.key + rel)
            self.store.put_bytes(f"{sha}  {size}  {rel.rsplit('/', 1)[-1]}\n".encode(), self.key + rel + ".sha256")
            entry = {"key": rel, "sha256": sha, "samples": meta["samples"], "segment": meta["segment"]}
            with self.lock:
                if meta["slot"] == "main":
                    if (self.state["main"] or {}).get("samples", -1) <= meta["samples"]:
                        self.state["main"] = entry
                elif meta["slot"] == "branch":
                    if meta["segment"] == self.state["branch"]:      # ignore a finished branch's stragglers
                        self.state["branch_ckpt"] = entry
                else:
                    self.state["finals"][meta["segment"]] = entry
            put_json(self.store, self.key + "state.json", self.snapshot())
            if meta["rolling"]:
                self._prune(meta["rolling"])
            staged.unlink()
        else:
            raise ValueError(f"unknown upload item {kind!r}")

    def _prune(self, prefix: str) -> None:
        keys = self.store.list(self.key + prefix)
        pts = sorted((k for k in keys if k.endswith(".pt")), key=lambda k: int(k.rsplit("/s", 1)[1][:-3]))
        for k in pts[:-self.p.rolling_keep]:
            self.store.delete(k + ".sha256")          # sidecar first: a half-pruned pair is never trusted
            self.store.delete(k)

    # ---- fetch (resume) ------------------------------------------------
    def fetch(self, rel: str, want_sha: str | None = None) -> Path:
        """A checkpoint is trusted only if its sidecar exists and matches (and matches state.json)."""
        self.drain()            # our own queued uploads land first (the branch's stable checkpoint)
        path = local_path(self.run_dir, rel)
        side = self.store.get_bytes(self.key + rel + ".sha256")
        if side is None:
            raise SystemExit(f"refusing {rel}: its .sha256 sidecar is missing")
        sha = side.decode().split()[0]
        if want_sha is not None and sha != want_sha:
            raise SystemExit(f"refusing {rel}: sidecar {sha[:12]} != state.json {want_sha[:12]}")
        if not (path.is_file() and sha256_file(path) == sha):
            tmp = path.with_name(path.name + ".download")
            self.store.get_file(self.key + rel, tmp)
            got = sha256_file(tmp)
            if got != sha:
                tmp.unlink()
                raise SystemExit(f"refusing {rel}: downloaded sha256 {got[:12]} != sidecar {sha[:12]}")
            os.replace(tmp, path)
        self.seen[path] = path.stat().st_mtime_ns
        return path

    # ---- watching a trainer ----------------------------------------------
    def scan(self) -> None:
        if self.seg is None:
            return
        if self.seg == "main":
            spots = [(self.run_dir / "ckpt", "ckpt/rolling/", "main", True),
                     (self.run_dir / "stable", "ckpt/stable/", "main", False)]
            for f in (*self.run_dir.glob("provenance*.json"), *self.run_dir.glob("code*.patch")):
                if f not in self.seen:
                    self.seen[f] = 0
                    self.q.put(("file", f.name, f))
        else:
            b = self.run_dir / "branches" / self.seg
            spots = [(b / "ckpt", f"branches/{self.seg}/rolling/", "branch", True),
                     (b, f"branches/{self.seg}/", "final", False)]
        for d, prefix, slot, rolling in spots:
            for f in ([d / "final.pt"] if slot == "final" else d.glob("s*.pt")):
                try:
                    mt = f.stat().st_mtime_ns
                except FileNotFoundError:
                    continue
                if self.seen.get(f) == mt:
                    continue
                self.seen[f] = mt
                # every log line written before this checkpoint reaches the store ahead of it,
                # so a resume from it never leaves a gap in the metrics (and a pause asked for
                # by an event just before it, the sanity check, lands on it)
                for ev in self.tail(self.log_path()):
                    self.on_event(ev)
                self.flush_metrics()
                self.stage.mkdir(parents=True, exist_ok=True)
                staged = self.stage / f"{self.seg}_{f.name}.{mt}"
                try:
                    os.link(f, staged)
                except FileNotFoundError:           # rotated away before we got to it
                    continue
                samples = self.seg_end if slot == "final" else int(f.stem[1:])
                with self.lock:
                    self.ckpts_queued += 1
                    lag, was = self.ckpts_queued, self.lagging
                    self.lagging = lag > self.p.max_upload_lag
                self.q.put(("ckpt", staged, prefix + f.name, {"slot": slot, "samples": samples,
                            "segment": self.seg, "rolling": prefix if rolling else None}))
                if lag > self.p.max_upload_lag and not was:
                    self.anomaly("upload_lag", f"{lag} checkpoints queued for upload")
                if rolling and self.pause_req and self.proc is not None:
                    self.halt("pause")

    def tail(self, path: Path) -> list:
        if not path.exists():
            return []
        off = self.offsets.get(path, 0)
        with open(path, "rb") as f:
            f.seek(off)
            data = f.read()
        end = data.rfind(b"\n") + 1
        self.offsets[path] = off + end
        return [json.loads(ln) for ln in data[:end].splitlines() if ln.strip()]

    def log_path(self) -> Path:
        base = self.run_dir if self.seg == "main" else self.run_dir / "branches" / self.seg
        return base / "logs" / "train.jsonl"

    def on_event(self, ev: dict) -> None:
        ev["segment"] = self.seg
        self.metrics.append(json.dumps(ev))
        e, n = ev["event"], ev.get("samples")
        if e == "step":
            self.last_step = ev
        elif e == "quick_eval":
            self.put(f"evals/{self.seg}_quick_s{n}.json", ev)
            if self.seg == "main":
                with self.lock:
                    self.state["quick_kl"][str(n)] = ev["policy_kl"]
                self.check_references()
        elif e in ("full_eval", "memorization_gap"):
            kind = "full" if e == "full_eval" else "memgap"
            tag = f"{self.seg}_{kind}_s{n}"
            doc = self.evals.setdefault(tag, {"segment": self.seg, "samples": n, "reason": ev["reason"], "sets": {}})
            doc["sets"].setdefault(ev["set"], {})[ev["weights"]] = _metrics_only(ev)
            self.put(f"evals/{tag}.json", doc)
        elif e == "anomaly":
            self.anomaly("nonfinite", f"trainer: {ev}")

    def references(self) -> list:
        s = self.p.sanity
        primary = ([{"source": "primary", "ref_kl": s.ref_kl, "samples": s.samples, "tol": s.tol,
                     "ref_run": s.ref_run}] if s.samples and s.ref_kl is not None else [])
        return primary + self.state["references"]

    def check_references(self) -> None:
        """Each reference once per run, as soon as production has a main-line quick-val at
        its sample count: the primary (config) and any secondary from reference.json."""
        for ref in self.references():
            kl = self.state["quick_kl"].get(str(ref["samples"]))
            if kl is not None and ref["source"] not in self.state["sanity"]:
                self.sanity(ref, kl)

    def sanity(self, ref: dict, kl: float) -> None:
        dev = (kl - ref["ref_kl"]) / ref["ref_kl"]
        doc = {**ref, "quick_kl": kl, "rel_dev": dev, "pass": abs(dev) <= ref["tol"], "utc": utc()}
        self.put(f"evals/sanity_{ref['source']}_s{ref['samples']}.json", doc)
        with self.lock:
            self.state["sanity"][ref["source"]] = doc
        self.q.put(("state",))
        print(f"[prod] sanity ({ref['source']}) @ {ref['samples']:,}: quick-val KL {kl:.5f} vs "
              f"{ref['ref_kl']:.5f} ({dev:+.2%}, tol ±{ref['tol']:.0%}): {'pass' if doc['pass'] else 'FAIL'}",
              flush=True)
        if not doc["pass"]:
            reason = (f"sanity check failed ({ref['source']} reference): quick-val KL {kl:.5f} at "
                      f"{ref['samples']:,} is {dev:+.2%} from {ref['ref_kl']:.5f} (tol ±{ref['tol']:.0%})")
            self.anomaly("sanity", reason)
            self.set(pause_reason=reason)
            ctl = {"action": "pause", "by": "prod.py", "reason": reason, "utc": utc()}
            put_json(self.store, self.key + "control.json", ctl)     # synchronous: a poll must not race it
            self.control_seen = ctl
            self.request("pause")
            self.status()

    def poll_reference(self) -> None:
        """runs/<run>/reference.json, written after launch (C0's 40M quick-val, say):
        {"source": "C0", "ref_kl": float, "samples": int[, "tol": float, "ref_run": str]}."""
        try:
            doc = get_json(self.store, self.key + "reference.json")
        except STORE_ERRORS as e:
            print(f"[prod] reference.json poll failed ({type(e).__name__}: {e})", flush=True)
            return
        if doc is None or doc == self.reference_seen:
            return
        self.reference_seen = doc
        q = self.cfg.eval.quick_every_samples
        try:
            ref = {"source": str(doc["source"]), "ref_kl": float(doc["ref_kl"]), "samples": int(doc["samples"]),
                   "tol": float(doc.get("tol", self.p.sanity.tol)), "ref_run": str(doc.get("ref_run", ""))}
            ok = (ref["ref_kl"] > 0 and ref["samples"] > 0 and q > 0 and ref["samples"] % q == 0
                  and ref["source"] not in ("", "primary"))
        except (KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            self.anomaly("reference", f"ignored reference.json {doc!r}: needs source, ref_kl > 0 and "
                                      f"samples on the {q:,} quick-val grid")
            return
        print(f"[prod] secondary reference: {ref}", flush=True)
        with self.lock:
            self.state["references"] = [r for r in self.state["references"] if r["source"] != ref["source"]] + [ref]
        self.q.put(("state",))
        self.check_references()

    def request(self, action: str) -> None:
        print(f"[prod] control: {action}", flush=True)
        self.action = action
        if action == "continue":
            self.pause_req = self.stop_after = False
            if self.state.get("pause_reason"):
                self.set(pause_reason=None)
        elif action == "pause":
            self.pause_req = True
        elif action == "stop_after_branch":
            self.stop_after = True
        elif action == "stop_now":
            self.halt("stop_now")

    def poll_control(self) -> None:
        try:
            doc = get_json(self.store, self.key + "control.json")
        except STORE_ERRORS as e:
            print(f"[prod] control.json poll failed ({type(e).__name__}: {e})", flush=True)
            return
        if doc is None or doc == self.control_seen:
            return
        self.control_seen = doc
        if not isinstance(doc, dict) or doc.get("action") not in ACTIONS:
            self.anomaly("control", f"ignored control.json {doc!r}; action must be one of {ACTIONS}")
            return
        if "time_cap_h" in doc:
            self.set(time_cap_h=float(doc["time_cap_h"]))
        self.request(doc["action"])

    def poll(self) -> None:
        self.poll_control()
        self.poll_reference()

    def halt(self, why: str) -> None:
        self.killed = why
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()

    def status(self) -> None:
        s = self.last_step
        t = s.get("time_s") or {}
        iv = t.get("interval") or 0.0
        rate, done = s.get("samples_per_s"), s.get("samples")
        el = self.snapshot()["elapsed_s"] / 3600
        k = self.state["k"]
        seg_eta = nxt_eta = None
        if rate and done is not None and self.seg is not None:
            seg_eta = (self.seg_end - done) / rate / 3600
            name, end = branch_of(self.cfg, self.bps[min(k, len(self.bps) - 1)])
            left = (end - done) if self.seg != "main" else (self.bps[k] - done) + (end - self.bps[k])
            nxt_eta = left / rate / 3600
        self.put("status.json", {
            "run": self.cfg.run.name, "phase": self.state["phase"], "paused": self.pause_req,
            "pause_reason": self.state.get("pause_reason"),
            "segment": self.seg, "segment_end": self.seg_end, "next_branch_point": self.bps[min(k, len(self.bps) - 1)],
            "samples": done, "step": s.get("step"), "samples_per_s": rate, "lr": s.get("lr"),
            "loss": s.get("loss"), "grad_norm": s.get("grad_norm"), "loader_wait_frac": s.get("loader_wait_frac"),
            "step_time_s": t, "step_time_frac": {k2: v / iv for k2, v in t.items() if k2 != "interval"} if iv else None,
            "peak_vram_mib": s.get("peak_vram_mib"), "gpu": gpu_query(), "passes": s.get("passes"),
            "eta_segment_end_h": seg_eta, "eta_next_branch_end_h": nxt_eta, "elapsed_h": el,
            "time_cap_h": self.state["time_cap_h"], "cap_remaining_h": self.state["time_cap_h"] - el,
            "upload_queue": self.q.qsize(), "anomalies": len(self.state["anomalies"]),
            "control": self.action, "last_step_utc": s.get("utc"), "updated_utc": utc()})

    def flush_metrics(self) -> None:
        if not self.metrics:
            return
        with self.lock:
            seq = self.state["metrics_seq"]
            self.state["metrics_seq"] = seq + 1
        self.q.put(("bytes", f"metrics/train-{seq:06d}.jsonl", ("\n".join(self.metrics) + "\n").encode()))
        self.metrics = []

    def tick(self, force: bool = False) -> None:
        if self.seg is not None:            # log before checkpoints: a pause requested by an event
            for ev in self.tail(self.log_path()):   # (the sanity check) lands on the same checkpoint
                self.on_event(ev)
        self.scan()
        now = time.monotonic()
        for what, every, fn in (("status", self.p.status_every_s, self.status),
                                ("metrics", self.p.metrics_every_s, self.flush_metrics),
                                ("control", self.p.control_every_s, self.poll)):
            if force or now >= self.next[what]:
                self.next[what] = now + every
                fn()

    def supervise(self, args: list, seg: str, end: int) -> str:
        self.seg, self.seg_end, self.killed = seg, end, None
        print(f"[prod] {seg}: {' '.join(args)}", flush=True)
        self.proc = subprocess.Popen(PY + args, cwd=REPO)
        try:
            while True:
                rc = self.proc.poll()
                self.tick()
                if rc is not None:
                    break
                time.sleep(1.0)
            self.tick(force=True)
        finally:
            if self.proc.poll() is None:      # this driver is dying: never orphan a trainer
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if self.killed:
            return self.killed
        if rc != 0:
            self.anomaly("trainer_exit", f"{seg} exited {rc}")
            self.drain()
            raise SystemExit(f"trainer exited {rc}; the store has the state; resume with --resume {self.cfg.run.name}")
        return "done"

    # ---- segments ----------------------------------------------------------
    def seen_args(self) -> list:
        s = self.p.seen_eval
        return ["--seen-eval", f"{s.group}:{s.n}"] if s.n else []

    def run_main(self, k: int) -> str:
        bp = self.bps[k]
        args = ["-m", "training.v6.train", "--config", str(resolve(self.state["config"])),
                "--set", *self.state["sets"], "--stop-at-samples", str(bp), *self.seen_args()]
        if self.state["main"]:
            args += ["--resume", str(self.fetch(self.state["main"]["key"], self.state["main"]["sha256"]))]
        r = self.supervise(args, "main", bp)
        if r == "done":
            if not (self.run_dir / "stable" / f"s{bp}.pt").exists():
                raise SystemExit(f"main segment ended without stable/s{bp}.pt")
            start = (self.state["main_start"] or 0)
            self.set(trained_samples=self.state["trained_samples"] + bp - start)
        return r

    def run_branch(self, k: int) -> str:
        bp = self.bps[k]
        name, end = branch_of(self.cfg, bp)
        stable = self.fetch(f"ckpt/stable/s{bp}.pt")
        bdir = self.run_dir / "branches" / name
        shutil.rmtree(bdir, ignore_errors=True)         # only store-verified state is resumed
        args = ["-m", "training.v6.tools.branch_decay", str(stable), *self.seen_args()]
        if self.state["branch_ckpt"]:
            (bdir / "ckpt").mkdir(parents=True)
            self.fetch(self.state["branch_ckpt"]["key"], self.state["branch_ckpt"]["sha256"])
            args.append("--resume")
        return self.supervise(args, name, end)

    def close_branch(self, k: int) -> dict:
        self.drain()                        # final.pt's key and sha256 into state before the summary
        bp = self.bps[k]
        name, end = branch_of(self.cfg, bp)
        doc = self.evals.get(f"{name}_full_s{end}")
        if doc is None or set(doc["sets"].get("frozen90", {})) != {"raw", "ema"}:
            raise SystemExit(f"branch {name}: no raw+EMA frozen90 full eval at {end:,} in its log")
        rec = {"name": name, "from": bp, "end": end, "frozen90": doc["sets"]["frozen90"],
               "other_sets": {s: m for s, m in doc["sets"].items() if s != "frozen90"},
               "memgap": (self.evals.get(f"{name}_memgap_s{end}") or {}).get("sets")}
        prev = self.state["branches"][-1]["frozen90"] if self.state["branches"] else None
        d = decide(k, len(self.bps), rec["frozen90"], prev, self.p.min_kl_gain, self.state["anomalies"])
        d.update(branch=name, utc=utc())
        print(f"[prod] decision after {name}: {'CONTINUE' if d['go'] else 'STOP'} ({d['reason']})", flush=True)
        with self.lock:
            self.state["branches"].append(rec)
            self.state["decisions"].append(d)
            self.state["trained_samples"] += end - bp
        self.put(f"evals/decision_{name}.json", d)
        self.put("summary.json", self.summary())
        self.q.put(("state",))
        return d

    def summary(self) -> dict:
        st = self.snapshot()
        return {"run": self.cfg.run.name, "updated_utc": st["updated_utc"], "gpu_hours": st["elapsed_s"] / 3600,
                "trained_samples": st["trained_samples"],
                "branches": [{**b, "final": st["finals"].get(b["name"])} for b in st["branches"]],
                "decisions": st["decisions"], "shipped": st["shipped"], "stopped": st["stopped"],
                "anomalies": st["anomalies"]}

    def within_cap(self, k: int) -> bool:
        el = self.snapshot()["elapsed_s"]
        name, end = branch_of(self.cfg, self.bps[k])
        need = end - self.bps[k - 1]                     # the stable segment plus its decay
        rate = self.state["trained_samples"] / el if el > 0 else 0.0
        cap = self.state["time_cap_h"] * 3600
        ok = rate > 0 and el + need / rate <= cap
        doc = {"before_branch": name, "elapsed_h": el / 3600, "rate_samples_per_s": rate,
               "need_samples": need, "projected_h": (el + need / rate) / 3600 if rate else None,
               "time_cap_h": self.state["time_cap_h"], "ok": ok, "utc": utc()}
        self.put(f"evals/timecap_{name}.json", doc)
        print(f"[prod] time cap before {name}: projected {doc['projected_h']} h of {doc['time_cap_h']} h -> "
              f"{'start' if ok else 'STOP'}", flush=True)
        return ok

    def wait_paused(self) -> str:
        self.seg = None
        self.drain()            # the halting checkpoint, its evals and state land before "paused" does
        self.set(paused=True)
        self.drain()
        print("[prod] paused; write control.json {\"action\": \"continue\"} to resume", flush=True)
        while self.pause_req and self.action != "stop_now":
            self.tick()
            time.sleep(1.0)
        self.set(paused=False)
        return "stop_now" if self.action == "stop_now" else "continue"

    def finish(self, reason: str) -> None:
        self.seg = None
        self.set(stopped=reason)
        self.drain()
        ship = pick_ship(self.state["branches"])
        if ship and not self.state["shipped"]:
            fin = self.state["finals"].get(ship["branch"])
            if fin is None:
                raise SystemExit(f"shipped branch {ship['branch']}: its final.pt never reached the store")
            final = self.fetch(fin["key"], fin["sha256"])
            exp = self.run_dir / "export"
            before = set(exp.glob("*.pt"))
            r = subprocess.run(PY + ["-m", "training.v6.tools.export", str(final), "--weights", ship["weights"]], cwd=REPO)
            new = sorted(set(exp.glob("*.pt")) - before)
            if r.returncode or len(new) != 1:
                raise SystemExit(f"export failed (exit {r.returncode}, new files {new})")
            sha = sha256_file(new[0])
            self.q.put(("file", f"export/{new[0].name}", new[0]))
            self.q.put(("bytes", f"export/{new[0].name}.sha256", f"{sha}  {new[0].stat().st_size}  {new[0].name}\n".encode()))
            ship.update(export_key=f"export/{new[0].name}", export_sha256=sha, contract=self.cfg.model.contract)
            self.set(shipped=ship)
        self.set(phase="done")
        self.put("summary.json", self.summary())
        self.status()
        self.drain()
        print(f"[prod] done: {reason}; shipped {self.state['shipped']}", flush=True)

    def loop(self) -> None:
        self.tick(force=True)
        while True:
            if self.state["phase"] == "done":
                return
            if self.state["stopped"]:
                return self.finish(self.state["stopped"])
            if self.pause_req and self.wait_paused() == "stop_now":
                return self.finish("operator: stop_now")
            k = self.state["k"]
            if self.state["phase"] == "stable":
                if not self.state["segment_started"]:
                    if k > 0 and not self.within_cap(k):
                        return self.finish("time cap")
                    self.set(segment_started=True, main_start=self.bps[k - 1] if k else 0)
                r = self.run_main(k)
                if r == "done":
                    self.set(phase="branch", branch=branch_of(self.cfg, self.bps[k])[0], branch_ckpt=None,
                             segment_started=False)
            else:
                r = self.run_branch(k)
                if r == "done":
                    d = self.close_branch(k)
                    self.set(branch=None, branch_ckpt=None)
                    if not d["go"]:
                        return self.finish(d["reason"])
                    if self.stop_after:
                        return self.finish("operator: stop_after_branch")
                    self.set(phase="stable", k=k + 1)
            if r == "stop_now":
                return self.finish("operator: stop_now")


def new_state(cfg, config: str, sets: list, p: Settings) -> dict:
    return {"run": cfg.run.name, "created_utc": utc(), "config": config, "sets": list(sets),
            "phase": "stable", "k": 0, "segment_started": False, "main_start": 0, "main": None,
            "branch": None, "branch_ckpt": None, "finals": {}, "paused": False,
            "elapsed_s": 0.0, "trained_samples": 0, "metrics_seq": 0, "time_cap_h": p.time_cap_h,
            "sanity": {}, "references": [], "quick_kl": {}, "pause_reason": None,
            "anomalies": [], "branches": [], "decisions": [],
            "shipped": None, "stopped": None, "updated_utc": utc()}


def start(config: str, sets: list) -> None:
    cfg, p = load(config, sets)
    if p.sanity.samples and p.sanity.ref_kl is None:      # C0 is optional; the primary is not
        raise SystemExit(f"the primary sanity reference is missing: prod.sanity.ref_kl is null (A3's "
                         f"quick-val KL at {p.sanity.samples:,}; prod.yaml pins it)")
    pins = check_pins(cfg, p)
    check_dirty(git_state(), cfg.system.allow_dirty)
    store = open_store()
    key = f"{p.store_prefix}{cfg.run.name}/"
    if store.get_bytes(key + "state.json") is not None:
        raise SystemExit(f"{store.where}/{key}state.json exists; continue it with --resume {cfg.run.name}")
    d = Driver(cfg, p, store, {**new_state(cfg, config, sets, p), "pins": pins})
    d.q.put(("bytes", "config.resolved.yaml", yaml.safe_dump({**to_plain(cfg), "prod": asdict(p)}, sort_keys=True).encode()))
    d.q.put(("state",))
    _run(d)


def resume(run: str, config: str | None, sets: list) -> None:
    _, p0 = load(config or DEFAULT_CONFIG, sets)
    store = open_store()
    key = f"{p0.store_prefix}{run}/"
    state = get_json(store, key + "state.json")
    if state is None:
        raise SystemExit(f"{store.where}/{key}state.json does not exist")
    state["sets"] = state["sets"] + [s for s in sets if s not in state["sets"]]
    cfg, p = load(config or state["config"], state["sets"])
    if cfg.run.name != run:
        raise SystemExit(f"config resolves run.name={cfg.run.name!r}, not {run!r}")
    state["pins_resume"] = check_pins(cfg, p)
    for k, v in {"sanity": {}, "references": [], "quick_kl": {}, "pause_reason": None}.items():
        if state.get(k) is None:                    # a state.json from before these keys existed
            state[k] = v
    d = Driver(cfg, p, store, state)
    shutil.rmtree(d.stage, ignore_errors=True)
    for f in d.run_dir.rglob("*"):                  # local leftovers: never uploaded, never trusted
        if f.is_file():
            d.seen[f] = f.stat().st_mtime_ns
            if f.name == "train.jsonl":
                d.offsets[f] = f.stat().st_size
    print(f"[prod] resuming {run} from {store.where}/{key}: phase {state['phase']}, k {state['k']}, "
          f"main {state['main'] and state['main']['key']}, branch {state['branch_ckpt'] and state['branch_ckpt']['key']}",
          flush=True)
    _run(d)


def _run(d: Driver) -> None:
    try:
        d.loop()
    finally:
        d.halt("driver exit")
        d.flush_metrics()
        d.q.put(("state",))
        d.drain()


# ---------------------------------------------------------------- smoke

def smoke(config: str, sets: list, mbs: list, steps: int) -> dict:
    cfg, p = load(config, sets)
    check_pins(cfg, p)                  # a drifted corpus or definition fails here, before GPU time
    eff = cfg.optim.effective_batch
    root = REPO / "models" / "v6" / "_smoke"
    out = {"config": config, "steps": steps, "utc": utc(), "splits": {}}
    for mb in mbs:
        m, a = (int(x) for x in mb.split("x"))
        if m * a != eff:
            raise SystemExit(f"--mb {mb}: {m} x {a} != effective batch {eff}")
        name = f"smoke_{mb}"
        shutil.rmtree(root / name, ignore_errors=True)
        args = ["-m", "training.v6.train", "--config", str(resolve(config)), "--set", *sets,
                f"run.name={name}", f"run.out_root={root.as_posix()}", f"optim.micro_batch={m}",
                f"optim.accum={a}", "--stop-at-samples", str(steps * eff)]
        print(f"[smoke] {mb}: {steps} steps", flush=True)
        r = subprocess.run(PY + args, cwd=REPO)
        if r.returncode:
            raise SystemExit(f"smoke {mb}: trainer exited {r.returncode}")
        ev = [json.loads(ln) for ln in (root / name / "logs/train.jsonl").read_text().splitlines()]
        st = [e for e in ev if e["event"] == "step" and e["step"] > min(100, steps // 3)]   # past compile
        if not st:
            raise SystemExit(f"smoke {mb}: no step intervals after warm-up; raise --steps")
        rate = median(e["samples_per_s"] for e in st)
        tsum = {k: sum(e["time_s"][k] for e in st) for k in st[0]["time_s"]}
        eta, cum = {}, 0
        for i, bp in enumerate(p.branch_points):
            cum = bp + sum(branch_of(cfg, b)[1] - b for b in p.branch_points[:i + 1])
            eta[str(bp)] = cum / rate / 3600
        res = {"samples_per_s": rate, "loader_wait_frac": sum(e["time_s"]["data_wait"] for e in st) / tsum["interval"],
               "step_time_frac": {k: v / tsum["interval"] for k, v in tsum.items() if k != "interval"},
               "peak_vram_mib": max((e["peak_vram_mib"] or 0) for e in ev if e["event"] == "step") or None,
               "eta_h_cumulative": eta}
        res["go"] = {"loader_wait_lt_5pct": res["loader_wait_frac"] < 0.05,
                     "rate_ge_1p5x_local_a3": rate >= 1.5 * A3_LOCAL_SAMPLES_PER_S}
        out["splits"][mb] = res
    (root / "smoke.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"\n[smoke] {cfg.run.name}: {steps} steps per split (intervals after warm-up); go needs loader "
          f"wait < 5% and >= {1.5 * A3_LOCAL_SAMPLES_PER_S:,.0f} samples/s (1.5 x local A3)")
    for mb, r in out["splits"].items():
        f = r["step_time_frac"]
        print(f"  {mb:>7}: {r['samples_per_s']:>9,.0f} samples/s | peak VRAM {r['peak_vram_mib'] or 0:,.0f} MiB | "
              f"loader wait {r['loader_wait_frac']:.1%} | h2d {f['h2d']:.1%} fwd+bwd {f['fwd_bwd']:.1%} "
              f"optim {f['optim']:.1%} eval {f['eval']:.1%} | go {all(r['go'].values())}")
        print("           ETA (h, cumulative incl. decays) to branch " +
              ", ".join(f"{int(b) / 1e9:.2g}B: {h:.1f}" for b, h in r["eta_h_cumulative"].items()))
    print(f"  written: {root / 'smoke.json'}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help=f"default {DEFAULT_CONFIG} (resume: the run's own)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--resume", metavar="RUN")
    g.add_argument("--smoke", action="store_true")
    ap.add_argument("--mb", nargs="+", default=["512x2", "1024x1"], help="--smoke: micro_batch x accum splits")
    ap.add_argument("--steps", type=int, default=300, help="--smoke: optimizer steps per split")
    a = ap.parse_args(argv)
    if a.smoke:
        smoke(a.config or DEFAULT_CONFIG, a.set, a.mb, a.steps)
    elif a.resume:
        resume(a.resume, a.config, a.set)
    else:
        start(a.config or DEFAULT_CONFIG, a.set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
