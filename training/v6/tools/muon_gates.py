"""Batched-Muon gates on the GPU (VM harness follow-ups §1). Local 5070; never while C0 runs.

    python -m training.v6.tools.muon_gates parity     # gate 1 on CUDA: compiled batched vs reference
    python -m training.v6.tools.muon_gates bench      # optimizer-step breakdown, per component
    python -m training.v6.tools.muon_gates train      # gates 2-4: three 2,000-step A3 runs
    python -m training.v6.tools.muon_gates report     # gates 2-4 from those runs' logs

`train`: A3's resolved config, seed 20260802, 2,048,000 samples (2,000 steps x 1,024):
  ref  --muon-impl reference          bat  batched (compiled)
  res  batched, rolling checkpoints every 512,000, killed after step 1,250, resumed.
Gate 2: mean loss of steps 1,501-2,000, bat vs ref, within 0.3%. Gate 3: res's stream
digest equals bat's at every logged interval. Gate 4: optimizer share of step time
(GPU-stream `optim` / interval), steps > 200, ref vs bat; target <= 10%.
Results: models/v6/muon_gates/gates.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

import torch

import training.v6.train  # noqa: F401  (Inductor descriptive_names=False: MAX_PATH)
from core.guofish_net import build_model
from training.v6.config import load_config
from training.v6.data.formats import REPO
from training.v6.optim import EMA, build_optimizer

OUT = REPO / "models/v6/muon_gates"
A3 = "training/v6/config/configs/a3_resolved.yaml"
STEPS, EFF = 2000, 1024


def save(key: str, value) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "gates.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d[key] = value
    p.write_text(json.dumps(d, indent=1) + "\n")


def parity() -> dict:
    """Gate 1 on CUDA, TF32 off: compiled batched vs reference, fp64 and fp32, plus the
    reference's own fp32 floor (the batched path eager, as a cross-check)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = load_config(REPO / "training/v6/config/configs/prod.yaml")

    import training.v6.optim.muon as muon_mod
    ns = muon_mod.newton_schulz

    def ns_contiguous(g, n, ns_dtype=None):       # the reference, its transposed view made contiguous
        tall = g.size(0) > g.size(1)
        x = ns(g.T.contiguous() if tall else g, n, ns_dtype)
        return x.T if tall else x

    def run(dtype, impl_b, compile_b, reorder_b=False):
        a = build_model(cfg.model, cfg.run.init_seed).to("cuda", dtype)
        b = copy.deepcopy(a)
        oa = build_optimizer(a, cfg.optim, 3.5e-4, muon_impl="reference")
        ob = build_optimizer(b, cfg.optim, 3.5e-4, muon_impl=impl_b, compile=compile_b)
        oa.muon.ns_dtype = ob.muon.ns_dtype = dtype
        gen = torch.Generator(device="cuda").manual_seed(20260802)
        worst = 0.0
        for _ in range(5):
            pa = [p.detach().clone() for p in a.parameters()]
            pb = [p.detach().clone() for p in b.parameters()]
            for x, y in zip(a.parameters(), b.parameters()):
                g = (torch.randn(x.shape, generator=gen, device="cuda") * 1e-3).to(dtype)
                x.grad, y.grad = g.clone(), g.clone()
            oa.step()
            muon_mod.newton_schulz = ns_contiguous if reorder_b else ns
            try:
                ob.step()
            finally:
                muon_mod.newton_schulz = ns
            for p0, q0, x, y in zip(pa, pb, a.parameters(), b.parameters()):
                da, db = x.detach() - p0, y.detach() - q0
                worst = max(worst, float((db - da).norm() / da.norm()))
        return worst

    r = {"float64_compiled": run(torch.float64, "batched", True),
         "fp32_compiled": run(torch.float32, "batched", True),
         "fp32_eager": run(torch.float32, "batched", False),
         "fp32_floor_reference_reordered": run(torch.float32, "reference", False, reorder_b=True)}
    print("gate 1 on CUDA (TF32 off), worst relative update difference over 5 steps:",
          {k: f"{v:.1e}" for k, v in r.items()})
    save("gate1_cuda", r)
    return r


def bench(iters: int = 50) -> dict:
    """Mean ms per call of each optimizer-step component, production model, bf16 NS."""
    cfg = load_config(REPO / "training/v6/config/configs/prod.yaml")
    model = build_model(cfg.model, cfg.run.init_seed).cuda()
    params = list(model.parameters())
    for p in params:
        p.grad = torch.randn_like(p) * 1e-3
    ema = EMA(model, cfg.ema.half_life_samples, EFF)
    opts = {impl: build_optimizer(model, cfg.optim, 3.5e-4, muon_impl=impl, compile=comp)
            for impl, comp in (("reference", False), ("batched", False))}
    opts["batched_compiled"] = build_optimizer(model, cfg.optim, 3.5e-4, muon_impl="batched", compile=True)
    for o in opts.values():
        o.found_inf = torch.zeros((), device="cuda")

    def timed(fn):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / iters

    saved = [p.detach().clone() for p in params]
    r = {"clip_grad_norm": timed(lambda: torch.nn.utils.clip_grad_norm_(params, 1.0)),
         **{f"muon_{k}": timed(o.muon.step) for k, o in opts.items()},
         "adamw_fused": timed(opts["reference"].adamw.step),
         "ema_update": timed(lambda: ema.update(model))}
    for p, s in zip(params, saved):
        p.data.copy_(s)
    print("optimizer components, ms per step (mean of", iters, "):", {k: round(v, 3) for k, v in r.items()})
    save("bench_ms", r)
    return r


def train() -> None:
    base = [sys.executable, "-u", "-m", "training.v6.train", "--config", A3, "--set",
            f"run.out_root={OUT.as_posix()}", "system.allow_dirty=true"]
    stop = ["--stop-at-samples", str(STEPS * EFF)]

    def go(name, *extra, sets=(), check=0):
        log = OUT / f"{name}.log"
        cmd = base + [f"run.name={name}", *sets] + stop + list(extra)
        print(f"[muon_gates] {name}: {' '.join(cmd[3:])}", flush=True)
        with open(log, "a", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != check:
            raise SystemExit(f"{name}: exit {rc} (want {check}); see {log}")

    OUT.mkdir(parents=True, exist_ok=True)
    go("ref", "--muon-impl", "reference")
    go("bat")
    res_sets = ["ckpt.every_samples=512000"]
    go("res", "--crash-after-steps", "1250", sets=res_sets, check=17)
    go("res", "--resume", "latest", sets=res_sets)


def steps(name: str) -> dict:
    out = {}
    for ln in (OUT / name / "logs/train.jsonl").read_text().splitlines():
        e = json.loads(ln)
        if e["event"] == "step":
            out[e["step"]] = e                  # a replayed interval keeps its latest copy
    return out


def report() -> dict:
    ref, bat, res = steps("ref"), steps("bat"), steps("res")
    lm = lambda s: mean(x for e in s.values() if 1500 < e["step"] <= STEPS for x in e["step_losses"])  # noqa: E731
    l_ref, l_bat = lm(ref), lm(bat)
    g2 = {"mean_loss_1501_2000_ref": l_ref, "mean_loss_1501_2000_bat": l_bat,
          "rel_diff": (l_bat - l_ref) / l_ref, "pass": abs(l_bat - l_ref) / l_ref <= 0.003}
    same = [k for k in bat if k in res and bat[k]["stream_sha"] == res[k]["stream_sha"]]
    starts = [json.loads(ln)["step"] for ln in (OUT / "res/logs/train.jsonl").read_text().splitlines()
              if json.loads(ln)["event"] == "run_start"]
    g3 = {"intervals": len(bat), "res_intervals": len(res), "stream_sha_equal": len(same),
          "resume_starts": starts, "max_abs_dloss": max(abs(bat[k]["loss"] - res[k]["loss"]) for k in same),
          # one planned kill (step 1,250 -> resume at 1,000); any further resume (an unplanned
          # kill) must also land on the 500-step checkpoint grid
          "pass": len(same) == len(bat) == len(res) and starts[:2] == [0, 1000]
                  and all(s % 500 == 0 for s in starts) and starts == sorted(starts)}

    def share(s):
        w = [e for e in s.values() if e["step"] > 200]
        iv = sum(e["time_s"]["interval"] for e in w)
        return {k: sum(e["time_s"][k] for e in w) / iv for k in ("data_wait", "h2d", "fwd_bwd", "optim", "eval")} | \
               {"samples_per_s": mean(e["samples_per_s"] for e in w)}
    g4 = {"before_reference": share(ref), "after_batched": share(bat)}
    g4["pass"] = g4["after_batched"]["optim"] <= 0.10
    r = {"gate2": g2, "gate3": g3, "gate4": g4}
    print(json.dumps(r, indent=1))
    save("gates2_4", r)
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["parity", "bench", "train", "report"])
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("the Muon gates need the GPU (CUDA_VISIBLE_DEVICES hides it?)")
    {"parity": parity, "bench": bench, "train": train, "report": report}[a.what]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
