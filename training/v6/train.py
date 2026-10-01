"""v6 trainer (§8-§10).

    python -m training.v6.train --config training/v6/config/configs/base.yaml \
        [--set a.b=c ...] [--resume latest|<ckpt>] [--stop-at-samples N] [--seen-eval GROUP:N]

--stop-at-samples ends the run early at sample N without touching the schedule
(a production stable segment: the checkpoint at N is the caller's to request via
ckpt.stable_every_samples). --seen-eval evaluates N fixed training records of
mixture group GROUP at every full eval and logs the memorization gap against
frozen90; it is a diagnostic, so it stays out of the config and its hash.

The data stream, augmentation and LR are closed-form in the sample index, so
a resume at ANY step boundary (mid-pass or at a pass boundary) continues the
exact run: the checkpoint carries weights, EMA, optimizer, the global torch
RNG (dropout) and the sample index. No host sync per micro-batch: metrics
accumulate on device and are read once per `system.log_every` steps. A
non-finite window is skipped on device (fused AdamW's found_inf protocol);
the next log sync sees it, writes an emergency checkpoint of the last good
weights and exits non-zero.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path

# One BLAS thread (H2): numpy's OpenBLAS pre-allocates a buffer per core, ~490 MB
# private in every process that loads it. DataLoader workers inherit this
# environment and never call BLAS; nor does this process. Must precede `import numpy`
# to cover this process too; workers get it either way.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
import torch._inductor.config as inductor_config

from core.guofish_net import build_model
from training.v6.ckpt import (
    JsonlLog, atomic_save, check_dirty, checkpoint_blob, latest_checkpoint, prepare_run_dir,
    provenance, rotate, sha256_file, utc,
)
from training.v6.config import (
    RESUME_WHITELIST, build_config, config_hash, diff_paths, dump_yaml, load_config, to_plain,
)
from training.v6.data.batch import BatchBuilder, StreamDataset
from training.v6.data.formats import REPO
from training.v6.data.mixture import Mixture
from training.v6.data.reader import ShardSet
from training.v6.data.strata import DEFINITION_HASH, load_strata
from training.v6.eval import EVALSETS, EvalSet, evaluate, load_evalset, quick_subset
from training.v6.losses import LossFn, loss_normalizers
from training.v6.optim import EMA, Schedule, build_optimizer

# Plain Triton kernel names. A descriptive fused name here runs to 135 chars; with
# the cache's 52-char hash and 50-char temp dir no cache root keeps the path under
# MAX_PATH (LongPathsEnabled is 0) and Triton fails with FileNotFoundError. Names
# only: the generated code is otherwise unchanged. bench.py and s3_check.py import
# this module, so every compile path gets it.
inductor_config.triton.descriptive_names = False

METRICS = ("loss", "soft_kl", "n_soft", "hard_ce", "hard_nll", "n_hard", "value_loss", "value_se")
CRASH_EXIT = 17


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def verify_frozen(frozen_dir: Path) -> str:
    """Every frozen shard must still hash to its manifest entry."""
    mpath = frozen_dir / "manifest.json"
    man = json.loads(mpath.read_text())
    for s in man["shards"]:
        if sha256_file(frozen_dir / s["name"]) != s["sha256"]:
            raise SystemExit(f"frozen val shard {s['name']} no longer matches its manifest")
    return sha256_file(mpath)


def pinned_quick_subset(ev, n_frozen: int) -> np.ndarray:
    """The committed quick-val indices (follow-ups §2): independent of the strata, so a
    strata-definition change can't shift them. Refused unless sha256, size and range match."""
    path = resolve(ev.quick_indices)
    if sha256_file(path) != ev.quick_indices_sha256:
        raise SystemExit(f"{ev.quick_indices}: sha256 is not the pinned {ev.quick_indices_sha256[:12]}")
    idx = np.load(path)
    if (len(idx) != ev.quick_size or idx.dtype != np.int64 or (np.diff(idx) <= 0).any()
            or idx[0] < 0 or idx[-1] >= n_frozen):
        raise SystemExit(f"{ev.quick_indices}: not {ev.quick_size:,} sorted unique indices into "
                         f"the {n_frozen:,}-record frozen set")
    return idx


def _strata_meta(path: Path) -> dict:
    return json.loads(path.with_suffix(".json").read_text())


def run(cfg, *, run_dir: Path, resume: Path | None = None, branch: bool = False,
        crash_after_steps: int | None = None, stop_at: int | None = None,
        seen_eval: tuple[str, int] | None = None, muon_impl: str = "batched") -> dict:
    dev = torch.device(cfg.system.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("system.device=cuda but CUDA is not available")
    torch.backends.cuda.matmul.allow_tf32 = cfg.system.tf32
    torch.backends.cudnn.allow_tf32 = cfg.system.tf32
    for name, p in [("data.corpus", cfg.data.corpus), ("data.manifest", cfg.data.manifest),
                    ("data.strata", cfg.data.strata), ("eval.frozen_dir", cfg.eval.frozen_dir),
                    ("eval.frozen_strata", cfg.eval.frozen_strata),
                    *(("eval.extra_sets", EVALSETS / f"{n}.json") for n in cfg.eval.extra_sets)]:
        if not resolve(p).exists():
            raise SystemExit(f"{name}: {p} does not exist")

    plain, chash = to_plain(cfg), config_hash(cfg)
    manifest_path = resolve(cfg.data.manifest)
    strata_path = resolve(cfg.data.strata)
    hashes = {"corpus_manifest_sha256": sha256_file(manifest_path),
              "strata_definition_hash": DEFINITION_HASH,
              "strata_codes_sha256": _strata_meta(strata_path)["codes_sha256"],
              "frozen_val_manifest_sha256": verify_frozen(resolve(cfg.eval.frozen_dir))}
    prov, patch = provenance(cfg, chash, hashes)
    check_dirty(prov, cfg.system.allow_dirty)
    prepare_run_dir(run_dir, resume is not None)
    log = JsonlLog(run_dir / "logs/train.jsonl", run_dir / "logs/console.log")
    tag = "" if resume is None else f"_resume_{utc().replace(':', '')}"
    (run_dir / f"provenance{tag}.json").write_text(json.dumps(prov, indent=2) + "\n")
    (run_dir / f"code{tag}.patch").write_text(patch)
    if not (run_dir / "config.resolved.yaml").exists():      # fresh run or new branch
        (run_dir / "config.resolved.yaml").write_text(dump_yaml(cfg))

    # ---- data ----------------------------------------------------------
    shards = ShardSet(resolve(cfg.data.corpus), cfg.data.split, manifest_path)
    strata = np.asarray(load_strata(strata_path, len(shards), hashes["corpus_manifest_sha256"]))
    grouped = cfg.mixture.groups != "natural"
    mixture = Mixture(cfg.mixture, len(shards), cfg.optim.micro_batch, cfg.run.data_seed,
                      strata=strata if grouped else None,
                      members_dir=run_dir / "mixture" if grouped else None)
    t = cfg.targets
    man = shards.manifest
    if t.policy_soft.source == "stored" and "epsilon" in man and man["epsilon"] != t.policy_soft.epsilon:
        raise SystemExit(f"stored targets were built with epsilon {man['epsilon']}, "
                         f"config says {t.policy_soft.epsilon}")
    builder = BatchBuilder(cfg.model.token_scheme, t.mirror_prob, cfg.run.data_seed,
                           t.policy_soft.epsilon, t.policy_soft.temperature)
    eff, accum = cfg.optim.effective_batch, cfg.optim.accum
    sched = Schedule(cfg.schedule, cfg.optim.lr, eff)
    total_steps = sched.total_steps
    norms = loss_normalizers(cfg, mixture, strata)

    frozen = EvalSet("frozen90", resolve(cfg.eval.frozen_dir), "val", resolve(cfg.eval.frozen_strata),
                     cfg.model.token_scheme, cfg.eval.batch, cfg.eval.workers)
    quick_idx = (pinned_quick_subset(cfg.eval, len(frozen.indices)) if cfg.eval.quick_indices
                 else quick_subset(frozen.codes, cfg.eval.quick_size, cfg.eval.quick_seed))
    extra = [load_evalset(n, cfg.model.token_scheme, cfg.eval.batch, cfg.eval.workers)
             for n in cfg.eval.extra_sets]
    seen = None
    if seen_eval is not None:                # fixed training records of one group
        gname, n_seen = seen_eval
        if gname not in mixture.names:
            raise SystemExit(f"--seen-eval: no mixture group {gname!r} in {mixture.names}")
        members = np.asarray(mixture._member(mixture.names.index(gname)), dtype=np.int64)
        if n_seen > len(members):
            raise SystemExit(f"--seen-eval: {n_seen:,} > {len(members):,} records in {gname}")
        idx = np.sort(np.random.default_rng([cfg.run.data_seed, 0x5EE7]).choice(members, n_seen, replace=False))
        seen = EvalSet(f"seen_{gname}", resolve(cfg.data.corpus), cfg.data.split, strata_path,
                       cfg.model.token_scheme, cfg.eval.batch, cfg.eval.workers, indices=idx,
                       manifest=manifest_path)

    # ---- model / optimizer / resume -----------------------------------
    torch.manual_seed(cfg.run.init_seed)            # dropout (and v5_deepcopy's init draws)
    model = build_model(cfg.model, cfg.run.init_seed).to(dev)
    opt = build_optimizer(model, cfg.optim, sched.lr(0), muon_impl=muon_impl,
                          compile=cfg.system.compile and dev.type == "cuda")
    ema = EMA(model, cfg.ema.half_life_samples, eff) if cfg.ema.enabled else None
    loss_fn = LossFn(cfg, norms, dev)
    step0, best = 0, math.inf
    if resume is not None:
        ck = torch.load(resume, map_location="cpu", weights_only=True)
        # fields added to the schema since the checkpoint compare at their defaults
        changed = diff_paths(to_plain(build_config(ck["config"])), plain)
        allowed = RESUME_WHITELIST - (set() if branch else {"schedule.total_samples"})
        if changed - allowed:
            raise SystemExit(f"resume refused: config differs at {sorted(changed - allowed)}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if ema is not None:
            if ck["ema"] is None:
                raise SystemExit("config enables EMA but the checkpoint has none")
            ema.load_state_dict({k: v.to(dev) for k, v in ck["ema"].items()})
        torch.set_rng_state(ck["rng"]["torch"])
        if dev.type == "cuda" and "cuda" in ck["rng"]:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        step0, best = int(ck["step"]), float(ck["best"])
        if int(ck["samples"]) != step0 * eff:
            raise SystemExit("checkpoint sample index is not step x effective batch")
        if step0 >= total_steps:
            raise SystemExit(f"nothing to do: checkpoint at step {step0} of {total_steps}")
    train_fn = (torch.compile(model.forward_train, mode=cfg.system.compile_mode)
                if cfg.system.compile else model.forward_train)
    amp = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16))
           if cfg.system.precision == "bf16" else contextlib.nullcontext)
    n_params = sum(p.numel() for p in model.parameters())

    log.event("run_start", run=cfg.run.name, resume=str(resume) if resume else None, branch=branch,
              step=step0, samples=step0 * eff, config_hash=chash, params=n_params, muon_impl=muon_impl,
              schedule=sched.describe(), normalizers=norms, records=len(shards),
              record_format=shards.format, mixture={"names": mixture.names,
              "shares": mixture.shares.tolist(), "sizes": mixture.sizes,
              "rest_excluded": mixture.rest_excluded}, quick_subset=len(quick_idx),
              git_sha=prov["git_sha"], dirty_files=len(prov["dirty_files"]))
    log.say(f"[{cfg.run.name}] {n_params:,} params | {len(shards):,} {shards.format} records | "
            f"steps {step0}->{total_steps} x {eff} | cfg {chash[:12]} | norms {norms}")

    # ---- events --------------------------------------------------------
    ema_model = None
    state = {"best": best, "metrics": {}}

    def save(path: Path, done: int, reason: str):
        atomic_save(checkpoint_blob(
            model=model, ema=ema, optimizer=opt, samples=done, step=done // eff, cfg_plain=plain,
            cfg_hash=chash, model_cfg=cfg.model, prov=prov, metrics=state["metrics"],
            reason=reason, best=state["best"], device=dev.type), path)
        log.event("checkpoint", path=str(path.relative_to(run_dir)), samples=done, reason=reason)

    def full_eval(done: int, reason: str):
        nonlocal ema_model
        nets = {"raw": model}
        if ema is not None:
            if ema_model is None:
                ema_model = build_model(cfg.model).to(dev)
            ema_model.load_state_dict(ema.weights_for(model))
            nets["ema"] = ema_model
        results = {which: evaluate(net, frozen, dev, amp, mirror_n=cfg.eval.mirror_n,
                                   mirror_seed=cfg.eval.quick_seed) for which, net in nets.items()}
        for which, m in results.items():
            log.event("full_eval", samples=done, weights=which, reason=reason, set="frozen90", **m)
            v = m[cfg.eval.best_metric]
            if v < state["best"]:
                state["best"] = v
                weights = (model.state_dict() if which == "raw" else ema.weights_for(model))
                atomic_save({"state_dict": {k: w.detach().cpu() for k, w in weights.items()},
                             "weights": which, "samples": done, "metric": cfg.eval.best_metric,
                             "value": v, "metrics": m, "model_config": cfg.model.to_dict(),
                             "config_hash": chash}, run_dir / "best.pt")
                log.event("best", samples=done, weights=which, value=v)
        log.say(f"  full eval @ {done:,} ({reason}): " + " | ".join(
            f"{w} KL {m['policy_kl']:.5f} MSE {m['value_mse']:.5f} total {m['total']:.5f}"
            for w, m in results.items()))
        if seen is not None:                # memorization gap: (seen - frozen90) / frozen90
            passes = mixture.passes(done)[seen_eval[0]]
            for which, net in nets.items():
                m, f = evaluate(net, seen, dev, amp), results[which]
                gap = {k: (m[k] - f[k]) / f[k] for k in ("policy_kl", "value_mse")}
                log.event("memorization_gap", samples=done, weights=which, reason=reason, set=seen.name,
                          group_passes=passes, valid=passes >= 1, gap=gap, seen=m)
        for es in extra:                    # H3: reported, never used for best.pt
            for which, net in nets.items():
                m = evaluate(net, es, dev, amp)
                log.event("full_eval", samples=done, weights=which, reason=reason, set=es.name, **m)
                results[f"{which}/{es.name}"] = m
        state["metrics"] = results

    def events(done: int):
        at = lambda every: every > 0 and done % every == 0  # noqa: E731
        if at(cfg.eval.quick_every_samples):
            m = evaluate(model, frozen, dev, amp, indices=quick_idx)
            log.event("quick_eval", samples=done, **m)
            log.say(f"  quick eval @ {done:,}: KL {m['policy_kl']:.5f} MSE {m['value_mse']:.5f}")
        stable = (cfg.schedule.kind == "wsd" and at(cfg.ckpt.stable_every_samples)
                  and done <= sched.decay_start)
        if stable or at(cfg.eval.full_every_samples):
            full_eval(done, "stable" if stable else "periodic")
        if stable:
            save(run_dir / "stable" / f"s{done}.pt", done, "stable")
        if at(cfg.ckpt.every_samples) and done < total_steps * eff:   # the end saves its own
            save(run_dir / "ckpt" / f"s{done}.pt", done, "periodic")
            rotate(run_dir / "ckpt", cfg.ckpt.keep_last)

    # ---- loop ----------------------------------------------------------
    stop_steps = total_steps
    if stop_at is not None:
        if stop_at % eff or not step0 * eff < stop_at <= total_steps * eff:
            raise SystemExit(f"--stop-at-samples {stop_at}: must be a multiple of {eff} "
                             f"in ({step0 * eff}, {total_steps * eff}]")
        stop_steps = stop_at // eff
    cuda = dev.type == "cuda"

    def mark():
        """GPU-stream timestamp on CUDA (read at the log sync, no per-step host
        sync); wall clock on CPU, where every op is synchronous."""
        if not cuda:
            return time.perf_counter()
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    spans = []                          # (kind, start mark, end mark) since the last log
    ds = StreamDataset(shards, mixture, builder, total_steps * accum)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=None, sampler=range(step0 * accum, stop_steps * accum),
        num_workers=cfg.data.workers, prefetch_factor=cfg.data.prefetch_factor if cfg.data.workers else None,
        pin_memory=dev.type == "cuda", persistent_workers=False, generator=torch.Generator())
    it = iter(loader)
    acc = torch.zeros(len(METRICS), device=dev)
    gn_sum = torch.zeros((), device=dev)
    nonfinite = torch.zeros((), device=dev)
    step_losses = []
    stream = hashlib.sha256()
    t_int, wait, excl, n_int = time.perf_counter(), 0.0, 0.0, 0
    params = list(model.parameters())

    for step in range(step0, stop_steps):
        s = step * eff
        lr, b1 = sched.lr(s), sched.beta1(s)
        for g in opt.param_groups:
            g["lr"] = lr
            if b1 is not None and "betas" in g:
                g["betas"] = (b1, g["betas"][1])
        t0 = time.perf_counter()
        window = [next(it) for _ in range(accum)]
        wait += time.perf_counter() - t0
        wl = torch.zeros((), device=dev)
        for b in window:
            stream.update(b["record_index"].numpy().tobytes())
            stream.update(b["mirrored"].numpy().tobytes())
            m0 = mark()
            bd = {k: v.to(dev, non_blocking=True) for k, v in b.items()
                  if k not in ("record_index", "group", "sample_index", "mirrored")}
            m1 = mark()
            with amp():
                out = train_fn(bd["tokens"])
            loss, m = loss_fn(out, bd)
            loss.backward()
            wl += loss.detach()
            acc += torch.stack([m[k] for k in METRICS])
            spans += [("h2d", m0, m1), ("fwd_bwd", m1, mark())]
        m0 = mark()
        gn = torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
        found = (~torch.isfinite(wl) | ~torch.isfinite(gn)).float()
        opt.found_inf, opt.grad_scale = found, None
        opt.step()
        opt.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)
        spans.append(("optim", m0, mark()))
        gn_sum += torch.nan_to_num(gn, nan=0.0, posinf=0.0)
        nonfinite += found
        step_losses.append(wl)
        n_int += 1
        done = s + eff

        if (step + 1) % cfg.system.log_every == 0 or step + 1 in (total_steps, stop_steps):
            vals = torch.cat([acc, gn_sum[None], nonfinite[None], torch.stack(step_losses)]).tolist()
            a = dict(zip(METRICS, vals[:len(METRICS)]))
            gnm, nf, losses = vals[len(METRICS)] / n_int, vals[len(METRICS) + 1], vals[len(METRICS) + 2:]
            wall = time.perf_counter() - t_int - excl
            rows = n_int * eff
            if cuda:
                torch.cuda.synchronize()
            tm = {"h2d": 0.0, "fwd_bwd": 0.0, "optim": 0.0}
            for kind, x, y in spans:
                tm[kind] += x.elapsed_time(y) / 1e3 if cuda else y - x
            spans.clear()
            rec = {"samples": done, "step": step + 1, "lr": lr, "beta1": b1,
                   "loss": sum(losses) / len(losses), "step_losses": losses,
                   "soft_kl": a["soft_kl"] / a["n_soft"] if a["n_soft"] else None,
                   "hard_ce": a["hard_ce"] / a["n_hard"] if a["n_hard"] else None,
                   "hard_nll": a["hard_nll"] / a["n_hard"] if a["n_hard"] else None,
                   "value_loss": a["value_loss"] / rows, "value_mse": a["value_se"] / rows,
                   "n_soft": int(a["n_soft"]), "n_hard": int(a["n_hard"]),
                   "grad_norm": gnm, "nonfinite_steps": int(nf),
                   "samples_per_s": rows / wall if wall > 0 else None,
                   "loader_wait_frac": wait / wall if wall > 0 else None,
                   "peak_vram_mib": (torch.cuda.max_memory_allocated() / 2 ** 20
                                     if dev.type == "cuda" else None),
                   # seconds this interval: data_wait and eval (evals + checkpoint writes) are
                   # host time; h2d, fwd_bwd and optim are GPU-stream time (CUDA events)
                   "time_s": {"interval": wall + excl, "data_wait": wait, **tm, "eval": excl},
                   "passes": mixture.passes(done), "stream_sha": stream.hexdigest()[:16]}
            log.event("step", **rec)
            if step + 1 == step0 + cfg.system.log_every or (step + 1) % (cfg.system.log_every * 20) == 0:
                log.say(f"  step {step + 1:,}/{total_steps:,} loss {rec['loss']:.5f} lr {lr:.3e} "
                        f"{rec['samples_per_s'] or 0:,.0f} samples/s")
            if nf:
                path = run_dir / "ckpt" / f"emergency_s{done}.pt"
                save(path, done, f"non-finite loss or gradient in {int(nf)} step(s)")
                log.event("anomaly", samples=done, nonfinite_steps=int(nf))
                log.close()
                raise SystemExit(f"non-finite loss/gradient; emergency checkpoint {path}")
            acc.zero_()
            gn_sum.zero_()
            step_losses.clear()
            stream = hashlib.sha256()      # per-interval digest: comparable across a resume
            t_int, wait, excl, n_int = time.perf_counter(), 0.0, 0.0, 0

        te = time.perf_counter()
        events(done)
        excl += time.perf_counter() - te
        if crash_after_steps is not None and step + 1 == crash_after_steps:
            log.close()
            os._exit(CRASH_EXIT)          # simulated kill: no cleanup, no checkpoint

    if stop_steps < total_steps:
        log.event("run_stop", samples=stop_steps * eff)
        log.say(f"[{cfg.run.name}] stopped at {stop_steps * eff:,} samples (--stop-at-samples)")
        log.close()
        shards.close()
        return state["metrics"]
    end = total_steps * eff
    full_eval(end, "end")
    final = run_dir / ("final.pt" if branch else f"ckpt/s{end}.pt")
    save(final, end, "end")
    rotate(run_dir / "ckpt", cfg.ckpt.keep_last)
    log.event("run_end", samples=end, best=state["best"])
    log.say(f"[{cfg.run.name}] done at {end:,} samples; best {cfg.eval.best_metric} {state['best']:.5f}")
    log.close()
    shards.close()
    return state["metrics"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--resume", default=None, help="'latest' or a checkpoint path")
    ap.add_argument("--crash-after-steps", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--stop-at-samples", type=int, default=None)
    ap.add_argument("--seen-eval", default=None, type=parse_seen, metavar="GROUP:N")
    ap.add_argument("--muon-impl", default="batched", choices=["batched", "reference"],
                    help=argparse.SUPPRESS)     # reference: the per-matrix Muon, for the parity gates
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    run_dir = resolve(cfg.run.out_root) / cfg.run.name
    resume = None
    if args.resume == "latest":
        resume = latest_checkpoint(run_dir)
    elif args.resume:
        resume = Path(args.resume)
    run(cfg, run_dir=run_dir, resume=resume, crash_after_steps=args.crash_after_steps,
        stop_at=args.stop_at_samples, seen_eval=args.seen_eval, muon_impl=args.muon_impl)
    return 0


def parse_seen(spec: str) -> tuple[str, int]:
    group, _, n = spec.rpartition(":")
    if not group or not n.isdigit() or int(n) <= 0:
        raise argparse.ArgumentTypeError(f"--seen-eval {spec!r}: expected GROUP:N")
    return group, int(n)


if __name__ == "__main__":
    raise SystemExit(main())
