# Production run: operator runbook

One d384×10 with the A3 recipe, trained on a rented Ubuntu VM by `training/v6/tools/prod.py`, streaming everything to R2. The brief is `docs/capacity/training/prod/vm_harness_brief.md`, and the design decisions are in `training/v6/DECISIONS.md` under "VM harness".

**Launch only after the seed-t confirmation run (`Ct`) confirms the recipe.** The exported model is a contract-B net, and **it cannot play until the engine's contract-B support lands and passes `tools/s7_check.py` on this export.** That is separate work.

Commands below run from the repo root. `$PY` is `.venv/bin/python`.

---

## 1. Rent the VM

- **Card:** RunPod Secure Cloud, A100 SXM 80GB (16 vCPUs, 125 GB RAM, $1.59/hr).
- **Image:** any CUDA ≥ 12.6 Ubuntu image with `git` and `curl`. The RunPod PyTorch template works; `setup.sh` builds its own venv, so the image's Python doesn't matter.
- **Disk:** container disk ≥ 250 GB, on local NVMe. Clone the repo there, not onto a network volume. `setup.sh` needs ≥ 200 GB for `data/`, refuses a network filesystem, and needs `/dev/shm` ≥ 16 GB.
- `setup.sh` accepts any compute capability ≥ 8.0 card that passes its checks, but only the A100 is planned.

## 2. Clone and credentials

```bash
git clone https://<token>@github.com/elliasc04/GuoFish.git     # or a deploy key
cd GuoFish && git checkout v6-prod
cat > .env <<'EOF'
R2_ACCOUNT_ID=<cloudflare account id>
R2_BUCKET=guofishv6corpus
R2_ACCESS_KEY_ID=<R2 API token: Access Key ID>
R2_SECRET_ACCESS_KEY=<R2 API token: Secret Access Key>
EOF
```

- `.env` is gitignored. The run refuses a dirty checkout (`system.allow_dirty: false`), so never edit tracked files on the VM. Change settings with `--set` (recorded in `state.json` and `config.resolved.yaml`), or commit and pull.
- The token needs Object Read & Write on the bucket.

## 3. `setup.sh`

```bash
bash training/v6/setup.sh 2>&1 | tee setup.log
```

1. **Checks** (it fails with a message):
   - GPU compute capability ≥ 8.0;
   - driver CUDA ≥ 12.6;
   - ≥ 200 GB on a local filesystem;
   - RAM ≥ 64 GB (it warns below 96 GB). RAM and vCPUs are read from the container's cgroup limits, not the host's;
   - `/dev/shm` ≥ 16 GB;
   - ≥ 12 vCPUs.
2. **Environment:**
   - installs uv 0.10.12 and a Python 3.13 `.venv`;
   - installs `training/v6/requirements-prod.txt`, taking torch from the PyTorch index that matches the driver (cu129, cu128 or cu126);
   - prints the resolved versions.
3. **Data:**
   - pulls every file listed in `data/sha256.txt` from R2 `data/` into `data/`, 16 files at a time;
   - verifies each file's size and sha256, and fails on any mismatch;
   - re-running verifies without re-downloading.
4. **Smoke:**
   - runs the fast tests;
   - then runs 300 steps of `prod.yaml` at micro-batch 512×2 and at 1024×1, with no R2 writes;
   - prints samples/s, peak VRAM, the loader-wait fraction, the step-time split and the ETA to each branch point, and writes `models/v6/_smoke/smoke.json`.

`sha256.txt` lines are `sha256  size  relative_path`, with paths relative to `data/` (for example `processed/multipv_v3/train_0000.bin`). The object key is `data/<relative_path>`. The run needs:

- `processed/multipv_v3/` (shards and `manifest.json`) and `processed/strata/multipv_v3_train.strata2.{npy,json}`;
- `processed/val_frozen_90m_v2/` and `processed/strata/val_frozen_90m_v2_val.strata2.{npy,json}`;
- for the two v2 eval sets:
  - `processed/evalsets/{v2val_roots,v2val_derived}.json`;
  - `processed/multipv_v2/manifest.json` with its `val_*.bin` and `valderived_*.bin` shards;
  - `processed/strata/multipv_v2_{val,valderived}.strata2.{npy,json}`.

## 4. Go / no-go

Read the smoke table:

- **Go** only if loader wait is **< 5 %** and samples/s is **≥ 5,019**, which is 1.5× the local A3 proxy's 3,346 samples/s (SCREENING_REPORT.md).
- If throughput is below that, use the printed ETA (cumulative, decays included) against the 120 h cap to decide.
- **Pick the split** with the higher samples/s. 512×2 is the `prod.yaml` default; 1024×1 needs `--set optim.micro_batch=1024 optim.accum=1` at launch. The effective batch is 1,024 either way.
- **Loader wait ≥ 5 %:**
  - first try `--set data.workers=14`;
  - still ≥ 5 % → GPU-side target building (VM harness brief §7) is due, so don't launch.
- **Optimizer > 15 % of step time:** batched Muon (§7) is due. Launching anyway only costs time.

## 5. Launch

```bash
tmux new -s prod
$PY -m training.v6.tools.prod --set prod.sanity.ref_kl=<C0 quick-val KL at 40,960,000> \
    [optim.micro_batch=1024 optim.accum=1] 2>&1 | tee -a prod.log
# detach: Ctrl-b d;   reattach: tmux attach -t prod
```

- `prod.sanity.ref_kl` comes from `data/multiPV/CORPUS_V3_REPORT.md` (C0's quick-val KL at 40M). The driver refuses to start while it is null.
- The mixture in `prod.yaml` is the production pool. If C0 chose a different pool, commit that mixture to `prod.yaml` and pull before launching.
- **What happens:**
  - the stable phase runs to 300,001,280 samples, then the 300M branch's decay, then the stop rule, and so on at 600M, 1.2B and 2.4B;
  - each piece is a separate `train.py` / `branch_decay.py` process;
  - the driver refuses to start if `runs/prod_v6/state.json` already exists; use `--resume` for that.

## 6. The 40M sanity check

- At 40,960,000 samples the quick-val policy KL must be within ±3 % of `ref_kl`.
- The driver writes `evals/sanity_s40960000.json`.
- **On a failure:**
  - it writes `control.json` = `{"action": "pause", "by": "prod.py", "reason": …}`;
  - it stops the trainer at that checkpoint and waits;
  - `state.json` shows `"paused": true`, and `anomalies` has a `sanity` entry.
- Investigate, then write `continue` or `stop_now` (§8).

## 7. Watching `status.json`

```bash
$PY -m training.v6.r2 cat runs/prod_v6/status.json
```

It's written every 60 s. Check:

- `updated_utc` is recent (the driver is alive), and `last_step_utc` is recent (the trainer is alive).
- `samples_per_s` is near the smoke's rate.
- `step_time_frac.data_wait` < 0.05.
- `gpu.util_pct` is high.
- `upload_queue` is small, and `anomalies` is 0. Any anomaly is described in `state.json`.
- `eta_segment_end_h`, `eta_next_branch_end_h` and `cap_remaining_h` are the planning numbers.

## 8. `control.json`

```bash
echo '{"action": "pause"}' | $PY -m training.v6.r2 put runs/prod_v6/control.json -
```

The driver polls it every 5 min and acts once per change of its content.

| action | effect |
|---|---|
| `continue` | normal running. Resumes a pause and cancels a pending `stop_after_branch`. |
| `pause` | stops the trainer at its next rolling checkpoint (≤ 10,240,000 samples later), or at the end of the current branch, then waits for `continue`. The pod keeps billing while paused. |
| `stop_after_branch` | finishes the current stable segment and its branch, applies the rule, then stops, ships and exports. |
| `stop_now` | kills the trainer, ships the best finished branch (if any), exports it and exits. |

Add `"time_cap_h": 150` to any action to raise the cap. The cap is checked only before each new stable segment.

On `--resume`, the current `control.json` is acted on again: a `pause` or `stop_now` left there applies at once. Set it to `continue` first if that isn't what you want.

## 9. Resuming, on any VM

1. Stop the old driver if it's still running.
2. On the new VM: steps 1–3, with `--skip-smoke` if you like; the data pull is still needed.
3. Run:

```bash
tmux new -s prod
$PY -m training.v6.tools.prod --resume prod_v6 2>&1 | tee -a prod.log
```

- The driver reads `state.json`, downloads the latest checkpoint it names (main line, or the current branch's rolling checkpoint), and continues.
- It trusts a checkpoint only if its `.sha256` sidecar exists, matches `state.json`, and matches the downloaded bytes. Otherwise it refuses with `refusing …`.
- The launch's `--set` values come back from `state.json`. Extra `--set` values are appended. If `prod.store_prefix` was overridden at launch, pass it again.
- The data stream is closed-form in the sample index, so a resume continues the exact run.
- A local run directory left over from before is ignored; only store-verified checkpoints are used.
- A crashed trainer makes the driver exit non-zero after recording a `trainer_exit` anomaly. Resume the same way.

## 10. Fetching the export

```bash
$PY -m training.v6.r2 cat runs/prod_v6/summary.json | python -c "import json,sys; print(json.load(sys.stdin)['shipped'])"
$PY -m training.v6.r2 get runs/prod_v6/export/<file>.pt <file>.pt
$PY -m training.v6.r2 cat runs/prod_v6/export/<file>.pt.sha256; sha256sum <file>.pt
```

- The shipped branch is the best by frozen90 total, raw or EMA.
- The export is `tools/export.py` output (contract B, `value_scale` from the corpus manifest), already smoke-tested: 64 positions give identical output through `load_for_inference`.
- **It can't play until the engine's contract-B support lands.**

---

## 11. Store layout and schemas

Everything is under `runs/<run.name>/` (default `runs/prod_v6/`). A dashboard needs nothing else.

| Object | Written |
|---|---|
| `config.resolved.yaml` | at start: the trainer config plus the `prod:` section, `--set` applied |
| `provenance.json`, `code.patch` | at start (git sha, dirty files, versions, data hashes); `provenance_resume_<utc>.json` / `code_resume_<utc>.patch` on each resume |
| `status.json` | every 60 s |
| `state.json` | after every checkpoint and every phase change |
| `metrics/train-<seq:06d>.jsonl` | every 5 min, and before every checkpoint |
| `evals/<tag>.json` | as produced (tags below) |
| `ckpt/rolling/s<samples>.pt` + `.sha256` | every 10,240,000 samples on the main line; last 2 kept |
| `ckpt/stable/s<samples>.pt` + `.sha256` | every 150,000,640 samples, including every branch point; all kept |
| `branches/<name>/rolling/s<samples>.pt` + `.sha256` | every 10,240,000 samples inside a branch; last 2 kept (resume inside a branch) |
| `branches/<name>/final.pt` + `.sha256` | at each branch end; all kept |
| `summary.json` | at each branch end and at stop |
| `export/<file>.pt` + `.sha256` | at stop |
| `control.json` | written by the operator (or by the sanity check) |

**Names:**
- A branch is `s<from>_d<decay>`, for example `s300001280_d52941403`, as `branch_decay.py` names it. The decay is `ceil(from / 0.85) − from`.
- A segment is `main` (the stable line) or a branch name.
- Sample counts are always multiples of the 1,024 effective batch.

**Sidecar** (`*.sha256`): one line, `<sha256>  <size>  <file name>`. Upload order is always the checkpoint, then its sidecar, then `state.json`. A checkpoint without a matching sidecar is never trusted.

### `status.json`

```
run, phase ("stable" | "branch"), paused (bool), segment ("main" | <branch> | null while paused),
segment_end (samples), next_branch_point, samples, step, samples_per_s, lr, loss, grad_norm,
loader_wait_frac, step_time_s {interval, data_wait, h2d, fwd_bwd, optim, eval},
step_time_frac {data_wait, h2d, fwd_bwd, optim, eval}, peak_vram_mib,
gpu {util_pct, mem_used_mib, mem_total_mib} | null, passes {<group>: completed passes},
eta_segment_end_h, eta_next_branch_end_h, elapsed_h, time_cap_h, cap_remaining_h,
upload_queue, anomalies (count), control (last action), last_step_utc, updated_utc
```

All step numbers come from the trainer's latest log interval (50 steps). In `step_time_s`:
- `data_wait` and `eval` are host time. `eval` covers evals and checkpoint writes.
- `h2d`, `fwd_bwd` and `optim` are GPU-stream time from CUDA events, so the parts need not add up to `interval`.

### `state.json`

```
run, created_utc, updated_utc, config (path), sets [KEY=VALUE, …],
phase ("stable" | "branch" | "done"), k (index of the current branch point), segment_started,
main_start, main {key, sha256, samples, segment} | null   (latest main-line checkpoint),
branch (<name> | null), branch_ckpt {key, sha256, samples, segment} | null,
finals {<branch>: {key, sha256, samples, segment}}, paused (bool),
elapsed_s (wall time across sessions), trained_samples (stable + decay samples finished),
metrics_seq, time_cap_h, sanity (the sanity doc | null),
anomalies [{kind, detail, utc, segment}], branches […as summary], decisions […],
shipped {…} | null, stopped (reason | null)
```

Anomaly kinds:
- `nonfinite`: the trainer's non-finite step;
- `trainer_exit`: an unexpected exit;
- `sanity`;
- `upload_lag`: more than 3 checkpoints queued;
- `upload_error`;
- `control`: an ignored `control.json`.

Only `nonfinite` and `trainer_exit` stop the run at the first branch.

### `evals/<tag>.json`

| tag | contents |
|---|---|
| `<segment>_quick_s<N>` | the trainer's `quick_eval` event: `policy_kl`, `value_mse`, `total`, top-1/5, … on the fixed 32,768-record frozen90 subset, raw weights |
| `<segment>_full_s<N>` | `{segment, samples, reason, sets: {frozen90, v2val_roots, v2val_derived: {raw: {metrics}, ema: {metrics}}}}`, at every stable checkpoint and branch end |
| `<segment>_memgap_s<N>` | `{…, sets: {seen_policy: {raw\|ema: {gap: {policy_kl, value_mse}, group_passes, valid, seen: {metrics}}}}}`. The gap is (seen − frozen90) / frozen90 on 200,000 fixed policy-group training roots; `valid` once the group finished a pass (≈ 104M samples) |
| `decision_<branch>` | `{branch, branch_index, policy_kl, total, go, reason, [kl_gain, kl_ok, total_ok, prev_total], utc}`, where `policy_kl` and `total` are the better of raw and EMA |
| `timecap_<branch>` | before each stable segment after the first: `{elapsed_h, rate_samples_per_s, need_samples, projected_h, time_cap_h, ok}` |
| `sanity_s<N>` | `{samples, quick_kl, ref_kl, rel_dev, tol, pass}` |

The metric keys are `training/v6/eval/__init__.py`'s: `policy_kl`, `value_mse`, `total`, `policy_top1`, `sf_top1_pv0`, the `value/<stratum>/…` and `slice/…` cells, and so on.

### `summary.json`

```
run, updated_utc, gpu_hours, trained_samples,
branches [{name, from, end, frozen90 {raw, ema}, other_sets {<set>: {raw, ema}},
           memgap {seen_policy: {raw, ema}}, final {key, sha256, …} | null}],
decisions [as evals/decision_*], shipped {branch, weights, frozen90_total, export_key,
export_sha256, contract} | null, stopped (reason | null), anomalies […]
```

### `metrics/train-<seq>.jsonl`

These are the trainer's own JSONL lines (`training/v6/train.py`), each with a `segment` field added.

- `event: "step"` (every 50 steps) carries:
  - `samples`, `step`, `lr`, `loss`, `step_losses[50]`;
  - `soft_kl`, `value_mse`, `grad_norm`, `nonfinite_steps`;
  - `samples_per_s`, `loader_wait_frac`, `peak_vram_mib`;
  - `time_s {…}`, `passes {…}`, `stream_sha` (the digest of that interval's records).
- Other events: `run_start`, `quick_eval`, `full_eval`, `memorization_gap`, `checkpoint`, `best`, `anomaly`, `run_stop`, `run_end`.
- After a resume, the steps since the resume checkpoint appear twice with identical values. Keep the last copy per `(segment, step)`.

### `control.json`

```
{"action": "continue" | "pause" | "stop_after_branch" | "stop_now", "time_cap_h": <float, optional>, …}
```

Any other fields (`by`, `reason`, `utc`) are informational. A content change is what triggers an action. An invalid action is ignored and logged as a `control` anomaly.
