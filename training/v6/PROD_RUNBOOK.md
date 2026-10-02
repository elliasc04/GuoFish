# Production run: operator runbook

One d384×10 with the A3 recipe, trained on a rented Ubuntu VM by `training/v6/tools/prod.py`, streaming everything to R2. The brief is `docs/capacity/training/prod/vm_harness_brief.md`, and the design decisions are in `training/v6/DECISIONS.md` under "VM harness".

**The seed-t confirmation run (`Ct`) confirmed the recipe on 2026-09-30.** The exported model is a contract-B net, and **it cannot play until the engine's contract-B support lands and passes `tools/s7_check.py` on this export.** That is separate work.

**The workflow:**
1. Rent the VM and clone (§1–2).
2. `setup.sh --build-corpus` builds corpus v3 on the VM and runs the smoke (§3).
3. Go / no-go (§4).
4. Launch production *at risk* on the production pool (§5).
5. Upload v3 from a second pane (§5a).
6. Locally, `C0` runs on the 5070 once the upload is done. If it fails, restart on a fallback pool (§6a).

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
R2_ENDPOINT=https://<cloudflare account id>.r2.cloudflarestorage.com
R2_BUCKET=guofishv6corpus
R2_ACCESS_KEY_ID=<R2 API token: Access Key ID>
R2_SECRET_ACCESS_KEY=<R2 API token: Secret Access Key>
EOF
```

- `.env` is gitignored. The run refuses a dirty checkout (`system.allow_dirty: false`), so never edit tracked files on the VM. Change settings with `--set` (recorded in `state.json` and `config.resolved.yaml`), or commit and pull.
- The token needs Object Read & Write on the bucket.
- **Use the account's default endpoint.** The jurisdiction endpoint (`<id>.us.r2.cloudflarestorage.com`) answers with **no buckets** for this account. A `NoSuchBucket` error means the wrong endpoint.

## 3. `setup.sh`

```bash
bash training/v6/setup.sh --build-corpus 2>&1 | tee setup.log    # the first VM
bash training/v6/setup.sh --pull-corpus 2>&1 | tee setup.log     # a replacement VM, once v3 is uploaded (§5a)
```

1. **Checks** (it fails with a message):
   - GPU compute capability ≥ 8.0;
   - driver CUDA ≥ 12.6;
   - ≥ 200 GB on a local filesystem;
   - RAM ≥ 64 GB (it warns below 96 GB). RAM and vCPUs are read from the container's cgroup limits, not the host's;
   - `/dev/shm` ≥ 16 GB;
   - ≥ 12 vCPUs.
2. **Environment:**
   - installs uv 0.10.12 and a Python 3.13.7 `.venv`;
   - installs `training/v6/requirements-prod.txt`, taking torch from the PyTorch index that matches the driver (cu129, cu128 or cu126);
   - prints the resolved versions.
3. **Data, `--build-corpus`:**
   - **The build** runs `data/multiPV/vm_build_v3.sh --r2-prefix data/ --work data`, which:
     - pulls and verifies the inputs: dump, Pass A index, rate plan, expected counts, frozen val sets, eval sets;
     - checks the index-only selection against the committed exact counts;
     - builds corpus v3 (≈ 104 GB; ≥ 2 h, a progress line every 5 min);
     - runs the gates, then writes the strata, the sha256 list and `data/processed/v3_build/build_ok.json`.
   - **Its exits**, as `setup.sh` reports them:
     - **2**, inputs: missing, or failing their sha256;
     - **3**, count mismatch: the selection doesn't reproduce `v3_expected_counts.json`. Don't train;
     - **4**, the build itself: disk, builder or strata;
     - **5**, a gate failed. Don't train.

     The logs are in `data/processed/v3_build/`.
   - **On success:** `setup.sh` prints `build_ok.json`'s counts and hashes, then re-verifies frozen90 v2 and the two v2 eval sets against `data/sha256.txt`.
   - **Disk:** the build peaks at ≈ 103 GB, plus checkpoints; hence the 200 GB floor.
4. **Data, `--pull-corpus`:**
   - refuses unless `data/multipv_v3/upload_ok.json` is in R2;
   - otherwise pulls v3 and its strata from `data/multipv_v3/`, verifying every sha256, then `build_ok.json`, frozen90 v2 and the eval sets.
5. **Smoke:**
   - verifies `prod.yaml`'s pins (§6a);
   - runs the fast tests;
   - then runs 300 steps of `prod.yaml` on v3 at micro-batch 512×2 and at 1024×1, with no R2 writes;
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
- **Optimizer share:** batched Muon is built and on by default (VM_HARNESS_REPORT.md, "Follow-ups"). The smoke's `optim` fraction should be well under 15 %.

## 5. Launch (at risk, on the production pool)

```bash
tmux new -s prod
$PY -m training.v6.tools.prod [--set optim.micro_batch=1024 optim.accum=1] 2>&1 | tee -a prod.log
# detach: Ctrl-b d;   reattach: tmux attach -t prod
```

- **The pins:** `prod.yaml` pins A3's quick-val KL at 40,960,000 as the primary sanity reference, plus the strata definition, the quick-val subset and the corpus manifest (against `build_ok.json`). The driver verifies the pins at launch and on every resume, and refuses on any drift (§6a).
- **No `C0` needed to launch:** the driver refuses only without the primary reference.
- **What happens:**
  - the stable phase runs to 300,001,280 samples, then the 300M branch's decay, then the stop rule, and so on at 600M, 1.2B and 2.4B;
  - each piece is a separate `train.py` / `branch_decay.py` process;
  - the driver refuses to start if `runs/prod_v6/state.json` already exists; use `--resume` for that.

## 5a. Upload corpus v3 while training

Right after the launch, in a second `tmux` pane:

```bash
tmux new-window -t prod
nohup bash data/multiPV/vm_upload_v3.sh --work data > v3_upload.log 2>&1 &
```

- **What it uploads:** v3 and its strata go to `data/multipv_v3/`, two files at a time, so training isn't starved. Then it publishes `data/multipv_v3/sha256.txt` and `build_ok.json`.
- **Then it deletes** the local dump and index copies.
- **`upload_ok.json` comes last.**
- **Resumable:** re-run the same line, and files already uploaded at the right size are skipped.
- **Done when** this prints the uploader's stats JSON:

  ```bash
  $PY -m training.v6.r2 cat data/multipv_v3/upload_ok.json
  ```
- **Only then** can a replacement VM use `--pull-corpus`, and **only then does the local `C0` pool check start** on the 5070. It pulls v3 from R2. Until then, a lost VM means `--build-corpus` again. The shards come out the same, but the manifest hash doesn't (§9).

## 6. The 40M sanity references

- **Primary**, pinned in `prod.yaml`: A3's quick-val policy KL at 40,960,000 samples, **0.73995** (run A3, config `04cda02b8526`, its `train.jsonl`).
- **The check:** at 40,960,000, production's quick-val KL on the same 32,768 frozen90 records must be within ±3 %. Those records are pinned in `training/v6/config/quickval_frozen90_32k.npy`.
- **Secondary**, optional, after launch: `C0`'s KL at 40,960,000.

  ```bash
  echo '{"source": "C0", "ref_kl": <C0 quick-val KL>, "samples": 40960000, "ref_run": "<C0 run>"}' \
    | $PY -m training.v6.r2 put runs/prod_v6/reference.json -
  ```

  - The driver polls it every 5 min.
  - It checks it as soon as production has a main-line quick-val at that sample count, which is immediately if training has already passed it.
  - Each reference is checked once per run.
- **Every check** writes `evals/sanity_<source>_s<N>.json` (`source` is `primary` or the file's `source`).
- **On a failure:**
  - it writes `control.json` = `{"action": "pause", "by": "prod.py", "reason": …}`;
  - it stops the trainer at its next checkpoint and waits;
  - the reason is in `status.json` (`pause_reason`) and `state.json`, and `anomalies` has a `sanity` entry.
- Investigate, then write `continue` or `stop_now` (§8).

## 6a. Pins, and restarting on another pool

**The pins.** Each production config pins:
- the strata definition hash: v3, `5e5fdedaae85…`;
- the quick-val subset file's sha256;
- the A3 reference;
- the corpus manifest, through `build_ok.json`: the manifest must hash to its `manifest_sha256`. An optional literal pin is `prod.pins.corpus_manifest_sha256`.

A `build_ok.json` from a `--corpus-smoke` build is refused.

**The pool configs:**

| config | mixture | use when |
|---|---|---|
| `prod.yaml` (run `prod_v6`) | the production pool, `t20` included | the default launch; `C0` passes |
| `prod_no_t20.yaml` (`prod_v6_no_t20`) | the production pool without `t20` | `C0` fails, `C1` passes |
| `prod_a3pool.yaml` (`prod_v6_a3pool`) | A3's `in_90m` pool, on corpus v3 | `C0` (and `C1`) fail |

**To restart on another pool:**
1. Stop the running driver:

   ```bash
   echo '{"action": "stop_now"}' | $PY -m training.v6.r2 put runs/prod_v6/control.json -
   ```

   Wait for `phase: done` in `status.json`. It may ship a finished branch; that's harmless.
2. In the same `tmux` window, start a fresh run on the chosen config:

   ```bash
   $PY -m training.v6.tools.prod --config training/v6/config/configs/prod_no_t20.yaml 2>&1 | tee -a prod.log
   ```

   It's a new run name, so `runs/<name>/` starts empty. The old run's objects stay in R2 for the record. The corpus on the VM is the same, so nothing is rebuilt.

## 7. Watching `status.json`

```bash
$PY -m training.v6.r2 cat runs/prod_v6/status.json
```

It's written every 15 s (`prod.status_every_s`). Check:

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
2. On the new VM: steps 1–2, then `setup.sh --pull-corpus` (`--skip-smoke` if you like).
   - That needs v3's upload to have finished (§5a). If it hadn't, run `setup.sh --build-corpus` again. The shards come out the same (the build is deterministic), but the manifest records timestamps and the git sha, so **its hash changes on every build**. The pins follow the new `build_ok.json`, so resume still passes. Don't set a literal `prod.pins.corpus_manifest_sha256` unless you'll only ever *pull* this corpus.
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
| `status.json` | every 15 s |
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
run, phase ("stable" | "branch"), paused (bool), pause_reason (str | null: why the run paused itself),
segment ("main" | <branch> | null while paused),
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
metrics_seq, time_cap_h, quick_kl {<samples>: main-line quick-val KL},
sanity {<source>: the sanity doc}   ("primary", or reference.json's source),
references [{source, ref_kl, samples, tol, ref_run}]   (secondary, from reference.json),
pause_reason (str | null), pins {strata_definition_hash, corpus_manifest_sha256, build_ok {…},
quick_indices_sha256}   (what check_pins verified at launch; pins_resume on each resume),
anomalies [{kind, detail, utc, segment}], branches […as summary], decisions […],
shipped {…} | null, stopped (reason | null)
```

Anomaly kinds:
- `nonfinite`: the trainer's non-finite step;
- `trainer_exit`: an unexpected exit;
- `sanity`;
- `upload_lag`: more than 3 checkpoints queued;
- `upload_error`;
- `control`: an ignored `control.json`;
- `reference`: an ignored `reference.json`.

Only `nonfinite` and `trainer_exit` stop the run at the first branch.

### `evals/<tag>.json`

| tag | contents |
|---|---|
| `<segment>_quick_s<N>` | the trainer's `quick_eval` event: `policy_kl`, `value_mse`, `total`, top-1/5, … on the fixed 32,768-record frozen90 subset, raw weights |
| `<segment>_full_s<N>` | `{segment, samples, reason, sets: {frozen90, v2val_roots, v2val_derived: {raw: {metrics}, ema: {metrics}}}}`, at every stable checkpoint and branch end |
| `<segment>_memgap_s<N>` | `{…, sets: {seen_policy: {raw\|ema: {gap: {policy_kl, value_mse}, group_passes, valid, seen: {metrics}}}}}`. The gap is (seen − frozen90) / frozen90 on 200,000 fixed policy-group training roots; `valid` once the group finished a pass (≈ 104M samples) |
| `decision_<branch>` | `{branch, branch_index, policy_kl, total, go, reason, [kl_gain, kl_ok, total_ok, prev_total], utc}`, where `policy_kl` and `total` are the better of raw and EMA |
| `timecap_<branch>` | before each stable segment after the first: `{elapsed_h, rate_samples_per_s, need_samples, projected_h, time_cap_h, ok}` |
| `sanity_<source>_s<N>` | `{source, samples, ref_kl, ref_run, tol, quick_kl, rel_dev, pass, utc}`; `source` is `primary` (prod.yaml's A3 reference) or a `reference.json` source such as `C0` |

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

### `reference.json`

Written by the operator after launch: a secondary sanity reference, such as `C0`.

```
{"source": "C0", "ref_kl": <float > 0>, "samples": <int on the quick-val grid, e.g. 40960000>,
 "tol": <optional, default prod.sanity.tol>, "ref_run": <optional provenance string>}
```

It's polled with `control.json` and checked once per source. `primary` is reserved. An invalid file is ignored and logged as a `reference` anomaly.
