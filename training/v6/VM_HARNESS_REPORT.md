# VM harness report (2026-09-30)

Brief: `docs/capacity/training/prod/vm_harness_brief.md`. Branch: `v6-prod`, from `v6-harness`, not pushed. Decisions: `DECISIONS.md`, "VM harness". Operator steps and schemas: `PROD_RUNBOOK.md`.

Claims are tagged **[measured]**, **[read]** or **[inferred]**.

## Summary

- **What exists:**
  - the production driver, `tools/prod.py`;
  - the store layer, `r2.py`;
  - `setup.sh` and `requirements-prod.txt`;
  - `configs/prod.yaml`, built on a byte-for-byte copy of A3's config;
  - the runbook;
  - a CPU dry run that covers every stop path, both control actions, kill + wipe + resume, and sidecar refusal.
- **The full `training/v6` suite passes:**
  - locally on Windows [measured];
  - in an Ubuntu 26.04 container, CPU only, set up by the real `setup.sh` from a clean `git pull` of `v6-prod` [measured]. The exception is one float64 parity test that needs more than the Docker VM's 14 GB (§1).
  - Getting there, the container run found and fixed one real driver race and four Windows-only assumptions.
- **Not done yet:**
  1. **The dry run against R2 itself.** The session couldn't get S3 credentials. The dry run used a directory store (`PROD_STORE=file://…`) through the same code paths, except boto3's calls. See open question 1.
  2. **A GPU smoke of `prod.yaml`.** No local GPU per the ground rules, and corpus v3 doesn't exist yet; `setup.sh` runs it on the VM before launch. The new trainer code has already run on a GPU: Ct resumed on it overnight (§5).
- **The inputs from the corpus v3 brief:** corpus v3 in R2, the pool decision and C0's 40M quick-val KL. `prod.yaml` carries the production pool as placeholder, and `prod.sanity.ref_kl` is null, which the driver refuses at launch.

## 1. Linux port and test results

**Changes:**
- **PowerShell launchers → bash:** `setup.sh`, plus tmux in the runbook. The stale untracked `prod.sh` (L40S, 90M pool) is removed.
- **`Start-Process`:** no training/v6 code launched with it. `tools/screen.py` mentions it only in a docstring, and it is a local-box tool [read].
- **Priority:** the VM trainer runs at normal priority; it's the only job. There was no priority code to port [read].
- **Triton:** torch 2.8.0's own `triton==3.4.0` on Linux [measured: uv resolves it]. Inductor `descriptive_names=False` is kept.
- **Paths:** already `pathlib` throughout.

**What the container run found** [measured]:
- **`test_m2_config`** passed Windows' `SYSTEMROOT` to a subprocess, a `KeyError` on Linux. It now passes the full environment with the seed override.
- **`test_refusals`** assumed a dirty working tree. On a clean checkout, which is what the VM has, the dirty-tree refusal never fired. The test now adds an untracked probe file.
- **`test_m7`'s BLAS probe** read `psutil`'s Windows-only `private` field; it uses USS elsewhere. The 100 MiB drop H2 measured is a Windows commit-charge effect: Windows spawns workers, while Linux forks them (27 MiB USS, shared pages) and maps OpenBLAS's buffers lazily. So the threshold applies on Windows only; batch identity is checked everywhere.
- **A real race in `prod.py`:** a branch fetched its stable checkpoint from the store while that checkpoint's upload could still be queued, and was refused ("sidecar is missing"). Windows won the race every time; the container's slower IO lost it. `fetch()` now drains the upload queue first.
- **Only on the CPU torch wheel:** `test_cpu_checkpoint_never_initialises_cuda` patches `torch._C._cuda_init`, which that build lacks. `setup.sh --cpu` now installs the VM's cu129 wheels, which run without a GPU.

**Results:**
- **Windows, local:** `training/v6/tests`: 95 passed (dry run deselected, 28 min 50 s), and the 7 dry-run tests passed (5 min 24 s) [measured].
- **Ubuntu 26.04 container (Docker, CPU only), first full run** (`8b8c7cd`, CPU wheel): 83 passed, 9 failed, 6 errors, 4 skipped. The errors were a missing v5 checkpoint file. The failures are the five issues above, plus four dry-run tests that hit the race [measured].
- **After the fixes** (`cc4021e` and `771b6c1`), every test that had failed now passes in the container [measured]: the 7 dry-run scenarios (the kill/resume one bit-identical over 65 steps), `test_refusals`, the two eval-set tests and the BLAS probe. The reference run's decisions match Windows to the digit (KL gain +1.541%).
- **Final full run** (`771b6c1`, cu129 wheels, 14 GB cap: the Docker VM's whole budget): **99 passed, 1 skipped, 1 failed, 1 deselected** [measured].
  - **Skipped:** the A3 byte-copy check, since `runs/` is local to the screening box.
  - **Failed:** `test_dryrun_sanity_pause_then_stop_now`, whose pause landed on 6,400, not 5,120. The toy trainer writes a checkpoint every ~0.5 s against the driver's 1 s poll, so two checkpoints can show up in one poll; at production scale that is one checkpoint every ~17 min. The test now asserts the pause and that the run stopped (`8302ce2`), and passes in the container [measured].
  - **Deselected:** `test_m1_model.py::test_v5_parity_float64`, which the kernel OOM-killed at 14 GB. It passes on the 32 GB Windows box, and the VM has 125 GB [measured]. The other five `test_m1` tests, including both v5 parity paths (max |Δ| 6.2e-6), pass on Linux [measured].
  - **Net:** every test passes on Linux except that one, which is too big for this Docker VM.
- **The container's data:** frozen90 v1/v2 and the 90M corpus were copied in, because two `test_m3` reader-parity tests read 50k random 90M records. The synthetic dry-run set was pulled by `setup.sh`.
- **The container's `setup.sh --cpu` run** [measured]:
  - GPU, RAM (15 GB) and `/dev/shm` (0 GB) failed their checks, as warnings under `--cpu`;
  - it created the Python 3.13.7 venv and installed the pins below: torch `2.8.0+cpu` on the first pass, `2.8.0+cu129` after `--cpu` switched to the VM's wheels;
  - the pull verified all 12 files;
  - 29 fast tests passed;
  - the smoke ran both splits and wrote `smoke.json`.

## 2. The A3-equality assertion

- `configs/a3_resolved.yaml` is `runs/screening/configs/A3.yaml` copied with `cp`. Its sha256 `c369fe1e…` is the same as the ledger config and `models/v6/screen/A3/config.resolved.yaml` [measured].
- `prod.yaml` extends it.
- **`test_prod_equals_a3_except_the_overrides`** asserts that the set of keys where prod differs from A3 is exactly these, no more and no fewer [measured, passes]:
  - `run.name`, `run.out_root`;
  - `data.{corpus,manifest,strata,workers}`;
  - `mixture.groups`;
  - `schedule.{total_samples,warmup_samples,decay_frac}`;
  - `eval.{quick_every_samples,extra_sets}`;
  - `ckpt.{keep_last,stable_every_samples}`;
  - `system.allow_dirty`.
- **Seeds** are 20260802 for data and init, the same as A3's, so they aren't in the diff.
- **The micro-batch** default is A3's 512×2; 1024×1 is a launch-time `--set`.
- `test_a3_resolved_is_the_screening_config` checks the byte copy while `runs/` exists; it skips on the VM.

## 3. Dry run

**Setup:** `configs/dryrun_cpu.yaml` is `prod.yaml` at d64×2, with the branch grid scaled down 39,063×:
- stable checkpoints every 3,840 samples;
- branches at 7,680 / 15,360 / 30,720 / 61,440;
- rolling checkpoints and quick-val every 1,280;
- the sanity check at 5,120.

The data is `tools/make_synth.py`'s 16,384 train and 2,048 val records carved from frozen90 v2. Each scenario is a real `prod.py` process (`tests/test_prod.py`, `-k dryrun`) [measured, local and container]:

| scenario | result |
|---|---|
| reference (continue) | continued after branch 1 (always) and branch 2 (KL +1.54%, total improved), then stopped after branch 3 on **worse total** (the 16k-record set overfits). Shipped branch 2's EMA weights (best total) and exported them as contract B; the export's 64-position smoke test passed. |
| insufficient gain | `min_kl_gain=0.99`: stopped after branch 2, "insufficient gain" |
| worse total | value-negated corpus, value loss ×10, policy ×0.1: MSE 0.378 → 0.494, stopped after branch 2 on exactly "worse total", KL rule passing |
| time cap | cap 1e-6 h: branch 1 done, then `timecap_*.json` projected over the cap and the run stopped "time cap", shipping branch 1 |
| sanity pause | `ref_kl` 1.5× the true value: at the 5,120 check the run paused itself via `control.json` (`by: prod.py`, reason recorded) and halted on the first checkpoint after the check. That was 5,120, or 6,400 when the toy trainer's next checkpoint fell in the same poll. It logged a `sanity` anomaly, and `stop_now` then ended it with nothing to ship |
| pause / continue / stop_after_branch | `pause` stopped the trainer at its next rolling checkpoint (no progress in the next 3 s); `continue` resumed it; `stop_after_branch` finished branch 1 and stopped, shipping it. The replayed steps equal the reference's |
| kill + wipe + resume | prod.py and its trainer were killed mid segment 2, and the local run dir and staging were deleted. `--resume` refused a tampered sidecar ("sidecar … != state.json") and a missing one ("sidecar is missing"). With the sidecar restored it resumed from the store alone and finished. **All 65 logged steps across 4 segments are bit-identical to the reference in `step_losses` and `stream_sha`, and so are the decisions' reasons and totals.** |

The reference run also checked the store layout:
- rolling checkpoints: 2 kept, with sidecars;
- stable checkpoints: every one kept;
- present: `provenance.json`, `code.patch`, `config.resolved.yaml`, `summary.json`, the export and its sidecar, the decision and memorization-gap evals, and every documented `status.json` key;
- the memorization gap's `valid` equals `group_passes ≥ 1`: false at branch 1, true by the last.

The store's metrics are complete up to every checkpoint: the driver flushes them before queuing one. So the union of the metrics segments after a resume has every step, with replays identical [measured].

## 4. Pinned versions

From `requirements-prod.txt`. The local env is Python 3.13.7; the container got these exact versions [measured]:
- **Brief's list:** torch 2.8.0 (cu129 / cu128 / cu126 by driver; cu129 for `--cpu`), numpy 2.3.3, chess (python-chess) 1.11.2, PyYAML 6.0.3, zstandard 0.25.0, pytest 9.1.1, boto3 1.43.105.
- **Added:** psutil 7.1.0, which `test_m7` imports.
- **Transitive:** botocore 1.43.105, s3transfer 0.19.2, jmespath 1.1.0, python-dateutil 2.9.0.post0, six 1.17.0, urllib3 2.5.0, iniconfig 2.3.0, packaging 25.0, pluggy 1.6.0, Pygments 2.19.2, filelock 3.20.0, fsspec 2025.9.0, Jinja2 3.1.6, MarkupSafe 3.0.3, mpmath 1.3.0, networkx 3.5, sympy 1.14.0, typing_extensions 4.15.0, setuptools 80.9.0.
- **Tooling:** uv 0.10.12.
- **Resolved but not installed (no GPU):** `--torch-backend cu129` gives `torch 2.8.0+cu129`, `triton 3.4.0`, `nvidia-cublas-cu12 12.9.1.4`, `nvidia-cudnn-cu12 9.10.2.21`, `nvidia-nccl-cu12 2.27.3`; cu126 gives the 12.6 set [measured: `uv pip install --dry-run` in an empty 3.13.7 venv].

## 5. Optional speed-ups

**None built.** Their triggers are loader wait > 5% and optimizer > 15% of step time. The brief ties both to the VM smoke, which prints them. The runbook's go/no-go says what to do if either fires.

**An early reading.** Ct, the confirmation arm, resumed tonight on this branch's `train.py`: the same d384×10 A3 recipe at 512×2, on the local RTX 5070. Its new step-time split reads [measured, steady intervals from step 10,100]:
- fwd/bwd 79 % and **optimizer 21 %**;
- loader wait 0.1 %;
- about 3,450 samples/s.

On this card the batched-Muon trigger (> 15 %) already fires. Newton–Schulz on 384-wide matrices is small, launch-bound matmuls, which an A100 speeds up less than it speeds up fwd/bwd, so the fraction should be no lower there [inferred]. Recommendation: build batched Muon (with its 1e-6 gate) before renting, rather than finding out from the smoke on a billed VM. The loader has plenty of headroom.

## 6. Deviations

1. **Step-aligned grid** (details in DECISIONS.md):
   - branch points 300,001,280 / 600,002,560 / 1,200,005,120 / 2,400,010,240;
   - stable checkpoints every 150,000,640;
   - rolling checkpoints and quick-val every 10,240,000;
   - the sanity check at 40,960,000.

   The brief's round numbers aren't multiples of the 1,024 batch.
2. **The memorization gap is a trainer flag (`--seen-eval`), not a config field.** A schema field would change every existing config's hash and plain form, and old checkpoints (Ct's) would be refused on resume.
3. **Store layout additions:** branch rolling checkpoints (`branches/<name>/rolling/`, last 2), so a lost VM resumes inside a 12 h decay; `provenance_resume_*` / `code_resume_*`; `evals/timecap_*` and `evals/sanity_*`.
4. **The dry run used a directory store, not R2 `runs/_test/`** (credentials), and synthetic shards, not corpus v2. That covers everything except boto3 talking to R2.
5. **`setup.sh --cpu`**, a flag the brief doesn't list, makes the dry run possible on a CPU box. There's also `--set` for the smoke on non-default data.
6. **`prod.sanity.ref_kl` is supplied at launch** (`--set`) or committed. The driver refuses a null.
7. **The step-time split, in seconds per interval:**
   - `data_wait` and `eval` are host time; `eval` also covers checkpoint writes;
   - `h2d`, `fwd_bwd` and `optim` are GPU-stream time from CUDA events, which are read at the existing 50-step log sync, so there is no per-step host sync;
   - first run on a GPU by Ct (§5): the split is sensible, and Ct's throughput matches A3's.

## 7. Open questions

1. **R2 S3 credentials.** The token supplied is a Cloudflare API token (`cfut_…`). It lists the account's buckets, and the one bucket is `guofishv6corpus` [measured]. The S3 API needs an Access Key ID / Secret Access Key pair. I stopped short of deriving that pair from the token, since that's the owner's credential call. With the pair in `.env` (runbook §2), the dry run runs against R2 as `PROD_DRYRUN_STORE=r2 pytest training/v6/tests/test_prod.py -k dryrun`, under `runs/_test/<random>/`.
2. **`t20` strata and the definition hash.** The corpus brief adds `t20` to `depth_tier` and bumps the strata definition. The definition lives in `training/v6/data/strata.py` and `config/schema.py`, which are harness-owned. Any change to it changes `DEFINITION_HASH`, and `load_strata` then refuses **every** existing sidecar: frozen90 v2's (used by Ct and by prod's evals) and the v2 eval sets'. It needs one coordinated change: rebuild those sidecars, or version the check. I haven't made it.
3. **The `sha256.txt` contract with the corpus brief.** Paths are relative to `data/` (key `data/<path>`, local `data/<path>`). The upload must include the eval-set sidecars (`processed/evalsets/*.json`), `multipv_v2`'s manifest and its `val_*` / `valderived_*` shards, and their strata (runbook §3). Otherwise the smoke fails at start.
4. **Corpus v3's manifest must record `value_scale`.** The export copies it, and the engine refuses an export without one.
5. **The briefs disagree.** `training_brief.md` says it supersedes this brief, and the corpus brief calls it the master. I followed `vm_harness_brief.md`, as the owner said.
6. **Ct: done, and confirmed.**
   - Requeued at the owner's request at 04:28 UTC, it resumed from `s10240000` on this branch's `train.py` and finished at 08:43 UTC.
   - The ledger says `confirmation.confirmed: true`:

     | metric | seed s | seed t |
     |---|---:|---:|
     | KL delta vs A0t | 21.31 % | 21.64 % |
     | MSE delta vs A0t | 37.10 % | 37.31 % |

     [read, `runs/screening/ledger.jsonl`]
   - **The production launch condition is met.**
   - The replayed steps 10,050–12,100 matched the pre-crash log's data digests in all 6 intervals, with loss differences ≤ 3.4e-3 from GPU nondeterminism [measured].

## 8. Contract B

**The shipped model is a contract-B export (`canonical_65` input). The engine's contract-B support must land, and pass `tools/s7_check.py` on the final export, before the shipped model can play.** That work is separate from this harness.
