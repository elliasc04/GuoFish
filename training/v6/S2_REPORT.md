# v6 GPU smoke checks (G1–G3) and gate S2 — report

Branch `v6-harness`. Session 2026-09-25 20:50 → 2026-09-26 (local time, EDT; the logs
are UTC).

Tags: **[measured]** means I ran it and read the output. **[read]** means from code, logs
or manifests. **[inferred]** means my judgment.

Decisions: `training/v6/DECISIONS.md`, "Brief of 2026-09-25".

---

## Summary

| check | verdict | key number |
|---|---|---|
| G1: throughput, bf16 + compile | **PASS** | 6,978 samples/s, 109–112% of v5 |
| G3: micro-batch sweep | **PASS** | no OOM at 1,024 (7.1 GB) |
| G2: S3 resume on GPU | **FAILS the 1e-3 bar**, from nondeterminism | resume itself exact (`rng_check`: bit-identical) |
| **S2: compat reproduction** | **PASS** | `ref` inside the c1/c2 band on KL and MSE |
| `ref` cross-check vs `gates.score_baseline` | PASS at fp32; MSE misses at bf16 | the metric code agrees to 1e-9; the gap is the forward path |

**d, the screening noise band (O3):** measured from one pair of seeds (c1 = 20260802,
c2 = 20260925), same code, same data, 60M samples:

| metric | d = \|c1 − c2\| | relative |
|---|---:|---:|
| frozen90 policy KL | **0.00913** | **0.97%** |
| frozen90 value MSE | **0.00173** | **2.31%** |
| frozen90 top-1 | 0.00246 | 0.61% |
| total (KL + MSE) | 0.01086 | 1.07% |

A single pair gives a noisy estimate. For two draws, E|c1 − c2| ≈ 1.13σ, so the per-run
seed σ is ≈ 0.9 d. [inferred] An arm's difference from A0 smaller than d on these metrics
is inside the noise measured here.

All three runs finished on their first attempt: 0 restarts, 0 non-finite steps. [measured]

## G-0 pre-flight [measured]

- **GPU:** `nvidia-smi` showed no compute process, only desktop C+G clients, and the GPU
  was idle.
- **Frozen val v1:** all 12 shards (452,405 records) hash to their manifest.

## G1 — d384×6, bf16 + `torch.compile`, 500 steps: **PASS** [measured]

- **Result:** a median of **6,978 samples/s** over the `step` events after step 100 (range
  6,964–7,006). That is 109–112% of v5's 6,247–6,400, so above the 95% flag line.
- **Other bars:**
  - `loader_wait_frac` max 0.0021 (bar < 0.05);
  - `peak_vram_mib` 3,676;
  - `run_end`, with no `anomaly` event and 0 non-finite steps.
- **End-of-run full eval** (500 steps, 512k samples): KL 1.75401, MSE 0.21498.
- **Window:** 01:06:46–01:10:15 UTC. Track C was only editing code; no Track C process ran.
- **Found on the way (fixed, `5d87e7c`):** the first compile died with a `FileNotFoundError`
  in Triton's cache. A descriptive fused-kernel name was 135 characters, and the path
  reached 283 even under `%USERPROFILE%\ti`, which is v5's MAX_PATH fix. It cannot fit, and
  `LongPathsEnabled` is 0. `train.py` now sets
  `inductor_config.triton.descriptive_names = False`. That only renames kernels, and it
  covers `bench.py` and `s3_check.py` too, which import `train.py`.

## G3 — micro-batch sweep, d384×6: **PASS** (no OOM at 1,024) [measured]

| micro-batch | samples/s | step ms | peak VRAM MiB |
|---:|---:|---:|---:|
| 256 | 6,752 | 37.9 | 1,917 |
| 512 | 7,008 | 73.1 | 3,633 |
| 768 | 7,007 | 109.6 | 5,339 |
| 1,024 | 7,060 | 145.0 | 7,070 |

- **Window:** 01:10:28–~01:11:40 UTC. Track C was idle.
- **1,024 × 1 is only 0.75% faster than 512 here.** The bench steps the optimizer on every
  micro-batch, which slightly favours 1,024 over 512 × 2.
- **The shape config is left unset.** S2 must keep v5's 512 × 2, and the difference is
  within the bench's resolution. There is no d384×6 shape file to set it in. For
  non-compat runs, 512 × 2 halves peak VRAM (3.6 vs 7.1 GB) at ≤ 1% cost. [inferred]

## G2 — S3 on GPU (bf16 + compile): **FAILS its 1e-3 criterion**, from nondeterminism, not resume [measured]

`s3_check --steps 300 --kill-after 157 --ckpt-every 25 --tol 1e-3`, with EMA on and
dropout 0.1 (runs `s3a/s3b_20260925_211150`):

- **Verdict:** `"passed": false`, with a max relative per-step loss difference of
  **0.0545**. The stream digests (record indices and mirror flags) match on all 307
  compared steps.
- **The divergence comes before the kill.**
  - **Before the kill:** comparing run B's steps 1–157 with the uninterrupted run A, steps
    1–3 are identical and step 4 differs by 2.1e-6. The difference grows to **5.4%**
    (around peak LR) with no resume involved.
  - **After the resume:** the resumed segment (steps 151–300) peaks at **1.1%**.
- **Conclusion:** two uninterrupted bf16 + compile runs do not reproduce each other.
  Backward-pass atomics reorder sums, and the tiny difference is amplified. A per-step
  1e-3 bar cannot hold on this stack; on CPU fp32, S3 is bit-identical.
- **Follow-up that isolates resume, `tools/rng_check.py`: PASS.** It checks the one
  GPU-specific piece of resume state, the CUDA RNG under compiled dropout:
  - Process 1 saves weights and the torch + CUDA RNG state the way a checkpoint does, then
    runs a compiled, dropout-on bf16 forward (A1) and repeats it without restoring (A2).
  - A fresh process restores the state and runs the forward (B).
  - **B equals A1 bit for bit** (max |Δ| 0.0 over 2,097,664 outputs), while A2 differs by
    up to 1.047, so dropout is live. Resume on GPU is exact in everything it restores.
  - Run at 04:11:36 UTC, in the `ref`→`c1` gap.
- **Full evals of the two G2 runs** (raw weights): A KL 1.77856 / MSE 0.23115, B (resumed)
  1.77790 / 0.22960.
- **Recommendation:** replace G2's per-step tolerance with "resumed vs uninterrupted is no
  worse than uninterrupted vs uninterrupted", or run S3 under deterministic kernels. The
  current bar fails a correct implementation. [inferred]

## S2 — protocol, as run [read]

`training/v6/tools/s2.py` ran all three back to back from one driver, each wrapped in an
auto-resume loop (at most 2 restarts):

| run | command | seed | output |
|---|---|---|---|
| `ref` | `train_v5.py --config training/v5_multiPV/configs/corpus90m.yaml --epochs 1 --max-steps 58594 --no-h2h-gate --gap-probe 0 --seed 20260802 --out-dir models/v6/s2/ref --run-name ref` | 20260802 | `models/v6/s2/ref` |
| `c1` | `python -m training.v6.train --config training/v6/config/configs/v5_compat.yaml --set run.out_root=models/v6/s2 schedule.total_samples=60000256 ema.enabled=false system.allow_dirty=true run.name=c1 run.seed=20260802` | 20260802 | `models/v6/s2/c1` |
| `c2` | same as `c1`, with `run.name=c2 run.seed=20260925` | 20260925 | `models/v6/s2/c2` |

**Confirmed against the brief:**
- d384×6 on the 90M corpus.
- Exactly **58,594 optimizer steps**, i.e. 60,000,256 samples at 1,024.
- **OneCycle is compressed, not truncated.** `train_v5.py` caps OneCycleLR's `total_steps`
  at `--max-steps`, and v5's log confirms "total 58,594 (capped from 87,965)".
- v5 hyperparameters, EMA off, no h2h gate, no gap probe.

**Fixed from GPU_TODO's version:**
1. `ref` lacked `--gap-probe 0`, and `corpus90m.yaml` sets 200k.
2. `ref` wrote outside `models/v6/`.
3. `c2`'s seed was 20260803.

**The schedules agree exactly** with one torch OneCycleLR replay (§S2 table).

**One known compat difference: the policy normalizer.** v5 measures coverage on 200k
records (0.603120 this run); v6 uses the exact corpus coverage (0.602146), H15. That is a
0.16% difference in the soft-KL denominator.

## `ref` cross-check: v6 evaluator vs v5's `gates.score_baseline` [measured]

`ref`'s final checkpoint (`v5_10.9M_ep1.pt`, step 58,594) is loaded through the M1
converter for the v6 evaluator. v5's `score_baseline` loads it natively. Same 452,405
records.

| precision | v6 KL | v5 KL | rel | v6 MSE | v5 MSE | rel | within 1e-4? |
|---|---:|---:|---:|---:|---:|---:|---|
| bf16 autocast (the trainer's eval setting) | 0.9474877 | 0.9474576 | 3.2e-5 | 0.0753652 | 0.0753526 | **1.67e-4** | KL yes, **MSE no** |
| fp32, TF32 off | 0.9474642 | 0.9474646 | 3.9e-7 | 0.0753546 | 0.0753601 | 7.3e-5 | **yes, both** |

**Decomposition.** v5's own `probe_metrics`, run on the **v6 forward** (the converted
model) in fp32, gives KL 0.94746420 and MSE 0.07535464. v6's evaluator on the same forward
gives 0.94746420 and 0.07535464. They differ by **8e-10 and 3e-10 relative**.

- **The metric code is identical**, so every residual is the model path: v5's fused
  `TransformerEncoderLayer` kernel against v6's modules, on the same weights.
- **bf16 rounding moves value MSE by ~1e-4 relative on each side, in opposite
  directions.** That is the bf16 miss.
- **v5's own end-of-training full validation** reported total 1.02282. The v6 evaluator
  gives 1.02285 (bf16) and 1.02282 (fp32).

**Verdict:** the evaluators agree. The cross-check passes at fp32 and misses the value-MSE
bar by 1.67e-4 at bf16, for a reason unrelated to the evaluator. All three S2 runs are
scored the same way (v6 evaluator, v6 forward, bf16), so the forward-path offset does not
enter the S2 comparison. It is 130× smaller than d_MSE anyway.

## S2 results [measured]

Scoring (`python -m training.v6.tools.s2 analyze`, which calls `tools/score.py`):
- all three final checkpoints, raw weights;
- v6 evaluator, bf16 autocast, TF32, batch 1024;
- `frozen90` (v1): 452,405 records, 271,876 of them with policy;
- `ref` goes through the M1 converter.

Output: `models/v6/s2/analyze_official.json`.

| run | seed | steps | frozen90 KL | frozen90 MSE | top-1 | train KL + MSE, final 1,069,056 samples | wall clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ref` (v5) | 20260802 | 58,594 | **0.947488** | **0.075365** | 0.40140 | 1.004586 + 0.083414 = **1.088000** | 2 h 48 m |
| `c1` | 20260802 | 58,594 | 0.937419 | 0.073879 | 0.40356 | 0.993717 + 0.081378 = 1.075095 | 2 h 35 m |
| `c2` | 20260925 | 58,594 | 0.946552 | 0.075608 | 0.40110 | 1.003060 + 0.082539 = 1.085599 | 2 h 30 m |

**Pass rule:** `ref` must lie in [min(c1, c2) − d, max(c1, c2) + d].

| metric | band | ref | where |
|---|---|---:|---|
| policy KL | [0.928286, 0.955685] | 0.947488 | inside, 0.00094 above c2 |
| value MSE | [0.072150, 0.077337] | 0.075365 | inside, between c1 and c2 |
| top-1 (not gating) | [0.398634, 0.406027] | 0.401400 | inside, between c1 and c2 |

**S2 passes.**

- **Training loss.** The window is the last 1,044 steps, the last 21 v6 log rows. It is
  aggregated by count: KL per policy row, MSE per row. v5 uses its per-micro-batch rows, v6
  its per-interval `soft_kl` / `n_soft`. `ref`'s 1.0880 sits 0.0024 above c2, inside
  |c1 − c2| = 0.0105, the same picture as frozen90.
- **The trainer's own end-of-run eval** for c1/c2 equals `score.py` to 6 decimals.
- **Schedules.** Every logged LR and β1 of the three runs equals one torch OneCycleLR
  replay: max relative difference **0.0**, over 1,171 / 1,172 / 1,172 points.
- **The cross-check's forward-path offset is small against d.** It is 1.7e-4 relative on
  bf16 MSE, 130× smaller than d_MSE (2.3e-2), so it cannot move the verdict.
- **Other checks:** peak VRAM for c1/c2 was 3,679 MiB. There were no `anomaly` events.
  Provenance: c1 at `39a098b`, c2 at `1ecaafe`, both `allow_dirty` with the patch
  recorded; `ref` at `35115ea`.

**Found at the end, fixed (`s2.py`).** The first `analyze` (run by the `after_s2.py`
watcher, which is not mine) crashed on two bugs in my training-loss window:
- v5's `micro` rows carry 0-based step numbers.
- 58,594 is not a multiple of v6's 50-step log interval, so a 1,000-step window can't align
  with v6's log rows. It is now 1,044 steps.

The watcher's headless review diagnosed both and recomputed the table on CPU from the
trainers' own evals, with the same numbers. That review could not write `REVIEW.md` in
`models/v6/s2/` (its permissions); its copy is in its own scratchpad.

## Restarts [read]

`driver.jsonl`: `ref`, `c1` and `c2` each ran once (attempt 0) and exited 0:
- `ref` 10,128 s;
- `c1` 9,317 s;
- `c2` 9,029 s.

There were no restarts. The driver waited on a hold file between `ref` and `c1` for the
loader benchmark and `rng_check` (00:08:41–00:12:11 local), so the GPU was idle for 3.5 min.

## Throughput and contention [measured]

Median `samples_per_s` over the log intervals, split by whether other work overlapped:

| run | while other work ran | idle box | what overlapped |
|---|---:|---:|---|
| `ref` (v5) | 5,983 (1,045 intervals) | **6,333** (126) | corpus v2 build: 8 workers, below normal, 21:15–23:39 local; then S6/strata/counts to 23:51 |
| `c1` (v6) | 6,881 (195) | **6,986** (975) | 00:12–00:42 local: my frozen90 scoring on the same GPU (~9 min), then the 20-min CPU test regression |
| `c2` (v6) | none | **6,989** (1,170) | none |

- **v6 compat throughput is ~110% of v5's** on an idle box (6,989 vs 6,333), consistent with
  G1. Above the 95% flag.
- **The build cost `ref` ~5.5%** of throughput.
- **GPU-sharing dips.** `c1`'s intervals that overlapped the GPU scoring ran at ~3,000
  samples/s; loader wait stayed ≤ 0.3%, so this was GPU sharing, not data.
- **Memory.** `ref` held ~26 GB private: main 9.1 GB, 8 loader workers at 1.48 GB and 4
  persistent val workers at 1.34 GB. Available RAM sat at ~0.5–2 GB throughout, because its
  memory-mapped 34 GB corpus fills the page cache. Page writes were 0/s, so this was not
  swapping. The v6 trainer held ~13 GB (main 1.8 GB, workers 1.37 GB).
- **Session effect.** During `c1`, Claude Code stopped two of my idle background shell
  waits for low memory. The training runs were unaffected.
