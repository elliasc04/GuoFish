# v6 harness — GPU work, prepared, not run

None of this was run on the GPU. The capacity campaign owns it. `s3_check.py` and `bench.py`
were exercised on CPU with the tiny config, so their plumbing is known to work. Every command runs from the
repo root in PowerShell. `system.allow_dirty=true` is included because the tree currently
carries unrelated uncommitted files (`tools/capacity_suite.py`, `tools/capacity_campaign.py`,
…); drop it once the tree is clean. Timings for v5 come from the recon report (§1.11).

## G0 — prerequisite (CPU, after the GPU run finishes): 90M train strata

G1–G3 and S2 train on the 90M corpus, and the trainer requires its strata sidecar.
This pass reads all 34 GB of train shards sequentially, which is why it waits: it would
evict the running trainer's page cache. The output stays outside the live corpus
directory.

```powershell
$env:CUDA_VISIBLE_DEVICES=""
python -m training.v6.tools.build_strata --shards data/processed/multipv_90m --split train `
  --manifest data/multiPV/manifests/dataset_manifest_90m.json `
  --out data/processed/strata/multipv_90m_train_v1.npy
```

- **Expected:** a few minutes, and 90,076,117 codes.
- **Pass:** the JSON `counts.label.multipv` equals 54,239,523, the train policy count from the
  recon report (§0 item 1).

## G1 — d384×6, bf16 + `torch.compile`, 500 steps: throughput vs v5

```powershell
python -m training.v6.train --config training/v6/config/configs/v5_compat.yaml --set `
  run.name=g1_d384x6_500 schedule.total_samples=512000 eval.quick_every_samples=102400 `
  ckpt.every_samples=102400 system.allow_dirty=true
```

- **Expected:** about 10 min, covering the first compile (1–3 min on Windows), 500 × ~0.16 s,
  and one full eval of about 35 s.
- **Pass:**
  1. The run ends with `run_end` and no `anomaly` event.
  2. The median `samples_per_s` over `step` events after step 100 is at least 6,000, which is
     within ~5% of v5's 6,247–6,400 at the same shape, batch and precision. Below that is a
     finding to profile, not a pass.
  3. The same events show `loader_wait_frac` below 0.05.
  4. Record `peak_vram_mib`. v5's A1 smoke footprint was ~4.3 GB above baseline.

```powershell
python -c "import json,statistics as s; r=[json.loads(l) for l in open('models/v6/g1_d384x6_500/logs/train.jsonl')]; st=[x for x in r if x['event']=='step' and x['step']>100]; print('samples/s', s.median(x['samples_per_s'] for x in st), 'wait', max(x['loader_wait_frac'] for x in st), 'vram', max(x['peak_vram_mib'] for x in st))"
```

## G2 — S3 on GPU (bf16 + compile): **PASS** under the amended criterion (brief of 2026-09-26)

**Amended criterion.** The per-step 1e-3 bar below cannot hold on bf16 + compile, because
two *uninterrupted* runs already diverge (backward atomics). G2 now passes when both of these hold:
1. the resumed run's divergence from the uninterrupted run is no larger than the divergence
   between two uninterrupted runs;
2. `tools/rng_check.py`, which restores the CUDA RNG under compiled dropout, is bit-identical.

**Evidence** (S2_REPORT.md, G2 section), with stream digests identical on all 307 compared steps:
1. The resumed segment peaks at 1.1% relative per-step loss difference. The same run's
   uninterrupted first 157 steps had already diverged from run A by 5.4%.
2. `rng_check` was bit-identical: max |Δ| 0.0 over 2,097,664 outputs.

The original command and bar are kept below for reference.

```powershell
python -m training.v6.tools.s3_check --config training/v6/config/configs/v5_compat.yaml `
  --steps 300 --kill-after 157 --ckpt-every 25 --tol 1e-3 --set system.allow_dirty=true `
  ema.enabled=true ema.half_life_samples=1e6 eval.quick_every_samples=102400
```

- **Expected:** about 10 min: three trainer processes, each paying the compile, plus two
  end-of-run full evals of raw and EMA.
- **Pass:** printed `"passed": true`. That means every step's stream digest (record indices
  and mirror flags) is identical, and every step's loss is within 1e-3 relative of the
  uninterrupted run. Run B is killed after step 157 and resumed at step 150; dropout 0.1
  exercises the CUDA RNG restore.
- On CPU the same tool with `--tol 0` is bit-identical. `test_m5_trainer.py` also covers
  resuming at a group's pass boundary.

## G3 — micro-batch sweep for d384×6

```powershell
python -m training.v6.tools.bench --config training/v6/config/configs/v5_compat.yaml `
  --set system.allow_dirty=true --micro 256 512 768 1024 --warmup 15 --steps 40
```

- **Expected:** about 15 min, because each micro-batch size recompiles.
- **Pass:** one JSON line per size with `samples_per_s` and `peak_vram_mib`, and no OOM at
  1,024. Pick micro 512 × accum 2 or 1,024 × 1, whichever is faster, and set it in the shape
  config. The config decides this per §8.4.

## G4 — S7 engine smoke

**Unblocked (2026-09-27), contract A.** `playing/v6/evaluator.load_default_model` now loads
v6 exports through `core.guofish_net.load_for_inference`. v5 and legacy checkpoints keep
their loader; root `DECISIONS.md`, "S7", says why. The run is:

```powershell
python -m training.v6.tools.export <run>/ckpt/s<N>.pt --weights raw
python tools/s7_check.py <run>/export/<file>.pt      # GPU; writes runs/s7/<file>.json
```

It covers steps 1–3 below. Step 4 (contract B) is built; see "Contract B" below.

**Result on A7 (2026-09-28):**
- UCI and forward: pass. The forward is bit-exact.
- Numerics: 98.60% (493/500), against the 98.75% criterion.
- The owner accepted it; the ruling and the disagreements are in the root `DECISIONS.md`.
- A9 (`canonical_65`) was adopted, so the final recipe is a contract-B net. Step 4 now blocks the
  confirmation match.

**Contract B (2026-09-29):** the C++ `canonical_65` tokenizer, policy remap and value sign exist,
selected by the export's declared contract. B1–B3 pass (`tests/test_s7_contract_b.py`; B2 needs
the GPU, ~4 min beside a training run). What remains is `tools/s7_check.py` on the final
contract-B export, in a HOLD gap: UCI, forward and the ≥ 98.75% numerics, as for A7. Root
`DECISIONS.md`, "S7 contract B", has the details.

The original plan:

1. `python -m training.v6.tools.export <ckpt> --weights ema`. This already smoke-tests the
   export through `load_for_inference` on 64 fixed positions.
2. The UCI handshake, capture up to `max_batch` 128, and `go nodes 800` on 20 positions.
3. Contract-A numerics: compiled and captured forward vs eager, ≥ 98.75% move agreement on
   `golden/c10_corpus.json`.
4. Contract B additionally needs B1–B3 (§11.3), which need the C++ `canonical_65` path.

## Not scaffolding: S2 (Phase 1 stack validation, ~8.5 h)

Listed only. This is validation and needs a decision on budget before it runs. It
compares two `v5_compat` seeds against `train_v5.py` at the same 60M-sample proxy
budget (58,594 steps × 1,024) on `frozen90` policy KL and value MSE.

```powershell
python -m training.v6.train --config training/v6/config/configs/v5_compat.yaml --set `
  run.name=s2_compat_seed1 schedule.total_samples=60000256 system.allow_dirty=true
python -m training.v6.train --config training/v6/config/configs/v5_compat.yaml --set `
  run.name=s2_compat_seed2 run.seed=20260803 schedule.total_samples=60000256 system.allow_dirty=true
python -u training/v5_multiPV/train_v5.py --config training/v5_multiPV/configs/corpus90m.yaml `
  --epochs 1 --max-steps 58594 --no-h2h-gate --out-dir models/s2_v5_60M --run-name s2_v5_60M
```

- **Pass:** the v5 run lands within the two compat seeds' noise band on both metrics (§12).
- Score all three on `frozen90`. v6 does that itself at run end. For v5, use
  `gates.score_baseline`, which the capacity campaign already uses (`a5-val`).
