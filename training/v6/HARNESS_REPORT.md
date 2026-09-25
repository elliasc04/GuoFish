# v6 training harness — report

Branch `v6-harness`, from `509dda0`. Session 2026-09-24 22:xx → 2026-09-25 local.
Tags: **[measured]** means I ran it and read the output; **[read]** means from code, logs or
manifests; **[inferred]** means my judgment.
Design doc: `docs/capacity/training_stack.md` ("the doc"). Deviations: `training/v6/DECISIONS.md`.

---

## ⚠ Read first

1. **The capacity campaign is BLOCKED and the GPU has been idle since 23:16.**
   `d384x10` died at step ~24,850 of 87,965 with `torch.AcceleratorError: CUDA error: out of
   memory` inside `loss.backward()` (`models/capacity_brief/d384x10/logs/d384x10.log`,
   last write 23:14:49). The campaign logged `BLOCKED: training d384x10 exited 3221226505 …
   (re-running resumes from its latest checkpoint)` at 23:16:00
   (`runs/capacity_campaign/chain2.out`). The last rolling checkpoint is step 20,000. I did not
   touch the campaign, its files, or the GPU. **Restarting it is yours.** [read]
2. **My GPU hiding was ineffective for most of the session.** In PowerShell 5.1,
   `$env:CUDA_VISIBLE_DEVICES = ""` deletes the variable, so processes I launched from
   PowerShell had the GPU visible. That breaks ground rule 1. [measured: `Test-Path env:X` is
   False after an empty assignment] What that did and did not do:
   - Only one v6 code path initialises CUDA: the checkpoint writer, via
     `torch.cuda.get_rng_state_all()` "if available". It ran with the GPU visible once, in the manual
     tiny run at 23:49–23:56. That was **after** the crash, and it created a CUDA context on
     an idle GPU. [measured timestamps; read code]
   - **The crash window.** My M1 test run started at 23:14:43, six seconds before the trainer
     died. Re-running that suite with CUDA hidden, but with `is_available()` forced true and
     CUDA initialisation replaced by a recorder, records **0 CUDA-init attempts**. A positive
     control on the checkpoint path records 1. So it took no GPU memory. [measured]
   - **What I cannot exclude:** host-RAM pressure. The box had ~2.1 GB free during the
     campaign [measured at session start], and that test process was loading torch and a
     130 MB checkpoint. Under WDDM, system-memory pressure can surface as a CUDA OOM. The error
     type fits either that or an external allocation: it is `AcceleratorError`, not the
     allocator's `OutOfMemoryError`. The open PCIe Gen5-on-Gen4 fault on this host is another
     candidate. [inferred]
   - **Fixed:**
     - CUDA RNG is saved only when `system.device == "cuda"`, with a regression test.
     - `training/v6/tests/conftest.py` sets `CUDA_VISIBLE_DEVICES=-1` for every test and its
       subprocesses, and fails the session if CUDA was initialised.
     - My launcher uses `-1`.

## Milestones

| | Status | Commit |
|---|---|---|
| M0 freeze val | **done**: 12 shards, 452,405 records, sha256 manifest; every copy verified against its source with a second hasher (`Get-FileHash`) [measured] | `b12fe76` |
| M1 model package | **done**: all gates pass [measured] | `97b3b34` |
| M2 config system | **done**: 21 tests; `v5_compat.yaml` checked field by field against v5's `corpus90m.yaml` and manifest [measured] | `b84307a` |
| M3 data layer | **done**: S1, S4, S5 pass; strata built on frozen val and synthetic shards only [measured] | `6291185` |
| M4 losses/optim/schedule/EMA | **done** [measured]; `branch_decay.py` landed with M5 (it needs the trainer) | `ea6434d` |
| M5 trainer/eval/ckpt/export | **done**: tiny run, S3 on CPU, export round trip, branch decay, refusals, NaN guard [measured] | `5411db9` |
| M6 corpus v2 builder | **done** (lower priority): synthetic S9/S6, smoke `--limit 200000`, depth-24 index check, partial real-data S6 [measured]. Full build **not** run, as instructed | `0d1e117` |
| GPU items | **prepared, not run**: `training/v6/GPU_TODO.md` | report commit |

## Gates

| Gate | Result | Evidence |
|---|---|---|
| **S1** reader parity | **PASS** [measured] | 100,000 seeded records (50k frozen val, 50k 90M train by random reads). 60,091 had policy. Mirror off and on (50,093 mirrored rows). Tokens, dense targets, legal masks, values, `value_cp`, `pv_idx`, `n_pv` and `n_legal` are bit-identical to `MultiPVDataset` + `color_mirror`. S1 first **failed**: 1-ulp target differences on 30 of 2,000 records, because v5's ε spread adds in float64 (a NumPy `add.at` detail). Fixed to match. `test_m3_data.py::test_s1_reader_parity_with_v5` |
| **S3** resume, CPU fp32 | **PASS, bit-identical** [measured] | 200-step run with dropout 0.1, grouped mixture, quick evals in between. Killed (`os._exit`, no checkpoint) after steps 57 and 117; resumed at 55 (mid-pass) and at 115, where the 4,485-record group had consumed exactly one pass (the pass boundary). All 204 resumed-run steps match: max \|Δloss\| = 0.0. Stream digests (record indices and mirror flags), final weights, EMA and full-eval metrics are identical. `test_m5_trainer.py::test_s3_kill_and_resume_is_bit_identical` |
| **S4** sampler shares | **PASS** [measured] | 10,000 micro-batches of 512, groups 0.65/0.20/rest 0.15 on frozen-val strata. Every batch has exact allocated counts (floor or ceiling of share × 512). Cumulative counts stay within 1 sample of target at every batch; final shares are 0.65/0.20/0.15. Passes reached: 12, 17 and 6. Every record's own strata match the group it was drawn for. Each pass is a permutation of its group. The stream is identical for a fresh sampler, with 2 worker processes, and when resumed at the batch where a group wraps |
| **S5** canonical transform | **PASS** [measured] | 1,000,000 v1 records (all frozen val plus 547,595 train, read in contiguous blocks), 492,458 of them Black to move. canonical(x) = canonical(mirror(x)) on tokens, targets, legal mask, value, `value_cp` and `hard_move`. Mirror round-trips exactly. The value sign is side-to-move. The `hard_move` remap was exercised on 856,934 synthetic hard moves. On 100,000 records the worker tokens equal the python-chess reference on the FEN decoded from the tokens. Real data had 144 ep-file rows, all with a legal capture, so the illegal-ep branches are pinned with 5 hand positions, including two pins |
| **S8** bias wiring | **PASS, exact** [measured] | The deployed v5 weights were loaded into static-bias and smolgen models with zeroed bias output. Both reproduce the plain model with max \|Δ\| = **0.0**. Making the bias live moves policy logits by 2.46 (static) and 7.34 (smolgen). Found and fixed on the way: CPU SDPA takes a different kernel for a 3-D or grad-requiring mask (5.2e-6 drift) |
| **S6** frozen val survives rebuild | **PASS on synthetic data; partial PASS on real data** [measured] | Synthetic dump: all 205 frozen records re-materialised with shared fields byte-identical, and a single changed byte is caught. Real smoke: all **460** frozen records from the first 200k dump lines, rebuilt by `pass_b_v2.py`, are byte-identical in every shared field (`pv_score` via float16) to the v1 frozen shards, which v1 Pass B wrote. The full S6 (452,405) needs the full build |
| **S9** corpus v2 build | **PASS on synthetic data; smoke consistent** [measured] | Synthetic: nesting drops 0 lines; hard_move on 99.67% of roots; planted failures counted by reason; derived hard moves legal, depths = root − k, values = the root's (mates shortened); no derived record duplicates a root or another derived record; a second run into the same directory is refused. Real smoke: nesting drops 0 (scope: first 200k rows); hard_move on 100.00%; rejection and dedup counts in the manifest |
| S2, S7 | not run | S2 is validation (~8.5 GPU-h), listed in GPU_TODO. S7 is blocked on the engine loader change |

Other checks [measured]:
- **OneCycle:** the compat schedule equals `torch.optim.lr_scheduler.OneCycleLR` at all
  **351,858** steps, LR and β1 both with a max difference of exactly 0.
- **WSD:** matches hand-computed points for all three decay shapes.
- **EMA:** a 10M-sample half-life at batch 1,024 gives **0.99992902** per step (≈ 0.99993),
  and a 100-step half-life moves exactly halfway in 100 steps.
- **Param groups:** v5's split is reproduced exactly (29 tensors / 10,830,336 params
  decayed; 55 / 57,345 not).
- **Losses:** soft KL is bit-identical to v5's `policy_kl_per_sample`. Two micro-batches of
  256 vs one of 512 differ by 7e-7 relative in the gradient.
- **Tiny run:** 300 steps (d64×2, fp32, frozen val as train data) in about 2,340 samples/s.
  Mean loss over the first 30 steps is 2.2095 and over the last 30 is 2.1312. It covers
  stable and rolling checkpoints, quick and full evals (raw and EMA), `best.pt`,
  provenance, `code.patch`, export and a branch decay.
- **Quick-val subset:** 32,768 records, stratified over 64 strata cells. Policy share is
  0.6010, the same as full val; the old source-order prefix gave 0.772 (H3).

## M1 parity numbers [measured]

Deployed `models/guofish5_90M/v5_10.9M_best.pt` through the key-map converter. 1,000 seeded
frozen-val records, CPU, eval mode, compared with `training/v5_multiPV/model_v5.py`:

| Compared against | policy max \|Δ\| | value max \|Δ\| |
|---|---|---|
| v5 fused fast path (`no_grad`), fp32 | 7.868e-06 | 9.909e-07 |
| v5 plain modules (grad on), fp32 | 9.060e-06 | 7.451e-07 |
| float64, relative (v6 casts outputs to fp32) | 5.96e-08 | 1.01e-07 (fp32 ulp 1.19e-07) |

- v5-compatible config: exactly **10,887,681** parameters.
- The fp32 residual is kernel rounding, not a mapping error: in float64, v6's output is
  v5's output rounded to fp32. Its |logit| max is 6.76.
- The fp32 policy margin against the 1e-5 bar is only ~10%.

## Value-depth-24 index check (§6.4) [measured]

- **Run:** 51 s, one process, below-normal priority.
- **Self-check passed.** The reproduced 90M selection gives 91,350,634 lines,
  54,832,831 policy and 36,517,803 value-only, and 60,692,417 policy-eligible rows at
  depth 26. All four equal the manifest.
- **New policy-eligible rows** (24 ≤ max_depth < 26, policy_depth ≥ 20): **22,920,896**, plus
  29,735,457 new value-only rows. Total policy-eligible at depth 24 is 83,613,313, matching
  `feasibility_scan.md`.

| decile | ≤5 | 6–14 | 15–27 | ≥28 | total |
|---:|---:|---:|---:|---:|---:|
| 0 | 21,739 | 274,537 | 1,122,518 | 893,151 | 2,311,945 |
| 1 | 19,361 | 298,913 | 1,235,436 | 782,977 | 2,336,687 |
| 2 | 18,334 | 309,633 | 1,304,201 | 721,662 | 2,353,830 |
| 3 | 17,941 | 297,074 | 1,047,149 | 473,285 | 1,835,449 |
| 4 | 25,075 | 376,335 | 1,296,707 | 595,968 | 2,294,085 |
| 5 | 25,145 | 435,750 | 1,643,726 | 716,571 | 2,821,192 |
| 6 | 28,076 | 467,946 | 1,653,970 | 672,874 | 2,822,866 |
| 7 | 28,947 | 433,924 | 1,508,166 | 642,910 | 2,613,947 |
| 8 | 22,842 | 298,535 | 1,219,621 | 597,273 | 2,138,271 |
| 9 | 36,588 | 290,388 | 775,049 | 290,599 | 1,392,624 |
| **share** | **1.1%** | **15.2%** | **55.9%** | **27.9%** | 22,920,896 |

**≤5-piece rows do not dominate** (1.1%), so the doc's cap is not triggered. Source-order
deciles are roughly flat (6.1%–12.3% each). Unlike the depth-26 headroom (78.5% ≤5), the new
depth-24 rows are middlegame-heavy. [measured; interpretation inferred]

## Corpus v2 smoke (`--limit 200000`) [measured]

- **Run:** 25 s, 3 workers.
- **Selection:** 108,078 lines selected, nesting dropping 0.
- **Roots:** 107,674 (522 val), with `hard_move` on 100%.
- **Derived candidates:** 106,686 at k=1 and 106,195 at k=2. At rate 0.125, 26,487 were
  sampled. Dedup dropped 8,235 as duplicates of a root (31%) and 262 as duplicates of an
  earlier derived record, leaving 17,905 train and 85 valderived records.
- **Rejections:** 372 Chess960, 8 invalid FEN, 24 invariant violations (0.022%).
- **Strata over the smoke's train split:** 17,905 derived, 32,608 `hard_only`, 0
  `value_only`.

The first 200k lines are opening-heavy (57% ≥28 pieces), so none of this is a statistic
about the full build.

## Deviations (details and reasons in `DECISIONS.md`)

**M1**
- `init: v5_deepcopy` matches v5's distribution, not its RNG stream.
- The static bias goes to SDPA as a 4-D tensor, detached when grad is off (the S8 fix).
- `aux_policy_head` is a `ModelConfig` field, cross-checked against `targets.policy_hard.head`.

**M2**
- Paths are checked at run start, not at config load.
- The resume whitelist adds `data.workers` and `data.prefetch_factor`.

**M3**
- Mixture groups must be disjoint. The doc's §6.7 example overlaps and would be refused.
- Within-group order is a Feistel permutation, not a stored array.
- Per-micro-batch counts use cumulative largest remainder.
- canonical_65 reads ep legality from the stored legal moves.

**M4**
- Per-term normalizers are the expected per-window counts, computed from config + strata.
- Hard-target ε is spread over legal moves (the doc's wording), not over all 4,096 as in v4.

**M5**
- The non-finite guard is a device-side skip, detected at the next log sync.
- The torch RNG is checkpointed, and each DataLoader gets its own generator.
- Export names add `_s<samples>_<weights>` (D-7).
- `lr_range.py` and the v2 eval sets are not built.

**M6**
- Derived sampling is independent per ply.
- No resume.
- `--limit` selects over the first N index rows.
- Exit code 3 on the S9 coverage fail.

## Open questions for the operator

1. **Restart the capacity campaign.** Its log says a re-run resumes from the latest
   checkpoint (step 20,000). Consider whether ~2 GB of free RAM during those runs is safe
   for anything else running alongside.
2. **Mixture example (§6.7).** Should `hard` be `{label: hard_only, origin: root}` so the
   groups are disjoint, or should overlaps have a precedence rule instead of being an error?
3. **Normalizer semantics.** Expected counts from config + strata (v6 now) versus realised
   per-window counts. They are equal for label-aligned groups and differ slightly under
   `natural`.
4. **Hard-label smoothing.** ε over legal moves (doc) or over all 4,096 (v4 Phase 3)?
5. **90M train strata (G0).** The output is `data/processed/strata/multipv_90m_train_v1.npy`,
   outside the live corpus directory. Is that the right place? It is not built yet.
6. **Corpus v2 sizing.** The defaults are `--target 121.1M`, `policy_share 0.65` and
   `--derived-rate 0.125`. From the smoke, derived records ≈ roots × 1.98 × rate × 0.68,
   so 30M derived at 120M roots needs a rate of about 0.19; 0.125 gives about 20M. That is
   inferred from the dump's head, which may not be representative. The doc expects a ~69%
   policy share; which value of `--policy-share` should the full build use?
7. **The dirty tree.** `tools/capacity_suite.py` is modified and `tools/capacity_campaign.py`
   is untracked, so every v6 run needs `system.allow_dirty=true` until they are committed.
   Should they be committed?
8. **S2.** Approve ~8.5 GPU-h (GPU_TODO lists the commands).
9. **Engine loader.** Switching the evaluator to `load_for_inference` unblocks S7/G4. The
   package lives at `core/guofish_net/`; confirm the location (it is a doc open question).

## Commits (branch `v6-harness`, not pushed)

`b12fe76` M0 · `97b3b34` M1 · `b84307a` M2 · `6291185` M3 · `ea6434d` M4 · `5411db9` M5 ·
`0d1e117` M6 · report commit (this file, GPU_TODO.md). No data, models or run outputs are
committed; the untracked files that were there before are untouched.

## Final regression [measured]

`python -m pytest training/v6/tests`, with `CUDA_VISIBLE_DEVICES=-1` set by the conftest,
at below-normal priority: **71 passed in 1,105.9 s (18 min 25 s)**. That includes the
session-end check that no test initialised CUDA. Every gate number above reproduced exactly
in this run. The GPU_TODO tools were also run on the tiny CPU config:
- `bench.py` printed one JSON line per micro-batch (32/64/128 → 1,671 / 2,463 / 3,283
  samples/s on CPU).
- `s3_check.py --tol 0` passed with 43 steps compared and a max relative loss difference of
  0.0, on the natural mixture. The test-suite S3 uses a grouped mixture.

## Where the evidence is

- **Tests:** `training/v6/tests/test_m{1..6}_*.py`. The final all-suite regression result
  is at the end of this file.
- **Frozen val:** `data/processed/val_frozen_90m_v1/` (`manifest.json`, `strata_val_v1.npy` and
  its `.json`).
- **Scratch runs** (not committed; the session scratchpad): the tiny run, the smoke corpus
  `v2_smoke/`, the index check `index_check/v2_index_check.{json,csv}`, and the logs.
