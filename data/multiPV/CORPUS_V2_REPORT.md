# Corpus v2 — build report

Branch `v6-harness`. Session 2026-09-25 20:50 → 2026-09-26 (local time, EDT).

Tags: **[measured]** means I ran it and read the output. **[read]** means from code, logs
or manifests. **[inferred]** means my judgment.

Decisions and deviations are logged in `training/v6/DECISIONS.md`, under "Brief of
2026-09-25".

---

## Summary

- **Built, and every gate passes.** `data/processed/multipv_v2/` holds 114,268,954 roots
  and 32,137,001 derived records: 145,673,383 train, 571,232 val and 161,340 valderived.
  It is 56.7 GB. The build exited with code 0 in 2 h 08 m on 8 workers, beside the v5 S2
  run. [measured]
- **S9 passes in full.** 0 of the 91,350,634 replayed 90M lines were dropped, across the
  whole index. `hard_move` is on **100.00%** of roots. Rejection and dedup counts are
  present, and a second run into the directory is refused. [measured]
- **S6 passes in full.** `val_frozen_90m_v2` has exactly **452,405** records, all 12 shard
  counts match, and every shared field is byte-identical to v1 (0 mismatches;
  `pv_score` via float16). [measured]
- **Counts match the plan.** The old tier is exact, and the total selection is within
  0.00025% of the expectation. The realised policy share is **68.48%** of roots.
  Derived records: 32.1M, inside 30M ± 30%. [measured]
- **Loader benchmark: 32,208 samples/s** at 8 workers with the 0.65 / 0.25 / 0.10
  mixture. The bar was 7,600. [measured]
- **Three stopped attempts before the good build.** They were stopped for memory beside
  the v5 trainer, not because they failed. The fixes cut the builder's footprint from
  ~12.5 GB to ~1.5 GB (§3). [measured]

## 1. Selection plan (C1): expected against actual

The plan (`data/multiPV/rate_plan_v2.json`, sha256 `b989d2c9…`) is the brief's table.
- **The 90M cells use the manifest's full-precision rates.** Three of the brief's 8-decimal
  values are slightly *below* the 90M floats: old ≤5 policy, old ≤5 value-only and old
  15–27 value-only. The builder's nesting check would refuse them. [read]
- **Tiers** come from the Pass A index's `max_depth`: old ≥ 26, new 24–25.
- **Other settings:** `value_min_depth` 24, `policy_min_depth` 20, seed 20260802, v1's `u`
  stream.

The dry run took 31.5 s. The builder repeats the same selection and printed the same
115,315,219 lines. [measured]

**Selected index rows** (the "expected" column is the brief's derivation):

| tier / label | ≤5 | 6–14 | 15–27 | ≥28 | total | expected |
|---|---:|---:|---:|---:|---:|---:|
| old, policy | 593,484 | 20,728,798 | 25,009,155 | 9,760,936 | **56,092,373** | 56,092,373 (exact) ✓ |
| old, value-only | 318,991 | 11,247,714 | 12,786,952 | 12,164,146 | **36,517,803** | 36,517,803 (exact) ✓ |
| new, policy | 28,195 | 3,483,035 | 12,806,543 | 6,387,270 | **22,705,043** | ≈ 22,704,751 |
| new, value-only | 0 | 0 | 0 | 0 | 0 | 0 |
| **all** | | | | | **115,315,219** | ≈ 115,314,927 (+0.00025%) |

- **The new tier's +292 all sits in the random ≤5 cell:** 28,195 against 244,048 × 0.1143 ≈
  27,903, which is +1.9σ.
- **The other new-tier buckets are exact.** They equal the depth-24 index check's counts
  (e.g. 6–14 is 3,483,035). [measured]

**Actual roots after conversion** (train + val, from the strata sidecars). Here "policy"
means `has_policy` after conversion, and rejections are removed:

| tier / label | ≤5 | 6–14 | 15–27 | ≥28 | total |
|---|---:|---:|---:|---:|---:|
| old, policy | 592,844 | 20,688,645 | 24,841,846 | 9,639,211 | 55,762,546 |
| old, value-only | 318,958 | 11,242,732 | 12,616,574 | 11,838,859 | 36,017,123 |
| new, policy | 28,093 | 3,474,995 | 12,701,096 | 6,285,101 | 22,489,285 |
| new, value-only | 0 | 0 | 0 | 0 | 0 |

- **Totals reconcile.** The rows sum to 114,268,954 roots. The 1,046,265 rejected lines
  make up the difference to the selection. [measured]
- **Realised policy share:** 78,251,831 / 114,268,954 = **68.48%** of roots. The selection's
  share was 68.33%, and the doc expected ~69%. [measured]
- **Value-only roots are exactly the 90M set.** The old tier's value-only roots in train
  number **35,836,594**, the 90M train value-only count. That is expected: the old-tier
  value-only rates equal the 90M rates on the same `u` draws, and the new tier has no
  value-only rows. [measured]

## 2. Derived records (C2, `--derived-rate 0.19`)

| stage | k=1 | k=2 | total | per root |
|---|---:|---:|---:|---:|
| candidates | 112,912,713 | 112,247,348 | 225,160,061 | 1.970 |
| sampled (rate 0.18998 realised) | 21,452,198 | 21,324,756 | 42,776,954 | 0.374 |
| dropped: duplicate of a root | | | 10,105,150 (23.6%) | |
| dropped: duplicate of an earlier derived | | | 534,803 (1.25%) | |
| **written** | 14,905,067 | 17,231,934 | **32,137,001** | 0.281 |

- **Written counts** are train 31,975,661 plus valderived 161,340. By origin: ply1
  14,830,022 + 75,045, ply2 17,145,639 + 86,295. [measured]
- **Unroll stops:** 2,012,612 at a terminal position, 8,993 at the end of the line, and 1
  where the mate was used up. [read]
- **The smoke over-predicted dedup** (33.5% dropped). The dump's opening-heavy head
  transposes more, so the smoke projected ~28.6M derived, where the full build gave 32.1M.
  [measured; the reason is inferred]

## 3. Build: exit code, runtime, restarts

- **Command:**
  `python data/multiPV/pass_b_v2.py --rate-plan data/multiPV/rate_plan_v2.json --out-dir data/processed/multipv_v2 --manifest-copy data/multiPV/manifests/dataset_manifest_v2.json --derived-rate 0.19 --workers 8`,
  with `CUDA_VISIBLE_DEVICES=-1`. Priority was BelowNormal, inherited by the workers
  (verified).
- **Exit code 0**, captured through a held process handle. The run took **7,671.7 s**
  (21:31:36 → 23:39:29), converting at a steady ~16,000 roots/s. It overlapped all of the
  v5 `ref` run. [measured]
- **The timing smoke** (`--limit 2000000`, 8 workers, beside G2) ran 59.7 s, ~33.5k
  lines/s, ~15k roots/s. It projected a conservative ~3.3 h; the build took 2.1 h.
  [measured]

**Stopped attempts** (deviation, all mine and deliberate):

| attempt | ran | stopped because | fix, committed before the next attempt |
|---|---|---|---|
| 1 | 21:15:54, ~6 min | commit charge reached **59.4 of 60.5 GB**; the build held 12.5 GB (1.28 GB per worker) beside v5's ~26 GB | dedup post-pass made lean (`672a607`): `searchsorted` in place of `np.isin`, a vectorised routing hash, the selection freed first |
| 2 | 21:23:25, ~2 min | the workers still held 1.28 GB each | torch kept out of the workers (`cf5614b`): lazy imports in `data/pgn_parallel.py` and for `training.v6.ckpt`, −770 MB per process |
| 3 | 21:28:23, ~3 min | still 513 MB per worker | `OPENBLAS_NUM_THREADS=1` (`6bac6ef`): numpy's OpenBLAS pre-allocated a buffer per core, −492 MB per process |
| **4** | 21:31:36 → 23:39:29 | completed | workers ~31 MB each, main ~1.0–1.7 GB |

Each time, only the partial `train_*` / `val_*` shards and `_derived_staging.bin` were
deleted. No manifest or manifest copy existed. [measured]

## 4. S9 in full: **PASS** [measured]

`python data/multiPV/corpus_v2_gates.py s9`:

| check | result |
|---|---|
| nesting over the whole index; 90M replay reproduces 91,350,634 rows; 0 dropped | ✓ |
| `hard_move` coverage ≥ 99% of roots | ✓ **100.00%** (114,268,954 of 114,268,954; no failures by reason) |
| rejection histogram present | ✓ chess960 967,261 · invalid FEN 31,428 · invariant violation 47,576 (rate 0.041%, bar 0.1%) |
| dedup counts present | ✓ (§2) |
| shard counts add up; roots + rejections = selected lines | ✓ |
| manifest copy byte-identical (`dataset_manifest_v2.json`) | ✓ sha256 `eeba5c6b…` |
| a second run into the same directory is refused | ✓ "already holds a manifest; refusing (H1)" |

## 5. Frozen val re-materialisation and S6 in full: **PASS** [measured]

- **Command:** `extract_frozen_v2.py --corpus data/processed/multipv_v2 --out data/processed/val_frozen_90m_v2`, 22 s.
- **Count:** 452,405, the same in v1 and v2, and all 12 per-shard counts match.
- **Shared fields:** 0 mismatches in each of tokens, value, value_cp, has_policy, n_pv,
  pv_idx, pv_prob, pv_score (compared via float16), n_legal and legal_idx.
- **Manifest sha256:** `20c38314…`.

The v2 frozen set adds `hard_move` on all 452,405 records. The 180,529 value-only ones now
carry a hard label.

## 6. Strata (definition v2)

Every sidecar is `data/processed/strata/<corpus>_<split>.strata2.npy` plus `.json`, with
definition hash `a9e7a62e9880`. [measured]

| sidecar | records | time |
|---|---:|---:|
| `multipv_90m_train` | 90,076,117 | 144 s |
| `val_frozen_90m_v1_val` | 452,405 | 0.5 s |
| `multipv_v2_train`, `_val`, `_valderived` | 145,673,383 / 571,232 / 161,340 | 407 s total, including the 16 s index context |
| `val_frozen_90m_v2_val` | 452,405 | 26 s |

**Checks on the 90M sidecars:**
- `multipv_90m_train` has 54,239,523 multipv, which passes GPU_TODO's G0 bar.
- `val_frozen_90m_v1_val`'s low 9 bits equal the old v1 sidecar exactly.

**`depth_tier` × label (train):**

| | multipv | hard_only | value_only |
|---|---:|---:|---:|
| old | 55,484,304 | 60,914,761 | 0 |
| new | 22,376,824 | 6,897,494 | 0 |

- **`hard_only`** covers the value-only roots plus every derived record.
- **`value_only` is empty** because `hard_move` coverage is 100%.

**`in_90m`:**

| split | in_90m = 1 | in_90m = 0 |
|---|---|---|
| train roots | **90,076,117**, exactly the 90M train set | 23,621,605: 22,376,824 new-tier policy, plus 1,244,781 old-tier rows from raising 15–27 policy from 0.9497 to 1.0 |
| train derived | 24,752,428 | 7,223,233 |
| val roots | **452,405**, exactly frozen val | 118,827 (112,461 new-tier + 6,366 old) |

No new-tier row is `in_90m`, which is correct by construction. Frozen v2 is 100% old tier,
`in_90m` and root.

**Material classes:**

| split | level | ahead | compensated |
|---|---:|---:|---:|
| train | 102,165,816 (70.1%) | 30,623,726 (21.0%) | 12,883,841 (8.8%) |
| val | 405,946 | 115,520 | 49,766 |
| valderived | 108,326 | 38,233 | 14,781 |
| frozen v2 | 313,277 | 98,430 | 40,698 |

Frozen v2's material classes are identical to frozen v1's.

## 7. Loader benchmark: **PASS** [measured]

`CUDA_VISIBLE_DEVICES=-1 python -m training.v6.tools.loader_bench --config training/v6/config/configs/mix_v2.yaml --seconds 120`

- **Result: 32,208 samples/s** over 121.5 s (3,914,752 samples, after 50 warm-up
  micro-batches; worker spawn and member lists took 18.6 s). The bar was ≥ 7,600.
- **Setup:** v6 loader and collate, v2 train, 8 workers, micro-batch 512.
- **Shares were exact:** policy 0.65, hard (`hard_only`, roots) 0.25, derived (ply1 or ply2)
  0.10.
- **Group sizes:** 77,861,128 / 35,836,594 / 31,975,661.
- **Timing:** 00:08:58–00:11:27, in the S2 gap between `ref` and `c1`. The driver was held;
  there was no trainer and no Track C job.
- **Caveat:** part of the corpus was still in page cache from the strata pass ~25 min
  earlier. 56.7 GB does not fit in 32 GB, so a cold loader will be slower. The margin (4.2×)
  is large. [inferred]

## 8. Disk, runtime, workers, provenance

**Disk:** [measured]
- `multipv_v2`: 56.66 GB. The derived staging file (~16.6 GB) was deleted by the builder.
- `val_frozen_90m_v2`: 168 MB.
- `data/processed/strata`: 453 MB.
- C: now has 84 GB free (143 GB at start).

**Runtime:** build 2 h 07 m 52 s, S6 22 s, strata 7.5 min. Workers: 8, below normal.

**Provenance** (manifest): [read]
- **Code:** `git_sha` `6bac6ef`, the commit the build ran. None of the builder's code or
  dependencies differ from it.
- **Dirty files** (`diff_sha256` `fcafc53b…`): only the pre-existing untracked and modified
  files (`tools/capacity_*`, `benchmarking/…`), my then-uncommitted
  `training/v6/tools/s2.py`, `DECISIONS.md` and reports, and `training/v6/tools/after_s2.py`,
  which is not mine.
- **Inputs:** source dump sha256 `2046f874…`, rate plan sha256 `b989d2c9…`.
- **The builder records git state at the end of the build.** I made no commits while it
  ran.

## 9. Deviations

1. **Full-precision 90M rates in the plan** instead of the 8-decimal display (§1).
2. **Three stopped build attempts**, and three builder fixes for memory (§3).
3. **`data/pgn_parallel.py`** (shared code) now imports torch inside `parse_game_block`
   instead of at module level. Behaviour is identical, and v5's vocab tests pass.
4. **`--manifest-copy`** writes `dataset_manifest_v2.json` from the builder itself, refusing
   an existing file, rather than being copied afterwards.
5. **`source_order_quantiles`** is no longer in the v2 manifest.
6. **The timing smoke ran beside G2**, a GPU trainer that is not throughput-sensitive,
   rather than on an idle box.

## 10. For the screening plan

1. **The hard-label roots are exactly the 90M value-only positions** (35,836,594). A
   `w_hard > 0` arm on the `hard` group therefore trains the same positions with a new
   label. Any gain is attributable to the label, not to new positions. New positions
   arrive only through the new-tier policy roots (22.49M) and derived records (32.1M).
2. **frozen90 contains no new-tier and no derived positions.** It cannot see
   new-tier-specific effects. The v2 val split has 112,461 new-tier roots and valderived
   has 161,340 records, but the `v2val_roots` / `v2val_derived` eval sets are still
   unbuilt (M5). They should exist before an arm that changes the mixture is judged.
3. **Passes at the base 360M-sample budget with `mix_v2`:** policy ≈ 3.0, hard ≈ 2.5,
   derived ≈ 1.1. The hard group is the one repeated most for its size. [inferred from the
   group sizes]
4. **Derived records are 22% of train rows.** The derived-rate lever is well calibrated:
   0.19 gave 32.1M against a 30M target.
5. **Memory on this box.** Each v6 trainer loader worker holds 1.37 GB private; the
   `ref`/`c1` breakdown is in `training/v6/S2_REPORT.md`. ~0.5 GB of that is OpenBLAS's
   per-core buffers, the same effect fixed in the builder. `OPENBLAS_NUM_THREADS=1` for
   trainer workers would free ~4 GB at 8 workers. That matters for running a build or a
   match beside training. It was not changed mid-S2. [measured per process; the trainer
   saving is inferred]
