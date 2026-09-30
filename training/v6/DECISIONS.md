# v6 harness — decisions and deviations

Deviations from *GuoFish v6 Training Stack — Design* (`docs/capacity/training_stack.md`,
"the doc") and judgment calls it leaves open. Each entry: what, why.

---

## M0 — frozen val

- **Manifest lives in the shard directory** (`val_frozen_90m_v1/manifest.json`) and uses
  Pass B's `record_dtype` spelling, so the v6 reader identifies the format from it the
  same way it does for the corpus manifests.

## M1 — model package (`core/guofish_net/`)

- **`init: v5_deepcopy` matches v5's distribution, not its RNG stream.** It applies
  PyTorch-default init (Embedding N(0,1), Linear Kaiming-uniform, `in_proj` Xavier-uniform
  with zero biases, the `out_proj` bias zeroed, positions randn×0.02) and copies block 0 into every block.
  The same seed does not produce v5's exact initial weights. S2 is judged against a seed-noise band,
  so bit-identical init buys nothing there.
- **Static bias is passed to SDPA as a 4-D tensor, detached when grad is off.** Measured:
  CPU `scaled_dot_product_attention` leaves its fused kernel when the mask is 3-D *or*
  has `requires_grad=True`, even under `no_grad`. Either one alone gave a 5.2e-6
  policy difference at zero bias, which failed S8. With both fixed, S8 is exact (0.0) for static and
  smolgen.
- **Smolgen details the doc leaves open.** The second LayerNorm normalizes the whole
  `heads × 128` vector, before the reshape; the doc lists LN before the reshape. The shared
  128→4096 projection has no bias, so a zero weight gives a zero bias exactly.
- **`aux_policy_head` is a `ModelConfig` field.** The doc puts `policy_hard.head: main | aux`
  under `targets`, but the builder has to know whether to create the head. The config
  loader requires the two to agree (`targets.policy_hard.head == "aux"` ⇔
  `model.aux_policy_head`).
- **hlgauss constants are code constants, not config:** 101 bins over [−1, 1],
  σ = 0.75 × bin width. They are part of the architecture (`arch_version`), not a tunable.
- **`forward` returns fp32 even for a float64 model.** The doc's contract is fp32
  outputs, cast once after the forward. The float64 parity test accounts for this cast.
- **`strict.py` (strict nested frozen-dataclass builder) lives in `core/`** because
  `ModelConfig.from_dict` needs it, and the trainer's config loader reuses it rather than
  keeping a second copy. It parses numeric strings for int/float fields, because
  PyYAML reads `360e6` and `7.2e6` as strings.
- **`load_for_inference` accepts export files only**, not training checkpoints. One format
  reaches the engine.
- **The module keeps a `seq_length` attribute**, which the v5 engine reads off a loaded
  module.

## M2 — config system (`training/v6/config/`)

- **Config files live in `training/v6/config/configs/`**, following the doc's §3 table
  ("`training/v6/config/` | … `configs/*.yaml`").
- **Paths are checked at run start, not at config load.** `base.yaml` points at the corpus v2
  paths the doc gives, and those don't exist yet. Checking at load would stop any config from
  resolving or hashing on a machine without the data.
- **Resume whitelist** is the doc's (output root, log cadence, `total_samples` for branching),
  plus `data.workers` and `data.prefetch_factor`. The sample stream is a function of
  (seed, sample index) only, so worker count cannot change it.
- **Cadences are multiples of the effective batch** (`ckpt.*_samples`, `eval.*_samples`).
  `onecycle` also requires `total_samples` to be one, because torch's OneCycle is defined
  over whole steps. WSD does not: the doc's own `360e6` is not a multiple of 1,024. The LR
  is a function of the sample index, and the run ends at the first step boundary at or
  past `total_samples`.
- **`policy_soft.source` and `temperature` are both kept** (the doc's example has both) and
  cross-checked: `pv_score` requires a temperature, `stored` forbids one.
- **`optim.decay_embedding`** (default false, per §8.1): v5 decayed `embedding.weight`, so
  `v5_compat.yaml` sets it true.
- **`v5_compat.yaml` `total_samples` = 360,302,592** (4 × 90,075,648, v5's drop_last epoch),
  i.e. 351,858 optimizer steps. v5 itself ran 351,860 steps, because each epoch ended on a
  half window of one micro-batch. v6 has only full windows. The 2-step difference is at the
  end of the schedule, where LR is ~1.4e-9.
- **bf16 requires `system.device: cuda`**: a hard error rather than a silent CPU autocast.

## M3 — data layer (`training/v6/data/`)

- **v1 upcast sentinels.** v1 records carry neither `value_depth` nor `src_line`. The reader
  reports `value_depth = 0` (a real depth is ≥ 20) and `src_line = 0xFFFFFFFF`, alongside the
  doc's `hard_move = −1`, `origin = 0`. v1 `pv_score` (float16 of an int) converts to
  int16 exactly, and the reader checks this on every read.
- **The ε spread accumulates in float64, per add, as v5 does.** Found by S1: v5's
  `np.add.at(policy, li, eps / n_legal)` passes a Python float, so NumPy adds in float64 and
  rounds to float32 after every add. A float32 share was 1 ulp off on 30 of 2,000 records
  wherever entries accumulate (promotion squares, PV moves). v6 passes a float64 share.
  PV mass is added as float32, as in v5.
- **Mirroring happens on the record, before dense targets are built.** Tokens, `pv_idx`,
  `legal_idx`, `hard_move`, `value` and `value_cp` are permuted and negated in the record.
  The same float32 values then accumulate in the same order at permuted positions, so the
  result equals v5's dense-then-gather mirror bit for bit (S1, mirror on).
- **Per-micro-batch counts use cumulative largest remainder.** The count for micro-batch k is
  A(B·(k+1)) − A(B·k), where A(N) apportions N samples by largest remainder, ties broken by
  group order. Every micro-batch gets the floor or the ceiling of share × B, summing to B.
  The cumulative count stays within one sample of share × N forever, and the whole thing is
  closed-form in k. This is how "remainders rotated deterministically" is implemented. Each
  group needs share × micro_batch ≥ 1, validated at config load.
- **Within-group order is a seeded Feistel permutation, not a stored array.** It is a
  6-round balanced Feistel network with cycle walking, keyed by (seed, group, pass). Each
  pass gets a fresh key. That keeps O(1) memory per worker at 150M records, and makes the
  position at any sample index closed-form.
- **Mixture groups must be disjoint**, and an overlap is a hard error naming the counts. The
  doc's §6.7 example overlaps: derived rows have label `hard_only` *and* origin `derived`.
  As written it would be refused. It needs `where: {label: hard_only, origin: root}` on the
  `hard` group. A group with a share and no records is also an error.
- **Grouped mixtures write member lists** as uint32 `.npy` files in the run directory. Workers
  memmap them, so they share page cache rather than each holding a pickled copy.
- **canonical_65 ep legality is read from the stored legal moves.** There is no board in the worker. A
  capture onto the target from an adjacent rank-5 pawn must be in `legal_idx`. Caveat: a
  position with more than 128 legal moves stores a truncated list (200 records in 90M) and
  could lose its ep capture. None was seen in S5. The illegal-ep branches (no capturer, a
  pinned capturer) never occur in real data, where 144 of 144 ep squares were legal, so
  they are pinned with hand positions against python-chess.
- **Strata sidecars.** The frozen-val sidecar lives next to the frozen shards (the M0
  directory). For the 90M train split, `v5_compat.yaml` points at
  `data/processed/strata/multipv_90m_train_v1.npy`, outside the live corpus directory,
  and that file has not been built (see GPU_TODO).
- **The value stratum for λ weights is computed per batch from `value_cp`.** It is the same rule as the
  sidecar, so the training path needs no strata lookup.

## M4 — losses, optimization, schedule, EMA

- **Per-term normalizers are expected counts, computed once from config + strata.** §7 says
  each term divides by "its own row count in the optimizer window", and §4 says the
  normalizer is "computed from the config, logged at start, and stored". The two agree when
  mixture groups align with label kinds, where counts are constant per window. v6 uses
  N_term = effective batch × Σ_g share_g × (fraction of group g eligible for the term). That is exact for
  label-aligned groups. Under `natural` it is v5's constant C = window × coverage, but with
  the corpus coverage measured exactly from the strata (H15). A constant denominator
  keeps gradient accumulation exact.
- **Hard target ε is spread over the unique legal indices** (the dense bool mask). §7 says
  "spread over legal moves". v4 Phase 3, which §7 cites, used `CrossEntropyLoss(label_smoothing)`
  over all 4,096 classes, and the doc's own wording is followed instead. The target is
  (1 − ε) on `hard_move` plus ε/n over the n legal indices, which sums to 1. A `hard_move` outside
  the stored legal set is unioned in, as with the soft target's truncation repair.
- **The hard term applies to rows with `has_policy = 0` and `hard_move ≥ 0`** (hard-only and
  derived rows, §7). Multi-PV rows get only the soft term. With `policy_hard.head: aux`, the
  hard term trains the aux head and the soft term trains the main head.
- **The masked log-softmax is re-implemented, not imported from v5.** v5's has a
  `bool(empty.any())` host sync per call (H19). v6 unions the empty-row repair
  unconditionally. `test_soft_kl_matches_v5` shows the KL is bit-identical to v5's function.
- **WSD warmup is `peak × s / warmup_samples`**, where s is the sample index at the step's
  start, so LR is 0 at step 0 (the common linear-warmup convention).
- **Muon details.** Muon covers the 2-D weights inside blocks: fused QKV as one matrix,
  attention output, both FFN matrices, and smolgen's compress/fc1/fc2. The shared smolgen
  projection sits outside the blocks and gets AdamW. Update scale is
  0.2·√max(rows, cols) (Liu et al. 2025), so the AdamW LR and weight decay (decoupled,
  `p ← p(1 − lr·wd)`) carry over. Newton–Schulz runs in bf16 on CUDA and fp32 on CPU.
- **`tools/branch_decay.py` landed with M5**, not M4. It is the trainer's resume path with a new
  `total_samples` and output directory, so it needs the trainer.

## M5 — trainer, eval, checkpoints, export

- **Non-finite guard without a per-step sync.** Each window's finite flag is computed on
  device and handed to the optimizer as `found_inf`. That is the GradScaler protocol fused AdamW
  already honours; v6's Muon honours it too. A bad step is skipped on device. The next log
  sync (every `log_every` steps) sees the count, writes `ckpt/emergency_s<N>.pt` with the last
  good weights and exits non-zero. Up to `log_every − 1` further windows can be skipped before
  exit.
- **The checkpoint stores the global torch RNG (and CUDA's on a CUDA run).** The doc says no
  *worker* RNG is needed, and that holds because augmentation is hash-based. But dropout
  draws from the main-process RNG, so it is saved. Every DataLoader gets its own
  `torch.Generator()`. Without one, creating an iterator draws a seed from the global RNG,
  and a resume creates its iterator at a different point in the stream, which would shift
  dropout masks. S3 runs with dropout 0.1 to prove both.
- **`--crash-after-steps N` (hidden)** simulates a kill with `os._exit`: no cleanup and no
  checkpoint. S3 uses it so the kill point is deterministic.
- **The step log carries a per-interval stream digest**: sha256 of the window's record indices
  and mirror flags. That is how S3 shows "same sample indices and augmentation decisions".
- **`torch.compile` wraps `model.forward_train`**, the training path. The engine compiles
  `forward` itself. Eval runs the uncompiled module, and EMA eval uses a separate uncompiled
  instance.
- **Resume.** A config diff outside the whitelist is refused. `schedule.total_samples` is
  accepted only in branch mode (`tools/branch_decay.py`), per §10.2. A finished run refuses
  to resume ("nothing to do"), which covers H10. Each resume writes its own
  `provenance_resume_<utc>.json` and `code_resume_<utc>.patch`, alongside the originals.
- **`code.patch` is `git diff HEAD --binary`.** Untracked files are listed in
  `dirty_files` but their contents are not captured. `system.allow_dirty: false` refuses a
  dirty tree anyway.
- **Export names add `_s<samples>_<weights>`** to the doc's `<run>_<shape>_<cfghash8>`. Two
  exports of one run (EMA and raw, or two checkpoints) would otherwise collide, which is D-7 again.
  An existing file is never overwritten. `best.pt` stores whichever weight set (raw or EMA)
  scored best, and `--weights best` exports it.
- **Eval cadence.** Quick-val evaluates raw weights on the stratified seeded subset
  (`eval.quick_size`, default 32,768) every `quick_every_samples`. Full eval covers raw and
  EMA at stable checkpoints, at `full_every_samples` if set, and at run end. Mirror
  consistency uses a seeded random `mirror_n` subset rather than a prefix (H3).
- **Not built:** the `v2val_roots` / `v2val_derived` eval sets, because corpus v2 doesn't
  exist (only `frozen90` is wired), and `tools/lr_range.py` (in the doc's layout, not in the
  brief).
- **INCIDENT: GPU hiding was ineffective in this session.** In PowerShell 5.1,
  `$env:CUDA_VISIBLE_DEVICES = ""` *deletes* the variable (measured), so every process
  launched from PowerShell, directly or through the low-priority launcher, had the GPU
  visible. Test subprocesses given `env={..., "": ""}` did see an empty value, so they had it
  hidden. Only one code path initialises CUDA: the checkpoint writer called
  `torch.cuda.get_rng_state_all()` whenever CUDA was available. It ran with the GPU
  visible once, in the manual tiny run at 23:49. That was after the capacity trainer had
  already crashed, at 23:14:49. Fixes:
  1. CUDA RNG is saved only when `system.device == "cuda"`, with a regression test that
     simulates a visible GPU.
  2. `training/v6/tests/conftest.py` sets `CUDA_VISIBLE_DEVICES=-1` for every test and its
     subprocesses, and fails the session if CUDA was initialised.
  3. The launcher uses `-1`.

  See HARNESS_REPORT.md for the crash timeline.

## M6 — corpus v2 builder (`data/multiPV/pass_b_v2.py`), frozen-val extractor

- **v1 code is imported, not copied:** `labels.py` (every label rule), `pass_b_convert`
  (`build_selection`, routing hash, `file_hash`), `feasibility_scan` (rates), and
  `pass_a_index.iter_lines`. The v1 CLI is never run by tests, because its defaults point at the live
  corpus (H1). The synthetic "90M" build calls v1's `build_selection` and `convert_one`
  in-process.
- **Position key for dedup:** blake2b-64 of the first four FEN fields (placement, side,
  castling, *legal* ep), computed from the parsed board for roots and derived alike.
- **Derived sampling is independent per ply.** Every k = 1..`max_ply` candidate that passes
  the §6.3 conditions is kept with probability `--derived-rate`, using a hash of
  (line, k, seed). A k = 2 candidate can be kept when k = 1 was not. Unrolling stops at the
  first failed condition, and each stop reason is counted.
- **Mate shortening:** |mate| drops by the number of the mating side's moves among the k
  plies. A saturated distance (≥ 1,000) is kept as stored. A mate used up without a terminal
  position stops the unroll (`derived_stop_mate_used_up`).
- **Derived train records are appended after all roots** in the train shards, routed by a
  hash of (line, k). Roots keep v1's order and routing, so shard contents are deterministic.
- **No resume.** A failed build is re-run into a new, empty `--out-dir`, as the H1 fix requires.
- **`--limit N` selects over the first N index rows only.** The u stream is positional, so
  that equals the full build's selection of those rows, without holding ~120M line
  numbers in RAM. `selection_scope` in the manifest says so.
- **Defaults are starting points for the operator:** `--target 121,100,000` (about 120M roots at
  the 90M yield), 90M shares and `policy_share` 0.65, `--derived-rate 0.125`, 256/16/4
  train/val/valderived shards. The smoke measured 213k derived candidates per 108k roots
  and 32% of sampled ones dropped as root duplicates. Size the rate from those numbers.
- **S9 exit codes:** 1 for an invariant rate above 0.1% (v1's gate) or broken nesting; 3 for
  `hard_move` on fewer than 99% of roots. The manifest is still written, per "investigate
  before training".
- **`extract_frozen_v2.py` reproduces v1's val order** (routing hash, then source order) so the
  comparison is positional. It requires the full index to replay the 90M selection
  (730 MB of line numbers).
- **`v2_index_check.py` is a new file**, not the recon's `headroom_scan.py` (which is untracked,
  in the gitignored `recon/`). It keeps that script's method, including self-checking the
  reproduced 90M selection against the manifest before reporting anything.

## Brief of 2026-09-25 — corpus v2 build and S2

This brief changes the doc in two places, recorded here rather than edited into it:
**§6.4's single `policy_share` rule is superseded** by a per-tier rate plan (C1), and
**§6.6's strata fields are extended** (C3).

### C1 — per-tier selection plan (`pass_b_v2.py --rate-plan`)

- **Rates are per (tier, has_policy, bucket)**, read from `data/multiPV/rate_plan_v2.json`
  (`{tier: {label: {bucket: rate}}}`). `--target`, `--shares`, `--policy-share` and
  `--no-spill` are gone, with the v1 rate derivation they fed. The manifest records the plan,
  its file, its sha256, and the selected counts per tier × label × bucket.
- **Tiers come from the Pass A index's `max_depth`.** `old` is `max_depth` ≥ the floor
  manifest's `value_min_depth` (26). `new` is [`--value-min-depth` (24), 26). The same `u`
  stream as v1 (`default_rng(seed)`, one draw per index row), so each cell's selection only
  grows with its rate.
- **Nesting is checked twice.** Before any scan, an `old` rate below the 90M manifest's rate
  for its cell is refused. The scan then replays the 90M selection on the same draws,
  self-checks the replay against the manifest's `selected_lines` (full index only), and
  refuses the build if any replayed line is not selected.
- **The plan uses the 90M manifest's full-precision rates**, not the brief's 8-decimal
  display. Three displayed values are *below* the 90M floats: old ≤5 policy 0.11433605 <
  0.114336054…, old ≤5 value-only 0.05731881 < 0.057318811…, old 15–27 value-only 0.29495813 <
  0.294958134…. The nesting check would refuse those, as it should: the brief's intent is
  "the 90M rate" in those cells. New ≤5 policy uses the same float for consistency. Against
  the rounded value that changes the expected count by ~0.001 rows.
- **`--dry-run`** runs the selection only and prints the counts; it writes nothing.
  **`--manifest-copy`** also writes the manifest to `dataset_manifest_v2.json`, and refuses
  an existing file before the build starts. **`--limit`** no longer writes an index-head file
  into the output directory; the selection simply stops after N rows.
- **`source_order_quantiles` is dropped** from the v2 manifest. It was v1's §7 diagnostic and
  nothing reads it.

### C2 — derived rate and provenance

- **`--derived-rate 0.19`** is passed on the command line; the default stays 0.125.
- **Provenance was already in the manifest** (`git_sha`, `dirty_files`, `diff_sha256` via
  `ckpt.git_state`). Nothing was added.

### C3 — strata definition v2

- **Layout, 13 bits of the uint16:** bucket 0–1, label 2–3, value 4–5, material 6–7,
  **origin 8–9** (root, ply1, ply2), **depth_tier 10–11** (old, new, v1), **in_90m 12**.
  `DEFINITION.version` is 2, so `load_strata` refuses every v1 sidecar.
- **`origin` is widened, not duplicated.** v1's `root`/`derived` became `root`/`ply1`/`ply2`.
  "Derived" is now `origin: [ply1, ply2]`. An origin above 2 is an error: `max_ply` 2 is the
  build.
- **`depth_tier` is the index's `max_depth` at `src_line`**, the brief's definition. It is
  not the record's `value_depth` (+ `origin`). The value block is the deepest block *with
  PVs*, so a row whose deepest block is empty would be tiered differently. Derived records
  inherit their root's tier and `in_90m`, through `src_line`. v1 records (the reader's
  `value_depth = 0` sentinel) are tier `v1` with `in_90m = 1`.
- **`in_90m` values are the ints 0 and 1**, because YAML 1.1 reads `yes`/`no` as booleans.
- **`build_strata`** takes several splits per call, sharing one index context between
  them: the 90M replay plus the `max_depth` column, about 1.5 GB. Output is named
  `data/processed/strata/<corpus dir>_<split>.strata2.npy`, with a `.strata2.json` sidecar.
  The old `val_frozen_90m_v1/strata_val_v1.npy` stays where it is (that directory is
  read-only) but is now refused as stale.
- **Mixture example.** `configs/mix_v2.yaml` is the doc's §6.7 mixture made disjoint:
  `hard: {label: hard_only, origin: root}` and `derived: {origin: [ply1, ply2]}`. Overlap
  stays a hard error. There were no grouped examples in `configs/` to change.

### Track G — GPU smoke checks and S2

- **Triton kernel names are plain** (`inductor_config.triton.descriptive_names = False` in
  `train.py`). G1's first compile failed with `FileNotFoundError` in Triton's cache. One
  fused kernel name was 135 characters, and the cache path came to 283 even under
  `%USERPROFILE%\ti`. That is v5's MAX_PATH fix, tried first and not enough on its own, so
  it was not kept. Only kernel names change.
- **The S2 driver (`tools/s2.py`) corrects GPU_TODO's S2 commands against the brief's protocol.**
  1. `ref` gets `--gap-probe 0`. `corpus90m.yaml` sets 200k, so the probe would have run.
  2. `ref` gets an explicit `--seed 20260802` and `--out-dir models/v6/s2/ref`, inside the
     allowed write area.
  3. `c2`'s seed is 20260925, not 20260803.
  4. `run.out_root=models/v6/s2`, and `ema.enabled=false` is explicit.

  Confirmed unchanged: `--max-steps` caps OneCycleLR's `total_steps` in `train_v5.py`, so the
  schedule is compressed, not truncated. `total_samples` 60,000,256 is 58,594 × 1,024.
- **Known compat difference in S2: v5's policy denominator.** v5 measures coverage on
  200,000 records (0.603120 this run); v6 uses the exact corpus coverage from strata
  (0.602146), H15. That is a 0.16% difference in the soft-KL normalizer.
- **G2 (S3 on GPU) fails its 1e-3 criterion through run-to-run nondeterminism, not resume.**
  Stream digests (record indices and mirror flags) match on all 307 compared steps. But
  run B's own first 157 steps, *before* the kill, already differ from run A. Steps 1–3 are
  identical, step 4 differs by 2e-6 relative, and the gap grows to 5.4% near peak LR. The
  resumed segment's maximum is 1.1%. On bf16 + compile, two uninterrupted runs do not
  reproduce each other (atomics in backward), so a per-step 1e-3 bar cannot hold.
  `tools/rng_check.py` checks the one GPU-specific resume state (CUDA RNG under compiled
  dropout) directly.

### Memory on the shared box (found while starting the build beside `ref`)

- **Commit charge reached 59.4 of 60.5 GB** with the v5 `ref` trainer and the first build
  attempt running. The pagefile is system-managed, so the limit can grow, but allocations
  can fail while it grows. v5's footprint is ~26 GB private: the main process at 9.1 GB,
  8 loader workers at 1.48 GB each, and 4 persistent val-loader workers at 1.34 GB each.
  `training/v5_multiPV/` is read-only.
- **The builder's footprint went from ~12.5 GB to ~1.2 GB**, in three committed fixes:
  1. **Every spawned worker imported torch** (+770 MB each). It came in through
     `data/pgn_parallel.py`'s module-level `import torch`, used only by `parse_game_block`,
     and through `training.v6.ckpt`. Both imports are now lazy, and a test pins that
     `import pass_b_v2` loads no torch.
  2. **numpy's OpenBLAS pre-allocates a buffer per core** (+492 MB per process on 16
     threads). The builder sets `OPENBLAS_NUM_THREADS=1` (setdefault) before importing
     numpy; it uses no BLAS.
  3. **The dedup post-pass is lean.** It frees the selection first, finds root duplicates
     with `searchsorted` on the sorted unique root keys instead of `np.isin`, and routes with
     a vectorised splitmix64 (equivalence-tested against the scalar hash).
- **Build restarts caused by this:** attempts 1–3 were stopped after 1–6 minutes of
  conversion. Their partial outputs (train/val shards and staging only) were deleted
  before each relaunch. Attempt 4 runs `6bac6ef`.
- **The builder records `git_state` at the end, when it writes the manifest**, not at
  start. No commits were made while the build ran, so the manifest's SHA is the code that
  ran.

### S2 results and analysis

- **S2's training-loss window is the last 1,044 steps** (1,069,056 samples), not 1,000.
  58,594 = 1,171 × 50 + 44, so only a multiple of v6's 50-step log rows plus the last
  44-step row aligns exactly; that is 21 rows. The window is applied identically to v5's
  per-micro-batch rows, which carry 0-based step numbers.
  The first `analyze` crashed on both points. It was run by `training/v6/tools/after_s2.py`,
  a watcher started outside this session and left uncommitted.
- **S2 verdict and d** are in `training/v6/S2_REPORT.md`. S2 passes. d is 0.97% relative on
  frozen90 KL and 2.31% on MSE, from one seed pair.
- **G2's criterion needs replacing** (proposal in the S2 report). On GPU, uninterrupted runs
  are not reproducible at 1e-3 per step. `rng_check` shows the resume path itself is exact.

## Brief of 2026-09-26 — screening pass (Phases 2–3) at d384×10

The brief (`docs/capacity/screening_pass.md`) supersedes the doc on six points. They are
recorded here and not edited into the doc:
1. **Shape d384×10**, with 6 heads, d_ff 1536 and micro-batch 512 × 2 (`configs/shapes/d384x10.yaml`).
2. **A2 (learning rate) dropped.** Peak LR stays 3.5e-4.
3. **G2 amended and PASS.** `GPU_TODO.md` G2 carries the new criterion and its evidence.
4. **Phase 2 on corpus v2, `in_90m` roots only** (the `base90` group). That group is exactly
   90,076,117 records, measured against the train strata.
5. **Paired-seed decision rules** (§4 of the brief), implemented in `tools/screen.py`.
6. **Proxy EMA half-life 2M samples.**

### H1 — data and init seeds; name-keyed init

- **`run.seed` is gone.** `run.data_seed` drives the sampler and mirror flags;
  `run.init_seed` drives the weights. An old `run.seed` key is a hard error (unknown key).
  - **`v5_compat.yaml`** sets both seeds to 20260802, so S2's `c1` resolves as before.
  - **`s2.py`** passes both seeds.
- **Independent init is keyed per parameter.** Each Linear/Embedding weight and
  `pos_embedding` draws N(0, 0.02) from its own `torch.Generator`. The generator is seeded
  from sha256(`"<init_seed>:<parameter name>"`), keeping 63 bits. The residual 1/√(2L)
  scaling, the zero biases, and the LayerNorm ones/zeros are unchanged.
  - **What the old stream did:** consecutive draws from the global RNG, so adding a module
    shifted every later tensor.
  - **What it gives now:** a paired arm starts from its reference's exact weights under
    every shared name and shape. Tested for plain, static, smolgen and flatten-head d384×10.
- **`build_model` leaves the global RNG where it was** for independent init. It builds
  under `torch.random.fork_rng`, because the module constructors still draw PyTorch's
  default inits before they are overwritten.
  - **Consequence:** the dropout stream starts at `torch.manual_seed(init_seed)` whatever
    the architecture.
  - **Also fixed:** building the EMA eval model at a full eval no longer moves the dropout
    stream. Before, a resume across a stable checkpoint could shift later dropout masks.
- **Dropout draws from the global RNG seeded with `init_seed`.** The brief assigns dropout
  to neither seed, and init was the closer fit. With one seed pair on every arm, the choice
  does not matter for pairing.
- **`init: v5_deepcopy` is untouched.** It still draws from the global RNG and ignores `init_seed`.
- **`build_model(cfg, init_seed=0)`:** callers that load weights afterwards (EMA model,
  `score.py`, export, `load_for_inference`) keep the default.

### H2 — one OpenBLAS thread in the trainer

- **`train.py` sets `OPENBLAS_NUM_THREADS=1` (setdefault) before importing numpy.** It covers
  the trainer process, every DataLoader worker (they inherit the environment), and
  `bench.py` / `s3_check.py`, which import `train.py`. Nothing on these paths calls BLAS.
- **Measured** (`test_trainer_workers_run_one_blas_thread`): a spawned worker holds
  **1,280 → 881 MiB** private, a **399 MiB** drop. That is ≈ 3.2 GB at 8 workers,
  against the corpus report's ~4 GB estimate. The batches are byte-identical.

### H3 — named eval sets

- **`data/processed/evalsets/<name>.json`, built by `tools/build_evalsets.py`.** The records
  stay in their corpus. Each sidecar pins:
  - every shard's sha256 (the v2 manifest has no per-shard hashes);
  - the manifest's sha256;
  - the strata sidecar, checked against that manifest.

  `eval.load_evalset` refuses a changed shard. An existing sidecar is never overwritten.
- **Three sets:**
  - `frozen90` → `val_frozen_90m_v2` (452,405);
  - `v2val_roots` → corpus v2 `val` (571,232: 458,771 old tier + 112,461 new tier);
  - `v2val_derived` → `valderived` (161,340; 126,615 old + 34,725 new tier).

  The counts match the corpus report.
- **`frozen90` means the v2 copy from now on.** Every field it shares with v1 is
  byte-identical (S6), and it adds `hard_move` for the hard-label metrics. `score.py`'s
  `--v5-crosscheck` still hands v5's `score_baseline` the v1 copy, which v5's dataset can read.
- **New eval slice `tier/{old,new,v1}/{n,policy_kl,value_mse}`.** It serves B0: new-tier
  positions are 20% of `v2val_roots`, and the aggregate would dilute them.
- **Trainer wiring: `eval.extra_sets`** (default `[]`).
  - Each named set is evaluated for raw and EMA at every full eval.
  - Results are logged as `full_eval` events with `set=<name>`; frozen90 events now carry
    `set=frozen90`.
  - Extra sets never decide `best.pt`.
  - The proxy keeps the list empty: `screen.py` scores all three sets with `score.py`, and
    running them in the trainer too would repeat ~3 min of GPU per run.
- **`score.py` rewrite:**
  - `--weights raw|ema` and `--sets`;
  - it uses the model's own token scheme (it used to hard-code `v5_68`, which would have
    mis-scored A9);
  - it builds the model from the checkpoint's `model` section only, so checkpoints written
    before H1 still load;
  - its output nests every metric under `sets`, and keeps the first set's headline keys at
    the top level for `s2.py`;
  - NaN is written as null.

### H4 — queue driver (`tools/screen.py`, `training/v6/screen_queue.yaml`)

- **Arms compose.** An entry's `set` is applied on top of its `base` arm's full override
  list (default: the current best), and the arm is judged against `ref` (default: its base).
  The whole chain is stored in each ledger row, so a later arm reproduces any adoption.
- **The key is `base`, not the first draft's `on`.** YAML 1.1 reads `on:` as the boolean
  `True`, and the queue test caught it. `load_queue` now refuses any unknown key.
- **The driver process imports no torch** (measured: 1.28 GB private with the config and
  torch loaded). It sits idle for days, so config resolution runs in a
  `screen.py resolve` subprocess, like training, scoring and fwd_cost.
- **§4 as implemented:**
  - Δ is relative to the reference, positive is better.
  - Calibration r = |A0r − A0| / A0 on frozen90, per metric and weight set.
  - Primary weights are whichever of raw and EMA gives A0 the lower frozen90 total.
  - Floors: f_KL = max(r_KL, 0.25%) and f_MSE = max(r_MSE, 1.7%). PV-restricted KL gets
    its own floor, max(r_PVKL, 0.25%).
  - Adoption uses the brief's two-branch rule. An architecture arm's 100·ΔKL must also
    exceed its required gain.
  - Anything else keeps the reference. For every arm in this queue, the reference is also
    the cheaper option on a tie.
- **Hard-label arms use PV metrics.** Any arm training hard labels (`policy_hard.weight > 0`,
  which covers B1–B3 and a confirmation of a recipe with hard labels) is judged on
  PV-restricted KL and MSE. Top-1 vs SF's best move (`sf_top1_hard`) is reported as Δ but
  not gated, because the brief sets no floor for it.
- **Confirmation (§3.4).** The rule is applied per primary metric on which the recipe gained
  at seed s: its seed-t Δ against A0t must be positive and at least half the seed-s Δ.
  Metrics the recipe did not gain on at s are reported but not gated.
- **B0 keeps the adopted mixture and widens its pool** from the 90M roots to all v2 roots.
  If A8 is not adopted, that is the brief's natural mixture over roots. If A8 is adopted,
  its 0.75 / 0.25 policy / value-only split is kept over all roots. Dropping A8 there would
  change two things at once. The queue has both variants, selected by `only_if`.
- **A8's groups:**
  - `policy90` = `{in_90m: 1, origin: root, label: multipv}`;
  - `vonly90` = `{…, label: [hard_only, value_only]}`.

  "Value-only" means no policy; with 100% `hard_move` coverage those rows are all `hard_only`.
- **The queue pauses after B1b** (`phase3_review`), which creates HOLD. The pre-registered rules
  do not settle three things on their own:
  - B2 runs only if B1's best lost PV-restricted KL;
  - B3 needs B1's best `w_hard`, which may belong to an arm that was not adopted;
  - B4's shares depend on the mixture current at the time.

  Writing those as queue logic was more machinery than a review costs. The confirmation
  entries (A0t, Ct) are already queued after the pause.
- **After scoring**, the driver deletes the rolling checkpoints and `best.pt` (a duplicate
  of one weight set) and keeps the final `ckpt/s60000256.pt`, which holds raw, EMA and
  optimizer state. Mixture member lists (`mixture/*.npy`, 360 MB per run) are left in place;
  the brief's disk rule covers checkpoints only.
- **Stop conditions** each write `runs/screening/STOP`, and the driver will not relaunch
  while that file exists:
  - an anomaly event or non-finite step (checked after every attempt, and never restarted);
  - an arm failing its restarts;
  - a failed score or fwd_cost;
  - C: below 25 GB;
  - calibration r_KL > 0.5%;
  - an arm reached before calibration exists.
- **`system.allow_dirty: true` in `proxy.yaml`.** The tree carries untracked files that are
  not this work's (`benchmarking/…`, `after_s2.py`). Provenance lists them and `code.patch`
  holds the tracked diff, as in S2.

### H5 — `tools/fwd_cost.py`

- **Method = the campaign's A1:**
  - the forward is compiled with Inductor in default mode and captured as one CUDA graph
    per batch size;
  - bf16 autocast, with 20 warm-up and 200 timed replays;
  - batch sizes 24 and 128, on real frozen-val positions in each model's token scheme;
  - random keyed weights.

  The two engine Inductor settings (static launcher off, pointwise autotune off) are
  restated from `playing/v6/graphs.configure_inductor`, not imported, because that module
  imports the C++ core at the top.
- **Interleaved timing.** Plain d384×10, the arm and its reference are timed in one process,
  interleaved over 5 rounds, and the median is kept.
- **Conversion = the campaign's A4 exchange, not a bare log₂:**
  - required gain = log₂(1 + f(r − 1)) × 6.50% KL, with f = 0.908 (the GPU-bound share)
    and r the batch-24 ratio;
  - it reproduces A4's +4.21% for d384×10 (r = 1.624);
  - a bare log₂ would overstate small costs by about 10%.
- **Marginal cost is what gates.** The verdict uses the arm against its reference; the ratio
  against plain d384×10 is reported alongside. If an earlier architecture change was
  adopted, the reference already carries its cost.

## S7 (2026-09-27) — exports carry `value_scale`

- **`tools/export.py` records the training corpus manifest's `value_scale`** in the export
  (290.6806 for corpus v2). `load_for_inference` attaches it to the module as
  `value_scale`. The engine needs it for `cp = value_scale * atanh(value)`, which every
  resign and adjudication threshold reads, and refuses an export without one.
- **A manifest without one exports null**, with a warning. The frozen val set, which
  `tiny_cpu.yaml` trains on, has no `value_scale`. `test_export_round_trip` caught the
  first draft raising `KeyError` there.
- **The engine side** is recorded in the root `DECISIONS.md`, "S7".

## Phase 3 review (2026-09-29): B2, B3, B4

The queue paused at `phase3_review` after B1b, 05:50 UTC. The owner handed these calls over
("giving you the reins to make decisions here and log them"). They are written into
`screen_queue.yaml` above the confirmation pair. The best at the pause was **A3**:
dropout 0, smolgen, `canonical_65`, the A8 policy / value-only mixture and Muon.
Phase 3 so far:
- **B0** (all v2 roots): rejected. frozen90 was flat (+0.16% KL, −0.87% MSE), but new-tier
  positions on `v2val_roots` improved (+1.6% KL, +3.6% MSE).
- **B1a / B1b** (main-head hard labels): both rejected.

**B1's best is B1a (`w_hard` 0.25).** Against A3:

| arm | PV-restricted KL | MSE | top-1 vs SF | policy entropy |
|---|---:|---:|---:|---:|
| B1a (0.25) | −5.86% | +0.19% | +1.17% | 1.7490 → 1.7206 |
| B1b (0.5) | −12.10% | −3.54% | +0.43% | 1.7490 → 1.7130 |

B1a is better on every primary metric. The main-head hard labels sharpen the prior, which is
the §14 risk the doc names: SF top-1 and hard NLL improve while the PV-move distribution
drifts from the multi-PV target.

**B2 runs, because its pre-registered trigger fired:** B1's best lost PV-restricted KL.
- **Config:** `model.aux_policy_head=true`, `targets.policy_hard.head=aux`,
  `targets.policy_hard.weight=0.25`, on A3, judged against A3.
- **Metrics:** PV metrics, since hard labels are in the loss (the driver's default).
- **Not gated on inference cost.** The engine never reads the aux head, so a production
  export should drop it: `aux_policy_head=false` and no `aux_*` weights.
  - This is not built yet: `load_for_inference` builds whatever the export's config says, and
    the forward computes the aux logits when the head exists.
  - It is only needed if B2 is adopted.

**B3 decision: derived records at 10% of samples, on top of B2's aux-head hard labels.**
- **The pool.** B0 was rejected, so the pool stays the 90M roots, and the derived group is
  those roots' derived records: `{in_90m: 1, origin: [ply1, ply2]}`, 24,752,428 records.
- **Shares.** The root groups are scaled by 0.9 (policy90 0.675, vonly90 0.225), plus
  derived90 at 0.10.
- **Hard labels are required.** Derived rows carry only a hard label and a value, so
  without hard labels B3 would test them as value-only data. That misses the question §6.3
  asks.
- **The brief's "B1's best `w_hard`" is taken on B2's aux head, not the main head.** B1
  showed that the main-head configuration hurts PV-KL for a reason unrelated to derived
  data. B3 on it would be rejected for the known reason and teach nothing.
- **Two variants, chosen by `only_if`:**
  - *B2 adopted:* B3 builds on the best (B2) and is judged against it, which is one change.
  - *B2 not adopted:* B3 builds on B2's overrides and is judged against the best (A3),
    with its delta against B2 reported (`also: [B2]`). A B3 adoption then means
    "aux-head hard labels plus derived beats A3". The B2 delta shows how much of that is
    the derived data.

**B4 decision: one combined arm, as §13.2 specifies.**
- **Value loss:** exact-zero λ = 2 (`targets.value.stratum_weights.exact_zero`).
- **Value-only sampling rates:** 6–14 ×1.5 and mates ×0.5. 6–14 mates get ×0.75, both
  multipliers applied.
- **The multipliers are rates within the value-only share,** and the adopted split is kept:
  0.75 policy / 0.25 value-only, or 0.675 / 0.225 / 0.10 with derived.
  - Applying them to the whole mixture and renormalising would also move A8's adopted
    policy share, a second change in the same arm.
- **Cell counts** (in_90m value-only roots, train strata):

  | cell | records | rate | share of the 0.25 value-only budget |
  |---|---:|---:|---:|
  | 6–14, non-mate | 7,285,717 | ×1.5 | 0.073256 |
  | 6–14, mate | 3,900,463 | ×0.75 | 0.019609 |
  | other buckets, mate | 2,416,956 | ×0.5 | 0.008101 |
  | rest | 22,233,458 | ×1 | 0.149034 |

  Each budget's decimals sum to it exactly. The smallest group gets 3.7 samples per
  micro-batch.
- **Two variants:** with or without the derived group, chosen by whether B3 was adopted.
- **B4 builds on whatever is best,** so it inherits the hard-label settings as adopted.

**Validation before release.** Every branch was walked with the real ledger plus synthetic
verdicts: B2, B3 and B4 each adopted or not, 8 paths. On each path:
- the right variants are picked;
- every composed config resolves;
- the queue then reaches A0t and Ct, and ends.

Every distinct mixture was built against the real train strata: disjoint groups and the
expected sizes. `test_queue_walk_and_composition` now walks the Phase 3 entries.

**Found by that validation: YAML 1.1 reads `6_14` as the integer 614** (and `15_27` as 1527),
because underscores are allowed in YAML integers. It is the same class of trap as the `on:` key.
- The schema refused it with a clear error. The driver would have stopped at B4's config
  resolve, before any training.
- Bucket names are quoted in the queue now. They had never been used in a `where` clause before.

**Time left:** B2, B3 and B4 at Muon speed, then A0t and Ct. About 5.2 h each (A0t about
4.2 h), roughly 25 h. The confirmation match itself remains blocked on contract B.

## VM harness (2026-09-30): production driver and VM packaging

Brief: `docs/capacity/training/prod/vm_harness_brief.md`. Where `training_brief.md` differs
(it claims to supersede this brief; section and report names), the harness brief is followed,
on the owner's word. Operator steps and every store schema: `PROD_RUNBOOK.md`.

**Sample grid.** Every cadence must be a multiple of the 1,024 effective batch, because the
trainer's events fire on `done % every == 0`, and 300M, 150M and 10M are not.
- **Stable checkpoints** every 150,000,640 samples (146,485 steps, the first multiple ≥ 150M).
  **Branch points** are 2×, 4×, 8× and 16× that: 300,001,280 / 600,002,560 / 1,200,005,120 /
  2,400,010,240. Their decays are 52.9M / 105.9M / 211.8M / 423.5M, 3.195B samples in all.
- **Rolling checkpoints and quick-val** every 10,240,000, the grid the screening already used.
  The 40M sanity check reads 40,960,000, which is also on C0's 2,048,000 quick-val grid.
- **Warmup** is exactly 6,000,000; it needn't sit on the grid.

**One schedule for the whole main line.** `schedule.total_samples` = ceil(2,400,010,240 / 0.85),
so the main line holds peak LR through the last branch point. A stable segment is `train.py
--stop-at-samples <branch point>`: a clean exit that leaves the schedule alone. A normal
resume can't change `total_samples`, and the stop point is a run control, not a
hyperparameter. `branch_decay.py` computes the same total for the 2.4B branch, so the final
decay is just the last branch, with no special case.

**The memorization gap is a CLI flag (`--seen-eval GROUP:N`), not a config field.** Any new
schema field changes every existing config's plain form and hash. Old checkpoints (Ct's
included) would then be refused on resume, and `export --weights best` would refuse their
`best.pt`. The seen set is 200,000 roots of the `policy` group, drawn with
`default_rng([data_seed, 0x5EE7])`. The gap is logged for raw and EMA at every full eval,
with `valid` = the group finished a pass.

**Stop rule, as implemented:**
- **"Better of raw and EMA"** is taken separately for KL and for total, per branch.
- **"Anomaly" at the first branch** means `nonfinite` or `trainer_exit`. An upload lag, an
  ignored `control.json` or a sanity miss the operator chose to continue past doesn't stop
  the run.
- **After the last branch** the run always stops; the decision still records the rule's
  verdict.
- **Ship** is the lowest frozen90 total over every finished branch × {raw, EMA}.

**Store layout additions:**
- **Branch rolling checkpoints** go to `branches/<name>/rolling/` (last 2), so a VM lost
  12 h into the 2.4B decay resumes inside the branch rather than restarting it.
  `branch_decay.py --resume` continues from the branch's own latest checkpoint.
- `provenance_resume_*` / `code_resume_*` are uploaded on each resume.
- **Eval tags** are `evals/<segment>_{quick,full,memgap}_s<N>`, `decision_<branch>`,
  `timecap_<branch>` and `sanity_s<N>`.

**Integrity and ordering.** One upload thread, one FIFO queue: the checkpoint, then its
`.sha256`, then `state.json`. Before a checkpoint is queued, the log is read to its end and
the metrics are flushed, so the store never has a gap in the loss trace before a resume
point. That same ordering makes a pause asked for by the sanity check land on that check's
own checkpoint. Checkpoints are hard-linked into a staging directory beside the run dir, so
the trainer's rotation can't delete one mid-upload. Resume trusts only a checkpoint whose
sidecar exists and equals both `state.json` and the downloaded bytes; a local leftover run
directory is never trusted.

**Time cap.** The rate is trained samples (stable plus decay) over the driver's wall time
across sessions, so evals, compiles and restarts are counted: a conservative estimate. The
first segment always starts. The cap is checked before each later stable segment. The
operator raises it with `"time_cap_h"` in `control.json`.

**Control.** `control.json` is edge-triggered: it acts once per change of content.
- `pause` stops the trainer at its next rolling checkpoint or at the segment's end.
- The sanity check pauses by writing `control.json` synchronously, so a poll can't race a
  queued write.

**Production pool.** `{origin: root, label: multipv}` at 0.75 and
`{origin: root, label: [hard_only, value_only]}` at 0.25. This is A3's two groups without
`in_90m`, so `t20` policy roots fall into `policy` by label.

**VM packaging:**
- `requirements-prod.txt` holds the brief's pins plus the transitive pins from the local env,
  and `psutil` (for `test_m7`); scipy was dropped, since nothing imports it.
- torch comes through `uv pip install --torch-backend`, choosing cu129, cu128 or cu126 by
  the driver's CUDA version. Its metadata pins triton and the nvidia libraries.
- **boto3 is the one S3 client**, for both `setup.sh`'s parallel pull and the uploads.
- `setup.sh` reads RAM and vCPUs through cgroup limits: a RunPod pod is a container, and
  `/proc/meminfo` shows the host's RAM.
- `--cpu` turns the hardware checks into warnings for the dry run.
- `sha256.txt` paths are relative to `data/` (key = `data/<path>`). The pull verifies at pull
  time and leaves no copy in the tree, since that would dirty the checkout.

**Linux port.** No PowerShell launcher, `Start-Process` or priority code lives in
`training/v6` beyond `tools/screen.py`'s docstring (a local-box tool) and `tools/after_s2.py`
(a Windows one-off, untracked).
- The stale untracked `prod.sh` (L40S, 90M pool) is removed; `setup.sh` and the runbook
  replace it.
- The VM uses tmux, and the trainer runs at normal priority, since it is the VM's only job.
- Triton comes with torch; `descriptive_names=False` is kept.

**Dry-run data.** `tools/make_synth.py` carves 16,384 train and 2,048 val records out of
frozen90 v2, plus a value-negated copy. Training on that copy with value loss ×10 and policy
loss ×0.1 makes the total worse from branch to branch, which drives the "worse total" path
on purpose. The natural run also reaches "worse total" at its third branch, by overfitting.
