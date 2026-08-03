# GuoFish v5 — multi-PV student (~10.9M)

Training for the 10.9M-parameter transformer distilled from the multi-PV
Stockfish corpus in `data/processed/multipv`. Fresh init, no warm start.

The labels here are **distributions**, not best moves. The policy loss is a KL
against the whole 4096-wide target; nothing in this directory takes an argmax of
a label. That is the entire point of the dataset.

---

## Reproducing this run

```bash
# 0. tests (no pytest installed here; the runner is standalone)
python training/v5_multiPV/tests/run_all.py

# 1. pick the micro-batch empirically on this GPU
python training/v5_multiPV/bench_batch.py --json-out bench.json

# 2. re-derive max_lr at THAT effective batch (terminal mode — it destroys the weights)
python training/v5_multiPV/train_v5.py --config training/v5_multiPV/configs/base.yaml \
    --lr-range-test --micro-batch 512 --accum-steps 2 \
    --lr-range-plot models/guofish5/logs/lr_range.png

# 3. smoke run
python training/v5_multiPV/train_v5.py --config training/v5_multiPV/configs/base.yaml \
    --subset 50000 --epochs 1 --run-name smoke

# 4. full run
python training/v5_multiPV/train_v5.py --config training/v5_multiPV/configs/base.yaml
```

Steps 1→2 are ordered, not interchangeable: the LR is only meaningful for a
stated effective batch, so the batch has to be settled first. Every checkpoint
records `effective_batch` and `max_lr` together for exactly this reason.

Resume mid-epoch (dataloader position, optimizer, scheduler, RNG):

```bash
python training/v5_multiPV/train_v5.py --config training/v5_multiPV/configs/base.yaml \
    --resume models/guofish5/v5_10.9M_last.pt
```

Live monitoring — the JSONL is one event per line, flushed immediately:

```bash
tail -f models/guofish5/logs/<run-name>.jsonl
```

Event types: `run_start`, `micro`, `step`, `val_quick`, `epoch`, `checkpoint`,
`coverage_drift`, `lr_range`, `run_end`.

---

## Measured on this box (RTX 5070, 11.94 GiB, Windows/WDDM)

### Batch size — `bench_batch.py`

Throughput is **flat** across a 8× range of micro-batch, then falls off a cliff.

> **Read the sub-cliff rows as "flat", not as a ranking.** This box drifts:
> re-measuring one identical config later in the same process gave 6,034 then
> 4,941 samples/s, an 18% swing, while the 3 repeats *within* each timing block
> agreed to 0.15%. That is slow thermal/clock drift, and it is larger than every
> difference between the sub-cliff rows. So "192 is the peak" is noise; what is
> robust is the flat region and the cliff, which is a 47-91% collapse.

| micro | peak alloc | peak reserved | % VRAM | samples/s | steps/s |
|---|---|---|---|---|---|
| 128 | 1.08 GiB | 1.15 GiB | 9.6% | 4,474 | 34.95 |
| 192 | 1.52 GiB | 1.63 GiB | 13.6% | **4,599** | 23.96 |
| 256 | 1.97 GiB | 2.08 GiB | 17.4% | 4,523 | 17.67 |
| 384 | 2.83 GiB | 2.92 GiB | 24.5% | 4,377 | 11.40 |
| 512 | 3.71 GiB | 3.95 GiB | 33.1% | 4,376 | 8.55 |
| 768 | 5.47 GiB | 5.60 GiB | 46.9% | 4,383 | 5.71 |
| 1024 | 7.24 GiB | 7.40 GiB | 62.0% | 4,381 | 4.28 |
| 1536 | 10.77 GiB | 11.18 GiB | **93.7%** | 2,459 (−47%) | 1.60 |
| 2048 | 14.28 GiB | **15.09 GiB** | **126.4%** | 646 (−86%) | 0.32 |

**The WDDM spill is real and was reproduced.** At micro-batch 2048 the process
reserves 15.09 GiB on an 11.94 GiB card — 126% of physical VRAM — and *does not
raise OOM*. It silently pages to shared system memory over PCIe and throughput
falls 86%. A sweep that only watches for `torch.cuda.OutOfMemoryError` sees
nothing wrong here.

**`torch.compile` does not move the cliff, and makes falling off it worse.**
Measured on both paths:

| micro | eager reserved / samples/s | compiled reserved / samples/s |
|---|---|---|
| 1024 | 7.40 GiB (62.0%) / 4,381 | 7.64 GiB (64.0%) / 6,128 |
| **1536** | 11.18 GiB (93.7%) / 2,459 (−47%) | 11.40 GiB (95.5%) / 1,139 (**−81%**) |
| 2048 | 15.09 GiB (126.4%) / 646 (−86%) | 14.84 GiB (124.3%) / 573 (−91%) |

Peak *allocated* is near-identical on both paths (10.76 vs 10.77 GiB at 1536),
so fusion does not shrink activation memory for this model — and compiled
consistently *reserves* 2–3% more, which is the number that pressures the
driver. Past the cliff the compiled path is roughly half the speed of eager
(1,139 vs 2,459 at 1536): fused kernels hold larger intermediate buffers, so
more of the working set ends up paged. Compile is faster below the cliff and
slower above it, which makes headroom worth *more* under compile, not less.

`bench_batch.py` therefore stops on the **throughput knee**, and distinguishes
three things that look alike in a throughput column: a *plateau* (no gain,
low memory — the GPU is simply saturated), *saturation* (a few percent lost to
worse tiling), and a *spill* (a collapse **and** high memory pressure). A 5%
wobble at 62% reserved is not a spill, and calling it one is crying wolf.

Chosen: **micro-batch 512, accum 2 → effective batch 1024** (see the
torch.compile section; 512 was chosen with compile enabled).

The reasoning is deliberately *not* "512 was fastest". An interleaved
512/1024/512/1024 run put the two within 6.5% on the means while the same
config drifted 18% between its own two passes — the difference is unresolvable
here. Two things settle it instead:

- once accumulation targets a fixed effective batch, micro-batch does not change
  the optimizer-step count at all, so the tool's (b) rule ("smaller is cheaper
  in optimizer steps") has no force;
- **headroom does** — 512 sits at 32.9% reserved (2.9× below the cliff), 1024 at
  64.2% (1.5×), and falling off the cliff costs −81% under compile.

256/accum 4 is equally defensible. Anything ≥ 1536 is not.

Loader at micro-batch 256: **18,150 samples/s** vs GPU 4,523 — 4× headroom, so
the loader is not the constraint.

At ~4,500 samples/s eager, 10M positions is ≈37 min/epoch, ≈5.6 h for 9
epochs. **As configured** (compiled, micro 512 × accum 2) the run measures
5,830–5,880 samples/s in steady state: **9,766 optimizer steps and ≈28 min
per epoch, ≈4.2 h for 9 epochs.** The first log line of each epoch reports
`ETA pending` rather than a number — its window includes DataLoader worker
spawn and the one-off graph compilation, so extrapolating from it produces a
wildly wrong multi-hour figure.

### `torch.compile` — works, +10%, with three Windows landmines

Enabled by default (`compile: true`); disable with `--no-compile`. Getting there
took three separate fixes, all of which will recur on a fresh machine:

**1. Triton must match torch, exactly.** torch 2.8.0 imports
`triton.compiler.compiler.triton_key`, which Triton **3.7** removed (it is
`get_cache_key` now) — so the newest `triton-windows` fails with an `ImportError`
that says nothing about versions. torch 2.8 pairs with **triton 3.4.x**:

```bash
pip install "triton-windows==3.4.0.post21"    # for torch 2.8.x
```

If torch is ever upgraded, move Triton with it (2.9 → 3.5, and so on).

**2. Windows `MAX_PATH` (260 chars).** Inductor writes each kernel to

```
<root>/triton/0/<52-char hash>/tmp.pid_<50 chars>/<kernel name>.source
```

and this model's fused kernel names reach ~90 characters
(`triton_per_fused__to_copy_add_clone_embedding_native_dropout_native_layer_norm_0`),
so **~190 characters are spent before the root even begins**. The default root
`%TEMP%/torchinductor_<username>` lands right on the limit, and anything deeper
sails past it. `LongPathsEnabled` is `0` on this machine.

The failure surfaces as `FileNotFoundError` on a temp file inside the cache —
nothing mentions path length — and it only appears once a graph is big enough to
generate long kernel names, so a two-layer smoke model compiles fine and hides
it. Both `train_v5.py` and `bench_batch.py` now set a short
`TORCHINDUCTOR_CACHE_DIR` (`%USERPROFILE%\ti`) before importing torch, unless one
is already set. A username containing a space is *not* the cause — that was
ruled out by testing a space-free path that still failed.

**3. `cl.exe` and `LIB`/`INCLUDE`** must be present, i.e. MSVC + Windows SDK
installed and on the environment. Note that a shell started *before* those
variables were set will not see them.

**It is numerically faithful.** Same weights, same batch, eager vs compiled:

| precision | worst relative gradient difference |
|---|---|
| fp32 | **2.5e-6** |
| bf16 autocast | 9.0e-3 |

The fp32 agreement is the meaningful one: compile does not change the math. The
0.9% under bf16 is bf16's own ~0.4% per-element precision amplified through
reductions that fusion reorders — expected, not a defect. Policy KL differs by
2.1e-5 and value MSE by 4.3e-5.

**Speedup — measure this in one process, not across runs.** Comparing separate
`bench_batch.py` invocations suggested +17% to +27%; those numbers were wrong,
contaminated by run-to-run variance (each invocation sees different clocks and
thermal state). Back-to-back in a single process, 3 repeats of 30 steps, medians:

| micro | eager | compiled | speedup |
|---|---|---|---|
| 128 | 4,057 | 4,207 | 1.04× |
| 256 | 4,350 | 4,561 | 1.05× |
| 512 | 4,263 | **4,780** | **1.12×** |

Best-to-best — eager at 256 (4,350) → compiled at 512 (4,780) — is **+9.9%**,
about 30 minutes off a 5.6 h run. Compile also flattens the micro-batch curve:
eager gets *slower* from 256→512, compiled gets faster. Hence
`micro_batch: 512, accum_steps: 2` (effective batch still 1024, 33% of VRAM, far
below the 1536 spill point). 256/accum 4 is within noise and equally fine.

Costs: one-time graph compilation (~40–60 s) plus a recompile whenever an input
shape changes — validation's ragged final batch triggers a couple, then it
settles.

### LR range test — `--lr-range-test`

500 steps, 1e-5 → 5e-3, at effective batch 1024:

- smoothed-loss span 0.244 (11.8% of the minimum — enough signal to read)
- **minimum at 3.57e-4**; usable band ≈ 3.6e-5 .. 3.6e-4
- loss degrades above ~5e-4; no hard divergence up to 5e-3
- an earlier 250-step sweep over 1e-6..1e-2 put the minimum at 5.2e-4, so treat
  the optimum as **≈3.5e-4 – 5e-4**, noisy at this step budget

**Verdict: 5e-4 does not hold for this smaller net at effective batch 1024** —
it sits at or just above the loss minimum rather than inside the band.
`configs/base.yaml` ships **`max_lr: 3.5e-4`**, which is both the measured
minimum and where the batch-scaling heuristic lands.

Two estimator notes, because the first run got both wrong. Steepest descent is
fitted over a window and restricted to the region *at or before* the minimum —
an unwindowed whole-sweep search finds the largest one-step drop, which past the
minimum is just the recovery after a noise spike, and it returned 8.6e-3. And a
flat sweep now warns rather than reporting a "minimum" that is noise.

### Source order is not exchangeable — `--subset` is a permutation prefix

The Pass B shards are not in a random order. **Every shard is one sweep from
opening-like positions to endgames**, and the pattern repeats identically in
every shard (verified in shards 0, 7 and 30). Walking within-shard offsets:

| position in shard | coverage | mate | exact-zero | **mean pieces** | n_legal |
|---|---|---|---|---|---|
| head | 0.75 | 0.14 | 0.14 | **23.6** | 29.7 |
| middle | 0.19 | 0.22 | 0.16 | 17.9 | 26.0 |
| tail | 0.35 | **0.39** | **0.36** | **16.2** | 23.9 |

Piece count correlates with coverage at r = +0.83. So `has_policy` coverage is
just the *loudest* drifting property, not the only one — game phase and the
value strata move far more.

The consequence is entirely about **where a prefix ends**. A prefix that lands on
a whole number of shards is fine, because each shard is a complete sweep; a
prefix that ends mid-shard is a different corpus. Measured against the full
corpus across ten statistics:

| subset | coverage | mate | mean pieces | ≥28 pieces | worst deviation |
|---|---|---|---|---|---|
| full corpus | 0.402 | 0.203 | 18.89 | 0.236 | — |
| prefix 10M (21.6 shards) | 0.404 | 0.199 | 18.99 | 0.239 | **2.5%** |
| permuted 10M | 0.405 | 0.201 | 18.99 | 0.238 | 2.8% |
| **prefix 50k** (shard fragment) | 0.576 | 0.180 | 21.09 | **0.380** | **75%** |
| permuted 50k | 0.406 | 0.203 | 18.93 | 0.240 | 12% |

`--subset N` therefore selects a prefix of **one seeded global permutation** of
the corpus (`select_subset`, `--subset-seed`, default 20260802), not a prefix of
source order. That buys three things:

- **Representative by construction** at any N, instead of by an undocumented
  accident of shard-boundary alignment.
- **Nested subsets**: 5M ⊂ 10M ⊂ 20M exactly, for a fixed seed. A scaling study
  then varies data *quantity* with composition held fixed, instead of
  confounding quantity with whatever the second tranche happened to contain.
- **Usable smoke runs.** `--subset 50000` was previously an opening-heavy
  fragment (0.380 of its mass in ≥28-piece positions vs a corpus 0.236); it is
  now a miniature of the real run.

`--subset-seed` is deliberately separate from `--seed`: which records a run sees
should not change because the weight init changed. Both travel in every
checkpoint alongside `subset_size` and `corpus_size`.

Two consequences worth stating plainly:

- **The LR range test does not need redoing.** It ran at 10M, where a source
  prefix already matched the corpus to 2.5% on every statistic measured — so the
  permutation changes that run's composition negligibly.
- **The working set is now all 68 shards** (10.48 GiB) rather than the first
  third. Measured, with the real sampler and collate at micro-batch 512:
  prefix 28,663 samples/s, permuted 26,908 (first pass) and 28,094 (second) —
  a ≤6% cost, still **4.4× the GPU's ~6,100 samples/s demand**. This box has
  18.5 GiB free, so the shards sit in page cache. On a RAM-tight machine this
  would need rechecking.

### Coverage is measured, never assumed

`C = effective_batch * coverage`, so the coverage figure directly scales the
policy gradient. It is **always measured on the records actually selected**
(200k sampled, sorted into a forward scan, ~2 s) — there is no mode that
substitutes the corpus-wide figure, because there is no situation in which
guessing beats measuring the thing you are about to train on.

With the permutation in place, measured coverage should land on the corpus
figure at *any* subset size — which turns it into a free self-check. The 50k
smoke run measures 0.3996 against a corpus 0.4034 (ratio 0.9905); before the
permutation it measured 0.5741 (ratio 1.42). If it ever drifts again, the
warning says what that actually implies: the permutation is not reaching the
sampler, which means composition is skewed far beyond coverage alone.

Three independent guards: the startup ratio check, the rolling `[WARN]` every
`--drift-window` (200) effective batches, and a realised-vs-expected line in
every epoch summary — the last is what catches a short run, since 200 windows
never accumulate in one.

---

## Files

| file | what it owns |
|---|---|
| `model_v5.py` | the architecture + `ModelConfig`; checkpoint→config recovery for the engine |
| `losses.py` | masked KL, value SE, the policy denominator. Returns **sums**, never means |
| `metrics.py` | stratified value metrics, mirror consistency, policy top-k |
| `train_v5.py` | training loop, LR range test, logging, checkpointing, resume |
| `bench_batch.py` | VRAM / throughput sweep to choose `--micro-batch` |
| `configs/base.yaml` | this run's config |
| `tests/` | the properties that fail silently if broken |

Nothing the manifest states is hardcoded. `value_scale`, the policy coverage and
the value strata are read from
`data/multiPV/manifests/dataset_manifest.json` at startup and travel into every
checkpoint.

---

## Two deviations from `training/train.py`, both deliberate

`train.py`'s `ChessTransformer` is the ancestor of this model and the engine's
`ChessTransformerV2` matches it exactly. Two things in it are wrong for v5:

**1. No final LayerNorm.** `nn.TransformerEncoder(layer, n)` defaults to
`norm=None`. With `norm_first=True` (Pre-LN) that leaves the residual stream
unnormalised at the output, so both heads read a stream whose scale grows with
depth. v5 adds `final_norm` (a `nn.LayerNorm(d_model)`) after the last encoder
block and before both heads. `tests/test_model.py` checks it is in the path for
both heads, not just present. Disable with `--no-final-norm`.

**2. ReLU, not GELU, in the FFN.** `nn.TransformerEncoderLayer` defaults to
`activation=relu` and `train.py` never overrides it — so every previous GuoFish
generation has a **ReLU** FFN despite a GELU value head. The v5 brief specifies
GELU, and `model_v5.py` passes it explicitly.

Both add state-dict keys / change semantics relative to older checkpoints, so
**the engine's loader needs one change** before it can load a v5 checkpoint.
`playing/v5/playv5.py:load_model` currently hardcodes `ModelClass()` at
d_model=512 / 8 layers and infers only `seq_length` from `pos_encoder`. A v5
checkpoint carries its own `config` dict; use it:

```python
from training.v5_multiPV.model_v5 import load_from_checkpoint  # config-aware
model = load_from_checkpoint(ckpt)          # falls back to shape inference
```

`model_v5.config_from_checkpoint` also reconstructs the config from a bare state
dict (including whether `final_norm` exists), so there is one code path for both
old and new checkpoints. **This change has not been made — `playing/` is
untouched by this work.**

---

## The two places silent ELO is lost

### Masking

The legal mask is applied to the logits **before** `log_softmax`, with `-inf`,
which is exactly what the engine does at inference
(`ChessTransformerV5.forward`'s `legal_move_mask` branch). Masking after the
softmax instead renormalises over all 4096 and then zeroes, leaving every legal
probability too small by the illegal mass — a train/inference mismatch that
never shows up in a loss curve.

`losses.masked_log_softmax` is the single implementation. Tests pin that illegal
moves get *exactly* zero probability, that the result equals a `log_softmax`
computed over the legal subset alone, that post-hoc masking does **not** match,
and that the engine's `forward(x, mask)` path and the training path agree to
1e-6 on the same weights.

### Normalisation

Every micro-batch divides its policy KL by the **same constant**:

```
C = effective_batch * coverage        # coverage = 0.403396, from the manifest
```

Never by its own realised `has_policy` count. Because `backward()` accumulates
linearly, the accumulated gradient is `(sum of KL over the effective batch) / C`
— the per-policy-record mean at the expected coverage, identical whether the
effective batch arrived in 1 chunk or 8. Dividing each micro-batch by its own
count produces a mean-of-means: records landing in a sparse micro-batch get
up-weighted, and the effective policy LR jitters with per-batch coverage noise.

`tests/test_accum.py` pins accum=1 ≡ accum=4 ≡ accum=8 to <1e-6, and includes
the negative control — per-micro-batch count normalisation must **fail** that
test, otherwise the invariance test proves nothing.

The value loss divides by `micro_batch * accum_steps`. A short final
accumulation window (or a short final micro-batch under `--no-drop-last`) is
sized explicitly by `plan_epoch` rather than discovered at runtime, so no window
can silently reweight itself by a constant it did not actually see.

Realised vs expected `has_policy` is logged per micro-batch and per effective
batch. Sustained drift beyond `--drift-tol` (5%) over `--drift-window` (200
effective batches) warns — that is the signal that `--subset` has different
coverage than the manifest's global 40.34%.

`--policy-norm count` reproduces the original per-batch-count rule. It is
**not** accumulation-invariant and the script warns when it is combined with
`--accum-steps > 1`.

### `has_policy = 0` (59.65% of records)

These contribute exactly zero policy gradient and full value gradient. Masking
by multiplication is not sufficient: the all-zero target meets a `-inf`
log-probability, so the naive `target * log q` is `0 * -inf` → NaN in the
forward, and a `torch.where` written after the fact still propagates NaN through
the untaken branch in the backward. `losses.py` masks `log q` itself, before the
multiply, so the `-inf` never meets a zero. The test asserts the surviving
gradient is *bit-identical* to training on the `has_policy=1` subset alone.

---

## Validation is deterministic

**The colour mirror is OFF on the val split** (`--val-mirror-prob 0.0`), and must
stay off for every subsequent run — stochastic val augmentation makes val loss
incomparable across runs. Train-time mirroring stays at 50% per sample.

The mirrored half of the world is covered instead by the **mirror-consistency
diagnostic**, which is exact rather than sampled and free (the mirror is an
involution). On a fixed 10k-record held-out slice, every position is run both
ways and checked for:

- `v(mirror(x)) == -v(x)` → reports mean and max `|v(x) + v(mirror(x))|`
- `argmax(mirror(x)) == POLICY_PERM[argmax(x)]`, where
  `POLICY_PERM[from*64+to] = (from^56)*64 + (to^56)` → reports top-1 agreement

It also reports the **mirror-balanced prediction mean**,
`mean (v(x) + v(mirror(x))) / 2`. That is the deterministic version of
"predictions should sit near 0 with mirroring on": it is identically 0 for a
perfectly equivariant head regardless of how skewed the corpus is (the labels
sit at +0.077 pre-mirror). Drift means the model is leaning on absolute board
orientation — the exact capacity waste the augmentation exists to prevent.

---

## Metrics: value MSE is never headlined in aggregate

21.0% of targets are exactly zero and 20.2% are mates pinned at ±0.995. A head
collapsing toward zero scores well on a fifth of the data by doing nothing, and
that is invisible in an aggregate number. Every value metric is reported per
stratum, defined off the raw `value_cp` exactly as `pass_b_convert.py` counted
them (so they are mirror-invariant and survive a change of value scale):

| stratum | definition | share |
|---|---|---|
| exact-zero | `value_cp == 0` | 21.0% |
| mate | `abs(value_cp) >= 29000` | 20.2% |
| middle | everything else | 58.8% |

Per stratum: MSE, Pearson r, prediction std, label std. Plus overall prediction
std (Phase 2 tracked 0.352 → 0.501; the middle-only **label** std is 0.510).
Pearson r is `NaN`, not 0, when a prediction is constant — the honest answer for
a collapsed head.

**Epoch boundaries only** (never mid-epoch): stratified value metrics,
mirror-consistency, policy top-5.
**Mid-epoch** (`--val-every`, default 2000 steps, on the same
`--val-subset 20000` records every time, kept under ~5 s): policy KL, policy
top-1, value MSE, value prediction std.

---

## Checkpoints

Written every `--ckpt-every` steps (default 5000), at every epoch boundary, and
on best val — plus `_last.pt` for resume and an emergency save on SIGINT or an
unhandled exception. Step checkpoints are pruned to the most recent
`--keep-step-ckpts` (3).

Each carries: `model_state_dict`, `config` (the full `ModelConfig`), `epoch`,
`step`, `micro_in_epoch`, `samples_seen`, optimizer + scheduler state, RNG
state, `manifest_hash` (sha256 of the manifest file), `manifest_path`,
`git_sha`, `value_scale`, `coverage`, `effective_batch`, `micro_batch`,
`accum_steps`, `max_lr`, the loss weights, the full train config, metrics, and
`best_val`. All entries are `weights_only=True`-loadable.

Best-val is tracked on the **full** val split at epoch boundaries, using
`policy_weight * policy_kl + value_weight * value_mse`. The mid-epoch curve is
for watching, not for selection — it is a 20k subsample and not comparable to
the 148k full split.

### Resume caveats

- **The scheduler is rebuilt, not restored.** `OneCycleLR.state_dict()` bakes in
  `total_steps` and its phase boundaries, so `load_state_dict` silently
  overrides the current run's `--epochs`/`--max-steps` with whatever the
  checkpoint was written under — which either follows the wrong curve or hard-
  fails with `Tried to step N times`. The LR is a closed-form function of the
  step count, so it is replayed exactly from `last_epoch=step-1`. A mismatch
  against the checkpoint's `total_steps` warns.
- **Mirror augmentation resumes statistically, not bit-exactly.** The
  train-time mirror draws from each worker's RNG, which the DataLoader reseeds
  per iterator from the main process. The record ORDER is exact (the sampler is
  a seeded per-epoch permutation with an index-space skip, so resuming does not
  re-read skipped records); which 50% get mirrored is not.

---

## Tests

`python training/v5_multiPV/tests/run_all.py` — 48 tests, no pytest required
(each module also runs standalone and is pytest-compatible).

- `test_losses.py` — masked-KL correctness against a dense reference; masked
  logits contribute exactly zero probability; mask-before-log_softmax (with a
  negative control); engine-forward vs loss masking agreement; `has_policy=0`
  → zero policy gradient, no NaN, bit-identical to the subset; the loss is not
  argmax-based; normalisation by count under known coverage; the constant
  denominator's immunity to coverage jitter (with the `count` negative control);
  target mass outside the legal mask is caught rather than silently `+inf`.
- `test_accum.py` — accum 1 ≡ 4 ≡ 8 to <1e-6 with deliberately uneven
  per-chunk coverage; the negative control; `plan_epoch` window sizing including
  short final windows and short final micro-batches.
- `test_mirror.py` — involution of all three LUTs; the `(from^56)*64 + (to^56)`
  formula; batch-level `color_mirror` round-trip on every field; partial-mask
  isolation; the diagnostic returns the ideal numbers for a provably
  equivariant stub and flags an orientation-dependent one.
- `test_model.py` — final LayerNorm present *and in the path for both heads*;
  GELU not ReLU; parameter count; the policy head reproduced by hand from the
  encoder output; the value head reads CLS at 67; checkpoint→config recovery;
  stratification matches `pass_b_convert.py`; a collapsed head is caught by the
  strata but not the aggregate.

One note on the equivariant stub: the mirror identifies square `sq` with
`sq^56`, so **any** equivariant function must assign equal logits to two
mirror-paired, mirror-equivalent squares (e.g. two empty squares on the same
file) — a genuinely tied argmax. Tied rows are dropped before the top-1
assertion, because otherwise the comparison measures tie-breaking rather than
equivariance. A trained network hits this with probability zero.
