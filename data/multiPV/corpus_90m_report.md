# 90M policy-biased corpus rebuild — decisions and results

Companion to the generated `feasibility_scan.md` (the raw scan) and
`manifests/dataset_manifest_90m.json` (the build record). This file is the
argument: what the scan said, what was chosen because of it, and what came out.

Corpus: `data/processed/multipv_90m`, 192 train + 12 val shards, 31.95 GiB.
The 30M control at `data/processed/multipv` was not touched.

---

## 1. Two premises, checked

**Pass A was not re-run.** `index/pass_a_index.bin` (394,669,566 rows) was used
byte-for-byte. Confirmation that this was the right call is in §4: the verifier
independently reproduces the prior selection at exactly 29,994,655 lines, which
is only possible if the chunk boundaries — and therefore the index length — are
unchanged.

**Policy coverage is not a piece-count effect, so no stratified correction was
applied.** The scan confirms it at `value_min_depth 26 / policy_min_depth 20`:
coverage is 48.2 / 42.7 / 36.6 / 42.2% across `<=5 / 6-14 / 15-27 / >=28`. The
oversampling is therefore applied *within* each bucket, which leaves
`target_shares` untouched by construction rather than approximately — the
realised histogram lands at 1.01 / 35.27 / 40.00 / 23.73% against a target of
1 / 35 / 40 / 24.

---

## 2. `value_min_depth` stays at 26. The scan overrules the prior.

The brief's prior was "if depth 28 still clears 90M with headroom, take it".
It does not clear it. Per-bucket demand against the eligible pool:

| value_min_depth | `<=5` | `6-14` | `15-27` | `>=28` | total pool | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 24 | 12.26x | 1.75x | 2.69x | 1.73x | 203.4M | ok |
| **26** | **11.79x** | **1.52x** | **1.87x** | **1.05x** | **150.8M** | **ok** |
| 28 | 11.32x | 1.33x | 1.33x | **0.61x** | 114.9M | short |
| 30 | 10.80x | 1.18x | **0.96x** | **0.33x** | 89.9M | short |

The binding constraint is the `>=28` bucket, which is the *scarcest* pool at
every threshold and shrinks fastest with depth: 23.1M rows at 26 against a
21.9M demand — 1.05x, essentially exhausted — and only 13.4M at 28, a 39%
shortfall. Depth 28 loses 8.5M records outright and drops the policy ceiling
from 61.8% to 46.3%.

So the tension the brief flagged does not actually resolve in favour of cleaner
labels: at this target there is no depth above 26 that fits. Depth 24 fits
comfortably and would allow the full 65% policy share, but that is the wrong
direction — the whole motivation for the rebuild is that the value head is a
well-calibrated conditional mean whose residual variance is label noise, and
shallower labels are noisier labels. **Held at 26.** Nesting is preserved as a
result; nothing needed to be recorded as broken.

`policy_min_depth` held at 20. Dropping to 18 adds only 692k policy rows at
depth 26 (+1.1%, ceiling 61.8% → ~62.2%) in exchange for shallower policy
blocks. Not worth it.

## 3. `policy_share` 0.65 is the right ask, but the pool caps delivery at ~60%.

The policy pool at 26/20 is 60,692,417 rows against a demand of 0.65 × 91.355M
= 59.4M — 97.9% of every policy-bearing row in the dump. It does not fit
per-bucket, which is what matters:

| bucket | policy pool | policy demand @ 0.65 | outcome |
|---|---:|---:|---|
| `<=5` | 5,193,528 | 593,808 | fine (r = 0.114) |
| `6-14` | 20,728,798 | 20,783,263 | **clamps**, 54k short |
| `15-27` | 25,009,155 | 23,752,300 | fine, r = 0.950 |
| `>=28` | 9,760,936 | 14,251,380 | **clamps**, 4.49M short |

`>=28` is the casualty — not the `<=5` bucket the brief expected, which has
11.8x headroom and never binds. Its coverage caps at 44.5% no matter what is
requested.

Two things follow. First, the **coverage ceiling** at this target and share
vector is 61.75%, so 0.65 cannot be honoured and neither can anything above it;
requesting 0.65 lands 60.2%, within 1.5 points of the maximum the data can give.
Raising the request to 0.70 buys 1.2 points more and consumes the `15-27` policy
pool entirely; not worth the loss of value-only diversity in the largest bucket.
**Kept at 0.65 and reported the shortfall rather than hiding it.**

Second, a clamp had to be handled explicitly rather than left to
`min(want/avail, 1.0)`. The choice made — recorded as `policy_share_spill` in
the manifest, `--no-spill` to disable — is to make a bucket's unmet policy
demand up from its own value-only pool. This keeps the piece-count share vector
exact and lets coverage absorb the miss, which is the right way round: the share
vector is a modelling decision, the coverage is a preference. Without it `>=28`
would have under-delivered by 4.49M records and quietly violated its 24% share.
`selection_plan[*].spilled_to_value_only` records exactly how much moved
(4,490,444 in `>=28`, 54,465 in `6-14`).

The brief's other ceiling, `180M · (1 − c) ≥ value-only available`, is not
binding: 36.0M value-only records over 180M samples is ~2 views each.

---

## 4. Invariants

| invariant | result |
|---|---|
| Nesting | **strict superset**, 0 of 29,994,655 prior rows dropped |
| Val split | `sha1(fen) % 1000 < 5` unchanged; 4,129/4,129 prior-val records still val and still selected |
| Val leakage | 0 FENs on both sides across 818,794 distinct sampled FENs |
| D2 mate mapping | unchanged; violation rate 0.0425% against the 0.1% gate |
| D3 PV dedup | unchanged; 1,929,025 duplicates removed across 978,996 positions |
| D4 coverage on selected | unchanged (`post_selection_piece_histogram`) |
| D5 uniform draw | unchanged; still a random draw over the index, not a prefix |
| D1 `value_scale` | locked at 290.6806 |

Nesting is verified two ways: inside the build (`build_selection` replays the
prior mask against the same `u`) and independently by
`verify_corpus_90m.py`, which reconstructs both selections from the manifests
alone. The two-rate scheme required `r_policy` and `r_value_only` to each
dominate the old single rate in every bucket; they do, by wide margins
(worst case `<=5`: 0.0573 vs 0.0279).

## 5. Source-order diagnostic (§7 of the brief)

Line-number quantiles as a fraction of the dump:

| population | count | q05 | q25 | q50 | q75 | q95 |
|---|---:|---:|---:|---:|---:|---:|
| eligible, policy-bearing | 60,692,417 | 0.025 | 0.195 | 0.494 | 0.739 | 0.948 |
| **selected, policy-bearing** | 54,832,831 | 0.027 | 0.199 | 0.498 | 0.740 | 0.948 |
| eligible, value-only | 90,081,330 | 0.058 | 0.294 | 0.533 | 0.781 | 0.958 |
| **selected, value-only** | 36,517,803 | 0.064 | 0.295 | 0.531 | 0.773 | 0.954 |

**Read it as: the oversampling did not narrow anything, but it did widen a gap
that was already there.** Selected tracks its own eligible pool to within 0.005
at every quantile, so heavy policy sampling has *not* concentrated the selection
into a policy-dense band — the risk the brief raised does not materialise.

What the table does show is that the two *pools* differ in provenance:
policy-bearing rows sit systematically earlier in the file (q05 0.025 vs 0.058,
q25 0.195 vs 0.294). Since the corpus is now 60% policy-bearing rather than 40%,
the corpus as a whole has shifted earlier in source order, and the policy and
value supervision are drawn from measurably different regions. Given no engine
version field, that is a provenance difference of unknown sign. It is a
first-order effect only if SF-version mix varies along the file; nothing here
measures that, and nothing else in the pipeline would have surfaced the shift at
all. Flagged, not corrected.

---

## 6. Handoff (§9)

| quantity | 30M control | **90M rebuild** |
|---|---:|---:|
| train records | 29,555,094 | **90,076,117** |
| val records | 147,963 | **452,405** |
| selected lines | 29,994,655 | 91,350,634 |
| policy coverage | 40.4% | **60.2%** |
| value-only records | 17,720,968 | 36,017,123 |
| piece histogram `<=5 / 6-14 / 15-27 / >=28` | 1.01 / 35.31 / 39.96 / 23.73% | 1.01 / 35.27 / 40.00 / 23.73% |
| value strata zero / mate / middle | 21.01 / 20.15 / 58.84% | **21.48 / 20.50 / 58.01%** |
| invariant violation rate | 0.0278% | 0.0425% |
| shards | 64 + 4 | 192 + 12 |

Value strata moved as predicted, by about half a point each — policy
oversampling reweights the position mix. The stratified value metrics are on
slightly different mass than the baseline's; report against these shares.

Carried forward unchanged: eff batch 1024, `max_lr 3.5e-4`, 2 epochs, train
mirror 0.5 / val mirror 0.0. 90.08M × 2 = 180.15M samples = 175,929 steps at
eff batch 1024, against the baseline's 175,781 — matched to 0.08%.

Three things the training brief needs to settle:

1. `training/v5_multiPV/configs/base.yaml` still points `shards:` at
   `data/processed/multipv`. The Dataset globs `{split}_*.bin`, so 192/12 shards
   need no code change — only that path.
2. The val set is now 452,405 records and the old 147,963 are a subset of it.
   Evaluate both models on the **same** set — old subset or new whole — never one
   metric from each.
3. Whether the subset-permutation machinery is still needed: 2 epochs over the
   full corpus subsets nothing.

Judge the result by head-to-head against the 20M ep9 net at matched sims, not by
val loss.

---

## 7. Operational notes for the next rebuild

- **Worker count is now the bottleneck, not decompression.** The brief's
  `13,230 / 816 ≈ 17 workers` is right, and this machine has 16 cores, so 15
  workers was the ceiling. Realised 51,781 rec/s / ~226k source lines/s against
  a ~420k lines/s decompression capability: the run was worker-bound throughout
  and took 29.1 min. On >=18 cores it would be decompression-bound at ~14 min.
- Peak RSS ~4 GB, in `build_selection` (91.4M int64 line numbers plus the
  concatenate). Not the converter.
- 31.95 GiB written; 3.05x the 30M corpus's 10.48 GiB, as `record_size_bytes`
  (379, read from the manifest, not estimated) predicts.
