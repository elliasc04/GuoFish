# Value scale refit - decision table

> **DECISION (2026-08-02): the scale stays at 290.6806.** The refit was rejected.
> The "overall std 0.5" fit returns 849.70, which maps a +100cp edge to v=0.117 -
> a 55.9% expected score for being a clean pawn up. That is miscalibrated, not
> merely flat; 290.6806 puts the same edge at 66.6%. 290.6806 is the Lc0 WDL
> calibration constant (v ~ expected score), which is what MCTS Q backup and the
> tuned C_PUCT / VALUE_LOSS are denominated in. 11.5% cp saturation is accepted.
>
> The decision is reversible without reconversion: `value_cp` stores the raw
> White-POV score, so the scale is a collate-time transform (`dataset.revalue`).

- sample: **2,000,000** selected positions (1,599,157 cp / 400,843 mate / 0 unusable)
- value_min_depth: 26

## The complication: mates dominate the variance

- mate fraction: **20.04%** of value targets
- empirical mate sign split: **59.92% positive** (240,194+ / 160,649-) - measured, not assumed 50/50
- mates are pinned at +-0.995 regardless of scale, so they contribute FIXED variance 0.1984 = **79.4% of the 0.5 target's budget** from 20.0% of the data
- floor: as scale -> infinity the overall std cannot fall below **0.4437** (mates alone)

Fitting *overall* std therefore does not spread the centipawn values, it shrinks them until the mates stop overshooting. Read the `cp std` column, not just `overall std`.

## Decision table

| fit | scale | overall std | cp-only std | +100cp | +496cp (p95) | sat. cp | sat. all |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy (Lc0 WDL calibration) | **290.68** | 0.590 | 0.438 | +0.331 | +0.936 | 11.5% | 29.2% |
| fit incl. mates -> overall std 0.5 | **849.70** | 0.500 | 0.261 | +0.117 | +0.525 | 1.9% | 21.6% |
| fit cp-only -> cp std 0.3 | **667.20** | 0.517 | 0.300 | +0.149 | +0.631 | 2.1% | 21.8% |
| fit cp-only -> cp std 0.35 | **496.48** | 0.541 | 0.350 | +0.199 | +0.761 | 3.3% | 22.7% |
| fit cp-only -> cp std 0.4 | **369.41** | 0.568 | 0.400 | +0.264 | +0.872 | 7.0% | 25.6% |
| fit cp-only -> cp std 0.5 | **188.46** | 0.627 | 0.500 | +0.486 | +0.990 | 17.7% | 34.2% |

`sat.` = share of targets with |v| > 0.9, where tanh' < 0.19 and the sample contributes almost no gradient. `sat. all` includes the mate mass, which is 20.0% and saturated at ANY scale.

## Downstream consequences of changing the scale

`290.6806` is the Lc0 WDL calibration - it makes v ~ expected score. It was NOT fitted on our data. Changing it means:

- `benchmarking/player/acpl_elo_estimator.py` inverts with `290.6806*atanh(q)`; it would need the new constant or every ACPL Elo estimate is wrong by the ratio.
- C_PUCT (2.0) and the value weight (2.5x) were tuned against Q-as-expected-score; rescaling Q changes what those mean.
- `playing/uci_wrapper.py` reports `int(q*1000)`, a third mapping already inconsistent with both. Worth unifying whatever is chosen.

## Raw cp distribution (pre-tanh)

| quantile | cp |
|---|---:|
| p1 | -887 |
| p5 | -420 |
| p25 | -15 |
| p50 | 0 |
| p75 | 51 |
| p95 | 496 |
| p99 | 2,754 |

- exactly 0: **418,796** (26.19%) - dead-drawn positions. This atom lands in the `[0.00,+0.10)` histogram bin, which is why that bin looks lopsided; it is mostly binning, not a White bias.
- mean cp +38.7, median +0.0 (a mild genuine White skew remains on top of the zero atom)
- |cp| >= 2000 (clipped): 28,026 (1.753%)

## Emitted scale: **849.7011**

Selected as the with-mates fit to overall std 0.5. Pass --exclude-mates-from-fit to emit the cp-only fit instead.

## Value target histogram at the emitted scale (cp only)

| bin | count |
|---|---:|
| [-1.00, -0.90) | 12,919 |
| [-0.90, -0.80) | 2,235 |
| [-0.80, -0.70) | 7,272 |
| [-0.70, -0.60) | 16,495 |
| [-0.60, -0.50) | 28,835 |
| [-0.50, -0.40) | 28,390 |
| [-0.40, -0.30) | 28,311 |
| [-0.30, -0.20) | 32,149 |
| [-0.20, -0.10) | 59,791 |
| [-0.10, +0.00) | 277,126 |
| [+0.00, +0.10) | 796,382 |
| [+0.10, +0.20) | 93,158 |
| [+0.20, +0.30) | 46,760 |
| [+0.30, +0.40) | 39,106 |
| [+0.40, +0.50) | 38,459 |
| [+0.50, +0.60) | 39,193 |
| [+0.60, +0.70) | 22,559 |
| [+0.70, +0.80) | 9,481 |
| [+0.80, +0.90) | 2,956 |
| [+0.90, +1.00) | 17,580 |