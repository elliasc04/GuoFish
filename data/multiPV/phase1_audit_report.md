# Phase 1 audit - Lichess eval DB

- source: `data\multiPV\lichess_db_eval.jsonl.zst`
- lines streamed: **2,000,000** (limit 2000000)
- JSON parse errors: 0 | records with no evals: 0
- move-level subsample rate: 0.1 (200,257 positions fully parsed)

## 1. POV convention (headline)

### Verdict: **WHITE RELATIVE (absolute)**: Black-to-move multi-PV scores are overwhelmingly ASCENDING.

**Test (a) - multi-PV monotonicity, Black to move**

| pattern | count | share of directional |
|---|---:|---:|
| descending (=> stm-relative) | 748 | 0.081% |
| ascending (=> White-relative) | 927,492 | 99.919% |
| flat (all equal, uninformative) | 55,872 | - |
| non-monotonic (unordered) | 4,125 | - |

- directional blocks: **928,240**
- ambiguity rate (flat + non-monotonic) / all Black multi-PV blocks: **6.071%**

**Control - White to move** (both conventions predict descending): descending=99.925%, ascending=0.075% (of 1,003,655 directional blocks) -> OK, best-first ordering holds

**Test (b) - mate replay** (independent of move ordering)

| outcome | count | share |
|---:|---:|---:|
| consistent_with_both | 24,048 | 22.043% |
| inconclusive | 69,698 | 63.886% |
| white_relative | 15,351 | 14.071% |

## 2. Histograms

### depth (all eval blocks)
| depth | count | share |
|---:|---:|---:|
| 1 | 12,428 | 0.386% |
| 2 | 74 | 0.002% |
| 3 | 42 | 0.001% |
| 4 | 33 | 0.001% |
| 5 | 46 | 0.001% |
| 6 | 40 | 0.001% |
| 7 | 64 | 0.002% |
| 8 | 75 | 0.002% |
| 9 | 95 | 0.003% |
| 10 | 159 | 0.005% |
| 11 | 360 | 0.011% |
| 12 | 1,386 | 0.043% |
| 13 | 4,568 | 0.142% |
| 14 | 11,397 | 0.354% |
| 15 | 26,256 | 0.815% |
| 16 | 48,240 | 1.497% |
| 17 | 71,737 | 2.226% |
| 18 | 91,831 | 2.849% |
| 19 | 101,807 | 3.159% |
| 20 | 207,472 | 6.438% |
| 21 | 195,325 | 6.061% |
| 22 | 194,408 | 6.032% |
| 23 | 209,948 | 6.515% |
| 24 | 201,981 | 6.267% |
| 25 | 185,974 | 5.771% |
| 26 | 166,811 | 5.176% |
| 27 | 147,361 | 4.573% |
| 28 | 129,566 | 4.020% |
| 29 | 114,528 | 3.554% |
| 30 | 140,045 | 4.346% |
| 31 | 42,972 | 1.333% |
| 32 | 106,853 | 3.316% |
| 33 | 31,455 | 0.976% |
| 34 | 77,836 | 2.415% |
| 35 | 23,494 | 0.729% |
| 36 | 54,584 | 1.694% |
| 37 | 17,846 | 0.554% |
| 38 | 40,457 | 1.255% |
| 39 | 14,314 | 0.444% |
| 40 | 33,322 | 1.034% |
| _(94 more)_ | | |

### PVs per eval block
| n_pvs | count | share |
|---:|---:|---:|
| 1 | 1,149,041 | 35.654% |
| 2 | 316,782 | 9.830% |
| 3 | 658,739 | 20.441% |
| 4 | 180,000 | 5.585% |
| 5 | 843,932 | 26.187% |
| 6 | 2,239 | 0.069% |
| 7 | 12,455 | 0.386% |
| 8 | 709 | 0.022% |
| 9 | 53,409 | 1.657% |
| 10 | 5,145 | 0.160% |
| 11 | 8 | 0.000% |
| 12 | 4 | 0.000% |
| 13 | 11 | 0.000% |
| 14 | 2 | 0.000% |
| 15 | 18 | 0.001% |
| 16 | 1 | 0.000% |
| 17 | 7 | 0.000% |
| 18 | 8 | 0.000% |
| 19 | 33 | 0.001% |
| 20 | 72 | 0.002% |
| 21 | 7 | 0.000% |
| 22 | 3 | 0.000% |
| 23 | 1 | 0.000% |
| 24 | 2 | 0.000% |
| 25 | 4 | 0.000% |
| 26 | 2 | 0.000% |
| 27 | 7 | 0.000% |
| 28 | 6 | 0.000% |
| 29 | 9 | 0.000% |
| 30 | 13 | 0.000% |
| 31 | 5 | 0.000% |
| 32 | 7 | 0.000% |
| 33 | 2 | 0.000% |
| 34 | 5 | 0.000% |
| 35 | 2 | 0.000% |
| 36 | 1 | 0.000% |
| 37 | 5 | 0.000% |
| 38 | 2 | 0.000% |
| 39 | 3 | 0.000% |
| 41 | 1 | 0.000% |
| _(5 more)_ | | |

### max depth among evals with >=2 PVs (per position)
| depth | count | share |
|---:|---:|---:|
| 1 | 215 | 0.016% |
| 2 | 13 | 0.001% |
| 3 | 19 | 0.001% |
| 4 | 10 | 0.001% |
| 5 | 17 | 0.001% |
| 6 | 15 | 0.001% |
| 7 | 27 | 0.002% |
| 8 | 36 | 0.003% |
| 9 | 41 | 0.003% |
| 10 | 86 | 0.006% |
| 11 | 234 | 0.017% |
| 12 | 882 | 0.065% |
| 13 | 3,089 | 0.226% |
| 14 | 8,063 | 0.590% |
| 15 | 18,760 | 1.372% |
| 16 | 34,817 | 2.547% |
| 17 | 51,545 | 3.770% |
| 18 | 63,858 | 4.671% |
| 19 | 69,035 | 5.049% |
| 20 | 92,277 | 6.749% |
| 21 | 85,761 | 6.273% |
| 22 | 79,571 | 5.820% |
| 23 | 81,759 | 5.980% |
| 24 | 76,473 | 5.593% |
| 25 | 70,584 | 5.163% |
| 26 | 65,077 | 4.760% |
| 27 | 58,877 | 4.306% |
| 28 | 51,734 | 3.784% |
| 29 | 44,391 | 3.247% |
| 30 | 54,828 | 4.010% |
| 31 | 13,599 | 0.995% |
| 32 | 36,763 | 2.689% |
| 33 | 9,352 | 0.684% |
| 34 | 26,155 | 1.913% |
| 35 | 6,967 | 0.510% |
| 36 | 19,715 | 1.442% |
| 37 | 5,652 | 0.413% |
| 38 | 15,806 | 1.156% |
| 39 | 4,603 | 0.337% |
| 40 | 14,317 | 1.047% |
| _(89 more)_ | | |

- positions whose DEEPEST eval has exactly 1 PV: **1,149,041** (57.452%)
- positions with at least one >=2-PV block: **1,367,194** (68.360%)

## 3. Yield estimates

`value` = positions whose deepest eval meets value_min_depth (a sample is emitted). 
`policy` = subset that also has a >=2-PV block at policy_min_depth (has_policy=1).

| value_min_depth | policy_min_depth | value samples | share | with policy | policy share of emitted |
|---:|---:|---:|---:|---:|---:|
| 16 | 12 | 1,960,376 | 98.019% | 1,340,489 | 68.379% |
| 16 | 16 | 1,960,376 | 98.019% | 1,335,687 | 68.134% |
| 16 | 20 | 1,960,376 | 98.019% | 1,116,432 | 56.950% |
| 20 | 12 | 1,756,045 | 87.802% | 1,157,902 | 65.938% |
| 20 | 16 | 1,756,045 | 87.802% | 1,153,282 | 65.675% |
| 20 | 20 | 1,756,045 | 87.802% | 1,116,432 | 63.577% |
| 24 | 12 | 1,311,476 | 65.574% | 910,737 | 69.444% |
| 24 | 16 | 1,311,476 | 65.574% | 908,154 | 69.247% |
| 24 | 20 | 1,311,476 | 65.574% | 885,622 | 67.529% |

## 4. Mate scores and promotion collisions

- total PVs seen: **9,342,787**
- PVs scored `mate`: **1,085,786** (11.622%)
- PVs with NEITHER `cp` nor `mate`: **0** (0.000%) - these must drop move+score together
- PV first-moves that failed to parse: **26**
- positions where two PVs in one block share a (from,to) pair (**promotion collision**): **9,225** (4.607% of the move-parsed subsample)

## 5. Smoke tests

- **4-field FEN accepted by `chess.Board`**: 3/3 PASS (missing halfmove/fullmove default to 0/1)
- **UCI_Chess960 castling normalization**: `e1h1`->`e1g1`, `e1a1`->`e1c1` PASS
- **`_board_to_tokens`**: 3000 random positions, len==68 and all tokens in [0,42] -> 0 failures PASS

## 6. Phase 2 pre-flight

### 6a. `rel[0] == max(rel)` violation rate

Phase 2 asserts that, after conversion to stm-relative, PV 0 is the best move, skipping the sample on violation and hard-failing the job above 0.1%.

- multi-PV blocks checked: **2,073,669**
- max |cp| observed in the DB: **20,000**
- brief's mate mapping occupies the band **[5,000, 9,900]** -> it OVERLAPS the cp range, so a large cp can outrank a mate

| mate mapping | violations | rate | 0.1% gate |
|---|---:|---:|---|
| brief: `sign*(10000-100*min(|m|,50))` | 5,132 | **0.2475%** | **TRIPS** |
| disjoint bands (cp clamped to +-10,000, mates above) | 876 | **0.0422%** | **PASS** |

- blocks mixing `mate` and `cp` PVs: **90,835** (4.380%)
- share of the brief-mapping violations that occur in mixed blocks: **84.645%** - this is the root cause

### 6b. Chess960 contamination

- positions whose PV first move is illegal on a standard board but legal with `chess960=True`: **24** (0.012% of the move-parsed subsample)
- these are king-takes-rook castling moves in shuffled positions; the record is Chess960, not standard chess
