# GATE - promotion-collision investigation

- source: `data\multiPV\lichess_db_eval.jsonl.zst`  | lines scanned: 2,000,000 | positions move-parsed: **200,257**

## Verdict

The 4.61% rate is **REAL and reproduces exactly** (4.6066% of positions), but it is **not promotions** and **not a cross-block artifact**. Both hypotheses are wrong.

**96.2088% of colliding pairs are the SAME MOVE listed twice inside one eval block** - byte-identical `line` and identical score. Only 3.7912% are genuine underpromotion pairs.

The cross-block hypothesis is ruled out two ways: the Phase 1 detector already reset its `seen` set inside `for ev in evals` (it never had the bug), and a genuinely cross-block detector would have reported an ADDITIONAL 31.5989% of positions, not 4.6%.

Duplicated blocks look like a merge artifact: PV counts run 2k+1 (9 = 1+4x2, 7 = 1+3x2), consistent with two users' MultiPV submissions being concatenated rather than deduplicated.

### Consequence for the converter (action required)

`+=` accumulation is correct for underpromotions but **wrong for duplicated PVs**: it gives a duplicated move double softmax mass. Dedup on the full `chess.Move` (from, to, promotion) BEFORE accumulating into the 4096 index. That drops exact repeats while keeping e8=Q distinct from e8=N, so genuine promotion collisions still merge via `+=` as intended.

Note the `rel[0] == max(rel)` invariant does **not** catch this - the best move stays first; it is the *relative mass* that is corrupted.

## Measurements

| quantity | value |
|---|---:|
| positions with a WITHIN-block (from,to) collision | **9,225** (4.6066%) |
| positions where a (from,to) repeats only ACROSS blocks (what the hypothesised buggy detector would add) | 63,279 (31.5989%) |
| total colliding pairs | 30,228 |
| PV first-moves that are promotions | 2,818 of 933,636 (0.3018%) |

### Collision pair classification

| kind | count | share |
|---|---:|---:|
| duplicate_pv | 29,082 | 96.2088% |
| underpromotion_pair | 1,146 | 3.7912% |

### Promotion pieces involved in colliding pairs

| piece | count |
|---|---:|
| Q | 923 |
| R | 673 |
| B | 391 |
| N | 305 |

### Where collisions live (piece count)

| pieces | colliding positions | all sampled | collision rate |
|---:|---:|---:|---:|
| 3 | 26 | 3,837 | 0.6776% |
| 4 | 107 | 13,402 | 0.7984% |
| 5 | 90 | 5,485 | 1.6408% |
| 6 | 58 | 4,044 | 1.4342% |
| 7 | 44 | 3,554 | 1.2380% |
| 8 | 35 | 3,483 | 1.0049% |
| 9 | 37 | 3,906 | 0.9473% |
| 10 | 35 | 4,144 | 0.8446% |
| 11 | 31 | 4,240 | 0.7311% |
| 12 | 29 | 3,706 | 0.7825% |
| 13 | 24 | 3,708 | 0.6472% |
| 14 | 27 | 4,135 | 0.6530% |
| 15 | 25 | 4,168 | 0.5998% |
| 16 | 45 | 4,960 | 0.9073% |
| 17 | 30 | 4,554 | 0.6588% |
| 18 | 39 | 5,073 | 0.7688% |
| 19 | 40 | 4,578 | 0.8737% |
| 20 | 55 | 5,804 | 0.9476% |
| 21 | 51 | 5,155 | 0.9893% |
| 22 | 98 | 7,174 | 1.3660% |
| 23 | 76 | 5,637 | 1.3482% |
| 24 | 221 | 8,755 | 2.5243% |
| 25 | 161 | 6,140 | 2.6221% |
| 26 | 449 | 11,263 | 3.9865% |
| 27 | 289 | 6,193 | 4.6666% |
| 28 | 1,112 | 14,658 | 7.5863% |
| 29 | 613 | 6,448 | 9.5068% |
| 30 | 2,257 | 19,049 | 11.8484% |
| 31 | 632 | 5,229 | 12.0864% |
| 32 | 2,489 | 17,763 | 14.0123% |

## Impact on the policy target actually written

Measured on the block the converter would select (deepest with >=2 PVs at depth >= 20), T=30, amended mate mapping.

| quantity | value |
|---|---:|
| positions with a usable policy block | 111,857 |
| ...whose selected block contains DUPLICATE moves | **1,761** (1.5743%) |
| ...whose selected block contains a real promotion collision | 382 (0.3415%) |
| **policy targets whose ARGMAX changes if you don't dedup** | **1,080** (0.9655% of all policy labels; 61.3288% of affected) |
| worst single-move probability shift | 0.1716 |

### How identical are the duplicated entries?

| relationship | count |
|---|---:|
| identical_line_and_score | 3,581 |
| different_score | 23 |
| same_score_different_line | 1 |

## Dumped colliding positions (20)

#### `r1bq1rk1/pp2bppp/2n1pn2/6B1/2BP4/2N2N2/PP3PPP/R2QK2R w KQ -`

- side to move: **white**, pieces: **28**
- colliding pairs: `a2a3`+`a2a3` (duplicate_pv), `d1d2`+`d1d2` (duplicate_pv), `a1c1`+`a1c1` (duplicate_pv), `g5e3`+`g5e3` (duplicate_pv)

```json
eval[0] depth=29 knodes=87850 n_pvs=3
   pv[0]       cp=-10  e1h1 h7h6 g5f4 b7b6 f1e1 c8b7 a2a3 e7d6 f4g3 a8c8
   pv[1]       cp=-26  a2a3 f6d5 g5e7 c6e7 c3d5 e7d5 e1h1 b7b6 f3e5 c8b7
   pv[2]       cp=-32  d1d2 b7b6 a1d1 c8b7 e1h1 c6a5 c4e2 a8c8 f3e5 f6d5
eval[1] depth=25 knodes=94080 n_pvs=5
   pv[0]       cp=-21  e1h1 h7h6 g5h4 b7b6 a2a3 c8b7 d1d3 f6h5 h4g3 a8c8
   pv[1]       cp=-30  d1d2 a7a6 c4d3 b7b5 e1h1 c8b7 a1c1 c6b4 g5f6 e7f6
   pv[2]       cp=-35  g5e3 a7a6 a2a4 d8a5 e1h1 f8d8 d1b3 a5c7 c4d3 c6a5
   pv[3]       cp=-42  a1c1 b7b6 e1h1 c8b7 a2a3 h7h6 g5h4 f6h5 h4g3 e7f6
   pv[4]       cp=-45  c4b5 f6d5 b5c6 d5c3 g5e7 d8e7 b2c3 b7c6 e1h1 c6c5
eval[2] depth=22 knodes=12958 n_pvs=9
   pv[0]       cp=-20  e1h1 b7b6 a2a3 c8b7 d1d3 h7h6 g5h4 a8c8 h4g3 f6h5
   pv[1]       cp=-29  a2a3 f6d5 g5e7 c6e7 c3d5 e7d5 e1h1 b7b6 c4d3 c8b7
   pv[2]       cp=-29  a2a3 f6d5 g5e7 c6e7 c3d5 e7d5 e1h1 b7b6 c4d3 c8b7
   pv[3]       cp=-35  d1d2 a7a6 e1h1 b7b5 c4d3 c8b7 a1c1 c6b4 g5f6 e7f6
   pv[4]       cp=-35  d1d2 a7a6 e1h1 b7b5 c4d3 c8b7 a1c1 c6b4 g5f6 e7f6
   pv[5]       cp=-37  a1c1 h7h6 g5f4 b7b6 e1h1 c8b7 f1e1 a8c8 c4d3 e7d6
   pv[6]       cp=-37  a1c1 h7h6 g5f4 b7b6 e1h1 c8b7 f1e1 a8c8 c4d3 e7d6
   pv[7]       cp=-38  g5e3 a7a6 a2a3 b7b5 d4d5 e6d5 c4d5 d8c7 e1h1 c8b7
   pv[8]       cp=-38  g5e3 a7a6 a2a3 b7b5 d4d5 e6d5 c4d5 d8c7 e1h1 c8b7
```

#### `rnbqk2r/pp3ppp/4pn2/1Bbp4/8/2N1PN2/PPP2PPP/R1BQK2R b KQkq -`

- side to move: **black**, pieces: **30**
- colliding pairs: `b8c6`+`b8c6` (duplicate_pv), `c8d7`+`c8d7` (duplicate_pv), `f6d7`+`f6d7` (duplicate_pv)

```json
eval[0] depth=34 knodes=1547392 n_pvs=5
   pv[0]       cp=-65  b8d7 e1h1 e8h8 b2b3 a7a6 b5e2 b7b5 c3b1 c8b7 c1b2
   pv[1]       cp=-54  b8c6 e1h1 e8h8 b2b3 d8e7 c1b2 c8d7 c3e2 a7a6 b5c6
   pv[2]       cp=-33  c8d7 b5d3 d7c6 e1h1 e8h8 a2a3 c5d6 d1e2 b8d7 e3e4
   pv[3]         cp=0  f6d7 e3e4 d5e4 c3e4 d8a5 e4c3 a7a6 b5d3 e8h8 e1h1
   pv[4]        cp=53  e8e7 d1e2 d8c7 e1h1 a7a6 b5d3 c8d7 a1b1 d7c6 b2b4
eval[1] depth=26 knodes=116192 n_pvs=7
   pv[0]       cp=-50  b8d7 e1h1 e8h8 b5d3 h7h6 a2a3 e6e5 e3e4 d5e4 d3e4
   pv[1]       cp=-41  b8c6 e1h1 e8h8 b2b3 d8e7 c1b2 c8d7 c3a4 c5a3 c2c4
   pv[2]       cp=-41  b8c6 e1h1 e8h8 b2b3 d8e7 c1b2 c8d7 c3a4 c5a3 c2c4
   pv[3]       cp=-30  c8d7 b5d3 e8h8 e3e4 b8c6 e1h1 d5d4 e4e5 c6e5 f3e5
   pv[4]       cp=-30  c8d7 b5d3 e8h8 e3e4 b8c6 e1h1 d5d4 e4e5 c6e5 f3e5
   pv[5]        cp=12  f6d7 e3e4 d5e4 c3e4 d8a5 e4c3 a7a6 b5d3 b8c6 e1h1
   pv[6]        cp=12  f6d7 e3e4 d5e4 c3e4 d8a5 e4c3 a7a6 b5d3 b8c6 e1h1
```

#### `r1b2rk1/3nppbp/p2p2p1/Q1p5/2pPP3/5P2/PP4PP/1KN2B1R b - -`

- side to move: **black**, pieces: **26**
- colliding pairs: `g7d4`+`g7d4` (duplicate_pv), `c4c3`+`c4c3` (duplicate_pv), `h7h5`+`h7h5` (duplicate_pv), `d6d5`+`d6d5` (duplicate_pv)

```json
eval[0] depth=34 knodes=331295 n_pvs=5
   pv[0]      cp=-161  g7d4 f1c4 a8b8 b2b3 d7b6 c4a6 b8a8 a5b6 c8a6 h1d1
   pv[1]      cp=-152  a8b8 f1c4 g7d4 b2b3 d7b6 c4a6 b8a8 a5b6 c8a6 h1d1
   pv[2]      cp=-137  c4c3 b2c3 c5d4 c3d4 a8b8 c1b3 g7d4 f1c4 d7e5 h1c1
   pv[3]       cp=-47  h7h5 f1c4 g7d4 h1d1 d4f6 d1d3 d7e5 c4d5 e5d3 c1d3
   pv[4]       cp=-41  d6d5 e4d5 c4c3 a5c3 g7d4 c3a3 d4f6 c1b3 a8b8 f1d3
eval[1] depth=22 knodes=52705 n_pvs=9
   pv[0]      cp=-165  a8b8 f1c4 g7d4 c1b3 d4g7 h1c1 d7b6 c4a6 b8a8 a5b6
   pv[1]      cp=-159  g7d4 f1c4 a8b8 b2b3 d7b6 c4a6 b8a8 a5b6 c8a6 b1c2
   pv[2]      cp=-159  g7d4 f1c4 a8b8 b2b3 d7b6 c4a6 b8a8 a5b6 c8a6 b1c2
   pv[3]      cp=-146  c4c3 a5c3 g7d4 c3a5 a8b8 b2b3 d7e5 a5d2 e5c6 f1c4
   pv[4]      cp=-146  c4c3 a5c3 g7d4 c3a5 a8b8 b2b3 d7e5 a5d2 e5c6 f1c4
   pv[5]       cp=-29  h7h5 f1c4 g7d4 h1d1 d4f6 b2b3 a8b8 c1d3 d7b6 c4a6
   pv[6]       cp=-29  h7h5 f1c4 g7d4 h1d1 d4f6 b2b3 a8b8 c1d3 d7b6 c4a6
   pv[7]       cp=-21  d6d5 e4d5 g7d4 f1c4 d7e5 c4e2 c8f5 b1a1 f5c2 a5d2
   pv[8]       cp=-21  d6d5 e4d5 g7d4 f1c4 d7e5 c4e2 c8f5 b1a1 f5c2 a5d2
```

#### `r1bqk2r/pppp1ppp/5n2/2b1p3/2P1P3/2N2Q1P/PPP2PP1/R1B1K1NR b KQkq -`

- side to move: **black**, pieces: **30**
- colliding pairs: `d7d6`+`d7d6` (duplicate_pv), `b7b5`+`b7b5` (duplicate_pv), `a7a6`+`a7a6` (duplicate_pv), `d8e7`+`d8e7` (duplicate_pv)

```json
eval[0] depth=27 knodes=107178 n_pvs=9
   pv[0]        cp=10  c7c6 b2b3 d7d6 g1e2 c8e6 c1e3 c5e3 f3e3 d8b6 e3d3
   pv[1]        cp=12  d7d6 c1e3 f6d7 g1e2 b7b6 h3h4 h7h5 e3g5 f7f6 g5e3
   pv[2]        cp=12  d7d6 c1e3 f6d7 g1e2 b7b6 h3h4 h7h5 e3g5 f7f6 g5e3
   pv[3]        cp=22  b7b5 c4b5 a7a6 b5b6 c5b6 c1g5 h7h6 g5f6 d8f6 f3f6
   pv[4]        cp=22  b7b5 c4b5 a7a6 b5b6 c5b6 c1g5 h7h6 g5f6 d8f6 f3f6
   pv[5]        cp=25  a7a6 g1e2 d7d6 c1e3 c5e3 f3e3 c8e6 b2b3 f6d7 g2g3
   pv[6]        cp=25  a7a6 g1e2 d7d6 c1e3 c5e3 f3e3 c8e6 b2b3 f6d7 g2g3
   pv[7]        cp=28  d8e7 g1e2 d7d6 c1e3 c5e3 f3e3 a7a5 b2b3 c8e6 a2a4
   pv[8]        cp=28  d8e7 g1e2 d7d6 c1e3 c5e3 f3e3 a7a5 b2b3 c8e6 a2a4
```

#### `rnb1kbnr/ppp1pppp/8/3q4/8/8/PPPP1PPP/RNBQKBNR w KQkq -`

- side to move: **white**, pieces: **30**
- colliding pairs: `g1f3`+`g1f3` (duplicate_pv), `h2h3`+`h2h3` (duplicate_pv), `d2d4`+`d2d4` (duplicate_pv), `d2d3`+`d2d3` (duplicate_pv)

```json
eval[0] depth=60 knodes=129621416 n_pvs=5
   pv[0]        cp=52  b1c3 d5d6 g1f3 g8f6 d2d4 c7c6 f1d3 c8g4 c1e3 b8d7
   pv[1]        cp=30  g1f3 c8g4 b1c3 d5e6 d1e2 e6e2 f1e2 b8c6 h2h3 g4h5
   pv[2]        cp=16  d2d4 e7e5 b1c3 d5d4 d1d4 e5d4 c3b5 f8b4 c1d2 b4d2
   pv[3]         cp=0  h2h3 c7c5 g1f3 b8c6 b1c3 d5d8 f1b5 c8d7 e1h1 e7e6
   pv[4]         cp=0  d2d3 c8f5 g1f3 b8c6 b1c3 d5d7 d3d4 c6b4 f1b5 b4c2
eval[1] depth=25 knodes=52216 n_pvs=7
   pv[0]        cp=53  b1c3 d5d6 d2d4 g8f6 g1f3 g7g6 c1g5 a7a6 d1d2 b7b5
   pv[1]        cp=30  g1f3 c8g4 b1c3 d5e6 d1e2 b8c6 e2e6 g4e6 f1b5 a7a6
   pv[2]        cp=28  d2d4 e7e5 g1f3 e5d4 d1d4 d5d4 f3d4 g8f6 b1c3 f8c5
   pv[3]        cp=14  h2h3 d5e6 g1e2 b7b6 d2d4 c8b7 c2c4 e6c4 e2f4 c4b4
   pv[4]         cp=0  a2a3 b8c6 b1c3 d5e6 g1e2 c6d4 d2d3 g8f6 c3e4 d4f5
   pv[5]       cp=-10  d2d3 b8c6 b1c3 d5d6 g1f3 e7e5 f1e2 g8f6 e1h1 c8f5
   pv[6]       cp=-10  a2a4 b8c6 b1c3 d5e5 f1e2 c8g4 d2d4 g4e2 g1e2 e5h5
eval[2] depth=24 knodes=54125 n_pvs=9
   pv[0]        cp=57  b1c3 d5d6 g1f3 g8f6 d2d4 g7g6 f1e2 f8g7 c3b5 d6b6
   pv[1]        cp=25  g1f3 c8g4 b1c3 d5e6 d1e2 b8c6 e2e6 g4e6 f1b5 e6d7
   pv[2]        cp=25  g1f3 c8g4 b1c3 d5e6 d1e2 b8c6 e2e6 g4e6 f1b5 e6d7
   pv[3]        cp=14  h2h3 b8c6 g1f3 c8f5 b1c3 d5d7 d2d4 e7e6 f1b5 f8d6
   pv[4]        cp=14  h2h3 b8c6 g1f3 c8f5 b1c3 d5d7 d2d4 e7e6 f1b5 f8d6
   pv[5]        cp=11  d2d4 b8c6 g1f3 c8g4 f1e2 e8a8 c1e3 e7e5 b1c3 d5a5
   pv[6]        cp=11  d2d4 b8c6 g1f3 c8g4 f1e2 e8a8 c1e3 e7e5 b1c3 d5a5
   pv[7]         cp=0  d2d3 b8c6 b1c3 d5d6 g1f3 e7e5 d3d4 c8f5 d4d5 c6b4
   pv[8]         cp=0  d2d3 b8c6 b1c3 d5d6 g1f3 e7e5 d3d4 c8f5 d4d5 c6b4
eval[3] depth=23 knodes=91737 n_pvs=10
   pv[0]        cp=57  b1c3 d5d6 d2d4 g8f6 g1f3 g7g6 f1e2 f8g7 c3b5 d6d8
   pv[1]        cp=35  d2d4 b8c6 g1f3 e7e5 b1c3 f8b4 c1d2 b4c3 d2c3 e5e4
   pv[2]        cp=31  g1f3 c8g4 b1c3 d5e6 d1e2 e6e2 f1e2 b8c6 h2h3 g4h5
   pv[3]        cp=24  h2h3 b8c6 g1f3 c8f5 b1c3 d5d6 f1c4 e7e6 e1h1 f8e7
   pv[4]         cp=5  d2d3 b8c6 b1c3 d5d6 g1f3 e7e5 f1e2 c8f5 e1h1 e8a8
   pv[5]        cp=-5  g1e2 c8f5 b1c3 d5d7 e2g3 f5g6 d2d4 b8c6 c1e3 e7e6
   pv[6]        cp=-7  a2a3 b8c6 b1c3 d5e6 g1e2 c6d4 d2d3 g8f6 c3e4 d4f5
   pv[7]       cp=-12  a2a4 b8c6 b1c3 d5e5 f1e2 c8g4 d2d4 g4e2 d1e2 e5e2
   pv[8]       cp=-20  d1e2 a7a6 b1c3 d5a5 c3d1 e7e5 c2c3 g8f6 b2b4 a5d5
   pv[9]       cp=-23  d1f3 g8f6 f3d5 f6d5 b1c3 d5c3 d2c3 e7e5 c1e3 c8e6
```

#### `8/3kPK2/8/8/8/8/8/8 w - -`

- side to move: **white**, pieces: **3**
- colliding pairs: `e7e8q`+`e7e8r` (underpromotion_pair), `e7e8q`+`e7e8r` (underpromotion_pair)

```json
eval[0] depth=245 knodes=34596 n_pvs=1
   pv[0]       mate=6  e7e8q d7c7 e8b5 c7d8 f7e6 d8c7 e6e7 c7c8 e7d6 c8d8
eval[1] depth=185 knodes=85528 n_pvs=2
   pv[0]       mate=6  e7e8q d7c7 e8b5 c7d8 f7e6 d8c7 e6e7 c7c8 e7d6 c8d8
   pv[1]       mate=9  f7f8 d7e6 e7e8q e6d5 e8e3 d5c4 f8e7 c4b4 e7d6 b4c4
eval[2] depth=100 knodes=163049 n_pvs=3
   pv[0]       mate=6  e7e8q d7c7 e8b5 c7d8 b5c5 d8d7 f7f6 d7d8 f6e6 d8e8
   pv[1]       mate=9  f7f8 d7c6 e7e8q c6d5 e8e3 d5d6 e3g5 d6c6 f8e7 c6b6
   pv[2]      mate=15  e7e8r d7d6 f7g6 d6d5 g6f5 d5d4 e8d8 d4c4 f5e5 c4c3
eval[3] depth=55 knodes=72958 n_pvs=5
   pv[0]       mate=6  e7e8q d7c7 e8b5 c7d6 f7f6 d6c7 f6e6 c7c8 e6d6 c8d8
   pv[1]       mate=9  f7f8 d7d6 e7e8q d6d5 e8e3 d5c4 f8e7 c4b4 e7d6 b4c4
   pv[2]      mate=16  e7e8r d7d6 f7g6 d6d5 g6f5 d5d4 e8d8 d4c4 f5e4 c4c5
   pv[3]         cp=0  f7g6 d7e7 g6g5 e7e6 g5f4 e6d5 f4f5 d5c5 f5g4 c5b5
   pv[4]         cp=0  f7f6 d7e8 f6e5 e8e7 e5f4 e7e6 f4e4 e6d6 e4f5 d6c5
```

#### `8/1kP1K3/8/8/8/8/8/8 w - -`

- side to move: **white**, pieces: **3**
- colliding pairs: `c7c8r`+`c7c8q` (underpromotion_pair), `c7c8b`+`c7c8r` (underpromotion_pair), `c7c8b`+`c7c8q` (underpromotion_pair)

```json
eval[0] depth=245 knodes=189933 n_pvs=2
   pv[0]       mate=7  e7d7 b7b6 c7c8q b6b5 c8c3 b5a4 d7d6 a4b5 c3b3 b5a5
   pv[1]       mate=9  e7d8 b7c6 c7c8q c6d5 c8g4 d5e5 d8c7 e5d5 g4f4 d5c5
eval[1] depth=65 knodes=232323 n_pvs=3
   pv[0]       mate=7  e7d7 b7a6 c7c8q a6b5 c8c2 b5b4 d7c6 b4a3 c2b1 a3a4
   pv[1]       mate=9  e7d8 b7c6 c7c8q c6d5 c8g4 d5e5 d8c7 e5d5 g4f4 d5c5
   pv[2]         cp=0  c7c8r b7c8 e7e6 c8d8 e6f5 d8e7 f5e5 e7d7 e5f4 d7e6
eval[2] depth=50 knodes=99856 n_pvs=5
   pv[0]       mate=7  e7d7 b7b6 c7c8q b6b5 c8c3 b5a4 d7c7 a4b5 c3d4 b5a5
   pv[1]       mate=9  e7d8 b7c6 c7c8q c6d5 c8g4 d5c5 d8c7 c5d5 g4f4 d5c5
   pv[2]         cp=0  c7c8r b7c8 e7e6 c8c7 e6e5 c7c6 e5f4 c6d6 f4e4 d6c5
   pv[3]         cp=0  c7c8q b7c8 e7e6 c8c7 e6e5 c7c6 e5f4 c6d6 f4e4 d6c5
   pv[4]         cp=0  e7d6 b7c8 d6e5 c8c7 e5e4 c7d6 e4f3 d6e6 f3f2 e6e5
eval[3] depth=48 knodes=32500 n_pvs=10
   pv[0]       mate=7  e7d7 b7b6 c7c8q b6b5 c8c2 b5b4 d7c6 b4a3 c2b1 a3a4
   pv[1]       mate=9  e7d8 b7c6 c7c8q c6d5 c8g4 d5e5 d8c7 e5d5 g4f4 d5e6
   pv[2]         cp=0  e7e6 b7c7 e6d5 c7b7 d5d6 b7b6 d6d5 b6c7 d5c5 c7d7
   pv[3]         cp=0  c7c8b b7c8 e7d6 c8b7 d6d7 b7b8 d7c6 b8a7 c6c7 a7a6
   pv[4]         cp=0  c7c8r b7c8 e7d6 c8b7 d6d7 b7b8 d7c6 b8a7 c6c7 a7a6
   pv[5]         cp=0  e7f6 b7c7 f6e6 c7c6 e6e5 c6c5 e5f6 c5c4 f6g7 c4d3
   pv[6]         cp=0  e7d6 b7c8 d6d5 c8c7 d5c5 c7b7 c5d6 b7b6 d6d7 b6c5
   pv[7]         cp=0  c7c8q b7c8 e7d6 c8b7 d6d7 b7b8 d7c6 b8a7 c6c7 a7a6
   pv[8]         cp=0  e7f7 b7c7 f7e7 c7c6 e7f6 c6c5 f6g6 c5d4 g6g5 d4e5
   pv[9]         cp=0  e7f8 b7c7 f8e7 c7c6 e7f6 c6c5 f6g6 c5d4 g6g5 d4e5
```

#### `8/6PK/5k2/8/8/8/8/6r1 w - -`

- side to move: **white**, pieces: **4**
- colliding pairs: `g7g8n`+`g7g8r` (underpromotion_pair), `g7g8n`+`g7g8b` (underpromotion_pair), `g7g8n`+`g7g8b` (underpromotion_pair), `g7g8n`+`g7g8q` (underpromotion_pair), `g7g8n`+`g7g8r` (underpromotion_pair)

```json
eval[0] depth=80 knodes=2741740 n_pvs=5
   pv[0]         cp=0  g7g8n f6f7 g8h6 f7e7 h6g8 e7f8 g8h6
   pv[1]      mate=-4  h7h8 f6f7 g7g8q g1g8 h8h7 g8g6 h7h8 g6h6
   pv[2]      mate=-4  h7g8 g1g7 g8h8 g7f7 h8g8 f6g6 g8h8 f7f8
   pv[3]      mate=-1  g7g8r g1h1
   pv[4]      mate=-1  g7g8b g1h1
eval[1] depth=46 knodes=48647 n_pvs=7
   pv[0]       cp=-40  g7g8n f6e6 g8h6 g1g5 h6g8 e6f7 g8h6 f7e7 h6g8 e7f8
   pv[1]      mate=-4  h7h8 f6f7 g7g8b g1g8 h8h7 g8g6 h7h8 g6h6
   pv[2]      mate=-4  h7g8 g1g7 g8h8 g7g6 h8h7 f6f7 h7h8 g6h6
   pv[3]      mate=-1  g7g8b g1h1
   pv[4]      mate=-1  g7g8q g1h1
   pv[5]      mate=-1  g7g8r g1h1
   pv[6]      mate=-1  h7h6 g1h1
```

#### `r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1BPP1N2/PP3PPP/RNBQK2R w KQ -`

- side to move: **white**, pieces: **32**
- colliding pairs: `e1g1`+`e1g1` (duplicate_pv), `a2a4`+`a2a4` (duplicate_pv), `d1e2`+`d1e2` (duplicate_pv), `h2h3`+`h2h3` (duplicate_pv)

```json
eval[0] depth=38 knodes=153923 n_pvs=1
   pv[0]         cp=9  b1d2 d7d5 e1h1 c8e6 f1e1 d8d6 e4d5 e6d5 d2e4 d5b3
eval[1] depth=34 knodes=51166 n_pvs=2
   pv[0]         cp=4  e1h1 d7d5 b1d2 c8e6 f1e1 d8d6 e4d5 e6d5 d2e4 f6e4
   pv[1]         cp=4  b1d2 d7d5 e1h1 c8e6 f1e1 d8d6 e4d5 e6d5 d2e4 d5b3
eval[2] depth=32 knodes=156551 n_pvs=3
   pv[0]         cp=7  b1d2 d7d5 e1h1 c8e6 f1e1 d8d6 e4d5 e6d5 d2e4 f6e4
   pv[1]         cp=4  e1h1 d7d5 b1d2 c8e6 f1e1 d8d6 e4d5 e6d5 d2e4 f6e4
   pv[2]        cp=-1  d1e2 h7h6 e1h1 f8e8 a2a4 a8b8 a4b5 a6b5 d3d4 e5d4
eval[3] depth=26 knodes=75054 n_pvs=9
   pv[0]         cp=7  b1d2 d7d5 e1h1 c8e6 f1e1 d8d6 e4d5 e6d5 d2e4 f6e4
   pv[1]         cp=6  e1h1 d7d5 b1d2 c8e6 f1e1 d8d6 e4d5 e6d5 d2e4 f6e4
   pv[2]         cp=6  e1h1 d7d5 b1d2 c8e6 f1e1 d8d6 e4d5 e6d5 d2e4 f6e4
   pv[3]        cp=-2  a2a4 b5b4 b1d2 d7d5 e1h1 c8e6 f1e1 a8b8 a4a5 d5e4
   pv[4]        cp=-2  a2a4 b5b4 b1d2 d7d5 e1h1 c8e6 f1e1 a8b8 a4a5 d5e4
   pv[5]        cp=-4  d1e2 d7d5 e1h1 c8e6 c1g5 h7h6 g5f6 e7f6 b1d2 d8d7
   pv[6]        cp=-4  d1e2 d7d5 e1h1 c8e6 c1g5 h7h6 g5f6 e7f6 b1d2 d8d7
   pv[7]       cp=-10  h2h3 d7d5 e4d5 f6d5 e1h1 a6a5 f1e1 c8b7 a2a4 b5b4
   pv[8]       cp=-10  h2h3 d7d5 e4d5 f6d5 e1h1 a6a5 f1e1 c8b7 a2a4 b5b4
```

#### `rnbqkb1r/pppppppp/5n2/8/2P5/2N5/PP1PPPPP/R1BQKBNR b KQkq -`

- side to move: **black**, pieces: **32**
- colliding pairs: `d7d5`+`d7d5` (duplicate_pv), `e7e6`+`e7e6` (duplicate_pv), `c7c5`+`c7c5` (duplicate_pv), `c7c6`+`c7c6` (duplicate_pv)

```json
eval[0] depth=50 knodes=16992381 n_pvs=3
   pv[0]        cp=12  e7e5 g1f3 b8c6 e2e4 f8c5 f3e5 c6e5 d2d4 c5b4 d4e5
   pv[1]        cp=17  e7e6 g1f3 d7d5 d2d4 f8b4 c1g5 h7h6 g5f6 d8f6 e2e3
   pv[2]        cp=25  c7c5 g1f3 e7e6 g2g3 b8c6 f1g2 d8b6 e1h1 f8e7 b2b3
eval[1] depth=46 knodes=3834247 n_pvs=5
   pv[0]        cp=10  e7e5 e2e3 f8b4 g1e2 d7d5 c4d5 f6d5 d1b3 d5c3 b2c3
   pv[1]        cp=11  e7e6 g1f3 d7d5 d2d4 f8e7 c1f4 e8h8 e2e3 b8d7 c4c5
   pv[2]        cp=21  c7c6 g1f3 d7d5 d2d4 e7e6 c1g5 h7h6 g5h4 f8e7 e2e3
   pv[3]        cp=24  d7d5 c4d5 f6d5 g1f3 b8c6 e2e4 d5c3 b2c3 e7e5 f1b5
   pv[4]        cp=30  c7c5 g1f3 d7d5 c4d5 f6d5 d2d4 d5c3 b2c3 g7g6 e2e4
eval[2] depth=36 knodes=468504 n_pvs=7
   pv[0]        cp=11  e7e5 g1f3 b8c6 e2e4 f8b4 d2d3 d7d6 a2a3 b4c5 b2b4
   pv[1]        cp=16  e7e6 g1f3 d7d5 d2d4 f8e7 c1g5 e8h8 e2e3 h7h6 g5f6
   pv[2]        cp=23  c7c5 g1f3 e7e6 g2g3 f8e7 f1g2 d7d5 c4d5 e6d5 d2d4
   pv[3]        cp=29  d7d5 c4d5 f6d5 g1f3 d5c3 b2c3 g7g6 d2d4 f8g7 e2e4
   pv[4]        cp=30  c7c6 d2d4 d7d5 g1f3 e7e6 c1g5 h7h6 g5h4 f8e7 e2e3
   pv[5]        cp=33  g7g6 e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 f8g7 d4c6
   pv[6]        cp=33  b7b6 d2d4 c8b7 d1c2 d7d5 c4d5 f6d5 e2e4 d5c3 b2c3
eval[3] depth=22 knodes=6414 n_pvs=9
   pv[0]        cp=12  e7e5 g1f3 b8c6 g2g3 f8b4 f1g2 e8h8 e1h1 b4c3 b2c3
   pv[1]        cp=22  d7d5 c4d5 f6d5 g1f3 c7c5 d2d4 d5c3 b2c3 c5d4 c3d4
   pv[2]        cp=22  d7d5 c4d5 f6d5 g1f3 c7c5 d2d4 d5c3 b2c3 c5d4 c3d4
   pv[3]        cp=27  e7e6 e2e4 d7d5 c4d5 e6d5 e4e5 d5d4 e5f6 d4c3 d2c3
   pv[4]        cp=27  e7e6 e2e4 d7d5 c4d5 e6d5 e4e5 d5d4 e5f6 d4c3 d2c3
   pv[5]        cp=31  c7c5 g1f3 d7d5 c4d5 f6d5 e2e4 d5b4 f1b5 b8c6 d2d4
   pv[6]        cp=31  c7c5 g1f3 d7d5 c4d5 f6d5 e2e4 d5b4 f1b5 b8c6 d2d4
   pv[7]        cp=34  c7c6 e2e4 d7d5 e4e5 d5d4 e5f6 d4c3 b2c3 e7f6 d2d4
   pv[8]        cp=34  c7c6 e2e4 d7d5 e4e5 d5d4 e5f6 d4c3 b2c3 e7f6 d2d4
eval[4] depth=21 knodes=12670 n_pvs=10
   pv[0]        cp=19  e7e5 g1f3 b8c6 g2g3 d7d5 c4d5 f6d5 f1g2 d5b6 a2a3
   pv[1]        cp=24  c7c5 g1f3 e7e6 e2e4 b8c6 f1e2 d7d5 e4d5 e6d5 d2d4
   pv[2]        cp=25  e7e6 e2e4 d7d5 e4e5 d5d4 e5f6 d4c3 b2c3 d8f6 g1f3
   pv[3]        cp=29  d7d5 c4d5 f6d5 g1f3 c7c5 d2d4 d5c3 b2c3 c5d4 c3d4
   pv[4]        cp=30  c7c6 g1f3 d7d5 d2d4 e7e6 c1g5 b8d7 e2e4 d5e4 c3e4
   pv[5]        cp=48  d7d6 d2d4 g7g6 e2e4 f8g7 f1e2 e8h8 h2h3 e7e5 d4d5
   pv[6]        cp=48  g7g6 e2e4 d7d6 d2d4 f8g7 f1e2 e8h8 h2h3 a7a5 c1e3
   pv[7]        cp=49  b8c6 d2d4 d7d5 g1f3 e7e6 c1g5 f8e7 e2e3 h7h6 g5f4
   pv[8]        cp=58  b7b6 e2e4 c8b7 e4e5 f6e4 d1f3 d7d5 d2d4 e7e6 g1h3
   pv[9]        cp=68  h7h6 d2d4 d7d5 c4d5 f6d5 e2e4 d5c3 b2c3 e7e5 g1f3
```

#### `rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/6P1/PP2PP1P/RNBQKBNR w KQkq -`

- side to move: **white**, pieces: **32**
- colliding pairs: `f1g2`+`f1g2` (duplicate_pv), `b1c3`+`b1c3` (duplicate_pv), `c4d5`+`c4d5` (duplicate_pv), `a2a3`+`a2a3` (duplicate_pv)

```json
eval[0] depth=47 knodes=2987855 n_pvs=5
   pv[0]        cp=20  g1f3 f8b4 c1d2 b4e7 f1g2 e8h8 e1h1 c7c6 d1c2 b7b6
   pv[1]        cp=18  f1g2 f8b4 c1d2 b4e7 g1f3 e8h8 e1h1 c7c6 d1c2 b7b6
   pv[2]        cp=-9  b1c3 c7c5 d4c5 d5d4 c3a4 f8c5 a4c5 d8a5 c1d2 a5c5
   pv[3]       cp=-10  b1d2 d5c4 d1a4 d8d7 a4c4 b7b5 c4d3 c8b7 g1f3 c7c5
   pv[4]       cp=-19  a2a3 d5c4 d1a4 b8d7 f1g2 a7a6 a4c4 c7c5 d4c5 f8c5
eval[1] depth=22 knodes=26050 n_pvs=6
   pv[0]        cp=25  g1f3 f8b4 c1d2 b4e7 f1g2 c7c6 e1h1 e8h8 d1c2 b8d7
   pv[1]        cp=21  f1g2 f8b4 b1d2 e8h8 g1f3 d5c4 a2a3 b4d2 c1d2 c8d7
   pv[2]        cp=-7  b1c3 c7c5 d4c5 d5d4 c3a4 b8c6 f1g2 e6e5 g1f3 e5e4
   pv[3]       cp=-17  c1g5 d5c4 g1f3 b8c6 f1g2 a8b8 e1h1 f8e7 d1c1 h7h6
   pv[4]       cp=-18  a2a3 d5c4 d1a4 c7c6 a4c4 d8d5 c4d5 c6d5 g1f3 b8c6
   pv[5]       cp=-22  e2e3 d5c4 f1c4 f8e7 b1d2 b8d7 g1f3 a7a6 c4e2 e8h8
eval[2] depth=20 knodes=38064 n_pvs=8
   pv[0]        cp=26  g1f3 f8b4 c1d2 b4e7 f1g2 e8h8 e1h1 c7c6 d1c2 a7a5
   pv[1]        cp=20  f1g2 f8b4 c1d2 b4e7 g1f3 c7c6 e1h1 e8h8 d1c2 b7b6
   pv[2]        cp=-4  b1c3 c7c5 d4c5 f8c5 c4d5 e6d5 g1f3 b8c6 f1g2 d5d4
   pv[3]       cp=-14  c1g5 d5c4 b1c3 f8e7 f1g2 e8h8 g1f3 b8d7 e1h1 f6d5
   pv[4]       cp=-16  c4d5 e6d5 c1f4 c7c6 d1c2 g7g6 b1c3 a7a5 g1f3 c8f5
   pv[5]       cp=-17  a2a3 d5c4 d1a4 b8d7 a4c4 c7c5 f1g2 c5d4 g1f3 e6e5
   pv[6]       cp=-20  e2e3 d5c4 f1c4 c7c5 g1f3 f8e7 e1h1 c5d4 d1d4 d8d4
   pv[7]       cp=-25  b1d2 d5c4 d1a4 d8d7 a4c4 b7b5 c4b3 c7c5 f1g2 c8b7
eval[3] depth=19 knodes=15166 n_pvs=9
   pv[0]        cp=31  g1f3 f8b4 c1d2 b4e7 f1g2 c7c6 e1h1 e8h8 d1b3 b8d7
   pv[1]        cp=24  f1g2 f8b4 b1d2 e8h8 g1f3 b7b6 e1h1 c8b7 b2b3 a7a5
   pv[2]       cp=-14  b1c3 c7c5 d4c5 f8c5 c4d5 e6d5 f1g2 b8c6 a2a3 e8h8
   pv[3]       cp=-22  c4d5 e6d5 g1f3 h7h6 f1g2 f8e7 d1c2 e8h8 b1c3 c7c6
   pv[4]       cp=-29  a2a3 d5c4 d1a4 b8d7 f1g2 c7c5 a4c4 c5d4 c4d4 f8c5
   pv[5]        cp=24  f1g2 f8b4 b1d2 e8h8 g1f3 b7b6 e1h1 c8b7 b2b3 a7a5
   pv[6]       cp=-14  b1c3 c7c5 d4c5 f8c5 c4d5 e6d5 f1g2 b8c6 a2a3 e8h8
   pv[7]       cp=-22  c4d5 e6d5 g1f3 h7h6 f1g2 f8e7 d1c2 e8h8 b1c3 c7c6
   pv[8]       cp=-29  a2a3 d5c4 d1a4 b8d7 f1g2 c7c5 a4c4 c5d4 c4d4 f8c5
```

#### `rnbqk2r/pp3ppp/2p2n2/2bpp3/2B1PP2/2NP4/PPP3PP/R1BQK1NR w KQkq -`

- side to move: **white**, pieces: **32**
- colliding pairs: `f4e5`+`f4e5` (duplicate_pv)

```json
eval[0] depth=36 knodes=179807 n_pvs=2
   pv[0]         cp=6  e4d5 e8h8 f4e5 c6d5 c4b3 c5g1 h1g1 c8g4 d1d2 f8e8
   pv[1]       cp=-33  f4e5 d5c4 e5f6 d8f6 g1f3 c4d3 d1d3 e8h8 c1e3 f8d8
eval[1] depth=29 knodes=96382 n_pvs=3
   pv[0]         cp=0  e4d5 e8h8 f4e5 f6g4 g1f3 g4f2 d1e2 f2h1 c1g5 d8a5
   pv[1]       cp=-31  f4e5 d5c4 e5f6 d8f6 g1f3 c4d3 d1d3 e8h8 c1e3 f8d8
   pv[2]       cp=-31  f4e5 d5c4 e5f6 d8f6 g1f3 c4d3 d1d3 e8h8 c1e3 f8d8
eval[2] depth=28 knodes=331602 n_pvs=5
   pv[0]         cp=9  e4d5 e8h8 f4e5 c6d5 c4b3 c5g1 h1g1 c8g4 d1d2 f8e8
   pv[1]       cp=-22  f4e5 d5c4 e5f6 d8f6 g1f3 c4d3 d1d3 e8h8 c1e3 f8d8
   pv[2]      cp=-113  c4b3 f6g4 d1d2 c5e3 d2e2 e5f4 c1e3 g4e3 c3d1 e3d1
   pv[3]      cp=-157  g1f3 d5c4 f4e5 f6d7 d3d4 c5e7 d1e2 b7b5 a2a4 b5b4
   pv[4]      cp=-233  c4d5 c6d5 f4e5 c8g4 d1d2 d5d4 c3b5 f6d7 d2g5 d8g5
```

#### `r1b1r1k1/pp3pbp/6p1/2nBp1B1/4P3/8/PP1N1PPP/R3K2R w KQ -`

- side to move: **white**, pieces: **24**
- colliding pairs: `e1g1`+`e1g1` (duplicate_pv), `g2g3`+`g2g3` (duplicate_pv), `e1c1`+`e1c1` (duplicate_pv), `a1d1`+`a1d1` (duplicate_pv)

```json
eval[0] depth=43 knodes=148848 n_pvs=1
   pv[0]         cp=0  d2c4 g7f8 e1h1 c8e6 f1d1 e6d5 e4d5 b7b6 g2g4 a8c8
eval[1] depth=40 knodes=221998 n_pvs=2
   pv[0]         cp=0  e1h1 c8e6 f1d1 e6d5 e4d5 g7f8 d2c4 a8c8 a1c1 b7b6
   pv[1]         cp=0  d2c4 g7f8 e1h1 c8e6 f1d1 e6d5 e4d5 a8c8 a1c1 b7b6
eval[2] depth=38 knodes=212242 n_pvs=3
   pv[0]         cp=0  d2c4 g7f8 e1h1 c8e6 f1d1 e6d5 e4d5 b7b6 a1c1 a8c8
   pv[1]         cp=0  e1h1 c8e6 f1d1 g7f8 d2c4 e6d5 e4d5 b7b6 a1c1 a8c8
   pv[2]         cp=0  e1a1 c8e6 d5e6 c5e6 g5e3 f7f5 f2f3 a8c8 c1b1 b7b6
eval[3] depth=34 knodes=556521 n_pvs=5
   pv[0]         cp=0  d2c4 g7f8 e1h1 c8e6 f1d1 a8c8 a1c1 b7b6 f2f3 e6d5
   pv[1]         cp=0  e1h1 c8e6 f1d1 e6d5 e4d5 h7h6 g5e3 c5d3 d2e4 d3b2
   pv[2]        cp=-6  g2g3 c8e6 e1e2 e6d5 e4d5 e5e4 a1c1 b7b6 d5d6 g7b2
   pv[3]        cp=-7  e1a1 c5e6 g5e3 e6f4 e3f4 e5f4 c1b1 c8e6 d5e6 e8e6
   pv[4]       cp=-15  a1d1 c5e6 g5e3 e6f4 e3f4 e5f4 d2c4 a8b8 e1e2 c8e6
eval[4] depth=24 knodes=38086 n_pvs=9
   pv[0]         cp=9  d2c4 g7f8 e1h1 c8e6 f1d1 a8c8 a1c1 b7b6 f2f3 e6d5
   pv[1]         cp=0  e1h1 c8e6 f1d1 f7f5 a1c1 a8c8 d2c4 c5e4 d5e4 c8c4
   pv[2]         cp=0  g2g3 c8e6 e1e2 e6d5 e4d5 b7b6 a1c1 e5e4 d5d6 g7b2
   pv[3]         cp=0  e1h1 c8e6 f1d1 f7f5 a1c1 a8c8 d2c4 c5e4 d5e4 c8c4
   pv[4]         cp=0  g2g3 c8e6 e1e2 e6d5 e4d5 b7b6 a1c1 e5e4 d5d6 g7b2
   pv[5]        cp=-3  e1a1 c8e6 d5e6 c5e6 g5e3 f7f5 f2f3 a8c8 c1b1 b7b6
   pv[6]        cp=-3  e1a1 c8e6 d5e6 c5e6 g5e3 f7f5 f2f3 a8c8 c1b1 b7b6
   pv[7]        cp=-7  a1d1 c8e6 e1e2 g7f8 h2h4 f7f5 h4h5 e6d5 e4d5 h7h6
   pv[8]        cp=-7  a1d1 c8e6 e1e2 g7f8 h2h4 f7f5 h4h5 e6d5 e4d5 h7h6
```

#### `r1bqk2r/p1pn1ppp/5n2/1pbPp1B1/2p1P3/2N2N2/PP3PPP/R2QKB1R w KQkq -`

- side to move: **white**, pieces: **31**
- colliding pairs: `c3b5`+`c3b5` (duplicate_pv), `g5h4`+`g5h4` (duplicate_pv), `d1c1`+`d1c1` (duplicate_pv)

```json
eval[0] depth=48 knodes=701297 n_pvs=1
   pv[0]       cp=-85  f1e2 e8h8 e1h1 a8b8 b2b3 c4b3 c3b5 h7h6 g5f6 d7f6
eval[1] depth=36 knodes=976128 n_pvs=3
   pv[0]       cp=-82  f1e2 e8h8 e1h1 a8b8 b2b3 c4b3 c3b5 h7h6 g5f6 d7f6
   pv[1]      cp=-118  g5h4 a7a6 a2a4 a8b8 f1e2 e8h8 e1h1 h7h6 a4b5 a6b5
   pv[2]      cp=-126  d1c2 a7a6 f1e2 h7h6 g5h4 e8h8 a2a4 a8b8 e1h1 c5d6
eval[2] depth=26 knodes=113335 n_pvs=5
   pv[0]       cp=-94  f1e2 e8h8 e1h1 a8b8 b2b3 c4b3 a2b3 a7a6 d1d3 c8b7
   pv[1]      cp=-108  g5h4 a7a6 f1e2 e8h8 a2a4 a8b8 e1h1 h7h6 a4b5 a6b5
   pv[2]      cp=-112  c3b5 c5f2 e1f2 f6e4 f2g1 e4g5 f3g5 d8g5 d1c1 g5d8
   pv[3]      cp=-112  d1c2 a7a6 f1e2 e8h8 a2a4 a8b8 a4b5 a6b5 e1h1 h7h6
   pv[4]      cp=-122  a1c1 a7a6 f1e2 h7h6 g5h4 c5d6 e1h1 d7c5 f3d2 g7g5
eval[3] depth=22 knodes=22235 n_pvs=7
   pv[0]       cp=-82  f1e2 e8h8 e1h1 a8b8 b2b3 c4b3 a2b3 a7a6 d1d3 h7h6
   pv[1]      cp=-111  c3b5 c5f2 e1f2 f6e4 f2g1 e4g5 f3g5 d8g5 d1c1 g5d8
   pv[2]      cp=-111  c3b5 c5f2 e1f2 f6e4 f2g1 e4g5 f3g5 d8g5 d1c1 g5d8
   pv[3]      cp=-115  g5h4 a7a6 f1e2 e8h8 e1h1 h7h6 b2b3 c4b3 d1b3 f8e8
   pv[4]      cp=-115  g5h4 a7a6 f1e2 e8h8 e1h1 h7h6 b2b3 c4b3 d1b3 f8e8
   pv[5]      cp=-117  d1c1 a7a6 f1e2 c8b7 e1h1 c7c6 b2b3 c4b3 a2b3 e8h8
   pv[6]      cp=-117  d1c1 a7a6 f1e2 c8b7 e1h1 c7c6 b2b3 c4b3 a2b3 e8h8
```

#### `rnb1kb1r/ppppqppp/8/4N3/4Q3/8/PPPP1PPP/RNB1KB1R b KQkq -`

- side to move: **black**, pieces: **29**
- colliding pairs: `f7f6`+`f7f6` (duplicate_pv), `b8c6`+`b8c6` (duplicate_pv), `d7d5`+`d7d5` (duplicate_pv)

```json
eval[0] depth=47 knodes=335043 n_pvs=1
   pv[0]        cp=55  d7d6 d2d4 d6e5 d4e5 b8c6 b1c3 e7e5 e4e5 c6e5 c3b5
eval[1] depth=46 knodes=3574566 n_pvs=5
   pv[0]        cp=60  d7d6 d2d4 d6e5 d4e5 b8c6 b1c3 e7e5 e4e5 c6e5 c1f4
   pv[1]       cp=144  f7f6 d2d4 f6e5 e4e5 e7e5 d4e5 b8c6 c1f4 c6b4 e1d2
   pv[2]       cp=284  b8c6 d2d4 d7d6 c1g5 f7f6 f1b5 c8d7 b1c3 d6e5 c3d5
   pv[3]       cp=401  d7d5 e4e2 c8e6 d2d4 g7g6 e5f3 b8d7 b1c3 f8g7 c1g5
   pv[4]       cp=424  c7c6 e4e3 d7d5 d2d4 f7f6 e5f3 c8f5 c2c4 b8a6 e3e7
eval[2] depth=24 knodes=20610 n_pvs=7
   pv[0]        cp=57  d7d6 d2d4 d6e5 d4e5 b8c6 b1c3 e7e5 e4e5 c6e5 c3b5
   pv[1]       cp=116  f7f6 d2d4 f6e5 e4e5 e7e5 d4e5 b8c6 c1f4 b7b6 b1c3
   pv[2]       cp=116  f7f6 d2d4 f6e5 e4e5 e7e5 d4e5 b8c6 c1f4 b7b6 b1c3
   pv[3]       cp=170  b8c6 d2d4 d7d6 c1g5 f7f6 f1b5 c8d7 b1c3 d6e5 c3d5
   pv[4]       cp=170  b8c6 d2d4 d7d6 c1g5 f7f6 f1b5 c8d7 b1c3 d6e5 c3d5
   pv[5]       cp=376  d7d5 e4e2 c8e6 d2d4 b8d7 b1c3 d7e5 e2e5 e7d7 c1f4
   pv[6]       cp=376  d7d5 e4e2 c8e6 d2d4 b8d7 b1c3 d7e5 e2e5 e7d7 c1f4
```

#### `r1b2rk1/pp1Nqppp/2nbp3/2pp4/3P4/2PBP1B1/PP1N1PPP/R2QK2R b KQ -`

- side to move: **black**, pieces: **31**
- colliding pairs: `c8d7`+`c8d7` (duplicate_pv), `c5c4`+`c5c4` (duplicate_pv), `c5d4`+`c5d4` (duplicate_pv)

```json
eval[0] depth=34 knodes=1002350 n_pvs=5
   pv[0]        cp=56  e7d7 d1e2 e6e5 d4e5 c6e5 d3c2 f8e8 e1a1 b7b5 d2f3
   pv[1]        cp=71  c8d7 g3d6 e7d6 d4c5 d6c7 d2f3 c6e7 d1c2 e6e5 e3e4
   pv[2]       cp=464  c5c4 d7f8 c4d3 f8h7 g8h7 d2f3 f7f5 g3d6 e7d6 d1d3
   pv[3]       cp=496  f8d8 g3d6 e7d6 d7c5 e6e5 e1h1 e5e4 d3e2 b7b6 c5a4
   pv[4]       cp=497  c5d4 g3d6 e7d6 d7f8 d4c3 b2c3 d6f8 e1h1 f8c5 a1c1
eval[1] depth=27 knodes=91763 n_pvs=7
   pv[0]        cp=31  e7d7 e1h1 d7e7 d3c2 g7g6 f1e1 d6g3 h2g3 f8d8 f2f4
   pv[1]        cp=46  c8d7 g3d6 e7d6 d4c5 d6c7 b2b4 c6e5 d3e2 b7b6 f2f4
   pv[2]        cp=46  c8d7 g3d6 e7d6 d4c5 d6c7 b2b4 c6e5 d3e2 b7b6 f2f4
   pv[3]       cp=421  c5c4 d7f8 c4d3 f8h7 g8h7 d2f3 h7g8 g3d6 e7d6 d1d3
   pv[4]       cp=421  c5c4 d7f8 c4d3 f8h7 g8h7 d2f3 h7g8 g3d6 e7d6 d1d3
   pv[5]       cp=446  c5d4 g3d6 e7d6 d7f8 d4c3 b2c3 d6f8 e1h1 e6e5 a1c1
   pv[6]       cp=446  c5d4 g3d6 e7d6 d7f8 d4c3 b2c3 d6f8 e1h1 e6e5 a1c1
```

#### `rnbqkbnr/pp3ppp/4p3/2ppP3/3P4/8/PPP2PPP/RNBQKBNR w KQkq -`

- side to move: **white**, pieces: **32**
- colliding pairs: `g1f3`+`g1f3` (duplicate_pv), `f1d3`+`f1d3` (duplicate_pv), `d4c5`+`d4c5` (duplicate_pv), `b1d2`+`b1d2` (duplicate_pv)

```json
eval[0] depth=60 knodes=83116829 n_pvs=2
   pv[0]        cp=20  c2c3 b8c6 g1f3 d8b6 a2a3 c5c4 f1e2 c6a5 b1d2 c8d7
   pv[1]         cp=0  g1f3 c5d4 f3d4 g8e7 f1d3 b8c6 d4c6 e7c6 d1e2 d8c7
eval[1] depth=55 knodes=1732462 n_pvs=3
   pv[0]        cp=32  c2c3 b8c6 g1f3 c8d7 f1e2 g8e7 e1h1 e7g6 b1a3 f8e7
   pv[1]         cp=0  g1f3 c5d4 f3d4 g8e7 f1d3 e7c6 d4c6 b8c6 d1e2 d8c7
   pv[2]       cp=-12  f1d3 c5d4 g1f3 b8c6 e1h1 f8c5 b1d2 g8e7 d2b3 c5b6
eval[2] depth=46 knodes=3454803 n_pvs=5
   pv[0]        cp=23  c2c3 d8b6 g1f3 b8c6 a2a3 g8h6 f1d3 c8d7 e1h1 c5d4
   pv[1]         cp=0  g1f3 c5d4 f3d4 g8e7 f1d3 b8c6 d4c6 e7c6 d1e2 d8c7
   pv[2]       cp=-11  f1d3 b8c6 g1f3 c5d4 e1h1 f8c5 b1d2 g8e7 d2b3 c5b6
   pv[3]       cp=-15  f2f4 b8c6 g1f3 d8b6 f1d3 g8h6 e1h1 c8d7 g1h1 h6g4
   pv[4]       cp=-15  b1d2 b8c6 g1f3 d8c7 c2c4 c5d4 c4d5 e6d5 f1e2 c8f5
eval[3] depth=25 knodes=83328 n_pvs=9
   pv[0]        cp=31  c2c3 b8c6 g1f3 g8e7 b1a3 c8d7 a3c2 d8a5 c1d2 a5b6
   pv[1]        cp=-3  g1f3 c5d4 f1d3 b8c6 e1h1 f8c5 b1d2 g8e7 d2b3 c5b6
   pv[2]        cp=-3  g1f3 c5d4 f1d3 b8c6 e1h1 f8c5 b1d2 g8e7 d2b3 c5b6
   pv[3]        cp=-9  f1d3 c5d4 g1f3 b8c6 e1h1 f8c5 b1d2 g8e7 d2b3 c5b6
   pv[4]        cp=-9  f1d3 c5d4 g1f3 b8c6 e1h1 f8c5 b1d2 g8e7 d2b3 c5b6
   pv[5]       cp=-17  d4c5 b8c6 g1f3 f8c5 b1c3 a7a6 f1d3 g8e7 e1h1 c8d7
   pv[6]       cp=-17  d4c5 b8c6 g1f3 f8c5 b1c3 a7a6 f1d3 g8e7 e1h1 c8d7
   pv[7]       cp=-24  b1d2 b8c6 g1f3 c5d4 d2b3 d8c7 b3d4 c6e5 c1f4 c7a5
   pv[8]       cp=-24  b1d2 b8c6 g1f3 c5d4 d2b3 d8c7 b3d4 c6e5 c1f4 c7a5
eval[4] depth=20 knodes=49754 n_pvs=20
   pv[0]        cp=43  c2c3 c8d7 g1f3 d8b6 a2a3 g8e7 b2b4 c5d4 c3d4 d7b5
   pv[1]        cp=-6  g1f3 c5d4 f3d4 g8e7 f1d3 e7c6 d4c6 b8c6 d1e2 d8c7
   pv[2]        cp=-7  f1d3 c5d4 g1f3 b8c6 e1h1 f8c5 b1d2 b7b6 f1e1 g8e7
   pv[3]       cp=-29  b1d2 b8c6 g1f3 c5d4 d2b3 d8c7 b3d4 c6e5 c1f4 c7a5
   pv[4]       cp=-32  d4c5 b8c6 g1f3 f8c5 b1c3 d8c7 c1f4 c7b6 d1d2 c5f2
   pv[5]       cp=-33  f1b5 b8c6 g1e2 c5d4 e2d4 c8d7 d4c6 d7c6 b5c6 b7c6
   pv[6]       cp=-38  g1e2 b8c6 c2c3 f7f6 e5f6 g8f6 g2g3 f8d6 f1g2 e6e5
   pv[7]       cp=-41  f2f4 b8c6 g1f3 d8b6 f1d3 c8d7 e1h1 c5d4 g1h1 g8h6
   pv[8]       cp=-41  a2a3 c5d4 f2f4 b8c6 b2b4 a7a5 b4b5 c6e5 f4e5 d8h4
   pv[9]       cp=-47  b1c3 c5d4 c3b5 b8c6 g1f3 f8c5 a2a3 a7a6 b2b4 c5b6
   pv[10]       cp=-47  f1e2 c5d4 f2f4 g8h6 e2d3 b8c6 g1f3 f8c5 b1d2 c5b6
   pv[11]       cp=-49  a2a4 c5d4 f2f4 b8c6 b1d2 f8b4 f1d3 g8e7 g1f3 d8b6
   pv[12]       cp=-53  b1a3 c5d4 a3b5 b8c6 g1f3 f8c5 a2a3 a7a6 b2b4 c5b6
   pv[13]       cp=-53  h2h4 c5d4 g1f3 b8c6 c1f4 g8h6 f3d4 f8c5 d4b3 c5b6
   pv[14]       cp=-55  h2h3 c5d4 f1d3 b8c6 g1f3 f7f6 d1e2 f6e5 f3e5 g8f6
   pv[15]       cp=-57  c1e3 c5d4 e3d4 b8c6 g1f3 g8e7 c2c3 c8d7 d1d2 e7f5
   pv[16]       cp=-58  g2g3 c5d4 b1d2 b8c6 f2f4 a7a5 g1f3 c8d7 a2a4 f8b4
   pv[17]       cp=-67  c1f4 d8b6 b1d2 b8c6 g1f3 b6b2 d4c5 f8c5 f1d3 b2a3
   pv[18]       cp=-70  d1g4 c5d4 f1d3 b8c6 g1f3 h7h5 g4h3 d8a5 b1d2 a5c7
   pv[19]       cp=-71  c2c4 c5d4 g1f3 b8c6 f3d4 g8e7 d4c6 e7c6 f1e2 f8b4
```

#### `r1bqkb1r/ppp2ppp/2n5/3np3/2B5/3P1N2/PPP2PPP/RNBQK2R w KQkq -`

- side to move: **white**, pieces: **30**
- colliding pairs: `a2a4`+`a2a4` (duplicate_pv), `b1c3`+`b1c3` (duplicate_pv), `d1e2`+`d1e2` (duplicate_pv), `h2h3`+`h2h3` (duplicate_pv), `b1d2`+`b1d2` (duplicate_pv), `c4b3`+`c4b3` (duplicate_pv), `a2a3`+`a2a3` (duplicate_pv), `c2c3`+`c2c3` (duplicate_pv), `c1d2`+`c1d2` (duplicate_pv)

```json
eval[0] depth=55 knodes=1435159 n_pvs=2
   pv[0]        cp=21  e1h1 f8e7 f1e1 f7f6 c2c3 c8g4 h2h3 g4h5 d3d4 e5d4
   pv[1]        cp=11  d1e2 f7f6 e1h1 d5b6 c4b5 f8d6 a2a4 a7a5 c1e3 e8h8
eval[1] depth=40 knodes=3405069 n_pvs=3
   pv[0]        cp=38  e1h1 f8e7 f1e1 f7f6 c2c3 c8g4 h2h3 g4h5 d3d4 e5d4
   pv[1]        cp=17  a2a4 c8f5 e1h1 d8d7 f1e1 f7f6 c2c3 d5b6 c4b5 a7a6
   pv[2]        cp=16  d1e2 f7f6 e1h1 d5b6 c4b5 f8d6 a2a4 a7a5 c1e3 e8h8
eval[2] depth=36 knodes=469910 n_pvs=5
   pv[0]        cp=41  e1h1 f8e7 f1e1 f7f6 h2h3 d5b6 c4b3 c8f5 a2a4 a7a5
   pv[1]         cp=8  d1e2 f7f6 e1h1 d5b6 c4b5 f8d6 a2a4 a7a5 d3d4 e8h8
   pv[2]         cp=0  a2a4 c8f5 e1h1 d8d7 f1e1 f7f6 c2c3 d5b6 c4b5 a7a6
   pv[3]         cp=0  b1d2 c8f5 e1h1 d8d7 f3e5 c6e5 d1e2 d7e6 c4d5 e6d5
   pv[4]         cp=0  b1c3 c8e6 f3g5 d5c3 g5e6 c3d1 e6d8 a8d8 e1d1 c6a5
eval[3] depth=18 knodes=27427 n_pvs=19
   pv[0]        cp=40  e1h1 f8e7 f1e1 d5b6 c4b3 c8g4 h2h3 g4f3 d1f3 e8h8
   pv[1]         cp=7  a2a4 c8f5 a4a5 a7a6 e1h1 d8d7 f1e1 f7f6 c2c3 e8a8
   pv[2]         cp=7  a2a4 c8f5 a4a5 a7a6 e1h1 d8d7 f1e1 f7f6 c2c3 e8a8
   pv[3]         cp=1  b1c3 d5c3 b2c3 h7h6 e1h1 f8d6 a2a4 e8h8 a4a5 a7a6
   pv[4]         cp=1  b1c3 d5c3 b2c3 h7h6 e1h1 f8d6 a2a4 e8h8 a4a5 a7a6
   pv[5]         cp=0  d1e2 f7f6 e1h1 d5b6 c4b5 f8d6 a2a4 a7a6 b5c6 b7c6
   pv[6]         cp=0  d1e2 f7f6 e1h1 d5b6 c4b5 f8d6 a2a4 a7a6 b5c6 b7c6
   pv[7]        cp=-3  h2h3 c8f5 e1h1 d8d7 d1e2 f7f6 d3d4 e5e4 c4d5 d7d5
   pv[8]        cp=-3  h2h3 c8f5 e1h1 d8d7 d1e2 f7f6 d3d4 e5e4 c4d5 d7d5
   pv[9]        cp=-8  b1d2 c8f5 e1h1 d8d7 f3e5 c6e5 d1e2 f7f6 d3d4 d5f4
   pv[10]        cp=-8  b1d2 c8f5 e1h1 d8d7 f3e5 c6e5 d1e2 f7f6 d3d4 d5f4
   pv[11]       cp=-10  c4b3 c8g4 h2h3 g4h5 e1h1 f7f6 b1d2 f8e7 f1e1 e8h8
   pv[12]       cp=-10  a2a3 c8g4 e1h1 f8e7 h2h3 g4h5 f1e1 e8h8 b1c3 d5c3
   pv[13]       cp=-10  c4b3 c8g4 h2h3 g4h5 e1h1 f7f6 b1d2 f8e7 f1e1 e8h8
   pv[14]       cp=-10  a2a3 c8g4 e1h1 f8e7 h2h3 g4h5 f1e1 e8h8 b1c3 d5c3
   pv[15]       cp=-13  c2c3 d5b6 c4b5 f8d6 e1h1 e8h8 f1e1 c8g4 b1d2 a7a6
   pv[16]       cp=-13  c1d2 f7f6 b1c3 c8e6 d1e2 d5c3 d2c3 e6c4 d3c4 d8d7
   pv[17]       cp=-13  c2c3 d5b6 c4b5 f8d6 e1h1 e8h8 f1e1 c8g4 b1d2 a7a6
   pv[18]       cp=-13  c1d2 f7f6 b1c3 c8e6 d1e2 d5c3 d2c3 e6c4 d3c4 d8d7
```

#### `r1bq1rk1/1p1n2pp/p2p4/3Pppb1/2P5/2N1BP2/PP2B1PP/R2Q1RK1 w - -`

- side to move: **white**, pieces: **28**
- colliding pairs: `e3f2`+`e3f2` (duplicate_pv), `d1d2`+`d1d2` (duplicate_pv)

```json
eval[0] depth=35 knodes=121530 n_pvs=1
   pv[0]         cp=0  e3g5 d8g5 d1c1 g5d8 a1b1 d7f6 b2b4 c8d7 a2a4 a8c8
eval[1] depth=30 knodes=91539 n_pvs=2
   pv[0]         cp=4  e3g5 d8g5 d1c1 g5d8 a1b1 d8b6 g1h1 d7f6 b2b4 c8d7
   pv[1]       cp=-11  e3f2 d8f6 b2b4 f6h6 c4c5 d6c5 b4c5 d7c5 f2c5 g5e3
eval[2] depth=28 knodes=83393 n_pvs=5
   pv[0]         cp=2  e3g5 d8g5 d1c1 g5d8 a1b1 d7f6 b2b4 c8d7 a2a4 a8c8
   pv[1]       cp=-14  e3f2 d8f6 d1c2 f6h6 a1d1 g5f4 g2g3 f4e3 b2b4 b7b6
   pv[2]       cp=-14  e3f2 d8f6 d1c2 f6h6 a1d1 g5f4 g2g3 f4e3 b2b4 b7b6
   pv[3]       cp=-24  d1d2 g5e3 d2e3 a6a5 f3f4 f8e8 a1e1 e5f4 e3f4 d7c5
   pv[4]       cp=-24  d1d2 g5e3 d2e3 a6a5 f3f4 f8e8 a1e1 e5f4 e3f4 d7c5
```

#### `r1bqkb1r/ppp2ppp/2n5/4p3/4p3/3B1N2/PPPP1PPP/R1BQK2R w KQkq -`

- side to move: **white**, pieces: **29**
- colliding pairs: `d3b5`+`d3b5` (duplicate_pv), `d3e2`+`d3e2` (duplicate_pv), `f3e5`+`f3e5` (duplicate_pv), `d3b5`+`d3b5` (duplicate_pv), `d3c4`+`d3c4` (duplicate_pv), `d3e2`+`d3e2` (duplicate_pv), `f3e5`+`f3e5` (duplicate_pv)

```json
eval[0] depth=46 knodes=4523640 n_pvs=5
   pv[0]       cp=-18  d3e4 f8d6 d2d4 e5d4 e4c6 b7c6 d1d4 e8h8 c1e3 c6c5
   pv[1]      cp=-461  d3b5 e4f3 b5c6 b7c6 d1f3 d8d5 d2d3 f8d6 h2h4 e8h8
   pv[2]      cp=-471  d3e2 e4f3 e2f3 f8d6 e1h1 e8h8 f3c6 b7c6 d2d3 a8b8
   pv[3]      cp=-473  d3c4 d8f6 c4d5 e4f3 d5f3 c8f5 d2d3 e8a8 c1d2 e5e4
   pv[4]      cp=-499  f3e5 e4d3 e5c6 b7c6 e1h1 d8f6 c2d3 f8d6 f1e1 c8e6
eval[1] depth=25 knodes=55110 n_pvs=7
   pv[0]        cp=-7  d3e4 f8d6 d2d4 e5d4 e4c6 b7c6 d1d4 e8h8 c1e3 a8b8
   pv[1]      cp=-461  d3b5 e4f3 b5c6 b7c6 d1f3 d8d5 d2d3 f7f6 c1e3 c8e6
   pv[2]      cp=-461  d3b5 e4f3 b5c6 b7c6 d1f3 d8d5 d2d3 f7f6 c1e3 c8e6
   pv[3]      cp=-467  d3e2 e4f3 e2f3 f8d6 e1h1 e8h8 f1e1 c8f5 c2c3 d8d7
   pv[4]      cp=-467  d3e2 e4f3 e2f3 f8d6 e1h1 e8h8 f1e1 c8f5 c2c3 d8d7
   pv[5]      cp=-478  f3e5 e4d3 e5c6 b7c6 e1h1 c8e6 f1e1 d8f6 e1e3 e8a8
   pv[6]      cp=-478  f3e5 e4d3 e5c6 b7c6 e1h1 c8e6 f1e1 d8f6 e1e3 e8a8
eval[2] depth=24 knodes=28848 n_pvs=9
   pv[0]       cp=-20  d3e4 f8d6 d2d4 e5d4 e4c6 b7c6 d1d4 e8h8 e1h1 c6c5
   pv[1]      cp=-457  d3b5 e4f3 b5c6 b7c6 d1f3 d8d5 d2d3 f7f6 c1e3 f8b4
   pv[2]      cp=-457  d3b5 e4f3 b5c6 b7c6 d1f3 d8d5 d2d3 f7f6 c1e3 f8b4
   pv[3]      cp=-464  d3c4 e4f3 d1f3 d8f6 f3f6 g7f6 c2c3 h8g8 g2g3 c8e6
   pv[4]      cp=-464  d3c4 e4f3 d1f3 d8f6 f3f6 g7f6 c2c3 h8g8 g2g3 c8e6
   pv[5]      cp=-474  d3e2 e4f3 e2f3 f8d6 e1h1 e8h8 c2c3 e5e4 f3e4 d8h4
   pv[6]      cp=-474  d3e2 e4f3 e2f3 f8d6 e1h1 e8h8 c2c3 e5e4 f3e4 d8h4
   pv[7]      cp=-476  f3e5 e4d3 e5c6 b7c6 e1h1 c8e6 f1e1 d8f6 e1e3 f8d6
   pv[8]      cp=-476  f3e5 e4d3 e5c6 b7c6 e1h1 c8e6 f1e1 d8f6 e1e3 f8d6
```
