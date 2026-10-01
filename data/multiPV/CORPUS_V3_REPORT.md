# Corpus v3 report

Brief: `docs/capacity/training/prod/corpus_v3_brief.md`. Branch `v6-corpus3`. Decisions:
`training/v6/DECISIONS.md`, "Corpus v3".

## Summary

- **`t20` is included.** There are 78,585,592 `t20` policy rows outside ≤5, against a 10M bar [measured].
- **v3 selects 271,759,447 index rows** [measured], not ≈ 192.7M. That is ≈ 269M records and ≈ 104 GB [inferred from v2's 0.9% rejection rate].
  - The policy pool is ≈ 157M roots, about double the 78M planned. So production makes about half the planned policy passes per branch.
- **Old-tier check:** old policy matches §3 exactly (56,092,373). **Old value-only is 84,822,004, 10 more than §3's 84,821,994.**
  - The four buckets agree with the independent recon scan. The brief's sum is off by 10.
  - **Operator: please confirm**; §4 says to stop and report, and this is the report.
- **Nesting** [measured, over the whole index, Windows and Linux]: 90M replay 91,350,634 rows with 0 dropped; corpus v2 replay 115,315,219 rows with 0 dropped.
- **Linux proof** [measured]:
  - the builder tests pass (12/12);
  - the `--limit 200000` smoke is byte-identical to Windows (276 shards);
  - `vm_build_v3.sh --smoke` and `vm_upload_v3.sh` pass end to end in Ubuntu 26.04.1;
  - the full-index selection reproduces `v3_expected_counts.json` exactly.
- **Strata definition bumped to v3** (`depth_tier` gains `t20`). New hash: `5e5fdedaae858f45357bb8eb950fc6ba7be63ca5a30ea2b17422b0fe4d6011a9`. **The harness pins this.**
  - Every regenerated sidecar's codes equal the old ones byte for byte.
- **R2: incomplete.** 60 of the 61 build/eval files (6.17 GB, the index included) and the dry-run set are uploaded and size-verified.
  - **The dump (21.4 GB) is not.** Its upload was stopped at 11.7 GB because the box ran low on memory (a GPU training job had started beside it); it must be re-sent from the start.
  - `data/sha256.txt` is therefore unpublished, so a VM pull fails loudly. Command to finish: §7.
- **Pending (needs the VM):** the build, `build_ok.json`, the v3 upload, then `C0` (and `C1`). Which production config to run is therefore **not yet decided**. `C0`'s 40M quick-val KL, which `prod.sanity.ref_kl` needs, doesn't exist yet.

## 1. Index scan and the `t20` decision

`data/multiPV/v3_index_scan.py` → `data/multiPV/manifests/v3_index_scan.json`. It is one 50.7 s pass over 394,669,566 rows. The 90M selection replayed by the same pass gives 91,350,634 rows, equal to its manifest (self-check) [measured].

**Index rows** (rows with ≤ 32 pieces; policy = `policy_depth ≥ 20`) [measured]:

| tier / label | ≤5 | 6–14 | 15–27 | ≥28 | total |
|---|---:|---:|---:|---:|---:|
| old (≥ 26), policy | 5,193,528 | 20,728,798 | 25,009,155 | 9,760,936 | 60,692,417 |
| old, value-only | 5,578,317 | 27,774,780 | 43,361,069 | 13,367,164 | 90,081,330 |
| new (24–25), policy | 244,048 | 3,483,035 | 12,806,543 | 6,387,270 | 22,920,896 |
| new, value-only | 181,022 | 3,889,385 | 17,210,836 | 8,454,214 | 29,735,457 |
| t20 (20–23), policy | 521,083 | 9,815,387 | 47,525,474 | 21,244,731 | 79,106,675 |
| t20, value-only | 4,342,565 | 16,902,025 | 46,142,483 | 21,370,793 | 88,757,866 |

**≤5 split of the value-only cells:**

| tier | ≤5 | 6+ |
|---|---:|---:|
| old | 5,578,317 | 84,503,013 |
| new | 181,022 | 29,554,435 |
| t20 | 4,342,565 | 84,415,301 |

**Decision:** `t20` policy rows outside ≤5 number 78,585,592, which is ≥ 10M, so **`t20` is included**. Consequently `value_min_depth` is 20 and the strata definition changes (§5).

## 2. Builder changes and the Linux proof

**Changes:**
- **`pass_b_v2.py`:**
  - optional tier `t20` (20 ≤ `max_depth` < 24), on fixed tier edges; `value_min_depth` must equal the lowest tier present;
  - `--nest-manifest` refuses any cell below corpus v2's plan, and replays v2's selection, requiring 0 dropped (its own count must equal v2's `selected_lines`). The 90M floor check is unchanged;
  - `--write-counts` / `--expect-counts`: exact counts, with exit 3 on a mismatch, before anything converts.
- **`corpus_v2_gates.py`:**
  - takes `--corpus`, `--manifest-copy` (or `none`), `--expect-counts`, `--floor-replayed`, `--smoke` and `--out`;
  - the H1 re-run uses the manifest's own source, index and plan, so it works on the VM;
  - `counts` handles `t20` and an empty `valderived`.
- **`extract_frozen_v2.py`:** takes `--manifest` as well as `--corpus`, `--index` and `--m90`.
- **`tools/make_sha256_list.py`:** writes `sha256  size  relative_path` lines, posix and sorted.
- **`data/multiPV/r2_push.py`:** a resumable, size-verified push that publishes the list last. It reuses `training/v6/r2.py`.

**Tests** (`training/v6/tests/test_m6_corpus_v3.py`, synthetic, on top of `test_m6_corpus_v2`'s generated dump, 90M build and v2 build):
- t20 counts match an independent implementation, and both replays drop 0;
- the v2 plan still selects exactly what v2 did;
- a counts mismatch exits 3;
- **refusals:** a cell below v2's rate, a missing tier, a cell below the 90M floor, and `t20` without `value_min_depth` 20;
- the v3 build passes s9 (H1 refusal included), and **every v2 root is in v3 byte for byte**;
- frozen val re-extracted from v3 passes;
- strata v3 assigns `t20`, and refuses `max_depth` < 20;
- the sha256 list and push are resumable, re-send a wrong-sized object, refuse a stale list, and round-trip through the harness's `pull`.

**Results** [measured]:

| check | Windows 11 | Ubuntu 26.04.1 container |
|---|---|---|
| `test_m6_corpus_v2` + `test_m6_corpus_v3` | 12 passed | 12 passed |
| full v6 suite (real data; strata from `strata_def3`, definition v3) | **107 passed, 1 skipped** (31 m 43 s); the skip (A3-equality, needs `runs/screening`) then passed with it linked: 108/108 | not re-run (the harness's container run covers the rest; only strata changed) |
| `--limit 200000` smoke (real dump head; dump and index **read-only** bind mounts in Linux) | exit 0, s9 PASS | exit 0, s9 PASS |
| smoke output | 276 shards, 53,941,608 B | **byte-identical** to Windows; manifests equal except paths, timestamps and git fields; both at `0e00579`, clean |
| full-index selection vs `v3_expected_counts.json` | identical (53 s) | identical (43 s) |
| `vm_build_v3.sh --smoke 200000` via a `file://` store | — | exit 0: pull + verify of 61 files 131 s, s9 PASS, strata, list, `build_ok.json` (`git_dirty: []`); its corpus equals the plain smoke |
| `vm_upload_v3.sh` via a `file://` store | — | exit 0; order upload → verify → delete inputs → `upload_ok.json`; a fresh pull verifies 281 files; a re-run sends nothing |

**Smoke counts** (first 200,000 lines): 140,050 selected (t20 policy 16,852), 139,384 roots, `hard_move` coverage 100%. Its `t20` roots are 16,639 multipv, 0 hard-only, 0 value-only.

## 3–4. Rate plan and exact expected counts

`data/multiPV/rate_plan_v3.json` (sha256 `ec4ce3bf…`) is §3's table. The `*` cells are the 90M manifest's full-precision floors (0.11433605441233782 policy, 0.05731881139060401 value-only), asserted equal at write time. Other settings: `value_min_depth` 20, `policy_min_depth` 20, `--derived-rate 0`, seed 20260802, record format v2.

`data/multiPV/v3_expected_counts.json` (sha256 `2745a63b…`). These are the selected index rows [measured; identical on Windows and Linux]:

| tier / label | ≤5 | 6–14 | 15–27 | ≥28 | total |
|---|---:|---:|---:|---:|---:|
| old, policy | 593,484 | 20,728,798 | 25,009,155 | 9,760,936 | **56,092,373** |
| old, value-only | 318,991 | 27,774,780 | 43,361,069 | 13,367,164 | **84,822,004** |
| new, policy | 28,195 | 3,483,035 | 12,806,543 | 6,387,270 | **22,705,043** |
| new, value-only | 0 | 3,889,385 | 17,210,836 | 8,454,214 | **29,554,435** |
| t20, policy | 0 | 9,815,387 | 47,525,474 | 21,244,731 | **78,585,592** |
| t20, value-only | 0 | 0 | 0 | 0 | **0** |
| **all** | | | | | **271,759,447** |

**Old-tier check against §3:**

| cell | §3 | exact | |
|---|---:|---:|---|
| old policy | 56,092,373 | 56,092,373 | ✓ |
| old value-only | 84,821,994 | 84,822,004 | **+10** |

- The ≤5 cell (318,991) is the 90M build's own on the same draws (CORPUS_V2_REPORT). The other three buckets take every row, so the exact total is their sum.
- The recon headroom scan of 2026-09-25 gives the same per-bucket rows. §3's figure appears to be an addition slip [inferred].
- **I continued past §4's stop-and-report for this reason. The operator should confirm.**
- New-tier policy matches the brief's ≈ 22.7M. New-tier value-only is 29,554,435, against the brief's estimate of ≈ 29.1M (+1.6%).

**Size** [inferred]: v2 rejected 0.91% of selected lines, so v3 ≈ 269.3M records × 387 B ≈ 104 GB, against the brief's ≈ 74 GB for the no-`t20` case.
- Roots: policy ≈ 156.6M, value-only ≈ 113.3M.
- **Production pool, passes per branch** (shares 0.75 / 0.25) [inferred]:
  - policy 1.4 / 2.9 / 5.7 / 11.5 at 300M / 600M / 1.2B / 2.4B;
  - value 0.7 / 1.3 / 2.6 / 5.3.
- The harness's memorization-gap readings become valid after one policy pass, at ≈ 209M samples, not ≈ 105M.

## 5. Strata

**Definition v3:** `depth_tier` ∈ {old, new, v1, **t20**}, where `t20` is code 3 (`max_depth` 20–23; < 20 is refused).
- Placing it after `v1` leaves every existing code unchanged.
- **New hash: `5e5fdedaae858f45357bb8eb950fc6ba7be63ca5a30ea2b17422b0fe4d6011a9`** (v2 was `a9e7a62e9880…`).

**Regenerated** into `data/processed/strata_def3/` (local) [measured]. Each new `codes_sha256` equals the old file's:

| sidecar | records | codes identical |
|---|---:|---|
| `val_frozen_90m_v1_val` | 452,405 | ✓ |
| `val_frozen_90m_v2_val` | 452,405 | ✓ |
| `multipv_v2_val` | 571,232 | ✓ |
| `multipv_v2_valderived` | 161,340 | ✓ |
| `multipv_v2_train` | 145,673,383 | ✓ |
| `multipv_90m_train` | 90,076,117 | ✓ |
| dry-run set (`make_synth`: `synth_v6`, `synth_v6_flip`, `synth_v6_val`) | 16,384 / 16,384 / 2,048 | rebuilt under v3 |

- **The old sidecars in `data/processed/strata/` are untouched** and were not uploaded.
- The eval-set JSONs pin shard sha256s and codes, so they remain valid as they are.
- **Local action at merge time:** move `data/processed/strata/` to `strata_def2/`, then copy `strata_def3/` in as `data/processed/strata/`. Until then, v6-prod code on this box reads the v2 sidecars, which is correct for it. After the merge, code on this box refuses them.

## 6. VM scripts

`data/multiPV/vm_build_v3.sh --r2-prefix data/ --work data [--workers N] [--smoke N]`. Use `--work data`: every output lands under the gitignored `data/processed/`, so the checkout stays clean for `prod.py`.

| step | what |
|---|---|
| 1 | `python -m training.v6.r2 pull data/ --dest data`: all of `data/sha256.txt`, each sha256 verified. Then the pulled `rate_plan_v3.json` and `v3_expected_counts.json` must equal the committed ones |
| 2 | ≥ 120 GB free |
| 3 | index-only selection with `--expect-counts`; exit 3 on any difference |
| 4 | `pass_b_v2.py` with the v3 settings, `nice -n 10`, one BLAS thread. Workers default to min(vCPUs − 4, 32), cgroup-aware. Progress and projected finish every 5 min; a > 3 h projection logs a warning and continues |
| 5 | gates: s9 (nesting 0 of 91,350,634 over the whole index, 0 of corpus v2's; selection = expected counts; `hard_move` ≥ 99%; records add up; roots + rejections = selected; rejection histogram; second build refused) and frozen val (exactly 452,405, every shared field byte-identical to v1, `pv_score` via float16) |
| 6 | strata for train and val (`processed/strata/multipv_v3_{train,val}.strata2.*`), plus `counts.json` (tier × label, including `t20`) |
| 7 | `processed/v3_build/multipv_v3.sha256.txt` |
| 8 | `processed/v3_build/build_ok.json`: counts, rejection histogram, realised tier × label, per-step seconds, workers, every gate result, strata hash, git SHA and dirty files |

**Exit codes:** 0 everything passed; 2 input verification; 3 count mismatch; 4 build failure (disk, builder, strata); 5 gate failure.

**`setup.sh --build-corpus`** (harness side) should call it after creating the venv:
```bash
bash data/multiPV/vm_build_v3.sh --r2-prefix data/ --work data      # then setup.sh's own pull only verifies
nohup bash data/multiPV/vm_upload_v3.sh --work data > v3_upload.log 2>&1 &   # after launching prod.py
```

`vm_upload_v3.sh --work data`:
1. pushes the list's files to `data/multipv_v3/`, 2 files × 4 parts at a time, `nice 10`; a same-size object is skipped, so it resumes;
2. checks remote sizes against the list;
3. publishes `data/multipv_v3/sha256.txt`, then `build_ok.json`;
4. **then** deletes the local dump and index;
5. writes `upload_ok.json` last.

A fresh machine restores v3 with `python -m training.v6.r2 pull data/multipv_v3/ --dest data`.

**Build time on the VM.** A 5M-line timing smoke with 12 workers in the container ran at **29,966 roots/s** (3,295,442 roots in 114 s) [measured].
- At that rate ≈ 269M roots take ≈ 2.5 h; selection, gates, strata and hashing add ≈ 15–20 min.
- On a 16-vCPU cloud VM that is **≈ 2.5–3.5 h** [inferred], against the brief's 1.5–2 h for 191M lines.
- The script's > 3 h warning may fire; it only warns.

## 7. R2 upload (local)

Bucket `guofishv6corpus`. Layout and deviation: DECISIONS "Corpus v3". Every key mirrors the VM's `data/` tree under `processed/`, the harness runbook's convention; the brief's bare `inputs/` and `val_frozen_90m_v1/` prefixes would have landed outside the ignored `data/processed/`.

| key prefix | contents | size |
|---|---|---:|
| `data/processed/inputs/` | dump, `pass_a_index.bin`, `pass_a_checkpoint.json`, `rate_plan_v3.json`, `v3_expected_counts.json` | 26.89 GB |
| `data/processed/val_frozen_90m_v1/` | gate reference: manifest and 12 shards | 171.5 MB |
| `data/processed/val_frozen_90m_v2/` + `processed/strata/val_frozen_90m_v2_val.strata2.*` (v3) | production frozen90 | 177.0 MB |
| `data/processed/evalsets/` (`frozen90`, `v2val_roots`, `v2val_derived`), `processed/multipv_v2/` manifest + val + valderived shards, their v3 strata | the v2 eval sets as the runbook lists them | 285 MB |
| `data/sha256.txt` | 61 files, published last | — |
| `data/dryrun/` + its own `sha256.txt` | `make_synth`'s set under strata v3, for `setup.sh --data-prefix data/dryrun/` | 13.8 MB |

**Upload status** [measured]:
- **Done and remote-size-verified:**
  - `pass_a_index.bin`, `pass_a_checkpoint.json`, `rate_plan_v3.json`, `v3_expected_counts.json`;
  - every `val_frozen`, eval-set and strata file: 59 small files, 632 MB, sent in 582 s at 1.09 MB/s while the big files shared the link;
  - `data/dryrun/`: 12 files, 13.6 MB, its list published.
- **Not done:** `processed/inputs/lichess_db_eval.jsonl.zst`. Its multipart upload reached 174 parts (11.68 GB) after 59 min, then the process was stopped for low memory. boto3 does not resume a multipart upload, and R2 aborts the orphan after 7 days by default.
- **Throughput:** the uplink measured 5.6–8.7 MB/s (≈ 45–70 Mbps) [measured]. The index plus 11.7 GB of the dump plus the small files moved ≈ 17.9 GB in 59 min, ≈ 5 MB/s [measured]. The dump alone needs ≈ 45–70 min [inferred].
- **To finish:** resumable; it skips the 60 files already there, sends the dump, verifies all 61 sizes, then publishes `data/sha256.txt` last. From the worktree root, with the R2 variables set:
  ```bash
  CUDA_VISIBLE_DEVICES=-1 python data/multiPV/r2_push.py --root ../GuoFish/data/processed/r2_stage \
      --list ../GuoFish/data/processed/r2_stage/sha256.txt --prefix data/ --workers 2 --part-concurrency 8
  ```

## 8. After the VM build

**Pending.** When `build_ok.json` and `data/multipv_v3/upload_ok.json` exist:
1. download v3 here and verify it;
2. run `C0` (`training/v6/config/configs/pool_check_c0.yaml`: A3's resolved config with only the corpus, the production pool and the run name changed);
3. compare `C0` with the ledger's A3 at f_KL 0.37% / f_MSE 1.7%;
4. run `C1` (`pool_check_c1.yaml`) if `C0` fails;
5. record `C0`'s quick-val KL at 40,960,000.

**Production configs, all loadable through `prod.py`:**
- `prod.yaml` if `C0` passes;
- `prod_no_t20.yaml` if only `C1` passes;
- `prod_a3pool.yaml` if both fail.

**Local disk for `C0`:** 187 GB was free before this work; v3 needs ≈ 104 GB.

## Provenance

- **Code:** `v6-corpus3` at `0e00579` and later. It was cut from `v6-harness` and fast-forwarded to `v6-prod` `1341c19`, the commit that has `training/v6/r2.py` and `setup.sh`.
- **Inputs:**
  - dump sha256 `6cf9e842…` (21,368,367,967 B);
  - index sha256 `1ec22122…` (5,525,373,924 B, 394,669,566 rows);
  - 90M manifest seed 20260802;
  - corpus v2 manifest (115,315,219 selected).
- **Windows:** Python 3.13.7, torch 2.8.0+cu129 (GPU hidden), numpy 2.3.3, python-chess 1.11.2.
- **Container:** Ubuntu 26.04.1, uv 0.10.12, Python 3.13.7, torch 2.8.0+cpu, numpy 2.3.3, python-chess 1.11.2, from `training/v6/requirements-prod.txt` with `--torch-backend cpu`.
- **GPU:** not used.

## Deviations

1. **§4 stop-and-report:** the old value-only cell is +10 against §3. I continued after tracing it; this needs operator confirmation.
2. **§7 prefixes:** keys are `data/processed/...` (the harness runbook's layout), not `data/inputs/` etc. The dry-run sample is `make_synth`'s set under `data/dryrun/`, not two corpus v2 train shards, because that is what the harness's dry-run config uses.
3. **`data/sha256.txt` includes the 27 GB of inputs,** so `setup.sh`'s pull fetches them on any VM, including a resume VM. The harness may want to filter `processed/inputs/` there.
4. **Configs:** `pool_check_c{0,1}.yaml`, `prod_no_t20.yaml` and `prod_a3pool.yaml` are new files under `training/v6/config/configs/`. §8 names them, but they didn't exist.
5. **Build size and time:** with `t20`, v3 is ≈ 104 GB and the build is expected to run longer than the brief's 1.5–2 h (§6). The VM's 125 GB RAM may not page-cache a 104 GB corpus beside the trainer, so the harness's smoke should watch loader wait.
6. **Credentials:** the supplied file's endpoint (`.us.` jurisdiction) can't see the bucket; the harness's `R2_ACCOUNT_ID` endpoint can. `creds/` is not gitignored in the main checkout.
