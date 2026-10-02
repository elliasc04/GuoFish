#!/usr/bin/env bash
# Build corpus v3 on the rented VM (corpus v3 brief §6). From the repo root, after setup.sh has
# made .venv, with the R2_* variables in the environment or .env:
#
#   bash data/multiPV/vm_build_v3.sh --r2-prefix data/ --work data [--workers N] [--smoke N]
#
# --smoke N: the same steps on the first N dump lines (a --limit build): no exact-count compare,
# whole-index gate checks reported n/a, no frozen-val gate, a 1 GB disk floor. For containers.
#
# <work> is the directory holding `processed/` (use `data`: everything lands under the gitignored
# data/processed/, so the checkout stays clean for prod.py). It gets:
#   processed/inputs/            dump, Pass A index, rate plan, expected counts (pulled)
#   processed/val_frozen_90m_v1  the frozen-val gate reference (pulled; with the rest of
#                                <r2-prefix>sha256.txt: frozen v2, eval sets, their strata)
#   processed/multipv_v3/        corpus v3 (train + val; no valderived, --derived-rate 0)
#   processed/strata/multipv_v3_{train,val}.strata2.{npy,json}
#   processed/v3_build/          logs, gate results, multipv_v3.sha256.txt, build_ok.json
#
# Steps: 1 pull + verify every sha256 | 2 >= 120 GB free | 3 index-only selection == the
# committed v3_expected_counts.json | 4 build (nice 10, one BLAS thread, progress every 5 min)
# | 5 gates | 6 strata | 7 sha256 list | 8 build_ok.json.
# Exit: 0 all passed; 2 input verification; 3 count mismatch; 4 build failure; 5 gate failure.
# An existing build_ok.json means the build is done: it prints it and exits 0.
set -euo pipefail
cd "$(dirname "$0")/../.."

PREFIX=data/; WORK=""; WORKERS=""; SMOKE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --r2-prefix) PREFIX="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --smoke) SMOKE="$2"; shift 2 ;;
    *) sed -n 2,24p "$0"; exit 1 ;;
  esac
done
[ -n "$WORK" ] || { sed -n 2,24p "$0"; exit 1; }
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python3
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
MPV=data/multiPV
IN=$WORK/processed/inputs
OUT=$WORK/processed/multipv_v3
LOG=$WORK/processed/v3_build
STRATA=$WORK/processed/strata
M90=$MPV/manifests/dataset_manifest_90m.json
mkdir -p "$LOG"
T0=$(date +%s)
declare -A SECS
step() { echo "== $(date -u +%H:%M:%SZ) $*" | tee -a "$LOG/steps.log"; }
die() { echo "VM BUILD FAILED ($1): $2" | tee -a "$LOG/steps.log" >&2; exit "$1"; }

if [ -f "$LOG/build_ok.json" ]; then
  echo "corpus v3 already built:"; cat "$LOG/build_ok.json"; exit 0
fi

step "1/8 pull ${PREFIX} and verify every sha256"
t=$(date +%s)
"$PY" -m training.v6.r2 pull "$PREFIX" --dest "$WORK" --workers 16 2>&1 | tee "$LOG/pull.log" \
  || die 2 "pull/verify failed (see $LOG/pull.log)"
for f in rate_plan_v3.json v3_expected_counts.json; do
  cmp -s "$IN/$f" "$MPV/$f" || die 2 "$IN/$f differs from the committed $MPV/$f"
done
SECS[pull]=$(( $(date +%s) - t ))

step "2/8 disk"
FREE_GB=$(( $(df -Pk "$WORK" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
echo "  $FREE_GB GB free in $WORK"
NEED_GB=120; [ -n "$SMOKE" ] && NEED_GB=1
[ "$FREE_GB" -ge "$NEED_GB" ] || die 4 "$FREE_GB GB free in $WORK < $NEED_GB GB"

if [ -z "$WORKERS" ]; then
  CPUS=$(nproc)
  if [ -r /sys/fs/cgroup/cpu.max ]; then                      # a container's own CPU quota
    read -r Q P < /sys/fs/cgroup/cpu.max
    [ "$Q" != max ] && [ $(( Q / P )) -lt "$CPUS" ] && CPUS=$(( Q / P ))
  fi
  WORKERS=$(( CPUS - 4 )); [ "$WORKERS" -gt 32 ] && WORKERS=32; [ "$WORKERS" -lt 1 ] && WORKERS=1
fi
BUILD=("$MPV/pass_b_v2.py" --rate-plan "$MPV/rate_plan_v3.json" --source "$IN/lichess_db_eval.jsonl.zst"
       --index "$IN/pass_a_index.bin" --floor-manifest "$M90"
       --nest-manifest "$MPV/manifests/dataset_manifest_v2.json"
       --value-min-depth 20 --policy-min-depth 20 --derived-rate 0 --seed 20260802)
EXPECT_ARGS=(--expect-counts "$MPV/v3_expected_counts.json")
if [ -n "$SMOKE" ]; then BUILD+=(--limit "$SMOKE"); EXPECT_ARGS=(); echo "  SMOKE: first $SMOKE lines"; fi

step "3/8 index-only selection against v3_expected_counts.json"
t=$(date +%s)
set +e
"$PY" "${BUILD[@]}" "${EXPECT_ARGS[@]}" --dry-run > "$LOG/selection.json" 2> "$LOG/selection.err"; rc=$?
set -e
cat "$LOG/selection.err"
[ "$rc" = 3 ] && die 3 "the selection does not reproduce v3_expected_counts.json"
[ "$rc" = 0 ] || die 4 "selection failed (rc $rc; see $LOG/selection.err)"
SECS[selection]=$(( $(date +%s) - t ))

step "4/8 build: $WORKERS workers, nice 10"
EXPECT=$("$PY" -c "import json; print(json.load(open('$LOG/selection.json'))['selected_lines'])")
t=$(date +%s)
(   # progress and projected finish every 5 min; the builder logs "roots N | ... | R roots/s"
  set +e
  warned=0
  while sleep 300; do
    ln=$(grep -a "roots/s" "$LOG/build.log" 2>/dev/null | tail -1) || true
    [ -n "$ln" ] || continue
    roots=$(echo "$ln" | sed -E 's/^ *roots ([0-9,]+).*/\1/' | tr -d ,)
    el=$(( $(date +%s) - t ))
    [ "$roots" -gt 0 ] || continue
    eta=$(( el * (EXPECT - roots) / roots ))
    tot_h=$(awk -v a="$el" -v b="$eta" 'BEGIN {printf "%.2f", (a + b) / 3600}')
    msg="  progress: $roots / ~$EXPECT roots ($(( 100 * roots / EXPECT ))%) after $(( el / 60 )) min; projected finish $(date -u -d "+${eta} sec" +%H:%MZ) (build ${tot_h} h)"
    echo "$msg" | tee -a "$LOG/progress.log"
    if [ "$warned" = 0 ] && awk -v h="$tot_h" 'BEGIN {exit !(h > 3)}'; then
      echo "  WARNING: projected build time ${tot_h} h > 3 h; continuing (the operator decides)" | tee -a "$LOG/progress.log"
      warned=1
    fi
  done
) &
MON=$!
set +e
nice -n 10 "$PY" "${BUILD[@]}" "${EXPECT_ARGS[@]}" --out-dir "$OUT" --workers "$WORKERS" > "$LOG/build.log" 2>&1; rc=$?
set -e
kill "$MON" 2>/dev/null || true
tail -n 25 "$LOG/build.log"
SECS[build]=$(( $(date +%s) - t ))
# 0 ok; 3 = hard_move coverage < 99%, which the gates report; 1 (invariant violations) or anything else = build failure
[ "$rc" = 0 ] || [ "$rc" = 3 ] || die 4 "builder exited $rc (see $LOG/build.log)"
[ -f "$OUT/manifest.json" ] || die 4 "no manifest after the build"

step "5/8 gates"
t=$(date +%s)
GATES=0
S9_ARGS=("${EXPECT_ARGS[@]}"); [ -n "$SMOKE" ] && S9_ARGS=(--smoke)
"$PY" "$MPV/corpus_v2_gates.py" s9 --corpus "$OUT" --manifest-copy none "${S9_ARGS[@]}" \
    --out "$LOG/gate_s9.json" > /dev/null || GATES=1
"$PY" -c "import json; d=json.load(open('$LOG/gate_s9.json')); print('  s9:', 'PASS' if d['passed'] else 'FAIL', d['checks'])" || GATES=1
rm -rf "$LOG/val_frozen_from_v3"
if [ -n "$SMOKE" ]; then echo '{"skipped": "smoke", "passed": true}' > "$LOG/gate_frozen.json"; echo "  frozen val: n/a (smoke)"
else "$PY" "$MPV/extract_frozen_v2.py" --corpus "$OUT" --out "$LOG/val_frozen_from_v3" \
    --frozen-v1 "$WORK/processed/val_frozen_90m_v1" --index "$IN/pass_a_index.bin" --m90 "$M90" \
    --expect 452405 > "$LOG/gate_frozen.json" || GATES=1
"$PY" -c "import json; d=json.load(open('$LOG/gate_frozen.json')); print('  frozen val:', 'PASS' if d['passed'] else 'FAIL', d['count_v2'], d['field_mismatches'])" || GATES=1
fi
SECS[gates]=$(( $(date +%s) - t ))
[ "$GATES" = 0 ] || die 5 "a gate failed (see $LOG/gate_*.json)"

step "6/8 strata (train, val)"
t=$(date +%s)
"$PY" -m training.v6.tools.build_strata --shards "$OUT" --split train val --out-dir "$STRATA" \
    --index "$IN/pass_a_index.bin" --m90 "$M90" > "$LOG/strata.log" 2>&1 || die 4 "build_strata failed (see $LOG/strata.log)"
"$PY" "$MPV/corpus_v2_gates.py" counts --corpus "$OUT" --frozen-v2 none --strata-dir "$STRATA" \
    > "$LOG/counts.json" || die 4 "the strata counts failed"
SECS[strata]=$(( $(date +%s) - t ))

step "7/8 sha256 list"
t=$(date +%s)
rm -f "$LOG/multipv_v3.sha256.txt"
"$PY" tools/make_sha256_list.py --root "$WORK" --workers 16 --out "$LOG/multipv_v3.sha256.txt" \
    processed/multipv_v3 processed/strata/multipv_v3_train.strata2.npy processed/strata/multipv_v3_train.strata2.json \
    processed/strata/multipv_v3_val.strata2.npy processed/strata/multipv_v3_val.strata2.json
SECS[sha256]=$(( $(date +%s) - t ))

step "8/8 build_ok.json"
SECS_JSON=$(for k in "${!SECS[@]}"; do printf '"%s": %s,' "$k" "${SECS[$k]}"; done)
"$PY" - "$OUT" "$LOG" "$STRATA" "$WORKERS" "{${SECS_JSON%,}}" "$(( $(date +%s) - T0 ))" "${SMOKE:-0}" <<'EOF'
import hashlib, json, subprocess, sys, time
from pathlib import Path
out, log, strata, workers, secs, total = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), json.loads(sys.argv[5]), int(sys.argv[6])
smoke = int(sys.argv[7]) or None
m = json.loads((out / "manifest.json").read_text())
s9, fz = json.loads((log / "gate_s9.json").read_text()), json.loads((log / "gate_frozen.json").read_text())
cnt = json.loads((log / "counts.json").read_text())
st = {s: json.loads((strata / f"multipv_v3_{s}.strata2.json").read_text()) for s in ("train", "val")}
hashes = {v["definition_hash"] for v in st.values()}
assert len(hashes) == 1, hashes
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
ok = {
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "git_sha": git("rev-parse", "HEAD"), "git_dirty": git("status", "--porcelain", "--untracked-files=all").splitlines(),
    "smoke_limit_lines": smoke, "workers": workers, "seconds": {**secs, "total": total},
    "counts": {k: m[k] for k in ("selected_lines", "roots", "records_train", "records_val", "records_valderived",
                                  "selected_counts", "hard_move_coverage", "invariant_violation_rate")},
    "rejection_histogram": m["rejection_histogram"],
    "realised_roots_tier_x_label": cnt["selected_vs_actual_roots"],
    "realised_policy_share_roots": cnt["realised_policy_share_roots"],
    "depth_tier_x_label": {s: v["depth_tier_x_label"] for s, v in cnt["per_split"].items()},
    "gates": {"s9": s9["checks"], "s9_passed": s9["passed"],
              "frozen_val": fz if smoke else {k: fz[k] for k in ("count_v2", "expected", "shard_counts_match",
                                                                     "field_mismatches", "passed")}},
    "strata_definition_hash": hashes.pop(),
    "manifest_sha256": hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest(),
    "sha256_list": "processed/v3_build/multipv_v3.sha256.txt",
}
assert ok["gates"]["s9_passed"] and ok["gates"]["frozen_val"]["passed"]
(log / "build_ok.json").write_text(json.dumps(ok, indent=1) + "\n")
print(json.dumps({k: ok[k] for k in ("seconds", "workers", "strata_definition_hash", "git_sha")}, indent=1))
EOF
step "corpus v3 built and gated: $LOG/build_ok.json"
