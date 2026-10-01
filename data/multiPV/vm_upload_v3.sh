#!/usr/bin/env bash
# Upload corpus v3 from the VM to R2, alongside training (corpus v3 brief §6). From the repo root:
#
#   nohup bash data/multiPV/vm_upload_v3.sh --work data > v3_upload.log 2>&1 &
#
# Needs <work>/processed/v3_build/build_ok.json (vm_build_v3.sh). Uploads every file of
# multipv_v3.sha256.txt (corpus v3 and its train/val strata) to data/multipv_v3/<path>, two
# files and four parts at a time so training isn't starved. Objects already there at the right
# size are skipped (sync semantics: re-run to resume). Then: remote sizes are checked against the
# list, the list is published as data/multipv_v3/sha256.txt, build_ok.json is uploaded, and
# ONLY THEN are the local dump and index copies deleted. data/multipv_v3/upload_ok.json is last.
# A fresh machine restores v3 with:  python -m training.v6.r2 pull data/multipv_v3/ --dest data
set -euo pipefail
cd "$(dirname "$0")/../.."

WORK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --work) WORK="$2"; shift 2 ;;
    *) sed -n 2,13p "$0"; exit 1 ;;
  esac
done
[ -n "$WORK" ] || { sed -n 2,13p "$0"; exit 1; }
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python3
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
LOG=$WORK/processed/v3_build
DEST=data/multipv_v3/
[ -f "$LOG/build_ok.json" ] || { echo "no $LOG/build_ok.json: run vm_build_v3.sh first" >&2; exit 1; }

echo "== $(date -u +%H:%M:%SZ) upload corpus v3 to $DEST"
nice -n 10 "$PY" data/multiPV/r2_push.py --root "$WORK" --list "$LOG/multipv_v3.sha256.txt" \
    --prefix "$DEST" --workers 2 --part-concurrency 4 | tee "$LOG/upload.log"
"$PY" -m training.v6.r2 put "${DEST}build_ok.json" "$LOG/build_ok.json"

echo "== $(date -u +%H:%M:%SZ) remote verified; deleting the local dump and index copies"
rm -f "$WORK/processed/inputs/lichess_db_eval.jsonl.zst" "$WORK/processed/inputs/pass_a_index.bin"

tail -n 1 "$LOG/upload.log" > "$LOG/upload_ok.json"          # r2_push's last line: its stats JSON
"$PY" -m training.v6.r2 put "${DEST}upload_ok.json" "$LOG/upload_ok.json"
echo "== $(date -u +%H:%M:%SZ) done: ${DEST}upload_ok.json"
