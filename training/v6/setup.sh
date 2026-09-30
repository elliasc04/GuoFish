#!/usr/bin/env bash
# Production VM setup (VM harness brief §6). From the repo root, on the v6-prod checkout,
# with .env in place (training/v6/PROD_RUNBOOK.md):
#
#   bash training/v6/setup.sh                   # checks, venv, data pull + verify, smoke
#
# Options:
#   --data-prefix P    store prefix to pull (default data/); holds sha256.txt
#   --config C         config the smoke runs (default training/v6/config/configs/prod.yaml)
#   --mb "A B"         smoke splits, micro_batch x accum (default "512x2 1024x1")
#   --steps N          smoke steps per split (default 300)
#   --set KEY=VALUE    passed to the smoke's trainer (repeatable)
#   --cpu              dry run on a CPU box: hardware checks warn instead of failing, CPU torch
#   --skip-data | --skip-smoke
set -euo pipefail
cd "$(dirname "$0")/../.."

PREFIX=data/; CONFIG=training/v6/config/configs/prod.yaml; MB="512x2 1024x1"; STEPS=300
CPU=0; SKIP_DATA=0; SKIP_SMOKE=0; SETS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --data-prefix) PREFIX="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --mb) MB="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --set) SETS+=("$2"); shift 2 ;;
    --cpu) CPU=1; shift ;;
    --skip-data) SKIP_DATA=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    *) sed -n 2,17p "$0"; exit 2 ;;
  esac
done

fail() { echo "SETUP FAILED: $*" >&2; exit 1; }
check() { if [ "$CPU" = 1 ]; then echo "  WARNING (--cpu): $*"; else fail "$*"; fi; }
ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }     # version $1 >= $2

echo "== 1/4 checks"
for c in git curl df awk sort; do command -v "$c" >/dev/null || fail "$c is not installed"; done
CUDA=0
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
  echo "  GPU: $GPU, compute capability $CC, driver supports CUDA $CUDA"
  ge "$CC" 8.0 || check "compute capability $CC < 8.0 (bf16 and the tested kernels need Ampere or newer)"
  ge "$CUDA" 12.6 || check "the driver supports CUDA $CUDA < 12.6"
else
  check "no working nvidia-smi: no NVIDIA GPU or driver"
fi
read -r FSTYPE AVAIL_KB <<< "$(df -PTk . | awk 'NR==2 {print $2, $5}')"
HAVE_KB=$(du -sk data/processed 2>/dev/null | awk '{print $1}'); HAVE_KB=${HAVE_KB:-0}
DISK_GB=$(( (AVAIL_KB + HAVE_KB) / 1024 / 1024 ))
echo "  disk: $FSTYPE, $(( AVAIL_KB / 1024 / 1024 )) GB free + $(( HAVE_KB / 1024 / 1024 )) GB already in data/processed"
case "$FSTYPE" in
  nfs*|cifs|smb*|fuse*|9p|ceph*|glusterfs|mfs|lustre) check "the repo is on a network filesystem ($FSTYPE); clone it onto local NVMe" ;;
esac
[ "$DISK_GB" -ge 200 ] || check "$DISK_GB GB for data < 200 GB"
MEM_KB=$(awk '/^MemTotal/ {print $2}' /proc/meminfo)
for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do     # a container's own limit
  if [ -r "$f" ] && [ "$(cat "$f")" != max ]; then L=$(( $(cat "$f") / 1024 )); [ "$L" -lt "$MEM_KB" ] && MEM_KB=$L; fi
done
MEM_GB=$(( MEM_KB / 1024 / 1024 ))
echo "  RAM: $MEM_GB GB"
[ "$MEM_GB" -ge 64 ] || check "RAM $MEM_GB GB < 64 GB"
[ "$MEM_GB" -ge 96 ] || echo "  WARNING: RAM $MEM_GB GB < 96 GB: the corpus will not fully page-cache"
SHM_GB=$(df -Pk /dev/shm | awk 'NR==2 {print int($2 / 1024 / 1024)}')
echo "  /dev/shm: $SHM_GB GB"
[ "$SHM_GB" -ge 16 ] || check "/dev/shm is $SHM_GB GB < 16 GB (DataLoader workers need it; a container's default is 64 MB)"
CPUS=$(nproc)
if [ -r /sys/fs/cgroup/cpu.max ]; then
  read -r Q P < /sys/fs/cgroup/cpu.max
  [ "$Q" != max ] && [ $(( Q / P )) -lt "$CPUS" ] && CPUS=$(( Q / P ))
fi
echo "  vCPUs: $CPUS"
[ "$CPUS" -ge 12 ] || check "$CPUS vCPUs < 12"

echo "== 2/4 environment"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/0.10.12/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -x .venv/bin/python ] || uv venv --python 3.13 .venv
if [ "$CPU" = 1 ]; then BACKEND=cpu
elif ge "$CUDA" 12.9; then BACKEND=cu129
elif ge "$CUDA" 12.8; then BACKEND=cu128
else BACKEND=cu126; fi
echo "  torch wheels: $BACKEND"
uv pip install --python .venv/bin/python --torch-backend "$BACKEND" -r training/v6/requirements-prod.txt
PY=.venv/bin/python
$PY - "$CPU" <<'EOF'
import sys, platform, torch, numpy, chess, yaml, zstandard, boto3, pytest
print(f"  python {platform.python_version()} | torch {torch.__version__} (CUDA {torch.version.cuda}) | "
      f"numpy {numpy.__version__} | chess {chess.__version__} | PyYAML {yaml.__version__} | "
      f"zstandard {zstandard.__version__} | boto3 {boto3.__version__} | pytest {pytest.__version__}")
try:
    import triton
    print(f"  triton {triton.__version__}")
except ImportError:
    print("  triton: not installed")
if sys.argv[1] != "1":
    assert torch.cuda.is_available(), "torch sees no CUDA device"
    print(f"  torch device: {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}")
EOF
[ "$CPU" = 1 ] && export CUDA_VISIBLE_DEVICES=-1

echo "== 3/4 data"
if [ "$SKIP_DATA" = 1 ]; then echo "  skipped"
else $PY -m training.v6.r2 pull "$PREFIX" --dest data --workers "$CPUS"; fi

echo "== 4/4 smoke"
if [ "$SKIP_SMOKE" = 1 ]; then echo "  skipped"
else
  $PY -m pytest -q -x training/v6/tests/test_m2_config.py training/v6/tests/test_prod.py -k "not dryrun"
  # shellcheck disable=SC2086
  $PY -m training.v6.tools.prod --smoke --config "$CONFIG" --mb $MB --steps "$STEPS" --set ${SETS[@]+"${SETS[@]}"}
fi
echo "setup OK"
