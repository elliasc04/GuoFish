# C12 Step 1b — Nsight Compute on the graphed forward.
#
# MUST BE RUN FROM AN ELEVATED POWERSHELL. Nsight Compute reads GPU performance
# counters, which are admin-only on this machine (ERR_NVGPUCTRPERM;
# HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak
# \RmProfilingAdminOnly is unset, i.e. the default of 1). Everything else in C12
# runs unprivileged; this one step does not.
#
#   Right-click PowerShell -> Run as administrator, then:
#     cd 'C:\Users\Ethan Guo\Github\GuoFish'
#     powershell -ExecutionPolicy Bypass -File tools\run_ncu_c12.ps1
#
# It writes runs\c12\ncu_shape24.ncu-rep and runs\c12\ncu_shape128.ncu-rep.
# Nothing else in the repo is touched. Expect ~2-6 minutes per shape: `--set
# full` replays every kernel several times to collect all metric passes.
#
# WHY THE NVTX FILTER IS NOT OPTIONAL
# ===================================
# Building the evaluator captures nine graph shapes and warms each of them, so
# an unfiltered run would profile a few thousand kernels that are not the thing
# under test and take most of an hour. `--nvtx-include` restricts profiling to
# the single steady-state forward that tools/profile_c12.py brackets, and
# `--iters 2` makes that exactly one forward: iteration 0 is inside a `warmup`
# range, iteration 1 is inside `FORWARD:shape<N>`.

$ErrorActionPreference = 'Stop'

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error ("This shell is not elevated. Nsight Compute will fail with " +
                 "ERR_NVGPUCTRPERM. Re-run from an administrator PowerShell.")
}

$repo = Split-Path -Parent $PSScriptRoot
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.0\target\windows-desktop-win7-x64\ncu.exe'
if (-not (Test-Path $ncu)) { Write-Error "ncu.exe not found at $ncu" }

$outDir = Join-Path $repo 'runs\c12'
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

Set-Location $repo

# Shape 24 is what the fresh-root regime actually runs (21.2 rows per crossing,
# padded up to 24). Shape 128 is the knee from BENCH.md C10b-3a, and the
# comparison between the two is what says whether the forward is occupancy-bound
# at the shape the engine ships at.
foreach ($shape in 24, 128) {
    $out = Join-Path $outDir "ncu_shape$shape"
    Write-Host "=== ncu shape $shape -> $out.ncu-rep ===" -ForegroundColor Cyan
    & $ncu --target-processes all `
           --graph-profiling node `
           --set full `
           --nvtx --nvtx-include "FORWARD:shape$shape/" `
           --export $out --force-overwrite `
           python tools/profile_c12.py forward --shape $shape --iters 2
    if ($LASTEXITCODE -ne 0) { Write-Error "ncu failed for shape $shape (exit $LASTEXITCODE)" }
}

Write-Host ""
Write-Host "Done. Reports:" -ForegroundColor Green
Get-ChildItem (Join-Path $outDir 'ncu_shape*.ncu-rep') | ForEach-Object { Write-Host "  $($_.FullName)" }
Write-Host "Tell Claude they exist; parsing them needs no privileges."
