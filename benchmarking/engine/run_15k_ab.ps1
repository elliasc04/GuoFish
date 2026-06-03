<#
A/B benchmark: two MCTS configs vs Stockfish 2800 at a TRUE 15k-sim budget.

Each config plays 100 games (50 paired openings, colors reversed via -repeat)
over the SAME sequential opening slice, so the A-vs-B comparison is paired on
openings (lower-variance delta than two independent absolute ratings).

The two configs are the statistically-tied leaders from the 10k sweep, now
re-measured at the budget we actually ship:
    A = C_PUCT 2.00 / VL 3   / depth 60   (baseline champion, 2878 @ 10k)
    B = C_PUCT 2.00 / VL 2.5 / depth 80   (best alternative,  2871 @ 10k)

Budget is forced with --sims 15000 (bypasses the GUI 'nodes'/--sim-cap path
entirely) so it cannot be silently clamped. Stockfish settings, openings, and
adjudication match the existing 2878 runs so the numbers stay comparable.

Run from the repo root:
    powershell -ExecutionPolicy Bypass -File benchmarking/engine/run_15k_ab.ps1
#>

$ErrorActionPreference = "Stop"

# --- Paths ---
# Auto-derive repo root from this script's location (benchmarking/engine/..).
$Repo      = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Cutechess = Join-Path $Repo "cutechess-1.4.0-win64\cutechess-cli.exe"
$Ordo      = Join-Path $Repo "ordo-win64.exe"
$Stockfish = "stockfish-windows-x86-64-avx2.exe"   # resolved from engine dir
$Model     = "models/guofish4/guofish4_25.6M_policy_final.pt"
$Openings  = "assets/8moves_v3.pgn"

if (-not (Test-Path $Cutechess)) { throw "cutechess not found: $Cutechess" }

# Fresh timestamped output dir so we never append to / clobber prior PGNs.
$Stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = Join-Path $Repo "benchmarking/engine/games/v4/ab_15k_$Stamp"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host "Output dir: $OutDir`n"

# --- Configs (engine name must be unique so ordo rates them separately) ---
$Configs = @(
  @{ Name = "Guofish4_c2.0_vl3_d60";   CPuct = "2.0"; VL = "3";   Depth = "60" },
  @{ Name = "Guofish4_c2.0_vl2.5_d80"; CPuct = "2.0"; VL = "2.5"; Depth = "80" }
)

function Invoke-Match($cfg) {
  $pgn = Join-Path $OutDir ("{0}.pgn" -f $cfg.Name)
  Write-Host "=== $($cfg.Name)  (C_PUCT=$($cfg.CPuct) VL=$($cfg.VL) depth=$($cfg.Depth)) ==="
  Write-Host "    -> $pgn"

  $ccArgs = @(
    "-engine", "name=$($cfg.Name)", "cmd=python",
      "arg=playing/uci_wrapper.py",
      "arg=--model",         "arg=$Model",
      "arg=--workers",       "arg=8",
      "arg=--sims",          "arg=15000",          # force the budget, no clamp
      "arg=--c-puct",        "arg=$($cfg.CPuct)",
      "arg=--virtual-loss",  "arg=$($cfg.VL)",
      "arg=--max-tree-depth","arg=$($cfg.Depth)",
      "dir=$Repo", "proto=uci", "tc=inf", "nodes=15000", "timemargin=300000",
    "-engine", "name=Stockfish", "cmd=$Stockfish", "dir=$Repo", "proto=uci",
      "option.UCI_LimitStrength=true", "option.UCI_Elo=2800", "option.Threads=1",
      "tc=10+0.1",
    "-openings", "file=$Openings", "format=pgn", "order=sequential", "plies=16",
    "-resign", "movecount=3", "score=600",
    "-draw",   "movenumber=40", "movecount=8", "score=10",
    "-concurrency", "7",
    "-rounds", "50", "-games", "2", "-repeat",
    "-pgnout", $pgn
  )

  # Pipe cutechess stdout to the console via Out-Host so it streams live AND
  # does NOT land in the function's output stream. Without this, PowerShell
  # folds cutechess's "Started game N of 100..." lines into the return value
  # and the caller collects log text instead of the .pgn path.
  & $Cutechess @ccArgs | Out-Host
  if ($LASTEXITCODE -ne 0) { throw "cutechess exited $LASTEXITCODE for $($cfg.Name)" }
  return $pgn
}

# --- Run both courses sequentially ---
$pgns = @()
foreach ($cfg in $Configs) { $pgns += (Invoke-Match $cfg) }

# --- Pool both PGNs; Stockfish is the shared anchor that links the two configs ---
$pooled = Join-Path $OutDir "pooled.pgn"
$combined = ($pgns | ForEach-Object { Get-Content -Raw $_ }) -join "`n"
Set-Content -Path $pooled -Value $combined -Encoding utf8
Write-Host "`nPooled PGN: $pooled"

# --- Joint ratings ---
# Anchor Stockfish at its configured UCI_Elo (2800 here). The anchor is a pure
# additive offset on the whole scale, so it MUST match the strength SF was
# actually set to, or every rating is shifted by the difference. (-a 2500 only
# made sense for runs where SF was set to ~2500.)
$SFElo = 2800
if (Test-Path $Ordo) {
  Write-Host "`n=== Ordo ratings (Stockfish anchored at $SFElo) ==="
  & $Ordo -p $pooled -A "Stockfish" -a $SFElo
} else {
  Write-Host "`nordo not found at $Ordo; rate manually:"
  Write-Host "  `"$Ordo`" -p `"$pooled`" -A `"Stockfish`" -a $SFElo"
}

Write-Host "`nDone."
