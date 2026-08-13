#requires -Version 5.1
<#
    Two v6 matches, run SERIALLY. One GPU, one match at a time.

        powershell -ExecutionPolicy Bypass -File tools\run_90M_vs_stockfish.ps1
        powershell -ExecutionPolicy Bypass -File tools\run_90M_vs_stockfish.ps1 -Skip3000
        powershell -ExecutionPolicy Bypass -File tools\run_90M_vs_stockfish.ps1 -PreflightOnly

    SUITE 3000  90M-corpus at 50k nodes vs Stockfish (UCI_Elo 3000) at 100k nodes,
                50 games, concurrency 2. Runs FIRST.
    SUITE 2900  90M-corpus at 50k nodes vs Stockfish (UCI_Elo 2900) at 100k nodes,
                50 games, concurrency 2. Runs second.

    Structure, adjudication, node/arena convention and house discipline (preflight,
    superseding stale artifacts, command recording, serial GPU ownership) are all
    copied from tools/run_head2head.ps1's Suite B, which established the 50k/100k
    node split against Stockfish for a v5-architecture 10.9M-param net. The 90M
    checkpoint is the same architecture and carries the same locked value_scale
    (290.6806 -- see data/multiPV/corpus_90m_report.md D1), so that convention
    applies unchanged; nothing here was re-derived for the 90M corpus specifically.

    WHY SERIAL. GPU share is 80-96% across the C10b grid, so two matches at once
    corrupt both. This script never starts the 2900 suite until the 3000 suite's
    process has exited.

    B RUNS EVEN IF A FAILS, and the summary says so. See run_head2head.ps1 for the
    rationale -- two independent verdicts beat one abandoned night.
#>
[CmdletBinding()]
param(
    [switch]$Skip2900,
    [switch]$Skip3000,
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Paths and the shipping configuration
# ---------------------------------------------------------------------------

$Repo      = 'C:\Users\Ethan Guo\Github\GuoFish'
$Cutechess = Join-Path $Repo 'cutechess-1.4.0-win64\cutechess-cli.exe'
$Openings  = Join-Path $Repo 'assets\8moves_v3.pgn'
$Stockfish = Join-Path $Repo 'stockfish-windows-x86-64-avx2.exe'
$Model90M  = Join-Path $Repo 'models\guofish5_90M\v5_10.9M_best.pt'
$GamesRoot = Join-Path $Repo 'benchmarking\engine\games\v6\head2head'

# C10b-3g's selection. Passed explicitly on every arm even though it IS the
# EngineConfig default, so an arm that silently inherited a changed default
# would be unreadable later.
$ShipWorkers   = 1
$ShipOutstand  = 24
$ShipMaxBatch  = 128
$ShipAffinity  = 'none'

# 90M is v5-architecture (same 10.9M-param net), so it takes the same
# v5 Q-denominated constants as the 10M/20M corpus checkpoints.
$CPuctInit = 1.43
$FpuTree   = 0.3

# ARENA. capacity_suite.py's ARENA_NODES_PER_SIM = 75 -- see run_head2head.ps1
# for why the default 37.5/sim is unsafe. 50k nodes is above every budget
# capacity_suite measured, so this margin is the convention carried over, not a
# fit specific to the 90M checkpoint.
$ArenaPerSim = 75

function Get-Arena { param([int]$Nodes) return [int]($Nodes * $ArenaPerSim) }

$GuoFishNodes   = 50000
$StockfishNodes = 100000
$GuoFishArena   = Get-Arena $GuoFishNodes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Log {
    param([string]$Message = '')
    if ($Message -eq '') { Write-Host ''; return }
    $stamp = (Get-Date).ToString('HH:mm:ss')
    Write-Host "[$stamp] $Message"
}

function Format-Duration {
    param([TimeSpan]$Span)
    return ('{0:d2}:{1:d2}:{2:d2}' -f [int]$Span.TotalHours, $Span.Minutes, $Span.Seconds)
}

function Test-Preflight {
    <#
        Every path this run depends on, checked before the first game.
        Returns the list of problems; empty means good to go.
    #>
    $problems = @()
    $required = [ordered]@{
        'cutechess-cli'    = $Cutechess
        'openings book'    = $Openings
        'stockfish'        = $Stockfish
        '90M-corpus model' = $Model90M
        'uci wrapper'      = (Join-Path $Repo 'playing\uci_wrapper_v6.py')
    }
    foreach ($name in $required.Keys) {
        $path = $required[$name]
        if (-not (Test-Path -LiteralPath $path)) {
            $problems += "MISSING $name : $path"
        }
    }
    # `cmd=python` resolves through PATH inside cutechess, so a python that this
    # shell cannot see is a python cutechess cannot see either.
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) { $problems += 'MISSING python on PATH (engines are launched as cmd=python)' }
    return $problems
}

function Move-StaleArtifacts {
    <#
        Move a previous run of this event aside. REQUIRED FOR CORRECTNESS, not
        tidiness: `-pgnout` appends and `stderr=` accumulates, so re-running in
        place silently blends two runs into one artifact.

        Moved, never deleted.
    #>
    param([string]$Directory, [string]$Event)

    if (-not (Test-Path -LiteralPath $Directory)) { return }
    $stale = @(Get-ChildItem -LiteralPath $Directory -File -ErrorAction SilentlyContinue |
               Where-Object { $_.Name -like "$Event.*" -or $_.Name -like '*.stderr.log' })
    if ($stale.Count -eq 0) { return }

    $attic = Join-Path $Directory ('_superseded_' + (Get-Date).ToString('yyyyMMddTHHmmss'))
    New-Item -ItemType Directory -Path $attic -Force | Out-Null
    foreach ($file in $stale) {
        Move-Item -LiteralPath $file.FullName -Destination (Join-Path $attic $file.Name) -Force
    }
    Write-Log ("    superseded {0} artifact(s) from a previous run -> {1}\" -f $stale.Count, (Split-Path $attic -Leaf))
}

function New-GuoFishArm {
    <#
        The GuoFish engine block for the 90M-corpus checkpoint. THE SHIPPING
        CONFIG IS THE DATACLASS DEFAULT; it is passed explicitly so the artifact
        is self-describing.

        Book and Syzygy are left ON (the C11b default) for this bench, by
        request -- unlike run_head2head.ps1's Suite A/B, which force them off
        because both BYPASS MCTS and would contaminate that script's
        delivered-sims telemetry. Here the goal is realistic strength against
        Stockfish, and both moves being zero-simulation whenever the engine's
        own book/tablebase fires is expected, not a defect. `-openings` still
        governs the first 16 plies of each game; book/tablebase only act after
        that, and only on positions each source actually covers.
    #>
    param([string]$StderrPath)
    return @(
        '-engine', 'name=v5-90M', 'cmd=python',
        'arg=-u', 'arg=playing/uci_wrapper_v6.py',
        'arg=--model', "arg=$Model90M",
        'arg=--threads', "arg=$ShipWorkers",
        'arg=--max-outstanding', "arg=$ShipOutstand",
        'arg=--max-batch', "arg=$ShipMaxBatch",
        'arg=--affinity', "arg=$ShipAffinity",
        'arg=--c-puct-init', "arg=$CPuctInit",
        'arg=--fpu-tree', "arg=$FpuTree",
        'arg=--arena-capacity', "arg=$GuoFishArena",
        "dir=$Repo", 'proto=uci',
        'tc=inf', "nodes=$GuoFishNodes", 'timemargin=300000',
        "stderr=$StderrPath"
    )
}

function New-StockfishArm {
    param([int]$Elo, [string]$StderrPath)
    return @(
        '-engine', 'name=Stockfish', "cmd=$Stockfish",
        "dir=$Repo", 'proto=uci',
        'option.UCI_LimitStrength=true', "option.UCI_Elo=$Elo", 'option.Threads=1',
        'tc=inf', "nodes=$StockfishNodes", 'timemargin=300000',
        "stderr=$StderrPath"
    )
}

function Invoke-Match {
    <#
        Run one cutechess match to completion, teeing its output to a log and
        echoing only the score lines. Returns a result object.
    #>
    param(
        [string]$Event,
        [string]$Directory,
        [string[]]$Arguments
    )

    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
    Move-StaleArtifacts -Directory $Directory -Event $Event

    $logPath     = Join-Path $Directory "$Event.cutechess.log"
    $commandPath = Join-Path $Directory "$Event.command.txt"

    # The house convention: the exact invocation, recorded beside its own PGN.
    $rendered = ($Arguments | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join ' '
    Set-Content -LiteralPath $commandPath -Value ('"' + $Cutechess + '" ' + $rendered) -Encoding utf8

    Write-Log "    $($Arguments.Count) tokens -> $Event.cutechess.log"
    $started = [System.Diagnostics.Stopwatch]::StartNew()
    $games = 0

    # Native stdout is streamed through the pipeline so the log is written and
    # flushed as the match runs -- a match killed at hour three must still leave
    # a readable log. 2>&1 is deliberately NOT used: it would wrap cutechess's
    # stderr lines in ErrorRecords under PowerShell 5.1 and set $? to false on a
    # clean exit.
    & $Cutechess @Arguments | ForEach-Object {
        $_ | Out-File -LiteralPath $logPath -Append -Encoding utf8
        if ($_ -match '^Score of' -or $_ -match 'SPRT') {
            if ($_ -match '^Score of') { $games++ }
            Write-Log "      [$games] $_"
        }
    }
    $code = $LASTEXITCODE
    $started.Stop()

    Write-Log ("    $Event exited $code after {0} ({1} score lines)" -f (Format-Duration $started.Elapsed), $games)
    return [pscustomobject]@{
        Event    = $Event
        ExitCode = $code
        Elapsed  = $started.Elapsed
        Log      = $logPath
        Games    = $games
    }
}

function Invoke-StockfishSuite {
    param([int]$Elo)

    $event     = "90M50K_vs_SF$Elo"
    $directory = Join-Path $GamesRoot $event

    Write-Log ''
    Write-Log ('=' * 74)
    Write-Log "SUITE $Elo -- 90M-corpus @ ${GuoFishNodes} nodes vs Stockfish UCI_Elo $Elo @ ${StockfishNodes} nodes, 50 games"
    Write-Log ('=' * 74)
    Write-Log "    arena $GuoFishArena nodes ($ArenaPerSim/sim; carried over from run_head2head.ps1 Suite B)"

    New-Item -ItemType Directory -Path $directory -Force | Out-Null

    $arguments = @()
    $arguments += New-GuoFishArm -StderrPath (Join-Path $directory 'v5-90M.stderr.log')
    $arguments += New-StockfishArm -Elo $Elo -StderrPath (Join-Path $directory 'stockfish.stderr.log')
    $arguments += @('-openings', "file=$Openings", 'format=pgn', 'order=sequential', 'plies=16')
    # House adjudication. Valid here because value_scale (290.6806) is locked
    # across every v5-architecture checkpoint including the 90M corpus one, so
    # `score cp = value_scale * atanh(q)` puts the GuoFish arm on the same
    # scale as the 10M/20M matches and -resign fires symmetrically.
    $arguments += @('-resign', 'movecount=3', 'score=600')
    $arguments += @('-draw', 'movenumber=40', 'movecount=8', 'score=10')
    $arguments += @('-recover', '-concurrency', '2', '-rounds', '25', '-games', '2', '-repeat',
                    '-event', $event,
                    '-pgnout', (Join-Path $directory "$event.pgn"))

    return Invoke-Match -Event $event -Directory $directory -Arguments $arguments
}

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

$suiteStarted = [System.Diagnostics.Stopwatch]::StartNew()

Write-Log ('=' * 74)
Write-Log 'v6 90M-corpus vs Stockfish -- two matches, serial. One GPU, one match at a time.'
Write-Log ('=' * 74)

$problems = Test-Preflight
if ($problems.Count -gt 0) {
    Write-Log ''
    foreach ($problem in $problems) { Write-Log "  $problem" }
    Write-Log ''
    throw "Preflight failed with $($problems.Count) problem(s); nothing was run."
}
Write-Log '  preflight: all binaries, checkpoints and the opening book resolve'

if ($PreflightOnly) {
    Write-Log '  -PreflightOnly: stopping here.'
    return
}

$results = @()

if (-not $Skip3000) { $results += Invoke-StockfishSuite -Elo 3000 }
else                { Write-Log 'SUITE 3000 skipped (-Skip3000)' }

# SERIAL. Invoke-Match does not return until the child has exited, so the 2900
# suite cannot start while the 3000 suite still owns the GPU. It runs even if
# 3000 failed -- two independent verdicts beat one abandoned night.
if (-not $Skip2900) { $results += Invoke-StockfishSuite -Elo 2900 }
else                { Write-Log 'SUITE 2900 skipped (-Skip2900)' }

$suiteStarted.Stop()

Write-Log ''
Write-Log ('=' * 74)
Write-Log 'SUMMARY'
Write-Log ('=' * 74)
foreach ($result in $results) {
    $verdict = if ($result.ExitCode -eq 0) { 'ok' } else { "EXIT $($result.ExitCode)" }
    Write-Log ('  {0,-22} {1,-10} {2}  ({3} score lines)' -f `
               $result.Event, $verdict, (Format-Duration $result.Elapsed), $result.Games)
    Write-Log "      $($result.Log)"
}
Write-Log ('  total wall: {0}' -f (Format-Duration $suiteStarted.Elapsed))

$failed = @($results | Where-Object { $_.ExitCode -ne 0 })
if ($failed.Count -gt 0) {
    Write-Log ''
    Write-Log "  {0} match(es) exited non-zero -- read the logs above before trusting any PGN." -f $failed.Count
    exit 1
}
exit 0
