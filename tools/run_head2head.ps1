#requires -Version 5.1
<#
    Two v6 head-to-head matches, run SERIALLY. One GPU, one match at a time.

        powershell -ExecutionPolicy Bypass -File tools\run_head2head.ps1
        powershell -ExecutionPolicy Bypass -File tools\run_head2head.ps1 -SkipA
        powershell -ExecutionPolicy Bypass -File tools\run_head2head.ps1 -PreflightOnly

    SUITE A  10M-corpus vs 20M-corpus, same v5 architecture, 12k nodes each,
             100 games, concurrency 2.
    SUITE B  20M-corpus at 50k nodes vs Stockfish (UCI_Elo 2900) at 100k nodes,
             50 games, concurrency 2.

    WHY SERIAL. GPU share is 80-96% across the C10b grid (87% at the shipping
    point), so two matches at once corrupt both. This script never starts B
    until A's process has exited.

    WHAT THIS DOES THAT A BARE .bat DOES NOT
    ========================================
    1. PREFLIGHT. Every binary, checkpoint and opening book is checked BEFORE
       the first game, because the alternative is discovering a typo'd path
       four hours in. `-PreflightOnly` runs just this.

    2. CREATES THE OUTPUT DIRECTORIES. cutechess-cli does not create parents for
       `-pgnout` or `stderr=`; it fails or silently drops the stream instead.

    3. SUPERSEDES A PREVIOUS RUN. `-pgnout` APPENDS and the per-arm `stderr=`
       files accumulate, so re-running in place would leave a 400-game PGN whose
       halves came from different configurations, and would let a previous run's
       `[error] command=go` lines be read as this run's fallback moves. Stale
       artifacts are MOVED, never deleted -- they are the evidence for whatever
       the re-run is fixing. Same discipline as
       tools/capacity_suite.py:supersede_previous.

    4. RECORDS THE COMMAND. Each suite writes its own `<event>.command.txt`
       exactly as capacity_suite.py does, so the artifact is self-describing a
       month from now.

    B RUNS EVEN IF A FAILS, and the summary says so. A four-hour overnight run
    that abandons its second half because the first half exited 1 is worse than
    one that reports two independent verdicts.
#>
[CmdletBinding()]
param(
    [switch]$SkipA,
    [switch]$SkipB,
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
$Model10M  = Join-Path $Repo 'models\guofish5_10M\v5_10.9M_best.pt'
$Model20M  = Join-Path $Repo 'models\guofish5_20M\v5_10.9M_best.pt'
$GamesRoot = Join-Path $Repo 'benchmarking\engine\games\v6\head2head'

# C10b-3g's selection. Passed explicitly on every arm even though it IS the
# EngineConfig default, so an arm that silently inherited a changed default
# would be unreadable later.
$ShipWorkers   = 1
$ShipOutstand  = 24
$ShipMaxBatch  = 128
$ShipAffinity  = 'none'

# Both nets are v5-architecture, so both take the v5 Q-denominated constants.
# (A v4-involving match would need the Q_RATIO_V4_OVER_V5-scaled pair instead;
# see capacity_suite.py.)
$CPuctInit = 1.43
$FpuTree   = 0.3

# ARENA. capacity_suite.py's ARENA_NODES_PER_SIM = 75, chosen because the
# default 37.5/sim exhausted the arena mid-search and 647 moves across 133 of
# 150 games came back as first-legal-move fallbacks -- which inverted the sign
# of an ELO figure. 75/sim is the headroom the 16k arm demonstrably ran on.
$ArenaPerSim = 75

function Get-Arena { param([int]$Nodes) return [int]($Nodes * $ArenaPerSim) }

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
        'cutechess-cli'      = $Cutechess
        'openings book'      = $Openings
        'stockfish'          = $Stockfish
        '10M-corpus model'   = $Model10M
        '20M-corpus model'   = $Model20M
        'uci wrapper'        = (Join-Path $Repo 'playing\uci_wrapper_v6.py')
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
        One GuoFish engine block. THE SHIPPING CONFIG IS THE DATACLASS DEFAULT;
        it is passed explicitly so the artifact is self-describing.

        `--no-book --no-syzygy` on every arm: both default ON since C11b and
        both BYPASS MCTS, which would nullify `-openings` and contaminate the
        delivered-sims telemetry with zero-simulation moves.
    #>
    param(
        [string]$Name,
        [string]$Model,
        [int]$Nodes,
        [int]$Arena,
        [string]$StderrPath
    )
    return @(
        '-engine', "name=$Name", 'cmd=python',
        'arg=-u', 'arg=playing/uci_wrapper_v6.py',
        'arg=--model', "arg=$Model",
        'arg=--threads', "arg=$ShipWorkers",
        'arg=--max-outstanding', "arg=$ShipOutstand",
        'arg=--max-batch', "arg=$ShipMaxBatch",
        'arg=--affinity', "arg=$ShipAffinity",
        'arg=--c-puct-init', "arg=$CPuctInit",
        'arg=--fpu-tree', "arg=$FpuTree",
        'arg=--no-book', 'arg=--no-syzygy',
        'arg=--arena-capacity', "arg=$Arena",
        "dir=$Repo", 'proto=uci',
        'tc=inf', "nodes=$Nodes", 'timemargin=300000',
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

# ---------------------------------------------------------------------------
# Suite A -- 10M-corpus vs 20M-corpus, 12k nodes each
# ---------------------------------------------------------------------------

function Invoke-SuiteA {
    $event     = '10M_vs_20M_12k'
    $directory = Join-Path $GamesRoot '10M_vs_20M'
    $nodes     = 12000
    $arena     = Get-Arena $nodes

    Write-Log ''
    Write-Log ('=' * 74)
    Write-Log 'SUITE A -- 10M-corpus vs 20M-corpus, v5 architecture, 12k nodes, 100 games'
    Write-Log ('=' * 74)
    Write-Log "    arena $arena nodes both arms ($ArenaPerSim/sim)"

    New-Item -ItemType Directory -Path $directory -Force | Out-Null

    $arguments = @()
    $arguments += New-GuoFishArm -Name 'v5-10M' -Model $Model10M -Nodes $nodes -Arena $arena `
                                 -StderrPath (Join-Path $directory 'v5-10M.stderr.log')
    $arguments += New-GuoFishArm -Name 'v5-20M' -Model $Model20M -Nodes $nodes -Arena $arena `
                                 -StderrPath (Join-Path $directory 'v5-20M.stderr.log')
    $arguments += @('-openings', "file=$Openings", 'format=pgn', 'order=sequential', 'plies=16')
    # House adjudication. Valid here because both checkpoints carry the same
    # value_scale (290.6806), so `score cp = value_scale * atanh(q)` puts both
    # arms on ONE scale and -resign fires symmetrically.
    $arguments += @('-resign', 'movecount=3', 'score=600')
    $arguments += @('-draw', 'movenumber=40', 'movecount=8', 'score=10')
    $arguments += @('-recover', '-concurrency', '2', '-rounds', '50', '-games', '2', '-repeat',
                    '-event', $event,
                    '-pgnout', (Join-Path $directory "$event.pgn"))

    return Invoke-Match -Event $event -Directory $directory -Arguments $arguments
}

# ---------------------------------------------------------------------------
# Suite B -- 20M-corpus at 50k nodes vs Stockfish 2900
# ---------------------------------------------------------------------------

function Invoke-SuiteB {
    $event     = '20M50K_vs_SF2900'
    $directory = Join-Path $GamesRoot '20M50K_vs_SF2900'
    $nodes     = 50000
    $arena     = Get-Arena $nodes

    Write-Log ''
    Write-Log ('=' * 74)
    Write-Log 'SUITE B -- 20M-corpus @ 50k nodes vs Stockfish UCI_Elo 2900 @ 100k nodes, 50 games'
    Write-Log ('=' * 74)
    Write-Log "    arena $arena nodes ($ArenaPerSim/sim; 50k is above every budget"
    Write-Log '      capacity_suite measured, so the margin is the convention, not a fit)'

    New-Item -ItemType Directory -Path $directory -Force | Out-Null

    $arguments = @()
    # NAMED v5-20M, and pointed at the 20M-corpus checkpoint. Both halves matter:
    # the original draft carried `name=v5-10M` with the 20M model, which would
    # have labelled every PGN and every score line with the wrong net.
    $arguments += New-GuoFishArm -Name 'v5-20M' -Model $Model20M -Nodes $nodes -Arena $arena `
                                 -StderrPath (Join-Path $directory 'v5-20M.stderr.log')
    # Stockfish. `stderr=` is set here too -- without it there is no record of
    # what the opponent did, and a UCI option that failed to apply is silent.
    $arguments += @(
        '-engine', 'name=Stockfish', "cmd=$Stockfish",
        "dir=$Repo", 'proto=uci',
        'option.UCI_LimitStrength=true', 'option.UCI_Elo=2900', 'option.Threads=1',
        'tc=inf', 'nodes=100000', 'timemargin=300000',
        "stderr=$(Join-Path $directory 'stockfish.stderr.log')"
    )
    $arguments += @('-openings', "file=$Openings", 'format=pgn', 'order=sequential', 'plies=16')
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
Write-Log 'v6 head-to-head -- two matches, serial. One GPU, one match at a time.'
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

if (-not $SkipA) { $results += Invoke-SuiteA }
else             { Write-Log 'SUITE A skipped (-SkipA)' }

# SERIAL. Invoke-Match does not return until the child has exited, so B cannot
# start while A still owns the GPU. B runs even if A failed -- two independent
# verdicts beat one abandoned night.
if (-not $SkipB) { $results += Invoke-SuiteB }
else             { Write-Log 'SUITE B skipped (-SkipB)' }

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
