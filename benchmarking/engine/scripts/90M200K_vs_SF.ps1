#requires -Version 5.1
<#
    One v6 match: 90M-corpus at 200k nodes vs Stockfish (UCI_Elo 3000) at 100k
    nodes, 100 games, concurrency 4 (P6's measured best for GvSF: 23.25 s/game,
    1.58x over c=1, at 93.2% GPU utilisation).

        powershell -ExecutionPolicy Bypass -File benchmarking\engine\scripts\90M200K_vs_SF.ps1
        powershell -ExecutionPolicy Bypass -File benchmarking\engine\scripts\90M200K_vs_SF.ps1 -PreflightOnly

    THREE DELIBERATE DEPARTURES from the scripts this was copied from:

      * `tscale=10` on the GuoFish arm (D-5), forced by concurrency 4. Without
        it, four engines starting at once miss cutechess's ping timeout and
        the match dies with `no response to ping` at returncode 0.
      * `--sim-cap 200000` on the GuoFish arm, forced by the 200k budget.
        Without it, `sim_cap`'s 60,000 default silently clamps every
        `go nodes 200000` down to 60,000 -- see New-GuoFishArm.
      * `--arena-capacity` is NO LONGER PINNED; playv6's computed default sizes
        it (and now scales correctly with `--sim-cap`). See the ARENA block
        below.

    Structure, adjudication and house discipline (preflight, superseding stale
    artifacts, command recording) are otherwise copied from
    run_head2head.ps1's Suite B / run_90M_vs_stockfish.ps1 (both in this same
    scripts/ directory), which established the 50k/100k node split against
    Stockfish for a v5-architecture
    10.9M-param net. The 90M checkpoint is the same architecture and carries the
    same locked value_scale (290.6806 -- see data/multiPV/corpus_90m_report.md
    D1), so the adjudication convention applies unchanged.

    GuoFish's node budget is deliberately bumped to 200k for this bench (up
    from the 50k the convention above used). Stockfish's 100k is left AS-IS,
    not scaled with it: probe/RESULTS.md never measured a minimum node count
    for Stockfish under a UCI_Elo cap (P7's node-only-weakening ladder
    deliberately avoids UCI_Elo entirely), so there is no measured basis for a
    ratio here. 100k is far above what even full-strength Stockfish needs to
    play well (P3/P7: 100k+ nodes were dropped from the ladder because
    full-strength SF was already overwhelming at that budget), so it is not
    expected to be the bottleneck under an Elo=3000 cap either.
#>
[CmdletBinding()]
param(
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

# ARENA. NOT PINNED. C11c made `--arena-capacity` optional-valued, and leaving
# it unset selects playv6's computed default:
#
#     arena_nodes = 60 x (sim_cap + ponder ceiling)
#
# which floors on `sim_cap` rather than on the fixed budget -- deliberately, per
# playv6.py's `arena_nodes` docstring, because an ACCUMULATING tree outgrows an
# arena sized off one move's allowance. DECISIONS.md's C12b entry records that
# exact failure: a reuse arm needed 1,436,625 nodes, exhausted a 1.2M arena
# sized off a single move, and delivered 6,666 of 20,000 simulations.
#
# The older scripts in this directory pin it via capacity_suite.py's
# ARENA_NODES_PER_SIM = 75. That convention predates the computed default and is
# not carried forward here -- the engine sizes itself better than a constant
# multiplied by this script's node count can, and `SearchOutcome.arena_exhausted`
# remains the backstop for a position that beats the estimate anyway.

$GuoFishNodes   = 200000
$StockfishNodes = 100000

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
        request -- unlike this directory's run_head2head.ps1 Suite A/B, which force them off
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
        # SimCap MUST match the node budget. `_plan()`'s `go nodes` branch
        # (uci_wrapper_v6.py) computes `budget = min(nodes, cfg.sim_cap)` --
        # SimCap's 60,000 default is well under this script's 200k, so every
        # move was silently clamped to 60,000 (with a
        # `[go] nodes 200000 exceeds SimCap 60000; clamped` line to stderr on
        # each one) until this was set. It also fixes the arena's computed
        # default, which sizes off `sim_cap`, not off `nodes=`.
        'arg=--sim-cap', "arg=$GuoFishNodes",
        "dir=$Repo", 'proto=uci',
        # tscale=10 -- D-5. The first `isready` takes 14.52 s (checkpoint load
        # ~4 s, then Inductor compile + CUDA-graph capture ~8.05 s), which sits
        # right at cutechess's ping timeout and produces
        # `Warning: Engine v5-90M(N): no response to ping` -- with an empty PGN
        # and RETURNCODE 0 (D-6), so the failure is silent to anything checking
        # the exit code. probe/RESULTS.md D-5 states outright that a campaign
        # spawning engines at concurrency 4 hits this nondeterministically
        # unless it sets tscale or pre-warms. Safe here because `tc=inf nodes=N`
        # has no clock for a time scale to distort.
        'tscale=10',
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

    $event     = "90M200K_vs_SF$Elo"
    $directory = Join-Path $GamesRoot $event

    Write-Log ''
    Write-Log ('=' * 74)
    Write-Log "SUITE $Elo -- 90M-corpus @ ${GuoFishNodes} nodes vs Stockfish UCI_Elo $Elo @ ${StockfishNodes} nodes, 100 games"
    Write-Log ('=' * 74)
    Write-Log "    arena: not pinned -- playv6 computes it (60 x (sim_cap + ponder ceiling))"
    Write-Log "    tscale=10 on the GuoFish arm (D-5: first isready ~14.5 s vs cutechess's ping timeout)"

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
    $arguments += @('-recover', '-concurrency', '4', '-rounds', '50', '-games', '2', '-repeat',
                    '-event', $event,
                    '-pgnout', (Join-Path $directory "$event.pgn"))

    return Invoke-Match -Event $event -Directory $directory -Arguments $arguments
}

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

$suiteStarted = [System.Diagnostics.Stopwatch]::StartNew()

Write-Log ('=' * 74)
Write-Log 'v6 90M-corpus @ 200k nodes vs Stockfish UCI_Elo 3000 @ 100k nodes, 100 games'
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
$results += Invoke-StockfishSuite -Elo 3000

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
