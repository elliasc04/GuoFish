@echo off
REM ============================================================================
REM Phase 3 C_PUCT finalist benchmark: top candidates vs Stockfish, 200 games.
REM
REM Follow-up to tune_cpuct.bat. The sweep narrowed the optimum to the
REM {2.00, 2.25, 2.50} band; this run takes those three candidates and
REM plays each for 200 games vs Stockfish (UCI_Elo=2500). At 200 games per
REM candidate the ordo 95% CI tightens to roughly +/- 20-30 ELO, enough to
REM rank the three with confidence.
REM
REM No Phase-2 head-to-head leg here -- absolute ELO vs the same SF anchor
REM is the only signal needed to pick a finalist.
REM
REM Total games: 3 * 200 = 600. At concurrency=7 and ~25-30s/SF-game,
REM est. wall-clock ~40-60 min.
REM ============================================================================

setlocal EnableDelayedExpansion

REM --- Auto-derive project root from script location --------------------------
pushd "%~dp0..\.."
set "PROJECT_ROOT=%CD%"
popd

REM --- Binaries -------------------------------------------------------------
set "CUTECHESS=%PROJECT_ROOT%\cutechess-1.4.0-win64\cutechess-cli.exe"
set "STOCKFISH=%PROJECT_ROOT%\stockfish-windows-x86-64-avx2.exe"
set "UCI_SCRIPT=playing\uci_wrapper.py"
set "OPENING_BOOK=assets\8moves_v3.pgn"
set "PARSER=benchmarking\engine\parse_cpuct_results.py"

REM --- Checkpoint -----------------------------------------------------------
REM Same Phase 3 checkpoint convention as tune_cpuct.bat. Override by setting
REM PHASE3_CKPT before invoking.
if not defined PHASE3_CKPT (
    set "PHASE3_CKPT=models\guofish4\guofish4_25.6M_policy_final.pt"
)

REM --- Match settings -------------------------------------------------------
set "SIMS=800"
set "GAMES_VS_SF=200"
set "CONCURRENCY=7"
set "WORKERS=8"
set "SF_TC=10+0.1"
set "SF_THREADS=1"
set "SF_HASH=128"
set "SF_ANCHOR_ELO=2500"
set "PGN_DIR=benchmarking\engine\games\cpuct_phase3_finalists"
set "LOG_DIR=benchmarking\engine\logs"

set "CUDA_VISIBLE_DEVICES=0"

REM --- Setup ----------------------------------------------------------------
if not exist "%PROJECT_ROOT%\%PGN_DIR%" mkdir "%PROJECT_ROOT%\%PGN_DIR%"
if not exist "%PROJECT_ROOT%\%LOG_DIR%" mkdir "%PROJECT_ROOT%\%LOG_DIR%"

if not exist "%CUTECHESS%" (
    echo ERROR: cutechess-cli.exe not found at %CUTECHESS%
    exit /b 1
)
if not exist "%STOCKFISH%" (
    echo ERROR: Stockfish binary not found: %STOCKFISH%
    exit /b 1
)
if not exist "%PROJECT_ROOT%\%PHASE3_CKPT%" (
    echo ERROR: Phase 3 checkpoint not found: %PHASE3_CKPT%
    exit /b 1
)
if not exist "%PROJECT_ROOT%\%PARSER%" (
    echo ERROR: parser script missing: %PARSER%
    exit /b 1
)
if not exist "%PROJECT_ROOT%\%OPENING_BOOK%" (
    echo WARN: opening book missing; cutechess will start from initial position only.
)

echo.
echo ============================================================
echo Phase 3 C_PUCT finalist benchmark
echo ------------------------------------------------------------
echo Phase 3 candidate: %PHASE3_CKPT%
echo Candidates:        2.00, 2.25, 2.50 (3)
echo Games per cand:    %GAMES_VS_SF% vs Stockfish ^(anchor %SF_ANCHOR_ELO%^)
echo Total games:       600   Concurrency: %CONCURRENCY%   Est: ~40-60 min
echo ============================================================
echo.

REM --- Sweep ----------------------------------------------------------------
for %%C in (2.00 2.25 2.50) do (
    set "CANDIDATE=%%C"
    set "TAG=cpuct_%%C"
    set "PGN_VS_SF=%PGN_DIR%\!TAG!_phase3_vs_sf.pgn"

    call :check_complete "%PROJECT_ROOT%\!PGN_VS_SF!" %GAMES_VS_SF%
    if "!_SKIP!" NEQ "1" (
        echo.
        echo ============================================================
        echo Matchup: Phase3 C_PUCT=!CANDIDATE!  vs  Stockfish ^(tc=%SF_TC%^)
        echo Games:   %GAMES_VS_SF%   Sims/move: %SIMS%   PGN: !PGN_VS_SF!
        echo ============================================================

        "%CUTECHESS%" ^
          -engine name="Phase3_cpuct_!CANDIDATE!" cmd=python arg=%UCI_SCRIPT% arg=--c-puct arg=!CANDIDATE! arg=--checkpoint arg=%PHASE3_CKPT% arg=--sims arg=%SIMS% arg=--workers arg=%WORKERS% proto=uci dir="%PROJECT_ROOT%" tc=inf stderr="%PROJECT_ROOT%\%LOG_DIR%\finalist_!TAG!_phase3_sf.stderr.log" ^
          -engine name="Stockfish" cmd="%STOCKFISH%" proto=uci tc=%SF_TC% option.Threads=%SF_THREADS% option.Hash=%SF_HASH% option.UCI_LimitStrength=true option.UCI_Elo=%SF_ANCHOR_ELO% stderr="%PROJECT_ROOT%\%LOG_DIR%\finalist_!TAG!_sf.stderr.log" ^
          -games %GAMES_VS_SF% ^
          -repeat ^
          -recover ^
          -concurrency %CONCURRENCY% ^
          -openings file="%PROJECT_ROOT%\%OPENING_BOOK%" format=pgn order=random plies=8 ^
          -pgnout "%PROJECT_ROOT%\!PGN_VS_SF!" ^
          -resign movecount=5 score=600 ^
          -draw movenumber=40 movecount=10 score=10 ^
          -event "cpuct_phase3_finalist_!CANDIDATE!_vs_sf"

        if errorlevel 1 (
            echo WARN: cutechess returned non-zero for C_PUCT=!CANDIDATE! vs sf. Continuing.
        )
    )
)

echo.
echo ============================================================
echo All matchups complete. Parsing results...
echo ============================================================
REM Pass --out so this run's report does not overwrite the sweep's report.
REM Head-to-head and SF-inferred-delta columns will show n/a since there is
REM no Phase 2 leg in this run -- the absolute-ELO column is the headline.
python "%PROJECT_ROOT%\%PARSER%" ^
  --pgn-dir "%PROJECT_ROOT%\%PGN_DIR%" ^
  --sf-anchor %SF_ANCHOR_ELO% ^
  --out "%PROJECT_ROOT%\%LOG_DIR%\cpuct_phase3_finalists_results.md" ^
  --plot "%PROJECT_ROOT%\%LOG_DIR%\cpuct_phase3_finalists_elo_curve.png"

endlocal
exit /b 0

REM ============================================================================
REM Subroutines.
REM ============================================================================

:check_complete
REM Args: %~1 = absolute PGN path, %~2 = target game count.
REM Sets _SKIP=1 if PGN already has >= target games. Otherwise sets _SKIP=0 and
REM deletes any partial PGN so the upcoming cutechess run starts from a clean
REM file. Cutechess has no native resume; mixing partial+resumed runs in one
REM PGN would corrupt color balance and opening rotation, so we sacrifice the
REM partial games for statistical cleanliness.
set "_SKIP=0"
set "_DONE=0"
if exist "%~1" (
    for /f %%n in ('python "%~dp0_count_pgn_games.py" "%~1"') do set "_DONE=%%n"
)
if !_DONE! GEQ %~2 (
    echo [skip] %~nx1: !_DONE!/%~2 games already complete.
    set "_SKIP=1"
    goto :eof
)
if exist "%~1" (
    echo [reset] %~nx1: !_DONE!/%~2 games found, deleting partial and re-running.
    del "%~1"
)
goto :eof
