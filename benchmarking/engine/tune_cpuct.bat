@echo off
REM ============================================================================
REM C_PUCT tuning sweep: guofish4 (Phase 3) vs guofish3 (Phase 2 ref) vs Stockfish.
REM
REM Revision history:
REM   v1: 4 coarse candidates {0.75, 1.00, 1.25, 1.50} against the Phase 2
REM       checkpoint. Showed monotonic improvement with rising C_PUCT but
REM       didn't bracket the peak.
REM   v2 (this file): 9 finer-grained candidates {0.75 .. 3.00} against the
REM       Phase 3 (guofish4) checkpoint. Phase 2 final stays as the fixed
REM       reference opponent so head-to-head numbers remain comparable with v1.
REM       The 3.00 endpoint is intentionally past the expected optimum to
REM       confirm we've found the peak rather than capping the sweep at a
REM       still-improving value.
REM
REM Total games: 9 * (80 vs Phase 2 + 50 vs SF) + 50 (Phase 2 ref vs SF leg)
REM            = 1220 games. At concurrency=5 and ~35-45s/game, est. 2-2.5h.
REM
REM Stockfish anchor stays at 2500 ELO (UCI_LimitStrength), tc=10+0.1,
REM Threads=1, Hash=128 -- matches v1 so absolute ELO is comparable.
REM
REM Run manually so GPU memory can be monitored. No CLI args needed.
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

REM --- Checkpoints ----------------------------------------------------------
REM Phase 3 candidate: latest models\guofish4\policy_finetune_phase3_*.pt by
REM modification time. Override by setting PHASE3_CKPT before invoking.
set "PHASE3_CKPT=models\guofish4\guofish4_25.6M_policy_final.pt"
REM Phase 2 reference opponent: held fixed so head-to-head ELO deltas are
REM directly comparable to v1's measurements.
set "PHASE2_REF_CKPT=models\guofish3\guofish3_25.6M_final_0.0691.pt"

REM --- Match settings -------------------------------------------------------
set "SIMS=800"
set "GAMES_VS_REF=50"
set "GAMES_VS_SF=50"
set "BASELINE_CPUCT=1.5"
REM CONCURRENCY=5: validated by the prior sweep. With 5 parallel games and
REM up to 2 guofish processes per game (Leg 1), peak resident count is 10
REM guofish workers; at 25.6M params (~307MB) that's still inside VRAM budget.
set "CONCURRENCY=7"
set "WORKERS=8"
set "SF_TC=10+0.1"
set "SF_THREADS=1"
set "SF_HASH=128"
set "SF_ANCHOR_ELO=2500"
set "PGN_DIR=benchmarking\engine\games\cpuct_phase3_tuning"
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
if not defined PHASE3_CKPT (
    echo ERROR: no Phase 3 checkpoint found matching models\guofish4\policy_finetune_phase3_*.pt
    echo Set PHASE3_CKPT manually if your checkpoint lives elsewhere.
    exit /b 1
)
if not exist "%PROJECT_ROOT%\%PHASE3_CKPT%" (
    echo ERROR: Phase 3 checkpoint not found: %PHASE3_CKPT%
    exit /b 1
)
if not exist "%PROJECT_ROOT%\%PHASE2_REF_CKPT%" (
    echo ERROR: Phase 2 reference checkpoint not found: %PHASE2_REF_CKPT%
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
echo Phase 3 C_PUCT sweep
echo ------------------------------------------------------------
echo Phase 3 candidate: %PHASE3_CKPT%
echo Phase 2 reference: %PHASE2_REF_CKPT%
echo Candidates:        0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 3.00 (9)
echo Games per cand:    %GAMES_VS_REF% vs Phase 2 + %GAMES_VS_SF% vs SF
echo Plus calibration:  Phase 2 ref vs SF (%GAMES_VS_SF% games, once)
echo Total games:       1220   Concurrency: %CONCURRENCY%   Est: ~2-2.5h
echo ============================================================
echo.

REM --- Calibration leg: Phase 2 ref vs Stockfish ----------------------------
REM Runs once. Anchors today's hardware/SF version to the v1 sweep so the
REM Phase-2-baseline absolute ELO is comparable across runs and we can detect
REM hardware drift independently of the Phase 3 candidate results.
set "PGN_BASELINE_VS_SF=%PGN_DIR%\phase2_baseline_vs_sf.pgn"
call :check_complete "%PROJECT_ROOT%\%PGN_BASELINE_VS_SF%" %GAMES_VS_SF%
if "!_SKIP!" NEQ "1" (
    echo.
    echo ============================================================
    echo Calibration: Phase 2 ref C_PUCT=%BASELINE_CPUCT%  vs  Stockfish ^(tc=%SF_TC%^)
    echo Games:   %GAMES_VS_SF%   Sims/move: %SIMS%   PGN: %PGN_BASELINE_VS_SF%
    echo ============================================================

    "%CUTECHESS%" ^
      -engine name="Phase2_ref" cmd=python arg=%UCI_SCRIPT% arg=--c-puct arg=%BASELINE_CPUCT% arg=--checkpoint arg=%PHASE2_REF_CKPT% arg=--sims arg=%SIMS% arg=--workers arg=%WORKERS% proto=uci dir="%PROJECT_ROOT%" tc=inf stderr="%PROJECT_ROOT%\%LOG_DIR%\phase2_baseline_vs_sf_p2.stderr.log" ^
      -engine name="Stockfish" cmd="%STOCKFISH%" proto=uci tc=%SF_TC% option.Threads=%SF_THREADS% option.Hash=%SF_HASH% option.UCI_LimitStrength=true option.UCI_Elo=%SF_ANCHOR_ELO% stderr="%PROJECT_ROOT%\%LOG_DIR%\phase2_baseline_vs_sf_sf.stderr.log" ^
      -games %GAMES_VS_SF% ^
      -repeat ^
      -recover ^
      -concurrency %CONCURRENCY% ^
      -openings file="%PROJECT_ROOT%\%OPENING_BOOK%" format=pgn order=random plies=8 ^
      -pgnout "%PROJECT_ROOT%\%PGN_BASELINE_VS_SF%" ^
      -resign movecount=5 score=600 ^
      -draw movenumber=40 movecount=10 score=10 ^
      -event "cpuct_phase3_sweep_phase2_ref_vs_sf"

    if errorlevel 1 (
        echo WARN: cutechess returned non-zero for Phase 2 ref vs sf calibration. Continuing.
    )
)

REM --- Sweep ----------------------------------------------------------------
for %%C in (0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 3.00) do (
    set "CANDIDATE=%%C"
    set "TAG=cpuct_%%C"
    set "PGN_VS_REF=%PGN_DIR%\!TAG!_phase3_vs_phase2.pgn"
    set "PGN_VS_SF=%PGN_DIR%\!TAG!_phase3_vs_sf.pgn"

    REM --- Leg 1: Phase 3 candidate vs Phase 2 reference -------------------
    call :check_complete "%PROJECT_ROOT%\!PGN_VS_REF!" %GAMES_VS_REF%
    if "!_SKIP!" NEQ "1" (
        echo.
        echo ============================================================
        echo Matchup: Phase3 C_PUCT=!CANDIDATE!  vs  Phase2 ref C_PUCT=%BASELINE_CPUCT%
        echo Games:   %GAMES_VS_REF%   Sims/move: %SIMS%   PGN: !PGN_VS_REF!
        echo ============================================================

        "%CUTECHESS%" ^
          -engine name="Phase3_cpuct_!CANDIDATE!" cmd=python arg=%UCI_SCRIPT% arg=--c-puct arg=!CANDIDATE! arg=--checkpoint arg=%PHASE3_CKPT% arg=--sims arg=%SIMS% arg=--workers arg=%WORKERS% proto=uci dir="%PROJECT_ROOT%" stderr="%PROJECT_ROOT%\%LOG_DIR%\!TAG!_phase3.stderr.log" ^
          -engine name="Phase2_ref" cmd=python arg=%UCI_SCRIPT% arg=--c-puct arg=%BASELINE_CPUCT% arg=--checkpoint arg=%PHASE2_REF_CKPT% arg=--sims arg=%SIMS% arg=--workers arg=%WORKERS% proto=uci dir="%PROJECT_ROOT%" stderr="%PROJECT_ROOT%\%LOG_DIR%\!TAG!_phase2.stderr.log" ^
          -each tc=inf ^
          -games %GAMES_VS_REF% ^
          -repeat ^
          -recover ^
          -concurrency %CONCURRENCY% ^
          -openings file="%PROJECT_ROOT%\%OPENING_BOOK%" format=pgn order=random plies=8 ^
          -pgnout "%PROJECT_ROOT%\!PGN_VS_REF!" ^
          -resign movecount=5 score=600 ^
          -draw movenumber=40 movecount=10 score=10 ^
          -event "cpuct_phase3_sweep_!CANDIDATE!_vs_phase2"

        if errorlevel 1 (
            echo WARN: cutechess returned non-zero for C_PUCT=!CANDIDATE! vs phase2. Continuing.
        )
    )

    REM --- Leg 2: Phase 3 candidate vs Stockfish (calibration anchor) -----
    call :check_complete "%PROJECT_ROOT%\!PGN_VS_SF!" %GAMES_VS_SF%
    if "!_SKIP!" NEQ "1" (
        echo.
        echo ============================================================
        echo Matchup: Phase3 C_PUCT=!CANDIDATE!  vs  Stockfish ^(tc=%SF_TC%^)
        echo Games:   %GAMES_VS_SF%   Sims/move: %SIMS%   PGN: !PGN_VS_SF!
        echo ============================================================

        "%CUTECHESS%" ^
          -engine name="Phase3_cpuct_!CANDIDATE!" cmd=python arg=%UCI_SCRIPT% arg=--c-puct arg=!CANDIDATE! arg=--checkpoint arg=%PHASE3_CKPT% arg=--sims arg=%SIMS% arg=--workers arg=%WORKERS% proto=uci dir="%PROJECT_ROOT%" tc=inf stderr="%PROJECT_ROOT%\%LOG_DIR%\!TAG!_phase3_sf.stderr.log" ^
          -engine name="Stockfish" cmd="%STOCKFISH%" proto=uci tc=%SF_TC% option.Threads=%SF_THREADS% option.Hash=%SF_HASH% option.UCI_LimitStrength=true option.UCI_Elo=%SF_ANCHOR_ELO% stderr="%PROJECT_ROOT%\%LOG_DIR%\!TAG!_sf.stderr.log" ^
          -games %GAMES_VS_SF% ^
          -repeat ^
          -recover ^
          -concurrency %CONCURRENCY% ^
          -openings file="%PROJECT_ROOT%\%OPENING_BOOK%" format=pgn order=random plies=8 ^
          -pgnout "%PROJECT_ROOT%\!PGN_VS_SF!" ^
          -resign movecount=5 score=600 ^
          -draw movenumber=40 movecount=10 score=10 ^
          -event "cpuct_phase3_sweep_!CANDIDATE!_vs_sf"

        if errorlevel 1 (
            echo WARN: cutechess returned non-zero for C_PUCT=!CANDIDATE! vs sf. Continuing.
        )
    )
)

echo.
echo ============================================================
echo All matchups complete. Parsing results...
echo ============================================================
python "%PROJECT_ROOT%\%PARSER%" --pgn-dir "%PROJECT_ROOT%\%PGN_DIR%" --sf-anchor %SF_ANCHOR_ELO%

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
