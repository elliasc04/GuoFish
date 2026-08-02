@echo off
REM ============================================================================
REM GuoFish UCI launcher for lichess-bot (and any UCI GUI).
REM
REM lichess-bot runs this through `cmd /c` (see the `interpreter` field in
REM lichess-bot/config.yml) because python-chess's popen_uci uses CreateProcess,
REM which cannot execute a .bat directly. This wrapper just execs the project's
REM system Python on uci_wrapper.py with production-safe arguments.
REM
REM Everything is absolute so the launch is independent of the working dir
REM lichess-bot sets. uci_wrapper.py itself resolves the model/book/syzygy from
REM its own __file__ location, but we pass --model explicitly because the
REM auto-detector only matches guofish2_*/guofish_* names, not the guofish4
REM champion checkpoint we actually want to play.
REM
REM Search budget: fixed 5000 sims/move (no time scaling in the wrapper). At the
REM measured ~800 sims/s worst case that is ~6.3s/move, ~3.8s at the ~1300/s peak
REM -- safe for blitz 5+3 and slower once the opening book and Syzygy bypass take
REM the (instant) opening and <=5-piece endgame moves off MCTS. Keep
REM go_commands.nodes in config.yml in sync with --sim-cap below.
REM ============================================================================

set "PYTHON=C:\Users\Ethan Guo\AppData\Local\Programs\Python\Python313\python.exe"
set "REPO=C:\Users\Ethan Guo\Github\GuoFish"
set "MODEL=%REPO%\models\guofish4\guofish4_25.6M_policy_final.pt"

REM Pin to GPU 0 and keep stdio unbuffered so UCI lines flush promptly.
set "CUDA_VISIBLE_DEVICES=0"
set "PYTHONUNBUFFERED=1"

REM --workers 32: documented single-game default (concurrency is 1 on Lichess).
REM --sim-cap 10000: hard ceiling on sims/move; also the value go_commands.nodes
REM   forces. Even if lichess-bot ever sends a larger `nodes`, it is clamped here.
REM --ponder: engine's own background pondering on the opponent's clock. NOTE:
REM   this is NOT the UCI ponder handshake, so config.yml keeps `ponder: false`
REM   (otherwise python-chess would send `go ponder` and mis-handle the reply).
REM Opening book (assets\gm2001.bin) and Syzygy (assets\syzygy\) are ON by
REM   default in uci_wrapper.py, so no flags are needed to enable them.
"%PYTHON%" "%REPO%\playing\uci_wrapper.py" --model "%MODEL%" --workers 32 --sim-cap 50000 --ponder
