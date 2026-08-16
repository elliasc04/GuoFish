@echo off
REM ============================================================================
REM GuoFish v6 UCI launcher for lichess-bot (and any UCI GUI).
REM
REM This is the C++ engine (guofish_core) driven through playing/v6/playv6.py.
REM It is NOT run_guofish.bat, which is kept alongside it and launches the old
REM pure-Python guofish2-era wrapper (playing/uci_wrapper.py on core.mctsv3,
REM guofish4 checkpoint). The two are separate engines with separate configs:
REM
REM   run_guofish.bat     + lichess-bot/config_legacy.yml   legacy, mctsv3
REM   run_guofishv6.bat   + lichess-bot/config.yml          v6, C++ core
REM
REM lichess-bot launches this .bat directly (python-chess executed it without an
REM interpreter in testing). Everything is absolute so the launch does not
REM depend on the working directory lichess-bot sets.
REM
REM WHAT CHANGED FROM THE LEGACY LAUNCHER, and why the old flags are not here
REM ============================================================================
REM   --workers 32   GONE. v6 has no such flag. Parallelism is --threads (W) and
REM                  --max-outstanding (W*K); the shipping defaults are W=1,
REM                  K=24 from BENCH.md C10b-3g. 32 was a v3-era number.
REM   --sims         MEANS WHAT IT MEANT IN v5 AGAIN. It sets the per-move
REM                  budget and ignores the clock, exactly as
REM                  uci_wrapper_v5.py's --sims did. From C11 to C12b it was
REM                  quietly bound to DefaultSims instead -- the lowest-priority
REM                  branch, reached only when a `go` carried neither a clock
REM                  nor nodes -- so the flag that used to override the GUI
REM                  became the one the GUI overrides. Collapsed back in C12c;
REM                  --fixed-sims survives as an alias for the same option.
REM                  NOT SET HERE: a rated game should let the GUI decide.
REM   --sim-cap      Ceiling on `go nodes N`, and the quantity the ARENA is
REM                  sized from. Sizing deliberately does NOT follow --sims:
REM                  see the arena_nodes docstring in playv6.py.
REM   time vs nodes  v6 does REAL TIME MANAGEMENT (UCIEngine._allot reads
REM                  wtime/btime/winc/movestogo), which the legacy wrapper did
REM                  not. config.yml still sets go_commands.nodes, now for a
REM                  different reason: it is what keeps the per-move budget
REM                  FIXED. Drop it and the budget becomes clock-driven and
REM                  variable. With it set both bounds are live -- the node
REM                  target decides the move, the deadline is the backstop.
REM                  That is where sims/move is configured for Lichess.
REM   --ponder       SET HERE, AND ONLY HERE. The ENGINE's default is OFF and
REM                  stays off: this launcher is for competitive play, where
REM                  thinking on the opponent's clock is free strength and
REM                  Lichess neither forbids it nor checks for it. Every other
REM                  caller of uci_wrapper_v6.py -- the cutechess benches,
REM                  tools/smoke_c11.py, tools/uci_conform_c11.py -- invokes the
REM                  wrapper directly, does not pass this flag, and therefore
REM                  measures the ponder-off engine the throughput anchors were
REM                  recorded against. Do not move this default into
REM                  EngineConfig: a bench that silently ponders reports a
REM                  sims/s measured against a clock it never spent.
REM
REM                  Unlike the legacy launcher's flag, this is the real UCI
REM                  handshake (go ponder / ponderhit, C11c), not a background
REM                  thread of the engine's own.
REM
REM                  WHAT THE FLAG ACTUALLY DOES: it makes the engine APPEND a
REM                  ponder move to its bestmove ("bestmove e2e4 ponder e7e5").
REM                  Without that suggestion a GUI has nothing to ponder on and
REM                  never sends `go ponder` at all. Handling of an incoming
REM                  `go ponder` does not depend on it.
REM
REM                  UNDER lichess-bot THIS FLAG IS OVERRIDDEN, and harmlessly:
REM                  `Ponder` is a UCI option this engine advertises, and
REM                  python-chess sets it explicitly from config.yml's `ponder:`
REM                  on every play call. config.yml says true, so the two agree.
REM                  The flag is what makes any OTHER GUI -- cutechess, Arena, a
REM                  bare pipe -- ponder by default, and it is the state the
REM                  engine reports on its own [config] ponder line at startup.
REM                  Set config.yml's `ponder: false` to turn it off for Lichess
REM                  alone; drop this flag to turn it off everywhere else.
REM   --switch-interval
REM                  NOT SET. The default 0.0005 is scope 2.1's GIL mitigation
REM                  and is what C10/C10b measured throughput against. Leave it.
REM
REM UCI OPTIONS vs FLAGS: both work, and the split matters
REM ============================================================================
REM Every EngineConfig field is a UCI option (uci_wrapper_v6.OPTIONS is checked
REM against the dataclass at startup and the process refuses to start if one is
REM missing), so most of what is settable here is also settable from config.yml's
REM uci_options. lichess-bot sends every setoption immediately after launch and
REM before the first isready, so even the read-once options are reachable there.
REM
REM Set it HERE when it must hold no matter what a GUI does, and when you want it
REM on the process's own [config] startup line. Set it in config.yml when it is a
REM per-deployment number you expect to retune.
REM ============================================================================

set "PYTHON=C:\Users\Ethan Guo\AppData\Local\Programs\Python\Python313\python.exe"
set "REPO=C:\Users\Ethan Guo\Github\GuoFish"

REM The v5 90M student. This is ALREADY playv6.DEFAULT_MODEL; it is passed
REM explicitly so the checkpoint a game was played with is on the launch line
REM rather than resolved from a constant that may move.
set "MODEL=%REPO%\models\guofish5_90M\v5_10.9M_best.pt"

REM Pin to GPU 0 and keep stdio unbuffered so UCI lines flush promptly.
REM v6 requires CUDA: Engine.ensure_ready raises if torch.cuda.is_available()
REM is false. There is no measured CPU path.
set "CUDA_VISIBLE_DEVICES=0"
set "PYTHONUNBUFFERED=1"

REM ---------------------------------------------------------------------------
REM TUNABLES, deliberately left at the v6 defaults for now. Uncomment and append
REM to the launch line below once the values are settled.
REM
REM   --sim-cap N        Ceiling on `go nodes N`, and THE ARENA DRIVER. The
REM                      arena is 60 x (sim_cap + ponder ceiling) nodes, both
REM                      halves of the ping-pong pair reserved, and it COMMITS
REM                      rather than reserves, so this is real RSS:
REM
REM                          sim-cap 60000 (default)  7.20M nodes   535.6 MB
REM                          sim-cap 50000            6.00M nodes   446.3 MB
REM                          sim-cap 25000            3.00M nodes   223.2 MB
REM                          sim-cap 10000            1.20M nodes    89.3 MB
REM
REM                      SIZED FROM THE CAP, NOT FROM --sims. A fixed budget is
REM                      one move's allowance; the arena's real consumer is the
REM                      accumulating tree across moves, and sizing it off one
REM                      move is the mistake DECISIONS.md's C12b entry records
REM                      (a reuse arm needed 1,436,625 nodes and exhausted a
REM                      1.2M arena). Over-provisioning costs address space;
REM                      under-provisioning costs a game. The engine prints the
REM                      resolved figure on its [config] memory line at
REM                      startup -- check it there.
REM   --sims N           THE PER-MOVE BUDGET: force every move to N sims and
REM                      ignore the clock (_plan returns deadline=None).
REM                      Outranks go_commands.nodes and the clock alike. v5's
REM                      contract, restored in C12c; --fixed-sims is an alias
REM                      for the same option.
REM                      For BENCHMARKING. For rated play prefer
REM                      go_commands.nodes in config.yml, which pins the same
REM                      budget but keeps the deadline as a backstop, so a
REM                      pathological position cannot flag the game.
REM                      NOTE it means two things depending on the path: _plan
REM                      treats it as an ABSOLUTE root-visit target, while
REM                      _plan_after_ponderhit treats it as N FRESH sims on the
REM                      pondered tree. Both are deliberate; they differ, and
REM                      it only shows when ponder is also on.
REM   --threads N        W. Default 1.
REM   --max-outstanding N  W*K. Default 24. K is derived as this over threads.
REM   --move-overhead-ms N  Subtracted from every allotted move time. Default
REM                      100. This is the ENGINE's own margin and is separate
REM                      from lichess-bot's `move_overhead: 2000`.
REM   --no-compile       Fall back to the unfused eager forward, i.e. tag
REM                      GUOFISH_NUMERICS_BASELINE's numerics (C12b). Slower on
REM                      fresh roots; bit-exact against the frozen baseline.
REM                      Use it to bisect a suspected numerics regression.
REM   --no-book / --no-syzygy
REM                      Both features are ON by default and both BYPASS MCTS.
REM                      Leave them on for play; turn them off for any
REM                      measurement of the search itself. The engine tallies
REM                      search/book/tablebase per game on stderr either way.
REM   --ponder-max-sims N / --ponder-decay F
REM                      Only relevant once config.yml sets `ponder: true`. The
REM                      defaults keep decay x ponder_max_sims at most
REM                      sims_per_move, which is the coupling that stops a
REM                      pondered tree out-weighing the fresh search it feeds.
REM ---------------------------------------------------------------------------

"%PYTHON%" "%REPO%\playing\uci_wrapper_v6.py" --model "%MODEL%" --ponder
