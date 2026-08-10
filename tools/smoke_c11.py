#!/usr/bin/env python
"""C11 acceptance 2 — the 20-game Cutechess smoke run, and its verdict.

    python tools/smoke_c11.py                       # run 20 games, then verify
    python tools/smoke_c11.py --games 4 --tc 4+0.1  # a quick shakeout
    python tools/smoke_c11.py --verify-only runs/smoke_c11   # re-read a run

WHAT THE SMOKE RUN IS TESTING
=============================
Not strength — C13 does that. This asks whether the v6 wrapper survives a real
GUI driving it under a real clock, and the brief names four things the logs must
prove: zero illegal moves, zero null bestmoves (`bestmove 0000`), zero engine
crashes, zero timeouts. All four are read out of Cutechess's own output rather
than asserted from the engine's point of view, because an engine that has
crashed is in no position to report it.

WHERE EACH VERDICT COMES FROM
=============================
  illegal moves  Cutechess writes `{White makes an illegal move: e2e5}` into the
                 PGN result comment and prints `Illegal move` on stdout. Both are
                 scanned. The PGN is ALSO replayed move by move through
                 python-chess, so a move Cutechess accepted but the rules do not
                 is caught independently of Cutechess agreeing with itself.
  null bestmoves `bestmove 0000` in the engine's own stderr/debug transcript, and
                 the `no legal moves` / `terminated` PGN terminations it produces.
                 A 0000 at a genuinely finished position is legal UCI and is not
                 what this counts — the run is checked for the ones that end a
                 game that was not over.
  crashes        a non-zero exit, `Terminating process`, `connection stalls`, or
                 a Python traceback in the engine's stderr. The traceback check
                 matters most: the wrapper catches exceptions per command and
                 keeps playing, so a crash CAN be invisible in the result column.
  timeouts       `loses on time` / `Terminating` in Cutechess's output and
                 `time forfeit` in the PGN.

Writes to `--out` only (default `benchmarking/engine/games/c11_smoke/`). Nothing
under golden/ is touched (Global Rule 2).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chess  # noqa: E402
import chess.pgn  # noqa: E402

CUTECHESS = REPO_ROOT / "cutechess-1.4.0-win64" / "cutechess-cli.exe"
STOCKFISH = REPO_ROOT / "stockfish-windows-x86-64-avx2.exe"
OPENINGS = REPO_ROOT / "assets" / "8moves_v3.pgn"
WRAPPER = REPO_ROOT / "playing" / "uci_wrapper_v6.py"
DEFAULT_OUT = REPO_ROOT / "benchmarking" / "engine" / "games" / "c11_smoke"

# Patterns Cutechess prints for the four conditions. Kept as a table so the
# verdict names the line it fired on rather than a boolean.
CRASH_PATTERNS = (
    ("process crashed or terminated", re.compile(r"Terminating process", re.I)),
    ("connection stalled", re.compile(r"connection stalls|does not respond", re.I)),
    ("protocol error", re.compile(r"Unknown result|invalid.*bestmove", re.I)),
)
TIMEOUT_PATTERNS = (
    ("lost on time", re.compile(r"loses on time|time forfeit", re.I)),
)
ILLEGAL_PATTERNS = (
    ("illegal move", re.compile(r"illegal move", re.I)),
)
# A Python traceback in the engine's stderr. The wrapper survives exceptions by
# design, so this is the only way a per-command failure becomes visible.
TRACEBACK = re.compile(r"^Traceback \(most recent call last\)", re.M)


def build_command(args, out_dir: Path) -> list[str]:
    """The cutechess-cli invocation, as a list (no shell quoting to get wrong).

    THE HOUSE SYNTAX, from benchmarking/engine/run_15k_ab.ps1 and the v5 A/B
    command line: `cmd=python` with a REPO-RELATIVE script path and `dir=` set
    to the repo root, one `arg=` per argument, `-resign`/`-draw` adjudication,
    `-rounds R -games 2 -repeat`. Deviating from it would make this run's PGNs
    incomparable with every other run in benchmarking/engine/games/.

    TWO CLOCK REGIMES, because the acceptance criteria need both:

      nodes  `tc=inf nodes=N timemargin=300000` — the v5 fixed-budget form. It
             is how every strength run in this repo is played, and it is the
             regime C12/C13 will use. It cannot produce a timeout by
             construction, which is exactly why it cannot be the only regime a
             "zero timeouts" criterion is checked against.
      clock  a real time control on both engines. This is the only regime in
             which "zero timeouts" and the wrapper's time management mean
             anything, and it is the one that exercises `stop`, the deadline
             and the slice loop.

    NO `-debug`. This build of cutechess-cli 1.4 rejects it in any form — bare,
    it warns `Empty value for option "-debug"` and abandons the run; with a
    value, `Invalid value`. Verified at the command line in both positions. The
    evidence it would have given is given instead by the engine:
    `uci_wrapper_v6._emit_bestmove` mirrors every bestmove to stderr and
    cutechess forwards engine stderr into the match log. That is strictly better
    for the null-bestmove check, because the mirror carries the FEN the move was
    made in — so a legitimate 0000 at a finished position is distinguishable
    from one that ended a live game.
    """
    engine_args = [
        "-u", "playing/uci_wrapper_v6.py",
        "--threads", str(args.threads),
        "--max-outstanding", str(args.max_outstanding),
        "--max-batch", str(args.max_batch),
        "--virtual-loss", str(args.virtual_loss),
        "--sim-cap", str(args.sim_cap),
    ]
    if args.fixed_sims:
        engine_args += ["--fixed-sims", str(args.fixed_sims)]

    if args.mode == "nodes":
        # `tc=inf nodes=N` makes cutechess send `go nodes N` with no clock, and
        # `timemargin` is what stops it adjudicating a long think as a loss.
        guofish_clock = ["tc=inf", f"nodes={args.nodes}",
                         f"timemargin={args.timemargin}"]
        stockfish_clock = list(guofish_clock)
    else:
        guofish_clock = [f"tc={args.tc}"]
        stockfish_clock = [f"tc={args.tc}"]

    # One `arg=` per argument, never joined: an argument containing a space
    # would otherwise be split and the engine would launch on its defaults.
    command = [str(CUTECHESS),
               "-engine", "name=guofish-v6", "cmd=python"]
    command += [f"arg={a}" for a in engine_args]
    command += [f"dir={REPO_ROOT}", "proto=uci", *guofish_clock,
                f"stderr={out_dir / 'guofish.stderr.log'}"]

    command += ["-engine", "name=stockfish", f"cmd={STOCKFISH}",
                f"dir={REPO_ROOT}", "proto=uci", *stockfish_clock,
                "option.Threads=1", "option.Hash=64",
                "option.UCI_LimitStrength=true", f"option.UCI_Elo={args.sf_elo}",
                f"stderr={out_dir / 'stockfish.stderr.log'}"]

    command += [
        "-openings", f"file={OPENINGS}", "format=pgn", "order=random",
        f"plies={args.opening_plies}",
        "-resign", "movecount=3", "score=600",
        "-draw", "movenumber=40", "movecount=8", "score=10",
        "-recover",
        "-concurrency", str(args.concurrency),
        "-rounds", str(max(1, args.games // 2)), "-games", "2", "-repeat",
        "-event", f"c11_smoke_{args.mode}",
        "-pgnout", str(out_dir / "smoke.pgn"),
    ]
    return command


def run_match(args, out_dir: Path) -> tuple[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "cutechess.log"
    command = build_command(args, out_dir)
    (out_dir / "command.txt").write_text(" ".join(command), encoding="utf-8",
                                         newline="\n")
    print(f"running: {' '.join(command[:6])} ... ({len(command)} tokens)", flush=True)
    print(f"log: {log_path}", flush=True)

    with log_path.open("w", encoding="utf-8", errors="replace", newline="\n") as sink:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace",
                                cwd=str(REPO_ROOT), bufsize=1)
        for line in proc.stdout:
            sink.write(line)
            # Progress and anything alarming, echoed. The full transcript is very
            # large under -debug (every UCI line of every game).
            if line.startswith("Finished game") or line.startswith("Score of"):
                print(line.rstrip(), flush=True)
        code = proc.wait()
    print(f"cutechess exited {code}", flush=True)
    return code, log_path


def verify(out_dir: Path, expected_games: int) -> dict:
    """Read the run back and produce the four verdicts plus the evidence."""
    log_path = out_dir / "cutechess.log"
    pgn_path = out_dir / "smoke.pgn"
    findings: dict = {"checks": [], "evidence": {}}

    def check(name: str, ok: bool, detail: str = "") -> None:
        findings["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f" — {detail}" if detail else ""), flush=True)

    if not log_path.exists():
        check("cutechess log exists", False, str(log_path))
        return findings
    text = log_path.read_text(encoding="utf-8", errors="replace")

    # The engine's stderr goes to its own file (cutechess `stderr=` per engine,
    # the house convention). Both streams are searched together: cutechess's own
    # verdicts live in one and the engine's self-report in the other, and a check
    # that read only one of them would be checking half the run.
    engine_log = out_dir / "guofish.stderr.log"
    engine_text = (engine_log.read_text(encoding="utf-8", errors="replace")
                   if engine_log.exists() else "")
    check("the engine's stderr was captured", bool(engine_text.strip()),
          str(engine_log))
    lines = text.splitlines() + engine_text.splitlines()

    def hits(patterns) -> list[tuple[str, str]]:
        out = []
        for label, pattern in patterns:
            for line in lines:
                if pattern.search(line):
                    out.append((label, line.strip()))
        return out

    # --- games completed --------------------------------------------------
    finished = [l for l in lines if l.startswith("Finished game")]
    check(f"{expected_games} games completed",
          len(finished) >= expected_games,
          f"{len(finished)} finished")

    # --- illegal moves ----------------------------------------------------
    illegal = hits(ILLEGAL_PATTERNS)
    check("zero illegal moves (cutechess)", not illegal,
          "; ".join(line for _, line in illegal[:3]))

    # --- null bestmoves ---------------------------------------------------
    # From the engine's own `[bestmove] ...` mirror, which carries the FEN.
    nulls = [l for l in lines if re.search(r"\bbestmove\s+0000\b", l)]
    check("zero null bestmoves", not nulls, "; ".join(nulls[:3]))
    mirrored = [l for l in lines if "[bestmove]" in l]
    check("the engine mirrored its bestmoves to stderr", bool(mirrored),
          f"{len(mirrored)} move(s) recorded")

    # --- crashes ----------------------------------------------------------
    crashes = hits(CRASH_PATTERNS)
    tracebacks = TRACEBACK.findall(text) + TRACEBACK.findall(engine_text)
    check("zero engine crashes / stalls", not crashes,
          "; ".join(line for _, line in crashes[:3]))
    check("zero Python tracebacks in the engine's output", not tracebacks,
          f"{len(tracebacks)} traceback(s)")

    # --- timeouts ---------------------------------------------------------
    timeouts = hits(TIMEOUT_PATTERNS)
    check("zero timeouts", not timeouts,
          "; ".join(line for _, line in timeouts[:3]))

    # --- the PGN, replayed independently ----------------------------------
    if not pgn_path.exists():
        check("PGN written", False, str(pgn_path))
    else:
        games = 0
        bad_moves = 0
        bad_terminations: list[str] = []
        with pgn_path.open(encoding="utf-8", errors="replace") as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                games += 1
                board = game.board()
                for move in game.mainline_moves():
                    if move not in board.legal_moves:
                        bad_moves += 1
                        break
                    board.push(move)
                reason = game.headers.get("Termination", "")
                if reason and reason.lower() not in ("normal", "adjudication"):
                    bad_terminations.append(
                        f"game {games}: {reason} ({game.headers.get('Result')})")
        check("every PGN game replays legally under python-chess",
              bad_moves == 0, f"{games} games, {bad_moves} with an illegal move")
        check("no abnormal PGN termination", not bad_terminations,
              "; ".join(bad_terminations[:3]))
        findings["evidence"]["pgn_games"] = games

    # --- the engine's own telemetry ---------------------------------------
    # Proof that the delivered-sims correction is live in a real match, not only
    # in the conformance harness.
    telemetry = [l for l in lines if "delivered_sims_per_s=" in l]
    check("the engine logged delivered-sims telemetry during the match",
          bool(telemetry), f"{len(telemetry)} move telemetry line(s)")
    configs = [l for l in lines if "[config]" in l]
    check("the engine logged its resolved configuration during the match",
          bool(configs), f"{len(configs)} config line(s)")

    findings["evidence"].update({
        "finished_games": len(finished),
        "illegal": illegal[:10],
        "nulls": nulls[:10],
        "crashes": crashes[:10],
        "timeouts": timeouts[:10],
        "tracebacks": len(tracebacks),
        "telemetry_lines": len(telemetry),
        "config_lines": len(configs),
    })
    findings["ok"] = all(c["ok"] for c in findings["checks"])
    return findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--mode", choices=("nodes", "clock"), default="nodes",
                        help="'nodes' is the house fixed-budget form "
                             "(tc=inf nodes=N timemargin=M); 'clock' plays a "
                             "real time control, which is the only regime in "
                             "which 'zero timeouts' means anything")
    parser.add_argument("--nodes", type=int, default=800,
                        help="simulations per move in --mode nodes (the v5 A/B "
                             "runs used 800)")
    parser.add_argument("--timemargin", type=int, default=300000,
                        help="cutechess timemargin in ms for --mode nodes")
    parser.add_argument("--tc", type=str, default="10+0.1",
                        help="time control for BOTH engines in --mode clock")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="1 by default: a smoke run is about protocol "
                             "behaviour, and concurrent games share one GPU")
    parser.add_argument("--sf-elo", type=int, default=1800,
                        help="Stockfish UCI_Elo. Low on purpose — the games "
                             "should last, not end in 20 moves")
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-outstanding", type=int, default=24)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--virtual-loss", type=float, default=2.5)
    parser.add_argument("--sim-cap", type=int, default=60000)
    parser.add_argument("--fixed-sims", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify-only", type=Path, default=None,
                        help="skip the match and verify an existing run directory")
    args = parser.parse_args(argv)

    if args.verify_only is not None:
        print(f"verifying {args.verify_only}", flush=True)
        findings = verify(args.verify_only, args.games)
        return 0 if findings.get("ok") else 1

    for binary in (CUTECHESS, STOCKFISH, OPENINGS, WRAPPER):
        if not binary.exists():
            print(f"ERROR: missing {binary}", file=sys.stderr)
            return 2

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out / stamp
    code, _ = run_match(args, out_dir)

    print("\nverifying", flush=True)
    findings = verify(out_dir, args.games)
    findings["cutechess_exit"] = code
    (out_dir / "verdict.json").write_text(json.dumps(findings, indent=1),
                                          encoding="utf-8", newline="\n")
    print(f"\nverdict -> {out_dir / 'verdict.json'}", flush=True)
    failures = [c for c in findings["checks"] if not c["ok"]]
    print(f"{len(findings['checks']) - len(failures)}/{len(findings['checks'])} "
          f"checks passed", flush=True)
    return 0 if not failures and code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
