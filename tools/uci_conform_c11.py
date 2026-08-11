#!/usr/bin/env python
"""C11 acceptance 1 — UCI conformance against playing/uci_wrapper_v6.py.

    python tools/uci_conform_c11.py
    python tools/uci_conform_c11.py --engine playing/uci_wrapper_v5.py   # contrast
    python tools/uci_conform_c11.py --keep-log runs/conform.log

Drives the wrapper as a REAL SUBPROCESS over pipes, which is the only way to
test the thing Cutechess talks to: the stdin reader thread, the flush discipline,
the stop path and the process's own exit are all outside anything an in-process
test can reach.

WHAT EACH CHECK IS FOR
======================
  handshake       `uci` -> id/option lines/uciok, and every advertised option
                  parses as `option name <N> type <T> ...`.
  option-echo     C11's stated validation: set an UNUSUAL VirtualLoss and
                  CPuctFactor, then assert both appear in the configuration the
                  engine logs to stderr on `isready`. This is the check that
                  fails on v5 by construction — v5 logs no configuration at all.
  temperature     C11b. PolicyTemperature 0.8 is ACCEPTED, reaches the resolved
                  configuration, and rebuilds the search — dropping the tree AND
                  the cache, both of which hold priors materialised at the old
                  temperature. This check was the opposite one until C11b
                  implemented the knob.
  refusals        DirichletEpsilon is advertised and REFUSED at anything but
                  0.0, loudly, with the previous value surviving; so is
                  PolicyTemperature 0, which would be a division by zero. A
                  silent accept is the defect.
  bypass          C11b, in a SECOND engine process with the shipping defaults
                  ON. The book and Syzygy states are in the config header; a
                  book move is announced, reports zero delivered simulations and
                  carries the per-game tally; `UseBook false` is honoured and the
                  same position is then searched. The main run above deliberately
                  disables both — see `run`.
  isready-load    `isready` before anything else answers `readyok` (and pays the
                  model load), then answers again in under a second.
  position/go     `position startpos moves ...` then `go nodes N` returns a
                  bestmove that is legal in the resulting position.
  go-nodes-budget the `info` line's `nodes` is DELIVERED simulations, not the
                  budget: after a tree reuse it must be strictly below `go
                  nodes N`'s N while the search still happened.
  stop            `go infinite`, then `stop`, must produce a bestmove promptly.
                  Run twice: once after a delay, once IMMEDIATELY after the go,
                  which is the ordering that catches a wrapper clearing its own
                  stop flag on dispatch.
  isready-load-b  `isready` DURING a `go infinite` answers `readyok` while the
                  search is still running, then the search still ends on `stop`.
  ponderhit       `go ponder` + `ponderhit` converts to a timed search and
                  returns a bestmove; `go ponder` + `stop` returns one too.
  under-load      20 position/go rounds back to back with a competing formatter
                  thread, asserting a legal bestmove every time and no hang.
  game-over       a position with no legal move answers `bestmove 0000` rather
                  than hanging.
  quit            the process exits within the grace period.

Exit status is 0 only if every check passed. Writes nothing to golden/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chess  # noqa: E402

DEFAULT_ENGINE = REPO_ROOT / "playing" / "uci_wrapper_v6.py"

# Deliberately not any default: the point of the option-echo check is that a
# value nobody would pick by accident travels from `setoption` to the log.
PROBE_VIRTUAL_LOSS = "3.7"
PROBE_C_PUCT_FACTOR = "1.23"

# A short opening line, so the reuse checks have a tree to inherit from.
OPENING = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]


class EngineProcess:
    """A UCI engine over pipes, with both output streams drained continuously.

    ONE reader thread per stream, for the whole process lifetime, pushing into a
    queue. A `read_until` that started its own reader would race every other one
    for the same file object and lose whichever lines the loser had already
    buffered — and the lost line is always the `bestmove`.

    stderr must be drained too, and for a harder reason than tidiness: the
    wrapper writes a dozen configuration lines per `isready` and a telemetry line
    per move, and a full pipe buffer would block the engine mid-search. That hang
    would look exactly like the engine bug this script is here to detect.
    """

    def __init__(self, engine: Path, extra_args: list[str]):
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(engine), *extra_args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", cwd=str(REPO_ROOT),
            bufsize=1)
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []
        self._stdout_q: "queue.Queue[str | None]" = queue.Queue()
        self._lock = threading.Lock()
        for stream, sink in ((self.proc.stdout, self._stdout_q), (self.proc.stderr, None)):
            threading.Thread(target=self._drain, args=(stream, sink),
                             daemon=True).start()

    def _drain(self, stream, sink) -> None:
        for raw in stream:
            line = raw.rstrip("\n")
            with self._lock:
                (self.stdout_lines if sink is not None else self.stderr_lines).append(line)
            if sink is not None:
                sink.put(line)
        if sink is not None:
            sink.put(None)     # EOF sentinel

    def stderr_text(self) -> str:
        with self._lock:
            return "\n".join(self.stderr_lines)

    def send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, predicate, timeout: float) -> list[str]:
        """Lines from stdout up to and including the first satisfying `predicate`.

        Raises TimeoutError naming what was seen, so a hang is a diagnosis rather
        than a hung harness.
        """
        collected: list[str] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"nothing matched within {timeout:.1f}s; saw "
                    f"{len(collected)} line(s), last: {collected[-5:]}")
            try:
                line = self._stdout_q.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            if line is None:
                raise TimeoutError(f"engine stdout closed; saw: {collected[-5:]}")
            collected.append(line)
            if predicate(line):
                return collected

    def close(self, grace: float = 15.0) -> int:
        self.send("quit")
        try:
            return self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return -9


class Report:
    """Pass/fail accounting. Every check records a reason, passing or not."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        return ok

    @property
    def failures(self) -> list[dict]:
        return [r for r in self.rows if not r["ok"]]


def is_bestmove(line: str) -> bool:
    return line.startswith("bestmove")


def bestmove_of(lines: list[str]) -> str:
    for line in lines:
        if line.startswith("bestmove"):
            return line.split()[1]
    raise AssertionError(f"no bestmove in {lines[-5:]}")


def info_nodes(lines: list[str]) -> list[int]:
    out = []
    for line in lines:
        if not line.startswith("info ") or " nodes " not in line:
            continue
        parts = line.split()
        out.append(int(parts[parts.index("nodes") + 1]))
    return out


def info_strings(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("info string")]


class Competitor:
    """A Python thread formatting strings, as a stand-in for a busy host.

    Imported in spirit from tests/test_c0b_contention.py's UciFormatterLoad: the
    point is a competing bytecode stream during a search, which is what the C10
    acquire-wait histogram was measured against. Reimplemented here rather than
    imported because this file drives a SUBPROCESS and the load has to run in
    the harness, not in the engine.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        n = 0
        while not self._stop.is_set():
            n += 1
            _ = f"info depth {n % 40} nodes {n * 7} score cp {n % 300 - 150}"

    def __enter__(self) -> "Competitor":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def run(engine_path: Path, extra_args: list[str], *, go_timeout: float,
        load_timeout: float, rounds: int, keep_log: Path | None = None) -> Report:
    """The protocol checks, run with the BOOK AND SYZYGY OFF.

    C11b defaults both ON, and that is right for deployment and wrong here. Half
    the checks below are statements about the SEARCH — that the reported `nodes`
    is delivered simulations and not the budget, that a reused root does less
    work, that the inflation ratio is finite and above 1. A book move bypasses
    MCTS and delivers zero simulations, and the opening this script drives
    (`OPENING`, a Ruy Lopez) is exactly the kind of line an opening book covers
    to a dozen plies. With the book on, those checks would be measuring the
    book.

    So this run disables both and `run_bypass_checks` turns them on in a second
    process to test the bypass on its own terms. That split IS the measurement
    discipline C11b requires, applied to C11b's own harness: a run that mixes
    bypassed moves into a search measurement has to say so, and the cleanest way
    to say so is not to mix them.

    A caller who passes their own `--book-path`/`--syzygy-path`/`--no-book`
    arguments overrides this, because then they have said what they want.
    """
    report = Report()
    if not any(a.startswith(("--book", "--no-book", "--syzygy", "--no-syzygy"))
               for a in extra_args):
        extra_args = ["--no-book", "--no-syzygy", *extra_args]
    engine = EngineProcess(engine_path, extra_args)

    try:
        # --- handshake ----------------------------------------------------
        print("handshake", flush=True)
        engine.send("uci")
        lines = engine.read_until(lambda l: l.strip() == "uciok", timeout=60.0)
        options = [l for l in lines if l.startswith("option name ")]
        report.check("uciok", any(l.strip() == "uciok" for l in lines))
        report.check("id name", any(l.startswith("id name ") for l in lines))
        report.check("option lines present", len(options) > 0,
                     f"{len(options)} advertised")
        malformed = [l for l in options if " type " not in l]
        report.check("every option advertises a type", not malformed,
                     f"malformed: {malformed[:3]}")
        names = {l.split()[2] for l in options}
        for required in ("VirtualLoss", "CPuctFactor", "PolicyTemperature",
                         "Threads", "MaxOutstanding", "MaxBatch", "CacheEntries",
                         "CPuctInit", "CPuctBase", "FPURoot", "FPUTree",
                         "MaxTreeDepth",
                         # C11b's five.
                         "UseBook", "BookPath", "BookSeed", "UseSyzygy",
                         "SyzygyPath"):
            report.check(f"option {required} advertised", required in names)

        # --- option echo (C11's stated validation) ------------------------
        print("option-echo", flush=True)
        engine.send(f"setoption name VirtualLoss value {PROBE_VIRTUAL_LOSS}")
        engine.send(f"setoption name CPuctFactor value {PROBE_C_PUCT_FACTOR}")
        engine.send("isready")
        engine.read_until(lambda l: l.strip() == "readyok", timeout=load_timeout)
        time.sleep(0.3)   # let the stderr drain thread catch up
        text = engine.stderr_text()
        report.check("stderr carries a configuration log", "[config]" in text)
        report.check(f"VirtualLoss={PROBE_VIRTUAL_LOSS} in the logged config",
                     f"virtual_loss={PROBE_VIRTUAL_LOSS}" in text,
                     "searched for 'virtual_loss=3.7'")
        report.check(f"CPuctFactor={PROBE_C_PUCT_FACTOR} in the logged config",
                     f"c_puct_factor={PROBE_C_PUCT_FACTOR}" in text,
                     "searched for 'c_puct_factor=1.23'")

        # --- temperature: accepted, then restored (C11b) -------------------
        #
        # This section used to live under "refusals" and asserted that
        # `PolicyTemperature 0.8` was REFUSED, because the C++ core had no
        # temperature. C11b implemented it, so the check is now the opposite
        # one: the value must travel to the resolved configuration, and the
        # stderr must say the search was rebuilt — because a temperature change
        # invalidates every prior in the tree AND in the transposition cache.
        print("temperature", flush=True)
        before = len(engine.stderr_lines)
        engine.send("setoption name PolicyTemperature value 0.8")
        engine.send("isready")
        engine.read_until(lambda l: l.strip() == "readyok", timeout=load_timeout)
        time.sleep(0.3)
        temp_text = "\n".join(engine.stderr_lines[before:])
        report.check("PolicyTemperature 0.8 is accepted",
                     "REFUSED" not in temp_text or "PolicyTemperature" not in temp_text,
                     "C11b implements it; refusing it now would be the "
                     "mirror-image defect")
        report.check("PolicyTemperature 0.8 reaches the resolved config",
                     "policy_temperature=0.8" in temp_text,
                     "searched for 'policy_temperature=0.8'")
        report.check("a temperature change rebuilds the search "
                     "(tree AND cache dropped)",
                     "rebuilding the search" in temp_text
                     and "TRANSPOSITION CACHE" in temp_text)
        engine.send("setoption name PolicyTemperature value 1.0")
        engine.send("isready")
        engine.read_until(lambda l: l.strip() == "readyok", timeout=load_timeout)

        # --- refusals -----------------------------------------------------
        print("refusals", flush=True)
        before = len(engine.stderr_lines)
        # 0 is still refused, and for a sharper reason than "unimplemented":
        # the temperature DIVIDES the logits.
        engine.send("setoption name PolicyTemperature value 0")
        engine.send("setoption name DirichletEpsilon value 0.25")
        engine.send("setoption name VirtualLoss value not-a-number")
        engine.send("setoption name NoSuchOption value 3")
        engine.send("isready")
        engine.read_until(lambda l: l.strip() == "readyok", timeout=load_timeout)
        time.sleep(0.3)
        new_text = "\n".join(engine.stderr_lines[before:])
        report.check("PolicyTemperature 0 refused loudly (it divides the logits)",
                     "REFUSED" in new_text and "PolicyTemperature" in new_text)
        report.check("DirichletEpsilon 0.25 refused loudly",
                     "REFUSED" in new_text and "DirichletEpsilon" in new_text)
        report.check("a non-numeric value is refused, not coerced",
                     "not-a-number" in new_text and "REFUSED" in new_text)
        report.check("an unknown option is refused loudly",
                     "NoSuchOption" in new_text)
        report.check("the refused values did not stick",
                     "policy_temperature=1 " in new_text
                     and "dirichlet_epsilon=0 " in new_text
                     and f"virtual_loss={PROBE_VIRTUAL_LOSS} " in new_text,
                     "the post-refusal config log still shows the identity "
                     "values and the surviving VirtualLoss")

        # --- the resolved book/Syzygy state is in the log (C11b) -----------
        #
        # This run has both features OFF (see `run`'s docstring), and the point
        # of the check is that the log says so RATHER THAN saying nothing. The
        # convention C11b requires is that the resolved state is always present,
        # so that a benchmark artifact cannot be produced without it; a run with
        # the features off has to demonstrate the same convention as a run with
        # them on.
        #
        # TWO PLACES, AND THEY HAVE DIFFERENT GUARANTEES. The one-line `as_kv()`
        # record is emitted on EVERY `isready`, so `use_book=`/`use_syzygy=` must
        # be in the lines this section just produced. The grouped human-readable
        # block is emitted only when the configuration CHANGED — a GUI sends
        # `isready` before every move and eleven identical lines per move is 660
        # lines for two games — so it is checked against the whole transcript
        # rather than against this section's slice.
        all_text = engine.stderr_text()
        report.check("every isready records the resolved book state",
                     "use_book=" in new_text and "book_seed=" in new_text,
                     "the one-line as_kv() record, which is emitted every time")
        report.check("every isready records the resolved Syzygy state",
                     "use_syzygy=" in new_text and "syzygy_path=" in new_text)
        report.check("the grouped config block names the resolved book path",
                     "[config] book" in all_text
                     and str(REPO_ROOT / "assets") in all_text,
                     "emitted on change only, so checked over the whole run")
        report.check("the grouped config block names the resolved Syzygy path",
                     "[config] syzygy" in all_text)

        # --- isready is fast once loaded ----------------------------------
        started = time.perf_counter()
        engine.send("isready")
        engine.read_until(lambda l: l.strip() == "readyok", timeout=load_timeout)
        report.check("isready answers in under a second once loaded",
                     time.perf_counter() - started < 1.0,
                     f"{1000 * (time.perf_counter() - started):.0f} ms")

        # --- position / go ------------------------------------------------
        print("position/go", flush=True)
        engine.send("ucinewgame")
        engine.send("position startpos moves " + " ".join(OPENING))
        engine.send("go nodes 3000")
        lines = engine.read_until(is_bestmove, timeout=go_timeout)
        board = chess.Board()
        for uci in OPENING:
            board.push(chess.Move.from_uci(uci))
        move = bestmove_of(lines)
        report.check("bestmove is legal in the searched position",
                     move != "0000" and chess.Move.from_uci(move) in board.legal_moves,
                     f"bestmove {move} in {board.fen()}")
        report.check("an info line was emitted", bool(info_nodes(lines)))
        report.check("info string carries both sim counts",
                     any("delivered=" in s and "nominal=" in s
                         for s in info_strings(lines)))

        # --- delivered, not nominal ---------------------------------------
        print("go-nodes-budget", flush=True)
        board.push(chess.Move.from_uci(move))
        played = OPENING + [move]
        # Search the SAME position again at the same budget. The tree is already
        # there, so the honest node count is far below the budget; a wrapper
        # reporting the budget would print 3000 both times.
        engine.send("position startpos moves " + " ".join(played))
        engine.send("go nodes 3000")
        first = engine.read_until(is_bestmove, timeout=go_timeout)
        engine.send("position startpos moves " + " ".join(played))
        engine.send("go nodes 3000")
        second = engine.read_until(is_bestmove, timeout=go_timeout)
        nodes_second = info_nodes(second)
        report.check("the second search reports fewer nodes than the budget",
                     bool(nodes_second) and max(nodes_second) < 3000,
                     f"reported nodes {max(nodes_second) if nodes_second else None} "
                     f"against a budget of 3000 (first search reported "
                     f"{max(info_nodes(first)) if info_nodes(first) else None})")
        # A LARGER budget on the same reused root, so the search does real work
        # and the reported inflation is a finite number strictly above 1 — the
        # 3000-then-3000 pair above delivers nothing at all and reports `inf`.
        engine.send("position startpos moves " + " ".join(played))
        engine.send("go nodes 6000")
        third = engine.read_until(is_bestmove, timeout=go_timeout)
        nodes_third = info_nodes(third)
        report.check("a larger budget on the reused root delivers real, "
                     "under-budget work",
                     bool(nodes_third) and 0 < max(nodes_third) < 6000,
                     f"reported nodes {max(nodes_third) if nodes_third else None} "
                     f"against a budget of 6000")
        inflations = [float(s.split("inflation=")[1].split()[0])
                      for s in info_strings(third) if "inflation=" in s]
        report.check("the reported inflation is finite and above 1 on the "
                     "reused root",
                     bool(inflations) and 1.0 < inflations[-1] < float("inf"),
                     f"inflation={inflations[-1] if inflations else None}")

        # --- stop ----------------------------------------------------------
        print("stop", flush=True)
        for label, delay in (("after 300 ms", 0.30), ("immediately", 0.0)):
            engine.send("position startpos moves " + " ".join(OPENING))
            engine.send("go infinite")
            if delay:
                time.sleep(delay)
            engine.send("stop")
            started = time.perf_counter()
            try:
                lines = engine.read_until(is_bestmove, timeout=go_timeout)
                took = time.perf_counter() - started
                stopped_move = bestmove_of(lines)
                report.check(f"stop {label} returns a bestmove",
                             stopped_move != "0000", f"{took * 1000:.0f} ms, "
                                                     f"bestmove {stopped_move}")
                report.check(f"stop {label} returns promptly", took < 3.0,
                             f"{took * 1000:.0f} ms")
            except TimeoutError as exc:
                report.check(f"stop {label} returns a bestmove", False, str(exc))

        # --- isready during a search ---------------------------------------
        print("isready-under-search", flush=True)
        engine.send("position startpos moves " + " ".join(OPENING))
        engine.send("go infinite")
        time.sleep(0.2)
        started = time.perf_counter()
        engine.send("isready")
        try:
            engine.read_until(lambda l: l.strip() == "readyok", timeout=5.0)
            report.check("isready is answered while a search is running",
                         True, f"{1000 * (time.perf_counter() - started):.0f} ms")
        except TimeoutError as exc:
            report.check("isready is answered while a search is running",
                         False, str(exc))
        engine.send("stop")
        try:
            engine.read_until(is_bestmove, timeout=go_timeout)
            report.check("the search still ends after an isready mid-search", True)
        except TimeoutError as exc:
            report.check("the search still ends after an isready mid-search",
                         False, str(exc))

        # --- ponderhit -----------------------------------------------------
        print("ponderhit", flush=True)
        engine.send("position startpos moves " + " ".join(OPENING))
        engine.send("go ponder wtime 30000 btime 30000 winc 300 binc 300")
        time.sleep(0.3)
        engine.send("ponderhit")
        started = time.perf_counter()
        try:
            lines = engine.read_until(is_bestmove, timeout=go_timeout)
            took = time.perf_counter() - started
            report.check("ponderhit converts to a timed search and returns",
                         bestmove_of(lines) != "0000", f"{took * 1000:.0f} ms")
        except TimeoutError as exc:
            report.check("ponderhit converts to a timed search and returns",
                         False, str(exc))

        engine.send("position startpos moves " + " ".join(OPENING))
        engine.send("go ponder")
        time.sleep(0.2)
        engine.send("stop")
        try:
            lines = engine.read_until(is_bestmove, timeout=go_timeout)
            report.check("a pondering search answers stop",
                         bestmove_of(lines) != "0000")
        except TimeoutError as exc:
            report.check("a pondering search answers stop", False, str(exc))

        # --- under load ----------------------------------------------------
        print(f"under-load ({rounds} rounds)", flush=True)
        engine.send("ucinewgame")
        board = chess.Board()
        played = []
        illegal = 0
        nulls = 0
        with Competitor():
            for _ in range(rounds):
                if board.is_game_over(claim_draw=False):
                    break
                engine.send("position startpos"
                            + (" moves " + " ".join(played) if played else ""))
                engine.send("go wtime 4000 btime 4000 winc 100 binc 100")
                try:
                    lines = engine.read_until(is_bestmove, timeout=go_timeout)
                except TimeoutError as exc:
                    report.check("under load: every go answered", False, str(exc))
                    break
                uci = bestmove_of(lines)
                if uci == "0000":
                    nulls += 1
                    break
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    illegal += 1
                    break
                board.push(move)
                played.append(uci)
        report.check("under load: no illegal move", illegal == 0,
                     f"{len(played)} moves played")
        report.check("under load: no null bestmove", nulls == 0)

        # --- a finished game -----------------------------------------------
        print("game-over", flush=True)
        # Fool's mate: black is checkmated, no legal moves at all.
        engine.send("position fen "
                    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        engine.send("go nodes 200")
        try:
            lines = engine.read_until(is_bestmove, timeout=go_timeout)
            report.check("a position with no legal move answers bestmove 0000",
                         bestmove_of(lines) == "0000", bestmove_of(lines))
        except TimeoutError as exc:
            report.check("a position with no legal move answers bestmove 0000",
                         False, str(exc))

        # --- quit -----------------------------------------------------------
        code = engine.close(grace=20.0)
        report.check("quit exits cleanly", code == 0, f"exit code {code}")

    finally:
        if engine.proc.poll() is None:
            engine.proc.kill()
        if keep_log is not None:
            keep_log.parent.mkdir(parents=True, exist_ok=True)
            keep_log.write_text(engine.stderr_text(), encoding="utf-8", newline="\n")
            print(f"engine stderr -> {keep_log}", flush=True)

    return report


def run_bypass_checks(engine_path: Path, extra_args: list[str], *,
                      go_timeout: float, load_timeout: float,
                      keep_log: Path | None = None) -> Report:
    """C11b's bypass, in a second process with the shipping defaults ON.

    Everything here is about ATTRIBUTION rather than about chess. A book move is
    a move this port and the Python reference would both play, by construction,
    so the only thing that keeps it from silently diluting a strength or
    throughput measurement is that the engine SAYS it took one — on the move, in
    the per-game tally, and in the configuration header. These are the checks
    that the saying happens.
    """
    report = Report()
    engine = EngineProcess(engine_path, extra_args)
    try:
        print("bypass (defaults ON)", flush=True)
        engine.send("uci")
        engine.read_until(lambda l: l.strip() == "uciok", timeout=60.0)
        engine.send("isready")
        engine.read_until(lambda l: l.strip() == "readyok", timeout=load_timeout)
        time.sleep(0.3)
        text = engine.stderr_text()

        report.check("book and Syzygy default ON",
                     "use_book=true" in text and "use_syzygy=true" in text,
                     "the resolved config log, not the dataclass default")
        report.check("BookSeed defaults to the deterministic 0",
                     "book_seed=0" in text and "DETERMINISTIC" in text)
        report.check("the book reader opened or said why not",
                     "[book] opened" in text or "[book] WARNING" in text)
        report.check("the Syzygy reader opened or said why not",
                     "[syzygy] opened" in text or "[syzygy] WARNING" in text)

        book_open = "[book] opened" in text
        engine.send("ucinewgame")
        engine.send("position startpos")
        engine.send("go nodes 400")
        lines = engine.read_until(is_bestmove, timeout=go_timeout)
        strings = info_strings(lines)
        move = bestmove_of(lines)
        board = chess.Board()
        report.check("the start position answers a legal move",
                     chess.Move.from_uci(move) in board.legal_moves, move)

        if book_open:
            report.check("a book move is announced as one",
                         any("string book " in s for s in strings)
                         or any("source=book" in s for s in strings),
                         f"info strings: {strings[:3]}")
            report.check("a book move reports zero delivered simulations",
                         any("delivered=0" in s for s in strings),
                         f"info strings: {strings[:3]}")
            report.check("a book move carries the per-game tally",
                         any("game_counts" in s and "book=1" in s for s in strings),
                         f"info strings: {strings[:3]}")
            # And the tally is reported when the game ends.
            before = len(engine.stderr_lines)
            engine.send("ucinewgame")
            engine.send("isready")
            engine.read_until(lambda l: l.strip() == "readyok", timeout=load_timeout)
            time.sleep(0.3)
            ended = "\n".join(engine.stderr_lines[before:])
            report.check("ucinewgame reports the finished game's decision counts",
                         "[game] decided" in ended and "book=" in ended,
                         ended.splitlines()[:2])
        else:
            report.check("book checks skipped: no book file", True,
                         "assets/gm2001.bin is absent; the engine warned and "
                         "disabled it, which is the required behaviour")

        # Turning the book off must be honoured, and must show in the log.
        before = len(engine.stderr_lines)
        engine.send("setoption name UseBook value false")
        engine.send("isready")
        engine.read_until(lambda l: l.strip() == "readyok", timeout=load_timeout)
        time.sleep(0.3)
        off_text = "\n".join(engine.stderr_lines[before:])
        report.check("UseBook false is accepted and logged",
                     "use_book=false" in off_text)

        engine.send("position startpos")
        engine.send("go nodes 400")
        lines = engine.read_until(is_bestmove, timeout=go_timeout)
        nodes = info_nodes(lines)
        report.check("with the book off, the start position is SEARCHED",
                     bool(nodes) and max(nodes) > 0,
                     f"info nodes: {nodes[:5]}")

        code = engine.close(grace=20.0)
        report.check("quit exits cleanly (bypass run)", code == 0,
                     f"exit code {code}")
    finally:
        if engine.proc.poll() is None:
            engine.proc.kill()
        if keep_log is not None:
            keep_log.parent.mkdir(parents=True, exist_ok=True)
            keep_log.write_text(engine.stderr_text(), encoding="utf-8", newline="\n")
            print(f"bypass engine stderr -> {keep_log}", flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--engine-arg", action="append", default=[],
                        help="extra argument for the engine process; repeatable")
    parser.add_argument("--go-timeout", type=float, default=60.0)
    parser.add_argument("--load-timeout", type=float, default=300.0,
                        help="the first isready pays the checkpoint load and the "
                             "CUDA graph capture")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--keep-log", type=Path, default=None,
                        help="write the engine's stderr here")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--skip-bypass", action="store_true",
                        help="skip the C11b book/Syzygy section, which starts a "
                             "second engine process with the shipping defaults ON")
    args = parser.parse_args(argv)

    print(f"engine: {args.engine} {' '.join(args.engine_arg)}", flush=True)
    report = run(args.engine, args.engine_arg, go_timeout=args.go_timeout,
                 load_timeout=args.load_timeout, rounds=args.rounds,
                 keep_log=args.keep_log)

    if not args.skip_bypass:
        bypass_log = (args.keep_log.with_suffix(".bypass" + args.keep_log.suffix)
                      if args.keep_log else None)
        bypass = run_bypass_checks(args.engine, args.engine_arg,
                                   go_timeout=args.go_timeout,
                                   load_timeout=args.load_timeout,
                                   keep_log=bypass_log)
        report.rows.extend(bypass.rows)

    passed = len(report.rows) - len(report.failures)
    print(f"\n{passed}/{len(report.rows)} checks passed", flush=True)
    for row in report.failures:
        print(f"  FAILED {row['check']}: {row['detail']}", flush=True)

    if args.json_out:
        args.json_out.write_text(json.dumps(report.rows, indent=1),
                                 encoding="utf-8", newline="\n")
    return 0 if not report.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
