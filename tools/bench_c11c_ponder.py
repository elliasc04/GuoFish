#!/usr/bin/env python
"""C11c measurements — stop latency, arena high-water, and the ponder-hit rate.

    python tools/bench_c11c_ponder.py                       # everything
    python tools/bench_c11c_ponder.py --only stop-latency
    python tools/bench_c11c_ponder.py --only hit-rate \
        --log benchmarking/engine/games/c11c_ponder/guofish.stderr.log

WHAT IS HERE AND WHY IT IS NOT IN THE TEST SUITE
================================================
`tests/test_c11c_ponder.py` asserts the BEHAVIOUR — a stop is answered, an
undersized arena degrades, a hit retains its simulations. What it deliberately
does not do is measure DISTRIBUTIONS, because a test that asserted a p99 would
either be flaky or be so loose it asserted nothing. These are the three figures
C11c's acceptance asks to be reported rather than bounded:

  stop-latency  A HISTOGRAM, NOT A MEAN — the same discipline C10d applied to
                GIL acquisition, and for the same reason: the cost of a stop is
                entirely in its tail. C11 measured 7-109 ms for `stop` under
                `go infinite`; C11c puts that on the ponder MISS critical path,
                where the opponent has already moved and the engine is burning
                its own clock until it answers. So the tail is now a clock cost
                and not merely an annoyance.

  arena         Measured high-water against the PREDICTED `arena_nodes`, as a
                utilisation percentage, over a full game at a real time control.
                Requirement 3d: "A safety factor nobody has seen engage is an
                assumption." Reported both here, where it should not engage, and
                by `--only degradation`, where it is made to.

  hit-rate      The share of ponders the opponent walked into, read out of an
                ordinary match log. It is the number that decides whether the
                salvage-on-miss design this chunk rejected is ever worth
                revisiting (DECISIONS.md, C11c), and it has to be recoverable
                from a run somebody already did rather than from an instrumented
                one nobody will repeat.

Every subprocess here is driven over a REAL PIPE, for C11c's standing reason:
`go ponder` behind a pipe and `go ponder` at a terminal are different programs,
and only the first is the one Cutechess runs.

Writes to `--out` (default `runs/c11c/`) and to nothing else. `golden/` is never
touched (Global Rule 2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# THIS SCRIPT PRINTS ENGINE OUTPUT VERBATIM, and a Windows console defaults to
# cp1252. The engine's stdout is ASCII by protocol, but the pipe is read with
# `errors="replace"`, so a byte that arrives mangled becomes U+FFFD — which
# cp1252 then cannot encode, and the run dies in `print` rather than in
# anything it was measuring. Reconfiguring here costs nothing and removes a
# whole class of failure that has nothing to do with the measurement.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):                     # pragma: no cover
        pass

import chess  # noqa: E402
import guofish_core  # noqa: E402

from playing.v6.playv6 import EngineConfig  # noqa: E402

# The pipe harness the acceptance tests use, imported rather than reimplemented
# so a change to one cannot leave the other testing a different transport.
sys.path.insert(0, str(REPO_ROOT / "tests"))
from test_c11c_ponder import Pipe, _boot  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "runs" / "c11c"
OPENING = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]


def percentiles(samples: list[float]) -> dict:
    """p50/p90/p95/p99/max plus the count. A mean is included and is the number
    NOT to make a decision on — it is there so a reader can see how far it sits
    from p99 and therefore how much a mean would have hidden."""
    if not samples:
        return {"n": 0}
    ordered = sorted(samples)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        index = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
        return ordered[index]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def histogram(samples: list[float], bins: list[float], unit: str = "ms") -> str:
    """A text histogram over explicit bin EDGES.

    Explicit rather than computed, because the interesting comparison is against
    a fixed prior — C11's 7-109 ms range — and auto-scaled bins would move the
    boundaries every run and make two runs incomparable.
    """
    if not samples:
        return "  (no samples)"
    counts = [0] * (len(bins) + 1)
    for value in samples:
        placed = False
        for i, edge in enumerate(bins):
            if value < edge:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    widest = max(counts) or 1
    lines = []
    low = 0.0
    for i, edge in enumerate(bins):
        bar = "#" * int(40 * counts[i] / widest)
        lines.append(f"  [{low:8.1f}, {edge:8.1f}) {unit}  {counts[i]:5d}  {bar}")
        low = edge
    bar = "#" * int(40 * counts[-1] / widest)
    lines.append(f"  [{low:8.1f},      inf) {unit}  {counts[-1]:5d}  {bar}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Stop latency during ponder
# ---------------------------------------------------------------------------


def measure_stop_latency(samples: int, think_s: float) -> dict:
    """`stop` -> `bestmove`, measured over a pipe, with a ponder in flight.

    THE CLOCK STARTS AT THE `stop` WRITE and stops at the `bestmove` READ, so it
    includes everything a GUI would wait for: the pipe, the reader thread, the
    interrupt reaching the workers, the in-flight batch draining, the principal
    variation walk, and the final `info` formatting. Measuring inside the engine
    would produce a smaller number that no opponent ever experiences.

    IT WALKS A GAME rather than re-pondering one position, and the first version
    of this function did the latter and produced a monotonically rising series —
    7.6 ms on the first sample and 227.5 ms on the twenty-fifth. That was not
    noise. Re-pondering the same root never advances the position, so the tree
    accumulates every sample's simulations and the end-of-search PV walk, which
    is O(visited nodes), grows with it. The number that measurement produced
    describes a state no game reaches.

    So each sample plays the previous sample's move, `apply_move` compacts the
    tree the way a real game does, and `root_visits` at the moment of the stop
    is recorded ALONGSIDE the latency. The correlation between the two is the
    finding, not a nuisance: if stop latency scales with tree size, then the
    tail is a property of the position rather than of the mechanism, and a
    single percentile would hide that.

    `think_s` is how long the ponder runs before the stop. A stop landing
    between slices is nearly free and one landing mid-batch waits for the GPU,
    so the delay is JITTERED across a slice period rather than fixed.
    """
    pipe = _boot("--no-book", "--no-syzygy", "--ponder")
    latencies: list[float] = []
    tree_sizes: list[int] = []
    slice_s = EngineConfig().slice_seconds
    board = chess.Board()
    moves: list[str] = []
    for uci in OPENING:
        board.push(chess.Move.from_uci(uci))
        moves.append(uci)
    try:
        pipe.send("ucinewgame")
        for i in range(samples):
            if board.is_game_over():
                board = chess.Board()
                moves = []
                for uci in OPENING:
                    board.push(chess.Move.from_uci(uci))
                    moves.append(uci)
                pipe.send("ucinewgame")

            pipe.send("position startpos moves " + " ".join(moves))
            pipe.send("go ponder wtime 30000 btime 30000 winc 300 binc 300")
            time.sleep(think_s + slice_s * (i % 7) / 7.0)
            started = time.perf_counter()
            pipe.send("stop")
            uci, lines = pipe.bestmove(timeout=60.0)
            latencies.append((time.perf_counter() - started) * 1000.0)

            visits = 0
            for line in pipe.stderr_lines[-30:]:
                found = re.search(r"root_visits=(\d+)", line)
                if found:
                    visits = int(found.group(1))
            tree_sizes.append(visits)
            print(f"  stop {i + 1}/{samples}: {latencies[-1]:7.1f} ms  "
                  f"(root_visits {visits:,})", flush=True)

            # ADVANCE THE GAME. The move the engine just returned is played, so
            # the next ponder starts from a compacted tree rather than from an
            # ever-growing one.
            move = chess.Move.from_uci(uci)
            if uci == "0000" or move not in board.legal_moves:
                board = chess.Board()
                moves = []
                pipe.send("ucinewgame")
                continue
            board.push(move)
            moves.append(uci)
    finally:
        pipe.close()

    stats = percentiles(latencies)
    print()
    print("stop latency during ponder (ms), from the `stop` write to the "
          "`bestmove` read")
    print(histogram(latencies, [1, 2, 5, 10, 20, 50, 100, 200]))
    print(f"  {stats}")
    print(f"  root_visits at stop: {percentiles([float(v) for v in tree_sizes])}")
    print("  C11 baseline for `stop` under `go infinite`: 7-109 ms. This now "
          "sits on the ponder MISS critical path, where the opponent has "
          "already moved and the engine is spending its OWN clock until it "
          "answers.")
    return {"samples_ms": latencies, "root_visits": tree_sizes,
            "stats": stats,
            "root_visits_stats": percentiles([float(v) for v in tree_sizes])}


# ---------------------------------------------------------------------------
# 2. Arena high-water over a full game at a real time control
# ---------------------------------------------------------------------------


_ARENA = re.compile(r"arena_hw=(\d+)/(\d+)")
_PONDER_SIMS = re.compile(r"\bponder_sims=(\d+)")
_SEARCH_SIMS = re.compile(r"\bsearch_sims=(\d+)")


def measure_arena_over_a_game(plies: int, movetime_ms: int, ponder: bool) -> dict:
    """Self-play `plies` half-moves at a real clock, tracking arena occupancy.

    A GAME rather than a position, because the figure the sizing formula has to
    survive is the peak across a whole game with tree reuse in play — a single
    position understates it, and the arena is never reset between moves within
    one game.

    Pondering is driven the way a GUI drives it: `go ponder` on the position
    after the move this side is about to face, then `ponderhit`, because the
    self-play driver knows both moves and can therefore always produce a hit.
    THAT IS THE WORST CASE FOR THE ARENA and is the point — a 100% hit rate
    means every move carries its ponder's nodes into the timed search, which is
    exactly the demand the `sims_per_move + ponder_max_sims` term is sized for.
    """
    args = ["--no-book", "--no-syzygy"] + (["--ponder"] if ponder else [])
    pipe = _boot(*args)
    board = chess.Board()
    moves: list[str] = []
    rows: list[dict] = []
    try:
        pipe.send("ucinewgame")
        for ply in range(plies):
            if board.is_game_over():
                break
            line = ("position startpos"
                    + (" moves " + " ".join(moves) if moves else ""))
            pipe.send(line)
            pipe.send(f"go movetime {movetime_ms}")
            uci, out = pipe.bestmove(timeout=120.0)
            if uci == "0000":
                break
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise SystemExit(f"illegal move {uci} in {board.fen()}")
            board.push(move)
            moves.append(uci)

            text = "\n".join(pipe.stderr_lines[-40:])
            found = _ARENA.findall(text)
            high_water, capacity = (int(found[-1][0]), int(found[-1][1])) if found else (0, 0)
            rows.append({"ply": ply + 1, "move": uci,
                         "arena_high_water": high_water,
                         "arena_capacity": capacity})
            print(f"  ply {ply + 1:3d} {uci}  arena {high_water:,}/{capacity:,} "
                  f"({(high_water / capacity if capacity else 0):.1%})", flush=True)

            if ponder and not board.is_game_over():
                # The GUI's ponder handshake: search the position after the
                # reply we are about to make, then confirm it. Predicting our
                # own next move is what makes this a guaranteed hit.
                pipe.send("position startpos moves " + " ".join(moves))
                pipe.send("go ponder wtime 20000 btime 20000 winc 200 binc 200")
                time.sleep(movetime_ms / 1000.0)
                pipe.send("ponderhit")
                pipe.bestmove(timeout=120.0)
    finally:
        pipe.close()

    peak = max((r["arena_high_water"] for r in rows), default=0)
    capacity = max((r["arena_capacity"] for r in rows), default=0)
    config = EngineConfig()
    print()
    print(f"arena high-water over {len(rows)} plies at movetime {movetime_ms} ms, "
          f"ponder={'on' if ponder else 'off'}")
    print(f"  predicted arena_nodes : {config.arena_nodes:,} "
          f"(60 x ({config.sims_per_move:,} + {config.ponder_max_sims_resolved:,}))")
    print(f"  engine capacity       : {capacity:,}")
    print(f"  measured high-water   : {peak:,}")
    print(f"  utilisation           : {(peak / capacity if capacity else 0):.1%}")
    print("  Requirement 3d: a safety factor nobody has seen engage is an "
          "assumption. Utilisation well under 100% here is the formula holding; "
          "`--only degradation` is the same mechanism made to fire.")
    return {"rows": rows, "peak": peak, "capacity": capacity,
            "predicted": config.arena_nodes,
            "utilisation": peak / capacity if capacity else 0.0}


# ---------------------------------------------------------------------------
# 3. The degradation path, made to fire
# ---------------------------------------------------------------------------


def measure_degradation(capacity: int, budget: int) -> dict:
    """Requirement 3d's second half: prove the backstop actually engages.

    Run in a SEPARATE PROCESS at a deliberately undersized arena, over a pipe,
    so what is measured is the engine a GUI would be talking to rather than a
    library call.
    """
    pipe = _boot("--no-book", "--no-syzygy", "--arena-capacity", str(capacity))
    try:
        pipe.send("position startpos moves " + " ".join(OPENING))
        pipe.send(f"go nodes {budget}")
        uci, lines = pipe.bestmove(timeout=120.0)
        info = [l for l in lines if l.startswith("info string")]
        delivered = 0
        exhausted = False
        for line in info:
            for token in line.split():
                if token.startswith("delivered="):
                    delivered = int(token.split("=")[1])
                if token == "arena_exhausted=true":
                    exhausted = True
        notice = [l for l in info if "arena exhausted at" in l]

        board = chess.Board()
        for played in OPENING:
            board.push(chess.Move.from_uci(played))
        legal = uci != "0000" and chess.Move.from_uci(uci) in board.legal_moves

        print()
        print(f"degradation at arena_capacity={capacity:,}, budget {budget}")
        print(f"  bestmove          : {uci}  (legal: {legal})")
        print(f"  delivered         : {delivered} of {budget}")
        print(f"  arena_exhausted   : {exhausted}")
        print(f"  stdout notice     : {notice[0] if notice else 'MISSING'}")
        print(f"  traceback in log  : {'Traceback' in pipe.stderr_text()}")
        return {"capacity": capacity, "budget": budget, "bestmove": uci,
                "legal": legal, "delivered": delivered,
                "arena_exhausted": exhausted,
                "notice": notice[0] if notice else None,
                "traceback": "Traceback" in pipe.stderr_text()}
    finally:
        pipe.close()


# ---------------------------------------------------------------------------
# 4. Ponder-hit rate, from a match log
# ---------------------------------------------------------------------------


_VERDICT = re.compile(r"\[ponder\] verdict=(hit|miss) ponder_sims=(\d+) "
                      r"ponder_wall_ms=([\d.]+) bypassed=(\S+)")


def measure_hit_rate(log: Path) -> dict:
    """Count `[ponder] verdict=` lines out of an engine stderr log.

    One canonical line per resolved ponder, emitted by
    `uci_wrapper_v6.handle_go_ponder`. Reading the log rather than
    instrumenting a special run is deliberate: this number has to be
    recoverable from the 20-game smoke somebody already ran.

    A `bypassed=` verdict is counted SEPARATELY and excluded from the rate. Those
    are ponders that never happened — the book or the tablebase already answered
    the position — so folding them in would report a rate that describes which
    openings came up rather than how well the engine predicts a reply.
    """
    if not log.is_file():
        raise SystemExit(f"no such log: {log}")
    text = log.read_text(encoding="utf-8", errors="replace")
    hits = misses = bypassed = 0
    ponder_sims: list[int] = []
    for verdict, sims, _wall, bypass in _VERDICT.findall(text):
        if bypass != "no":
            bypassed += 1
            continue
        if verdict == "hit":
            hits += 1
            ponder_sims.append(int(sims))
        else:
            misses += 1
    total = hits + misses
    rate = hits / total if total else 0.0
    print()
    print(f"ponder-hit rate from {log}")
    print(f"  ponders resolved : {total}  (+{bypassed} skipped: book/tablebase)")
    print(f"  hits             : {hits}")
    print(f"  misses           : {misses}")
    print(f"  hit rate         : {rate:.1%}")
    if ponder_sims:
        print(f"  sims carried into a hit: {percentiles([float(s) for s in ponder_sims])}")
    print("  This is the number that decides whether salvage-on-miss is worth "
          "revisiting. See DECISIONS.md, C11c: a miss costs nothing that was "
          "otherwise being used, so a LOW rate is not by itself an argument for "
          "salvage — it has to be low AND the lost work has to matter.")
    return {"log": str(log), "hits": hits, "misses": misses,
            "bypassed": bypassed, "rate": rate,
            "ponder_sims": percentiles([float(s) for s in ponder_sims])}


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", default=[],
                        choices=("stop-latency", "arena", "degradation",
                                 "hit-rate"),
                        help="run one section; repeatable. Default: all but "
                             "hit-rate, which needs a match log")
    parser.add_argument("--stop-samples", type=int, default=25)
    parser.add_argument("--think", type=float, default=0.45,
                        help="seconds of ponder before the stop")
    parser.add_argument("--plies", type=int, default=40)
    parser.add_argument("--movetime", type=int, default=400,
                        help="ms per move for the arena game; 400 ms is about "
                             "what a 10+0.1 clock allots")
    parser.add_argument("--small-arena", type=int, default=32768)
    parser.add_argument("--degrade-budget", type=int, default=4000)
    parser.add_argument("--log", type=Path, default=None,
                        help="engine stderr log for --only hit-rate")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    sections = args.only or ["stop-latency", "arena", "degradation"]
    results: dict = {
        "build": guofish_core.build_info(),
        "config": EngineConfig().as_dict(),
    }

    if "stop-latency" in sections:
        print("=== stop latency during ponder ===", flush=True)
        results["stop_latency"] = measure_stop_latency(args.stop_samples,
                                                       args.think)
    if "arena" in sections:
        print("\n=== arena high-water, ponder ON, full game ===", flush=True)
        results["arena_ponder_on"] = measure_arena_over_a_game(
            args.plies, args.movetime, ponder=True)
    if "degradation" in sections:
        print("\n=== graceful degradation ===", flush=True)
        results["degradation"] = measure_degradation(args.small_arena,
                                                     args.degrade_budget)
    if "hit-rate" in sections:
        print("\n=== ponder-hit rate ===", flush=True)
        if args.log is None:
            raise SystemExit("--only hit-rate needs --log <engine stderr log>")
        results["hit_rate"] = measure_hit_rate(args.log)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "bench_c11c_ponder.json"
    path.write_text(json.dumps(results, indent=1, default=str),
                    encoding="utf-8", newline="\n")
    print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
