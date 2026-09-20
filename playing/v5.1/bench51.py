"""CPU benchmarks for the v5.1 engine. The measurement this tree did not have.

    python playing/v5.1/bench51.py --verify      # numerics, run this first
    python playing/v5.1/bench51.py --breakdown   # where the time goes
    python playing/v5.1/bench51.py --search      # the A/B that decides things
    python playing/v5.1/bench51.py --forward     # forward-only scaling
    python playing/v5.1/bench51.py --all

WHY THIS FILE EXISTS
====================
BENCH.md is the C++ port's, measured on an i5-12600K desktop, and every
breakdown in it lands on a CUDA forward. Its one Python-CPU figure -- 236 us/sim
(C5, "vs Python CPU") -- is the traverse loop against a REPLAY evaluator, i.e.
an unordered_map lookup where the network should be. Nothing in that document
says what a CPU forward costs, so nothing in it can rank a CPU optimization.
This file measures the thing that was missing.

THE ONE RULE THIS HARNESS FOLLOWS
=================================
**Arms are interleaved, never run in blocks.** This is a 15 W laptop part. A
forward measured cold is 19.5 ms and the same forward after a sustained search
is 24.6 ms -- 26% drift, entirely from clocks, in the direction of whichever arm
ran later. Blocked arms would hand the first arm a systematic win of roughly the
size of the effect being measured. Repeats alternate A/B/A/B and the reported
figure is the per-repeat MEDIAN, so the drift divides out of the ratio instead
of adding to it. C12b-4 does the same thing for the same reason on the GPU side
("arms interleaved repeat by repeat so the ~4% GPU-clock drift divides out").

Absolute numbers from this file are worth less than the ratios. Quote the
ratios.
"""

import argparse
import io
import contextlib
import random
import statistics
import sys
import time
from pathlib import Path

import chess
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import core.mctsv4 as mctsv4
import core.mctsv5 as mctsv5
from playing.v5 import playv5
from playv51 import jit_wrap

CHECKPOINT = _PROJECT_ROOT / "models" / "v5_10.9M_best_fp16.pt"

# Four positions spanning what a game actually asks the net: an open tactical
# middlegame, two quiet closed ones, and a pawn endgame where the tree goes deep
# and the branching collapses. One position would measure one branching factor.
POSITIONS = [
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2P1PN2/PP1NBPPP/R1BQ1RK1 w - - 0 10",
    "2rq1rk1/pb1nbppp/1p2pn2/8/2BP4/2N1PN2/PP3PPP/R1BQR1K1 w - - 4 13",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]


def load(device=torch.device("cpu")):
    """The eager INT8 model and a traced copy of it."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eager = playv5.load_model(CHECKPOINT, device)
        traced = jit_wrap(eager, device)
    torch.set_num_threads(1)
    return eager, traced


def corpus(n=250, seed=7):
    """Random legal walks. Not a curated set -- the point is coverage of piece
    counts, castling states and en-passant squares, which is what the tokenizer
    and the traced graph could plausibly disagree on."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(0, 80)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
            if b.is_game_over():
                break
        if not b.is_game_over():
            out.append(b.copy())
    return out


# --------------------------------------------------------------------------
# verify: numerics. Nothing else in this file means anything if this fails.
# --------------------------------------------------------------------------

def bench_verify():
    eager, traced = load()
    boards = corpus()
    print(f"=== verify: {len(boards)}-position corpus ===\n")

    # 1. The tokenizer rewrite must be bit-identical to v4's.
    mismatched = sum(1 for b in boards
                     if not torch.equal(mctsv4.board_to_tokens(b),
                                        mctsv5.board_to_tokens(b)))
    print(f"tokenizer  mctsv5 vs mctsv4: {len(boards) - mismatched}/{len(boards)} "
          f"identical  {'OK' if not mismatched else 'FAIL'}")

    # 2. The traced forward must be bit-identical to the eager one. Not "close":
    #    identical. A traced module that merely agrees to 1e-6 would be a
    #    different engine, and would need the Gate 2'-style move-agreement
    #    re-certification that C12b needed for Inductor. This one does not.
    bad_p = bad_v = 0
    max_dp = max_dv = 0.0
    with torch.no_grad():
        for b in boards:
            t = mctsv5.board_to_tokens(b).unsqueeze(0)
            pe, ve = eager(t)
            pt, vt = traced(t)
            dp = (pe - pt).abs().max().item()
            dv = (ve - vt).abs().max().item()
            max_dp, max_dv = max(max_dp, dp), max(max_dv, dv)
            bad_p += dp != 0.0
            bad_v += dv != 0.0
    print(f"forward    policy: {len(boards) - bad_p}/{len(boards)} bit-identical  "
          f"max|dlogit|={max_dp:.3e}")
    print(f"forward    value:  {len(boards) - bad_v}/{len(boards)} bit-identical  "
          f"max|dvalue|={max_dv:.3e}")

    # 3. A batch-1 trace must stay correct at other widths. Inline mode only ever
    #    uses 1, so this is headroom rather than a requirement -- but a trace that
    #    had baked in a batch dimension would be a trap for anyone who later
    #    batches, and it is one assert to find out.
    print()
    with torch.no_grad():
        for bs in (1, 2, 8, 24):
            t = torch.cat([mctsv5.board_to_tokens(b).unsqueeze(0)
                           for b in boards[:bs]])
            pe, ve = eager(t)
            pt, vt = traced(t)
            print(f"batch {bs:>2}: max|dlogit|={(pe - pt).abs().max():.3e}  "
                  f"max|dvalue|={(ve - vt).abs().max():.3e}")

    # 4. The raw-policy path takes a mask the traced graph never saw. JitForward
    #    routes it to the eager module; if that routing ever breaks this raises.
    print()
    mask = torch.zeros(1, 4096, dtype=torch.bool)
    mask[0, :64] = True
    with torch.no_grad():
        t = mctsv5.board_to_tokens(boards[0]).unsqueeze(0)
        masked, _ = traced(t, legal_move_mask=mask)
    finite = int(torch.isfinite(masked).sum())
    print(f"masked forward routes to eager: {finite} finite logits of 4096 "
          f"(expect 64)  {'OK' if finite == 64 else 'FAIL'}")
    print(f"parameters() reachable through the wrapper: "
          f"{next(traced.parameters()).dtype}")

    ok = not mismatched and not bad_p and not bad_v and finite == 64
    print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
    return ok


# --------------------------------------------------------------------------
# breakdown: where the time goes. Requires GUOFISH_INSTR=1 at import.
# --------------------------------------------------------------------------

def bench_breakdown(sims=300):
    if not mctsv5._INSTR:
        print("--breakdown needs the instrumentation compiled in. Re-run as:\n"
              "    GUOFISH_INSTR=1 python playing/v5.1/bench51.py --breakdown\n"
              "  (PowerShell: $env:GUOFISH_INSTR=1; python ...)")
        return False
    eager, traced = load()

    # The instrumentation times everything EXCEPT the forward -- its evaluator
    # probes only cover the batched GPU path, and inline mode does not go
    # through them. Wrap eval_inline to close that gap; this is the measurement
    # the harness was missing and the reason 95% was not visible before.
    fwd = {"s": 0.0, "n": 0}
    original = mctsv5.BatchedEvaluator.eval_inline

    def timed(self, tokens):
        t0 = time.perf_counter()
        out = original(self, tokens)
        fwd["s"] += time.perf_counter() - t0
        fwd["n"] += 1
        return out

    mctsv5.BatchedEvaluator.eval_inline = timed
    try:
        engine = mctsv5.ParallelMCTS(traced, torch.device("cpu"), num_workers=1)
        engine.search(chess.Board(), num_simulations=32)   # warm
        mctsv5.instr_reset()
        fwd["s"], fwd["n"] = 0.0, 0
        wall, n = 0.0, 0
        for fen in POSITIONS:
            engine.reset()
            engine.cache = mctsv5.TranspositionCache(max_size=100_000)
            t0 = time.perf_counter()
            engine.search(chess.Board(fen), num_simulations=sims)
            wall += time.perf_counter() - t0
            n += sims
    finally:
        mctsv5.BatchedEvaluator.eval_inline = original

    d = mctsv5.instr_merge()
    rows = [
        ("NN forward (inline)", fwd["s"],
         f"{fwd['n']} evals, {fwd['s'] / max(fwd['n'], 1) * 1e3:.1f} ms each"),
        ("board_to_tokens", d.get("board_to_tokens_s", 0),
         f"{int(d.get('board_to_tokens_calls', 0))} calls"),
        ("expand: children", d.get("expand_children_s", 0),
         f"{d.get('expand_children_n', 0) / max(d.get('expand_calls', 1), 1):.1f} children avg"),
        ("expand: policy softmax", d.get("expand_softmax_s", 0), ""),
        ("expand_root", d.get("expand_root_s", 0), ""),
        ("repetition history", d.get("rep_history_s", 0), ""),
        ("virtual-loss reset", d.get("vloss_reset_s", 0), ""),
    ]
    accounted = sum(r[1] for r in rows)

    print(f"=== breakdown: W=1, {n} sims, {wall:.1f}s = {wall / n * 1e3:.1f} ms/sim "
          f"({n / wall:.1f} sims/s) ===\n")
    print(f"{'phase':<26}{'ms/sim':>9}{'% wall':>9}   note")
    for name, secs, note in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:<26}{secs / n * 1e3:>9.3f}{100 * secs / wall:>8.1f}%   {note}")
    print(f"{'select+backup+copy+terminal':<26}{(wall - accounted) / n * 1e3:>9.3f}"
          f"{100 * (wall - accounted) / wall:>8.1f}%   "
          f"{d.get('select_steps', 0) / n:.1f} select steps/sim (by subtraction)")
    print("\nInstrumentation inflates the Python phases it wraps and not the "
          "forward,\nso the forward's true share is if anything higher than shown.")
    return True


# --------------------------------------------------------------------------
# search: the A/B. Interleaved.
# --------------------------------------------------------------------------

def _one_search_repeat(model, workers, sims):
    engine = mctsv5.ParallelMCTS(model, torch.device("cpu"), num_workers=workers)
    engine.search(chess.Board(), num_simulations=32)   # warm, untimed
    total = 0.0
    for fen in POSITIONS:
        engine.reset()
        engine.cache = mctsv5.TranspositionCache(max_size=100_000)
        t0 = time.perf_counter()
        engine.search(chess.Board(fen), num_simulations=sims)
        total += time.perf_counter() - t0
    cache = engine.cache
    hit = 100 * cache.hits / max(1, cache.hits + cache.misses)
    return len(POSITIONS) * sims / total, hit


def bench_search(sims=400, repeats=3, worker_counts=(2, 4, 6, 8)):
    eager, traced = load()
    print(f"=== search A/B: {len(POSITIONS)} positions x {sims} sims, "
          f"median of {repeats}, arms interleaved ===\n")
    print(f"{'workers':>8} {'eager sims/s':>13} {'traced sims/s':>14} "
          f"{'speedup':>8} {'cache hit':>10}")
    for workers in worker_counts:
        e_runs, t_runs, hits = [], [], []
        for _ in range(repeats):
            # A then B inside the repeat: consecutive, so they see the most
            # similar thermal state the machine can offer.
            sps, _ = _one_search_repeat(eager, workers, sims)
            e_runs.append(sps)
            sps, hit = _one_search_repeat(traced, workers, sims)
            t_runs.append(sps)
            hits.append(hit)
        e, t = statistics.median(e_runs), statistics.median(t_runs)
        print(f"{workers:>8} {e:>13.1f} {t:>14.1f} {t / e:>7.2f}x "
              f"{statistics.median(hits):>9.1f}%")
    print(f"\ndefault workers on this machine: {mctsv5.cpu_worker_default()} "
          f"(physical cores; os.cpu_count()={__import__('os').cpu_count()})")


# --------------------------------------------------------------------------
# forward: scaling, isolated from the search.
# --------------------------------------------------------------------------

def bench_forward(seconds=3.0):
    import threading
    eager, traced = load()
    x = mctsv5.board_to_tokens(chess.Board()).unsqueeze(0)

    def throughput(model, nthreads):
        counts = [0] * nthreads
        stop = threading.Event()

        def work(i):
            with torch.no_grad():
                while not stop.is_set():
                    model(x)
                    counts[i] += 1
        with torch.no_grad():
            for _ in range(8):
                model(x)
        threads = [threading.Thread(target=work, args=(i,)) for i in range(nthreads)]
        t0 = time.perf_counter()
        for th in threads:
            th.start()
        time.sleep(seconds)
        stop.set()
        for th in threads:
            th.join()
        return sum(counts) / (time.perf_counter() - t0)

    print(f"=== forward throughput, {seconds}s per cell, arms interleaved ===\n")
    print(f"{'threads':>8} {'eager pos/s':>12} {'traced pos/s':>13} {'speedup':>8}")
    base_e = base_t = None
    for n in (1, 2, 4, 8):
        e = throughput(eager, n)
        t = throughput(traced, n)
        base_e = base_e or e
        base_t = base_t or t
        print(f"{n:>8} {e:>12.1f} {t:>13.1f} {t / e:>7.2f}x")
    print("\nTokenizer, same machine:")
    board = chess.Board(POSITIONS[1])
    for label, fn in (("mctsv4 (36 dispatches)", mctsv4.board_to_tokens),
                      ("mctsv5 (1 dispatch)", mctsv5.board_to_tokens)):
        n = 2000
        t0 = time.perf_counter()
        for _ in range(n):
            fn(board)
        print(f"  {label:<24}{(time.perf_counter() - t0) / n * 1e6:>8.1f} us/call")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if not any((args.verify, args.breakdown, args.search, args.forward, args.all)):
        ap.print_help()
        return
    if args.verify or args.all:
        bench_verify()
        print()
    if args.breakdown or args.all:
        bench_breakdown()
        print()
    if args.forward or args.all:
        bench_forward()
        print()
    if args.search or args.all:
        bench_search(sims=args.sims, repeats=args.repeats)


if __name__ == "__main__":
    main()
