#!/usr/bin/env python
"""C12 Step 1 — the workload Nsight profiles, and nothing else.

    # Nsight Systems: a real search, timeline + CUDA trace
    nsys profile -o runs/c12/nsys_fresh --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
        python tools/profile_c12.py search --regime fresh --sims 20000

    # Nsight Compute: the graphed forward, isolated
    ncu --set full --graph-profiling node -o runs/c12/ncu_shape24 \
        python tools/profile_c12.py forward --shape 24 --iters 12

WHY THIS IS A SEPARATE TOOL AND NOT A FLAG ON bench_c10b.py
===========================================================
Two reasons, and both are about not contaminating the thing being measured.

`bench_c10b.py` builds and tears down a few hundred searches per invocation and
publishes medians over repeats. Under nsys that produces a trace dominated by
capture, warmup and allocator churn, and the section that matters — one search,
steady state — is a few percent of it. This tool runs ONE search, with the
capture and the warmup outside the NVTX region nsys is told to report on.

Second, the callback is where the GIL-held window lives (BENCH.md C10b-4b), and
NVTX pushes inside it are not free. They are therefore OFF by default and behind
`--nvtx-batches`; the default run instruments only the search boundary, and the
per-batch structure is recovered from the CUDA trace's own kernel timestamps,
which cost the callback nothing. `--nvtx-batches` exists so the two can be
compared and the instrumentation's own cost stated rather than assumed small.

WHAT THE MODES ARE FOR
======================
  search   The Gate 4 workload: W=1/K=24, max_batch 128, the shipping
           configuration from BENCH.md C10b-3g. `--regime reuse` plays six
           plies of the engine's own moves first, which is the regime a real
           game spends most of its moves in and where 92% of leaves never reach
           the network (C10h). GPU busy fraction is read off the CUDA trace
           between the NVTX region's endpoints.

  forward  The graphed forward alone, replayed `--iters` times at one captured
           shape, with the H2D and D2H copies the production callback does. No
           search, no C++ threads, no cache — so ncu attributes every kernel to
           the forward rather than to whatever the dispatcher was doing. Shape
           24 is what the fresh-root regime actually runs (21.2 rows/crossing,
           padded to 24); shape 128 is the knee.

Writes nothing to golden/ (Global Rule 2) and nothing to BENCH.md; it prints a
JSON summary that the reporting step reads.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import statistics
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

# The same two positions bench_c10b.py measures Gate 4 on, restated by import
# rather than by copy so the profile and the throughput table cannot drift onto
# different boards.
from tools.bench_c10b import (  # noqa: E402
    ARENA_CAPACITY, CACHE_SLOTS, ENDGAME_FEN, FRESH_ROOT_FEN,
)


def log(msg: str = "") -> None:
    print(msg, file=sys.stderr, flush=True)


class Nvtx:
    """torch's NVTX, or a no-op when it is unavailable or switched off.

    A class rather than two module-level functions so the enabled/disabled
    choice is made once, at construction, instead of by a branch on every push
    inside the callback.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        if enabled:
            import torch

            self._push = torch.cuda.nvtx.range_push
            self._pop = torch.cuda.nvtx.range_pop

    def push(self, name: str) -> None:
        if self.enabled:
            self._push(name)

    def pop(self) -> None:
        if self.enabled:
            self._pop()


def build_evaluator(args):
    from playing.v6 import evaluator as evaluator_module

    started = time.perf_counter()
    # `load_model` reports what it detected through a `log` callable that defaults
    # to `print`, i.e. to stdout — and stdout here is the JSON summary. The same
    # collision `playv6.Engine.ensure_ready` solves, solved the same way, because
    # a tool whose machine-readable output has "Loading ..." in front of it is a
    # tool whose output cannot be piped.
    with contextlib.redirect_stdout(sys.stderr):
        evaluator = evaluator_module.build(args.max_batch, model_path=args.model,
                                           graphs=not args.no_graphs)
    seconds = time.perf_counter() - started
    log(f"[setup] evaluator ready in {seconds:.2f} s; "
        f"{evaluator.graph_report.describe() if evaluator.graph_report else 'eager'}; "
        f"pinned={evaluator.pinned}")
    return evaluator


def instrument_callback(evaluator, nvtx: Nvtx):
    """Wrap `graph.run` in three NVTX ranges: stage, replay, and the copies.

    Wraps the GRAPH rather than the evaluator's `_evaluate`, because `_evaluate`
    is what C++ holds a bound reference to — replacing it after construction
    would leave the C++ side calling the original. The graph object is consulted
    through `self.graph` on every batch, so swapping its `run` is picked up.
    """
    if not nvtx.enabled or evaluator.graph is None:
        return
    graph = evaluator.graph
    original = graph.run

    def instrumented(count, source, poison=False):
        nvtx.push(f"batch[{count}->{graph.pad_to(count)}]")
        try:
            nvtx.push("stage")
            padded = graph.stage(count, source)
            nvtx.pop()
            nvtx.push("replay")
            graph.replay(padded)
            nvtx.pop()
            if poison:
                graph.poison_pad(count, padded)
            graph.rows += count
            graph.padded_rows += padded
            graph.by_size[padded] += 1
            return graph.outputs(count, padded)
        finally:
            nvtx.pop()

    graph.run = instrumented
    del original


def make_search(evaluator, args):
    config = guofish_core.SearchConfig(virtual_loss=args.virtual_loss,
                                       arena_capacity=args.arena_capacity)
    config.cache_slots = CACHE_SLOTS
    # verify_compaction OFF: the C12 brief requires the Gate 4 measurement made
    # without it (C8 measured it at +8.1 ms per apply_move at 184k nodes), and a
    # profile of a configuration nobody ships is a profile of nothing.
    config.verify_compaction = False
    search = guofish_core.ReplaySearchQ32(config)
    search.set_evaluator(evaluator.core)
    return search


def mode_search(args) -> dict:
    """One search, profiled. Setup and warmup are outside the NVTX region."""
    nvtx = Nvtx(args.nvtx)
    evaluator = build_evaluator(args)
    instrument_callback(evaluator, Nvtx(args.nvtx_batches))

    parallel = guofish_core.ParallelConfig(
        workers=args.workers, in_flight=args.in_flight, max_batch=args.max_batch,
        affinity=args.affinity, collect_histograms=True)

    fen = FRESH_ROOT_FEN if args.regime == "fresh" else ENDGAME_FEN
    plies = 0 if args.regime == "fresh" else args.reuse_plies

    search = make_search(evaluator, args)
    search.set_position(fen)

    # The reuse regime is produced the way a game produces it — search, play the
    # engine's own move, search again — and every ply of it is OUTSIDE the
    # measured region. It is the state the measured search starts from, not part
    # of what is being measured.
    nvtx.push("setup:reuse-plies")
    for ply in range(plies):
        stats = search.search_parallel(args.sims, parallel)
        search.apply_move(stats["best_move"])
        log(f"[setup] reuse ply {ply + 1}/{plies}: {stats['best_move']}")
    nvtx.pop()

    # A warmup search on a THROWAWAY tree, so the profiled search is not the one
    # paying for first-touch on the arena's pages or for a cold cache. Discarded
    # by rebuilding the search below rather than by hoping it left no trace.
    if args.warmup:
        nvtx.push("setup:warmup")
        warm = make_search(evaluator, args)
        warm.set_position(fen)
        warm.search_parallel(min(args.sims, 2000), parallel)
        warm.set_evaluator(None)
        del warm
        nvtx.pop()

    evaluator.graph_counters_reset()
    nvtx.push(f"SEARCH:{args.regime}")
    started = time.perf_counter()
    search.search_parallel(args.sims, parallel)
    wall = time.perf_counter() - started
    nvtx.pop()

    par = search.parallel_stats()
    evals = search.eval_stats()
    audit = search.audit()
    assert par["delivered"] == par["requested"], (
        f"delivered {par['delivered']} != requested {par['requested']}; a short "
        f"search is not a profile of the search that was asked for")
    assert audit["vloss_total"] == 0, "virtual loss stranded"
    assert audit["conservation_failures"] == 0, "visits do not conserve"

    summary = {
        "mode": "search",
        "regime": args.regime,
        "workers": args.workers,
        "in_flight": args.in_flight,
        "max_batch": args.max_batch,
        "sims_requested": args.sims,
        "sims_delivered": par["delivered"],
        "wall_s": wall,
        "sims_per_s": par["delivered"] / wall,
        "crossings": evals["batches"],
        "rows": evals["rows"],
        "rows_per_crossing": evals["mean_rows"],
        "cache_skipped": evals["cache_skipped"],
        "cache_rate": evals["cache_skipped"] / max(1, evals["cache_skipped"] + evals["rows"]),
        "pad_ratio": evaluator.graph_pad_ratio(),
        "call_ns": evals["call_ns"],
        "gpu_share": evals["call_ns"] / max(1, par["wall_ns"]),
        "nvtx_batches": args.nvtx_batches,
        "shape_histogram": (dict(evaluator.graph.by_size) if evaluator.graph else {}),
    }
    search.set_evaluator(None)
    del search
    evaluator.close()
    return summary


def mode_forward(args) -> dict:
    """`--iters` graph replays at one captured shape, plus the production copies.

    This is the production callback's body with the search removed: the same H2D
    of pinned int32 rows, the same replay, the same two D2H copies. Nothing else
    runs, so every kernel ncu reports belongs to the forward.
    """
    import numpy as np
    import torch

    evaluator = build_evaluator(args)
    if evaluator.graph is None:
        raise SystemExit("forward mode profiles the CAPTURED graph; --no-graphs "
                         "leaves nothing to profile")
    shape = args.shape
    if shape not in evaluator.graph.sizes:
        raise SystemExit(f"shape {shape} is not captured; sizes are "
                         f"{list(evaluator.graph.sizes)}")

    # Real board tokens, not zeros: the embedding gather's address pattern is an
    # input-dependent property and a block of identical rows would flatter it.
    evaluator._input_np[:shape] = guofish_core.tokens(FRESH_ROOT_FEN)

    nvtx = Nvtx(args.nvtx)
    durations = []
    # Untimed, unprofiled warmup is impossible under ncu — it profiles every
    # launch — so the iteration index is recorded instead and the reporting step
    # drops the first. Under nsys the warmup is real.
    for index in range(args.iters):
        if index == 0:
            nvtx.push("warmup")
        elif index == 1:
            nvtx.pop()
            nvtx.push(f"FORWARD:shape{shape}")
        torch.cuda.synchronize()
        started = time.perf_counter()
        evaluator._evaluate(shape)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    if args.iters > 1:
        nvtx.pop()

    steady = durations[1:] or durations
    summary = {
        "mode": "forward",
        "shape": shape,
        "iters": args.iters,
        "median_ms": 1000 * statistics.median(steady),
        "min_ms": 1000 * min(steady),
        "max_ms": 1000 * max(steady),
        "pos_per_s": shape / statistics.median(steady),
        "policy_checksum": float(np.asarray(evaluator._value_np[:shape]).sum()),
    }
    evaluator.close()
    return summary


MODES = {"search": mode_search, "forward": mode_forward}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--regime", choices=("fresh", "reuse"), default="fresh")
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--reuse-plies", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--in-flight", type=int, default=24)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--affinity", default="none")
    parser.add_argument("--virtual-loss", type=float, default=2.5)
    parser.add_argument("--arena-capacity", type=int, default=ARENA_CAPACITY)
    parser.add_argument("--shape", type=int, default=24, help="forward mode only")
    parser.add_argument("--iters", type=int, default=12, help="forward mode only")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", default=True)
    parser.add_argument("--nvtx", action="store_true", default=True,
                        help="region markers (cheap, outside the callback)")
    parser.add_argument("--no-nvtx", dest="nvtx", action="store_false")
    parser.add_argument("--nvtx-batches", action="store_true",
                        help="per-batch markers INSIDE the GIL-held callback; costs "
                             "what it costs, and the point is to measure that")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    info = guofish_core.build_info()
    if info["asan"] or info["ubsan"] or info["asserts"]:
        raise SystemExit(f"refusing to profile an instrumented build: {info}. "
                         f"C10's incident — a set of instrumented numbers published "
                         f"as if they were Release — is why this is a refusal.")

    summary = MODES[args.mode](args)
    summary["build"] = info
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
