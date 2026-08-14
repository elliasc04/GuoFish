#!/usr/bin/env python
"""C12b — Gate 4 re-run on the Inductor forward, in BOTH regimes, reported apart.

    python tools/bench_c12b.py --sections gate4 --markdown
    python tools/bench_c12b.py --sections forward --markdown
    python tools/bench_c12b.py --markdown --json-out runs/c12b/bench_c12b.json

WHY THIS IS A NEW FILE AND NOT A SECTION IN bench_c12.py
========================================================
The same reason bench_c12.py is not a section in bench_c10b.py: editing the tool
that produced a published table destroys the ability to reproduce that table and
see the difference. `tools/bench_c12.py` produced C12-5's 15,342 / 334,315 and is
left exactly as it was; this file measures the same two rows on the changed
forward and imports bench_c12's own `timed_search` so that "the same row" means
the same code and not a re-spelling of it.

**1.19x IS A FRESH-ROOT NUMBER AND MUST NOT BE QUOTED AS A GAME-WIDE ONE.** That
is the brief's Part 4 and it is why every table below has a regime column and no
total. In the reuse-heavy regime the GPU is 26.3% busy and one crossing of two
rows serves the whole search (C12-1), so Inductor buys close to nothing there;
the game-weighted gain is somewhere between 1.10x and 1.19x depending on the mix,
which is roughly +14 to +23 ELO at 91.8 ELO/doubling — and that curve was
measured at 800->2000 simulations, so the true figure at 15k is likely at the low
end. Aggregating the two regimes into one headline number would erase exactly the
distinction that makes the estimate honest, so this tool does not compute one.

  gate4    Delivered sims/s, fresh midgame root and reuse-heavy endgame, with
           `compile=False` and `compile=True` measured in the SAME process and
           interleaved, because the run-to-run spread is ~4% and drifts with GPU
           clock — two blocks of N would compare two machine states rather than
           two forwards. `delivered > 0` is asserted before any row is admitted
           (bench_c12's `timed_search` does it), as is no recompilation.

  forward  The graphed forward's DEVICE time per captured shape, by CUDA events,
           with the fusion term beside it: how many bf16 policy words Inductor
           moved and by how much. Device time is where the change actually is;
           the sims/s above is that change after the dispatcher, the C++ descent
           and the padding have all diluted it.

Writes nothing to golden/ and nothing to baseline/. Refuses to publish from an
instrumented build, for the reason recorded in DECISIONS.md, C10.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

from tools.bench_c10b import (  # noqa: E402
    CACHE_SLOTS, ENDGAME_FEN, FRESH_ROOT_FEN, GATE4_FLOOR,
    GATE4_STRETCH, PYTHON_BASELINE,
)
from tools.bench_c12 import (  # noqa: E402
    make_search, parallel_config, reused_search, timed_search,
)

# C12-5's published figures, so every row below carries its own comparison rather
# than leaving the reader to find them. Both are delivered sims/s at W=1/K=24,
# `max_batch` 128, `verify_compaction` off.
#
# THEY WERE TAKEN ON THE 20M-CORPUS CHECKPOINT AND THIS FILE RUNS THE 90M ONE, so
# the `vs C12-5` column is indicative rather than paired. It is still worth
# printing: the two nets are the same v5 architecture down to the parameter count
# (10,887,681), the same 68 tokens and the same captured ladder, so a large gap in
# that column would mean something changed that is not the weights. The number to
# quote is the PAIRED speedup, which is measured here with both arms on the same
# checkpoint in the same process.
C12_FRESH_ROOT = 15_342
C12_REUSE_HEAVY = 334_315

ARMS = (("eager", False), ("inductor", True))

# C11c's sizing constant, restated from `playv6.EngineConfig.arena_nodes` rather
# than imported so this harness does not drag the whole UCI surface in. The
# engine's rule is `60 x (sims_per_move + ponder_max_sims)`; 60 comes from C8's
# measured 38.9 and 39.7 nodes per simulation, fitted at 40 and multiplied by 1.5
# for branching variance.
NODES_PER_SIM = 60


def resolve_arena(args) -> int:
    """C11c's nodes-per-simulation rule, applied to what this harness actually runs.

    **THE RULE IS A FUNCTION OF SIMULATIONS AND IS MODEL-INDEPENDENT. THE OLD
    CONSTANT WAS NEITHER, AND THAT IS THE BUG THIS REPLACES.** `bench_c10b`'s
    `ARENA_CAPACITY = 1,200,000` predates C11c. It coincides with `60 x 20,000`,
    i.e. the rule's answer for ONE search of the default budget — but the
    reuse-heavy arm runs `reuse_plies + 1` searches of `sims` NEW simulations each
    on a single accumulating tree, so one move's allowance was never the right
    budget for it. Applying the rule to the workload gives
    `60 x sims x (reuse_plies + 1)`.

    IT FIT BEFORE FOR A REASON THAT WAS NEVER A GUARANTEE. Measured over the six
    reuse plies at 20,000 NEW simulations each, on the same positions and the same
    budget, the two checkpoints' trees behave completely differently:

        20M   261,111 -> 390,075 -> 415,635 -> 203,807 -> 156,615 -> 156,126
        90M   308,188 -> 530,925 -> 646,246 -> 654,081 -> 864,057 -> 1,104,746

    The 20M net's reuse tree **converges** — it peaks at ply 3 and shrinks as the
    endgame simplifies — so it stayed under 1.2M by luck of shape. The 90M net's
    grows every ply and reaches 1,436,625 by the measured search, so 1.2M exhausts
    and the search delivers 6,666 of 20,000. C11c makes a short delivery
    legitimate on exhaustion and `timed_search` correctly refuses to publish the
    row; it surfaces as `delivered != requested` only because that assertion comes
    first.

    **This is not evidence that the engine's own sizing is model-sensitive, and
    the same data says so.** Per SEARCH — which is what `arena_nodes` covers — the
    90M net's first 20,000-simulation search builds 308,188 nodes, i.e. **15.4
    nodes per simulation against the 60 the rule assumes**. The rule has roughly
    4x headroom on this checkpoint. What accumulates here is an artifact of a
    benchmark that deliberately never lets go of the tree, not of a move the
    engine plays.
    """
    if args.arena_capacity is not None:
        return int(args.arena_capacity)
    return NODES_PER_SIM * args.sims * (args.reuse_plies + 1)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def build_evaluator(args, compiled: bool):
    from playing.v6 import evaluator as evaluator_module

    # `args.model` defaults to SHIPPING_MODEL (the 90M-corpus net the engine
    # loads), NOT to `evaluator.DEFAULT_MODEL` (the 20M net C10's goldens are
    # anchored to). Throughput is optimised for the checkpoint that will be tuned
    # and played. Same v5 architecture either way, so the ladder is unchanged.
    model, device = evaluator_module.load_default_model(args.model, None)
    return evaluator_module.TorchEvaluator(model, device, args.max_batch, graphs=True,
                                           compile=compiled)


# --- sections ---------------------------------------------------------------


def section_gate4(args, payload, sections):
    """Both regimes, both arms, interleaved repeat by repeat."""
    evaluators = {name: build_evaluator(args, compiled) for name, compiled in ARMS}
    parallel = parallel_config(args)
    samples = {(name, regime): []
               for name, _ in ARMS for regime in ("fresh", "reuse")}

    for repeat in range(args.repeats):
        # INTERLEAVED, not two blocks. The run-to-run spread is ~4% and drifts
        # with GPU clock, so a block of N eager followed by a block of N inductor
        # would be comparing two machine states as much as two forwards. C12-6
        # made the same choice for the same reason and reported 15/15 paired wins.
        for name, _ in ARMS:
            evaluator = evaluators[name]

            search = make_search(evaluator, args)
            search.set_position(FRESH_ROOT_FEN)
            samples[(name, "fresh")].append(
                timed_search(search, evaluator, args.sims, parallel, absolute=False))
            evaluator.assert_no_recompilation("during the fresh-root Gate 4 row")
            search.set_evaluator(None)
            del search

            search = reused_search(evaluator, args, parallel, absolute=False)
            samples[(name, "reuse")].append(
                timed_search(search, evaluator, args.sims, parallel, absolute=False))
            evaluator.assert_no_recompilation("during the reuse-heavy Gate 4 row")
            search.set_evaluator(None)
            del search
        log(f"  repeat {repeat + 1}/{args.repeats} done")

    rows = []
    for regime, label, c12 in (("fresh", "fresh midgame root", C12_FRESH_ROOT),
                               ("reuse", "reuse-heavy endgame", C12_REUSE_HEAVY)):
        entry = {"regime": label, "c12_baseline": c12, "arms": {}}
        for name, _ in ARMS:
            taken = samples[(name, regime)]
            rates = [s["sims_per_s"] for s in taken]
            entry["arms"][name] = {
                "sims_per_s": statistics.median(rates),
                "min": min(rates), "max": max(rates),
                "delivered": int(statistics.median(s["delivered"] for s in taken)),
                "inherited": int(statistics.median(s["inherited"] for s in taken)),
                "wall_ms": 1000 * statistics.median(s["wall_s"] for s in taken),
                "rows_per_crossing": statistics.fmean(s["rows_per_crossing"] for s in taken),
                "pad_ratio": statistics.fmean(s["pad_ratio"] for s in taken),
                "cache_rate": statistics.fmean(s["cache_rate"] for s in taken),
                "gpu_share_call": statistics.fmean(s["gpu_share_call"] for s in taken),
            }
        # PAIRED, repeat by repeat, because that is what the interleaving buys:
        # each ratio compares two arms measured seconds apart on the same machine
        # state, so the ~4% clock drift divides out instead of adding.
        pairs = [b["sims_per_s"] / a["sims_per_s"]
                 for a, b in zip(samples[("eager", regime)],
                                 samples[("inductor", regime)])]
        entry["speedup_median"] = statistics.median(pairs)
        entry["speedup_min"] = min(pairs)
        entry["speedup_max"] = max(pairs)
        entry["inductor_wins"] = sum(1 for p in pairs if p > 1.0)
        entry["pairs"] = len(pairs)
        rows.append(entry)
        log(f"  {label:<24} eager {entry['arms']['eager']['sims_per_s']:>9,.0f} -> "
            f"inductor {entry['arms']['inductor']['sims_per_s']:>9,.0f} sims/s   "
            f"{entry['speedup_median']:.3f}x  "
            f"({entry['inductor_wins']}/{entry['pairs']} paired wins)")

    sections += [
        "### C12b-4 — Gate 4 on the Inductor forward, both regimes, REPORTED SEPARATELY", "",
        f"W={args.workers}, K={args.in_flight}, `max_batch` {args.max_batch}, virtual "
        f"loss {args.virtual_loss}, `verify_compaction` **off**, {args.repeats} repeats "
        f"(median), arms interleaved repeat by repeat. No opening book and no tablebase "
        f"are attached on this path, so no move is bypassed and there is nothing to "
        f"exclude from the rate. `delivered > 0` is asserted before a row is admitted, "
        f"and so is no recompilation.", "",
        "**There is deliberately no combined number.** The two regimes are different "
        "machines: on a fresh root the GPU is the bottleneck and Inductor moves it; "
        "under deep reuse the GPU is ~26% busy and one crossing serves the whole "
        "search, so there is almost nothing for a faster forward to buy. A "
        "game-weighted figure depends on the mix and belongs to Gate 5.", "",
        "| regime | arm | inherited | delivered | wall | **delivered sims/s** | min | max "
        "| rows/crossing | pad waste | cache hit | vs C12-5 | paired speedup |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for entry in rows:
        for name, _ in ARMS:
            arm = entry["arms"][name]
            speed = (f"**{entry['speedup_median']:.3f}x** "
                     f"({entry['speedup_min']:.3f}–{entry['speedup_max']:.3f})"
                     if name == "inductor" else "—")
            sections.append(
                f"| {entry['regime']} | `compile={name == 'inductor'}` "
                f"| {arm['inherited']:,} | {arm['delivered']:,} "
                f"| {arm['wall_ms']:,.1f} ms | **{arm['sims_per_s']:,.0f}** "
                f"| {arm['min']:,.0f} | {arm['max']:,.0f} "
                f"| {arm['rows_per_crossing']:.1f} | {arm['pad_ratio']:.2f}x "
                f"| {100 * arm['cache_rate']:.1f}% "
                f"| {arm['sims_per_s'] / entry['c12_baseline']:.3f}x | {speed} |")
    sections += [
        "",
        f"Gate 4's floor is {GATE4_FLOOR:,} delivered sims/s and its stretch target "
        f"{GATE4_STRETCH:,}; the Python reference measured {PYTHON_BASELINE:,}.", ""]
    payload["gate4"] = rows
    for evaluator in evaluators.values():
        evaluator.close()
    return rows


def section_forward(args, payload, sections):
    """Device time per captured shape, and how far the fusion moved the logits."""
    import torch

    evaluators = {name: build_evaluator(args, compiled) for name, compiled in ARMS}
    shapes = evaluators["eager"].graph_sizes
    assert evaluators["inductor"].graph_sizes == shapes

    # A REAL POSITION IN EVERY ROW, not the pad. `GraphedForward.tokens` is
    # initialised to the start position and left there unless a batch stages
    # something, so timing and differencing against it would measure one
    # position's forward repeated `shape` times — which is fine for device time
    # and misleading for the fusion term, since a 4096-wide row that is identical
    # across rows understates how many distinct words the fusion moves.
    row_tokens = torch.from_numpy(guofish_core.tokens(FRESH_ROOT_FEN))
    for evaluator in evaluators.values():
        evaluator.graph.tokens[:] = row_tokens.to(evaluator.graph.tokens.device)

    rows = []
    for shape in shapes:
        timings = {}
        for name, _ in ARMS:
            graph = evaluators[name].graph
            for _ in range(args.warmup_replays):
                graph.replay(shape)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(args.timed_replays):
                graph.replay(shape)
            end.record()
            torch.cuda.synchronize()
            timings[name] = start.elapsed_time(end) / args.timed_replays * 1000

        # The fusion term, from inside one process: same rows, same shape, same
        # autocast context, Inductor against unfused ATen.
        graph = evaluators["inductor"].graph
        fused, _ = graph.eager(shape)
        unfused, _ = graph.eager_unfused(shape)
        torch.cuda.synchronize()
        words = int((fused.view(torch.uint16) != unfused.view(torch.uint16)).sum())
        max_delta = float((fused.float() - unfused.float()).abs().max())

        row = {"shape": shape, "eager_us": timings["eager"],
               "inductor_us": timings["inductor"],
               "speedup": timings["eager"] / timings["inductor"],
               "differing_words": words, "total_words": int(fused.numel()),
               "max_abs_dlogit": max_delta}
        rows.append(row)
        log(f"  shape {shape:>4}: eager {row['eager_us']:8.1f} us   "
            f"inductor {row['inductor_us']:8.1f} us   {row['speedup']:.3f}x   "
            f"{words:,}/{row['total_words']:,} words differ")

    sections += [
        "### C12b-3 — The graphed forward's device time, and the fusion term", "",
        f"CUDA events, {args.timed_replays} timed replays after {args.warmup_replays} "
        f"warm ones, `max_batch` {args.max_batch}. Device time is where the change "
        f"actually is; C12b-4's sims/s is this after the dispatcher, the C++ descent "
        f"and the padding have diluted it.", "",
        # ASCII ONLY in anything `log()` prints: stdout is cp1252 on this box and a
        # stray micro sign or delta kills the run at the last line, after every
        # measurement has been taken and before any of it is written out.
        "| shape | eager us | inductor us | speedup | policy words differing "
        "| max abs dlogit |",
        "|--:|---:|---:|---:|---:|---:|"]
    for row in rows:
        marker = "**" if row["shape"] == args.in_flight else ""
        sections.append(
            f"| {marker}{row['shape']}{marker} | {row['eager_us']:,.1f} "
            f"| {row['inductor_us']:,.1f} | {marker}{row['speedup']:.3f}x{marker} "
            f"| {row['differing_words']:,} of {row['total_words']:,} "
            f"({100 * row['differing_words'] / row['total_words']:.1f}%) "
            f"| {row['max_abs_dlogit']:.4g} |")
    sections.append("")
    payload["forward"] = rows
    for evaluator in evaluators.values():
        evaluator.close()
    return rows


def section_capture(args, payload, sections):
    """What adopting Inductor costs at engine start, and in device memory."""
    entries = {}
    for name, compiled in ARMS:
        started = time.perf_counter()
        evaluator = build_evaluator(args, compiled)
        wall = time.perf_counter() - started
        report = evaluator.graph_report
        entries[name] = {
            "method": report.method,
            "sizes": list(report.sizes),
            "capture_s": report.seconds,
            "construct_s": wall,
            "reserved_delta_mib": report.reserved_delta / 2 ** 20,
            "warmup_rounds": dict(getattr(evaluator.graph, "warmup_rounds", {})),
        }
        log(f"  {name:9s} {report.describe()}  (construct {wall:.1f} s)")
        evaluator.close()

    sections += [
        "### C12b-5 — What adoption costs at engine start", "",
        "`ensure_ready` pays this once per process. Inductor's compile is cached "
        "under `TORCHINDUCTOR_CACHE_DIR`, so a cold cache costs substantially more "
        "than the warm figure below — the first run on a new machine, a new GPU or a "
        "new torch is the expensive one.", "",
        "| arm | method | capture | construct | device memory | warmup rounds per shape |",
        "|---|---|---:|---:|---:|---|"]
    for name, _ in ARMS:
        entry = entries[name]
        rounds = (", ".join(f"{k}:{v}" for k, v in sorted(entry["warmup_rounds"].items()))
                  or "—")
        sections.append(
            f"| `compile={name == 'inductor'}` | `{entry['method']}` "
            f"| {entry['capture_s']:.1f} s | {entry['construct_s']:.1f} s "
            f"| +{entry['reserved_delta_mib']:.0f} MiB | {rounds} |")
    sections.append("")
    payload["capture"] = entries
    return entries


SECTIONS = {"gate4": section_gate4, "forward": section_forward,
            "capture": section_capture}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", nargs="+", default=sorted(SECTIONS),
                        choices=sorted(SECTIONS))
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reuse-plies", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--in-flight", type=int, default=24)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--affinity", default="none")
    parser.add_argument("--virtual-loss", type=float, default=2.5)
    # DERIVED FROM C11c's RULE, NOT HARDCODED — see `_resolve_arena` below.
    parser.add_argument("--arena-capacity", type=int, default=None,
                        help="nodes per arena; default is C11c's 60 nodes/sim "
                             "applied to this harness's accumulated workload, "
                             "i.e. 60 x sims x (reuse_plies + 1)")
    parser.add_argument("--warmup-replays", type=int, default=20)
    parser.add_argument("--timed-replays", type=int, default=200)
    parser.add_argument("--model", type=Path, default=None,
                        help="defaults to evaluator.SHIPPING_MODEL, the 90M-corpus "
                             "checkpoint the engine loads")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--allow-instrumented", action="store_true")
    args = parser.parse_args()

    from playing.v6.evaluator import SHIPPING_MODEL

    if args.model is None:
        args.model = SHIPPING_MODEL
    # Resolved once and stamped back onto `args`, because `make_search` and
    # `reused_search` are bench_c12's and read `args.arena_capacity` directly.
    args.arena_capacity = resolve_arena(args)

    info = guofish_core.build_info()
    log(f"build: {info}")
    log(f"model: {args.model}")
    log(f"arena: {args.arena_capacity:,} nodes "
        f"({NODES_PER_SIM} nodes/sim x {args.sims:,} sims x "
        f"{args.reuse_plies + 1} searches on the reuse tree)")
    log(f"cache_slots {CACHE_SLOTS:,}, endgame {ENDGAME_FEN}")
    if (info["asan"] or info["ubsan"] or info["asserts"]) and not args.allow_instrumented:
        raise SystemExit("refusing to publish from an instrumented build; see "
                         "DECISIONS.md, C10. Pass --allow-instrumented to override.")

    payload, sections = {}, []
    for name in args.sections:
        log(f"\n== {name} ==")
        SECTIONS[name](args, payload, sections)

    if args.markdown:
        log("\n" + "=" * 72 + "\n")
        log("\n".join(sections))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
