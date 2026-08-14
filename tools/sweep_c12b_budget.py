#!/usr/bin/env python
"""C12b — does Gate 2''s move agreement depend on the simulation budget?

    python tools/sweep_c12b_budget.py --budgets 1600 6400 --markdown
    python tools/sweep_c12b_budget.py --budgets 6400 --json-out runs/c12b/budget.json

WHY THIS EXISTS
===============
Gate 2' at Gate 2b's 1,600-simulation budget FAILS: 514/520 = 98.85% against a
>= 99% criterion, and four of the six disagreements are at top-two visit margins
of 9.5% to 38.8% rather than the under-2% the criterion calls a near-tie. That is
reported rather than worked around (Global Rule 10), and this tool is the
measurement that says what it means.

**THE NEAR-TIE CRITERION WAS IMPORTED FROM A GATE WHERE IT MEASURED SOMETHING
ELSE.** Gate 2b compared two engines that agreed to 1e-6, so the only mechanism
available for a disagreement was a genuine coin flip and a small top-two margin
was the signature of one. Under Inductor's ~1e-3 prior shift the mechanism is
different: an early selection flips, the two trees diverge structurally from
there, and each arm then converges CONFIDENTLY on its own answer. A visit margin
measured at the end of that says how concentrated one arm's tree became — not how
close the decision was.

The diagnostic that makes this concrete, and the reason this tool sweeps a
budget rather than reporting one: at 1,600 simulations the EAGER arm itself has
not settled on three of the four decisive positions. It answers e4d5 at 1,600 and
e3b6 from 3,200 up; e5g4 at 1,600 and h7h6 from 3,200 up; e8g8 at 1,600 and c7c5
by 12,800. A budget at which the reference arm is still changing its own mind is
not a budget at which a differential measures agreement between two engines.

So the honest question is whether the disagreements are a property of the fusion
or of the budget, and it is answerable: run both arms over the whole corpus at
more than one budget and see which disagreements survive. The engine ships at
roughly 15,000 simulations per move, so the converged end of this sweep is much
closer to how it plays than 1,600 is.

WHAT IT DOES NOT DO
===================
It does not move Gate 2''s criteria. >= 99% agreement with every disagreement
under a 2% margin is what the brief sets and what `tests/test_c12b_gate2prime.py`
asserts, at the budget the frozen baseline was generated with. This tool reports;
the gate gates.

Both arms are driven from one process with one evaluator each, position by
position, so neither arm's answer can depend on which order the two were run in.
Writes nothing to golden/ and nothing to baseline/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

BASELINE_SEARCH = REPO_ROOT / "baseline" / "c12b_eager_search.json"

MIN_AGREEMENT = 0.99
MAX_DISAGREEMENT_MARGIN = 0.02


def log(msg: str = "") -> None:
    print(msg, flush=True)


def make_arm(compiled: bool, recorded: dict, cache_slots: int):
    from playing.v6 import evaluator as live_evaluator

    config = guofish_core.SearchConfig()
    config.c_init = recorded["c_init"]
    config.c_base = recorded["c_base"]
    config.fpu_root = recorded["fpu_root"]
    config.fpu_tree = recorded["fpu_tree"]
    config.virtual_loss = recorded["virtual_loss"]
    config.max_tree_depth = recorded["max_tree_depth"]
    config.cache_slots = cache_slots
    # The 90M-corpus checkpoint the engine ships, named rather than defaulted.
    evaluator = live_evaluator.build(1, model_path=live_evaluator.SHIPPING_MODEL,
                                     compile=compiled)
    search = guofish_core.ReplaySearchDouble(config)
    search.set_evaluator(evaluator.core)
    return evaluator, search


def answer(search, fen: str, sims: int, parallel) -> dict:
    search.set_position(fen)
    stats = search.search_parallel(sims, parallel)
    arrays = search.dump_tree_arrays(0)
    visits = {guofish_core.move_to_uci(int(move)): int(count)
              for depth, move, count in zip(arrays["depth"], arrays["move"],
                                            arrays["visits"]) if depth == 1}
    counts = sorted(visits.values(), reverse=True)
    total = sum(counts)
    margin = (1.0 if len(counts) == 1 else
              (counts[0] - counts[1]) / total if total else 0.0)
    return {"move": stats["best_move"], "margin": margin, "visits": visits,
            "root_visits": int(stats["root_visits"]),
            "mating_move": stats["mating_move"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1600, 6400])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    baseline = json.loads(BASELINE_SEARCH.read_text(encoding="utf-8"))
    recorded = baseline["search_config"]
    fens = [r["fen"] for r in baseline["records"]]
    sources = {r["fen"]: r["source"] for r in baseline["records"]}
    if args.limit:
        fens = fens[:args.limit]

    parallel = guofish_core.ParallelConfig(workers=1, in_flight=1, max_batch=1)
    arms = {name: make_arm(compiled, recorded, baseline["cache_slots"])
            for name, compiled in (("eager", False), ("inductor", True))}

    payload = {"budgets": {}, "positions": len(fens),
               "search_config": recorded, "cache_slots": baseline["cache_slots"]}
    try:
        for sims in args.budgets:
            log(f"\n== {sims} simulations, {len(fens)} positions, both arms ==")
            rows = []
            started = time.perf_counter()
            for index, fen in enumerate(fens):
                eager = answer(arms["eager"][1], fen, sims, parallel)
                inductor = answer(arms["inductor"][1], fen, sims, parallel)
                rows.append({"fen": fen, "source": sources[fen],
                             "eager": eager, "inductor": inductor})
                if index and index % 50 == 0:
                    rate = (index + 1) / (time.perf_counter() - started)
                    log(f"  {index + 1}/{len(fens)}  {rate:.2f} pos/s  "
                        f"eta {(len(fens) - index - 1) / rate:.0f}s")
            elapsed = time.perf_counter() - started

            disagreements = [r for r in rows
                             if r["eager"]["move"] != r["inductor"]["move"]]
            decisive = [r for r in disagreements
                        if r["eager"]["margin"] >= MAX_DISAGREEMENT_MARGIN
                        or r["inductor"]["margin"] >= MAX_DISAGREEMENT_MARGIN]
            rate = (len(rows) - len(disagreements)) / len(rows)
            payload["budgets"][str(sims)] = {
                "seconds": elapsed,
                "positions": len(rows),
                "agreement": rate,
                "disagreements": [
                    {"fen": r["fen"], "source": r["source"],
                     "eager_move": r["eager"]["move"],
                     "eager_margin": r["eager"]["margin"],
                     "inductor_move": r["inductor"]["move"],
                     "inductor_margin": r["inductor"]["margin"],
                     "decisive": r in decisive}
                    for r in disagreements],
                "decisive": len(decisive),
            }
            log(f"  agreement {len(rows) - len(disagreements)}/{len(rows)} = "
                f"{rate:.4%}  ({'PASS' if rate >= MIN_AGREEMENT else 'FAIL'} vs "
                f"{MIN_AGREEMENT:.0%})")
            log(f"  disagreements {len(disagreements)}, of which decisive "
                f"(>= {MAX_DISAGREEMENT_MARGIN:.0%} on one arm) {len(decisive)} "
                f"({'PASS' if not decisive else 'FAIL'})")
            for r in disagreements:
                mark = "DECISIVE" if r in decisive else "near-tie"
                log(f"    [{mark}] {r['fen']}")
                log(f"      eager    {r['eager']['move']:6s} "
                    f"margin {r['eager']['margin']:7.3%}")
                log(f"      inductor {r['inductor']['move']:6s} "
                    f"margin {r['inductor']['margin']:7.3%}")
            log(f"  {elapsed:.0f}s ({len(rows) / elapsed:.2f} pos/s)")
    finally:
        for evaluator, search in arms.values():
            search.set_evaluator(None)
            evaluator.close()

    if args.markdown:
        log("\n" + "=" * 72 + "\n")
        log("| simulations | agreement | >= 99%? | disagreements | of which decisive "
            "| every disagreement a near-tie? |")
        log("|---:|---:|:--|---:|---:|:--|")
        for sims in args.budgets:
            entry = payload["budgets"][str(sims)]
            log(f"| {sims:,} | {entry['agreement']:.4%} "
                f"| {'yes' if entry['agreement'] >= MIN_AGREEMENT else '**NO**'} "
                f"| {len(entry['disagreements'])} | {entry['decisive']} "
                f"| {'yes' if not entry['decisive'] else '**NO**'} |")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
