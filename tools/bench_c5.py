"""Populate the C5 section of BENCH.md — search throughput on the replay evaluator.

NOTE: this is a benchmark, not a golden-data generator. Nothing here writes to
golden/ and nothing here is a reference implementation (Global Rule 2).

Usage:
    python tools/bench_c5.py [--trials N] [--markdown]

What this number is, and what it is not
---------------------------------------
It is the cost of the TRAVERSE LOOP: descent with PUCT selection over ~35
siblings per step, virtual-loss apply and repay, tokenization and `nn_key` at
every leaf, a hash lookup, expansion of ~35 children, and backup to the root.
Everything Gate 1 compares, and nothing else.

It is NOT a Gate 4 projection, and quoting it as one would be dishonest in the
flattering direction. Two things are missing that dominate the real engine:

  * **No network.** The replay evaluator is an `unordered_map` lookup. In
    production a leaf costs a batched GPU forward — measured at 4.86 ms for
    batch 64, i.e. ~13.2k evals/s — which is the actual ceiling. What this
    measures is how much CPU the search spends *around* that.
  * **No cache, no tree reuse, no concurrency.** C7, C8 and C9.

So the useful reading is a headroom check against scope 2.2's claim that CPU
descent capacity is not the constraint: Python's real CPU work was 236 us/sim
single-threaded, and the projection assumed ~10x for C++. This measures whether
that held.

The per-simulation cost is reported against the reference's own 236 us so the
ratio is stated rather than left to the reader, and the ratio is the number that
feeds the C9 worker-count decision.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

# pytest.ini's `pythonpath = .` only applies under pytest, so make the built
# extension importable however this script is invoked.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

import guofish_core  # noqa: E402

DUMP = REPO_ROOT / "golden" / "gate1_dump.npz"
TREES = REPO_ROOT / "golden" / "gate1_trees.npz"
MANIFEST = REPO_ROOT / "golden" / "gate1_manifest.json"

# core.mctsv4's measured single-threaded CPU cost per simulation, excluding the
# forward pass (recon: 479 ms of real CPU work per 2,030 simulations).
PYTHON_US_PER_SIM = 236.0


def load():
    for path in (DUMP, TREES, MANIFEST):
        if not path.exists():
            sys.exit(f"missing golden file: {path}\n"
                     "Run `python tools/gen_gate1_golden.py` first.")
    with np.load(DUMP) as data:
        dump = {k: data[k] for k in data.files}
    with np.load(TREES) as data:
        trees = {k: data[k] for k in data.files}
    with open(MANIFEST, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return dump, trees, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    dump, trees, manifest = load()

    runs = [(i, r) for i, r in enumerate(manifest["runs"]) if not r["full_tree"]]
    capacity = 0
    for index, _ in runs:
        begin = int(trees["run_offset"][index])
        end = int(trees["run_offset"][index + 1])
        capacity = max(capacity, 1 + int(trees["children"][begin:end].sum()))

    print(f"{platform.platform()}")
    print(f"module: {guofish_core.build_info()}")
    print(f"dump: {len(dump['keys'])} entries, {len(dump['moves'])} moves")
    print(f"arena capacity: {capacity} nodes")
    print()

    rows = []
    for virtual_loss in (0.0, 2.5):
        selected = [(i, r) for i, r in runs if r["virtual_loss"] == virtual_loss]
        search = guofish_core.ReplaySearchDouble(
            guofish_core.SearchConfig(virtual_loss=virtual_loss,
                                      arena_capacity=capacity))

        load_start = time.perf_counter()
        search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                         dump["moves"], dump["priors"], dump["values"])
        load_s = time.perf_counter() - load_start

        # One untimed pass so the first run is not paying for cold pages in the
        # arena and the dump's hash table.
        search.set_position(selected[0][1]["fen"])
        search.search(selected[0][1]["sims"])

        per_trial = []
        sims_per_trial = 0
        for _ in range(args.trials):
            elapsed = 0.0
            delivered = 0
            for _index, run in selected:
                search.set_position(run["fen"])
                start = time.perf_counter()
                stats = search.search(run["sims"])
                elapsed += time.perf_counter() - start
                delivered += stats["simulations"]
            per_trial.append(elapsed)
            sims_per_trial = delivered

        median = statistics.median(per_trial)
        sims_per_s = sims_per_trial / median
        us_per_sim = median * 1e6 / sims_per_trial
        rows.append({
            "virtual_loss": virtual_loss,
            "positions": len(selected),
            "sims": sims_per_trial,
            "median_s": median,
            "sims_per_s": sims_per_s,
            "us_per_sim": us_per_sim,
            "speedup": PYTHON_US_PER_SIM / us_per_sim,
            "load_s": load_s,
        })
        print(f"VL {virtual_loss:<4} {len(selected)} positions, "
              f"{sims_per_trial} sims: median {median:.3f} s -> "
              f"{sims_per_s:,.0f} sims/s ({us_per_sim:.2f} us/sim, "
              f"{PYTHON_US_PER_SIM / us_per_sim:.0f}x Python's CPU work)")
        print(f"        dump load {load_s:.2f} s")

    if args.markdown:
        print()
        print("| virtual loss | positions | sims | median s | sims/s | us/sim | "
              "vs Python CPU |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            print(f"| {row['virtual_loss']} | {row['positions']} | {row['sims']} | "
                  f"{row['median_s']:.3f} | {row['sims_per_s']:,.0f} | "
                  f"{row['us_per_sim']:.2f} | {row['speedup']:.0f}x |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
