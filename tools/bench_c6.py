"""Populate the C6 section of BENCH.md — what terminal handling costs.

NOTE: this is a benchmark, not a golden-data generator. Nothing here writes to
golden/ and nothing here is a reference implementation (Global Rule 2).

Usage:
    python tools/bench_c6.py [--trials N] [--markdown]

What this measures
------------------
Two things, and they answer different questions.

**The tax on a quiet position.** C6 added work to every descent step that C5 did
not do: a `ParsedFen` rebuild, an FNV-1a over ~90 bytes for `rep_key`, a linear
scan of the path's repetition tally, and a halfmove-clock update; plus, at every
leaf, a check-and-legal-move-count test against `outcome_of`. None of that fires
on a quiet position — no draw is ever claimed, no terminal is ever found — so the
whole of it is overhead there. Running the C5 corpus under the C6 build and
comparing against the numbers recorded in BENCH.md's C5 section is the honest way
to price it, and it is what `--quiet` does (it is exactly `tools/bench_c5.py`'s
measurement, repeated here so the two appear side by side).

**Throughput on positions where the machinery does fire.** The terminal corpus,
where up to 90% of simulations end in a claimable draw at ply four to six. Those
simulations are CHEAPER than a normal one, not dearer — they never reach the
leaf, never tokenize, never look anything up in the dump and never expand — so
this number is faster than the quiet one and says so. What it is good for is the
opposite of a headroom check: it is the number that would collapse if the
repetition tally were accidentally quadratic in the path length.

Neither number is a Gate 4 projection. There is no network, no cache, no tree
reuse and no concurrency; see the C5 section's much longer version of this
warning, which applies here unchanged.
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

CORPORA = {
    "quiet": (REPO_ROOT / "golden" / "gate1_dump.npz",
              REPO_ROOT / "golden" / "gate1_trees.npz",
              REPO_ROOT / "golden" / "gate1_manifest.json"),
    "terminal": (REPO_ROOT / "golden" / "gate1_terminal_dump.npz",
                 REPO_ROOT / "golden" / "gate1_terminal_trees.npz",
                 REPO_ROOT / "golden" / "gate1_terminal_manifest.json"),
}

# core.mctsv4's measured single-threaded CPU cost per simulation, excluding the
# forward pass (recon: 479 ms of real CPU work per 2,030 simulations). Carried
# over from tools/bench_c5.py so both sections quote the same baseline.
PYTHON_US_PER_SIM = 236.0


def load(corpus):
    dump_path, trees_path, manifest_path = CORPORA[corpus]
    for path in (dump_path, trees_path, manifest_path):
        if not path.exists():
            sys.exit(f"missing golden file: {path}\n"
                     f"Run `python tools/gen_gate1_golden.py"
                     f"{' --corpus terminal' if corpus == 'terminal' else ''}` first.")
    with np.load(dump_path) as data:
        dump = {k: data[k] for k in data.files}
    with np.load(trees_path) as data:
        trees = {k: data[k] for k in data.files}
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return dump, trees, manifest


def measure(corpus, trials):
    dump, trees, manifest = load(corpus)
    positions = {p["index"]: p for p in manifest["positions"]}

    runs = [(i, r) for i, r in enumerate(manifest["runs"]) if not r["full_tree"]]
    capacity = 0
    for index, _ in runs:
        begin = int(trees["run_offset"][index])
        end = int(trees["run_offset"][index + 1])
        capacity = max(capacity, 1 + int(trees["children"][begin:end].sum()))

    print(f"[{corpus}] dump: {len(dump['keys'])} entries, "
          f"{len(dump['moves'])} moves; arena capacity {capacity} nodes")

    rows = []
    for virtual_loss in (0.0, 2.5):
        selected = [(i, r) for i, r in runs if r["virtual_loss"] == virtual_loss]

        # One engine per (virtual loss, cap): both are fixed at construction. The
        # terminal corpus runs four positions at a lowered cap, so this is a
        # dict rather than a single search.
        engines = {}
        load_s = 0.0
        for _index, run in selected:
            cap = run.get("max_tree_depth", 80)
            if cap in engines:
                continue
            search = guofish_core.ReplaySearchDouble(
                guofish_core.SearchConfig(virtual_loss=virtual_loss,
                                          max_tree_depth=cap,
                                          arena_capacity=capacity))
            start = time.perf_counter()
            search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                             dump["moves"], dump["priors"], dump["values"])
            load_s += time.perf_counter() - start
            engines[cap] = search

        def history_of(run):
            return positions[run["position"]].get("history", [])

        # One untimed pass so the first run is not paying for cold pages in the
        # arena and the dump's hash table.
        for _index, run in selected:
            search = engines[run.get("max_tree_depth", 80)]
            search.set_position(run["fen"], history_of(run))
            search.search(run["sims"])

        per_trial = []
        sims_per_trial = 0
        draws = 0
        terminals = 0
        for _ in range(trials):
            elapsed = 0.0
            delivered = 0
            draws = 0
            terminals = 0
            for _index, run in selected:
                search = engines[run.get("max_tree_depth", 80)]
                search.set_position(run["fen"], history_of(run))
                start = time.perf_counter()
                stats = search.search(run["sims"])
                elapsed += time.perf_counter() - start
                delivered += stats["simulations"]
                draws += stats["draw_by_rule_hits"]
                terminals += (stats["checkmates"] + stats["stalemates"] +
                              stats["insufficient_material"])
            per_trial.append(elapsed)
            sims_per_trial = delivered

        median = statistics.median(per_trial)
        us_per_sim = median * 1e6 / sims_per_trial
        rows.append({
            "corpus": corpus,
            "virtual_loss": virtual_loss,
            "positions": len(selected),
            "sims": sims_per_trial,
            "median_s": median,
            "sims_per_s": sims_per_trial / median,
            "us_per_sim": us_per_sim,
            "speedup": PYTHON_US_PER_SIM / us_per_sim,
            "draw_share": draws / sims_per_trial,
            "terminal_marks": terminals,
            "load_s": load_s,
        })
        print(f"  VL {virtual_loss:<4} {len(selected)} positions, "
              f"{sims_per_trial} sims: median {median:.3f} s -> "
              f"{sims_per_trial / median:,.0f} sims/s ({us_per_sim:.2f} us/sim, "
              f"{PYTHON_US_PER_SIM / us_per_sim:.0f}x Python's CPU work); "
              f"{draws / sims_per_trial:.0%} of sims ended in a claimable draw, "
              f"{terminals} terminal marks")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--corpus", choices=("quiet", "terminal", "both"),
                        default="both")
    args = parser.parse_args()

    info = guofish_core.build_info()
    print(platform.platform())
    print(f"module: {info}")
    if info.get("asan") or info.get("ubsan"):
        print("\n*** WARNING: this is a SANITIZED build. These numbers are "
              "roughly 7x slow and do not belong in BENCH.md. ***\n")
    print()

    corpora = ("quiet", "terminal") if args.corpus == "both" else (args.corpus,)
    rows = []
    for corpus in corpora:
        rows.extend(measure(corpus, args.trials))
        print()

    if args.markdown:
        print("| corpus | virtual loss | positions | sims | median s | sims/s | "
              "us/sim | vs Python CPU | sims ending in a claimable draw |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            print(f"| {row['corpus']} | {row['virtual_loss']} | "
                  f"{row['positions']} | {row['sims']} | {row['median_s']:.3f} | "
                  f"{row['sims_per_s']:,.0f} | {row['us_per_sim']:.2f} | "
                  f"{row['speedup']:.0f}x | {row['draw_share']:.0%} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
