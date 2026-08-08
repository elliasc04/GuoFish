#!/usr/bin/env python3
"""C7 golden data — the Gate 1 corpora re-run with the reference's cache ON.

WHY THIS FILE EXISTS
====================
C7's second acceptance criterion is "Gate 1 re-run (with cache ON against a
Python reference with cache ON) remains bit-exact". The Gate 1 golden trees C5
and C6 were judged against were produced at ``cache_size=1`` — the equivalence
configuration, which the brief chose because ``cache_size=0`` raises IndexError
and there is no cache-off flag. So they are not, on their face, a cache-on
reference.

The brief also states, from the port recon, that the Python cache is
result-invariant with tablebases off: ``cache=1`` and ``cache=100k`` give
bit-identical trees. If that is true then the existing golden trees ARE the
cache-on reference and nothing needs generating.

**That is an argument, and this file replaces it with a measurement.** It runs
the reference over both Gate 1 corpora with a 100,000-entry cache, writes the
resulting trees as new golden files, and records — per run, in the manifest —
whether the tree came out byte-identical to the cache=1 golden tree, along with
the reference's own cache hit and miss counts.

The result is two facts rather than one claim:

  * the C++ cache-on run has a genuine cache-on Python reference to be compared
    against, which is what the criterion asks for;
  * the invariance the brief asserts is now checked on this corpus, so if a
    future change to the reference breaks it, a named test fails rather than a
    footnote quietly becoming wrong.

The reference's per-run hit rate is recorded for a third reason: it is the only
independent evidence of what hit rate a *correct* cache achieves on these
positions. C++'s hit rate is asserted against a floor, and a floor picked out of
the air is a floor that tells you nothing.

WHAT IT DOES NOT WRITE
======================
No new replay dump. The existing ``golden/gate1_dump.npz`` and
``golden/gate1_terminal_dump.npz`` are produced from the cache=1 runs, where
every leaf was evaluated fresh, so they hold an entry for every position either
corpus ever expands — a superset of what a cache-on run evaluates. That matters:
the C++ side's cache and the reference's do not have to hit in the same places,
because a C++ miss falls through to a dump that covers it either way.

Global Rules 1 and 2: this writes only NEW files under ``golden/``, never over
an existing one, and refuses to run without ``--force`` if its own outputs are
already there. Every number in them comes from the Python reference.

Usage
-----
    python tools/gen_c7_cache_golden.py               # both corpora
    python tools/gen_c7_cache_golden.py --corpus quiet
    python tools/gen_c7_cache_golden.py --limit 2     # smoke run, NOT acceptance-grade
"""

import argparse
import json
import platform
import struct
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(TOOLS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chess  # noqa: E402
import torch  # noqa: E402

from core import mctsv4  # noqa: E402
from core.mctsv4 import ParallelMCTS, SearchParams  # noqa: E402

# Everything structural is imported from the C5/C6 generator rather than
# re-implemented. This file must not be a second opinion about what a Gate 1 run
# is; it is the same run with one setting changed.
import gen_gate1_golden as gen  # noqa: E402
from playing.v5.playv5 import load_model  # noqa: E402

# The reference's own default (`ParallelMCTS.__init__`: cache_size=100_000).
# Large enough that no eviction occurs inside a 5,000-simulation search, so the
# ring buffer's replacement policy is not part of the answer.
CACHE_SIZE = 100_000

QUIET_MANIFEST = REPO_ROOT / "golden" / "gate1_manifest.json"
TERMINAL_MANIFEST = REPO_ROOT / "golden" / "gate1_terminal_manifest.json"
QUIET_TREES = REPO_ROOT / "golden" / "gate1_trees.npz"
TERMINAL_TREES = REPO_ROOT / "golden" / "gate1_terminal_trees.npz"

OUT_TREES = {
    "quiet": REPO_ROOT / "golden" / "gate1_cache_trees.npz",
    "terminal": REPO_ROOT / "golden" / "gate1_cache_terminal_trees.npz",
}
OUT_MANIFEST = REPO_ROOT / "golden" / "gate1_cache_manifest.json"


def board_for(corpus: str, spec: dict) -> chess.Board:
    """The board a run starts from, WITH its move stack.

    The stack is not decoration: ``build_repetition_history`` walks back over it
    to seed the threefold counter, so a terminal-corpus position rebuilt from its
    root FEN alone would have an empty history and a different tree. The quiet
    corpus has no pre-root history by construction (C5 rejected any position
    whose history counted anything twice), so a plain FEN is correct there and is
    what the C5 generator used.
    """
    if corpus == "terminal":
        return gen.build_line(spec["base_fen"], spec["moves"])
    return chess.Board(spec["fen"])


def golden_records(trees: dict, index: int):
    """One run's node records out of a golden CSR block, in write_trees' tuple
    order, so they can be compared element-for-element against a fresh run."""
    begin = int(trees["run_offset"][index])
    end = int(trees["run_offset"][index + 1])
    terminal = trees["terminal"][begin:end] if "terminal" in trees else None
    terminal_value = trees["terminal_value"][begin:end] if "terminal_value" in trees else None
    out = []
    for i in range(end - begin):
        out.append((
            int(trees["depth"][begin + i]),
            int(trees["move"][begin + i]),
            int(trees["visits"][begin + i]),
            float(trees["value_sum"][begin + i]),
            float(trees["prior"][begin + i]),
            int(trees["children"][begin + i]),
            int(terminal[i]) if terminal is not None else 0,
            float(terminal_value[i]) if terminal_value is not None else 0.0,
        ))
    return out


def identical(golden, fresh) -> bool:
    """Bit-exact equality of two record lists.

    The float fields are compared on their BIT PATTERN, not with ``==``. This is
    the whole point of the exercise: two trees that agree to fifteen decimal
    places are two trees that would diverge structurally at simulation 6,000, and
    ``==`` would call them equal.
    """
    if len(golden) != len(fresh):
        return False
    for a, b in zip(golden, fresh):
        if a[:3] != b[:3] or a[5] != b[5] or a[6] != b[6]:
            return False
        if struct.pack("<d", a[3]) != struct.pack("<d", b[3]):
            return False
        if struct.pack("<f", np.float32(a[4])) != struct.pack("<f", np.float32(b[4])):
            return False
        if struct.pack("<f", np.float32(a[7])) != struct.pack("<f", np.float32(b[7])):
            return False
    return True


def run_corpus(corpus: str, manifest_path: Path, golden_trees_path: Path, model, device,
               limit: int) -> tuple[list, list]:
    """Re-run every recorded run of one corpus with the cache on.

    Returns (runs, positions). Each run carries its records plus an audit that
    includes the reference's cache counters and whether the tree matched the
    cache=1 golden one.
    """
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    with np.load(golden_trees_path) as data:
        golden = {k: data[k] for k in data.files}

    positions = {p["index"]: p for p in manifest["positions"]}
    recorded = manifest["runs"]
    if limit:
        recorded = recorded[:limit]

    mcts = ParallelMCTS(
        model=model,
        device=device,
        num_workers=1,
        cache_size=CACHE_SIZE,
        tablebase=None,
        params=SearchParams(),
    )

    runs = []
    for index, spec in enumerate(recorded):
        position = positions[spec["position"]]
        board = board_for(corpus, position)
        # The reference reads both of these as module globals at every step, so
        # they are set per run exactly as the C5/C6 generator sets them.
        mctsv4.VIRTUAL_LOSS = spec["virtual_loss"]
        mctsv4.MAX_TREE_DEPTH = spec.get("max_tree_depth", 80)

        # A fresh Recorder per run. Its `record_entry` still asserts that any key
        # expanded twice was expanded identically — which under a live cache is a
        # sharper check than it was at cache_size=1, because the second expansion
        # is now served FROM the cache and would surface a payload that had been
        # mutated in it.
        recorder = gen.Recorder()
        started = time.perf_counter()
        records, audit = gen.run_reference(
            mcts, board, spec["sims"], recorder,
            visited_only=not spec["full_tree"], strict=False)
        elapsed = time.perf_counter() - started

        cache = mcts.cache
        fresh_matches = identical(golden_records(golden, index), records)

        audit.update(
            virtual_loss=spec["virtual_loss"],
            sims=spec["sims"],
            full_tree=spec["full_tree"],
            position=spec["position"],
            fen=spec["fen"],
            max_tree_depth=spec.get("max_tree_depth", 80),
            name=spec.get("name"),
            cache_hits=int(cache.hits),
            cache_misses=int(cache.misses),
            cache_hit_rate=float(cache.hit_rate),
            cache_size=int(cache.size),
            matches_cache1_golden=fresh_matches,
            seconds=round(elapsed, 3),
        )
        runs.append({"records": records, "audit": audit})

        flag = "==" if fresh_matches else "!! DIFFERS FROM cache=1 GOLDEN"
        print(f"[{corpus} {index + 1}/{len(recorded)}] "
              f"{spec.get('name') or 'pos%02d' % spec['position']} "
              f"VL={spec['virtual_loss']:<4} n={spec['sims']:<5} "
              f"{len(records):>7} nodes  hits={int(cache.hits):>5} "
              f"misses={int(cache.misses):>5} "
              f"rate={cache.hit_rate:.3f}  {elapsed:5.1f}s  {flag}", flush=True)

    return runs, manifest["positions"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=("quiet", "terminal", "both"), default="both")
    parser.add_argument("--model", type=Path, default=gen.DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N runs per corpus (smoke runs; NOT "
                             "acceptance-grade, and the manifest says so)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite this generator's own outputs")
    args = parser.parse_args()

    corpora = ("quiet", "terminal") if args.corpus == "both" else (args.corpus,)

    targets = [OUT_TREES[c] for c in corpora] + [OUT_MANIFEST]
    existing = [p for p in targets if p.exists()]
    if existing and not args.force:
        print("refusing to overwrite existing golden files without --force:", file=sys.stderr)
        for p in existing:
            print(f"  {p}", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    model = load_model(args.model, device)

    # THE flag, same as C5/C6: without it the reference resolves PUCT ties by
    # python-chess's generation order, which C++ does not reproduce.
    mctsv4.GATE1_CANONICAL_ORDER = True
    default_max_depth = mctsv4.MAX_TREE_DEPTH

    out = {}
    try:
        for corpus in corpora:
            manifest_path = QUIET_MANIFEST if corpus == "quiet" else TERMINAL_MANIFEST
            golden_path = QUIET_TREES if corpus == "quiet" else TERMINAL_TREES
            runs, positions = run_corpus(corpus, manifest_path, golden_path, model, device,
                                         args.limit)
            stats = gen.write_trees(OUT_TREES[corpus], runs)
            out[corpus] = {
                "trees": str(OUT_TREES[corpus].relative_to(REPO_ROOT)),
                "trees_sha256": gen.sha256_of(OUT_TREES[corpus]),
                "tree_stats": stats,
                "positions": positions,
                "runs": [r["audit"] for r in runs],
            }
    finally:
        mctsv4.GATE1_CANONICAL_ORDER = False
        mctsv4.MAX_TREE_DEPTH = default_max_depth

    divergent = {c: [r["name"] or "pos%02d" % r["position"]
                     for r in out[c]["runs"] if not r["matches_cache1_golden"]]
                 for c in out}
    total_hits = sum(r["cache_hits"] for c in out for r in out[c]["runs"])
    total_misses = sum(r["cache_misses"] for c in out for r in out[c]["runs"])

    manifest = {
        "provenance": {
            "generator": "tools/gen_c7_cache_golden.py",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version(),
            "python_build": sys.version,
            "platform": platform.platform(),
            "python_chess": chess.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device": (torch.cuda.get_device_name(0) if device.type == "cuda" else None),
            "model": str(args.model.relative_to(REPO_ROOT)),
            "model_sha256": gen.sha256_of(args.model),
            "reference_sha256": gen.sha256_of(REPO_ROOT / "core" / "mctsv4.py"),
            "argv": sys.argv[1:],
        },
        "corpus": "gate1-cache-on",
        "config": {
            "workers": 1,
            # The one setting that differs from the C5/C6 manifests.
            "cache_size": CACHE_SIZE,
            "tablebase": False,
            "dirichlet": False,
            "canonical_order_patch": True,
            "virtual_losses": list(gen.VIRTUAL_LOSSES),
            "max_tree_depth": default_max_depth,
            "search_params": {
                "c_init": SearchParams().c_init,
                "c_base": SearchParams().c_base,
                "fpu_root": SearchParams().fpu_root,
                "fpu_tree": SearchParams().fpu_tree,
                "policy_temperature": SearchParams().policy_temperature,
            },
        },
        # The measurement this file exists for. `cache_invariant` is the brief's
        # recon claim, restated as a fact about THIS corpus and THIS checkpoint.
        "cache_invariance": {
            "cache_invariant": all(not names for names in divergent.values()),
            "divergent_runs": divergent,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": (total_hits / (total_hits + total_misses)
                         if total_hits + total_misses else 0.0),
        },
        "complete": args.limit == 0 and args.corpus == "both",
        "limit": args.limit,
        "corpora": out,
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print()
    for corpus, block in out.items():
        print(f"{corpus:<9} {block['trees']}  "
              f"{block['tree_stats']['runs']} runs, {block['tree_stats']['nodes']} nodes")
    print(f"manifest  {OUT_MANIFEST}")
    print(f"cache     {total_hits} hits / {total_misses} misses "
          f"({manifest['cache_invariance']['hit_rate']:.1%})")
    if manifest["cache_invariance"]["cache_invariant"]:
        print("invariance: every cache-on tree is BIT-IDENTICAL to its cache=1 golden tree")
    else:
        print("invariance: FAILED — see cache_invariance.divergent_runs", file=sys.stderr)
        return 1
    if not manifest["complete"]:
        print("NOTE: this run was limited and is NOT acceptance-grade", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
