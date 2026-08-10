#!/usr/bin/env python
"""C10 Gate 2b — the Python reference's move, over the shared corpus.

WHAT THIS RECORDS AND WHY IT IS A SEPARATE STEP FROM THE TEST
=============================================================
The acceptance criterion is a differential one: two engines, one thread, a fixed
simulation budget, >= 99% move agreement. Running both sides inside pytest would
put python-chess and the reference's search in the test process, which Amendment
A exists to prevent — the reference reaches a test only through golden files. So
the reference runs here, once, and `tests/test_c10_gate2b.py` runs the C++ engine
against what it wrote.

Per position:

  best_move     the reference's answer, chosen exactly as it chooses it:
                `max(root.children.items(), key=visit_count)`, which keeps the
                FIRST maximum in python-chess's generation order (dict insertion
                order). C++ keeps the first maximum in CANONICAL order. That
                difference is deliberate and is one of the things Gate 2b is
                measuring -- see scope §2.6.
  visits        every root child's visit count, keyed by UCI. The margin the
                criterion asks about is computed from these, on both sides,
                rather than recorded as a single number, so the test can say what
                the reference thought of the move C++ chose and not only what it
                thought of its own.
  root_visits   for the delivered-vs-requested check.

EVERY SEARCH PARAMETER IS RECORDED, AND THAT IS NOT BOOKKEEPING
===============================================================
The manifest carries the **full resolved configuration** — `c_init`, `c_base`,
`fpu_root`, `fpu_tree`, `policy_temperature`, `virtual_loss`, `max_tree_depth` —
and `tests/test_c10_gate2b.py` builds the C++ `SearchConfig` *from those values*
rather than from its own defaults, asserting that every field is present.

That is a direct consequence of getting it wrong. The first Gate 2b run compared
the reference at `mctsv4.VIRTUAL_LOSS = 2.5` (its production default) against a
C++ engine at `SearchConfig.virtual_loss = 0.0` (its default, deliberately, so
that C5's certification at `cache_size=1`/VL=0 cannot be changed by a later
chunk). The two defaults differ, nothing complained, and the result was 12
decisive move disagreements that looked like a numerical failure and were a
configuration mismatch.

Virtual loss is not neutral at one worker, which is what made the mismatch
plausible enough to survive review: `MCTSNode.select_child` takes
`parent_visits` from **`effective_visits`** — `visit_count + vloss_count *
VIRTUAL_LOSS` — so the descent's own in-flight loss inflates `sqrt(N)` in the
exploration term of every child of every node on the current path. At VL 2.5 a
parent with 4 real visits explores as though it had 6.5. See DECISIONS.md, C10.

THE REST OF THE CONFIGURATION
=============================
  1 worker      The criterion says one thread.
  cache ON      Production. It cannot move the answer: a cache hit returns
                exactly the priors and value a fresh evaluation would have, on
                both sides, so switching it on only removes forward passes.
  tablebase off Out of the comparison. C7 preserved the reference's Syzygy
                behaviour including its defects; mixing it in here would put a
                third source of disagreement into a two-way test.
  no Dirichlet  `add_dirichlet_noise` defaults to False and stays there. Noise
                would make the reference's own answer non-reproducible.
  NO CANONICAL-ORDER PATCH. Gate 1 ran the reference with
                `GATE1_CANONICAL_ORDER = True` so both sides gathered in the same
                order. This run does NOT: scope §2.6 makes Gate 2b the test that
                establishes the ordering choice does not move search behaviour,
                and it can only do that against unpatched Python.

Global Rule 2: every number here comes from the Python reference. No column has
seen C++ output.

Usage:
    python tools/gen_c10_gate2b_golden.py --sims 400
    python tools/gen_c10_gate2b_golden.py --sims 400 --limit 20 --out /tmp/pilot.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import chess
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import mctsv4  # noqa: E402
from core.mctsv4 import ParallelMCTS, SearchParams  # noqa: E402
from playing.v5.playv5 import load_model  # noqa: E402

DEFAULT_CORPUS = REPO_ROOT / "golden" / "c10_corpus.json"
DEFAULT_MODEL = REPO_ROOT / "models" / "guofish5_20M" / "v5_10.9M_best.pt"
DEFAULT_OUT = REPO_ROOT / "golden" / "c10_gate2b.json"
DEFAULT_MANIFEST = REPO_ROOT / "golden" / "c10_gate2b_manifest.json"

# THIS IS THE RUN'S MEMORY BUDGET, and it is the reference's cache that spends it.
#
# The reference caches 4096-wide bf16 POLICY LOGIT ROWS — scope §2.5's measured
# 8 KiB/entry — so 400,000 slots is ~3.1 GiB of resident tensors, and a full run
# plateaus around 4.3-4.5 GB once torch, the CUDA context and the model are added.
# The plateau is the LRU cap engaging, not a leak: this many slots is reached and
# then held.
#
# Eviction is harmless and was never the reason for the number. A cache hit
# returns exactly what a fresh evaluation would have returned, on both sides, so
# the recorded answers do not depend on whether a given probe hit — only on how
# many forward passes were spent. The size is chosen to keep the run fast, and
# it can be cut to 100_000 (~800 MiB, the reference's own default) on a smaller
# machine with no effect on the golden data.
#
# The C++ side of the comparison stores *gathered priors* rather than 4096-wide
# rows — ~140 B/entry, about 100x smaller — so `cache_slots = 400_000` costs it
# roughly 56 MB, not 3 GiB. The asymmetry is scope §2.5's single largest
# structural win and it shows up here as a factor of fifty in resident memory.
CACHE_SIZE = 400_000


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top_two_margin(visits: dict[str, int]) -> float:
    """(v1 - v2) / total, over the root's children. 1.0 when there is only one.

    The criterion the acceptance names -- "every disagreement traced to a top-two
    visit margin of under 2%" -- is a statement about how close the decision was,
    so it is normalised by the visits actually distributed rather than by the
    budget: a search that spent visits on terminal fast paths has fewer to
    distribute and a raw difference would read as more decisive than it was.
    """
    counts = sorted(visits.values(), reverse=True)
    total = sum(counts)
    if total == 0:
        return 0.0
    if len(counts) == 1:
        return 1.0
    return (counts[0] - counts[1]) / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--limit", type=int, default=0, help="pilot runs only")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.limit == 0:
        for path in (args.out, args.manifest):
            if path.exists() and not args.force:
                print(f"refusing to overwrite {path} without --force", file=sys.stderr)
                return 2

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    positions = corpus["positions"]
    if args.limit:
        positions = positions[:args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    mctsv4.require_v5_config(model)

    # Asserted rather than assumed: Gate 1's patch reorders the gather, and a
    # Gate 2b run under it would be measuring the wrong thing while looking
    # exactly the same.
    if getattr(mctsv4, "GATE1_CANONICAL_ORDER", False):
        print("ERROR: mctsv4.GATE1_CANONICAL_ORDER is set. Gate 2b must run against "
              "UNPATCHED Python -- see scope §2.6.", file=sys.stderr)
        return 1

    params = SearchParams()
    mcts = ParallelMCTS(model=model, device=device, num_workers=1,
                        cache_size=CACHE_SIZE, tablebase=None, params=params)

    # The full resolved configuration, read off the objects that will actually
    # run rather than restated from the defaults they were built from. The
    # consuming test constructs the C++ SearchConfig from exactly these numbers;
    # a field added here and not there fails loudly, and a default that drifts on
    # either side cannot go unnoticed. See the module docstring for the incident.
    search_config = {
        "c_init": float(params.c_init),
        "c_base": float(params.c_base),
        "fpu_root": float(params.fpu_root),
        "fpu_tree": float(params.fpu_tree),
        "policy_temperature": float(params.policy_temperature),
        # Module globals, not SearchParams fields: both are read on the hot path
        # by code holding no SearchParams reference. Captured at run time for
        # that reason.
        "virtual_loss": float(mctsv4.VIRTUAL_LOSS),
        "max_tree_depth": int(mctsv4.MAX_TREE_DEPTH),
    }
    print("resolved reference configuration:")
    for key, value in search_config.items():
        print(f"  {key:20s} {value}")

    records = []
    started = time.perf_counter()
    try:
        for i, entry in enumerate(positions):
            board = chess.Board(entry["fen"])
            # No tree reuse across corpus positions: each one is its own search
            # from a cold root, which is what the C++ side does too.
            mcts.reset()
            best = mcts.search(board, num_simulations=args.sims)
            if best is None:
                print(f"ERROR: the reference returned no move at {entry['fen']}",
                      file=sys.stderr)
                return 1
            visits = {move.uci(): int(child.visit_count)
                      for move, child in mcts.root.children.items()}
            records.append({
                "fen": entry["fen"],
                "best_move": best.uci(),
                "root_visits": int(mcts.root.visit_count),
                "visits": visits,
                "margin": top_two_margin(visits),
            })
            if (i + 1) % 25 == 0 or i + 1 == len(positions):
                rate = (i + 1) / (time.perf_counter() - started)
                print(f"  {i + 1}/{len(positions)}  {rate:.2f} pos/s  "
                      f"eta {(len(positions) - i - 1) / rate:.0f}s", flush=True)
    finally:
        mcts.shutdown() if hasattr(mcts, "shutdown") else None

    elapsed = time.perf_counter() - started
    payload = {
        "sims": args.sims,
        "records": records,
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")

    manifest = {
        "generator": "tools/gen_c10_gate2b_golden.py",
        "python": platform.python_version(),
        "python_chess": chess.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "platform": platform.platform(),
        "model": {"path": str(args.model.relative_to(REPO_ROOT)),
                  "sha256": sha256_of(args.model)},
        "corpus": {"path": str(args.corpus.relative_to(REPO_ROOT)),
                   "sha256": sha256_of(args.corpus),
                   "positions": len(records)},
        "sims": args.sims,
        "workers": 1,
        "cache_size": CACHE_SIZE,
        "tablebase": None,
        "dirichlet": False,
        "canonical_order_patch": False,
        "search_config": search_config,
        "seconds": elapsed,
        "dump_sha256": sha256_of(args.out),
    }
    args.manifest.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8",
                             newline="\n")

    margins = sorted(r["margin"] for r in records)
    print(f"wrote {args.out}")
    print(f"wrote {args.manifest}")
    print(f"  {len(records)} positions at {args.sims} sims in {elapsed:.0f}s "
          f"({len(records) / elapsed:.2f} pos/s)")
    print(f"  top-two margin: min {margins[0]:.4f}  p10 {margins[len(margins) // 10]:.4f}  "
          f"median {margins[len(margins) // 2]:.4f}")
    print(f"  under 2%: {sum(1 for m in margins if m < 0.02)}/{len(margins)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
