#!/usr/bin/env python3
"""C8 golden data — the reference playing whole games with its tree reused.

WHAT THIS FILE IS FOR
=====================
Every gate before C8 handed the C++ side one position, ran one search, and
compared one tree. C8's subject is what happens BETWEEN searches: a move is
played, the unselected branches are discarded, the chosen child becomes the new
root, and the next search continues from whatever visits that child already had.
The reference does this with a pointer assignment (`ParallelMCTS.apply_move`
detaches the child and drops the parent); C++ does it by compacting-copying the
surviving subtree into an alternate arena and remapping every `children_offset`.
Those are wildly different mechanisms that have to produce the same tree, node
for node and bit for bit, and they have to keep producing it for the length of a
game rather than for one move.

So the corpus here is a GAME, not a position. Each game runs

    set_position(root)
    repeat:
        search(sims)          -> snapshot the tree          ("search" snapshot)
        apply_move(best)      -> snapshot the tree          ("apply"  snapshot)

and every snapshot is written out. The acceptance criterion only asks about the
post-`apply_move` trees; the post-`search` trees are recorded as well because
without them a failure at ply 12 cannot be attributed. A search snapshot that
already differs is a C5/C6/C7 regression that tree reuse merely carried forward;
an apply snapshot that differs where the search snapshot before it matched is a
compaction bug. Those are different chunks' problems and the corpus should be
able to tell them apart.

THREE THINGS ARE RECORDED PER SNAPSHOT, AND THE THIRD IS THE ONE C8 NEEDS
========================================================================
1. ``records`` — the VISITED subtree, in canonical DFS preorder, exactly the
   layout `golden/gate1_trees.npz` uses: (depth, move, visits, value_sum, prior,
   children, terminal, terminal_value). This is the bit-exactness surface.

2. ``full_nodes`` — the number of nodes in the WHOLE subtree, visited or not.
   The C++ arena holds exactly the copied subtree after a compaction, so
   `search.nodes == full_nodes` is a direct measurement of what the compaction
   copied. A copy that dropped an unvisited child, or copied one twice, moves
   this number and nothing in (1) would notice.

3. ``full_shape`` — a SHA-256 over the full-tree DFS preorder of
   (depth, packed move, children_count, terminal bit). This is the structural
   diff the mutation drill targets. A `children_offset` that is off by one
   produces a tree whose visited nodes can still look plausible while its
   traversal order and child counts do not; hashing the traversal catches that
   in one comparison instead of hoping a numeric field happened to move.

WHAT IS DELIBERATELY UNCHANGED FROM C5/C6/C7
============================================
The engine is built by `gen_gate1_golden.build_engine` and the canonical-order
patch is the same one. The evaluator hooks are the same. `walk_tree` is the same
function — imported, not reimplemented — so the record layout cannot drift from
the one the earlier gates were certified against.

ONE THING HAD TO CHANGE, AND IT IS A REAL DIFFERENCE, NOT A TIDY-UP
===================================================================
`gen_gate1_golden.walk_tree` records a dump entry with ``is_root = (depth == 0)``.
That was correct when every run began with `_expand_root` on a fresh node. Under
tree reuse it is not: at ply 5 the root is a node that was expanded as an
INTERIOR leaf during the ply-4 search, so its priors are the CPU softmax and
tagging them `is_root` would file the CPU priors in the table the GPU priors
live in. The reference's two tables disagree by up to ~2e-9 (see
`Recorder.record_entry`), and the C++ side would then be handed the wrong ones
the first time a promoted root really is expanded as a root.

So the walk here is given a recorder PROXY that overrides the flag: a node is
`is_root` only if `_expand_root` was actually called on that node object. The
hook records the node objects it was called on (strong references — MCTSNode has
`__slots__` and cannot be weak-referenced), and identity decides.

Global Rules 1 and 2: every number here comes from the Python reference. This
writes only NEW files under ``golden/`` and refuses to overwrite its own outputs
without ``--force``.

Usage
-----
    python tools/gen_c8_reuse_golden.py
    python tools/gen_c8_reuse_golden.py --only quiet-80
    python tools/gen_c8_reuse_golden.py --plies 4        # smoke; NOT acceptance-grade
"""

import argparse
import hashlib
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

import gen_gate1_golden as gen  # noqa: E402
from gen_key_golden import fen_of  # noqa: E402
from playing.v5.playv5 import load_model  # noqa: E402

OUT_TREES = REPO_ROOT / "golden" / "c8_reuse_trees.npz"
OUT_DUMP = REPO_ROOT / "golden" / "c8_reuse_dump.npz"
OUT_MANIFEST = REPO_ROOT / "golden" / "c8_reuse_manifest.json"

# The reference's own default. The cache LIVES ACROSS MOVES here, which is the
# configuration C7 shipped and the one tree reuse actually runs in: a game is
# where a transposition cache earns its keep, and clearing it per move would test
# a configuration nobody runs.
CACHE_SIZE = 100_000

# TWO SIMULATION BUDGETS, AND THE SPLIT IS DELIBERATE.
#
# The chunk table calls C8 "Gate 1 across apply_move", and Gate 1 is specified at
# N >= 5000 (scope §5). So the acceptance game — `gate1-20`, the brief's 20-move
# sequence — runs at 5,000, unchanged. Nothing about the budget is negotiated
# away on the run the criterion names.
#
# The other games run at 2,000, and that is a stated trade rather than a quiet
# one. What they add is SEAMS: 170 more `apply_move` calls across four tree
# shapes, two virtual-loss magnitudes and both path-dependent draw rules. What
# they would not add at 5,000 is anything that only appears at simulation 4,999 —
# the per-search machinery at that budget is already certified by C5/C6/C7 over
# 48 runs. At 2,000 the whole corpus is ~25 minutes of reference time and a dump
# that stays regenerable; at 5,000 it is most of an afternoon.
#
# The consequence for the memory budget is real and is handled by measuring
# nodes-per-simulation rather than by pretending the number scales for free —
# scope §2.3 sizes the 2-3M budget from ~40 nodes/sim at 15k sims, and BENCH.md
# reports the measured rate beside the extrapolation rather than instead of it.
GATE1_SIMS = 5000
SIMS = 2000

# ---------------------------------------------------------------------------
# The corpus
#
# Four games, each chosen for a different thing the seam can get wrong.
# ---------------------------------------------------------------------------

GAMES = [
    # THE ACCEPTANCE GAME. The brief's "full 20-move sequence", at the Gate 1
    # simulation budget, from a benchmark midgame (C5 corpus position 1). This is
    # the run the first two acceptance criteria are about; everything below it
    # widens the coverage rather than replacing it.
    {
        "name": "gate1-20",
        "base_fen": "r2q1rk1/pB1pbpp1/1p2pn1p/4N3/1PP2B2/2n3P1/P1Q1PP1P/R2R2K1 b - - 0 15",
        "moves": [],
        "plies": 20,
        "sims": GATE1_SIMS,
        "virtual_loss": 0.0,
        "max_tree_depth": 80,
        "why": "the brief's 20-move sequence, at the Gate 1 budget (C5 position 1)",
    },
    # THE MEMORY-BUDGET GAME. 80 plies is the figure the acceptance criterion
    # names, and it is played from a real benchmark midgame (C5 corpus position
    # 0) so the branching factor and the tree shape are the ones the budget was
    # sized against rather than an endgame's.
    {
        "name": "quiet-80",
        "base_fen": "r3kr2/1p1nnpp1/1bp1p1p1/p2pP1N1/P2P3P/BPP3P1/5PB1/R3K2R w KQ - 1 21",
        "moves": [],
        "plies": 80,
        "sims": SIMS,
        "virtual_loss": 0.0,
        "max_tree_depth": 80,
        "why": "80-ply memory-budget game from a benchmark midgame (C5 position 0)",
    },
    # THE SAME SEAM AT VIRTUAL LOSS 2.5. Gate 1 runs every position at both
    # magnitudes because VL changes which child selection takes, and therefore
    # which subtree survives an apply_move. A compaction that is correct on one
    # shape is not thereby correct on another.
    {
        "name": "quiet-vl",
        "base_fen": "r2r2k1/pq1pbpp1/1p2pn1p/2n5/2PNP3/P1N3P1/1PQ2PKP/R1BR4 w - - 1 21",
        "moves": [],
        "plies": 40,
        "sims": SIMS,
        "virtual_loss": 2.5,
        "max_tree_depth": 80,
        "why": "the seam at VIRTUAL_LOSS 2.5 (C5 corpus position 2)",
    },
    # THE PATH/HISTORY PARTITION, REPETITION HALF. Implementation scope item 2:
    # a repetition that a simulation counted in its PATH tally becomes, once the
    # move is really played, part of the pre-root HISTORY tally. The two
    # partitions must sum identically or a node that was a threefold draw stops
    # being one (or starts being one) purely because a move was applied. The
    # eight priming plies are C6's `threefold-pawn-chain` and put every position
    # on the loop at two occurrences before the game even starts.
    {
        "name": "threefold-walk",
        "base_fen": "4k3/8/8/3pP3/3P4/8/8/4K3 w - - 0 1",
        "moves": ["e1e2", "e8e7", "e2e1", "e7e8", "e1e2", "e8e7", "e2e1", "e7e8"],
        "plies": 30,
        "sims": SIMS,
        "virtual_loss": 0.0,
        "max_tree_depth": 80,
        "why": "repetition history crossing the apply_move seam (C6 threefold-pawn-chain line)",
    },
    # THE PATH/HISTORY PARTITION, FIFTY-MOVE HALF. The clock is at 94 and no
    # move in a dead-drawn R+K vs R+K resets it, so it crosses 100 a few plies
    # in — WHILE moves are being applied, which is the case where the C++ side's
    # own halfmove clock (tracked as an int through make_move, and now through
    # apply_move) has to agree with python-chess's across the seam.
    {
        "name": "fifty-walk",
        "base_fen": "4rk2/8/8/8/8/8/8/3R1K2 w - - 94 70",
        "moves": ["d1d2", "e8e7"],
        "plies": 20,
        "sims": SIMS,
        "virtual_loss": 0.0,
        "max_tree_depth": 80,
        "why": "the halfmove clock crossing 100 across the apply_move seam (C6 fifty-rooks line)",
    },
]


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class RootTagger:
    """Which node objects `_expand_root` was actually called on.

    Strong references on purpose. `MCTSNode` declares `__slots__` without
    `__weakref__`, so it cannot be weak-referenced and cannot be tagged with a
    new attribute either; identity against a held list is what is left. The list
    is bounded by the number of searches in a game (at most one `_expand_root`
    per root, and roots only move forward), so holding the nodes costs a handful
    of subtrees that would otherwise be freed — irrelevant next to the tree
    itself, and it also guarantees no `id()` can be recycled underneath us.
    """

    def __init__(self):
        self.nodes = []

    def add(self, node):
        self.nodes.append(node)

    def contains(self, node) -> bool:
        return any(node is held for held in self.nodes)


class RootAwareRecorder:
    """A `gen.Recorder` proxy that fixes `is_root` for a REUSED root.

    `gen.walk_tree` derives the flag from the depth, which is right only when
    every root was expanded by `_expand_root`. Under tree reuse most roots were
    not; they were expanded as interior leaves and carry the CPU softmax's
    priors. Forwarding `depth == 0` unchanged would file those under the root
    table, and the C++ side — which reads the root table exactly when it has to
    run `expand_root` on a promoted node — would be served priors that are ~2e-9
    away from the ones it needs.

    Everything else, including the two conflict assertions, is the real
    recorder's.
    """

    def __init__(self, inner: gen.Recorder, tagger: RootTagger):
        self.inner = inner
        self.tagger = tagger
        self.root_is_root = False

    def begin_snapshot(self, root) -> None:
        self.root_is_root = self.tagger.contains(root)

    # gen.walk_tree only ever calls this one.
    def record_entry(self, key: int, is_root: bool, moves, priors) -> None:
        self.inner.record_entry(key, bool(is_root) and self.root_is_root, moves, priors)


def install_root_hook(recorder: gen.Recorder, tagger: RootTagger):
    """`gen.install_hooks`, plus a note of WHICH node `_expand_root` ran on.

    Wrapping rather than reimplementing: the value-recording behaviour has to be
    the C5 one bit for bit, and the only addition is the tagger call.
    """
    uninstall_inner = gen.install_hooks(recorder)
    hooked = ParallelMCTS._expand_root

    def patched(self, root, board):
        hooked(self, root, board)
        tagger.add(root)

    ParallelMCTS._expand_root = patched

    def uninstall():
        ParallelMCTS._expand_root = hooked
        uninstall_inner()

    return uninstall


# ---------------------------------------------------------------------------
# The full-tree walk
# ---------------------------------------------------------------------------

def full_tree_shape(root) -> tuple[int, str, int]:
    """(node count, SHA-256 of the shape, max depth) over the WHOLE subtree.

    Canonical DFS preorder over every node, visited or not — the same order
    `ReplaySearch::dump_tree(0)` produces, because both iterate children in
    canonical order and both emit a node before descending into it.

    The hashed record is (depth, packed move, children_count, terminal). Those
    four are the SHAPE: what the tree looks like, independent of what the search
    learned. Visit counts and values are compared separately and exactly, in
    `records`; putting them in here as well would make a hash mismatch ambiguous
    between "the arithmetic diverged" and "the layout is corrupt", which is
    exactly the distinction the mutation drill needs this digest to draw.
    """
    digest = hashlib.sha256()
    count = 0
    max_depth = 0
    # (node, depth), with children pushed in reverse so pop() yields canonical
    # order. Iterative for the same reason gen.walk_tree is: a depth violation
    # should be reportable rather than a RecursionError.
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        count += 1
        if depth > max_depth:
            max_depth = depth
        packed = 0 if depth == 0 else gen.pack_move(node.move)
        digest.update(struct.pack("<HHHB", depth, packed, len(node.children),
                                  1 if node.is_terminal else 0))
        children = list(node.children.values())
        for child in reversed(children):
            stack.append((child, depth + 1))
    return count, digest.hexdigest(), max_depth


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------

def snapshot(mcts, board, recorder_proxy, kind: str, ply: int, move_uci, seconds,
             stats) -> dict:
    """Serialise the tree as it stands, plus the audit that goes with it."""
    root = mcts.root
    recorder_proxy.begin_snapshot(root)
    records, max_depth, expanded, census = gen.walk_tree(
        root, board, recorder_proxy, visited_only=True, strict=False)
    full_nodes, full_shape, full_depth = full_tree_shape(root)

    audit = {
        "kind": kind,
        "ply": ply,
        "move": move_uci,
        "fen": fen_of(board),
        "halfmove_clock": board.halfmove_clock,
        "root_visits": int(root.visit_count),
        "root_children": len(root.children),
        "root_is_terminal": bool(root.is_terminal),
        "root_terminal_value": float(root.terminal_value),
        "nodes_recorded": len(records),
        "expanded_nodes": expanded,
        "max_depth": max_depth,
        "full_nodes": full_nodes,
        "full_max_depth": full_depth,
        "full_shape_sha256": full_shape,
        "census": census,
        "seconds": round(seconds, 3),
    }
    if stats is not None:
        audit["stats"] = stats
    return {"records": records, "audit": audit}


def play_game(spec: dict, model, device, recorder: gen.Recorder, tagger: RootTagger,
              plies_override: int) -> dict:
    """One game, snapshotted at every seam.

    The board carries its full move stack throughout, because that is what the
    UCI layer hands the reference and what `build_repetition_history` walks. A
    game played on a stackless board would have no repetition history at all and
    would exercise none of what games A/C/D are here for.
    """
    board = gen.build_line(spec["base_fen"], spec["moves"])
    root_fen = fen_of(board)
    history = gen.history_fens(board)

    mctsv4.VIRTUAL_LOSS = spec["virtual_loss"]
    mctsv4.MAX_TREE_DEPTH = spec["max_tree_depth"]

    mcts = gen.build_engine(model, device, CACHE_SIZE)
    mcts.reset()
    mcts.clear_cache()

    proxy = RootAwareRecorder(recorder, tagger)
    snapshots = []
    plies = min(spec["plies"], plies_override) if plies_override else spec["plies"]
    played: list[str] = []
    ended = None

    for ply in range(plies):
        if board.is_game_over():
            ended = f"game over before ply {ply}: {board.result()}"
            break

        started = time.perf_counter()
        with gen.DrawRuleCounter() as draws:
            best = mcts.search(board.copy(stack=True), num_simulations=spec["sims"])
        elapsed = time.perf_counter() - started
        if best is None:
            ended = f"search returned no move at ply {ply}"
            break

        stats = {
            "draw_by_rule_hits": draws.hits,
            "requested_sims": spec["sims"],
            "best_move": best.uci(),
        }
        snapshots.append(snapshot(mcts, board, proxy, "search", ply, best.uci(),
                                  elapsed, stats))

        # === the seam ===
        applied_started = time.perf_counter()
        mcts.apply_move(best)
        board.push(best)
        applied = time.perf_counter() - applied_started

        # The reference's own root board must have tracked ours move for move;
        # if it has not, every snapshot after this one is a tree over a position
        # nobody asked about.
        if fen_of(mcts.root_board) != fen_of(board):
            raise AssertionError(
                f"{spec['name']} ply {ply}: apply_move left the engine at "
                f"{fen_of(mcts.root_board)} but the game is at {fen_of(board)}")
        if mcts.root is None:
            raise AssertionError(
                f"{spec['name']} ply {ply}: apply_move discarded the tree "
                f"({best.uci()} was not a child of the root)")

        played.append(best.uci())
        snapshots.append(snapshot(mcts, board, proxy, "apply", ply, best.uci(),
                                  applied, None))

    if ended is None and board.is_game_over():
        ended = f"game over after the last ply: {board.result()}"

    return {
        "name": spec["name"],
        "why": spec["why"],
        "base_fen": spec["base_fen"],
        "setup_moves": list(spec["moves"]),
        "root_fen": root_fen,
        "history": history,
        "sims": spec["sims"],
        "virtual_loss": spec["virtual_loss"],
        "max_tree_depth": spec["max_tree_depth"],
        "requested_plies": plies,
        "played": played,
        "ended": ended,
        "final_fen": fen_of(board),
        "cache_hits": int(mcts.cache.hits),
        "cache_misses": int(mcts.cache.misses),
        "cache_hit_rate": float(mcts.cache.hit_rate),
        "snapshots": snapshots,
    }


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def write_trees(path: Path, games: list[dict]) -> dict:
    """Every snapshot of every game, in one CSR block.

    `gen.write_trees` takes a flat list of runs and that is exactly what a
    flattened snapshot list is, so the file layout is byte-for-byte the one the
    earlier gates use and `golden_records`-style readers work unchanged. The
    manifest carries the (game, snapshot) -> index mapping.
    """
    flat = []
    index = 0
    for game in games:
        game["snapshot_index"] = [index + i for i in range(len(game["snapshots"]))]
        for snap in game["snapshots"]:
            flat.append(snap)
            index += 1
    return gen.write_trees(path, flat)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=gen.DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--only", default="",
                        help="comma-separated game names")
    parser.add_argument("--plies", type=int, default=0,
                        help="cap every game at N plies (smoke runs; NOT "
                             "acceptance-grade, and the manifest says so)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    specs = GAMES
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        unknown = wanted - {g["name"] for g in GAMES}
        if unknown:
            print(f"unknown game(s): {sorted(unknown)}", file=sys.stderr)
            return 2
        specs = [g for g in GAMES if g["name"] in wanted]

    targets = [OUT_TREES, OUT_DUMP, OUT_MANIFEST]
    existing = [p for p in targets if p.exists()]
    if existing and not args.force:
        print("refusing to overwrite existing golden files without --force:", file=sys.stderr)
        for p in existing:
            print(f"  {p}", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    model = load_model(args.model, device)

    # THE flag. Without it the reference resolves PUCT ties by python-chess's
    # generation order, which C++ does not reproduce.
    mctsv4.GATE1_CANONICAL_ORDER = True
    default_max_depth = mctsv4.MAX_TREE_DEPTH
    default_vl = mctsv4.VIRTUAL_LOSS

    recorder = gen.Recorder()
    tagger = RootTagger()
    uninstall = install_root_hook(recorder, tagger)

    games = []
    try:
        for spec in specs:
            started = time.perf_counter()
            game = play_game(spec, model, device, recorder, tagger, args.plies)
            game["seconds"] = round(time.perf_counter() - started, 1)
            games.append(game)
            snaps = game["snapshots"]
            nodes = sum(len(s["records"]) for s in snaps)
            peak = max((s["audit"]["full_nodes"] for s in snaps), default=0)
            print(f"[{spec['name']}] {len(game['played'])} plies, {len(snaps)} snapshots, "
                  f"{nodes} recorded nodes, peak full-tree {peak}, "
                  f"cache {game['cache_hit_rate']:.1%}, {game['seconds']}s"
                  + (f"  [{game['ended']}]" if game["ended"] else ""), flush=True)
    finally:
        uninstall()
        mctsv4.GATE1_CANONICAL_ORDER = False
        mctsv4.MAX_TREE_DEPTH = default_max_depth
        mctsv4.VIRTUAL_LOSS = default_vl

    tree_stats = write_trees(OUT_TREES, games)
    dump_stats = gen.write_dump(OUT_DUMP, recorder)

    manifest = {
        "provenance": {
            "generator": "tools/gen_c8_reuse_golden.py",
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
        "corpus": "c8-tree-reuse",
        "config": {
            "workers": 1,
            "cache_size": CACHE_SIZE,
            "cache_cleared_between_moves": False,
            "tablebase": False,
            "dirichlet": False,
            "canonical_order_patch": True,
            "search_params": {
                "c_init": SearchParams().c_init,
                "c_base": SearchParams().c_base,
                "fpu_root": SearchParams().fpu_root,
                "fpu_tree": SearchParams().fpu_tree,
                "policy_temperature": SearchParams().policy_temperature,
            },
        },
        "trees": str(OUT_TREES.relative_to(REPO_ROOT)),
        "trees_sha256": gen.sha256_of(OUT_TREES),
        "tree_stats": tree_stats,
        "dump": str(OUT_DUMP.relative_to(REPO_ROOT)),
        "dump_sha256": gen.sha256_of(OUT_DUMP),
        "dump_stats": dump_stats,
        "complete": args.plies == 0 and not args.only,
        "plies_cap": args.plies,
        "games": [
            {k: v for k, v in game.items() if k != "snapshots"} |
            {"snapshots": [s["audit"] for s in game["snapshots"]]}
            for game in games
        ],
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print()
    print(f"trees     {OUT_TREES}  {tree_stats['runs']} snapshots, "
          f"{tree_stats['nodes']} nodes, {tree_stats['terminal_nodes']} terminal")
    print(f"dump      {OUT_DUMP}  {dump_stats['entries']} entries "
          f"({dump_stats['root_entries']} root), {dump_stats['moves']} moves")
    print(f"manifest  {OUT_MANIFEST}")
    if not manifest["complete"]:
        print("NOTE: this run was limited and is NOT acceptance-grade", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
