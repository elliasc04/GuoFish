"""C8 acceptance — tree reuse across `apply_move`, on ping-pong arenas.

WHAT IS BEING TESTED, AND WHY IT IS A DIFFERENT KIND OF THING FROM C5-C7
=======================================================================
Every gate so far compared ONE tree built from ONE position. This one compares a
sequence: five games, 190 applied moves, and after each of them the C++ tree has
to be the reference's tree — node for node, bit for bit — with the two sides
having arrived there by mechanisms that share nothing.

The reference promotes a child by assigning a pointer (`self.root = new_root`)
and letting Python's collector deal with the rest. C++ has no pointers and no
collector: the tree is a bump-allocated index space and the surviving subtree is
scattered through it among the ~70% of nodes that just became garbage. So the
promotion is a compacting copy into the standby arena, every `children_offset`
remapped, and a swap (scope §2.3). Those two implementations agreeing once is
weak evidence; agreeing 190 times across four tree shapes, two virtual-loss
magnitudes and both path-dependent draw rules is the claim.

THE SECTIONS BELOW MAP ONTO THE ACCEPTANCE CRITERIA
===================================================
1.  **Bit-exactness across the seam.** ``test_every_snapshot_matches_the_reference``
    is criterion 2 and is the whole chunk. It replays each game step for step
    and compares the visited subtree field by field, floats on their bit pattern.

    It also compares two things the record list cannot see, and both are about
    the compaction rather than about the search:

      * ``nodes`` — the arena's occupancy — against the reference's FULL subtree
        size, visited or not. A compaction that dropped an unvisited child, or
        copied one twice, moves this and nothing in the records would notice.
      * a SHA-256 over the full-tree DFS of (depth, move, children_count,
        terminal). This is the structural diff scope §7 asks for, on the Python
        side of the boundary: the visited records can look plausible while the
        layout underneath them is stitched out of the wrong siblings.

2.  **The memory budget.** ``test_the_arena_high_water_stays_inside_the_budget``
    is criterion 3, measured over the 80-ply game, reported in BENCH.md, and
    stated at the simulation budget it was measured at rather than at the one it
    would be nice to claim.

3.  **The path/history partition** (implementation scope item 2) —
    ``test_the_repetition_history_repartitions_correctly_across_the_seam``. When
    a move is applied, occurrences a simulation was counting on its PATH become
    occurrences the root counts as HISTORY, and the halfmove-clock horizon moves
    with them. The two games built for this (`threefold-walk`, `fifty-walk`)
    exercise both rules; the test re-derives the expected counter independently
    from the reference's own position trail and compares it to the engine's.

4.  **Terminal promotion** (the C6 invariant, now across a real seam) —
    ``test_a_terminal_root_promoted_by_apply_move_is_not_frozen``. `fifty-walk`
    promotes a node that is marked terminal on 17 of its 20 plies, so this is
    not a constructed probe: the corpus contains the case and the reference and
    C++ have to agree on it both before and after the mark is withdrawn.

5.  **No `_reset_virtual_loss`** (implementation scope item 4) —
    ``test_there_is_no_production_virtual_loss_reset`` and
    ``test_the_tree_is_quiescent_at_every_seam``. The first says the Release
    module does not contain a defensive walk. The second, in a build that has
    the debug-only audit, says it does not need one.

6.  **Ponder-root decay** (item 3) — ``test_ponder_decay_*``. Default 1.0 and
    provably a no-op at that setting, because the gate above would fail on ply 1
    otherwise.

WHAT THIS FILE DOES NOT DO
==========================
It does not corrupt a `children_offset`. The engine's own structural diff
(``SearchConfig.verify_compaction``, always on where asserts are) is what would
catch that, and a test cannot reach into the standby arena to break it without a
backdoor in production code. ``tools/drill_c8_reuse.py`` does it properly: it
mutates the fixup arithmetic in a scratch COPY of the source, rebuilds, and
requires both the engine's diff and this file's digest comparison to fail loudly.
Amendment B, applied to C++ rather than to a golden file.

Environment overrides
---------------------
``GUOFISH_GOLDEN_C8_TREES`` / ``_DUMP`` / ``_MANIFEST``, so the drill can point
this suite at corrupted copies in a scratch directory without `golden/` being
written to.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent

TREES = (REPO_ROOT / "golden" / "c8_reuse_trees.npz", "GUOFISH_GOLDEN_C8_TREES")
DUMP = (REPO_ROOT / "golden" / "c8_reuse_dump.npz", "GUOFISH_GOLDEN_C8_DUMP")
MANIFEST = (REPO_ROOT / "golden" / "c8_reuse_manifest.json", "GUOFISH_GOLDEN_C8_MANIFEST")

# Above the largest full tree the corpus builds (189,236 nodes, gate1-20) with
# room for the search that follows a compaction. The ping-pong pair is two of
# these, so this is ~41 MB of node payload, not 41 MB total — see
# `reuse_stats()["arena_bytes_reserved"]`, which the budget test reports.
ARENA_CAPACITY = 1 << 19

# C7's reasoning, unchanged: the C++ table is direct-mapped where the
# reference's is a hash map with ring-buffer eviction, so sizing well above the
# working set keeps slot collisions from being a confound. The reference ran at
# 100,000 (golden/c8_reuse_manifest.json records it).
CACHE_SLOTS = 1 << 20

# Scope §2.3: "with game-long reuse, budget 2-3M nodes peak". The lower figure is
# what the measurement is asserted against, because a budget passed only at its
# generous end is not a budget.
NODE_BUDGET = 2_000_000


def _path(spec):
    default, env = spec
    return Path(os.environ.get(env, default))


def _load_npz(spec, what):
    path = _path(spec)
    if not path.exists():
        pytest.fail(
            f"{what} missing: {path}\n"
            "Global Rule 2: regenerate with `python tools/gen_c8_reuse_golden.py`. "
            "It is produced by the Python reference and never from C++ output.")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def manifest():
    path = _path(MANIFEST)
    if not path.exists():
        pytest.fail(f"C8 manifest missing: {path}\n"
                    "Regenerate with `python tools/gen_c8_reuse_golden.py`.")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["complete"], (
        "golden/c8_reuse_manifest.json was written by a limited run "
        f"(--plies {payload['plies_cap']}) and is not acceptance-grade. "
        "Regenerate without --plies or --only.")
    return payload


@pytest.fixture(scope="session")
def trees():
    return _load_npz(TREES, "C8 reference trees")


@pytest.fixture(scope="session")
def dump():
    return _load_npz(DUMP, "C8 replay dump")


RECORD_FIELDS = ("depth", "move", "visits", "value_sum", "prior", "children",
                 "terminal", "terminal_value")


def _golden_snapshot(trees, index):
    begin = int(trees["run_offset"][index])
    end = int(trees["run_offset"][index + 1])
    return {name: trees[name][begin:end] for name in RECORD_FIELDS}


def _shape_digest(arrays) -> str:
    """SHA-256 over the FULL-tree DFS of (depth, move, children_count, terminal).

    Byte-for-byte what ``gen_c8_reuse_golden.full_tree_shape`` hashes: a packed
    little-endian ``<HHHB`` per node, in preorder. Built through a packed NumPy
    structured dtype rather than a per-node ``struct.pack`` because the trees run
    to 190,000 nodes and there are 380 of them.
    """
    n = len(arrays["depth"])
    packed = np.empty(n, dtype=np.dtype([("d", "<u2"), ("m", "<u2"),
                                         ("c", "<u2"), ("t", "u1")]))
    assert packed.itemsize == 7, "the structured dtype must be packed, not aligned"
    packed["d"] = arrays["depth"]
    packed["m"] = arrays["move"]
    packed["c"] = arrays["children"]
    packed["t"] = arrays["terminal"]
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _first_difference(golden, arrays):
    """A precise description of where two record sets first diverge, or None.

    Precise because a bare "trees differ" over 190,000 nodes is not a finding,
    it is the start of an investigation. The DFS index is reported alongside the
    field and both values, and the floats are compared on their BIT PATTERN —
    two value_sums that agree to fifteen decimal places are two trees that would
    have diverged structurally a thousand simulations later.
    """
    if len(golden["depth"]) != len(arrays["depth"]):
        return (f"node count {len(arrays['depth'])} where the reference has "
                f"{len(golden['depth'])}")
    for field in ("depth", "move", "visits", "children", "terminal"):
        bad = np.flatnonzero(np.asarray(arrays[field]) != np.asarray(golden[field]))
        if bad.size:
            i = int(bad[0])
            return (f"{field} differs at DFS index {i} (depth "
                    f"{int(golden['depth'][i])}, move "
                    f"{guofish_core.move_to_uci(int(golden['move'][i]))}): "
                    f"C++ {arrays[field][i]!r} vs reference {golden[field][i]!r} "
                    f"[{bad.size} node(s) differ]")
    pairs = (("value_sum", np.float64, np.uint64), ("prior", np.float32, np.uint32),
             ("terminal_value", np.float32, np.uint32))
    for field, dt, ut in pairs:
        mine = np.asarray(arrays[field], dtype=dt).view(ut)
        theirs = np.asarray(golden[field], dtype=dt).view(ut)
        bad = np.flatnonzero(mine != theirs)
        if bad.size:
            i = int(bad[0])
            return (f"{field} differs BITWISE at DFS index {i} (depth "
                    f"{int(golden['depth'][i])}, move "
                    f"{guofish_core.move_to_uci(int(golden['move'][i]))}): "
                    f"C++ {arrays[field][i]!r} (0x{int(mine[i]):x}) vs reference "
                    f"{golden[field][i]!r} (0x{int(theirs[i]):x}) "
                    f"[{bad.size} node(s) differ]")
    return None


def _new_search(game, dump, *, cache_slots, ponder_decay=1.0):
    config = guofish_core.SearchConfig(
        virtual_loss=game["virtual_loss"],
        max_tree_depth=game["max_tree_depth"],
        arena_capacity=ARENA_CAPACITY,
        cache_slots=cache_slots,
        # Scope §7's mitigation, on for the whole gate. In a build with asserts
        # it is on regardless; this makes the Release run carry it too, so the
        # acceptance figures come from a run that checked its own compactions.
        verify_compaction=True,
        ponder_decay=ponder_decay,
    )
    search = guofish_core.ReplaySearchDouble(config)
    search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                     dump["moves"], dump["priors"], dump["values"])
    search.set_position(game["root_fen"], game["history"])
    return search


def _replay(game, trees, dump, *, cache_slots):
    """One game, step for step, with every snapshot compared as it is produced.

    Returns a list of per-snapshot facts. The comparison happens HERE rather
    than in the tests because the arrays are large — 190,000 nodes at the peak,
    380 snapshots across the corpus — and keeping them all alive to assert over
    later would cost gigabytes for no benefit. What survives is the verdict plus
    the handful of scalars the other tests need.
    """
    search = _new_search(game, dump, cache_slots=cache_slots)
    out = []
    for offset, audit in enumerate(game["snapshots"]):
        index = game["snapshot_index"][offset]
        if audit["kind"] == "search":
            search.search(game["sims"])
        else:
            reused = search.apply_move(audit["move"])
        golden = _golden_snapshot(trees, index)
        visited = search.dump_tree_arrays(1)
        full = search.dump_tree_arrays(0)
        out.append({
            "audit": audit,
            "index": index,
            "difference": _first_difference(golden, visited),
            "nodes": int(search.nodes),
            "shape": _shape_digest(full),
            "root_fen": search.root_fen,
            "root_visits": int(search.root_visits),
            "reused": None if audit["kind"] == "search" else reused,
            "rep_history": dict(search.rep_history()),
        })
    return search, out


@pytest.fixture(scope="session")
def replayed(manifest, trees, dump):
    """Every game, replayed once with the cache on — the reference's own config.

    Session-scoped: this is the expensive thing in the file and eight tests read
    it. `golden/c8_reuse_manifest.json` records that the reference ran with a
    100,000-entry cache that was NOT cleared between moves, which is what makes
    a game the configuration a transposition cache is actually for.
    """
    out = {}
    for game in manifest["games"]:
        search, snapshots = _replay(game, trees, dump, cache_slots=CACHE_SLOTS)
        out[game["name"]] = {
            "game": game,
            "snapshots": snapshots,
            "reuse": search.reuse_stats(),
            "cache": search.cache_stats(),
            "search": search,
        }
    return out


# ---------------------------------------------------------------------------
# 1. THE ACCEPTANCE CRITERION — Gate 1 across apply_move
# ---------------------------------------------------------------------------


def test_every_snapshot_matches_the_reference(replayed):
    """CRITERION 2, over 380 snapshots and 190 applied moves.

    "After every single apply_move call, the serialized C++ tree must
    bit-exactly match the serialized Python tree after Python's apply_move."

    The post-SEARCH snapshots are compared too, and that is not scope creep — it
    is what makes a failure attributable. A search snapshot that already differs
    is a C5/C6/C7 regression that tree reuse merely carried forward. An apply
    snapshot that differs where the search snapshot immediately before it matched
    is a compaction bug, and it is C8's. Reporting the first divergence with its
    kind and ply says which.
    """
    failures = []
    seams = 0
    for name, block in replayed.items():
        for snap in block["snapshots"]:
            audit = snap["audit"]
            where = f"{name} ply {audit['ply']} ({audit['kind']}, {audit['move']})"
            if audit["kind"] == "apply":
                seams += 1
            if snap["difference"] is not None:
                failures.append(f"{where}: {snap['difference']}")
            if snap["nodes"] != audit["full_nodes"]:
                failures.append(
                    f"{where}: the arena holds {snap['nodes']} nodes where the "
                    f"reference's subtree has {audit['full_nodes']} — the "
                    f"compaction copied the wrong set, not merely the wrong values")
            if snap["shape"] != audit["full_shape_sha256"]:
                failures.append(
                    f"{where}: full-tree shape digest {snap['shape'][:16]} != "
                    f"{audit['full_shape_sha256'][:16]} — the visited records may "
                    f"still agree, but the layout underneath them does not")
            if snap["root_fen"] != audit["fen"]:
                failures.append(
                    f"{where}: the engine is at {snap['root_fen']} and the game "
                    f"is at {audit['fen']}")
            if snap["root_visits"] != audit["root_visits"]:
                failures.append(
                    f"{where}: root visits {snap['root_visits']} != "
                    f"{audit['root_visits']}")

    assert seams == sum(len(b["game"]["played"]) for b in replayed.values())
    assert seams >= 20, (
        f"only {seams} applied moves in the corpus; the criterion asks for a "
        f"20-move sequence and this suite is supposed to exceed it")
    assert not failures, (
        f"{len(failures)} divergence(s) from the reference across {seams} "
        f"applied moves:\n  " + "\n  ".join(failures[:20]) +
        ("" if len(failures) <= 20 else f"\n  ... and {len(failures) - 20} more"))


def test_the_twenty_move_gate1_sequence_ran_at_the_gate1_budget(replayed):
    """The criterion names a 20-move sequence; Gate 1 names N >= 5000.

    Asserted rather than assumed because the rest of the corpus runs at 2,000 to
    keep 170 extra seams affordable, and a reader who saw only that number could
    reasonably conclude the gate had been quietly relaxed. It has not: the game
    the criterion is about runs at the Gate 1 budget, and this is where that is
    checked against the manifest rather than against a comment.
    """
    block = replayed["gate1-20"]
    game = block["game"]
    assert game["sims"] >= 5000, (
        f"the acceptance game ran at {game['sims']} simulations; scope §5 "
        f"specifies Gate 1 at N >= 5000")
    assert len(game["played"]) == 20
    assert game["ended"] is None, (
        f"the acceptance game did not play 20 full moves: {game['ended']}")
    searches = [s for s in block["snapshots"] if s["audit"]["kind"] == "search"]
    assert all(s["root_visits"] == game["sims"] for s in searches), (
        "a search in the acceptance game exited early; that is legal (the "
        "depth-1 mate short-circuit does it) but it means the gate ran at a "
        "smaller budget than it claims, so it has to be visible")


def test_the_corpus_covers_the_shapes_the_seam_can_get_wrong(replayed, manifest):
    """The gate is only as good as what it was run over. Stated as measurements.

    Every clause here is a property of the CORPUS, not of the port, and it fails
    if a future regeneration quietly produces a corpus that no longer exercises
    something. A gate that passes over a corpus containing no terminal
    promotions has not tested terminal promotion.
    """
    names = set(replayed)
    assert {"gate1-20", "quiet-80", "quiet-vl", "threefold-walk", "fifty-walk"} <= names

    vls = {b["game"]["virtual_loss"] for b in replayed.values()}
    assert {0.0, 2.5} <= vls, f"the corpus runs at virtual losses {vls}; both are required"

    census = {}
    for block in replayed.values():
        for snap in block["snapshots"]:
            for key, value in snap["audit"]["census"].items():
                census[key] = census.get(key, 0) + value
    assert census["terminal_nodes"] > 0, "no terminal node anywhere in the corpus"
    assert census["terminal_win"] > 0, "no checkmate anywhere in the corpus"
    assert census["terminal_draw"] > 0, "no drawn terminal anywhere in the corpus"
    assert census["terminal_with_children"] == 0, (
        "the reference produced a node that is terminal AND has children — the "
        "bestmove 0000 state. The C++ arena cannot represent it, so this corpus "
        "cannot be replayed and the gate above is meaningless")

    promoted_terminal = sum(
        1 for block in replayed.values() for snap in block["snapshots"]
        if snap["audit"]["kind"] == "apply" and snap["audit"]["root_is_terminal"])
    assert promoted_terminal > 0, (
        "no apply_move in the corpus promoted a node that was marked terminal, "
        "so the promotion invariant the C6 brief carried into C8 is untested "
        "here")

    reexpanded = sum(block["reuse"]["terminal_marks_cleared"]
                     for block in replayed.values())
    assert reexpanded > 0, (
        "no promoted terminal root was ever re-expanded, so the "
        "clear-the-stale-mark path never ran")


# ---------------------------------------------------------------------------
# 2. THE COMPACTION ITSELF
# ---------------------------------------------------------------------------


def test_apply_move_actually_frees_the_dead_branches(replayed):
    """A compaction that copied everything would pass the equivalence gate.

    It would also grow the arena by a search's worth of nodes every ply and hit
    the capacity ceiling somewhere around move 30, which is the failure the
    ping-pong design exists to prevent. So the discard is asserted as a
    quantity, not inferred from the tree matching.
    """
    for name, block in replayed.items():
        reuse = block["reuse"]
        assert reuse["applies"] == len(block["game"]["played"]), (
            f"{name}: {reuse['applies']} compactions for "
            f"{len(block['game']['played'])} applied moves")
        assert reuse["nodes_dropped"] > 0, (
            f"{name}: no node was ever discarded, so the compaction is copying "
            f"the whole arena")
        assert reuse["nodes_copied"] > 0
        assert reuse["standby_allocated"] is True, (
            f"{name}: no standby arena was ever allocated, so nothing "
            f"ping-ponged")

    # And the specific consequence, on the game long enough to show it: the
    # arena does not grow monotonically over 80 plies.
    nodes = [s["nodes"] for s in replayed["quiet-80"]["snapshots"]]
    assert min(nodes[1:]) < max(nodes), "the arena never shrank across 80 plies"


def test_the_structural_diff_ran_on_every_compaction(replayed):
    """Scope §7's mitigation is engine behaviour, so it has to have executed.

    `verify_compaction=True` is set for every search in this file; a build with
    asserts on runs the diff regardless. If this count ever fell behind the
    compaction count it would mean the acceptance figures came from runs that
    did not check their own fixups — which is precisely the state scope §7 says
    not to be in.

    That the diff can FAIL is not demonstrated here and cannot be: breaking a
    remapped offset from Python would need a backdoor in production code.
    `tools/drill_c8_reuse.py` mutates the fixup arithmetic in a scratch copy of
    the source and rebuilds, which is the honest form of the same question.
    """
    for name, block in replayed.items():
        reuse = block["reuse"]
        assert reuse["verifications"] == reuse["applies"], (
            f"{name}: {reuse['verifications']} structural diffs for "
            f"{reuse['applies']} compactions")


def test_the_arena_high_water_stays_inside_the_budget(replayed):
    """CRITERION 3, measured over the 80-ply game.

    `arena_high_water` is the peak of EITHER arena, which is the honest figure:
    during a compaction the source subtree and its copy are alive at the same
    instant, so the moment of greatest occupancy is the moment the two overlap.

    The number is asserted against scope §2.3's LOWER bound (2M of the "2-3M
    nodes peak" budget), because a budget passed only at its generous end is not
    a budget. BENCH.md carries the figure itself, the nodes-per-simulation rate
    behind it, and the extrapolation to the 15k-simulation production budget —
    which is an extrapolation and is labelled as one, because the reference
    corpus this replays was generated at 2,000.
    """
    block = replayed["quiet-80"]
    game = block["game"]
    assert len(game["played"]) == 80, (
        f"the memory-budget game played {len(game['played'])} plies, not 80: "
        f"{game['ended']}")

    peak = block["reuse"]["arena_high_water"]
    assert 0 < peak <= NODE_BUDGET, (
        f"arena high-water {peak} nodes over an 80-ply game at {game['sims']} "
        f"simulations exceeds the {NODE_BUDGET} node budget (scope §2.3)")

    # Sanity on the measurement itself: a high-water that never exceeded the
    # largest single tree would mean the counter is reporting the current size
    # rather than the peak.
    largest = max(s["nodes"] for s in block["snapshots"])
    assert peak >= largest, (
        f"high-water {peak} is below the largest observed tree {largest}; the "
        f"counter is not tracking a peak")

    # The figure BENCH.md reports, computed here so the two cannot drift.
    sims = sum(s["audit"]["root_visits"] for s in block["snapshots"]
               if s["audit"]["kind"] == "search")
    print(f"\n[C8 memory] quiet-80: high-water {peak} nodes, largest tree {largest}, "
          f"{sims} root visits over 80 plies, "
          f"{block['reuse']['arena_bytes_reserved'] / (1 << 20):.1f} MiB reserved "
          f"for the ping-pong pair at capacity {ARENA_CAPACITY}")


# ---------------------------------------------------------------------------
# 3. THE PATH / HISTORY PARTITION — implementation scope item 2
# ---------------------------------------------------------------------------


def test_the_repetition_history_repartitions_correctly_across_the_seam(replayed):
    """Item 2, asserted at its source rather than through its consequences.

    "Repetition and 50-move draws rely on counts that partition differently once
    a move is formally applied... the history + path occurrences must sum
    identically."

    The engine's `rep_history` is `build_repetition_history(board)` for the
    current root: every prior position inside the halfmove-clock horizon, plus
    the root itself. When a move is applied, the position being left enters that
    map and the horizon moves — by one for a reversible move, all the way to
    zero for a capture or a pawn push, because nothing before a zeroing move can
    ever repeat again.

    The expectation is re-derived here from the REFERENCE's own position trail —
    the FENs it recorded at every snapshot — through python-chess's documented
    rule, and keyed with C3's `rep_key`, which was certified against
    `_transposition_key()` by partition equality over 100k positions. So this
    compares two independent derivations rather than the engine against itself.
    """
    checked = 0
    for name, block in replayed.items():
        game = block["game"]
        # Most recent first, exactly as `history_fens` and `set_position` want
        # it. Seeded with the pre-root history the game was handed.
        trail = list(game["history"])
        previous_fen = game["root_fen"]
        for snap in block["snapshots"]:
            audit = snap["audit"]
            if audit["kind"] != "apply":
                continue
            trail.insert(0, previous_fen)
            previous_fen = audit["fen"]

            clock = audit["halfmove_clock"]
            window = min(clock, len(trail))
            expected = {guofish_core.rep_key(audit["fen"]): 1}
            for fen in trail[:window]:
                key = guofish_core.rep_key(fen)
                expected[key] = expected.get(key, 0) + 1

            assert snap["rep_history"] == expected, (
                f"{name} ply {audit['ply']} ({audit['move']}): the engine's "
                f"repetition history after the seam is {snap['rep_history']} "
                f"where the reference's position trail gives {expected} "
                f"(halfmove clock {clock}, {len(trail)} plies behind the root)")
            checked += 1

    assert checked >= 190, f"only {checked} seams checked"


def test_the_two_rule_games_actually_exercise_the_two_rules(replayed):
    """The test above is vacuous on a corpus where no clock ever moves.

    `threefold-walk` starts eight plies into a repetition loop and
    `fifty-walk` starts at halfmove clock 96, so one exercises the repetition
    partition and the other the clock horizon — including the zeroing case,
    where the window must collapse rather than shrink by one.
    """
    fifty = replayed["fifty-walk"]
    clocks = [s["audit"]["halfmove_clock"] for s in fifty["snapshots"]]
    assert max(clocks) >= 100, (
        f"the fifty-move game's clock only reached {max(clocks)}; it never "
        f"crossed the 100 that makes a position drawn")

    three = replayed["threefold-walk"]
    repeated = [s for s in three["snapshots"]
                if s["audit"]["kind"] == "apply" and max(s["rep_history"].values()) >= 2]
    assert repeated, (
        "no root in the threefold game ever had a position occurring twice in "
        "its history, so the repetition partition was never non-trivial")

    draws = sum(s["audit"]["census"]["terminal_draw"] for s in three["snapshots"])
    draws += sum(s["audit"]["census"]["terminal_draw"] for s in fifty["snapshots"])
    assert draws > 0, "neither rule game produced a drawn terminal"

    # A zeroing move somewhere in the corpus, so the collapse case is covered.
    zeroed = False
    for block in replayed.values():
        clocks = [s["audit"]["halfmove_clock"] for s in block["snapshots"]
                  if s["audit"]["kind"] == "apply"]
        zeroed = zeroed or any(b == 0 for b in clocks)
    assert zeroed, (
        "no applied move in the corpus was a capture or a pawn push, so the "
        "horizon never collapsed and the zeroing branch is untested")


# ---------------------------------------------------------------------------
# 4. TERMINAL PROMOTION ACROSS A REAL SEAM — the C6 invariant, carried forward
# ---------------------------------------------------------------------------


def test_a_terminal_root_promoted_by_apply_move_is_not_frozen(replayed):
    """The risk the brief names, and the corpus contains it 17 times.

    "Ensure the ping-pong copy does not accidentally freeze or lose the ability
    to expand these nodes if they become the new root."

    `fifty-walk` promotes a node the search had marked drawn by the fifty-move
    rule on 17 of its 20 plies. What has to be true of each one, and is checked
    against the reference at every step by the gate above, is a sequence:

      after apply_move   terminal bit SET, no children, Unexpanded — the state
                         C6 established is recoverable, preserved by the copy;
      after the search   the host having declined the claim, the mark is GONE
                         and the node has children and a legal move.

    A copy that dropped the mark would fail the first; one that froze the node
    would fail the second with `bestmove 0000`, which is the defect the whole
    terminal design exists to make unrepresentable.
    """
    promotions = 0
    reexpansions = 0
    for name, block in replayed.items():
        snapshots = block["snapshots"]
        for i, snap in enumerate(snapshots):
            audit = snap["audit"]
            if audit["kind"] != "apply" or not audit["root_is_terminal"]:
                continue
            promotions += 1
            assert audit["root_children"] == 0, (
                f"{name} ply {audit['ply']}: the reference promoted a node that "
                f"is terminal AND has {audit['root_children']} children")

            # The next snapshot is the search that follows the promotion. If the
            # game ended, there is none — and then there is nothing to expand.
            if i + 1 >= len(snapshots):
                continue
            after = snapshots[i + 1]
            assert after["audit"]["kind"] == "search"
            legal = guofish_core.legal_moves(snap["root_fen"])
            if not legal:
                # Genuine checkmate or stalemate: "still yields a move" is not a
                # property anyone can have. C6 separates these for the same
                # reason.
                continue
            assert after["audit"]["root_children"] > 0, (
                f"{name} ply {audit['ply']}: a terminal node promoted to root by "
                f"apply_move produced no children at {snap['root_fen']}, which "
                f"has {len(legal)} legal moves. This is bestmove 0000")
            assert after["audit"]["root_is_terminal"] is False, (
                f"{name} ply {audit['ply']}: the promoted root kept its terminal "
                f"mark after being expanded; the reference clears it in "
                f"_expand_root and a node that is both is unrepresentable here")
            reexpansions += 1

    assert promotions >= 17, (
        f"only {promotions} terminal promotions in the corpus; fifty-walk alone "
        f"is supposed to supply 17")
    assert reexpansions > 0, (
        "every promoted terminal was a position with no legal moves, so none of "
        "them can demonstrate that promotion is recoverable")


def test_expand_root_assigns_the_promoted_visit_count_rather_than_adding_to_it(replayed):
    """The one-line bug this section would otherwise miss.

    `_expand_root` writes `root.visit_count = 1`. Until tree reuse existed the
    distinction was invisible: a fresh root came out of the arena cleared, so
    `+= 1` and `= 1` were the same operation. A PROMOTED root is not cleared — a
    fifty-move draw arrives carrying hundreds of visits from the terminal fast
    path — and `+= 1` would leave the tree claiming it had been searched that
    many times.

    The gate above would catch it, at every node above the root, as a
    bit-exactness failure. This says what it was.
    """
    found = 0
    for name, block in replayed.items():
        snapshots = block["snapshots"]
        for i, snap in enumerate(snapshots[:-1]):
            audit = snap["audit"]
            if audit["kind"] != "apply" or audit["root_children"] != 0:
                continue
            if audit["root_visits"] <= 1:
                continue  # nothing to be lost by adding
            after = snapshots[i + 1]["audit"]
            if after["root_children"] == 0:
                continue  # never expanded; the game ended here
            assert after["root_visits"] == snapshots[i + 1]["root_visits"], (
                f"{name} ply {audit['ply']}: C++ and the reference disagree on "
                f"the re-seeded root's visit count")
            found += 1
    assert found > 0, (
        "no promoted root in the corpus arrived unexpanded with more than one "
        "visit, so the assignment-versus-accumulation distinction is untested")


# ---------------------------------------------------------------------------
# 5. NO _reset_virtual_loss — implementation scope item 4
# ---------------------------------------------------------------------------


def test_there_is_no_production_virtual_loss_reset():
    """Item 4: "Do NOT implement a production equivalent of _reset_virtual_loss."

    The reference walks the whole tree writing `vloss_count = 0` before every
    search — 3.4 ms at 2k sims, 937 ms over a game (scope §2.3) — because
    nothing guaranteed a previous search had repaid what it applied. Here
    repayment is scope-guaranteed by RAII, so the walk buys nothing and would
    hide a bug if it found anything to do.

    Checked as an absence, which is awkward to assert and worth asserting
    anyway: there is no resetting entry point under any name, and the only
    full-tree virtual-loss code that exists is a READ, compiled only behind
    GUOFISH_DEBUG_VL. `guofish_core.DEBUG_VL` reports that flag, so this is a
    statement about the module rather than an inference from a missing
    attribute — which a typo would satisfy just as well.
    """
    search = guofish_core.ReplaySearchDouble(
        guofish_core.SearchConfig(arena_capacity=1 << 12))
    for forbidden in ("reset_virtual_loss", "_reset_virtual_loss", "clear_virtual_loss"):
        assert not hasattr(search, forbidden), (
            f"ReplaySearch exposes {forbidden}; C8 forbids a production "
            f"equivalent of the reference's defensive walk")

    assert isinstance(guofish_core.DEBUG_VL, bool)
    assert hasattr(search, "debug_total_vloss") == guofish_core.DEBUG_VL, (
        "guofish_core.DEBUG_VL and the presence of the audit disagree; one of "
        "the two is lying about what this module contains")


def test_the_tree_is_quiescent_at_every_seam(replayed):
    """The invariant the reference's defensive walk was standing in for.

    A quiescent tree holds exactly ZERO in-flight virtual losses, at any
    virtual-loss magnitude, because `vloss_count` is an integer count and the
    penalty is applied at read time — apply and repay are exact inverses with no
    floating-point residue. The debug audit scans the whole arena rather than
    traversing the tree, so a loss stranded on an unreachable node (which is
    exactly what a compaction bug could leave behind) is counted too.

    Runs only where the audit is compiled, which by default is Debug builds —
    i.e. the sanitized run Global Rule 5 requires the suite to pass under. In a
    Release build there is nothing to read, and `test_there_is_no_production_
    virtual_loss_reset` asserts that absence positively.
    """
    if not guofish_core.DEBUG_VL:
        pytest.skip("built without GUOFISH_DEBUG_VL; the audit is not compiled in")
    for name, block in replayed.items():
        total = block["search"].debug_total_vloss()
        assert total == 0, (
            f"{name}: {total} virtual losses are still in flight after the game, "
            f"at VIRTUAL_LOSS {block['game']['virtual_loss']}. RAII repayment is "
            f"what lets C8 delete the defensive walk; this says it did not happen")


# ---------------------------------------------------------------------------
# 6. PONDER-ROOT DECAY — implementation scope item 3
# ---------------------------------------------------------------------------


def _decay_case(manifest, dump, decay):
    """One search and one applied move at a given decay. Returns (tree, search).

    The move is the one the reference played, so the promoted subtree is the same
    one every other test in this file looks at — the only variable is the decay.
    """
    game = next(g for g in manifest["games"] if g["name"] == "fifty-walk")
    search = _new_search(game, dump, cache_slots=0, ponder_decay=decay)
    search.search(game["sims"])
    search.apply_move(game["played"][0], from_ponder=decay != 1.0)
    return search.dump_tree_arrays(0), search


def test_ponder_decay_defaults_to_a_no_op(manifest, dump):
    """The default has to be provably inert, not merely documented as inert.

    The reference has no decay, so a default that touched anything would fail
    Gate 1 on ply 1 — which is the correct behaviour for a knob that changed the
    engine without being asked, and is why the default is 1.0. Checked directly
    as well, because "the gate would have caught it" is an argument and this is
    a measurement: `from_ponder=True` at the default decay must produce the same
    tree as `from_ponder=False`.
    """
    game = next(g for g in manifest["games"] if g["name"] == "fifty-walk")
    trees = []
    for from_ponder in (False, True):
        search = _new_search(game, dump, cache_slots=0)
        search.search(game["sims"])
        search.apply_move(game["played"][0], from_ponder=from_ponder)
        trees.append(search.dump_tree_arrays(0))
    plain, pondered = trees
    assert _first_difference(plain, pondered) is None, (
        "ponder_decay defaults to 1.0, so promoting a pondered root must be "
        "byte-identical to promoting an ordinary one: "
        + str(_first_difference(plain, pondered)))


def test_ponder_decay_scales_visits_and_leaves_q_alone(manifest, dump):
    """Item 3, and the shape of the thing matters as much as that it fires.

    Decay is a statement about CONFIDENCE, not about evaluation. Scope §8's
    reason for wanting it — 30k fresh simulations cannot redistribute against
    64k+ inherited ones — is about weight, and a node the ponder search scored
    at +0.4 should still score +0.4 afterwards, just with less of the tree's
    attention nailed to it. So visits and value_sum are scaled by the same
    ratio and Q comes out unchanged.

    Scaling visits alone would divide every inherited Q by the decay factor and
    turn a quiet +0.4 into a winning +0.8 at d=0.5, which is the opposite of the
    intent and would be invisible to any test that only counted visits.
    """
    after, search = _decay_case(manifest, dump, 0.5)
    assert search.reuse_stats()["decays"] == 1

    # The promoted subtree, as it stood before the move, is the comparison set.
    # Taking the root's own numbers is enough to see both halves: it is the node
    # with the most visits and the only one whose Q is worth reading.
    root_visits = int(after["visits"][0])
    assert root_visits >= 1

    plain = _new_search(next(g for g in manifest["games"] if g["name"] == "fifty-walk"),
                        dump, cache_slots=0)
    game = next(g for g in manifest["games"] if g["name"] == "fifty-walk")
    plain.search(game["sims"])
    plain.apply_move(game["played"][0])
    undecayed = plain.dump_tree_arrays(0)

    assert len(undecayed["visits"]) == len(after["visits"]), (
        "decay changed the SHAPE of the tree; it must only change its weights")

    scaled = 0
    for i in range(len(after["visits"])):
        old = int(undecayed["visits"][i])
        new = int(after["visits"][i])
        if old == 0:
            assert new == 0, "an unvisited node acquired visits"
            continue
        expected = max(1, int(round(old * 0.5)))
        # Python's round() is banker's rounding and C++ uses llround (half away
        # from zero), so allow the one-unit disagreement at exact halves rather
        # than encoding one language's tie rule as the specification.
        assert abs(new - expected) <= 1, (
            f"node {i}: {old} visits decayed to {new}, expected ~{expected}")
        if new != old:
            scaled += 1
        if new > 0 and old > 0:
            q_old = float(undecayed["value_sum"][i]) / old
            q_new = float(after["value_sum"][i]) / new
            assert abs(q_new - q_old) < 1e-9, (
                f"node {i}: Q moved from {q_old} to {q_new} under decay; decay "
                f"must scale confidence, not evaluation")
    assert scaled > 0, "decay at 0.5 changed no visit count at all"


def test_ponder_decay_rejects_a_factor_outside_its_range(manifest, dump):
    """0 and negatives are not "aggressive decay", they are a broken tree."""
    game = next(g for g in manifest["games"] if g["name"] == "fifty-walk")
    for bad in (0.0, -0.5, 1.5):
        search = _new_search(game, dump, cache_slots=0, ponder_decay=bad)
        search.search(game["sims"])
        with pytest.raises(ValueError):
            search.apply_move(game["played"][0], from_ponder=True)


# ---------------------------------------------------------------------------
# 7. THE CACHE ACROSS THE SEAM — C7 x C8
# ---------------------------------------------------------------------------


def test_the_cache_does_not_change_the_tree_across_a_game(manifest, trees, dump, replayed):
    """C7's invariance claim, re-measured where it is most likely to break.

    C7 established that with tablebases off the cache is result-invariant. That
    was measured within single searches. A game is the first configuration where
    the cache OUTLIVES the tree it was filled from: entries written at ply 3 are
    served at ply 11 to a search whose root is somewhere else entirely. If a
    cached payload were ever mutated in place, or keyed by anything the position
    does not determine, this is where it would show.

    Run on the two cheapest games rather than all five — the point is the seam,
    and 50 seams over two tree shapes says as much about it as 190 would.
    """
    for name in ("fifty-walk", "threefold-walk"):
        game = next(g for g in manifest["games"] if g["name"] == name)
        _, cold = _replay(game, trees, dump, cache_slots=0)
        warm = replayed[name]["snapshots"]
        for a, b in zip(cold, warm):
            assert a["difference"] is None, (
                f"{name} ply {a['audit']['ply']} ({a['audit']['kind']}) with the "
                f"cache OFF: {a['difference']}")
            assert a["shape"] == b["shape"], (
                f"{name} ply {a['audit']['ply']} ({a['audit']['kind']}): the "
                f"cache changed the tree's shape across the seam")
            assert a["nodes"] == b["nodes"]

    hits = sum(block["cache"]["hits"] for block in replayed.values())
    misses = sum(block["cache"]["misses"] for block in replayed.values())
    assert hits > 0 and hits / (hits + misses) > 0.05, (
        f"the cache hit {hits} times in {hits + misses} probes across the whole "
        f"corpus; a cache that never hits passes every test in this section "
        f"vacuously")


# ---------------------------------------------------------------------------
# 8. THE EDGES OF apply_move
# ---------------------------------------------------------------------------


def test_apply_move_rejects_an_illegal_move(manifest, dump):
    """A caller that has desynchronised from the game must find out here.

    The alternative is worse than a crash: the engine would advance to a
    position nobody is in and every subsequent answer would be about that
    position, confidently.
    """
    game = next(g for g in manifest["games"] if g["name"] == "fifty-walk")
    search = _new_search(game, dump, cache_slots=0)
    search.search(50)
    with pytest.raises(ValueError):
        search.apply_move("a1a2")
    with pytest.raises(ValueError):
        search.apply_move("not-a-move")


def test_apply_move_without_a_tree_still_advances_the_position(manifest, dump):
    """The reference's "nothing to reuse" branch, and the UCI layer's normal one.

    `move not in self.root.children` sets `self.root = None` and the next search
    builds a fresh tree. Here the root exists but is unexpanded — no search has
    run — so there is no child to promote, and the position still has to move.
    Returning False rather than raising is what lets a host replay a whole game's
    moves into the engine before asking it to search any of them, which is what
    `position startpos moves ...` is.

    So this replays the fifty-walk game with NO searching at all: twenty
    discards in a row. Every one has to leave a single fresh root and a position
    that tracks the reference's, which is a check on the board and the halfmove
    clock with the tree taken out of the picture entirely.
    """
    game = next(g for g in manifest["games"] if g["name"] == "fifty-walk")
    search = _new_search(game, dump, cache_slots=0)
    applied = [s for s in game["snapshots"] if s["kind"] == "apply"]

    for audit in applied:
        assert search.apply_move(audit["move"]) is False, (
            f"ply {audit['ply']}: something was reused, but nothing was searched")
        assert search.nodes == 1, "a discarded tree should leave a single fresh root"
        assert search.root_fen == audit["fen"], (
            f"ply {audit['ply']}: the engine advanced to {search.root_fen} where "
            f"the reference is at {audit['fen']}")
    assert search.reuse_stats() == {**search.reuse_stats(),
                                    "discards": len(applied), "applies": 0}

    # And the fresh root is searchable, which is the point of not raising. The
    # position has to be one the reference expanded as a ROOT, or the replay
    # evaluator has nothing to answer with — that is a limit of the dump, not of
    # the engine, so it is selected for rather than tripped over.
    root_keys = {int(k) for k, r in zip(dump["keys"], dump["is_root"]) if r}
    for audit in applied:
        if guofish_core.nn_key(audit["fen"]) not in root_keys:
            continue
        probe = _new_search(game, dump, cache_slots=0)
        for step in applied:
            probe.apply_move(step["move"])
            if step is audit:
                break
        stats = probe.search(50)
        assert stats["root_visits"] == 50
        assert stats["best_move"] in guofish_core.legal_moves(probe.root_fen)
        return
    pytest.fail(
        "no position on the fifty-walk move list was expanded as a root by the "
        "reference, so a rebuilt tree could not be searched against the replay "
        "dump. The corpus is supposed to contain 17 of them.")


def test_the_promoted_root_carries_no_move(replayed):
    """A root is a root, whatever it used to be.

    The tree serialisation writes 0 at depth 0 on both sides, so a promoted node
    that kept the move it was reached by would put the arena one field away from
    what every reader of it believes. Every game in the corpus ends on an
    `apply_move`, so the final tree of each one is a promoted tree — read off the
    arena rather than off the serialisation, because the serialisation is what
    would paper over it.
    """
    for name, block in replayed.items():
        assert block["snapshots"][-1]["audit"]["kind"] == "apply"
        arrays = block["search"].dump_tree_arrays(0)
        assert int(arrays["move"][0]) == 0, (
            f"{name}: the promoted root still carries a move "
            f"({guofish_core.move_to_uci(int(arrays['move'][0]))})")
