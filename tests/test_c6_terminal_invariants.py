"""C6 acceptance — the terminal state cannot be spelled wrong.

This file is about STRUCTURE, not about numbers. `test_c6_gate1_full.py` proves
the C++ search reproduces the reference's trees bit for bit; this one proves that
the state which produced the engine's worst historical defect is *unrepresentable*
rather than merely absent.

The defect
----------
`bestmove 0000` — a forfeit, on a board with legal moves. The Python engine
reached it because `MCTSNode` overloaded one flag: a checkmate leaf was written
`is_expanded = True` with an empty `children` dict, since inside a search that
combination is harmless (selection stops on the empty dict either way). It stops
being harmless the moment such a node is PROMOTED TO ROOT. `apply_move` makes it
the root, `search()` sees `is_expanded` and skips `_expand_root`, then finds no
children and returns None. The comment at core/mctsv4.py:1222 is the reference's
own account of it.

Python's fix is a convention: the fifty-move / threefold path deliberately leaves
`is_expanded` alone so a promoted node is forced to expand, and `search()` /
`get_policy()` carry an `or not root.children` recovery for the checkmate path
that does set it. Conventions hold until someone edits them.

The C++ fix is structural, and it is what this file checks:

    NodeState is a three-valued LIFECYCLE in the low bits of an atomic byte.
    Terminal is a separate BIT in the high bit. They answer different questions
    and neither can stand in for the other.

    set_children()  refuses a zero count, refuses a terminal node, refuses a
                    node that is already expanded, refuses an out-of-range range.
    mark_terminal() refuses a node that has children.

So "expanded with no children" and "terminal with children" are both rejected at
the point of writing, not asserted about afterwards.

The three claims, and how each is shown
---------------------------------------
1. A node cannot be both terminal and expanded-with-children.
   Shown twice: on the arena directly, in both orders, and over every node of
   every tree the C6 corpus produces — a claim about the API and a claim about
   what the search actually built.

2. A terminal node promoted to root still yields a legal move.
   Shown by DOING IT. `terminal_nodes()` hands back each terminal node's FEN,
   carrying the search's own raw en-passant square AND its own halfmove clock —
   which is the part that matters, since a fifty-move draw node whose FEN said
   `0` would be a different position on the one axis that made it a draw. The
   FEN goes back into `set_position()` on a fresh search and the search must
   expand it and return a legal move.

3. The draw-by-rule value never leaves the tree node.
   Path-dependence is not a detail: the same position reached by a different line
   is not drawn. C7 owns the cache; what C6 owes is that the value the draw path
   produces is handed to `backpropagate` and to nothing else, which is visible
   here as a drawn terminal node whose position, evaluated on its own, is not
   over.

Environment overrides (Amendment B) are the same as the gate's, so a mutation
drill can point this file at a corrupted copy too.

Nothing here imports ``chess`` or ``torch``.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TREES = REPO_ROOT / "golden" / "gate1_terminal_trees.npz"
DEFAULT_DUMP = REPO_ROOT / "golden" / "gate1_terminal_dump.npz"
DEFAULT_MANIFEST = REPO_ROOT / "golden" / "gate1_terminal_manifest.json"

TREES_ENV = "GUOFISH_GOLDEN_C6_TREES"
DUMP_ENV = "GUOFISH_GOLDEN_C6_DUMP"
MANIFEST_ENV = "GUOFISH_GOLDEN_C6_MANIFEST"


def _path(env, default):
    return Path(os.environ.get(env, default))


def _missing(path, what):
    pytest.fail(
        f"{what} missing: {path}\n"
        "Global Rule 2: regenerate with "
        "`python tools/gen_gate1_golden.py --corpus terminal`, which runs the "
        "Python reference. It is never produced from C++ output."
    )


@pytest.fixture(scope="session")
def manifest():
    path = _path(MANIFEST_ENV, DEFAULT_MANIFEST)
    if not path.exists():
        _missing(path, "C6 terminal manifest")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def dump():
    path = _path(DUMP_ENV, DEFAULT_DUMP)
    if not path.exists():
        _missing(path, "C6 terminal replay dump")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def trees():
    path = _path(TREES_ENV, DEFAULT_TREES)
    if not path.exists():
        _missing(path, "C6 terminal reference trees")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# 1a. The arena API, driven directly
#
# These do not need golden data and do not need a search. They are the whole
# structural claim, in the smallest form it can be made.
# ---------------------------------------------------------------------------


def _arena(capacity=64):
    return guofish_core.NodeArenaDouble(capacity)


def test_a_node_cannot_be_expanded_with_zero_children():
    """THE DEFECT ITSELF. `bestmove 0000` is a root that reported itself expanded
    and had nothing to pick from; here the write is refused."""
    arena = _arena()
    parent = arena.allocate(1)
    with pytest.raises(ValueError) as excinfo:
        arena.set_children(parent, parent, 0)
    assert "zero children" in str(excinfo.value)
    assert "bestmove 0000" in str(excinfo.value), (
        "the rejection should name the defect it prevents, so a reader who hits "
        "it knows why the check is there")
    assert arena.children_count(parent) == 0
    assert arena.state(parent) == 0, "a refused write must not have changed the state"


def test_a_terminal_node_cannot_then_be_expanded():
    arena = _arena()
    node = arena.allocate(1)
    kids = arena.allocate(3)
    arena.mark_terminal(node, 1.0)
    assert arena.is_terminal(node)

    with pytest.raises(RuntimeError) as excinfo:
        arena.set_children(node, kids, 3)
    assert "terminal" in str(excinfo.value)
    assert arena.children_count(node) == 0
    assert arena.is_terminal(node), "the terminal bit must survive the refusal"


def test_an_expanded_node_cannot_then_be_marked_terminal():
    """The other order. A node with children has moves to play, so it is not a
    game result — and if this were permitted, the first order's guarantee could be
    walked around by doing the two writes the other way."""
    arena = _arena()
    node = arena.allocate(1)
    kids = arena.allocate(3)
    arena.set_children(node, kids, 3)

    with pytest.raises(RuntimeError) as excinfo:
        arena.mark_terminal(node, 0.0)
    assert "children" in str(excinfo.value)
    assert not arena.is_terminal(node)
    assert arena.children_count(node) == 3


def test_terminal_is_a_bit_beside_the_lifecycle_not_a_lifecycle_value():
    """The representation, not just the guards.

    If TERMINAL were a fourth NodeState the two questions would share one field
    and 'terminal' would necessarily overwrite 'unexpanded' — which is exactly
    what makes a promoted terminal node unable to say 'I have never been
    expanded, expand me'. So a terminal node must still report its lifecycle.
    """
    arena = _arena()
    node = arena.allocate(1)
    assert arena.lifecycle(node) == guofish_core.STATE_UNEXPANDED
    arena.mark_terminal(node, 0.0)
    assert arena.is_terminal(node)
    assert arena.lifecycle(node) == guofish_core.STATE_UNEXPANDED, (
        "a terminal node must remain UNEXPANDED, or promoting it to a root "
        "cannot force the expansion that finds its legal moves")
    assert arena.state(node) & guofish_core.TERMINAL_BIT, (
        "terminal must live in the high bit")
    assert arena.state(node) & guofish_core.STATE_LIFECYCLE_MASK == 0, (
        "the lifecycle bits must be untouched")
    assert not arena.is_expanded(node)


def test_the_terminal_bit_survives_a_pending_claim_cycle():
    """`try_claim_pending` moves Unexpanded -> Pending and `release_pending` moves
    it back. Neither may disturb the terminal bit, which lives in a different
    part of the byte — C9 runs both concurrently with terminal marking."""
    arena = _arena()
    node = arena.allocate(1)
    assert arena.try_claim_pending(node)
    assert arena.lifecycle(node) == guofish_core.STATE_PENDING
    arena.release_pending(node)
    assert arena.lifecycle(node) == guofish_core.STATE_UNEXPANDED
    arena.mark_terminal(node, 1.0)
    # A terminal node is not claimable: there is nothing to evaluate.
    assert arena.try_claim_pending(node) is False, (
        "a terminal node has no evaluation to claim")
    assert arena.is_terminal(node)
    assert arena.terminal_value(node) == 1.0


# ---------------------------------------------------------------------------
# 1b. The same claim over every node the C6 corpus actually built
# ---------------------------------------------------------------------------


def _golden_run(trees, index):
    """The per-node arrays for one recorded run, sliced out of the CSR block."""
    begin = int(trees["run_offset"][index])
    end = int(trees["run_offset"][index + 1])
    return {name: array[begin:end] for name, array in trees.items()
            if name != "run_offset"}


@pytest.fixture(scope="session")
def searched(dump, trees, manifest):
    """Every recorded run, re-run in C++. One engine per (virtual loss, cap).

    Virtual loss and max_tree_depth are both fixed at construction, so the engines
    are keyed by the pair. Arena capacity comes from the reference's own children
    column — `1 + sum(children)` is the exact node total, since an expansion
    allocates exactly its child count — so an over-allocating C++ expansion shows
    up as a failure rather than being absorbed by slack.
    """
    capacity = 0
    for index in range(len(manifest["runs"])):
        run = _golden_run(trees, index)
        capacity = max(capacity, 1 + int(run["children"].sum()))

    engines = {}
    results = {}
    positions = {p["index"]: p for p in manifest["positions"]}
    for index, run in enumerate(manifest["runs"]):
        key = (run["virtual_loss"], run["max_tree_depth"])
        if key not in engines:
            config = guofish_core.SearchConfig(virtual_loss=run["virtual_loss"],
                                               max_tree_depth=run["max_tree_depth"],
                                               arena_capacity=capacity)
            search = guofish_core.ReplaySearchDouble(config)
            search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                             dump["moves"], dump["priors"], dump["values"])
            engines[key] = search
        search = engines[key]
        search.set_position(run["fen"], positions[run["position"]]["history"])
        stats = search.search(run["sims"])
        results[index] = {
            "stats": stats,
            "arrays": search.dump_tree_arrays(0 if run["full_tree"] else 1),
            "terminals": search.terminal_nodes(),
            "run": run,
        }
    return results


def test_no_node_in_any_corpus_tree_is_terminal_with_children(searched):
    """THE ACCEPTANCE CRITERION, over real trees.

    The API guards above say the state cannot be written. This says the search
    never tried — across every node of every run, including the drawn terminals
    the fifty-move and threefold paths create, which are the ones Python leaves
    unexpanded on purpose.
    """
    seen_terminal = 0
    for index, case in searched.items():
        arrays = case["arrays"]
        both = np.flatnonzero((arrays["terminal"] != 0) & (arrays["children"] != 0))
        assert both.size == 0, (
            f"run {index} ({case['run']['name']}, VL {case['run']['virtual_loss']}): "
            f"{both.size} node(s) are terminal AND have children — this is the "
            f"bestmove 0000 state. First at DFS index {int(both[0])}, "
            f"{int(arrays['children'][both[0]])} children.")
        seen_terminal += int((arrays["terminal"] != 0).sum())

    assert seen_terminal > 0, (
        "no terminal node in the whole corpus, so this assertion proved nothing. "
        "The C6 corpus is supposed to be full of them.")


def test_every_terminal_node_reports_itself_unexpanded(searched):
    """The bit and the lifecycle, read back off real nodes.

    `terminal_nodes()` reports `expanded` from the lifecycle field, separately
    from the terminal bit. Every terminal node must be Unexpanded — that is what
    makes promotion recoverable, and it is the one place C++ deliberately does
    NOT copy the reference (which sets `is_expanded = True` on the checkmate
    path). The behaviour is identical because selection stops on the empty child
    list either way; the difference is that here the bad state does not exist.
    """
    total = 0
    for index, case in searched.items():
        for node in case["terminals"]:
            assert node["children"] == 0, (
                f"run {index}: terminal node at {node['path']} has "
                f"{node['children']} children")
            assert node["expanded"] is False, (
                f"run {index}: terminal node at {node['path']} reports itself "
                f"expanded")
            total += 1
    assert total > 0, "no terminal nodes found; the corpus is not what it claims"


# ---------------------------------------------------------------------------
# 2. Promotion
# ---------------------------------------------------------------------------


def _drawn_terminal_probes(searched, limit=40):
    """Terminal nodes carrying value 0.0, with their FENs.

    Value 0.0 is the interesting class. A +1.0 terminal is a checkmate and has no
    legal moves BY DEFINITION, so "promoted to root, still yields a move" is not
    a claim anyone can make about it. 0.0 covers stalemate (also no moves — and
    the test below separates the two on that basis), insufficient material, the
    fifty-move rule and threefold repetition. The last three are exactly the
    positions a host may decline to end, which is when a promotion happens for
    real.
    """
    seen = {}
    for case in searched.values():
        for node in case["terminals"]:
            if node["value"] == 0.0 and node["fen"] not in seen:
                seen[node["fen"]] = node
            if len(seen) >= limit:
                return list(seen.values())
    return list(seen.values())


def test_a_terminal_node_promoted_to_root_still_yields_a_legal_move(searched, dump,
                                                                    manifest):
    """THE ACCEPTANCE CRITERION, performed rather than asserted.

    Take a node the search marked terminal by a CLAIMABLE rule — fifty-move or
    threefold, the two a host can decline — hand its FEN back as a root, and
    require a real expansion and a real move. This is the exact sequence that
    produced `bestmove 0000`: `apply_move` promotes the node, `search()` runs on
    it, and something has to generate moves.

    Positions with no legal moves (checkmate, stalemate) are excluded here and
    checked separately below, because for them "yields a move" is not a property
    anyone can have.
    """
    probes = _drawn_terminal_probes(searched)
    assert probes, "no drawn terminal nodes in the corpus to promote"

    playable = [p for p in probes if guofish_core.legal_moves(p["fen"])]
    assert playable, (
        "every drawn terminal in the corpus is a position with no legal moves, "
        "so none of them can demonstrate the promotion property")

    # Only positions the reference actually expanded as a ROOT can be replayed —
    # the dump is the evaluator, and a position it never saw at the root is a
    # miss by design. The generator runs a promotion probe for exactly this.
    root_keys = set(int(k) for k, r in zip(dump["keys"], dump["is_root"]) if r)
    promoted = 0
    for probe in playable:
        if guofish_core.nn_key(probe["fen"]) not in root_keys:
            continue
        search = guofish_core.ReplaySearchDouble(
            guofish_core.SearchConfig(arena_capacity=1 << 16))
        search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                         dump["moves"], dump["priors"], dump["values"])
        # Promoted with NO history: the point is that the position alone, with
        # the claim declined and the game restarted from here, is playable.
        search.set_position(probe["fen"])
        stats = search.search(2)

        assert stats["root_visits"] >= 1
        arrays = search.dump_tree_arrays(0)
        assert int(arrays["children"][0]) > 0, (
            f"a terminal node promoted to root produced no children: "
            f"{probe['fen']} (was terminal by value {probe['value']} at "
            f"{probe['path']})")
        assert int(arrays["terminal"][0]) == 0, (
            "a promoted root that expanded must not still carry the terminal "
            "bit; the reference clears it in _expand_root for the same reason")
        assert stats["best_move"] is not None, (
            f"promoted root {probe['fen']} returned no best move — this is "
            f"bestmove 0000")
        assert stats["best_move"] in guofish_core.legal_moves(probe["fen"]), (
            f"promoted root {probe['fen']} returned {stats['best_move']}, which "
            f"is not legal there")
        promoted += 1

    assert promoted > 0, (
        "no drawn terminal node in the corpus was also expanded as a root by the "
        "reference, so the promotion could not be performed against the replay "
        "evaluator. The generator is supposed to guarantee at least one.")


def test_a_checkmate_or_stalemate_terminal_genuinely_has_no_moves(searched):
    """The other half of the promotion story, and the reason the engine is
    allowed to concede on these.

    `bestmove 0000` was wrong because legal moves existed. Where they genuinely do
    not, there is nothing to play and the position really is over. Separating the
    two is what keeps the test above honest: without this, 'excluded because it
    has no legal moves' would be an untested excuse.
    """
    checked = 0
    for case in searched.values():
        for node in case["terminals"]:
            if node["value"] != 1.0:
                continue
            assert guofish_core.legal_moves(node["fen"]) == [], (
                f"a node marked terminal with a WIN has legal moves: "
                f"{node['fen']} at {node['path']}")
            assert guofish_core.terminal_reason(node["fen"]) == "checkmate", (
                f"a +1.0 terminal must be checkmate; "
                f"{node['fen']} is {guofish_core.terminal_reason(node['fen'])}")
            checked += 1
    assert checked > 0, "no checkmate terminals in the corpus"


# ---------------------------------------------------------------------------
# 3. The claimable draws are path-dependent and stay on the node
# ---------------------------------------------------------------------------


def test_a_claimable_draw_is_not_a_property_of_the_position(searched):
    """The type discipline C7 has to inherit, demonstrated on real nodes.

    A fifty-move or threefold node is drawn because of the PATH that reached it.
    Ask the position on its own — which is all a position-keyed cache could ever
    do — and it says the game is not over. So the value cannot be stored anywhere
    keyed by position, and here it is not: `draw_by_rule` returns a bool, the
    caller backs 0.0 up and marks the tree node, and nothing else sees it.
    """
    path_dependent = 0
    for case in searched.values():
        for node in case["terminals"]:
            if node["value"] != 0.0:
                continue
            if guofish_core.terminal_reason(node["fen"]) != "none":
                # Stalemate or insufficient material: intrinsically over, and the
                # reference caches those on the node too. Not this test's case.
                continue
            assert guofish_core.legal_moves(node["fen"]), (
                f"{node['fen']} is not over by any positional rule and yet has "
                f"no legal moves")
            path_dependent += 1

    assert path_dependent > 0, (
        "no path-dependent draw in the corpus. The fifty-move and threefold "
        "specs are supposed to produce them, and without one this test proves "
        "nothing about path dependence.")


def test_the_recorded_history_is_load_bearing(dump, manifest):
    """The pre-root history is an INPUT, and dropping it must change the answer.

    This is the only thing in the repository that can catch the history argument
    being ignored. Every position in C5's quiet corpus has an empty history, so a
    ``set_position`` that silently dropped it would pass the whole of Gate 1's
    quiet half — and would then, in a real game, decline threefolds that had
    actually occurred.

    Every position carrying a history is run twice, once with it and once
    without, and at least one must answer differently. It is deliberately "at
    least one" rather than "all": whether a given root's history is reachable
    inside search range is a property of that position's tree, not of the
    plumbing. Measured on this corpus, two of the three threefold specs diverge
    (both by walking into a position the reference never evaluated, which is a
    ReplayMiss and therefore the loudest possible difference) and the third does
    not — its draws all come from repetitions deeper in the line, where the
    pre-root history contributes nothing.
    """
    positions = {p["index"]: p for p in manifest["positions"]}
    with_history_runs = [p for p in positions.values() if p["history"]]
    assert with_history_runs, "no position in the corpus carries a history"

    def verdict(run, history, sims):
        search = guofish_core.ReplaySearchDouble(
            guofish_core.SearchConfig(virtual_loss=run["virtual_loss"],
                                      max_tree_depth=run["max_tree_depth"],
                                      arena_capacity=1 << 19))
        search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                         dump["moves"], dump["priors"], dump["values"])
        search.set_position(run["fen"], history)
        try:
            stats = search.search(sims)
        except RuntimeError as exc:
            # A replay miss: without the history the search walked into a
            # position the reference never evaluated. That IS the divergence.
            return ("replay-miss", str(exc).splitlines()[0])
        return (stats["draw_by_rule_hits"], stats["nodes"])

    differing = []
    report = []
    for spec in with_history_runs:
        run = next(r for r in manifest["runs"]
                   if r["position"] == spec["index"] and not r["full_tree"]
                   and r["virtual_loss"] == 0.0)
        # A short run: the point is made in the first few hundred simulations and
        # this fixture is not the gate.
        sims = min(200, run["sims"])
        kept = verdict(run, spec["history"], sims)
        dropped = verdict(run, [], sims)
        report.append(f"  {spec['name']:24s} with={kept}  without={dropped}")
        if kept != dropped:
            differing.append(spec["name"])

    assert differing, (
        "dropping the pre-root repetition history changed nothing on any "
        "position that has one, which means either set_position is ignoring the "
        "argument or no corpus history is reachable inside search range:\n" +
        "\n".join(report))


def test_the_threefold_branch_actually_fires(searched):
    """``_draw_by_rule`` has two branches and they must be told apart.

    A C++ that answered the fifty-move question correctly and never reached the
    repetition question would match ``draw_by_rule_hits`` on the fifty-move specs
    and fail only on the threefold ones — so the split is asserted rather than the
    total.
    """
    threefold = 0
    fifty = 0
    for (index, case) in searched.items():
        stats = case["stats"]
        threefold += stats["threefold_hits"]
        fifty += stats["fifty_move_hits"]
    assert threefold > 0, "the repetition branch of _draw_by_rule never fired"
    assert fifty > 0, "the fifty-move branch of _draw_by_rule never fired"


# ---------------------------------------------------------------------------
# The depth cap is NOT a terminal, which is a separate structural claim
# ---------------------------------------------------------------------------


def test_the_depth_cap_does_not_mark_a_node_terminal(searched, manifest):
    """A capped node is not a game result.

    The reference backs up 0.0 and leaves the node alone — no terminal mark, no
    expansion — precisely so that the position remains searchable if it is ever
    reached again with budget left, or promoted. Marking it would be a lie about
    the position that survives into whatever reads the tree next.

    Checked on the depth-cap specs, where the cap is the ONLY thing that can
    fire: those positions are C5 corpus midgames the C5 audit already certified
    as reaching no terminal and no draw at full depth.
    """
    capped_runs = [i for i, r in enumerate(manifest["runs"])
                   if "depthcap" in r["categories"]]
    assert capped_runs, "no depth-cap runs in the corpus"

    total_hits = 0
    for index in capped_runs:
        case = searched[index]
        stats = case["stats"]
        arrays = case["arrays"]
        run = case["run"]

        assert int(arrays["terminal"].sum()) == 0, (
            f"run {index} ({run['name']}) marked "
            f"{int(arrays['terminal'].sum())} node(s) terminal. The cap must "
            f"not mark, and these positions have no terminal in range.")
        assert stats["depth_cap_hits"] > 0, (
            f"run {index} ({run['name']}) is a depth-cap spec at cap "
            f"{run['max_tree_depth']} and the cap never fired")
        assert stats["max_depth"] <= run["max_tree_depth"], (
            f"run {index} descended to {stats['max_depth']}, past its cap of "
            f"{run['max_tree_depth']}")
        total_hits += stats["depth_cap_hits"]

    assert total_hits > 0


def test_a_capped_node_is_left_expandable(searched, manifest):
    """The consequence of not marking: a node the cap stopped at is Unexpanded
    with no children and no terminal bit, so it is indistinguishable from a leaf
    the search has simply not got to yet. That is the correct state — it IS one.
    """
    index = next(i for i, r in enumerate(manifest["runs"])
                 if "depthcap" in r["categories"] and not r["full_tree"])
    case = searched[index]
    arrays = case["arrays"]
    cap = case["run"]["max_tree_depth"]

    frontier = np.flatnonzero((arrays["depth"] == cap) & (arrays["visits"] > 0))
    assert frontier.size > 0, (
        f"no visited node at the cap depth {cap}; the cap cannot have fired")
    assert np.all(arrays["children"][frontier] == 0), (
        "a node the cap stopped at must have no children")
    assert np.all(arrays["terminal"][frontier] == 0), (
        "a node the cap stopped at must not be marked terminal")
    assert np.all(arrays["terminal_value"][frontier] == 0.0)
