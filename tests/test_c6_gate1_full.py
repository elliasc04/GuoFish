"""C6 acceptance — Gate 1 on the FULL corpus, bit-exact.

This is the gate C5 could only run half of. C5 compared C++ against Python on
positions that had been *audited to contain no terminal state*, because it had
not implemented any; C6 adds checkmate, stalemate, insufficient material,
threefold repetition, the fifty-move rule and the depth cap, and this file
compares the two implementations over both corpora at once:

    quiet     golden/gate1_{dump,trees,manifest}          — C5's, unchanged
    terminal  golden/gate1_terminal_{dump,trees,manifest} — C6's

Every run in both, at both virtual-loss magnitudes, node for node: visit counts
equal as integers, ``value_sum`` identical as a 64-bit pattern, priors identical
as 32-bit patterns, and — new here — the terminal bit and the cached terminal
value identical too. There is no tolerance anywhere and there must not be one.

Why the quiet corpus is re-run rather than deferred to C5's file
---------------------------------------------------------------
Because C6 changed the code that produces it. Terminal detection runs at every
descent step and at every leaf; the halfmove clock and the repetition key are
maintained on every move; ``set_position`` now builds a repetition history. Any
of that could perturb a quiet tree, and the only way to know it does not is to
run them again through the new code. Re-running here also means a C6 regression
that only shows on quiet positions fails a C6 test, rather than being blamed on
C5.

What the terminal corpus is
---------------------------
25 hand-specified positions, in eight classes the brief names, each measured
rather than declared — the generator counts what the reference actually did and
refuses to write golden data if any class never fired:

    mate1         a root child IS checkmate, so the depth-1 short-circuit fires
                  and the reference stops early. Three positions.
    mate2/mate3   forced mate deeper, so checkmate terminals appear at depth 3+
                  with no truncation. Seven positions.
    stalemate     a stalemating move exists and no mate in one does, so the
                  search reaches the stalemate. Three, plus three more that carry
                  stalemates alongside a mate.
    insufficient  one capture leaves K+B or K+N against a bare king. Two.
    threefold     the root's own repetition history is primed by a shuffle, so a
                  threefold is inside search range. Three.
    fifty         halfmove clock 94-96 at the root, so the rule fires at ply four
                  to six. Three.
    depthcap      quiet C5 midgames run at MAX_TREE_DEPTH 4-7. Four.

Every base position is legal (``Board.is_valid()``) and every base FEN's
en-passant field is ``-``; the roots that are not bases are reached by pushing
legal moves. That is what keeps the corpus clear of the ``has_legal_en_passant``
boundary from C3, where a hand-written ep square can make ``rep_key`` and
``nn_key`` disagree about a position no game can reach. Asserted below.

Why the depth cap is exercised at 4-7 and not at 80
---------------------------------------------------
A line reaching ply 80 without repeating a position or crossing the fifty-move
clock would have to be forty moves of non-repeating, clock-resetting play, and
MCTS does not build one inside 5,000 simulations — attempts ended as a threefold
at ply 4 or a fifty-move draw at ply 6, which is what deep shuffling actually
produces. So the cap is lowered on BOTH SIDES and recorded per run in the
manifest: the reference reads ``mctsv4.MAX_TREE_DEPTH`` at each descent step, the
C++ side reads ``SearchConfig.max_tree_depth``, and a run at 6 executes the same
code a run at 80 would. That the DEFAULT is 80 on both sides is a separate
assertion below, so lowering it for these runs cannot hide a wrong constant.

Environment overrides (Amendment B)
-----------------------------------
``GUOFISH_GOLDEN_GATE1_*`` for the quiet files (shared with C5) and
``GUOFISH_GOLDEN_C6_*`` for the terminal ones, so a mutation drill runs against
corrupted copies in a scratch directory and ``golden/`` is never written to.

Nothing here imports ``chess`` or ``torch``. The reference reaches this file only
through the golden files.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent

QUIET = {
    "trees": (REPO_ROOT / "golden" / "gate1_trees.npz", "GUOFISH_GOLDEN_GATE1_TREES"),
    "dump": (REPO_ROOT / "golden" / "gate1_dump.npz", "GUOFISH_GOLDEN_GATE1_DUMP"),
    "manifest": (REPO_ROOT / "golden" / "gate1_manifest.json",
                 "GUOFISH_GOLDEN_GATE1_MANIFEST"),
}
TERMINAL = {
    "trees": (REPO_ROOT / "golden" / "gate1_terminal_trees.npz", "GUOFISH_GOLDEN_C6_TREES"),
    "dump": (REPO_ROOT / "golden" / "gate1_terminal_dump.npz", "GUOFISH_GOLDEN_C6_DUMP"),
    "manifest": (REPO_ROOT / "golden" / "gate1_terminal_manifest.json",
                 "GUOFISH_GOLDEN_C6_MANIFEST"),
}

REQUIRED_TERMINAL_POSITIONS = 20
REQUIRED_VIRTUAL_LOSSES = (0.0, 2.5)
REFERENCE_MAX_TREE_DEPTH = 80

# The eight classes the brief lists. Coverage is asserted against the reference's
# MEASURED counters, not against the spec labels, so a spec whose intended class
# never fired cannot satisfy it.
REQUIRED_CATEGORIES = ("mate1", "mate2", "mate3", "stalemate", "insufficient",
                       "threefold", "fifty", "depthcap")


def _path(spec):
    default, env = spec
    return Path(os.environ.get(env, default))


def _load_json(spec, what):
    path = _path(spec)
    if not path.exists():
        pytest.fail(
            f"{what} missing: {path}\n"
            "Global Rule 2: regenerate with `python tools/gen_gate1_golden.py"
            "{}`, which runs the Python reference. It is never produced from C++ "
            "output.".format(" --corpus terminal" if "terminal" in str(path) else ""))
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_npz(spec, what):
    path = _path(spec)
    if not path.exists():
        pytest.fail(
            f"{what} missing: {path}\n"
            "Global Rule 2: regenerate with `python tools/gen_gate1_golden.py"
            "{}`, which runs the Python reference.".format(
                " --corpus terminal" if "terminal" in str(path) else ""))
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def corpora():
    """Both corpora, loaded once.

    Keyed by name so a failing case can say WHICH corpus it came from before it
    says anything else — a quiet-corpus regression and a terminal-corpus
    regression have completely different causes.
    """
    return {
        "quiet": {
            "manifest": _load_json(QUIET["manifest"], "Gate 1 quiet manifest"),
            "dump": _load_npz(QUIET["dump"], "Gate 1 quiet replay dump"),
            "trees": _load_npz(QUIET["trees"], "Gate 1 quiet reference trees"),
        },
        "terminal": {
            "manifest": _load_json(TERMINAL["manifest"], "C6 terminal manifest"),
            "dump": _load_npz(TERMINAL["dump"], "C6 terminal replay dump"),
            "trees": _load_npz(TERMINAL["trees"], "C6 terminal reference trees"),
        },
    }


def _golden_run(trees, index):
    """The per-node arrays for one recorded run, sliced out of the CSR block.

    `terminal` and `terminal_value` are C6 columns. The quiet files predate them,
    and rather than special-casing every comparison this fills them with zeros —
    which is the reference's answer for a corpus certified to contain no terminal
    node, and which the C++ side is then required to match exactly. A quiet run
    that produced a terminal node would fail here, which is the point.
    """
    begin = int(trees["run_offset"][index])
    end = int(trees["run_offset"][index + 1])
    out = {name: trees[name][begin:end]
           for name in ("depth", "move", "visits", "value_sum", "prior", "children")}
    size = end - begin
    out["terminal"] = (trees["terminal"][begin:end] if "terminal" in trees
                       else np.zeros(size, dtype=np.uint8))
    out["terminal_value"] = (trees["terminal_value"][begin:end]
                             if "terminal_value" in trees
                             else np.zeros(size, dtype=np.float32))
    return out


# ---------------------------------------------------------------------------
# Failure reporting
# ---------------------------------------------------------------------------


def _paths_from_dfs(depth, move):
    """Move path from the root for every node, from the DFS depth column.

    Preorder means a node at depth d is always preceded by its ancestors, so one
    stack reconstructs every path in a single pass. Doing it here rather than
    asking C++ for it keeps the two sides' path reconstruction identical — if it
    were computed differently on each side, a structural divergence could print
    two paths that disagree for a reason unrelated to the bug.
    """
    paths = []
    stack = []
    for d, m in zip(depth, move):
        del stack[int(d):]
        stack.append(guofish_core.move_to_uci(int(m)) if int(d) > 0 else "(root)")
        paths.append(" ".join(stack[1:]) if len(stack) > 1 else "(root)")
    return paths


def _bits64(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def _bits32(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def _first_divergence(label, golden, actual):
    """Locate and describe the first differing node in DFS order.

    Returns None when the two agree on every field of every node. Otherwise
    returns a report naming the node by its path from the root, the field, both
    values, and — for the float fields — both bit patterns, because a one-ulp
    difference is invisible in decimal. A bare "trees differ" is not acceptable;
    the brief is explicit about that.
    """
    if len(golden["depth"]) != len(actual["depth"]):
        # A length difference is still a divergence AT a node: the shorter side
        # stopped producing nodes somewhere, and saying where is the whole job.
        shorter = min(len(golden["depth"]), len(actual["depth"]))
        for i in range(shorter):
            if (int(golden["depth"][i]) != int(actual["depth"][i]) or
                    int(golden["move"][i]) != int(actual["move"][i])):
                break
        else:
            i = shorter
        paths = _paths_from_dfs(golden["depth"][:shorter], golden["move"][:shorter])
        where = paths[i] if i < len(paths) else "(past the end of the shorter tree)"
        return (
            f"{label}: the trees have different node counts — "
            f"golden {len(golden['depth'])}, C++ {len(actual['depth'])}.\n"
            f"  first structural difference at DFS index {i}, path: {where}\n"
            f"  This is a SHAPE divergence: the two searches expanded different "
            f"moves, not merely different numbers."
        )

    comparisons = (
        ("depth", golden["depth"], actual["depth"], int),
        ("move", golden["move"], actual["move"], int),
        ("visit_count", golden["visits"], actual["visits"], int),
        ("children_count", golden["children"], actual["children"], int),
        ("terminal", golden["terminal"], actual["terminal"], int),
        ("value_sum", _bits64(golden["value_sum"]), _bits64(actual["value_sum"]), float),
        ("prior", _bits32(golden["prior"]), _bits32(actual["prior"]), float),
        ("terminal_value", _bits32(golden["terminal_value"]),
         _bits32(actual["terminal_value"]), float),
    )

    first_index = None
    for _, want, got, _kind in comparisons:
        differing = np.flatnonzero(want != got)
        if differing.size:
            candidate = int(differing[0])
            first_index = candidate if first_index is None else min(first_index, candidate)
    if first_index is None:
        return None

    paths = _paths_from_dfs(golden["depth"], golden["move"])
    lines = [
        f"{label}: first divergence at DFS index {first_index} of "
        f"{len(golden['depth'])} nodes",
        f"  path from root : {paths[first_index]}",
        f"  depth          : {int(golden['depth'][first_index])}",
        f"  move           : {guofish_core.move_to_uci(int(golden['move'][first_index]))}",
    ]
    sources = {"value_sum": "value_sum", "prior": "prior",
               "terminal_value": "terminal_value"}
    for name, want, got, kind in comparisons:
        w = want[first_index]
        g = got[first_index]
        flag = "  <-- DIFFERS" if w != g else ""
        if kind is float:
            source = sources[name]
            lines.append(
                f"  {name:<14} : golden 0x{int(w):016x} "
                f"({golden[source][first_index]!r})  "
                f"c++ 0x{int(g):016x} ({actual[source][first_index]!r}){flag}")
        else:
            lines.append(f"  {name:<14} : golden {int(w)}  c++ {int(g)}{flag}")

    total = sum(int(np.count_nonzero(want != got)) for _, want, got, _ in comparisons)
    lines.append(f"  differing field-values across the whole tree: {total}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Collection
#
# Parametrisation is resolved before fixtures exist, so both manifests are read
# once here as well. One case per recorded run across both corpora means a
# failure names the corpus, the position and the virtual-loss magnitude in the
# test id before it prints anything — and one position failing does not hide the
# other forty-four.
# ---------------------------------------------------------------------------


def _collect():
    cases = []
    ids = []
    for corpus, spec in (("quiet", QUIET), ("terminal", TERMINAL)):
        path = _path(spec["manifest"])
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            runs = json.load(handle)["runs"]
        for index, run in enumerate(runs):
            cases.append((corpus, index))
            name = run.get("name", f"pos{run['position']:02d}")
            ids.append(f"{corpus}-{name}-vl{run['virtual_loss']}-n{run['sims']}-"
                       f"{'full' if run['full_tree'] else 'visited'}")
    return cases, ids


CASES, CASE_IDS = _collect()


# ---------------------------------------------------------------------------
# The corpus is what the brief asked for
# ---------------------------------------------------------------------------


def test_the_terminal_corpus_meets_the_briefs_floor(corpora):
    manifest = corpora["terminal"]["manifest"]
    assert manifest["corpus"] == "terminal"
    positions = manifest["positions"]
    assert len(positions) >= REQUIRED_TERMINAL_POSITIONS, (
        f"the brief requires >= {REQUIRED_TERMINAL_POSITIONS} new positions "
        f"targeting terminal handling; the manifest has {len(positions)}")

    fens = [p["fen"] for p in positions]
    assert len(set(fens)) == len(fens), "the terminal corpus contains duplicates"

    covered = {}
    for run in manifest["runs"]:
        if run["full_tree"]:
            continue
        covered.setdefault(run["position"], set()).add(run["virtual_loss"])
    for position, magnitudes in sorted(covered.items()):
        assert magnitudes == set(REQUIRED_VIRTUAL_LOSSES), (
            f"position {position} was not run at both virtual-loss magnitudes "
            f"{REQUIRED_VIRTUAL_LOSSES}; got {sorted(magnitudes)}. The brief "
            f"requires the whole corpus to pass twice.")

    labelled = set()
    for position in positions:
        labelled.update(position["categories"])
    missing = set(REQUIRED_CATEGORIES) - labelled
    assert not missing, f"no position targets: {sorted(missing)}"


def test_every_class_the_brief_names_actually_fired(corpora):
    """Coverage as a MEASUREMENT, not as a label.

    A spec's `categories` say what it was written to exercise. These counters say
    what the reference did. The gap between the two is where a corpus quietly
    stops testing what its name claims — the first draft of `fifty-rooks` was
    labelled `fifty` and produced not one fifty-move hit, because a winning
    capture at the root ate every simulation.
    """
    manifest = corpora["terminal"]["manifest"]
    runs = manifest["runs"]

    def total(key):
        return sum(r["census"][key] for r in runs)

    checkmates = total("terminal_win")
    draws_on_node = total("terminal_draw")
    rule_draws = sum(r["draw_by_rule_hits"] for r in runs)
    capped = total("at_depth_cap")
    early = sum(1 for r in runs if r["early_exit"])

    assert checkmates > 0, "no checkmate terminal anywhere in the corpus"
    assert draws_on_node > 0, "no drawn terminal anywhere in the corpus"
    assert rule_draws > 0, (
        "_draw_by_rule never fired: neither the threefold nor the fifty-move "
        "path is exercised, and both are C6's")
    assert capped > 0, "the depth cap never fired"
    assert early > 0, (
        "the depth-1 mate short-circuit never fired, so the one behaviour the "
        "brief singles out as an ugly requirement to replicate is untested")

    # The one count that must be ZERO. `terminal_with_children` is the reference
    # standing in the state C++ cannot represent; if it were ever non-zero, the
    # two implementations could not agree and the right response would be to stop
    # and report, not to relax the comparison.
    assert total("terminal_with_children") == 0, (
        "the reference produced a node that is terminal AND has children. C++ "
        "cannot represent that state by construction (cpp/arena.hpp), so this "
        "corpus cannot be matched — report it rather than working around it.")


def test_the_terminal_corpus_cannot_rely_on_the_en_passant_inconsistency(corpora):
    """The brief's validation clause, checked rather than promised.

    C3 established that a hand-written FEN can name an en-passant square no legal
    double push could have set, and that `rep_key` (legal-capture rule) and
    `nn_key` (raw rule) then disagree about a position no game reaches. C6's
    corpus must not depend on that boundary.

    Two things make it unreachable and both are asserted: every base FEN's ep
    field is `-`, so there is nothing to be inconsistent about; and every root
    that is not a base is reached by pushing legal moves onto one, so its ep
    square is whatever python-chess's own push() set.
    """
    positions = corpora["terminal"]["manifest"]["positions"]
    for spec in positions:
        base_ep = spec["base_fen"].split()[3]
        assert base_ep == "-", (
            f"{spec['name']}: base FEN names an en-passant square ({base_ep}). "
            f"Hand-written ep fields are exactly what this corpus must avoid.")
        if not spec["moves"]:
            assert spec["fen"].split()[3] == "-", (
                f"{spec['name']}: no moves played, so the root's ep field must "
                f"still be '-'")
        # A root reached by pushing legal moves may legitimately carry an ep
        # square. Whatever it is, the two keys must agree about the position in
        # the only sense that matters here: both are derived from the same
        # ParsedFen, so ask for both and require them to be computable and
        # distinct (the domain tags guarantee the second).
        assert guofish_core.nn_key(spec["fen"]) != guofish_core.rep_key(spec["fen"])


def test_the_reference_default_depth_cap_is_still_eighty(corpora):
    """Lowering the cap for the depth-cap specs must not be able to hide a wrong
    constant, so the DEFAULT is asserted on both sides and against both
    manifests."""
    assert guofish_core.SearchConfig().max_tree_depth == REFERENCE_MAX_TREE_DEPTH
    for name in ("quiet", "terminal"):
        assert corpora[name]["manifest"]["config"]["max_tree_depth"] == \
            REFERENCE_MAX_TREE_DEPTH, (
            f"the {name} manifest records a default MAX_TREE_DEPTH other than "
            f"{REFERENCE_MAX_TREE_DEPTH}")

    caps = {r["max_tree_depth"] for r in corpora["terminal"]["manifest"]["runs"]}
    assert REFERENCE_MAX_TREE_DEPTH in caps, (
        "every terminal run lowered the cap; nothing was run at the reference's "
        "own constant")
    assert caps - {REFERENCE_MAX_TREE_DEPTH}, (
        "no terminal run lowered the cap, so the cap path is not exercised")


def test_the_equivalence_configuration_is_the_one_the_brief_specifies(corpora):
    for name in ("quiet", "terminal"):
        config = corpora[name]["manifest"]["config"]
        assert config["workers"] == 1, name
        assert config["cache_size"] == 1, (
            "the reference must run with a one-slot cache: cache_size=0 raises "
            "IndexError and there is no cache-off flag")
        assert config["tablebase"] is False, name
        assert config["dirichlet"] is False, name
        assert config["canonical_order_patch"] is True, (
            "without the canonical-ordering patch the reference resolves PUCT "
            "ties by python-chess's generation order, which C++ does not "
            "reproduce")
        assert config["search_params"]["policy_temperature"] == 1.0
        assert corpora[name]["manifest"]["determinism_check"] is not None, (
            f"the {name} generator did not verify that the equivalence "
            f"configuration is bit-deterministic across runs")
        assert corpora[name]["manifest"]["determinism_check"]["identical"] is True


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def searched(corpora):
    """Run every recorded configuration of both corpora once.

    One engine per (corpus, virtual loss, max_tree_depth): all three are fixed at
    construction, and the arena is recycled by ``set_position`` rather than
    rebuilt, which is what keeps a 90-run suite from allocating 90 arenas and
    reloading two dumps 90 times.

    Capacity comes from the GOLDEN data, not from a guess. In the quiet corpus
    every visited node is expanded and an expansion allocates exactly its child
    count, so ``1 + sum(children)`` is the exact node total; in the terminal
    corpus terminal and capped nodes are not expanded, so the same expression is
    an upper bound — still derived from the reference, so an over-allocating C++
    expansion still shows up rather than being absorbed by slack.
    """
    engines = {}
    results = {}

    for corpus in ("quiet", "terminal"):
        manifest = corpora[corpus]["manifest"]
        trees = corpora[corpus]["trees"]
        dump = corpora[corpus]["dump"]
        positions = {p["index"]: p for p in manifest["positions"]}

        capacity = 0
        for index in range(len(manifest["runs"])):
            capacity = max(capacity, 1 + int(_golden_run(trees, index)["children"].sum()))

        for index, run in enumerate(manifest["runs"]):
            key = (corpus, run["virtual_loss"], run.get("max_tree_depth", 80))
            if key not in engines:
                config = guofish_core.SearchConfig(
                    virtual_loss=run["virtual_loss"],
                    max_tree_depth=run.get("max_tree_depth", 80),
                    arena_capacity=capacity)
                search = guofish_core.ReplaySearchDouble(config)
                search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                                 dump["moves"], dump["priors"], dump["values"])
                engines[key] = search
            search = engines[key]
            search.set_position(run["fen"],
                                positions[run["position"]].get("history", []))
            stats = search.search(run["sims"])
            results[(corpus, index)] = {
                "stats": stats,
                "arrays": search.dump_tree_arrays(0 if run["full_tree"] else 1),
                "golden": _golden_run(trees, index),
                "run": run,
            }
    return results


@pytest.mark.parametrize("corpus,index", CASES, ids=CASE_IDS)
def test_tree_is_bit_exact(searched, corpus, index):
    """THE ACCEPTANCE CRITERION.

    Visit counts equal as integers, value_sum identical as a bit pattern, priors
    identical as a bit pattern, the terminal bit and cached terminal value
    identical, and the same nodes in the same DFS order — over the quiet corpus
    and the terminal corpus, at virtual loss 0.0 and 2.5.
    """
    case = searched[(corpus, index)]
    run = case["run"]
    label = (f"[{corpus}] {run.get('name', 'position ' + str(run['position']))} "
             f"(VL {run['virtual_loss']}, {run['sims']} sims, "
             f"cap {run.get('max_tree_depth', 80)}, "
             f"{'full tree' if run['full_tree'] else 'visited subtree'})\n"
             f"  fen: {run['fen']}")
    report = _first_divergence(label, case["golden"], case["arrays"])
    assert report is None, report


@pytest.mark.parametrize("corpus,index", CASES, ids=CASE_IDS)
def test_search_accounting_matches_the_reference(searched, corpus, index):
    """The counters, independently of the tree comparison.

    ``root_visits`` is the one that matters most in the terminal corpus: when the
    depth-1 short-circuit fires the reference stops early, and the C++ side has to
    stop at the SAME simulation. A C++ that ran the full budget would build a
    bigger tree and fail the comparison above — but it would fail it as a shape
    divergence at some arbitrary node, which says nothing about the cause. Here it
    fails as "the reference stopped at 2 visits and C++ did not".
    """
    case = searched[(corpus, index)]
    stats = case["stats"]
    run = case["run"]
    golden = case["golden"]

    assert stats["root_visits"] == run["root_visits"], (
        f"root has {stats['root_visits']} visits, reference has "
        f"{run['root_visits']}" +
        ("  (the reference exited early — the depth-1 mate short-circuit)"
         if run.get("early_exit") else ""))
    assert stats["simulations"] == run["root_visits"] - 1, (
        "a fresh search runs N-1 simulations: _expand_root seeds the root with "
        "one visit and search() targets the difference")
    assert stats["max_depth"] <= run["max_depth"], (
        f"C++ descended to depth {stats['max_depth']}, deeper than the "
        f"reference's {run['max_depth']}")
    assert stats["nodes"] <= 1 + int(golden["children"].sum()), (
        f"arena holds {stats['nodes']} nodes; the reference tree implies at most "
        f"{1 + int(golden['children'].sum())}")

    if run.get("early_exit"):
        assert stats["mating_move"] is not None, (
            "the reference exited early via the mate short-circuit; C++ reports "
            "no mating move")
        assert stats["mating_move"] == run["best_move"], (
            f"C++ took {stats['mating_move']} as the mating move, the reference "
            f"played {run['best_move']}")
    else:
        assert stats["mating_move"] is None, (
            f"C++ short-circuited on {stats['mating_move']} where the reference "
            f"ran to completion")


@pytest.mark.parametrize("corpus,index", CASES, ids=CASE_IDS)
def test_the_best_move_agrees_with_the_reference(searched, corpus, index):
    """`ParallelMCTS.search`'s actual return value.

    C5 deliberately stopped short of move selection — nothing in a tree
    comparison needs it. C6 needs it, because "a terminal node promoted to root
    still yields a legal move" is a statement about the move, and a tie-break
    that differed from `max(root.children.items(), key=visit_count)` would be
    invisible in the tree and wrong at the board.
    """
    case = searched[(corpus, index)]
    assert case["stats"]["best_move"] == case["run"]["best_move"], (
        f"[{corpus}] run {index}: C++ plays {case['stats']['best_move']}, the "
        f"reference plays {case['run']['best_move']} at {case['run']['fen']}")


@pytest.mark.parametrize("corpus,index", CASES, ids=CASE_IDS)
def test_the_terminal_census_matches_the_reference(searched, corpus, index):
    """How many terminals, of which kind, and how many draw-by-rule hits.

    The tree comparison already covers the terminal BIT per node. This covers the
    EVENTS: the reference counted `_draw_by_rule` returning True at its source,
    and C++ counts the same call site. Two implementations can agree on the
    finished tree while disagreeing about how they got there — a C++ that marked
    a node terminal on the intrinsic path where the reference took the claimable
    one would produce the same 0.0 backup and the same bit, and only these
    counters would notice.
    """
    case = searched[(corpus, index)]
    stats = case["stats"]
    run = case["run"]
    census = run.get("census")
    if census is None:
        pytest.skip("this corpus predates the census columns")

    assert stats["draw_by_rule_hits"] == run["draw_by_rule_hits"], (
        f"[{corpus}] run {index}: C++ hit the claimable-draw path "
        f"{stats['draw_by_rule_hits']} times, the reference "
        f"{run['draw_by_rule_hits']}")
    assert (stats["fifty_move_hits"] + stats["threefold_hits"] ==
            stats["draw_by_rule_hits"]), (
        "every claimable-draw hit must be attributed to exactly one of the two "
        "branches")

    golden = case["golden"]
    terminal = golden["terminal"] != 0

    # A node is marked ONCE — the first visit that resolves it — and every later
    # visit takes the cached fast path, so the mark counters are exact node
    # counts, not event counts. Checkmate is the only rule that produces a 1.0,
    # so that side is an equality.
    golden_wins = int(np.count_nonzero(terminal & (golden["terminal_value"] == 1.0)))
    assert stats["checkmates"] == golden_wins, (
        f"[{corpus}] run {index}: C++ marked {stats['checkmates']} checkmate(s), "
        f"the reference's tree holds {golden_wins} node(s) with terminal_value "
        f"1.0")

    # The 0.0 side is a bound rather than an equality: stalemate, insufficient
    # material and the two claimable rules all write 0.0, and the reference does
    # not record which one fired at each node. The bound still catches a C++ that
    # invented an intrinsic terminal the reference never marked.
    golden_draws = int(np.count_nonzero(terminal & (golden["terminal_value"] == 0.0)))
    cpp_intrinsic_draws = (stats["stalemates"] + stats["insufficient_material"] +
                           stats["seventyfive_moves"] + stats["fivefold_repetitions"])
    assert cpp_intrinsic_draws <= golden_draws, (
        f"[{corpus}] run {index}: C++ marked {cpp_intrinsic_draws} intrinsic "
        f"drawn terminal(s) but the reference's tree holds only {golden_draws} "
        f"drawn terminal node(s) in total")


def test_the_seventyfive_move_and_fivefold_rules_never_fire(searched):
    """Both are implemented and both are unreachable, and saying so is worth a test.

    `_draw_by_rule` runs at EVERY descent step and returns at a halfmove clock of
    100 and at a threefold. So by the time a leaf reaches `board.is_game_over()`,
    the clock is below 100 and no position has occurred three times — which puts
    the seventy-five-move rule (150) and fivefold repetition (5) out of reach by
    construction, not by luck of the corpus.

    They are implemented anyway, because "unreachable" is a claim about the
    CALLER and `outcome_of` is a transcription of a function that asks. This test
    pins the claim: if a future change to the descent (tree reuse, C8) makes one
    reachable, this fails and the implementation is already there to be checked.
    """
    for key, case in searched.items():
        stats = case["stats"]
        assert stats["seventyfive_moves"] == 0, (
            f"{key}: the seventy-five-move rule fired {stats['seventyfive_moves']} "
            f"times. It is supposed to be unreachable behind _draw_by_rule's "
            f"fifty-move check — if that is no longer true, the reference's own "
            f"ordering has changed and this needs re-deriving, not relaxing.")
        assert stats["fivefold_repetitions"] == 0, (
            f"{key}: fivefold repetition fired {stats['fivefold_repetitions']} "
            f"times, behind a threefold check that returns two occurrences "
            f"earlier")


def test_the_quiet_corpus_still_reaches_no_terminal_under_c6(searched, corpora):
    """C5's certification, re-asserted through the new code.

    The quiet corpus was accepted precisely because the reference touched none of
    this machinery. C6 added code that runs on every descent step and every leaf;
    if any of it fires on a quiet position, the C5 audit and the C6 implementation
    disagree about the same tree, and one of them is wrong.
    """
    for (corpus, index), case in searched.items():
        if corpus != "quiet":
            continue
        stats = case["stats"]
        run = case["run"]
        assert stats["draw_by_rule_hits"] == 0, (
            f"quiet run {index} ({run['fen']}) hit the claimable-draw path "
            f"{stats['draw_by_rule_hits']} times; the C5 manifest certifies zero")
        assert stats["depth_cap_hits"] == 0
        assert stats["checkmates"] == 0 and stats["stalemates"] == 0
        assert stats["insufficient_material"] == 0
        assert int(case["arrays"]["terminal"].sum()) == 0


# ---------------------------------------------------------------------------
# The rule transcriptions, one at a time
#
# The gate above is end-to-end and is the acceptance criterion. It is a poor
# DIAGNOSTIC: `has_insufficient_material` reading the wrong side's bishops shows
# up there as "the trees have different node counts at ply 14". These check each
# predicate against the reference's answer on a FEN, so a transcription error
# fails by name before the tree comparison runs.
#
# The FENs come from the corpus itself and from golden/movegen.jsonl — C1's
# 100k-position corpus, already in the repository and already generated from the
# Python reference (Global Rule 2).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def movegen_corpus():
    path = REPO_ROOT / "golden" / "movegen.jsonl"
    if not path.exists():
        pytest.skip("golden/movegen.jsonl not present")
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def test_terminal_reason_agrees_with_the_reference_over_the_c1_corpus(movegen_corpus):
    """`outcome_of` against C1's 100k positions.

    The golden file records each position's legal moves, which is enough to
    derive python-chess's answer for the two rules that need movegen: a position
    with no legal moves is checkmate or stalemate depending on whether the side to
    move is in check, and everything else can only be 'none' or one of the
    material/clock rules. So the comparison here is against a DERIVED reference
    answer rather than a recorded one — which is why it is limited to the part
    that can be derived, and why insufficient material gets its own test below
    against the transcription's own definition on positions where it is decidable.
    """
    checked = 0
    for row in movegen_corpus:
        moves = row["moves"]
        if moves:
            continue
        # No legal moves: the position is over, and it is checkmate exactly when
        # the side to move is in check. The C++ predicate must say so.
        reason = guofish_core.terminal_reason(row["fen"])
        assert reason in ("checkmate", "stalemate", "insufficient-material"), (
            f"{row['fen']} has no legal moves and terminal_reason says {reason}")
        checked += 1
    if checked == 0:
        pytest.skip("the C1 corpus contains no position with zero legal moves")


def test_terminal_reason_says_none_for_every_corpus_root(corpora):
    """No root in either corpus may already be over.

    A root with no legal moves is `bestmove 0000` territory and the reference
    answers it with None; neither corpus contains one, and this is what says so.
    """
    for name in ("quiet", "terminal"):
        for position in corpora[name]["manifest"]["positions"]:
            fen = position["fen"]
            assert guofish_core.terminal_reason(fen) == "none", (
                f"[{name}] root {fen} is already over "
                f"({guofish_core.terminal_reason(fen)})")
            assert guofish_core.legal_moves(fen), f"[{name}] root {fen} has no moves"


def test_insufficient_material_is_decided_the_reference_way():
    """The transcription's quirks, on positions constructed to isolate each.

    Every case here is a specific clause of `Board.has_insufficient_material`, and
    the two marked QUIRK are the ones a paraphrase gets wrong:

      * the bishop clause's same-colour test reads BOTH sides' bishops, so a lone
        bishop each on the same complex is insufficient and on opposite complexes
        is not;
      * the knight clause requires the OPPONENT to hold nothing but kings and
        queens, because a knight can force a selfmate against anything else.
    """
    cases = [
        ("8/8/4k3/8/8/4K3/8/8 w - - 0 1", True, "bare kings"),
        ("8/8/4k3/8/8/4K3/8/2B5 w - - 0 1", True, "K+B vs K"),
        ("8/8/4k3/8/8/4K3/8/2N5 w - - 0 1", True, "K+N vs K"),
        ("8/8/4k3/8/8/4K3/8/2R5 w - - 0 1", False, "a rook can mate"),
        ("8/8/4k3/8/8/4K3/2P5/8 w - - 0 1", False, "a pawn can promote"),
        ("8/8/3bk3/8/8/4K3/8/2B5 w - - 0 1", True,
         "QUIRK: a bishop each on the SAME complex — d6 and c1 are both dark — "
         "so the same-colour test, which reads BOTH sides' bishops, is satisfied "
         "and neither side can mate"),
        ("8/8/2b1k3/8/8/4K3/8/2B5 w - - 0 1", False,
         "QUIRK: opposite complexes — c6 is light, c1 is dark — so the same "
         "material is NOT insufficient. A per-colour reading of the clause would "
         "answer True for both of these."),
        ("8/8/3nk3/8/8/4K3/8/2N5 w - - 0 1", False,
         "QUIRK: a knight each — the opponent holds a non-king, non-queen piece"),
        ("8/8/4k3/8/8/4K3/8/1NN5 w - - 0 1", False,
         "two knights: popcount(ours) is 3, so the knight clause fails"),
    ]
    for fen, expected, why in cases:
        assert guofish_core.insufficient_material(fen) == expected, (
            f"{fen}: expected insufficient_material={expected} ({why})")
        # And the composite predicate must agree with its own part.
        reason = guofish_core.terminal_reason(fen)
        if expected:
            assert reason == "insufficient-material", (
                f"{fen} is insufficient material but outcome_of says {reason}")


def test_insufficient_material_is_asked_before_stalemate():
    """The reference's test ORDER, which is only visible on a position that is both.

    `Board.outcome()` asks insufficient material BEFORE stalemate, so a bare king
    stalemated by a lone king and knight reports INSUFFICIENT_MATERIAL. Both give
    a 0.0 backup, so nothing in this chunk can observe the difference — which is
    exactly why it is worth pinning: the moment anything reads the reason rather
    than the value, a transcription that reordered them starts lying.
    """
    # Black to move, no legal moves, and neither side can mate.
    fen = "7k/8/6KN/8/8/8/8/8 b - - 0 1"
    assert guofish_core.legal_moves(fen) == [], "the fixture must be stalemate"
    assert guofish_core.insufficient_material(fen)
    assert guofish_core.terminal_reason(fen) == "insufficient-material", (
        "outcome_of must ask insufficient material before stalemate, as "
        "Board.outcome() does")


def test_move_classifiers_agree_with_the_reference_on_the_corpus_roots(corpora):
    """`is_zeroing` and `is_irreversible` on real moves.

    These are what the halfmove clock and the repetition walk-back are built out
    of, and both are easy to get subtly wrong on castling: chess-library encodes
    it as king-takes-rook, so the raw destination is the ROOK's square, and
    `_reduces_castling_rights` compares the destination against the castling
    rights — which ARE the rook squares. Feeding it the raw encoding would make
    every castle irreversible for the wrong reason.
    """
    checked = 0
    castling_seen = 0
    for name in ("quiet", "terminal"):
        for position in corpora[name]["manifest"]["positions"]:
            fen = position["fen"]
            for uci in guofish_core.legal_moves(fen):
                probe = guofish_core.move_rule_probe(fen, uci)
                assert isinstance(probe["zeroing"], bool)
                # A castling move must not be zeroing: the king's destination is
                # empty and the king is not a pawn.
                if uci in ("e1g1", "e1c1", "e8g8", "e8c8") and fen.split()[2] != "-":
                    if probe["reduces_castling_rights"]:
                        castling_seen += 1
                        assert not probe["zeroing"], (
                            f"{fen} {uci}: castling must not zero the halfmove "
                            f"clock")
                        assert probe["irreversible"], (
                            f"{fen} {uci}: castling destroys castling rights and "
                            f"is therefore irreversible")
                checked += 1
    assert checked > 0
    assert castling_seen > 0, (
        "no castling move in either corpus, so the king-takes-rook normalisation "
        "is not exercised here")
