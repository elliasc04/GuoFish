"""C5 acceptance — Gate 1 on quiet positions, bit-exact.

This is the equivalence gate. For every position in the corpus, at both virtual
loss magnitudes, the C++ tree must match the Python reference's node for node:
visit counts equal as integers, `value_sum` identical as a 64-bit pattern, and
priors identical as 32-bit patterns. There is no tolerance anywhere in this file
and there must not be one — a single ulp in one prior changes which child wins a
PUCT comparison, and from there the two trees diverge structurally rather than
numerically.

What makes the comparison meaningful
------------------------------------
The network is not in the loop. Python ran once with the real checkpoint and
dumped ``nn_key -> (legal moves, priors, value)``; C++ replays that dump as its
evaluator. So the two sides cannot disagree about what the network said, and any
divergence is provably in selection, expansion, virtual loss, backup or child
ordering — which is the only thing Gate 1 was ever meant to test.

That also means the dump lookup is itself a test, and the strongest one here.
C++ reaches each position by ``makeMove`` from the root and tokenizes its own
search state; if that tokenization derives a different en-passant square, the
key it computes is not in the dump and the search fails by name. See
``test_a_missing_dump_entry_fails_loudly_with_the_path``.

How a failure is reported
-------------------------
The brief is explicit that a bare "trees differ" is an immediate fail, so
``_first_divergence`` locates the first differing node in DFS order and prints
the move path from the root, the field that differs and both sides' values —
including the raw bit patterns for the float fields, because a difference of one
ulp prints identically in decimal.

What the corpus is
------------------
20 quiet midgame positions sampled from the 200-game fixed-node benchmark match,
each of which was RUN and audited before being accepted: any position whose
reference search touched a checkmate, stalemate, repetition, fifty-move or
depth-cap path was rejected, because C5 does not implement that machinery (it is
C6's). Seven candidates were rejected on exactly those grounds; they are listed
in the manifest. The runs are:

* 20 positions x 2 virtual-loss magnitudes at 5,000 simulations, recording the
  VISITED subtree (a 5,000-sim tree holds ~175,000 nodes, of which ~5,000 have
  any visits; the rest carry visit 0 and value_sum 0.0 on both sides by
  construction).
* 4 positions x 2 magnitudes at 800 simulations recording EVERY node, unvisited
  children included. Those runs are what make the whole-tree claim measured
  rather than argued: unvisited children have non-trivial priors, in an order
  that is exactly the canonical ordering under test.

Why VL 2.5 is not redundant
---------------------------
At VL 0.0 virtual loss contributes nothing to selection, so that run cannot fail
for a virtual-loss reason. At 2.5 it can: ``select_child`` reads the parent's
``effective_visits``, and a parent always carries one applied virtual loss
mid-descent, so even at one worker the exploration term sees ``N + 1*2.5``.
Single-threaded Python is still deterministic there, so it must also be
bit-exact, and it is the only place virtual loss's apply/repay ordering is
exercised before concurrency arrives in C9.

Environment overrides (Amendment B)
-----------------------------------
``GUOFISH_GOLDEN_GATE1_TREES``, ``GUOFISH_GOLDEN_GATE1_DUMP`` and
``GUOFISH_GOLDEN_GATE1_MANIFEST`` point the suite at copies elsewhere, so the
mutation drill runs against a corrupted copy in a scratch directory and
``golden/`` is never written to.

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
DEFAULT_TREES = REPO_ROOT / "golden" / "gate1_trees.npz"
DEFAULT_DUMP = REPO_ROOT / "golden" / "gate1_dump.npz"
DEFAULT_MANIFEST = REPO_ROOT / "golden" / "gate1_manifest.json"

TREES_ENV = "GUOFISH_GOLDEN_GATE1_TREES"
DUMP_ENV = "GUOFISH_GOLDEN_GATE1_DUMP"
MANIFEST_ENV = "GUOFISH_GOLDEN_GATE1_MANIFEST"

# The brief's floor. Asserted rather than assumed, so a golden file regenerated
# with smaller arguments cannot quietly turn this into a weaker gate.
REQUIRED_POSITIONS = 20
REQUIRED_SIMS = 5000
REQUIRED_VIRTUAL_LOSSES = (0.0, 2.5)


def _path(env, default):
    return Path(os.environ.get(env, default))


def _missing(path, what):
    pytest.fail(
        f"{what} missing: {path}\n"
        "Global Rule 2: regenerate with `python tools/gen_gate1_golden.py`, which "
        "runs the Python reference. It is never produced from C++ output."
    )


@pytest.fixture(scope="session")
def manifest():
    path = _path(MANIFEST_ENV, DEFAULT_MANIFEST)
    if not path.exists():
        _missing(path, "Gate 1 manifest")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def dump():
    path = _path(DUMP_ENV, DEFAULT_DUMP)
    if not path.exists():
        _missing(path, "Gate 1 replay dump")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def trees():
    path = _path(TREES_ENV, DEFAULT_TREES)
    if not path.exists():
        _missing(path, "Gate 1 reference trees")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _golden_run(trees, index):
    """The per-node arrays for one recorded run, sliced out of the CSR block."""
    begin = int(trees["run_offset"][index])
    end = int(trees["run_offset"][index + 1])
    return {name: trees[name][begin:end]
            for name in ("depth", "move", "visits", "value_sum", "prior", "children")}


@pytest.fixture(scope="session")
def engines(dump, trees, manifest):
    """One search per virtual-loss magnitude, dump loaded once.

    Virtual loss is fixed at construction (it is part of the config the arena is
    built with), so there is one engine per magnitude rather than one per run.
    The arena is sized to the largest run in the corpus and recycled by
    ``set_position``, which is what keeps a 48-run suite from allocating 48
    arenas and reloading a 17 MB dump 48 times.

    Capacity comes from the GOLDEN data, not from a guess: every visited node in
    a quiet tree is expanded, and an expansion allocates exactly its child count,
    so ``1 + sum(children)`` is the exact node total. Taking it from the
    reference means an over-allocating C++ expansion shows up as an equality
    failure below rather than being absorbed by slack.
    """
    capacity = 0
    for index in range(len(manifest["runs"])):
        run = _golden_run(trees, index)
        capacity = max(capacity, 1 + int(run["children"].sum()))

    built = {}
    for virtual_loss in REQUIRED_VIRTUAL_LOSSES:
        config = guofish_core.SearchConfig(virtual_loss=virtual_loss,
                                           arena_capacity=capacity)
        search = guofish_core.ReplaySearchDouble(config)
        search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                         dump["moves"], dump["priors"], dump["values"])
        built[virtual_loss] = search
    return built


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
    difference is invisible in decimal.
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

    # Compared in DFS order, field by field, so the first index that differs in
    # ANY field is the first divergent node. Bit patterns for the floats.
    comparisons = (
        ("depth", golden["depth"], actual["depth"], int),
        ("move", golden["move"], actual["move"], int),
        ("visit_count", golden["visits"], actual["visits"], int),
        ("children_count", golden["children"], actual["children"], int),
        ("value_sum", _bits64(golden["value_sum"]), _bits64(actual["value_sum"]), float),
        ("prior", _bits32(golden["prior"]), _bits32(actual["prior"]), float),
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
    for name, want, got, kind in comparisons:
        w = want[first_index]
        g = got[first_index]
        flag = "  <-- DIFFERS" if w != g else ""
        if kind is float:
            source = "value_sum" if name == "value_sum" else "prior"
            lines.append(
                f"  {name:<14} : golden 0x{int(w):016x} "
                f"({golden[source][first_index]!r})  "
                f"c++ 0x{int(g):016x} ({actual[source][first_index]!r}){flag}")
        else:
            lines.append(f"  {name:<14} : golden {int(w)}  c++ {int(g)}{flag}")

    total = sum(int(np.count_nonzero(want != got)) for _, want, got, _ in comparisons)
    lines.append(f"  differing field-values across the whole tree: {total}")
    return "\n".join(lines)


# Parametrisation has to be resolved at collection time, before fixtures exist,
# so the manifest is read once here as well. One case per recorded run means a
# failure names the position and the virtual-loss magnitude in the test id,
# before it prints anything — and it means one position failing does not hide
# the other nineteen.
def _collect_runs():
    path = _path(MANIFEST_ENV, DEFAULT_MANIFEST)
    if not path.exists():
        # The `manifest` fixture reports the missing file properly; this only
        # has to keep collection from raising.
        return [], []
    with open(path, encoding="utf-8") as handle:
        runs = json.load(handle)["runs"]
    ids = [f"pos{r['position']:02d}-vl{r['virtual_loss']}-n{r['sims']}-"
           f"{'full' if r['full_tree'] else 'visited'}" for r in runs]
    return list(range(len(runs))), ids


RUN_INDICES, RUN_IDS = _collect_runs()


# ---------------------------------------------------------------------------
# The corpus is what the brief asked for
# ---------------------------------------------------------------------------


def test_corpus_meets_the_briefs_floor(manifest):
    """A golden file regenerated with smaller arguments must not silently turn
    this suite into a weaker gate."""
    assert len(manifest["positions"]) >= REQUIRED_POSITIONS, (
        f"Gate 1 requires >= {REQUIRED_POSITIONS} positions, manifest has "
        f"{len(manifest['positions'])}")

    fens = [p["fen"] for p in manifest["positions"]]
    assert len(set(fens)) == len(fens), "the corpus contains duplicate positions"

    main_runs = [r for r in manifest["runs"] if not r["full_tree"]]
    assert main_runs, "no 5,000-simulation runs recorded"
    for run in main_runs:
        assert run["sims"] >= REQUIRED_SIMS, (
            f"position {run['position']} ran {run['sims']} simulations, "
            f"below the {REQUIRED_SIMS} the brief requires")

    covered = {}
    for run in main_runs:
        covered.setdefault(run["position"], set()).add(run["virtual_loss"])
    for position, magnitudes in sorted(covered.items()):
        assert magnitudes == set(REQUIRED_VIRTUAL_LOSSES), (
            f"position {position} was not run at both virtual-loss magnitudes "
            f"{REQUIRED_VIRTUAL_LOSSES}; got {sorted(magnitudes)}")


def test_corpus_is_certified_quiet(manifest):
    """Every accepted run touched none of the machinery C5 does not implement.

    This is what licenses the omission. The generator audits each finished
    reference tree and rejects the position outright on any terminal mark,
    depth-cap signature or draw-by-rule hit; the counters it recorded are
    re-asserted here so the licence travels with the data instead of living in a
    commit message.
    """
    for run in manifest["runs"]:
        assert run["draw_by_rule_hits"] == 0, (
            f"position {run['position']} at VL {run['virtual_loss']} hit "
            f"_draw_by_rule {run['draw_by_rule_hits']} times; repetition and "
            f"fifty-move are C6's and are not implemented in C5")
        assert run["max_depth"] < manifest["config"]["max_tree_depth"], (
            f"position {run['position']} reached depth {run['max_depth']}, at or "
            f"past MAX_TREE_DEPTH")
        assert run["root_visits"] == run["sims"]

    assert manifest["rejected"], (
        "no candidate positions were rejected, which means the quietness audit "
        "either never fired or was not applied — a corpus selected without it "
        "cannot be trusted to contain no terminals")


def test_the_equivalence_configuration_is_the_one_the_brief_specifies(manifest):
    config = manifest["config"]
    assert config["workers"] == 1
    assert config["cache_size"] == 1, (
        "the reference must run with a one-slot cache: cache_size=0 raises "
        "IndexError and there is no cache-off flag")
    assert config["tablebase"] is False
    assert config["dirichlet"] is False
    assert config["canonical_order_patch"] is True, (
        "without the canonical-ordering patch the reference resolves PUCT ties "
        "by python-chess's generation order, which C++ does not reproduce")
    assert manifest["determinism_check"] is not None, (
        "the generator did not verify that the equivalence configuration is "
        "bit-deterministic across runs")
    assert manifest["determinism_check"]["identical"] is True


def test_cpp_defaults_match_the_reference_search_params(manifest):
    """The C++ defaults are the reference's, so a run that forgets to pass a
    parameter is not silently searching with a different one."""
    reference = manifest["config"]["search_params"]
    config = guofish_core.SearchConfig()
    assert config.c_init == reference["c_init"]
    assert config.c_base == reference["c_base"]
    assert config.fpu_root == reference["fpu_root"]
    assert config.fpu_tree == reference["fpu_tree"]
    assert config.max_tree_depth == manifest["config"]["max_tree_depth"]
    assert config.c_factor == 1.0
    assert reference["policy_temperature"] == 1.0, (
        "the golden run used a policy temperature other than 1.0; C5 replays "
        "pre-softmaxed priors and cannot reproduce sharpening")


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def searched(engines, trees, manifest):
    """Run every recorded configuration once, and hand the results to the tests.

    Session-scoped because the assertions below slice the same trees several
    ways and re-running 240,000 simulations per assertion would be the only slow
    thing in the suite.
    """
    results = {}
    for index, run in enumerate(manifest["runs"]):
        search = engines[run["virtual_loss"]]
        search.set_position(run["fen"])
        stats = search.search(run["sims"])
        arrays = search.dump_tree_arrays(0 if run["full_tree"] else 1)
        results[index] = {"stats": stats, "arrays": arrays,
                          "golden": _golden_run(trees, index), "run": run}
    return results


@pytest.mark.parametrize("index", RUN_INDICES, ids=RUN_IDS)
def test_tree_is_bit_exact(searched, index):
    """THE ACCEPTANCE CRITERION.

    Visit counts equal as integers, value_sum identical as a bit pattern, priors
    identical as a bit pattern, and the same nodes in the same DFS order. One
    parametrised case per recorded run, so a failure names the position and the
    virtual-loss magnitude before it says anything else.
    """
    case = searched[index]
    run = case["run"]
    label = (f"position {run['position']} (VL {run['virtual_loss']}, "
             f"{run['sims']} sims, "
             f"{'full tree' if run['full_tree'] else 'visited subtree'})\n"
             f"  fen: {run['fen']}")
    report = _first_divergence(label, case["golden"], case["arrays"])
    assert report is None, report


@pytest.mark.parametrize("index", RUN_INDICES, ids=RUN_IDS)
def test_search_accounting_matches_the_reference(searched, index):
    """The counters, independently of the tree comparison.

    ``nodes`` is the one worth staring at: every visited node in a quiet tree is
    expanded and an expansion allocates exactly its child count, so the arena
    total is a pure function of the reference's own children column. An
    expansion that allocated the wrong width would show up here even if the DFS
    comparison somehow did not.
    """
    case = searched[index]
    stats = case["stats"]
    run = case["run"]
    golden = case["golden"]

    assert stats["root_visits"] == run["sims"], (
        f"root has {stats['root_visits']} visits, reference has {run['sims']}")
    assert stats["simulations"] == run["sims"] - 1, (
        "a fresh search runs N-1 simulations: _expand_root seeds the root with "
        "one visit and search() targets the difference")
    assert stats["depth_cap_hits"] == 0, (
        f"the depth cap fired {stats['depth_cap_hits']} times on a position the "
        f"reference certified as never reaching it")
    assert stats["max_depth"] <= run["max_depth"], (
        f"C++ descended to depth {stats['max_depth']}, deeper than the "
        f"reference's {run['max_depth']}")
    assert stats["nodes"] == 1 + int(golden["children"].sum()), (
        f"arena holds {stats['nodes']} nodes; the reference tree implies "
        f"{1 + int(golden['children'].sum())}")


def test_root_value_and_visit_seeding_match(searched, manifest):
    """_expand_root assigns rather than accumulates, and seeds the root's value
    in the mover's perspective. Checked separately because the root is the one
    node whose value does not arrive through backpropagate()."""
    for index, case in searched.items():
        golden = case["golden"]
        # The root is DFS index 0 in both serialisations, at depth 0.
        assert int(golden["depth"][0]) == 0
        assert int(case["arrays"]["depth"][0]) == 0
        assert _bits64(golden["value_sum"][:1])[0] == _bits64(
            case["arrays"]["value_sum"][:1])[0], (
            f"run {index}: root value_sum differs")


def test_dump_tree_tuple_surface_agrees_with_the_array_surface(engines, manifest):
    """The brief specifies dump_tree() as (move, visit_count, value_sum, prior)
    tuples in canonical DFS order. The gate above compares NumPy arrays for
    speed, so this checks the specified surface reports the same thing rather
    than being decorative."""
    run = min(manifest["runs"], key=lambda r: r["sims"])
    search = engines[run["virtual_loss"]]
    search.set_position(run["fen"])
    search.search(run["sims"])

    tuples = search.dump_tree(1)
    arrays = search.dump_tree_arrays(1)

    assert len(tuples) == len(arrays["depth"])
    for i, (move, visits, value_sum, prior) in enumerate(tuples):
        assert move == guofish_core.move_to_uci(int(arrays["move"][i]))
        assert visits == int(arrays["visits"][i])
        assert value_sum == arrays["value_sum"][i]
        assert prior == arrays["prior"][i]

    # min_visits=0 must be a superset: every node, unvisited children included.
    assert len(search.dump_tree(0)) > len(tuples)


def test_visited_and_full_traversals_agree_where_they_overlap(engines, manifest):
    """The visited subtree must be the full tree filtered, in the same order —
    otherwise the 5,000-simulation runs and the 800-simulation full-tree runs
    would be testing two different traversals."""
    run = min(manifest["runs"], key=lambda r: r["sims"])
    search = engines[run["virtual_loss"]]
    search.set_position(run["fen"])
    search.search(run["sims"])

    full = search.dump_tree_arrays(0)
    visited = search.dump_tree_arrays(1)
    keep = full["visits"] > 0

    assert np.array_equal(full["move"][keep], visited["move"])
    assert np.array_equal(full["visits"][keep], visited["visits"])
    assert np.array_equal(_bits64(full["value_sum"][keep]),
                          _bits64(visited["value_sum"]))
    # Depth is NOT expected to survive filtering in general — a visited node
    # whose parent is unvisited cannot exist, so here it does.
    assert np.array_equal(full["depth"][keep], visited["depth"])


# ---------------------------------------------------------------------------
# The failure paths, which the brief requires to be loud
# ---------------------------------------------------------------------------


def test_a_missing_dump_entry_fails_loudly_with_the_path(dump, manifest):
    """A lookup miss must never return a default value.

    Constructed deterministically: load a dump holding ONLY the root entries, so
    the first simulation descends exactly one ply and misses. The message then
    has to name a real move in its path, which is what proves the path is
    reported rather than merely mentioned.

    This is the mechanism that makes the replay dump audit in-search
    tokenization for free — but only if a miss is loud.
    """
    keep = dump["is_root"] != 0
    offsets = dump["move_offset"]
    moves = []
    priors = []
    new_offset = [0]
    for i in np.flatnonzero(keep):
        begin, end = int(offsets[i]), int(offsets[i + 1])
        moves.append(dump["moves"][begin:end])
        priors.append(dump["priors"][begin:end])
        new_offset.append(new_offset[-1] + (end - begin))

    search = guofish_core.ReplaySearchDouble(
        guofish_core.SearchConfig(arena_capacity=4096))
    search.load_dump(dump["keys"][keep], dump["is_root"][keep],
                     np.array(new_offset, dtype=np.uint64),
                     np.concatenate(moves), np.concatenate(priors),
                     dump["values"][keep])

    run = manifest["runs"][0]
    search.set_position(run["fen"])
    with pytest.raises(RuntimeError) as excinfo:
        search.search(run["sims"])

    message = str(excinfo.value)
    assert "replay dump miss" in message
    assert "nn_key : 0x" in message, "the miss must report the key in hex"
    assert "fen    :" in message, "the miss must report the position"

    path_line = next(line for line in message.splitlines() if "path   :" in line)
    path = path_line.split(":", 1)[1].strip()
    assert path != "(root)", (
        "the root entries were kept, so the miss must have happened at least one "
        f"ply down and the path must name that move; got {path!r}")
    assert 4 <= len(path) <= 5 and path[0].isalpha(), (
        f"expected a single UCI move on the path, got {path!r}")


def test_a_corrupted_dump_move_list_is_caught_rather_than_misaligned(dump, manifest):
    """Priors are matched to moves by position, so a move list that does not
    agree with the one C++ generated would silently land 37 priors on the wrong
    37 moves. The dump entry's move list is therefore compared, not trusted."""
    run = manifest["runs"][0]
    root_slots = np.flatnonzero(dump["is_root"] != 0)

    moves = dump["moves"].copy()
    # Corrupt the first move of every root entry. One of them is this position's.
    for i in root_slots:
        begin = int(dump["move_offset"][i])
        moves[begin] = np.uint16((int(moves[begin]) + 16) & 0xFFFF)

    search = guofish_core.ReplaySearchDouble(
        guofish_core.SearchConfig(arena_capacity=4096))
    search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                     moves, dump["priors"], dump["values"])
    search.set_position(run["fen"])

    with pytest.raises(RuntimeError) as excinfo:
        search.search(run["sims"])
    message = str(excinfo.value)
    assert "legal-move mismatch" in message
    assert "C++      :" in message and "golden   :" in message, (
        "the mismatch must print both move lists")


def _full_tree_run_index(manifest):
    for index, run in enumerate(manifest["runs"]):
        if run["full_tree"]:
            return index
    pytest.fail("no full-tree run recorded; the drill needs one to locate the "
                "dump entry a given position's root actually uses")
    raise AssertionError  # unreachable; keeps type checkers quiet


def _root_slot_for(dump, golden):
    """Which dump entry is THIS position's root entry.

    The dump is sorted by (nn_key, is_root), so entry order says nothing about
    which position an entry belongs to — the first attempt at this drill mutated
    a root entry belonging to a rejected candidate and changed nothing, which is
    exactly the way a mutation drill fails to be a drill.

    A full-tree run records every child of the root, so the root's children in
    canonical order are a fingerprint: find the root entry whose move list is
    that list. Root entries number in the tens, so the scan is free.
    """
    wanted = golden["move"][golden["depth"] == 1]
    for slot in np.flatnonzero(dump["is_root"] != 0):
        begin = int(dump["move_offset"][slot])
        end = int(dump["move_offset"][slot + 1])
        if end - begin == len(wanted) and np.array_equal(dump["moves"][begin:end], wanted):
            return int(slot), begin, end
    pytest.fail("no dump root entry matches this position's root children")
    raise AssertionError  # unreachable


@pytest.mark.parametrize("field", ["prior-one-ulp", "interior-values"])
def test_a_mutated_dump_diverges_the_tree(dump, trees, manifest, field):
    """The mutation drill, in memory: perturb the dump and the gate must fail.

    Amendment B keeps drills out of ``golden/`` entirely; doing it on the loaded
    arrays is stricter still, since nothing is written anywhere and the real
    file's bytes are never at risk. The point is that
    ``test_tree_is_bit_exact`` CAN fail — a comparison that cannot fail is not a
    test — and that when it does it names the divergent node by its path.

    Two cases, because the two fields fail through different routes:

    ``prior-one-ulp``   the smallest change the dump can express, on ONE prior
                        of the root's entry. A prior enters PUCT directly and is
                        also stored per node, so this is the surgical case.
    ``interior-values`` 1e-9 on every interior entry's value. A value reaches
                        the comparison only as an accumulated ``value_sum``,
                        via backpropagate(), so this is what shows the gate
                        checks arithmetic and not merely storage.

    THE PERTURBATION SIZES ARE NOT INTERCHANGEABLE, and the reason is worth
    recording. One ulp on the ROOT's value does NOT diverge the tree: the root's
    value_sum is seeded with it and then accumulates ~800 backups, and once the
    running sum passes ~1 its own ulp (2.2e-16) swallows a 1e-16 seed
    difference. That is a real property of the accumulator, not a gap in the
    gate — and it is exactly why the value case perturbs interior entries, whose
    values enter each node's Q and therefore selection, rather than only the one
    value that lands in a sum and is never read back. See DECISIONS.md.
    """
    index = _full_tree_run_index(manifest)
    run = manifest["runs"][index]
    golden = _golden_run(trees, index)
    _slot, begin, _end = _root_slot_for(dump, golden)

    priors = dump["priors"]
    values = dump["values"]
    if field == "prior-one-ulp":
        priors = priors.copy()
        before = priors[begin]
        priors[begin] = np.nextafter(before, np.float32(1.0), dtype=np.float32)
        assert priors[begin] != before
    else:
        values = values.copy()
        interior = dump["is_root"] == 0
        values[interior] = values[interior] + 1e-9
        assert not np.array_equal(values, dump["values"])

    capacity = 1 + int(golden["children"].sum())
    search = guofish_core.ReplaySearchDouble(
        guofish_core.SearchConfig(virtual_loss=run["virtual_loss"],
                                  arena_capacity=capacity))
    search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                     dump["moves"], priors, values)
    search.set_position(run["fen"])
    search.search(run["sims"])

    report = _first_divergence("mutation drill", golden, search.dump_tree_arrays(0))
    assert report is not None, (
        f"the {field} mutation produced a bit-identical tree, which means the "
        f"gate is not actually comparing what it claims to")
    assert "path from root" in report or "structural difference" in report, (
        f"the failure report must name the divergent node's path:\n{report}")


def test_the_unmutated_run_still_passes_after_a_drill(dump, trees, manifest):
    """The drill's control: the same run, same engine construction, unmutated
    arrays, must still be bit-exact. Without this a drill that 'failed' because
    the harness itself was misconfigured would read as a successful drill."""
    index = _full_tree_run_index(manifest)
    run = manifest["runs"][index]
    golden = _golden_run(trees, index)

    search = guofish_core.ReplaySearchDouble(
        guofish_core.SearchConfig(virtual_loss=run["virtual_loss"],
                                  arena_capacity=1 + int(golden["children"].sum())))
    search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                     dump["moves"], dump["priors"], dump["values"])
    search.set_position(run["fen"])
    search.search(run["sims"])

    report = _first_divergence("drill control", golden, search.dump_tree_arrays(0))
    assert report is None, report


def test_the_dump_is_required(manifest):
    """A search with no dump is a configuration error, not a search that quietly
    invents evaluations."""
    search = guofish_core.ReplaySearchDouble(
        guofish_core.SearchConfig(arena_capacity=64))
    search.set_position(manifest["runs"][0]["fen"])
    with pytest.raises(RuntimeError):
        search.search(10)


def test_set_position_rejects_a_fen_the_reference_would_refuse():
    search = guofish_core.ReplaySearchDouble(
        guofish_core.SearchConfig(arena_capacity=64))
    for bad in ("", "not a fen", "8/8/8/8/8/8/8/8 w - - 0 1"):
        with pytest.raises(ValueError):
            search.set_position(bad)
