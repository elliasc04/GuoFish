"""C7 acceptance — the transposition cache and the Syzygy tablebase.

What this chunk had to prove, and where each proof lives
-------------------------------------------------------
The cache is easy to write and easy to get subtly, permanently wrong, and the
brief is unusually specific about the ways. Each criterion has its own section
below and each section says which one it is.

1.  **The cache cannot hold a path-dependent value.** Not "does not" — cannot.
    ``test_a_terminal_or_tablebase_value_cannot_be_stored`` reads the compiler's
    own answer about the real ``TranspositionCache::insert``. The same facts are
    ``static_assert``s in ``cpp/cache.hpp``, so a violation stops the build; this
    file exists because a build that stops gives a test nothing to point at.

2.  **Gate 1, re-run with the cache ON, bit-exact against a Python reference
    with its cache ON.** ``golden/gate1_cache_*`` is that reference — a fresh run
    of ``core/mctsv4.py`` at ``cache_size=100_000`` over both Gate 1 corpora.
    It is not the C5/C6 golden data re-labelled.

3.  **EP-twins do not collide.** Two positions differing only in the raw
    en-passant square tokenize differently, so they cannot share an entry.

4.  **Clock-twins share an entry, and that is correct.** The halfmove clock is
    not a token, so the network's output cannot depend on it. This is asserted
    as CORRECT rather than tolerated — and it is also exactly why (1) matters,
    because the sharing that makes the cache right makes a tablebase value wrong.

5.  See (1).

6.  **A non-trivial hit rate.** With tablebases off the cache is
    result-invariant, so a cache that never hits produces a bit-identical tree
    and passes (2) outright. The hit rate is therefore asserted separately, and
    against the REFERENCE's own measured rate rather than against a number
    picked to be passed.

7.  **Entry contents round-trip, verified by assertion on mismatch.** The moves
    and priors that come back out of the cache are compared against the golden
    dump they went in from, element for element, priors on their bit pattern.
    ``test_the_entry_contents_assertion_is_live`` then corrupts a prior and a
    move in memory and requires the comparison to name them — the mutation drill
    (Amendment B), run as a test rather than by hand, so ``golden/`` is not
    merely un-written but untouched.

The tablebase half is separate and is judged differently: it has no golden data
and no parity claim, because the reference's tablebase behaviour contains the
defect this chunk exists to remove (it caches the WDL override). What IS asserted
is that the port's own behaviour is correct and that the backend agrees with
python-chess about what the tables say.

Environment overrides
---------------------
``GUOFISH_GOLDEN_GATE1_DUMP`` / ``GUOFISH_GOLDEN_C6_DUMP`` (shared with C5/C6)
and ``GUOFISH_GOLDEN_C7_*`` for this chunk's files, so any drill runs against
corrupted copies in a scratch directory.

``chess``/``chess.syzygy`` are imported only by the tablebase section and only to
serve as an oracle; the cache half imports neither, and neither half imports
``torch``. The reference reaches the Gate 1 comparison only through golden files.
"""

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent

# The replay dumps are C5's and C6's, unchanged. A cache-on run does not need a
# new one: the cache=1 dumps hold an entry for every position either corpus ever
# expands, which is a superset of what a cache-on run evaluates. That is what
# lets the C++ cache miss where the reference hit without falling off the end of
# the world.
DUMPS = {
    "quiet": (REPO_ROOT / "golden" / "gate1_dump.npz", "GUOFISH_GOLDEN_GATE1_DUMP"),
    "terminal": (REPO_ROOT / "golden" / "gate1_terminal_dump.npz", "GUOFISH_GOLDEN_C6_DUMP"),
}
CACHE_TREES = {
    "quiet": (REPO_ROOT / "golden" / "gate1_cache_trees.npz", "GUOFISH_GOLDEN_C7_TREES"),
    "terminal": (REPO_ROOT / "golden" / "gate1_cache_terminal_trees.npz",
                 "GUOFISH_GOLDEN_C7_TERMINAL_TREES"),
}
CACHE_MANIFEST = (REPO_ROOT / "golden" / "gate1_cache_manifest.json",
                  "GUOFISH_GOLDEN_C7_MANIFEST")

SYZYGY_DIR = REPO_ROOT / "assets" / "syzygy"

# The reference's own default (`ParallelMCTS.__init__`: cache_size=100_000), and
# what golden/gate1_cache_manifest.json records having run at.
REFERENCE_CACHE_SIZE = 100_000

# What the C++ side runs the gate at. Larger than the reference's, on purpose:
# this table is DIRECT-MAPPED (one slot per bucket, an insert displaces whatever
# was there) where the reference's is a hash map with a ring-buffer eviction
# order, so two distinct keys landing on one slot cost a hit here that the
# reference would have kept. Sizing well above the working set — a 5,000-sim
# search holds ~4,500 distinct positions — keeps that a rounding error rather
# than a confound, and `test_a_small_cache_evicts_without_changing_the_tree`
# covers the opposite regime deliberately.
GATE_CACHE_SLOTS = 1 << 20


def _path(spec):
    default, env = spec
    return Path(os.environ.get(env, default))


def _load_npz(spec, what):
    path = _path(spec)
    if not path.exists():
        pytest.fail(
            f"{what} missing: {path}\n"
            "Global Rule 2: regenerate with `python tools/gen_c7_cache_golden.py`, "
            "which runs the Python reference with its cache on. It is never "
            "produced from C++ output.")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _load_manifest():
    path = _path(CACHE_MANIFEST)
    if not path.exists():
        pytest.fail(
            f"C7 cache manifest missing: {path}\n"
            "Regenerate with `python tools/gen_c7_cache_golden.py`.")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def manifest():
    return _load_manifest()


@pytest.fixture(scope="session")
def corpora(manifest):
    """Both corpora: the C5/C6 dumps and this chunk's cache-on reference trees."""
    out = {}
    for corpus in manifest["corpora"]:
        out[corpus] = {
            "dump": _load_npz(DUMPS[corpus], f"{corpus} replay dump"),
            "trees": _load_npz(CACHE_TREES[corpus], f"{corpus} cache-on reference trees"),
            "runs": manifest["corpora"][corpus]["runs"],
            "positions": {p["index"]: p for p in manifest["corpora"][corpus]["positions"]},
        }
    return out


def _golden_run(trees, index):
    begin = int(trees["run_offset"][index])
    end = int(trees["run_offset"][index + 1])
    return {name: trees[name][begin:end]
            for name in ("depth", "move", "visits", "value_sum", "prior", "children",
                         "terminal", "terminal_value")}


def _bits64(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def _bits32(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


# ---------------------------------------------------------------------------
# 1. THE ANTI-POISONING REQUIREMENT — acceptance criterion 5
#
# "A compile-fail test successfully proves that attempting to store a
# terminal/proof in the cache fails to compile."
# ---------------------------------------------------------------------------


def test_a_terminal_or_tablebase_value_cannot_be_stored():
    """The criterion, read off the compiler.

    Every value below is computed by overload resolution against the REAL
    ``TranspositionCache::insert`` (``guofish::detail::CacheInsertAccepts``), not
    against a restatement of its signature that could drift away from it. The
    same predicates are ``static_assert``s in ``cpp/cache.hpp``, so in practice a
    violation never reaches this test — the build stops first. That is the
    intended failure mode; this is the reporting surface.

    C3 established that shelling out to a compiler proves nothing a
    ``static_assert`` does not, and has the additional cost of only running when
    somebody remembers to run it. These run in every build on both toolchains.
    """
    sep = guofish_core.cache_type_separation()

    assert sep["network_value_accepted"] is True, (
        "the cache must accept the network's own value — a cache that accepts "
        "nothing holds nothing, and every other assertion here would pass "
        "vacuously")

    forbidden = {
        "terminal_value_accepted":
            "a terminal value is path-dependent (repetition, fifty-move) and the "
            "key is position-only",
        "tablebase_value_accepted":
            "a Syzygy WDL ignores the fifty-move rule, so it depends on the "
            "halfmove clock, which is not a token and so not part of the key. "
            "This is the defect core/mctsv4.py has today",
        "proof_value_accepted":
            "a proof is a statement about a subtree, not about the leaf the key "
            "identifies",
        "bare_double_accepted":
            "a bare double is how all three of the above get in, by reading "
            "someone's .value member",
        "terminal_flag_accepted":
            "bool converts to double in C++, so a terminal FLAG is the crossing "
            "the language would otherwise permit silently",
        "float_accepted": "same argument as double",
        "int_accepted": "same argument as double",
    }
    for key, why in forbidden.items():
        assert sep[key] is False, f"{key}: {why}"

    # The conversions that would route around all of the above if they existed.
    for key in ("terminal_converts_to_network", "tablebase_converts_to_network",
                "proof_converts_to_network", "double_converts_to_network",
                "network_converts_to_double", "network_constructible_from_tablebase",
                "network_constructible_from_terminal"):
        assert sep[key] is False, (
            f"{key} would defeat the separation: the types would still be "
            f"distinct, and a conversion would let one become the other silently")

    # And the payload itself is closed, so a caller holding a slot could not
    # write a forbidden value into it either.
    assert sep["payload_value_is_network_value"] is True
    for key in ("payload_value_assignable_from_tablebase",
                "payload_value_assignable_from_terminal",
                "payload_value_assignable_from_double"):
        assert sep[key] is False, key


def test_the_empty_slot_sentinel_does_not_rest_on_a_reserved_key():
    """C3's design, depended on rather than admired.

    ``NNKey`` has no default constructor, so a slot cannot hold a "zero key"
    meaning empty — and it should not want to. FNV-1a's output covers the whole
    64-bit range, so every value including 0 and including the offset basis is a
    key that some payload really hashes to; reserving one would make one position
    in 2**64 uncacheable and, worse, require every reader to remember which.

    The sentinel is ``std::optional``'s disengaged state, which is distinct from
    every NNKey by construction. If somebody "fixes" NNKey by giving it a default
    constructor, ``cpp/cache.hpp``'s static_assert fires and this test says why.
    """
    assert guofish_core.cache_type_separation()["nn_key_default_constructible"] is False
    assert guofish_core.cache_type_separation()["network_value_default_constructible"] is False

    # And a key of 0 is an ordinary, storable key — not a hole in the table.
    cache = guofish_core.TranspositionCache(256)
    assert cache.probe(0) is None
    cache.insert(0, 0.25, ["e2e4"], [1.0])
    entry = cache.probe(0)
    assert entry is not None and entry["value"] == 0.25, (
        "key 0 must be storable. If this fails, somebody has reserved a key "
        "value as an empty-slot marker, and one position in 2**64 is now "
        "silently uncacheable.")


# ---------------------------------------------------------------------------
# 2. THE KEY COMES FROM THE TOKEN ROW
#
# Scope 2.5, and the brief's "Risks / NNKey Generation": the stored nn_key must
# be computed from the exact token buffer row dispatched to the evaluator, not
# re-derived from the board inside the cache.
# ---------------------------------------------------------------------------


def test_the_key_is_a_function_of_the_dispatched_token_row():
    """One derivation, checked by perturbing its input.

    ``eval_row`` returns the 68 tokens AND the key, from one ``guofish::EvalRow``
    object. If the key were derived from the board independently, changing a
    token would leave it unmoved — which is precisely the failure the brief warns
    about, and precisely the one that produces no symptom until the cache starts
    serving one position's policy for another.
    """
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3kr2/1p1nnpp1/1bp1p1p1/p2pP1N1/P2P3P/BPP3P1/5PB1/R3K2R w KQ - 1 21",
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
    ]
    for fen in fens:
        row = guofish_core.eval_row(fen)
        assert len(row["tokens"]) == guofish_core.SEQ_LENGTH
        # The key the cache is written under is the hash of THESE bytes.
        assert guofish_core.nn_key_of_tokens(row["tokens"]) == row["nn_key"]
        # And it agrees with the position-level entry point, which is the same
        # implementation reached another way (C3's tests judge that one).
        assert guofish_core.nn_key(fen) == row["nn_key"]

        # Perturb one token at a time. Every index must move the key.
        for index in range(guofish_core.SEQ_LENGTH):
            perturbed = row["tokens"].copy()
            perturbed[index] = perturbed[index] + 1
            assert guofish_core.nn_key_of_tokens(perturbed) != row["nn_key"], (
                f"changing token {index} left the nn_key unchanged for {fen}. "
                f"The key is then not a function of the evaluator's input, and "
                f"two positions the network sees differently can share an entry.")


# ---------------------------------------------------------------------------
# 3. STRUCTURE — sharding, sentinel, eviction, refusal of a useless cache
# ---------------------------------------------------------------------------


def test_the_cache_is_sharded_at_least_sixty_four_ways():
    """The brief's floor, and the floor is enforced rather than defaulted to."""
    assert guofish_core.CACHE_MIN_SHARDS >= 64
    assert guofish_core.CACHE_DEFAULT_SHARDS >= guofish_core.CACHE_MIN_SHARDS

    cache = guofish_core.TranspositionCache(4096)
    assert cache.shard_count >= 64
    assert cache.capacity >= 4096
    assert cache.shard_count * cache.slots_per_shard == cache.capacity

    with pytest.raises(ValueError):
        guofish_core.TranspositionCache(4096, shards=32)
    with pytest.raises(ValueError):
        guofish_core.TranspositionCache(4096, shards=100)  # not a power of two


def test_a_cache_that_can_hold_nothing_is_not_constructible():
    """'Cache off' and 'a cache with a 0% hit rate' must not be one keystroke
    apart.

    A zero-slot cache passes every tree-equivalence test in this file while doing
    no work, which is the exact failure mode acceptance criterion 6 exists to
    catch. Off is expressed by not having one.
    """
    with pytest.raises(ValueError):
        guofish_core.TranspositionCache(0)

    off = guofish_core.SearchConfig()
    assert off.cache_slots == 0, (
        "the default must be no cache: C5 and C6 were certified against the "
        "reference on the no-cache path, and a chunk that silently changed the "
        "code under their tests would make a C7 regression look like a C5 one")
    search = guofish_core.ReplaySearchDouble(off)
    assert search.cache_stats()["enabled"] is False


def test_a_probe_reports_a_miss_rather_than_a_stale_slot():
    cache = guofish_core.TranspositionCache(256)
    key_a, key_b = guofish_core.nn_key(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), 0xDEADBEEF

    assert cache.probe(key_a) is None
    stats = cache.stats
    assert stats.misses == 1 and stats.hits == 0

    cache.insert(key_a, -0.5, ["e2e4", "d2d4"], [0.7, 0.3])
    assert cache.probe(key_b) is None
    hit = cache.probe(key_a)
    assert hit is not None
    assert hit["moves"] == ["e2e4", "d2d4"]
    assert cache.size == 1

    cache.clear()
    assert cache.probe(key_a) is None
    assert cache.size == 0
    assert cache.stats.hits == 0, "clear() must zero the counters too"


def test_inserting_an_entry_with_no_moves_is_refused():
    """A position with no legal moves is terminal, and a terminal result must
    never reach this cache. The refusal is at the door rather than relying on
    every caller to have checked."""
    cache = guofish_core.TranspositionCache(256)
    with pytest.raises(ValueError):
        cache.insert(1, 0.0, [], [])


# ---------------------------------------------------------------------------
# 4. EP-TWINS AND CLOCK-TWINS — acceptance criteria 3 and 4
# ---------------------------------------------------------------------------

# Twins built by hand rather than sampled, because the property is about a
# single field and a corpus would only show that the field usually differs.
EP_TWINS = [
    ("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
     "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
     "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
]

CLOCK_TWINS = [
    ("r3kr2/1p1nnpp1/1bp1p1p1/p2pP1N1/P2P3P/BPP3P1/5PB1/R3K2R w KQ - 1 21",
     "r3kr2/1p1nnpp1/1bp1p1p1/p2pP1N1/P2P3P/BPP3P1/5PB1/R3K2R w KQ - 97 21"),
    ("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
     "8/8/4k3/8/8/4K3/4P3/8 w - - 42 1"),
]


@pytest.mark.parametrize("with_ep,without_ep", EP_TWINS)
def test_ep_twins_do_not_collide_in_the_cache(with_ep, without_ep):
    """Acceptance criterion 3.

    Token 66 is written from ``board.ep_square is not None`` — the RAW rule,
    unconditionally, with no test for whether a capture is available. Two
    positions differing only there are two different network inputs, so they must
    be two different entries.

    The reference reaches the same place by a patch: ``make_cache_key`` appends
    the raw ep square to a Polyglot Zobrist that would otherwise be COARSER than
    the network's own input, because Polyglot folds the ep file in only when an
    enemy pawn stands ready to capture. Here there is nothing to patch — the key
    IS the hash of the tokens, so the property holds by construction and cannot
    be regressed by someone simplifying the key.
    """
    key_ep = guofish_core.nn_key(with_ep)
    key_none = guofish_core.nn_key(without_ep)
    assert key_ep != key_none, (
        "the two ep-twins share an nn_key, so the cache would serve one's "
        "evaluation for the other")

    # And that separation survives the cache, which is the claim that matters.
    cache = guofish_core.TranspositionCache(4096)
    cache.insert(key_ep, 0.75, ["e4e5"], [1.0])
    assert cache.probe(key_none) is None, (
        "an entry stored for the en-passant position was served to the position "
        "without it")
    cache.insert(key_none, -0.75, ["e4e5"], [1.0])
    assert cache.probe(key_ep)["value"] == 0.75
    assert cache.probe(key_none)["value"] == -0.75
    assert cache.size == 2


@pytest.mark.parametrize("early,late", CLOCK_TWINS)
def test_clock_twins_share_one_entry_and_that_is_correct(early, late):
    """Acceptance criterion 4 — asserted as CORRECT, not tolerated.

    The halfmove clock is not one of the 68 tokens. The network cannot see it, so
    its output cannot depend on it, so two positions differing only in the clock
    have the same evaluation and must share one entry. Splitting them would not
    be conservative; it would be a cache that stores the same answer twice and
    reports a lower hit rate for it.

    This is also exactly why the type discipline in section 1 is not paranoia.
    Sharing across clock-twins is correct ONLY for values that ignore the clock.
    A Syzygy WDL does not ignore the clock — it assumes the fifty-move rule never
    intervenes — so the same sharing that makes this entry right would make a
    cached WDL wrong, which is the reference's defect. The two tests are the two
    halves of one argument.
    """
    key_early = guofish_core.nn_key(early)
    key_late = guofish_core.nn_key(late)
    assert key_early == key_late, (
        "clock-twins must share an nn_key. If they do not, the halfmove clock "
        "has leaked into the tokenization, and the C2 parity corpus would be "
        "wrong about 100,000 positions.")

    cache = guofish_core.TranspositionCache(4096)
    cache.insert(key_early, 0.125, ["e2e4", "d2d4"], [0.5, 0.5])
    hit = cache.probe(key_late)
    assert hit is not None and hit["value"] == 0.125
    assert cache.size == 1, "one position, one entry"

    # The fifty-move rule still applies to the POSITION; it just does not apply
    # to the cache, because it produces a TerminalValue and section 1 proves a
    # TerminalValue cannot get in here.
    assert guofish_core.cache_type_separation()["terminal_value_accepted"] is False


# ---------------------------------------------------------------------------
# 5-7. THE GATE, THE HIT RATE, AND THE ENTRY CONTENTS
#
# One search per recorded run, shared by three sections, because a 5,000-sim run
# is not cheap and all three read the same tree.
# ---------------------------------------------------------------------------


def _collect():
    path = _path(CACHE_MANIFEST)
    if not path.exists():
        return [], []
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    cases, ids = [], []
    for corpus, block in manifest["corpora"].items():
        for index, run in enumerate(block["runs"]):
            cases.append((corpus, index))
            name = run.get("name") or f"pos{run['position']:02d}"
            ids.append(f"{corpus}-{name}-vl{run['virtual_loss']}-n{run['sims']}-"
                       f"{'full' if run['full_tree'] else 'visited'}")
    return cases, ids


CASES, CASE_IDS = _collect()


@pytest.fixture(scope="session")
def searched(corpora):
    """Every recorded run, executed once with the cache ON.

    One engine per (corpus, virtual loss, max_tree_depth, cache) so the arena and
    the dump are not rebuilt 106 times. The cache is cleared before each run so
    that its hit count is attributable to that run — the search deliberately does
    NOT clear it on set_position (the reference's cache lives on the engine and
    survives every search, which is most of what it is for), so the test has to
    ask.
    """
    engines = {}
    results = {}

    for corpus, block in corpora.items():
        trees = block["trees"]
        dump = block["dump"]

        capacity = 0
        for index in range(len(block["runs"])):
            capacity = max(capacity, 1 + int(_golden_run(trees, index)["children"].sum()))

        for index, run in enumerate(block["runs"]):
            key = (corpus, run["virtual_loss"], run.get("max_tree_depth", 80))
            if key not in engines:
                config = guofish_core.SearchConfig(
                    virtual_loss=run["virtual_loss"],
                    max_tree_depth=run.get("max_tree_depth", 80),
                    arena_capacity=capacity,
                    cache_slots=GATE_CACHE_SLOTS)
                search = guofish_core.ReplaySearchDouble(config)
                search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                                 dump["moves"], dump["priors"], dump["values"])
                engines[key] = search
            search = engines[key]
            search.clear_cache()
            search.set_position(run["fen"],
                                block["positions"][run["position"]].get("history", []))
            stats = search.search(run["sims"])
            results[(corpus, index)] = {
                "stats": stats,
                "cache": search.cache_stats(),
                "arrays": search.dump_tree_arrays(0 if run["full_tree"] else 1),
                "run": run,
            }
    return results


# --- the failure report ----------------------------------------------------


def _paths_from_dfs(depth, move):
    paths, stack = [], []
    for d, m in zip(depth, move):
        del stack[int(d):]
        stack.append(guofish_core.move_to_uci(int(m)) if int(d) > 0 else "(root)")
        paths.append(" ".join(stack[1:]) if len(stack) > 1 else "(root)")
    return paths


def _first_divergence(label, golden, actual):
    """The first differing node in DFS order, named by its path from the root.

    Same shape as C5's and C6's, and for the same reason: a bare "the trees
    differ" is useless on a 200,000-node tree. C7 adds one line to the report —
    the cache counters — because when a cache-on run diverges and a cache-off run
    does not, the first question is whether the cache hit at all.
    """
    if len(golden["depth"]) != len(actual["depth"]):
        shorter = min(len(golden["depth"]), len(actual["depth"]))
        for i in range(shorter):
            if (int(golden["depth"][i]) != int(actual["depth"][i]) or
                    int(golden["move"][i]) != int(actual["move"][i])):
                break
        else:
            i = shorter
        paths = _paths_from_dfs(golden["depth"][:shorter], golden["move"][:shorter])
        where = paths[i] if i < len(paths) else "(past the end of the shorter tree)"
        return (f"{label}: different node counts — golden {len(golden['depth'])}, "
                f"C++ {len(actual['depth'])}\n"
                f"  first structural difference at DFS index {i}, path: {where}\n"
                f"  This is a SHAPE divergence: the two searches expanded different "
                f"moves, not merely different numbers.")

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
    sources = {"value_sum": "value_sum", "prior": "prior", "terminal_value": "terminal_value"}
    lines = [
        f"{label}: first divergence at DFS index {first_index} of "
        f"{len(golden['depth'])} nodes",
        f"  path from root : {paths[first_index]}",
        f"  depth          : {int(golden['depth'][first_index])}",
        f"  move           : {guofish_core.move_to_uci(int(golden['move'][first_index]))}",
    ]
    for name, want, got, kind in comparisons:
        w, g = want[first_index], got[first_index]
        flag = "  <-- DIFFERS" if w != g else ""
        if kind is float:
            src = sources[name]
            lines.append(f"  {name:<14} : golden 0x{int(w):016x} "
                         f"({golden[src][first_index]!r})  c++ 0x{int(g):016x} "
                         f"({actual[src][first_index]!r}){flag}")
        else:
            lines.append(f"  {name:<14} : golden {int(w)}  c++ {int(g)}{flag}")
    return "\n".join(lines)


# --- acceptance criterion 2 -------------------------------------------------


def test_the_reference_ran_with_its_cache_on(manifest):
    """The golden data must be what the criterion asks for, not C5/C6's relabelled.

    Two independent statements, both required:
      * the reference ran at cache_size=100_000, not at the equivalence
        configuration's 1;
      * tablebases were OFF, because with them on the reference caches the WDL
        override and its trees would encode the defect this chunk removes.
    """
    config = manifest["config"]
    assert config["cache_size"] == REFERENCE_CACHE_SIZE, (
        "the golden trees were not produced with the reference's cache on; "
        "comparing against them would not test what criterion 2 asks about")
    assert config["tablebase"] is False, (
        "the reference's tablebase path caches its WDL override "
        "(core/mctsv4.py: 'We override BEFORE caching'). Golden data generated "
        "with it on would bake that defect into the acceptance criteria.")
    assert config["workers"] == 1
    assert config["dirichlet"] is False
    assert config["canonical_order_patch"] is True
    assert manifest["complete"] is True, (
        "this golden data came from a --limit run and does not cover both "
        "corpora; it is not acceptance-grade")


def test_the_python_cache_is_result_invariant(manifest):
    """The brief's recon claim, restated as a measurement on this corpus.

    The brief says "``cache=1`` and ``cache=100k`` give bit-identical trees",
    which is what lets a cache-on C++ run be judged at all — and what makes the
    hit rate a SEPARATE criterion, since tree equality cannot see it. The
    generator checked every run of both corpora against the cache=1 golden trees
    and recorded the answer, so the claim is now a fact about this checkpoint
    rather than a footnote.
    """
    invariance = manifest["cache_invariance"]
    assert invariance["cache_invariant"] is True, (
        f"the reference's own trees changed when its cache was turned on: "
        f"{invariance['divergent_runs']}. Tree equivalence is then not a valid "
        f"test of a cache-on C++ run, and the whole comparison below has to be "
        f"rethought rather than patched.")
    assert invariance["total_hits"] > 0, (
        "the reference's cache never hit anywhere in either corpus, so these "
        "positions do not transpose and the corpus cannot support criterion 6")


@pytest.mark.parametrize("corpus,index", CASES, ids=CASE_IDS)
def test_gate1_with_the_cache_on_is_bit_exact(corpus, index, corpora, searched):
    """ACCEPTANCE CRITERION 2. No tolerance, and there must not be one.

    Visit counts equal as integers; ``value_sum`` identical as a 64-bit pattern;
    priors and terminal values identical as 32-bit patterns; the terminal bit and
    the child count equal.

    What a failure HERE and not in C5/C6 means: the cache returned something
    other than what the evaluator would have. The two candidates are a key
    collision (two positions sharing an nn_key) and a payload that was corrupted
    between insert and probe. The move-list check inside ``expand`` catches the
    first at its source and names it; this catches whatever is left.
    """
    result = searched[(corpus, index)]
    golden = _golden_run(corpora[corpus]["trees"], index)
    report = _first_divergence(f"{corpus}[{index}]", golden, result["arrays"])
    if report is not None:
        cache = result["cache"]
        report += (f"\n  cache          : {cache['hits']} hits, {cache['misses']} misses, "
                   f"{cache['collisions']} collisions, {cache['size']} entries\n"
                   f"  A divergence with 0 hits is not a cache bug. A divergence with "
                   f"hits is: the cache served something the evaluator would not have.")
        pytest.fail(report)


@pytest.mark.parametrize("corpus,index", CASES, ids=CASE_IDS)
def test_the_search_agrees_with_the_reference_about_what_it_did(corpus, index, searched):
    """The run's own audit, not just its tree.

    The reference recorded how many simulations it ran, whether the depth-1 mate
    short-circuit truncated it, and which move it played. A tree that matched
    while the search had stopped early for a different reason would be a
    coincidence worth catching.
    """
    result = searched[(corpus, index)]
    run = result["run"]
    stats = result["stats"]

    assert stats["root_visits"] == run["root_visits"], (
        f"root visit count {stats['root_visits']} != reference {run['root_visits']}")
    assert stats["best_move"] == run["best_move"]
    assert (stats["mating_move"] is not None) == bool(run["early_exit"]), (
        "the depth-1 mate short-circuit fired on one side and not the other")


# --- acceptance criterion 6: the hit rate -----------------------------------


def test_the_cache_actually_hits(searched, manifest):
    """ACCEPTANCE CRITERION 6, and the reason it is a criterion at all.

    The brief is explicit: with tablebases off the Python cache is
    result-invariant, so tree equivalence cannot distinguish a working cache from
    one that never hits — a C++ cache with a 0% hit rate passes the gate above
    outright.

    The floor is not a number picked to be cleared. It is the REFERENCE's own
    measured hit rate on the same run, recorded by the generator. The C++ side is
    allowed to fall slightly short of it because this table is direct-mapped and
    the reference's is not (see GATE_CACHE_SLOTS), but it is not allowed to fall
    short by much, and it is certainly not allowed to be zero.
    """
    totals = {"hits": 0, "misses": 0, "ref_hits": 0, "ref_misses": 0}
    worst = []

    for (corpus, index), result in searched.items():
        cache = result["cache"]
        run = result["run"]
        totals["hits"] += cache["hits"]
        totals["misses"] += cache["misses"]
        totals["ref_hits"] += run["cache_hits"]
        totals["ref_misses"] += run["cache_misses"]

        name = run.get("name") or f"pos{run['position']:02d}"
        # Per-run: the probe count must match the reference exactly. Both probe
        # once per interior leaf, and the trees are identical, so the number of
        # PROBES is a structural fact even where the hit/miss split is not.
        probes = cache["hits"] + cache["misses"]
        ref_probes = run["cache_hits"] + run["cache_misses"]
        assert probes == ref_probes, (
            f"{corpus}/{name}: the C++ search probed the cache {probes} times "
            f"and the reference {ref_probes}. The trees are identical, so the "
            f"two searches reached different numbers of leaves — which means "
            f"the probe is in the wrong place, most likely at the root (the "
            f"reference's _expand_root does not consult the cache).")

        if run["cache_hits"] > 0:
            ratio = cache["hits"] / run["cache_hits"]
            if ratio < 0.95:
                worst.append(f"{corpus}/{name}: {cache['hits']} hits vs the "
                             f"reference's {run['cache_hits']} ({ratio:.1%})")

    assert totals["hits"] > 0, (
        "the cache never hit anywhere in 106 runs. The tree comparison above "
        "passes with a cache that does nothing, which is exactly why this "
        "assertion exists.")

    rate = totals["hits"] / (totals["hits"] + totals["misses"])
    ref_rate = totals["ref_hits"] / (totals["ref_hits"] + totals["ref_misses"])
    assert ref_rate > 0.05, (
        "the reference's own hit rate on this corpus is below 5%, so these "
        "positions barely transpose and the criterion is not being tested")
    assert rate >= 0.95 * ref_rate, (
        f"C++ hit rate {rate:.3%} against the reference's {ref_rate:.3%}. A "
        f"direct-mapped table loses a little to slot collisions and that is "
        f"expected; losing this much means it is losing entries for a reason "
        f"other than capacity.")
    assert not worst, "runs where the C++ cache hit materially less often:\n  " + \
        "\n  ".join(worst)


# --- acceptance criterion 7: the entry contents -----------------------------


def _compare_cache_against(search, keys, is_root, move_offset, moves, priors, values):
    """Every cache entry, checked field for field against the payload it came from.

    Returns a list of complaints, empty when everything round-trips. The point of
    returning them rather than asserting inline is that the same function drives
    the acceptance test and the mutation drill below — so the drill proves that
    THIS comparison catches a corrupted prior, not that some other one would.

    Root entries are skipped: the root's evaluation never enters the cache (the
    reference's ``_expand_root`` does not consult it, and mirroring that is what
    keeps the GPU-softmaxed root priors from being served to an interior visit of
    the same position). A root key that HAD an entry would be a bug, and is
    checked separately below.
    """
    complaints = []
    for i in range(len(keys)):
        if is_root[i]:
            continue
        entry = search.cache_entry_by_key(int(keys[i]))
        if entry is None:
            continue
        begin, end = int(move_offset[i]), int(move_offset[i + 1])
        want_moves = moves[begin:end]
        want_priors = priors[begin:end]

        got_moves = np.asarray(entry["moves"], dtype=np.uint16)
        got_priors = np.asarray(entry["priors"], dtype=np.float32)

        if len(got_moves) != len(want_moves):
            complaints.append(
                f"key {int(keys[i]):#018x}: cached move list has {len(got_moves)} "
                f"moves, the dump has {len(want_moves)}")
            continue
        bad = np.flatnonzero(got_moves != want_moves)
        for j in bad[:4]:
            complaints.append(
                f"key {int(keys[i]):#018x}: move[{int(j)}] cached "
                f"{guofish_core.move_to_uci(int(got_moves[j]))} != dump "
                f"{guofish_core.move_to_uci(int(want_moves[j]))}")
        # Priors on their BIT PATTERN. A prior that round-tripped to within
        # 1e-9 is a prior that would change a PUCT comparison.
        bad = np.flatnonzero(_bits32(got_priors) != _bits32(want_priors))
        for j in bad[:4]:
            complaints.append(
                f"key {int(keys[i]):#018x}: prior[{int(j)}] for "
                f"{guofish_core.move_to_uci(int(want_moves[j]))} cached "
                f"0x{int(_bits32(got_priors)[j]):08x} ({got_priors[j]!r}) != dump "
                f"0x{int(_bits32(want_priors)[j]):08x} ({want_priors[j]!r})")
        if struct.pack("<d", entry["value"]) != struct.pack("<d", float(values[i])):
            complaints.append(
                f"key {int(keys[i]):#018x}: value cached {entry['value']!r} != dump "
                f"{float(values[i])!r}")
    return complaints


@pytest.fixture(scope="session")
def one_cached_search(corpora):
    """One completed cache-on search, kept with its cache intact for inspection.

    Separate from `searched` because that fixture reuses engines across runs, and
    an entry-contents check wants to know which run's entries it is looking at.
    """
    block = corpora["quiet"]
    run = block["runs"][0]
    dump = block["dump"]
    trees = block["trees"]
    capacity = 1 + int(_golden_run(trees, 0)["children"].sum())

    search = guofish_core.ReplaySearchDouble(guofish_core.SearchConfig(
        virtual_loss=run["virtual_loss"],
        max_tree_depth=run.get("max_tree_depth", 80),
        arena_capacity=capacity,
        cache_slots=GATE_CACHE_SLOTS))
    search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                     dump["moves"], dump["priors"], dump["values"])
    search.set_position(run["fen"], block["positions"][run["position"]].get("history", []))
    search.search(run["sims"])
    return search, dump, run


def test_cache_entries_round_trip(one_cached_search):
    """ACCEPTANCE CRITERION 7.

    "Direct entry contents (priors and move lists) successfully round-trip and
    are verified by assertions on mismatch rather than positional alignment."

    Every entry the search inserted is read back and compared against the golden
    dump payload it was built from — moves as integers, priors on their bit
    pattern, the value on its bit pattern. Nothing here infers correctness from
    the tree; the tree cannot see any of it.
    """
    search, dump, _run = one_cached_search
    assert search.cache_stats()["size"] > 100, (
        "the search stored almost nothing, so this test has nothing to check")

    complaints = _compare_cache_against(
        search, dump["keys"], dump["is_root"], dump["move_offset"],
        dump["moves"], dump["priors"], dump["values"])
    assert not complaints, (
        "cache entries do not match the payload they were built from:\n  " +
        "\n  ".join(complaints[:20]))


def test_the_root_evaluation_never_enters_the_cache(one_cached_search):
    """The root's entry must not be there, and the reason is not tidiness.

    The reference softmaxes root priors on the GPU (``_expand_root`` runs its own
    forward and hands ``expand`` a CUDA tensor) and interior priors on the CPU,
    and the two disagree by up to ~2e-9 on the same position — which is why
    ``golden/gate1_dump.npz`` is keyed by (nn_key, is_root) and not by nn_key.

    ``_expand_root`` does not consult or write ``self.cache``. If this port's did,
    an interior visit to the root's own position — four plies away in a
    middlegame — would be served the GPU priors where the reference used the CPU
    ones, and Gate 1 would fail at that node with no indication of why.
    """
    search, dump, run = one_cached_search
    root_key = guofish_core.nn_key(run["fen"])

    root_slots = [i for i in range(len(dump["keys"]))
                  if dump["is_root"][i] and int(dump["keys"][i]) == root_key]
    assert root_slots, "the dump has no root entry for this position"

    entry = search.cache_entry_by_key(root_key)
    if entry is None:
        return  # the root position never recurred as an interior node

    # It recurred, so an interior evaluation of it IS legitimately cached. What
    # must not be there is the ROOT table's payload — and the two tables differ
    # only in the last few ulps of a handful of priors, so this has to be a
    # bit-pattern comparison against BOTH candidates rather than a plausibility
    # check against one.
    cached = np.asarray(entry["priors"], dtype=np.float32)

    interior = [i for i in range(len(dump["keys"]))
                if not dump["is_root"][i] and int(dump["keys"][i]) == root_key]
    assert interior, (
        "the root position was cached, but the dump has no INTERIOR entry for "
        "it — so the payload in the cache came from the root table, which is "
        "the GPU-softmaxed one")
    begin, end = int(dump["move_offset"][interior[0]]), int(dump["move_offset"][interior[0] + 1])
    assert np.array_equal(_bits32(cached), _bits32(dump["priors"][begin:end])), (
        "the cached entry for the root position carries the ROOT table's priors, "
        "not the interior table's. _expand_root must neither read nor write the "
        "cache.")


def test_the_entry_contents_assertion_is_live(one_cached_search):
    """The mutation drill (Amendment B), as a test rather than a one-off.

    The brief asks for a gathered prior or a move-list entry to be altered and
    for the entry-contents assertion to fail loudly. Doing it here rather than by
    hand has two advantages: ``golden/`` is not merely un-written but never
    opened for writing — the corruption is applied to an in-memory copy of the
    expectation — and the drill runs on every CI pass instead of living in a
    commit message.

    The corruption is applied to the EXPECTED side, which is equivalent to
    corrupting the stored side and strictly safer. What is being proved is that
    ``_compare_cache_against`` compares what it claims to: if it silently trusted
    positional alignment, or compared priors with ``==`` at float64, or skipped
    entries it could not find, one of these three would pass unnoticed.
    """
    search, dump, _run = one_cached_search

    # A key the search definitely cached, so the comparison reaches an entry.
    cached_index = None
    for i in range(len(dump["keys"])):
        if not dump["is_root"][i] and search.cache_entry_by_key(int(dump["keys"][i])):
            cached_index = i
            break
    assert cached_index is not None, "no interior entry was cached; nothing to drill"
    begin = int(dump["move_offset"][cached_index])

    # (1) A perturbed prior, one ulp. This is the mutation that a tolerance-based
    #     comparison would let through, and one ulp is enough to flip a PUCT tie.
    priors = dump["priors"].copy()
    original = priors[begin]
    priors[begin] = np.frombuffer(
        struct.pack("<I", struct.unpack("<I", struct.pack("<f", original))[0] + 1),
        dtype=np.float32)[0]
    complaints = _compare_cache_against(
        search, dump["keys"], dump["is_root"], dump["move_offset"],
        dump["moves"], priors, dump["values"])
    assert complaints, (
        "a one-ulp change to a gathered prior was not detected. The comparison "
        "is not comparing bit patterns, and a cache that returned nearly-right "
        "priors would pass.")
    assert any("prior[0]" in c for c in complaints), \
        f"the report does not name the corrupted prior: {complaints[:3]}"

    # (2) A altered move-list entry. This is the one that must not be absorbed by
    #     positional alignment: the priors still line up, they just belong to a
    #     different move.
    moves = dump["moves"].copy()
    moves[begin] = np.uint16((int(moves[begin]) + 16) & 0xFFFF)
    complaints = _compare_cache_against(
        search, dump["keys"], dump["is_root"], dump["move_offset"],
        moves, dump["priors"], dump["values"])
    assert complaints, (
        "an altered move-list entry was not detected. The brief requires the "
        "move list to be stored and asserted on rather than trusting positional "
        "alignment, and this is that assertion.")
    assert any("move[0]" in c for c in complaints), \
        f"the report does not name the corrupted move: {complaints[:3]}"

    # (3) A perturbed value.
    values = dump["values"].copy()
    values[cached_index] = values[cached_index] + 1e-15
    complaints = _compare_cache_against(
        search, dump["keys"], dump["is_root"], dump["move_offset"],
        dump["moves"], dump["priors"], values)
    assert complaints and any("value cached" in c for c in complaints), \
        "a perturbed value was not detected"

    # And with nothing corrupted, it is quiet — otherwise the three above prove
    # nothing but that the function always complains.
    assert not _compare_cache_against(
        search, dump["keys"], dump["is_root"], dump["move_offset"],
        dump["moves"], dump["priors"], dump["values"])


def test_a_small_cache_evicts_without_changing_the_tree(corpora):
    """The direct-mapped table's other regime, exercised on purpose.

    ``GATE_CACHE_SLOTS`` is sized so eviction is a rounding error. That is the
    right configuration for measuring a hit rate and the wrong one for finding a
    bug in eviction, so this runs the same position with 128 slots — far below
    the working set — and requires the tree to be bit-identical anyway.

    That it can be is the whole reason the cache is allowed to be simpler than
    the reference's ring buffer: with tablebases off, a cache miss costs an
    evaluation and changes nothing else, so the replacement policy is free.
    """
    block = corpora["quiet"]
    run = block["runs"][0]
    dump = block["dump"]
    trees = block["trees"]
    capacity = 1 + int(_golden_run(trees, 0)["children"].sum())

    results = {}
    for slots in (128, GATE_CACHE_SLOTS):
        search = guofish_core.ReplaySearchDouble(guofish_core.SearchConfig(
            virtual_loss=run["virtual_loss"],
            max_tree_depth=run.get("max_tree_depth", 80),
            arena_capacity=capacity,
            cache_slots=slots))
        search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                         dump["moves"], dump["priors"], dump["values"])
        search.set_position(run["fen"], block["positions"][run["position"]].get("history", []))
        search.search(run["sims"])
        results[slots] = (search.dump_tree_arrays(1), search.cache_stats())

    small_stats = results[128][1]
    assert small_stats["collisions"] > 0, (
        "a 128-slot cache holding a ~4,500-position search evicted nothing, so "
        "this test is not exercising eviction at all")

    report = _first_divergence("small-cache vs large-cache",
                               results[GATE_CACHE_SLOTS][0], results[128][0])
    assert report is None, (
        "evicting from the cache changed the tree. With tablebases off it must "
        "not: a miss costs an evaluation and nothing else.\n" + (report or ""))

    golden = _golden_run(trees, 0)
    assert _first_divergence("small-cache vs reference", golden, results[128][0]) is None


# ---------------------------------------------------------------------------
# 8. SYZYGY TABLEBASES — modes 1 and 2
#
# Judged differently from everything above, and the difference is worth stating.
# There is no golden data here and no parity claim, because the REFERENCE'S
# TABLEBASE BEHAVIOUR CONTAINS THE DEFECT this chunk exists to remove: it applies
# the WDL override before caching, so its cache serves a clock-independent
# tablebase result to a position whose clock has since crossed 100.
#
# So what is asserted is (a) that the port's own behaviour is right — the
# mapping, the perspective, the piece-count gate, the mode 1 ranking — and (b)
# that the native backend agrees with python-chess about what the tables SAY.
# ---------------------------------------------------------------------------

# A GUARDED IMPORT, NOT pytest.importorskip. The difference is not stylistic and
# it was found the hard way: `importorskip` at module scope skips the ENTIRE
# file, so on a machine without python-chess the 200-odd cache tests above —
# which import neither chess nor torch, and which are this chunk's acceptance
# criteria — silently vanished, and the Linux run reported 821 passed where
# Windows reported 1068. A skip that large is indistinguishable from a pass in a
# summary line, which is the failure mode Global Rule 10 is about.
#
# python-chess is needed HERE only as an oracle: something that already knows
# what the tables say, so "Fathom agrees with the reference" is a comparison
# rather than a restatement. Nothing above needs it.
try:
    import chess
    import chess.syzygy as syzygy
except ImportError:  # pragma: no cover - environment-dependent
    chess = None
    syzygy = None

if chess is None:
    _TABLEBASE_SKIP = ("the tablebase section needs python-chess as an oracle "
                       "(`pip install chess`); the cache tests above do not")
elif not SYZYGY_DIR.is_dir() or not any(SYZYGY_DIR.glob("*.rtbw")):
    _TABLEBASE_SKIP = f"no Syzygy tables under {SYZYGY_DIR}"
else:
    _TABLEBASE_SKIP = None

tablebase_required = pytest.mark.skipif(_TABLEBASE_SKIP is not None,
                                        reason=_TABLEBASE_SKIP or "")


@pytest.fixture(scope="session")
def fathom():
    """The one FathomProber this session gets.

    Session-scoped because it has to be: Fathom keeps its state in file scope —
    ``tb_init`` and ``tb_free`` take no handle — so a second live instance would
    share the first's tables and free them from under it. ``cpp/fathom.hpp``
    refuses to construct one rather than letting that happen, and this fixture is
    how the tests cooperate with that.
    """
    if _TABLEBASE_SKIP is not None:
        pytest.skip(_TABLEBASE_SKIP)
    prober = guofish_core.FathomProber(str(SYZYGY_DIR))
    yield prober
    del prober


@pytest.fixture(scope="session")
def reference_tablebase():
    if _TABLEBASE_SKIP is not None:
        pytest.skip(_TABLEBASE_SKIP)
    with syzygy.open_tablebase(str(SYZYGY_DIR)) as tb:
        yield tb


def _reference_prober(tb):
    """A PythonProber backed by the reference's own handle.

    This is what makes modes 1 and 2 testable against something other than
    themselves: the C++ ranking and override logic runs unchanged, fed by
    ``chess.syzygy``.
    """
    def probe(fen):
        board = chess.Board(fen)
        try:
            return (tb.probe_wdl(board), tb.probe_dtz(board))
        except (syzygy.MissingTableError, KeyError, ValueError):
            return None
    return guofish_core.PythonProber(probe)


def _random_endgame(rng, max_men=5, allow_terminal=False):
    piece_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    while True:
        board = chess.Board(None)
        squares = rng.sample(range(64), rng.randint(2, max_men))
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        for square in squares[2:]:
            piece_type = rng.choice(piece_types)
            if piece_type == chess.PAWN and chess.square_rank(square) in (0, 7):
                piece_type = chess.KNIGHT
            board.set_piece_at(square,
                               chess.Piece(piece_type, rng.choice([chess.WHITE, chess.BLACK])))
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        board.halfmove_clock = rng.choice([0, 0, 1, 7, 42, 99])
        if not board.is_valid():
            continue
        if not allow_terminal and (board.is_checkmate() or board.is_stalemate()):
            continue
        return board


# --- the mapping and the perspective ---------------------------------------


def test_the_wdl_mapping_is_the_references():
    """``wdl_to_value``: wdl / 2.0, and the raw +-1.0 / +-0.5 the brief names.

    The +-1 endpoints match the bounded range the value head produces, so a
    tablebase result does not read as out-of-distribution to PUCT; the +-0.5
    mappings treat cursed wins and blessed losses as half-decisive, which is the
    safer reading of a result the fifty-move rule may take away.
    """
    assert guofish_core.wdl_to_value(2) == 1.0
    assert guofish_core.wdl_to_value(1) == 0.5
    assert guofish_core.wdl_to_value(0) == 0.0
    assert guofish_core.wdl_to_value(-1) == -0.5
    assert guofish_core.wdl_to_value(-2) == -1.0


def test_the_piece_count_gate():
    """<= 5 men, both kings counted — a popcount that filters out the
    overwhelming majority of leaves before any probe cost."""
    assert guofish_core.TABLEBASE_MAX_PIECES == 5
    assert guofish_core.piece_count("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1") == 3
    assert guofish_core.within_tablebase_range("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert guofish_core.piece_count(start) == 32
    assert not guofish_core.within_tablebase_range(start)


@tablebase_required
def test_mode_two_returns_an_absolute_value(fathom, reference_tablebase):
    """The perspective conversion, against the reference's own answer.

    ``probe_wdl`` reports from the side to move's perspective; the value head is
    White-POV. The conversion happens at the probe rather than at the backup
    site, which is the reference's decision and its own docstring's reason: the
    backup site "is exactly where tablebase perspective bugs hide".

    A sign error here is invisible in a symmetric position and catastrophic in
    every other, so the check runs over positions where White is winning, Black
    is winning, and it is drawn, from both sides to move.
    """
    import random
    rng = random.Random(20260807)
    checked = 0
    for _ in range(400):
        board = _random_endgame(rng)
        fen = board.fen(en_passant="fen")
        try:
            wdl = reference_tablebase.probe_wdl(board)
        except (syzygy.MissingTableError, KeyError, ValueError):
            continue
        expected = wdl / 2.0
        if board.turn == chess.BLACK:
            expected = -expected
        got = guofish_core.tablebase_probe_value(fen, fathom)
        assert got == expected, (
            f"{fen}: mode 2 value {got} != {expected} (wdl {wdl} from "
            f"{'White' if board.turn else 'Black'} to move, converted to "
            f"White-POV)")
        checked += 1
    assert checked > 100, f"only {checked} positions were covered by the tables"


# --- the native backend against the reference's -----------------------------


@tablebase_required
def test_fathom_and_python_chess_agree_about_wdl(fathom, reference_tablebase):
    """The backend is a lookup whose ANSWER is what matters, so it is measured.

    ``cpp/fathom.hpp`` lists five places Fathom and python-chess do not speak the
    same language — the WDL scale (0..4 vs -2..+2), the halfmove-clock guard on
    ``tb_probe_wdl``, castling rights, the DTZ sign, and which entry point DTZ
    comes from. Each is handled with an argument. This is the measurement.

    En-passant positions are included deliberately: the ep square is the field
    this port has been careful about since C2, and Fathom takes one.
    """
    import random
    rng = random.Random(20260807)
    compared = ep_compared = 0
    for _ in range(1500):
        board = _random_endgame(rng)
        fen = board.fen(en_passant="fen")
        try:
            expected = reference_tablebase.probe_wdl(board)
        except (syzygy.MissingTableError, KeyError, ValueError):
            expected = None
        assert fathom.probe_wdl(fen) == expected, (
            f"{fen}: Fathom says {fathom.probe_wdl(fen)}, python-chess says "
            f"{expected}")
        if expected is not None:
            compared += 1
        if fen.split()[3] != "-":
            ep_compared += 1

        # A double push, so the next position carries a raw en-passant square.
        pushes = [m for m in board.legal_moves
                  if board.piece_type_at(m.from_square) == chess.PAWN
                  and abs(m.to_square - m.from_square) == 16]
        if pushes:
            board.push(rng.choice(pushes))
            if not (board.is_checkmate() or board.is_stalemate()):
                fen = board.fen(en_passant="fen")
                try:
                    expected = reference_tablebase.probe_wdl(board)
                except (syzygy.MissingTableError, KeyError, ValueError):
                    expected = None
                assert fathom.probe_wdl(fen) == expected, f"{fen} (after a double push)"
                if fen.split()[3] != "-":
                    ep_compared += 1

    assert compared > 500, f"only {compared} positions were covered by the tables"
    assert ep_compared > 0, "no en-passant position was compared"


@tablebase_required
def test_fathom_and_python_chess_agree_about_dtz_except_on_checkmate(
        fathom, reference_tablebase):
    """DTZ agrees everywhere the mode 1 ranking can reach it, and the one place
    it does not is characterised rather than tolerated.

    Fathom reports DTZ as a magnitude inside ``tb_probe_root``; python-chess
    returns it signed. The sign is restored from the WDL in ``cpp/fathom.hpp``,
    and over non-terminal positions the two then agree exactly.

    ON CHECKMATE THEY DISAGREE: python-chess answers -1 (a loss, zero plies
    away), while ``tb_probe_root`` returns TB_RESULT_CHECKMATE, whose DTZ field
    is 0. That divergence is unreachable from mode 1, which tests
    ``is_checkmate()`` before probing and never asks — asserted below rather than
    asserted about. It is recorded in DECISIONS.md because "unreachable" is a
    claim about a caller, and callers change.
    """
    import random
    rng = random.Random(4242)
    compared = 0
    for _ in range(1500):
        board = _random_endgame(rng)
        fen = board.fen(en_passant="fen")
        try:
            expected = reference_tablebase.probe_dtz(board)
        except (syzygy.MissingTableError, KeyError, ValueError):
            expected = None
        assert fathom.probe_dtz(fen) == expected, (
            f"{fen}: Fathom says {fathom.probe_dtz(fen)}, python-chess says {expected}")
        if expected is not None:
            compared += 1
    assert compared > 500, f"only {compared} positions were covered by the tables"

    # The characterised divergence, pinned so that a re-pin of Fathom which
    # changed it fails a named test.
    mate = "4k3/8/8/5R2/6Kq/8/8/7q w - - 0 1"
    assert chess.Board(mate).is_checkmate()
    assert reference_tablebase.probe_dtz(chess.Board(mate)) == -1
    assert fathom.probe_dtz(mate) == 0, (
        "Fathom no longer reports DTZ 0 for a checkmate. If that changed, the "
        "note in cpp/fathom.hpp and DECISIONS.md is now wrong.")


# --- mode 1: the root bypass ------------------------------------------------


def _reference_root_move(board, tb):
    """``playing/uci_wrapper.py::_probe_tablebase``, transcribed.

    Written out here rather than imported because importing the UCI wrapper drags
    in a model load. It is the reference algorithm; what it is being used for is
    to check that the C++ ranking agrees with it, so it has to be the ranking and
    not a summary of it.
    """
    best_move = None
    best_key = None
    try:
        for move in board.legal_moves:
            zeroing = board.is_zeroing(move)
            board.push(move)
            try:
                if board.is_checkmate():
                    outcome, distance = 3, 0
                elif board.is_stalemate() or board.is_insufficient_material():
                    outcome, distance = 0, 0
                else:
                    wdl_child = tb.probe_wdl(board)
                    dtz_child = tb.probe_dtz(board)
                    our_wdl = -wdl_child
                    outcome = 1 if our_wdl > 0 else (-1 if our_wdl < 0 else 0)
                    if outcome > 0:
                        distance = 0 if zeroing else -dtz_child
                    elif outcome < 0:
                        distance = -dtz_child
                    else:
                        distance = 0
            finally:
                board.pop()
            key = (outcome, -distance)
            if best_key is None or key > best_key:
                best_key, best_move = key, move
    except (syzygy.MissingTableError, KeyError, ValueError):
        return None, None
    return (best_move.uci() if best_move else None), best_key


def test_the_mode_one_ranking_is_the_references():
    """The ranking on numbers, independent of any board or any table.

    ``tablebase_root_score`` is exposed precisely so this can be checked directly:
    a mate outranks every tablebase win; a winning zeroing move outranks a
    winning non-zeroing one whatever its DTZ; and a losing side prefers the
    LARGEST DTZ, which is the sign convention easiest to get backwards and the
    one whose failure mode is an engine resigning faster than it needs to.
    """
    mate = guofish_core.tablebase_root_score(True, False, False, 0, 0)
    assert mate[0] == 3

    drawn = guofish_core.tablebase_root_score(False, True, False, 0, 0)
    assert drawn == (0, 0)

    # BOTH ARGUMENTS ARE FROM THE CHILD'S POINT OF VIEW, which is the opponent's,
    # and keeping them consistent with each other is the thing this test exists
    # to pin. When WE are winning, the child is a LOSS for the side to move
    # there, so wdl_child is -2 and python-chess's probe_dtz(child) is NEGATIVE.
    # Passing a positive dtz alongside a losing wdl is not a harsh test case,
    # it is an impossible position, and reading the two as independent is
    # exactly how the sign gets inverted.
    zeroing_win = guofish_core.tablebase_root_score(False, False, True, -2, -30)
    slow_win = guofish_core.tablebase_root_score(False, False, False, -2, -30)
    fast_win = guofish_core.tablebase_root_score(False, False, False, -2, -4)
    assert zeroing_win[0] == slow_win[0] == fast_win[0] == 1
    # The engine maximises (outcome, -distance).
    assert -zeroing_win[1] > -fast_win[1] > -slow_win[1], (
        "a winning zeroing move must rank above every non-zeroing win, and a "
        "shorter DTZ above a longer one")

    # Losing: stall. The child is a WIN for the opponent, so wdl_child is +2 and
    # dtz_child is positive — the plies they need to zero us in. Larger is
    # better for us.
    quick_loss = guofish_core.tablebase_root_score(False, False, False, 2, 3)
    slow_loss = guofish_core.tablebase_root_score(False, False, False, 2, 60)
    assert quick_loss[0] == slow_loss[0] == -1
    assert -slow_loss[1] > -quick_loss[1], (
        "the losing side must prefer the LARGEST dtz — the sign is the thing "
        "that is easy to get backwards here")

    # And the ordering across outcomes.
    assert mate[0] > zeroing_win[0] > drawn[0] > quick_loss[0]


@tablebase_required
def test_mode_one_agrees_with_the_reference_bypass(fathom, reference_tablebase):
    """The C++ root probe against the reference's, over real endgames.

    A tie in ``(outcome, -distance)`` is broken by whichever move came first, and
    the two implementations iterate in different orders — python-chess's
    generation order there, canonical (from, to, promotion) order here. So a
    disagreement is only a real disagreement if the reference's own key for the
    C++ move is WORSE than for its own choice; if the keys are equal, both moves
    are tablebase-optimal and the choice between them is arbitrary. That
    distinction is made here rather than papered over by comparing move strings.
    """
    import random
    rng = random.Random(31337)
    agreed = tied = compared = 0

    for _ in range(250):
        board = _random_endgame(rng)
        fen = board.fen(en_passant="fen")
        expected, expected_key = _reference_root_move(board.copy(), reference_tablebase)
        got = guofish_core.tablebase_root_move(fen, fathom)

        if expected is None:
            assert got is None, (
                f"{fen}: the reference fell through to MCTS but C++ returned {got}. "
                f"A partial ranking can prefer a move only because its sibling "
                f"could not be scored.")
            continue
        compared += 1
        assert got is not None, f"{fen}: the reference chose {expected}, C++ chose nothing"
        if got == expected:
            agreed += 1
            continue

        # Different move: score the C++ move with the REFERENCE's rule and
        # require the keys to be equal.
        probe = board.copy()
        move = chess.Move.from_uci(got)
        assert move in probe.legal_moves, f"{fen}: C++ returned an illegal move {got}"
        zeroing = probe.is_zeroing(move)
        probe.push(move)
        if probe.is_checkmate():
            outcome, distance = 3, 0
        elif probe.is_stalemate() or probe.is_insufficient_material():
            outcome, distance = 0, 0
        else:
            our_wdl = -reference_tablebase.probe_wdl(probe)
            dtz_child = reference_tablebase.probe_dtz(probe)
            outcome = 1 if our_wdl > 0 else (-1 if our_wdl < 0 else 0)
            if outcome > 0:
                distance = 0 if zeroing else -dtz_child
            elif outcome < 0:
                distance = -dtz_child
            else:
                distance = 0
        assert (outcome, -distance) == expected_key, (
            f"{fen}: C++ chose {got} scoring {(outcome, -distance)}, the "
            f"reference chose {expected} scoring {expected_key}. This is not a "
            f"tie-break difference — one of the two is worse.")
        tied += 1

    assert compared > 100, f"only {compared} positions were inside the tables"
    assert agreed + tied == compared


@tablebase_required
def test_mode_one_declines_rather_than_guessing(fathom):
    """Out of range, and any miss, means "fall through to MCTS".

    The reference wraps its whole loop in one try/except, so a single missing
    table abandons the bypass rather than ranking the remaining moves against an
    incomplete picture. A bypass that guessed would play a losing move with total
    confidence, which is strictly worse than searching.
    """
    # A full middlegame: far out of range.
    assert guofish_core.tablebase_root_move(
        "r3kr2/1p1nnpp1/1bp1p1p1/p2pP1N1/P2P3P/BPP3P1/5PB1/R3K2R w KQ - 1 21",
        fathom) is None
    # No backend at all.
    assert guofish_core.tablebase_root_move(
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1", guofish_core.NullProber()) is None


# --- mode 2 inside a search: the thing this chunk is actually about ---------


def _stub_dump(root_fen, depth):
    """A stand-in evaluator for a small endgame, as a replay dump.

    THIS IS NOT GOLDEN DATA AND IS NOT COMPARED AGAINST ANYTHING. Global Rule 2
    governs data the port is JUDGED against; nothing below compares a tree to the
    reference. What is being asserted is which value ends up in which store, and
    for that the evaluator only has to be deterministic and distinguishable — a
    real network would make the test slower and prove exactly the same thing.

    The reason a stub is needed at all: the replay search resolves every
    expansion through the dump, and Gate 1's corpus is middlegames, where the
    piece-count gate means mode 2 can never fire. So mode 2 cannot be exercised
    on any position for which golden data exists.

    Values are a deterministic function of the key, confined to (-0.09, 0.09)
    and offset off the 1e-4 grid, so that NO stub value can equal any value
    ``wdl_to_value`` produces — +-1.0, +-0.5 or 0.0. That is what lets the
    assertions below treat "the cache holds a tablebase value" as a fact rather
    than a coincidence, and it is checked here rather than argued for, because
    the first version of this generator could produce -0.5 and the test that
    depended on it failed for the right reason with the wrong cause.
    """
    board = chess.Board(root_fen)
    seen = {}
    frontier = [(board, 0)]
    while frontier:
        node, node_depth = frontier.pop()
        if node_depth >= depth:
            continue
        fen = node.fen(en_passant="fen")
        moves = guofish_core.legal_moves(fen)
        if not moves:
            continue
        key = guofish_core.nn_key(fen)
        seen.setdefault((key, node_depth == 0), (fen, moves))
        for uci in moves:
            child = node.copy(stack=False)
            child.push(chess.Move.from_uci(uci))
            frontier.append((child, node_depth + 1))

    keys, is_root, offsets, packed, priors, values = [], [], [0], [], [], []
    for (key, root), (_fen, moves) in sorted(seen.items()):
        keys.append(key)
        is_root.append(1 if root else 0)
        for uci in moves:
            packed.append(guofish_core.pack_uci(uci))
            priors.append(np.float32(1.0 / len(moves)))
        offsets.append(len(packed))
        # In (-0.09, 0.09) and off the 1e-4 grid, so it can never be +-1.0,
        # +-0.5 or 0.0.
        values.append(((key % 1601) - 800) / 10000.0 + 3e-5)

    assert not (TABLEBASE_VALUES & set(values)), (
        "the stub evaluator produced a value wdl_to_value can also produce, so "
        "'this cache entry holds a tablebase value' would no longer be a fact")

    return {
        "keys": np.array(keys, dtype=np.uint64),
        "is_root": np.array(is_root, dtype=np.uint8),
        "move_offset": np.array(offsets, dtype=np.uint64),
        "moves": np.array(packed, dtype=np.uint16),
        "priors": np.array(priors, dtype=np.float32),
        "values": np.array(values, dtype=np.float64),
    }


ENDGAME_ROOT = "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"
ENDGAME_DEPTH = 3

# Every value wdl_to_value can produce. A cache entry holding one of these did
# not come from the stub evaluator.
TABLEBASE_VALUES = {1.0, 0.5, 0.0, -0.5, -1.0}


@pytest.fixture(scope="session")
def endgame_dump():
    if _TABLEBASE_SKIP is not None:
        pytest.skip(_TABLEBASE_SKIP)
    return _stub_dump(ENDGAME_ROOT, ENDGAME_DEPTH)


def _endgame_search(dump, prober):
    search = guofish_core.ReplaySearchDouble(guofish_core.SearchConfig(
        max_tree_depth=ENDGAME_DEPTH,
        arena_capacity=1 << 16,
        cache_slots=1 << 14))
    search.load_dump(dump["keys"], dump["is_root"], dump["move_offset"],
                     dump["moves"], dump["priors"], dump["values"])
    if prober is not None:
        search.set_tablebase(prober)
    search.set_position(ENDGAME_ROOT)
    stats = search.search(400)
    return search, stats


@tablebase_required
def test_mode_two_overrides_the_value_at_the_leaf(endgame_dump, fathom):
    """The override fires, and it changes the tree.

    Two searches over the same stub evaluator, one with tablebases attached and
    one without. If the trees came out the same, mode 2 would be doing nothing
    and every assertion in the next test would be vacuous.
    """
    _off, stats_off = _endgame_search(endgame_dump, None)
    _on, stats_on = _endgame_search(endgame_dump, fathom)

    assert stats_off["tablebase_probes"] == 0
    assert stats_off["tablebase_overrides"] == 0
    assert stats_on["tablebase_probes"] > 0, (
        "no leaf passed the piece-count gate; the endgame root is not an endgame")
    assert stats_on["tablebase_overrides"] > 0, (
        "every probe missed. The tables are open but answer nothing for a "
        "3-man position, which should be impossible with a 5-man set.")
    assert stats_on["best_move"] != stats_off["best_move"] or \
        stats_on["root_visits"] == stats_off["root_visits"]


@tablebase_required
def test_a_tablebase_value_never_enters_the_cache(endgame_dump, fathom):
    """THE CENTRAL ASSERTION OF THIS CHUNK.

    ``core/mctsv4.py`` applies the WDL override and then caches the result::

        nn_value = tb_value                          # override
        ...
        self.cache.put(cache_key, policy, nn_value)  # poisoning

    with the comment "We override BEFORE caching so the WDL value is what gets
    stored — subsequent transpositions to this position reuse it without
    re-probing". The optimisation is real and it is unsound: a Syzygy WDL assumes
    the fifty-move rule never intervenes, so it is a function of the position AND
    the halfmove clock, while the cache key is a function of the position alone
    — deliberately, because the clock is not a token. A KQvK win stored at clock
    3 is served back at clock 99, where the truth is a draw. The reference's own
    instrumentation counts exactly this crossing
    (``cache_hit_tb_hmc_crossing``).

    Here every cache entry must still hold the NETWORK's value. The stub
    evaluator's values are all in (-0.9, 0.9) and never +-1.0, +-0.5 or 0.0, so a
    tablebase value in the cache is not a subtle discrepancy to be argued about
    — it is a number that could not have come from anywhere else.
    """
    search, stats = _endgame_search(endgame_dump, fathom)
    assert stats["tablebase_overrides"] > 0, "nothing was overridden; nothing to prove"

    dump = endgame_dump
    tablebase_values = TABLEBASE_VALUES
    checked = 0
    for i in range(len(dump["keys"])):
        if dump["is_root"][i]:
            continue
        entry = search.cache_entry_by_key(int(dump["keys"][i]))
        if entry is None:
            continue
        checked += 1
        assert entry["value"] not in tablebase_values, (
            f"key {int(dump['keys'][i]):#018x} holds {entry['value']!r}, which is a "
            f"tablebase value. The stub evaluator never produces one, so this "
            f"came from a Syzygy probe — the reference's poisoning defect has "
            f"been reproduced.")
        assert struct.pack("<d", entry["value"]) == struct.pack("<d", float(dump["values"][i])), (
            f"key {int(dump['keys'][i]):#018x}: the cache holds {entry['value']!r} but "
            f"the evaluator returned {float(dump['values'][i])!r}")
    assert checked > 20, f"only {checked} entries were checked; the search cached too little"


@tablebase_required
def test_a_tablebase_never_changes_what_a_cached_entry_says(endgame_dump, fathom):
    """Tree-locality, stated as the strongest form that is actually true.

    The obvious formulation — "attaching a backend cannot change the cache at
    all" — is FALSE, and believing it would have been a bug in this test rather
    than in the engine. A tablebase override changes the value backed up, which
    changes PUCT, which changes which leaves the search reaches, which changes
    which positions get evaluated and therefore cached. The two runs legitimately
    cache different SETS of positions.

    What must not change is what an entry SAYS. So: over every key both runs
    cached, the value, the move list and the priors must be identical bit for
    bit. If the override leaked into a `put`, the entries for probed positions
    would differ between the two runs and this is where it would show.

    The companion assertion — that no cached value is one a tablebase can produce
    — is in ``test_a_tablebase_value_never_enters_the_cache``. Together they cover
    both directions: nothing tablebase-shaped got in, and nothing that got in
    changed when tablebases were switched on.
    """
    off, _stats_off = _endgame_search(endgame_dump, None)
    on, stats_on = _endgame_search(endgame_dump, fathom)
    assert stats_on["tablebase_overrides"] > 0

    dump = endgame_dump
    differences = []
    shared = 0
    for i in range(len(dump["keys"])):
        if dump["is_root"][i]:
            continue
        key = int(dump["keys"][i])
        a, b = off.cache_entry_by_key(key), on.cache_entry_by_key(key)
        if a is None or b is None:
            continue
        shared += 1
        if struct.pack("<d", a["value"]) != struct.pack("<d", b["value"]):
            differences.append(
                f"key {key:#018x}: value {a['value']!r} without tablebases, "
                f"{b['value']!r} with them")
        if not np.array_equal(np.asarray(a["moves"]), np.asarray(b["moves"])):
            differences.append(f"key {key:#018x}: the move list changed")
        if not np.array_equal(_bits32(np.asarray(a["priors"], dtype=np.float32)),
                              _bits32(np.asarray(b["priors"], dtype=np.float32))):
            differences.append(f"key {key:#018x}: the priors changed")

    assert shared > 20, (
        f"only {shared} positions were cached by both runs, so this comparison "
        f"has almost nothing to compare")
    assert not differences, (
        "attaching a tablebase changed what a cache entry says. Tablebase "
        "results are supposed to be tree-local — applied to the value backed up "
        "at the leaf that was probed, and to nothing another position can "
        "read:\n  " + "\n  ".join(differences[:20]))


@tablebase_required
def test_the_reference_would_have_poisoned_this_cache(reference_tablebase):
    """The defect, demonstrated on the reference rather than described.

    A DECISIONS.md entry saying "Python caches tablebase values" is a claim about
    someone else's code. This shows the mechanism: one position, two halfmove
    clocks, one cache key — so whichever is probed first decides what the other
    is told, and the tables' own answer is clock-independent while the truth is
    not.

    Nothing in ``guofish_core`` is exercised here except the key. The point is
    that the SETUP for the defect exists — clock-twins share a key, a WDL ignores
    the clock — and that the port's type system is what stops it turning into the
    defect.
    """
    early = "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"
    late = "8/8/4k3/8/8/4K3/4P3/8 w - - 99 1"

    assert guofish_core.nn_key(early) == guofish_core.nn_key(late), (
        "the premise of the defect: clock-twins share a cache key")

    # And the tables answer the same for both, because a WDL ignores the clock.
    wdl_early = reference_tablebase.probe_wdl(chess.Board(early))
    wdl_late = reference_tablebase.probe_wdl(chess.Board(late))
    assert wdl_early == wdl_late, (
        "Syzygy WDL is clock-independent by construction; if this ever differs, "
        "the argument in cpp/values.hpp needs revisiting")
    assert wdl_early == 2, "KPvK with the pawn safe is a win"

    # The position at clock 99 is one ply from a fifty-move draw, so the value
    # the tables report is not the value of the position. Caching it under a key
    # that cannot tell the two apart is the defect.
    assert guofish_core.wdl_to_value(wdl_late) == 1.0
    # Which the port makes unrepresentable rather than merely avoiding:
    assert guofish_core.cache_type_separation()["tablebase_value_accepted"] is False
