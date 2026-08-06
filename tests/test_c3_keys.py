"""C3 acceptance tests — the two position keys, and the proof they cannot mix.

`golden/keys.jsonl` and `golden/keys_adversarial.jsonl` are the reference
(Global Rule 2). They were produced by `tools/gen_key_golden.py` from
`core.mctsv4.board_to_tokens` and python-chess's `Board._transposition_key()`,
and they are never regenerated to agree with the C++ side. If the two disagree,
the C++ side is wrong.

What is actually being tested
-----------------------------
Not "does the hash work". A hash that works is not the deliverable; two keys
that answer two *different* questions and are never confused for one another is.
The engine's history here is a Polyglot Zobrist — a perfectly good hash — used
for both, and the resulting bugs were invisible: a cache hit returning another
position's policy, or a threefold claim on a position that never repeated.

So the suite is organised by the ways that failure can come back:

  * **Partition equivalence** over the 100k corpus, for each key separately.
    This is the acceptance criterion in the chunk brief, and it is the property
    the cache and the repetition table actually depend on: `golden[i] ==
    golden[j]` if and only if `cpp[i] == cpp[j]`. It is computed as a groupby
    per side and one comparison of the two groupings, not an O(n^2) sweep.
  * **Value equality**, which is stronger than required and currently holds.
    See `test_values_match_golden_exactly` for why it is asserted anyway and
    what it would mean if it were the only thing to fail.
  * **The adversarial pairs**, which are the only coverage of the three
    en-passant rules that does not depend on the corpus happening to contain
    the right accident. Each pair carries the relation it is meant to prove, so
    the expectation travels with the data instead of being restated here.
  * **Corpus coverage**, asserting the 100k file really contains the cases the
    parity run would otherwise pass vacuously — en-passant positions with no
    legal capture, and groups of positions that differ only in the clock.
  * **Type separation**, criterion 3, read out of the compiler rather than
    argued about. `guofish_core.key_type_separation()` reports what
    `<type_traits>` answered while the module was built.

Nothing here imports `chess`. The reference's answers reach this file only
through the golden data, so the tests cannot drift into re-deriving the
expectation from the same library that produced it.

Pointing the parity tests at a different file
---------------------------------------------
``GUOFISH_GOLDEN_KEYS`` and ``GUOFISH_GOLDEN_KEYS_ADVERSARIAL`` override the
inputs. This exists so the diagnostic output can be exercised against a
deliberately corrupted copy without writing to ``golden/`` (Global Rules 1
and 2); see DECISIONS.md, "C3 / mutation check". The corpus-coverage tests skip
when the override is set.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = REPO_ROOT / "golden" / "keys.jsonl"
DEFAULT_ADVERSARIAL = REPO_ROOT / "golden" / "keys_adversarial.jsonl"

GOLDEN_ENV = "GUOFISH_GOLDEN_KEYS"
ADVERSARIAL_ENV = "GUOFISH_GOLDEN_KEYS_ADVERSARIAL"

# How many mismatching positions are reported in full before the report is
# truncated. A systematically wrong key produces all 100k of them and the first
# few are enough to name the bug.
MAX_REPORTED = 10

SAME = "same"
DIFFER = "differ"


# ---------------------------------------------------------------------------
# Golden data
# ---------------------------------------------------------------------------


def _jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def golden():
    path = Path(os.environ.get(GOLDEN_ENV, DEFAULT_GOLDEN))
    if not path.exists():
        pytest.fail(
            f"golden key data missing: {path}\n"
            "Global Rule 2: regenerate it with `python tools/gen_key_golden.py`, "
            "which reads the Python reference. It is never produced from C++ output."
        )
    rows = _jsonl(path)
    assert rows, f"{path} is empty"
    return {
        "path": path,
        "fens": [r["fen"] for r in rows],
        "nn": np.array([r["nn_key"] for r in rows], dtype=np.uint64),
        "rep": np.array([r["rep_key"] for r in rows], dtype=np.uint64),
    }


@pytest.fixture(scope="module")
def adversarial():
    path = Path(os.environ.get(ADVERSARIAL_ENV, DEFAULT_ADVERSARIAL))
    if not path.exists():
        pytest.fail(
            f"adversarial key pairs missing: {path}\n"
            "Global Rule 2: regenerate with `python tools/gen_key_golden.py`."
        )
    pairs = _jsonl(path)
    assert pairs, f"{path} is empty"
    return pairs


@pytest.fixture(scope="module")
def computed(golden):
    """Both keys for every corpus FEN, from the batched C++ entry point."""
    pairs = guofish_core.key_pairs(golden["fens"])
    assert pairs.dtype == np.uint64
    assert pairs.shape == (len(golden["fens"]), 2)
    return {"nn": pairs[:, 0], "rep": pairs[:, 1]}


def using_override(env):
    return env in os.environ


# ---------------------------------------------------------------------------
# Partition equivalence — the stated acceptance criterion
# ---------------------------------------------------------------------------


def partition(values):
    """Canonical labelling of `values`: each entry maps to the first index that
    carried its value.

    Two sequences have the same partition exactly when they agree about which
    positions are equal to which, whatever the values themselves are. Built with
    a dict, so this is O(n) rather than the O(n^2) pairwise sweep the property is
    usually stated as.
    """
    first = {}
    return [first.setdefault(value, index) for index, value in enumerate(values)]


def report_partition_divergence(name, fens, golden_values, cpp_values):
    golden_part = partition(golden_values)
    cpp_part = partition(cpp_values)
    if golden_part == cpp_part:
        return None

    lines = [f"{name}: C++ does not group the corpus the way the reference does."]
    shown = 0
    for i, (g, c) in enumerate(zip(golden_part, cpp_part)):
        if g == c:
            continue
        shown += 1
        if shown > MAX_REPORTED:
            break
        if g < i and c == i:
            lines.append(
                f"  row {i}: the reference says this position is the same as row {g}, "
                f"C++ says it is distinct\n"
                f"    row {i}: {fens[i]}\n"
                f"    row {g}: {fens[g]}\n"
                f"    C++ keys {int(cpp_values[i]):#018x} vs {int(cpp_values[g]):#018x}"
            )
        elif c < i and g == i:
            lines.append(
                f"  row {i}: C++ says this position is the same as row {c}, "
                f"the reference says it is distinct\n"
                f"    row {i}: {fens[i]}\n"
                f"    row {c}: {fens[c]}\n"
                f"    reference keys {int(golden_values[i]):#018x} vs {int(golden_values[c]):#018x}"
            )
        else:
            lines.append(f"  row {i}: reference groups with {g}, C++ groups with {c}\n    {fens[i]}")

    total = sum(1 for g, c in zip(golden_part, cpp_part) if g != c)
    lines.append(
        f"  {total} of {len(fens)} rows are grouped differently; "
        f"reference has {len(set(golden_part))} distinct positions, C++ has {len(set(cpp_part))}."
    )
    return "\n".join(lines)


def test_nn_key_partition_matches_reference(golden, computed):
    """The C3 criterion for nn_key: same equivalence classes as the reference.

    This is what a transposition cache depends on. A key that grouped two
    differently-tokenized positions together would serve one position's policy
    and value for the other — the original defect — and a key that split a
    position from itself would merely lose hits.
    """
    failure = report_partition_divergence("nn_key", golden["fens"], golden["nn"], computed["nn"])
    assert failure is None, failure


def test_rep_key_partition_matches_reference(golden, computed):
    """The same criterion for rep_key, where the consequence is a wrong draw.

    Grouping two positions that are not the same position lets a threefold be
    claimed that did not happen; splitting one that is loses a real draw.
    """
    failure = report_partition_divergence("rep_key", golden["fens"], golden["rep"], computed["rep"])
    assert failure is None, failure


def test_values_match_golden_exactly(golden, computed):
    """Stronger than the criterion above, and it currently holds for all 100k.

    The brief asks only for partition equivalence, because both Python keys are
    tuples and there was never a value to match; the golden file's numbers are
    the generator's own FNV-1a serialisation. Asserting equality anyway pins that
    serialisation, which is worth pinning: C7's cache will persist these, and a
    silent change to the byte layout would invalidate stored data while every
    partition test still passed.

    If this is the ONLY test in the file that fails, the C++ side changed how it
    serialises a payload without changing which positions it considers equal.
    That is a decision, not a bug — but it is a decision, and it should be made
    on purpose in DECISIONS.md rather than discovered later.
    """
    for name in ("nn", "rep"):
        mismatched = np.nonzero(golden[name] != computed[name])[0]
        if mismatched.size == 0:
            continue
        lines = [f"{name}_key: {mismatched.size} of {len(golden['fens'])} positions differ from the reference."]
        for i in mismatched[:MAX_REPORTED]:
            lines.append(
                f"  row {int(i)}: {golden['fens'][int(i)]}\n"
                f"    reference {int(golden[name][int(i)]):#018x}\n"
                f"    C++       {int(computed[name][int(i)]):#018x}"
            )
        if mismatched.size > MAX_REPORTED:
            lines.append(f"  ... and {mismatched.size - MAX_REPORTED} more")
        pytest.fail("\n".join(lines))


def test_single_position_entry_point_agrees_with_batch(golden, computed):
    """`nn_key(fen)` and `nn_keys(fens)` are different code paths into the same
    encoder, and only the batched one is what the search will use. A sample is
    enough: the two share everything below the argument marshalling, so a
    divergence would be systematic rather than positional.
    """
    step = max(1, len(golden["fens"]) // 500)
    for i in range(0, len(golden["fens"]), step):
        fen = golden["fens"][i]
        assert guofish_core.nn_key(fen) == int(computed["nn"][i]), fen
        assert guofish_core.rep_key(fen) == int(computed["rep"][i]), fen
        assert guofish_core.keys(fen) == (int(computed["nn"][i]), int(computed["rep"][i])), fen


def test_batch_helpers_agree_with_key_pairs(golden, computed):
    fens = golden["fens"][:4096]
    np.testing.assert_array_equal(guofish_core.nn_keys(fens), computed["nn"][:4096])
    np.testing.assert_array_equal(guofish_core.rep_keys(fens), computed["rep"][:4096])


# ---------------------------------------------------------------------------
# The two keys are never each other
# ---------------------------------------------------------------------------


def test_the_two_keys_never_coincide(golden, computed):
    """The runtime half of "never mixed": a swapped key cannot compare equal.

    The payloads carry different domain tags, so `nn_key(p) == rep_key(p)` would
    take a genuine 64-bit collision. Over 100k positions the chance is ~5e-10; if
    this fires, a tag is missing rather than luck having run out.
    """
    coincide = np.nonzero(computed["nn"] == computed["rep"])[0]
    assert coincide.size == 0, (
        f"{coincide.size} positions have nn_key == rep_key, e.g. row {int(coincide[0])}: "
        f"{golden['fens'][int(coincide[0])]} -> {int(computed['nn'][int(coincide[0])]):#018x}. "
        "The domain tags are not doing their job."
    )


def test_keys_are_deterministic(golden):
    """Same FEN, same key, every time. Rules out anything order- or
    allocation-dependent leaking into a payload."""
    sample = golden["fens"][:256]
    np.testing.assert_array_equal(guofish_core.key_pairs(sample), guofish_core.key_pairs(sample))
    np.testing.assert_array_equal(guofish_core.key_pairs(sample), guofish_core.key_pairs(list(reversed(sample)))[::-1])


# ---------------------------------------------------------------------------
# Compile-time separation — acceptance criterion 3
# ---------------------------------------------------------------------------


def test_nn_key_and_rep_key_are_incompatible_cpp_types():
    """Criterion 3: passing one where the other is expected is a compile error.

    Read out of `<type_traits>` as the compiler answered it while building this
    module, not asserted about at runtime. `nn_accepted_as_rep` is literally
    `std::is_invocable_v<decltype(void(RepKey)), NNKey>` — the criterion's own
    wording, in code.

    cpp/keys.hpp static_asserts every one of these, so a build in which any is
    wrong does not produce a module to import. This test is what makes that
    visible from the suite, where someone is looking.
    """
    facts = guofish_core.key_type_separation()

    must_be_false = [
        "nn_accepted_as_rep",
        "rep_accepted_as_nn",
        "nn_converts_to_rep",
        "rep_converts_to_nn",
        "nn_constructible_from_rep",
        "rep_constructible_from_nn",
        "nn_assignable_from_rep",
        "rep_assignable_from_nn",
        "uint64_converts_to_nn",
        "uint64_converts_to_rep",
        "nn_converts_to_uint64",
        "rep_converts_to_uint64",
        "nn_comparable_to_rep",
        "rep_comparable_to_nn",
    ]
    violated = [name for name in must_be_false if facts[name]]
    assert not violated, (
        "the two key types are not separate: " + ", ".join(violated) + ". "
        "A raw `using NNKey = uint64_t;` alias would report exactly this."
    )

    # ... and they are still usable as themselves. A type nothing can be done
    # with would pass every assertion above.
    for name in ("nn_accepted_as_nn", "rep_accepted_as_rep", "nn_comparable_to_nn", "rep_comparable_to_rep"):
        assert facts[name], f"{name} should hold: the key types must remain usable"

    # No space is paid for the safety.
    assert facts["nn_size"] == facts["rep_size"] == 8


# ---------------------------------------------------------------------------
# Adversarial pairs
#
# Each record carries the relation it is meant to prove, so these tests state
# the property once and let the data supply the cases. The keys are recomputed
# from the FEN by C++; the golden values are used only through the declared
# relation, which is what makes a pair a test of behaviour rather than of
# arithmetic.
# ---------------------------------------------------------------------------


def relation(a, b):
    return SAME if a == b else DIFFER


def test_adversarial_file_covers_the_required_kinds(adversarial):
    kinds = {}
    for pair in adversarial:
        kinds[pair["kind"]] = kinds.get(pair["kind"], 0) + 1

    assert len(adversarial) >= 18, f"the brief asks for ~20 constructed pairs, found {len(adversarial)}"
    for required in ("ep_twin", "clock_twin", "transposition"):
        assert kinds.get(required, 0) >= 3, f"only {kinds.get(required, 0)} {required} pairs: {kinds}"


def test_adversarial_pairs_hold_under_cpp(adversarial):
    """Every declared relation, recomputed by C++ from the two FENs."""
    failures = []
    for pair in adversarial:
        for key_name, expected, fn in (
            ("nn_key", pair["expect_nn"], guofish_core.nn_key),
            ("rep_key", pair["expect_rep"], guofish_core.rep_key),
        ):
            a, b = fn(pair["a"]["fen"]), fn(pair["b"]["fen"])
            got = relation(a, b)
            if got != expected:
                failures.append(
                    f"{pair['pair_id']} ({pair['kind']}): {key_name} should {expected}, C++ says {got}\n"
                    f"    {pair['desc']}\n"
                    f"    a: {pair['a']['fen']} -> {a:#018x}\n"
                    f"    b: {pair['b']['fen']} -> {b:#018x}"
                )
    assert not failures, "\n".join(failures)


def test_adversarial_pair_values_match_golden(adversarial):
    """The pairs are also plain parity cases; a relation can hold for two wrong
    keys, and these positions are hand-built precisely because they are the ones
    most likely to be wrong."""
    failures = []
    for pair in adversarial:
        for role in ("a", "b"):
            side = pair[role]
            nn, rep = guofish_core.keys(side["fen"])
            if nn != side["nn_key"] or rep != side["rep_key"]:
                failures.append(
                    f"{pair['pair_id']}.{role}: {side['fen']}\n"
                    f"    reference nn={side['nn_key']:#018x} rep={side['rep_key']:#018x}\n"
                    f"    C++       nn={nn:#018x} rep={rep:#018x}"
                )
    assert not failures, "\n".join(failures)


def test_clock_twins_share_an_nn_key(adversarial):
    """Acceptance criterion 4, and the one most likely to be "fixed" later.

    The halfmove clock is not one of the 68 tokens. The network cannot see it,
    so two positions differing only in the clock get the same evaluation, and a
    cache that distinguished them would simply waste entries. This is correct
    and deliberate. It is also exactly what a future reader will look at and
    call a bug, so it is pinned here with a comment they will find first.

    (The scope's standing constraint is what makes it safe: no proof and no
    path-dependent value may ever enter this cache. A cached mate score WOULD
    depend on the clock. Priors and a static value do not.)
    """
    twins = [p for p in adversarial if p["kind"] == "clock_twin"]
    assert twins, "no clock-twin pairs in the adversarial data"

    for pair in twins:
        a_fen, b_fen = pair["a"]["fen"], pair["b"]["fen"]
        assert a_fen.split()[:4] == b_fen.split()[:4], (
            f"{pair['pair_id']} is not a clock twin: the first four FEN fields differ\n"
            f"  a: {a_fen}\n  b: {b_fen}"
        )
        assert a_fen.split()[4] != b_fen.split()[4], (
            f"{pair['pair_id']} does not actually vary the halfmove clock: {a_fen}"
        )
        assert guofish_core.nn_key(a_fen) == guofish_core.nn_key(b_fen), (
            f"{pair['pair_id']}: nn_key must ignore the halfmove clock\n  a: {a_fen}\n  b: {b_fen}"
        )
        assert guofish_core.rep_key(a_fen) == guofish_core.rep_key(b_fen), (
            f"{pair['pair_id']}: rep_key must ignore the halfmove clock too — "
            f"_transposition_key() has no clock field\n  a: {a_fen}\n  b: {b_fen}"
        )


def test_en_passant_twins_follow_two_different_rules(adversarial):
    """The heart of the chunk. Same placement, one FEN with an ep square and one
    without:

      * `nn_key` must ALWAYS split them. Token 66 is written from
        `ep_square is not None`, unconditionally, so the two are different
        network inputs whatever the position looks like.
      * `rep_key` must split them only when an en-passant capture is actually
        legal, because that is `_transposition_key()`'s rule.

    A single implementation shared between the two keys cannot pass this, and
    neither can one built on Polyglot's pseudo-legal adjacency rule — the
    `ep_pinned_*` pairs are corpus positions where a pawn attacks the ep square
    but the capture is illegal, and Polyglot disagrees with both rules there.
    """
    twins = [p for p in adversarial if p["kind"] == "ep_twin"]
    assert twins, "no en-passant-twin pairs in the adversarial data"

    for pair in twins:
        a_fen, b_fen = pair["a"]["fen"], pair["b"]["fen"]
        assert guofish_core.nn_key(a_fen) != guofish_core.nn_key(b_fen), (
            f"{pair['pair_id']}: nn_key must distinguish ANY en-passant square\n"
            f"  a: {a_fen}\n  b: {b_fen}"
        )
        assert relation(guofish_core.rep_key(a_fen), guofish_core.rep_key(b_fen)) == pair["expect_rep"], (
            f"{pair['pair_id']}: rep_key should {pair['expect_rep']} here — {pair['desc']}\n"
            f"  a: {a_fen}\n  b: {b_fen}"
        )

    # Both branches of the legal-ep rule must be present, or the test above is
    # satisfied by an implementation that always answers the same way.
    outcomes = {p["expect_rep"] for p in twins}
    assert outcomes == {SAME, DIFFER}, (
        f"the ep twins only exercise rep_key {outcomes}; a rep_key that ignored the ep "
        "square entirely, or one that used the raw square, would pass"
    )


def test_transpositions_reach_one_position_by_two_routes(adversarial):
    """Both keys must be functions of the position and nothing else. A key that
    picked up anything path-dependent — a move counter, an insertion order, a
    stale field left over from the route taken — splits these."""
    pairs = [p for p in adversarial if p["kind"] == "transposition"]
    assert pairs, "no transposition pairs in the adversarial data"

    for pair in pairs:
        a_fen, b_fen = pair["a"]["fen"], pair["b"]["fen"]
        assert guofish_core.keys(a_fen) == guofish_core.keys(b_fen), (
            f"{pair['pair_id']}: {pair['desc']}\n  a: {a_fen}\n  b: {b_fen}"
        )


def test_near_misses_are_not_transpositions(adversarial):
    """Positions that look like transpositions and are not. Without these, an
    implementation that dropped the castling rights or the ep file altogether
    would pass every pair above."""
    pairs = [p for p in adversarial if p["kind"] == "near_miss"]
    assert pairs, "no near-miss pairs in the adversarial data"

    for pair in pairs:
        a_fen, b_fen = pair["a"]["fen"], pair["b"]["fen"]
        assert relation(guofish_core.nn_key(a_fen), guofish_core.nn_key(b_fen)) == pair["expect_nn"], (
            f"{pair['pair_id']}: nn_key should {pair['expect_nn']} — {pair['desc']}"
        )
        assert relation(guofish_core.rep_key(a_fen), guofish_core.rep_key(b_fen)) == pair["expect_rep"], (
            f"{pair['pair_id']}: rep_key should {pair['expect_rep']} — {pair['desc']}"
        )


# ---------------------------------------------------------------------------
# Corpus coverage
#
# The parity tests above are only as good as what the corpus contains. These
# assert that it contains the hard cases, so a green run means something.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(using_override(GOLDEN_ENV), reason="corpus claims apply to golden/keys.jsonl only")
def test_corpus_is_the_full_shared_c1_corpus(golden):
    assert len(golden["fens"]) == 100_000, f"expected the 100k C1 corpus, found {len(golden['fens'])}"
    assert len(set(golden["fens"])) == len(golden["fens"]), "the corpus contains duplicate FENs"

    movegen = REPO_ROOT / "golden" / "movegen.jsonl"
    if movegen.exists():
        with open(movegen, encoding="utf-8") as handle:
            shared = [json.loads(line)["fen"] for line in handle if line.strip()]
        assert golden["fens"] == shared, (
            "golden/keys.jsonl is not keyed on the same corpus, in the same order, as "
            "golden/movegen.jsonl — C1, C2 and C3 are supposed to share one position set"
        )


@pytest.mark.skipif(using_override(GOLDEN_ENV), reason="corpus claims apply to golden/keys.jsonl only")
def test_corpus_contains_en_passant_positions_of_both_kinds(golden, computed):
    """The corpus must contain ep positions where the capture is NOT legal, or
    the difference between the two keys' ep rules is never exercised by the
    parity sweep.

    Counted without a chess library: an ep FEN whose `rep_key` equals the
    `rep_key` of the same FEN with the ep field cleared is one where the legal-ep
    rule found no capture. Its `nn_key` must differ, because token 66 does not
    care.
    """
    ep_rows = [(i, fen) for i, fen in enumerate(golden["fens"]) if fen.split()[3] != "-"]
    assert len(ep_rows) > 1000, f"only {len(ep_rows)} corpus positions carry an en-passant square"

    no_capture = 0
    for i, fen in ep_rows:
        fields = fen.split()
        fields[3] = "-"
        stripped = " ".join(fields)
        if guofish_core.rep_key(stripped) == int(computed["rep"][i]):
            no_capture += 1
            assert guofish_core.nn_key(stripped) != int(computed["nn"][i]), (
                f"nn_key ignored the en-passant square on {fen} — this is the original defect"
            )

    assert no_capture > 100, (
        f"only {no_capture} of {len(ep_rows)} en-passant positions have no legal capture; "
        "the corpus does not exercise the gap between the two ep rules"
    )
    assert no_capture < len(ep_rows), (
        "every en-passant position in the corpus lacks a legal capture; rep_key's ep field "
        "is never populated, so a rep_key that dropped it would pass"
    )


@pytest.mark.skipif(using_override(GOLDEN_ENV), reason="corpus claims apply to golden/keys.jsonl only")
def test_corpus_contains_positions_that_share_a_key(golden):
    """`nn_key` has fewer distinct values than the corpus has rows, and that is
    correct: the corpus is deduplicated on the whole FEN, so it holds positions
    that differ only in the halfmove clock or the move number — neither of which
    is in either key.

    This matters for the partition test. If every key were unique, the partition
    would be the identity on both sides and would compare equal no matter what
    the implementation did with the fields that are shared.
    """
    distinct_nn = len(set(golden["nn"].tolist()))
    distinct_rep = len(set(golden["rep"].tolist()))
    assert distinct_nn < len(golden["fens"]), (
        "every nn_key in the corpus is unique, so the partition test is vacuous"
    )
    assert distinct_rep < distinct_nn, (
        f"rep_key ({distinct_rep} classes) should be coarser than nn_key ({distinct_nn}): it "
        "ignores en-passant squares that cannot be captured, and nn_key does not"
    )


# ---------------------------------------------------------------------------
# Rejected input
#
# Both entry points must refuse what the reference refuses. Returning a key for
# a FEN python-chess would not parse means returning a key for a position that
# has no reference answer.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen",
    [
        "",
        "not a fen",
        # nine files on the first rank
        "rnbqkbnrr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        # seven ranks
        "rnbqkbnr/pppppppp/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        # two subsequent digits
        "44/8/8/8/8/8/8/8 w - - 0 1",
        # side to move is neither w nor b
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",
        # unparseable en-passant field
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e9 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq ee 0 1",
        # a seventh field
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 1",
    ],
)
def test_bad_fens_raise(fen):
    with pytest.raises(ValueError):
        guofish_core.nn_key(fen)
    with pytest.raises(ValueError):
        guofish_core.rep_key(fen)
    with pytest.raises(ValueError):
        guofish_core.keys(fen)
    with pytest.raises(ValueError):
        guofish_core.nn_keys([fen])


def test_batch_rejects_non_strings():
    with pytest.raises(TypeError):
        guofish_core.nn_keys([1, 2, 3])


def test_empty_batch_is_empty_not_an_error():
    assert guofish_core.nn_keys([]).shape == (0,)
    assert guofish_core.key_pairs([]).shape == (0, 2)
