"""C4 acceptance tests — the struct-of-arrays node arena.

There is no golden file for this chunk, and there could not be: what C4
delivers is a *shape*, not an answer. So the suite is not a parity run. It is a
set of assertions about properties that are expensive or impossible to recover
later, organised by the way each one fails.

What is actually being tested
-----------------------------
* **Alignment, in a Release build.** `assert()` is compiled out with NDEBUG, so
  the constructor's alignment checks are live in exactly the builds that do not
  matter. `array_info()` reports the addresses the arrays actually landed at,
  and the tests below check them in whatever build is running. The failure this
  guards is silent and enormous — a `lock xadd` straddling a cache line
  escalates to a bus lock, roughly 100x the aligned cost — and it would surface
  as a profile nobody can explain rather than as a wrong answer.

* **Lock-freedom**, read out of the compiler rather than assumed. The build
  already fails if `is_always_lock_free` is false anywhere (static_asserts in
  `cpp/arena.hpp`); `arena_layout()` puts the same facts where a person is
  looking.

* **The terminal invariant.** The Python engine could emit `bestmove 0000`: a
  node reporting itself expanded while having no children, leaving the move
  selector nothing to pick. Here that is not a bug that was fixed but a state
  that cannot be spelled, and the tests try to spell it several different ways.

* **Q32 round-trip, exhaustively.** Every IEEE-754 float bit pattern in [-1, 1],
  both signs — 2,130,706,434 of them — through `to_q32` and back. The criterion
  is the chunk brief's: exact to the stated 2.3e-10 resolution.

* **Both accumulators, in one process.** Every structural test is parametrized
  over `NodeArenaQ32` and `NodeArenaDouble`. Both are compiled into every build
  precisely so that neither can rot behind an `#ifdef` that nobody flips, and
  the two tests that distinguish them — associativity, and exact fixed-point
  accumulation — assert the difference rather than tolerating it.

* **Canonical move order.** The packed uint16 move field sorts by *square
  index*, which is rank-major, while C1's canonical order is the UCI string's,
  which is file-major. `canonical_move_key()` bridges them, and the test drives
  it against `golden/movegen.jsonl` — the same reference C1 was judged on. This
  is here because getting it wrong does not fail loudly: PUCT breaks ties by
  child order, so C++ and Python would simply explore different moves at equal
  priors.

Nothing here imports `chess`. The one golden file this suite reads is C1's, and
it is read, never written (Global Rules 1 and 2).

Environment overrides
---------------------
``GUOFISH_Q32_STRIDE``  stride for the exhaustive float sweep. Default 1, i.e.
exhaustive. Raising it makes the sweep a sample; the test says so in its
failure message so a strided pass is never mistaken for the real thing.
"""

import json
import math
import os
import struct
from pathlib import Path

import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_MOVEGEN = REPO_ROOT / "golden" / "movegen.jsonl"

# Positions read from the C1 corpus for the canonical-ordering test. The
# property is per-position, so this is about coverage of move shapes
# (promotions, castling, both colours) rather than about corpus size.
ORDERING_POSITIONS = 4000

BUILD = guofish_core.build_info()

ARENA_CLASSES = {
    "q32": guofish_core.NodeArenaQ32,
    "double": guofish_core.NodeArenaDouble,
}


@pytest.fixture(params=sorted(ARENA_CLASSES), ids=sorted(ARENA_CLASSES))
def arena_class(request):
    """Every structural test runs against both accumulators."""
    return ARENA_CLASSES[request.param]


@pytest.fixture
def arena(arena_class):
    return arena_class(1024)


# ---------------------------------------------------------------------------
# Layout and lock-freedom
# ---------------------------------------------------------------------------


def test_every_atomic_is_always_lock_free():
    """The chunk's compile-time criterion, reported where a person can see it.

    A `std::atomic<T>` that is not lock-free silently takes a mutex. That would
    put a lock back into the node — the exact thing this layout removes from
    Python, where a per-node `threading.Lock` was 56 B/node and 20% of
    per-child construction cost.
    """
    layout = guofish_core.arena_layout()
    for field in (
        "int32_always_lock_free",
        "int64_always_lock_free",
        "uint8_always_lock_free",
        "uint32_always_lock_free",
        "double_always_lock_free",
    ):
        assert layout[field] is True, f"{field} is False — see static_asserts in cpp/arena.hpp"


def test_atomics_are_no_wider_than_their_payload():
    """The cache-line budget in the SoA argument assumes this.

    `sizeof(atomic<T>) == sizeof(T)` is not guaranteed by the standard. If it
    were ever false the arrays would still be *correct* — the element type is
    what gets indexed — but a 32-sibling scan would touch more lines than the
    design claims, which is the entire justification for SoA.
    """
    layout = guofish_core.arena_layout()
    assert layout["sizeof_atomic_int32"] == 4
    assert layout["sizeof_atomic_int64"] == 8
    assert layout["sizeof_atomic_double"] == 8
    assert layout["sizeof_atomic_uint8"] == 1
    assert layout["alignof_atomic_int64"] == 8
    assert layout["alignof_atomic_double"] == 8


def test_per_node_width_matches_the_scope_figure():
    """33 bytes of payload per node, which is the number scope 2.3 quotes.

    The point of the figure is the comparison it enables: array-of-structs pads
    those same 33 bytes to 40 under the 8-byte accumulator's alignment, so an
    AoS scan strides 40 B and touches ~20 cache lines for 32 siblings. SoA pays
    no padding at all because there is no struct.
    """
    layout = guofish_core.arena_layout()
    assert layout["bytes_per_node_q32"] == 33
    assert layout["bytes_per_node_double"] == 33
    assert guofish_core.NodeArenaQ32.bytes_per_node == 33
    assert guofish_core.NodeArenaDouble.bytes_per_node == 33


def test_default_accumulator_is_q32_and_names_a_real_class():
    """`NodeArena` is an alias, and both real classes exist regardless.

    The `#ifdef` selects which one the engine uses; it must not be able to make
    the other one disappear, because a type that is not compiled is a type
    whose tests do not run.
    """
    assert guofish_core.DEFAULT_ACCUMULATOR in ("q32", "double")
    assert guofish_core.arena_layout()["default_accumulator"] == guofish_core.DEFAULT_ACCUMULATOR
    assert guofish_core.NodeArena is ARENA_CLASSES[guofish_core.DEFAULT_ACCUMULATOR]

    assert guofish_core.NodeArenaQ32.accumulator == "q32"
    assert guofish_core.NodeArenaDouble.accumulator == "double"


def test_the_atomics_in_this_arena_are_lock_free(arena):
    """The per-object claim, composed from the two things that establish it.

    There is no `is_lock_free()` accessor to call, and that is deliberate: on
    libstdc++ it is an out-of-line call into libatomic, which is not linked by
    default and made the Clang build fail to import — for an answer the standard
    already guarantees. `is_always_lock_free` being true means *every* object of
    the type is lock-free, and the alignment entries below show these particular
    objects are where the hardware needs them. Both halves, asserted together.
    """
    layout = guofish_core.arena_layout()
    assert layout["int32_always_lock_free"] is True
    assert layout["int64_always_lock_free"] is True
    assert layout["double_always_lock_free"] is True

    by_field = {entry["field"]: entry for entry in arena.array_info()}
    for field in ("visit_count", "value_sum", "vloss_count", "state"):
        assert by_field[field]["naturally_aligned"] is True


# ---------------------------------------------------------------------------
# Alignment — the runtime half, live in Release
# ---------------------------------------------------------------------------


EXPECTED_ARRAYS = [
    ("visit_count", 4),
    ("value_sum", 8),
    ("vloss_count", 4),
    ("prior", 4),
    ("move", 2),
    ("children_offset", 4),
    ("children_count", 2),
    ("state", 1),
    ("terminal_value", 4),
]


def test_arena_reports_one_array_per_field(arena):
    """Nine arrays, nine fields, no struct anywhere."""
    info = arena.array_info()
    assert [(e["field"], e["element_size"]) for e in info] == EXPECTED_ARRAYS


@pytest.mark.parametrize("capacity", [1, 2, 3, 7, 15, 33, 64, 1000, 65536])
def test_every_array_base_is_naturally_aligned(arena_class, capacity):
    """The runtime assertion the scope asks for, at a spread of sizes.

    Odd and prime capacities are in the list on purpose: an allocator bug that
    strides arrays out of one block would show up on a size that is not a
    multiple of the widest element, and not on a round one.
    """
    a = arena_class(capacity)
    for entry in a.array_info():
        assert entry["naturally_aligned"], (
            f"{entry['field']} ({entry['element_type']}) at 0x{entry['address']:x} "
            f"is not aligned to {entry['element_align']} — a lock xadd here can straddle "
            f"a cache line and escalate to a bus lock"
        )
        assert entry["address"] % entry["element_align"] == 0


@pytest.mark.parametrize("capacity", [1, 33, 1000])
def test_every_array_base_is_cache_line_aligned(arena_class, capacity):
    """Stronger than required, and the reason a sibling scan reads no extra line.

    Natural alignment is what correctness needs. Starting each array on a line
    boundary is what makes 32 siblings cost the minimum number of lines instead
    of one extra at each end.
    """
    a = arena_class(capacity)
    for entry in a.array_info():
        assert entry["requested_align"] >= guofish_core.CACHE_LINE
        assert entry["cache_line_aligned"], (
            f"{entry['field']} at 0x{entry['address']:x} is not {guofish_core.CACHE_LINE}-byte "
            f"aligned"
        )


def test_alignment_holds_across_repeated_allocation(arena_class):
    """One arena landing on a boundary by luck proves nothing.

    C0 found exactly this: a 16-byte-aligned allocation passed a 64-byte
    assertion once, by accident. Sizes are varied so consecutive allocations do
    not all come from the same size class.
    """
    for i in range(64):
        a = arena_class(1 + i * 7)
        for entry in a.array_info():
            assert entry["naturally_aligned"]
            assert entry["cache_line_aligned"]
            # The two booleans above are answered against absolute constants,
            # but `requested_align` is what was actually asked of the allocator
            # — check it too, or a build that requested nothing would pass by
            # answering its own question. The mutation drill caught exactly that.
            assert entry["requested_align"] >= guofish_core.CACHE_LINE
            assert entry["address"] % guofish_core.CACHE_LINE == 0


def test_arrays_do_not_overlap(arena):
    """Each field is a separate allocation, which is what SoA means here.

    If two arrays shared a block, writing `prior` could corrupt `value_sum` —
    and the tests that write one field and read another would still pass, since
    they would be reading the value that was just written.
    """
    info = arena.array_info()
    spans = []
    for entry in info:
        start = entry["address"]
        end = start + entry["element_size"] * arena.capacity
        spans.append((start, end, entry["field"]))

    spans.sort()
    for (a_start, a_end, a_field), (b_start, b_end, b_field) in zip(spans, spans[1:]):
        assert a_end <= b_start, f"{a_field} [{a_start:x},{a_end:x}) overlaps {b_field}"


# ---------------------------------------------------------------------------
# Q32 fixed point
# ---------------------------------------------------------------------------


def test_q32_constants():
    """The scale is 2^32, and that is what makes the conversions exact.

    Not an arbitrary constant: a power-of-two scale means `q * 2^-32` only
    changes a double's exponent, so it is exact for every code in range, and the
    resolution is the 2.3e-10 the scope quotes against bf16's 8 mantissa bits.
    """
    assert guofish_core.Q32_SCALE == 2.0**32
    assert guofish_core.Q32_ONE == 2**32
    assert guofish_core.Q32_RESOLUTION == 2.0**-32
    assert guofish_core.Q32_RESOLUTION == pytest.approx(2.3283064365386963e-10, rel=0, abs=0)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, 0),
        (1.0, 2**32),
        (-1.0, -(2**32)),
        (0.5, 2**31),
        (-0.5, -(2**31)),
        (0.25, 2**30),
        (2.0**-32, 1),
        (-(2.0**-32), -1),
    ],
)
def test_q32_known_values(value, expected):
    assert guofish_core.q32_from_float(value) == expected
    assert guofish_core.q32_to_float(expected) == value


def test_q32_rounds_half_away_from_zero_and_symmetrically():
    """The tie rule is reachable, so it has to be pinned.

    A float in [2^-10, 2^-9) has an ulp of 2^-33, so `v * 2^32` lands exactly on
    k + 0.5 for half of them. `std::llround` rounds away from zero, which is
    symmetric about zero — an asymmetric quantizer would bias every backup in
    one direction by half a tick, which over 15k sims is not noise.
    """
    half_tick = 2.0**-33
    assert guofish_core.q32_from_float(half_tick) == 1
    assert guofish_core.q32_from_float(-half_tick) == -1
    assert guofish_core.q32_from_float(3 * half_tick) == 2
    assert guofish_core.q32_from_float(-3 * half_tick) == -2


def test_q32_overflow_headroom_is_two_billion_visits():
    """Where the accumulator actually runs out, stated as a number.

    The engine budgets 15k sims a move and 2-3M nodes; this is seven orders
    away. Worth asserting because it is the one property that would make Q32 the
    wrong choice, and it is a property of the scale, not of the code.
    """
    max_visits_at_full_magnitude = (2**63 - 1) // guofish_core.Q32_ONE
    assert max_visits_at_full_magnitude > 2_000_000_000


def test_q32_roundtrip_is_exhaustively_exact_to_the_stated_resolution():
    """Acceptance criterion 6, over every float in the range.

    2,130,706,434 IEEE-754 float bit patterns — every one whose value lies in
    [-1, 1], denormals included, both signs. Four separate claims:

      * the worst absolute round-trip error is 2^-33, exactly half the Q32
        resolution, which is what "exact to 2.3e-10" means for a
        round-to-nearest quantizer;
      * every float at or above 2^-9 comes back *bit-identical*, because at that
        magnitude a float's ulp is already 2^-32 or coarser and Q32 can hold it
        exactly. This is the strong claim, and the boundary is asserted rather
        than the count;
      * Q32 -> double -> Q32 is exact for every code the sweep produced;
      * the quantizer is symmetric about zero.
    """
    stride = int(os.environ.get("GUOFISH_Q32_STRIDE", "1"))
    result = guofish_core.q32_roundtrip_sweep(stride)

    assert result["exhaustive"] is (stride == 1), (
        f"GUOFISH_Q32_STRIDE={stride} makes this a sample of {result['floats_examined']:,} "
        f"floats, not the exhaustive sweep of 2,130,706,434"
    )
    if stride == 1:
        assert result["floats_examined"] == 2_130_706_434

    assert result["max_abs_error"] <= result["half_resolution"]
    assert result["max_abs_error"] == 2.0**-33
    assert result["code_mismatches"] == 0
    assert result["asymmetric"] == 0

    # Every float below 2^-9 has an ulp finer than the Q32 tick, so some of them
    # must be inexact; every float at or above it must be exact.
    assert result["largest_inexact"] < 2.0**-9


def as_float32(x):
    """Round a Python double to the nearest float32, and back to a double.

    The exactness claim below is about *float32* values, because that is what
    the network emits and what `prior` and `terminal_value` are stored as. A
    Python float is a double, and 0.1-as-a-double is not a multiple of 2^-32 —
    Q32 rounds it, correctly, to within half a tick.
    """
    return struct.unpack("<f", struct.pack("<f", x))[0]


def test_floats_at_or_above_the_quantization_boundary_are_bit_exact():
    """The boundary the sweep reports, checked directly on both sides of it.

    2^-9 is where a float32's ulp becomes 2^-32, i.e. exactly the Q32 tick.
    At or above it the conversion loses nothing at all; just below it, it must
    lose something or the sweep is not measuring what it claims to.
    """
    boundary = 2.0**-9
    boundary_bits = struct.unpack("<I", struct.pack("<f", boundary))[0]
    below = struct.unpack("<f", struct.pack("<I", boundary_bits - 1))[0]
    assert below < boundary

    for value in (boundary, 0.5, 1.0, -1.0, as_float32(0.1), as_float32(-0.7), as_float32(0.9999)):
        assert guofish_core.q32_to_float(guofish_core.q32_from_float(value)) == value, (
            f"{value!r} is a float32 at or above 2^-9 and must survive the round trip exactly"
        )

    assert guofish_core.q32_to_float(guofish_core.q32_from_float(below)) != below

    # A double that is not a float32 is quantized, and must land within half a
    # tick — which is the sweep's criterion, restated on a value a person would
    # actually type.
    assert guofish_core.q32_to_float(guofish_core.q32_from_float(0.1)) == pytest.approx(
        0.1, abs=2.0**-33
    )


def test_q32_code_roundtrip_over_the_code_range():
    """The other direction: Q32 -> double -> Q32.

    Cannot be exhaustive — the range is 8.6e9 codes — so this is a strided
    sweep plus every boundary code. The general case is a consequence rather
    than a measurement: `from_q32` multiplies by a power of two, which is exact
    in double for |q| <= 2^53, so `to_q32` is handed the integer back unchanged.
    The sweep is here to catch that premise breaking, not to establish it.
    """
    result = guofish_core.q32_code_sweep(4099)
    assert result["mismatches"] == 0, f"first mismatching code: {result['first_mismatch']}"
    assert result["codes_examined"] > 2_000_000


# ---------------------------------------------------------------------------
# Packed moves and canonical order
# ---------------------------------------------------------------------------


def test_move_packing_roundtrips_over_every_possible_move():
    """All 64 x 64 x 5 packings, which is the whole space the field can hold."""
    seen = set()
    for src in range(64):
        for dst in range(64):
            for promo in range(5):
                packed = guofish_core.pack_move(src, dst, promo)
                assert 0 <= packed <= 0xFFFF
                assert guofish_core.move_from(packed) == src
                assert guofish_core.move_to(packed) == dst
                assert guofish_core.move_promotion(packed) == promo
                seen.add(packed)
    assert len(seen) == 64 * 64 * 5


def test_move_packing_rejects_out_of_range_input():
    for bad in [(-1, 0, 0), (64, 0, 0), (0, -1, 0), (0, 64, 0), (0, 0, -1), (0, 0, 5)]:
        with pytest.raises(ValueError):
            guofish_core.pack_move(*bad)


def test_promotion_codes_are_alphabetical_not_by_piece_value():
    """b < n < q < r, and none is lowest.

    This looks wrong to a chess player and is exactly right for the sort: the
    canonical order is the UCI string's byte order, so the promotion letters
    compare alphabetically, and a four-character move sorts before any
    promotion sharing its from/to because the missing letter is "less than"
    every letter.
    """
    assert guofish_core.PROMO_NONE == 0
    assert guofish_core.PROMO_BISHOP == 1
    assert guofish_core.PROMO_KNIGHT == 2
    assert guofish_core.PROMO_QUEEN == 3
    assert guofish_core.PROMO_ROOK == 4

    letters = ["", "b", "n", "q", "r"]
    assert letters == sorted(letters)


def uci_to_packed(uci):
    """Parse a UCI move without importing python-chess.

    Squares are numbered as chess-library numbers them, rank * 8 + file, which
    is what `pack_move` documents.
    """
    promo_codes = {"": 0, "b": 1, "n": 2, "q": 3, "r": 4}
    src = (ord(uci[1]) - ord("1")) * 8 + (ord(uci[0]) - ord("a"))
    dst = (ord(uci[3]) - ord("1")) * 8 + (ord(uci[2]) - ord("a"))
    return guofish_core.pack_move(src, dst, promo_codes[uci[4:]])


def test_move_to_uci_inverts_the_parser():
    for uci in ["e2e4", "a1a2", "h7h8q", "b7a8n", "e1g1", "a7a8b", "h2h1r", "d4d5"]:
        assert guofish_core.move_to_uci(uci_to_packed(uci)) == uci


def golden_positions(limit):
    if not GOLDEN_MOVEGEN.exists():
        pytest.skip(f"golden corpus missing: {GOLDEN_MOVEGEN}")
    with GOLDEN_MOVEGEN.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if lineno > limit:
                return
            record = json.loads(line)
            yield lineno, record["fen"], record["moves"]


def test_canonical_move_key_reproduces_c1_ordering_on_the_golden_corpus():
    """The packed move's sort key must agree with C1's canonical order.

    `golden/movegen.jsonl` is C1's reference and its move lists are in canonical
    (from, to, promotion) order — byte-wise UCI order. Packing each move and
    sorting by `canonical_move_key` has to reproduce that list exactly.

    Read-only: Global Rules 1 and 2. Nothing here writes to `golden/`.
    """
    checked = 0
    for lineno, fen, moves in golden_positions(ORDERING_POSITIONS):
        assert moves == sorted(moves), f"line {lineno}: golden list is not sorted"
        packed = [uci_to_packed(m) for m in moves]
        by_key = sorted(packed, key=guofish_core.canonical_move_key)
        assert [guofish_core.move_to_uci(p) for p in by_key] == moves, (
            f"line {lineno} ({fen}): canonical_move_key does not reproduce C1's order"
        )
        checked += 1
    assert checked > 100


def test_sorting_packed_moves_directly_is_the_wrong_order():
    """The whole reason `canonical_move_key` exists, demonstrated on real data.

    Square indices are rank-major (a1, b1, ... h1, a2) and UCI string order is
    file-major (a1, a2, ... a8, b1). If this test ever stops finding a
    disagreement, the two orders have converged and the helper is pointless —
    which is not something that can happen, so the failure would mean the test
    is no longer reading real move lists.

    The disagreement matters because it does not announce itself: PUCT resolves
    ties by child order, so the two languages would explore different moves at
    equal priors and simply diverge.
    """
    disagreements = 0
    positions = 0
    for _lineno, _fen, moves in golden_positions(ORDERING_POSITIONS):
        positions += 1
        packed = [uci_to_packed(m) for m in moves]
        if sorted(packed) != sorted(packed, key=guofish_core.canonical_move_key):
            disagreements += 1

    assert positions > 100
    assert disagreements > positions // 2, (
        f"only {disagreements}/{positions} positions distinguish raw-packed order from "
        f"canonical order — the corpus is not exercising the difference"
    )


def test_no_move_sentinel_is_zero():
    """The root's move slot, and a1a1 is not a legal move.

    There is no spare bit pattern for "no move" — every one of the 65,536 is a
    well-formed (from, to, promo) triple — so the sentinel is a legal *pattern*
    that is not a legal *move*. Stated here so nobody later reads a root's move
    field as a real move.
    """
    assert guofish_core.NO_MOVE == 0
    assert guofish_core.move_from(0) == 0
    assert guofish_core.move_to(0) == 0
    assert guofish_core.move_to_uci(0) == "a1a1"


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def test_allocation_is_contiguous_and_monotonic(arena):
    """Children are a contiguous index range, which is why the scan is sequential."""
    assert arena.size == 0
    first = arena.allocate(1)
    assert first == 0
    assert arena.size == 1

    block = arena.allocate(32)
    assert block == 1
    assert arena.size == 33

    later = arena.allocate(5)
    assert later == 33
    assert arena.size == 38

    # No block overlaps another.
    ranges = [(first, 1), (block, 32), (later, 5)]
    covered = set()
    for start, count in ranges:
        indices = set(range(start, start + count))
        assert not (covered & indices)
        covered |= indices
    assert len(covered) == 38


def test_allocation_refuses_to_exceed_capacity(arena_class):
    a = arena_class(10)
    assert a.allocate(10) == 0
    assert a.size == 10
    with pytest.raises(RuntimeError, match="exhausted"):
        a.allocate(1)
    # A failed allocation must not consume anything.
    assert a.size == 10


def test_try_allocate_reports_exhaustion_without_raising(arena_class):
    """The form the search uses; a full arena is a condition, not a bug."""
    a = arena_class(4)
    assert a.try_allocate(3) == 0
    assert a.try_allocate(3) is None
    assert a.size == 3
    assert a.try_allocate(1) == 3
    assert a.size == 4


def test_zero_capacity_is_rejected(arena_class):
    with pytest.raises(ValueError):
        arena_class(0)


def test_reset_recycles_the_arena_and_hands_back_clean_nodes(arena):
    """C8's ping-pong reuse depends on this, and it is the quiet failure.

    A recycled node still carrying the previous search's visit count and value
    sum does not crash. It biases selection toward a node that was never visited
    in this search, and the effect is invisible in any single position.
    """
    idx = arena.allocate(4)
    for i in range(idx, idx + 4):
        arena.add_visits(i, 17)
        arena.add_value(i, 0.75)
        arena.add_vloss(i, 3)
        arena.set_prior(i, 0.5)
        arena.set_move(i, uci_to_packed("e2e4"))
    arena.mark_terminal(idx, -1.0)
    arena.set_children(idx + 1, idx + 2, 2)

    arena.reset()
    assert arena.size == 0

    again = arena.allocate(4)
    assert again == 0
    for i in range(again, again + 4):
        assert arena.visit_count(i) == 0
        assert arena.value_sum(i) == 0.0
        assert arena.value_sum_raw(i) == 0
        assert arena.vloss_count(i) == 0
        assert arena.prior(i) == 0.0
        assert arena.move(i) == guofish_core.NO_MOVE
        assert arena.children_offset(i) == 0
        assert arena.children_count(i) == 0
        assert arena.terminal_value(i) == 0.0
        assert arena.state(i) == guofish_core.STATE_UNEXPANDED
        assert arena.is_terminal(i) is False


def test_indices_are_bounded_by_size_not_capacity(arena):
    """An unallocated slot is not a node, even though the memory exists.

    Reading one would be reading whatever the last search left there. Bounding
    on `size` rather than `capacity` is what makes `reset()` safe to be O(1).
    """
    arena.allocate(3)
    for accessor in (arena.visit_count, arena.value_sum, arena.prior, arena.state):
        with pytest.raises(IndexError):
            accessor(3)
    with pytest.raises(IndexError):
        arena.set_prior(3, 1.0)
    with pytest.raises(IndexError):
        arena.add_visits(99, 1)


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


def test_every_field_round_trips(arena):
    """Set and read back each of the nine fields at a specific index.

    Values are chosen so no two fields could be confused for one another — a
    bug that indexed the wrong array would otherwise read back a plausible
    number.
    """
    base = arena.allocate(8)
    i = base + 3

    arena.add_visits(i, 12345)
    arena.add_vloss(i, 7)
    arena.set_prior(i, 0.375)
    arena.set_move(i, uci_to_packed("g7g8q"))

    assert arena.visit_count(i) == 12345
    assert arena.vloss_count(i) == 7
    assert arena.prior(i) == 0.375
    assert guofish_core.move_to_uci(arena.move(i)) == "g7g8q"

    # The neighbours are untouched: an off-by-one in the stride would show here.
    for other in range(base, base + 8):
        if other == i:
            continue
        assert arena.visit_count(other) == 0
        assert arena.vloss_count(other) == 0
        assert arena.prior(other) == 0.0
        assert arena.move(other) == guofish_core.NO_MOVE


def test_prior_is_a_float_not_a_double(arena):
    """The field is `float`, and the truncation is deliberate.

    Priors come from a softmax over a bf16 network output; storing them at
    double width would double that array's contribution to the sibling scan's
    working set to buy precision the source does not have.
    """
    i = arena.allocate(1)
    arena.set_prior(i, 0.1)
    assert arena.prior(i) == pytest.approx(0.1, rel=1e-7)
    assert arena.prior(i) != 0.1  # exactly the float/double gap


def test_visit_and_vloss_counts_accumulate_and_subtract(arena):
    """Virtual loss is repaid by adding a negative delta, not by a separate call."""
    i = arena.allocate(1)
    for _ in range(10):
        arena.add_visits(i, 1)
    assert arena.visit_count(i) == 10

    arena.add_vloss(i, 1)
    arena.add_vloss(i, 1)
    assert arena.vloss_count(i) == 2
    arena.add_vloss(i, -2)
    assert arena.vloss_count(i) == 0


def test_value_sum_accumulates(arena):
    i = arena.allocate(1)
    arena.add_value(i, 0.5)
    arena.add_value(i, 0.25)
    arena.add_value(i, -1.0)
    assert arena.value_sum(i) == pytest.approx(-0.25, abs=1e-9)


def test_negative_and_extreme_values_are_stored(arena):
    """+-1.0 is a real evaluation — a forced win or loss — not an edge case."""
    base = arena.allocate(2)
    arena.add_value(base, 1.0)
    arena.add_value(base + 1, -1.0)
    assert arena.value_sum(base) == 1.0
    assert arena.value_sum(base + 1) == -1.0


# ---------------------------------------------------------------------------
# The two accumulators, and what actually differs between them
# ---------------------------------------------------------------------------


def test_q32_stores_the_fixed_point_integer():
    a = guofish_core.NodeArenaQ32(4)
    i = a.allocate(1)
    a.add_value(i, 0.5)
    assert a.value_sum_raw(i) == 2**31
    a.add_value(i, 0.5)
    assert a.value_sum_raw(i) == 2**32
    assert a.value_sum(i) == 1.0


def test_double_stores_the_double():
    a = guofish_core.NodeArenaDouble(4)
    i = a.allocate(1)
    a.add_value(i, 0.5)
    assert a.value_sum_raw(i) == 0.5
    assert isinstance(a.value_sum_raw(i), float)


def test_q32_accumulation_is_order_independent_and_double_is_not():
    """The reason Q32 is the production accumulator, as a two-line difference.

    Integer addition is associative, so a Q32 total does not depend on the
    order the threads happened to arrive in and the multithreaded engine is
    bit-reproducible — a property the Python engine never had and which the
    original brief wrote off as unreachable. Double addition is not associative,
    which is why the same three values in two orders give two different answers
    below.

    The magnitudes are chosen to make the point at a sum size a real search
    reaches: at 2^21 visits of magnitude 1, a double's ulp has grown past the
    Q32 tick, so one more tick-sized backup is simply lost.
    """
    big = float(2**21)
    tick = guofish_core.Q32_RESOLUTION

    def run(cls, order):
        a = cls(4)
        i = a.allocate(1)
        for value in order:
            if abs(value) > 1.0:
                a.add_value_raw(i, value if cls is guofish_core.NodeArenaDouble
                                else int(value * guofish_core.Q32_ONE))
            else:
                a.add_value(i, value)
        return a.value_sum_raw(i)

    forward = [big, tick, -big]
    swapped = [big, -big, tick]

    q32_forward = run(guofish_core.NodeArenaQ32, forward)
    q32_swapped = run(guofish_core.NodeArenaQ32, swapped)
    assert q32_forward == q32_swapped == 1

    dbl_forward = run(guofish_core.NodeArenaDouble, forward)
    dbl_swapped = run(guofish_core.NodeArenaDouble, swapped)
    assert dbl_forward != dbl_swapped
    assert dbl_forward == 0.0
    assert dbl_swapped == tick


def test_both_accumulators_agree_within_the_q32_resolution(arena_class):
    """They must not disagree by more than the quantization, over a real workload.

    Gate 2b is the swap from the double accumulator to Q32; if the two ever
    drifted by more than a tick per backup, that swap would change the engine's
    play and the equivalence build would stop being evidence about production.
    """
    values = [(-1.0) ** k * (k % 977) / 977.0 for k in range(2000)]

    a = arena_class(4)
    i = a.allocate(1)
    for v in values:
        a.add_value(i, v)

    exact = math.fsum(values)
    assert a.value_sum(i) == pytest.approx(exact, abs=len(values) * guofish_core.Q32_RESOLUTION)


# ---------------------------------------------------------------------------
# Child ranges
# ---------------------------------------------------------------------------


def test_child_slices_are_correct_and_disjoint(arena):
    """The structural criterion in the brief, over a two-level tree.

    Children are addressed as (offset, count) — one uint32 and one uint16 per
    node rather than a pointer per child — so a wrong offset does not crash,
    it silently reads another node's siblings.
    """
    root = arena.allocate(1)
    kids = arena.allocate(4)
    arena.set_children(root, kids, 4)

    grandchildren = {}
    for k in range(4):
        child = arena.child(root, k)
        block = arena.allocate(3)
        arena.set_children(child, block, 3)
        grandchildren[child] = list(range(block, block + 3))

    assert arena.children(root) == (kids, 4)
    assert arena.children_indices(root) == [kids + k for k in range(4)]

    seen = set()
    for child, expected in grandchildren.items():
        assert arena.children_indices(child) == expected
        assert arena.children_count(child) == 3
        assert arena.children_offset(child) == expected[0]
        assert not (seen & set(expected)), "two nodes share a child slot"
        seen |= set(expected)

    assert len(seen) == 12
    assert seen.isdisjoint(arena.children_indices(root))


def test_child_index_is_bounded_by_children_count(arena):
    root = arena.allocate(1)
    kids = arena.allocate(3)
    arena.set_children(root, kids, 3)

    assert [arena.child(root, k) for k in range(3)] == [kids, kids + 1, kids + 2]
    with pytest.raises(IndexError):
        arena.child(root, 3)


def test_unexpanded_node_has_no_children(arena):
    i = arena.allocate(1)
    assert arena.children_count(i) == 0
    assert arena.children_indices(i) == []
    with pytest.raises(IndexError):
        arena.child(i, 0)


def test_child_range_outside_the_allocated_region_is_refused(arena):
    """The C8 pointer-fixup bug class, caught at the write rather than the read.

    Compacting-copy remaps every child index. An off-by-one there produces a
    range that points past the end of the live region, and the nodes it names
    are last search's, not this one's.
    """
    root = arena.allocate(1)
    kids = arena.allocate(4)

    with pytest.raises(IndexError):
        arena.set_children(root, kids, 5)  # one past the allocated region
    with pytest.raises(IndexError):
        arena.set_children(root, arena.size, 1)

    # ...and the node is unchanged by the refusal.
    assert arena.children_count(root) == 0
    assert arena.is_expanded(root) is False


# ---------------------------------------------------------------------------
# State — including the invariant this chunk exists to make unrepresentable
# ---------------------------------------------------------------------------


def test_fresh_node_is_unexpanded_and_not_terminal(arena):
    i = arena.allocate(1)
    assert arena.state(i) == guofish_core.STATE_UNEXPANDED
    assert arena.lifecycle(i) == guofish_core.STATE_UNEXPANDED
    assert arena.is_terminal(i) is False
    assert arena.is_expanded(i) is False


def test_pending_is_claimed_by_exactly_one_caller(arena):
    """The CAS that resolves two threads selecting the same leaf.

    The loser unwinds its virtual loss and retries (scope 2.2). Single-threaded
    here, but the transition is the same one: it must be a compare-exchange and
    not a load-then-store, or both threads would expand the same node and one
    child block would be orphaned with its visits still in the tree.
    """
    i = arena.allocate(1)
    assert arena.try_claim_pending(i) is True
    assert arena.lifecycle(i) == guofish_core.STATE_PENDING
    assert arena.try_claim_pending(i) is False

    arena.release_pending(i)
    assert arena.lifecycle(i) == guofish_core.STATE_UNEXPANDED
    assert arena.try_claim_pending(i) is True


def test_releasing_a_node_that_is_not_pending_is_an_error(arena):
    i = arena.allocate(1)
    with pytest.raises(RuntimeError, match="not pending"):
        arena.release_pending(i)


def test_expansion_publishes_children_and_marks_expanded(arena):
    root = arena.allocate(1)
    kids = arena.allocate(2)
    assert arena.try_claim_pending(root) is True
    arena.set_children(root, kids, 2)

    assert arena.is_expanded(root) is True
    assert arena.lifecycle(root) == guofish_core.STATE_EXPANDED
    assert arena.is_terminal(root) is False


def test_a_node_cannot_be_expanded_with_no_children(arena):
    """`bestmove 0000`, made unspellable.

    The Python engine could reach a node that reported itself expanded while
    having no children, so the move selector had nothing to choose and the
    engine emitted a null move. This is that state, and it is refused at the
    only door into it.
    """
    root = arena.allocate(1)
    with pytest.raises(ValueError, match="zero children"):
        arena.set_children(root, 0, 0)
    assert arena.is_expanded(root) is False
    assert arena.children_count(root) == 0


def test_a_terminal_node_cannot_be_expanded(arena):
    """Terminal-ness is a distinct bit, and it is one-way.

    A node where the game ends has no moves. If it could also be marked
    expanded, "expanded with zero children" would be reachable by a second
    route.
    """
    i = arena.allocate(1)
    kids = arena.allocate(2)
    arena.mark_terminal(i, -1.0)

    assert arena.is_terminal(i) is True
    assert arena.lifecycle(i) == guofish_core.STATE_UNEXPANDED

    with pytest.raises(RuntimeError, match="terminal"):
        arena.set_children(i, kids, 2)
    assert arena.children_count(i) == 0
    assert arena.is_expanded(i) is False


def test_an_expanded_node_cannot_become_terminal(arena):
    """The other direction of the same invariant."""
    root = arena.allocate(1)
    kids = arena.allocate(2)
    arena.set_children(root, kids, 2)

    with pytest.raises(RuntimeError, match="children"):
        arena.mark_terminal(root, 0.0)
    assert arena.is_terminal(root) is False
    assert arena.children_count(root) == 2


def test_a_node_cannot_be_expanded_twice(arena):
    """Double expansion orphans the first child block, visits and all.

    Nothing crashes: the tree simply loses a subtree's statistics into
    unreachable arena slots, and the parent's visit count no longer equals the
    sum of its children's.
    """
    root = arena.allocate(1)
    first = arena.allocate(2)
    second = arena.allocate(3)
    arena.set_children(root, first, 2)

    with pytest.raises(RuntimeError, match="already expanded"):
        arena.set_children(root, second, 3)
    assert arena.children(root) == (first, 2)


def test_terminal_is_a_distinct_bit_from_the_lifecycle(arena):
    """Both facts live in one byte and neither overwrites the other.

    Collapsing them into a single enum is how the defect above happened: a
    reader asking "is this expanded" and a reader asking "does the game end
    here" were consulting the same value.
    """
    i = arena.allocate(1)
    arena.mark_terminal(i, 1.0)

    raw = arena.state(i)
    assert raw & guofish_core.TERMINAL_BIT
    assert raw & guofish_core.STATE_LIFECYCLE_MASK == guofish_core.STATE_UNEXPANDED
    assert guofish_core.TERMINAL_BIT & guofish_core.STATE_LIFECYCLE_MASK == 0
    assert arena.terminal_value(i) == 1.0


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_terminal_value_records_the_game_result(arena, value):
    i = arena.allocate(1)
    arena.mark_terminal(i, value)
    assert arena.terminal_value(i) == value
    assert arena.is_terminal(i) is True


def test_a_pending_node_can_still_be_marked_terminal(arena):
    """A leaf is claimed before anyone knows whether the game ends there.

    The claim happens on selection; terminality is discovered when the position
    is examined. So PENDING -> terminal has to be legal, and the terminal bit
    must survive alongside the lifecycle value.
    """
    i = arena.allocate(1)
    assert arena.try_claim_pending(i) is True
    arena.mark_terminal(i, 0.0)
    assert arena.is_terminal(i) is True
    assert arena.lifecycle(i) == guofish_core.STATE_PENDING


# ---------------------------------------------------------------------------
# A small tree, end to end
# ---------------------------------------------------------------------------


def test_a_two_level_tree_is_structurally_consistent(arena_class):
    """Everything above, assembled once into the shape C5 will actually build.

    Not a search: no PUCT, no evaluation. It checks that a tree built through
    the public surface can be walked back down and every field found where it
    was put — which is the whole of what C4 owes C5.
    """
    arena = arena_class(4096)
    root = arena.allocate(1)
    arena.set_move(root, guofish_core.NO_MOVE)

    root_moves = ["b1c3", "e2e4", "g1f3"]
    kids = arena.allocate(len(root_moves))
    for k, uci in enumerate(root_moves):
        arena.set_move(kids + k, uci_to_packed(uci))
        arena.set_prior(kids + k, 1.0 / len(root_moves))
    arena.set_children(root, kids, len(root_moves))

    # One child is a terminal position; the others get grandchildren.
    terminal_child = arena.child(root, 1)
    arena.mark_terminal(terminal_child, -1.0)

    expanded = []
    for k in (0, 2):
        child = arena.child(root, k)
        block = arena.allocate(2)
        for j, uci in enumerate(["a7a6", "b7b6"]):
            arena.set_move(block + j, uci_to_packed(uci))
        arena.set_children(child, block, 2)
        expanded.append(child)

    # Walk it back down.
    assert arena.children_count(root) == 3
    assert [guofish_core.move_to_uci(arena.move(c)) for c in arena.children_indices(root)] == root_moves

    for child in arena.children_indices(root):
        if child == terminal_child:
            assert arena.is_terminal(child)
            assert arena.children_count(child) == 0
            assert arena.terminal_value(child) == -1.0
        else:
            assert child in expanded
            assert arena.is_expanded(child)
            assert arena.children_count(child) == 2
            assert not arena.is_terminal(child)

    # Every index that was handed out is distinct, and every child range lies
    # inside the allocated region.
    for i in range(arena.size):
        offset, count = arena.children(i)
        assert offset + count <= arena.size

    # No node is both terminal and expanded — the invariant, restated over the
    # whole tree rather than at one node.
    for i in range(arena.size):
        assert not (arena.is_terminal(i) and arena.is_expanded(i))
        if arena.is_expanded(i):
            assert arena.children_count(i) > 0


# ---------------------------------------------------------------------------
# Benchmark plumbing — the numbers themselves are BENCH.md's, not a criterion
# ---------------------------------------------------------------------------


def test_sibling_scan_bench_reads_every_node_it_claims_to():
    """The benchmark must not be timing an empty loop.

    A scan whose reads were optimised away would report a wonderful number and
    decide the accumulator question wrongly. The checksum is derived from the
    argmax of each block, so it is zero only if nothing was read.
    """
    result = guofish_core.sibling_scan_bench(blocks=64, block_size=32, repeats=4)
    assert result["nodes"] == 64 * 32
    assert result["scans"] == 64 * 4
    assert result["checksum"] > 0
    assert result["q32_ns_per_scan"] > 0
    assert result["double_ns_per_scan"] > 0
    assert result["hot_bytes_per_node_q32"] == result["hot_bytes_per_node_double"] == 16


def test_sibling_scan_bench_rejects_degenerate_shapes():
    for kwargs in ({"blocks": 0}, {"block_size": 0}, {"repeats": 0}):
        args = {"blocks": 4, "block_size": 32, "repeats": 1}
        args.update(kwargs)
        with pytest.raises(ValueError):
            guofish_core.sibling_scan_bench(**args)
