"""C0 acceptance tests — toolchain spike.

These are the acceptance criteria for C0 (Global Rule 1). They are deliberately
written so that they fail if the C++ side quietly does the easy thing instead of
the required thing:

  * a *copy* handed back to NumPy would still pass a naive shape/dtype check, so
    the zero-copy tests assert on ``base`` / ``OWNDATA`` *and* prove that a write
    made in Python is visible to a C++ read;
  * ``alignas(64)`` on a ``std::vector`` member aligns the vector object rather
    than its heap allocation, so alignment is asserted on the address NumPy
    actually exposes, across several shapes;
  * a ``roundtrip_bench`` that never entered Python would report excellent
    latency, so the callback counts its own invocations.

C0b amendment: ``roundtrip_bench`` now passes the live zero-copy view to the
callback, so every callback here takes one argument and touches the array. The
signature change is the only edit C0b made to this file; no assertion was
weakened. See DECISIONS.md, "C0b / Rule 1".
"""

import numpy as np
import pytest

import guofish_core

ALIGNMENT = 64

# Shapes cover a power-of-two batch, the real tokenizer width (68 int32 = 272
# bytes, deliberately not a multiple of 64), and a single row.
SHAPES = [(32, 68), (64, 68), (256, 68), (1, 1), (17, 3)]


def data_ptr(arr):
    """The base address NumPy would hand to a C consumer."""
    return arr.__array_interface__["data"][0]


# ---------------------------------------------------------------------------
# Import surface
# ---------------------------------------------------------------------------


def test_ping():
    assert guofish_core.ping() == "pong"


# ---------------------------------------------------------------------------
# make_buffer: shape, dtype, alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_make_buffer_shape_and_dtype(rows, cols):
    buf = guofish_core.make_buffer(rows, cols)

    assert isinstance(buf, np.ndarray)
    assert buf.shape == (rows, cols)
    assert buf.dtype == np.int32
    assert buf.flags.c_contiguous
    assert buf.flags.writeable, "the evaluator must be able to write into this view"


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_buffer_is_64_byte_aligned(rows, cols):
    buf = guofish_core.make_buffer(rows, cols)
    addr = data_ptr(buf)

    assert addr % ALIGNMENT == 0, (
        f"buffer at 0x{addr:x} is {addr % ALIGNMENT} bytes past a "
        f"{ALIGNMENT}-byte boundary"
    )


def test_alignment_holds_across_repeated_allocation():
    """One aligned allocation can be luck; a hundred in a row cannot."""
    for i in range(100):
        buf = guofish_core.make_buffer(1 + i, 68)
        assert data_ptr(buf) % ALIGNMENT == 0


# ---------------------------------------------------------------------------
# Zero copy
# ---------------------------------------------------------------------------


def test_buffer_is_a_view_not_a_copy():
    buf = guofish_core.make_buffer(32, 68)

    assert buf.base is not None, "no base object: NumPy owns this memory, so it is a copy"
    assert not buf.flags.owndata, "OWNDATA set: NumPy allocated this, so it is a copy"


def test_python_write_is_visible_to_cpp():
    buf = guofish_core.make_buffer(32, 68)
    assert guofish_core.buffer_checksum() == 0

    buf[0, 0] = 42
    buf[1, 5] = 100

    assert guofish_core.buffer_checksum() == 142


def test_checksum_tracks_successive_writes():
    rows, cols = 16, 68
    buf = guofish_core.make_buffer(rows, cols)

    buf[:] = 1
    assert guofish_core.buffer_checksum() == rows * cols

    buf[3, :] = 7
    assert guofish_core.buffer_checksum() == (rows - 1) * cols + 7 * cols

    buf[:] = 0
    assert guofish_core.buffer_checksum() == 0

    # Negative values must not be read back as unsigned.
    buf[0, 0] = -5
    assert guofish_core.buffer_checksum() == -5


def test_checksum_reads_the_whole_buffer():
    """Guards against a checksum that only walks the first row."""
    rows, cols = 8, 68
    buf = guofish_core.make_buffer(rows, cols)
    buf[:] = 0
    buf[rows - 1, cols - 1] = 1234

    assert guofish_core.buffer_checksum() == 1234


def test_make_buffer_rebinds_the_checksum_target():
    first = guofish_core.make_buffer(8, 68)
    first[:] = 1
    assert guofish_core.buffer_checksum() == 8 * 68

    guofish_core.make_buffer(4, 68)
    assert guofish_core.buffer_checksum() == 0, "checksum still reading the previous allocation"


def test_old_view_survives_reallocation():
    """The NumPy view must keep the C++ allocation alive on its own.

    If the module hands Python a raw pointer it also owns elsewhere, this is a
    use-after-free (and a double free at teardown). Under ASan it aborts here.
    """
    first = guofish_core.make_buffer(8, 8)
    first[:] = 3

    guofish_core.make_buffer(4, 4)  # rebinds the module-level owner

    assert first.sum() == 3 * 64, "read through a stale view returned garbage"
    first[0, 0] = 99
    assert first[0, 0] == 99


# ---------------------------------------------------------------------------
# roundtrip_bench
# ---------------------------------------------------------------------------

REQUIRED_BENCH_KEYS = {"median_us", "p99_us", "median_ms", "p99_ms", "mean_us", "iters", "rows"}


def test_roundtrip_bench_actually_calls_into_python():
    calls = 0

    def callback(arr):
        nonlocal calls
        calls += 1

    iters = 500
    guofish_core.roundtrip_bench(32, iters, callback)

    assert calls == iters, "callback was not invoked once per iteration"


def test_roundtrip_bench_passes_a_live_zero_copy_view():
    """C0b: the payload must be the real buffer, not a copy and not a stub."""
    rows = 32
    seen = []

    def callback(arr):
        if not seen:
            seen.append(arr)
        assert arr is seen[0], "a fresh object per call means this is not one stable view"

    guofish_core.roundtrip_bench(rows, 50, callback)

    arr = seen[0]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (rows, 68), "view must be sized to `rows` x the tokenizer width"
    assert arr.dtype == np.int32
    assert arr.base is not None, "no base object: NumPy owns this memory, so it is a copy"
    assert not arr.flags.owndata, "OWNDATA set: NumPy allocated this, so it is a copy"
    assert data_ptr(arr) % ALIGNMENT == 0

    # The C++ side writes row (i % rows) on iteration i, so after 50 iterations
    # of a 32-row buffer every row has been written and nothing is still zero.
    assert arr.sum() != 0, "callback was handed a buffer C++ never wrote to"


def test_roundtrip_bench_view_outlives_the_call():
    """A view captured by the callback must not dangle once the bench returns."""
    captured = []
    guofish_core.roundtrip_bench(8, 20, captured.append)

    arr = captured[0]
    assert arr.sum() == arr.sum(), "read through a stale view returned garbage"
    arr[0, 0] = 99
    assert arr[0, 0] == 99


def test_roundtrip_bench_reports_required_statistics():
    res = guofish_core.roundtrip_bench(32, 500, lambda arr: None)

    missing = REQUIRED_BENCH_KEYS - set(res)
    assert not missing, f"missing keys: {sorted(missing)}"

    assert res["median_us"] > 0.0
    assert res["p99_us"] >= res["median_us"]
    assert res["max_us"] >= res["p99_us"]
    assert res["min_us"] <= res["median_us"]
    assert res["p99_ms"] == pytest.approx(res["p99_us"] / 1000.0)


def test_roundtrip_bench_rejects_degenerate_arguments():
    with pytest.raises(ValueError):
        guofish_core.roundtrip_bench(0, 10, lambda arr: None)
    with pytest.raises(ValueError):
        guofish_core.roundtrip_bench(32, 0, lambda arr: None)


def test_roundtrip_bench_propagates_callback_exceptions():
    class Boom(Exception):
        pass

    def callback(arr):
        raise Boom

    with pytest.raises(Boom):
        guofish_core.roundtrip_bench(32, 10, callback)


def test_roundtrip_bench_leaves_the_gil_consistent():
    """After the bench returns, Python must still be fully usable."""
    guofish_core.roundtrip_bench(64, 200, lambda arr: None)
    assert sum(range(1000)) == 499500
    assert guofish_core.ping() == "pong"


@pytest.mark.parametrize("rows", [32, 64, 128, 256])
def test_roundtrip_latency_gate(rows):
    """Go/no-go for the scope 2.1 dispatcher design.

    C0 brief: if p99 round-trip exceeds ~2 ms at batch 256, the dispatcher needs
    revisiting before C5. Asserted at every batch size because the round trip
    should not depend on batch size at all.
    """
    calls = 0

    def callback(arr):
        nonlocal calls
        calls += 1
        arr.sum()  # touch the payload so the transfer is not optimised away

    iters = 2000
    res = guofish_core.roundtrip_bench(rows, iters, callback)

    assert calls == iters
    assert res["p99_ms"] < 2.0, (
        f"batch {rows}: p99 round-trip {res['p99_ms']:.3f} ms exceeds the 2 ms gate "
        f"(median {res['median_ms']:.3f} ms)"
    )
