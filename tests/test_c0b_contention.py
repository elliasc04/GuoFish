"""C0b acceptance tests — contended GIL acquisition.

C0 measured the GIL round trip with nothing else running and got ~100 ns. That
number is a floor, not a prediction: the dispatcher in scope 2.1 shares the
interpreter with a Python thread formatting UCI ``info`` lines, and CPython will
not hand the GIL back until that thread reaches an eval-breaker check.

Three configurations, per the C0b brief:

    A   no background thread                       (control)
    B   background thread, switch interval 5 ms    (default — the risk)
    C   background thread, switch interval 0.5 ms  (the proposed mitigation)

Two things are asserted, and they fail for opposite reasons:

  * **The simulation must work.** If B is indistinguishable from A the harness is
    not creating contention and every other number here is meaningless. This is
    the more important of the two — a broken contention test reporting "no
    contention" is exactly the false green this chunk exists to rule out.

  * **The mitigation must work.** In C, p99 ``acquire_wait_us`` < 1 ms and max
    < 2 ms at batch 256. If this fails, ``sys.setswitchinterval`` is not
    sufficient and UCI output must move into C++ before C10.

Two properties of the harness are load-bearing and were both found the hard way:

``DISPATCH_GAP_US`` — the GIL-free interval between callbacks — must be
non-zero. With no gap the dispatcher re-requests the GIL within ~100 ns of
releasing it and usually wins the re-acquire before the competing thread is even
scheduled, so ~90% of iterations see no contention at all and config B's *median*
looks identical to config A's. The contention is then only visible in the tail.
A real dispatcher waits for search threads to fill the next batch, so a gap is
both more realistic and strictly more adversarial. See DECISIONS.md.

The background load is **pure Python**: a C extension holding the GIL inside one
long call would not yield at bytecode boundaries at all, so the switch interval
would have no effect and the experiment would measure the wrong thing.
"""

import platform
import statistics
import sys
import threading
import time
import warnings

import pytest

import guofish_core

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SWITCH_INTERVAL = 0.005  # CPython's default, restated for the record
FAST_SWITCH_INTERVAL = 0.0005  # the mitigation under test

BATCH_SIZES = (32, 256)

# The GIL-free gap between one batch callback and the next: what the dispatcher
# spends waiting for search threads. 200 us is the order of magnitude of
# assembling a 256-position batch across a handful of search threads. The exact
# value is not critical — tools/bench_c0b.py sweeps it, and everything from
# 10 us to 1000 us behaves the same. What matters is that it is not zero; see
# the module docstring.
DISPATCH_GAP_US = 200.0

# Iteration counts differ per config because the configs differ in cost by three
# orders of magnitude: in B every iteration waits out a GIL handoff (~5 ms
# here), so 200 iterations already costs ~1 s per batch size. Chosen to keep the
# module near ~5 s while leaving p99 backed by enough samples to mean something
# (nearest-rank p99 of 200 is the 2nd worst sample; of 2000, the 20th).
# tools/bench_c0b.py runs the same harness at a higher scale for BENCH.md.
ITERS = {"A": 1500, "B": 200, "C": 2000}

GATE_P99_US = 1000.0  # 1 ms
GATE_MAX_US = 2000.0  # 2 ms

BUILD = dict(guofish_core.build_info())

# Which configuration the ``max`` half of the gate is enforced on.
#
# ``max`` is a single worst sample out of thousands, so it measures the OS
# scheduler's tail rather than GIL behaviour, and instrumentation moves it a
# long way. Measured over 20,000 samples per configuration (BENCH.md):
#
#     Windows Release   max 86-285 us     0/10 runs over 2 ms
#     Windows ASan      max 127-2036 us   1/10 runs over 2 ms
#     WSL2 Release      max 754-1987 us   0/10 runs over 2 ms
#     WSL2 ASan         max 817-3314 us   3/10 runs over 2 ms
#
# p99 by contrast never exceeded 733 us in any of those 80,000 samples.
#
# The C0b brief designates Windows as authoritative for the gate and WSL2 as a
# timing sanity check only. Asserting ``max`` on the sanitized or WSL2 builds
# would therefore make the suite intermittently red for a reason the brief has
# already ruled out of scope -- so on those builds the excursion is *reported*
# (as a warning, and in BENCH.md) and a relative bound is asserted instead:
# config C's tail must still beat config B's, which catches an actual
# regression in the mitigation. p99 is asserted everywhere, unconditionally.
#
# This is the one place C0b interprets rather than follows the brief. See
# DECISIONS.md, "Where the max criterion is enforced".
GATE_MAX_IS_AUTHORITATIVE = platform.system() == "Windows" and not BUILD["asan"]

# A plausible principal variation. Length matters: this is what the background
# thread formats and joins on every pass, and a real ``info`` line carries a PV
# of roughly this depth.
PV_MOVES = [
    "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
    "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6", "c2c3", "e8g8",
    "h2h3", "c6b8", "d2d4", "b8d7", "b1d2", "c7c5", "d4c5", "d6c5",
]


class UciFormatterLoad:
    """A pure-Python CPU burner shaped like UCI ``info`` emission.

    Pure Python on purpose (see module docstring): the thread must yield only at
    bytecode boundaries so that ``sys.setswitchinterval`` actually governs how
    long it can hold the GIL.

    ``iterations`` is published continuously rather than only at exit, so the
    context manager can confirm the thread is making progress *before* anything
    is timed. A background thread that silently died would make config B look
    identical to config A and quietly invalidate the whole chunk.
    """

    #: Seconds to let the thread reach a steady state before measuring.
    WARMUP_S = 0.15

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self.iterations = 0

    def _burn(self):
        n = 0
        sink = []
        while not self._stop.is_set():
            # 40 formats between stop checks: enough that the loop is dominated
            # by bytecode rather than by Event.is_set(), short enough to shut
            # down promptly.
            for depth in range(1, 41):
                pv = " ".join([f"{move}" for move in PV_MOVES])
                line = (
                    f"info depth {depth} seldepth {depth + 4} multipv 1 "
                    f"score cp {depth * 7 - 140} nodes {depth * 123457} "
                    f"nps {depth * 54321} hashfull {depth * 25} "
                    f"tbhits 0 time {depth * 13} pv {pv}"
                )
                sink.append(line[-8:])
                n += 1
            self.iterations = n
            if len(sink) > 4000:
                del sink[:]

    def __enter__(self):
        self._thread = threading.Thread(target=self._burn, name="uci-info-load", daemon=True)
        self._thread.start()
        time.sleep(self.WARMUP_S)
        assert self._thread.is_alive(), "background load thread died before measurement"
        assert self.iterations > 0, (
            "background load thread produced no output during warm-up; it is not "
            "running and any contention measured against it is invalid"
        )
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10.0)
        assert not self._thread.is_alive(), "background load thread did not stop"
        return False


def _callback_factory(counter):
    def callback(arr):
        counter[0] += 1
        # Touch the payload so the transfer across the boundary is real work and
        # cannot be optimised or elided.
        counter[1] += int(arr.sum())

    return callback


def _measure(rows, iters, work_us=DISPATCH_GAP_US):
    counter = [0, 0]
    res = guofish_core.contention_bench(rows, iters, _callback_factory(counter), work_us)
    assert counter[0] == iters, f"callback ran {counter[0]} times, expected {iters}"
    res = dict(res)
    res["callback_calls"] = counter[0]
    res["payload_sum"] = counter[1]
    return res


def run_configurations(iters=None, gap_us=DISPATCH_GAP_US):
    """Run A, B and C at each batch size. Returns {(config, rows): result}.

    Public because ``tools/bench_c0b.py`` calls it to produce the BENCH.md
    tables. The published numbers and the asserted numbers therefore come from
    the same harness by construction, and cannot drift apart.
    """
    iters = dict(ITERS if iters is None else iters)
    original_interval = sys.getswitchinterval()
    out = {
        "clock": dict(guofish_core.clock_info(200000)),
        "bg_iterations": {},
        "iters": iters,
        "gap_us": gap_us,
    }

    try:
        # Warm up: the first call pays import/branch-predictor cold-start costs.
        _measure(32, 200)

        # --- A: control, no competing Python thread ------------------------
        sys.setswitchinterval(DEFAULT_SWITCH_INTERVAL)
        for rows in BATCH_SIZES:
            out[("A", rows)] = _measure(rows, iters["A"], gap_us)

        # --- B: contention at the default switch interval ------------------
        sys.setswitchinterval(DEFAULT_SWITCH_INTERVAL)
        with UciFormatterLoad() as load:
            for rows in BATCH_SIZES:
                out[("B", rows)] = _measure(rows, iters["B"], gap_us)
            # Same config, no GIL-free gap. Not a fourth configuration: this is
            # the control proving the gap is what exposes the contention, and it
            # is the regime C0's benchmark accidentally measured.
            out[("B-nogap", 256)] = _measure(256, iters["B"], 0.0)
        out["bg_iterations"]["B"] = load.iterations

        # --- C: same contention, bounded switch interval -------------------
        sys.setswitchinterval(FAST_SWITCH_INTERVAL)
        with UciFormatterLoad() as load:
            for rows in BATCH_SIZES:
                out[("C", rows)] = _measure(rows, iters["C"], gap_us)
        out["bg_iterations"]["C"] = load.iterations
    finally:
        sys.setswitchinterval(original_interval)

    return out


@pytest.fixture(scope="module")
def measurements():
    return run_configurations()


# ---------------------------------------------------------------------------
# Shape of the result
# ---------------------------------------------------------------------------

PHASES = ("acquire_wait_us", "call_us", "release_us")
STATS = ("p50", "p95", "p99", "min", "max", "mean", "n")


def test_contention_bench_reports_three_phases():
    res = _measure(32, 200, 0.0)

    for phase in PHASES:
        assert phase in res, f"missing phase {phase}"
        missing = set(STATS) - set(res[phase])
        assert not missing, f"{phase} is missing {sorted(missing)}"

        s = res[phase]
        assert s["n"] == 200
        assert s["min"] <= s["p50"] <= s["p95"] <= s["p99"] <= s["max"]
        assert s["min"] >= 0.0


def test_contention_bench_phases_sum_to_the_wall_clock():
    """The three phases must partition the loop, not overlap or leave a gap.

    If they did not, ``acquire_wait_us`` could be measuring something other than
    the wait — the failure mode where the gate passes for the wrong reason.
    """
    iters = 400
    res = _measure(32, iters, 0.0)

    accounted = sum(res[phase]["mean"] for phase in PHASES) * iters
    assert accounted <= res["wall_us"], "phases account for more time than elapsed"
    # The unaccounted remainder is the C++ row write plus loop overhead. It is
    # tiny in absolute terms; assert it is not hiding a fourth phase.
    assert res["wall_us"] - accounted < 0.5 * res["wall_us"] + 5000.0


def test_work_us_actually_consumes_the_requested_time():
    """``work_us`` is load-bearing, so it must do what it says."""
    iters = 200
    gap = 300.0

    without = _measure(32, iters, 0.0)
    with_gap = _measure(32, iters, gap)

    added_us = with_gap["wall_us"] - without["wall_us"]
    expected = iters * gap
    assert added_us > 0.7 * expected, (
        f"work_us={gap} over {iters} iterations added only {added_us:.0f} us, "
        f"expected roughly {expected:.0f} us"
    )
    assert with_gap["work_us"] == gap, "the gap actually used is not reported back"


def test_contention_bench_rejects_degenerate_arguments():
    with pytest.raises(ValueError):
        guofish_core.contention_bench(0, 10, lambda arr: None)
    with pytest.raises(ValueError):
        guofish_core.contention_bench(32, 0, lambda arr: None)
    with pytest.raises(ValueError):
        guofish_core.contention_bench(32, 10, lambda arr: None, -1.0)
    with pytest.raises(ValueError):
        guofish_core.contention_bench(32, 10, lambda arr: None, float("nan"))


def test_contention_bench_propagates_callback_exceptions():
    class Boom(Exception):
        pass

    def callback(arr):
        raise Boom

    with pytest.raises(Boom):
        guofish_core.contention_bench(32, 10, callback)


def test_contention_bench_leaves_the_gil_consistent():
    guofish_core.contention_bench(64, 100, lambda arr: None)
    assert sum(range(1000)) == 499500
    assert guofish_core.ping() == "pong"


# ---------------------------------------------------------------------------
# Clock resolution — every microsecond figure below depends on this
# ---------------------------------------------------------------------------


def test_build_info_is_self_consistent():
    """The gate reads ``asan`` to decide how strictly to judge the tail, so a
    wrong answer here would silently change what the acceptance test means."""
    info = guofish_core.build_info()

    assert set(info) == {"asan", "ubsan", "asserts", "compiler", "cpp_standard"}
    assert isinstance(info["asan"], bool)
    assert info["cpp_standard"] >= 201703, "built below C++17; over-aligned new is not guaranteed"
    assert info["compiler"] != "unknown"

    # Rule 5 requires the sanitized build to keep debug asserts live. If ASan is
    # on, NDEBUG must be off — CMake strips it, and this is what proves it.
    if info["asan"]:
        assert info["asserts"], "ASan build has NDEBUG set; Global Rule 5 asserts are compiled out"


def test_clock_is_steady_and_fine_enough_to_resolve_the_gate():
    info = guofish_core.clock_info(200000)

    assert info["is_steady"], "steady_clock is not steady; durations are unreliable"

    # The gate is written in milliseconds. A clock that cannot resolve better
    # than ~10 us would make a sub-millisecond verdict meaningless.
    assert info["measured_tick_ns"] < 10_000.0, (
        f"clock resolution {info['measured_tick_ns']:.1f} ns is too coarse to "
        f"defend a 1 ms p99 gate"
    )


# ---------------------------------------------------------------------------
# Validity: the contention simulation must actually contend
# ---------------------------------------------------------------------------


def test_background_thread_actually_ran(measurements):
    for config in ("B", "C"):
        n = measurements["bg_iterations"][config]
        assert n > 1000, (
            f"config {config}: background thread completed only {n} format "
            f"passes; it is not producing CPU load and the contention "
            f"measurement is invalid"
        )


@pytest.mark.parametrize("rows", BATCH_SIZES)
def test_config_b_visibly_degrades_against_config_a(measurements, rows):
    """The experiment's own control.

    Per the C0b brief: if A, B and C are indistinguishable the simulation is
    broken and the chunk FAILS regardless of what the gate says. B must be
    visibly worse than A.
    """
    a = measurements[("A", rows)]["acquire_wait_us"]
    b = measurements[("B", rows)]["acquire_wait_us"]

    assert b["p50"] > 20.0 * max(a["p50"], 0.05), (
        f"batch {rows}: contended p50 acquire wait {b['p50']:.3f} us is not "
        f"meaningfully worse than uncontended {a['p50']:.3f} us — the "
        f"background thread is not competing for the GIL"
    )
    assert b["p50"] > 100.0, (
        f"batch {rows}: contended p50 acquire wait is only {b['p50']:.3f} us. "
        f"A thread holding the GIL for a 5 ms switch interval should cost far "
        f"more than this; the simulation is suspect"
    )
    assert b["p99"] > a["p99"]


@pytest.mark.parametrize("rows", BATCH_SIZES)
def test_uncontended_acquire_is_essentially_free(measurements, rows):
    """Config A must reproduce the C0 result, or A is not a valid control."""
    a = measurements[("A", rows)]["acquire_wait_us"]
    assert a["p50"] < 5.0, (
        f"batch {rows}: uncontended acquire wait p50 is {a['p50']:.3f} us; C0 "
        f"measured ~0.1 us, so something else is holding the GIL"
    )


def test_the_dispatch_gap_is_never_less_adversarial(measurements):
    """Documents why ``DISPATCH_GAP_US`` is not zero.

    The variable that actually matters is the length of the GIL-free interval
    between callbacks. Below roughly a microsecond the dispatcher re-requests
    the GIL before the competing thread is scheduled and wins the re-acquire, so
    the median iteration never contends and the cost hides in the tail — that is
    the regime C0's benchmark accidentally measured, and it understates the p50
    by four orders of magnitude.

    Two things are asserted, and neither depends on how fast this build is:

      * a real gap is never *less* adversarial than no gap, so measuring with
        one cannot flatter the result;
      * even with no explicit gap the contention shows up in the tail, so the
        no-gap regime hides the median rather than the phenomenon.

    The size of the difference is build-dependent and is therefore reported
    rather than asserted: in an optimised build the no-gap p50 collapses to
    ~1 us, but under ASan the instrumented row write is itself a gap of several
    microseconds and the no-gap p50 stays in the milliseconds. See BENCH.md.
    """
    nogap = measurements[("B-nogap", 256)]["acquire_wait_us"]
    gap = measurements[("B", 256)]["acquire_wait_us"]

    assert gap["p50"] > 0.8 * nogap["p50"], (
        f"adding a {measurements['gap_us']} us GIL-free gap made the median "
        f"acquire wait *better* ({gap['p50']:.1f} vs {nogap['p50']:.1f} us). The "
        f"gap is supposed to be the adversarial case; if it is not, the gate is "
        f"being measured against the easy regime"
    )
    assert nogap["p99"] > 1000.0, (
        f"even without an explicit gap the p99 acquire wait should show the GIL "
        f"handoff cost; got {nogap['p99']:.1f} us"
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", BATCH_SIZES)
def test_config_c_meets_the_acquire_wait_gate(measurements, rows):
    """Go/no-go for the scope 2.1 dispatcher under contention.

    C0b brief: in configuration C, p99 ``acquire_wait_us`` must be < 1 ms, and
    max < 2 ms at batch 256. Failing this means bounding the switch interval is
    not enough and UCI emission must move into C++ before C10.
    """
    c = measurements[("C", rows)]["acquire_wait_us"]

    # Asserted on every platform and every build. This is the robust half of the
    # gate and it has never been close to failing anywhere.
    assert c["p99"] < GATE_P99_US, (
        f"batch {rows}: config C p99 acquire wait {c['p99']:.1f} us exceeds the "
        f"{GATE_P99_US:.0f} us gate (p50 {c['p50']:.1f}, max {c['max']:.1f})"
    )

    if rows != 256:
        return

    over_by = c["max"] - GATE_MAX_US
    if GATE_MAX_IS_AUTHORITATIVE:
        assert c["max"] < GATE_MAX_US, (
            f"batch 256: config C max acquire wait {c['max']:.1f} us exceeds the "
            f"{GATE_MAX_US:.0f} us gate (p99 {c['p99']:.1f}). This is the "
            f"authoritative configuration, so this is a real gate failure"
        )
    else:
        if over_by > 0:
            warnings.warn(
                f"batch 256: config C max acquire wait {c['max']:.1f} us exceeds "
                f"the {GATE_MAX_US:.0f} us gate by {over_by:.1f} us on a "
                f"non-authoritative build "
                f"(platform={platform.system()}, asan={BUILD['asan']}). "
                f"Not failed here; see BENCH.md. p99 was {c['p99']:.1f} us.",
                stacklevel=2,
            )
        # Relative bound, asserted everywhere the absolute one is not: whatever
        # the scheduler is doing to the tail, the mitigation must still beat the
        # unmitigated configuration.
        b = measurements[("B", rows)]["acquire_wait_us"]
        assert c["max"] < b["max"], (
            f"batch 256: config C max acquire wait {c['max']:.1f} us is no better "
            f"than config B's {b['max']:.1f} us — the mitigation has stopped working"
        )


@pytest.mark.parametrize("rows", BATCH_SIZES)
def test_shorter_switch_interval_improves_contended_latency(measurements, rows):
    """C must be better than B, or the mitigation does nothing."""
    b = measurements[("B", rows)]["acquire_wait_us"]
    c = measurements[("C", rows)]["acquire_wait_us"]

    assert c["p50"] < b["p50"], (
        f"batch {rows}: setswitchinterval(0.0005) did not reduce the p50 "
        f"acquire wait ({c['p50']:.1f} us vs {b['p50']:.1f} us)"
    )
    assert c["p99"] < b["p99"]


# ---------------------------------------------------------------------------
# Human-readable dump — this is what BENCH.md is transcribed from
# ---------------------------------------------------------------------------

REPORT_ROWS = (("A", 32), ("A", 256), ("B", 32), ("B", 256), ("B-nogap", 256), ("C", 32), ("C", 256))


def print_report(measurements):
    """Emit the markdown tables BENCH.md carries. Shared with tools/bench_c0b.py."""
    clock = measurements["clock"]

    print(f"platform : {platform.system()} {platform.machine()}")
    print(f"python   : {platform.python_version()} ({sys.implementation.name})")
    print(
        f"build    : {BUILD['compiler']}, asan={BUILD['asan']} ubsan={BUILD['ubsan']} "
        f"asserts={BUILD['asserts']} "
        f"(max gate {'ENFORCED' if GATE_MAX_IS_AUTHORITATIVE else 'reported only'})"
    )
    print(
        f"clock    : steady_clock nominal tick {clock['nominal_tick_ns']:.1f} ns, "
        f"measured {clock['measured_tick_ns']:.1f} ns "
        f"({clock['zero_delta_fraction'] * 100:.1f}% of back-to-back reads identical)"
    )
    print(f"switch   : A/B {DEFAULT_SWITCH_INTERVAL} s, C {FAST_SWITCH_INTERVAL} s")
    print(f"gap      : {measurements['gap_us']} us GIL-free per iteration "
          f"(B-nogap row uses 0)")
    print(f"bg load  : B={measurements['bg_iterations']['B']:,} "
          f"C={measurements['bg_iterations']['C']:,} format passes")
    print()
    print("acquire_wait_us")
    print("| config | rows | iters | p50 | p95 | p99 | max |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for config, rows in REPORT_ROWS:
        s = measurements[(config, rows)]["acquire_wait_us"]
        print(
            f"| {config} | {rows} | {s['n']} | {s['p50']:.3f} | "
            f"{s['p95']:.3f} | {s['p99']:.3f} | {s['max']:.3f} |"
        )
    print()
    print("call_us / release_us")
    print("| config | rows | call p50 | call p99 | release p50 | release p99 |")
    print("|---|---:|---:|---:|---:|---:|")
    for config, rows in REPORT_ROWS:
        r = measurements[(config, rows)]
        call, rel = r["call_us"], r["release_us"]
        print(
            f"| {config} | {rows} | {call['p50']:.3f} | {call['p99']:.3f} | "
            f"{rel['p50']:.3f} | {rel['p99']:.3f} |"
        )


def test_report(measurements, capsys):
    """Not an assertion; prints the table. Run with `-s` to see it."""
    with capsys.disabled():
        print()
        print_report(measurements)

    # Keep this a real test: assert the three configs did not collapse into one
    # measurement, which is the failure the brief calls out explicitly.
    p50s = [measurements[(c, 256)]["acquire_wait_us"]["p50"] for c in ("A", "B", "C")]
    assert statistics.pstdev(p50s) > 0.0
