// GuoFish C0 / C0b — toolchain spike and GIL contention probe.
//
// No engine logic lives here. This translation unit exists to prove four
// things before any of it gets written:
//
//   1. the extension builds warning-clean and imports on MSVC and Clang,
//   2. a C++-owned, 64-byte-aligned buffer reaches NumPy with no copy, so the
//      evaluator in scope 2.1 can write batches straight into Python's view,
//   3. the GIL acquire -> Python callback -> return round trip is fast enough
//      that the dispatcher thread is not the bottleneck (go/no-go for C5),
//   4. (C0b) what that round trip costs when a *second* Python thread is
//      burning bytecode, which is the condition scope 2.1 actually runs in:
//      UCI `info` strings are formatted in Python while search continues.
//
// (3) measures the mechanism. (4) measures the queue in front of it. They are
// different numbers by four orders of magnitude; see BENCH.md.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// Vendored per Global Rule 7. C0 only proves it compiles and links; C1 is the
// chunk that actually generates moves with it.
#include <chess.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <ratio>
#include <stdexcept>
#include <string>
#include <vector>

// Whether this translation unit was built with AddressSanitizer. Clang answers
// via __has_feature; MSVC and GCC define __SANITIZE_ADDRESS__. Both spellings
// are checked because the project builds on both toolchains (Global Rule 8).
//
// This is not decoration: the C0b acquire-wait tail is a scheduling artifact
// that instrumentation makes materially worse, so the gate assertion needs to
// know whether it is looking at a production build. See BENCH.md.
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define GUOFISH_ASAN_BUILD 1
#  endif
#endif
#if !defined(GUOFISH_ASAN_BUILD) && defined(__SANITIZE_ADDRESS__)
#  define GUOFISH_ASAN_BUILD 1
#endif
#if !defined(GUOFISH_ASAN_BUILD)
#  define GUOFISH_ASAN_BUILD 0
#endif

#if defined(__has_feature)
#  if __has_feature(undefined_behavior_sanitizer)
#    define GUOFISH_UBSAN_BUILD 1
#  endif
#endif
#if !defined(GUOFISH_UBSAN_BUILD)
#  define GUOFISH_UBSAN_BUILD 0
#endif

namespace py = pybind11;

namespace {

// 64 bytes: one x86 cache line, and the alignment AVX-512 loads want.
constexpr std::size_t kAlignment = 64;

// The tokenizer width this port targets (scope C2: tokens(fen) -> int32[68]).
// The benchmarks write rows of this width so the simulated work matches the
// shape of the real dispatcher batch rather than being an arbitrary memset.
constexpr std::size_t kTokenWidth = 68;

using clock_type = std::chrono::steady_clock;

double micros_between(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

// A C++-owned, 64-byte-aligned int32 matrix.
//
// Alignment comes from C++17 over-aligned operator new rather than
// _aligned_malloc (MSVC) or posix_memalign (POSIX): it is the one spelling that
// is portable across both toolchains with no preprocessor branching, and unlike
// std::aligned_alloc it does not require the size to be a multiple of the
// alignment. See DECISIONS.md.
class AlignedBuffer {
public:
    AlignedBuffer(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), count_(checked_count(rows, cols)), data_(allocate(count_)) {
        std::fill_n(data_, count_, std::int32_t{0});
        // reinterpret_cast: pointer -> integer solely to check alignment; the
        // value is never dereferenced or converted back to a pointer.
        assert(reinterpret_cast<std::uintptr_t>(data_) % kAlignment == 0);
    }

    ~AlignedBuffer() { ::operator delete(data_, std::align_val_t(kAlignment)); }

    AlignedBuffer(const AlignedBuffer &) = delete;
    AlignedBuffer &operator=(const AlignedBuffer &) = delete;
    AlignedBuffer(AlignedBuffer &&) = delete;
    AlignedBuffer &operator=(AlignedBuffer &&) = delete;

    std::int32_t *data() noexcept { return data_; }
    const std::int32_t *data() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }
    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }

    std::int32_t *row(std::size_t r) noexcept {
        assert(r < rows_);
        return data_ + r * cols_;
    }

private:
    static std::size_t checked_count(std::size_t rows, std::size_t cols) {
        constexpr std::size_t kMax = (std::numeric_limits<std::size_t>::max)();
        if (rows != 0 && cols > kMax / rows) {
            throw std::overflow_error("guofish_core: buffer dimensions overflow size_t");
        }
        const std::size_t count = rows * cols;
        if (count > kMax / sizeof(std::int32_t)) {
            throw std::overflow_error("guofish_core: buffer byte size overflows size_t");
        }
        return count;
    }

    static std::int32_t *allocate(std::size_t count) {
        void *raw = ::operator new(count * sizeof(std::int32_t), std::align_val_t(kAlignment));
        return static_cast<std::int32_t *>(raw);
    }

    std::size_t rows_;
    std::size_t cols_;
    std::size_t count_;
    std::int32_t *data_;
};

// The buffer most recently handed out by make_buffer(). buffer_checksum() reads
// this and nothing else, which is how the test proves that a write performed in
// Python landed in memory C++ owns rather than in a copy.
std::shared_ptr<AlignedBuffer> g_buffer;

// Wrap a buffer in a NumPy array that aliases it — no copy, and the allocation
// outlives whichever reference dies first.
//
// The capsule owns its own shared_ptr copy, so Python may `del` the array while
// C++ still holds the buffer, and C++ may drop the buffer while an older view is
// still alive. Requires the GIL.
py::array_t<std::int32_t> make_view(const std::shared_ptr<AlignedBuffer> &buffer) {
    auto *owner = new std::shared_ptr<AlignedBuffer>(buffer);
    py::capsule base(owner, [](void *p) { delete static_cast<std::shared_ptr<AlignedBuffer> *>(p); });

    const auto r = static_cast<py::ssize_t>(buffer->rows());
    const auto c = static_cast<py::ssize_t>(buffer->cols());
    const auto item = static_cast<py::ssize_t>(sizeof(std::int32_t));

    // Passing a non-null base makes pybind11 build the array around our pointer
    // with NPY_ARRAY_OWNDATA clear — i.e. a view, not a copy.
    return py::array_t<std::int32_t>({r, c}, {c * item, item}, buffer->data(), base);
}

double median_of(const std::vector<double> &sorted) {
    const std::size_t n = sorted.size();
    assert(n > 0);
    if (n % 2 == 1) {
        return sorted[n / 2];
    }
    return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

// Nearest-rank percentile: the smallest sample at or above the q-th position.
// No interpolation, so every reported number is a latency that actually
// occurred rather than an average of two that did not.
double percentile_of(const std::vector<double> &sorted, double q) {
    const std::size_t n = sorted.size();
    assert(n > 0);
    auto rank = static_cast<std::size_t>(std::ceil(q * static_cast<double>(n)));
    if (rank == 0) {
        rank = 1;
    }
    if (rank > n) {
        rank = n;
    }
    return sorted[rank - 1];
}

// Sorts `samples` in place and summarises it. p50 is the nearest-rank 50th
// percentile, not the interpolated median, so every key in the returned dict is
// a value that was actually observed.
py::dict summarize(std::vector<double> &samples) {
    assert(!samples.empty());

    double total = 0.0;
    for (double s : samples) {
        total += s;
    }
    std::sort(samples.begin(), samples.end());

    py::dict d;
    d["p50"] = percentile_of(samples, 0.50);
    d["p95"] = percentile_of(samples, 0.95);
    d["p99"] = percentile_of(samples, 0.99);
    d["min"] = samples.front();
    d["max"] = samples.back();
    d["mean"] = total / static_cast<double>(samples.size());
    d["n"] = samples.size();
    return d;
}

std::string ping() { return "pong"; }

// Provenance for anything transcribed into BENCH.md, and the input the C0b gate
// uses to decide whether it is looking at a production build or an instrumented
// one. `asserts` reports whether NDEBUG is off, i.e. whether Global Rule 5's
// "all debug asserts" are actually live in the module under test.
py::dict build_info() {
    py::dict d;
    d["asan"] = static_cast<bool>(GUOFISH_ASAN_BUILD);
    d["ubsan"] = static_cast<bool>(GUOFISH_UBSAN_BUILD);
#if defined(NDEBUG)
    d["asserts"] = false;
#else
    d["asserts"] = true;
#endif
#if defined(_MSC_VER)
    d["compiler"] = "MSVC " + std::to_string(_MSC_FULL_VER);
#elif defined(__clang__)
    d["compiler"] = "Clang " __clang_version__;
#elif defined(__GNUC__)
    d["compiler"] = "GCC " __VERSION__;
#else
    d["compiler"] = "unknown";
#endif
    // _MSVC_LANG is MSVC's always-correct answer; __cplusplus only agrees when
    // /Zc:__cplusplus is passed (CMakeLists.txt does pass it). Reading
    // _MSVC_LANG first means this stays truthful even if that flag is dropped,
    // which matters because the gate below trusts what this function reports.
#if defined(_MSVC_LANG)
    d["cpp_standard"] = static_cast<long>(_MSVC_LANG);
#else
    d["cpp_standard"] = static_cast<long>(__cplusplus);
#endif
    return d;
}

// Nominal and empirically measured resolution of the clock every number in
// BENCH.md comes from. Without this a "0.100 us median" cannot be told apart
// from "one tick, i.e. below the noise floor" — which is exactly what the
// Windows C0 numbers turned out to be.
py::dict clock_info(std::size_t probes) {
    if (probes < 2) {
        throw py::value_error("guofish_core.clock_info: probes must be >= 2");
    }

    constexpr double kNominalNs =
        1e9 * static_cast<double>(clock_type::period::num) / static_cast<double>(clock_type::period::den);

    double min_nonzero_ns = std::numeric_limits<double>::infinity();
    std::size_t zero_deltas = 0;

    {
        py::gil_scoped_release released;
        auto prev = clock_type::now();
        for (std::size_t i = 1; i < probes; ++i) {
            const auto cur = clock_type::now();
            const double ns = std::chrono::duration<double, std::nano>(cur - prev).count();
            if (ns > 0.0) {
                min_nonzero_ns = (std::min)(min_nonzero_ns, ns);
            } else {
                ++zero_deltas;
            }
            prev = cur;
        }
    }

    py::dict d;
    d["nominal_tick_ns"] = kNominalNs;
    d["measured_tick_ns"] = std::isfinite(min_nonzero_ns) ? min_nonzero_ns : 0.0;
    d["zero_delta_fraction"] = static_cast<double>(zero_deltas) / static_cast<double>(probes - 1);
    d["is_steady"] = clock_type::is_steady;
    d["probes"] = probes;
    return d;
}

py::array_t<std::int32_t> make_buffer(std::size_t rows, std::size_t cols) {
    auto buffer = std::make_shared<AlignedBuffer>(rows, cols);
    g_buffer = buffer;
    return make_view(buffer);
}

std::int64_t buffer_checksum() {
    if (!g_buffer) {
        return 0;
    }
    const std::int32_t *d = g_buffer->data();
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < g_buffer->size(); ++i) {
        sum += d[i];
    }
    return sum;
}

py::dict roundtrip_bench(std::size_t rows, std::size_t iters, const py::object &callback) {
    if (rows == 0) {
        throw py::value_error("guofish_core.roundtrip_bench: rows must be > 0");
    }
    if (iters == 0) {
        throw py::value_error("guofish_core.roundtrip_bench: iters must be > 0");
    }

    // Local, not g_buffer: benchmarking must not disturb the buffer the
    // zero-copy tests are inspecting.
    auto scratch = std::make_shared<AlignedBuffer>(rows, kTokenWidth);

    // C0b: the callback now receives the live zero-copy view, so the measured
    // segment includes handing real data across the boundary rather than
    // calling a nullary function. Built once, outside the loop — the buffer
    // address is stable, so a production dispatcher would reuse the view too,
    // and rebuilding it per iteration would time NumPy object construction
    // instead of the GIL round trip. See DECISIONS.md.
    py::array_t<std::int32_t> view = make_view(scratch);

    std::vector<double> call_us;
    std::vector<double> scope_us;
    call_us.reserve(iters);
    scope_us.reserve(iters);

    std::int64_t work = 0;

    {
        // The dispatcher thread does its own work without the GIL.
        py::gil_scoped_release released;

        for (std::size_t i = 0; i < iters; ++i) {
            // Simulated C++ work: fill one row, as the evaluator would.
            std::int32_t *dst = scratch->row(i % scratch->rows());
            for (std::size_t j = 0; j < scratch->cols(); ++j) {
                dst[j] = static_cast<std::int32_t>(i + j);
            }

            const auto t0 = clock_type::now();
            double acquire_call_return = 0.0;
            {
                py::gil_scoped_acquire acquired;
                callback(view);
                // Sampled inside the scope: this is the "acquire -> call ->
                // return" segment the brief asks for, excluding the release.
                acquire_call_return = micros_between(t0, clock_type::now());
            }
            const auto t1 = clock_type::now();

            call_us.push_back(acquire_call_return);
            scope_us.push_back(micros_between(t0, t1));
        }

        // Consume the writes so the optimiser cannot delete the simulated work
        // as dead stores and leave us timing an empty loop.
        for (std::size_t i = 0; i < scratch->size(); ++i) {
            work += scratch->data()[i];
        }
    }

    double total_us = 0.0;
    for (double sample : call_us) {
        total_us += sample;
    }

    std::sort(call_us.begin(), call_us.end());
    std::sort(scope_us.begin(), scope_us.end());

    const double median_us = median_of(call_us);
    const double p99_us = percentile_of(call_us, 0.99);

    py::dict res;
    res["rows"] = rows;
    res["iters"] = iters;

    // Primary metric: GIL acquire -> callback -> return.
    res["median_us"] = median_us;
    res["p99_us"] = p99_us;
    res["median_ms"] = median_us / 1000.0;
    res["p99_ms"] = p99_us / 1000.0;
    res["mean_us"] = total_us / static_cast<double>(iters);
    res["min_us"] = call_us.front();
    res["max_us"] = call_us.back();

    // Same segment plus the GIL release at scope exit — the full per-batch cost
    // the dispatcher actually pays.
    res["scope_median_us"] = median_of(scope_us);
    res["scope_p99_us"] = percentile_of(scope_us, 0.99);

    // Returned only to keep the simulated work observable; see above.
    res["work_checksum"] = work;

    return res;
}

// Burn `us` microseconds of CPU without the GIL, touching `buf` so the loop is
// not empty and cannot be optimised away. Must be called with the GIL released.
//
// A spin rather than a sleep: this models the search threads doing MCTS work
// between batches, which occupies a core. Sleeping would hand the core back to
// the OS and change the scheduling picture it is meant to reproduce.
void burn_without_gil(double us, AlignedBuffer &buf, std::int64_t &sink) {
    if (us <= 0.0) {
        return;
    }
    const auto start = clock_type::now();
    std::size_t r = 0;
    for (;;) {
        std::int32_t *dst = buf.row(r % buf.rows());
        for (std::size_t j = 0; j < buf.cols(); ++j) {
            dst[j] = static_cast<std::int32_t>(r + j);
        }
        sink += dst[0];
        ++r;
        if (micros_between(start, clock_type::now()) >= us) {
            return;
        }
    }
}

// C0b. Same loop as roundtrip_bench, but the three phases are timed separately
// so a wait for the GIL can be told apart from the cost of the callback itself.
//
// The split matters because the two have different fixes. Time in `call_us` is
// paid down by making the Python callback cheaper; time in `acquire_wait_us` is
// paid down only by bounding how long *some other* thread holds the GIL —
// sys.setswitchinterval, or moving that thread's work out of Python entirely.
//
// `work_us` is the GIL-free gap between one callback and the next: the time the
// real dispatcher spends waiting for search threads to fill the next batch. It
// is not a detail. At work_us = 0 the loop re-requests the GIL within ~100 ns of
// releasing it, so it usually wins the re-acquire before the competing Python
// thread has even been scheduled, and the contention this function exists to
// measure mostly does not happen. See BENCH.md, "why the gap matters".
py::dict contention_bench(std::size_t rows, std::size_t iters, const py::object &callback,
                          double work_us) {
    if (rows == 0) {
        throw py::value_error("guofish_core.contention_bench: rows must be > 0");
    }
    if (iters == 0) {
        throw py::value_error("guofish_core.contention_bench: iters must be > 0");
    }
    if (!(work_us >= 0.0)) {  // also rejects NaN
        throw py::value_error("guofish_core.contention_bench: work_us must be >= 0");
    }

    auto scratch = std::make_shared<AlignedBuffer>(rows, kTokenWidth);
    py::array_t<std::int32_t> view = make_view(scratch);

    std::vector<double> acquire_us;
    std::vector<double> call_us;
    std::vector<double> release_us;
    acquire_us.reserve(iters);
    call_us.reserve(iters);
    release_us.reserve(iters);

    std::int64_t work = 0;
    const auto wall_start = clock_type::now();

    {
        py::gil_scoped_release released;

        for (std::size_t i = 0; i < iters; ++i) {
            std::int32_t *dst = scratch->row(i % scratch->rows());
            for (std::size_t j = 0; j < scratch->cols(); ++j) {
                dst[j] = static_cast<std::int32_t>(i + j);
            }

            // The GIL-free gap. Everything the search threads do lives here.
            burn_without_gil(work_us, *scratch, work);

            // t0 -> t1  the GIL is requested but not yet held (acquire wait)
            // t1 -> t2  the GIL is held and the callback runs
            // t2 -> t3  the GIL is handed back at scope exit
            //
            // clock_type::now() needs no GIL, so t0 and t3 can be sampled from
            // outside the acquire scope without perturbing what they measure.
            const auto t0 = clock_type::now();
            clock_type::time_point t1;
            clock_type::time_point t2;
            {
                py::gil_scoped_acquire acquired;
                t1 = clock_type::now();
                callback(view);
                t2 = clock_type::now();
            }
            const auto t3 = clock_type::now();

            acquire_us.push_back(micros_between(t0, t1));
            call_us.push_back(micros_between(t1, t2));
            release_us.push_back(micros_between(t2, t3));
        }

        for (std::size_t i = 0; i < scratch->size(); ++i) {
            work += scratch->data()[i];
        }
    }

    const double wall_us = micros_between(wall_start, clock_type::now());

    py::dict res;
    res["rows"] = rows;
    res["iters"] = iters;
    res["work_us"] = work_us;
    res["acquire_wait_us"] = summarize(acquire_us);
    res["call_us"] = summarize(call_us);
    res["release_us"] = summarize(release_us);
    res["wall_us"] = wall_us;
    res["work_checksum"] = work;
    return res;
}

}  // namespace

PYBIND11_MODULE(guofish_core, m) {
    m.doc() = "GuoFish C++ core — C0 toolchain spike, C0b GIL contention probe";

    m.def("ping", &ping, "Import health check; returns \"pong\".");

    m.def("build_info", &build_info,
          "Toolchain, sanitizer and assert status of this module. Provenance for BENCH.md.");

    m.def("make_buffer", &make_buffer, py::arg("rows"), py::arg("cols"),
          "Allocate a 64-byte-aligned, C++-owned int32 matrix and return it as a "
          "zero-copy NumPy view.");

    m.def("buffer_checksum", &buffer_checksum,
          "Sum the buffer from the last make_buffer() call, reading C++ memory directly.");

    m.def("clock_info", &clock_info, py::arg("probes") = 200000,
          "Nominal and measured resolution of the steady_clock every benchmark here uses.");

    m.def("roundtrip_bench", &roundtrip_bench, py::arg("rows"), py::arg("iters"), py::arg("callback"),
          "Time `iters` GIL acquire -> callback(view) -> return round trips; returns latency stats. "
          "`view` is a zero-copy rows x 68 int32 NumPy view of a C++-owned buffer.");

    m.def("contention_bench", &contention_bench, py::arg("rows"), py::arg("iters"), py::arg("callback"),
          py::arg("work_us") = 0.0,
          "As roundtrip_bench, but reports the GIL acquire wait, the callback, and the GIL "
          "release as three separately timed phases (p50/p95/p99/max each). `work_us` is the "
          "GIL-free gap between callbacks, i.e. how long the search threads run between "
          "batches; it strongly affects who wins the GIL and must not be left at 0 when "
          "modelling a real dispatcher.");
}
