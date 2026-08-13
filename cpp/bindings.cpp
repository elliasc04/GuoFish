// GuoFish C0 / C0b / C1 — toolchain spike, GIL contention probe, movegen parity.
//
// The C0/C0b half of this translation unit holds no engine logic. It exists to
// prove four things before any of it gets written:
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
//
// C1 adds the first real engine surface: legal_moves(fen). The generation and
// UCI normalisation live in cpp/movegen.hpp; this file only binds them.
//
// C2 adds the tokenizer: tokens(fen) for one position and TokenBatch for the
// batched path the evaluator will actually use, which writes straight into the
// same kind of C++-owned aligned buffer C0 proved out. The encoding itself
// lives in cpp/tokens.hpp; this file only binds it and owns the memory.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Vendored per Global Rule 7.
#include <chess.hpp>

#include "arena.hpp"
#include "cache.hpp"
#include "fathom.hpp"
#include "keys.hpp"
#include "movegen.hpp"
#include "search.hpp"
#include "tablebase.hpp"
#include "tokens.hpp"
#include "values.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
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

// C9. Same three-step, and it has to be three steps for the same reason: MSVC
// has no `__has_feature` and its preprocessor does not short-circuit `&&`, so a
// single `defined(__has_feature) && __has_feature(...)` is a hard parse error
// there rather than a false.
#if defined(__has_feature)
#  if __has_feature(thread_sanitizer)
#    define GUOFISH_TSAN_BUILD 1
#  endif
#endif
#if !defined(GUOFISH_TSAN_BUILD) && defined(__SANITIZE_THREAD__)
#  define GUOFISH_TSAN_BUILD 1
#endif
#if !defined(GUOFISH_TSAN_BUILD)
#  define GUOFISH_TSAN_BUILD 0
#endif

namespace py = pybind11;

namespace {

// 64 bytes: one x86 cache line, and the alignment AVX-512 loads want.
constexpr std::size_t kAlignment = 64;

// The tokenizer width, taken from the encoding itself rather than restated:
// C0's benchmarks write rows of this width so the simulated work matches the
// shape of the real dispatcher batch, and C2's buffers are this wide because
// that is what the network's input is.
constexpr std::size_t kTokenWidth = static_cast<std::size_t>(guofish::kSeqLength);

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
    // C8's GUOFISH_DEBUG_VL flag is deliberately NOT reported here. It belongs
    // in this dict by subject matter, and tests/test_c0b_contention.py pins the
    // exact key set — Global Rule 1 makes that pin the specification, not a
    // detail to be updated. It is `guofish_core.DEBUG_VL` instead.
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

// ---------------------------------------------------------------------------
// C2 — tokenization
// ---------------------------------------------------------------------------

// One position, as a fresh 68-element int32 array.
//
// This allocates a NumPy array per call and is the *validation* signature, not
// the search path — parity against golden/tokens.npz is judged one position at
// a time. TokenBatch below is the shape the evaluator uses.
py::array_t<std::int32_t> tokens(std::string_view fen) {
    py::array_t<std::int32_t> out(static_cast<py::ssize_t>(guofish::kSeqLength));
    guofish::tokenize_into(fen, out.mutable_data());
    return out;
}

// Materialise an arbitrary iterable of FENs into a list this call owns.
//
// Two things follow, and the batch path needs both: a generator is consumed
// exactly once, here, rather than re-entered from inside a GIL-released region;
// and every string object stays alive at a fixed address for the duration of
// the call, which is what makes the cached UTF-8 pointers below safe to read
// without the GIL.
py::list materialize(const py::object &fens) {
    PyObject *raw = PySequence_List(fens.ptr());
    if (raw == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::list>(raw);
}

// Borrow each element's UTF-8 buffer. Must be called with the GIL held, and the
// returned views are valid only while `items` is alive.
//
// PyUnicode_AsUTF8AndSize caches the encoded form inside the str object, so the
// pointer it hands back lives as long as the object does. Python strings are
// immutable, so nothing can move or rewrite that buffer underneath a reader —
// which is why the tokenize loop may then run with the GIL released. It makes
// no Python API calls at all; it only reads bytes.
std::vector<std::string_view> borrow_utf8(const py::list &items) {
    const auto n = static_cast<std::size_t>(py::len(items));

    std::vector<std::string_view> views;
    views.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        PyObject *item = PyList_GetItem(items.ptr(), static_cast<py::ssize_t>(i));
        if (item == nullptr) {
            throw py::error_already_set();
        }
        if (!PyUnicode_Check(item)) {
            throw py::type_error("guofish_core: expected a str FEN at index " + std::to_string(i));
        }
        py::ssize_t length = 0;
        const char *data = PyUnicode_AsUTF8AndSize(item, &length);
        if (data == nullptr) {
            throw py::error_already_set();
        }
        views.emplace_back(data, static_cast<std::size_t>(length));
    }

    return views;
}

// A C++-owned [capacity, 68] int32 matrix that FENs are tokenized straight into.
//
// This is the batch interface the brief asks for. The important property is
// what does *not* happen: no per-position array, no list of arrays, no
// np.stack, and no copy between the parser and the network's input buffer. The
// parser's only output destination is a row of this allocation, and `view()`
// hands Python a NumPy array aliasing that same memory.
//
// The buffer is 64-byte aligned for the same reason C0's is — see AlignedBuffer.
class TokenBatch {
public:
    explicit TokenBatch(std::size_t capacity) : buffer_(std::make_shared<AlignedBuffer>(capacity, kTokenWidth)) {
        if (capacity == 0) {
            throw py::value_error("guofish_core.TokenBatch: capacity must be > 0");
        }
    }

    std::size_t capacity() const noexcept { return buffer_->rows(); }

    py::array_t<std::int32_t> view() const { return make_view(buffer_); }

    // Tokenize `fens` into rows [row_offset, row_offset + len(fens)). Returns
    // the number of rows written.
    //
    // On a FEN the parser refuses this throws ValueError, and the rows written
    // before it keep their new contents while the rest keep their old ones.
    // Recovering a half-filled batch is not attempted: the caller's batch is
    // wrong, and quietly leaving stale rows in place for the network to read
    // would be worse than a partially updated buffer plus an exception.
    std::size_t fill(const py::object &fens, std::size_t row_offset) {
        py::list items = materialize(fens);
        const std::vector<std::string_view> views = borrow_utf8(items);

        if (row_offset > buffer_->rows() || views.size() > buffer_->rows() - row_offset) {
            throw py::value_error("guofish_core.TokenBatch.fill: " + std::to_string(views.size()) +
                                  " FENs at row_offset " + std::to_string(row_offset) +
                                  " do not fit a batch of capacity " + std::to_string(buffer_->rows()));
        }

        {
            // Safe because the loop touches no Python object: see borrow_utf8.
            // `items` is held on this frame and keeps every buffer alive; the
            // release guard reacquires the GIL on the way out, including while
            // unwinding from a bad FEN.
            py::gil_scoped_release released;
            for (std::size_t i = 0; i < views.size(); ++i) {
                guofish::tokenize_into(views[i], buffer_->row(row_offset + i));
            }
        }

        return views.size();
    }

private:
    std::shared_ptr<AlignedBuffer> buffer_;
};

// Single-threaded tokenization throughput, for BENCH.md.
//
// The FENs are copied into C++-owned strings up front and the timed region
// touches no Python object at all, so what this reports is the cost of the
// encoder rather than the cost of the language boundary. `TokenBatch.fill` pays
// that boundary once per batch; tools/bench_c2.py measures it separately and
// BENCH.md carries both numbers, because quoting only the faster one would
// overstate what the dispatcher will actually see.
py::dict tokenize_bench(const py::object &fens, std::size_t repeats) {
    if (repeats == 0) {
        throw py::value_error("guofish_core.tokenize_bench: repeats must be > 0");
    }

    py::list items = materialize(fens);
    const std::vector<std::string_view> borrowed = borrow_utf8(items);
    if (borrowed.empty()) {
        throw py::value_error("guofish_core.tokenize_bench: need at least one FEN");
    }

    const std::vector<std::string> owned(borrowed.begin(), borrowed.end());

    // One row per position: writing every result to the same row would let the
    // whole buffer stay in L1 and measure something the real batch never sees.
    AlignedBuffer scratch(owned.size(), kTokenWidth);

    double elapsed_us = 0.0;
    std::int64_t checksum = 0;

    {
        py::gil_scoped_release released;
        const auto start = clock_type::now();
        for (std::size_t r = 0; r < repeats; ++r) {
            for (std::size_t i = 0; i < owned.size(); ++i) {
                guofish::tokenize_into(owned[i], scratch.row(i));
            }
        }
        elapsed_us = micros_between(start, clock_type::now());

        // Consume the writes so the encoder cannot be optimised away wholesale.
        for (std::size_t i = 0; i < scratch.size(); ++i) {
            checksum += scratch.data()[i];
        }
    }

    const auto positions = static_cast<double>(owned.size()) * static_cast<double>(repeats);

    py::dict d;
    d["positions"] = owned.size();
    d["repeats"] = repeats;
    d["total_positions"] = static_cast<std::size_t>(positions);
    d["elapsed_s"] = elapsed_us / 1e6;
    d["ns_per_position"] = (elapsed_us * 1000.0) / positions;
    d["positions_per_second"] = positions / (elapsed_us / 1e6);
    d["checksum"] = checksum;
    return d;
}

// ---------------------------------------------------------------------------
// C3 — the two keys
//
// Both cross the boundary as plain Python ints. The strong typing this chunk is
// about is a C++ property and cannot survive the trip: Python has no way to
// reject `cache[rep_key(fen)]`. What protects the Python side instead is the
// domain tag in each payload — the two keys for one position are never equal —
// and the fact that no Python code in the engine will hold either once C7 owns
// the cache. `key_type_separation()` below reports the compile-time facts so the
// acceptance test can check them rather than take them on trust.
// ---------------------------------------------------------------------------

std::uint64_t nn_key(std::string_view fen) { return guofish::nn_key(fen).value; }

std::uint64_t rep_key(std::string_view fen) { return guofish::rep_key(fen).value; }

// Both keys for one FEN, so a caller that needs the pair pays for one parse.
// This is also the shape C6/C7 will use internally.
py::tuple keys(std::string_view fen) {
    const guofish::ParsedFen parsed = guofish::parse_fen(fen);
    return py::make_tuple(guofish::nn_key(parsed).value, guofish::rep_key(parsed).value);
}

// The batch path, mirroring TokenBatch.fill(): the FENs are materialised and
// their UTF-8 buffers borrowed under the GIL, then the whole sweep runs with the
// GIL released. 100k positions through the one-at-a-time entry point would spend
// most of their time in pybind11's argument marshalling rather than in the keys.
//
// `which` selects the key rather than there being two near-identical functions,
// because the expensive half — parse_fen — is shared, and a caller wanting both
// should not parse twice.
enum class WhichKeys { Nn, Rep, Both };

py::array_t<std::uint64_t> key_batch(const py::object &fens, WhichKeys which) {
    py::list items = materialize(fens);
    const std::vector<std::string_view> views = borrow_utf8(items);

    const auto rows = static_cast<py::ssize_t>(views.size());
    const py::ssize_t cols = (which == WhichKeys::Both) ? 2 : 1;

    py::array_t<std::uint64_t> out({rows, cols});
    std::uint64_t *data = out.mutable_data();

    {
        // Touches no Python object: see borrow_utf8. `items` keeps every string
        // alive on this frame for the duration.
        py::gil_scoped_release released;
        for (std::size_t i = 0; i < views.size(); ++i) {
            const guofish::ParsedFen parsed = guofish::parse_fen(views[i]);
            std::uint64_t *row = data + static_cast<std::size_t>(cols) * i;
            switch (which) {
                case WhichKeys::Nn:
                    row[0] = guofish::nn_key(parsed).value;
                    break;
                case WhichKeys::Rep:
                    row[0] = guofish::rep_key(parsed).value;
                    break;
                case WhichKeys::Both:
                    row[0] = guofish::nn_key(parsed).value;
                    row[1] = guofish::rep_key(parsed).value;
                    break;
            }
        }
    }

    if (which == WhichKeys::Both) {
        return out;
    }
    // A [n, 1] column is an awkward thing to compare against a golden list;
    // reshape to [n]. This is a view, not a copy.
    return out.reshape({rows});
}

py::array_t<std::uint64_t> nn_keys(const py::object &fens) { return key_batch(fens, WhichKeys::Nn); }

py::array_t<std::uint64_t> rep_keys(const py::object &fens) { return key_batch(fens, WhichKeys::Rep); }

py::array_t<std::uint64_t> key_pairs(const py::object &fens) { return key_batch(fens, WhichKeys::Both); }

// What the compiler was able to prove about NNKey and RepKey while building this
// module. Every value is a constant folded at compile time from <type_traits>,
// not a runtime experiment — the C++ side already refuses to build if any of
// them is wrong (see the static_asserts in cpp/keys.hpp). Exposing them lets
// tests/test_c3_keys.py state acceptance criterion 3 as an assertion in the same
// place as the rest of the suite, instead of as a claim about a build nobody
// re-runs.
py::dict key_type_separation() {
    using guofish::NNKey;
    using guofish::RepKey;
    namespace gd = guofish::detail;

    py::dict d;
    d["nn_accepted_as_rep"] = std::is_invocable_v<decltype(gd::takes_rep_key), NNKey>;
    d["rep_accepted_as_nn"] = std::is_invocable_v<decltype(gd::takes_nn_key), RepKey>;
    d["nn_accepted_as_nn"] = std::is_invocable_v<decltype(gd::takes_nn_key), NNKey>;
    d["rep_accepted_as_rep"] = std::is_invocable_v<decltype(gd::takes_rep_key), RepKey>;
    d["nn_converts_to_rep"] = std::is_convertible_v<NNKey, RepKey>;
    d["rep_converts_to_nn"] = std::is_convertible_v<RepKey, NNKey>;
    d["nn_constructible_from_rep"] = std::is_constructible_v<NNKey, RepKey>;
    d["rep_constructible_from_nn"] = std::is_constructible_v<RepKey, NNKey>;
    d["nn_assignable_from_rep"] = std::is_assignable_v<NNKey &, RepKey>;
    d["rep_assignable_from_nn"] = std::is_assignable_v<RepKey &, NNKey>;
    d["uint64_converts_to_nn"] = std::is_convertible_v<std::uint64_t, NNKey>;
    d["uint64_converts_to_rep"] = std::is_convertible_v<std::uint64_t, RepKey>;
    d["nn_converts_to_uint64"] = std::is_convertible_v<NNKey, std::uint64_t>;
    d["rep_converts_to_uint64"] = std::is_convertible_v<RepKey, std::uint64_t>;
    d["nn_comparable_to_rep"] = gd::EqComparable<NNKey, RepKey>::value;
    d["rep_comparable_to_nn"] = gd::EqComparable<RepKey, NNKey>::value;
    d["nn_comparable_to_nn"] = gd::EqComparable<NNKey, NNKey>::value;
    d["rep_comparable_to_rep"] = gd::EqComparable<RepKey, RepKey>::value;
    d["nn_size"] = sizeof(NNKey);
    d["rep_size"] = sizeof(RepKey);
    return d;
}

// ---------------------------------------------------------------------------
// C4 — the SoA node arena
//
// The layout, the invariants and the two accumulators live in cpp/arena.hpp.
// This section is the Python surface tests/test_c4_arena.py drives: allocate,
// set and read every field, walk child ranges, and read back the two things
// that are otherwise only visible to the compiler — where each array actually
// landed in memory, and whether the atomics are lock-free.
//
// None of this is a search API. C5 will call NodeArena<> directly in C++; these
// bindings exist so that a structural property can be asserted in the same
// place as the rest of the suite instead of in a build log nobody re-reads.
// ---------------------------------------------------------------------------

// Reinterpreting a uint32 bit pattern as a float. std::memcpy rather than a
// reinterpret_cast or a union, because it is the one spelling that is not
// type-punning UB in C++17 and both compilers fold it to a register move.
float float_from_bits(std::uint32_t bits) {
    static_assert(sizeof(float) == sizeof(std::uint32_t), "unexpected float width");
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::string move_to_uci(std::uint16_t packed) {
    const int from = guofish::move_from(packed);
    const int to = guofish::move_to(packed);
    std::string uci;
    uci.reserve(5);
    uci.push_back(static_cast<char>('a' + (from & 7)));
    uci.push_back(static_cast<char>('1' + (from >> 3)));
    uci.push_back(static_cast<char>('a' + (to & 7)));
    uci.push_back(static_cast<char>('1' + (to >> 3)));
    const char promo = guofish::promotion_letter(guofish::move_promotion(packed));
    if (promo != '\0') {
        uci.push_back(promo);
    }
    return uci;
}

// The inverse of move_to_uci, for the C7 cache binding: a Python caller
// supplying a move list speaks UCI, and the cache stores packed moves.
//
// It validates rather than trusting, because the cache's move list is the thing
// that turns a key collision into a named failure — a silently mis-parsed move
// would defeat exactly the check it feeds. No board is consulted: this is a
// syntactic conversion, and the search's own path packs from a generated move.
std::uint16_t packed_from_uci(const std::string &uci) {
    const auto bad = [&uci](const char *why) {
        throw py::value_error("guofish_core: '" + uci + "' is not a UCI move (" + why + ")");
    };
    if (uci.size() != 4 && uci.size() != 5) {
        bad("expected 4 or 5 characters");
    }
    const auto square = [&](std::size_t i) {
        const char file = uci[i];
        const char rank = uci[i + 1];
        if (file < 'a' || file > 'h' || rank < '1' || rank > '8') {
            bad("square out of range");
        }
        return (rank - '1') * 8 + (file - 'a');
    };
    const int from = square(0);
    const int to = square(2);

    int promo = static_cast<int>(guofish::Promotion::None);
    if (uci.size() == 5) {
        switch (uci[4]) {
            case 'b': promo = static_cast<int>(guofish::Promotion::Bishop); break;
            case 'n': promo = static_cast<int>(guofish::Promotion::Knight); break;
            case 'q': promo = static_cast<int>(guofish::Promotion::Queen); break;
            case 'r': promo = static_cast<int>(guofish::Promotion::Rook); break;
            default: bad("promotion piece must be one of bnqr"); break;
        }
    }
    return guofish::pack_move(from, to, static_cast<guofish::Promotion>(promo));
}

std::uint16_t pack_move(int from, int to, int promo) {
    if (promo < 0 || promo >= static_cast<int>(guofish::kPromotionCount)) {
        throw py::value_error("guofish_core.pack_move: promotion code must be in [0, 5)");
    }
    return guofish::pack_move(from, to, static_cast<guofish::Promotion>(promo));
}

// Exhaustive float -> Q32 -> double round trip over the representable range.
//
// "Representable" means every IEEE-754 float bit pattern whose value lies in
// [-1, 1], which is 2,130,706,434 of them counting both signs — the domain a
// network value actually comes from. `stride` walks the positive bit patterns
// in steps; stride 1 is the exhaustive sweep and anything larger is a sample
// (the ASan build runs a strided sweep, since instrumentation makes the full
// one take minutes rather than seconds).
//
// Three separate things are counted, because they fail differently:
//
//   max_abs_error       the criterion. Must not exceed half the Q32 resolution,
//                       2^-33 = 1.164e-10, which is what "exact to 2.3e-10"
//                       means for a round-to-nearest quantizer.
//   inexact             floats that do not come back bit-identical. Every float
//                       at or above 2^-9 has an ulp of at least 2^-32 and is
//                       therefore *exactly* representable in Q32, so all of
//                       these must be smaller than that — reported as
//                       largest_inexact so the test can check the boundary
//                       rather than just the count.
//   code_mismatches     to_q32(from_q32(q)) != q for the codes this sweep
//                       produced. Must be zero: from_q32 multiplies by a power
//                       of two, which is exact in double for |q| <= 2^53, so
//                       the value handed back to to_q32 is the integer q
//                       itself. This is a proof rather than a measurement; the
//                       counter is here to catch the proof's premises breaking.
//
// Also checked: to_q32(-v) == -to_q32(v) for every sampled v. Round-half-away-
// from-zero is symmetric about zero, and an asymmetric quantizer would bias
// every backup in one direction by half a tick.
py::dict q32_roundtrip_sweep(std::uint32_t stride) {
    if (stride == 0) {
        throw py::value_error("guofish_core.q32_roundtrip_sweep: stride must be > 0");
    }

    // 0x3F800000 is 1.0f; bit patterns 0 .. 0x3F800000 are exactly the
    // non-negative floats in [0, 1], denormals included.
    constexpr std::uint32_t kOneBits = 0x3F800000u;

    double max_abs_error = 0.0;
    double max_error_at = 0.0;
    std::uint64_t examined = 0;
    std::uint64_t inexact = 0;
    std::uint64_t code_mismatches = 0;
    std::uint64_t asymmetric = 0;
    double largest_inexact = 0.0;

    {
        py::gil_scoped_release released;
        for (std::uint64_t b = 0; b <= kOneBits; b += stride) {
            const float f = float_from_bits(static_cast<std::uint32_t>(b));
            const double v = static_cast<double>(f);

            const std::int64_t q = guofish::to_q32(v);
            const double back = guofish::from_q32(q);
            const double err = back > v ? back - v : v - back;

            if (err > max_abs_error) {
                max_abs_error = err;
                max_error_at = v;
            }
            if (back != v) {
                ++inexact;
                if (v > largest_inexact) {
                    largest_inexact = v;
                }
            }
            if (guofish::to_q32(back) != q) {
                ++code_mismatches;
            }
            if (guofish::to_q32(-v) != -q) {
                ++asymmetric;
            }
            ++examined;
        }
    }

    py::dict d;
    d["stride"] = stride;
    d["exhaustive"] = (stride == 1);
    // Both signs: every sample above was checked at +v and -v.
    d["floats_examined"] = examined * 2;
    d["max_abs_error"] = max_abs_error;
    d["max_error_at"] = max_error_at;
    d["inexact"] = inexact * 2;
    d["largest_inexact"] = largest_inexact;
    d["code_mismatches"] = code_mismatches;
    d["asymmetric"] = asymmetric;
    d["resolution"] = guofish::kQ32Resolution;
    d["half_resolution"] = guofish::kQ32Resolution * 0.5;
    return d;
}

// The other direction, over Q32 codes rather than floats: q -> double -> q.
//
// This one cannot be exhaustive — [-2^32, 2^32] is 8.6e9 codes — so it walks
// the range at `stride` and additionally pins every boundary code by hand. See
// the note in q32_roundtrip_sweep about why the general case is a consequence
// of the scale being a power of two rather than something a sweep establishes.
py::dict q32_code_sweep(std::uint64_t stride) {
    if (stride == 0) {
        throw py::value_error("guofish_core.q32_code_sweep: stride must be > 0");
    }

    std::uint64_t examined = 0;
    std::uint64_t mismatches = 0;
    std::int64_t first_mismatch = 0;

    {
        py::gil_scoped_release released;
        for (std::int64_t q = -guofish::kQ32One; q <= guofish::kQ32One;
             q += static_cast<std::int64_t>(stride)) {
            const double v = guofish::from_q32(q);
            if (guofish::to_q32(v) != q) {
                if (mismatches == 0) {
                    first_mismatch = q;
                }
                ++mismatches;
            }
            ++examined;
        }

        // The edges, unconditionally: +-1.0, +-1 tick, 0, and the codes either
        // side of them. A stride that steps over the extremes would otherwise
        // certify the interior of a range whose ends are where clamping and
        // saturation bugs live.
        const std::int64_t edges[] = {
            -guofish::kQ32One, -guofish::kQ32One + 1, -2, -1, 0, 1, 2,
            guofish::kQ32One - 1, guofish::kQ32One,
        };
        for (std::int64_t q : edges) {
            const double v = guofish::from_q32(q);
            if (guofish::to_q32(v) != q) {
                if (mismatches == 0) {
                    first_mismatch = q;
                }
                ++mismatches;
            }
            ++examined;
        }
    }

    py::dict d;
    d["stride"] = stride;
    d["codes_examined"] = examined;
    d["mismatches"] = mismatches;
    d["first_mismatch"] = first_mismatch;
    d["q32_one"] = guofish::kQ32One;
    return d;
}

// What the compiler proved about the arena's atomics while building this
// module, plus the layout constants the SoA design is judged on. Every value is
// folded at compile time; the build already fails if any is_always_lock_free
// entry is false (see the static_asserts in cpp/arena.hpp).
py::dict arena_layout() {
    py::dict d;
    d["int32_always_lock_free"] = std::atomic<std::int32_t>::is_always_lock_free;
    d["int64_always_lock_free"] = std::atomic<std::int64_t>::is_always_lock_free;
    d["uint8_always_lock_free"] = std::atomic<std::uint8_t>::is_always_lock_free;
    d["uint32_always_lock_free"] = std::atomic<std::uint32_t>::is_always_lock_free;
    d["double_always_lock_free"] = std::atomic<double>::is_always_lock_free;

    d["sizeof_atomic_int32"] = sizeof(std::atomic<std::int32_t>);
    d["sizeof_atomic_int64"] = sizeof(std::atomic<std::int64_t>);
    d["sizeof_atomic_double"] = sizeof(std::atomic<double>);
    d["sizeof_atomic_uint8"] = sizeof(std::atomic<std::uint8_t>);
    d["alignof_atomic_int64"] = alignof(std::atomic<std::int64_t>);
    d["alignof_atomic_double"] = alignof(std::atomic<double>);

    d["bytes_per_node_q32"] = guofish::Q32Arena::bytes_per_node();
    d["bytes_per_node_double"] = guofish::DoubleArena::bytes_per_node();
    d["cache_line"] = guofish::kCacheLine;
    d["default_accumulator"] = std::string(guofish::kDefaultAccumulator);
    return d;
}

// The sibling scan, under both accumulators, over the same logical tree.
//
// This is the C4 design decision the chunk brief calls out: the Q32 -> double
// conversion sits in the hottest read path in the engine, and if it does not
// vanish into the loop the layout has to change now rather than in C12. The
// loop reads exactly what PUCT selection reads — visit_count, value_sum and
// prior for a contiguous block of siblings — and reduces them to an argmax so
// the reads are observable and cannot be optimised away.
//
// It is deliberately NOT the PUCT formula (out of scope for this chunk): there
// is no cpuct, no sqrt(parent visits) and no virtual loss. That makes the
// per-child arithmetic *cheaper* than the real thing, so the conversion's share
// of the loop is overstated here rather than flattered — the conservative
// direction for a decision about whether it is affordable.
template <class Arena>
double scan_arena(const Arena &arena, std::size_t blocks, std::size_t block_size,
                  std::size_t repeats, std::uint64_t &sink) {
    using Acc = typename Arena::accumulator_type;

    const std::atomic<std::int32_t> *visits = arena.visit_count_data();
    const auto *values = arena.value_sum_data();
    const float *priors = arena.prior_data();

    std::uint64_t accumulated = 0;
    const auto start = clock_type::now();

    for (std::size_t r = 0; r < repeats; ++r) {
        for (std::size_t b = 0; b < blocks; ++b) {
            const std::size_t base = b * block_size;
            double best = -std::numeric_limits<double>::infinity();
            std::size_t best_child = 0;

            for (std::size_t k = 0; k < block_size; ++k) {
                const std::int32_t n = visits[base + k].load(std::memory_order_relaxed);
                const double sum = Acc::decode(values[base + k].load(std::memory_order_relaxed));
                const double q = (n > 0) ? sum / static_cast<double>(n) : 0.0;
                const double score = q + static_cast<double>(priors[base + k]);
                if (score > best) {
                    best = score;
                    best_child = k;
                }
            }
            accumulated += best_child;
        }
    }

    const double elapsed_us = micros_between(start, clock_type::now());
    sink += accumulated;
    return elapsed_us;
}

// Deterministic filler so both arenas hold identical logical contents and the
// two timings are comparable. A hand-rolled LCG rather than <random> because
// the sequence has to be identical on MSVC and Clang, and the standard pins the
// engines but not the distributions.
template <class Arena>
void fill_arena(Arena &arena, std::size_t nodes) {
    const std::uint32_t first = arena.allocate(nodes);
    assert(first == 0);
    (void)first;

    std::uint64_t rng = 0x9E3779B97F4A7C15ull;
    for (std::size_t i = 0; i < nodes; ++i) {
        rng = rng * 6364136223846793005ull + 1442695040888963407ull;
        const auto bits = static_cast<std::uint32_t>(rng >> 33);

        // Visit counts spanning the range a real search produces, values in
        // [-1, 1], priors in [0, 1).
        const std::int32_t n = static_cast<std::int32_t>(bits % 4096);
        const double v = static_cast<double>(bits % 2001) / 1000.0 - 1.0;
        const float p = static_cast<float>(bits % 1024) / 1024.0f;

        const auto idx = static_cast<std::uint32_t>(i);
        arena.add_visits(idx, n);
        arena.add_value(idx, v);
        arena.set_prior(idx, p);
    }
}

py::dict sibling_scan_bench(std::size_t blocks, std::size_t block_size, std::size_t repeats) {
    if (blocks == 0 || block_size == 0 || repeats == 0) {
        throw py::value_error(
            "guofish_core.sibling_scan_bench: blocks, block_size and repeats must all be > 0");
    }
    const std::size_t nodes = blocks * block_size;
    if (nodes / blocks != block_size) {
        throw py::value_error("guofish_core.sibling_scan_bench: blocks * block_size overflows");
    }

    guofish::Q32Arena q32(nodes);
    guofish::DoubleArena f64(nodes);
    fill_arena(q32, nodes);
    fill_arena(f64, nodes);

    std::uint64_t sink = 0;
    double q32_us = 0.0;
    double f64_us = 0.0;

    {
        py::gil_scoped_release released;
        // One untimed pass each so the comparison is not a measurement of which
        // arena happened to be paged in first.
        std::uint64_t warm = 0;
        scan_arena(q32, blocks, block_size, 1, warm);
        scan_arena(f64, blocks, block_size, 1, warm);
        sink += warm;

        q32_us = scan_arena(q32, blocks, block_size, repeats, sink);
        f64_us = scan_arena(f64, blocks, block_size, repeats, sink);
    }

    const auto scans = static_cast<double>(blocks) * static_cast<double>(repeats);
    const double children = scans * static_cast<double>(block_size);

    py::dict d;
    d["blocks"] = blocks;
    d["block_size"] = block_size;
    d["repeats"] = repeats;
    d["nodes"] = nodes;
    d["scans"] = static_cast<std::uint64_t>(scans);
    // Bytes of the three scanned fields per node; what the working set is made
    // of, and the number that decides whether this run is L1-resident.
    d["hot_bytes_per_node_q32"] =
        sizeof(std::atomic<std::int32_t>) + sizeof(std::atomic<std::int64_t>) + sizeof(float);
    d["hot_bytes_per_node_double"] =
        sizeof(std::atomic<std::int32_t>) + sizeof(std::atomic<double>) + sizeof(float);
    d["q32_ns_per_scan"] = q32_us * 1000.0 / scans;
    d["double_ns_per_scan"] = f64_us * 1000.0 / scans;
    d["q32_ns_per_child"] = q32_us * 1000.0 / children;
    d["double_ns_per_child"] = f64_us * 1000.0 / children;
    d["q32_total_s"] = q32_us / 1e6;
    d["double_total_s"] = f64_us / 1e6;
    d["q32_over_double"] = f64_us > 0.0 ? q32_us / f64_us : 0.0;
    d["checksum"] = sink;
    return d;
}

// Bind one NodeArena instantiation. Called twice, so the Q32 and double arenas
// have identical Python surfaces and a test can drive both through the same
// code path — which is what "both accumulator builds are exercised by tests"
// requires, and what an #ifdef alone would not give.
template <class Acc>
void bind_arena(py::module_ &m, const char *name, const char *doc) {
    using Arena = guofish::NodeArena<Acc>;

    py::class_<Arena>(m, name, doc)
        .def(py::init<std::size_t>(), py::arg("capacity"))
        .def_property_readonly("capacity", &Arena::capacity, "Nodes this arena can ever hold.")
        .def_property_readonly("size", &Arena::size, "Nodes allocated so far.")
        .def_property_readonly_static(
            "accumulator", [](const py::object &) { return std::string(Acc::name()); },
            "\"q32\" or \"double\" — which representation value_sum is stored in.")
        .def_property_readonly_static(
            "bytes_per_node", [](const py::object &) { return Arena::bytes_per_node(); },
            "Sum of the per-node field widths. There is no node struct; this is what one "
            "would cost if there were.")

        .def("allocate", &Arena::allocate, py::arg("count"),
             "Claim `count` consecutive nodes and return the index of the first. The block is "
             "reset to a clean unexpanded state first. Raises RuntimeError if the arena is "
             "exhausted.")
        .def(
            "try_allocate",
            [](Arena &self, std::size_t count) -> py::object {
                const std::uint32_t first = self.try_allocate(count);
                if (first == guofish::kNoNode) {
                    return py::none();
                }
                return py::cast(first);
            },
            py::arg("count"),
            "As allocate(), but returns None instead of raising when the arena is full. This "
            "is the form the search uses.")
        .def("reset", &Arena::reset,
             "Drop every allocation. O(1) — the storage is cleared by allocate(), not here.")

        .def("visit_count", &Arena::visit_count, py::arg("index"))
        .def("value_sum", &Arena::value_sum, py::arg("index"),
             "value_sum as a double, decoded from whatever the accumulator stores.")
        .def("value_sum_raw", &Arena::value_sum_raw, py::arg("index"),
             "value_sum in the accumulator's own representation: the Q32 integer under the "
             "fixed-point accumulator, the same double under the floating-point one.")
        .def("vloss_count", &Arena::vloss_count, py::arg("index"))
        .def("prior", &Arena::prior, py::arg("index"))
        .def("move", &Arena::move, py::arg("index"))
        .def("children_offset", &Arena::children_offset, py::arg("index"))
        .def("children_count", &Arena::children_count, py::arg("index"))
        .def("terminal_value", &Arena::terminal_value, py::arg("index"))

        .def("state", &Arena::state_bits, py::arg("index"),
             "The raw state byte: lifecycle in the low two bits, TERMINAL_BIT in the high one.")
        .def(
            "lifecycle",
            [](const Arena &self, std::uint32_t i) {
                return static_cast<std::uint8_t>(self.lifecycle(i));
            },
            py::arg("index"), "STATE_UNEXPANDED, STATE_PENDING or STATE_EXPANDED.")
        .def("is_terminal", &Arena::is_terminal, py::arg("index"))
        .def(
            "is_expanded",
            [](const Arena &self, std::uint32_t i) {
                return self.lifecycle(i) == guofish::NodeState::Expanded;
            },
            py::arg("index"))

        .def("set_prior", &Arena::set_prior, py::arg("index"), py::arg("prior"))
        .def("set_move", &Arena::set_move, py::arg("index"), py::arg("packed_move"),
             "Store a packed move. The field is a raw uint16; use pack_move() to build one.")
        .def("add_visits", &Arena::add_visits, py::arg("index"), py::arg("delta"))
        .def("add_value", &Arena::add_value, py::arg("index"), py::arg("value"),
             "Add a value in [-1, 1] to the accumulator: one lock xadd under Q32, a CAS loop "
             "under double.")
        .def("add_value_raw", &Arena::add_value_raw, py::arg("index"), py::arg("raw"),
             "Add a quantity already in the accumulator's representation, so a test can put an "
             "exact Q32 integer in without going through the conversion it is testing.")
        .def("add_vloss", &Arena::add_vloss, py::arg("index"), py::arg("delta"))

        .def("try_claim_pending", &Arena::try_claim_pending, py::arg("index"),
             "CAS UNEXPANDED -> PENDING. False means another thread claimed this leaf first.")
        .def("release_pending", &Arena::release_pending, py::arg("index"),
             "PENDING -> UNEXPANDED. Raises RuntimeError if the node is not pending.")
        .def("set_children", &Arena::set_children, py::arg("index"), py::arg("offset"),
             py::arg("count"),
             "Publish a child range and mark the node expanded. Raises ValueError on a count of "
             "zero — an expanded node with no children is the bestmove 0000 defect and is not "
             "representable here — IndexError if the range leaves the allocated region, and "
             "RuntimeError if the node is terminal or already expanded.")
        .def("mark_terminal", &Arena::mark_terminal, py::arg("index"), py::arg("value"),
             "Set the TERMINAL bit and record the game result. Raises RuntimeError if the node "
             "already has children.")

        .def("child", &Arena::child, py::arg("index"), py::arg("k"),
             "The arena index of the k-th child. Raises IndexError past children_count.")
        .def(
            "children",
            [](const Arena &self, std::uint32_t i) {
                return py::make_tuple(self.children_offset(i), self.children_count(i));
            },
            py::arg("index"), "(children_offset, children_count).")
        .def(
            "children_indices",
            [](const Arena &self, std::uint32_t i) {
                const std::uint16_t count = self.children_count(i);
                py::list out;
                for (std::uint16_t k = 0; k < count; ++k) {
                    out.append(self.child(i, k));
                }
                return out;
            },
            py::arg("index"), "Every child index, in order. Empty for an unexpanded node.")

        .def(
            "array_info",
            [](const Arena &self) {
                py::list out;
                self.for_each_array([&out](const typename Arena::ArrayInfo &info) {
                    py::dict d;
                    d["field"] = std::string(info.field);
                    d["element_type"] = std::string(info.element_type);
                    d["address"] = info.address;
                    d["element_size"] = info.element_size;
                    d["element_align"] = info.element_align;
                    d["requested_align"] = info.requested_align;
                    d["naturally_aligned"] = info.naturally_aligned;
                    d["cache_line_aligned"] = info.cache_line_aligned;
                    out.append(d);
                });
                return out;
            },
            "One entry per SoA array, in layout order: where it landed in memory and whether "
            "that address satisfies its element's alignment. The runtime half of the guarantee "
            "the static_asserts start — and unlike assert(), this is live in a Release build.");
    // Deliberately no per-object `is_lock_free()` accessor. It would report
    // nothing new — the standard says is_always_lock_free is true only if every
    // object of the type is lock-free, and the static_asserts already require
    // that — and on libstdc++ it is an out-of-line call to
    // __atomic_is_lock_free, which lives in libatomic and is not linked by
    // default. The Clang build failed to import with an undefined symbol. See
    // DECISIONS.md; the property is covered by arena_layout() plus the
    // alignment entries above.
}

// ---------------------------------------------------------------------------
// C5 — the search core on a replay evaluator
//
// The search itself, the raw-en-passant board state and the replay dump live in
// cpp/search.hpp. This section is the surface tests/test_c5_gate1_quiet.py
// drives, plus two probes that exist to pin things the search silently depends
// on: `c_puct` (so a libm difference cannot enter the equivalence path) and
// `ep_pin_probe` (so a chess-library re-pin fails a test rather than a parity
// run).
// ---------------------------------------------------------------------------

// Find the legal move in `board` whose NORMALISED UCI is `uci`.
//
// Normalised: C1's castling rewrite is applied before comparing, so "e1g1"
// matches the king-takes-rook move chess-library actually generated. Matching
// on the raw encoding instead would make every castling probe fail with
// "no such legal move", which reads as a bad test rather than as the encoding
// mismatch it is.
chess::Move find_legal_move(const chess::Board &board, std::string_view uci) {
    chess::Movelist movelist;
    chess::movegen::legalmoves(movelist, board);
    for (const auto &move : movelist) {
        if (guofish::move_to_uci(board, move) == uci) {
            return move;
        }
    }
    throw py::value_error("guofish_core: no legal move " + std::string(uci) + " in this position");
}

std::string square_or_dash(int square) {
    if (square < 0) {
        return "-";
    }
    std::string out;
    out += static_cast<char>('a' + (square & 7));
    out += static_cast<char>('1' + (square >> 3));
    return out;
}

int square_index_or_none(chess::Square square) {
    return (square == chess::Square::NO_SQ) ? -1 : square.index();
}

// THE PIN TEST'S DATA SOURCE.
//
// Reports, for one (position, move) pair, what each of the four en-passant
// conventions in play says. The point is not that any one of them is right —
// each is right for its own consumer — but that they DIFFER, and that the
// search must therefore carry its own. tests/test_c5_ep_pin.py asserts the
// specific values at chess-library revision 53e6a84, so an upstream re-pin that
// changes `makeMove`'s behaviour fails a named test instead of quietly moving
// token 66 on ~3% of positions.
py::dict ep_pin_probe(std::string_view fen, std::string_view uci) {
    py::dict d;

    // What the FEN literally says, via C2's own parser.
    const guofish::ParsedFen parsed = guofish::parse_fen(fen);
    d["fen_ep"] = square_or_dash(parsed.ep_square);

    // What chess-library kept after setFen(), i.e. after isEpSquareValid().
    chess::Board board;
    guofish::set_fen_or_throw(board, fen);
    d["library_ep_after_setfen"] = square_or_dash(square_index_or_none(board.enpassantSq()));

    // Each block below re-finds the move on its own board. A `chess::Move`
    // carries square indices, not a board reference, so one lookup would do —
    // but the three boards are deliberately independent, and sharing a move
    // between them would make it easy to later change one board's FEN and have
    // the others silently keep playing the old position's move.

    // makeMove<false> — the default. Sets ep on PSEUDO-LEGAL adjacency: an
    // enemy pawn merely stands beside the pushed pawn.
    chess::Board pseudo;
    guofish::set_fen_or_throw(pseudo, fen);
    const chess::Move pseudo_move = find_legal_move(pseudo, uci);
    pseudo.makeMove(pseudo_move);
    d["library_ep_default"] = square_or_dash(square_index_or_none(pseudo.enpassantSq()));
    pseudo.unmakeMove(pseudo_move);
    d["library_ep_unmade"] = square_or_dash(square_index_or_none(pseudo.enpassantSq()));

    // makeMove<true> — the EXACT template. Additionally requires the capture to
    // be legal.
    chess::Board exact;
    guofish::set_fen_or_throw(exact, fen);
    const chess::Move exact_move = find_legal_move(exact, uci);
    exact.makeMove<true>(exact_move);
    d["library_ep_exact"] = square_or_dash(square_index_or_none(exact.enpassantSq()));

    // And what the search carries: the RAW rule, derived from the move.
    guofish::SearchBoard search_board;
    search_board.set_fen(fen);
    const chess::Move raw_move = find_legal_move(search_board.board(), uci);
    d["raw_ep_before"] = square_or_dash(search_board.raw_ep());
    search_board.make_move(raw_move, guofish::is_zeroing(parsed, raw_move.from().index(),
                                                        guofish::uci_destination(
                                                            search_board.board(), raw_move).index()));
    d["raw_ep"] = square_or_dash(search_board.raw_ep());
    // Token 66 is the only thing that consumes it, so report that too: this is
    // the number the network actually sees and the number nn_key hashes.
    std::int32_t tokens[guofish::kSeqLength];
    guofish::tokenize_into(search_board.parsed(), tokens);
    d["token_66"] = tokens[guofish::kIdxEnPassant];
    d["nn_key"] = guofish::nn_key(search_board.parsed()).value;
    search_board.unmake_move(raw_move);
    d["raw_ep_unmade"] = square_or_dash(search_board.raw_ep());

    return d;
}

// ---------------------------------------------------------------------------
// C6 — the rules that end a game, exposed one at a time
//
// The Gate 1 comparison is end-to-end: two trees, node for node. That is the
// acceptance criterion and it is the right one, but it is a poor DIAGNOSTIC —
// `has_insufficient_material` reading `self.bishops` instead of this colour's
// bishops shows up there as "the trees have different node counts at ply 14".
//
// So each predicate is also bound on its own, over a FEN, and
// tests/test_c6_gate1_full.py diffs them against python-chess across the
// existing C1 movegen corpus. A transcription error then fails as
// "insufficient_material disagrees on <fen>" before the tree comparison ever
// runs. Nothing in the search calls these; they are the same inline functions
// the search calls, reached through a FEN instead of through search state.
// ---------------------------------------------------------------------------

// python-chess's `Board.outcome()` reason, as a lowercase string, for a position
// given by FEN alone.
//
// FIVEFOLD REPETITION IS NOT ASKED, and cannot be: it is a property of the game
// leading to the position, which a FEN does not carry. The search evaluates it
// against its own path (ReplaySearch::is_repetition); here it is passed as
// False, which is what python-chess itself answers for a board built from a FEN
// with no move stack. Same for the seventy-five-move rule's dependence on the
// clock — that one IS in the FEN, and is read from it.
std::string terminal_reason(std::string_view fen) {
    const guofish::ParsedFen parsed = guofish::parse_fen(fen);
    chess::Board board;
    guofish::set_fen_or_throw(board, fen);

    chess::Movelist movelist;
    chess::movegen::legalmoves(movelist, board);

    return guofish::terminal_reason_name(
        guofish::outcome_of(parsed, board.inCheck(), static_cast<std::size_t>(movelist.size()),
                            static_cast<int>(board.halfMoveClock()),
                            /*fivefold_repetition=*/false));
}

bool insufficient_material(std::string_view fen) {
    return guofish::is_insufficient_material(guofish::parse_fen(fen).placement);
}

// `Board.is_zeroing(move)` and `Board.is_irreversible(move)`, on the NORMALISED
// UCI destination — g1/c1 for castling, which is the square python-chess indexes
// its bitboards with. Feeding chess-library's king-takes-rook `to()` here would
// make `_reduces_castling_rights` compare the rook square against the castling
// rights, which are the rook squares, and every castle would read as
// irreversible for the wrong reason.
py::dict move_rule_probe(std::string_view fen, std::string_view uci) {
    const guofish::ParsedFen parsed = guofish::parse_fen(fen);
    chess::Board board;
    guofish::set_fen_or_throw(board, fen);
    const chess::Move move = find_legal_move(board, uci);

    const int from = move.from().index();
    const int to = guofish::uci_destination(board, move).index();

    py::dict d;
    d["zeroing"] = guofish::is_zeroing(parsed, from, to);
    d["reduces_castling_rights"] = guofish::reduces_castling_rights(parsed, from, to);
    d["has_legal_en_passant"] = guofish::has_legal_en_passant(parsed);
    d["irreversible"] = guofish::is_irreversible(parsed, from, to);
    return d;
}

// ---------------------------------------------------------------------------
// C7 — the transposition cache and the tablebase backend
// ---------------------------------------------------------------------------

// The compile-time facts of acceptance criterion 5, as data a test can assert
// on. Same shape and same reasoning as C3's `key_type_separation`: a
// `static_assert` that fires stops the build, which is the right behaviour but
// gives the acceptance test nothing to point at, so the predicates are also
// evaluated here and reported.
//
// Every value below is computed by the compiler from the REAL
// `TranspositionCache::insert` through real overload resolution — see
// guofish::detail::CacheInsertAccepts. Nothing here restates a signature.
py::dict cache_type_separation() {
    namespace gd = guofish::detail;
    using guofish::NetworkValue;
    using guofish::ProofValue;
    using guofish::TablebaseValue;
    using guofish::TerminalValue;

    py::dict d;
    // The one that must be true: the cache holds the network's output.
    d["network_value_accepted"] = gd::CacheInsertAccepts<NetworkValue>::value;

    // The ones that must be false. Each is a distinct way the historical defect
    // gets in, and each fails to COMPILE rather than failing at runtime.
    d["terminal_value_accepted"] = gd::CacheInsertAccepts<TerminalValue>::value;
    d["tablebase_value_accepted"] = gd::CacheInsertAccepts<TablebaseValue>::value;
    d["proof_value_accepted"] = gd::CacheInsertAccepts<ProofValue>::value;
    d["bare_double_accepted"] = gd::CacheInsertAccepts<double>::value;
    d["terminal_flag_accepted"] = gd::CacheInsertAccepts<bool>::value;
    d["float_accepted"] = gd::CacheInsertAccepts<float>::value;
    d["int_accepted"] = gd::CacheInsertAccepts<int>::value;

    // The conversions that would route around the above if they existed.
    d["terminal_converts_to_network"] = std::is_convertible_v<TerminalValue, NetworkValue>;
    d["tablebase_converts_to_network"] = std::is_convertible_v<TablebaseValue, NetworkValue>;
    d["proof_converts_to_network"] = std::is_convertible_v<ProofValue, NetworkValue>;
    d["double_converts_to_network"] = std::is_convertible_v<double, NetworkValue>;
    d["network_converts_to_double"] = std::is_convertible_v<guofish::NetworkValue, double>;
    d["network_constructible_from_tablebase"] =
        std::is_constructible_v<NetworkValue, TablebaseValue>;
    d["network_constructible_from_terminal"] =
        std::is_constructible_v<NetworkValue, TerminalValue>;

    // The payload itself is closed too, so a caller who somehow held a Slot
    // could not write a forbidden value into it either.
    d["payload_value_assignable_from_tablebase"] =
        std::is_assignable_v<decltype(guofish::CachedEval::value) &, TablebaseValue>;
    d["payload_value_assignable_from_terminal"] =
        std::is_assignable_v<decltype(guofish::CachedEval::value) &, TerminalValue>;
    d["payload_value_assignable_from_double"] =
        std::is_assignable_v<decltype(guofish::CachedEval::value) &, double>;
    d["payload_value_is_network_value"] =
        std::is_same_v<decltype(guofish::CachedEval::value), NetworkValue>;

    // C3's design, which the empty-slot sentinel rests on.
    d["nn_key_default_constructible"] = std::is_default_constructible_v<guofish::NNKey>;
    d["network_value_default_constructible"] = std::is_default_constructible_v<NetworkValue>;
    return d;
}

// The evaluator's input row and the key derived from it, as a pair a test can
// compare. This is the surface for "the stored nn_key is computed from the EXACT
// same token buffer row dispatched to the evaluator": the caller gets the tokens
// and the key from ONE object, so it can check that the key really is a function
// of those bytes and of nothing else.
py::dict eval_row(std::string_view fen) {
    const guofish::EvalRow row(guofish::parse_fen(fen));

    py::array_t<std::int32_t> tokens(guofish::kSeqLength);
    std::copy(row.tokens(), row.tokens() + guofish::kSeqLength, tokens.mutable_data());

    py::dict d;
    d["tokens"] = tokens;
    d["nn_key"] = row.key().value;
    return d;
}

// The key a token row hashes to, for a row the caller supplies. Exists so a test
// can perturb ONE token and watch the key move — which is what makes "the key is
// a function of the tokens" checkable rather than assertable.
std::uint64_t nn_key_of_tokens(
    py::array_t<std::int32_t, py::array::c_style | py::array::forcecast> tokens) {
    if (tokens.size() != guofish::kSeqLength) {
        throw py::value_error("guofish_core.nn_key_of_tokens: expected " +
                              std::to_string(guofish::kSeqLength) + " tokens, got " +
                              std::to_string(tokens.size()));
    }
    return guofish::nn_key_of_tokens(tokens.data()).value;
}

// A TablebaseProber backed by a Python callable.
//
// THIS IS WHAT MAKES MODES 1 AND 2 TESTABLE WITHOUT VENDORING FATHOM. The
// callable is handed a FEN and returns (wdl, dtz) or None; the reference's own
// `chess.syzygy.Tablebase` over assets/syzygy is exactly such a callable, so the
// C++ override values, the perspective conversion, the piece-count gate and —
// the thing that matters — the fact that no probed value ever reaches the cache
// are all exercised against the REAL 5-piece tables.
//
// The FEN it passes carries the search's own RAW en-passant square and its own
// halfmove clock (guofish::fen_of), so the position python-chess reconstructs is
// the position the search is standing on, not chess-library's filtered view of
// it. That is the same discipline C5 established and the reason this takes a
// ParsedFen rather than a chess::Board.
//
// It calls into Python, so it requires the GIL. Every caller today holds it (the
// search bindings do not release it). C9 introduces threads that will not, and a
// Fathom backend — which touches no Python — is the answer there rather than a
// GIL acquisition in a leaf probe. Noted here because it is the one thing about
// this class that does not generalise.
class PythonProber final : public guofish::TablebaseProber {
public:
    explicit PythonProber(py::function probe) : probe_(std::move(probe)) {}

    std::optional<int> probe_wdl(const guofish::ParsedFen &parsed,
                                 int halfmove_clock) const override {
        return field(parsed, halfmove_clock, 0);
    }

    std::optional<int> probe_dtz(const guofish::ParsedFen &parsed,
                                 int halfmove_clock) const override {
        return field(parsed, halfmove_clock, 1);
    }

    std::string backend() const override { return "python"; }

    std::int64_t calls() const noexcept { return calls_; }

private:
    std::optional<int> field(const guofish::ParsedFen &parsed, int halfmove_clock,
                             int index) const {
        ++calls_;
        // Fullmove 1: no rule this file implements reads it, and python-chess
        // does not use it for a Syzygy probe. The halfmove clock is real and is
        // passed through, because a fifty-move-aware backend needs it.
        const std::string fen = guofish::fen_of(parsed, halfmove_clock, 1);
        const py::object result = probe_(fen);
        if (result.is_none()) {
            return std::nullopt;
        }
        const py::sequence pair = result.cast<py::sequence>();
        if (py::len(pair) != 2) {
            throw py::value_error(
                "guofish_core: a tablebase probe callback must return (wdl, dtz) or None");
        }
        return pair[static_cast<py::size_t>(index)].cast<int>();
    }

    py::function probe_;
    mutable std::int64_t calls_ = 0;
};

// ---------------------------------------------------------------------------
// C9 — the parallel search's Python surface
// ---------------------------------------------------------------------------

// A numpy int64 view of a vector, for the histograms scope §6.2 asks for.
// Percentiles are computed in Python from these rather than in C++, because the
// C9 brief asks for a histogram and not a mean: the tail is the entire question.
py::array_t<std::int64_t> int64_array(const std::vector<std::int64_t> &values) {
    py::array_t<std::int64_t> out(static_cast<py::ssize_t>(values.size()));
    std::copy(values.begin(), values.end(), out.mutable_data());
    return out;
}

py::dict parallel_stats_dict(const guofish::ParallelStats &stats) {
    py::dict d;
    d["requested"] = stats.requested;
    d["delivered"] = stats.delivered;
    d["queued_leaves"] = stats.queued_leaves;
    d["worker_terminals"] = stats.worker_terminals;
    d["select_collisions"] = stats.select_collisions;
    d["batches"] = stats.batches;
    d["largest_batch"] = stats.largest_batch;
    d["worker_waits"] = stats.worker_waits;
    d["wall_ns"] = stats.wall_ns;
    d["workers"] = stats.workers;
    d["in_flight"] = stats.in_flight;
    d["max_outstanding"] = stats.max_outstanding;
    d["affinity"] = stats.affinity;
    d["pinned_cpus"] = stats.pinned_cpus;
    d["batch_sizes"] = int64_array(stats.batch_sizes);
    d["outstanding_at_drain"] = int64_array(stats.outstanding_at_drain);
    d["hook_wait_ns"] = stats.hook_wait_ns;
    d["hook_wait_ns_samples"] = int64_array(stats.hook_wait_ns_samples);
    // C11c. The two legitimate reasons `delivered < requested`, and the arena
    // occupancy a sizing report checks the formula against. A benchmark harness
    // that admits a row only on `delivered == requested` must consult these
    // before rejecting the row as broken — and must still reject it, rather than
    // publish a short one. See ParallelStats in cpp/search.hpp.
    d["deadline_hit"] = stats.deadline_hit;
    d["arena_exhausted"] = stats.arena_exhausted;
    d["arena_exhausted_at"] = stats.arena_exhausted_at;
    d["arena_high_water"] = stats.arena_high_water;
    d["arena_capacity"] = stats.arena_capacity;
    d["arena_utilisation"] =
        stats.arena_capacity > 0 ? static_cast<double>(stats.arena_high_water) /
                                       static_cast<double>(stats.arena_capacity)
                                 : 0.0;
    d["mean_batch"] = stats.batches > 0
                          ? static_cast<double>(stats.queued_leaves) /
                                static_cast<double>(stats.batches)
                          : 0.0;
    d["sims_per_s"] = stats.wall_ns > 0 ? static_cast<double>(stats.delivered) * 1e9 /
                                              static_cast<double>(stats.wall_ns)
                                        : 0.0;
    return d;
}

// C10. The evaluation boundary's own counters, for the LAST search only.
//
// `acquire_wait_ns_samples` is the whole point and is returned raw: the C10
// trigger is a p99 and a share of wall time, and both are properties of the
// distribution. A mean is offered as a convenience and is the number NOT to
// make a decision on — see EvalStats in cpp/search.hpp.
py::dict eval_stats_dict(const guofish::EvalStats &stats) {
    py::dict d;
    d["batches"] = stats.batches;
    d["rows"] = stats.rows;
    d["cache_skipped"] = stats.cache_skipped;
    d["acquire_wait_ns"] = stats.acquire_wait_ns;
    d["call_ns"] = stats.call_ns;
    d["max_acquire_wait_ns"] = stats.max_acquire_wait_ns;
    d["acquire_wait_ns_samples"] = int64_array(stats.acquire_wait_ns_samples);
    d["call_ns_samples"] = int64_array(stats.call_ns_samples);
    d["mean_rows"] = stats.batches > 0 ? static_cast<double>(stats.rows) /
                                             static_cast<double>(stats.batches)
                                       : 0.0;
    return d;
}

py::dict tree_audit_dict(const guofish::TreeAudit &audit) {
    py::dict d;
    d["nodes"] = audit.nodes;
    d["expanded_nodes"] = audit.expanded_nodes;
    d["visited_nodes"] = audit.visited_nodes;
    d["vloss_total"] = audit.vloss_total;
    d["vloss_nonzero_nodes"] = audit.vloss_nonzero_nodes;
    d["conservation_failures"] = audit.conservation_failures;
    d["first_bad_node"] = audit.first_bad_node;
    d["first_bad_expected"] = audit.first_bad_expected;
    d["first_bad_actual"] = audit.first_bad_actual;
    d["root_visits"] = audit.root_visits;
    d["root_children_visits"] = audit.root_children_visits;
    return d;
}

py::dict topology_dict(const guofish::Topology &topo) {
    py::dict d;
    d["pcore_physical"] = topo.pcore_physical;
    d["pcore_all"] = topo.pcore_all;
    d["ecore_all"] = topo.ecore_all;
    d["all_logical"] = topo.all_logical;
    d["hybrid"] = topo.hybrid;
    d["source"] = topo.source;
    d["hardware_concurrency"] = static_cast<int>(std::thread::hardware_concurrency());
    return d;
}

// THE GIL-ACQUISITION PROBE, and it is a prediction test rather than a
// measurement for its own sake.
//
// C0b established that the pybind11 boundary itself costs ~60 ns uncontended, so
// all of the GIL cost scope §2.1 projects is CONTENTION and none of it is the
// mechanism. Under §2.1's discipline the only Python-relevant threads during a
// search are the dispatcher and a main thread blocked on `stdin`, which releases
// the GIL while blocked. The prediction that follows is specific: acquisition
// latency stays near the uncontended floor, and any excursion toward
// milliseconds means Python bytecode is running during a search that should not
// be.
//
// This installs the measurement on the REAL C9 topology — a C++-created
// dispatcher thread holding no GIL, W pinned search threads that touch no Python
// at all, and whatever the caller has left running on the interpreter — one
// chunk before C10 makes the answer load-bearing. `guofish_core.GilProbe` is
// handed to `set_batch_hook`; the per-batch waits come back in
// parallel_stats()["hook_wait_ns_samples"].
class GilProbe final : public guofish::BatchHook {
public:
    std::int64_t before_batch(std::size_t batch_size) override {
        const auto before = std::chrono::steady_clock::now();
        // Scoped to this function: acquired here, released on return, exactly
        // as C10's evaluator callback will bracket its torch calls.
        py::gil_scoped_acquire acquire;
        const auto after = std::chrono::steady_clock::now();
        const std::int64_t wait =
            std::chrono::duration_cast<std::chrono::nanoseconds>(after - before).count();
        ++acquisitions_;
        rows_ += static_cast<std::int64_t>(batch_size);
        return wait;
    }

    std::int64_t acquisitions() const noexcept { return acquisitions_; }
    std::int64_t rows() const noexcept { return rows_; }

private:
    std::int64_t acquisitions_ = 0;
    std::int64_t rows_ = 0;
};

// ---------------------------------------------------------------------------
// C10 — the live evaluator
// ---------------------------------------------------------------------------

// A C++-owned, 64-byte-aligned array of anything. AlignedBuffer above is the
// int32 matrix C0 built and every existing caller wants; the policy and value
// buffers are uint16 and float, so the allocation discipline is factored rather
// than the class copied. Same rationale as AlignedBuffer's: C++17 over-aligned
// operator new is the one spelling portable across MSVC and clang with no
// preprocessor branching.
template <class T>
class AlignedArray {
public:
    explicit AlignedArray(std::size_t count) : count_(count), data_(allocate(count)) {
        std::fill_n(data_, count_, T{});
        // reinterpret_cast: pointer -> integer solely to check alignment; the
        // value is never dereferenced or converted back to a pointer.
        assert(reinterpret_cast<std::uintptr_t>(data_) % kAlignment == 0);
    }

    ~AlignedArray() { ::operator delete(data_, std::align_val_t(kAlignment)); }

    AlignedArray(const AlignedArray &) = delete;
    AlignedArray &operator=(const AlignedArray &) = delete;
    AlignedArray(AlignedArray &&) = delete;
    AlignedArray &operator=(AlignedArray &&) = delete;

    T *data() noexcept { return data_; }
    const T *data() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }
    std::size_t nbytes() const noexcept { return count_ * sizeof(T); }

private:
    static T *allocate(std::size_t count) {
        constexpr std::size_t kMax = (std::numeric_limits<std::size_t>::max)();
        if (count > kMax / sizeof(T)) {
            throw std::overflow_error("guofish_core: buffer byte size overflows size_t");
        }
        return static_cast<T *>(::operator new(count * sizeof(T), std::align_val_t(kAlignment)));
    }

    std::size_t count_;
    T *data_;
};

// A NumPy view over an AlignedArray, with the same capsule discipline make_view
// uses: the capsule owns a shared_ptr copy, so either side may drop its
// reference first. Requires the GIL.
template <class T>
py::array_t<T> make_array_view(const std::shared_ptr<AlignedArray<T>> &buffer,
                               std::vector<py::ssize_t> shape) {
    auto *owner = new std::shared_ptr<AlignedArray<T>>(buffer);
    py::capsule base(owner,
                     [](void *p) { delete static_cast<std::shared_ptr<AlignedArray<T>> *>(p); });

    std::vector<py::ssize_t> strides(shape.size());
    py::ssize_t stride = static_cast<py::ssize_t>(sizeof(T));
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return py::array_t<T>(shape, strides, buffer->data(), base);
}

// THE PRODUCTION EVALUATION BOUNDARY.
//
// Owns the three buffers of scope §2.1 and the one Python callable that turns
// tokens into logits. `run` is called by the dispatcher — a C++-created thread
// holding no GIL — once per batch.
//
// WHY THE SWITCH INTERVAL IS SET HERE AND NOT LEFT TO THE CALLER
// --------------------------------------------------------------
// Because "before any search thread spawns" is a requirement (scope §2.1,
// settled by C0b's measurement) and a constructor is the only place in this
// design that is guaranteed to run before one. Under the default 5 ms interval
// C0b measured the dispatcher's contended acquire wait at a MEDIAN of 15.25 ms
// on Windows against a competing pure-Python thread — a hard ceiling of ~60
// batches per second, which at any real batch size reads as a slow network
// rather than as a lock. At 0.0005 the same measurement is a p99 of 78 µs.
//
// It is a process-global interpreter setting, so setting it from a library
// constructor is a real imposition on the host, and it is done anyway: an
// engine that silently runs 250× slower because its host forgot a line is worse
// than one that changed a setting and said so. `switch_interval_before` reports
// what was overwritten.
//
// THE WINDOWS MARGIN IS AN ARTIFACT AND MUST NOT BE SPENT. CPython's Windows
// condvar takes a DWORD of milliseconds by integer division, so 0.0005 s
// truncates to a 0 ms timeout and the interval sweep is a step function with a
// cliff at exactly 1 ms. Linux's ns-resolution wait shows the honest ~1.3×.
class LiveEvaluator final : public guofish::BatchEvaluator {
public:
    LiveEvaluator(std::size_t max_batch, py::object callback, double switch_interval)
        : max_batch_(max_batch),
          input_(std::make_shared<AlignedBuffer>(max_batch, kTokenWidth)),
          policy_(std::make_shared<AlignedArray<std::uint16_t>>(max_batch * guofish::kPolicySize)),
          value_(std::make_shared<AlignedArray<float>>(max_batch)),
          callback_(std::move(callback)) {
        if (max_batch == 0) {
            throw py::value_error("guofish_core.LiveEvaluator: max_batch must be > 0");
        }
        if (!PyCallable_Check(callback_.ptr())) {
            throw py::type_error("guofish_core.LiveEvaluator: callback must be callable");
        }
        py::object sys = py::module_::import("sys");
        switch_interval_before_ = sys.attr("getswitchinterval")().cast<double>();
        if (switch_interval > 0.0) {
            sys.attr("setswitchinterval")(switch_interval);
            switch_interval_ = switch_interval;
        } else {
            switch_interval_ = switch_interval_before_;
        }
    }

    ~LiveEvaluator() override {
        // Runs with the GIL held: pybind11 destroys held objects from the
        // interpreter's own teardown of the wrapper. The teardown hook is where
        // a host that page-locked these buffers unregisters them, and it has to
        // happen HERE — before the allocations below are freed — or CUDA keeps a
        // mapping onto memory the allocator has handed back.
        if (teardown_) {
            try {
                teardown_();
            } catch (py::error_already_set &e) {
                e.discard_as_unraisable("guofish_core.LiveEvaluator teardown");
            }
        }
    }

    LiveEvaluator(const LiveEvaluator &) = delete;
    LiveEvaluator &operator=(const LiveEvaluator &) = delete;

    std::size_t max_batch() const noexcept override { return max_batch_; }

    std::int32_t *token_row(std::size_t i) noexcept override { return input_->row(i); }

    const std::uint16_t *policy_row(std::size_t i) const noexcept override {
        return policy_->data() + i * guofish::kPolicySize;
    }

    float value_at(std::size_t i) const noexcept override { return value_->data()[i]; }

    guofish::EvalTiming run(std::size_t count) override {
        guofish::EvalTiming timing;
        // clock_type::now() needs no GIL, so t0 is sampled from OUTSIDE the
        // acquire scope. That is what makes `acquire_wait_ns` the clean
        // contention metric: everything between t0 and t1 is waiting for some
        // other thread to let go, and nothing else.
        const auto t0 = clock_type::now();
        {
            py::gil_scoped_acquire acquired;
            const auto t1 = clock_type::now();
            timing.acquire_wait_ns = nanos_between(t0, t1);
            try {
                callback_(count);
            } catch (py::error_already_set &e) {
                // Converted to a plain C++ exception WHILE THE GIL IS HELD. A
                // py::error_already_set escaping from here would travel up
                // through the dispatcher's std::exception_ptr, be rethrown on a
                // thread that has released the GIL, and destroy its Python
                // objects without one. The message survives; the exception type
                // does not, which is the trade.
                const std::string what = e.what();
                e.discard_as_unraisable("guofish_core.LiveEvaluator callback");
                throw std::runtime_error(
                    "guofish: the evaluator callback raised, at batch size " +
                    std::to_string(count) + "\n" + what);
            }
            timing.call_ns = nanos_between(t1, clock_type::now());
        }
        return timing;
    }

    py::array_t<std::int32_t> input_view() const { return make_view(input_); }

    py::array_t<std::uint16_t> policy_view() const {
        return make_array_view<std::uint16_t>(
            policy_, {static_cast<py::ssize_t>(max_batch_),
                      static_cast<py::ssize_t>(guofish::kPolicySize)});
    }

    py::array_t<float> value_view() const {
        return make_array_view<float>(value_, {static_cast<py::ssize_t>(max_batch_)});
    }

    // (address, nbytes) per buffer, for a host that wants to page-lock them.
    // Pinning C++-owned memory needs `cudaHostRegister`, and calling it from
    // here would make the CUDA runtime a build dependency of a module that is
    // deliberately torch-free (Global Rule 7) — torch already exposes cudart to
    // Python, so the host does it and hands back an unregister via
    // `set_teardown`.
    py::dict buffer_spans() const {
        py::dict out;
        // reinterpret_cast: pointer -> integer, because cudaHostRegister's
        // Python binding takes an address as an int. Nothing converts back to a
        // pointer on this side, and the buffers outlive the registration by
        // construction — the teardown hook runs in the destructor, before the
        // allocations are released.
        out["input"] = py::make_tuple(reinterpret_cast<std::uintptr_t>(input_->data()),
                                      input_->size() * sizeof(std::int32_t));
        // reinterpret_cast: as above.
        out["policy"] = py::make_tuple(reinterpret_cast<std::uintptr_t>(policy_->data()),
                                       policy_->nbytes());
        // reinterpret_cast: as above.
        out["value"] = py::make_tuple(reinterpret_cast<std::uintptr_t>(value_->data()),
                                      value_->nbytes());
        return out;
    }

    void set_teardown(py::object hook) { teardown_ = std::move(hook); }

    double switch_interval() const noexcept { return switch_interval_; }
    double switch_interval_before() const noexcept { return switch_interval_before_; }

private:
    static std::int64_t nanos_between(clock_type::time_point a, clock_type::time_point b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    }

    std::size_t max_batch_;
    std::shared_ptr<AlignedBuffer> input_;
    std::shared_ptr<AlignedArray<std::uint16_t>> policy_;
    std::shared_ptr<AlignedArray<float>> value_;
    py::object callback_;
    py::object teardown_;
    double switch_interval_ = 0.0;
    double switch_interval_before_ = 0.0;
};

// Gate 2's probe: one position, one recorded 4096-wide bf16 policy row, out come
// the canonical move list and the priors the search would have stored.
//
// This is the PRODUCTION gather — the same `gather_softmax_canonical` the
// dispatcher calls, over the same generation-order index — exposed so the test
// can compare it against ATen without standing up a model, a GPU or a search.
// If it were a reimplementation the gate would be measuring the wrong function.
py::tuple gather_softmax(const std::string &fen,
                         py::array_t<std::uint16_t, py::array::c_style | py::array::forcecast>
                             policy_row,
                         float temperature) {
    if (policy_row.ndim() != 1 ||
        static_cast<std::size_t>(policy_row.shape(0)) != guofish::kPolicySize) {
        throw py::value_error("guofish_core.gather_softmax: policy_row must be a 1-D array of " +
                              std::to_string(guofish::kPolicySize) +
                              " bf16 bit patterns (dtype uint16)");
    }
    if (!(temperature > 0.0f)) {
        throw py::value_error(
            "guofish_core.gather_softmax: temperature must be > 0 (it divides the logits), got " +
            std::to_string(temperature));
    }

    chess::Board board;
    guofish::set_fen_or_throw(board, fen);

    std::vector<std::uint16_t> packed;
    std::vector<std::uint16_t> raw;
    std::vector<std::uint16_t> generation;
    guofish::generate_canonical_moves(board, packed, raw, &generation);
    if (packed.empty()) {
        throw py::value_error("guofish_core.gather_softmax: the position has no legal moves");
    }

    std::vector<float> scratch;
    std::vector<float> priors(packed.size());
    guofish::gather_softmax_canonical(policy_row.data(), packed.data(), generation.data(),
                                      packed.size(), scratch, priors.data(), temperature);

    std::vector<std::string> moves;
    moves.reserve(packed.size());
    for (const std::uint16_t p : packed) {
        moves.push_back(move_to_uci(p));
    }

    py::array_t<float> out(static_cast<py::ssize_t>(priors.size()));
    std::copy(priors.begin(), priors.end(), out.mutable_data());
    return py::make_tuple(moves, out);
}

// The same position's legal moves in chess-library's GENERATION order, as
// normalised UCI.
//
// Exposed for exactly one purpose: Gate 2 needs to reduce ATen's softmax in the
// SAME order the C++ gather does, so that the permutation variance scope §2.6
// measured (up to 3e-7 over 200 random permutations) can be separated from the
// softmax-implementation difference. Reported as two distributions rather than
// one, because they have different causes and only one of them is this port's
// to answer for. Nothing in the search reads this.
std::vector<std::string> generation_order(const std::string &fen) {
    chess::Board board;
    guofish::set_fen_or_throw(board, fen);

    std::vector<std::uint16_t> packed;
    std::vector<std::uint16_t> raw;
    std::vector<std::uint16_t> generation;
    guofish::generate_canonical_moves(board, packed, raw, &generation);

    std::vector<std::string> out(packed.size());
    for (std::size_t k = 0; k < packed.size(); ++k) {
        out[generation[k]] = move_to_uci(packed[k]);
    }
    return out;
}

// The search() / search_parallel() return value. One builder so the two entry
// points cannot drift into reporting different things about the same tree.
template <class Search>
py::dict search_stats_dict(const Search &self, const guofish::SearchStats &stats) {
    py::dict d;
    d["simulations"] = stats.simulations;
    d["expansions"] = stats.expansions;
    d["depth_cap_hits"] = stats.depth_cap_hits;
    d["max_depth"] = stats.max_depth;
    d["select_steps"] = stats.select_steps;
    d["root_visits"] = self.arena().visit_count(self.root());
    d["nodes"] = self.arena().size();
    // C6.
    d["draw_by_rule_hits"] = stats.draw_by_rule_hits;
    d["fifty_move_hits"] = stats.fifty_move_hits;
    d["threefold_hits"] = stats.threefold_hits;
    d["checkmates"] = stats.checkmates;
    d["stalemates"] = stats.stalemates;
    d["insufficient_material"] = stats.insufficient_material;
    d["seventyfive_moves"] = stats.seventyfive_moves;
    d["fivefold_repetitions"] = stats.fivefold_repetitions;
    d["terminal_fast_path_hits"] = stats.terminal_fast_path_hits;
    d["mate_short_circuits"] = stats.mate_short_circuits;
    // C7. Per-SEARCH, unlike `cache_stats`, which is per-cache lifetime — the
    // cache survives set_position, as the reference's does.
    d["cache_hits"] = stats.cache_hits;
    d["cache_misses"] = stats.cache_misses;
    d["cache_inserts"] = stats.cache_inserts;
    d["tablebase_probes"] = stats.tablebase_probes;
    d["tablebase_overrides"] = stats.tablebase_overrides;
    // C9. Must be 0 for anything compared against Python.
    d["synthetic_evaluations"] = stats.synthetic_evaluations;
    d["mating_move"] = self.mating_move() == guofish::kNoMove
                           ? py::none().cast<py::object>()
                           : py::cast(move_to_uci(self.mating_move()));
    d["best_move"] = self.best_move() == guofish::kNoMove
                         ? py::none().cast<py::object>()
                         : py::cast(move_to_uci(self.best_move()));
    return d;
}

// Bind one ReplaySearch instantiation, exactly as C4 binds one NodeArena: twice,
// so neither accumulator can rot behind an #ifdef and a test can drive both
// through one code path. Gate 1 runs the double build, which is the one that has
// to reproduce Python's float arithmetic.
template <class Acc>
void bind_replay_search(py::module_ &m, const char *name, const char *doc) {
    using Search = guofish::ReplaySearch<Acc>;

    py::class_<Search>(m, name, doc)
        .def(py::init<const guofish::SearchConfig &>(), py::arg("config"))

        .def(
            "load_dump",
            [](Search &self,
               py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast> keys,
               py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> is_root,
               py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast> move_offset,
               py::array_t<std::uint16_t, py::array::c_style | py::array::forcecast> moves,
               py::array_t<float, py::array::c_style | py::array::forcecast> priors,
               py::array_t<double, py::array::c_style | py::array::forcecast> values) {
                const auto entries = static_cast<std::size_t>(keys.size());
                const auto move_count = static_cast<std::size_t>(moves.size());
                if (static_cast<std::size_t>(is_root.size()) != entries ||
                    static_cast<std::size_t>(values.size()) != entries ||
                    static_cast<std::size_t>(move_offset.size()) != entries + 1) {
                    throw py::value_error(
                        "guofish_core: keys, is_root and values must be the same length and "
                        "move_offset must be one longer");
                }
                if (static_cast<std::size_t>(priors.size()) != move_count) {
                    throw py::value_error("guofish_core: moves and priors must be the same length");
                }
                self.dump().load(keys.data(), is_root.data(), move_offset.data(), entries,
                                 moves.data(), priors.data(), move_count, values.data());
            },
            py::arg("keys"), py::arg("is_root"), py::arg("move_offset"), py::arg("moves"),
            py::arg("priors"), py::arg("values"),
            "Load golden/gate1_dump.npz. `is_root` selects between the two tables: the "
            "reference softmaxes root priors on the GPU and interior priors on the CPU, and "
            "the two disagree by up to ~2e-9 on the same position, so one nn_key can carry "
            "two different sets of priors.")

        .def(
            "set_position",
            [](Search &self, const std::string &fen, const std::vector<std::string> &history) {
                self.set_position(fen, history);
            },
            py::arg("fen"), py::arg("history") = std::vector<std::string>{},
            "Reset the tree and load a root position. The RAW en-passant square is taken "
            "from the FEN text, never from chess-library's filtered board state. Raises "
            "RuntimeError if the board-derived tokenization disagrees with the FEN-derived "
            "one.\n\n"
            "`history` is the pre-root game history that `build_repetition_history` walks: "
            "the FENs of the positions before each of the last min(halfmove_clock, plies) "
            "moves, most recent first. The root's own position is counted internally, so an "
            "empty history is a position with no game behind it — not a position whose "
            "history is unknown.")

        .def(
            "search",
            [](Search &self, int num_simulations) {
                return search_stats_dict(self, self.search(num_simulations));
            },
            py::arg("num_simulations"),
            "Run until the root has `num_simulations` visits, matching "
            "ParallelMCTS.search's target_new_sims arithmetic — unless the depth-1 mate "
            "short-circuit fires, which ends the search early exactly as the reference's "
            "completion_event does. Raises RuntimeError on a replay-dump miss (naming the "
            "FEN, the move path and the key) or if the ROOT itself has no legal moves.")

        // --- C9: W workers, K in flight -------------------------------------

        .def(
            "search_parallel",
            [](Search &self, int num_simulations, const guofish::ParallelConfig &pc) {
                guofish::SearchStats stats;
                {
                    // THE GIL IS RELEASED FOR THE WHOLE SEARCH, and it has to
                    // be: this call starts W+1 C++ threads and joins them, and
                    // holding the interpreter lock across that would serialise
                    // nothing (no worker touches Python) while making the batch
                    // hook — which acquires the GIL to measure how long
                    // acquiring it takes — deadlock on the caller's own hold.
                    py::gil_scoped_release release;
                    stats = self.search_parallel(num_simulations, pc);
                }
                return search_stats_dict(self, stats);
            },
            py::arg("num_simulations"), py::arg("config"),
            "Run to `num_simulations` root visits with W search threads and K in-flight paths "
            "each, an MPSC leaf queue and one dispatcher.\n\n"
            "Returns the same dict as search(), so a caller can compare the two directly. The "
            "C9-specific counters are in parallel_stats().\n\n"
            "Search threads never block on evaluation: a worker descends, applies virtual "
            "loss, CASes the leaf from Unexpanded to PENDING, submits it and immediately "
            "begins the next simulation. A worker that loses that CAS unwinds its own virtual "
            "loss, discards the simulation and retries — counted as select_collisions. "
            "Expansion happens only on the dispatcher.\n\n"
            "The dispatcher drains min(available, max_batch) with no minimum-batch floor and "
            "no straggler timeout; it drains when the queue is non-empty and no search thread "
            "can add to it. That trigger is what makes W=1 reproducible run to run (acceptance "
            "layer 2) while still sizing batches by the outstanding-leaf count rather than by "
            "the worker count. See DECISIONS.md, C9.")

        .def(
            "parallel_stats",
            [](const Search &self) { return parallel_stats_dict(self.parallel_stats()); },
            "The counters from the last search_parallel(). All zero after a serial search().\n\n"
            "`delivered` is the number of simulations actually backed up into the root, "
            "counted with a fetch_add by whichever thread performed the backup — not the "
            "requested budget. The reference's `stats['simulations'] += 1` was an "
            "unsynchronized read-modify-write over the same quantity across 32 threads, which "
            "is why its throughput figures could not be trusted to the last few percent.")

        // --- C11c: the mutable deadline ------------------------------------
        //
        // THE ONLY BINDINGS IN THIS FILE THAT ARE MEANT TO BE CALLED FROM
        // ANOTHER PYTHON THREAD WHILE A SEARCH IS RUNNING, and the property
        // that makes that safe is that each is a single atomic load or store
        // with nothing else in it. In particular NONE of them takes
        // `py::gil_scoped_release`: they are too short for the release to pay
        // for itself, and the calling thread — the UCI reader, blocked in
        // `readline` the rest of its life — already holds the GIL while the
        // searching thread has released it.
        .def(
            "set_deadline_in",
            [](Search &self, double seconds) { self.set_deadline_in(seconds); },
            py::arg("seconds"),
            "Arm the mutable deadline `seconds` from now. Callable from another thread while "
            "search_parallel() is in flight — that is what it is for.\n\n"
            "This is how `ponderhit` converts an infinite-budget search into a timed one "
            "WITHOUT stopping and restarting it. Restarting would discard the tree, which is "
            "the entire point of having pondered.\n\n"
            "A duration rather than an instant, deliberately: std::steady_clock and Python's "
            "time.monotonic() are both monotonic and are NOT the same epoch on Windows, so a "
            "caller is never asked to name a time in a clock it cannot read. Zero or negative "
            "is an already-expired deadline, which is the correct reading of a ponderhit on a "
            "flagged clock.\n\n"
            "The workers re-read it at their existing abort point, so it takes effect within "
            "one simulation. It is NOT cleared by search_parallel(): the host arms it once and "
            "the slice loop enters the search many times. Clear it in a finally.")

        .def("clear_deadline", &Search::clear_deadline,
             "Disarm the mutable deadline. The search then runs to its node budget.\n\n"
             "MUST be called when a timed search ends, because the deadline outlives the "
             "search_parallel() call that observed it. A host that forgets leaves the next "
             "search pre-expired.")

        .def_property_readonly(
            "deadline_remaining_s",
            [](const Search &self) {
                return static_cast<double>(self.deadline_remaining_ns()) * 1e-9;
            },
            "Seconds until the armed deadline; 0.0 when none is armed, negative once it has "
            "passed. The two zeros are distinguishable through `deadline_armed`.")

        .def_property_readonly(
            "deadline_armed",
            [](const Search &self) { return self.deadline_ns() != Search::kNoDeadline; },
            "Whether a deadline is currently armed. On the engine's telemetry line so a "
            "search that ended instantly can be told from one that was never given time.")

        // --- C11c: graceful arena exhaustion --------------------------------
        .def_property_readonly(
            "arena_exhausted", [](const Search &self) { return self.arena_exhausted(); },
            "Did an allocation fail during the last search()/search_parallel()?\n\n"
            "TRUE MEANS THE SEARCH DEGRADED RATHER THAN FAILED. The workers stopped "
            "descending at their abort point, the in-flight simulations unwound without "
            "backing anything up, and the search returned the best move it already had. "
            "`delivered < requested` is then a legitimate outcome — and a benchmark harness "
            "must REJECT such a row rather than publish a short one, which is why this is "
            "readable at all.\n\n"
            "Cleared at the top of each search call. A host that slices a move ORs it across "
            "the slices, because the statement worth logging is 'this move degraded'.")

        .def_property_readonly(
            "arena_exhausted_at",
            [](const Search &self) { return self.arena_exhausted_at(); },
            "Nodes allocated when the first allocation failed, or 0 if none did. The N in "
            "`info string arena exhausted at N nodes`.")

        .def_property_readonly(
            "arena_capacity", [](const Search &self) { return self.arena_capacity(); },
            "Nodes the ACTIVE arena can hold. The denominator for arena_high_water; the "
            "ping-pong pair reserves two of these. See arena_bytes_reserved.")

        .def_property_readonly(
            "arena_high_water", [](const Search &self) { return self.arena_high_water(); },
            "Peak occupancy of EITHER ping-pong arena, across every reset. Also in "
            "reuse_stats(); lifted out here because the C11c sizing verification reads it "
            "per move and does not want the rest of that dict.")

        .def(
            "audit", [](const Search &self) { return tree_audit_dict(self.audit()); },
            "The exact conservation invariants, with no epsilon anywhere.\n\n"
            "`vloss_total` is a FLAT SCAN of the arena, not a traversal: a virtual loss "
            "stranded on a node that is no longer reachable is exactly the kind of leak this "
            "is looking for, and a traversal would not see it. It must be 0 at the end of any "
            "search, at any virtual-loss magnitude, because the counts are integers.\n\n"
            "`conservation_failures` counts expanded nodes where visits != 1 + sum(children "
            "visits). The 1 is the simulation that expanded the node; nothing else can stop "
            "at an expanded node. `first_bad_node` names the first one in arena order so a "
            "failure can be reported rather than merely counted.\n\n"
            "The visit invariant assumes a tree built by search alone. SearchConfig."
            "ponder_decay rescales inherited visit counts on promotion and would break it by "
            "design; it is 1.0 by default and the C9 tests do not enable it.")

        .def_property(
            "synthetic_fallback", &Search::synthetic_fallback, &Search::set_synthetic_fallback,
            "C9. Answer a replay-dump miss with a deterministic stand-in evaluator instead of "
            "raising.\n\n"
            "OFF BY DEFAULT, AND NOTHING COMPARED AGAINST PYTHON MAY TURN IT ON. The Gate 1 "
            "dump holds exactly the positions the SERIAL reference evaluated, which is what "
            "makes a miss the strongest test in C5. With K in-flight paths, virtual loss "
            "steers descents onto branches the serial reference never opened — that is leaf "
            "parallelism working, not failing — and the first miss arrives about five plies "
            "in at W=1, K=8.\n\n"
            "So acceptance layer 1 (W=1, K=1, bit-exact against Python) runs with this off, "
            "and layers 2 and 3 — which assert reproducibility and conservation, not agreement "
            "with Python — run with it on. Real dump entries are still used wherever they "
            "exist. search()['synthetic_evaluations'] counts the holes it filled, so a Gate 1 "
            "run asserts 0 rather than trusting the flag.")

        .def(
            "set_batch_hook",
            [](Search &self, GilProbe *probe) { self.set_batch_hook(probe); },
            py::arg("hook").none(true), py::keep_alive<1, 2>(),
            "Install a per-batch hook on the dispatcher, or clear it with None.\n\n"
            "The only implementation is GilProbe, which measures how long the dispatcher — a "
            "C++-created thread holding no GIL — waits to acquire it, once per batch. That is "
            "the measurement scope §2.1's discipline rests on and C10 depends on being "
            "cheap; C9 takes it on the real thread topology first.")

        // --- C10: the live evaluator ----------------------------------------

        .def(
            "set_evaluator",
            [](Search &self, LiveEvaluator *evaluator) { self.set_evaluator(evaluator); },
            py::arg("evaluator").none(true), py::keep_alive<1, 2>(),
            "Install the live evaluator, or clear it with None.\n\n"
            "Installing one takes the replay dump out of the evaluation path ENTIRELY, root "
            "included, and there is no mode in which a search consults both. The dump's value "
            "is that a miss is a hard failure proving a divergence; a live fallback would turn "
            "every such proof into a silent 'the network answered instead'.\n\n"
            "It also unifies the softmax. The reference evaluates root priors on the GPU "
            "(`_expand_root` hands `expand()` a CUDA tensor) and interior priors on the CPU "
            "(BatchedEvaluator's bulk `.cpu()`); those disagree by up to 1.9e-9 across 6 of 37 "
            "priors, which is enough to flip a best move at 200 sims once the root position "
            "recurs as an interior node. With an evaluator installed every node — root "
            "included — goes down one path and through the same cache. That is a deliberate "
            "divergence from the reference at roots, arbitrated by Gate 2b. See DECISIONS.md, "
            "C10.")

        .def(
            "eval_stats", [](const Search &self) { return eval_stats_dict(self.eval_stats()); },
            "The evaluation boundary's counters for the LAST search, reset at its start.\n\n"
            "`acquire_wait_ns_samples` is one entry per boundary crossing and is the "
            "measurement C10's contingency is decided on: implement C++-side `info` emission "
            "if its p99 exceeds 200 us, or if `acquire_wait_ns` exceeds 1% of the move's wall "
            "time. It is deliberately not summarised here — a mean hides the tail behind the "
            "many fast acquisitions, and the tail is the entire cost.\n\n"
            "`call_ns` is NOT a contention metric and must not be read as one: torch and numpy "
            "re-acquire the GIL inside a call, so a handoff wait lands in the middle of it. "
            "The ratio between the two is what says whether a slow batch was contention or "
            "was just the model.\n\n"
            "`rows` counts leaves actually sent to the network and `cache_skipped` those the "
            "transposition cache answered before the batch was formed; `batches` counts "
            "boundary crossings. rows/batches is the mean batch the model actually saw.")

        .def_static(
            "topology", [] { return topology_dict(Search::topology()); },
            "What this build can see of the machine's CPU topology, and where it came from.\n\n"
            "`source` is the API or the sysfs path, or the reason there are no numbers. A "
            "kernel that does not report the hybrid split (WSL2 does not expose cpu_capacity) "
            "reports hybrid=False and every core as a performance core, rather than a guess: "
            "pinning threads to a layout that was inferred would be worse than not pinning "
            "them, because the resulting benchmark row would claim something untrue.")

        // --- C8: tree reuse -------------------------------------------------

        .def(
            "apply_move",
            [](Search &self, const std::string &uci, bool from_ponder) {
                return self.apply_move(packed_from_uci(uci), from_ponder);
            },
            py::arg("uci"), py::arg("from_ponder") = false,
            "Play `uci` and keep the subtree under it, compacting-copying the survivors into "
            "the standby arena and swapping (scope §2.3's ping-pong).\n\n"
            "Returns True if a subtree was reused, False if the tree had to be thrown away — "
            "the reference's `move not in self.root.children` branch, reachable when the root "
            "is an unexpanded promoted leaf or terminal. Either way the position advances, and "
            "an illegal move raises ValueError rather than desynchronising the engine from the "
            "game.\n\n"
            "The move must be in NORMALISED UCI: castling is e1g1, never chess-library's "
            "king-takes-rook e1h1. That is the form best_move returns and the form the arena "
            "stores.\n\n"
            "`from_ponder=True` is the caller's assertion that the promoted subtree was built "
            "by a ponder search rather than by a search of this position; only then is "
            "SearchConfig.ponder_decay applied. Raises RuntimeError (TreeCorruption) if the "
            "post-compaction structural diff finds the copy is not the tree it copied.")

        .def(
            "reuse_stats",
            [](const Search &self) {
                const guofish::ReuseStats stats = self.reuse_stats();
                py::dict d;
                d["applies"] = stats.applies;
                d["discards"] = stats.discards;
                d["nodes_copied"] = stats.nodes_copied;
                d["nodes_dropped"] = stats.nodes_dropped;
                d["verifications"] = stats.verifications;
                d["decays"] = stats.decays;
                d["terminal_promotions"] = stats.terminal_promotions;
                d["terminal_marks_cleared"] = stats.terminal_marks_cleared;
                d["largest_copy"] = stats.largest_copy;
                d["arena_high_water"] = self.arena_high_water();
                d["arena_bytes_reserved"] = self.arena_bytes_reserved();
                d["standby_allocated"] = self.has_standby_arena();
                return d;
            },
            "The tree-reuse counters since the last set_position — the span of a game, not of "
            "a search.\n\n"
            "`arena_high_water` is the peak occupancy of EITHER arena. That is the honest "
            "figure for the memory budget: during a compaction the source subtree and its copy "
            "are alive at the same moment, and reporting only the active arena would miss "
            "exactly the instant the budget is about.")

        .def(
            "rep_history",
            [](const Search &self) {
                py::dict d;
                for (const auto &entry : self.rep_history()) {
                    d[py::cast(entry.first)] = entry.second;
                }
                return d;
            },
            "rep_key -> occurrences in the game BEFORE this search, the root included: "
            "`build_repetition_history(board)`'s answer for the current root.\n\n"
            "C8's requirement that the path and history partitions sum identically across an "
            "applied move is otherwise only visible as a threefold appearing or disappearing "
            "somewhere deep in a tree. This is where it comes from, so a divergence can be "
            "attributed instead of hunted.")

        .def_property_readonly(
            "root_fen", [](const Search &self) { return self.root_fen(); },
            "The FEN of the position the tree is rooted at, carrying THIS search's raw "
            "en-passant square and its own halfmove clock. After apply_move this is how a "
            "harness checks that the engine and the game are still on the same position.")

#if defined(GUOFISH_DEBUG_VL)
        .def("debug_total_vloss", &Search::debug_total_vloss,
             "DEBUG BUILDS ONLY. The sum of every node's in-flight virtual-loss count over the "
             "whole arena — a READ of what the reference's `_reset_virtual_loss` used to "
             "overwrite.\n\n"
             "C8's brief forbids a production equivalent of that defensive walk: repayment "
             "here is scope-guaranteed by RAII (run_simulation's Unwind destructor), so a "
             "reset would hide a bug rather than fix one. This exists so a test can assert the "
             "invariant instead of trusting it, and it is compiled only under "
             "-DGUOFISH_DEBUG_VL (on by default in Debug builds, which is where the sanitized "
             "run lives). A quiescent tree returns exactly 0 at any virtual-loss magnitude, "
             "because the counts are integers.")
#endif

        .def(
            "terminal_nodes",
            [](Search &self) {
                py::list out;
                for (const guofish::TerminalNode &node : self.terminal_nodes()) {
                    py::dict d;
                    d["path"] = node.path;
                    d["fen"] = node.fen;
                    d["value"] = node.value;
                    d["visits"] = node.visits;
                    d["children"] = node.children;
                    d["expanded"] = node.expanded != 0;
                    d["depth"] = node.depth;
                    out.append(d);
                }
                return out;
            },
            "Every node carrying the terminal bit, with the FEN it stands for.\n\n"
            "The FEN is re-loadable — it carries the search's own raw ep square and its own "
            "halfmove clock — so a fifty-move draw node can be handed straight back to "
            "set_position(). That is what a promotion probe is, and it is how "
            "'a terminal node promoted to root still yields a legal move' is demonstrated "
            "rather than asserted.")

        .def(
            "dump_tree",
            [](const Search &self, std::int32_t min_visits) {
                const std::vector<guofish::TreeRecord> records = self.dump_tree(min_visits);
                py::list out;
                for (const auto &record : records) {
                    out.append(py::make_tuple(move_to_uci(record.move), record.visits,
                                              record.value_sum, record.prior));
                }
                return out;
            },
            py::arg("min_visits") = 0,
            "The tree in canonical DFS order as (move, visit_count, value_sum, prior) "
            "tuples. min_visits=0 emits every node; 1 emits the visited subtree.")

        .def(
            "dump_tree_arrays",
            [](const Search &self, std::int32_t min_visits) {
                const std::vector<guofish::TreeRecord> records = self.dump_tree(min_visits);
                const auto n = static_cast<py::ssize_t>(records.size());

                py::array_t<std::uint16_t> depth(n);
                py::array_t<std::uint16_t> move(n);
                py::array_t<std::int32_t> visits(n);
                py::array_t<double> value_sum(n);
                py::array_t<float> prior(n);
                py::array_t<std::uint16_t> children(n);
                py::array_t<std::uint8_t> terminal(n);
                py::array_t<float> terminal_value(n);

                std::uint16_t *depth_p = depth.mutable_data();
                std::uint16_t *move_p = move.mutable_data();
                std::int32_t *visits_p = visits.mutable_data();
                double *value_sum_p = value_sum.mutable_data();
                float *prior_p = prior.mutable_data();
                std::uint16_t *children_p = children.mutable_data();
                std::uint8_t *terminal_p = terminal.mutable_data();
                float *terminal_value_p = terminal_value.mutable_data();

                for (std::size_t i = 0; i < records.size(); ++i) {
                    depth_p[i] = records[i].depth;
                    move_p[i] = records[i].move;
                    visits_p[i] = records[i].visits;
                    value_sum_p[i] = records[i].value_sum;
                    prior_p[i] = records[i].prior;
                    children_p[i] = records[i].children;
                    terminal_p[i] = records[i].terminal;
                    terminal_value_p[i] = records[i].terminal_value;
                }

                py::dict d;
                d["depth"] = depth;
                d["move"] = move;
                d["visits"] = visits;
                d["value_sum"] = value_sum;
                d["prior"] = prior;
                d["children"] = children;
                d["terminal"] = terminal;
                d["terminal_value"] = terminal_value;
                return d;
            },
            py::arg("min_visits") = 0,
            "The same traversal as dump_tree(), as NumPy arrays. This is the comparison "
            "surface: golden/gate1_trees.npz stores the reference in the same layout, so a "
            "run of 200,000 nodes is compared without materialising 200,000 Python tuples, "
            "and value_sum can be compared on its bit pattern rather than by ==.\n\n"
            "`terminal` is the arena's terminal BIT, not a lifecycle value. A terminal node "
            "is Unexpanded, so `terminal == 1 and children > 0` is not a state this arena "
            "can represent — which is the C6 structural requirement, checked over the whole "
            "corpus by tests/test_c6_terminal_invariants.py.")

        .def("c_puct", &Search::c_puct_of, py::arg("parent_visits"),
             "This instance's c(N), including its c_factor. Exposed so a harness can use "
             "the C++ implementation on both sides of a comparison.")

        // --- C7: the transposition cache -----------------------------------

        .def(
            "cache_stats",
            [](const Search &self) {
                const guofish::CacheStats stats = self.cache_stats();
                py::dict d;
                d["hits"] = stats.hits;
                d["misses"] = stats.misses;
                d["inserts"] = stats.inserts;
                d["collisions"] = stats.collisions;
                d["refreshes"] = stats.refreshes;
                d["hit_rate"] = stats.hit_rate();
                d["size"] = self.cache_size();
                d["capacity"] = self.cache_capacity();
                d["shards"] = self.cache_shard_count();
                d["enabled"] = self.has_cache();
                return d;
            },
            "The cache's LIFETIME counters, which span every set_position since the last "
            "clear_cache() — the reference's cache lives on the engine, not on the search, "
            "and this one does too. For one search's numbers, read cache_hits / "
            "cache_misses out of search()'s return value instead.\n\n"
            "The hit rate is an acceptance criterion in its own right: with tablebases off "
            "the cache is result-invariant, so a cache that never hits produces a "
            "bit-identical tree and cannot be distinguished by tree comparison.")

        .def("clear_cache", &Search::clear_cache,
             "Drop every entry and zero the cache's counters. Not called by set_position, "
             "deliberately: a cache that reset per position would never see a transposition "
             "across moves of a game, which is most of what it is for.")

        .def(
            "cache_entry_by_key",
            [](Search &self, std::uint64_t key) -> py::object {
                guofish::CachedEval entry;
                if (!self.cache_probe(guofish::NNKey(key), entry)) {
                    return py::none();
                }
                py::array_t<std::uint16_t> moves(
                    static_cast<py::ssize_t>(entry.moves.size()));
                std::copy(entry.moves.begin(), entry.moves.end(), moves.mutable_data());
                py::array_t<float> priors(static_cast<py::ssize_t>(entry.priors.size()));
                std::copy(entry.priors.begin(), entry.priors.end(), priors.mutable_data());

                py::dict d;
                d["value"] = entry.value.value;
                d["moves"] = moves;   // PACKED, to compare against a dump slice directly
                d["priors"] = priors;
                return d;
            },
            py::arg("nn_key"),
            "The entry for a raw nn_key, with PACKED moves — the layout "
            "golden/gate1_dump.npz stores, so a round-trip check compares two arrays "
            "rather than two lists of formatted strings.\n\n"
            "Counts as a hit or a miss in cache_stats(), exactly as the search's own probe "
            "does.")

        .def(
            "cache_entry",
            [](Search &self, const std::string &fen) -> py::object {
                guofish::CachedEval entry;
                const guofish::EvalRow row(guofish::parse_fen(fen));
                if (!self.cache_probe(row.key(), entry)) {
                    return py::none();
                }
                py::list moves;
                for (const std::uint16_t packed : entry.moves) {
                    moves.append(move_to_uci(packed));
                }
                py::array_t<float> priors(static_cast<py::ssize_t>(entry.priors.size()));
                std::copy(entry.priors.begin(), entry.priors.end(), priors.mutable_data());

                py::dict d;
                d["nn_key"] = row.key().value;
                d["value"] = entry.value.value;
                d["moves"] = moves;
                d["priors"] = priors;
                return d;
            },
            py::arg("fen"),
            "The cache entry for `fen`, or None on a miss. THIS IS THE ROUND-TRIP SURFACE: "
            "the brief requires entry contents — priors and move list — to be verified "
            "directly and by assertion on mismatch, rather than inferred from the tree, "
            "because the tree cannot see them.\n\n"
            "NOTE that probing through this counts as a hit or a miss in cache_stats(), "
            "exactly as the search's own probe does. Read the statistics before inspecting "
            "entries, or clear and re-run.")

        // --- C7: tablebases -------------------------------------------------

        .def(
            "set_tablebase",
            [](Search &self, guofish::TablebaseProber *prober) { self.set_tablebase(prober); },
            py::arg("prober").none(true),
            // The search BORROWS the prober (the reference borrows its handle
            // too — one open Tablebase serves the UCI layer and every search).
            // keep_alive<1, 2> ties the backend's lifetime to this search's, so
            // a caller who drops their only reference does not leave a dangling
            // pointer in the leaf path.
            py::keep_alive<1, 2>(),
            "Attach a tablebase backend, or None to detach. Mode 2 then fires at every leaf "
            "with <= 5 men.\n\n"
            "THE SHIPPING DEFAULT IS None. cpp/tablebase.hpp explains why the Syzygy decoder "
            "is a backend rather than part of this chunk (Global Rule 7 and the brief's "
            "non-blocking clause), and what is implemented and tested without it.")

        .def_property_readonly(
            "tablebase_backend",
            [](const Search &self) -> py::object {
                const guofish::TablebaseProber *prober = self.tablebase();
                return prober == nullptr ? py::none().cast<py::object>()
                                         : py::cast(prober->backend());
            },
            "The attached backend's name, or None when tablebases are off.")

        .def_property_readonly(
            "dump_size", [](const Search &self) { return self.dump().size(); },
            "Entries in the loaded replay dump, both tables.")
        .def_property_readonly(
            "dump_root_size", [](const Search &self) { return self.dump().root_size(); })
        .def_property_readonly(
            "accumulator", [](const Search &) { return std::string(Acc::name()); })
        .def_property_readonly(
            "nodes", [](const Search &self) { return self.arena().size(); },
            "Arena nodes allocated so far.")
        .def_property_readonly(
            "root_visits",
            [](const Search &self) { return self.arena().visit_count(self.root()); })
        .def_property_readonly(
            "root_value_sum",
            [](const Search &self) { return self.arena().value_sum(self.root()); })
        .def_property_readonly(
            "best_move",
            [](const Search &self) {
                const std::uint16_t move = self.best_move();
                return move == guofish::kNoMove ? py::none().cast<py::object>()
                                                : py::cast(move_to_uci(move));
            },
            "ParallelMCTS.search's answer: the mating move if the depth-1 hack fired, "
            "otherwise the most-visited root child, ties going to the first in canonical "
            "order — which is what `max(root.children.items(), key=visit_count)` returns.")
        .def_property_readonly(
            "mating_move",
            [](const Search &self) {
                const std::uint16_t move = self.mating_move();
                return move == guofish::kNoMove ? py::none().cast<py::object>()
                                                : py::cast(move_to_uci(move));
            },
            "The move the depth-1 mate short-circuit seized on, or None. Not None means the "
            "search stopped early, so root_visits is below the requested count.");
}

}  // namespace

PYBIND11_MODULE(guofish_core, m) {
    m.doc() = "GuoFish C++ core — C0 toolchain spike, C0b GIL contention probe, C1 movegen, C2 tokenizer";

    m.attr("SEQ_LENGTH") = guofish::kSeqLength;

    // C8. Whether this module carries the debug-only full-tree virtual-loss
    // audit (`ReplaySearch.debug_total_vloss`). A module attribute rather than a
    // `build_info()` key: it belongs in that dict by subject, but
    // tests/test_c0b_contention.py pins build_info()'s exact key set and Global
    // Rule 1 makes that pin the specification.
    //
    // Reported at all so that "the Release module does not contain a defensive
    // virtual-loss walk" is a statement the acceptance test can assert
    // positively, rather than an inference from a missing attribute — which
    // would also be satisfied by a typo.
    // C9. Same reasoning, same place: build_info()'s key set is pinned, and
    // "the ThreadSanitizer acceptance run really was a ThreadSanitizer build"
    // has to be assertable rather than inferred from a log file's filename.
    m.attr("TSAN") = static_cast<bool>(GUOFISH_TSAN_BUILD);

#if defined(GUOFISH_DEBUG_VL)
    m.attr("DEBUG_VL") = true;
#else
    m.attr("DEBUG_VL") = false;
#endif

    // C1. std::invalid_argument is translated to ValueError by pybind11's stock
    // exception translator, so a bad FEN surfaces in Python as ValueError
    // without a custom registration.
    m.def("legal_moves", &guofish::legal_moves, py::arg("fen"),
          "Every legal move in `fen` as standard UCI strings, in canonical "
          "(from, to, promotion) order. Castling is normalised away from "
          "chess-library's king-takes-rook encoding (e1h1 -> e1g1). Raises "
          "ValueError on a FEN that cannot be parsed or has no king.");

    // C2. The 68-token encoding the v5 network was trained on.
    m.def("tokens", &tokens, py::arg("fen"),
          "The 68-token encoding of `fen` as an int32 NumPy array: 64 squares, side to "
          "move, castling rights, en-passant target file, CLS. Byte-for-byte identical to "
          "core.mctsv4.board_to_tokens. Index 66 reports the en-passant file whenever the "
          "FEN carries one, whether or not the capture is playable. Raises ValueError on a "
          "FEN python-chess would itself refuse.");

    py::class_<TokenBatch>(m, "TokenBatch",
                           "A C++-owned, 64-byte-aligned [capacity, 68] int32 buffer that FENs are "
                           "tokenized directly into. view() aliases it with no copy.")
        .def(py::init<std::size_t>(), py::arg("capacity"))
        .def_property_readonly("capacity", &TokenBatch::capacity, "Rows in the buffer.")
        .def("view", &TokenBatch::view,
             "A zero-copy [capacity, 68] int32 NumPy view of the buffer. Slice it to the "
             "rows fill() reported; a NumPy slice is itself a view.")
        .def("fill", &TokenBatch::fill, py::arg("fens"), py::arg("row_offset") = 0,
             "Tokenize an iterable of FENs into rows [row_offset, row_offset + n) and return n. "
             "Releases the GIL while encoding. Raises ValueError if the FENs do not fit, or on "
             "a FEN that cannot be parsed — in which case rows already written keep their new "
             "contents.");

    // C3. The magic bitboard tables `has_legal_en_passant` reads are filled by a
    // dynamic initializer inside chess.hpp. It runs at load time, before any of
    // this is reachable from Python, so this call is redundant today — it is
    // here so that the dependency is stated by the module that has it, rather
    // than inherited from an unrelated inline variable in another header. It is
    // idempotent and costs 64 iterations once.
    chess::attacks::initAttacks();

    m.def("nn_key", &nn_key, py::arg("fen"),
          "The NN cache key for `fen` as a uint64: FNV-1a over the 68-token encoding. Two "
          "positions share a key exactly when they tokenize identically, which is the same "
          "partition core.mctsv4.make_cache_key induces. Ignores the halfmove clock — the "
          "network does not see it. Distinguishes any en-passant square the FEN carries, "
          "whether or not the capture is playable.");

    m.def("rep_key", &rep_key, py::arg("fen"),
          "The repetition/draw key for `fen` as a uint64: FNV-1a over the fields of "
          "python-chess's Board._transposition_key(). Counts an en-passant square only when "
          "an en-passant capture is actually LEGAL — a different rule from nn_key's and from "
          "Polyglot's, and the three disagree on real positions.");

    m.def("keys", &keys, py::arg("fen"),
          "(nn_key, rep_key) for `fen`, sharing one parse. Never equal: the two payloads "
          "carry different domain tags, so a swapped key cannot compare equal by luck.");

    m.def("nn_keys", &nn_keys, py::arg("fens"),
          "nn_key over an iterable of FENs as a uint64 NumPy array. Releases the GIL for the "
          "sweep.");

    m.def("rep_keys", &rep_keys, py::arg("fens"),
          "rep_key over an iterable of FENs as a uint64 NumPy array. Releases the GIL for the "
          "sweep.");

    m.def("key_pairs", &key_pairs, py::arg("fens"),
          "Both keys over an iterable of FENs as an [n, 2] uint64 NumPy array (column 0 "
          "nn_key, column 1 rep_key), parsing each FEN once.");

    m.def("key_type_separation", &key_type_separation,
          "Compile-time facts about the C++ NNKey/RepKey types, as <type_traits> answered "
          "them while this module was built. Every 'accepted', 'converts', 'assignable' and "
          "'comparable' entry crossing the two types must be False; the build itself fails "
          "if one is not.");

    // -----------------------------------------------------------------------
    // C4 — the SoA node arena.
    // -----------------------------------------------------------------------

    m.attr("Q32_SCALE") = guofish::kQ32Scale;
    m.attr("Q32_ONE") = guofish::kQ32One;
    m.attr("Q32_RESOLUTION") = guofish::kQ32Resolution;
    m.attr("CACHE_LINE") = guofish::kCacheLine;
    m.attr("NO_NODE") = guofish::kNoNode;
    m.attr("NO_MOVE") = guofish::kNoMove;
    m.attr("DEFAULT_ACCUMULATOR") = std::string(guofish::kDefaultAccumulator);

    m.attr("STATE_UNEXPANDED") = static_cast<std::uint8_t>(guofish::NodeState::Unexpanded);
    m.attr("STATE_PENDING") = static_cast<std::uint8_t>(guofish::NodeState::Pending);
    m.attr("STATE_EXPANDED") = static_cast<std::uint8_t>(guofish::NodeState::Expanded);
    m.attr("STATE_LIFECYCLE_MASK") = guofish::kLifecycleMask;
    m.attr("TERMINAL_BIT") = guofish::kTerminalBit;

    // Alphabetical by UCI letter, not by piece value: canonical move order is
    // the byte order of the UCI string, so b < n < q < r, and None is 0 so a
    // four-character move sorts before any promotion of the same from/to.
    m.attr("PROMO_NONE") = static_cast<std::uint16_t>(guofish::Promotion::None);
    m.attr("PROMO_BISHOP") = static_cast<std::uint16_t>(guofish::Promotion::Bishop);
    m.attr("PROMO_KNIGHT") = static_cast<std::uint16_t>(guofish::Promotion::Knight);
    m.attr("PROMO_QUEEN") = static_cast<std::uint16_t>(guofish::Promotion::Queen);
    m.attr("PROMO_ROOK") = static_cast<std::uint16_t>(guofish::Promotion::Rook);

    m.def(
        "q32_from_float", [](double v) { return guofish::to_q32(v); }, py::arg("value"),
        "A value in [-1, 1] as a Q32 fixed-point integer, rounding half away from zero. "
        "Outside that range the debug build asserts; the domain is a network value, not a sum.");

    m.def(
        "q32_to_float", [](std::int64_t q) { return guofish::from_q32(q); }, py::arg("q"),
        "A Q32 integer back as a double. Exact for |q| <= 2^53, which is every value and every "
        "sum below 2.1M visits at full magnitude.");

    m.def("q32_roundtrip_sweep", &q32_roundtrip_sweep, py::arg("stride") = 1,
          "float -> Q32 -> double over every IEEE-754 float bit pattern in [-1, 1] (both signs) "
          "at the given stride; stride 1 is the exhaustive sweep of all 2,130,706,434 of them. "
          "Reports the largest absolute round-trip error, how many values are not bit-exact and "
          "the largest of those, and whether the quantizer is symmetric about zero.");

    m.def("q32_code_sweep", &q32_code_sweep, py::arg("stride") = 4099,
          "Q32 -> double -> Q32 over the code range [-2^32, 2^32] at the given stride, plus "
          "every boundary code unconditionally. Must report zero mismatches.");

    m.def("arena_layout", &arena_layout,
          "Compile-time facts about the arena's atomics and per-node widths, as the compiler "
          "answered them while building this module. Every '*_always_lock_free' entry must be "
          "True; the build fails if one is not.");

    m.def("pack_move", &pack_move, py::arg("from_square"), py::arg("to_square"),
          py::arg("promotion") = 0,
          "Pack (from, to, promotion) into the uint16 the arena's move field holds: "
          "from << 10 | to << 4 | promo, with squares numbered rank * 8 + file.");

    m.def(
        "move_from", [](std::uint16_t p) { return guofish::move_from(p); }, py::arg("packed_move"));
    m.def(
        "move_to", [](std::uint16_t p) { return guofish::move_to(p); }, py::arg("packed_move"));
    m.def(
        "move_promotion",
        [](std::uint16_t p) { return static_cast<std::uint16_t>(guofish::move_promotion(p)); },
        py::arg("packed_move"));

    m.def(
        "canonical_move_key", [](std::uint16_t p) { return guofish::canonical_move_key(p); },
        py::arg("packed_move"),
        "A uint16 whose integer order is exactly the byte order of the move's UCI string, i.e. "
        "C1's canonical (from, to, promotion) order. Sorting packed moves directly does NOT "
        "give that order — square indices are rank-major and UCI is file-major — and because "
        "PUCT breaks ties by child order, getting it wrong makes C++ and Python explore "
        "different moves at equal priors rather than failing loudly.");

    m.def("pack_uci", &packed_from_uci, py::arg("uci"),
          "The inverse of move_to_uci: a UCI string to the arena's packed 16-bit move. "
          "Syntactic — no board is consulted — so it is the right tool for building a "
          "move list to hand to the cache, and the wrong one for asking whether a move is "
          "legal.");

    m.def("move_to_uci", &move_to_uci, py::arg("packed_move"),
          "A packed move as its UCI string. No castling normalisation: that is a property of a "
          "generated move and belongs at the movegen boundary (C1), not in the packing.");

    bind_arena<guofish::Q32Accumulator>(
        m, "NodeArenaQ32",
        "The production node arena: struct-of-arrays, value_sum in Q32 fixed-point "
        "atomic<int64>. Integer addition is associative, so multithreaded backup is "
        "bit-reproducible.");

    bind_arena<guofish::DoubleAccumulator>(
        m, "NodeArenaDouble",
        "The Gate 1 equivalence arena: identical layout, value_sum in atomic<double> so C++ "
        "reproduces Python's float arithmetic exactly. Addition is a CAS loop and is not "
        "associative.");

    // Which one this build's engine uses. Both classes exist in every build so
    // neither can rot behind an #ifdef; this alias is what GUOFISH_VALUE_SUM
    // switches.
#if defined(GUOFISH_VALUE_SUM_DOUBLE)
    m.attr("NodeArena") = m.attr("NodeArenaDouble");
#else
    m.attr("NodeArena") = m.attr("NodeArenaQ32");
#endif

    m.def("sibling_scan_bench", &sibling_scan_bench, py::arg("blocks"), py::arg("block_size") = 32,
          py::arg("repeats") = 1,
          "Time the PUCT-shaped sibling scan — visit_count, value_sum and prior over a "
          "contiguous block of siblings — under both accumulators over identical data. The "
          "number this exists to produce is q32_over_double: whether the Q32 to double "
          "conversion disappears into the loop.");

    // -----------------------------------------------------------------------
    // C5 — search core on a replay evaluator.
    // -----------------------------------------------------------------------

    // The revision chess-library is pinned to, taken from the CMake variable
    // that pins it rather than restated here, so the pin test cannot pass
    // against a build of some other revision.
    m.attr("CHESS_LIBRARY_PIN") = std::string(GUOFISH_CHESS_LIBRARY_PIN);
    // C7. Pinned for a sharper reason than the other two: a tablebase probe is
    // a lookup whose ANSWER is under test, so a re-pin could change the WDL a
    // position reports with nothing in this repo changing.
    m.attr("FATHOM_PIN") = std::string(GUOFISH_FATHOM_PIN);

    py::class_<guofish::SearchConfig>(
        m, "SearchConfig",
        "The search knobs. Every default is core.mctsv4's, except c_factor, which does not "
        "exist in the reference at all: scope 3 adds it as a tunable and it defaults to 1.0, "
        "applied only when it is not 1.0 so the default path is bit-identical.")
        .def(py::init([](double c_init, double c_base, double c_factor, double fpu_root,
                         double fpu_tree, double virtual_loss, int max_tree_depth,
                         std::size_t arena_capacity, std::size_t cache_slots,
                         std::size_t cache_shards, double ponder_decay,
                         bool verify_compaction, float policy_temperature) {
                 guofish::SearchConfig config;
                 config.c_init = c_init;
                 config.c_base = c_base;
                 config.c_factor = c_factor;
                 config.fpu_root = fpu_root;
                 config.fpu_tree = fpu_tree;
                 config.virtual_loss = virtual_loss;
                 config.max_tree_depth = max_tree_depth;
                 config.arena_capacity = arena_capacity;
                 config.cache_slots = cache_slots;
                 config.cache_shards = cache_shards;
                 config.ponder_decay = ponder_decay;
                 config.verify_compaction = verify_compaction;
                 config.policy_temperature = policy_temperature;
                 return config;
             }),
             py::arg("c_init") = 1.43, py::arg("c_base") = 19652.0, py::arg("c_factor") = 1.0,
             py::arg("fpu_root") = 0.0, py::arg("fpu_tree") = 0.30,
             py::arg("virtual_loss") = 0.0, py::arg("max_tree_depth") = 80,
             py::arg("arena_capacity") = static_cast<std::size_t>(1u << 21),
             py::arg("cache_slots") = static_cast<std::size_t>(0),
             py::arg("cache_shards") = guofish::kDefaultCacheShards,
             py::arg("ponder_decay") = 1.0, py::arg("verify_compaction") = false,
             // C11b. APPENDED, not inserted. Every call site in the repo passes
             // keywords, but a positional one elsewhere would silently receive
             // this value in another field's slot, and a search running at the
             // wrong virtual loss because a parameter was added in the middle is
             // precisely the class of failure Gate 2b's first run was.
             py::arg("policy_temperature") = 1.0f)
        .def_readwrite("c_init", &guofish::SearchConfig::c_init)
        .def_readwrite("c_base", &guofish::SearchConfig::c_base)
        .def_readwrite("c_factor", &guofish::SearchConfig::c_factor)
        .def_readwrite("fpu_root", &guofish::SearchConfig::fpu_root)
        .def_readwrite("fpu_tree", &guofish::SearchConfig::fpu_tree)
        .def_readwrite("virtual_loss", &guofish::SearchConfig::virtual_loss)
        .def_readwrite("max_tree_depth", &guofish::SearchConfig::max_tree_depth)
        .def_readwrite("arena_capacity", &guofish::SearchConfig::arena_capacity)
        .def_readwrite("cache_slots", &guofish::SearchConfig::cache_slots,
                       "C7. Total transposition-cache entries. 0 is NO CACHE (the default, so "
                       "C5/C6's certified code path is unchanged); a cache that can hold "
                       "nothing is not constructible, because it would pass every "
                       "tree-equivalence test while doing no work.")
        .def_readwrite("cache_shards", &guofish::SearchConfig::cache_shards,
                       "C7. Spinlock-protected shards; a power of two, at least 64.")
        .def_readwrite("ponder_decay", &guofish::SearchConfig::ponder_decay,
                       "C8. The fraction of a PONDERED root's inherited visits that survives "
                       "promotion, in (0, 1]. 1.0 — the default — is no decay, which is what "
                       "the reference does and what Gate 1 across apply_move requires. Applied "
                       "only when apply_move is told the promotion is a ponder promotion; the "
                       "search cannot know that by itself. Visits AND value_sum are scaled by "
                       "the same ratio, so Q is unchanged and only the confidence behind it "
                       "falls.")
        .def_readwrite("verify_compaction", &guofish::SearchConfig::verify_compaction,
                       "C8. Run the full-tree structural diff after every compaction even with "
                       "asserts off. Builds with asserts on run it unconditionally. Scope §7 "
                       "names this as the mitigation for ping-pong fixup bugs, so it is engine "
                       "behaviour rather than a test fixture; the flag is how a Release build "
                       "opts in. Costs one extra O(nodes) pass per apply_move.")
        .def_readwrite("policy_temperature", &guofish::SearchConfig::policy_temperature,
                       "C11b. T in softmax(logits / T), applied at the ROOT and at every "
                       "interior node alike — there is exactly one live expansion path and "
                       "this is threaded into it, so the root/interior sharpness split the "
                       "reference has cannot reappear here. T < 1 sharpens, T > 1 flattens, "
                       "1.0 is the identity and skips the divide so the temperature-absent "
                       "path and the T = 1.0 path are the same instructions.\n\n"
                       "It DIVIDES rather than multiplying by a precomputed reciprocal: "
                       "cross-toolchain bit-identity (measured on MSVC 19.51 and Clang "
                       "18.1.3, C10a) is worth more than one division per logit.\n\n"
                       "A `float`, alone among these fields, because the reference divides a "
                       "float32 tensor by a Python float and torch's weak-scalar rule does "
                       "that in float32. A double would round twice and compute a different "
                       "number.\n\n"
                       "Must be > 0. Refused outright on a search with no live evaluator (the "
                       "replay dump's priors are already softmaxed, so it could only be "
                       "ignored) and in the GUOFISH_VALUE_SUM=double equivalence build.");

    // -----------------------------------------------------------------------
    // C9 — concurrency
    // -----------------------------------------------------------------------

    // Where the W search threads are pinned, as STRINGS rather than a bound
    // enum. See guofish::affinity_policy_from_name: `py::enum_` is the only
    // binding shape in this module that leaks its registration at import
    // (measured: 4,402 bytes, against zero without it), and README_BUILD.md's
    // leak discriminator — "no leaked allocation's stack should mention
    // guofish_core" — is worth more than an enum's ergonomics.
    //
    // Pinning is not a micro-optimisation on this machine. A worker parked on an
    // E-core by Windows' Thread Director holds the ROOT's contended atomics for
    // longer, and the root is touched by every descent and every backup, so one
    // slow thread in that path costs more than its own throughput. Measured at
    // 20,000 sims: +14.9% at W=4, +24.5% at W=6 (BENCH.md C9d).
    m.attr("AFFINITY_POLICIES") = py::make_tuple(
        "none",            // let the OS place the threads
        "pcore_physical",  // one thread per performance core, SMT siblings idle
        "pcore_smt",       // every logical processor of every performance core
        "all_logical");    // every logical processor, efficiency cores included

    py::class_<guofish::ParallelConfig>(
        m, "ParallelConfig",
        "W search threads, K in-flight paths each, and the dispatcher's batch ceiling.\n\n"
        "The defaults are not Python's. The reference ran 32 workers because it was GIL-bound "
        "and a worker spent almost all of its life blocked, so oversubscribing 16 hardware "
        "threads 2:1 cost nothing; C++ search threads actually run, and 32 of them would mean "
        "virtual loss held across preemption by descheduled threads. `max_batch` defaults to "
        "the measured GPU knee (BENCH.md, C9), which on this machine is set by HOST-side ATen "
        "dispatch rather than by the GPU: per-batch cost is flat at ~4 ms out to batch 64 and "
        "rises linearly after 128.")
        .def(py::init([](int workers, int in_flight, std::size_t max_batch,
                         const std::string &affinity, bool collect_histograms) {
                 guofish::ParallelConfig pc;
                 pc.workers = workers;
                 pc.in_flight = in_flight;
                 pc.max_batch = max_batch;
                 // Throws ValueError naming the valid policies; an unknown
                 // policy must not silently become "none", because the run
                 // would then report an affinity it did not apply.
                 pc.affinity = guofish::affinity_policy_from_name(affinity);
                 pc.collect_histograms = collect_histograms;
                 return pc;
             }),
             py::arg("workers") = 4, py::arg("in_flight") = 8,
             py::arg("max_batch") = static_cast<std::size_t>(128),
             py::arg("affinity") = "pcore_physical",
             py::arg("collect_histograms") = true)
        .def_readwrite("workers", &guofish::ParallelConfig::workers, "W: search threads.")
        .def_readwrite("in_flight", &guofish::ParallelConfig::in_flight,
                       "K: paths in flight per search thread. Every one of them holds virtual "
                       "loss on 10-30 nodes, which is the quality cost of parallelism and the "
                       "reason to minimise W*K rather than maximise it.")
        .def_readwrite("max_batch", &guofish::ParallelConfig::max_batch,
                       "The most leaves the dispatcher expands in one drain. A ceiling, not a "
                       "floor: there is no minimum-batch rule and no straggler timeout, which "
                       "is what removes both of the Python collector's measured pathologies.")
        .def_property(
            "affinity",
            [](const guofish::ParallelConfig &pc) {
                return std::string(guofish::affinity_policy_name(pc.affinity));
            },
            [](guofish::ParallelConfig &pc, const std::string &name) {
                pc.affinity = guofish::affinity_policy_from_name(name);
            },
            "One of guofish_core.AFFINITY_POLICIES. Reads back as the same string "
            "parallel_stats()['affinity'] reports, so a benchmark row and the configuration "
            "that produced it cannot disagree about what was asked for.")
        .def_readwrite("collect_histograms", &guofish::ParallelConfig::collect_histograms,
                       "Record every batch size and outstanding-leaf count at drain time "
                       "(scope §6.2). On by default; a few thousand int64s per search.")
        .def_property_readonly("max_outstanding", &guofish::ParallelConfig::max_outstanding,
                               "W * K — the leaf count the whole engine throttles on, and the "
                               "quantity scope §2.2 makes the batch-governing knob in place of "
                               "the worker count.");

    m.def(
        "race_probe",
        [](int iterations, int threads) {
            // A DELIBERATE DATA RACE. Nothing in the engine calls this and
            // nothing may: it exists so that "ThreadSanitizer reported nothing"
            // can be distinguished from "ThreadSanitizer was not looking".
            //
            // A clean sanitizer run is evidence only if the sanitizer is
            // demonstrably able to fail, which is the same argument the mutation
            // drill makes about golden data — a test that cannot fail is not a
            // test. So this is the positive control for the C9 acceptance run:
            // under a TSan build it must produce a report, and if it does not,
            // the clean run above proves nothing about the search.
            //
            // Plain `int`, no atomics, no lock, incremented from several
            // threads. That is undefined behaviour by construction and is the
            // point; it is quarantined behind an explicit call that only a test
            // makes.
            int counter = 0;
            std::vector<std::thread> pool;
            pool.reserve(static_cast<std::size_t>(threads));
            {
                py::gil_scoped_release release;
                for (int t = 0; t < threads; ++t) {
                    pool.emplace_back([&counter, iterations] {
                        for (int i = 0; i < iterations; ++i) {
                            counter = counter + 1;
                        }
                    });
                }
                for (std::thread &thread : pool) {
                    thread.join();
                }
            }
            return counter;
        },
        py::arg("iterations") = 10000, py::arg("threads") = 4,
        "SANITIZER SELF-TEST. Increments a plain int from several threads with no "
        "synchronisation, which is a data race by construction.\n\n"
        "Never called by the engine. It exists because a clean ThreadSanitizer run is "
        "evidence only if the sanitizer can be shown to fail — the same argument the "
        "mutation drill makes about golden data. Under a TSan build this must produce a "
        "report; if it does not, the clean C9 acceptance run proves nothing. The return "
        "value is the racy counter, which is also usually wrong, and that is expected.");

    m.def(
        "steady_now_ns",
        [] { return guofish::Q32ReplaySearch::steady_now_ns(); },
        "std::steady_clock::now() in nanoseconds — THE CLOCK THE MUTABLE DEADLINE USES.\n\n"
        "Exposed so a stop-latency measurement can be taken against the same clock the "
        "engine aborts on, rather than against time.monotonic(). The two are both monotonic "
        "and are NOT the same epoch on Windows, so a difference between one reading of each "
        "is meaningless while a difference between two readings of this is not.\n\n"
        "Nothing in the engine needs it: set_deadline_in() takes a duration precisely so "
        "that no caller has to name an instant in this epoch.");

    m.def(
        "deadline_race_probe",
        [](int iterations) {
            // C11c's positive control for the MUTABLE DEADLINE, and the reason
            // it is separate from race_probe above.
            //
            // race_probe proves TSan is looking at all. This one proves it is
            // looking at the shape of access the deadline actually has: one
            // thread storing a 64-bit integer while another polls it in a loop,
            // which is exactly ReplaySearch::set_deadline_ns against
            // ReplaySearch::deadline_passed. Written with a PLAIN int64 rather
            // than an atomic, so under TSan it must report — and if it does
            // not, a clean run over the real deadline says nothing, because the
            // sanitizer was not able to see that pattern.
            //
            // Quarantined behind an explicit call. Nothing in the engine calls
            // it and nothing may.
            std::int64_t deadline = 0;
            std::int64_t observed = 0;
            {
                py::gil_scoped_release release;
                std::thread poller([&deadline, &observed, iterations] {
                    for (int i = 0; i < iterations; ++i) {
                        observed += deadline;
                    }
                });
                for (int i = 0; i < iterations; ++i) {
                    deadline = i;
                }
                poller.join();
            }
            return observed;
        },
        py::arg("iterations") = 200000,
        "SANITIZER SELF-TEST for the C11c mutable deadline. One thread writes a plain "
        "int64 while another reads it in a loop — the unsynchronised version of "
        "set_deadline_ns() against the workers' abort-point poll.\n\n"
        "race_probe proves ThreadSanitizer is looking; this proves it is looking at THIS "
        "access pattern. A TSan build must report on it. Never called by the engine.");

    py::class_<GilProbe>(
        m, "GilProbe",
        "Measures how long the dispatcher waits to acquire the GIL, once per batch.\n\n"
        "This tests a PREDICTION rather than collecting a number. C0b established that the "
        "pybind11 boundary itself costs ~60 ns uncontended, so all of the GIL cost scope §2.1 "
        "projects is contention. Under §2.1's discipline the only Python-relevant threads "
        "during a search are the dispatcher and a main thread blocked on stdin — which "
        "releases the GIL while blocked — so acquisition should stay near the uncontended "
        "floor. An excursion toward milliseconds means Python bytecode is running during the "
        "search that should not be, and the answer is to move it to C++ rather than to tune "
        "around it.\n\n"
        "Install with search.set_batch_hook(probe); read the per-batch waits out of "
        "parallel_stats()['hook_wait_ns_samples'].")
        .def(py::init<>())
        .def_property_readonly("acquisitions", &GilProbe::acquisitions)
        .def_property_readonly("rows", &GilProbe::rows,
                               "Total leaves across every batch the probe saw.");

    // -----------------------------------------------------------------------
    // C10 — the live evaluator
    // -----------------------------------------------------------------------

    m.attr("POLICY_SIZE") = guofish::kPolicySize;
    m.attr("DEFAULT_SWITCH_INTERVAL") = 0.0005;

    py::class_<LiveEvaluator>(
        m, "LiveEvaluator",
        "The three zero-copy buffers of scope §2.1, plus the one Python callable that turns "
        "tokens into logits.\n\n"
        "    input_buffer   [max_batch, 68]    int32    C++ writes, Python reads\n"
        "    policy_buffer  [max_batch, 4096]  bf16     Python writes, C++ reads\n"
        "    value_buffer   [max_batch]        float32  Python writes, C++ reads\n\n"
        "All three are C++-owned and exposed as NumPy views over the same memory; nothing is "
        "copied at the boundary and no 4096-wide row is ever materialised per leaf — the "
        "gather pulls only the ~26-38 legal logits it needs.\n\n"
        "`policy_view()` has dtype uint16 because NumPy has no bfloat16. The values ARE bf16 "
        "bit patterns: view it as `torch.bfloat16` (same 2-byte width) and copy the model's "
        "bf16 logits straight in. Do not convert through float32 — matching Python's dtype is "
        "what keeps the production gather numerically close to the reference's, which is what "
        "Gate 2 measures.\n\n"
        "CONSTRUCTING ONE SETS sys.setswitchinterval. That is a process-global interpreter "
        "setting and it is done anyway, in the constructor, because 'before any search thread "
        "spawns' is a requirement settled by measurement: at the default 5 ms interval C0b "
        "clocked the dispatcher's contended acquire wait at a median of 15.25 ms on Windows — "
        "a ceiling of ~60 batches/s that reads as a slow network rather than as a lock. At "
        "0.0005 the same measurement is a p99 of 78 us. `switch_interval_before` reports what "
        "was overwritten.")
        .def(py::init<std::size_t, py::object, double>(), py::arg("max_batch"),
             py::arg("callback"), py::arg("switch_interval") = 0.0005,
             "`callback(count)` is called once per batch, with the GIL held, on a "
             "C++-created thread. It must read rows [0, count) of input_buffer and write rows "
             "[0, count) of policy_buffer and value_buffer.\n\n"
             "KEEP IT MINIMAL — a handful of torch calls on pre-allocated tensors, no Python "
             "loops. Torch releases the GIL inside CUDA dispatch, but tensor construction, "
             "`.to()` and indexing hold it, and every microsecond held is a microsecond the "
             "dispatcher is not launching kernels.\n\n"
             "`switch_interval` <= 0 leaves the interpreter's setting alone. That exists for "
             "the C10 measurement of the setting's own cost (BENCH.md) and for nothing else; "
             "production takes the default.")
        .def_property_readonly("max_batch", &LiveEvaluator::max_batch)
        .def("input_view", &LiveEvaluator::input_view,
             "int32 [max_batch, 68], zero-copy. C++ writes it; read it, do not write it.")
        .def("policy_view", &LiveEvaluator::policy_view,
             "uint16 [max_batch, 4096] holding bf16 bit patterns, zero-copy. Write it.")
        .def("value_view", &LiveEvaluator::value_view,
             "float32 [max_batch], zero-copy. Write it with the network's ABSOLUTE "
             "(White-POV) value; the side-to-move flip is search's job.")
        .def("buffer_spans", &LiveEvaluator::buffer_spans,
             "{name: (address, nbytes)} for a host that wants to page-lock the buffers.\n\n"
             "Pinning C++-owned memory needs cudaHostRegister, and calling it from this module "
             "would make the CUDA runtime a build dependency of an extension that is "
             "deliberately torch-free. Torch already exposes cudart to Python, so the host "
             "registers and hands back the matching unregister via set_teardown().")
        .def("set_teardown", &LiveEvaluator::set_teardown, py::arg("hook"),
             "A callable invoked once, with the GIL held, immediately BEFORE the buffers are "
             "freed. This is where a host that page-locked them unregisters; doing it later "
             "leaves CUDA holding a mapping onto memory the allocator has taken back.")
        .def_property_readonly("switch_interval", &LiveEvaluator::switch_interval,
                               "The interpreter switch interval this evaluator set.")
        .def_property_readonly("switch_interval_before", &LiveEvaluator::switch_interval_before,
                               "What sys.getswitchinterval() returned before it did.");

    m.def("gather_softmax", &gather_softmax, py::arg("fen"), py::arg("policy_row"),
          py::arg("temperature") = 1.0f,
          "Gate 2's probe: (canonical UCI moves, priors) for one position and one recorded "
          "4096-wide bf16 policy row (dtype uint16).\n\n"
          "This is the PRODUCTION gather, not a restatement of it — the same function the "
          "dispatcher calls, over the same generation-order index. Softmax over "
          "GENERATION-order logits, then permute the probabilities into canonical order; never "
          "gather in sorted order. `torch.softmax` is not permutation-invariant (scope §2.6 "
          "measured 109/200 permutations reproducing bit-identical priors, max delta 3e-7), so "
          "sorting before the softmax changes the priors on a large fraction of nodes.\n\n"
          "C11b. `temperature` divides the logits before the softmax and defaults to 1.0, "
          "which skips the divide entirely — so the default call is bit-identical to the "
          "pre-C11b one and Gate 2's recorded figures stand without being re-measured. "
          "Passing a temperature here drives the SAME code the search drives, which is what "
          "lets the Gate 2 extension over T in {0.7, 1.0, 1.5} be a statement about the "
          "engine rather than about a test helper.");

    m.def("generation_order", &generation_order, py::arg("fen"),
          "The position's legal moves as normalised UCI, in chess-library's GENERATION "
          "order.\n\n"
          "For Gate 2 only. Reducing ATen's softmax in this order isolates the "
          "softmax-implementation difference from the permutation variance that comes of "
          "python-chess and chess-library generating in different orders — two causes, only "
          "one of which is this port's to answer for. Nothing in the search reads it: "
          "reproducing a library's internal generation order is exactly the fragile dependency "
          "canonical ordering was chosen to avoid.");

    // -----------------------------------------------------------------------
    // C7 — cache and tablebase
    // -----------------------------------------------------------------------

    m.attr("CACHE_MIN_SHARDS") = guofish::kMinCacheShards;
    m.attr("CACHE_DEFAULT_SHARDS") = guofish::kDefaultCacheShards;
    m.attr("MAX_LEGAL_MOVES") = guofish::kMaxLegalMoves;
    m.attr("TABLEBASE_MAX_PIECES") = guofish::kTablebaseMaxPieces;

    m.def("cache_type_separation", &cache_type_separation,
          "The compile-time facts of C7's anti-poisoning requirement, as a dict.\n\n"
          "Every entry is evaluated by the compiler against the REAL "
          "TranspositionCache::insert through real overload resolution — not against a "
          "restatement of its signature that could drift. `network_value_accepted` must be "
          "True and every other `*_accepted` must be False; a False in the first or a True "
          "in the rest means a terminal value, a proof or a tablebase result can be stored "
          "in a position-keyed cache, which is the defect this chunk exists to prevent.\n\n"
          "The same predicates are static_asserts in cpp/cache.hpp, so a violation stops the "
          "build. This function exists because a build that stops gives an acceptance test "
          "nothing to point at.");

    m.def("eval_row", &eval_row, py::arg("fen"),
          "The 68-token evaluator input row for `fen` AND the nn_key computed from exactly "
          "those tokens, as one dict.\n\n"
          "They come from one guofish::EvalRow, which is the whole point: the cache key must "
          "be a function of the bytes handed to the network, never of a second derivation "
          "from the board that happens to agree today (scope 2.5, and the C7 brief's "
          "'Risks / NNKey Generation').");

    m.def("nn_key_of_tokens", &nn_key_of_tokens, py::arg("tokens"),
          "The nn_key for a caller-supplied 68-token row. Perturb one token and the key "
          "moves; that is how 'the key is a function of the tokens' is checked rather than "
          "asserted.");

    py::class_<guofish::CacheStats>(m, "CacheStats", "Counters for one TranspositionCache.")
        .def_readonly("hits", &guofish::CacheStats::hits)
        .def_readonly("misses", &guofish::CacheStats::misses)
        .def_readonly("inserts", &guofish::CacheStats::inserts)
        .def_readonly("collisions", &guofish::CacheStats::collisions,
                      "Inserts that displaced a DIFFERENT key — the direct-mapped table's "
                      "eviction count. Distinguishes 'no transpositions in this workload' "
                      "from 'the table is too small'.")
        .def_readonly("refreshes", &guofish::CacheStats::refreshes,
                      "Inserts that rewrote the SAME key with the same payload. Counted "
                      "separately so it cannot inflate `collisions`.")
        .def_property_readonly("hit_rate", &guofish::CacheStats::hit_rate);

    py::class_<guofish::TranspositionCache>(
        m, "TranspositionCache",
        "The NN evaluation cache, standalone. Sharded (>= 64 shards, spinlock each), "
        "direct-mapped within a shard, keyed by nn_key.\n\n"
        "It holds a value, the gathered legal priors and the move list those priors are "
        "aligned with. It CANNOT hold a terminal mark, a proof or a tablebase result: "
        "insert() takes a network value and there is no conversion from the others. See "
        "cache_type_separation().")
        .def(py::init<std::size_t, std::size_t>(), py::arg("slots"),
             py::arg("shards") = guofish::kDefaultCacheShards,
             "`slots` is rounded up to a power of two per shard. `slots == 0` raises: "
             "'cache off' is expressed by not having a cache, because a cache that can hold "
             "nothing passes every tree-equivalence test while doing no work.")
        .def_property_readonly("capacity", &guofish::TranspositionCache::capacity)
        .def_property_readonly("shard_count", &guofish::TranspositionCache::shard_count)
        .def_property_readonly("slots_per_shard", &guofish::TranspositionCache::slots_per_shard)
        .def_property_readonly("size", &guofish::TranspositionCache::size,
                               "Occupied slots. Walks the whole table under each shard's "
                               "lock; a diagnostic, not a hot-path call.")
        .def_property_readonly("stats", &guofish::TranspositionCache::stats)
        .def("clear", &guofish::TranspositionCache::clear)
        .def(
            "insert",
            [](guofish::TranspositionCache &self, std::uint64_t key, double value,
               const std::vector<std::string> &moves, const std::vector<float> &priors) {
                if (moves.size() != priors.size()) {
                    throw py::value_error(
                        "guofish_core: moves and priors must be the same length");
                }
                std::vector<std::uint16_t> packed;
                packed.reserve(moves.size());
                for (const std::string &uci : moves) {
                    packed.push_back(packed_from_uci(uci));
                }
                // The bare double arrives from Python and is wrapped HERE, at
                // the language boundary, which is the only place a wrap is
                // legitimate — Python has no type system to carry the
                // distinction across. Inside C++ the wrap is what `insert`
                // demands and cannot be bypassed.
                self.insert(guofish::NNKey(key), guofish::NetworkValue(value), packed.data(),
                            priors.data(), static_cast<std::uint16_t>(packed.size()));
            },
            py::arg("nn_key"), py::arg("value"), py::arg("moves"), py::arg("priors"),
            "Store one evaluation. `moves` are UCI strings in canonical order and `priors` "
            "is aligned with them.")
        .def(
            "probe",
            [](guofish::TranspositionCache &self, std::uint64_t key) -> py::object {
                guofish::CachedEval entry;
                if (!self.probe(guofish::NNKey(key), entry)) {
                    return py::none();
                }
                py::list moves;
                for (const std::uint16_t packed : entry.moves) {
                    moves.append(move_to_uci(packed));
                }
                py::array_t<float> priors(static_cast<py::ssize_t>(entry.priors.size()));
                std::copy(entry.priors.begin(), entry.priors.end(), priors.mutable_data());

                py::dict d;
                d["value"] = entry.value.value;
                d["moves"] = moves;
                d["priors"] = priors;
                return d;
            },
            py::arg("nn_key"), "The entry for `nn_key`, or None. Counts as a hit or a miss.");

    py::class_<guofish::TablebaseProber>(
        m, "TablebaseProber",
        "Base class for a Syzygy backend. Three exist: FathomProber (native, production), "
        "PythonProber (a Python callable, for cross-checking against the reference) and "
        "NullProber (misses on everything, which is what 'tablebases off' means "
        "concretely).")
        .def_property_readonly("backend", &guofish::TablebaseProber::backend);

    py::class_<guofish::FathomProber, guofish::TablebaseProber>(
        m, "FathomProber",
        "The native Syzygy backend, on jdart1/Fathom (pinned; see FATHOM_PIN).\n\n"
        "ONE PER PROCESS. Fathom keeps its state in file scope — tb_init and tb_free take "
        "no handle — so constructing a second one while the first is alive raises rather "
        "than silently sharing and double-freeing its tables. Hold one and hand it to "
        "whoever needs it, exactly as the reference hands one chess.syzygy.Tablebase to the "
        "UCI layer and to every search.\n\n"
        "Five places Fathom and python-chess disagree — the WDL scale, the halfmove-clock "
        "guard on tb_probe_wdl, castling rights, the DTZ sign, and which entry point DTZ "
        "comes from — are reconciled in cpp/fathom.hpp and MEASURED against the reference "
        "over the real tables by tests/test_c7_cache.py.\n\n"
        "probe_dtz is NOT thread safe (tb_probe_root is not). It is used only by mode 1, "
        "which runs once per move at the UCI layer. Mode 2, the one on the search path, "
        "needs WDL alone, and tb_probe_wdl IS thread safe.")
        .def(py::init<std::string>(), py::arg("path"),
             "Open the tables under `path`. Raises if they cannot be opened, if the "
             "directory holds no tables at all (which would otherwise give a prober that "
             "reports itself open and misses on everything), or if another FathomProber is "
             "already alive.")
        .def_property_readonly("largest", &guofish::FathomProber::largest,
                               "The most men any loaded table covers — Fathom's TB_LARGEST. "
                               "assets/syzygy is a 5-man set.")
        .def_property_readonly("path", &guofish::FathomProber::path)
        .def(
            "probe_wdl",
            [](const guofish::FathomProber &self, std::string_view fen) -> py::object {
                const guofish::ParsedFen parsed = guofish::parse_fen(fen);
                chess::Board board;
                guofish::set_fen_or_throw(board, fen);
                const auto wdl =
                    self.probe_wdl(parsed, static_cast<int>(board.halfMoveClock()));
                return wdl.has_value() ? py::cast(*wdl) : py::none();
            },
            py::arg("fen"),
            "Raw Syzygy WDL in [-2, +2] from the side to move's perspective, or None on a "
            "miss. Clock-independent, matching python-chess's probe_wdl — see "
            "cpp/fathom.hpp note 2 for why that is correct here and is not the "
            "cache-poisoning defect returning.")
        .def(
            "probe_dtz",
            [](const guofish::FathomProber &self, std::string_view fen) -> py::object {
                const guofish::ParsedFen parsed = guofish::parse_fen(fen);
                chess::Board board;
                guofish::set_fen_or_throw(board, fen);
                const auto dtz =
                    self.probe_dtz(parsed, static_cast<int>(board.halfMoveClock()));
                return dtz.has_value() ? py::cast(*dtz) : py::none();
            },
            py::arg("fen"),
            "SIGNED distance to zero from the side to move's perspective, or None on a "
            "miss — positive when winning, negative when losing, matching python-chess's "
            "probe_dtz. Fathom reports a magnitude; the sign is restored from the WDL.");

    py::class_<guofish::NullProber, guofish::TablebaseProber>(
        m, "NullProber",
        "A backend that misses on everything: what 'tablebases off' means concretely. Exists "
        "so the probe path — piece-count gate, probe, miss, keep the neural value — can be "
        "exercised in a build with no tables, which is the path production takes on every "
        "position with six or more men.")
        .def(py::init<>());

    py::class_<PythonProber, guofish::TablebaseProber>(
        m, "PythonProber",
        "A backend backed by a Python callable: fen -> (wdl, dtz) or None.\n\n"
        "`chess.syzygy.open_tablebase(...)` wrapped in three lines is exactly such a "
        "callable, so this is what lets modes 1 and 2 be tested against the REAL 5-piece "
        "tables — real WDL, real DTZ, real perspective — without vendoring a decoder. The "
        "FEN it passes carries the search's own raw en-passant square and halfmove clock, so "
        "the position Python reconstructs is the one the search is standing on.\n\n"
        "It calls into Python and therefore needs the GIL. Every caller today holds it; C9's "
        "threads will not, and a native backend rather than a GIL acquisition in a leaf "
        "probe is the answer there.")
        .def(py::init<py::function>(), py::arg("probe"))
        .def_property_readonly("calls", &PythonProber::calls,
                               "How many times the callable was invoked. A probe of one "
                               "position costs two calls (WDL then DTZ) in mode 1 and one "
                               "(WDL) in mode 2.");

    m.def(
        "wdl_to_value", &guofish::wdl_to_value, py::arg("wdl"),
        "The reference's wdl_to_value: wdl / 2.0, so +-2 -> +-1.0 (decisive), +-1 -> +-0.5 "
        "(cursed win / blessed loss, treated as half-decisive), 0 -> 0.0. The +-1 endpoints "
        "match the bounded range the value head produces, so a tablebase result does not "
        "read as out-of-distribution to PUCT.");

    m.def(
        "tablebase_probe_value",
        [](std::string_view fen, const guofish::TablebaseProber &prober) -> py::object {
            const guofish::ParsedFen parsed = guofish::parse_fen(fen);
            chess::Board board;
            guofish::set_fen_or_throw(board, fen);
            const auto value = guofish::probe_tablebase_value(
                prober, parsed, static_cast<int>(board.halfMoveClock()));
            return value.has_value() ? py::cast(value->value) : py::none();
        },
        py::arg("fen"), py::arg("prober"),
        "Mode 2's value for `fen` in ABSOLUTE (White) perspective, or None on a miss. The "
        "perspective conversion happens here rather than at the backup site, which is the "
        "reference's decision and its own docstring's reason: the backup site 'is exactly "
        "where tablebase perspective bugs hide'.");

    m.def(
        "within_tablebase_range",
        [](std::string_view fen) {
            return guofish::within_tablebase_range(guofish::parse_fen(fen));
        },
        py::arg("fen"),
        "The piece-count gate: <= TABLEBASE_MAX_PIECES men, both kings counted. A popcount "
        "that filters out the overwhelming majority of leaves before any probe cost.");

    m.def(
        "piece_count",
        [](std::string_view fen) { return guofish::piece_count(guofish::parse_fen(fen).placement); },
        py::arg("fen"), "Men on the board, both kings included.");

    m.def(
        "tablebase_root_score",
        [](bool mate, bool drawn_terminal, bool zeroing, int wdl_child, int dtz_child) {
            const guofish::TablebaseRootScore score =
                guofish::tablebase_root_score(mate, drawn_terminal, zeroing, wdl_child, dtz_child);
            return py::make_tuple(score.outcome, score.distance);
        },
        py::arg("mate"), py::arg("drawn_terminal"), py::arg("zeroing"), py::arg("wdl_child"),
        py::arg("dtz_child"),
        "Mode 1's per-move ranking pair (outcome, distance), transcribed from "
        "playing/uci_wrapper.py::_probe_tablebase. The engine maximises (outcome, -distance). "
        "Exposed as a pure function so the ranking can be checked on numbers rather than only "
        "through a board.");

    m.def(
        "tablebase_root_move",
        [](std::string_view fen, const guofish::TablebaseProber &prober) -> py::object {
            guofish::SearchBoard board;
            board.set_fen(fen);
            const std::optional<std::uint16_t> move = guofish::tablebase_root_move(board, prober);
            return move.has_value() ? py::cast(move_to_uci(*move)) : py::none();
        },
        py::arg("fen"), py::arg("prober"),
        "Mode 1: the tablebase-optimal move for a <= 5-man root, or None on any miss (out of "
        "range, no tables, or a table the backend cannot answer from). None means 'fall "
        "through to MCTS', which is the reference's behaviour and the only safe one.\n\n"
        "Runs on a SearchBoard, so the position handed to the backend carries the RAW "
        "en-passant square rather than chess-library's filtered one.");

    m.def("c_puct", &guofish::c_puct, py::arg("parent_visits"), py::arg("c_init") = 1.43,
          py::arg("c_base") = 19652.0,
          "AlphaZero's visit-scaled exploration constant, c(N) = c_init + "
          "log((N + c_base + 1) / c_base), associated exactly as Python associates it. "
          "Exposed so an equivalence harness can put the SAME log on both sides of a "
          "comparison, rather than discovering that CPython's libm and this translation "
          "unit's disagree in the last ulp.");

    m.def("ep_pin_probe", &ep_pin_probe, py::arg("fen"), py::arg("uci"),
          "What each en-passant convention says about one (position, move) pair: the FEN's "
          "own field, what chess-library kept after setFen, what makeMove<false> and "
          "makeMove<true> set, and the RAW square the search derives. They disagree, which "
          "is why the search carries its own. tests/test_c5_ep_pin.py pins the library's "
          "answers at CHESS_LIBRARY_PIN.");

    // -----------------------------------------------------------------------
    // C6 — the terminal rules, one predicate at a time.
    // -----------------------------------------------------------------------

    m.def("terminal_reason", &terminal_reason, py::arg("fen"),
          "python-chess's Board.outcome() reason for this FEN — 'none', 'checkmate', "
          "'insufficient-material', 'stalemate', 'seventy-five-moves' — in the reference's "
          "own test order (insufficient material is asked BEFORE stalemate). Fivefold "
          "repetition is a property of the game and not of the FEN, so it is answered False "
          "here, exactly as python-chess answers it for a board with no move stack; the "
          "search evaluates it against its own path instead.");

    m.def("insufficient_material", &insufficient_material, py::arg("fen"),
          "Board.is_insufficient_material(): neither side can force a mate. Transcribed "
          "including the quirk that the bishop clause tests BOTH colours' bishops for the "
          "same-colour-complex question.");

    m.def("move_rule_probe", &move_rule_probe, py::arg("fen"), py::arg("uci"),
          "Board.is_zeroing / _reduces_castling_rights / has_legal_en_passant / "
          "is_irreversible for one legal move, on the NORMALISED UCI destination. These are "
          "what the halfmove clock and the repetition walk-back are built out of.");

    bind_replay_search<guofish::DoubleAccumulator>(
        m, "ReplaySearchDouble",
        "The Gate 1 equivalence search: value_sum in atomic<double>, so C++ reproduces "
        "Python's float arithmetic exactly. Single-threaded; the evaluator is a lookup into "
        "the golden dump, never a network.");

    bind_replay_search<guofish::Q32Accumulator>(
        m, "ReplaySearchQ32",
        "The same search over the production Q32 accumulator. Bound so the fixed-point "
        "instantiation cannot rot behind an #ifdef; it does NOT reproduce Python's float "
        "arithmetic and is not what Gate 1 compares.");

#if defined(GUOFISH_VALUE_SUM_DOUBLE)
    m.attr("ReplaySearch") = m.attr("ReplaySearchDouble");
#else
    m.attr("ReplaySearch") = m.attr("ReplaySearchQ32");
#endif

    m.def("tokenize_bench", &tokenize_bench, py::arg("fens"), py::arg("repeats") = 1,
          "Single-threaded tokenization throughput over `fens`, repeated `repeats` times. The "
          "FENs are copied into C++ strings first and the timed region touches no Python "
          "object, so this measures the encoder and not the language boundary.");

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
