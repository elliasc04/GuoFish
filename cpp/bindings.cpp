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

#include "keys.hpp"
#include "movegen.hpp"
#include "tokens.hpp"

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

}  // namespace

PYBIND11_MODULE(guofish_core, m) {
    m.doc() = "GuoFish C++ core — C0 toolchain spike, C0b GIL contention probe, C1 movegen, C2 tokenizer";

    m.attr("SEQ_LENGTH") = guofish::kSeqLength;

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
