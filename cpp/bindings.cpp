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
#include "keys.hpp"
#include "movegen.hpp"
#include "tokens.hpp"

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
