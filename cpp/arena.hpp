// GuoFish C4 — the struct-of-arrays node arena.
//
// This header holds no search logic. It holds the *shape* of the tree, and the
// shape is the part that is expensive to change later: C5 traverses it, C8
// compacting-copies it, and C9 runs eight threads over it. Getting the layout
// wrong is not a bug that shows up as a wrong answer — it shows up as an engine
// that is three times slower than it should be, or as a torn read that appears
// once an hour under load.
//
//
// WHY STRUCT-OF-ARRAYS
// -------------------
// PUCT selection scans *every* sibling of a node, and that is the hottest loop
// in the engine. Python's array-of-structs node is 33 B of payload that pads to
// 40 B under the double's alignment, so scanning 32 siblings strides 40 B and
// touches ~20 cache lines, of which it uses three fields' worth. Here the same
// scan is three sequential, prefetchable streams — `visit_count`, `value_sum`,
// `prior` — over ~10 lines total, and the hardware prefetcher can see them
// coming.
//
// SoA also dissolves the alignment question rather than answering it. Each
// array is a separate over-aligned allocation, so no atomic can be
// under-aligned by struct padding, by `#pragma pack` (forbidden by Global
// Rule 6 anyway), or by stride arithmetic in an allocator. That matters because
// the failure is silent and enormous: a `lock xadd` whose operand straddles a
// cache line escalates to a **bus lock**, roughly 100x the cost of the aligned
// form, and MSVC's response to an under-aligned atomic is torn access and UB
// rather than a fallback lock. The static_asserts and the runtime alignment
// checks below are the net under that.
//
//
// WHY TWO ACCUMULATORS
// -------------------
// `value_sum` is templated on an accumulator policy rather than fixed:
//
//   Q32Accumulator     atomic<int64_t> holding fixed-point value * 2^32.
//                      Production. `fetch_add` is one `lock xadd` with no retry
//                      loop, and integer addition is *associative*, so the
//                      multithreaded engine becomes bit-reproducible — a
//                      property the Python engine never had.
//   DoubleAccumulator  atomic<double>. The Gate 1 equivalence build, where the
//                      C++ tree must match Python's float arithmetic exactly.
//                      Addition is a CAS loop, and it is not associative, so
//                      this one is only reproducible single-threaded.
//
// Both are instantiated in every build (see the explicit instantiations at the
// bottom) so neither can rot behind an `#ifdef`, and both are exercised by
// tests/test_c4_arena.py in one process. GUOFISH_VALUE_SUM_DOUBLE selects which
// one `DefaultArena` names, i.e. which one the engine will use.
//
//
// WHY TERMINAL IS A BIT AND NOT A STATE
// ------------------------------------
// The Python engine could produce `bestmove 0000`: a node that reported itself
// expanded while having no children, so the move selector had nothing to pick
// from. Here that is not a bug to be fixed but a state that cannot be spelled.
// `set_children()` refuses a count of zero and refuses a terminal node;
// `mark_terminal()` refuses a node that already has children. So
// "expanded with no children" and "terminal and expanded" are both
// unrepresentable, and the invariant is checked on the way in rather than
// asserted about afterwards.

#ifndef GUOFISH_ARENA_HPP
#define GUOFISH_ARENA_HPP

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace guofish {

// ---------------------------------------------------------------------------
// Q32 fixed point
//
// A value in [-1, 1] is stored as round(v * 2^32) in an int64. Three properties
// are load-bearing, and all three are why this is not simply "a scaled int":
//
//   * The scale is a power of two, so `q * 2^-32` is exact in IEEE-754 double
//     for every q with |q| <= 2^53, and the int -> double -> int round trip is
//     exact over the whole representable range. Nothing here is approximate in
//     the direction that would accumulate.
//   * Resolution is 2^-32 = 2.3283e-10. The network emits bf16, which carries 8
//     mantissa bits; this resolves seven orders of magnitude finer than its own
//     input, so the quantization is free in the only sense that matters.
//   * Overflow needs sum > 2^63, i.e. 2^31 = 2.1e9 visits at |v| = 1. The
//     engine budgets 2-3M nodes and 15k sims per move.
// ---------------------------------------------------------------------------

// 2^32, as both the double scale factor and the int64 image of 1.0.
inline constexpr double kQ32Scale = 4294967296.0;
inline constexpr std::int64_t kQ32One = INT64_C(4294967296);

// 2^-32 = 2.3283064365386963e-10. Exact; this is a power of two.
inline constexpr double kQ32Resolution = 1.0 / kQ32Scale;

// v -> Q32. Rounds half away from zero, because that is what std::llround is
// defined to do and the rule has to be identical on MSVC and Clang for the
// golden comparisons in C5 onward to mean anything. Ties are reachable: a float
// in [2^-10, 2^-9) has an ulp of 2^-33, so v * 2^32 lands exactly on k + 0.5.
//
// The domain is [-1, 1] — a network value, not a sum. Sums are accumulated in
// Q32 and converted back with from_q32(), which has no domain restriction.
inline std::int64_t to_q32(double v) {
    // NaN has no fixed-point image and would convert to an unspecified integer.
    assert(!(v != v));
    assert(v >= -1.0 && v <= 1.0);
    return std::llround(v * kQ32Scale);
}

// Q32 -> double. Exact for |q| <= 2^53, which covers every single value and
// every sum the engine can reach before int64 itself overflows... up to 2^53,
// after which the *double* loses precision while the int64 sum stays exact.
// That is 2^21 = 2.1M visits at |v| = 1, so a long-running root's Q is read
// back with slightly less precision than it is stored. Deliberate: the
// alternative is to keep the sum in double and lose associativity, which is the
// whole reason for Q32.
inline constexpr double from_q32(std::int64_t q) noexcept {
    return static_cast<double>(q) * kQ32Resolution;
}

// ---------------------------------------------------------------------------
// Packed moves
//
// 16 bits: from(6) << 10 | to(6) << 4 | promo(4), with squares numbered as
// chess-library numbers them (rank * 8 + file, a1 = 0).
//
// CAREFUL: a raw integer sort of this packing is NOT the canonical
// `(from, to, promotion)` order C1 established. Square indices are rank-major
// (a1, b1, ... h1, a2) while UCI string order is file-major (a1, a2, ... a8,
// b1). The two disagree on almost every position, and because PUCT resolves
// ties by child order, sorting on the wrong one does not fail loudly — C++ and
// Python simply explore different moves at equal priors. Sort on
// canonical_move_key() instead; see C1 in DECISIONS.md for the full argument.
// ---------------------------------------------------------------------------

enum class Promotion : std::uint16_t {
    None = 0,
    Bishop = 1,
    Knight = 2,
    Queen = 3,
    Rook = 4,
};

// Alphabetical by UCI promotion letter (b < n < q < r), NOT by piece value, and
// None is 0 so a four-character move sorts before any five-character move with
// the same from/to. That makes the promotion field of a packed move already in
// canonical order, so canonical_move_key() can pass it through unchanged.
inline constexpr char promotion_letter(Promotion p) noexcept {
    switch (p) {
        case Promotion::Bishop: return 'b';
        case Promotion::Knight: return 'n';
        case Promotion::Queen: return 'q';
        case Promotion::Rook: return 'r';
        case Promotion::None: break;
    }
    return '\0';
}

inline constexpr int kSquareCount = 64;
inline constexpr std::uint16_t kPromotionCount = 5;

// The packing has no "no move" encoding to spare — 0 is a1a1, which is not a
// legal move but is a legal *pattern*. The root's move slot holds this.
inline constexpr std::uint16_t kNoMove = 0;

inline std::uint16_t pack_move(int from, int to, Promotion promo) {
    if (from < 0 || from >= kSquareCount || to < 0 || to >= kSquareCount) {
        throw std::invalid_argument("guofish::pack_move: square out of range [0, 64)");
    }
    if (static_cast<std::uint16_t>(promo) >= kPromotionCount) {
        throw std::invalid_argument("guofish::pack_move: promotion code out of range [0, 5)");
    }
    return static_cast<std::uint16_t>((static_cast<unsigned>(from) << 10) |
                                      (static_cast<unsigned>(to) << 4) |
                                      static_cast<unsigned>(promo));
}

inline constexpr int move_from(std::uint16_t packed) noexcept {
    return static_cast<int>((packed >> 10) & 0x3Fu);
}

inline constexpr int move_to(std::uint16_t packed) noexcept {
    return static_cast<int>((packed >> 4) & 0x3Fu);
}

inline constexpr Promotion move_promotion(std::uint16_t packed) noexcept {
    return static_cast<Promotion>(packed & 0x0Fu);
}

// A sort key whose unsigned integer order is exactly the byte order of the
// move's UCI string, i.e. C1's canonical (from, to, promotion) order.
//
//   from_file(3) << 13 | from_rank(3) << 10 | to_file(3) << 7 |
//   to_rank(3) << 4 | promo(4)
//
// A UCI move is [file][rank][file][rank][promo?] over a fixed ASCII alphabet
// where 'a'..'h' and '1'..'8' are both ascending, so comparing those five
// fields in that order is the same comparison std::string makes byte by byte —
// including the shorter-first rule, which falls out of promo 0 being lowest.
inline constexpr std::uint16_t canonical_move_key(std::uint16_t packed) noexcept {
    const unsigned from = (packed >> 10) & 0x3Fu;
    const unsigned to = (packed >> 4) & 0x3Fu;
    const unsigned promo = packed & 0x0Fu;
    const unsigned from_file = from & 7u;
    const unsigned from_rank = from >> 3;
    const unsigned to_file = to & 7u;
    const unsigned to_rank = to >> 3;
    return static_cast<std::uint16_t>((from_file << 13) | (from_rank << 10) | (to_file << 7) |
                                      (to_rank << 4) | promo);
}

// ---------------------------------------------------------------------------
// Node state
//
// Two independent things live in one atomic byte: a three-valued lifecycle in
// the low bits, and a terminal flag in the high bit. They are independent
// because they answer different questions — "has anyone expanded this yet" and
// "does the game end here" — and collapsing them into one enum is exactly how
// `bestmove 0000` happened.
// ---------------------------------------------------------------------------

enum class NodeState : std::uint8_t {
    Unexpanded = 0,
    Pending = 1,   // one thread has claimed this leaf and is evaluating it
    Expanded = 2,  // children_offset / children_count are published and valid
};

inline constexpr std::uint8_t kLifecycleMask = 0x03;
inline constexpr std::uint8_t kTerminalBit = 0x80;

inline constexpr NodeState lifecycle_of(std::uint8_t state) noexcept {
    return static_cast<NodeState>(state & kLifecycleMask);
}

inline constexpr bool terminal_of(std::uint8_t state) noexcept {
    return (state & kTerminalBit) != 0;
}

// Sentinel index. Also the arena's hard capacity ceiling, so that a valid index
// can never equal it.
inline constexpr std::uint32_t kNoNode = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Compile-time guarantees — Global Rule 5's half of the alignment story.
//
// If any of these is false on some future target, the module does not build.
// The alternative is an atomic that silently takes a mutex, which would cost
// far more than the search can afford and would announce itself only as a
// profile nobody can explain.
// ---------------------------------------------------------------------------

static_assert(std::atomic<std::int64_t>::is_always_lock_free,
              "value_sum requires a lock-free 64-bit atomic");
static_assert(std::atomic<std::int32_t>::is_always_lock_free,
              "visit_count and vloss_count require a lock-free 32-bit atomic");
static_assert(std::atomic<std::uint8_t>::is_always_lock_free,
              "state requires a lock-free 8-bit atomic");
static_assert(std::atomic<double>::is_always_lock_free,
              "the equivalence build's value_sum requires a lock-free 64-bit atomic");
static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
              "the arena bump pointer requires a lock-free 32-bit atomic");

// sizeof(atomic<T>) == sizeof(T) is not guaranteed by the standard, but the SoA
// stride arithmetic below assumes the arrays are exactly as wide as their
// payload. If this were ever false the arrays would still be correct — the
// element type is what is indexed — but the cache-line budget in the comment at
// the top of this file would be wrong, so it is worth knowing.
static_assert(sizeof(std::atomic<std::int32_t>) == sizeof(std::int32_t), "unexpected atomic width");
static_assert(sizeof(std::atomic<std::int64_t>) == sizeof(std::int64_t), "unexpected atomic width");
static_assert(sizeof(std::atomic<double>) == sizeof(double), "unexpected atomic width");
static_assert(alignof(std::atomic<std::int64_t>) >= alignof(std::int64_t),
              "atomic<int64_t> must be at least as aligned as int64_t");

// ---------------------------------------------------------------------------
// Accumulator policies
// ---------------------------------------------------------------------------

struct Q32Accumulator {
    using value_type = std::int64_t;
    using atomic_type = std::atomic<std::int64_t>;

    static constexpr const char *name() noexcept { return "q32"; }
    static constexpr bool is_fixed_point() noexcept { return true; }

    static value_type encode(double v) { return to_q32(v); }
    static constexpr double decode(value_type raw) noexcept { return from_q32(raw); }

    // One `lock xadd`. No retry loop, no failure path, and — the reason this is
    // the production accumulator — associative, so the total does not depend on
    // the order the threads happened to arrive in.
    static void add(atomic_type &slot, double v) {
        slot.fetch_add(encode(v), std::memory_order_relaxed);
    }
};

struct DoubleAccumulator {
    using value_type = double;
    using atomic_type = std::atomic<double>;

    static constexpr const char *name() noexcept { return "double"; }
    static constexpr bool is_fixed_point() noexcept { return false; }

    static constexpr value_type encode(double v) noexcept { return v; }
    static constexpr double decode(value_type raw) noexcept { return raw; }

    // C++17 has no fetch_add for floating-point atomics (it arrived in C++20),
    // so this is the CAS loop the scope names as the cost of the equivalence
    // build: the retry rate climbs with thread count, and the root's
    // accumulator is touched by every single backup. Single-threaded — which is
    // what Gate 1 runs — it never retries.
    static void add(atomic_type &slot, double v) {
        double expected = slot.load(std::memory_order_relaxed);
        while (!slot.compare_exchange_weak(expected, expected + v, std::memory_order_relaxed,
                                           std::memory_order_relaxed)) {
            // expected has been reloaded by compare_exchange_weak.
        }
    }
};

// ---------------------------------------------------------------------------
// One over-aligned array. The unit the arena is built out of.
//
// Alignment is C++17 over-aligned operator new for the same reason C0 chose it:
// it is the one spelling that compiles unmodified on MSVC and Clang, with no
// platform branch and no size-must-be-a-multiple-of-alignment restriction
// (which std::aligned_alloc imposes and MSVC does not provide at all).
//
// The arrays are aligned to a full cache line rather than to alignof(T). Two
// reasons, and only the second is about correctness: it guarantees the natural
// alignment every atomic needs with room to spare, and it means the first
// element of every array starts a line, so a 32-sibling scan reads the minimum
// possible number of lines instead of one extra at each end.
// ---------------------------------------------------------------------------

inline constexpr std::size_t kCacheLine = 64;

template <class T>
class AlignedArray {
public:
    AlignedArray() = default;

    explicit AlignedArray(std::size_t count) : count_(count) {
        if (count_ == 0) {
            return;
        }
        if (count_ > (std::numeric_limits<std::size_t>::max)() / sizeof(T)) {
            throw std::overflow_error("guofish::AlignedArray: byte size overflows size_t");
        }
        void *raw = ::operator new(count_ * sizeof(T), std::align_val_t(alignment()));
        data_ = static_cast<T *>(raw);
        for (std::size_t i = 0; i < count_; ++i) {
            // Value-initialised: std::atomic's default constructor leaves the
            // value indeterminate in C++17, so an arena whose fields were merely
            // default-constructed would hand out nodes full of garbage visit
            // counts, and ASan would not see it because the memory is
            // legitimately owned.
            ::new (static_cast<void *>(data_ + i)) T();
        }
        assert(is_cache_line_aligned());
    }

    ~AlignedArray() { release(); }

    AlignedArray(const AlignedArray &) = delete;
    AlignedArray &operator=(const AlignedArray &) = delete;

    AlignedArray(AlignedArray &&other) noexcept
        : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    AlignedArray &operator=(AlignedArray &&other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    static constexpr std::size_t alignment() noexcept {
        return alignof(T) > kCacheLine ? alignof(T) : kCacheLine;
    }

    T *data() noexcept { return data_; }
    const T *data() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }

    T &operator[](std::size_t i) noexcept {
        assert(i < count_);
        return data_[i];
    }

    const T &operator[](std::size_t i) const noexcept {
        assert(i < count_);
        return data_[i];
    }

    // The address, as an integer, for the runtime alignment checks. Returning it
    // from here keeps the one pointer-to-integer conversion in this file in one
    // place instead of repeating it per array.
    std::uintptr_t address() const noexcept {
        // reinterpret_cast: pointer -> integer solely to test its alignment. The
        // result is never dereferenced and never converted back to a pointer.
        return reinterpret_cast<std::uintptr_t>(data_);
    }

    // Both questions are asked against absolute constants rather than against
    // alignment(). Answering "is it aligned to what I asked for" would make the
    // report agree with itself no matter what was requested, which is exactly
    // the kind of check that passes while the memory is wrong — the C4 mutation
    // drill caught this reporting it correctly aligned after alignment() had
    // been cut to alignof(T).
    bool is_cache_line_aligned() const noexcept {
        return data_ == nullptr || address() % kCacheLine == 0;
    }

    // The weaker property the hardware actually requires: naturally aligned for
    // T, so no atomic operation on any element can straddle a cache line.
    bool is_naturally_aligned() const noexcept {
        return data_ == nullptr || address() % alignof(T) == 0;
    }

private:
    void release() noexcept {
        if (data_ == nullptr) {
            return;
        }
        for (std::size_t i = 0; i < count_; ++i) {
            data_[i].~T();
        }
        ::operator delete(static_cast<void *>(data_), std::align_val_t(alignment()));
        data_ = nullptr;
        count_ = 0;
    }

    T *data_ = nullptr;
    std::size_t count_ = 0;
};

// ---------------------------------------------------------------------------
// The arena
//
// Capacity is fixed at construction and the arena never reallocates. That is
// not laziness: growth would move every base pointer while other threads are
// mid-scan, and there is no way to make that safe without a level of
// indirection on the hottest read path in the engine. The scope sizes this from
// measurement — ~40 nodes/sim, 15k sims, game-long reuse — and budgets 2-3M
// nodes peak, so the arena is allocated once at that size and the bump pointer
// runs out rather than the allocator being asked for more.
// ---------------------------------------------------------------------------

template <class Accumulator>
class NodeArena {
public:
    using accumulator_type = Accumulator;
    using accum_value_type = typename Accumulator::value_type;
    using accum_atomic_type = typename Accumulator::atomic_type;

    explicit NodeArena(std::size_t capacity)
        : capacity_(capacity),
          visit_count_(capacity),
          value_sum_(capacity),
          vloss_count_(capacity),
          prior_(capacity),
          move_(capacity),
          children_offset_(capacity),
          children_count_(capacity),
          state_(capacity),
          terminal_value_(capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("guofish::NodeArena: capacity must be > 0");
        }
        // kNoNode must not be a valid index, or "no child" and "child at the
        // last slot" become the same answer.
        if (capacity >= static_cast<std::size_t>(kNoNode)) {
            throw std::invalid_argument("guofish::NodeArena: capacity must be < 2^32 - 1");
        }
        // Global Rule 5 / scope 2.3: assert every base pointer, every build.
        // These are the last line of defence against a misaligned `lock xadd`,
        // whose failure mode is a ~100x bus lock rather than a crash.
        assert(visit_count_.is_naturally_aligned());
        assert(value_sum_.is_naturally_aligned());
        assert(vloss_count_.is_naturally_aligned());
        assert(prior_.is_naturally_aligned());
        assert(move_.is_naturally_aligned());
        assert(children_offset_.is_naturally_aligned());
        assert(children_count_.is_naturally_aligned());
        assert(state_.is_naturally_aligned());
        assert(terminal_value_.is_naturally_aligned());
    }

    std::size_t capacity() const noexcept { return capacity_; }

    std::size_t size() const noexcept {
        return static_cast<std::size_t>(next_.load(std::memory_order_acquire));
    }

    // C8. The largest `size()` this arena has ever held, across every reset.
    //
    // The bump pointer only ever rises between resets, so the peak is `next_`
    // just before each reset plus wherever it stands now — no per-allocation
    // bookkeeping and nothing to keep in step on the hot path. It is here rather
    // than in the search because the ping-pong pair is two arenas and the
    // interesting figure is the peak of EITHER: a compaction's high-water is the
    // moment both the source subtree and its copy exist, and reporting only the
    // active one would under-report exactly the instant the budget is about.
    std::size_t high_water() const noexcept {
        const std::size_t now = size();
        const std::size_t peak = high_water_.load(std::memory_order_relaxed);
        return now > peak ? now : peak;
    }

    static constexpr const char *accumulator_name() noexcept { return Accumulator::name(); }

    // ---- allocation -------------------------------------------------------

    // Claim `count` consecutive nodes and return the index of the first, or
    // kNoNode if the arena is full. Consecutive because a node's children are
    // addressed as (offset, count): one uint32 and one uint16 per node instead
    // of a pointer per child, and a sibling scan that is sequential by
    // construction.
    //
    // The block is reset to a clean unexpanded state before it is returned, so
    // a reused arena (C8 ping-pong) cannot hand out a node still carrying the
    // previous search's visit counts.
    std::uint32_t try_allocate(std::size_t count) noexcept {
        if (count == 0) {
            return kNoNode;
        }
        std::uint32_t first = next_.load(std::memory_order_relaxed);
        for (;;) {
            if (count > capacity_ - static_cast<std::size_t>(first)) {
                return kNoNode;
            }
            const auto end = static_cast<std::uint32_t>(static_cast<std::size_t>(first) + count);
            if (next_.compare_exchange_weak(first, end, std::memory_order_acq_rel,
                                            std::memory_order_relaxed)) {
                break;
            }
        }
        reset_range(first, count);
        return first;
    }

    // The throwing form, for callers that treat exhaustion as a bug rather than
    // as a condition. C5's expansion path uses try_allocate.
    std::uint32_t allocate(std::size_t count) {
        const std::uint32_t first = try_allocate(count);
        if (first == kNoNode) {
            if (count == 0) {
                throw std::invalid_argument("guofish::NodeArena::allocate: count must be > 0");
            }
            throw std::runtime_error("guofish::NodeArena::allocate: arena exhausted (" +
                                     std::to_string(count) + " requested, " +
                                     std::to_string(capacity_ - size()) + " free of " +
                                     std::to_string(capacity_) + ")");
        }
        return first;
    }

    // Drop every allocation. The storage is not touched — try_allocate() clears
    // each block on the way out — so this is O(1) and a 2M-node arena can be
    // recycled between searches without a 19 MB memset nobody asked for.
    void reset() noexcept {
        const std::uint32_t used = next_.load(std::memory_order_relaxed);
        if (static_cast<std::size_t>(used) > high_water_.load(std::memory_order_relaxed)) {
            high_water_.store(static_cast<std::size_t>(used), std::memory_order_relaxed);
        }
        next_.store(0, std::memory_order_release);
    }

    // C8, ping-pong. Exchange this arena's STORAGE with another arena's.
    //
    // Nine pointer swaps and one integer exchange — no node is touched and
    // nothing is copied. That is the whole point: `apply_move` compacts the
    // surviving subtree into the standby arena and then makes it the active one,
    // and if the "swap" were a copy the compaction would have been pointless.
    //
    // Written as a member rather than as a `std::swap` of two NodeArenas because
    // NodeArena is deliberately not movable: it holds a `std::atomic<uint32_t>`
    // bump pointer, and a type whose move constructor silently loaded and stored
    // an atomic would be a type C9 could accidentally move while threads were
    // running in it. Swapping the arrays under an explicit call has no such
    // reading.
    //
    // The capacities must match. They do by construction — the standby arena is
    // allocated at the active one's capacity — and a mismatch would mean one of
    // the two could not hold what the other was sized for, which is a bug rather
    // than a case to handle.
    void swap_storage(NodeArena &other) {
        if (capacity_ != other.capacity_) {
            throw std::invalid_argument(
                "guofish::NodeArena::swap_storage: capacities differ (" +
                std::to_string(capacity_) + " vs " + std::to_string(other.capacity_) + ")");
        }
        using std::swap;
        swap(visit_count_, other.visit_count_);
        swap(value_sum_, other.value_sum_);
        swap(vloss_count_, other.vloss_count_);
        swap(prior_, other.prior_);
        swap(move_, other.move_);
        swap(children_offset_, other.children_offset_);
        swap(children_count_, other.children_count_);
        swap(state_, other.state_);
        swap(terminal_value_, other.terminal_value_);

        const std::uint32_t mine = next_.load(std::memory_order_relaxed);
        next_.store(other.next_.load(std::memory_order_relaxed), std::memory_order_release);
        other.next_.store(mine, std::memory_order_release);

        // The high-water marks travel with the STORAGE, not with the object, so
        // that "the peak this arena ever held" keeps meaning the peak of the
        // memory rather than the peak of whichever role the memory is currently
        // playing.
        const std::size_t peak = high_water_.load(std::memory_order_relaxed);
        high_water_.store(other.high_water_.load(std::memory_order_relaxed),
                          std::memory_order_relaxed);
        other.high_water_.store(peak, std::memory_order_relaxed);
    }

    // ---- reads ------------------------------------------------------------

    std::int32_t visit_count(std::uint32_t i) const {
        check(i);
        return visit_count_[i].load(std::memory_order_relaxed);
    }

    // The accumulator's own representation: int64 Q32 under Q32Accumulator,
    // double under DoubleAccumulator. Exposed alongside value_sum() so a test
    // can see the fixed-point integer rather than only its decoded value.
    accum_value_type value_sum_raw(std::uint32_t i) const {
        check(i);
        return value_sum_[i].load(std::memory_order_relaxed);
    }

    double value_sum(std::uint32_t i) const { return Accumulator::decode(value_sum_raw(i)); }

    std::int32_t vloss_count(std::uint32_t i) const {
        check(i);
        return vloss_count_[i].load(std::memory_order_relaxed);
    }

    float prior(std::uint32_t i) const {
        check(i);
        return prior_[i];
    }

    std::uint16_t move(std::uint32_t i) const {
        check(i);
        return move_[i];
    }

    std::uint32_t children_offset(std::uint32_t i) const {
        check(i);
        return children_offset_[i];
    }

    std::uint16_t children_count(std::uint32_t i) const {
        check(i);
        return children_count_[i];
    }

    std::uint8_t state_bits(std::uint32_t i) const {
        check(i);
        return state_[i].load(std::memory_order_acquire);
    }

    NodeState lifecycle(std::uint32_t i) const { return lifecycle_of(state_bits(i)); }

    bool is_terminal(std::uint32_t i) const { return terminal_of(state_bits(i)); }

    float terminal_value(std::uint32_t i) const {
        check(i);
        return terminal_value_[i];
    }

    // The k-th child of `i`. Bounds-checked against the arena, not merely
    // against children_count: a child range that points outside the allocated
    // region is the C8 pointer-fixup bug class, and it should be caught where
    // it is read rather than where it produces a wrong move.
    std::uint32_t child(std::uint32_t i, std::uint16_t k) const {
        check(i);
        const std::uint16_t count = children_count_[i];
        if (k >= count) {
            throw std::out_of_range("guofish::NodeArena::child: child index " + std::to_string(k) +
                                    " out of range for node " + std::to_string(i) + " with " +
                                    std::to_string(count) + " children");
        }
        const std::uint32_t first = children_offset_[i];
        const std::uint32_t idx = first + k;
        // Cannot fire while set_children() is the only writer — it validates the
        // whole range — but this is the read side of the invariant and it costs
        // one compare on a path that is already touching the child's cache line.
        assert(static_cast<std::size_t>(idx) < size());
        return idx;
    }

    // ---- writes -----------------------------------------------------------

    void set_prior(std::uint32_t i, float p) {
        check(i);
        prior_[i] = p;
    }

    void set_move(std::uint32_t i, std::uint16_t packed) {
        check(i);
        move_[i] = packed;
    }

    void add_visits(std::uint32_t i, std::int32_t n) {
        check(i);
        visit_count_[i].fetch_add(n, std::memory_order_relaxed);
    }

    void add_value(std::uint32_t i, double v) {
        check(i);
        Accumulator::add(value_sum_[i], v);
    }

    // The raw form, so a test can put an exact fixed-point quantity into the
    // accumulator without going through the double conversion it is trying to
    // test.
    void add_value_raw(std::uint32_t i, accum_value_type raw) {
        check(i);
        if constexpr (Accumulator::is_fixed_point()) {
            value_sum_[i].fetch_add(raw, std::memory_order_relaxed);
        } else {
            Accumulator::add(value_sum_[i], raw);
        }
    }

    void add_vloss(std::uint32_t i, std::int32_t n) {
        check(i);
        vloss_count_[i].fetch_add(n, std::memory_order_relaxed);
    }

    // ---- C8: assignment, not accumulation ---------------------------------
    //
    // Two callers need to OVERWRITE these fields rather than add to them, and
    // both are on the tree-reuse path where the node in question already carries
    // a previous search's numbers:
    //
    //   the compaction  copies a node's exact totals into a fresh slot;
    //   expand_root     re-seeds a PROMOTED root, and the reference assigns
    //                   there (`root.visit_count = 1`, `root.value_sum = ...`)
    //                   rather than incrementing. On a fresh node the two are
    //                   the same and C5 could use add_*; on a promoted node that
    //                   arrived with 400 visits from the fifty-move fast path
    //                   they are not, and add_* would leave 401.
    //
    // Not `store`-named because the arena's vocabulary is add_/set_, and not
    // folded into add_* with a flag because "assign" and "accumulate" are the
    // distinction the bug above is made of.
    void set_visits(std::uint32_t i, std::int32_t n) {
        check(i);
        visit_count_[i].store(n, std::memory_order_relaxed);
    }

    void set_value(std::uint32_t i, double v) {
        check(i);
        value_sum_[i].store(Accumulator::encode(v), std::memory_order_relaxed);
    }

    // The accumulator's own representation, unconverted. The compaction uses
    // this: a Q32 total copied through `double` would round twice, and the whole
    // claim being made about a compacted tree is that it is bit-identical.
    void set_value_raw(std::uint32_t i, accum_value_type raw) {
        check(i);
        value_sum_[i].store(raw, std::memory_order_relaxed);
    }

    // ---- state transitions ------------------------------------------------

    // Claim an unexpanded leaf for evaluation. Returns false if someone else
    // got there first, which is the "two threads selected the same leaf" case
    // scope 2.2 resolves by making the loser unwind its virtual loss and retry.
    //
    // A terminal node cannot be claimed: there is nothing to evaluate.
    bool try_claim_pending(std::uint32_t i) {
        check(i);
        std::uint8_t expected = static_cast<std::uint8_t>(NodeState::Unexpanded);
        return state_[i].compare_exchange_strong(expected,
                                                 static_cast<std::uint8_t>(NodeState::Pending),
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire);
    }

    // Give up a claim without expanding — the evaluation failed, or the
    // simulation was discarded. Terminal nodes are never Pending, so the
    // terminal bit cannot be lost here.
    void release_pending(std::uint32_t i) {
        check(i);
        std::uint8_t expected = static_cast<std::uint8_t>(NodeState::Pending);
        if (!state_[i].compare_exchange_strong(expected,
                                               static_cast<std::uint8_t>(NodeState::Unexpanded),
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
            throw std::logic_error("guofish::NodeArena::release_pending: node " + std::to_string(i) +
                                   " is not pending (state " + std::to_string(expected) + ")");
        }
    }

    // Publish a child range and mark the node expanded.
    //
    // This is where `bestmove 0000` is made unrepresentable. Every rejection
    // below is a state the Python engine could reach:
    //
    //   count == 0        an expanded node with no children — the defect itself.
    //   already terminal  a node with a game result cannot also have moves.
    //   already expanded  double expansion would orphan the first child block
    //                     and silently leak its visits out of the tree.
    //   range invalid     children outside the allocated region.
    //
    // The store is a release, and every read of `state` is an acquire, so a
    // thread that sees Expanded also sees the offset, the count, and whatever
    // the expander wrote into the children themselves.
    void set_children(std::uint32_t parent, std::uint32_t offset, std::uint16_t count) {
        check(parent);
        if (count == 0) {
            throw std::invalid_argument(
                "guofish::NodeArena::set_children: node " + std::to_string(parent) +
                " cannot be expanded with zero children — a node with no moves is terminal, "
                "not expanded (this is the bestmove 0000 defect)");
        }
        const std::uint8_t current = state_[parent].load(std::memory_order_acquire);
        if (terminal_of(current)) {
            throw std::logic_error("guofish::NodeArena::set_children: node " +
                                   std::to_string(parent) + " is terminal and cannot be expanded");
        }
        if (lifecycle_of(current) == NodeState::Expanded) {
            throw std::logic_error("guofish::NodeArena::set_children: node " +
                                   std::to_string(parent) + " is already expanded");
        }
        const std::size_t allocated = size();
        if (static_cast<std::size_t>(offset) + count > allocated) {
            throw std::out_of_range("guofish::NodeArena::set_children: child range [" +
                                    std::to_string(offset) + ", " +
                                    std::to_string(static_cast<std::size_t>(offset) + count) +
                                    ") is outside the allocated region [0, " +
                                    std::to_string(allocated) + ")");
        }

        children_offset_[parent] = offset;
        children_count_[parent] = count;
        // Release: publishes the two lines above, and the children's own fields,
        // to any thread that subsequently sees Expanded.
        state_[parent].store(static_cast<std::uint8_t>(
                                 (current & kTerminalBit) |
                                 static_cast<std::uint8_t>(NodeState::Expanded)),
                             std::memory_order_release);
    }

    // Record that the game ends at this node.
    //
    // The terminal bit is set *beside* the lifecycle rather than replacing it,
    // so "terminal" and "unexpanded" are both true at once and no code has to
    // ask which of the two a single enum value meant. A terminal node is never
    // expanded, so `set_children` refuses it from here on.
    void mark_terminal(std::uint32_t i, float value) {
        check(i);
        const std::uint8_t current = state_[i].load(std::memory_order_acquire);
        if (lifecycle_of(current) == NodeState::Expanded || children_count_[i] != 0) {
            throw std::logic_error("guofish::NodeArena::mark_terminal: node " + std::to_string(i) +
                                   " already has children and cannot be terminal");
        }
        terminal_value_[i] = value;
        state_[i].store(static_cast<std::uint8_t>(current | kTerminalBit),
                        std::memory_order_release);
    }

    // C9. Record a game result on a node this thread has ALREADY claimed.
    //
    // `mark_terminal` above is a load, a plain store to `terminal_value_` and a
    // release store, which is exactly right for one thread and a data race for
    // two: W workers can reach the same first-visit terminal leaf at the same
    // instant and both write the same float, which is benign in practice,
    // undefined behaviour in the standard, and a ThreadSanitizer report either
    // way. The fix is not a lock — it is to notice that the PENDING claim the
    // parallel descent already takes before evaluating a leaf is exactly the
    // exclusivity this needs.
    //
    // So this is the terminal-marking half of that claim: the caller holds
    // PENDING, writes the value, and publishes Unexpanded|TERMINAL in one
    // release store. Unexpanded, not Pending, because the claim is over — and
    // TERMINAL beside it rather than instead of it, which is the C6 invariant
    // this arena exists to make unrepresentable in the first place.
    void mark_terminal_pending(std::uint32_t i, float value) {
        check(i);
        const std::uint8_t current = state_[i].load(std::memory_order_acquire);
        if (lifecycle_of(current) != NodeState::Pending) {
            throw std::logic_error("guofish::NodeArena::mark_terminal_pending: node " +
                                   std::to_string(i) + " is not pending (state " +
                                   std::to_string(current) + ")");
        }
        if (children_count_[i] != 0) {
            throw std::logic_error("guofish::NodeArena::mark_terminal_pending: node " +
                                   std::to_string(i) +
                                   " already has children and cannot be terminal");
        }
        terminal_value_[i] = value;
        state_[i].store(static_cast<std::uint8_t>(
                            static_cast<std::uint8_t>(NodeState::Unexpanded) | kTerminalBit),
                        std::memory_order_release);
    }

    // C9. Claim and mark in one call, for the caller that did not already hold
    // the node — the claimable-draw case, where the descent discovers a
    // fifty-move or threefold draw at a node it was only passing through.
    //
    // Returns false if someone else got there first. That is not an error and
    // the caller does not retry: within one search, `draw_by_rule` is a function
    // of the node (its path from the root is unique and the repetition history
    // is fixed for the search), so a second thread that reaches the same node
    // computes the same answer and is marking the same value. The loser simply
    // backs up the draw it independently derived.
    bool try_mark_terminal(std::uint32_t i, float value) {
        if (!try_claim_pending(i)) {
            return false;
        }
        mark_terminal_pending(i, value);
        return true;
    }

    // C8. Withdraw a terminal mark. THE ONLY LEGITIMATE CALLER IS ROOT
    // PROMOTION, and the reason is a rule of chess rather than a convenience.
    //
    // The fifty-move rule and threefold repetition are CLAIMABLE: the game ends
    // only if someone claims it. C6 marks such a node terminal and leaves it
    // Unexpanded precisely so that a host which declines the claim can hand the
    // position back as the one to move from. When that happens the mark is
    // stale — we are playing on — and it has to come off before the node can be
    // expanded, because `set_children` refuses a terminal node by design.
    //
    // The reference does exactly this, in `_expand_root`:
    //
    //     if root.children:
    //         root.is_terminal = False
    //
    // gated on children existing, so a genuine checkmate or stalemate (where
    // expansion produced nothing) keeps its mark and its value. The gate lives
    // at the call site here too, for the same reason.
    //
    // `terminal_value_` is deliberately LEFT ALONE. The reference leaves it too,
    // and the tree serialisation writes it whether or not the bit is set — so
    // clearing it here would put C++ one field away from the reference on every
    // promoted draw. The value is dead state on an unmarked node; the tests read
    // it anyway, which is why it has to be dead in the same way on both sides.
    void clear_terminal(std::uint32_t i) {
        check(i);
        const std::uint8_t current = state_[i].load(std::memory_order_acquire);
        state_[i].store(static_cast<std::uint8_t>(current & ~kTerminalBit),
                        std::memory_order_release);
    }

    // ---- diagnostics ------------------------------------------------------

    struct ArrayInfo {
        const char *field;
        const char *element_type;
        std::uintptr_t address;
        std::size_t element_size;
        std::size_t element_align;
        std::size_t requested_align;
        bool naturally_aligned;
        bool cache_line_aligned;
    };

    // Nine entries, one per array, in layout order. The runtime half of the
    // alignment guarantee: the static_asserts prove the atomics are lock-free,
    // this proves the memory they live in is where it has to be. Reported to
    // Python rather than only asserted so that a Release build — where assert()
    // is compiled out — is checked too.
    template <class F>
    void for_each_array(F &&fn) const {
        emit(fn, "visit_count", "atomic<int32>", visit_count_);
        emit(fn, "value_sum", Accumulator::is_fixed_point() ? "atomic<int64>" : "atomic<double>",
             value_sum_);
        emit(fn, "vloss_count", "atomic<int32>", vloss_count_);
        emit(fn, "prior", "float", prior_);
        emit(fn, "move", "uint16", move_);
        emit(fn, "children_offset", "uint32", children_offset_);
        emit(fn, "children_count", "uint16", children_count_);
        emit(fn, "state", "atomic<uint8>", state_);
        emit(fn, "terminal_value", "float", terminal_value_);
    }

    // Bytes of node payload per node, summed over the arrays. Not sizeof(node):
    // there is no node struct, which is the point.
    static constexpr std::size_t bytes_per_node() noexcept {
        return sizeof(std::atomic<std::int32_t>) + sizeof(accum_atomic_type) +
               sizeof(std::atomic<std::int32_t>) + sizeof(float) + sizeof(std::uint16_t) +
               sizeof(std::uint32_t) + sizeof(std::uint16_t) + sizeof(std::atomic<std::uint8_t>) +
               sizeof(float);
    }

    // Direct array access for the sibling-scan benchmark, which must read the
    // three hot fields with no per-element bounds check or it would be timing
    // the check. Not for general use: everything else goes through the
    // accessors above.
    const std::atomic<std::int32_t> *visit_count_data() const noexcept { return visit_count_.data(); }
    const accum_atomic_type *value_sum_data() const noexcept { return value_sum_.data(); }
    const float *prior_data() const noexcept { return prior_.data(); }

private:
    void check(std::uint32_t i) const {
        if (static_cast<std::size_t>(i) >= size()) {
            throw std::out_of_range("guofish::NodeArena: node index " + std::to_string(i) +
                                    " out of range [0, " + std::to_string(size()) + ")");
        }
    }

    void reset_range(std::uint32_t first, std::size_t count) noexcept {
        for (std::size_t k = 0; k < count; ++k) {
            const std::size_t i = static_cast<std::size_t>(first) + k;
            visit_count_[i].store(0, std::memory_order_relaxed);
            value_sum_[i].store(accum_value_type{}, std::memory_order_relaxed);
            vloss_count_[i].store(0, std::memory_order_relaxed);
            prior_[i] = 0.0f;
            move_[i] = kNoMove;
            children_offset_[i] = 0;
            children_count_[i] = 0;
            terminal_value_[i] = 0.0f;
            // Last, and a release: a thread that observes the fresh Unexpanded
            // state observes the cleared fields with it.
            state_[i].store(static_cast<std::uint8_t>(NodeState::Unexpanded),
                            std::memory_order_release);
        }
    }

    template <class F, class T>
    static void emit(F &fn, const char *field, const char *type, const AlignedArray<T> &arr) {
        ArrayInfo info{};
        info.field = field;
        info.element_type = type;
        info.address = arr.address();
        info.element_size = sizeof(T);
        info.element_align = alignof(T);
        info.requested_align = AlignedArray<T>::alignment();
        info.naturally_aligned = arr.is_naturally_aligned();
        info.cache_line_aligned = arr.is_cache_line_aligned();
        fn(info);
    }

    std::size_t capacity_;
    std::atomic<std::uint32_t> next_{0};
    // C8. Peak `next_` over this storage's whole life; see high_water().
    std::atomic<std::size_t> high_water_{0};

    // One array per field. Declaration order is initialisation order; it is
    // also the order for_each_array() reports, so the two cannot drift.
    AlignedArray<std::atomic<std::int32_t>> visit_count_;
    AlignedArray<accum_atomic_type> value_sum_;
    AlignedArray<std::atomic<std::int32_t>> vloss_count_;
    AlignedArray<float> prior_;
    AlignedArray<std::uint16_t> move_;
    AlignedArray<std::uint32_t> children_offset_;
    AlignedArray<std::uint16_t> children_count_;
    AlignedArray<std::atomic<std::uint8_t>> state_;
    AlignedArray<float> terminal_value_;
};

using Q32Arena = NodeArena<Q32Accumulator>;
using DoubleArena = NodeArena<DoubleAccumulator>;

// Which accumulator the engine is built with. Both types exist in every build —
// this only names the default, so that flipping the switch for the Gate 1
// equivalence build changes one alias rather than any call site.
#if defined(GUOFISH_VALUE_SUM_DOUBLE)
using DefaultArena = DoubleArena;
inline constexpr const char *kDefaultAccumulator = "double";
#else
using DefaultArena = Q32Arena;
inline constexpr const char *kDefaultAccumulator = "q32";
#endif

}  // namespace guofish

#endif  // GUOFISH_ARENA_HPP
