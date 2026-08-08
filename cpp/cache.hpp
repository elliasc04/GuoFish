// GuoFish C7 — the transposition cache for neural-network evaluations.
//
// A position-keyed cache in front of the evaluator. It holds exactly three
// things and it is incapable of holding a fourth:
//
//     value    the network's value-head output, White-POV     (NetworkValue)
//     priors   the gathered legal priors, canonical order     (float[])
//     moves    the move list those priors are aligned with    (packed uint16[])
//
// Everything interesting about this file is a consequence of one fact: THE KEY
// IS A FUNCTION OF THE POSITION AND OF NOTHING ELSE. Two positions share an
// `nn_key` exactly when they tokenize to the same 68 integers (C3). So anything
// the cache stores must also be a function of the tokenization — and the
// network's output is, by definition, because the tokenization is its input.
//
// A terminal mark is not. A repetition draw is not. A Syzygy WDL is not. Storing
// one of those here is the historical defect this chunk exists to prevent, and
// cpp/values.hpp is the mechanism: `insert` takes a `NetworkValue`, and a
// `TerminalValue`, a `TablebaseValue`, a `ProofValue`, a bare `double` and a
// `bool` are all things this program will not compile with. See the
// static_asserts at the bottom of this file — they ARE acceptance criterion 5.
//
//
// WHAT THE CACHE MAY AND MAY NOT DISTINGUISH
// ------------------------------------------
// Two properties, both required, and they pull in opposite directions:
//
//   EP-TWINS MUST NOT COLLIDE. Two positions differing only in the raw
//   en-passant square tokenize differently at index 66, so their nn_keys
//   differ, so they get different entries. This is the correction C3 was
//   written around; it holds here for free because the key is the token hash.
//   The reference reaches the same place by a patch (`make_cache_key` appends
//   the ep square to a Zobrist that would otherwise be coarser than the
//   network's own input); here it is not patchable, because there is nothing
//   to patch.
//
//   CLOCK-TWINS MUST SHARE AN ENTRY, AND THAT IS CORRECT. The halfmove clock is
//   not a token. The network cannot see it, so its output cannot depend on it,
//   so two positions differing only in the clock have the same evaluation and
//   must share one. Anyone reading this and reaching for "but the fifty-move
//   rule!" is right about the rule and wrong about the cache: the fifty-move
//   rule produces a TerminalValue, and a TerminalValue cannot be in here.
//
// The two together are why the type discipline is not paranoia. Sharing an
// entry across clock-twins is correct *only* for values that ignore the clock.
// The moment a clock-dependent value is admitted, the same sharing that makes
// the cache correct makes it wrong.
//
//
// SHAPE: SHARDED, DIRECT-MAPPED, SPINLOCK PER SHARD
// -------------------------------------------------
// >= 64 shards with a spinlock each, per the brief; C9 is what they are for.
// Within a shard the table is direct-mapped — one slot per bucket, an insert
// overwrites whatever was there. Alternatives considered are in DECISIONS.md;
// the short version is that the reference's hash-map-plus-ring-buffer needs two
// data structures and a write to a shared ring pointer on every insert, and
// with TB off the eviction policy cannot affect the result at all, so the
// simplest structure that never allocates in steady state wins.

#ifndef GUOFISH_CACHE_HPP
#define GUOFISH_CACHE_HPP

#include "keys.hpp"
#include "values.hpp"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#define GUOFISH_HAS_PAUSE 1
#include <immintrin.h>
#else
#define GUOFISH_HAS_PAUSE 0
#endif

namespace guofish {

// ---------------------------------------------------------------------------
// Spinlock
//
// A test-and-test-and-set lock over `std::atomic<bool>`. `std::atomic_flag` is
// the more obvious choice and is the wrong one in C++17: it has no non-modifying
// `test()` before C++20, so the read half of test-and-test-and-set cannot be
// written and every waiter's `test_and_set` bounces the cache line. That is the
// whole reason TTAS exists.
//
// Critical sections here are a memcpy of ~150 bytes with no syscall, no
// allocation (the slot's vectors are reused) and no branch that can block, so a
// waiter is never waiting on anything but a few hundred cycles. A mutex would
// be correct too and would cost a futex round trip on contention; the brief
// specifies a spinlock and for this critical-section length it is right.
// ---------------------------------------------------------------------------

inline void cpu_relax() noexcept {
#if GUOFISH_HAS_PAUSE
    _mm_pause();
#else
    // No portable pause instruction. Yielding is heavier than a pause but is
    // correct everywhere, and Global Rule 8 asks for a portable fallback rather
    // than for the fallback to be equally fast.
    std::this_thread::yield();
#endif
}

class Spinlock {
public:
    Spinlock() = default;

    Spinlock(const Spinlock &) = delete;
    Spinlock &operator=(const Spinlock &) = delete;

    void lock() noexcept {
        for (;;) {
            // The "test-and-set" half. On an uncontended lock this is the only
            // atomic operation performed.
            if (!held_.exchange(true, std::memory_order_acquire)) {
                return;
            }
            // The "test" half: spin on a plain load, which keeps the line in a
            // shared state instead of ping-ponging it exclusively between
            // waiters.
            while (held_.load(std::memory_order_relaxed)) {
                cpu_relax();
            }
        }
    }

    bool try_lock() noexcept { return !held_.exchange(true, std::memory_order_acquire); }

    void unlock() noexcept { held_.store(false, std::memory_order_release); }

private:
    std::atomic<bool> held_{false};
};

// RAII, so a throw inside a critical section cannot strand a shard locked. The
// bodies below do not throw, but "does not throw today" is not a property worth
// betting the engine's liveness on.
class SpinlockGuard {
public:
    explicit SpinlockGuard(Spinlock &lock) noexcept : lock_(lock) { lock_.lock(); }
    ~SpinlockGuard() { lock_.unlock(); }

    SpinlockGuard(const SpinlockGuard &) = delete;
    SpinlockGuard &operator=(const SpinlockGuard &) = delete;

private:
    Spinlock &lock_;
};

// ---------------------------------------------------------------------------
// The payload
//
// What a probe hands back. It is a COPY, not a pointer into the table, and that
// is deliberate: under C9 a pointer into a slot is valid only until some other
// thread evicts it, which is a use-after-free with a stochastic reproduction
// rate — the worst kind. The copy is ~150 bytes out of a lock that is already
// held; the vectors are reused across probes by the caller, so a steady-state
// search performs no allocation here.
// ---------------------------------------------------------------------------

struct CachedEval {
    // The declared type is the enforcement. `NetworkValue` has no default
    // constructor by design (cpp/values.hpp), so this member initialiser is the
    // only way to have one before a probe fills it, and 0.0 is never mistaken
    // for a real entry because `probe` returns false rather than leaving this
    // readable.
    NetworkValue value{0.0};
    std::vector<std::uint16_t> moves;   // packed, canonical order
    std::vector<float> priors;          // canonical order, aligned with `moves`

    void clear() {
        moves.clear();
        priors.clear();
    }
};

static_assert(std::is_same_v<decltype(CachedEval::value), NetworkValue>,
              "the cache payload's value must be typed as a network value, not as a double");

// ---------------------------------------------------------------------------
// Statistics
//
// `hits` and `misses` are the acceptance criterion the brief adds specifically
// because tree equivalence cannot see them: with TB off the cache is
// result-invariant, so a cache that never hits produces a bit-identical tree.
// Every number here is therefore load-bearing evidence rather than telemetry.
// ---------------------------------------------------------------------------

struct CacheStats {
    std::int64_t hits = 0;
    std::int64_t misses = 0;
    std::int64_t inserts = 0;
    // An insert that landed on a slot already holding a DIFFERENT key. This is
    // the direct-mapped table's eviction count, and it is reported because a
    // disappointing hit rate is either a workload with no transpositions or a
    // table that is too small, and these two numbers tell them apart.
    std::int64_t collisions = 0;
    // An insert that landed on a slot already holding the SAME key: a rewrite of
    // an identical payload. Non-zero is normal (two threads racing the same
    // miss); it is counted separately so it cannot inflate `collisions`.
    std::int64_t refreshes = 0;

    double hit_rate() const noexcept {
        const std::int64_t total = hits + misses;
        return total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
    }
};

// ---------------------------------------------------------------------------
// The cache
// ---------------------------------------------------------------------------

inline constexpr std::size_t kMinCacheShards = 64;   // the brief's floor
inline constexpr std::size_t kDefaultCacheShards = 64;

// The largest legal move count in any chess position is 218, so a payload never
// exceeds that. Checked on insert rather than assumed: a count that overflowed
// would silently truncate a move list, and a truncated move list is a wrong
// prior on every move after the cut.
inline constexpr std::uint16_t kMaxLegalMoves = 218;

class TranspositionCache {
public:
    // `slots` is the total number of entries, rounded UP to a power of two per
    // shard. `shards` must be a power of two and at least kMinCacheShards.
    //
    // A cache with zero slots is not constructible: "cache off" is expressed by
    // not having one (an empty std::optional in the search), not by having one
    // that silently answers every probe with a miss. A zero-slot cache would
    // pass every equivalence test in this chunk while doing nothing, which is
    // exactly the failure mode the hit-rate criterion exists to catch, and it
    // should not be one keystroke away.
    explicit TranspositionCache(std::size_t slots, std::size_t shards = kDefaultCacheShards) {
        if (shards < kMinCacheShards) {
            throw std::invalid_argument(
                "guofish::TranspositionCache: shards must be at least " +
                std::to_string(kMinCacheShards) + " (the C7 brief's floor); got " +
                std::to_string(shards));
        }
        if ((shards & (shards - 1)) != 0) {
            throw std::invalid_argument(
                "guofish::TranspositionCache: shards must be a power of two; got " +
                std::to_string(shards));
        }
        if (slots == 0) {
            throw std::invalid_argument(
                "guofish::TranspositionCache: slots must be > 0. A cache that can hold "
                "nothing is not 'cache off' — it is a cache with a 0% hit rate, which "
                "passes tree-equivalence tests while doing no work. Express 'off' by not "
                "constructing one.");
        }

        // Round the per-shard slot count up to a power of two so the index is a
        // mask rather than a modulo. A modulo by a runtime value is a hardware
        // divide on the hot path for no benefit.
        std::size_t per_shard = 1;
        while (per_shard * shards < slots) {
            per_shard <<= 1;
        }

        shard_mask_ = shards - 1;
        slot_mask_ = per_shard - 1;
        // `std::unique_ptr<Shard[]>` rather than `std::vector<Shard>`: a Shard
        // holds a Spinlock, a Spinlock is neither copyable nor movable (an
        // atomic that could be moved out from under a waiter would not be a
        // lock), and `std::vector::resize` requires MoveInsertable even when it
        // is not going to reallocate. The array is allocated once and never
        // grows, so a vector buys nothing here anyway. C++17's aligned `new[]`
        // honours the `alignas(64)` on Shard.
        shard_count_ = shards;
        shards_ = std::make_unique<Shard[]>(shards);
        for (std::size_t i = 0; i < shard_count_; ++i) {
            shards_[i].slots.resize(per_shard);
        }
    }

    std::size_t shard_count() const noexcept { return shard_count_; }
    std::size_t slots_per_shard() const noexcept { return slot_mask_ + 1; }
    std::size_t capacity() const noexcept { return shard_count() * slots_per_shard(); }

    // Occupied slots. Walks the whole table under each shard's lock, so it is a
    // diagnostic, not something to call in a loop.
    std::size_t size() const {
        std::size_t total = 0;
        for (std::size_t i = 0; i < shard_count_; ++i) {
            const Shard &shard = shards_[i];
            SpinlockGuard guard(shard.lock);
            for (const Slot &slot : shard.slots) {
                if (slot.key.has_value()) {
                    ++total;
                }
            }
        }
        return total;
    }

    // Copy the entry for `key` into `out` and return true, or return false and
    // leave `out` cleared.
    //
    // `out` is an in/out parameter rather than an optional return so a search
    // can keep one CachedEval alive across simulations and reuse its vectors'
    // capacity. At ~35 moves a fresh return value would be two allocations per
    // cache hit.
    //
    // Not `const`, because it is not: a probe writes a hit or a miss into the
    // shard's counters, and those counters are the acceptance criterion. Spelling
    // it const and casting the constness away inside would be a lie to every
    // caller and undefined behaviour on a genuinely const cache.
    bool probe(NNKey key, CachedEval &out) {
        Shard &shard = shard_for(key);
        Slot &slot = shard.slots[slot_index(key)];

        SpinlockGuard guard(shard.lock);
        // `slot.key` is std::optional: the disengaged state IS the empty-slot
        // sentinel, and `has_value()` is the check. See the note on Slot.
        if (!slot.key.has_value() || !(*slot.key == key)) {
            ++shard.stats.misses;
            out.clear();
            return false;
        }
        ++shard.stats.hits;
        out.value = slot.value;
        out.moves.assign(slot.moves.begin(), slot.moves.end());
        out.priors.assign(slot.priors.begin(), slot.priors.end());
        return true;
    }

    // Store one evaluation.
    //
    // THE VALUE PARAMETER IS `NetworkValue` AND THAT IS THE POINT. Passing a
    // TerminalValue, a TablebaseValue, a ProofValue, a bare double or a bool is
    // a compile error, which is acceptance criterion 5. The static_asserts below
    // this class prove it in every build.
    //
    // `key` is the caller's, computed from the token row that was dispatched to
    // the evaluator (guofish::EvalRow). It is NOT recomputed here, deliberately:
    // recomputing it would introduce a second derivation of the key, which is
    // precisely the trap the brief names under "Risks".
    void insert(NNKey key, NetworkValue value, const std::uint16_t *moves, const float *priors,
                std::uint16_t count) {
        if (count == 0) {
            throw std::invalid_argument(
                "guofish::TranspositionCache::insert: an entry with no moves. A position "
                "with no legal moves is terminal, and a terminal result must never reach "
                "this cache.");
        }
        if (count > kMaxLegalMoves) {
            throw std::invalid_argument(
                "guofish::TranspositionCache::insert: " + std::to_string(count) +
                " moves exceeds the maximum legal move count (" +
                std::to_string(kMaxLegalMoves) + ")");
        }
        assert(moves != nullptr && priors != nullptr);

        Shard &shard = shard_for(key);
        Slot &slot = shard.slots[slot_index(key)];

        SpinlockGuard guard(shard.lock);
        if (slot.key.has_value()) {
            if (*slot.key == key) {
                ++shard.stats.refreshes;
            } else {
                ++shard.stats.collisions;
            }
        }
        slot.key = key;
        slot.value = value;
        slot.moves.assign(moves, moves + count);
        slot.priors.assign(priors, priors + count);
        ++shard.stats.inserts;
    }

    // Drop every entry. Statistics are zeroed too: a hit rate that spans a clear
    // is a hit rate for two different caches.
    void clear() {
        for (std::size_t i = 0; i < shard_count_; ++i) {
            Shard &shard = shards_[i];
            SpinlockGuard guard(shard.lock);
            for (Slot &slot : shard.slots) {
                slot.key.reset();
                slot.moves.clear();
                slot.priors.clear();
            }
            shard.stats = CacheStats{};
        }
    }

    // Summed across shards. Each shard's counters are read under its own lock,
    // so this is a consistent snapshot per shard and an approximate one across
    // shards — which is what a statistics call can offer without stopping the
    // world, and is exact in the single-threaded builds this chunk ships.
    CacheStats stats() const {
        CacheStats total;
        for (std::size_t i = 0; i < shard_count_; ++i) {
            const Shard &shard = shards_[i];
            SpinlockGuard guard(shard.lock);
            total.hits += shard.stats.hits;
            total.misses += shard.stats.misses;
            total.inserts += shard.stats.inserts;
            total.collisions += shard.stats.collisions;
            total.refreshes += shard.stats.refreshes;
        }
        return total;
    }

private:
    // One entry.
    //
    // THE EMPTY-SLOT SENTINEL. C3 gives NNKey no default constructor on purpose,
    // so a slot cannot hold a "zero key" meaning empty — and it should not want
    // to: FNV-1a's output covers the whole 64-bit range, so every value
    // including 0 and including the offset basis is a key some payload really
    // hashes to. Reserving one would mean one position in 2^64 is uncacheable
    // and, worse, that a reader has to remember which.
    //
    // `std::optional<NNKey>` moves the question into the type system: the
    // disengaged state is distinct from EVERY NNKey by construction rather than
    // by convention, it costs one byte plus padding beside a key we were storing
    // anyway, and it cannot be forged from a key value. A separate `bool
    // occupied` would be the same idea spelled by hand, with the extra property
    // that the key would be readable while meaningless.
    struct Slot {
        std::optional<NNKey> key;
        // Non-default-constructible, so this needs an initialiser. The value is
        // never read while `key` is disengaged — `probe` returns before touching
        // it — so 0.0 here is inert.
        NetworkValue value{0.0};
        std::vector<std::uint16_t> moves;
        std::vector<float> priors;
    };

    // One shard's state. Split out from `Shard` only so the padding below can be
    // computed from its size.
    struct ShardState {
        mutable Spinlock lock;
        CacheStats stats;
        std::vector<Slot> slots;
    };

    // Cache-line aligned so two shards' locks never share a line. Without this
    // the sharding buys nothing under C9: 64 spinlocks in 512 bytes would
    // false-share into roughly 8 locks, and 64 threads would contend as if
    // there were 8.
    //
    // The explicit tail padding is neither decoration nor a micro-optimisation.
    // `alignas(64)` alone makes MSVC insert exactly this padding itself and then
    // warn (C4324) that it did — correctly, since silent growth from 72 bytes to
    // 128 is worth saying out loud. Global Rule 4 forbids suppressing the
    // warning with a pragma, and rightly. Declaring the padding removes it by
    // making the padding intentional and visible in the source, which is what it
    // always was.
    //
    // The `64 - (size % 64)` form is never zero, so the array is never
    // zero-length (ill-formed); when the state already fills whole lines it adds
    // one spare line rather than none.
    struct alignas(64) Shard : ShardState {
        char cache_line_padding[64 - (sizeof(ShardState) % 64)];
    };

    static_assert(sizeof(Shard) % 64 == 0,
                  "a shard must occupy a whole number of cache lines, or two shards' "
                  "spinlocks can share one and the sharding buys nothing");
    static_assert(alignof(Shard) == 64);

    // SplitMix64's finalizer. FNV-1a is a good hash but its avalanche is
    // weakest in the low bits, and this table takes both its shard index and its
    // slot index from the same word — the shard from the top, the slot from the
    // bottom. Running one finalizer first means neither index inherits a bias,
    // and it is three multiplies on a path that is about to take a lock.
    static std::uint64_t mix(std::uint64_t x) noexcept {
        x ^= x >> 30;
        x *= 0xBF58'476D'1CE4'E5B9ULL;
        x ^= x >> 27;
        x *= 0x94D0'49BB'1331'11EBULL;
        x ^= x >> 31;
        return x;
    }

    // Top bits for the shard, bottom bits for the slot: disjoint fields of one
    // mixed word, so a shard is not systematically fed one region of the slot
    // space. 56 rather than 58 so the field stays put if the shard count is
    // raised; `& shard_mask_` selects however many of those bits are in use.
    std::size_t shard_index(NNKey key) const noexcept {
        return static_cast<std::size_t>(mix(key.value) >> 56) & shard_mask_;
    }

    Shard &shard_for(NNKey key) noexcept { return shards_[shard_index(key)]; }

    std::size_t slot_index(NNKey key) const noexcept {
        return static_cast<std::size_t>(mix(key.value)) & slot_mask_;
    }

    std::unique_ptr<Shard[]> shards_;
    std::size_t shard_count_ = 0;
    std::size_t shard_mask_ = 0;
    std::size_t slot_mask_ = 0;
};

// ---------------------------------------------------------------------------
// ACCEPTANCE CRITERION 5, proved at compile time
//
// "A test that attempts to store a terminal/proof in the cache FAILS TO
// COMPILE." C3 established that shelling out to a compiler proves nothing these
// do not, so this is the whole of it — and unlike a shelled-out compile-fail
// test, it runs on every build, on both toolchains, for free.
//
// CacheInsertAccepts<V> asks the real `TranspositionCache::insert`, through
// exactly the overload resolution a caller would get. It is not a re-statement
// of the signature that could drift away from it: if somebody adds a
// `double`-taking overload, or gives TablebaseValue a conversion to
// NetworkValue, the corresponding assert fires here.
// ---------------------------------------------------------------------------

namespace detail {

template <typename V, typename = void>
struct CacheInsertAccepts : std::false_type {};

template <typename V>
struct CacheInsertAccepts<
    V, std::void_t<decltype(std::declval<TranspositionCache &>().insert(
           std::declval<NNKey>(), std::declval<V>(), std::declval<const std::uint16_t *>(),
           std::declval<const float *>(), std::declval<std::uint16_t>()))>> : std::true_type {};

}  // namespace detail

// The network's own output goes in. Nothing else does.
static_assert(detail::CacheInsertAccepts<NetworkValue>::value,
              "the cache must accept the network's value — otherwise it holds nothing");

static_assert(!detail::CacheInsertAccepts<TerminalValue>::value,
              "a TERMINAL value must not be storable in a position-keyed cache: checkmate "
              "and stalemate are properties of the position, but repetition and the "
              "fifty-move rule are properties of the PATH, and the cache cannot tell them "
              "apart");
static_assert(!detail::CacheInsertAccepts<TablebaseValue>::value,
              "a TABLEBASE value must not be storable: Syzygy WDL ignores the fifty-move "
              "rule, so it is a function of the position AND the halfmove clock, and the "
              "clock is not a token and therefore not part of the key. This is the exact "
              "defect core/mctsv4.py has today — see cpp/values.hpp");
static_assert(!detail::CacheInsertAccepts<ProofValue>::value,
              "a PROOF must not be storable: it is a statement about a subtree, not about "
              "the leaf the key identifies");
static_assert(!detail::CacheInsertAccepts<double>::value,
              "a bare double must not be storable — that is how all three of the above get "
              "in, by reading someone's .value member");
static_assert(!detail::CacheInsertAccepts<bool>::value,
              "a terminal FLAG must not be storable; bool converts to double in C++, so "
              "this crossing is the one the language would otherwise permit silently");
static_assert(!detail::CacheInsertAccepts<float>::value);
static_assert(!detail::CacheInsertAccepts<int>::value);

// The payload type is likewise closed. Even if a caller got past `insert`, there
// is nothing in a CachedEval or a Slot that a forbidden value could be assigned
// to.
static_assert(!std::is_assignable_v<decltype(CachedEval::value) &, TerminalValue>);
static_assert(!std::is_assignable_v<decltype(CachedEval::value) &, TablebaseValue>);
static_assert(!std::is_assignable_v<decltype(CachedEval::value) &, ProofValue>);
static_assert(!std::is_assignable_v<decltype(CachedEval::value) &, double>);
static_assert(!std::is_constructible_v<CachedEval, TerminalValue>);
static_assert(!std::is_constructible_v<CachedEval, TablebaseValue>);

// C3's design, depended on here rather than merely admired: the empty-slot
// sentinel is `std::optional`'s disengaged state PRECISELY BECAUSE NNKey cannot
// be default-constructed. If somebody adds a default constructor, this fires
// and the reader is sent to the comment on Slot.
static_assert(!std::is_default_constructible_v<NNKey>,
              "NNKey must not become default-constructible: the cache's empty-slot "
              "sentinel exists because it is not, and a zero key would otherwise mean "
              "'the payload that hashed to zero'");

}  // namespace guofish

#endif  // GUOFISH_CACHE_HPP
