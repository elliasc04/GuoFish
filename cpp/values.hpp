// GuoFish C7 — the value taxonomy, and the type system that keeps the
// path-dependent values out of the position-keyed cache.
//
// C3 gave the port two KEYS that cannot be mixed. This file does the same job
// for the four kinds of VALUE the search handles, for the same reason and with
// the same mechanism: they are all a `double` in [-1, +1], they are all
// backed up through the same `backpropagate`, and exactly one of them may be
// written to a cache that is keyed by position alone.
//
//
// THE DEFECT THIS EXISTS TO MAKE UNREPRESENTABLE
// ----------------------------------------------
// `core/mctsv4.py`, in `MCTSWorker._run_simulation`:
//
//     if self.tablebase is not None and count_pieces(board) <= TABLEBASE_MAX_PIECES:
//         tb_value = probe_tablebase_value(self.tablebase, board)
//         if tb_value is not None:
//             nn_value = tb_value          # <- the override
//             _was_tb = True
//     if policy is not None:
//         self.cache.put(cache_key, policy, nn_value)   # <- the poisoning
//
// with the comment "We override BEFORE caching so the WDL value is what gets
// stored". The cache is keyed by `make_cache_key`, which is (Zobrist, ep
// square) — a function of the POSITION. A Syzygy WDL is not a function of the
// position: the tables report the result under the assumption that the
// fifty-move rule does not intervene, and whether it intervenes depends on the
// halfmove clock, which the key deliberately does not carry (that is what
// makes clock-twins share an entry, which for a NETWORK value is correct,
// because the clock is not a token).
//
// So a tablebase entry stored at clock 3 is served to the same position at
// clock 99, where the true result is a draw. The reference's own instrumentation
// counts this: `cache_hit_tb_hmc_crossing` exists precisely to measure hits
// where the stored and asked-about clocks straddle 100.
//
// The same argument condemns the other two:
//
//   terminal values   a threefold repetition or a fifty-move draw is a property
//                     of the PATH from the root and of the game before the root.
//                     The same position reached down another line is not drawn.
//                     C6 already keeps these off any shared structure; this
//                     file is what makes "keeps" into "cannot".
//   proofs            a solved/proved score (a mate distance, a solved subtree)
//                     is a statement about a SUBTREE, not about the leaf, and it
//                     is not what the evaluator returned. Nothing produces one
//                     yet; the type exists so that whoever adds proof-number or
//                     mate-distance backup in a later chunk finds the door
//                     already locked rather than discovering it is open.
//
//
// HOW IT IS ENFORCED
// ------------------
// Four distinct struct types, each with an explicit constructor and no
// conversion to or from `double` or to each other. `TranspositionCache::insert`
// takes a `NetworkValue`. Handing it a `TablebaseValue` is then not a mistake
// that a code reviewer has to catch, or that a runtime assertion has to catch,
// or that a comment has to warn about — it is a program that does not compile.
// cpp/cache.hpp asserts exactly that, at compile time, in every build.
//
// The cost is nil: each is a single double with no padding and no vtable,
// passed in a register exactly as the bare double would be, and the asserts at
// the bottom of this file say so.

#ifndef GUOFISH_VALUES_HPP
#define GUOFISH_VALUES_HPP

#include <type_traits>

namespace guofish {

// ---------------------------------------------------------------------------
// The four kinds of value
//
// Each carries the ABSOLUTE (White-POV) convention the v5 value head produces,
// EXCEPT TerminalValue, which is the reference's terminal convention: 1.0 for
// "the side that just moved won", 0.0 for a draw. That difference is itself an
// argument for the separation — two doubles in the same range that are not even
// in the same frame of reference should not be interchangeable.
// ---------------------------------------------------------------------------

// What the evaluator returned for a position: the network's value head output,
// White-POV. THE ONLY VALUE THE TRANSPOSITION CACHE MAY HOLD.
struct NetworkValue {
    double value;

    explicit constexpr NetworkValue(double v) noexcept : value(v) {}
};

// A game result the rules of chess produced: checkmate, stalemate, insufficient
// material, the fifty-move rule, repetition. Mover-POV (1.0 = whoever moved to
// this node won), and PATH-DEPENDENT for the claimable half.
struct TerminalValue {
    double value;

    explicit constexpr TerminalValue(double v) noexcept : value(v) {}
};

// A Syzygy WDL result mapped onto the value scale, White-POV. Exact, and exactly
// as path-dependent as the fifty-move rule it ignores. Tree-local: applied at
// the leaf that was probed and nowhere else.
struct TablebaseValue {
    double value;

    explicit constexpr TablebaseValue(double v) noexcept : value(v) {}
};

// A proved score for a SUBTREE — a mate distance, a solved node. Nothing
// produces one today. The type is here so that the chunk which introduces one
// cannot store it in a leaf cache without deleting a static_assert and
// explaining itself in DECISIONS.md.
struct ProofValue {
    double value;

    explicit constexpr ProofValue(double v) noexcept : value(v) {}
};

// ---------------------------------------------------------------------------
// The separation, proved at compile time
//
// These fire in every build on both toolchains. If one ever trips, somebody has
// weakened the taxonomy — most likely by adding an implicit conversion "for
// convenience" — and the build stops before the poisoning can reach a game.
// ---------------------------------------------------------------------------

namespace detail {

// Stand-ins for "a function that wants one kind of value". Declared, never
// defined, never called: they exist to be asked about with is_invocable_v.
void takes_network_value(NetworkValue);
void takes_terminal_value(TerminalValue);
void takes_tablebase_value(TablebaseValue);
void takes_proof_value(ProofValue);

}  // namespace detail

// Each accepts its own kind...
static_assert(std::is_invocable_v<decltype(detail::takes_network_value), NetworkValue>);
static_assert(std::is_invocable_v<decltype(detail::takes_terminal_value), TerminalValue>);
static_assert(std::is_invocable_v<decltype(detail::takes_tablebase_value), TablebaseValue>);
static_assert(std::is_invocable_v<decltype(detail::takes_proof_value), ProofValue>);

// ...and no other, in either direction. The NetworkValue row is the one that
// matters — it is the cache's parameter type — but all of them are asserted,
// because a taxonomy with one hole is a taxonomy nobody can rely on.
static_assert(!std::is_invocable_v<decltype(detail::takes_network_value), TerminalValue>,
              "a terminal value must not be accepted where a network value is expected");
static_assert(!std::is_invocable_v<decltype(detail::takes_network_value), TablebaseValue>,
              "a tablebase value must not be accepted where a network value is expected");
static_assert(!std::is_invocable_v<decltype(detail::takes_network_value), ProofValue>,
              "a proof must not be accepted where a network value is expected");
static_assert(!std::is_invocable_v<decltype(detail::takes_terminal_value), NetworkValue>);
static_assert(!std::is_invocable_v<decltype(detail::takes_terminal_value), TablebaseValue>);
static_assert(!std::is_invocable_v<decltype(detail::takes_terminal_value), ProofValue>);
static_assert(!std::is_invocable_v<decltype(detail::takes_tablebase_value), NetworkValue>);
static_assert(!std::is_invocable_v<decltype(detail::takes_tablebase_value), TerminalValue>);
static_assert(!std::is_invocable_v<decltype(detail::takes_proof_value), NetworkValue>);
static_assert(!std::is_invocable_v<decltype(detail::takes_proof_value), TerminalValue>);

// A bare double is not any of them. This is the assertion that stops the
// separation being routed around with `insert(key, terminal.value, ...)`: the
// `.value` member is public and readable, but the READ does not produce
// something the cache will take.
static_assert(!std::is_invocable_v<decltype(detail::takes_network_value), double>,
              "a bare double must not be accepted where a network value is expected");
static_assert(!std::is_convertible_v<double, NetworkValue>);
static_assert(!std::is_convertible_v<NetworkValue, double>);
static_assert(!std::is_convertible_v<double, TerminalValue>);
static_assert(!std::is_convertible_v<double, TablebaseValue>);
static_assert(!std::is_convertible_v<double, ProofValue>);

// A terminal FLAG is a bool, and a bool converts to double in C, so this is the
// one crossing the language would otherwise permit silently.
static_assert(!std::is_invocable_v<decltype(detail::takes_network_value), bool>,
              "a terminal flag must not be accepted where a network value is expected");

// No cross-assignment either; is_convertible alone does not cover operator=,
// which is implicitly declared.
static_assert(!std::is_assignable_v<NetworkValue &, TerminalValue>);
static_assert(!std::is_assignable_v<NetworkValue &, TablebaseValue>);
static_assert(!std::is_assignable_v<NetworkValue &, ProofValue>);
static_assert(!std::is_assignable_v<NetworkValue &, double>);
static_assert(!std::is_assignable_v<TerminalValue &, NetworkValue>);
static_assert(!std::is_assignable_v<TablebaseValue &, NetworkValue>);

// No construction across the taxonomy.
static_assert(!std::is_constructible_v<NetworkValue, TerminalValue>);
static_assert(!std::is_constructible_v<NetworkValue, TablebaseValue>);
static_assert(!std::is_constructible_v<NetworkValue, ProofValue>);
static_assert(!std::is_constructible_v<TablebaseValue, NetworkValue>);

// Free at runtime: the wrapper is the double.
static_assert(sizeof(NetworkValue) == sizeof(double));
static_assert(sizeof(TerminalValue) == sizeof(double));
static_assert(sizeof(TablebaseValue) == sizeof(double));
static_assert(sizeof(ProofValue) == sizeof(double));
static_assert(std::is_trivially_copyable_v<NetworkValue>);
static_assert(std::is_trivially_copyable_v<TerminalValue>);
static_assert(std::is_trivially_copyable_v<TablebaseValue>);
static_assert(std::is_trivially_copyable_v<ProofValue>);

// Deliberately NOT default-constructible, for the reason C3 gives for NNKey: a
// zero-initialised value is a valid-looking value (0.0 is "drawn"), so a slot
// that was never written must be distinguishable by its TYPE rather than by its
// contents. cpp/cache.hpp's empty-slot sentinel rests on this.
static_assert(!std::is_default_constructible_v<NetworkValue>);
static_assert(!std::is_default_constructible_v<TerminalValue>);
static_assert(!std::is_default_constructible_v<TablebaseValue>);
static_assert(!std::is_default_constructible_v<ProofValue>);

}  // namespace guofish

#endif  // GUOFISH_VALUES_HPP
