// GuoFish C3 — the two position keys, and the type system that keeps them apart.
//
// A position has two identities in this engine and they are not the same
// identity:
//
//   nn_key   what the network sees. Two positions share an nn_key exactly when
//            they tokenize to the same 68 integers, so a cache hit can never
//            hand back an evaluation computed for a different input.
//   rep_key  what the rules of chess call the same position for repetition and
//            50-move purposes: python-chess's `Board._transposition_key()`.
//
// The historical defect this chunk exists to prevent was not a wrong hash. It
// was the right hash used for the wrong question — a Polyglot Zobrist, which
// follows a *third* en-passant rule, standing in for both. Nothing crashes when
// that happens. The cache serves one position's policy for another, or a draw
// is claimed on a position that never repeated, and the engine simply plays
// slightly worse forever.
//
//
// THE THREE EN-PASSANT RULES
// --------------------------
// All three are correct for their own consumer and no two agree:
//
//   raw            `board.ep_square is not None`. What token 66 is written
//                  from, so what `nn_key` distinguishes. 3,610 of the 100k
//                  corpus positions carry an ep square under this rule.
//   legal capture  `board.has_legal_en_passant()` — some legal move is an ep
//                  capture. What `_transposition_key()` uses, so what `rep_key`
//                  distinguishes. 407 of the same 100k.
//   pseudo-legal   an enemy pawn merely stands adjacent; Polyglot's own spec
//   adjacency      says "legality of the potential capture is irrelevant". 410
//                  of the same 100k, and *not* a superset relation that would
//                  make it a safe approximation — it disagrees with the legal
//                  rule on 3 corpus positions in both directions at once.
//
// Those 3 positions are in golden/keys_adversarial.jsonl as ep_pinned_1..3. They
// are the reason "just use the Zobrist" is wrong rather than merely imprecise.
//
//
// WHY FNV-1a AND NOT ZOBRIST
// --------------------------
// tools/gen_key_golden.py defines both keys as FNV-1a-64 over an explicit byte
// serialisation, and this file reproduces that byte for byte. A Zobrist
// realisation of either rule would have to *re-derive* the ep handling inside
// an incremental update, which is the exact silent divergence above; hashing a
// serialisation puts the rule in one visible place. It also needs no shared
// random table between Python and C++, and is a few lines of standard C++
// (Global Rule 7).
//
// The two domain tags are not decoration. They guarantee nn_key(p) != rep_key(p)
// for every position, so a key that has been swapped by mistake can never
// compare equal by luck. That is the runtime half of "two keys, never mixed";
// the compile-time half is the two struct types below.

#ifndef GUOFISH_KEYS_HPP
#define GUOFISH_KEYS_HPP

#include <chess.hpp>

#include "tokens.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace guofish {

// ---------------------------------------------------------------------------
// The two key types
//
// THE STRUCTURAL REQUIREMENT OF THIS CHUNK. `using NNKey = std::uint64_t;` would
// satisfy every test in tests/test_c3_keys.py and none of the point: an alias
// is the same type, so a repetition check fed a cache key compiles silently and
// is wrong at runtime. These are distinct struct types with explicit
// constructors, so:
//
//   * NNKey does not convert to RepKey or the other way round,
//   * neither converts to or from a bare std::uint64_t implicitly,
//   * `nn == rep` does not compile — there is no heterogeneous operator==,
//   * a function taking RepKey cannot be called with an NNKey.
//
// All four are asserted below, at compile time, so the property is checked by
// every build rather than by a test that would have to be believed.
//
// The cost is nil: each is a single uint64 member with no padding and no vtable,
// passed in a register exactly as the raw integer would be. C7's cache and C6's
// repetition stack store these, not integers.
// ---------------------------------------------------------------------------

// Cache key: two positions share one exactly when they tokenize identically.
struct NNKey {
    std::uint64_t value;

    explicit constexpr NNKey(std::uint64_t v) noexcept : value(v) {}
};

// Repetition key: two positions share one exactly when the rules of chess call
// them the same position.
struct RepKey {
    std::uint64_t value;

    explicit constexpr RepKey(std::uint64_t v) noexcept : value(v) {}
};

constexpr bool operator==(NNKey a, NNKey b) noexcept { return a.value == b.value; }
constexpr bool operator!=(NNKey a, NNKey b) noexcept { return a.value != b.value; }
constexpr bool operator==(RepKey a, RepKey b) noexcept { return a.value == b.value; }
constexpr bool operator!=(RepKey a, RepKey b) noexcept { return a.value != b.value; }

namespace detail {

// True when `A == B` compiles. Used only to assert that it does not.
template <typename A, typename B, typename = void>
struct EqComparable : std::false_type {};

template <typename A, typename B>
struct EqComparable<A, B, std::void_t<decltype(std::declval<const A &>() == std::declval<const B &>())>>
    : std::true_type {};

// The acceptance criterion, spelled as code: a function that wants one key type
// must reject the other. These exist to be asked about, never to be called.
void takes_nn_key(NNKey);
void takes_rep_key(RepKey);

}  // namespace detail

// --- Acceptance criterion 3, proved at compile time ------------------------
// If any of these ever fires, the separation has been weakened and the build
// stops. That is the intended failure mode: a silent key swap is the one bug
// this chunk cannot allow to reach runtime.

// A function expecting one key type rejects the other, and accepts its own.
static_assert(!std::is_invocable_v<decltype(detail::takes_rep_key), NNKey>,
              "an NNKey must not be accepted where a RepKey is expected");
static_assert(!std::is_invocable_v<decltype(detail::takes_nn_key), RepKey>,
              "a RepKey must not be accepted where an NNKey is expected");
static_assert(std::is_invocable_v<decltype(detail::takes_rep_key), RepKey>);
static_assert(std::is_invocable_v<decltype(detail::takes_nn_key), NNKey>);

// No conversion in either direction, and none to or from a raw integer. The
// last two are what stop `NNKey k = some_uint64;` and `foo(rep.value)` style
// mistakes from type-checking their way back in.
static_assert(!std::is_convertible_v<NNKey, RepKey>);
static_assert(!std::is_convertible_v<RepKey, NNKey>);
static_assert(!std::is_constructible_v<NNKey, RepKey>);
static_assert(!std::is_constructible_v<RepKey, NNKey>);
static_assert(!std::is_convertible_v<std::uint64_t, NNKey>);
static_assert(!std::is_convertible_v<std::uint64_t, RepKey>);
static_assert(!std::is_convertible_v<NNKey, std::uint64_t>);
static_assert(!std::is_convertible_v<RepKey, std::uint64_t>);

// Assignment across the two is not possible either — the check `is_convertible`
// alone does not cover, since operator= is implicitly declared.
static_assert(!std::is_assignable_v<NNKey &, RepKey>);
static_assert(!std::is_assignable_v<RepKey &, NNKey>);

// Comparing one against the other does not compile, while comparing like with
// like does. Without this a swapped key would still be *usable* in an `if`.
static_assert(!detail::EqComparable<NNKey, RepKey>::value,
              "NNKey and RepKey must not be comparable to each other");
static_assert(!detail::EqComparable<RepKey, NNKey>::value);
static_assert(!detail::EqComparable<NNKey, std::uint64_t>::value);
static_assert(!detail::EqComparable<RepKey, std::uint64_t>::value);
static_assert(detail::EqComparable<NNKey, NNKey>::value);
static_assert(detail::EqComparable<RepKey, RepKey>::value);

// Free at runtime: the wrapper is the integer.
static_assert(sizeof(NNKey) == sizeof(std::uint64_t));
static_assert(sizeof(RepKey) == sizeof(std::uint64_t));
static_assert(std::is_trivially_copyable_v<NNKey>);
static_assert(std::is_trivially_copyable_v<RepKey>);

// ---------------------------------------------------------------------------
// FNV-1a, 64-bit
//
// Offset basis and prime from the reference definition, which is what
// tools/gen_key_golden.py uses. The multiply is on an unsigned type, so the
// wraparound the algorithm depends on is defined behaviour rather than UB.
// ---------------------------------------------------------------------------

class Fnv1a64 {
public:
    void byte(std::uint8_t b) noexcept {
        hash_ ^= static_cast<std::uint64_t>(b);
        hash_ *= kPrime;
    }

    // Little-endian, written out rather than memcpy'd from the object
    // representation. The golden payloads are defined as little-endian byte
    // strings (`struct.pack("<...")`, `astype("<i4")`), and spelling the byte
    // order removes the host's from the answer. Global Rule 8 asks for both
    // toolchains; this asks for nothing of the hardware either.
    void u64_le(std::uint64_t v) noexcept {
        for (int i = 0; i < 8; ++i) {
            byte(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFFU));
        }
    }

    void i32_le(std::int32_t v) noexcept {
        const auto u = static_cast<std::uint32_t>(v);  // two's complement, value-preserving bit pattern
        for (int i = 0; i < 4; ++i) {
            byte(static_cast<std::uint8_t>((u >> (8 * i)) & 0xFFU));
        }
    }

    // A domain tag as a string literal. `N` counts the terminating NUL, and the
    // NUL is deliberately hashed: the Python tags are written b"...\0" so that
    // no tag can be a prefix of another.
    template <std::size_t N>
    void tag(const char (&literal)[N]) noexcept {
        static_assert(N >= 2, "a domain tag must not be empty");
        for (std::size_t i = 0; i < N; ++i) {
            byte(static_cast<std::uint8_t>(literal[i]));
        }
    }

    std::uint64_t value() const noexcept { return hash_; }

private:
    static constexpr std::uint64_t kOffsetBasis = 0xCBF2'9CE4'8422'2325ULL;
    static constexpr std::uint64_t kPrime = 0x0000'0100'0000'01B3ULL;

    std::uint64_t hash_ = kOffsetBasis;
};

// The tags, exactly as tools/gen_key_golden.py spells them.
inline constexpr char kNNKeyTag[] = "guofish/nn_key/v1";
inline constexpr char kRepKeyTag[] = "guofish/rep_key/v1";

// rep_key's "no en-passant capture is available" sentinel. 0xFF cannot collide
// with a square index, which is 0..63.
inline constexpr std::uint8_t kRepKeyNoEp = 0xFF;

// ---------------------------------------------------------------------------
// The legal-en-passant rule
//
// python-chess:
//
//     has_legal_en_passant() = ep_square is not None and any(generate_legal_ep())
//
// and `generate_legal_ep` is `generate_pseudo_legal_ep` filtered by
// `not is_into_check(move)`. The pseudo-legal half is three conditions on
// bitboards, transcribed directly below. The legality half is the question "is
// the mover's own king attacked once the capture has been played", and it is
// answered here by building the resulting occupancy and asking, rather than by
// transcribing python-chess's decomposition of it into pin masks and skewers.
//
// WHY NOT JUST ASK chess-library. `Board::setFen` runs `isEpSquareValid` and
// silently nulls an ep square with no legal capture, so `enpassantSq() != NO_SQ`
// looks like a free implementation of this rule. It is very nearly one, and it
// was rejected for two reasons. Its ep test omits the diagonal skewer —
// python-chess checks it, with a comment that it cannot arise in a real game,
// and "cannot arise in a real game" is not the same as "cannot arise in a FEN
// handed to a public entry point". And it costs a full setFen — castling paths,
// Zobrist, movegen tables — for a question that is a handful of bitboard
// operations, on a path C6 will call once per node.
//
// What is borrowed from chess-library is only `attacks::` — the magic bitboard
// tables. Those are pure functions of (square, occupancy) with no board state
// and no opinion about en passant.
// ---------------------------------------------------------------------------

// Rank 5 for White, rank 4 for Black: python-chess's `BB_RANKS[4 if turn else 3]`,
// the rank a pawn must stand on to capture en passant at all.
inline constexpr Bitboard kRank5 = 0x0000'00FF'0000'0000ULL;
inline constexpr Bitboard kRank4 = 0x0000'0000'FF00'0000ULL;

inline chess::Bitboard bb(Bitboard value) { return chess::Bitboard(value); }

// Is the square `king_square` attacked by a piece of the colour opposite to
// `white`, given the post-capture position described by the arguments?
//
// `occupied_after` is the full occupancy including the moved pawn; `enemy` is
// the attacking side's occupancy with the captured pawn already removed. The two
// are passed separately because a slider's ray depends on every piece, while
// whether the ray *ends* on an attacker depends only on the enemy's.
inline bool square_is_attacked(const Placement &placement, int king_square, bool white, Bitboard occupied_after,
                               Bitboard enemy) {
    const chess::Square ksq(king_square);
    const chess::Bitboard occ = bb(occupied_after);
    const chess::Bitboard foes = bb(enemy);

    if (chess::attacks::bishop(ksq, occ) & foes & bb(placement.bishops | placement.queens)) {
        return true;
    }
    if (chess::attacks::rook(ksq, occ) & foes & bb(placement.rooks | placement.queens)) {
        return true;
    }
    if (chess::attacks::knight(ksq) & foes & bb(placement.knights)) {
        return true;
    }
    // A pawn attacks `ksq` exactly when a pawn of the king's own colour standing
    // on `ksq` would attack it back.
    const chess::Color us = white ? chess::Color::WHITE : chess::Color::BLACK;
    if (chess::attacks::pawn(us, ksq) & foes & bb(placement.pawns)) {
        return true;
    }
    if (chess::attacks::king(ksq) & foes & bb(placement.kings)) {
        return true;
    }
    return false;
}

// python-chess's `has_legal_en_passant()`.
inline bool has_legal_en_passant(const ParsedFen &parsed) {
    const int ep = parsed.ep_square;
    if (ep < 0) {
        return false;
    }

    const Placement &placement = parsed.placement;
    const bool white = parsed.white_to_move;
    const Bitboard occupied = placement.occupied();

    // `if BB_SQUARES[self.ep_square] & self.occupied: return`. A FEN may name an
    // occupied square here; python-chess declines to invent a capture into it.
    if (occupied & square_bb(ep)) {
        return false;
    }

    // The pawns that could make the capture: ours, attacking the ep square, and
    // standing on the only rank from which that is possible.
    const Bitboard capturer_rank = white ? kRank5 : kRank4;
    const chess::Color them = white ? chess::Color::BLACK : chess::Color::WHITE;
    const Bitboard capturers = placement.pawns & placement.occupied_of(white) &
                               chess::attacks::pawn(them, chess::Square(ep)).getBits() & capturer_rank;
    if (capturers == 0) {
        return false;
    }

    // The square the captured pawn stands on — one rank *behind* the ep square
    // from the mover's point of view. python-chess computes it the same way and,
    // notably, does not check that a pawn is actually there.
    const int captured_square = white ? ep - 8 : ep + 8;
    assert(captured_square >= 0 && captured_square < 64);

    // python-chess's `king(color)`: the highest non-promoted king of that
    // colour. With no such king, `is_into_check` answers False for every move,
    // so every pseudo-legal ep capture is legal — matched here rather than
    // treated as an error, because this function must agree with the reference
    // on every FEN the reference accepts, including kingless ones.
    const int king_square = msb_index(placement.kings & placement.occupied_of(white) & ~placement.promoted);
    if (king_square < 0) {
        return true;
    }

    Bitboard remaining = capturers;
    while (remaining != 0) {
        const Bitboard from = lsb_mask(remaining);
        remaining &= remaining - 1;

        // The position after the capture: the capturer leaves, the captured pawn
        // leaves, the ep square is now occupied by the capturer.
        const Bitboard occupied_after = (occupied & ~from & ~square_bb(captured_square)) | square_bb(ep);
        const Bitboard enemy = placement.occupied_of(!white) & ~square_bb(captured_square);

        if (!square_is_attacked(placement, king_square, white, occupied_after, enemy)) {
            return true;
        }
    }

    return false;
}

// ---------------------------------------------------------------------------
// The keys
// ---------------------------------------------------------------------------

// The NN cache key: FNV-1a over the tag and the 68 tokens as little-endian
// int32, which is the byte image of the buffer C2's batch path already holds.
//
// The halfmove clock is absent because it is not a token, and that is the
// correct behaviour rather than an oversight: the network's input does not
// contain the clock, so two positions differing only in the clock must share an
// evaluation. tests/test_c3_keys.py pins it so that a later reader cannot
// "fix" it into a proof-caching bug.
inline NNKey nn_key(const ParsedFen &parsed) {
    std::int32_t tokens[kSeqLength];
    tokenize_into(parsed, tokens);

    Fnv1a64 hash;
    hash.tag(kNNKeyTag);
    for (int i = 0; i < kSeqLength; ++i) {
        hash.i32_le(tokens[i]);
    }
    return NNKey(hash.value());
}

// The repetition key: FNV-1a over the tag and the fields of python-chess's
// `Board._transposition_key()`, in its order —
//
//     (pawns, knights, bishops, rooks, queens, kings,
//      occupied_co[WHITE], occupied_co[BLACK], turn, clean_castling_rights(),
//      ep_square if has_legal_en_passant() else None)
//
// serialised as eight little-endian uint64, one byte of side to move, one
// little-endian uint64 of castling rights and one byte of en passant.
inline RepKey rep_key(const ParsedFen &parsed) {
    const Placement &placement = parsed.placement;

    Fnv1a64 hash;
    hash.tag(kRepKeyTag);
    hash.u64_le(placement.pawns);
    hash.u64_le(placement.knights);
    hash.u64_le(placement.bishops);
    hash.u64_le(placement.rooks);
    hash.u64_le(placement.queens);
    hash.u64_le(placement.kings);
    hash.u64_le(placement.occupied_white);
    hash.u64_le(placement.occupied_black);
    hash.byte(parsed.white_to_move ? 1U : 0U);
    hash.u64_le(parsed.castling);
    hash.byte(has_legal_en_passant(parsed) ? static_cast<std::uint8_t>(parsed.ep_square) : kRepKeyNoEp);

    return RepKey(hash.value());
}

// FEN entry points. Both throw std::invalid_argument (ValueError in Python) on a
// FEN python-chess would itself refuse.
inline NNKey nn_key(std::string_view fen) { return nn_key(parse_fen(fen)); }
inline RepKey rep_key(std::string_view fen) { return rep_key(parse_fen(fen)); }

}  // namespace guofish

#endif  // GUOFISH_KEYS_HPP
