// GuoFish C5/C6 — the single-threaded search core, on a replay evaluator.
//
// This is the chunk the port fails on if it fails. Everything before it was
// parity on a pure function of a FEN: hand both sides the same string, compare
// the answer. Here the two implementations run a stateful loop 5,000 times and
// have to agree on every node of the tree it builds, to the last bit of a
// double. There is no tolerance to hide in — a difference of one ulp in one
// prior changes which child wins a PUCT comparison, and from there the trees
// diverge structurally.
//
//
// WHY THERE IS NO NETWORK IN HERE
// ------------------------------
// Gate 1 asks whether the traverse loop is right, not whether C++ can reproduce
// ATen's softmax. It cannot: ATen's CPU path reduces through a vectorised exp
// (SLEEF/AVX) rather than libm's, so a hand-rolled softmax differs in the last
// ulp and that difference compounds through selection. So Python runs once with
// the real checkpoint and dumps
//
//     nn_key -> (legal moves in canonical order, priors in canonical order, value)
//
// and this file replays it. The NN side is then identical by construction and
// any divergence is *provably* in selection, backup, virtual loss or ordering.
//
// The lookup is also the strongest test in the chunk, and it is free. C++
// reaches a position by makeMove from the root, tokenizes its own search state,
// and hashes those tokens into an nn_key. If its in-search tokenization derives
// so much as a different en-passant square, that key is not in the dump and the
// lookup misses. So a miss is a hard, named failure carrying the FEN, the move
// path and the key — never a default value. See ReplayMiss.
//
//
// THE EN-PASSANT TRAP, WHICH IS THE REASON THIS FILE HAS ITS OWN BOARD STATE
// -------------------------------------------------------------------------
// Token 66 is written from python-chess's `board.ep_square`, which is set after
// ANY double pawn push, with no test for whether a capture is available. That is
// the RAW rule, it is the trained contract (C2), and `nn_key` inherits it.
//
// chess-library does not have that rule anywhere. From `makeMove` at the pinned
// revision 53e6a84:
//
//     if (Square::value_distance(move.to(), move.from()) == 16) {
//         Bitboard ep_mask = attacks::pawn(stm_, move.to().ep_square());
//         if (ep_mask & pieces(PAWN, ~stm_)) {          // <- pseudo-legal adjacency
//             if constexpr (EXACT) { ...isEpSquareValid... }   // <- legal capture
//             if (found != 0) ep_sq_ = move.to().ep_square();
//         }
//     }
//
// so `makeMove<false>` (the default) sets ep on pseudo-legal adjacency and
// `makeMove<true>` on the legal rule. **Neither is the raw rule.** A search that
// read `Board::enpassantSq()` for token 66 would mis-key every position where a
// pawn double-pushed with no enemy pawn beside it — which is most of them — and
// nothing would crash; the dump lookup would simply miss, or worse, in
// production the cache would serve one position's policy for another.
//
// Two more routes look free and are also wrong, so both are prohibited rather
// than merely avoided:
//
//   Board::fen() / setFen()   `fen()` emits `ep_sq_` unfiltered and `setFen()`
//                             re-filters it through `isEpSquareValid`, so a
//                             round trip is lossy on exactly the positions that
//                             matter (3,203 of the C1 corpus's 3,610 ep FENs).
//   Board::hash()             carries the pseudo-legal ep rule, i.e. a third
//                             convention again, and it is not either key.
//
// SearchBoard below therefore carries its own `raw_ep_`, derived from the move
// just made — double push always sets it, which is python-chess's rule by
// construction — with a stack so `unmake_move` restores it exactly. The
// library's own ep state is left to do what it is for: generating moves.
// tests/test_c5_ep_pin.py pins the library behaviour above so that a future
// re-pin fails a test rather than a parity run.
//
//
// C6: HOW A GAME ENDS, AND WHY THAT IS TWO QUESTIONS AND NOT ONE
// --------------------------------------------------------------
// The reference splits the ways a game can end across two code paths that look
// alike and are not:
//
//   INTRINSIC   `board.is_game_over()` — checkmate, stalemate, insufficient
//               material, seventy-five moves, fivefold repetition. A property of
//               the position, needing no claim from anybody. The node is marked
//               terminal and the result is CACHED ON THE NODE for later visits.
//   CLAIMABLE   `_draw_by_rule` — the fifty-move rule and threefold repetition.
//               NOT a property of the position: it depends on the path from the
//               root and on the game history before the root. The node is marked
//               terminal with 0.0, and the value must never reach a
//               position-keyed cache (C7's problem; the type discipline starts
//               here — see draw_by_rule, which returns a value the caller backs
//               up directly and hands to nothing else).
//
// The depth cap is a third thing again and is neither: it backs up 0.0 and does
// NOT mark the node, because a capped node is not a game result and marking it
// would misreport the position if the search ever resumed from it.
//
// The structural fix is in cpp/arena.hpp and is inherited rather than added
// here: TERMINAL is a bit beside the lifecycle, not a lifecycle value. Python
// wrote `is_expanded = True` on a checkmate node whose `children` dict was
// empty, and `bestmove 0000` followed the first time such a node was promoted to
// a search root. Here `set_children` refuses a zero count outright, so the state
// cannot be spelled — and a terminal node is left Unexpanded, so promoting it to
// a root forces a real expansion and a real move. See
// tests/test_c6_terminal_invariants.py.
//
// This file does not implement the claimable draws by asking chess-library. Its
// `isGameOver()` reports the threefold/fifty-move pair — the claimable ones —
// and not the seventy-five/fivefold pair python-chess ends a game on, and its
// repetition count runs over its own move stack rather than over the game
// history the root was handed. See cpp/terminal.hpp.

#ifndef GUOFISH_SEARCH_HPP
#define GUOFISH_SEARCH_HPP

#include <chess.hpp>

#include "arena.hpp"
#include "cache.hpp"
#include "evaluator.hpp"
#include "keys.hpp"
#include "movegen.hpp"
#include "parallel.hpp"
#include "tablebase.hpp"
#include "terminal.hpp"
#include "tokens.hpp"
#include "values.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace guofish {

// ---------------------------------------------------------------------------
// Configuration
//
// Every field is the reference's, at the reference's default, except
// `c_factor`, which does not exist in Python at all. Scope 3 adds it as a
// tunable for the later tuning chat; it defaults to 1.0 and — see c_puct below —
// is applied in a way that cannot perturb the default result.
// ---------------------------------------------------------------------------

struct SearchConfig {
    double c_init = 1.43;         // C_PUCT_INIT
    double c_base = 19652.0;      // C_PUCT_BASE
    double c_factor = 1.0;        // NEW; not in the reference
    double fpu_root = 0.0;        // FPU_ROOT
    double fpu_tree = 0.30;       // FPU_TREE
    double virtual_loss = 0.0;    // VIRTUAL_LOSS; production is 2.5
    int max_tree_depth = 80;      // MAX_TREE_DEPTH
    std::size_t arena_capacity = 1u << 21;

    // C7. `cache_slots == 0` means NO CACHE — the search holds an empty
    // std::optional and never probes. That is the default, deliberately:
    //
    //  * C5 and C6 were certified against the reference at `cache_size=1`, and
    //    a chunk that silently changed the code path under their tests would
    //    make a C7 regression look like a C5 one;
    //  * "cache off" and "a cache that can hold nothing" must not be the same
    //    object, because the second passes every tree-equivalence test while
    //    doing no work (see cpp/cache.hpp's constructor).
    //
    // tests/test_c7_cache.py turns it on explicitly and re-runs the whole Gate 1
    // corpus with it on, which is the chunk's acceptance criterion.
    std::size_t cache_slots = 0;
    std::size_t cache_shards = kDefaultCacheShards;

    // C8. What fraction of a PONDERED root's inherited visits survives its
    // promotion. 1.0 is no decay and is the default, because the reference has
    // no such mechanism and Gate 1 across `apply_move` has to reproduce the
    // reference exactly — a decay that fired by default would fail the gate on
    // the first ply, which is the correct outcome for a knob that changed
    // behaviour without being asked.
    //
    // The knob exists because the brief requires it and because scope §8 says
    // why: 30k fresh simulations cannot redistribute against 64k+ inherited
    // ones, so a wrong ponder prediction leaves the next search dominated by
    // evaluations of a line nobody played. Whether it should be below 1.0, and
    // by how much, is deferred to post-port measurement by that same section —
    // this chunk provides the lever, not the setting.
    //
    // It is applied only when the caller says the promotion is a ponder
    // promotion (`apply_move(move, from_ponder=true)`). The search cannot know
    // that by itself: pondering lives above this layer.
    double ponder_decay = 1.0;

    // C11b. T in `softmax(logits / T)`, applied at the root and at every
    // interior node alike. 1.0f is the identity, is the default, and SKIPS the
    // divide — see `apply_policy_temperature` in cpp/evaluator.hpp for the
    // three properties that depend on that (divide not reciprocal, bit-identity
    // at 1.0, and float rather than double).
    //
    // THE ONE `float` IN A STRUCT OF `double`s, and that is the point. The
    // reference divides a float32 tensor by a Python float, which torch's
    // weak-scalar rule performs in float32; a double here would promote every
    // logit, divide in double and round back, which is a different number.
    //
    // TWO PATHS REFUSE IT RATHER THAN IGNORE IT. The replay dump replays
    // priors that were softmaxed elsewhere, so a temperature could not reach
    // them; and the Gate 1 equivalence build replays a T = 1.0 reference dump,
    // where a non-identity temperature would be a silent mismatch. Both throw
    // at construction or at search entry — see the constructor.
    float policy_temperature = 1.0f;

    // C8. Run the full-tree structural diff after every compaction, even in a
    // build with asserts off.
    //
    // Scope §7 names "ping-pong arena pointer fixup bugs" as a standing risk and
    // "validate by full-tree structural diff against the pre-copy tree" as the
    // mitigation, so the diff is not a test fixture — it is the mitigation, and
    // it lives in the engine. Builds with asserts on run it unconditionally
    // (see compact_and_promote); this flag is how a Release build opts in. It
    // costs one extra O(nodes) pass over two arenas, i.e. roughly what the
    // compaction itself costs.
    bool verify_compaction = false;
};

// AlphaZero's visit-scaled exploration constant, exactly as SearchParams.c_puct
// spells it:
//
//     c(N) = c_init + log((N + c_base + 1) / c_base)
//
// The association is written out rather than left to the compiler because the
// reference's is fixed by Python's left-to-right evaluation of
// `parent_visits + self.c_base + 1.0`, and floating-point addition is not
// associative. `(pv + c_base) + 1.0`, then the divide, then the log.
//
// EXPOSED TO PYTHON on purpose. Both sides of the equivalence harness can then
// use this one implementation, so a difference between CPython's `math.log` and
// this translation unit's `std::log` cannot fail the gate by itself.
// tests/test_c5_ep_pin.py measures whether such a difference exists at all on
// this toolchain; as of C5 it does not, so the golden generator was left on
// pure Python and Global Rule 2 stays untouched.
inline double c_puct(double parent_visits, double c_init, double c_base) noexcept {
    return c_init + std::log((parent_visits + c_base + 1.0) / c_base);
}

// ---------------------------------------------------------------------------
// Failures
//
// Both of these are conditions the brief requires to be loud and named. They
// derive from std::runtime_error so pybind11 surfaces them as RuntimeError with
// the message intact.
// ---------------------------------------------------------------------------

class ReplayMiss : public std::runtime_error {
public:
    explicit ReplayMiss(const std::string &what) : std::runtime_error(what) {}
};

// C5 threw this on reaching a position with no legal moves. C6 handles those
// positions, so the only thing left that can raise it is a ROOT with no legal
// moves — a search asked to move from a finished game. The reference answers
// that with `bestmove 0000` via a None return; here it is an exception, because
// the replay dump has no way to represent a position with no children and
// pretending otherwise is how the original defect got in.
class TerminalReached : public std::runtime_error {
public:
    explicit TerminalReached(const std::string &what) : std::runtime_error(what) {}
};

// C8. The compacting copy produced a tree that is not the tree it copied.
//
// Separate from every other failure here because it means something different
// from all of them: not "the reference and this disagree" but "this arena no
// longer describes a tree". An off-by-one in a remapped `children_offset` does
// not crash and does not produce a wrong move — it produces a subtree stitched
// out of the wrong siblings, which the next search then explores confidently.
// Scope §7 lists exactly this as the ping-pong risk and a full-tree structural
// diff as the mitigation; this is what that diff throws.
class TreeCorruption : public std::runtime_error {
public:
    explicit TreeCorruption(const std::string &what) : std::runtime_error(what) {}
};

// ---------------------------------------------------------------------------
// Board state with the raw en-passant square
// ---------------------------------------------------------------------------

// Convert a chess-library promotion piece into the arena's promotion code.
// The arena's codes are ALPHABETICAL by UCI letter (b < n < q < r) so that the
// packed move's low nibble is already in canonical order; the library's are
// ordered by piece value. Getting this backwards would reorder the four
// promotion children, which — because they share one policy index and therefore
// one prior — is exactly the case where PUCT ties are decided by child order.
inline Promotion promotion_code(chess::PieceType pt) {
    switch (pt.internal()) {
        case chess::PieceType::underlying::QUEEN:
            return Promotion::Queen;
        case chess::PieceType::underlying::ROOK:
            return Promotion::Rook;
        case chess::PieceType::underlying::BISHOP:
            return Promotion::Bishop;
        case chess::PieceType::underlying::KNIGHT:
            return Promotion::Knight;
        default:
            break;
    }
    throw std::logic_error("guofish: promotion move carries a non-promotable piece type");
}

// A generated move as the arena stores it: (from, NORMALISED to, promotion).
//
// `uci_destination` is C1's castling normalisation — chess-library encodes
// castling as king-takes-rook, so the raw `to()` is the ROOK's square. Packing
// the raw square would sort castling under the wrong key and, worse, would not
// match the move list the golden dump records, so the mismatch check in
// expand() would fire on every castling position.
inline std::uint16_t packed_of(const chess::Board &board, chess::Move move) {
    const Promotion promo = (move.typeOf() == chess::Move::PROMOTION)
                                ? promotion_code(move.promotionType())
                                : Promotion::None;
    return pack_move(move.from().index(), uci_destination(board, move).index(), promo);
}

// Generate `board`'s legal moves, normalise them, and put them in canonical
// `(from, to, promotion)` order.
//
// Free function rather than a ReplaySearch member because C7 gives it a second
// caller — the tablebase root probe, which is a UCI-layer bypass and has no
// search. One implementation matters here for the usual reason: PUCT resolves
// ties by child order, so two orderings that agree "almost always" are the kind
// of divergence Gate 1 exists to prevent.
//
// Sorting a small vector of tuples rather than the Movelist itself keeps
// chess-library's generation order out of the answer entirely — except where
// C10 needs it back, which is the fourth field.
//
// `generation_index`, when asked for, comes out parallel to `packed`:
// `(*generation_index)[k]` is the position move k held in chess-library's
// generation order. It exists for exactly one caller — the live evaluator's
// gather, which must softmax in generation order and only then permute into
// canonical order (scope §2.6, cpp/evaluator.hpp's header). Nothing else may
// read it: reproducing a library's internal generation order is the fragile
// dependency canonical ordering was chosen to avoid, and the ONLY thing this
// index is allowed to decide is a floating-point reduction order.
inline void generate_canonical_moves(const chess::Board &board, std::vector<std::uint16_t> &packed,
                                     std::vector<std::uint16_t> &raw,
                                     std::vector<std::uint16_t> *generation_index = nullptr) {
    chess::Movelist movelist;
    chess::movegen::legalmoves(movelist, board);

    // (canonical key, packed move, raw library move, generation index).
    std::vector<std::array<std::uint16_t, 4>> entries;
    entries.reserve(static_cast<std::size_t>(movelist.size()));
    for (int i = 0; i < movelist.size(); ++i) {
        const chess::Move move = movelist[i];
        const std::uint16_t p = packed_of(board, move);
        entries.push_back({canonical_move_key(p), p, move.move(),
                           static_cast<std::uint16_t>(i)});
    }
    std::sort(entries.begin(), entries.end(),
              [](const std::array<std::uint16_t, 4> &a, const std::array<std::uint16_t, 4> &b) {
                  return a[0] < b[0];
              });

    packed.clear();
    raw.clear();
    packed.reserve(entries.size());
    raw.reserve(entries.size());
    if (generation_index != nullptr) {
        generation_index->clear();
        generation_index->reserve(entries.size());
    }
    for (const auto &entry : entries) {
        packed.push_back(entry[1]);
        raw.push_back(entry[2]);
        if (generation_index != nullptr) {
            generation_index->push_back(entry[3]);
        }
    }
}

class SearchBoard {
public:
    // Load `fen`, taking the RAW en-passant square from the FEN text rather
    // than from the Board.
    //
    // Order matters: parse_fen() is C2's direct FEN parser and is the only thing
    // in the process that sees the ep field verbatim. set_fen_or_throw() hands
    // the same string to chess-library, which filters the ep square on the way
    // in — correctly, for its own purposes, since it only wants to know whether
    // to generate the capture.
    void set_fen(std::string_view fen) {
        const ParsedFen parsed = parse_fen(fen);
        set_fen_or_throw(board_, fen);
        raw_ep_ = parsed.ep_square;
        ep_stack_.clear();
        clock_stack_.clear();

        // The halfmove clock is read off chess-library rather than re-parsed,
        // because the FEN parser in cpp/tokens.hpp deliberately consumes the
        // field without validating it (C2: "knowingly laxer than the
        // reference"). setFen has already validated it here.
        halfmove_clock_ = static_cast<int>(board_.halfMoveClock());

        // Self-audit, every position, both builds. `parsed()` below rebuilds a
        // ParsedFen from the Board's bitboards; this asserts that the rebuild
        // agrees with the FEN parser C2 verified against 100k positions. If the
        // two ever disagree — a castling-rights cleaning difference, a square
        // numbering slip — the search would tokenize every node of the tree
        // differently from the reference, and the only symptom would be a dump
        // miss at some arbitrary depth. Catching it here names the cause.
        if (nn_key(parsed) != nn_key(parsed_from_board())) {
            throw std::logic_error(
                "guofish::SearchBoard::set_fen: the board-derived tokenization "
                "disagrees with the FEN-derived one for: " + std::string(fen));
        }
    }

    const chess::Board &board() const noexcept { return board_; }

    bool white_to_move() const noexcept { return board_.sideToMove() == chess::Color::WHITE; }

    // python-chess's `board.ep_square`: a square index, or -1 for None.
    int raw_ep() const noexcept { return raw_ep_; }

    // python-chess's `board.halfmove_clock`. Tracked here as an `int` rather
    // than read off `Board::halfMoveClock()` at each step: chess-library stores
    // it in a `std::uint8_t`, and while the search cannot in fact drive it past
    // 150 (a node at 100 is a fifty-move draw and is never descended through),
    // a silent wrap at 255 in a rule this chunk is *about* is not a dependency
    // worth taking on a field's width.
    int halfmove_clock() const noexcept { return halfmove_clock_; }

    // Make `move`, deriving the next raw ep square from the move itself.
    //
    // A double pawn push ALWAYS sets it. That is python-chess's rule verbatim
    // (`Board.push`: `if piece_type == PAWN and abs(diff) == 16: self.ep_square
    // = ...`), with no adjacency test and no legality test, and it is the rule
    // token 66 needs. Deriving it from the move rather than reading it back off
    // the Board is the whole point — see this file's header.
    //
    // `zeroing` is python-chess's `Board.is_zeroing(move)`, computed by the
    // caller from the position BEFORE the move (which the caller is holding as
    // a ParsedFen anyway). Passing it in rather than recomputing it here avoids
    // a second full rebuild of that ParsedFen per descent step.
    void make_move(chess::Move move, bool zeroing) {
        ep_stack_.push_back(raw_ep_);
        clock_stack_.push_back(halfmove_clock_);

        int next_ep = -1;
        if (board_.at<chess::PieceType>(move.from()) == chess::PieceType::PAWN) {
            const int from = move.from().index();
            const int to = move.to().index();
            const int delta = to - from;
            if (delta == 16 || delta == -16) {
                next_ep = (from + to) / 2;
            }
        }

        board_.makeMove(move);
        raw_ep_ = next_ep;
        // `Board.push`: increment unconditionally, then zero if the move was a
        // capture or a pawn move. Written in that order because that is the
        // reference's order, not because the two-step matters arithmetically.
        halfmove_clock_ += 1;
        if (zeroing) {
            halfmove_clock_ = 0;
        }
    }

    // Undo `move`. chess-library restores its own ep square from its
    // `prev_states_` stack — verified — and this restores ours from ours, so
    // the two stay independent all the way back up.
    void unmake_move(chess::Move move) {
        assert(!ep_stack_.empty());
        assert(!clock_stack_.empty());
        board_.unmakeMove(move);
        raw_ep_ = ep_stack_.back();
        ep_stack_.pop_back();
        halfmove_clock_ = clock_stack_.back();
        clock_stack_.pop_back();
    }

    // C8. Make the position we now stand on the FLOOR of the undo stacks.
    //
    // `apply_move` plays a move that is never taken back: the game has moved on.
    // Dropping the two stacks says so structurally — a descent that tried to
    // unmake past the new root would now trip `unmake_move`'s assert instead of
    // quietly restoring a pre-root en-passant square and halfmove clock, which
    // is a state no simulation should ever be able to reach. It also keeps the
    // stacks from growing one entry per ply for the length of a game.
    //
    // chess-library's own `prev_states_` is not cleared: it is the library's
    // business, `unmakeMove` is never called past the floor anyway, and reaching
    // into it would be exactly the coupling this class exists to avoid.
    void commit() {
        ep_stack_.clear();
        clock_stack_.clear();
    }

    // The position in the terms C2's tokenizer and C3's keys are defined in.
    ParsedFen parsed() const { return parsed_from_board(); }

    // Is the side to move in check? Delegated to chess-library, which is the
    // only thing here that owns attack tables; python-chess's `is_check()` is
    // the same question and there is no convention to disagree about.
    bool in_check() const { return board_.inCheck(); }

    // A FEN for diagnostics AND for promotion probes — see
    // ReplaySearch::terminal_nodes. Never fed back into THIS board (that is the
    // prohibited round trip), and it prints OUR raw ep square rather than the
    // library's, which is what makes it both correct to re-load elsewhere and
    // useful when a dump miss has to be explained.
    std::string diagnostic_fen() const;

private:
    ParsedFen parsed_from_board() const {
        ParsedFen out;
        Placement &placement = out.placement;

        placement.pawns = board_.pieces(chess::PieceType::PAWN).getBits();
        placement.knights = board_.pieces(chess::PieceType::KNIGHT).getBits();
        placement.bishops = board_.pieces(chess::PieceType::BISHOP).getBits();
        placement.rooks = board_.pieces(chess::PieceType::ROOK).getBits();
        placement.queens = board_.pieces(chess::PieceType::QUEEN).getBits();
        placement.kings = board_.pieces(chess::PieceType::KING).getBits();
        placement.occupied_white = board_.us(chess::Color::WHITE).getBits();
        placement.occupied_black = board_.us(chess::Color::BLACK).getBits();

        // python-chess's `promoted` bitboard tracks the X-FEN "~" suffix and the
        // squares its own push() promoted onto. It is left empty here, and that
        // is exact rather than approximate: every reader of `promoted` in C2 and
        // C3 masks it against `kings` (`self.kings & ~self.promoted`, in
        // clean_castling_rights and in both has_*_castling_rights), and a
        // promoted piece is never a king. See DECISIONS.md.
        placement.promoted = 0;

        for (int square = 0; square < 64; ++square) {
            const chess::Piece piece = board_.at(chess::Square(square));
            const int index = static_cast<int>(piece.internal());
            // Piece::underlying is WHITEPAWN..WHITEKING, BLACKPAWN..BLACKKING,
            // NONE — 0..11 then 12. The token encoding is piece_type (1..6) for
            // White and piece_type + 6 for Black, which is index + 1 for both
            // halves, so no colour branch is needed.
            placement.square_token[square] =
                (index >= 12) ? kTokenEmpty : static_cast<std::int32_t>(index + 1);
        }

        out.white_to_move = white_to_move();
        out.castling = clean_castling_rights(castling_rook_mask(), placement);
        out.ep_square = raw_ep_;
        return out;
    }

    // The rook squares chess-library's castling rights refer to, as the
    // bitboard python-chess's clean_castling_rights() consumes. Reading the
    // rook FILE rather than assuming a1/h1/a8/h8 keeps this correct if the
    // library is ever handed a Chess960 board; the cleaning pass then discards
    // anything the standard rules would not allow.
    Bitboard castling_rook_mask() const {
        Bitboard mask = 0;
        // Returned by value by chess-library; copying four Files is free.
        const chess::Board::CastlingRights rights = board_.castlingRights();
        const chess::Color colors[2] = {chess::Color::WHITE, chess::Color::BLACK};
        const chess::Board::CastlingRights::Side sides[2] = {
            chess::Board::CastlingRights::Side::KING_SIDE,
            chess::Board::CastlingRights::Side::QUEEN_SIDE,
        };
        for (int c = 0; c < 2; ++c) {
            const int back_rank = (colors[c] == chess::Color::WHITE) ? 0 : 7;
            for (int s = 0; s < 2; ++s) {
                if (!rights.has(colors[c], sides[s])) {
                    continue;
                }
                const int file = static_cast<int>(rights.getRookFile(colors[c], sides[s]));
                mask |= square_bb(back_rank * 8 + file);
            }
        }
        return mask;
    }

    chess::Board board_;
    int raw_ep_ = -1;
    int halfmove_clock_ = 0;
    std::vector<int> ep_stack_;
    std::vector<int> clock_stack_;
};

// ---------------------------------------------------------------------------
// Diagnostic FEN
//
// `fen_of` itself moved to cpp/tokens.hpp in C7, beside the parser it inverts,
// because the tablebase prober needs to format a position without including the
// search. Its two jobs here are unchanged: it prints the ep square THIS FILE
// derived rather than chess-library's — when a dump miss is being explained,
// "which ep square did the search think it was at" is the first question — and
// it carries the halfmove clock, so the FEN of a fifty-move draw node is
// re-loadable as the same position that made it a draw (C6's promotion probes).
// ---------------------------------------------------------------------------

inline std::string SearchBoard::diagnostic_fen() const {
    return fen_of(parsed_from_board(), halfmove_clock_,
                  static_cast<int>(board_.fullMoveNumber()));
}

// ---------------------------------------------------------------------------
// C7, Mode 1 — the tablebase root bypass
//
// `playing/uci_wrapper.py::_probe_tablebase`, transcribed. For a root with <= 5
// men the tables play perfectly, so the engine returns the tablebase move and
// never starts a search. The ranking rule itself is in cpp/tablebase.hpp
// (`tablebase_root_score`), where it can be tested on numbers; this is the board
// walk that feeds it.
//
// Returns the packed, NORMALISED move (castling as e1g1, never e1h1), or nullopt
// on any miss — out of range, no tables, or a table the backend cannot answer
// from. Every one of those means "fall through to MCTS", which is the
// reference's behaviour and the only safe one: a bypass that guessed would play
// a losing move with total confidence.
//
// It runs on a SearchBoard rather than a chess::Board so the position handed to
// the backend carries the RAW en-passant square. A Syzygy probe of a position
// with a pawn structure that admits en passant is a different probe; going
// through chess-library's filtered ep state would be the same class of quiet
// error this port has been avoiding since C2.
//
// ONE KNOWN DIVERGENCE, and it is in tie-breaking only. The reference iterates
// `board.legal_moves` — python-chess's generation order — and keeps the first
// move achieving the maximum. This iterates canonical `(from, to, promotion)`
// order and does the same. Where two moves have an identical (outcome,
// -distance) key the two implementations can therefore pick differently. Both
// are tablebase-optimal by construction, so the game-theoretic result is
// unchanged; canonical order is chosen because it is the order this port uses
// everywhere else and is reproducible from the move alone. Recorded in
// DECISIONS.md.
// ---------------------------------------------------------------------------

inline std::optional<std::uint16_t> tablebase_root_move(SearchBoard &board,
                                                        const TablebaseProber &prober) {
    const ParsedFen before = board.parsed();
    // The reference's gate, at the call site rather than inside the helper.
    // Repeated here so the function is safe to call on any position.
    if (!within_tablebase_range(before)) {
        return std::nullopt;
    }

    std::vector<std::uint16_t> packed;
    std::vector<std::uint16_t> raw;
    generate_canonical_moves(board.board(), packed, raw);
    if (packed.empty()) {
        // Checkmate or stalemate at the root: there is no move to return, and
        // the reference's loop would leave best_move as None.
        return std::nullopt;
    }

    std::optional<std::uint16_t> best;
    TablebaseRootScore best_score;

    // Unmakes on ANY exit from the iteration, including a throw out of the
    // backend. A prober is allowed to fail — PythonProber propagates whatever
    // the callable raised — and without this the board would be left one move
    // deep, which for a caller that owns their SearchBoard is silent corruption
    // rather than a failed probe.
    struct Unmake {
        SearchBoard *board;
        chess::Move move;
        bool armed = true;
        ~Unmake() {
            if (armed) {
                board->unmake_move(move);
            }
        }
    };

    for (std::size_t k = 0; k < raw.size(); ++k) {
        const chess::Move move(raw[k]);
        const int from = move.from().index();
        const int to = uci_destination(board.board(), move).index();
        const bool zeroing = guofish::is_zeroing(before, from, to);

        board.make_move(move, zeroing);
        Unmake unmake{&board, move};

        chess::Movelist replies;
        chess::movegen::legalmoves(replies, board.board());
        const ParsedFen after = board.parsed();

        const bool no_replies = replies.size() == 0;
        const bool mate = no_replies && board.in_check();
        const bool drawn_terminal =
            (no_replies && !board.in_check()) || is_insufficient_material(after.placement);

        int wdl_child = 0;
        int dtz_child = 0;
        bool probed = true;
        if (!mate && !drawn_terminal) {
            const std::optional<int> wdl = prober.probe_wdl(after, board.halfmove_clock());
            const std::optional<int> dtz = prober.probe_dtz(after, board.halfmove_clock());
            if (wdl.has_value() && dtz.has_value()) {
                wdl_child = *wdl;
                dtz_child = *dtz;
            } else {
                probed = false;
            }
        }

        unmake.armed = false;
        board.unmake_move(move);

        // The reference wraps its WHOLE loop in one try/except, so a single
        // missing table abandons the bypass rather than ranking the remaining
        // moves against an incomplete picture. That is the conservative reading
        // and it is the right one: a partial ranking can prefer a move only
        // because its sibling could not be scored.
        if (!probed) {
            return std::nullopt;
        }

        const TablebaseRootScore score =
            tablebase_root_score(mate, drawn_terminal, zeroing, wdl_child, dtz_child);
        if (!best.has_value() || score.better_than(best_score)) {
            best = packed[k];
            best_score = score;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// The replay evaluator's dump
//
// Two tables, not one, because the reference genuinely has two answers for one
// position depending on where it is expanded:
//
//   _expand_root  runs its own forward and hands expand() a CUDA tensor, so the
//                 gather and the softmax run on the GPU.
//   interior      BatchedEvaluator does a bulk .cpu() first, so they run on the
//                 CPU.
//
// Those disagree — measured 6 of 37 priors, max delta 1.9e-9 — and it becomes
// visible the moment the root position recurs deeper in the tree, which a
// middlegame reaches in four plies. A dump keyed by nn_key alone could not hold
// both, and picking either one would put the C++ tree a few ulps away from the
// reference's at a node PUCT is comparing. So the root's expansion is looked up
// in its own table. See DECISIONS.md; it is a preserved property of the
// reference, not something this chunk gets to fix.
// ---------------------------------------------------------------------------

class ReplayDump {
public:
    struct Entry {
        const std::uint16_t *moves;   // packed, canonical order
        const float *priors;          // canonical order, aligned with `moves`
        std::uint16_t count;
        double value;                 // the network's ABSOLUTE (White-POV) value
    };

    // CSR in, exactly as golden/gate1_dump.npz stores it. `keys` and `is_root`
    // are parallel; `move_offset` has one more element than they do.
    void load(const std::uint64_t *keys, const std::uint8_t *is_root,
              const std::uint64_t *move_offset, std::size_t entry_count,
              const std::uint16_t *moves, const float *priors, std::size_t move_count,
              const double *values) {
        moves_.assign(moves, moves + move_count);
        priors_.assign(priors, priors + move_count);
        interior_.clear();
        root_.clear();
        interior_.reserve(entry_count);

        for (std::size_t i = 0; i < entry_count; ++i) {
            const std::uint64_t begin = move_offset[i];
            const std::uint64_t end = move_offset[i + 1];
            if (end < begin || end > move_count) {
                throw std::invalid_argument(
                    "guofish::ReplayDump::load: move_offset is not a valid CSR index at entry " +
                    std::to_string(i));
            }
            const std::uint64_t width = end - begin;
            if (width == 0 || width > 0xFFFFu) {
                throw std::invalid_argument(
                    "guofish::ReplayDump::load: entry " + std::to_string(i) + " has " +
                    std::to_string(width) + " moves; expected 1..65535");
            }

            Entry entry{};
            entry.moves = moves_.data() + begin;
            entry.priors = priors_.data() + begin;
            entry.count = static_cast<std::uint16_t>(width);
            entry.value = values[i];

            auto &table = (is_root[i] != 0) ? root_ : interior_;
            if (!table.emplace(keys[i], entry).second) {
                throw std::invalid_argument(
                    "guofish::ReplayDump::load: duplicate key in the dump");
            }
        }
    }

    // Never returns a default. A null result is the caller's cue to throw a
    // ReplayMiss carrying the position that produced the key.
    const Entry *find(NNKey key, bool at_root) const {
        const auto &table = at_root ? root_ : interior_;
        const auto it = table.find(key.value);
        return (it == table.end()) ? nullptr : &it->second;
    }

    std::size_t size() const noexcept { return interior_.size() + root_.size(); }
    std::size_t root_size() const noexcept { return root_.size(); }
    bool empty() const noexcept { return interior_.empty() && root_.empty(); }

private:
    std::vector<std::uint16_t> moves_;
    std::vector<float> priors_;
    std::unordered_map<std::uint64_t, Entry> interior_;
    std::unordered_map<std::uint64_t, Entry> root_;
};

// ---------------------------------------------------------------------------
// C9 — the stand-in evaluator, and why one is needed at all
//
// THIS IS NOT GOLDEN DATA AND NOTHING COMPARED AGAINST PYTHON MAY USE IT.
//
// The Gate 1 dump contains exactly the positions the SERIAL Python reference
// evaluated. That is what makes a dump miss the strongest test in C5: the search
// can only stay inside the dump if it walks the same tree, so a miss proves a
// divergence. C9 breaks the premise. With K in-flight paths, virtual loss steers
// descents onto branches the serial reference never opened — that is the entire
// mechanism of leaf parallelism, and it is working correctly when it happens.
// Measured: at W=1, K=8 the first miss arrives five plies in.
//
// So the acceptance layers split, and the split is the honest one:
//
//   layer 1  W=1, K=1. Explores exactly the serial tree, stays inside the dump,
//            and is compared bit-for-bit against the Python trees. The fallback
//            is OFF and `synthetic_evaluations` is asserted to be 0 — which is
//            what keeps the dump-miss test's teeth.
//   layers 2 and 3  W>=1, K>1. These assert REPRODUCIBILITY and CONSERVATION,
//            not agreement with Python: bit-identical trees across two runs of
//            the same configuration, exact virtual-loss return to zero, exact
//            subtree visit sums, delivered sims equal to the budget. None of
//            those is a claim about what the network said, so an evaluator that
//            is merely deterministic and position-keyed serves them exactly.
//            Real dump entries are still used wherever they exist; this fills
//            only the holes, and the count of holes is reported.
//
// The alternative — regenerating a dump wide enough to cover every position a
// parallel search might reach — is not available: the set depends on the
// scheduling of the run being tested, so producing it would mean running the
// implementation under test to decide what the reference should contain. That
// is circular in exactly the way Global Rule 2 exists to prevent.
//
// The requirements on the stand-in are three, and only the first is obvious:
//
//   1. DETERMINISTIC per position. The same position must always produce the
//      same value and the same priors, or the transposition cache and the tree
//      would disagree with each other and a "reproducibility" test would be
//      measuring the evaluator.
//   2. A well-formed distribution over the move list the search generated.
//   3. SMOOTH ACROSS POSITIONS — and this one is not optional, it is what makes
//      layer 3's root-stability criterion measurable at all.
//
// The first version of this hashed the nn_key into a value, which satisfies (1)
// and (2) and violates (3) completely: two positions one move apart get
// uncorrelated values. A real network does not behave that way, and the
// difference is not cosmetic. Two parallel runs open slightly different
// branches, so under a hashing evaluator they draw different random numbers and
// the root distribution moves for a reason that has nothing to do with
// concurrency. Measured, over 8 runs per cell at W=4/K=8:
//
//   position  stand-in share of expansions   worst pairwise root TV distance
//   0                             2.3%                                 5.9%
//   4                            11.1%                                 9.9%
//   5                            49.4%                                50.2%
//   5                            72.0%                                77.2%
//
// The spread tracks the stand-in's SHARE, not the simulation count — which is
// the signature of the evaluator driving it rather than the engine.
//
// So the stand-in is a chess evaluation instead: material with a small
// positional term, and priors that prefer captures by victim value. It is a weak
// evaluator and it is not meant to be a good one. What it is, is CONTINUOUS in
// the way a network is — two positions one move apart differ by at most one
// captured piece, so their values are close — which is the property that makes
// "the root distribution barely moves between two runs" a statement about the
// search rather than about the noise.
class SyntheticEvaluator {
public:
    // Splitmix64's finaliser, used ONLY as a tiny tie-break below. It is not the
    // evaluation; see the class comment for why that matters.
    static std::uint64_t mix(std::uint64_t x) noexcept {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    // A double in [0, 1) from the top 53 bits: the division is by a power of
    // two, so it introduces no rounding of its own.
    static double unit(std::uint64_t bits) noexcept {
        return static_cast<double>(bits >> 11) * 0x1.0p-53;
    }

    // Centipawn-ish weights, indexed by the token encoding cpp/tokens.hpp uses:
    // 1..6 is White pawn..king, 7..12 is Black pawn..king, 0 is empty. The king
    // is 0 because both sides always have exactly one and it would cancel.
    static double piece_value(std::int32_t token) noexcept {
        switch (token) {
            case 1: case 7: return 1.0;    // pawn
            case 2: case 8: return 3.0;    // knight
            case 3: case 9: return 3.2;    // bishop
            case 4: case 10: return 5.0;   // rook
            case 5: case 11: return 9.0;   // queen
            default: return 0.0;           // king, empty
        }
    }

    // The absolute (White-POV) value in (-1, 1), from material plus a small
    // centralisation term.
    //
    // `tanh` rather than a clamp so the function is smooth everywhere: a clamp
    // would make every position past the cut-off score identically, and a search
    // cannot distinguish two positions it scores the same.
    static double value_of(const ParsedFen &parsed) noexcept {
        double material = 0.0;
        double centre = 0.0;
        for (int square = 0; square < 64; ++square) {
            const std::int32_t token = parsed.placement.square_token[square];
            if (token == kTokenEmpty || token == 0) {
                continue;
            }
            const double value = piece_value(token);
            const bool white = token <= 6;
            material += white ? value : -value;
            // Distance from the centre of the board, in files and ranks. Small
            // enough not to swamp material, large enough that two positions with
            // identical material are not identically scored.
            const int file = square & 7;
            const int rank = square >> 3;
            const double dx = 3.5 - static_cast<double>(file < 4 ? 3 - file : file - 4);
            const double dy = 3.5 - static_cast<double>(rank < 4 ? 3 - rank : rank - 4);
            const double bonus = 0.01 * (dx + dy);
            centre += white ? bonus : -bonus;
        }
        return std::tanh((material + centre) / 4.0);
    }

    // Fill `priors` over `moves`, for the position `parsed`.
    //
    // Captures score by the victim's value, promotions by the promoted piece's,
    // and everything else scores 0 — then a softmax. The key-derived jitter is
    // ~1% of a pawn and exists only to break exact ties: without it every quiet
    // move in a position gets the identical prior, PUCT resolves the tie by child
    // order, and the search fans out in a shape no real policy produces.
    void evaluate(NNKey key, const ParsedFen &parsed, const std::uint16_t *moves,
                  std::size_t count, std::vector<float> &priors) const {
        priors.resize(count);
        double best = -1e30;
        scores_.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            const int to = move_to(moves[i]);
            double score = piece_value(parsed.placement.square_token[to]);
            switch (move_promotion(moves[i])) {
                case Promotion::Queen: score += 8.0; break;
                case Promotion::Rook: score += 4.0; break;
                case Promotion::Bishop: score += 2.2; break;
                case Promotion::Knight: score += 2.0; break;
                case Promotion::None: break;
            }
            score += 0.01 * unit(mix(key.value ^ (static_cast<std::uint64_t>(moves[i]) *
                                                  0x9E3779B97F4A7C15ULL)));
            scores_[i] = score;
            best = score > best ? score : best;
        }
        double total = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            // Shifted by the maximum before exponentiating, which is the
            // standard guard against overflow and is what ATen does too.
            const double weight = std::exp((scores_[i] - best) / 2.0);
            scores_[i] = weight;
            total += weight;
        }
        for (std::size_t i = 0; i < count; ++i) {
            priors[i] = static_cast<float>(scores_[i] / total);
        }
    }

private:
    // Scratch. Only ever touched on the thread that expands, which is the
    // dispatcher in a parallel search and the caller in a serial one, because
    // expansion is single-threaded by construction.
    mutable std::vector<double> scores_;
};

// ---------------------------------------------------------------------------
// Statistics
//
// The C6 counters are not decoration. The Gate 1 corpus is *measured* rather
// than assumed on every axis it claims — C5's manifest records that the quiet
// corpus reached none of this machinery, and C6's records how many times each
// path fired, so "this corpus exercises stalemate" is a number in a file rather
// than a sentence in a commit message.
// ---------------------------------------------------------------------------

struct SearchStats {
    std::int64_t simulations = 0;
    std::int64_t expansions = 0;
    std::int64_t depth_cap_hits = 0;
    std::int64_t max_depth = 0;
    std::int64_t select_steps = 0;

    // C6.
    std::int64_t draw_by_rule_hits = 0;     // fifty-move OR threefold, at the call site
    std::int64_t fifty_move_hits = 0;       // of those, the halfmove-clock branch
    std::int64_t threefold_hits = 0;        // of those, the repetition branch
    std::int64_t checkmates = 0;            // first-visit intrinsic terminals, by reason
    std::int64_t stalemates = 0;
    std::int64_t insufficient_material = 0;
    std::int64_t seventyfive_moves = 0;
    std::int64_t fivefold_repetitions = 0;
    std::int64_t terminal_fast_path_hits = 0;  // a later visit to a marked node
    std::int64_t mate_short_circuits = 0;      // the depth-1 hack fired

    // C7. The cache counters are per-SEARCH; TranspositionCache::stats() is the
    // per-CACHE lifetime total, and the two differ whenever a cache outlives a
    // set_position — which it does, exactly as the reference's does.
    //
    // These exist because the brief points out that tree equivalence cannot see
    // them: with tablebases off the cache is result-invariant, so a cache with a
    // 0% hit rate produces a bit-identical tree and passes the gate. The hit
    // rate is therefore asserted separately, and it is asserted on THIS number.
    std::int64_t cache_hits = 0;
    std::int64_t cache_misses = 0;
    std::int64_t cache_inserts = 0;

    // C7, mode 2. `tablebase_probes` counts positions that passed the
    // piece-count gate and reached the backend; `tablebase_overrides` counts
    // those the backend actually answered. The gap is the miss rate, which for a
    // 5-piece table set on a 6-piece leaf is 100% — a distinction worth having
    // when a run reports no overrides.
    std::int64_t tablebase_probes = 0;
    std::int64_t tablebase_overrides = 0;

    // C9. Leaves the replay dump did not contain, answered by the stand-in
    // evaluator instead. MUST BE ZERO for anything compared against Python —
    // see SyntheticEvaluator for why it exists and what it is not. The counter
    // is the whole safeguard: without it, "the fallback was off" is an assertion
    // about a flag, and with it, it is an assertion about what happened.
    std::int64_t synthetic_evaluations = 0;
};

// C8. Counters for the tree-reuse seam. Not part of SearchStats because they do
// not belong to a search: they span every `apply_move` since the last
// `set_position`, which is the span of a game.
struct ReuseStats {
    std::int64_t applies = 0;          // apply_move() calls that reused a subtree
    std::int64_t discards = 0;         // ...that could not, and rebuilt from scratch
    std::int64_t nodes_copied = 0;     // total nodes moved by every compaction
    std::int64_t nodes_dropped = 0;    // total nodes the compactions left behind
    std::int64_t verifications = 0;    // structural diffs actually run
    std::int64_t decays = 0;           // promotions that applied a ponder decay
    std::int64_t terminal_promotions = 0;   // promoted roots that arrived marked
    std::int64_t terminal_marks_cleared = 0;  // ...and were then expanded anyway
    // The largest single compaction, in nodes. This is the figure the memory
    // budget is really about: a compaction holds the source subtree and its copy
    // at the same time.
    std::int64_t largest_copy = 0;
};

// A serialized node, in canonical DFS preorder.
struct TreeRecord {
    std::uint16_t depth;
    std::uint16_t move;        // packed, normalised; 0 at the root
    std::int32_t visits;
    double value_sum;
    float prior;
    std::uint16_t children;
    // C6. `terminal` is the arena's bit, NOT a lifecycle value — a terminal node
    // is Unexpanded, so `terminal == 1 && children > 0` is unrepresentable and
    // tests/test_c6_terminal_invariants.py asserts it over the whole corpus.
    std::uint8_t terminal;
    float terminal_value;
};

// One terminal node, as the invariants test and the promotion probes read it.
struct TerminalNode {
    std::string path;     // normalised UCI moves from the root
    std::string fen;      // re-loadable: our raw ep square, our halfmove clock
    float value;
    std::int32_t visits;
    std::uint16_t children;
    std::uint8_t expanded;   // lifecycle == Expanded; must be 0 for every terminal
    std::uint16_t depth;
};

// ---------------------------------------------------------------------------
// C9 — the parallel search's configuration
//
// W AND K ARE NOT INHERITED FROM PYTHON, AND THE REASON IS ARITHMETIC.
// The reference ran 32 workers because it was GIL-bound: a worker spent almost
// all of its life blocked on `request.event.wait()`, so oversubscribing 16
// hardware threads 2:1 cost nothing. C++ search threads actually run. Thirty-two
// of them on this machine would mean real context switching with VIRTUAL LOSS
// HELD ACROSS PREEMPTION by descheduled threads, which inflates in-flight VL
// exposure and flattens the root visit distribution — the exact distortion the
// chunk is trying to bound.
//
// The defaults below come from the measured curves, both of them:
//
//   * the GPU knee (tools/bench_c9_knee.py, BENCH.md C9): per-batch cost is FLAT
//     at ~3.8-4.0 ms from batch 8 to batch 64 and then rises linearly, because
//     below batch 128 the forward pass is bound by HOST-side ATen dispatch
//     (~17-34 us per op on this machine, ~110 ops per forward), not by the GPU.
//     So evals/s rises almost proportionally with batch size up to 64, reaches
//     ~18k at 128 and stops. Small batches are actively expensive here — batch
//     32 buys only ~8.4k evals/s — which is the opposite of what the "minimise
//     outstanding" reading of the older curve suggested.
//   * the CPU descent rate (C5, BENCH.md): 178k sims/s single-threaded. One
//     thread can nearly saturate the GPU by itself, so W is small.
//
// `max_batch` therefore defaults to the knee (128) and `workers * in_flight`
// defaults to 32 — the brief's default, which the W x K grid in BENCH.md is the
// evidence for or against. Throughput is not the only axis: a configuration 10%
// faster and visibly flatter at the root is the wrong trade, and the grid
// reports both.
// ---------------------------------------------------------------------------

struct ParallelConfig {
    int workers = 4;                 // W
    int in_flight = 8;               // K, per worker
    std::size_t max_batch = 128;     // the dispatcher never drains more at once
    AffinityPolicy affinity = AffinityPolicy::PCorePhysical;
    // Record every batch size and every outstanding-leaf count at drain time.
    // On by default: scope 6.2 asks for the outstanding-leaf histogram and the
    // saturated-vs-starved regime label as first-class instrumentation, and a
    // few thousand int64s per search is not a cost worth a flag being off.
    bool collect_histograms = true;

    int max_outstanding() const noexcept { return workers * in_flight; }
};

// C9 counters. Deliberately NOT part of SearchStats: these describe the
// machinery that ran the simulations, not the simulations, and every one of
// them is identically zero for a serial search.
struct ParallelStats {
    std::int64_t requested = 0;          // simulations asked for
    std::int64_t delivered = 0;          // simulations backed up into the root
    std::int64_t queued_leaves = 0;      // leaves that went through the queue
    std::int64_t worker_terminals = 0;   // leaves resolved on a worker, never queued
    std::int64_t select_collisions = 0;  // descents discarded on a lost PENDING CAS
    std::int64_t batches = 0;
    std::int64_t largest_batch = 0;
    std::int64_t worker_waits = 0;       // times a worker had to wait for a drain
    std::int64_t hook_wait_ns = 0;       // total time the batch hook spent waiting
    std::int64_t wall_ns = 0;
    int workers = 0;
    int in_flight = 0;
    std::int64_t max_outstanding = 0;
    // Which logical processors the workers were pinned to, in worker order.
    // Empty means no pinning happened — see Topology::source for why.
    std::vector<int> pinned_cpus;
    std::string affinity = "none";
    // scope 6.2's histograms.
    std::vector<std::int64_t> batch_sizes;
    std::vector<std::int64_t> outstanding_at_drain;
    // The batch hook's wait, per batch, in nanoseconds. Empty unless a hook is
    // installed; under the GIL probe this is the acquisition-latency histogram
    // the C9 brief asks for.
    std::vector<std::int64_t> hook_wait_ns_samples;
};

// C10 — the live evaluator's contention instrumentation, and why it is a
// HISTOGRAM and not a mean.
//
// A mean acquire wait hides the tail behind the many fast acquisitions, and the
// tail is the entire cost: C0b measured a contended dispatcher at a median
// 15.25 ms under the default switch interval, which is a hard ceiling of ~60
// batches per second, while the same run's cheap acquisitions were sub-microsecond.
// Averaging those together produces a number that is neither. So every batch's
// wait is kept, and the C10 trigger — implement C++-side `info` emission if p99
// exceeds 200 µs, or if the total exceeds 1% of the move's wall time — is
// evaluated against the distribution.
//
// `total_ns` is deliberately NOT the callback's duration. torch and numpy
// re-acquire the GIL inside a call, so a handoff wait lands in the middle of
// `call_ns` and inflates it with something that is not contention; only the
// wait measured from OUTSIDE the acquire scope is clean (scope §2.1).
// `call_ns` is kept anyway, because the ratio between the two is what says
// whether a slow batch was contention or was just the model.
struct EvalStats {
    std::int64_t batches = 0;      // boundary crossings
    std::int64_t rows = 0;         // leaves actually sent to the network
    std::int64_t cache_skipped = 0;  // leaves that never needed a row
    std::int64_t acquire_wait_ns = 0;
    std::int64_t call_ns = 0;
    std::int64_t max_acquire_wait_ns = 0;
    // One entry per boundary crossing. Always collected: a batch is at least a
    // few hundred microseconds of model, so an 8-byte sample per batch cannot
    // be the thing that costs.
    std::vector<std::int64_t> acquire_wait_ns_samples;
    std::vector<std::int64_t> call_ns_samples;

    void reset() {
        batches = 0;
        rows = 0;
        cache_skipped = 0;
        acquire_wait_ns = 0;
        call_ns = 0;
        max_acquire_wait_ns = 0;
        acquire_wait_ns_samples.clear();
        call_ns_samples.clear();
    }
};

// The exact conservation invariants acceptance layer 3 is made of.
//
// These are ASSERTIONS ABOUT INTEGERS, with no epsilon anywhere, and that is
// only possible because C9 moves value_sum to the Q32 `atomic<int64>`
// accumulator and virtual loss is an integer COUNT. Under a floating-point
// accumulator "the subtree sums match" would be a statement about rounding; here
// it is a statement about arithmetic.
//
// The visit invariant is exact and worth stating precisely: for every EXPANDED
// node, `visits == 1 + sum(visits of children)`. The 1 is the simulation that
// expanded it. Nothing else can stop at an expanded node — the descent loop only
// exits at a node that is unexpanded, terminal or depth-capped, and a
// depth-capped node is at MAX_TREE_DEPTH and is therefore never expanded.
struct TreeAudit {
    std::int64_t nodes = 0;
    std::int64_t expanded_nodes = 0;
    std::int64_t visited_nodes = 0;
    // Virtual loss, summed by a FLAT SCAN of the arena rather than a traversal:
    // a loss stranded on a node that is no longer reachable is precisely the
    // kind this is looking for, and a traversal would not see it.
    std::int64_t vloss_total = 0;
    std::int64_t vloss_nonzero_nodes = 0;
    std::int64_t conservation_failures = 0;
    // The first failure, in arena order, so a report can name a node rather
    // than a count.
    std::int64_t first_bad_node = -1;
    std::int64_t first_bad_expected = 0;
    std::int64_t first_bad_actual = 0;
    std::int64_t root_visits = 0;
    std::int64_t root_children_visits = 0;  // sum over the root's children
};

// ---------------------------------------------------------------------------
// The search
// ---------------------------------------------------------------------------

template <class Accumulator>
class ReplaySearch {
public:
    explicit ReplaySearch(const SearchConfig &config)
        : config_(config),
          arena_(config.arena_capacity),
          parent_(config.arena_capacity, kNoNode),
          raw_move_(config.arena_capacity, 0) {
        if (config.c_base <= 0.0) {
            throw std::invalid_argument("guofish::ReplaySearch: c_base must be > 0 (log domain)");
        }
        if (config.max_tree_depth < 1) {
            throw std::invalid_argument("guofish::ReplaySearch: max_tree_depth must be >= 1");
        }
        // C11b. It DIVIDES the logits, so zero and negatives are not
        // "unusual settings" — they are a division by zero and an inverted
        // policy. The reference validates it the same way and with the same
        // parenthetical (SearchParams.__post_init__).
        if (!(config.policy_temperature > 0.0f)) {
            throw std::invalid_argument(
                "guofish::ReplaySearch: policy_temperature must be > 0 (it divides the "
                "logits), got " + std::to_string(config.policy_temperature));
        }
#ifdef GUOFISH_VALUE_SUM_DOUBLE
        // C11b requirement 5: THE EQUIVALENCE BUILD ASSERTS T = 1.0, it does
        // not document it.
        //
        // This is the build Gate 1 runs, and Gate 1 replays priors from a
        // reference dump recorded at POLICY_TEMPERATURE = 1.0. Those priors are
        // already post-softmax, so a temperature set here would not reach them
        // and would not change a single node of the tree — the run would pass
        // while claiming to have been sharpened. That is the silent mismatch,
        // and a build whose entire purpose is bit-exactness against Python is
        // the worst possible place to leave one available.
        if (config.policy_temperature != 1.0f) {
            throw std::invalid_argument(
                "guofish::ReplaySearch: the Gate 1 equivalence build "
                "(GUOFISH_VALUE_SUM=double) accepts only policy_temperature = 1.0, got " +
                std::to_string(config.policy_temperature) +
                ".\n  Gate 1 replays priors a T = 1.0 reference already softmaxed; a "
                "temperature set here would be silently ignored rather than applied. "
                "Build with GUOFISH_VALUE_SUM=q32 to search at a temperature.");
        }
#endif
        if (config.cache_slots != 0) {
            cache_.emplace(config.cache_slots, config.cache_shards);
        }
        // The serial descent drives the SEARCH's board, not a copy: apply_move,
        // terminal_nodes and root_fen all read it after search() returns, and a
        // descent that unwound a copy would leave the real one behind.
        serial_.board = &board_;
        serial_.stats = &stats_;
    }

    ReplayDump &dump() noexcept { return dump_; }
    const ReplayDump &dump() const noexcept { return dump_; }

    // C9. Answer a dump miss with the deterministic stand-in evaluator instead
    // of throwing. OFF BY DEFAULT, and nothing compared against Python may turn
    // it on. See SyntheticEvaluator; `SearchStats::synthetic_evaluations` counts
    // every leaf it answered, so "the fallback was off" is checkable after the
    // fact rather than only before it.
    void set_synthetic_fallback(bool on) noexcept { synthetic_fallback_ = on; }
    bool synthetic_fallback() const noexcept { return synthetic_fallback_; }
    const SearchConfig &config() const noexcept { return config_; }
    const SearchStats &stats() const noexcept { return stats_; }
    const NodeArena<Accumulator> &arena() const noexcept { return arena_; }
    std::uint32_t root() const noexcept { return root_; }

    // --- C7: the cache -----------------------------------------------------

    bool has_cache() const noexcept { return cache_.has_value(); }

    CacheStats cache_stats() const {
        return cache_.has_value() ? cache_->stats() : CacheStats{};
    }

    std::size_t cache_size() const { return cache_.has_value() ? cache_->size() : 0u; }
    std::size_t cache_capacity() const { return cache_.has_value() ? cache_->capacity() : 0u; }
    std::size_t cache_shard_count() const {
        return cache_.has_value() ? cache_->shard_count() : 0u;
    }

    // The cache is NOT cleared by set_position, matching the reference — one
    // `TranspositionCache` lives on the `ParallelMCTS` instance and survives
    // every `search()` call, which is what makes it worth having across moves of
    // a game. A test that wants a hit rate attributable to one position calls
    // this first.
    void clear_cache() {
        if (cache_.has_value()) {
            cache_->clear();
        }
    }

    // Read one entry back out, by key. The acceptance criteria ask for the entry
    // CONTENTS to be checked directly — priors and move list round-tripping —
    // rather than for correctness to be inferred from the tree, so the contents
    // have to be reachable.
    bool cache_probe(NNKey key, CachedEval &out) {
        if (!cache_.has_value()) {
            return false;
        }
        return cache_->probe(key, out);
    }

    // --- C7: tablebases ----------------------------------------------------

    // Borrowed, not owned: the caller keeps the prober alive. That is the
    // reference's arrangement too (`self.tablebase` is a handle the UCI layer
    // opened and hands to both modes), and it is what lets one set of open
    // tables serve several searches.
    //
    // `nullptr` is tablebases OFF, which is the default and what the build ships
    // with — see cpp/tablebase.hpp on why the Syzygy decoder is a backend and
    // not part of this chunk.
    void set_tablebase(const TablebaseProber *prober) noexcept { tablebase_ = prober; }
    const TablebaseProber *tablebase() const noexcept { return tablebase_; }

    // The move the depth-1 hack seized on, packed; kNoMove when it did not fire.
    std::uint16_t mating_move() const noexcept {
        return mating_move_.load(std::memory_order_relaxed);
    }

    double c_puct_of(double parent_visits) const noexcept {
        double c = guofish::c_puct(parent_visits, config_.c_init, config_.c_base);
        // Guarded rather than multiplied unconditionally. `1.0 * x == x` holds
        // exactly in IEEE-754 for every finite x, so an unguarded multiply would
        // also be safe — but the reference has no factor at all, and spelling
        // "the default path does not touch the value" as a branch makes that
        // property visible to a reader instead of resting on a footnote. The
        // reference itself guards policy_temperature the same way.
        if (config_.c_factor != 1.0) {
            c *= config_.c_factor;
        }
        return c;
    }

    // Reset the tree and load a root position. The arena is recycled rather
    // than reallocated; try_allocate() clears each block as it hands it out.
    //
    // `history` is the pre-root game history, as the FENs of the positions
    // `ParallelMCTS.search` walks back over when it calls
    // `build_repetition_history(board)` — that is, the positions BEFORE each of
    // the last `min(halfmove_clock, len(move_stack))` moves, most recent first.
    // The root's own position is counted here rather than being passed in,
    // exactly as the reference seeds its counter with `counter[root_key] = 1`.
    //
    // Passing FENs rather than keys is deliberate: a caller that computed
    // rep_keys itself would be a second implementation of the rule C3 exists to
    // have exactly one of. These go through the same `parse_fen` -> `rep_key`
    // path as everything else, so the raw-ep discipline is inherited rather than
    // re-stated.
    void set_position(std::string_view fen, const std::vector<std::string> &history) {
        arena_.reset();
        board_.set_fen(fen);
        root_ = arena_.allocate(1);
        parent_[root_] = kNoNode;
        raw_move_[root_] = 0;
        root_expanded_ = false;
        stats_ = SearchStats{};
        reuse_ = ReuseStats{};
        mating_move_ = kNoMove;

        root_parsed_ = board_.parsed();
        root_rep_key_ = rep_key(root_parsed_).value;
        root_occupied_ = root_parsed_.placement.occupied();

        // C8. The history is kept, not merely consumed. `apply_move` has to
        // rebuild `rep_history_` for the NEW root — the same position counted
        // once as the root plus its predecessors inside the halfmove-clock
        // horizon — and it can only do that if the predecessors are still around
        // as keys. Most recent first, which is the order the caller supplies and
        // the order `build_repetition_history` walks.
        history_keys_.clear();
        history_keys_.reserve(history.size());
        for (const std::string &position : history) {
            history_keys_.push_back(rep_key(parse_fen(position)).value);
        }

        // Counted verbatim rather than through apply_move's windowing rule, and
        // that is deliberate: this is C5/C6/C7's certified behaviour and the two
        // agree for every conforming caller anyway. The contract is that
        // `history` already IS the window — the FENs of the last
        // min(halfmove_clock, plies) positions — so windowing here would be a
        // second application of a filter the caller has applied.
        rep_history_.clear();
        rep_history_[root_rep_key_] = 1;
        for (const std::uint64_t key : history_keys_) {
            rep_history_[key] += 1;
        }
    }

    void set_position(std::string_view fen) { set_position(fen, {}); }

    // Run until the root has `num_simulations` visits, matching
    // ParallelMCTS.search's `target_new_sims = num_simulations - existing`.
    // _expand_root seeds the root with one visit, so a fresh N-simulation search
    // runs N-1 simulations and the root ends at exactly N.
    //
    // ...unless the depth-1 mate hack fires, in which case it ends wherever the
    // hack fired. `mating_move()` says whether that happened; the reference does
    // the same by setting its completion_event.
    SearchStats search(int num_simulations) {
        require_position();
        require_evaluation_source("search");
        eval_stats_.reset();
        // The reference builds a fresh `stats` defaultdict per search() call, so
        // a mating move found by a previous call does not stop this one before
        // it starts.
        mating_move_ = kNoMove;
        if (!root_expanded_) {
            expand_root();
        }
        const int existing = arena_.visit_count(root_);
        for (int i = existing; i < num_simulations; ++i) {
            // MCTSWorker._work_loop checks completion_event at the TOP of the
            // loop, so the simulation that set it is counted and the next one
            // never starts.
            if (mating_move_.load(std::memory_order_relaxed) != kNoMove) {
                break;
            }
            run_simulation(serial_);
        }
        return stats_;
    }

    // ---- C9: W workers, K in flight each ----------------------------------

    // Run `num_simulations` the way production will: W search threads that
    // never block on evaluation, an MPSC leaf queue, and one dispatcher that
    // expands what they submit.
    //
    // THE THREE THINGS THIS CHANGES ABOUT A SIMULATION, and nothing else:
    //
    //  1. A worker that reaches an unexpanded leaf does not evaluate it. It
    //     CASes the node from Unexpanded to PENDING, hands the leaf to the
    //     queue *with its virtual loss still applied*, and starts the next
    //     simulation immediately. Scope 2.2: search threads never block on
    //     evaluation. The virtual loss is what keeps the next descent off the
    //     same branch, and it is repaid by whoever backs the leaf up.
    //  2. A worker that LOSES that CAS has descended into a path another
    //     simulation already owns. It unwinds its own virtual loss, discards
    //     the simulation, and retries — counted as `select_collisions`, which
    //     scope 2.2 names as the VL-too-low signal.
    //  3. Expansion happens on the dispatcher and only there, so the arena's
    //     bump allocator has one writer and `set_children` has one caller per
    //     node. Everything else — selection, the draw rules, terminal
    //     classification, backup — runs on the workers, lock-free, against the
    //     same atomics C4 built for it.
    //
    // WHEN THE DISPATCHER DRAINS, which is the one design decision here that a
    // reader should not have to reverse-engineer.
    //
    // The brief requires the drain to be `min(available, max_batch)` with no
    // minimum-batch floor and no straggler timeout — both of the Python
    // collector's measured pathologies — and it separately requires W=1, K=8 to
    // produce BIT-IDENTICAL trees across two runs. Those two requirements
    // interact: a dispatcher that drains the instant a leaf appears makes the
    // handoff point a function of thread scheduling, and at W=1 that changes
    // whether simulation N+1 sees an expanded node or a virtual loss. The tree
    // then differs run to run for a reason that is not a bug, and the only clean
    // test of the in-flight machinery is lost.
    //
    // So the drain TRIGGER is "the queue is non-empty and no search thread can
    // currently make progress" — every worker is either throttled, waiting after
    // a collision, or finished. This is not a minimum-batch floor: the drain
    // takes whatever is there, which at W=1/K=8/VL=0 is a single leaf, because
    // at zero virtual loss the second descent re-selects the first descent's
    // leaf, collides, and waits. And it is not a timeout: there is no clock
    // anywhere in the dispatcher. Batch size is set by the outstanding-leaf
    // count, which is what scope 2.2 asks for. See DECISIONS.md, C9.
    //
    // `pc.workers == 1 && pc.in_flight == 1` is the acceptance-layer-1 case and
    // still runs through all of this machinery — the queue, the dispatcher, the
    // PENDING CAS — rather than falling back to `search()`. A fallback would
    // make layer 1 a test of code layer 3 does not use.
    SearchStats search_parallel(int num_simulations, const ParallelConfig &pc) {
        require_position();
        require_evaluation_source("search_parallel");
        eval_stats_.reset();
        // Checked BEFORE any thread starts, so an over-wide batch is a named
        // argument error on the caller's thread rather than a logic_error raised
        // on the dispatcher and re-thrown from a join.
        if (evaluator_ != nullptr && pc.max_batch > evaluator_->max_batch()) {
            throw std::invalid_argument(
                "guofish::ReplaySearch::search_parallel: max_batch is " +
                std::to_string(pc.max_batch) + " but the evaluator's buffers hold " +
                std::to_string(evaluator_->max_batch()) +
                " rows. Allocate the evaluator with at least max_batch rows.");
        }
        if (pc.workers < 1) {
            throw std::invalid_argument(
                "guofish::ReplaySearch::search_parallel: workers must be >= 1");
        }
        if (pc.in_flight < 1) {
            throw std::invalid_argument(
                "guofish::ReplaySearch::search_parallel: in_flight must be >= 1");
        }
        if (pc.max_batch < 1) {
            throw std::invalid_argument(
                "guofish::ReplaySearch::search_parallel: max_batch must be >= 1");
        }

        mating_move_.store(kNoMove, std::memory_order_relaxed);
        if (!root_expanded_) {
            expand_root();
        }

        par_ = ParallelStats{};
        par_.workers = pc.workers;
        par_.in_flight = pc.in_flight;
        par_.max_outstanding = pc.max_outstanding();
        par_.affinity = affinity_policy_name(pc.affinity);

        const int existing = arena_.visit_count(root_);
        target_ = num_simulations - existing;
        par_.requested = target_ > 0 ? target_ : 0;
        if (target_ <= 0) {
            return stats_;
        }

        run_workers(pc);
        return stats_;
    }

    const ParallelStats &parallel_stats() const noexcept { return par_; }

    // Install a per-batch hook on the dispatcher, or clear it with nullptr.
    // Borrowed, not owned. See BatchHook in cpp/parallel.hpp: the only
    // implementation is the GIL probe, which exists to test scope 2.1's
    // prediction on the real C9 thread topology.
    void set_batch_hook(BatchHook *hook) noexcept { hook_ = hook; }

    // C10. Install the live evaluator, or clear it with nullptr. Borrowed, not
    // owned — it holds the three buffers Python has numpy views onto, and their
    // lifetime is the caller's problem for the same reason the tablebase
    // prober's is.
    //
    // Installing one takes the replay dump OUT of the evaluation path entirely,
    // root included. There is deliberately no mode in which a search consults
    // both: the dump's whole value is that a miss is a hard failure proving a
    // divergence, and a live fallback would turn every such proof into a silent
    // "the network answered instead".
    void set_evaluator(BatchEvaluator *evaluator) noexcept { evaluator_ = evaluator; }
    const BatchEvaluator *evaluator() const noexcept { return evaluator_; }
    const EvalStats &eval_stats() const noexcept { return eval_stats_; }

    // The topology this build can see, whatever it is. Reported rather than
    // assumed so a benchmark table can say what it ran on and a machine that
    // does not report a hybrid split says so instead of pretending.
    static const Topology &topology() {
        static const Topology topo = detect_topology();
        return topo;
    }

    // The exact conservation invariants. See TreeAudit.
    //
    // O(nodes) over the arena, so this is a test and instrumentation entry
    // point, not something a search calls. It is exposed rather than asserted
    // internally because acceptance layer 3 has to be able to REPORT the
    // failure, and an assert that fires inside a worker thread reports a
    // process abort.
    TreeAudit audit() const {
        require_position();
        TreeAudit out;
        out.nodes = static_cast<std::int64_t>(arena_.size());
        for (std::size_t i = 0; i < arena_.size(); ++i) {
            const auto node = static_cast<std::uint32_t>(i);
            const std::int32_t vloss = arena_.vloss_count(node);
            out.vloss_total += vloss;
            if (vloss != 0) {
                ++out.vloss_nonzero_nodes;
            }
            if (arena_.visit_count(node) > 0) {
                ++out.visited_nodes;
            }
            const std::uint16_t count = arena_.children_count(node);
            if (arena_.lifecycle(node) != NodeState::Expanded || count == 0) {
                continue;
            }
            ++out.expanded_nodes;
            std::int64_t sum = 0;
            const std::uint32_t first = arena_.children_offset(node);
            for (std::uint16_t k = 0; k < count; ++k) {
                sum += arena_.visit_count(first + static_cast<std::uint32_t>(k));
            }
            if (node == root_) {
                out.root_children_visits = sum;
                out.root_visits = arena_.visit_count(node);
            }
            const std::int64_t expected = sum + 1;
            const std::int64_t actual = arena_.visit_count(node);
            if (expected != actual) {
                ++out.conservation_failures;
                if (out.first_bad_node < 0) {
                    out.first_bad_node = static_cast<std::int64_t>(node);
                    out.first_bad_expected = expected;
                    out.first_bad_actual = actual;
                }
            }
        }
        return out;
    }

    // ---- C8: the seam between moves ---------------------------------------

    // Play `packed` and keep the subtree under it.
    //
    // `ParallelMCTS.apply_move` transcribed — with the one structural difference
    // this chunk exists for. The reference promotes a child by reassigning a
    // pointer and detaching the parent, and Python's garbage collector reclaims
    // the discarded branches whenever it gets round to it. There are no pointers
    // here and there is no collector: the tree is a bump-allocated index space,
    // and the surviving subtree is scattered through it among the ~70% of nodes
    // that are about to become garbage. So the promotion is a COMPACTING COPY
    // into the standby arena, followed by a swap (scope §2.3).
    //
    // What that buys, beyond not leaking:
    //
    //   * the arena is contiguous again, so the next search's sibling scans are
    //     sequential rather than strided across the holes the dead branches left;
    //   * the bump pointer goes back to (nodes that survived), which is what
    //     keeps a game-long tree inside the 2-3M budget instead of growing by a
    //     search's worth of nodes every ply;
    //   * every index in the new arena is derived, in one pass, from one
    //     traversal — so a fixup bug is a *structural* difference against the
    //     source tree, and `verify_compaction` is a diff rather than a guess.
    //
    // Returns true if a subtree was reused. False means the tree was thrown
    // away and rebuilt around the new position, which is the reference's
    // "nothing to reuse" branch (`move not in self.root.children`) — reachable
    // when the root is an unexpanded promoted leaf, or terminal, or when the
    // caller plays a move this search never generated.
    //
    // `from_ponder` is the caller's assertion that the subtree being promoted
    // was built by a ponder search rather than by a search of this position.
    // Only then is `config_.ponder_decay` applied. The search cannot work this
    // out for itself; pondering is a layer above.
    bool apply_move(std::uint16_t packed, bool from_ponder = false) {
        require_position();
        if (packed == kNoMove) {
            throw std::invalid_argument(
                "guofish::ReplaySearch::apply_move: kNoMove is not a move");
        }
        const double decay = from_ponder ? config_.ponder_decay : 1.0;
        if (!(decay > 0.0 && decay <= 1.0)) {
            throw std::invalid_argument(
                "guofish::ReplaySearch::apply_move: ponder_decay must be in (0, 1], got " +
                std::to_string(decay));
        }

        const std::uint32_t child = find_root_child(packed);

        // The library move for the promotion, and the two classifiers, read
        // BEFORE anything moves. `is_zeroing` is evaluated on the position the
        // move is played from — the current root — which is python-chess's own
        // evaluation point and the reason `make_move` takes it as an argument.
        std::uint16_t raw = 0;
        if (child != kNoNode) {
            raw = raw_move_[child];
        } else {
            // No subtree to keep. The move still has to be played, and it still
            // has to be a legal one — a caller that hands us an illegal move has
            // desynchronised from the game, and continuing would search a
            // position nobody is in.
            std::vector<std::uint16_t> packed_moves;
            std::vector<std::uint16_t> raw_moves;
            generate_canonical(packed_moves, raw_moves);
            std::size_t k = 0;
            for (; k < packed_moves.size(); ++k) {
                if (packed_moves[k] == packed) {
                    break;
                }
            }
            if (k == packed_moves.size()) {
                throw std::invalid_argument(
                    "guofish::ReplaySearch::apply_move: " + uci_of(packed) +
                    " is not legal at " + board_.diagnostic_fen());
            }
            raw = raw_moves[k];
        }
        const chess::Move move(raw);
        const bool zeroing = guofish::is_zeroing(root_parsed_, move.from().index(),
                                                 uci_destination(board_.board(), move).index());

        // === advance the position =========================================
        //
        // The old root joins the history BEFORE the board moves, because from
        // the new root's point of view the position we are leaving is the most
        // recent previous position — the first thing
        // `build_repetition_history`'s walk-back would meet.
        history_keys_.insert(history_keys_.begin(), root_rep_key_);
        board_.make_move(move, zeroing);
        board_.commit();

        root_parsed_ = board_.parsed();
        root_rep_key_ = rep_key(root_parsed_).value;
        root_occupied_ = root_parsed_.placement.occupied();
        rebuild_rep_history();
        mating_move_ = kNoMove;

        // === move the tree ================================================
        if (child == kNoNode) {
            ++reuse_.discards;
            arena_.reset();
            root_ = arena_.allocate(1);
            parent_[root_] = kNoNode;
            raw_move_[root_] = 0;
            root_expanded_ = false;
            return false;
        }

        compact_and_promote(child, decay);
        ++reuse_.applies;
        return true;
    }

    // The reuse counters since the last set_position. See ReuseStats.
    const ReuseStats &reuse_stats() const noexcept { return reuse_; }

    // `build_repetition_history(board)`'s answer for the CURRENT root, read out.
    //
    // Exposed because C8's second implementation requirement — that the
    // path-dependent draw counts partition differently across an applied move
    // but sum identically — is otherwise only observable through its
    // consequences. It would show up as a threefold that appears or disappears
    // somewhere in a tree, which is a bit-exactness failure a hundred nodes away
    // from its cause. This is the cause, in one map.
    //
    // Read-only. The map is rebuilt by set_position and apply_move and is
    // treated as immutable by every simulation, which is what will let C9 share
    // it across threads without a lock.
    const std::unordered_map<std::uint64_t, int> &rep_history() const noexcept {
        return rep_history_;
    }

    // The position the tree is rooted at, in the FEN dialect this file owns —
    // OUR raw en-passant square, OUR halfmove clock. After `apply_move` this is
    // the cheapest possible check that the engine and the game agree, and it is
    // a check worth having: every rule C6 implements reads one of those two
    // fields, and a seam that advanced the board while losing the clock would
    // produce a tree that is internally consistent and about the wrong position.
    std::string root_fen() const {
        require_position();
        return board_.diagnostic_fen();
    }

    // The most nodes either arena has ever held. The peak of BOTH is the honest
    // figure: during a compaction the source subtree and its copy are alive at
    // the same time, so the moment of greatest occupancy is the moment the two
    // arenas overlap, and reporting only the active one would miss it by
    // construction. See NodeArena::high_water.
    std::size_t arena_high_water() const noexcept {
        const std::size_t active = arena_.high_water();
        const std::size_t standby = standby_ ? standby_->high_water() : 0u;
        return active > standby ? active : standby;
    }

    // Bytes of node payload the ping-pong pair has reserved. The standby arena
    // does not exist until the first apply_move, so this answers "what has been
    // committed", not "what could be".
    std::size_t arena_bytes_reserved() const noexcept {
        const std::size_t per_node = NodeArena<Accumulator>::bytes_per_node() +
                                     sizeof(std::uint32_t) + sizeof(std::uint16_t);
        return per_node * config_.arena_capacity * (standby_ ? 2u : 1u);
    }

    bool has_standby_arena() const noexcept { return standby_ != nullptr; }

#if defined(GUOFISH_DEBUG_VL)
    // THE DEFENSIVE FULL-TREE VIRTUAL-LOSS WALK, AND IT EXISTS ONLY HERE.
    //
    // The reference calls `_reset_virtual_loss(root)` at the top of every search
    // and every get_policy: a recursive walk of the whole tree writing
    // `vloss_count = 0` on every node, defensively, because nothing guaranteed
    // that a previous search had repaid what it applied. Scope §2.3 prices it at
    // 3.4 ms at 2k sims, 36 ms at 8k and 937 ms over a game — a full-tree write
    // pass per move, buying a property that should be a property.
    //
    // C8's brief is explicit: do NOT implement a production equivalent. Here
    // repayment is scope-guaranteed by RAII — `run_simulation`'s Unwind
    // destructor repays every applied loss on every exit, including an exception
    // mid-descent — so there is nothing to reset, and a walk that reset anything
    // would be hiding a bug rather than fixing one.
    //
    // So this READS rather than writes, and it is compiled only when
    // GUOFISH_DEBUG_VL is defined (CMake turns it on for Debug builds, which is
    // where the sanitized test run lives). It returns the sum, so a test can
    // assert the invariant the reference merely hoped for: a quiescent tree
    // holds exactly zero in-flight losses, at any virtual-loss magnitude,
    // because the counts are integers.
    std::int64_t debug_total_vloss() const {
        require_position();
        std::int64_t total = 0;
        // Flat scan, not a traversal: after a compaction the arena holds exactly
        // the reachable tree in [0, size()), and before one it holds the tree
        // plus nothing else. A stranded loss on an UNREACHABLE node is precisely
        // the kind this is looking for, and a traversal would not see it.
        for (std::size_t i = 0; i < arena_.size(); ++i) {
            total += arena_.vloss_count(static_cast<std::uint32_t>(i));
        }
        return total;
    }
#endif

    // `ParallelMCTS.search`'s return value: the mating move if the hack fired,
    // otherwise the most-visited root child. kNoMove if the root has no
    // children, which after a successful expansion cannot happen.
    //
    // `max(root.children.items(), key=...)` returns the FIRST maximal element in
    // dict order, and dict order is insertion order, which the Gate 1 patch
    // makes canonical order. A strict `>` over the children in arena order is
    // the same tie-break.
    std::uint16_t best_move() const {
        require_position();
        const std::uint16_t mate = mating_move_.load(std::memory_order_relaxed);
        if (mate != kNoMove) {
            return mate;
        }
        const std::uint16_t count = arena_.children_count(root_);
        if (count == 0) {
            return kNoMove;
        }
        const std::uint32_t first = arena_.children_offset(root_);
        std::uint32_t best = first;
        for (std::uint16_t k = 1; k < count; ++k) {
            const std::uint32_t child = first + static_cast<std::uint32_t>(k);
            if (arena_.visit_count(child) > arena_.visit_count(best)) {
                best = child;
            }
        }
        return arena_.move(best);
    }

    // Canonical DFS preorder. `min_visits` of 0 emits every node; 1 emits the
    // visited subtree, which is what the 5,000-simulation golden trees record —
    // a 5,000-sim tree holds ~175,000 nodes of which ~5,000 have any visits, and
    // the rest carry visit 0 / value_sum 0.0 on both sides by construction.
    std::vector<TreeRecord> dump_tree(std::int32_t min_visits) const {
        require_position();
        std::vector<TreeRecord> out;
        // Frame: (node, next child slot). An explicit stack rather than
        // recursion so the traversal cost is independent of MAX_TREE_DEPTH.
        std::vector<std::pair<std::uint32_t, std::uint16_t>> stack;

        emit(out, root_, 0);
        stack.emplace_back(root_, 0);

        while (!stack.empty()) {
            auto &frame = stack.back();
            const std::uint16_t count = arena_.children_count(frame.first);
            if (frame.second >= count) {
                stack.pop_back();
                continue;
            }
            const std::uint32_t child = arena_.child(frame.first, frame.second);
            ++frame.second;
            if (arena_.visit_count(child) < min_visits) {
                continue;
            }
            emit(out, child, static_cast<std::uint16_t>(stack.size()));
            stack.emplace_back(child, 0);
        }
        return out;
    }

    // Every node carrying the terminal bit, with the FEN it stands for.
    //
    // The FEN is the point. A node marked terminal by the fifty-move rule is one
    // a host may refuse to end, so it can arrive back as the position we must
    // move from — and the only way to show that "still yields a legal move" is
    // not merely asserted is to hand the FEN to a fresh search and watch it
    // expand. That is what a promotion probe is. The walk therefore re-derives
    // the board along each path with the same make/unmake the search uses,
    // rather than reconstructing a position from node payload.
    std::vector<TerminalNode> terminal_nodes() {
        require_position();
        std::vector<TerminalNode> out;
        std::vector<std::uint16_t> path_moves;

        // (node, next child slot, the ParsedFen at this node).
        struct Frame {
            std::uint32_t node;
            std::uint16_t next_child;
            ParsedFen parsed;
        };
        std::vector<Frame> stack;
        stack.push_back(Frame{root_, 0, root_parsed_});

        collect_terminal(out, root_, 0, path_moves);

        while (!stack.empty()) {
            Frame &frame = stack.back();
            const std::uint16_t count = arena_.children_count(frame.node);
            if (frame.next_child >= count) {
                if (stack.size() > 1) {
                    board_.unmake_move(chess::Move(raw_move_[frame.node]));
                    path_moves.pop_back();
                }
                stack.pop_back();
                continue;
            }
            const std::uint32_t child = arena_.child(frame.node, frame.next_child);
            ++frame.next_child;

            const chess::Move move(raw_move_[child]);
            const ParsedFen before = frame.parsed;
            const int from = move.from().index();
            const int to = uci_destination(board_.board(), move).index();
            board_.make_move(move, guofish::is_zeroing(before, from, to));
            path_moves.push_back(arena_.move(child));

            collect_terminal(out, child, static_cast<std::uint16_t>(stack.size()), path_moves);
            stack.push_back(Frame{child, 0, board_.parsed()});
        }
        return out;
    }

private:
    // One step of a descent. The raw library move is what unwinds the board;
    // the arena index is what names the move in a diagnostic; the key, the
    // occupancy and the irreversibility flag are what the two repetition rules
    // read. Kept together so they can never fall out of step.
    struct PathStep {
        std::uint16_t raw_move;
        std::uint32_t node;
        std::uint64_t rep_key;
        Bitboard occupied;
        // `is_irreversible(move)` for the move that REACHED this position,
        // evaluated on the position it was played from — which is python-chess's
        // own evaluation point, after the pop.
        bool irreversible_before;
    };

    // C9. EVERYTHING ONE SIMULATION MUTATES, IN ONE OBJECT.
    //
    // Through C8 these were plain members of the search: one board, one path,
    // one applied-virtual-loss list, one scratch move buffer, because there was
    // one descent at a time. C9 runs W of them at once, so the state has to be
    // per-thread — and the honest way to make that true is to move it into a
    // struct rather than to add a lock, because a lock around the descent would
    // delete the entire point of the chunk.
    //
    // The serial path is not a special case of the parallel one and is not
    // reimplemented for it: `search()` builds ONE Descent pointing at the
    // search's own board and its own SearchStats, and runs exactly the code
    // C5-C8 were certified on. Acceptance layer 1 (W=1, K=1 still bit-exact
    // against Gate 1) is therefore a test of this refactor, and it is the first
    // thing that would fail if the split were wrong.
    //
    // `board` is a POINTER because the serial descent must drive the search's
    // own board — `apply_move`, `terminal_nodes` and `root_fen` all read it
    // afterwards — while a worker drives its own copy, positioned at the root
    // and returned there by the RAII unwind at the end of every simulation.
    struct Descent {
        SearchBoard own_board;              // a worker's copy; unused when serial
        SearchBoard *board = nullptr;
        SearchStats *stats = nullptr;

        std::vector<std::uint32_t> applied;   // nodes carrying this descent's VL
        std::vector<PathStep> path;
        // The same path as NORMALISED PACKED MOVES. Kept alongside `path`
        // rather than derived from it on demand because a failure message needs
        // it after the descent has been handed off to another thread, at which
        // point the arena indices in `path` are still valid but the board that
        // could interpret them is halfway back up the tree.
        std::vector<std::uint16_t> path_moves;
        // `path_counts` in the reference: rep_key -> occurrences on THIS path.
        // A flat vector rather than a hash map — the path is at most
        // MAX_TREE_DEPTH long, the scan is linear over a contiguous 12-byte
        // record, and clearing a vector between simulations does not touch the
        // allocator.
        std::vector<std::pair<std::uint64_t, int>> path_counts;

        // Scratch for the leaf's legal moves, reused across simulations.
        std::vector<std::uint16_t> packed;
        std::vector<std::uint16_t> raw;
        // C10. Parallel to `packed`: where each move stood in chess-library's
        // GENERATION order. Only the live evaluator's gather reads it, and only
        // to fix a softmax reduction order (cpp/evaluator.hpp). Filled on every
        // descent regardless, because a branch here would be a branch in the
        // hottest loop in the engine to save one 38-byte memcpy.
        std::vector<std::uint16_t> generation;
    };

    // Where a descent was standing, in the terms a failure message needs, and
    // WITHOUT a board.
    //
    // C5-C8 built these strings from `board_` at the point of failure, which
    // worked because the board was still at the leaf. Under C9 the leaf is
    // evaluated on the dispatcher, which has no board at all and could not be
    // given one cheaply — so the four things a diagnostic actually needs travel
    // with the leaf instead. `fen_of(parsed, clock, fullmove)` reproduces
    // `SearchBoard::diagnostic_fen()` exactly, by construction: that function is
    // the same call on the same ParsedFen.
    struct LeafDiag {
        const ParsedFen *parsed;
        int halfmove_clock;
        int fullmove_number;
        const std::uint16_t *path_moves;
        std::size_t path_len;
    };

    static std::string diag_fen(const LeafDiag &diag) {
        return fen_of(*diag.parsed, diag.halfmove_clock, diag.fullmove_number);
    }

    static std::string diag_path(const LeafDiag &diag) {
        if (diag.path_len == 0) {
            return "(root)";
        }
        return uci_list(diag.path_moves, diag.path_len);
    }

    // `parsed` is passed in rather than read off the board because the descent
    // is already holding it — one ParsedFen build per step, not three (see
    // run_simulation) — and rebuilding it here to make a diagnostic would be a
    // second derivation of the very thing the diagnostic is about.
    static LeafDiag diag_of(const Descent &d, const ParsedFen &parsed) {
        return LeafDiag{&parsed, d.board->halfmove_clock(),
                        static_cast<int>(d.board->board().fullMoveNumber()),
                        d.path_moves.data(), d.path_moves.size()};
    }

    void require_position() const {
        if (root_ == kNoNode) {
            throw std::logic_error("guofish::ReplaySearch: no position set");
        }
    }

    void collect_terminal(std::vector<TerminalNode> &out, std::uint32_t node, std::uint16_t depth,
                          const std::vector<std::uint16_t> &path_moves) const {
        if (!arena_.is_terminal(node)) {
            return;
        }
        TerminalNode record;
        record.path = uci_list(path_moves.data(), path_moves.size());
        if (record.path.empty()) {
            record.path = "(root)";
        }
        record.fen = board_.diagnostic_fen();
        record.value = arena_.terminal_value(node);
        record.visits = arena_.visit_count(node);
        record.children = arena_.children_count(node);
        record.expanded = arena_.lifecycle(node) == NodeState::Expanded ? 1u : 0u;
        record.depth = depth;
        out.push_back(std::move(record));
    }

    void emit(std::vector<TreeRecord> &out, std::uint32_t node, std::uint16_t depth) const {
        TreeRecord record{};
        record.depth = depth;
        record.move = arena_.move(node);
        record.visits = arena_.visit_count(node);
        record.value_sum = arena_.value_sum(node);
        record.prior = arena_.prior(node);
        record.children = arena_.children_count(node);
        record.terminal = arena_.is_terminal(node) ? 1u : 0u;
        record.terminal_value = arena_.terminal_value(node);
        out.push_back(record);
    }

    // --- virtual loss ------------------------------------------------------
    //
    // An integer in-flight COUNT, never a magnitude. The penalty is applied at
    // read time in q() and effective_visits(), so apply and repay are exact
    // inverses at any magnitude and a quiescent tree returns to exactly zero
    // with no floating-point residue. Mutating the stored value_sum instead —
    // which is the obvious implementation — would leave a residue at VL 2.5 and
    // put Gate 1 out of reach on the run that matters most.

    void apply_vloss(Descent &d, std::uint32_t node) {
        arena_.add_vloss(node, 1);
        d.applied.push_back(node);
    }

    void repay(Descent &d) {
        while (!d.applied.empty()) {
            arena_.add_vloss(d.applied.back(), -1);
            d.applied.pop_back();
        }
    }

    // C9. Repay a path that was handed to the dispatcher rather than resolved
    // in place. Same arithmetic, different owner: the list arrived with the
    // leaf, and the thread repaying it is not the thread that applied it.
    //
    // The counts are integers and `add_vloss` is a `fetch_add`, so apply and
    // repay commute with every other thread's apply and repay and the total
    // returns to exactly zero at quiescence regardless of interleaving. That is
    // the property acceptance layer 3 asserts, and it is the reason virtual
    // loss is a COUNT here and not a magnitude added into value_sum.
    void repay_list(const std::vector<std::uint32_t> &applied) {
        for (std::size_t i = applied.size(); i-- > 0;) {
            arena_.add_vloss(applied[i], -1);
        }
    }

    double effective_visits(std::uint32_t node) const {
        return static_cast<double>(arena_.visit_count(node)) +
               static_cast<double>(arena_.vloss_count(node)) * config_.virtual_loss;
    }

    // MCTSNode.q_value, transcribed. The `total == 0` guard is the reference's
    // and it is reachable: at VL 0.0 an in-flight unvisited child has
    // penalty 0.0 and total 0.0, and Python returns 0.0 there rather than fpu.
    double q_value(std::uint32_t node) const {
        const double penalty = static_cast<double>(arena_.vloss_count(node)) * config_.virtual_loss;
        const double total = static_cast<double>(arena_.visit_count(node)) + penalty;
        if (total == 0.0) {
            return 0.0;
        }
        return (arena_.value_sum(node) - penalty) / total;
    }

    // --- selection ---------------------------------------------------------
    //
    // MCTSNode.select_child + ucb_score, with the parent-dependent terms hoisted
    // exactly where the reference hoists them.
    //
    // Two things here are bit-level requirements rather than style:
    //
    //   * the multiply/divide order. Python evaluates
    //     `c_puct * self.prior * sqrt_parent_visits / (1 + self.effective_visits)`
    //     left to right, i.e. ((c*P)*sqrt)/(1+eff). Floating-point multiplication
    //     is not associative, so any other grouping is a different number.
    //   * the strict `>`. Ties go to the FIRST child in iteration order, and
    //     ~1% of selection steps are exact ties. This is why child order has to
    //     be canonical on both sides — see scope 2.6.
    std::uint32_t select_child(Descent &d, std::uint32_t parent, bool at_root) {
        const double parent_visits = effective_visits(parent);
        const double c = c_puct_of(parent_visits);
        const double sqrt_parent_visits = std::sqrt(parent_visits);
        const double fpu = at_root ? config_.fpu_root : config_.fpu_tree;

        const std::uint32_t first = arena_.children_offset(parent);
        const std::uint16_t count = arena_.children_count(parent);

        double best_score = -std::numeric_limits<double>::infinity();
        std::uint32_t best_child = kNoNode;

        for (std::uint16_t k = 0; k < count; ++k) {
            const std::uint32_t child = first + static_cast<std::uint32_t>(k);

            // FPU applies only to a child that is both unvisited AND has no
            // simulation in flight through it. The second half is not optional
            // in the reference and is not optional here: an in-flight node must
            // keep reading its virtual loss, or every descent would score it at
            // fpu simultaneously.
            const double q = (arena_.visit_count(child) == 0 && arena_.vloss_count(child) == 0)
                                 ? fpu
                                 : q_value(child);

            const double exploration = c * static_cast<double>(arena_.prior(child)) *
                                       sqrt_parent_visits / (1.0 + effective_visits(child));
            const double score = q + exploration;
            if (score > best_score) {
                best_score = score;
                best_child = child;
            }
        }

        ++d.stats->select_steps;
        return best_child;
    }

    // --- backup ------------------------------------------------------------
    //
    // MCTSNode.backpropagate: increment, add, negate, step to the parent. The
    // negation is what makes every node's Q read from the perspective of the
    // player who moved TO it, which is what lets selection take a plain max.
    void backpropagate(std::uint32_t node, double value) {
        std::uint32_t current = node;
        while (current != kNoNode) {
            arena_.add_visits(current, 1);
            arena_.add_value(current, value);
            value = -value;
            current = parent_[current];
        }
    }

    // --- expansion ---------------------------------------------------------

    // Generate, normalise and canonically order this position's legal moves.
    // The body moved to the free `generate_canonical_moves` in C7 so the
    // tablebase root probe shares it; behaviour is unchanged.
    void generate_canonical(std::vector<std::uint16_t> &packed, std::vector<std::uint16_t> &raw,
                            std::vector<std::uint16_t> *generation_index = nullptr) const {
        generate_canonical_moves(board_.board(), packed, raw, generation_index);
    }

    // C9. The same, for a worker driving its own board.
    static void generate_canonical(const SearchBoard &source, std::vector<std::uint16_t> &packed,
                                   std::vector<std::uint16_t> &raw,
                                   std::vector<std::uint16_t> *generation_index = nullptr) {
        generate_canonical_moves(source.board(), packed, raw, generation_index);
    }

    // Look the current position up in the dump, or fail by name.
    //
    // THE MISS PATH IS A TEST, which is why it carries this much context. A
    // board-path tokenization that derives the wrong ep square produces a key
    // the Python-generated dump does not contain, and the FEN printed here shows
    // the ep square the search believed it was at.
    //
    // C7 splits the key out into a parameter. The caller now builds an EvalRow —
    // the 68 tokens plus the key computed FROM those tokens — and passes the key
    // to the cache probe, to this lookup and to the cache insert, so all three
    // are keyed by one derivation. Recomputing `nn_key(parsed)` here would put a
    // second derivation back in, which is the trap the brief names under
    // "Risks / NNKey Generation".
    const ReplayDump::Entry &lookup(NNKey key, bool at_root, const LeafDiag &diag) {
        const ReplayDump::Entry *entry = dump_.find(key, at_root);
        if (entry != nullptr) {
            return *entry;
        }
        const int raw_ep = diag.parsed->ep_square;
        throw ReplayMiss(
            "guofish: replay dump miss (" + std::string(at_root ? "root" : "interior") +
            " table)\n  nn_key : 0x" + hex64(key.value) +
            "\n  fen    : " + diag_fen(diag) +
            "\n  raw ep : " + (raw_ep < 0 ? std::string("-") : square_name(raw_ep)) +
            "\n  path   : " + diag_path(diag) +
            "\n  The dump is generated from the Python reference, so a miss means "
            "this search reached a position the reference never evaluated, or "
            "tokenized a position it did evaluate differently.");
    }

    // Publish `node`'s children from a (moves, priors) pair, given the move list
    // the caller has already generated.
    //
    // C5 generated the moves inside here. C6 hoists that out, because the
    // terminal test needs the same list first — `outcome_of` asks how many legal
    // moves there are, and generating them twice per leaf would be both slower
    // and a second place for the two answers to disagree.
    //
    // C7 makes the payload a pair of pointers rather than a ReplayDump::Entry,
    // because it now arrives from two places — the dump on a cache miss, the
    // cache on a hit — and BOTH have to pass the move-list check below.
    //
    // `source_phrase` names which one in the first line and `source_label` is
    // the column heading on the move list. Two strings rather than one because
    // the label is padded to the width of "C++      " so the two lists line up
    // under each other, and tests/test_c5_gate1_quiet.py pins that layout by
    // asserting on the literal headings.
    void expand(std::uint32_t node, const std::uint16_t *moves, const float *priors,
                std::size_t count, const std::vector<std::uint16_t> &packed,
                const std::vector<std::uint16_t> &raw, const char *source_phrase,
                const char *source_label, const LeafDiag &diag, SearchStats &stats) {
        assert(!packed.empty());
        assert(moves != nullptr && priors != nullptr);

        // ASSERT ON MISMATCH RATHER THAN TRUSTING POSITIONAL ALIGNMENT — the
        // brief's words, and it applies to the cache for a sharper reason than
        // it applied to the dump. From the dump, a mismatch means movegen
        // disagrees with python-chess. From the cache, it means two positions
        // that are NOT the same position were given the same nn_key: a hash
        // collision, or a key derived from something other than the tokens. The
        // moves are stored precisely so that this is a named failure and not 37
        // priors landing on the wrong 37 moves.
        if (packed.size() != count || !std::equal(packed.begin(), packed.end(), moves)) {
            throw ReplayMiss(
                std::string("guofish: legal-move mismatch against ") + source_phrase + " at " +
                diag_fen(diag) + "\n  path     : " + diag_path(diag) +
                "\n  C++      : " + uci_list(packed.data(), packed.size()) +
                "\n  " + source_label + ": " + uci_list(moves, count));
        }

        const std::uint32_t offset = arena_.allocate(packed.size());
        for (std::size_t k = 0; k < packed.size(); ++k) {
            const std::uint32_t child = offset + static_cast<std::uint32_t>(k);
            arena_.set_move(child, packed[k]);
            arena_.set_prior(child, priors[k]);
            parent_[child] = node;
            raw_move_[child] = raw[k];
        }
        arena_.set_children(node, offset, static_cast<std::uint16_t>(packed.size()));
        ++stats.expansions;
    }

    // ParallelMCTS._expand_root: expand, seed one visit, seed the value.
    //
    // The reference assigns rather than accumulates (`root.visit_count = 1`,
    // `root.value_sum = ...`), which is the same thing on a node the arena has
    // just cleared.
    //
    // THE ROOT NEITHER READS NOR WRITES THE CACHE, AND THAT IS NOT AN OVERSIGHT.
    // `_expand_root` in the reference runs its own unbatched forward and never
    // touches `self.cache` — verified by reading it, not assumed — and the
    // consequence is load-bearing rather than incidental. The reference
    // softmaxes root priors on the GPU and interior priors on the CPU, and the
    // two disagree by up to ~2e-9 on the same position (see ReplayDump's header
    // and DECISIONS.md, C5), which is why the dump has two tables keyed by
    // (nn_key, is_root).
    //
    // A cache that served the root's entry to an interior visit of the same
    // position would hand it the GPU priors where the reference used the CPU
    // ones, and Gate 1 would fail at whatever depth the root position first
    // recurs — four plies, in a middlegame. Mirroring the reference's omission
    // keeps the two tables from ever meeting.
    //
    // It also costs nothing: a root is expanded once per set_position.
    void expand_root() {
        std::vector<std::uint16_t> packed;
        std::vector<std::uint16_t> raw;
        std::vector<std::uint16_t> generation;
        generate_canonical(packed, raw, &generation);
        if (packed.empty()) {
            // The reference answers this with `bestmove 0000` — `expand()`
            // returns having set is_expanded on an empty children dict, and
            // `search()` finds nothing to pick. There is no dump entry that can
            // stand for it (an entry has at least one move), so the honest
            // answer here is a named failure rather than an empty tree.
            throw TerminalReached(
                "guofish: the root position has no legal moves (checkmate or stalemate) at " +
                board_.diagnostic_fen() +
                "\n  A finished game has no move to search for. The reference returns "
                "None here, which the UCI layer reports as bestmove 0000.");
        }
        const LeafDiag diag{&root_parsed_, board_.halfmove_clock(),
                            static_cast<int>(board_.board().fullMoveNumber()), nullptr, 0};

        // C8. A PROMOTED root can arrive already marked terminal — that is what
        // C6's claimable draws are for: `draw_by_rule` marks the node and leaves
        // it unexpanded precisely so a host that declines the claim can hand the
        // position back as the one to move from. We are here, with legal moves
        // in `packed`, so the host has declined and we are playing on. The mark
        // is stale and has to come off before `set_children` will accept the
        // node.
        //
        // The reference does the same thing, in the same place and with the same
        // guard (`if root.children: root.is_terminal = False`); its guard is
        // after expansion because Python's expand() can produce no children,
        // while here the empty case has already thrown above. `terminal_value`
        // is deliberately left in place on both sides — see
        // NodeArena::clear_terminal.
        if (arena_.is_terminal(root_)) {
            arena_.clear_terminal(root_);
            ++reuse_.terminal_marks_cleared;
        }

        // C10. THE ROOT GOES DOWN THE SAME PATH AS EVERY OTHER NODE, and that is
        // the whole of the softmax unification.
        //
        // The reference does not: `_expand_root` runs its own unbatched forward
        // and hands `expand()` a CUDA tensor, so the root's softmax reduces on
        // the GPU while `BatchedEvaluator`'s bulk `.cpu()` puts every interior
        // node's on the CPU. Those disagree by up to 1.9e-9 across 6 of 37
        // priors — enough to flip a best move at 200 sims once the root position
        // recurs as an interior node, which a middlegame reaches in four plies.
        // The replay path below preserves that split faithfully, keyed
        // (nn_key, is_root), because Gate 1 is a bit-exactness claim about the
        // reference as it is.
        //
        // Production is not allowed both answers. `evaluate_and_expand` is the
        // one evaluation path, and the root now uses it — including the cache,
        // which the reference's root deliberately avoided ONLY because the two
        // tables must never meet. With one table there is nothing to keep apart:
        // a root cache hit is bit-identical to the fresh evaluation it replaces.
        // See DECISIONS.md, C10, for the divergence and its measured size.
        NetworkValue root_value(0.0);
        if (evaluator_ != nullptr) {
            const EvalRow row(root_parsed_);
            root_value = evaluate_and_expand(root_, row, packed, raw, generation, diag,
                                             cache_hit_, stats_);
        } else {
            const ReplayDump::Entry &entry =
                lookup(EvalRow(root_parsed_).key(), /*at_root=*/true, diag);
            expand(root_, entry.moves, entry.priors, entry.count, packed, raw, "the replay dump",
                   "golden   ", diag, stats_);
            root_value = NetworkValue(entry.value);
        }
        // ASSIGNED, not accumulated. `_expand_root` writes `root.visit_count = 1`
        // and `root.value_sum = ...`, and until C8 the distinction could not be
        // observed: a fresh root came out of the arena cleared, so += and = were
        // the same operation. A promoted root does not. One that spent the last
        // search as a fifty-move draw arrives carrying several hundred visits
        // from the terminal fast path, and `add_visits(1)` would leave the tree
        // claiming the root had been searched 400 times when the reference says
        // once.
        arena_.set_visits(root_, 1);
        arena_.set_value(root_, mover_value(root_value, root_parsed_.white_to_move));
        root_expanded_ = true;
    }

    // The v5 value head is White-POV by construction, and every node's Q is from
    // the perspective of whoever moved TO it — the opponent of the side now to
    // move. So the value is negated exactly when White is to move.
    //
    // Two overloads rather than one taking a double, so the conversion from
    // "absolute value of some kind" to "the number backed up" is the ONE place
    // the taxonomy in cpp/values.hpp is unwrapped, and it is a place that names
    // which kind it was handed. A `TerminalValue` has no overload here on
    // purpose: terminal values are already mover-POV in the reference and are
    // backed up without conversion, so an overload would invite a double
    // negation that no test would catch (0.0 and 1.0 both survive it on the
    // corpus).
    static double mover_value(NetworkValue absolute, bool white_to_move) noexcept {
        return white_to_move ? -absolute.value : absolute.value;
    }

    static double mover_value(TablebaseValue absolute, bool white_to_move) noexcept {
        return white_to_move ? -absolute.value : absolute.value;
    }

    // --- C6: the claimable draws -------------------------------------------
    //
    // `MCTSWorker._draw_by_rule`, transcribed including the order of its two
    // branches and the fact that the fifty-move branch returns BEFORE the
    // repetition key is counted into the path tally. That ordering is not
    // observable — the caller ends the simulation either way — but a
    // transcription that quietly tidies it is a transcription a later reader
    // cannot check against the reference line by line.
    //
    // The value is 0.0 and it is returned to the caller to back up directly. It
    // is not written anywhere a position could look it up: a draw here is a
    // property of the PATH, and the same position reached by another line is not
    // drawn. That is the discipline C7's cache has to inherit.
    bool draw_by_rule(Descent &d, std::uint64_t key) {
        if (d.board->halfmove_clock() >= 100) {
            ++d.stats->fifty_move_hits;
            return true;
        }
        int seen = 1;
        for (auto &entry : d.path_counts) {
            if (entry.first == key) {
                seen = entry.second + 1;
                entry.second = seen;
                break;
            }
        }
        if (seen == 1) {
            d.path_counts.emplace_back(key, 1);
        }
        // `rep_history_` is written only by set_position and apply_move, both of
        // which run with no search in flight, so every worker reads it without
        // a lock and without a copy. That was designed for in C8; this is where
        // it is collected.
        const auto it = rep_history_.find(key);
        const int history = (it == rep_history_.end()) ? 0 : it->second;
        if (history + seen >= 3) {
            ++d.stats->threefold_hits;
            return true;
        }
        return false;
    }

    // `Board.is_repetition(count)` over THIS simulation's path.
    //
    // The reference's simulation board is `root_board.copy(stack=False)`, so
    // python-chess's own move stack holds exactly the moves this simulation has
    // pushed and nothing before them — the pre-root game history is invisible to
    // it. This walk therefore runs over `path_` plus the sim root, and stops at
    // the same places python-chess stops: at an irreversible move, and at the
    // point where too few moves remain to reach `count`.
    //
    // Only `is_fivefold_repetition()` (count = 5) reaches this, and on the
    // corpus it never fires — `draw_by_rule` catches the threefold two
    // occurrences earlier, on every descent step. It is implemented rather than
    // asserted away because "unreachable" is a claim about the caller, and the
    // caller is `outcome_of`, which is a transcription of a function that asks.
    bool is_repetition(const Descent &d, int count) const {
        const std::vector<PathStep> &path_ = d.path;
        if (path_.empty()) {
            // The leaf is the root, which only happens if the root was handed
            // back unexpanded. Nothing has been played, so nothing has repeated.
            return false;
        }
        const Bitboard occupied_now = path_.back().occupied;
        const std::uint64_t target = path_.back().rep_key;

        // python-chess's fast pre-check, over the same set of previous positions
        // (`self._stack`): occupancy alone, which every real repetition shares.
        int maybe = 1;
        for (std::size_t j = path_.size() - 1; j-- > 0;) {
            if (path_[j].occupied == occupied_now && ++maybe >= count) {
                break;
            }
        }
        if (maybe < count && root_occupied_ == occupied_now) {
            ++maybe;
        }
        if (maybe < count) {
            return false;
        }

        // The full replay. `position` indexes the path: 0 is the sim root,
        // k is the position after path_[k - 1]'s move.
        int remaining = count;
        int position = static_cast<int>(path_.size());
        for (;;) {
            if (remaining <= 1) {
                return true;
            }
            // `if len(self.move_stack) < count - 1: break` — the move stack
            // holds exactly `position` moves at this point.
            if (position < remaining - 1) {
                break;
            }
            // Pop. `is_irreversible` is evaluated on the position the move was
            // played FROM, which is the one we have just stepped back to, and
            // the break happens BEFORE that position's key is compared.
            const bool irreversible = path_[static_cast<std::size_t>(position) - 1].irreversible_before;
            --position;
            if (irreversible) {
                break;
            }
            const std::uint64_t key =
                (position == 0) ? root_rep_key_ : path_[static_cast<std::size_t>(position) - 1].rep_key;
            if (key == target) {
                --remaining;
            }
        }
        return false;
    }

    // Record an intrinsic terminal on `node` and count it.
    //
    // `mark_terminal` is the arena's, and it refuses a node that already has
    // children. Nothing here can hand it one: a node with children is Expanded,
    // and the descent loop never stops at an Expanded node with children.
    static void count_terminal(SearchStats &stats, TerminalReason reason) {
        switch (reason) {
            case TerminalReason::Checkmate: ++stats.checkmates; break;
            case TerminalReason::Stalemate: ++stats.stalemates; break;
            case TerminalReason::InsufficientMaterial: ++stats.insufficient_material; break;
            case TerminalReason::SeventyFiveMoves: ++stats.seventyfive_moves; break;
            case TerminalReason::FivefoldRepetition: ++stats.fivefold_repetitions; break;
            case TerminalReason::None: break;
        }
    }

    void mark_terminal(SearchStats &stats, std::uint32_t node, TerminalReason reason,
                       double value) {
        count_terminal(stats, reason);
        arena_.mark_terminal(node, static_cast<float>(value));
    }

    // The depth-1 mate short-circuit — THE PYTHON HACK, replicated verbatim.
    //
    //     if depth == 1 and node.terminal_value == 1.0 and node.move is not None:
    //         self.stats['mating_move'] = node.move
    //         self.completion_event.set()
    //
    // A root child that is checkmate is a mate in one, so the search stops and
    // plays it. What makes it a hack rather than a feature is that it is welded
    // to `depth == 1` and to the exact double 1.0, it fires from inside a worker
    // by side-effecting a shared dict, and it truncates the search — the tree
    // the caller gets back is whatever had been built when it fired, which is
    // why the Gate 1 corpus has to record the reference's early exit rather than
    // assuming `root_visits == sims`.
    //
    // It is reproduced and NOT improved. See DECISIONS.md, C6.
    void maybe_mate_short_circuit(SearchStats &stats, int depth, double value,
                                  std::uint32_t node) {
        if (depth == 1 && value == 1.0 && arena_.move(node) != kNoMove) {
            // C9: a relaxed store. Two workers can find two different mates in
            // one on the same ply and the last writer wins, which is exactly
            // what the reference's `self.stats['mating_move'] = node.move` does
            // from inside a worker thread. Both are mates in one, so either is
            // a correct answer; the sequencing is not, and cannot be, defined.
            mating_move_.store(arena_.move(node), std::memory_order_relaxed);
            ++stats.mate_short_circuits;
        }
    }

    // --- one simulation ----------------------------------------------------

    void run_simulation(Descent &d) {
        // Restores the board and repays every applied virtual loss on ANY exit,
        // including an exception mid-descent. The reference does the same with a
        // `finally`; here it is a destructor, which is why C8 can delete the
        // defensive full-tree vloss reset entirely.
        struct Unwind {
            ReplaySearch *self;
            Descent *d;
            ~Unwind() {
                self->repay(*d);
                self->rewind_board(*d);
            }
        } unwind{this, &d};

        d.path_counts.clear();

        std::uint32_t node = root_;
        int depth = 0;
        // The position the descent currently stands on, in python-chess's terms.
        // Threaded rather than rebuilt: each step needs the position BEFORE the
        // move (for is_zeroing / is_irreversible) and the position AFTER it (for
        // rep_key, and at the leaf for nn_key), and the after of one step is the
        // before of the next. One ParsedFen build per descent step, not three.
        ParsedFen parsed = root_parsed_;
        apply_vloss(d, node);

        while (arena_.lifecycle(node) == NodeState::Expanded && arena_.children_count(node) > 0) {
            const std::uint32_t child = select_child(d, node, depth == 0);
            assert(child != kNoNode);
            const chess::Move move(raw_move_[child]);

            // python-chess's squares for this move: the NORMALISED destination,
            // never chess-library's king-takes-rook encoding. Both classifiers
            // below test the destination against bitboards the reference indexes
            // with g1/c1, and `_reduces_castling_rights` compares it against the
            // castling-rights ROOK squares, where the two encodings differ.
            const int from = move.from().index();
            const int to = uci_destination(d.board->board(), move).index();
            const bool zeroing = guofish::is_zeroing(parsed, from, to);
            const bool irreversible = guofish::is_irreversible(parsed, from, to);

            d.board->make_move(move, zeroing);
            parsed = d.board->parsed();
            const std::uint64_t key = rep_key(parsed).value;

            d.path.push_back(PathStep{move.move(), child, key, parsed.placement.occupied(),
                                      irreversible});
            d.path_moves.push_back(arena_.move(child));
            node = child;
            ++depth;
            apply_vloss(d, node);

            if (depth > d.stats->max_depth) {
                d.stats->max_depth = depth;
            }

            // === Draw by rule: fifty-move or threefold repetition ===
            //
            // Checked on EVERY descent step, not only at the leaf, because the
            // reference does — with tree reuse an interior node can have become
            // a draw since it was expanded. The node is marked terminal and left
            // UNEXPANDED, which is what makes it recoverable if a host declines
            // the claim and hands it back as the position to move from.
            if (draw_by_rule(d, key)) {
                ++d.stats->draw_by_rule_hits;
                repay(d);
                arena_.mark_terminal(node, 0.0f);
                backpropagate(node, 0.0);
                ++d.stats->simulations;
                return;
            }

            // The depth cap. Backs up 0.0 WITHOUT marking the node terminal,
            // exactly as the reference does — a capped node is not a game
            // result, and marking it would make it unrepresentable as a future
            // search root.
            if (depth >= config_.max_tree_depth) {
                ++d.stats->depth_cap_hits;
                repay(d);
                backpropagate(node, 0.0);
                ++d.stats->simulations;
                return;
            }
        }

        // === Cached terminal: a node an earlier simulation already resolved ===
        if (arena_.is_terminal(node)) {
            ++d.stats->terminal_fast_path_hits;
            const double value = arena_.terminal_value(node);
            repay(d);
            backpropagate(node, value);
            maybe_mate_short_circuit(*d.stats, depth, value, node);
            ++d.stats->simulations;
            return;
        }

        // === Intrinsic terminal: first visit ===
        //
        // The legal moves are generated once and used twice — `outcome_of` needs
        // the count, `expand` needs the list. Generating them separately would
        // be a second opportunity for the two to disagree about the position.
        std::vector<std::uint16_t> &packed = d.packed;
        std::vector<std::uint16_t> &raw = d.raw;
        generate_canonical(*d.board, packed, raw, &d.generation);

        const TerminalReason reason =
            outcome_of(parsed, d.board->in_check(), packed.size(), d.board->halfmove_clock(),
                       /*fivefold_repetition=*/packed.empty() ? false : is_repetition(d, 5));
        if (reason != TerminalReason::None) {
            const double value = terminal_value_of(reason);
            mark_terminal(*d.stats, node, reason, value);
            repay(d);
            backpropagate(node, value);
            maybe_mate_short_circuit(*d.stats, depth, value, node);
            ++d.stats->simulations;
            return;
        }

        // === Expansion and evaluation ===
        //
        // ONE derivation of the key, from the exact token row an evaluator would
        // be handed. The cache probe, the dump lookup and the cache insert all
        // use `row.key()`; nothing below recomputes it. See EvalRow in
        // cpp/keys.hpp for why that is a requirement and not a tidiness.
        const EvalRow row(parsed);
        const LeafDiag diag = diag_of(d, parsed);
        const NetworkValue nn_value =
            evaluate_and_expand(node, row, packed, raw, d.generation, diag, cache_hit_, *d.stats);

        // === Mode 2: the tablebase value override ===
        //
        // TREE-LOCAL, AND AFTER THE INSERT. The reference overrides first and
        // caches the result, with the comment that this saves a re-probe on a
        // later transposition. It does, and it is unsound: a Syzygy WDL ignores
        // the fifty-move rule, so it is a function of the position AND the
        // halfmove clock, while the cache key is a function of the position
        // alone — deliberately, because the clock is not a token. The stored WDL
        // is then served to the same position at a clock where the truth is a
        // draw. cpp/tablebase.hpp has the full argument; the reference's own
        // instrumentation counts the crossing.
        //
        // Here the override touches `backup_value` and nothing else, so it
        // reaches exactly one node — the leaf that was probed — through exactly
        // one backup. Nothing another position could read ever sees it, and the
        // cache entry written above already holds the network's own value. This is
        // not a discipline that has to be maintained: `probe_tablebase_value`
        // returns a `TablebaseValue`, `insert` takes a `NetworkValue`, and there
        // is no conversion, so moving the probe above the insert does not
        // compile.
        // `backup_value` is in the MOVER's perspective, which is what
        // backpropagate consumes; both branches convert from their own absolute
        // (White-POV) value through the overload named for its kind.
        const double backup_value =
            apply_tablebase(nn_value, parsed, d.board->halfmove_clock(), *d.stats);

        // Repay BEFORE the backup, which is the reference's ordering: backprop
        // always runs with zero in-flight loss on the path it walks.
        repay(d);
        backpropagate(node, backup_value);
        ++d.stats->simulations;
    }

    // Unwind a descent's board to the root. Split out of the RAII guard so the
    // parallel worker — which hands its path off rather than resolving it — can
    // reuse exactly the same walk.
    void rewind_board(Descent &d) {
        while (!d.path.empty()) {
            d.board->unmake_move(chess::Move(d.path.back().raw_move));
            d.path.pop_back();
        }
        d.path_moves.clear();
    }

    // The cache probe, the dump lookup, the expansion and the cache insert, in
    // the reference's order.
    //
    // C9 lifts this out of run_simulation unchanged so that BOTH callers are one
    // piece of code: the serial descent, and the dispatcher expanding a leaf a
    // worker handed it. That is not tidiness. `expand` is where the legal-move
    // list is checked against the priors it is about to be paired with, and
    // where the cache's poisoning class would surface; two copies of it would be
    // two places for the leaf and its evaluation to stop agreeing.
    //
    // ONE derivation of the key, from the exact token row an evaluator would be
    // handed. The cache probe, the dump lookup and the cache insert all use
    // `row.key()`; nothing here recomputes it. See EvalRow in cpp/keys.hpp for
    // why that is a requirement and not a tidiness.
    NetworkValue evaluate_and_expand(std::uint32_t node, const EvalRow &row,
                                     const std::vector<std::uint16_t> &packed,
                                     const std::vector<std::uint16_t> &raw,
                                     const std::vector<std::uint16_t> &generation,
                                     const LeafDiag &diag, CachedEval &scratch,
                                     SearchStats &stats) {
        if (cache_.has_value() && cache_->probe(row.key(), scratch)) {
            ++stats.cache_hits;
            expand(node, scratch.moves.data(), scratch.priors.data(), scratch.moves.size(), packed,
                   raw, "the transposition cache", "cache    ", diag, stats);
            return scratch.value;
        }
        if (cache_.has_value()) {
            ++stats.cache_misses;
        }

        // C10. THE LIVE PATH, and it is the only one that reaches the network.
        //
        // This is the SERIAL entry — `search()`, and `expand_root()` through it —
        // so the batch is one row wide. The parallel dispatcher does not come
        // through here: it has a whole batch to send and calls
        // `expand_from_live_row` directly after one crossing for all of them,
        // which is the "ONE GIL acquisition per batch" the boundary exists for.
        //
        // The dump is not consulted at all when an evaluator is installed. That
        // is deliberate and it is the softmax unification (scope §2.5): the
        // reference has two answers for one position depending on whether it is
        // the root, and production has one. See DECISIONS.md, C10.
        if (evaluator_ != nullptr) {
            std::memcpy(evaluator_->token_row(0), row.tokens(),
                        sizeof(std::int32_t) * static_cast<std::size_t>(kSeqLength));
            note_eval_timing(evaluator_->run(1), 1);
            return expand_from_live_row(node, row.key(), 0, packed, raw, generation, diag, stats);
        }

        // C9. A leaf the SERIAL reference never reached, on a run that is
        // allowed to reach it. Off by default, so C5 through C8 keep the
        // hard-failure behaviour their acceptance rests on.
        const ReplayDump::Entry *entry = dump_.find(row.key(), /*at_root=*/false);
        if (entry == nullptr && synthetic_fallback_) {
            ++stats.synthetic_evaluations;
            // `diag.parsed` is the leaf's position, held by whichever caller we
            // are on — the descent's own `parsed` when serial, the leaf's copy
            // when the dispatcher expands it. Reading it from there rather than
            // rebuilding one keeps this on the same single derivation everything
            // else in the expansion path uses.
            const ParsedFen &parsed = *diag.parsed;
            synthetic_.evaluate(row.key(), parsed, packed.data(), packed.size(),
                                synthetic_priors_);
            const NetworkValue value(SyntheticEvaluator::value_of(parsed));
            expand(node, packed.data(), synthetic_priors_.data(), packed.size(), packed, raw,
                   "the stand-in evaluator", "stand-in ", diag, stats);
            if (cache_.has_value()) {
                // `insert` takes the move count as a uint16, and the move list
                // came from movegen, so it is bounded by kMaxLegalMoves — but
                // the narrowing is written out rather than left implicit,
                // because a silent truncation here would pair a move list with
                // the wrong priors, which is the one failure cpp/cache.hpp is
                // built to make impossible.
                assert(packed.size() <= kMaxLegalMoves);
                cache_->insert(row.key(), value, packed.data(), synthetic_priors_.data(),
                               static_cast<std::uint16_t>(packed.size()));
                ++stats.cache_inserts;
            }
            return value;
        }
        if (entry == nullptr) {
            // The named failure, unchanged. lookup() re-does the find so the
            // message stays in one place.
            lookup(row.key(), /*at_root=*/false, diag);
        }
        expand(node, entry->moves, entry->priors, entry->count, packed, raw, "the replay dump",
               "golden   ", diag, stats);
        const NetworkValue nn_value(entry->value);

        // The insert takes the NETWORK's value, before any tablebase override —
        // see apply_tablebase. This is the line the reference gets wrong.
        if (cache_.has_value()) {
            cache_->insert(row.key(), nn_value, entry->moves, entry->priors, entry->count);
            ++stats.cache_inserts;
        }
        return nn_value;
    }

    // C10. Turn one evaluated row into this node's children.
    //
    // The gather reads ~26-38 entries out of a 4096-wide bf16 row and never
    // materialises the rest (scope §2.1); the softmax runs in the order
    // chess-library GENERATED the moves and the probabilities are permuted into
    // canonical order afterwards, never before (scope §2.6). Both are
    // `gather_softmax_canonical`'s job — see cpp/evaluator.hpp for why they are
    // two separate statements.
    //
    // `live_priors_` and `live_scratch_` are members rather than locals because
    // this runs once per expansion; after the first few leaves they have grown
    // to the widest move list the search will see and never allocate again.
    // Single-threaded by construction: the dispatcher is the only expander
    // (scope §2.2), and in a serial search the caller is the only thread.
    NetworkValue expand_from_live_row(std::uint32_t node, NNKey key, std::size_t eval_row,
                                      const std::vector<std::uint16_t> &packed,
                                      const std::vector<std::uint16_t> &raw,
                                      const std::vector<std::uint16_t> &generation,
                                      const LeafDiag &diag, SearchStats &stats) {
        assert(evaluator_ != nullptr);
        assert(generation.size() == packed.size());
        if (generation.size() != packed.size()) {
            throw std::logic_error(
                "guofish: the live evaluator was handed a leaf whose generation-order index "
                "does not match its move list. The two are filled together by "
                "generate_canonical_moves and cannot disagree unless a caller passed the "
                "wrong leaf's scratch.");
        }

        live_priors_.resize(packed.size());
        // C11b. THE TEMPERATURE IS APPLIED HERE AND NOWHERE ELSE, and that one
        // fact is requirement 3 of the brief.
        //
        // Every live expansion in the process arrives at this line. The
        // dispatcher calls it directly after a batch crossing; the serial
        // descent reaches it through `evaluate_and_expand`; and the ROOT
        // reaches it through `evaluate_and_expand` too, because C10 unified the
        // root onto the interior path (see expand_root). So there is no second
        // place a temperature could be applied differently, which means the
        // root and every interior node are softmaxed at the same sharpness.
        //
        // That is not a nicety. The reference's root/interior split — GPU
        // softmax at the root, CPU softmax everywhere else, two answers for one
        // position — is the defect C10 measured at 1.9e-9 and unified away. A
        // temperature threaded into the interior path alone would re-create
        // exactly that inconsistency under a new name and at a far larger
        // magnitude, which is the worse version of a defect already paid for.
        gather_softmax_canonical(evaluator_->policy_row(eval_row), packed.data(),
                                 generation.data(), packed.size(), live_scratch_,
                                 live_priors_.data(), config_.policy_temperature);
        const NetworkValue value(static_cast<double>(evaluator_->value_at(eval_row)));

        expand(node, packed.data(), live_priors_.data(), packed.size(), packed, raw,
               "the live evaluator", "network  ", diag, stats);

        // The NETWORK's value, before any tablebase override — the same
        // discipline the replay path documents at its own insert, and for the
        // same reason: a Syzygy WDL is a function of the halfmove clock and the
        // cache key is not.
        if (cache_.has_value()) {
            assert(packed.size() <= kMaxLegalMoves);
            cache_->insert(key, value, packed.data(), live_priors_.data(),
                           static_cast<std::uint16_t>(packed.size()));
            ++stats.cache_inserts;
        }
        return value;
    }

    // A search needs exactly one source of network answers, and starting without
    // one is a configuration mistake worth naming rather than a tree of zeros.
    void require_evaluation_source(const char *entry) const {
        if (evaluator_ == nullptr && dump_.empty()) {
            throw std::logic_error(std::string("guofish::ReplaySearch::") + entry +
                                   ": no replay dump loaded and no live evaluator installed");
        }
        // C11b. The replay path never reaches `gather_softmax_canonical` — its
        // priors were softmaxed by whatever produced the dump — so a
        // temperature set here would be accepted and then have no effect. That
        // is the same failure the equivalence-build assert exists for, one
        // build wider, and the answer is the same: refuse loudly. Checked once
        // per search rather than per node, which costs nothing.
        if (evaluator_ == nullptr && config_.policy_temperature != 1.0f) {
            throw std::invalid_argument(
                std::string("guofish::ReplaySearch::") + entry +
                ": policy_temperature is " + std::to_string(config_.policy_temperature) +
                " but this search has no live evaluator.\n  The replay dump holds "
                "priors that were already softmaxed, so a temperature could only be "
                "ignored here, never applied. Install an evaluator with "
                "set_evaluator(), or leave policy_temperature at 1.0.");
        }
    }

    void note_eval_timing(const EvalTiming &timing, std::size_t rows) {
        ++eval_stats_.batches;
        eval_stats_.rows += static_cast<std::int64_t>(rows);
        eval_stats_.acquire_wait_ns += timing.acquire_wait_ns;
        eval_stats_.call_ns += timing.call_ns;
        if (timing.acquire_wait_ns > eval_stats_.max_acquire_wait_ns) {
            eval_stats_.max_acquire_wait_ns = timing.acquire_wait_ns;
        }
        eval_stats_.acquire_wait_ns_samples.push_back(timing.acquire_wait_ns);
        eval_stats_.call_ns_samples.push_back(timing.call_ns);
    }

    // === Mode 2: the tablebase value override ===
    //
    // TREE-LOCAL, AND AFTER THE INSERT. The reference overrides first and caches
    // the result, with the comment that this saves a re-probe on a later
    // transposition. It does, and it is unsound: a Syzygy WDL ignores the
    // fifty-move rule, so it is a function of the position AND the halfmove
    // clock, while the cache key is a function of the position alone —
    // deliberately, because the clock is not a token. The stored WDL is then
    // served to the same position at a clock where the truth is a draw.
    // cpp/tablebase.hpp has the full argument; the reference's own
    // instrumentation counts the crossing.
    //
    // Here the override touches the returned value and nothing else, so it
    // reaches exactly one node — the leaf that was probed — through exactly one
    // backup. Nothing another position could read ever sees it, and the cache
    // entry written above already holds the network's own value. This is not a
    // discipline that has to be maintained: `probe_tablebase_value` returns a
    // `TablebaseValue`, `insert` takes a `NetworkValue`, and there is no
    // conversion, so moving the probe above the insert does not compile.
    //
    // The return is in the MOVER's perspective, which is what backpropagate
    // consumes; both branches convert from their own absolute (White-POV) value
    // through the overload named for its kind.
    double apply_tablebase(NetworkValue nn_value, const ParsedFen &parsed, int halfmove_clock,
                           SearchStats &stats) {
        double backup_value = mover_value(nn_value, parsed.white_to_move);
        if (tablebase_ != nullptr && within_tablebase_range(parsed)) {
            ++stats.tablebase_probes;
            const std::optional<TablebaseValue> tb =
                probe_tablebase_value(*tablebase_, parsed, halfmove_clock);
            if (tb.has_value()) {
                ++stats.tablebase_overrides;
                backup_value = mover_value(*tb, parsed.white_to_move);
            }
            // On a miss the neural value stands, exactly as the reference does
            // it: a position the loaded tables do not cover is not a position
            // about which anything has been learned.
        }
        return backup_value;
    }

    // --- C8: the compacting copy -------------------------------------------

    // The root child holding `packed`, or kNoNode.
    //
    // A linear scan over a contiguous range of at most a few dozen uint16s. A
    // binary search over the canonical key would also work — the children ARE
    // sorted — but this runs once per move, and a scan cannot be wrong about the
    // ordering invariant while a binary search can.
    std::uint32_t find_root_child(std::uint16_t packed) const {
        const std::uint16_t count = arena_.children_count(root_);
        const std::uint32_t first = arena_.children_offset(root_);
        for (std::uint16_t k = 0; k < count; ++k) {
            const std::uint32_t child = first + static_cast<std::uint32_t>(k);
            if (arena_.move(child) == packed) {
                return child;
            }
        }
        return kNoNode;
    }

    // `build_repetition_history(board)` for the new root, from the keys we kept.
    //
    // The reference recomputes this from the board's move stack at the top of
    // every search: seed the root at 1, then pop
    // `min(halfmove_clock, len(move_stack))` moves and count each position on
    // the way back. `history_keys_` IS that walk-back, most recent first, so the
    // rule reduces to taking a prefix.
    //
    // THE WINDOW IS WHY THIS IS NOT JUST AN INCREMENT. The two partitions the
    // brief asks about — what a simulation counts on its PATH and what the root
    // counts as HISTORY — only sum identically if the horizon moves correctly
    // when the move is applied. A non-zeroing move raises the clock by one and
    // adds one position, so the window grows by exactly one and the position
    // that enters it is exactly the old root. A zeroing move drops the clock to
    // zero and the window with it, which is right: nothing before a capture or a
    // pawn move can ever repeat again, so those occurrences must leave the count
    // rather than linger in it.
    void rebuild_rep_history() {
        rep_history_.clear();
        rep_history_[root_rep_key_] = 1;
        const std::size_t clock = board_.halfmove_clock() > 0
                                      ? static_cast<std::size_t>(board_.halfmove_clock())
                                      : 0u;
        const std::size_t window = clock < history_keys_.size() ? clock : history_keys_.size();
        for (std::size_t i = 0; i < window; ++i) {
            rep_history_[history_keys_[i]] += 1;
        }
    }

    // Copy the subtree at `src` into the standby arena, verify it, swap, promote.
    void compact_and_promote(std::uint32_t src, double decay) {
        ensure_standby();
        standby_->reset();
        standby_parent_.assign(config_.arena_capacity, kNoNode);
        standby_raw_move_.assign(config_.arena_capacity, 0);

        const std::size_t before = arena_.size();
        const std::uint32_t dst_root = copy_subtree(src);
        assert(dst_root == 0);
        const std::size_t copied = standby_->size();

        // THE STRUCTURAL DIFF, against the tree that is still sitting in the
        // other arena. Scope §7's mitigation for the fixup-bug risk. Under
        // asserts it always runs; in a Release build it runs when the config
        // asks. Either way it happens BEFORE the swap and before the promotion
        // edits below, so what it compares is a faithful copy against its
        // original, with nothing yet done to either.
#if defined(NDEBUG)
        const bool verify = config_.verify_compaction;
#else
        const bool verify = true;
#endif
        if (verify) {
            verify_copy(src, dst_root);
            ++reuse_.verifications;
        }

        arena_.swap_storage(*standby_);
        parent_.swap(standby_parent_);
        raw_move_.swap(standby_raw_move_);
        // The old arena's storage is now the standby's. Dropping the bump
        // pointer is the whole of "free the dead branches": nothing is written,
        // and try_allocate() clears each block as it hands it out next time.
        standby_->reset();

        root_ = dst_root;
        parent_[root_] = kNoNode;
        // The root has no move. The reference's promoted node keeps its `move`
        // field, but nothing reads it there (only the depth-1 mate hack does,
        // and that inspects children) and the tree serialisation writes 0 at
        // depth 0 regardless. Clearing it makes the arena say the same thing the
        // serialisation does instead of relying on the reader to know.
        arena_.set_move(root_, kNoMove);
        raw_move_[root_] = 0;
        root_expanded_ = arena_.children_count(root_) > 0;

        if (arena_.is_terminal(root_)) {
            // Kept, not cleared. At this instant the mark is still true: the
            // node was a game result and nobody has yet declined to claim it.
            // `expand_root` clears it if and only if the position turns out to
            // have legal moves, which is the reference's rule and the C6
            // promotion property. Counted here because a promoted terminal root
            // is the case the whole promotion invariant is about, and a corpus
            // that never produced one should say so out loud.
            ++reuse_.terminal_promotions;
        }

        if (decay != 1.0) {
            apply_decay(decay);
            ++reuse_.decays;
        }

        reuse_.nodes_copied += static_cast<std::int64_t>(copied);
        reuse_.nodes_dropped += static_cast<std::int64_t>(before - copied);
        if (static_cast<std::int64_t>(copied) > reuse_.largest_copy) {
            reuse_.largest_copy = static_cast<std::int64_t>(copied);
        }
    }

    void ensure_standby() {
        if (standby_) {
            return;
        }
        // Allocated on FIRST USE rather than in the constructor. A ping-pong
        // pair at the 2M default is ~140 MB of node payload, and a search that
        // never applies a move — every test before this chunk, and the tablebase
        // root bypass — has no use for the second half of it. The allocation is
        // a few tens of milliseconds once per game, against a move that takes a
        // second.
        standby_ = std::make_unique<NodeArena<Accumulator>>(config_.arena_capacity);
    }

    // Breadth-first copy. Returns the new index of the subtree root, which is 0.
    //
    // BREADTH-FIRST, NOT DEPTH-FIRST, and the choice is about the next search
    // rather than about this one. A node's children are one contiguous block
    // either way — that is forced by the (offset, count) representation — but
    // BFS additionally lays whole DEPTH LEVELS out contiguously, so a descent
    // walks forward through memory instead of jumping to wherever a DFS happened
    // to have finished the previous branch. Selection reads a whole sibling
    // range per step; BFS is the order that keeps those ranges near the ranges
    // read just before and just after them.
    //
    // Nothing about correctness depends on it: `dump_tree` and the golden
    // comparison both traverse by (offset, count), so any allocation order that
    // keeps siblings contiguous and in canonical order produces the same tree.
    std::uint32_t copy_subtree(std::uint32_t src_root) {
        NodeArena<Accumulator> &dst = *standby_;
        const std::uint32_t dst_root = dst.allocate(1);
        copy_node(src_root, dst_root);
        standby_parent_[dst_root] = kNoNode;
        standby_raw_move_[dst_root] = raw_move_[src_root];

        // (source index, destination index). A vector used as a queue with a
        // read cursor: the pairs are never removed, so the storage is one
        // allocation for the whole subtree rather than a deque's chain of blocks.
        std::vector<std::pair<std::uint32_t, std::uint32_t>> queue;
        queue.reserve(64);
        queue.emplace_back(src_root, dst_root);

        for (std::size_t head = 0; head < queue.size(); ++head) {
            const std::uint32_t s = queue[head].first;
            const std::uint32_t d = queue[head].second;
            const std::uint16_t count = arena_.children_count(s);
            if (count == 0) {
                continue;
            }
            // Allocated as one block, which is what makes the fixup a single
            // number: every child's new index is `offset + k` for the same k it
            // had in the source. There is no per-child remap table and therefore
            // no per-child remap bug.
            const std::uint32_t offset = dst.allocate(count);
            const std::uint32_t src_first = arena_.children_offset(s);
            for (std::uint16_t k = 0; k < count; ++k) {
                const std::uint32_t sc = src_first + static_cast<std::uint32_t>(k);
                const std::uint32_t dc = offset + static_cast<std::uint32_t>(k);
                copy_node(sc, dc);
                standby_parent_[dc] = d;
                standby_raw_move_[dc] = raw_move_[sc];
                queue.emplace_back(sc, dc);
            }
            // Published only after every child is written, so the release inside
            // set_children covers the whole block.
            dst.set_children(d, offset, count);
        }
        return dst_root;
    }

    // One node's payload, source arena to standby arena.
    //
    // `value_sum` moves in the ACCUMULATOR'S OWN REPRESENTATION. Reading it as a
    // double and writing it back would round twice under Q32 — and the claim
    // being made about a compacted tree is that it is bit-identical, which a
    // round trip through a lossy intermediate cannot support even when it
    // happens to hold.
    void copy_node(std::uint32_t src, std::uint32_t dst) {
        NodeArena<Accumulator> &out = *standby_;
        out.set_move(dst, arena_.move(src));
        out.set_prior(dst, arena_.prior(src));
        out.set_visits(dst, arena_.visit_count(src));
        out.set_value_raw(dst, arena_.value_sum_raw(src));

        // A quiescent tree holds zero in-flight losses — that is the RAII
        // property that lets C8 delete the reference's defensive walk — so this
        // should always be zero. It is COPIED rather than assumed, because a
        // compaction that silently zeroed a non-zero count would erase the
        // evidence of the very bug the invariant is about, and asserted so that
        // a sanitized build says so at the seam instead of at some later
        // selection step.
        const std::int32_t vloss = arena_.vloss_count(src);
        assert(vloss == 0 && "apply_move on a tree with simulations in flight");
        if (vloss != 0) {
            out.add_vloss(dst, vloss);
        }

        if (arena_.is_terminal(src)) {
            // Before the children block exists, because mark_terminal refuses a
            // node that has children — the same invariant, read from the other
            // side. A source node that was somehow both would fail here rather
            // than propagate.
            out.mark_terminal(dst, arena_.terminal_value(src));
        }
    }

    // The full-tree structural diff (scope §7). Throws TreeCorruption naming the
    // DFS path and the field.
    //
    // Every field is compared, not a summary: a hash would say "different" and
    // this says "the fourth child of e2e4 g8f6 has prior 0.031 where the source
    // had 0.017", which is the difference between a five-minute diagnosis and a
    // day's. The float fields are compared as BIT PATTERNS via memcmp-free
    // integer punning avoidance — `==` on floats would pass two values that
    // Gate 1 would then fail on.
    void verify_copy(std::uint32_t src_root, std::uint32_t dst_root) const {
        const NodeArena<Accumulator> &dst = *standby_;
        // Every destination slot must be reached exactly once. A remapped offset
        // that points at a plausible-looking but wrong block leaves some slots
        // unvisited and visits others twice, and neither shows up in a
        // field-by-field comparison that only follows the corrupted links.
        std::vector<std::uint8_t> seen(dst.size(), 0);
        std::vector<std::uint16_t> path;
        std::size_t visited = 0;

        struct Frame {
            std::uint32_t src;
            std::uint32_t dst;
            std::uint16_t next_child;
        };
        std::vector<Frame> stack;

        const auto fail = [&](const std::string &what) {
            throw TreeCorruption(
                "guofish: the compacted tree does not match the tree it was copied from\n"
                "  path   : " + (path.empty() ? std::string("(root)")
                                              : uci_list(path.data(), path.size())) +
                "\n  detail : " + what +
                "\n  This is a ping-pong arena fixup bug: the surviving subtree was copied "
                "into the standby arena and the copy is not the original. Nothing above this "
                "point can be trusted.");
        };

        const auto compare = [&](std::uint32_t s, std::uint32_t d) {
            if (static_cast<std::size_t>(d) >= dst.size()) {
                fail("destination index " + std::to_string(d) + " is outside the copied region [0, " +
                     std::to_string(dst.size()) + ")");
            }
            if (seen[d] != 0) {
                fail("destination node " + std::to_string(d) + " is reachable twice");
            }
            seen[d] = 1;
            ++visited;
            if (arena_.move(s) != dst.move(d)) {
                fail("move " + uci_of(dst.move(d)) + " where the source has " +
                     uci_of(arena_.move(s)));
            }
            if (!bitwise_equal(arena_.prior(s), dst.prior(d))) {
                fail("prior differs bitwise on move " + uci_of(arena_.move(s)));
            }
            if (arena_.visit_count(s) != dst.visit_count(d)) {
                fail("visits " + std::to_string(dst.visit_count(d)) + " where the source has " +
                     std::to_string(arena_.visit_count(s)));
            }
            if (arena_.value_sum_raw(s) != dst.value_sum_raw(d)) {
                fail("value_sum differs in the accumulator's own representation");
            }
            if (arena_.vloss_count(s) != dst.vloss_count(d)) {
                fail("vloss_count differs");
            }
            if (arena_.children_count(s) != dst.children_count(d)) {
                fail("children_count " + std::to_string(dst.children_count(d)) +
                     " where the source has " + std::to_string(arena_.children_count(s)));
            }
            if (arena_.is_terminal(s) != dst.is_terminal(d)) {
                fail("terminal bit differs");
            }
            if (arena_.lifecycle(s) != dst.lifecycle(d)) {
                fail("lifecycle differs");
            }
            if (!bitwise_equal(arena_.terminal_value(s), dst.terminal_value(d))) {
                fail("terminal_value differs bitwise");
            }
            const std::uint16_t count = dst.children_count(d);
            if (count != 0) {
                const std::size_t first = dst.children_offset(d);
                if (first + count > dst.size()) {
                    fail("child range [" + std::to_string(first) + ", " +
                         std::to_string(first + count) + ") is outside the copied region [0, " +
                         std::to_string(dst.size()) + ")");
                }
            }
        };

        compare(src_root, dst_root);
        stack.push_back(Frame{src_root, dst_root, 0});

        while (!stack.empty()) {
            Frame &frame = stack.back();
            if (frame.next_child >= dst.children_count(frame.dst)) {
                stack.pop_back();
                if (!path.empty()) {
                    path.pop_back();
                }
                continue;
            }
            const std::uint16_t k = frame.next_child++;
            const std::uint32_t s = arena_.children_offset(frame.src) + k;
            const std::uint32_t d = dst.children_offset(frame.dst) + k;
            path.push_back(dst.move(d));
            compare(s, d);
            stack.push_back(Frame{s, d, 0});
        }

        if (visited != dst.size()) {
            path.clear();
            throw TreeCorruption(
                "guofish: the compacted tree reaches " + std::to_string(visited) + " of the " +
                std::to_string(dst.size()) +
                " nodes the compaction allocated\n"
                "  Unreachable slots mean a children_offset points somewhere other than the "
                "block that was allocated for it — the classic ping-pong fixup bug. The tree "
                "would still traverse; it would traverse the wrong nodes.");
        }
    }

    // Bit-pattern equality for a float, without type punning.
    //
    // `==` is the wrong operator here twice over: it calls +0.0 and -0.0 equal,
    // and it calls no NaN equal to itself. A prior that arrived as -0.0 on one
    // side and +0.0 on the other would pass, and Gate 1's golden comparison —
    // which packs the bytes — would then fail on a tree this function had
    // certified. Comparing through `std::memcmp` on the objects avoids both a
    // `reinterpret_cast` (Global Rule 6) and any aliasing question.
    static bool bitwise_equal(float a, float b) noexcept {
        return std::memcmp(&a, &b, sizeof(float)) == 0;
    }

    // Scale every node's inherited visits, and its value with them.
    //
    // A FLAT SCAN, not a traversal: after `copy_subtree` the arena holds exactly
    // the promoted tree in [0, size()) and nothing else, which is one of the
    // things compaction buys.
    //
    // BOTH fields move, by the same ratio, so Q is unchanged. That is the whole
    // intent — decay is a statement about CONFIDENCE, not about evaluation. A
    // node the ponder search visited 8,000 times and scored at +0.4 should come
    // out of the promotion still scoring +0.4, but with the weight of 8,000·d
    // visits rather than 8,000, so a few thousand fresh simulations can move it.
    // Scaling visits alone would divide the value by d instead and turn every
    // inherited node into a wild evaluation.
    //
    // A node with visits is floored at 1 rather than allowed to reach 0. A node
    // with children and zero visits is a state the search never otherwise builds
    // — selection treats it as FPU-eligible while it already has an expanded
    // subtree underneath — and manufacturing it here would be a decay that
    // changed the tree's kind, not its weight.
    void apply_decay(double decay) {
        for (std::size_t i = 0; i < arena_.size(); ++i) {
            const auto node = static_cast<std::uint32_t>(i);
            const std::int32_t visits = arena_.visit_count(node);
            if (visits <= 0) {
                continue;
            }
            std::int64_t scaled = std::llround(static_cast<double>(visits) * decay);
            if (scaled < 1) {
                scaled = 1;
            }
            if (scaled == visits) {
                continue;
            }
            const double ratio = static_cast<double>(scaled) / static_cast<double>(visits);
            const auto raw = arena_.value_sum_raw(node);
            arena_.set_visits(node, static_cast<std::int32_t>(scaled));
            if constexpr (Accumulator::is_fixed_point()) {
                arena_.set_value_raw(
                    node, static_cast<typename Accumulator::value_type>(
                              std::llround(static_cast<double>(raw) * ratio)));
            } else {
                arena_.set_value_raw(
                    node, static_cast<typename Accumulator::value_type>(raw * ratio));
            }
        }
    }

    // --- C9: the parallel engine -------------------------------------------

    enum class LeafOutcome : std::uint8_t {
        Delivered,   // the worker resolved and backed it up itself
        Submitted,   // handed to the dispatcher, virtual loss still applied
        Discarded,   // lost the PENDING claim; nothing was backed up
    };

    // One in-flight leaf. `MpscNode` is the queue link; deriving from it rather
    // than embedding it is what lets the dispatcher recover the leaf from what
    // the queue hands back with a `static_cast` down a non-virtual base, so
    // Global Rule 6's reinterpret_cast question never arises.
    //
    // Everything the dispatcher needs to evaluate, expand, repay and back up
    // travels in here, because by the time it runs the worker's board is
    // somewhere else entirely. In particular `applied` — the virtual-loss path —
    // is MOVED out of the worker's descent, so ownership of the repayment moves
    // with it and no thread can repay a loss it does not hold.
    struct LeafNode : MpscNode {
        std::uint32_t node = kNoNode;
        ParsedFen parsed;
        int halfmove_clock = 0;
        int fullmove_number = 1;
        std::vector<std::uint16_t> packed;
        std::vector<std::uint16_t> raw;
        // C10. See Descent::generation.
        std::vector<std::uint16_t> generation;
        std::vector<std::uint32_t> applied;
        std::vector<std::uint16_t> path_moves;
        // Produced by one worker, released by the dispatcher: SPSC per slot, so
        // one atomic flag is the whole protocol.
        std::atomic<bool> in_use{false};
    };

    void run_workers(const ParallelConfig &pc) {
        const int workers = pc.workers;
        const int per_worker = pc.in_flight;

        queue_ = std::make_unique<MpscQueue>();
        slots_.clear();
        slots_.reserve(static_cast<std::size_t>(workers) * static_cast<std::size_t>(per_worker));
        for (int i = 0; i < workers * per_worker; ++i) {
            slots_.push_back(std::make_unique<LeafNode>());
        }

        // Assigned before the Descents are built, because each Descent holds a
        // pointer into this vector and a later resize would dangle every one.
        worker_stats_.assign(static_cast<std::size_t>(workers), SearchStats{});
        dispatch_stats_ = SearchStats{};

        descents_.clear();
        descents_.reserve(static_cast<std::size_t>(workers));
        for (int i = 0; i < workers; ++i) {
            auto d = std::make_unique<Descent>();
            // A COPY of the search's board, positioned at the root. Copying is
            // what makes the descent lock-free: chess-library's make/unmake is
            // stateful, and W threads sharing one Board would need a lock around
            // the single hottest thing in the engine.
            d->own_board = board_;
            d->board = &d->own_board;
            d->stats = &worker_stats_[static_cast<std::size_t>(i)];
            descents_.push_back(std::move(d));
        }

        issued_.store(0, std::memory_order_relaxed);
        delivered_.store(0, std::memory_order_relaxed);
        outstanding_.store(0, std::memory_order_relaxed);
        queued_.store(0, std::memory_order_relaxed);
        drain_epoch_.store(0, std::memory_order_relaxed);
        aborted_.store(false, std::memory_order_relaxed);
        error_ = nullptr;
        waiting_workers_ = 0;
        running_workers_ = workers;
        collisions_ = 0;
        waits_ = 0;

        const std::vector<int> slots_for_affinity = affinity_slots(topology(), pc.affinity);
        // SIZED BEFORE ANY THREAD STARTS. Each worker writes its own element if
        // the platform refuses the pin request, and growing the vector while an
        // earlier worker holds a reference into it would reallocate under that
        // worker — a use-after-free that only fires on a machine where pinning
        // FAILS, i.e. never on the machine it was written on.
        par_.pinned_cpus.assign(static_cast<std::size_t>(workers), -1);
        for (int i = 0; i < workers; ++i) {
            par_.pinned_cpus[static_cast<std::size_t>(i)] =
                slots_for_affinity.empty()
                    ? -1
                    : slots_for_affinity[static_cast<std::size_t>(i) % slots_for_affinity.size()];
        }

        const auto started = std::chrono::steady_clock::now();

        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(workers) + 1);
        std::thread dispatcher([this, &pc] { dispatcher_loop(pc); });
        for (int i = 0; i < workers; ++i) {
            const int cpu = par_.pinned_cpus[static_cast<std::size_t>(i)];
            threads.emplace_back([this, &pc, i, cpu] {
                if (cpu >= 0 && !pin_current_thread(cpu)) {
                    // Reported, not thrown. A machine that will not honour an
                    // affinity request still runs the search correctly; what it
                    // must not do is let a BENCH.md row claim a pinning that did
                    // not happen, so the slot is rewritten to -1. Each worker
                    // writes only its own element of a vector that was sized
                    // before any thread existed.
                    par_.pinned_cpus[static_cast<std::size_t>(i)] = -1;
                }
                worker_loop(i, pc);
            });
        }
        for (std::thread &t : threads) {
            t.join();
        }
        dispatcher.join();

        par_.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - started)
                           .count();

        release_stranded_leaves();
        merge_worker_stats();
        par_.delivered = delivered_.load(std::memory_order_relaxed);
        par_.select_collisions = collisions_;
        par_.worker_waits = waits_;
        par_.worker_terminals = par_.delivered - par_.queued_leaves;

        if (error_) {
            std::exception_ptr error = error_;
            error_ = nullptr;
            std::rethrow_exception(error);
        }
    }

    // Nothing should be left here. The only way a slot survives the join is an
    // exception thrown out of the dispatcher mid-batch — and in that case the
    // leaf's virtual loss is still applied and its node is still PENDING, so
    // "the tree is now in an undefined state" would be a second failure caused
    // by the first. Unwinding here means the C9 conservation invariants hold
    // even on the error path, which is what makes them assertable at all.
    void release_stranded_leaves() {
        for (std::unique_ptr<LeafNode> &slot : slots_) {
            if (!slot->in_use.load(std::memory_order_acquire)) {
                continue;
            }
            repay_list(slot->applied);
            slot->applied.clear();
            // kNoNode is a slot that was acquired and then abandoned by a
            // throwing descent: it never reached a leaf, so there is no PENDING
            // claim to withdraw. Its virtual loss was already repaid by
            // run_simulation_parallel's RAII unwind on the way out.
            if (slot->node != kNoNode &&
                arena_.lifecycle(slot->node) == NodeState::Pending) {
                arena_.release_pending(slot->node);
            }
            slot->node = kNoNode;
            slot->in_use.store(false, std::memory_order_release);
        }
    }

    // Worker order then the dispatcher, always. Integer addition is
    // associative, so the merged totals are independent of how the threads
    // interleaved — the same property that lets Q32 make the tree itself
    // reproducible. `max_depth` is a maximum rather than a sum and is folded as
    // one.
    void merge_worker_stats() {
        const auto fold = [](SearchStats &into, const SearchStats &from) {
            into.simulations += from.simulations;
            into.expansions += from.expansions;
            into.depth_cap_hits += from.depth_cap_hits;
            into.select_steps += from.select_steps;
            into.max_depth = from.max_depth > into.max_depth ? from.max_depth : into.max_depth;
            into.draw_by_rule_hits += from.draw_by_rule_hits;
            into.fifty_move_hits += from.fifty_move_hits;
            into.threefold_hits += from.threefold_hits;
            into.checkmates += from.checkmates;
            into.stalemates += from.stalemates;
            into.insufficient_material += from.insufficient_material;
            into.seventyfive_moves += from.seventyfive_moves;
            into.fivefold_repetitions += from.fivefold_repetitions;
            into.terminal_fast_path_hits += from.terminal_fast_path_hits;
            into.mate_short_circuits += from.mate_short_circuits;
            into.cache_hits += from.cache_hits;
            into.cache_misses += from.cache_misses;
            into.cache_inserts += from.cache_inserts;
            into.tablebase_probes += from.tablebase_probes;
            into.tablebase_overrides += from.tablebase_overrides;
            into.synthetic_evaluations += from.synthetic_evaluations;
        };
        for (const SearchStats &s : worker_stats_) {
            fold(stats_, s);
        }
        fold(stats_, dispatch_stats_);
    }

    void record_error(std::exception_ptr error) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) {
                error_ = std::move(error);
            }
            // SET UNDER THE MUTEX. Both wait predicates read `aborted_`, so a
            // store outside the lock races the window between a waiter
            // evaluating its predicate and actually sleeping: the notify fires
            // into an empty condition variable and the waiter never wakes. That
            // is a deadlock reachable from any exception — an exhausted arena,
            // a dump miss — which is to say from the paths a test is most likely
            // to take.
            aborted_.store(true, std::memory_order_release);
        }
        worker_cv_.notify_all();
        dispatch_cv_.notify_one();
    }

    // A free slot of this worker's own K, or nullptr.
    //
    // PER-WORKER RINGS RATHER THAN A SHARED POOL. The brief phrases the throttle
    // two ways — "K in-flight paths per thread" and "stall only when the global
    // count hits W*K" — and they differ when the workers are unbalanced: a
    // worker can run out of its own K while the global count is below W*K. The
    // difference is not observable here, because the dispatcher drains only when
    // EVERY worker is blocked, so a worker that blocks early simply blocks
    // sooner into the same drain. What the rings buy is that a submission
    // touches no shared allocator at all — the hot path is one relaxed load and
    // one release store on a flag this worker is the only producer for.
    // `worker_waits` in ParallelStats counts how often it mattered.
    LeafNode *acquire_slot(int base, int per_worker) {
        for (int k = 0; k < per_worker; ++k) {
            LeafNode *slot = slots_[static_cast<std::size_t>(base + k)].get();
            if (!slot->in_use.load(std::memory_order_acquire)) {
                // Cleared on acquisition, not on release. A descent that throws
                // between here and the submission leaves the slot marked in use,
                // and release_stranded_leaves would otherwise find last
                // simulation's node index sitting in it and try to release a
                // PENDING claim that belongs to a leaf which was already
                // expanded. kNoNode is how that slot says "I hold nothing".
                slot->node = kNoNode;
                slot->applied.clear();
                slot->in_use.store(true, std::memory_order_release);
                return slot;
            }
        }
        return nullptr;
    }

    // Sleep until the dispatcher has finished a drain, or the search is over.
    //
    // `epoch` is read by the caller BEFORE it discovered it could not proceed,
    // so a drain that completed in between is not missed and this returns
    // immediately. Registering in `waiting_workers_` is also what ARMS the
    // dispatcher: its drain condition is "the queue is non-empty and every
    // worker is either waiting or finished".
    void worker_block(std::uint64_t epoch) {
        std::unique_lock<std::mutex> lock(mutex_);
        ++waiting_workers_;
        dispatch_cv_.notify_one();
        worker_cv_.wait(lock, [&] {
            return aborted_.load(std::memory_order_relaxed) ||
                   drain_epoch_.load(std::memory_order_relaxed) != epoch;
        });
        --waiting_workers_;
    }

    void worker_loop(int id, const ParallelConfig &pc) {
        Descent &d = *descents_[static_cast<std::size_t>(id)];
        const int base = id * pc.in_flight;
        std::int64_t collisions = 0;
        std::int64_t waits = 0;

        try {
            for (;;) {
                if (aborted_.load(std::memory_order_relaxed)) {
                    break;
                }
                // MCTSWorker._work_loop checks completion_event at the TOP of
                // the loop, so the simulation that set it is counted and the
                // next one never starts.
                if (mating_move_.load(std::memory_order_relaxed) != kNoMove) {
                    break;
                }

                // CLAIM A SIMULATION, EXACTLY ONE.
                //
                // `issued_` counts simulations that will each deliver exactly
                // one backup into the root. A thread that draws a slot at or
                // past the target gives it straight back and leaves; a thread
                // whose descent is DISCARDED gives its slot back too and loops,
                // so the slot is never lost — the discarding thread is still in
                // the loop and will re-take it if nobody else does. At
                // quiescence `issued_` is therefore exactly `target_`, which is
                // what makes "delivered sims exactly match the requested
                // budget" an equality rather than an approximation.
                const int slot = issued_.fetch_add(1, std::memory_order_acq_rel);
                if (slot >= target_) {
                    issued_.fetch_sub(1, std::memory_order_acq_rel);
                    break;
                }

                // THROTTLE. Read the epoch, THEN re-check for a slot, THEN
                // sleep — in that order, every time round.
                //
                // Hoisting the epoch read out of the loop looks like an
                // optimisation and is a busy-wait: after the first wake,
                // `drain_epoch_ != epoch` is permanently true, so every
                // subsequent worker_block returns instantly and the thread
                // spins on acquire_slot instead of sleeping. It only shows up
                // when a drain can fail to free one of THIS worker's slots —
                // W>1 with max_batch below the outstanding count — which is
                // exactly the configuration a throughput sweep reaches and a
                // W=1 test does not.
                LeafNode *item = nullptr;
                for (;;) {
                    const std::uint64_t epoch = drain_epoch_.load(std::memory_order_relaxed);
                    item = acquire_slot(base, pc.in_flight);
                    if (item != nullptr || aborted_.load(std::memory_order_relaxed)) {
                        break;
                    }
                    ++waits;
                    worker_block(epoch);
                }
                if (item == nullptr) {
                    issued_.fetch_sub(1, std::memory_order_acq_rel);
                    break;
                }

                const std::uint64_t before = drain_epoch_.load(std::memory_order_relaxed);
                const LeafOutcome outcome = run_simulation_parallel(d, item);
                if (outcome == LeafOutcome::Submitted) {
                    continue;  // the dispatcher owns the slot now
                }
                item->in_use.store(false, std::memory_order_release);
                if (outcome == LeafOutcome::Delivered) {
                    delivered_.fetch_add(1, std::memory_order_acq_rel);
                    continue;
                }
                // Discarded: another simulation owns the leaf we descended to.
                // Hand the budget slot back and wait for the dispatcher to
                // resolve it, rather than spinning on a descent that will make
                // the identical choice — which, at virtual loss 0, it would,
                // forever.
                ++collisions;
                issued_.fetch_sub(1, std::memory_order_acq_rel);
                ++waits;
                worker_block(before);
            }
        } catch (...) {
            record_error(std::current_exception());
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            --running_workers_;
            collisions_ += collisions;
            waits_ += waits;
        }
        dispatch_cv_.notify_one();
    }

    // One simulation on a worker thread.
    //
    // This is `run_simulation` with the leaf handling replaced and NOTHING else
    // changed: the same selection, the same descent-step draw checks in the same
    // order, the same depth cap, the same terminal classification, the same
    // backup. The two are deliberately not merged behind a flag — the serial one
    // is the code C5 through C8 were certified on and it is left alone, and
    // acceptance layer 1 is what checks that this one agrees with it.
    LeafOutcome run_simulation_parallel(Descent &d, LeafNode *item) {
        struct Unwind {
            ReplaySearch *self;
            Descent *d;
            ~Unwind() {
                // A submitted leaf swapped its applied list out, so this repays
                // nothing for it — ownership of the repayment left with the
                // leaf. Everything else, including an exception mid-descent,
                // repays here.
                self->repay(*d);
                self->rewind_board(*d);
            }
        } unwind{this, &d};

        d.path_counts.clear();

        std::uint32_t node = root_;
        int depth = 0;
        ParsedFen parsed = root_parsed_;
        apply_vloss(d, node);

        for (;;) {
            while (arena_.lifecycle(node) == NodeState::Expanded &&
                   arena_.children_count(node) > 0) {
                const std::uint32_t child = select_child(d, node, depth == 0);
                assert(child != kNoNode);
                const chess::Move move(raw_move_[child]);

                const int from = move.from().index();
                const int to = uci_destination(d.board->board(), move).index();
                const bool zeroing = guofish::is_zeroing(parsed, from, to);
                const bool irreversible = guofish::is_irreversible(parsed, from, to);

                d.board->make_move(move, zeroing);
                parsed = d.board->parsed();
                const std::uint64_t key = rep_key(parsed).value;

                d.path.push_back(PathStep{move.move(), child, key, parsed.placement.occupied(),
                                          irreversible});
                d.path_moves.push_back(arena_.move(child));
                node = child;
                ++depth;
                apply_vloss(d, node);

                if (depth > d.stats->max_depth) {
                    d.stats->max_depth = depth;
                }

                if (draw_by_rule(d, key)) {
                    ++d.stats->draw_by_rule_hits;
                    repay(d);
                    // May lose the claim to a thread that reached the same node
                    // and derived the same draw; see try_mark_terminal. Either
                    // way the node ends up terminal with 0.0 and this
                    // simulation backs up 0.0.
                    arena_.try_mark_terminal(node, 0.0f);
                    backpropagate(node, 0.0);
                    ++d.stats->simulations;
                    return LeafOutcome::Delivered;
                }

                if (depth >= config_.max_tree_depth) {
                    ++d.stats->depth_cap_hits;
                    repay(d);
                    backpropagate(node, 0.0);
                    ++d.stats->simulations;
                    return LeafOutcome::Delivered;
                }
            }

            if (arena_.is_terminal(node)) {
                ++d.stats->terminal_fast_path_hits;
                const double value = arena_.terminal_value(node);
                repay(d);
                backpropagate(node, value);
                maybe_mate_short_circuit(*d.stats, depth, value, node);
                ++d.stats->simulations;
                return LeafOutcome::Delivered;
            }

            // THE SELECT COLLISION, scope 2.2's one remaining race.
            if (arena_.try_claim_pending(node)) {
                break;
            }
            // Lost it. Three ways, and only one of them is a collision:
            //   * the node became terminal    -> take the fast path above;
            //   * the node became expanded    -> keep descending through it;
            //   * the node is PENDING         -> a real collision.
            if (arena_.is_terminal(node)) {
                continue;
            }
            if (arena_.lifecycle(node) == NodeState::Expanded &&
                arena_.children_count(node) > 0) {
                continue;
            }
            return LeafOutcome::Discarded;
        }

        // We hold `node` as PENDING; nothing else can expand or mark it.
        generate_canonical(*d.board, d.packed, d.raw, &d.generation);

        const TerminalReason reason =
            outcome_of(parsed, d.board->in_check(), d.packed.size(), d.board->halfmove_clock(),
                       /*fivefold_repetition=*/d.packed.empty() ? false : is_repetition(d, 5));
        if (reason != TerminalReason::None) {
            const double value = terminal_value_of(reason);
            count_terminal(*d.stats, reason);
            arena_.mark_terminal_pending(node, static_cast<float>(value));
            repay(d);
            backpropagate(node, value);
            maybe_mate_short_circuit(*d.stats, depth, value, node);
            ++d.stats->simulations;
            return LeafOutcome::Delivered;
        }

        // === Hand it to the dispatcher ===================================
        //
        // The virtual loss stays applied and the node stays PENDING. This is
        // the whole point of the chunk: the thread does not wait for the
        // evaluation, and the tree carries the fact that a simulation is in
        // flight through here until it is backed up.
        item->node = node;
        item->parsed = parsed;
        item->halfmove_clock = d.board->halfmove_clock();
        item->fullmove_number = static_cast<int>(d.board->board().fullMoveNumber());
        item->packed = d.packed;
        item->raw = d.raw;
        item->generation = d.generation;
        item->path_moves = d.path_moves;
        // MOVED, not copied: the descent must not repay what the dispatcher is
        // now responsible for, and a swap makes that structural rather than
        // remembered. `d.applied` comes back empty, so the RAII unwind above
        // repays nothing.
        item->applied.swap(d.applied);

        outstanding_.fetch_add(1, std::memory_order_acq_rel);
        // PUSH FIRST, THEN COUNT. The dispatcher pops exactly `queued_` items
        // and spins if the queue hands back nothing, so counting first would
        // open a window where it spins on a leaf that has not been linked yet.
        // The window is short and the spin yields, but it is avoidable for
        // free: after this order, a non-zero `queued_` means the items are
        // already reachable.
        queue_->push(item);
        queued_.fetch_add(1, std::memory_order_acq_rel);
        return LeafOutcome::Submitted;
    }

    void dispatcher_loop(const ParallelConfig &pc) {
        try {
            for (;;) {
                std::size_t available = 0;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    dispatch_cv_.wait(lock, [&] {
                        if (aborted_.load(std::memory_order_relaxed)) {
                            return true;
                        }
                        const int queued = queued_.load(std::memory_order_relaxed);
                        if (running_workers_ == 0) {
                            return true;
                        }
                        // No floor and no clock: whatever is there, once no
                        // search thread can add to it. See search_parallel.
                        return queued > 0 && waiting_workers_ >= running_workers_;
                    });
                    const int queued = queued_.load(std::memory_order_relaxed);
                    if (queued <= 0) {
                        if (running_workers_ == 0 || aborted_.load(std::memory_order_relaxed)) {
                            return;
                        }
                        continue;
                    }
                    if (aborted_.load(std::memory_order_relaxed)) {
                        // Leave the queue alone; release_stranded_leaves unwinds
                        // it once the threads are joined, which is the only
                        // place that can be done without racing a producer.
                        return;
                    }
                    available = static_cast<std::size_t>(queued);
                }

                const std::size_t take = available < pc.max_batch ? available : pc.max_batch;
                process_batch(take, pc);

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    drain_epoch_.fetch_add(1, std::memory_order_acq_rel);
                }
                worker_cv_.notify_all();
            }
        } catch (...) {
            record_error(std::current_exception());
        }
    }

    void process_batch(std::size_t take, const ParallelConfig &pc) {
        batch_.clear();
        batch_.reserve(take);
        for (std::size_t i = 0; i < take; ++i) {
            MpscNode *link = nullptr;
            for (;;) {
                link = queue_->try_pop();
                if (link != nullptr) {
                    break;
                }
                // A producer is between its exchange and its link store. It
                // cannot be blocked and it cannot be far away; yielding is
                // cheaper than any handshake that would prevent the window.
                std::this_thread::yield();
            }
            batch_.push_back(static_cast<LeafNode *>(link));
        }
        queued_.fetch_sub(static_cast<int>(take), std::memory_order_acq_rel);

        ++par_.batches;
        if (static_cast<std::int64_t>(take) > par_.largest_batch) {
            par_.largest_batch = static_cast<std::int64_t>(take);
        }
        if (pc.collect_histograms) {
            par_.batch_sizes.push_back(static_cast<std::int64_t>(take));
            par_.outstanding_at_drain.push_back(outstanding_.load(std::memory_order_relaxed));
        }
        if (hook_ != nullptr) {
            const std::int64_t wait = hook_->before_batch(take);
            par_.hook_wait_ns += wait;
            if (pc.collect_histograms) {
                par_.hook_wait_ns_samples.push_back(wait);
            }
        }

        // C10. ONE BOUNDARY CROSSING FOR THE WHOLE BATCH, which is the entire
        // reason the queue exists. Everything before the crossing is bookkeeping
        // that decides which leaves need a row; everything after it is the same
        // sequential expansion the replay path does.
        if (evaluator_ != nullptr) {
            prepare_live_batch();
        }

        // SEQUENTIALLY, on this thread. Expansion is single-threaded by
        // construction (scope 2.2), which is what leaves `set_children` with one
        // caller per node and the arena's bump allocator with one writer.
        for (std::size_t i = 0; i < batch_.size(); ++i) {
            expand_and_backup(*batch_[i], i);
        }
    }

    // C10. Probe the cache for every leaf in the batch, hand the misses a row of
    // the token buffer, and cross the boundary once.
    //
    // THE PROBE HAS TO HAPPEN BEFORE THE CROSSING, not inside the expansion loop
    // where the replay path does it, because the batch's width is decided here:
    // a leaf the cache can answer must not consume a row of the network's input.
    // At the reference's measured 24.7% hit rate that is a quarter of every
    // batch.
    //
    // The hit's payload is COPIED OUT rather than re-probed at expansion time.
    // It has to be: the misses in this same batch insert as they expand, and an
    // insert can evict the very slot an earlier leaf hit — the second probe
    // would then miss on a leaf that has no row to fall back on. `live_hits_`
    // costs a few dozen kilobytes that stop growing after the first batch.
    //
    // Leaves that transpose onto each other WITHIN one batch each take a row and
    // each insert. Deduplicating them would save network rows, but it would also
    // mean the first copy's insert decides what the second one expands from, and
    // the two are bit-identical anyway. Counted honestly (both are misses) and
    // left for C12 to measure rather than assumed to be worth fixing.
    void prepare_live_batch() {
        const std::size_t count = batch_.size();
        if (count > evaluator_->max_batch()) {
            throw std::logic_error(
                "guofish: the dispatcher drained " + std::to_string(count) +
                " leaves but the evaluator's buffers hold " +
                std::to_string(evaluator_->max_batch()) +
                ". ParallelConfig.max_batch must not exceed the evaluator's max_batch; "
                "search_parallel checks this before starting any thread, so reaching here "
                "means the evaluator was replaced mid-search.");
        }

        live_slot_.assign(count, kNoEvalRow);
        live_keys_.clear();
        live_keys_.reserve(count);
        if (live_hits_.size() < count) {
            live_hits_.resize(count);
        }

        std::size_t next_row = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const LeafNode &item = *batch_[i];
            const EvalRow row(item.parsed);
            live_keys_.push_back(row.key());

            if (cache_.has_value() && cache_->probe(row.key(), live_hits_[i])) {
                ++eval_stats_.cache_skipped;
                continue;
            }
            std::memcpy(evaluator_->token_row(next_row), row.tokens(),
                        sizeof(std::int32_t) * static_cast<std::size_t>(kSeqLength));
            live_slot_[i] = next_row;
            ++next_row;
        }

        if (next_row > 0) {
            note_eval_timing(evaluator_->run(next_row), next_row);
        }
    }

    void expand_and_backup(LeafNode &item, std::size_t index) {
        const LeafDiag diag{&item.parsed, item.halfmove_clock, item.fullmove_number,
                            item.path_moves.data(), item.path_moves.size()};
        NetworkValue nn_value(0.0);
        if (evaluator_ != nullptr) {
            const std::size_t slot = live_slot_[index];
            if (slot == kNoEvalRow) {
                ++dispatch_stats_.cache_hits;
                CachedEval &hit = live_hits_[index];
                expand(item.node, hit.moves.data(), hit.priors.data(), hit.moves.size(),
                       item.packed, item.raw, "the transposition cache", "cache    ", diag,
                       dispatch_stats_);
                nn_value = hit.value;
            } else {
                if (cache_.has_value()) {
                    ++dispatch_stats_.cache_misses;
                }
                nn_value = expand_from_live_row(item.node, live_keys_[index], slot, item.packed,
                                                item.raw, item.generation, diag, dispatch_stats_);
            }
        } else {
            const EvalRow row(item.parsed);
            nn_value = evaluate_and_expand(item.node, row, item.packed, item.raw, item.generation,
                                           diag, dispatch_cache_, dispatch_stats_);
        }
        const double backup_value =
            apply_tablebase(nn_value, item.parsed, item.halfmove_clock, dispatch_stats_);

        // Repay BEFORE the backup, which is the reference's ordering: backprop
        // always runs with zero in-flight loss on the path it walks — this
        // path, at least. Other threads' losses are on other paths and are
        // theirs to repay.
        repay_list(item.applied);
        item.applied.clear();
        backpropagate(item.node, backup_value);
        ++dispatch_stats_.simulations;
        ++par_.queued_leaves;

        delivered_.fetch_add(1, std::memory_order_acq_rel);
        outstanding_.fetch_sub(1, std::memory_order_acq_rel);
        item.in_use.store(false, std::memory_order_release);
    }

    // --- diagnostics -------------------------------------------------------

    static std::string hex64(std::uint64_t value) {
        static const char *digits = "0123456789abcdef";
        std::string out(16, '0');
        for (int i = 15; i >= 0; --i) {
            out[static_cast<std::size_t>(i)] = digits[value & 0xFULL];
            value >>= 4;
        }
        return out;
    }

    static std::string square_name(int square) {
        std::string out;
        out += static_cast<char>('a' + (square & 7));
        out += static_cast<char>('1' + (square >> 3));
        return out;
    }

    static std::string uci_of(std::uint16_t packed) {
        std::string out = square_name(move_from(packed)) + square_name(move_to(packed));
        const char promo = promotion_letter(move_promotion(packed));
        if (promo != '\0') {
            out += promo;
        }
        return out;
    }

    static std::string uci_list(const std::uint16_t *packed, std::size_t count) {
        std::string out;
        for (std::size_t i = 0; i < count; ++i) {
            if (i != 0) {
                out += ' ';
            }
            out += uci_of(packed[i]);
        }
        return out;
    }

    SearchConfig config_;
    NodeArena<Accumulator> arena_;
    // C8. The other half of the ping-pong pair. Null until the first
    // `apply_move`; see ensure_standby for why it is not allocated up front.
    // A unique_ptr rather than a value member because NodeArena is deliberately
    // immovable (it holds an atomic bump pointer) and because "the standby arena
    // does not exist yet" is a state worth being able to spell.
    std::unique_ptr<NodeArena<Accumulator>> standby_;

    // Parallel to the arena. Kept outside it because neither is node payload the
    // hot sibling scan reads: `parent_` is walked once per backup and
    // `raw_move_` once per descent step, so putting them in the SoA arena would
    // cost cache lines in the loop that matters without helping either.
    std::vector<std::uint32_t> parent_;
    std::vector<std::uint16_t> raw_move_;
    // C8. And their ping-pong partners. These have to be double-buffered for the
    // same reason the arena does, and it is not symmetry for its own sake: the
    // compaction READS `raw_move_[source]` while WRITING the destination's, and
    // the two index spaces overlap. Writing in place would corrupt source
    // entries that later siblings still need — a fixup bug in the one array the
    // structural diff could not see, because it is not node payload.
    std::vector<std::uint32_t> standby_parent_;
    std::vector<std::uint16_t> standby_raw_move_;

    ReplayDump dump_;
    SearchBoard board_;
    SearchStats stats_;
    ReuseStats reuse_;

    // C7. `std::optional` rather than a pointer or a zero-slot instance: the
    // three states a reader could imagine — off, on, and "on but useless" — are
    // reduced to two, because cpp/cache.hpp refuses to construct the third.
    std::optional<TranspositionCache> cache_;
    // Reused across leaves so a cache hit performs no allocation once the
    // vectors have grown to the largest move list seen.
    CachedEval cache_hit_;
    // Borrowed. nullptr is tablebases off, which is the shipping default.
    const TablebaseProber *tablebase_ = nullptr;

    // C10. The live evaluator, borrowed. nullptr is the replay build, which is
    // what every Gate 1 test runs and therefore the default.
    BatchEvaluator *evaluator_ = nullptr;
    EvalStats eval_stats_;
    // Per-batch, reused. `live_slot_[i]` is the evaluator row leaf i was given,
    // or kNoEvalRow if the cache answered it; `live_keys_[i]` is the key that
    // probe used, kept so the insert cannot re-derive it (cpp/keys.hpp's rule);
    // `live_hits_[i]` is the payload a hit copied out. See prepare_live_batch.
    std::vector<std::size_t> live_slot_;
    std::vector<NNKey> live_keys_;
    std::vector<CachedEval> live_hits_;
    // The gather's working set: `live_scratch_` holds the legal logits in
    // GENERATION order across the softmax, `live_priors_` the probabilities in
    // canonical order. One expander at a time, by construction.
    std::vector<float> live_scratch_;
    std::vector<float> live_priors_;

    // C9. The stand-in evaluator and its scratch. `synthetic_priors_` is only
    // ever touched on the thread that expands — the dispatcher in a parallel
    // search, the caller in a serial one — because expansion is single-threaded
    // by construction, which is what lets one buffer serve it.
    bool synthetic_fallback_ = false;
    SyntheticEvaluator synthetic_;
    std::vector<float> synthetic_priors_;

    std::uint32_t root_ = kNoNode;
    bool root_expanded_ = false;
    // C9. Atomic because W workers can find a mate in one at the same instant.
    // See maybe_mate_short_circuit.
    std::atomic<std::uint16_t> mating_move_{kNoMove};

    // The root position, in the three forms the descent needs it in. Computed
    // once by set_position rather than per simulation.
    ParsedFen root_parsed_;
    std::uint64_t root_rep_key_ = 0;
    Bitboard root_occupied_ = 0;

    // `build_repetition_history(board)`: rep_key -> occurrences in the game
    // BEFORE this search, including the root position itself. Read-only once
    // set_position has built it, which is what lets C9 share it across threads.
    std::unordered_map<std::uint64_t, int> rep_history_;

    // C8. The same history as a LIST, most recent first, so `apply_move` can
    // rebuild the map for a root one ply further on. The map alone cannot do it:
    // it has lost the order, and the halfmove-clock horizon is a rule about
    // order. Grows by one key per applied move — 80 keys over a game.
    std::vector<std::uint64_t> history_keys_;

    // The one descent `search()` runs, pointing at the search's own board and
    // its own counters. Scratch is reused across simulations so a
    // 5,000-simulation search performs no allocation after the first few
    // descents.
    Descent serial_;

    // --- C9 ----------------------------------------------------------------
    //
    // All of this is alive only between the first and last line of
    // `run_workers`. Nothing here is touched by `search()`, and every counter is
    // reset at the top of a parallel search rather than carried across one, so a
    // second `search_parallel` on the same tree starts from a known state.

    ParallelStats par_;
    // Borrowed. See BatchHook.
    BatchHook *hook_ = nullptr;

    std::vector<std::unique_ptr<Descent>> descents_;
    std::vector<SearchStats> worker_stats_;
    SearchStats dispatch_stats_;
    // The dispatcher's cache-hit scratch. Separate from `cache_hit_`, which
    // belongs to the serial descent, because both could otherwise be live at
    // once in a process that interleaved the two entry points.
    CachedEval dispatch_cache_;

    std::vector<std::unique_ptr<LeafNode>> slots_;
    std::unique_ptr<MpscQueue> queue_;
    // The dispatcher's working set for one drain. A member so a search performs
    // no allocation per batch once it has grown to max_batch.
    std::vector<LeafNode *> batch_;

    // Simulations claimed. See worker_loop for why this is a claim-and-return
    // protocol rather than a plain counter.
    std::atomic<int> issued_{0};
    // Simulations backed up into the root. Incremented by whichever thread
    // performed the backup — a worker for a terminal, the dispatcher for an
    // evaluated leaf — so it is the count of DELIVERED simulations and not of
    // requested ones. Python's `stats['simulations'] += 1` was an unsynchronized
    // read-modify-write over the same quantity across 32 threads; this is a
    // `fetch_add`, which is why C9 can assert an equality where the reference
    // could only report an estimate.
    std::atomic<std::int64_t> delivered_{0};
    // Leaves submitted and not yet backed up. This is the quantity scope 2.2
    // makes the batch-governing knob, and the one every in-flight virtual loss
    // belongs to.
    std::atomic<int> outstanding_{0};
    // Leaves in the queue and not yet popped. Read only by the dispatcher.
    std::atomic<int> queued_{0};
    std::atomic<std::uint64_t> drain_epoch_{0};
    std::atomic<bool> aborted_{false};
    int target_ = 0;

    std::mutex mutex_;
    std::condition_variable worker_cv_;
    std::condition_variable dispatch_cv_;
    int waiting_workers_ = 0;
    int running_workers_ = 0;
    std::int64_t collisions_ = 0;
    std::int64_t waits_ = 0;
    std::exception_ptr error_;
};

using DoubleReplaySearch = ReplaySearch<DoubleAccumulator>;
using Q32ReplaySearch = ReplaySearch<Q32Accumulator>;

}  // namespace guofish

#endif  // GUOFISH_SEARCH_HPP
