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
#include "keys.hpp"
#include "movegen.hpp"
#include "terminal.hpp"
#include "tokens.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
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
// Written by hand rather than taken from Board::fen() for two reasons. The
// prohibited round trip is the obvious one. The load-bearing one is that this
// has to print the ep square THIS FILE derived, not the library's — when a dump
// miss is being explained, "which ep square did the search think it was at" is
// the first question, and a FEN that answered with the library's state would
// hide exactly the bug the message exists to surface.
//
// C6 gives it a second job: the FEN of a terminal node is what a promotion probe
// re-loads, and there the halfmove clock is not decoration — a fifty-move draw
// node whose FEN said `0` would be a different position on the one axis that
// made it a draw.
// ---------------------------------------------------------------------------

inline std::string fen_of(const ParsedFen &parsed, int halfmove, int fullmove) {
    static const char kSymbols[] = "PNBRQKpnbrqk";

    std::string out;
    out.reserve(90);

    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            const std::int32_t token = parsed.placement.square_token[rank * 8 + file];
            if (token == kTokenEmpty) {
                ++empty;
                continue;
            }
            if (empty != 0) {
                out += static_cast<char>('0' + empty);
                empty = 0;
            }
            out += kSymbols[token - 1];
        }
        if (empty != 0) {
            out += static_cast<char>('0' + empty);
        }
        if (rank != 0) {
            out += '/';
        }
    }

    out += parsed.white_to_move ? " w " : " b ";

    const Placement &placement = parsed.placement;
    std::string castling;
    if (has_castling_right(parsed.castling, placement, true, true)) castling += 'K';
    if (has_castling_right(parsed.castling, placement, true, false)) castling += 'Q';
    if (has_castling_right(parsed.castling, placement, false, true)) castling += 'k';
    if (has_castling_right(parsed.castling, placement, false, false)) castling += 'q';
    out += castling.empty() ? "-" : castling;

    out += ' ';
    if (parsed.ep_square < 0) {
        out += '-';
    } else {
        out += static_cast<char>('a' + (parsed.ep_square & 7));
        out += static_cast<char>('1' + (parsed.ep_square >> 3));
    }

    out += ' ';
    out += std::to_string(halfmove);
    out += ' ';
    out += std::to_string(fullmove);
    return out;
}

inline std::string SearchBoard::diagnostic_fen() const {
    return fen_of(parsed_from_board(), halfmove_clock_,
                  static_cast<int>(board_.fullMoveNumber()));
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
    }

    ReplayDump &dump() noexcept { return dump_; }
    const ReplayDump &dump() const noexcept { return dump_; }
    const SearchConfig &config() const noexcept { return config_; }
    const SearchStats &stats() const noexcept { return stats_; }
    const NodeArena<Accumulator> &arena() const noexcept { return arena_; }
    std::uint32_t root() const noexcept { return root_; }

    // The move the depth-1 hack seized on, packed; kNoMove when it did not fire.
    std::uint16_t mating_move() const noexcept { return mating_move_; }

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
        mating_move_ = kNoMove;

        root_parsed_ = board_.parsed();
        root_rep_key_ = rep_key(root_parsed_).value;
        root_occupied_ = root_parsed_.placement.occupied();

        rep_history_.clear();
        rep_history_[root_rep_key_] = 1;
        for (const std::string &position : history) {
            rep_history_[rep_key(parse_fen(position)).value] += 1;
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
        if (dump_.empty()) {
            throw std::logic_error("guofish::ReplaySearch::search: no replay dump loaded");
        }
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
            if (mating_move_ != kNoMove) {
                break;
            }
            run_simulation();
        }
        return stats_;
    }

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
        if (mating_move_ != kNoMove) {
            return mating_move_;
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

    void apply_vloss(std::uint32_t node) {
        arena_.add_vloss(node, 1);
        applied_.push_back(node);
    }

    void repay() {
        while (!applied_.empty()) {
            arena_.add_vloss(applied_.back(), -1);
            applied_.pop_back();
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
    std::uint32_t select_child(std::uint32_t parent, bool at_root) {
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

        ++stats_.select_steps;
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
    void generate_canonical(std::vector<std::uint16_t> &packed,
                            std::vector<std::uint16_t> &raw) const {
        chess::Movelist movelist;
        chess::movegen::legalmoves(movelist, board_.board());

        // (canonical key, packed move, raw library move). Sorting a small vector
        // of triples rather than the Movelist itself keeps chess-library's
        // generation order out of the answer entirely.
        std::vector<std::array<std::uint16_t, 3>> entries;
        entries.reserve(static_cast<std::size_t>(movelist.size()));
        for (const auto &move : movelist) {
            const std::uint16_t p = packed_of(board_.board(), move);
            entries.push_back({canonical_move_key(p), p, move.move()});
        }
        std::sort(entries.begin(), entries.end(),
                  [](const std::array<std::uint16_t, 3> &a,
                     const std::array<std::uint16_t, 3> &b) { return a[0] < b[0]; });

        packed.clear();
        raw.clear();
        packed.reserve(entries.size());
        raw.reserve(entries.size());
        for (const auto &entry : entries) {
            packed.push_back(entry[1]);
            raw.push_back(entry[2]);
        }
    }

    // Look the current position up in the dump, or fail by name.
    //
    // THE MISS PATH IS A TEST, which is why it carries this much context. A
    // board-path tokenization that derives the wrong ep square produces a key
    // the Python-generated dump does not contain, and the FEN printed here shows
    // the ep square the search believed it was at.
    const ReplayDump::Entry &lookup(const ParsedFen &parsed, bool at_root) {
        const NNKey key = nn_key(parsed);
        const ReplayDump::Entry *entry = dump_.find(key, at_root);
        if (entry != nullptr) {
            return *entry;
        }
        throw ReplayMiss(
            "guofish: replay dump miss (" + std::string(at_root ? "root" : "interior") +
            " table)\n  nn_key : 0x" + hex64(key.value) +
            "\n  fen    : " + board_.diagnostic_fen() +
            "\n  raw ep : " + (board_.raw_ep() < 0 ? std::string("-") : square_name(board_.raw_ep())) +
            "\n  path   : " + path_from_root() +
            "\n  The dump is generated from the Python reference, so a miss means "
            "this search reached a position the reference never evaluated, or "
            "tokenized a position it did evaluate differently.");
    }

    // Publish `node`'s children from a dump entry, given the move list the
    // caller has already generated.
    //
    // C5 generated the moves inside here. C6 hoists that out, because the
    // terminal test needs the same list first — `outcome_of` asks how many legal
    // moves there are, and generating them twice per leaf would be both slower
    // and a second place for the two answers to disagree.
    void expand(std::uint32_t node, const ReplayDump::Entry &entry,
                const std::vector<std::uint16_t> &packed, const std::vector<std::uint16_t> &raw) {
        assert(!packed.empty());

        // The dump's move list is Python's, generated by python-chess and
        // written in canonical order. Comparing rather than trusting positional
        // alignment is what makes a movegen divergence a named failure instead
        // of 37 priors landing on the wrong 37 moves.
        if (packed.size() != entry.count ||
            !std::equal(packed.begin(), packed.end(), entry.moves)) {
            throw ReplayMiss(
                "guofish: legal-move mismatch against the replay dump at " +
                board_.diagnostic_fen() + "\n  path     : " + path_from_root() +
                "\n  C++      : " + uci_list(packed.data(), packed.size()) +
                "\n  golden   : " + uci_list(entry.moves, entry.count));
        }

        const std::uint32_t offset = arena_.allocate(packed.size());
        for (std::size_t k = 0; k < packed.size(); ++k) {
            const std::uint32_t child = offset + static_cast<std::uint32_t>(k);
            arena_.set_move(child, packed[k]);
            arena_.set_prior(child, entry.priors[k]);
            parent_[child] = node;
            raw_move_[child] = raw[k];
        }
        arena_.set_children(node, offset, static_cast<std::uint16_t>(packed.size()));
        ++stats_.expansions;
    }

    // ParallelMCTS._expand_root: expand, seed one visit, seed the value.
    //
    // The reference assigns rather than accumulates (`root.visit_count = 1`,
    // `root.value_sum = ...`), which is the same thing on a node the arena has
    // just cleared.
    void expand_root() {
        std::vector<std::uint16_t> packed;
        std::vector<std::uint16_t> raw;
        generate_canonical(packed, raw);
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
        const ReplayDump::Entry &entry = lookup(root_parsed_, /*at_root=*/true);
        expand(root_, entry, packed, raw);
        arena_.add_visits(root_, 1);
        arena_.add_value(root_, mover_value(entry.value, root_parsed_.white_to_move));
        root_expanded_ = true;
    }

    // The v5 value head is White-POV by construction, and every node's Q is from
    // the perspective of whoever moved TO it — the opponent of the side now to
    // move. So the value is negated exactly when White is to move.
    static double mover_value(double absolute, bool white_to_move) noexcept {
        return white_to_move ? -absolute : absolute;
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
    bool draw_by_rule(std::uint64_t key) {
        if (board_.halfmove_clock() >= 100) {
            ++stats_.fifty_move_hits;
            return true;
        }
        int seen = 1;
        for (auto &entry : path_counts_) {
            if (entry.first == key) {
                seen = entry.second + 1;
                entry.second = seen;
                break;
            }
        }
        if (seen == 1) {
            path_counts_.emplace_back(key, 1);
        }
        const auto it = rep_history_.find(key);
        const int history = (it == rep_history_.end()) ? 0 : it->second;
        if (history + seen >= 3) {
            ++stats_.threefold_hits;
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
    bool is_repetition(int count) const {
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
    void mark_terminal(std::uint32_t node, TerminalReason reason, double value) {
        switch (reason) {
            case TerminalReason::Checkmate: ++stats_.checkmates; break;
            case TerminalReason::Stalemate: ++stats_.stalemates; break;
            case TerminalReason::InsufficientMaterial: ++stats_.insufficient_material; break;
            case TerminalReason::SeventyFiveMoves: ++stats_.seventyfive_moves; break;
            case TerminalReason::FivefoldRepetition: ++stats_.fivefold_repetitions; break;
            case TerminalReason::None: break;
        }
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
    void maybe_mate_short_circuit(int depth, double value, std::uint32_t node) {
        if (depth == 1 && value == 1.0 && arena_.move(node) != kNoMove) {
            mating_move_ = arena_.move(node);
            ++stats_.mate_short_circuits;
        }
    }

    // --- one simulation ----------------------------------------------------

    void run_simulation() {
        // Restores the board and repays every applied virtual loss on ANY exit,
        // including an exception mid-descent. The reference does the same with a
        // `finally`; here it is a destructor, which is why C8 can delete the
        // defensive full-tree vloss reset entirely.
        struct Unwind {
            ReplaySearch *self;
            ~Unwind() {
                self->repay();
                while (!self->path_.empty()) {
                    self->board_.unmake_move(chess::Move(self->path_.back().raw_move));
                    self->path_.pop_back();
                }
            }
        } unwind{this};

        path_counts_.clear();

        std::uint32_t node = root_;
        int depth = 0;
        // The position the descent currently stands on, in python-chess's terms.
        // Threaded rather than rebuilt: each step needs the position BEFORE the
        // move (for is_zeroing / is_irreversible) and the position AFTER it (for
        // rep_key, and at the leaf for nn_key), and the after of one step is the
        // before of the next. One ParsedFen build per descent step, not three.
        ParsedFen parsed = root_parsed_;
        apply_vloss(node);

        while (arena_.lifecycle(node) == NodeState::Expanded && arena_.children_count(node) > 0) {
            const std::uint32_t child = select_child(node, depth == 0);
            assert(child != kNoNode);
            const chess::Move move(raw_move_[child]);

            // python-chess's squares for this move: the NORMALISED destination,
            // never chess-library's king-takes-rook encoding. Both classifiers
            // below test the destination against bitboards the reference indexes
            // with g1/c1, and `_reduces_castling_rights` compares it against the
            // castling-rights ROOK squares, where the two encodings differ.
            const int from = move.from().index();
            const int to = uci_destination(board_.board(), move).index();
            const bool zeroing = guofish::is_zeroing(parsed, from, to);
            const bool irreversible = guofish::is_irreversible(parsed, from, to);

            board_.make_move(move, zeroing);
            parsed = board_.parsed();
            const std::uint64_t key = rep_key(parsed).value;

            path_.push_back(PathStep{move.move(), child, key, parsed.placement.occupied(),
                                     irreversible});
            node = child;
            ++depth;
            apply_vloss(node);

            if (depth > stats_.max_depth) {
                stats_.max_depth = depth;
            }

            // === Draw by rule: fifty-move or threefold repetition ===
            //
            // Checked on EVERY descent step, not only at the leaf, because the
            // reference does — with tree reuse an interior node can have become
            // a draw since it was expanded. The node is marked terminal and left
            // UNEXPANDED, which is what makes it recoverable if a host declines
            // the claim and hands it back as the position to move from.
            if (draw_by_rule(key)) {
                ++stats_.draw_by_rule_hits;
                repay();
                arena_.mark_terminal(node, 0.0f);
                backpropagate(node, 0.0);
                ++stats_.simulations;
                return;
            }

            // The depth cap. Backs up 0.0 WITHOUT marking the node terminal,
            // exactly as the reference does — a capped node is not a game
            // result, and marking it would make it unrepresentable as a future
            // search root.
            if (depth >= config_.max_tree_depth) {
                ++stats_.depth_cap_hits;
                repay();
                backpropagate(node, 0.0);
                ++stats_.simulations;
                return;
            }
        }

        // === Cached terminal: a node an earlier simulation already resolved ===
        if (arena_.is_terminal(node)) {
            ++stats_.terminal_fast_path_hits;
            const double value = arena_.terminal_value(node);
            repay();
            backpropagate(node, value);
            maybe_mate_short_circuit(depth, value, node);
            ++stats_.simulations;
            return;
        }

        // === Intrinsic terminal: first visit ===
        //
        // The legal moves are generated once and used twice — `outcome_of` needs
        // the count, `expand` needs the list. Generating them separately would
        // be a second opportunity for the two to disagree about the position.
        std::vector<std::uint16_t> packed;
        std::vector<std::uint16_t> raw;
        generate_canonical(packed, raw);

        const TerminalReason reason =
            outcome_of(parsed, board_.in_check(), packed.size(), board_.halfmove_clock(),
                       /*fivefold_repetition=*/packed.empty() ? false : is_repetition(5));
        if (reason != TerminalReason::None) {
            const double value = terminal_value_of(reason);
            mark_terminal(node, reason, value);
            repay();
            backpropagate(node, value);
            maybe_mate_short_circuit(depth, value, node);
            ++stats_.simulations;
            return;
        }

        const ReplayDump::Entry &entry = lookup(parsed, /*at_root=*/false);
        expand(node, entry, packed, raw);

        // Repay BEFORE the backup, which is the reference's ordering: backprop
        // always runs with zero in-flight loss on the path it walks.
        repay();
        backpropagate(node, mover_value(entry.value, parsed.white_to_move));
        ++stats_.simulations;
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

    // The move path from the root to wherever the descent currently is, in
    // normalised UCI.
    //
    // Read off the ARENA's stored move rather than re-derived from the raw
    // library move: the arena already holds the normalised packing (castling as
    // e1g1, not e1h1), and re-deriving it would need the board as it stood
    // BEFORE the castle, which the descent has already moved past.
    std::string path_from_root() const {
        if (path_.empty()) {
            return "(root)";
        }
        std::string out;
        for (std::size_t i = 0; i < path_.size(); ++i) {
            if (i != 0) {
                out += ' ';
            }
            out += uci_of(arena_.move(path_[i].node));
        }
        return out;
    }

    SearchConfig config_;
    NodeArena<Accumulator> arena_;
    // Parallel to the arena. Kept outside it because neither is node payload the
    // hot sibling scan reads: `parent_` is walked once per backup and
    // `raw_move_` once per descent step, so putting them in the SoA arena would
    // cost cache lines in the loop that matters without helping either.
    std::vector<std::uint32_t> parent_;
    std::vector<std::uint16_t> raw_move_;

    ReplayDump dump_;
    SearchBoard board_;
    SearchStats stats_;

    std::uint32_t root_ = kNoNode;
    bool root_expanded_ = false;
    std::uint16_t mating_move_ = kNoMove;

    // The root position, in the three forms the descent needs it in. Computed
    // once by set_position rather than per simulation.
    ParsedFen root_parsed_;
    std::uint64_t root_rep_key_ = 0;
    Bitboard root_occupied_ = 0;

    // `build_repetition_history(board)`: rep_key -> occurrences in the game
    // BEFORE this search, including the root position itself. Read-only once
    // set_position has built it, which is what lets C9 share it across threads.
    std::unordered_map<std::uint64_t, int> rep_history_;

    // One step of the current descent. The raw library move is what unwinds the
    // board; the arena index is what names the move in a diagnostic; the key,
    // the occupancy and the irreversibility flag are what the two repetition
    // rules read. Kept together so they can never fall out of step.
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

    // Scratch, reused across simulations so a 5,000-simulation search performs
    // no allocation after the first few descents.
    std::vector<std::uint32_t> applied_;
    std::vector<PathStep> path_;
    // `path_counts` in the reference: rep_key -> occurrences on THIS path. A
    // flat vector rather than a hash map — the path is at most MAX_TREE_DEPTH
    // long, the scan is linear over a contiguous 12-byte record, and clearing a
    // vector between simulations does not touch the allocator.
    std::vector<std::pair<std::uint64_t, int>> path_counts_;
};

using DoubleReplaySearch = ReplaySearch<DoubleAccumulator>;
using Q32ReplaySearch = ReplaySearch<Q32Accumulator>;

}  // namespace guofish

#endif  // GUOFISH_SEARCH_HPP
