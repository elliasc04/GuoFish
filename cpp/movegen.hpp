// GuoFish C1 — legal move generation, normalised to standard UCI.
//
// This is the parity layer between Disservin/chess-library and python-chess.
// The MCTS tree in C4/C5 masks network priors by move index, so a single
// dropped, extra or differently-spelled move silently corrupts the search
// rather than crashing it. Everything here exists to make that class of bug
// impossible, and every normalisation below is a place where the library's
// internal representation and the UCI wire format genuinely disagree.
//
// Nothing in this header is on the hot path. C1 returns std::string because it
// is judged against a Python list of strings; the production search will carry
// the packed 16-bit chess::Move and only format on the way out to UCI.

#ifndef GUOFISH_MOVEGEN_HPP
#define GUOFISH_MOVEGEN_HPP

#include <chess.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace guofish {

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

// Append a square as two ASCII characters, file then rank ("e1").
//
// Written out rather than using chess-library's `operator std::string()` so
// that formatting a move costs one std::string and not five.
inline void append_square(std::string &out, chess::Square sq) {
    const int idx = sq.index();
    assert(idx >= 0 && idx < 64);
    out += static_cast<char>('a' + (idx & 7));
    out += static_cast<char>('1' + (idx >> 3));
}

// The UCI promotion suffix. Lower case, as UCI requires, regardless of colour.
inline char promotion_char(chess::PieceType pt) {
    switch (pt.internal()) {
        case chess::PieceType::underlying::QUEEN:
            return 'q';
        case chess::PieceType::underlying::ROOK:
            return 'r';
        case chess::PieceType::underlying::BISHOP:
            return 'b';
        case chess::PieceType::underlying::KNIGHT:
            return 'n';
        default:
            break;
    }
    // Unreachable for a move chess-library flagged as PROMOTION: promotionType()
    // masks two bits and offsets from KNIGHT, so it can only ever return one of
    // the four above. Kept as a throw rather than an assert so that a future
    // change to that encoding fails loudly in a release build too.
    throw std::logic_error("guofish: promotion move carries a non-promotable piece type");
}

// The square the *king* ends up on, which is what standard UCI names.
//
// THE CASTLING TRAP. chess-library encodes castling as king-takes-rook: for a
// White kingside castle, from() is e1 and to() is h1 — the rook's square, not
// the king's destination. That is the UCI_Chess960 convention, and it is what
// the library's own makeMove() expects back, but it is *not* what a standard
// UCI GUI or python-chess emits. python-chess's Board.uci() rewrites it to
// e1g1; so does this.
//
// The failure mode if it is skipped is quiet: "e1h1" is a well-formed UCI
// string that simply never appears in any reference move list, so it reads as
// one missing move plus one unknown move rather than as a formatting bug.
//
// Only two squares are ever produced, both on the king's own rank: file G when
// the rook is to the king's right, file C when it is to the left. The
// left/right test is on the rook square, not on a hard-coded e1/h1, so it is
// still correct for the shuffled back ranks of Chess960 — see below.
//
// When the board *is* a Chess960 board the king-takes-rook encoding is already
// the correct UCI (that is what UCI_Chess960 means), so it is passed through
// untouched. This mirrors python-chess's Board.uci(), which normalises only
// when self.chess960 is false. Every FEN in golden/movegen.jsonl is standard,
// so this branch is not exercised by the parity test; it is here so that the
// helper is not silently wrong the first time a 960 board reaches it.
inline chess::Square uci_destination(const chess::Board &board, chess::Move move) {
    if (move.typeOf() != chess::Move::CASTLING) {
        return move.to();
    }

    // chess-library guarantees these for a CASTLING move; assert rather than
    // branch, since a violation means the library changed under us.
    assert(board.at<chess::PieceType>(move.from()) == chess::PieceType::KING);
    assert(board.at<chess::PieceType>(move.to()) == chess::PieceType::ROOK);

    if (board.chess960()) {
        return move.to();
    }

    const bool king_side = move.to() > move.from();
    return chess::Square(king_side ? chess::File::FILE_G : chess::File::FILE_C, move.from().rank());
}

// One move as a standard UCI string: four characters, plus a promotion suffix
// on the fifth. The board is needed for the castling normalisation above.
inline std::string move_to_uci(const chess::Board &board, chess::Move move) {
    std::string out;
    out.reserve(5);

    append_square(out, move.from());
    append_square(out, uci_destination(board, move));

    if (move.typeOf() == chess::Move::PROMOTION) {
        out += promotion_char(move.promotionType());
    }

    return out;
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

// Every legal move in `board`, as UCI strings in canonical order.
//
// CANONICAL ORDER is byte-wise lexicographic order on the UCI string. Because
// every string is [file][rank][file][rank][promo?] over a fixed alphabet, that
// is exactly the tuple order (from_file, from_rank, to_file, to_rank, promo)
// the brief asks for, and it is the order the golden file is written in.
//
// It is deliberately *not* sorting by chess-library's square index. Index order
// is rank-major (a1, b1, ... h1, a2, ...) while UCI string order is file-major
// (a1, a2, ... a8, b1, ...), so the two disagree on nearly every position. The
// search's PUCT tie-break resolves by child order, so picking the wrong one
// would not fail loudly — it would just make C++ and Python explore different
// moves at equal priors.
//
// A 4-character move and a 5-character move can never share a from/to pair: any
// pawn arriving on the back rank must promote, so the prefix case in string
// comparison never arises between two real moves.
inline std::vector<std::string> legal_moves_uci(const chess::Board &board) {
    chess::Movelist movelist;
    chess::movegen::legalmoves(movelist, board);

    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(movelist.size()));
    for (const auto &move : movelist) {
        out.push_back(move_to_uci(board, move));
    }

    std::sort(out.begin(), out.end());

    // A duplicate means two distinct internal moves formatted to the same UCI
    // string, which is how a botched castling normalisation would show up if it
    // ever collided with a real king move.
    assert(std::adjacent_find(out.begin(), out.end()) == out.end());

    return out;
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

// Count the kings in a FEN's piece-placement field, without a Board.
//
// Text scanning rather than bitboards on purpose: this has to answer *before*
// the FEN is loaded. See set_fen_or_throw().
//
// The leading-space trim and the "up to the first space" split mirror what
// chess-library's own parser does, so this reads the same field it will.
inline void count_kings(std::string_view fen, int &white, int &black) {
    while (!fen.empty() && fen.front() == ' ') {
        fen.remove_prefix(1);
    }
    const std::string_view placement = fen.substr(0, fen.find(' '));

    white = 0;
    black = 0;
    for (char c : placement) {
        if (c == 'K') {
            ++white;
        } else if (c == 'k') {
            ++black;
        }
    }
}

// Load `fen` into `board`, throwing std::invalid_argument if it is not usable.
//
// The king count is checked FIRST, before setFen() is ever called, and that
// order is load-bearing rather than stylistic. setFen() does not merely accept
// a kingless position: on its way out it builds the castling paths, and that
// loop calls kingSq() for both colours unconditionally — before it consults the
// castling rights. kingSq() is `assert(pieces(KING, color) != 0ull)` followed
// by lsb() on the king bitboard, so on a board with no king it is an assertion
// failure in a debug build and an out-of-range square in a release one. Neither
// is something a caller's typo should be able to reach from Python.
//
// Validating after setFen() returned, which is the obvious way to write this,
// is therefore too late by one function call. It passed the release build and
// aborted the moment the suite ran against the ASan/debug module.
//
// The remaining rejection is setFen() itself returning false: a structurally
// broken FEN (bad piece letter, two pieces on one square, a side-to-move that
// is not w/b, a malformed ep square).
inline void set_fen_or_throw(chess::Board &board, std::string_view fen) {
    int white_kings = 0;
    int black_kings = 0;
    count_kings(fen, white_kings, black_kings);

    if (white_kings != 1 || black_kings != 1) {
        throw std::invalid_argument("guofish: FEN must place exactly one king per side (found " +
                                    std::to_string(white_kings) + " white, " + std::to_string(black_kings) +
                                    " black): " + std::string(fen));
    }

    if (!board.setFen(fen)) {
        throw std::invalid_argument("guofish: could not parse FEN: " + std::string(fen));
    }
}

// The C1 entry point: FEN in, canonically ordered UCI move list out.
inline std::vector<std::string> legal_moves(std::string_view fen) {
    chess::Board board;
    set_fen_or_throw(board, fen);
    return legal_moves_uci(board);
}

}  // namespace guofish

#endif  // GUOFISH_MOVEGEN_HPP
