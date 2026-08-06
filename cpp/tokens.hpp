// GuoFish C2 — the 68-token board encoding the v5 network was trained on.
//
// This is a parity layer, not a design. Every rule below is a transcription of
// `board_to_tokens()` in core/mctsv4.py running against python-chess 1.11.2,
// and the only correctness criterion is that the two agree on every FEN. The
// network never sees a FEN; it sees these 68 integers. A single wrong value is
// an out-of-distribution input, and the network answers it confidently with a
// wrong evaluation and wrong priors rather than failing.
//
//
// WHY THIS DOES NOT USE chess::Board
// ----------------------------------
// The obvious implementation — `board.setFen(fen)` and then read the pieces,
// side, castling rights and ep square off the Board — is WRONG, and wrong in a
// way that is invisible until a network eval is compared against Python.
//
// chess-library validates the en-passant square on the way in. From
// setFenCommon() in chess.hpp:
//
//     if (ep_sq_ != Square::NO_SQ) {
//         valid = movegen::isEpSquareValid<...>(*this, ep_sq_);
//         if (!valid) ep_sq_ = Square::NO_SQ;
//     }
//
// That is a reasonable thing for a *search* library to do — an ep square that
// no pawn can act on does not change the position — but it is not the FEN
// field, and it is not what python-chess's `board.ep_square` reports.
// python-chess stores the ep field verbatim and offers the legality question
// separately as `has_legal_en_passant()`.
//
// The gap is not hypothetical or rare. Of the 3,610 positions in
// golden/movegen.jsonl that carry an ep target, 3,203 have no legal ep capture
// available. chess-library would report NO_SQ for all 3,203 and this tokenizer
// would emit 31 where Python emits 32..39 — 3.2% of the corpus silently
// mis-encoded, on exactly the positions (a pawn just moved two squares) that
// occur constantly in real games.
//
// So the FEN is parsed here directly. That also removes setFen()'s zobrist
// hashing, castling-path construction and movegen from a hot path that needs
// none of them, which is most of why this runs as fast as it does.
//
//
// SQUARE NUMBERING is python-chess's: A1=0, B1=1, ... H8=63, i.e.
// `square = rank * 8 + file`. Token index i holds the piece on square i.
// FEN piece placement is written from rank 8 down, so the scan starts at 56.

#ifndef GUOFISH_TOKENS_HPP
#define GUOFISH_TOKENS_HPP

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace guofish {

// ---------------------------------------------------------------------------
// The contract. These values are frozen by the v5 training shards; they are not
// ours to renumber. Mirrors the constants block in core/mctsv4.py.
// ---------------------------------------------------------------------------

inline constexpr int kSeqLength = 68;

inline constexpr int kIdxSideToMove = 64;
inline constexpr int kIdxCastling = 65;
inline constexpr int kIdxEnPassant = 66;
inline constexpr int kIdxCls = 67;

inline constexpr std::int32_t kTokenEmpty = 0;
inline constexpr std::int32_t kTokenBlackOffset = 6;  // black piece = piece_type + 6
inline constexpr std::int32_t kTokenWhiteToMove = 13;
inline constexpr std::int32_t kTokenBlackToMove = 14;
inline constexpr std::int32_t kTokenCastlingBase = 15;
inline constexpr std::int32_t kTokenEpNone = 31;
inline constexpr std::int32_t kTokenEpBase = 32;
inline constexpr std::int32_t kTokenCls = 40;

// The 4-bit castling mask, in python-chess's order. Bit 3 is White kingside,
// bit 0 is Black queenside; getting this backwards produces a token in the
// legal 15..30 range for the *mirrored* rights, so nothing downstream would
// reject it.
inline constexpr int kCastleWhiteKing = 8;
inline constexpr int kCastleWhiteQueen = 4;
inline constexpr int kCastleBlackKing = 2;
inline constexpr int kCastleBlackQueen = 1;

// ---------------------------------------------------------------------------
// Bitboard helpers
//
// The castling rules below are a transliteration of python-chess, which works
// in 64-bit square masks and compares them as integers (`rook > king_mask`).
// Reproducing that shape exactly is deliberate: rewriting it in terms of square
// indices would be equivalent only under assumptions (one king, king on the
// back rank) that the source code does not itself make.
// ---------------------------------------------------------------------------

using Bitboard = std::uint64_t;

inline constexpr Bitboard kRank1 = 0x0000'0000'0000'00FFULL;
inline constexpr Bitboard kRank8 = 0xFF00'0000'0000'0000ULL;
inline constexpr Bitboard kFileA = 0x0101'0101'0101'0101ULL;
inline constexpr Bitboard kFileH = 0x8080'8080'8080'8080ULL;
inline constexpr Bitboard kSquareE1 = 1ULL << 4;
inline constexpr Bitboard kSquareE8 = 1ULL << 60;

inline constexpr Bitboard square_bb(int square) { return 1ULL << square; }

// Lowest set bit as a mask, or 0 for an empty board. python-chess spells this
// `bb & -bb`; on an unsigned type the negation is well defined and wraps, which
// is what makes the two's-complement trick legal here rather than UB.
inline constexpr Bitboard lsb_mask(Bitboard bb) { return bb & (~bb + 1ULL); }

// Index of the highest set bit, or -1 when empty — python-chess's `msb()`
// semantics, including the -1, which its castling parser relies on: `msb(0)`
// returning -1 is what makes the "no rook on the back rank" case fall into the
// else branch instead of indexing a bitboard.
//
// Written as a portable loop rather than _BitScanReverse64 / __builtin_clzll:
// Global Rule 8 forbids MSVC-only intrinsics without a fallback, and this is
// called at most four times per position, off the per-square hot path.
inline int msb_index(Bitboard bb) {
    if (bb == 0) {
        return -1;
    }
    int index = 63;
    while ((bb & square_bb(index)) == 0) {
        --index;
    }
    return index;
}

// python-chess's `lsb()`: index of the lowest set bit, -1 when empty. Its
// implementation is `(bb & -bb).bit_length() - 1`, and `(0).bit_length()` is 0,
// so the empty board really does answer -1 rather than raising.
inline int lsb_index(Bitboard bb) {
    if (bb == 0) {
        return -1;
    }
    int index = 0;
    while ((bb & square_bb(index)) == 0) {
        ++index;
    }
    return index;
}

// ---------------------------------------------------------------------------
// Parsed piece placement
// ---------------------------------------------------------------------------

// What one pass over the FEN's placement field produced: the 64 square tokens,
// plus the bitboards the castling rules need.
//
// `promoted` tracks the X-FEN "~" suffix. It exists for one reason: python-chess
// excludes promoted kings from every castling test (`self.kings & ~self.promoted`),
// so a board carrying `K~` has no castling rights at all. No FEN in the corpus
// uses the suffix, but parsing it as an ordinary character would place a piece
// on the wrong square for the entire rest of the rank, so it has to be handled
// either way — and once it is handled, handling it *correctly* is free.
struct Placement {
    std::int32_t square_token[64] = {};
    Bitboard occupied_white = 0;
    Bitboard occupied_black = 0;
    Bitboard rooks = 0;
    Bitboard kings = 0;
    Bitboard promoted = 0;

    Bitboard occupied_of(bool white) const { return white ? occupied_white : occupied_black; }
};

// python-chess's PIECE_SYMBOLS index: p=1, n=2, b=3, r=4, q=5, k=6. Returns 0
// for anything that is not a piece letter.
inline std::int32_t piece_type_of(char lower) {
    switch (lower) {
        case 'p':
            return 1;
        case 'n':
            return 2;
        case 'b':
            return 3;
        case 'r':
            return 4;
        case 'q':
            return 5;
        case 'k':
            return 6;
        default:
            return 0;
    }
}

[[noreturn]] inline void reject(std::string_view reason, std::string_view fen) {
    throw std::invalid_argument("guofish: " + std::string(reason) + ": " + std::string(fen));
}

// ---------------------------------------------------------------------------
// Field splitting
//
// python-chess splits on arbitrary whitespace runs (`fen.split()`) and supplies
// defaults for every field after the placement, so "8/8/8/8/8/8/8/8" alone is a
// legal FEN meaning "White to move, no rights, no ep". This mirrors that. It is
// not laxness for its own sake: `tokens()` and `legal_moves()` are fed from the
// same call sites, and two different ideas of what parses would be a bug that
// only shows up as a raised exception in one of them.
// ---------------------------------------------------------------------------

// Consume one whitespace-delimited field, advancing `rest` past it. Returns an
// empty view once the input is exhausted.
inline std::string_view next_field(std::string_view &rest) {
    const auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; };

    std::size_t begin = 0;
    while (begin < rest.size() && is_space(rest[begin])) {
        ++begin;
    }
    std::size_t end = begin;
    while (end < rest.size() && !is_space(rest[end])) {
        ++end;
    }

    const std::string_view field = rest.substr(begin, end - begin);
    rest.remove_prefix(end);
    return field;
}

// ---------------------------------------------------------------------------
// Placement field
// ---------------------------------------------------------------------------

// Validation mirrors python-chess's `_set_board_fen`: exactly 8 ranks, each
// summing to exactly 8 files, no two adjacent digits, "~" only after a piece.
// Rejecting the same inputs matters because a FEN that python-chess refuses has
// no golden tokens to be compared against — the reference has no answer, so
// neither may we invent one.
inline Placement parse_placement(std::string_view placement, std::string_view fen) {
    Placement out;

    int square = 56;   // a8; each '/' steps back a rank
    int rank_files = 0;
    int ranks_seen = 1;
    bool previous_was_digit = false;
    bool previous_was_piece = false;

    for (char c : placement) {
        if (c >= '1' && c <= '8') {
            if (previous_was_digit) {
                reject("two subsequent digits in the piece placement field", fen);
            }
            const int skip = c - '0';
            rank_files += skip;
            // Checked before `square` is advanced, not after: an over-long rank
            // is what would otherwise walk the write cursor off the end of the
            // 64-square array on the next piece letter.
            if (rank_files > 8) {
                reject("a rank in the piece placement field holds more than 8 files", fen);
            }
            square += skip;
            previous_was_digit = true;
            previous_was_piece = false;
            continue;
        }

        if (c == '/') {
            if (rank_files != 8) {
                reject("a rank in the piece placement field does not hold 8 files", fen);
            }
            if (ranks_seen == 8) {
                reject("more than 8 ranks in the piece placement field", fen);
            }
            ++ranks_seen;
            rank_files = 0;
            square -= 16;  // back to the start of the rank, then down one
            previous_was_digit = false;
            previous_was_piece = false;
            continue;
        }

        if (c == '~') {
            if (!previous_was_piece) {
                reject("'~' not after a piece in the piece placement field", fen);
            }
            // The piece it marks was placed on the previous square.
            assert(square >= 1);
            out.promoted |= square_bb(square - 1);
            previous_was_digit = false;
            previous_was_piece = false;
            continue;
        }

        const bool white = (c >= 'A' && c <= 'Z');
        const char lower = white ? static_cast<char>(c - 'A' + 'a') : c;
        const std::int32_t piece_type = piece_type_of(lower);
        if (piece_type == 0) {
            reject("invalid character in the piece placement field", fen);
        }

        ++rank_files;
        if (rank_files > 8) {
            reject("a rank in the piece placement field holds more than 8 files", fen);
        }
        // Guaranteed by the rank_files check above plus the '/' handling: within
        // a rank the cursor never advances past its 8th file, and there are only
        // ever 8 ranks.
        assert(square >= 0 && square < 64);

        out.square_token[square] = white ? piece_type : piece_type + kTokenBlackOffset;
        const Bitboard bit = square_bb(square);
        if (white) {
            out.occupied_white |= bit;
        } else {
            out.occupied_black |= bit;
        }
        if (piece_type == 4) {
            out.rooks |= bit;
        } else if (piece_type == 6) {
            out.kings |= bit;
        }

        ++square;
        previous_was_digit = false;
        previous_was_piece = true;
    }

    if (ranks_seen != 8) {
        reject("expected 8 ranks in the piece placement field", fen);
    }
    if (rank_files != 8) {
        reject("a rank in the piece placement field does not hold 8 files", fen);
    }

    return out;
}

// ---------------------------------------------------------------------------
// Castling rights
//
// THE SUBTLE FIELD. `board_to_tokens` does not read the FEN's castling letters;
// it calls `has_kingside_castling_rights()` / `has_queenside_castling_rights()`,
// and those answer a narrower question than "is the letter present". A FEN can
// claim "KQkq" with the white king on d1, and python-chess reports no white
// rights at all — the letters are cleaned against where the king and rooks
// actually stand.
//
// Every FEN in golden/movegen.jsonl happens to be self-consistent, so the
// letters and the cleaned rights agree on all 100k positions and a naive
// "is 'K' in the field" implementation would also pass the acceptance test.
// That is precisely why the real rule is implemented here: the corpus is not
// the contract, and the first inconsistent FEN reaching a search that used the
// naive rule would be mis-encoded with nothing to catch it. See DECISIONS.md.
//
// This is the standard-chess (non-Chess960) reading of python-chess, which is
// what `chess.Board(fen)` gives with chess960=False — the mode
// tools/gen_token_golden.py runs in.
// ---------------------------------------------------------------------------

// python-chess gates the castling field on
//     FEN_CASTLING_REGEX = re.compile(r"^(?:-|[KQABCDEFGH]{0,2}[kqabcdefgh]{0,2})$")
// before parsing it. Transcribed rather than approximated so that `tokens()`
// refuses exactly the fields the golden generator refuses: accepting a field
// the reference rejects would mean returning a 68-token encoding for a position
// that has no reference answer, which is worse than raising.
inline bool castling_field_is_well_formed(std::string_view castling) {
    const auto is_white = [](char c) { return c == 'K' || c == 'Q' || (c >= 'A' && c <= 'H'); };
    const auto is_black = [](char c) { return c == 'k' || c == 'q' || (c >= 'a' && c <= 'h'); };

    std::size_t i = 0;
    std::size_t white = 0;
    while (i < castling.size() && white < 2 && is_white(castling[i])) {
        ++i;
        ++white;
    }
    std::size_t black = 0;
    while (i < castling.size() && black < 2 && is_black(castling[i])) {
        ++i;
        ++black;
    }
    return i == castling.size();
}

// python-chess's `_set_castling_fen`: turn the letters into a bitboard of the
// rook squares they refer to, *before* any cleaning.
inline Bitboard parse_castling_field(std::string_view castling, const Placement &placement,
                                     std::string_view fen) {
    if (castling.empty() || castling == "-") {
        return 0;
    }
    if (!castling_field_is_well_formed(castling)) {
        reject("invalid castling field", fen);
    }

    Bitboard rights = 0;

    for (char flag : castling) {
        const bool white = (flag >= 'A' && flag <= 'Z');
        const char lower = white ? static_cast<char>(flag - 'A' + 'a') : flag;
        const Bitboard backrank = white ? kRank1 : kRank8;
        const Bitboard rooks = placement.occupied_of(white) & placement.rooks & backrank;

        // `Board.king(color)`: the highest non-promoted king of that colour, or
        // "None" (-1 here) if there is none.
        const int king = msb_index(placement.occupied_of(white) & placement.kings & ~placement.promoted);

        if (lower == 'q') {
            // The leftmost rook, if it is left of the king; otherwise the a-file
            // square is asserted whether or not a rook stands there. The
            // cleaning pass below is what discards that second case.
            if (king >= 0 && lsb_index(rooks) < king) {
                rights |= lsb_mask(rooks);
            } else {
                rights |= kFileA & backrank;
            }
        } else if (lower == 'k') {
            const int rook = msb_index(rooks);
            if (king >= 0 && king < rook) {
                rights |= square_bb(rook);
            } else {
                rights |= kFileH & backrank;
            }
        } else if (lower >= 'a' && lower <= 'h') {
            // Shredder-FEN file letters. python-chess accepts them on a
            // standard board too, where cleaning leaves only 'a' and 'h'
            // standing. No FEN in the corpus uses them; they are handled
            // because FEN_CASTLING_REGEX admits them and silently dropping a
            // letter the reference honoured would be a parity hole.
            rights |= (kFileA << (lower - 'a')) & backrank;
        } else {
            reject("invalid character in the castling field", fen);
        }
    }

    return rights;
}

// python-chess's `clean_castling_rights()`, standard-chess branch: a claimed
// right survives only if a rook of that colour really stands on a1/h1/a8/h8 and
// that colour's king really stands on e1/e8 unpromoted.
inline Bitboard clean_castling_rights(Bitboard claimed, const Placement &placement) {
    const Bitboard castling = claimed & placement.rooks;

    Bitboard white = castling & kRank1 & placement.occupied_white & (kFileA | kFileH);
    Bitboard black = castling & kRank8 & placement.occupied_black & (kFileA | kFileH);

    if ((placement.occupied_white & placement.kings & ~placement.promoted & kSquareE1) == 0) {
        white = 0;
    }
    if ((placement.occupied_black & placement.kings & ~placement.promoted & kSquareE8) == 0) {
        black = 0;
    }

    return white | black;
}

// python-chess's `has_kingside_castling_rights()` / `has_queenside_castling_rights()`.
//
// `king_mask` is compared as a whole 64-bit value rather than as a square index
// because that is what the source does; with the one king per side that real
// positions have, the mask is a single bit and the comparison is the square
// ordering. It is transcribed rather than simplified so that a board with two
// kings — which python-chess permits and which `tokens()` therefore has to
// answer for — lands on the same side of the comparison in both languages.
inline bool has_castling_right(Bitboard cleaned, const Placement &placement, bool white, bool king_side) {
    const Bitboard backrank = white ? kRank1 : kRank8;
    const Bitboard king_mask = placement.kings & placement.occupied_of(white) & backrank & ~placement.promoted;
    if (king_mask == 0) {
        return false;
    }

    Bitboard rights = cleaned & backrank;
    while (rights != 0) {
        const Bitboard rook = lsb_mask(rights);
        if (king_side ? (rook > king_mask) : (rook < king_mask)) {
            return true;
        }
        rights &= rights - 1;
    }
    return false;
}

// ---------------------------------------------------------------------------
// En-passant field
//
// THE TRAP THE BRIEF NAMES. python-chess assigns `board.ep_square` straight
// from the FEN's fourth field with no further test: not the rank, not whether
// an enemy pawn is placed to capture, not whether such a capture would be
// legal. `board_to_tokens` reads that attribute. So the token is a fact about
// the FEN string, and any "is this en passant actually playable" reasoning
// here — however sensible it looks — puts 3,203 of the corpus's 3,610
// ep positions out of distribution.
// ---------------------------------------------------------------------------

inline std::int32_t en_passant_token(std::string_view ep, std::string_view fen) {
    if (ep.empty() || ep == "-") {
        return kTokenEpNone;
    }

    // python-chess resolves the field with `SQUARE_NAMES.index(ep_part)`, so
    // exactly the 64 two-character names parse and everything else is an error.
    if (ep.size() != 2 || ep[0] < 'a' || ep[0] > 'h' || ep[1] < '1' || ep[1] > '8') {
        reject("invalid en passant field", fen);
    }

    // Only the file reaches the network: chess::square_file(ep_square).
    return kTokenEpBase + (ep[0] - 'a');
}

// ---------------------------------------------------------------------------
// The entry point
// ---------------------------------------------------------------------------

// Write the 68 tokens for `fen` into `out`, which must have room for
// `kSeqLength` int32s. Throws std::invalid_argument (ValueError in Python) on a
// FEN python-chess would itself refuse.
//
// `out` is written into directly rather than returned: this is the signature
// the batch path needs so a whole [N, 68] buffer can be filled with no
// per-position allocation and no intermediate copy.
inline void tokenize_into(std::string_view fen, std::int32_t *out) {
    assert(out != nullptr);

    std::string_view rest = fen;
    const std::string_view placement_field = next_field(rest);
    if (placement_field.empty()) {
        reject("empty FEN", fen);
    }

    const Placement placement = parse_placement(placement_field, fen);

    // Indices 0..63. Empty squares are 0, which parse_placement already left in
    // place, so this is a straight copy of the 64 square tokens.
    for (int square = 0; square < 64; ++square) {
        out[square] = placement.square_token[square];
    }

    // Index 64. python-chess defaults a missing side-to-move field to White.
    const std::string_view turn_field = next_field(rest);
    bool white_to_move = true;
    if (!turn_field.empty()) {
        if (turn_field == "w") {
            white_to_move = true;
        } else if (turn_field == "b") {
            white_to_move = false;
        } else {
            reject("expected 'w' or 'b' for the side-to-move field", fen);
        }
    }
    out[kIdxSideToMove] = white_to_move ? kTokenWhiteToMove : kTokenBlackToMove;

    // Index 65.
    const std::string_view castling_field = next_field(rest);
    const Bitboard cleaned = clean_castling_rights(parse_castling_field(castling_field, placement, fen), placement);
    const int mask = (has_castling_right(cleaned, placement, true, true) ? kCastleWhiteKing : 0) |
                     (has_castling_right(cleaned, placement, true, false) ? kCastleWhiteQueen : 0) |
                     (has_castling_right(cleaned, placement, false, true) ? kCastleBlackKing : 0) |
                     (has_castling_right(cleaned, placement, false, false) ? kCastleBlackQueen : 0);
    out[kIdxCastling] = kTokenCastlingBase + mask;

    // Index 66.
    out[kIdxEnPassant] = en_passant_token(next_field(rest), fen);

    // The halfmove clock and fullmove number are consumed but not read; see
    // below. python-chess refuses a seventh field ("expected 6 parts"), so a
    // FEN with trailing junk is refused here too rather than silently encoded.
    next_field(rest);
    next_field(rest);
    if (!next_field(rest).empty()) {
        reject("expected at most 6 space-separated fields in FEN", fen);
    }

    // Index 67. The halfmove clock and fullmove number are deliberately not
    // parsed: neither reaches the encoding, and the v5 network is blind to both.
    // python-chess does validate them as integers, so a FEN with a non-numeric
    // clock is accepted here and refused there. That is the one place this
    // parser is knowingly laxer than the reference, and it is safe in the only
    // direction that matters: no FEN the reference accepts is refused here, and
    // no accepted FEN produces a different token. See DECISIONS.md.
    out[kIdxCls] = kTokenCls;
}

}  // namespace guofish

#endif  // GUOFISH_TOKENS_HPP
