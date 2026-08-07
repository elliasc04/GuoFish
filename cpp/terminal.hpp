// GuoFish C6 — the rules that end a game, transcribed from python-chess.
//
// This header holds no search logic and no tree state. It holds the predicates
// `board.is_game_over()` is made of, as pure functions of a ParsedFen, plus the
// two move classifiers (`is_zeroing`, `is_irreversible`) that the repetition
// walk-back needs. The search in cpp/search.hpp composes them; the reason they
// live apart is that every one of them is a transcription of a specific
// python-chess function and each is testable on a FEN alone.
//
//
// WHY NOT chess-library's OWN isGameOver()
// ----------------------------------------
// It exists, it is one call, and it answers a different question. At the pinned
// revision it reports THREEFOLD_REPETITION and FIFTY_MOVE_RULE — the *claimable*
// draws. python-chess's `is_game_over()` with the default `claim_draw=False`
// reports neither: it ends a game on checkmate, stalemate, insufficient
// material, the SEVENTY-FIVE-move rule and FIVEFOLD repetition, and leaves the
// threefold/fifty-move pair to the caller because they need a claim.
//
// That difference is the whole shape of C6. The reference handles the claimable
// pair itself, in `MCTSWorker._draw_by_rule`, path-dependently and against a
// history the position alone does not carry — and it deliberately leaves those
// nodes unexpanded so a host that declines to claim can still be played from.
// Delegating to `isGameOver()` would collapse the two categories into one and
// reintroduce exactly the defect the chunk exists to remove.
//
// The library's own repetition counter is wrong for us for a second reason: it
// counts over `prev_states_`, which chess-library fills from `makeMove`. Our
// search's history is not the library's move stack — it is the game history
// handed in at the root plus the current simulation's path — so the count has to
// be taken there. See ReplaySearch::is_repetition.
//
//
// THE ORDER OF THE TESTS IS PART OF THE ANSWER
// --------------------------------------------
// `Board.outcome()` asks, in this order: checkmate, insufficient material,
// stalemate, seventy-five moves, fivefold repetition. Insufficient material
// comes BEFORE stalemate, so a position that is both — a bare king stalemated by
// a lone king and knight, say — is reported as insufficient material. Both give
// a 0.0 backup, so nothing in this chunk can observe the difference; it is
// transcribed in the reference's order anyway, because the moment anything reads
// the reason rather than the value the order stops being cosmetic.

#ifndef GUOFISH_TERMINAL_HPP
#define GUOFISH_TERMINAL_HPP

#include "keys.hpp"
#include "tokens.hpp"

#include <cstdint>

namespace guofish {

// ---------------------------------------------------------------------------
// Bit helpers
// ---------------------------------------------------------------------------

// Portable population count. Global Rule 8 forbids `__popcnt64` without a
// fallback, and the only caller is the insufficient-material test, which runs
// once per terminal leaf — far off the hot path, so the loop costs nothing worth
// measuring.
inline int popcount(Bitboard bb) {
    int count = 0;
    while (bb != 0) {
        bb &= bb - 1;
        ++count;
    }
    return count;
}

// python-chess's BB_DARK_SQUARES / BB_LIGHT_SQUARES. a1 (index 0) is a DARK
// square, and dark squares are those where (file + rank) is even, i.e. the low
// bit of (index ^ (index >> 3)) is 0.
inline constexpr Bitboard kDarkSquares = 0xAA55'AA55'AA55'AA55ULL;
inline constexpr Bitboard kLightSquares = ~kDarkSquares;

// ---------------------------------------------------------------------------
// Move classifiers
//
// Both take the position BEFORE the move and the move's python-chess squares —
// i.e. the NORMALISED destination, g1/c1 for castling, never chess-library's
// king-takes-rook encoding. The distinction is load-bearing here in a way it is
// not in the packed-move code: `is_zeroing` asks whether the touched squares
// meet the opponent's occupancy, and under the king-takes-rook encoding the
// destination holds OUR OWN rook. That is not the opponent's, so the answer
// happens to come out right either way — but `_reduces_castling_rights` tests
// `touched & castling_rights`, and the castling rights ARE the rook squares, so
// there the two encodings give different answers. Normalised in, always.
// ---------------------------------------------------------------------------

// `Board.is_zeroing`: a capture or a pawn move, i.e. a move that resets the
// halfmove clock. `move.drop` is a crazyhouse concept and is always None here.
inline bool is_zeroing(const ParsedFen &parsed, int from_square, int to_square) {
    const Bitboard touched = square_bb(from_square) ^ square_bb(to_square);
    const Placement &placement = parsed.placement;
    return (touched & placement.pawns) != 0 ||
           (touched & placement.occupied_of(!parsed.white_to_move)) != 0;
}

// `Board._reduces_castling_rights`. `parsed.castling` is already the CLEANED
// rights bitboard (rook squares), which is what python-chess compares against.
//
// The two king clauses are not redundant with the first. A king move from e1
// does not touch the rook squares, so `touched & cr` misses it; the second
// clause catches "this side still has rights and the king is on the move".
inline bool reduces_castling_rights(const ParsedFen &parsed, int from_square, int to_square) {
    const Bitboard cr = parsed.castling;
    const Bitboard touched = square_bb(from_square) ^ square_bb(to_square);
    const Placement &placement = parsed.placement;

    if ((touched & cr) != 0) {
        return true;
    }
    if ((cr & kRank1) != 0 &&
        (touched & placement.kings & placement.occupied_white & ~placement.promoted) != 0) {
        return true;
    }
    if ((cr & kRank8) != 0 &&
        (touched & placement.kings & placement.occupied_black & ~placement.promoted) != 0) {
        return true;
    }
    return false;
}

// `Board.is_irreversible`. The repetition walk-back stops at the first of these
// going backwards: nothing before an irreversible move can repeat what comes
// after it.
//
// The third clause is the surprising one and it is transcribed rather than
// reasoned about: python-chess calls a move irreversible if the position it is
// played FROM merely *has* a legal en-passant capture available, whether or not
// the move takes it. That is `has_legal_en_passant()` — the C3 rule, the one
// `rep_key` is built on — and not the raw ep square.
inline bool is_irreversible(const ParsedFen &parsed, int from_square, int to_square) {
    return is_zeroing(parsed, from_square, to_square) ||
           reduces_castling_rights(parsed, from_square, to_square) ||
           has_legal_en_passant(parsed);
}

// ---------------------------------------------------------------------------
// Insufficient material
// ---------------------------------------------------------------------------

// `Board.has_insufficient_material(color)`, verbatim, including the two quirks
// that make it worth transcribing rather than paraphrasing:
//
//   * the bishop clause tests `self.bishops` — BOTH colours' bishops — for the
//     same-colour-complex question, and `self.pawns` / `self.knights` globally.
//     So "I have only bishops" is judged against the whole board's bishops, not
//     against mine. That is the correct reading of the FIDE rule (a lone bishop
//     cannot mate even against another bishop on the same complex) and it is
//     what the reference does.
//   * the knight clause asks whether the OPPONENT has anything besides kings and
//     queens, because a knight can deliver a selfmate against those.
inline bool has_insufficient_material(const Placement &placement, bool white) {
    const Bitboard ours = placement.occupied_of(white);

    if ((ours & (placement.pawns | placement.rooks | placement.queens)) != 0) {
        return false;
    }

    if ((ours & placement.knights) != 0) {
        const Bitboard theirs = placement.occupied_of(!white);
        return popcount(ours) <= 2 && (theirs & ~placement.kings & ~placement.queens) == 0;
    }

    if ((ours & placement.bishops) != 0) {
        const bool same_colour = (placement.bishops & kDarkSquares) == 0 ||
                                 (placement.bishops & kLightSquares) == 0;
        return same_colour && placement.pawns == 0 && placement.knights == 0;
    }

    return true;
}

// `Board.is_insufficient_material()`: neither side can force a mate.
inline bool is_insufficient_material(const Placement &placement) {
    return has_insufficient_material(placement, true) && has_insufficient_material(placement, false);
}

// ---------------------------------------------------------------------------
// The outcome
// ---------------------------------------------------------------------------

// Why a game ended, in `Board.outcome()`'s own order. `None` means it has not.
//
// The claimable draws — threefold repetition and the fifty-move rule — are
// deliberately absent. They are not outcomes in python-chess's default sense and
// the reference detects them somewhere else entirely; conflating them is the
// defect this chunk removes. See MCTSWorker._draw_by_rule and
// ReplaySearch::draw_by_rule.
enum class TerminalReason : std::uint8_t {
    None = 0,
    Checkmate,
    InsufficientMaterial,
    Stalemate,
    SeventyFiveMoves,
    FivefoldRepetition,
};

inline const char *terminal_reason_name(TerminalReason reason) noexcept {
    switch (reason) {
        case TerminalReason::None: return "none";
        case TerminalReason::Checkmate: return "checkmate";
        case TerminalReason::InsufficientMaterial: return "insufficient-material";
        case TerminalReason::Stalemate: return "stalemate";
        case TerminalReason::SeventyFiveMoves: return "seventy-five-moves";
        case TerminalReason::FivefoldRepetition: return "fivefold-repetition";
    }
    return "?";
}

// `Board.outcome(claim_draw=False)`, minus the variant clauses (this engine is
// standard chess only, and python-chess's `is_variant_*` are constant there).
//
// The caller supplies the three facts that need a board rather than a placement:
// whether the side to move is in check, how many legal moves it has, and whether
// the position has occurred five times. Passing them in rather than computing
// them here is what keeps this header free of both movegen and search state —
// and the legal-move count is one the search has already paid for at the leaf.
inline TerminalReason outcome_of(const ParsedFen &parsed, bool in_check, std::size_t legal_move_count,
                                 int halfmove_clock, bool fivefold_repetition) {
    if (in_check && legal_move_count == 0) {
        return TerminalReason::Checkmate;
    }
    if (is_insufficient_material(parsed.placement)) {
        return TerminalReason::InsufficientMaterial;
    }
    if (legal_move_count == 0) {
        return TerminalReason::Stalemate;
    }
    // `_is_halfmoves(n)` is `halfmove_clock >= n AND a legal move exists`. The
    // second half is already established: legal_move_count == 0 returned above.
    if (halfmove_clock >= 150) {
        return TerminalReason::SeventyFiveMoves;
    }
    if (fivefold_repetition) {
        return TerminalReason::FivefoldRepetition;
    }
    return TerminalReason::None;
}

// The value the reference backs up for an outcome, from the perspective of the
// player who MOVED TO this position — the opponent of the side to move.
//
// `MCTSWorker._run_simulation` derives it from `board.result()`:
//
//     "1-0" -> +1.0 if Black is to move else -1.0
//     "0-1" -> +1.0 if White is to move else -1.0
//     draw  ->  0.0
//
// Checkmate is the only decisive outcome here, and its winner is `not turn` by
// definition, so the first branch is always the +1.0 one: the side that just
// moved is the side that mated. -1.0 is unreachable, and saying so is more
// useful than transcribing a branch that cannot be taken — a terminal node
// carrying -1.0 would mean the mated side had somehow just moved.
inline double terminal_value_of(TerminalReason reason) noexcept {
    return reason == TerminalReason::Checkmate ? 1.0 : 0.0;
}

}  // namespace guofish

#endif  // GUOFISH_TERMINAL_HPP
