// GuoFish C7 — Syzygy tablebases, and the reason their results never reach the
// transposition cache.
//
// Two modes, both from the reference:
//
//   MODE 1  playing/uci_wrapper.py, `_probe_tablebase`. At the UCI layer, a
//           ROOT position with <= 5 pieces is played straight out of the
//           tables — best WDL for us, then fastest progress toward a zeroing
//           move by DTZ — and MCTS is not run at all. Tablebase play is
//           perfect, so searching is strictly worse than not searching.
//   MODE 2  core/mctsv4.py, `MCTSWorker._run_simulation`. A LEAF with <= 5
//           pieces reached during a search of a larger root has its neural
//           value replaced by the exact WDL result. The policy is left alone;
//           the tables have no opinion about move ordering.
//
// The WDL -> value mapping is the reference's `wdl_to_value`, unchanged:
//
//     +2 win  -> +1.0      +1 cursed win    -> +0.5      0 draw -> 0.0
//     -2 loss -> -1.0      -1 blessed loss  -> -0.5
//
// i.e. `wdl / 2.0`. The +/-1 endpoints match the bounded range the value head
// produces, so a tablebase result does not read as out-of-distribution to PUCT,
// and the +/-0.5 mappings treat fifty-move-rule cursed/blessed results as half
// decisive.
//
//
// THE ONE THING THIS FILE IS REALLY FOR
// -------------------------------------
// The reference caches the override. From `_run_simulation`:
//
//     nn_value = tb_value          # override
//     ...
//     self.cache.put(cache_key, policy, nn_value)
//
// with the comment "We override BEFORE caching so the WDL value is what gets
// stored — subsequent transpositions to this position reuse it without
// re-probing". That is a real optimisation and it is unsound, for a reason the
// comment does not consider: a Syzygy WDL is computed under the assumption that
// the fifty-move rule does not intervene, so it is a function of the position
// AND the halfmove clock, while `make_cache_key` is a function of the position
// alone. A KQvK win stored at clock 3 is served back at clock 99, where the
// truth is a draw — and the reference's own instrumentation counts exactly this
// crossing (`cache_hit_tb_hmc_crossing`).
//
// This port does not carry it over. The override is applied to the value being
// backed up, at the leaf that was probed, AFTER the network's own value has been
// cached. It is tree-local: nothing that another node could read ever sees it.
// And it is not a discipline anyone has to remember — a `TablebaseValue` is not
// a `NetworkValue`, and `TranspositionCache::insert` takes the latter, so the
// reference's line does not compile here. See cpp/values.hpp and cpp/cache.hpp.
//
//
// WHAT IS NOT IN THIS FILE: THE SYZYGY FILE READER
// ------------------------------------------------
// Nothing here decodes a .rtbw or .rtbz. This file is the ENGINE's half — the
// piece-count gate, the WDL mapping, the perspective conversion, the mode 1
// ranking, the tree-locality and the type discipline — and it is written against
// an abstract `TablebaseProber` so that all of it can be exercised, and is, with
// no tables present at all.
//
// Three backends implement it:
//
//   FathomProber   cpp/fathom.hpp. The production one: native decoding via
//                  jdart1/Fathom, pinned, no Python in the loop, usable from
//                  C9's search threads. Fathom was authorised as a third
//                  dependency for this chunk (Global Rule 7).
//   PythonProber   cpp/bindings.cpp. A Python callable, so the reference's own
//                  `chess.syzygy` handle can serve as an oracle. This is what
//                  makes "Fathom and python-chess agree" a measurement rather
//                  than a claim — see cpp/fathom.hpp, which lists five places
//                  the two do not speak the same language.
//   NullProber     below. Misses on everything: what "tablebases off" means
//                  concretely, and the path production takes on every position
//                  with six or more men.
//
// Tablebases are still OFF unless a backend is attached and a path configured.
// `ReplaySearch::set_tablebase(nullptr)` is the default.

#ifndef GUOFISH_TABLEBASE_HPP
#define GUOFISH_TABLEBASE_HPP

#include "terminal.hpp"
#include "tokens.hpp"
#include "values.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace guofish {

// `TABLEBASE_MAX_PIECES` in core/mctsv4.py. Both kings counted. The downloaded
// set in assets/syzygy covers up to five.
inline constexpr int kTablebaseMaxPieces = 5;

// Syzygy WDL is an integer in [-2, +2] from the SIDE TO MOVE's perspective.
inline constexpr int kWdlLoss = -2;
inline constexpr int kWdlWin = 2;

// `count_pieces(board)`: every man on the board, both kings included.
inline int piece_count(const Placement &placement) { return popcount(placement.occupied()); }

// The reference's `wdl_to_value`. Spelled as the division rather than as a
// lookup table because that is how the reference spells it and the two must not
// be able to drift; `wdl / 2.0` on an int in [-2, 2] is exact in binary floating
// point, so there is no rounding to disagree about.
inline constexpr double wdl_to_value(int wdl) noexcept { return static_cast<double>(wdl) / 2.0; }

static_assert(wdl_to_value(2) == 1.0);
static_assert(wdl_to_value(1) == 0.5);
static_assert(wdl_to_value(0) == 0.0);
static_assert(wdl_to_value(-1) == -0.5);
static_assert(wdl_to_value(-2) == -1.0);

// ---------------------------------------------------------------------------
// The backend interface
//
// One position in, a WDL or DTZ out, or nothing on a miss. A miss is not an
// error: it is "these tables do not cover this position", and every caller's
// answer to it is to keep whatever it already had.
//
// `halfmove_clock` is passed even though Syzygy WDL ignores it, because a
// backend that speaks FEN needs it to produce a FEN, and because a future
// backend implementing the fifty-move-aware probe needs it for real. Passing it
// now means adding that backend does not change this interface.
// ---------------------------------------------------------------------------

class TablebaseProber {
public:
    virtual ~TablebaseProber() = default;

    // WDL in [-2, +2] from the side to move's perspective, or nullopt on a miss.
    virtual std::optional<int> probe_wdl(const ParsedFen &parsed, int halfmove_clock) const = 0;

    // Distance to zero, side-to-move perspective, or nullopt on a miss. Mode 1
    // only — mode 2 needs the outcome, not the distance.
    virtual std::optional<int> probe_dtz(const ParsedFen &parsed, int halfmove_clock) const = 0;

    // For diagnostics and for the manifest a benchmark writes.
    virtual std::string backend() const = 0;
};

// ---------------------------------------------------------------------------
// Mode 2: the leaf override
// ---------------------------------------------------------------------------

// The reference's `probe_tablebase_value`, returning ABSOLUTE (White-POV) value
// to match the network's own convention, or nullopt on a miss.
//
// `probe_wdl` reports from the side to move's perspective, so the result is
// negated when Black is to move. Doing the conversion HERE rather than at the
// backup site is the reference's decision and it is the right one — its own
// docstring says the backup site "is exactly where tablebase perspective bugs
// hide" — and it means the tablebase value flows through the same
// absolute-to-mover negation the network value does, instead of a second
// sign convention nobody can check.
//
// THE RETURN TYPE IS `TablebaseValue`, NOT `double`. That is what makes this
// result unable to reach the cache: cpp/cache.hpp's insert takes a
// `NetworkValue`, and there is no conversion between them.
inline std::optional<TablebaseValue> probe_tablebase_value(const TablebaseProber &prober,
                                                           const ParsedFen &parsed,
                                                           int halfmove_clock) {
    const std::optional<int> wdl = prober.probe_wdl(parsed, halfmove_clock);
    if (!wdl.has_value()) {
        return std::nullopt;
    }
    if (*wdl < kWdlLoss || *wdl > kWdlWin) {
        throw std::invalid_argument("guofish: tablebase backend returned a WDL of " +
                                    std::to_string(*wdl) + ", outside [-2, 2]");
    }
    const double stm_value = wdl_to_value(*wdl);
    return TablebaseValue(parsed.white_to_move ? stm_value : -stm_value);
}

// Is this position within tablebase range at all? Separated from the probe
// because it is a popcount and the probe is a file read: the reference calls it
// "a cheap popcount that filters out the overwhelming majority of middlegame
// leaves before any probe cost", and the ordering is worth keeping visible.
inline bool within_tablebase_range(const ParsedFen &parsed) {
    return piece_count(parsed.placement) <= kTablebaseMaxPieces;
}

// ---------------------------------------------------------------------------
// Mode 1: the root bypass, ranking half
//
// `_probe_tablebase` scores each legal move by a pair and takes the maximum:
//
//     outcome    3 immediate checkmate
//                1 we win (wdl 2 or 1 for us after the move)
//                0 draw, stalemate, or insufficient material
//               -1 we lose
//     distance   winning : 0 if the move zeroes the clock, else -dtz_child
//                losing  : -dtz_child   (larger dtz = the opponent needs longer)
//                else    : 0
//
//     key = (outcome, -distance), maximised.
//
// `dtz_child` is read AFTER the move, so it is from the opponent's perspective;
// the reference negates the WDL for that reason and leaves DTZ un-negated inside
// `distance`, which is why the two look inconsistent and are not.
//
// This struct is the pair, so the ranking can be tested directly on numbers
// rather than only through a board.
// ---------------------------------------------------------------------------

struct TablebaseRootScore {
    int outcome = 0;
    long long distance = 0;

    // `(outcome, -distance) > (other.outcome, -other.distance)`, i.e. Python's
    // tuple comparison on the key the reference builds. Strictly greater, so a
    // tie leaves the incumbent — which makes the answer a function of the order
    // moves are offered in, and is why the driver offers them in canonical
    // order. See DECISIONS.md.
    bool better_than(const TablebaseRootScore &other) const noexcept {
        if (outcome != other.outcome) {
            return outcome > other.outcome;
        }
        return -distance > -other.distance;
    }
};

// The scoring rule for one child, given what the position after the move looks
// like. `mate`, `drawn_terminal` and `zeroing` are the caller's; `wdl_child` and
// `dtz_child` are the backend's, both from the perspective of the side to move
// AFTER our move.
inline TablebaseRootScore tablebase_root_score(bool mate, bool drawn_terminal, bool zeroing,
                                               int wdl_child, int dtz_child) noexcept {
    TablebaseRootScore score;
    if (mate) {
        // Ranked above every tablebase win, distance 0. A mate on the board is
        // strictly better than a mate the tables promise in 40.
        score.outcome = 3;
        score.distance = 0;
        return score;
    }
    if (drawn_terminal) {
        score.outcome = 0;
        score.distance = 0;
        return score;
    }

    const int our_wdl = -wdl_child;
    if (our_wdl > 0) {
        score.outcome = 1;
        // Winning: make concrete progress. A zeroing move that keeps the win
        // resets the fifty-move counter and is the best kind of progress, so it
        // ranks at distance 0 ahead of any DTZ.
        score.distance = zeroing ? 0 : -static_cast<long long>(dtz_child);
    } else if (our_wdl < 0) {
        score.outcome = -1;
        // Losing: stall. A larger dtz_child means the opponent needs more plies
        // to zero us in, and negating makes the maximised key prefer it.
        score.distance = -static_cast<long long>(dtz_child);
    } else {
        score.outcome = 0;
        score.distance = 0;
    }
    return score;
}

// ---------------------------------------------------------------------------
// No backend
//
// The default, and what "tablebases are off" means concretely: a prober that
// misses on everything. It exists rather than a null pointer so that the
// tablebase code path can be exercised — piece-count gate, probe, miss, keep the
// neural value — in a build with no tables at all, which is the path production
// takes on every position with six or more men.
// ---------------------------------------------------------------------------

class NullProber final : public TablebaseProber {
public:
    std::optional<int> probe_wdl(const ParsedFen &, int) const override { return std::nullopt; }
    std::optional<int> probe_dtz(const ParsedFen &, int) const override { return std::nullopt; }
    std::string backend() const override { return "none"; }
};

}  // namespace guofish

#endif  // GUOFISH_TABLEBASE_HPP
