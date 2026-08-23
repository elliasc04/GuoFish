#ifndef GUOFISH_MOVE_STATS_HPP
#define GUOFISH_MOVE_STATS_HPP

// ---------------------------------------------------------------------------
// M1 — the `--move-stats` decision-trajectory emitter.
//
// WHAT THIS IS FOR. A 200,000-simulation search already contains the answer to
// "what would this engine have played at 25,000?" — it just throws it away on
// the way past. This records it: at nine points on a delivered-simulation
// ladder it snapshots the root's four most-visited children. One 200k search
// then prices every early-exit constant under consideration, offline, with no
// extra games.
//
// THE RULE THIS UNIT EXISTS TO OBEY. The flag may not change move selection,
// visit counts, or search order. Three properties make that structural rather
// than reviewed:
//
//   1. OFF IS A BOOL. `ReplaySearch` holds `move_stats_on_`, false unless a
//      caller armed it for this move. The whole of the hot-path cost when off
//      is one predictable branch per BATCH — not per simulation.
//   2. NOTHING HERE WRITES TO THE TREE. Every arena access below is a `const`
//      accessor. There is no path from this file to a visit count, a value sum,
//      a prior, or a state bit.
//   3. NO ALLOCATION AFTER `begin`. `begin` reserves the checkpoint vector to
//      the ladder's length plus the final rung; `capture` pushes into reserved
//      space and the top-4 selection is a fixed four-slot array on the stack.
//
// WHERE IT IS CALLED FROM, and it is one place in the search:
// `ReplaySearch::process_batch`, after the batch's leaves have been expanded and
// backed up. That is the dispatcher thread, which is not the selection loop, and
// it is the finest-grained point at which `delivered_` is a settled number.
//
// WHAT THE SNAPSHOT IS AND IS NOT. It is a read of a tree that W worker threads
// are still descending. Every field read is either an atomic load or is
// published behind the `Expanded` release store the acquire load below pairs
// with, so it is well-defined — but it is not a stopped-world snapshot: up to
// `max_outstanding` simulations are in flight and their backups have not landed.
// At the coarsest rung that is 24 visits in 150,000. The FINAL checkpoint is
// taken from `finish()`, which the host calls with no search in flight, and is
// therefore exact.
//
// LADDER RESOLUTION. The check fires once per batch, so a checkpoint is taken at
// the first batch boundary at or past its rung — `n` records where that actually
// was, and the analysis uses `n`, not the rung. A batch that crosses several
// rungs at once produces ONE checkpoint, tagged with the highest rung it
// crossed, because there is only one tree state to record. With `max_batch` at
// 128 that can only happen at the 1k and 2k rungs.
// ---------------------------------------------------------------------------

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "arena.hpp"

namespace guofish {

// One root child at one checkpoint. `move` is packed as `pack_move()` packs it;
// the host unpacks it to UCI, because that is where the move vocabulary lives.
struct RootChildSnapshot {
    std::uint16_t move = 0;
    std::int32_t visits = 0;
    // The CHILD's Q in the engine's own perspective, i.e. `value_sum / visits`
    // used as-is. The backup negates on the way up, so a child of the root is
    // already scored from the point of view of the side to move at the root —
    // the same convention as `Engine._principal_variation`'s
    // `last_best_child_q`. Contrast `root_q` below, which is negated.
    double q = 0.0;
};

struct MoveCheckpoint {
    // The ladder rung this fired for. `kFinalRung` for the end-of-move
    // checkpoint, which has no rung and is not on the ladder.
    std::int64_t rung = 0;
    // DELIVERED simulations when the snapshot was taken. This, not `rung`, is
    // the x-axis of every downstream curve.
    std::int64_t n = 0;
    std::int64_t root_visits = 0;
    // The ROOT's own Q, negated: a node's `value_sum` reads from the
    // perspective of the player who moved TO it, and at the root that is the
    // opponent.
    double root_q = 0.0;
    // How many children the root has, and how many of `top4` are populated.
    // Both, because "3 of 3" and "3 of 41" are different positions.
    std::int32_t root_children = 0;
    std::uint8_t top_n = 0;
    std::array<RootChildSnapshot, 4> top4{};

    std::uint16_t argmax() const noexcept { return top_n > 0 ? top4[0].move : 0; }
};

// `MoveCheckpoint::rung` for the end-of-move snapshot.
inline constexpr std::int64_t kFinalRung = -1;
// `MoveStatsRecord::n_lock` when the argmax was still moving at the final rung.
inline constexpr std::int64_t kNoLock = -1;

struct MoveStatsRecord {
    std::vector<MoveCheckpoint> checkpoints;
    // Argmax flips across consecutive checkpoints, the final one included.
    std::int64_t best_move_changes = 0;
    // `n` of the earliest checkpoint from which the argmax never changes again.
    // `kNoLock` when the argmax flipped at the final rung — i.e. the search had
    // not settled, which is a finding and not a missing value.
    std::int64_t n_lock = kNoLock;
    std::int64_t delivered = 0;
    // Batches that crossed more than one rung, so a reader can tell a hole in
    // the ladder from a bug.
    std::int64_t coalesced_rungs = 0;
};

// The ladder the brief specifies, delivered-denominated.
const std::vector<std::int64_t> &default_ladder();

// The two derivations that are free here and awkward anywhere else. Separated
// from the recorder so they are testable against a hand-built checkpoint list.
void derive_trajectory(MoveStatsRecord &record);

class MoveStatsRecorder {
public:
    // Arm for one move. `ladder` must be strictly increasing; an empty ladder
    // is legal and means "final checkpoint only".
    void begin(const std::vector<std::int64_t> &ladder) {
        ladder_ = ladder;
        next_ = 0;
        base_ = 0;
        coalesced_ = 0;
        checkpoints_.clear();
        checkpoints_.reserve(ladder_.size() + 1);
    }

    void clear() {
        ladder_.clear();
        checkpoints_.clear();
        checkpoints_.shrink_to_fit();
        next_ = 0;
        base_ = 0;
        coalesced_ = 0;
    }

    // Simulations delivered by slices of this move that have already returned.
    // `delivered_` inside the core is per-`search_parallel`; a move is many
    // calls, and the ladder is denominated in the move's total.
    std::int64_t base() const noexcept { return base_; }
    void add_base(std::int64_t n) noexcept { base_ += n; }

    std::size_t size() const noexcept { return checkpoints_.size(); }

    // THE HOT-PATH ENTRY POINT. One integer compare in the common case.
    template <class ArenaT>
    void maybe_capture(std::int64_t delivered, const ArenaT &arena,
                       std::uint32_t root) {
        if (next_ >= ladder_.size() || delivered < ladder_[next_]) {
            return;
        }
        const std::size_t first = next_;
        while (next_ < ladder_.size() && delivered >= ladder_[next_]) {
            ++next_;
        }
        if (next_ - first > 1) {
            coalesced_ += static_cast<std::int64_t>(next_ - first) - 1;
        }
        capture(ladder_[next_ - 1], delivered, arena, root);
    }

    // The end-of-move snapshot plus the derivations. Called by the host with no
    // search in flight, so this checkpoint alone is exact.
    template <class ArenaT>
    MoveStatsRecord finish(std::int64_t delivered, const ArenaT &arena,
                           std::uint32_t root) {
        capture(kFinalRung, delivered, arena, root);
        MoveStatsRecord out;
        out.checkpoints = checkpoints_;
        out.delivered = delivered;
        out.coalesced_rungs = coalesced_;
        derive_trajectory(out);
        return out;
    }

private:
    template <class ArenaT>
    void capture(std::int64_t rung, std::int64_t n, const ArenaT &arena,
                 std::uint32_t root) {
        MoveCheckpoint cp;
        cp.rung = rung;
        cp.n = n;
        cp.root_visits = arena.visit_count(root);
        if (cp.root_visits > 0) {
            cp.root_q = -arena.value_sum(root) / static_cast<double>(cp.root_visits);
        }

        // ACQUIRE, and it is what makes the two plain fields below readable from
        // this thread at all. `set_children` publishes `children_offset` and
        // `children_count` and THEN release-stores `Expanded`; reading the state
        // with acquire is the other half of that pair. Without it the two reads
        // would be a data race on a node another thread had just expanded.
        if (arena.lifecycle(root) != NodeState::Expanded) {
            checkpoints_.push_back(cp);
            return;
        }
        const std::uint32_t first = arena.children_offset(root);
        const std::uint16_t count = arena.children_count(root);
        cp.root_children = static_cast<std::int32_t>(count);

        // Top 4 by visits, by insertion into a four-slot array: O(4n) with no
        // allocation and no sort. Ties keep the EARLIER child, which is
        // canonical move order, so the snapshot is a deterministic function of
        // the visit counts it read.
        std::array<RootChildSnapshot, 4> best{};
        std::uint8_t filled = 0;
        for (std::uint16_t k = 0; k < count; ++k) {
            const std::uint32_t child = first + k;
            const std::int32_t visits = arena.visit_count(child);
            if (visits <= 0) {
                continue;   // a reply with no visits is not a branch
            }
            RootChildSnapshot snap;
            snap.move = arena.move(child);
            snap.visits = visits;
            snap.q = arena.value_sum(child) / static_cast<double>(visits);

            // Where it belongs: walk left while the entry to the left is
            // strictly smaller. Strictly, so an equal-visit child that was seen
            // earlier keeps its place.
            std::size_t pos = filled;
            while (pos > 0 && best[pos - 1].visits < snap.visits) {
                --pos;
            }
            if (pos >= best.size()) {
                continue;       // not in the top four
            }
            for (std::size_t j = (filled < best.size() ? filled : best.size() - 1);
                 j > pos; --j) {
                best[j] = best[j - 1];
            }
            best[pos] = snap;
            if (filled < best.size()) {
                ++filled;
            }
        }
        cp.top_n = filled;
        cp.top4 = best;
        checkpoints_.push_back(cp);
    }

    std::vector<std::int64_t> ladder_;
    std::vector<MoveCheckpoint> checkpoints_;
    std::size_t next_ = 0;
    std::int64_t base_ = 0;
    std::int64_t coalesced_ = 0;
};

}  // namespace guofish

#endif  // GUOFISH_MOVE_STATS_HPP
