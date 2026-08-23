// M1 — the parts of the move-stats emitter that do not need the arena.
//
// The ladder lives here rather than in a header constant so there is exactly one
// definition of it in the process and a test can compare against the same object
// the search uses. `derive_trajectory` lives here for the same reason it is
// separated at all: it is pure arithmetic over a checkpoint list, so it is
// testable without a tree.

#include "move_stats.hpp"

namespace guofish {

const std::vector<std::int64_t> &default_ladder() {
    // Delivered-denominated, strictly increasing, and it stops BELOW the
    // shipping 200k budget on purpose: the final checkpoint is the top rung and
    // is taken by `finish()`, where the tree is quiescent. A rung at 200,000
    // would race the end of the search for the same snapshot.
    static const std::vector<std::int64_t> ladder = {
        1000, 2000, 5000, 10000, 25000, 50000, 100000, 150000,
    };
    return ladder;
}

void derive_trajectory(MoveStatsRecord &record) {
    const std::vector<MoveCheckpoint> &cps = record.checkpoints;
    record.best_move_changes = 0;
    record.n_lock = kNoLock;
    if (cps.empty()) {
        return;
    }

    for (std::size_t i = 1; i < cps.size(); ++i) {
        if (cps[i].argmax() != cps[i - 1].argmax()) {
            ++record.best_move_changes;
        }
    }

    // THE LOCK POINT. Walk back from the end while the argmax still matches the
    // final one; the first checkpoint that does not match is the last flip, so
    // the one after it is where the search settled.
    //
    // `kNoLock` when the flip happened at the final rung itself. That is not a
    // missing measurement — it is the statement that 200,000 simulations were
    // not enough for this position to settle, which is exactly what a
    // smart-pruning floor needs to know about.
    const std::uint16_t final_move = cps.back().argmax();
    std::size_t i = cps.size();
    while (i > 0 && cps[i - 1].argmax() == final_move) {
        --i;
    }
    if (i + 1 >= cps.size()) {
        return;         // the argmax was still moving at the last rung
    }
    record.n_lock = cps[i].n;
}

}  // namespace guofish
