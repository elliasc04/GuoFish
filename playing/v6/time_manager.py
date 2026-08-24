#!/usr/bin/env python3
"""Optional, off-by-default time manager owned by the engine:
three offline fits (`docs/TC/TM_CONSTANTS.md`), one control law, and a shadow
mode that evaluates the whole thing in deployment before it is ever allowed to
pick a move.

    reserve      = POST_p95 + lichess-bot's move_overhead
    moves_left   = max(FLOOR, s * A * exp(-ply / tau))       # in OUR moves
    base_time    = (clock - reserve - emergency) / moves_left + increment
    nps_est      = smoothed[piece_bucket]
    target_nodes = base_time * nps_est
    target_total = max(current_root_visits, target_nodes)    # absolute

and then, once, at the smart-pruning floor:

    at delivered >= max(FLOOR_ABSOLUTE, FLOOR_FRACTION * target_nodes):
        leader unreachable and n_competing == 1  -> exit now
        otherwise                                -> target *= m(n_competing)

NO ENGINE IMPORTS. Everything here is arithmetic on numbers the caller hands
it, which is what lets `telemetry/tm_replay.py` run L0's eighty recorded games
through the identical code with no engine, no GPU and no clock.

THREE PLACES THIS DEPARTS FROM `tc_manager_brief.md`, each named where it
happens and each for a reason the brief itself supplies:

  1. `s` is 0.9714, not the ~0.90 the brief expected. Fitted, not assumed; see
     `TM_CONSTANTS.md` F1. Deployed games are slightly LONGER than the PGN set.
  2. `m(1)` is 0.92 and not 1.0, so the multiplier applies on every branch
     rather than only on `n_competing >= 2`. With `m(1)` pinned at 1.0 and
     every other rung above it, `E[m]` is 1.08 by construction — the brief's
     own mean-neutrality constraint cannot be satisfied. See `MULTIPLIERS`.
  3. The endgame nps bucket is `<= 10` pieces rather than the suggested
     `10-7 / <= 6` split, which is the occupancy adjustment the brief asks for.

LABELS (ground rule 5) are on every constant below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# --- the fitted constants --------------------------------------------------
#
# EVERY ONE OF THESE COMES FROM `docs/TC/TM_CONSTANTS.md`, which
# `telemetry/fit_tm_constants.py` regenerates from data on disk. They are
# duplicated here rather than read from the JSON because an engine that needs a
# file to plan a move has a new way to fail on move one; the JSON is the audit
# trail and `tests/test_tm_manager.py` asserts the two agree.

MOVES_LEFT_A = 98.98          # measured, MOVES_LEFT.md (3,185 PGN games)
MOVES_LEFT_TAU = 113.1        # measured, MOVES_LEFT.md
MOVES_LEFT_S = 0.9714         # fitted,   TM_CONSTANTS.md F1 (76 deployed games)
# In OUR moves, not plies, and ROUNDED UP from 44.95 plies = 22.47 of our
# moves. `moves_left` is a divisor: rounding up allocates less per move, which
# is the safe direction. See TM_CONSTANTS.md F1 for why there is a floor at all
# — `E[remaining | alive]` is non-monotone and bottoms out at ply 86.
MOVES_LEFT_FLOOR = 23         # derived,  TM_CONSTANTS.md F1

# (low, high, median nps). Inclusive piece-count bounds, checked in order.
# measured, TM_CONSTANTS.md F2, over L0's 1,805 full searches.
NPS_BUCKETS: tuple[tuple[int, int, float], ...] = (
    (25, 32, 16178.0),
    (17, 24, 17490.0),
    (11, 16, 23090.0),
    (0, 10, 30656.0),
)
NPS_GLOBAL = 17709.0          # measured, L0_LOGS.md §3 — the fallback
NPS_ALPHA = 0.474             # measured, L0_LOGS.md §8 (lag-1 rho 0.526)

THETA = 0.5                   # fitted (swept), TM_CONSTANTS.md F3
# The magnitudes are ESTIMATED; the ORDERING is measured — P(argmax at 25k
# differs from final) runs 7.08% / 30.39% / 39.29% / 50.47% across these four
# cells, monotone, over 5,172 moves.
#
# `m(1) < 1` IS THE MEAN-NEUTRALITY CONSTRAINT AND IS NOT A TYPO. A set that
# is 1.0 on the 70% case and larger everywhere else has `E[m] = 1.08`: a
# systematic 8% over-spend wearing an adaptive costume, which is the failure
# ANALYSIS.md §4a names and asks to be checked for. Dividing the hand set
# {1.00, 1.20, 1.35, 1.50} through by its own mean keeps the measured ordering
# and the ratios and moves only the level, which was hand-chosen anyway.
MULTIPLIERS: dict[int, float] = {1: 0.92, 2: 1.11, 3: 1.25, 4: 1.39}

RESERVE_MS = 2706.0           # derived: POST p95 706 + host move_overhead 2000
EMERGENCY_MS = 5000.0         # estimated, at 10+3; see TM_CONSTANTS.md
CLOCK_FRACTION_CAP = 0.40     # measured: `_allot`'s existing guard, restated
FLOOR_ABSOLUTE = 25000        # measured, ANALYSIS.md §3
FLOOR_FRACTION = 0.125        # estimated: 25,000 / the deployed 200,000
# The floor rule reads the top FOUR children and no more, because that is what
# F3 was fit on — `move_stats` records `top4` and `n_competing` saturates at 4
# there. Reading more here would apply a multiplier fitted on a different
# statistic than the one being computed.
TOP_N = 4


@dataclass(frozen=True)
class TMConstants:
    """One immutable bundle, so a test or a replay can vary one number."""

    a: float = MOVES_LEFT_A
    tau: float = MOVES_LEFT_TAU
    s: float = MOVES_LEFT_S
    moves_left_floor: int = MOVES_LEFT_FLOOR
    nps_buckets: tuple[tuple[int, int, float], ...] = NPS_BUCKETS
    nps_global: float = NPS_GLOBAL
    nps_alpha: float = NPS_ALPHA
    theta: float = THETA
    multipliers: tuple[tuple[int, float], ...] = tuple(sorted(MULTIPLIERS.items()))
    reserve_ms: float = RESERVE_MS
    emergency_ms: float = EMERGENCY_MS
    clock_fraction_cap: float = CLOCK_FRACTION_CAP
    floor_absolute: int = FLOOR_ABSOLUTE
    floor_fraction: float = FLOOR_FRACTION

    def multiplier(self, n_competing: int) -> float:
        table = dict(self.multipliers)
        if not table:
            return 1.0
        return table.get(min(max(n_competing, min(table)), max(table)), 1.0)

    def bucket(self, piece_count: Optional[int]) -> Optional[tuple[int, int]]:
        if piece_count is None:
            return None
        for lo, hi, _ in self.nps_buckets:
            if lo <= piece_count <= hi:
                return (lo, hi)
        return None

    def bucket_seed(self, key: Optional[tuple[int, int]]) -> float:
        for lo, hi, med in self.nps_buckets:
            if (lo, hi) == key:
                return med
        return self.nps_global


DEFAULT_CONSTANTS = TMConstants()


@dataclass
class TMPlan:
    """What the manager would allocate, whether or not anyone is listening.

    Every field here goes on the `--move-stats` record under a `tm_` prefix,
    which is the whole of Part 3: with the manager OFF these are computed,
    recorded and ignored, so a week of ordinary rated play prices the manager
    at zero risk before the switch is flipped.
    """

    # --- the allocation ---------------------------------------------------
    moves_left: float = 0.0
    moves_left_floored: bool = False
    base_time_ms: float = 0.0
    nps_est: float = 0.0
    bucket: Optional[tuple[int, int]] = None
    bucket_seeded: bool = True          # no observation yet this game
    target_nodes: int = 0
    target_total: int = 0
    clamped_by_clock: bool = False

    # --- the floor rule, filled in by `at_floor` --------------------------
    floor_at: int = 0
    n_competing: Optional[int] = None
    multiplier: Optional[float] = None
    target_after_multiplier: Optional[int] = None
    would_have_exited_at: Optional[int] = None
    floor_evaluated: bool = False

    # --- provenance -------------------------------------------------------
    applicable: bool = False            # was there a clock to plan against?
    note: str = ""

    def record(self) -> dict:
        """The eight fields the brief names, plus what makes them auditable."""
        return {
            "tm_moves_left": round(self.moves_left, 3),
            "tm_moves_left_floored": self.moves_left_floored,
            "tm_base_time_ms": round(self.base_time_ms, 1),
            "tm_nps_est": round(self.nps_est, 1),
            "tm_bucket": (None if self.bucket is None
                          else f"{self.bucket[0]}-{self.bucket[1]}"),
            "tm_bucket_seeded": self.bucket_seeded,
            "tm_target_nodes": self.target_nodes,
            "tm_target_total": self.target_total,
            "tm_clamped_by_clock": self.clamped_by_clock,
            "tm_floor_at": self.floor_at,
            "tm_n_competing": self.n_competing,
            "tm_multiplier": self.multiplier,
            "tm_target_after_multiplier": self.target_after_multiplier,
            "tm_would_have_exited_at": self.would_have_exited_at,
            "tm_applicable": self.applicable,
        }


@dataclass
class FloorDecision:
    """The one evaluation at the smart-pruning floor, and its three outcomes."""

    exit_now: bool
    n_competing: int
    multiplier: float
    new_target_total: int
    leader_unreachable: bool


class TimeManager:
    """Per-game state: one exponentially-smoothed nps estimate per bucket.

    THE STATE IS DELIBERATELY THIN. There is no piggybank and no accrual
    account: recomputing `(clock - reserve) / moves_left` every move means clock
    saved by an early exit automatically raises the next move's allocation, so
    the account is EMERGENT. That removes four Lc0 parameters and, given P5's
    near-flat surface, four opportunities to get one wrong.
    """

    def __init__(self, constants: TMConstants = DEFAULT_CONSTANTS) -> None:
        self.constants = constants
        self._nps: dict[tuple[int, int], float] = {}
        self._observations: dict[tuple[int, int], int] = {}

    # --- lifecycle --------------------------------------------------------

    def new_game(self) -> None:
        """`ucinewgame`. The smoother is per-game because its rate is.

        `alpha = 0.474` comes from a lag-1 autocorrelation computed WITHIN games
        and never across a game boundary (L0 §8). Carrying an estimate across
        games would use it outside the regime it was measured in, and the first
        move of a new game is exactly where a stale endgame estimate does the
        most damage.
        """
        self._nps.clear()
        self._observations.clear()

    def observe(self, piece_count: Optional[int], delivered: int,
                wall_s: float) -> Optional[float]:
        """Fold one completed search into its bucket's estimate.

        Returns the new estimate, or None when the move carried no usable rate.
        A move that delivered a handful of simulations in a few milliseconds —
        a `stop` at game end, a depth-1 mate short-circuit — has an nps that is
        a division by nothing, and L0's own statistics exclude it on the same
        `>= 1000 ms` rule.
        """
        key = self.constants.bucket(piece_count)
        if key is None or delivered <= 0 or wall_s < 1.0:
            return None
        rate = delivered / wall_s
        alpha = self.constants.nps_alpha
        prior = self._nps.get(key, self.constants.bucket_seed(key))
        updated = alpha * rate + (1.0 - alpha) * prior
        self._nps[key] = updated
        self._observations[key] = self._observations.get(key, 0) + 1
        return updated

    def nps_for(self, piece_count: Optional[int]) -> tuple[float, Optional[tuple[int, int]], bool]:
        """(estimate, bucket, is-still-the-seed). The F2 fallback chain."""
        key = self.constants.bucket(piece_count)
        if key is None:
            return self.constants.nps_global, None, True
        if key in self._nps:
            return self._nps[key], key, False
        return self.constants.bucket_seed(key), key, True

    # --- the control law --------------------------------------------------

    def moves_left(self, ply: int) -> tuple[float, bool]:
        """`max(FLOOR, s * A * exp(-ply/tau)) / 2`, in OUR moves.

        The estimator is fitted in plies and consumed in our moves, and the
        halving is the only conversion: we move on every other ply. The floor
        binds from roughly the median game length onward, so past the midpoint
        this is effectively "assume twenty-three more moves" — intentional, and
        the safe direction. `MOVES_LEFT.md` has the inspection paradox that
        makes it necessary.
        """
        c = self.constants
        plies = c.s * c.a * math.exp(-max(0, ply) / c.tau)
        ours = plies / 2.0
        if ours < c.moves_left_floor:
            return float(c.moves_left_floor), True
        return ours, False

    def plan(self, *, clock_ms: Optional[float], increment_ms: Optional[float],
             ply: int, piece_count: Optional[int], current_root_visits: int,
             n_legal: Optional[int] = None,
             node_cap: Optional[int] = None) -> TMPlan:
        """The allocation for one move. Pure; safe to call in shadow mode.

        `node_cap` is the GUI's own `go nodes N` when it sent one. With the
        manager LIVE that is honoured as an ABSOLUTE CAP and the manager is
        subordinate to it — the third row of the brief's switch table, and the
        rollback path if the manager misbehaves in deployment: it is reachable
        by editing `config.yml` alone.

        With no clock there is nothing to divide and `applicable` is False. The
        caller keeps its existing budget; the record still carries the plan so a
        shadow analysis can see that the move was not one the manager owns.
        """
        c = self.constants
        plan = TMPlan()
        nps, bucket, seeded = self.nps_for(piece_count)
        plan.nps_est, plan.bucket, plan.bucket_seeded = nps, bucket, seeded
        plan.moves_left, plan.moves_left_floored = self.moves_left(ply)

        if clock_ms is None or clock_ms <= 0:
            # NOT APPLICABLE, BUT STILL FLOORED. A fixed-node benchmark has no
            # clock and the manager has no opinion about its budget — but shadow
            # mode still wants `n_competing` MEASURED AT THE SAME PLACE Arm B
            # measured it, or the deployment-vs-Arm-B comparison the brief asks
            # for is comparing two different statistics. So the floor is set
            # from the GUI's own budget and the plan is marked inapplicable.
            plan.floor_at = self.floor_for(int(node_cap or 0), n_legal)
            plan.note = "no clock on the go line; the manager has nothing to divide"
            return plan

        inc = float(increment_ms or 0.0)
        # THE EMERGENCY EXISTS SO THE ESTIMATOR DOES NOT AIM AT ZERO. Spending
        # exactly `time_left/moves_left + increment` over exactly `moves_left`
        # moves lands on zero; a few seconds held back is the difference
        # between a tight finish and a flag.
        spendable = float(clock_ms) - c.reserve_ms - c.emergency_ms
        base = spendable / plan.moves_left + inc
        # CLAMPED AT 40% OF THE CLOCK, matching `_allot`'s guard, so the manager
        # cannot request what the deadline would refuse. Doing it here as well
        # as there is not redundant: the deadline would TRUNCATE such a move,
        # and a truncated move wastes the difference between the target it aimed
        # at and the one it could reach.
        cap = c.clock_fraction_cap * float(clock_ms)
        if base > cap:
            base, plan.clamped_by_clock = cap, True
        # A clock this thin has no allocation left to make; one slice is the
        # floor and the deadline is what will actually end the move.
        base = max(1.0, base)

        plan.base_time_ms = base
        plan.target_nodes = max(1, int(base / 1000.0 * nps))
        if node_cap is not None:
            plan.target_nodes = min(plan.target_nodes, max(1, int(node_cap)))
        # ABSOLUTE SEMANTICS, matching the core and `_plan`: `budget` is a
        # root-visit TARGET, and a root that already holds more than the target
        # correctly does no work rather than a negative amount of it.
        plan.target_total = max(int(current_root_visits), plan.target_nodes)
        plan.floor_at = self.floor_for(plan.target_nodes, n_legal)
        plan.applicable = True
        plan.note = (f"{plan.moves_left:.1f} moves left"
                     f"{' (floored)' if plan.moves_left_floored else ''}, "
                     f"{base:.0f} ms x {nps:,.0f} nps "
                     f"-> {plan.target_nodes:,} nodes"
                     f"{' (clamped at 40% of clock)' if plan.clamped_by_clock else ''}")
        return plan

    def floor_for(self, target_nodes: int, n_legal: Optional[int]) -> int:
        """Where the one floor evaluation happens, in DELIVERED simulations.

        FORCING MOVES BYPASS THE FLOOR. With two legal replies or fewer there is
        nothing a floor protects: the exit test is evaluated from the first
        slice, and with one legal reply it fires at once because the single
        visited child is trivially unreachable. Recaptures are NOT included —
        `n_lock` for that subset is computable from Arm B and is deliberately
        out of v1.
        """
        if n_legal is not None and n_legal <= 2:
            return 0
        c = self.constants
        return max(c.floor_absolute, int(c.floor_fraction * target_nodes))

    # --- the floor evaluation ---------------------------------------------

    def at_floor(self, plan: TMPlan, children: Sequence[tuple[str, int]],
                 delivered: int, current_root_visits: int,
                 governing_total: Optional[int] = None) -> FloorDecision:
        """One evaluation, three outcomes. Called ONCE per move.

        `children` is `(uci, visits)` for the root's visited children — what
        `Search.root_children()` returns, which reads the root's child slots and
        stops. It is O(branching), not O(tree), which is what makes it safe to
        do inside a timed move at all; the walk it would have replaced cost a
        measured 424 ms at the deployed pondered median.

        `leader unreachable` is `ANALYSIS.md`'s smart-pruning criterion: the
        leader's visit margin over the runner-up already exceeds every
        simulation the target has left, so no redistribution of the remainder
        can change the argmax. Measured over Arm B's 5,172 laddered moves it
        fires with `n_competing == 1` on 22.3% of them and the argmax changes on
        0 of those.
        """
        top = sorted(children, key=lambda row: row[1], reverse=True)[:TOP_N]
        plan.floor_evaluated = True
        if not top:
            plan.n_competing = 0
            plan.multiplier = 1.0
            plan.target_after_multiplier = plan.target_total
            return FloorDecision(False, 0, 1.0, plan.target_total, False)

        leader = top[0][1]
        n_competing = sum(1 for _, visits in top
                          if visits >= self.constants.theta * leader)
        runner_up = top[1][1] if len(top) > 1 else 0
        # THE BUDGET THE REACHABILITY TEST IS AGAINST, and which one it is
        # matters. When the manager has an opinion the counterfactual is its own
        # target — "would the leader still be unreachable if I had set the
        # budget". When it does not (no clock; a fixed-node benchmark) there is
        # no counterfactual and the honest answer is against the budget the move
        # is actually running to, which the caller passes in.
        total = (plan.target_total if plan.applicable
                 else (governing_total if governing_total is not None
                       else plan.target_total))
        remaining = max(0, total - current_root_visits)
        unreachable = (len(top) == 1) or (leader - runner_up > remaining)

        multiplier = self.constants.multiplier(n_competing)
        # The multiplier scales the NODE TARGET, then the absolute semantics are
        # reapplied — a target below what the root already holds is not a
        # smaller amount of work, it is a negative amount of it.
        scaled = max(1, int(plan.target_nodes * multiplier))
        new_total = max(int(current_root_visits), scaled)

        exit_now = unreachable and n_competing == 1
        plan.n_competing = n_competing
        plan.multiplier = multiplier
        plan.target_after_multiplier = scaled
        if exit_now:
            plan.would_have_exited_at = int(delivered)
        return FloorDecision(exit_now, n_competing, multiplier, new_total,
                             unreachable)


__all__ = [
    "TimeManager", "TMConstants", "TMPlan", "FloorDecision",
    "DEFAULT_CONSTANTS", "MULTIPLIERS", "NPS_BUCKETS",
]
