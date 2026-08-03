"""Pure label-construction functions for the GuoFish multiPV pipeline.

No I/O, no globals, no multiprocessing: everything here is a function of its
arguments so it can be unit-tested directly. Pass B imports this module; the
tests import the same functions the converter actually runs.

POV is WHITE-RELATIVE and hardcoded (Phase 1, 180x separation on the invariant
test). The two consumers use deliberately SEPARATE code paths - they do not
share a sign variable:

    value  : no conversion at all. Data is White-POV, the value head is White-POV.
    policy : multiply by stm (+1 White to move, -1 Black) before the softmax.
"""
from __future__ import annotations

import math
from typing import Optional

import chess

# --- policy scoring bands -------------------------------------------------
# The mate band sits strictly ABOVE the clamped cp band by construction, so no
# centipawn score can ever outrank a forced mate. (The original
# 10000-100*min(|m|,50) mapping overlapped the cp range - this DB emits |cp| up
# to 20000 sentinels - and tripped the 0.1% invariant gate at 0.2475%.)
CP_CLAMP = 10_000
MATE_BASE = 11_020
MATE_STEP = 20
MATE_MAX_DISTANCE = 50          # M1 -> 11000, M50 -> 10020

# --- value target ---------------------------------------------------------
VALUE_CP_CLIP = 2_000           # clip BEFORE tanh; different path from policy
VALUE_MATE = 0.995
VALUE_CLAMP = 0.995             # tanh head cannot reach +-1.0

# The value scale is a CALIBRATION, not a spread target: tanh(cp/290.6806) makes
# v approximate expected score (a +100cp edge -> 0.331 -> 66.6% expected score,
# which is about right at strong play). It is the Lc0 WDL calibration constant,
# also used by benchmarking/player/acpl_elo_estimator.py to invert Q back to cp.
# MCTS Q backup and the tuned C_PUCT / VALUE_LOSS live in these units, so it is
# fixed here rather than refit. Refitting to "overall std 0.5" produced 849.7,
# which flattens a pawn to 55.9% expected score - miscalibrated, not merely flat.
# 11.5% cp saturation is the accepted cost.
VALUE_SCALE = 290.6806

# Raw-score encoding for the record's `value_cp` field. Mates store DISTANCE,
# not a flat sentinel: sign * (30000 - |mate|), clamped at 1000 ply so the mate
# band is always >= 29000 and can never collide with a cp clipped to +-2000.
# int16 tops out at 32767, so 30000 fits with room to spare.
#
# Storing the raw score alongside the squashed one makes the value SCALE a
# collate-time transform, exactly as pv_score makes temperature one - the scale
# can be swept without re-running Pass B. Keeping mate distance means a future
# consumer can also taper M1 vs M40 instead of pinning both to +-0.995.
VALUE_MATE_BASE = 30_000
VALUE_MATE_MAX_DISTANCE = 1_000
VALUE_MATE_MIN = VALUE_MATE_BASE - VALUE_MATE_MAX_DISTANCE      # 29_000

# --- policy target --------------------------------------------------------
DEFAULT_TEMPERATURE = 30.0
DEFAULT_EPSILON = 0.05
MAX_PV = 8                      # pv_idx / pv_prob / pv_score width
MAX_LEGAL = 128                 # legal_idx width


def mate_to_policy_score(mate: int) -> int:
    """sign * (11020 - 20*min(|mate|,50)). Distance preserved: M1 outranks M8."""
    sign = 1 if mate > 0 else -1
    return sign * (MATE_BASE - MATE_STEP * min(abs(mate), MATE_MAX_DISTANCE))


def move_index(move: chess.Move) -> int:
    """Factored 64x64 policy index. Promotions are NOT representable and collide
    here by design - callers accumulate with += so the mass is preserved."""
    return move.from_square * 64 + move.to_square


def pv_move_and_score(board: chess.Board, pv: dict,
                      cp_clamp: int = CP_CLAMP) -> Optional[tuple[chess.Move, int]]:
    """Build (move, policy_score) as ONE unit, in White-POV score space.

    Returns None if EITHER half is unavailable, so a malformed PV can never
    contribute a move without a score or vice versa. This is the misalignment
    guard: there is no code path that appends to two parallel lists at
    different points.

    `mate: 0` is malformed (a mate in zero ply) and returns None.
    """
    line = pv.get("line")
    if not line:
        return None
    try:
        move = board.parse_uci(line.split(" ", 1)[0])
    except Exception:
        return None

    cp = pv.get("cp")
    if cp is not None:
        return move, max(-cp_clamp, min(cp_clamp, int(cp)))
    mate = pv.get("mate")
    if mate is not None:
        m = int(mate)
        if m == 0:
            return None
        return move, mate_to_policy_score(m)
    return None


def dedup_entries(entries: list[tuple[chess.Move, int]]
                  ) -> tuple[list[tuple[chess.Move, int]], int]:
    """Drop repeated PVs, keeping the FIRST occurrence.

    ~4.6% of positions have an eval block containing the same move twice
    (byte-identical line and score) - a merge artifact of two MultiPV
    submissions of the same search being concatenated. Left in, a duplicate
    doubles that move's exponentiated weight and overtakes the true best move
    whenever the gap is under T*ln2 (~20.8cp at T=30), which flips the argmax of
    ~1% of all policy labels.

    Keying on the FULL chess.Move (from, to, promotion) means e8=Q stays
    distinct from e8=N, so genuine underpromotion pairs survive here and merge
    later at the 4096 index via += - which is the intended behaviour.

    Keep-first is correct because PVs are score-ordered: the first occurrence
    carries the better score in the rare (0.6%) case where duplicates disagree.

    MUST be called before the softmax. Deduplicating after the exponentials are
    formed leaves the denominator already double-counted.
    """
    seen: set[chess.Move] = set()
    out: list[tuple[chess.Move, int]] = []
    removed = 0
    for move, score in entries:
        if move in seen:
            removed += 1
            continue
        seen.add(move)
        out.append((move, score))
    return out, removed


def parse_block(board: chess.Board, pvs: list[dict],
                cp_clamp: int = CP_CLAMP
                ) -> tuple[list[tuple[chess.Move, int]], int, int]:
    """Parse one eval block into deduped (move, score) entries.

    Returns (entries, n_dropped_pvs, n_duplicates_removed).
    """
    raw: list[tuple[chess.Move, int]] = []
    dropped = 0
    for pv in pvs:
        got = pv_move_and_score(board, pv, cp_clamp)
        if got is None:
            dropped += 1
            continue
        raw.append(got)
    entries, removed = dedup_entries(raw)
    return entries, dropped, removed


def select_policy_block(board: chess.Board, evals: list[dict],
                        policy_min_depth: int, cp_clamp: int = CP_CLAMP):
    """Deepest eval block that yields >= 2 UNIQUE moves after dedup.

    Selection runs on the DEDUPED list, not the raw pv count. A block whose
    n_pvs=2 is one move listed twice collapses to a single unique move; treating
    it as multi-PV would emit a one-hot policy target, which is exactly the
    hard-label regime this dataset exists to escape. Such a block is rejected
    and the search falls through to the next-deepest qualifying block.

    Returns (entries, depth, dropped, duplicates_removed) or None.
    """
    candidates = sorted(
        (e for e in evals if int(e.get("depth", -1)) >= policy_min_depth),
        key=lambda e: int(e.get("depth", -1)), reverse=True)
    total_removed = 0
    total_dropped = 0
    for ev in candidates:
        pvs = ev.get("pvs") or []
        if len(pvs) < 2:
            continue
        entries, dropped, removed = parse_block(board, pvs, cp_clamp)
        total_removed += removed
        total_dropped += dropped
        if len(entries) >= 2:
            return entries, int(ev.get("depth", -1)), total_dropped, total_removed
    return None


def select_value_block(evals: list[dict], value_min_depth: int) -> Optional[dict]:
    """Deepest eval block meeting value_min_depth. Value reads pvs[0] only.

    Scores are NOT comparable across blocks, so the caller must never mix this
    block with the policy block's scores.
    """
    best = None
    best_depth = -1
    for ev in evals:
        d = int(ev.get("depth", -1))
        if d >= value_min_depth and d > best_depth:
            pvs = ev.get("pvs") or []
            if pvs:
                best, best_depth = ev, d
    return best


def value_target(pv: dict, scale: float = VALUE_SCALE) -> Optional[float]:
    """White-POV value in [-0.995, +0.995] from the value block's pvs[0].

    Separate path from the policy mapping: cp is clipped to +-2000 and squashed
    by tanh(cp/scale); a mate is set directly to +-0.995. No stm conversion -
    the data is already White-POV and so is the head.
    """
    got = value_raw_cp(pv)
    if got is None:
        return None
    return value_from_raw_cp(got, scale)


def mate_to_value_cp(mate: int) -> int:
    """sign * (30000 - min(|mate|, 1000)). Distance survives the round trip."""
    sign = 1 if mate > 0 else -1
    return sign * (VALUE_MATE_BASE - min(abs(mate), VALUE_MATE_MAX_DISTANCE))


def value_cp_is_mate(raw_cp: int) -> bool:
    return abs(raw_cp) >= VALUE_MATE_MIN


def value_cp_mate_distance(raw_cp: int) -> Optional[int]:
    """Recover |mate| in ply from a stored raw score, or None if it is a cp."""
    if not value_cp_is_mate(raw_cp):
        return None
    return VALUE_MATE_BASE - abs(raw_cp)


def value_raw_cp(pv: dict) -> Optional[int]:
    """Raw White-POV score for the record's `value_cp` field: cp clipped to
    +-VALUE_CP_CLIP, or the mate-distance encoding for a forced mate. None if
    the PV carries neither."""
    cp = pv.get("cp")
    if cp is not None:
        return max(-VALUE_CP_CLIP, min(VALUE_CP_CLIP, int(cp)))
    mate = pv.get("mate")
    if mate is None or int(mate) == 0:
        return None
    return mate_to_value_cp(int(mate))


def value_from_raw_cp(raw_cp: int, scale: float = VALUE_SCALE) -> float:
    """Squash a stored raw score at `scale`. The inverse-free half of the
    round trip, so training can re-derive the target at any scale.

    Mate distance is stored but deliberately NOT used here - every mate maps to
    +-VALUE_MATE regardless of depth, because a forced mate is a win whether it
    takes 1 ply or 40 and tapering would teach the head to hedge on long ones.
    The distance is kept in the record so that choice stays reversible.
    """
    if value_cp_is_mate(raw_cp):
        v = VALUE_MATE if raw_cp > 0 else -VALUE_MATE
    else:
        v = math.tanh(max(-VALUE_CP_CLIP, min(VALUE_CP_CLIP, raw_cp)) / scale)
    return max(-VALUE_CLAMP, min(VALUE_CLAMP, v))


def build_policy_target(entries: list[tuple[chess.Move, int]],
                        stm_is_white: bool,
                        temperature: float = DEFAULT_TEMPERATURE,
                        epsilon: float = DEFAULT_EPSILON,
                        max_pv: int = MAX_PV):
    """Sparse policy target from DEDUPED entries.

    Returns (idxs, probs, scores, ok) where `probs` sum to (1 - epsilon); the
    remaining epsilon is spread uniformly over legal moves at reconstruction
    time (Phase 3), so the dense target sums to 1.

    `ok` is False when the rel[0] == max(rel) invariant fails - the caller SKIPS
    the sample and increments a counter. Never raises: an assert in a streaming
    path kills long jobs and vanishes under `python -O`.

    Entries are stored per-ENTRY, not per-4096-index, so pv_idx may legitimately
    contain a repeated index (an underpromotion pair). Phase 3 accumulates them
    with += into the dense target, which reproduces this function exactly and
    keeps temperature a true collate-time transform.
    """
    stm = 1 if stm_is_white else -1
    rel = [score * stm for _, score in entries]

    if rel[0] != max(rel):
        return None, None, None, False

    # Truncate to the storage width, keeping the highest-scoring entries.
    if len(entries) > max_pv:
        order = sorted(range(len(entries)), key=lambda i: rel[i], reverse=True)[:max_pv]
        order.sort()
        entries = [entries[i] for i in order]
        rel = [rel[i] for i in order]

    mx = max(rel)
    exps = [math.exp((r - mx) / temperature) for r in rel]
    total = sum(exps)
    scale = (1.0 - epsilon) / total

    idxs = [move_index(m) for m, _ in entries]
    probs = [e * scale for e in exps]
    return idxs, probs, rel, True


def dense_policy(idxs, probs, legal_idxs, epsilon: float = DEFAULT_EPSILON,
                 size: int = 4096):
    """Reconstruct the dense target. Accumulates with += in BOTH loops.

    Assignment would lose mass on an underpromotion pair, and the epsilon loop
    would hand a promotion square 4x its share (four legal promotions map to one
    index). Shared by Phase 3 and the unit tests.
    """
    import numpy as np
    out = np.zeros(size, dtype=np.float32)
    for i, p in zip(idxs, probs):
        out[i] += p
    if legal_idxs is not None and len(legal_idxs):
        share = epsilon / len(legal_idxs)
        for i in legal_idxs:
            out[i] += share
    return out
