"""Unit tests for the multiPV label pipeline.

Runs under pytest, or standalone:  python data/multiPV/test_pipeline.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

import chess
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from labels import (  # noqa: E402
    CP_CLAMP, DEFAULT_EPSILON, DEFAULT_TEMPERATURE, MATE_MAX_DISTANCE,
    build_policy_target, dedup_entries, dense_policy, mate_to_policy_score,
    move_index, parse_block, pv_move_and_score, select_policy_block,
    value_target,
)
from record_format import RECORD_DTYPE, shard_name  # noqa: E402

T = DEFAULT_TEMPERATURE
EPS = DEFAULT_EPSILON


# --------------------------------------------------------------- stm sign

def test_stm_sign_white_to_move():
    """White to move: White-POV scores need NO flip; higher cp = better."""
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - -")
    entries = [(board.parse_uci("e2e4"), 150), (board.parse_uci("e2e3"), 100)]
    idxs, probs, rel, ok = build_policy_target(entries, True, T, EPS)
    assert ok
    assert rel == [150, 100], rel
    assert probs[0] > probs[1], "best White-POV move must carry more mass"


def test_stm_sign_black_to_move():
    """Black to move: White-POV scores are NEGATED, so the most negative wins."""
    board = chess.Board("4k3/4p3/8/8/8/8/8/4K3 b - -")
    # White-POV: -150 is great for Black, -100 less so. PVs are score-ordered
    # best-first for the mover, so -150 comes first.
    entries = [(board.parse_uci("e7e5"), -150), (board.parse_uci("e7e6"), -100)]
    idxs, probs, rel, ok = build_policy_target(entries, False, T, EPS)
    assert ok
    assert rel == [150, 100], rel
    assert probs[0] > probs[1]


def test_stm_sign_round_trip_is_symmetric():
    """Same magnitudes, mirrored side: identical probability vector."""
    b_w = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - -")
    b_b = chess.Board("4k3/4p3/8/8/8/8/8/4K3 b - -")
    ew = [(b_w.parse_uci("e2e4"), 150), (b_w.parse_uci("e2e3"), 100)]
    eb = [(b_b.parse_uci("e7e5"), -150), (b_b.parse_uci("e7e6"), -100)]
    _, pw, _, _ = build_policy_target(ew, True, T, EPS)
    _, pb, _, _ = build_policy_target(eb, False, T, EPS)
    for a, b in zip(pw, pb):
        assert abs(a - b) < 1e-12


# ------------------------------------------------------- promotion accumulation

def test_promotion_accumulation():
    """e8=Q and e8=N are distinct moves that share a 4096 index: mass must ADD."""
    # black king on g8, NOT e8 - the pawn needs an empty promotion square
    board = chess.Board("6k1/4P3/8/8/8/8/8/4K3 w - -")
    q = board.parse_uci("e7e8q")
    n = board.parse_uci("e7e8n")
    assert q != n and move_index(q) == move_index(n)

    kept, removed = dedup_entries([(q, 900), (n, 300)])
    assert removed == 0, "dedup must NOT collapse different promotion pieces"
    assert len(kept) == 2

    idxs, probs, _, ok = build_policy_target(kept, True, T, EPS)
    assert ok and idxs[0] == idxs[1]

    dense = dense_policy(idxs, probs, [], epsilon=0.0)
    # dense_policy accumulates in float32, so compare at float32 resolution
    assert abs(float(dense[idxs[0]]) - (probs[0] + probs[1])) < 1e-6


def test_epsilon_loop_accumulates_on_promotion_squares():
    """Four legal promotions share one index; the eps loop must give that index
    4x the per-move share, not 1x (assignment would silently drop three)."""
    # black king on g8, NOT e8 - the pawn needs an empty promotion square
    board = chess.Board("6k1/4P3/8/8/8/8/8/4K3 w - -")
    legal = list(board.legal_moves)
    legal_idx = [move_index(m) for m in legal]
    dense = dense_policy([], [], legal_idx, epsilon=EPS)
    assert abs(float(dense.sum()) - EPS) < 1e-6
    promo_idx = move_index(board.parse_uci("e7e8q"))
    share = EPS / len(legal_idx)
    assert abs(float(dense[promo_idx]) - 4 * share) < 1e-6


# ----------------------------------------------------- missing cp and mate

def test_pv_missing_both_cp_and_mate_drops_the_pair():
    board = chess.Board()
    assert pv_move_and_score(board, {"line": "e2e4"}, CP_CLAMP) is None
    assert pv_move_and_score(board, {"line": "e2e4", "cp": None, "mate": None}) is None
    assert pv_move_and_score(board, {"cp": 20}, CP_CLAMP) is None       # no line
    assert pv_move_and_score(board, {"line": "e2e4", "mate": 0}) is None  # mate 0


def test_parse_block_stays_aligned_when_a_pv_is_malformed():
    """The malformed PV must remove BOTH its move and its score - never one."""
    board = chess.Board()
    pvs = [{"cp": 30, "line": "e2e4 e7e5"},
           {"line": "d2d4 d7d5"},            # no score at all
           {"cp": 10, "line": "g1f3 g8f6"}]
    entries, dropped, removed = parse_block(board, pvs, CP_CLAMP)
    assert dropped == 1 and removed == 0
    assert [m.uci() for m, _ in entries] == ["e2e4", "g1f3"]
    assert [s for _, s in entries] == [30, 10]


# --------------------------------------------------------- sums to one

def test_distribution_sums_to_one():
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - -")
    entries = [(board.parse_uci("e2e4"), 150), (board.parse_uci("e2e3"), 100)]
    idxs, probs, _, ok = build_policy_target(entries, True, T, EPS)
    assert ok
    assert abs(sum(probs) - (1.0 - EPS)) < 1e-9
    legal_idx = [move_index(m) for m in board.legal_moves]
    dense = dense_policy(idxs, probs, legal_idx, epsilon=EPS)
    assert abs(float(dense.sum()) - 1.0) < 1e-4


def test_distribution_sums_to_one_after_truncation():
    """>8 unique PVs are truncated; the stored mass must still be exactly 1-eps."""
    board = chess.Board()
    legal = list(board.legal_moves)[:12]
    entries = [(m, 100 - 5 * i) for i, m in enumerate(legal)]
    idxs, probs, _, ok = build_policy_target(entries, True, T, EPS, max_pv=8)
    assert ok and len(idxs) == 8
    assert abs(sum(probs) - (1.0 - EPS)) < 1e-9


# ------------------------------------------------- mate / cp band disjointness

def test_mate_cp_bands_are_disjoint_at_the_boundary():
    """The whole point of the amended mapping: the slowest mate must still
    outrank the largest possible centipawn score."""
    m50 = mate_to_policy_score(MATE_MAX_DISTANCE)      # 11020 - 1000 = 10020
    assert m50 == 10020
    assert m50 > CP_CLAMP, f"M50 ({m50}) must beat cp clamp ({CP_CLAMP})"
    assert mate_to_policy_score(-MATE_MAX_DISTANCE) < -CP_CLAMP
    # distance ordering preserved
    assert mate_to_policy_score(1) > mate_to_policy_score(8) > m50
    # and beyond the cap it saturates rather than inverting
    assert mate_to_policy_score(999) == m50


def test_cp_is_clamped_into_its_band():
    board = chess.Board()
    _, s = pv_move_and_score(board, {"line": "e2e4", "cp": 20000}, CP_CLAMP)
    assert s == CP_CLAMP
    _, s = pv_move_and_score(board, {"line": "e2e4", "cp": -20000}, CP_CLAMP)
    assert s == -CP_CLAMP


def test_mate_outranks_huge_cp_in_a_mixed_block():
    """Regression for the 0.2475% invariant failure under the old mapping."""
    board = chess.Board()
    pvs = [{"mate": 5, "line": "e2e4 e7e5"}, {"cp": 20000, "line": "d2d4 d7d5"}]
    entries, _, _ = parse_block(board, pvs, CP_CLAMP)
    _, _, rel, ok = build_policy_target(entries, True, T, EPS)
    assert ok, "mate listed first must remain the max after mapping"
    assert rel[0] > rel[1]


# ------------------------------------------------------------ duplicate PVs

def test_dedup_removes_repeated_pv_keeping_first():
    board = chess.Board()
    e4 = board.parse_uci("e2e4")
    d4 = board.parse_uci("d2d4")
    kept, removed = dedup_entries([(e4, 30), (d4, 25), (d4, 25)])
    assert removed == 1
    assert [m.uci() for m, _ in kept] == ["e2e4", "d2d4"]
    # keep-FIRST: PVs are score-ordered, so the first copy has the better score
    kept2, removed2 = dedup_entries([(e4, 30), (d4, 25), (d4, 10)])
    assert removed2 == 1 and kept2[1][1] == 25


def test_duplicate_pv_does_not_shift_mass():
    """The actual bug: a duplicate doubles that move's exponentiated weight and
    overtakes the true best move whenever the gap is under T*ln2 (~20.8cp)."""
    board = chess.Board()
    best, second = board.parse_uci("e2e4"), board.parse_uci("d2d4")
    clean = [(best, 30), (second, 21)]                 # 9cp gap, inside T*ln2
    dirty = [(best, 30), (second, 21), (second, 21)]   # merge artifact

    deduped, removed = dedup_entries(dirty)
    assert removed == 1 and deduped == clean

    i_c, p_c, _, _ = build_policy_target(clean, True, T, EPS)
    i_d, p_d, _, _ = build_policy_target(deduped, True, T, EPS)
    assert i_c == i_d and all(abs(a - b) < 1e-12 for a, b in zip(p_c, p_d))

    # and confirm the un-deduped target really would have flipped the argmax
    i_bad, p_bad, _, _ = build_policy_target(dirty, True, T, EPS)
    bad = dense_policy(i_bad, p_bad, [], epsilon=0.0)
    good = dense_policy(i_c, p_c, [], epsilon=0.0)
    assert int(np.argmax(bad)) == move_index(second), "expected the corruption"
    assert int(np.argmax(good)) == move_index(best), "dedup must restore it"


def test_block_selection_requires_two_unique_moves():
    """A 2-PV block that is one move twice must NOT be selected - that would
    emit a one-hot target, the hard-label regime this dataset exists to escape."""
    board = chess.Board()
    evals = [
        # deepest, but collapses to ONE unique move after dedup
        {"depth": 40, "pvs": [{"cp": 30, "line": "e2e4 e7e5"},
                              {"cp": 30, "line": "e2e4 e7e5"}]},
        # shallower but genuinely multi-PV
        {"depth": 25, "pvs": [{"cp": 30, "line": "e2e4 e7e5"},
                              {"cp": 20, "line": "d2d4 d7d5"}]},
    ]
    sel = select_policy_block(board, evals, 20, CP_CLAMP)
    assert sel is not None
    entries, depth, _dropped, removed = sel
    assert depth == 25, "must fall through to the next-deepest qualifying block"
    assert len(entries) == 2
    assert removed >= 1, "the rejected block's duplicate should still be counted"


def test_block_with_no_qualifying_multipv_yields_none():
    board = chess.Board()
    evals = [{"depth": 40, "pvs": [{"cp": 30, "line": "e2e4"},
                                   {"cp": 30, "line": "e2e4"}]}]
    assert select_policy_block(board, evals, 20, CP_CLAMP) is None


# ------------------------------------------------------------- invariant

def test_invariant_violation_returns_not_ok_and_never_raises():
    board = chess.Board()
    entries = [(board.parse_uci("e2e4"), 10), (board.parse_uci("d2d4"), 99)]
    idxs, probs, rel, ok = build_policy_target(entries, True, T, EPS)
    assert ok is False and idxs is None


# ----------------------------------------------------------------- value

def test_value_is_white_pov_with_no_stm_conversion():
    scale = 250.0
    assert value_target({"cp": 300}, scale) > 0      # good for White
    assert value_target({"cp": -300}, scale) < 0     # good for Black
    # identical cp gives identical value regardless of side to move
    assert value_target({"cp": 300}, scale) == value_target({"cp": 300}, scale)


def test_value_clamped_and_mate_pinned():
    scale = 250.0
    assert abs(value_target({"mate": 3}, scale) - 0.995) < 1e-12
    assert abs(value_target({"mate": -3}, scale) + 0.995) < 1e-12
    assert abs(value_target({"cp": 999999}, scale)) <= 0.995
    assert value_target({"mate": 0}, scale) is None
    assert value_target({}, scale) is None


# ------------------------------------------------------------- round trip

def test_roundtrip_hand_checked_position():
    """FEN -> converter -> shard file -> Dataset -> tensors, checked by hand."""
    from pass_b_convert import convert_one
    from dataset import MultiPVDataset

    fen = "4k3/8/8/8/8/8/4P3/4K3 w - -"
    board = chess.Board(fen)
    n_legal_expected = len(list(board.legal_moves))
    assert n_legal_expected == 6, n_legal_expected   # Ke1 x4, e2e3, e2e4

    scale = 250.0
    rec = {"fen": fen, "evals": [{"knodes": 1, "depth": 30, "pvs": [
        {"cp": 150, "line": "e2e4 e8e7"},
        {"cp": 100, "line": "e2e3 e8e7"}]}]}
    cfg = dict(value_min_depth=26, policy_min_depth=20, temperature=T,
               epsilon=EPS, cp_clamp=CP_CLAMP, value_scale=scale, val_permille=5,
               n_train_shards=1, n_val_shards=1, shard_seed=0, bucket_rates={})

    out, info = convert_one(json.dumps(rec).encode(), cfg, Counter())
    assert out is not None, info

    assert int(out["has_policy"][0]) == 1
    assert int(out["n_pv"][0]) == 2
    assert int(out["n_legal"][0]) == 6
    assert abs(float(out["value"][0]) - math.tanh(150 / scale)) < 1e-3

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        p = Path(td) / shard_name("train", 0)
        p.write_bytes(out.tobytes())
        with MultiPVDataset(td, split="train", epsilon=EPS) as ds:
            assert len(ds) == 1
            s = ds[0]

            assert tuple(s["tokens"].shape) == (68,)
            assert s["tokens"].dtype == torch.int64
            assert s["value"].dtype == torch.float32
        assert s["policy"].shape == (4096,) and s["legal_mask"].shape == (4096,)
        assert float(s["legal_mask"].sum()) == 6.0
        assert abs(float(s["policy"].sum()) - 1.0) < 1e-3

        i_e4 = move_index(board.parse_uci("e2e4"))
        i_e3 = move_index(board.parse_uci("e2e3"))
        assert i_e4 == 12 * 64 + 28 and i_e3 == 12 * 64 + 20
        assert int(s["policy"].argmax()) == i_e4

        # exact expected masses
        w = [math.exp(0.0), math.exp((100 - 150) / T)]
        tot = sum(w)
        share = EPS / 6
        assert abs(float(s["policy"][i_e4]) - ((w[0] / tot) * (1 - EPS) + share)) < 2e-3
        assert abs(float(s["policy"][i_e3]) - ((w[1] / tot) * (1 - EPS) + share)) < 2e-3

        # every legal move carries at least the epsilon floor
        for m in board.legal_moves:
            assert float(s["policy"][move_index(m)]) > 0.0


def test_retemperature_reproduces_stored_target():
    """Temperature must be a pure collate-time transform."""
    import torch
    from dataset import retemperature

    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - -")
    entries = [(board.parse_uci("e2e4"), 150), (board.parse_uci("e2e3"), 100)]
    idxs, probs, rel, ok = build_policy_target(entries, True, T, EPS)
    assert ok
    legal_idx = [move_index(m) for m in board.legal_moves]
    stored = dense_policy(idxs, probs, legal_idx, epsilon=EPS)

    legal_mask = torch.zeros(1, 4096)
    legal_mask[0, legal_idx] = 1.0
    pv_idx = torch.zeros(1, 8, dtype=torch.int64)
    pv_score = torch.zeros(1, 8)
    pv_idx[0, :2] = torch.tensor(idxs)
    pv_score[0, :2] = torch.tensor([float(r) for r in rel])
    batch = {"pv_idx": pv_idx, "pv_score": pv_score,
             "n_pv": torch.tensor([2]), "has_policy": torch.tensor([1.0]),
             "legal_mask": legal_mask}

    got = retemperature(batch, T, EPS)[0].numpy()
    assert np.allclose(got, stored, atol=2e-3), np.abs(got - stored).max()

    # a different T must actually change the target, and still sum to 1
    hot = retemperature(batch, 200.0, EPS)[0].numpy()
    assert abs(float(hot.sum()) - 1.0) < 1e-3
    assert not np.allclose(hot, stored, atol=1e-3)


def test_record_dtype_is_packed_as_documented():
    # 377 for the brief's layout + 2 for value_cp (raw score, so the value scale
    # is a collate-time transform like temperature)
    assert RECORD_DTYPE.itemsize == 379, RECORD_DTYPE.itemsize


def test_raw_cp_round_trips_with_mate_sign_and_distance():
    from labels import (
        VALUE_CP_CLIP, VALUE_MATE_BASE, VALUE_MATE_MAX_DISTANCE, VALUE_MATE_MIN,
        value_cp_is_mate, value_cp_mate_distance, value_from_raw_cp, value_raw_cp,
    )
    assert value_raw_cp({"cp": 150}) == 150
    assert value_raw_cp({"cp": 99999}) == VALUE_CP_CLIP          # clipped
    assert value_raw_cp({"mate": 0}) is None
    assert value_raw_cp({}) is None

    # mate DISTANCE survives the round trip, and sign with it
    for m in (1, 3, 8, 40, 999, 5000):
        for sign in (1, -1):
            raw = value_raw_cp({"mate": sign * m})
            assert value_cp_is_mate(raw)
            assert (raw > 0) == (sign > 0)
            assert value_cp_mate_distance(raw) == min(m, VALUE_MATE_MAX_DISTANCE)
    assert value_raw_cp({"mate": 1}) == VALUE_MATE_BASE - 1
    # nearer mates encode LARGER magnitudes, so the ordering is not inverted
    assert value_raw_cp({"mate": 1}) > value_raw_cp({"mate": 40})

    # the mate band can never collide with a clipped cp
    assert VALUE_MATE_MIN > VALUE_CP_CLIP
    assert not value_cp_is_mate(VALUE_CP_CLIP)
    assert not value_cp_is_mate(-VALUE_CP_CLIP)
    assert value_cp_mate_distance(150) is None
    # and it fits in the int16 the record declares
    assert VALUE_MATE_BASE <= np.iinfo(np.int16).max

    for s in (203.0, 290.6806, 891.86):
        assert abs(value_from_raw_cp(150, s) - math.tanh(150 / s)) < 1e-12
        # every mate pins to +-0.995 regardless of distance
        assert value_from_raw_cp(value_raw_cp({"mate": 1}), s) == 0.995
        assert value_from_raw_cp(value_raw_cp({"mate": 40}), s) == 0.995
        assert value_from_raw_cp(value_raw_cp({"mate": -40}), s) == -0.995


def test_revalue_matches_stored_value_and_tracks_scale():
    """Changing the value scale must NOT require re-running Pass B."""
    from dataset import revalue
    from labels import value_from_raw_cp, value_raw_cp

    m_fast = value_raw_cp({"mate": 1})
    m_slow = value_raw_cp({"mate": -40})
    raw = [150, -400, 2000, m_fast, m_slow, 0]
    batch = {"value_cp": torch.tensor(raw, dtype=torch.float32)}
    for scale in (290.6806, 891.8631, 203.0):
        got = revalue(batch, scale).numpy()
        want = [value_from_raw_cp(r, scale) for r in raw]
        assert np.allclose(got, want, atol=1e-6), (scale, got, want)
    # a different scale genuinely moves the non-mate targets...
    a = revalue(batch, 290.6806).numpy()
    b = revalue(batch, 891.8631).numpy()
    assert abs(a[0] - b[0]) > 0.05
    # ...but never the mates, near or far
    assert a[3] == b[3] == 0.995 and a[4] == b[4] == -0.995


def test_value_scale_default_is_the_calibration_constant():
    """290.6806 is the Lc0 WDL calibration; acpl_elo_estimator inverts Q with it.
    If this changes, MCTS Q backup and the tuned C_PUCT change meaning."""
    from labels import VALUE_SCALE
    assert abs(VALUE_SCALE - 290.6806) < 1e-9
    # a pawn should read as roughly a two-thirds expected score, not a coin flip
    from labels import value_from_raw_cp
    assert 0.60 < (value_from_raw_cp(100, VALUE_SCALE) + 1) / 2 < 0.72


# ------------------------------------------------------------ colour mirror

def test_castling_lut_swaps_the_colour_pairs():
    from mirror import _castling_mirror
    #      K Q k q  =  8 4 2 1
    assert _castling_mirror(0b1000) == 0b0010      # K -> k
    assert _castling_mirror(0b0100) == 0b0001      # Q -> q
    assert _castling_mirror(0b1111) == 0b1111      # all rights: fixed point
    assert _castling_mirror(0b0000) == 0b0000
    assert _castling_mirror(0b1100) == 0b0011      # KQ -> kq
    for b in range(16):
        assert _castling_mirror(_castling_mirror(b)) == b


def test_mirror_is_an_involution_on_tokens_and_policy():
    """mirror(mirror(x)) == x, exactly, on every field that carries orientation."""
    from data.pgn_parallel import _board_to_tokens
    from mirror import (
        POLICY_PERM, VOCAB_MIRROR, mirror_dense_np, mirror_policy_index_np,
        mirror_tokens_np,
    )
    # the LUTs themselves
    assert np.array_equal(VOCAB_MIRROR[VOCAB_MIRROR], np.arange(len(VOCAB_MIRROR)))
    assert np.array_equal(POLICY_PERM[POLICY_PERM], np.arange(4096))

    fens = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    ]
    for fen in fens:
        t = np.asarray(_board_to_tokens(chess.Board(fen)), dtype=np.int64)
        assert np.array_equal(mirror_tokens_np(mirror_tokens_np(t)), t), fen

    idx = np.arange(4096)
    assert np.array_equal(mirror_policy_index_np(mirror_policy_index_np(idx)), idx)
    dense = np.random.default_rng(0).random(4096).astype(np.float32)
    assert np.array_equal(mirror_dense_np(mirror_dense_np(dense)), dense)


def test_mirror_matches_python_chess_board_mirror():
    """The token transform must agree with chess.Board.mirror(), which is the
    independent reference implementation of exactly this operation."""
    from data.pgn_parallel import _board_to_tokens
    from mirror import mirror_tokens_np

    fens = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ]
    for fen in fens:
        board = chess.Board(fen)
        got = mirror_tokens_np(
            np.asarray(_board_to_tokens(board), dtype=np.int64))
        want = np.asarray(_board_to_tokens(board.mirror()), dtype=np.int64)
        assert np.array_equal(got, want), (fen, got.tolist(), want.tolist())


def test_mirror_remaps_moves_the_way_board_mirror_does():
    """A move on the mirrored board must land on the mirrored policy index."""
    from mirror import mirror_policy_index_np

    board = chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    mb = board.mirror()
    for move in board.legal_moves:
        want = chess.Move(move.from_square ^ 56, move.to_square ^ 56,
                          promotion=move.promotion)
        assert want in mb.legal_moves, move.uci()
        got = int(mirror_policy_index_np(
            np.array([move_index(move)], dtype=np.int64))[0])
        assert got == move_index(want), move.uci()


def test_color_mirror_batch_is_consistent_and_reversible():
    """The collate-time transform: value negates, policy permutes, and applying
    it twice to the same rows restores the batch exactly."""
    from data.pgn_parallel import _board_to_tokens
    from dataset import color_mirror

    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - -")
    entries = [(board.parse_uci("e2e4"), 150), (board.parse_uci("e2e3"), 100)]
    idxs, probs, rel, ok = build_policy_target(entries, True, T, EPS)
    assert ok
    legal_idx = [move_index(m) for m in board.legal_moves]

    tokens = torch.tensor(
        [_board_to_tokens(board), _board_to_tokens(board)], dtype=torch.int64)
    policy = torch.zeros(2, 4096)
    policy[:] = torch.from_numpy(dense_policy(idxs, probs, legal_idx, EPS))
    legal_mask = torch.zeros(2, 4096)
    legal_mask[:, legal_idx] = 1.0
    pv_idx = torch.zeros(2, 8, dtype=torch.int64)
    pv_idx[:, :2] = torch.tensor(idxs)
    batch = {
        "tokens": tokens, "value": torch.tensor([0.5, 0.5]),   # exact in float32
        "value_cp": torch.tensor([150.0, 150.0]),
        "policy": policy, "legal_mask": legal_mask, "pv_idx": pv_idx,
        "n_pv": torch.tensor([2, 2]), "has_policy": torch.tensor([1.0, 1.0]),
    }
    mask = torch.tensor([True, False])
    out = color_mirror(batch, mask)

    # row 1 untouched
    assert torch.equal(out["tokens"][1], batch["tokens"][1])
    assert float(out["value"][1]) == 0.5
    # row 0 mirrored: value and raw cp negate, mass is conserved, argmax moves
    assert float(out["value"][0]) == -0.5
    assert float(out["value_cp"][0]) == -150.0
    assert abs(float(out["policy"][0].sum()) - 1.0) < 1e-4
    assert float(out["legal_mask"][0].sum()) == 6.0
    want_argmax = move_index(chess.Move(chess.E2 ^ 56, chess.E4 ^ 56))
    assert int(out["policy"][0].argmax()) == want_argmax

    # tokens agree with python-chess's own mirror
    want_tokens = torch.tensor(_board_to_tokens(board.mirror()), dtype=torch.int64)
    assert torch.equal(out["tokens"][0], want_tokens)

    # involution on the batch
    back = color_mirror(out, mask)
    for k in ("tokens", "value", "value_cp", "policy", "legal_mask", "pv_idx"):
        assert torch.equal(back[k], batch[k]), k


def test_retemperature_survives_the_mirror():
    """A mirrored batch must stay self-consistent: rebuilding the dense policy
    from the mirrored sparse entries reproduces the mirrored dense target."""
    from data.pgn_parallel import _board_to_tokens
    from dataset import color_mirror, retemperature

    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - -")
    entries = [(board.parse_uci("e2e4"), 150), (board.parse_uci("e2e3"), 100)]
    idxs, probs, rel, ok = build_policy_target(entries, True, T, EPS)
    legal_idx = [move_index(m) for m in board.legal_moves]

    pv_idx = torch.zeros(1, 8, dtype=torch.int64)
    pv_score = torch.zeros(1, 8)
    pv_idx[0, :2] = torch.tensor(idxs)
    pv_score[0, :2] = torch.tensor([float(r) for r in rel])
    legal_mask = torch.zeros(1, 4096)
    legal_mask[0, legal_idx] = 1.0
    batch = {
        "tokens": torch.tensor([_board_to_tokens(board)], dtype=torch.int64),
        "value": torch.tensor([0.4]), "value_cp": torch.tensor([150.0]),
        "policy": torch.from_numpy(
            dense_policy(idxs, probs, legal_idx, EPS)).unsqueeze(0),
        "legal_mask": legal_mask, "pv_idx": pv_idx, "pv_score": pv_score,
        "n_pv": torch.tensor([2]), "has_policy": torch.tensor([1.0]),
    }
    out = color_mirror(batch, torch.tensor([True]))
    rebuilt = retemperature(out, T, EPS)[0].numpy()
    assert np.allclose(rebuilt, out["policy"][0].numpy(), atol=2e-3), \
        np.abs(rebuilt - out["policy"][0].numpy()).max()


def test_collate_applies_mirror_at_the_requested_rate():
    from data.pgn_parallel import _board_to_tokens
    from dataset import MultiPVCollate

    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - -")
    tokens = torch.tensor(_board_to_tokens(board), dtype=torch.int64)
    sample = {
        "tokens": tokens, "value": torch.tensor(0.4),
        "value_cp": torch.tensor(150.0), "has_policy": torch.tensor(0.0),
        "policy": torch.zeros(4096), "legal_mask": torch.zeros(4096),
        "pv_idx": torch.zeros(8, dtype=torch.int64),
        "pv_score": torch.zeros(8), "n_pv": torch.tensor(0),
        "n_legal": torch.tensor(6),
    }
    samples = [dict(sample) for _ in range(512)]

    off = MultiPVCollate(mirror_prob=0.0)(samples)
    assert bool((off["value"] > 0).all()), "mirror_prob=0 must be a no-op"

    b = MultiPVCollate(mirror_prob=0.5, seed=7)(samples)
    frac = float((b["value"] < 0).float().mean())
    assert 0.4 < frac < 0.6, frac
    assert MultiPVCollate(mirror_prob=1.0)(samples)["value"].max() < 0


def test_dataset_does_not_pickle_its_memmaps():
    """Regression: Windows spawns DataLoader workers, so the Dataset is pickled.
    numpy pickles a memmap by MATERIALISING it - with every shard touched in the
    parent (which a num_workers=0 pass does), that pushed 11 GB through the spawn
    pipe and died as OSError [Errno 22]. __getstate__ must drop the handles."""
    import pickle
    from dataset import MultiPVDataset

    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - -")
    rec = np.zeros(64, dtype=RECORD_DTYPE)
    rec["n_legal"] = 6
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        for s in range(3):
            (Path(td) / shard_name("train", s)).write_bytes(rec.tobytes())
        with MultiPVDataset(td, split="train") as ds:
            _ = ds[0], ds[70], ds[140]              # force every memmap open
            assert all(m is not None for m in ds._maps)
            blob = pickle.dumps(ds)
            # the payload must be metadata only, not 3 shards of records
            assert len(blob) < 4096, len(blob)
            revived = pickle.loads(blob)
            try:
                assert all(m is None for m in revived._maps)
                assert len(revived) == len(ds)
                assert torch.equal(revived[70]["tokens"], ds[70]["tokens"])
            finally:
                revived.close()


def test_subset_is_deterministic_and_bounded():
    from dataset import MultiPVDataset
    rec = np.zeros(50, dtype=RECORD_DTYPE)
    rec["n_legal"] = 1
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        for s in range(2):
            (Path(td) / shard_name("train", s)).write_bytes(rec.tobytes())
        with MultiPVDataset(td, split="train") as full:
            assert len(full) == 100
        with MultiPVDataset(td, split="train", subset=30) as sub:
            assert len(sub) == 30
            try:
                sub[30]
                raise AssertionError("subset must not index past its length")
            except IndexError:
                pass
        with MultiPVDataset(td, split="train", subset=10**9) as big:
            assert len(big) == 100, "subset larger than the corpus must clamp"


# ------------------------------------------------------------------ runner

def _main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            import traceback
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
