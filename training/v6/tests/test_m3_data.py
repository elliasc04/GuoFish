"""M3 gates: S1 (reader parity with v5), S4 (sampler), S5 (canonical transform),
plus the v2 reader and strata on synthetic shards.

S1/S5 read the frozen val set and the live 90M train shards (read-only,
random or small contiguous reads). They take a few minutes at CPU.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from core.guofish_net.tokenizers import board_from_v5_tokens, tokens_canonical_65
from training.v6.config import load_config
from training.v6.config.schema import MixtureConfig
from training.v6.data.batch import (
    BatchBuilder, StreamDataset, dense_legal, dense_policy, mirror_records,
)
from training.v6.data.formats import REPO, V2_DTYPE, dtype_descr
from training.v6.data.mixture import Mixture, apportion, feistel_perm
from training.v6.data.reader import ShardSet
from training.v6.data.strata import field_of, load_strata, match

FROZEN = REPO / "data/processed/val_frozen_90m_v1"
FROZEN_STRATA = REPO / "data/processed/strata/val_frozen_90m_v1_val.strata2.npy"
C90 = REPO / "data/processed/multipv_90m"
M90 = REPO / "data/multiPV/manifests/dataset_manifest_90m.json"
sys.path.insert(0, str(REPO / "data/multiPV"))
from dataset import MultiPVDataset, color_mirror  # noqa: E402
from mirror import POLICY_PERM  # noqa: E402


def _digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------- S1

def _s1_compare(v5_ds, ss, idx, builder_on, builder_off, stats):
    samples = [v5_ds[int(i)] for i in idx]
    v5 = default_collate(samples)
    rec = ss.read(idx)
    for builder in (builder_off, builder_on):
        sample_idx = stats["next_sample"] + np.arange(len(idx))
        v6 = builder(rec, sample_idx)
        ref = color_mirror(v5, v6["mirrored"]) if v6["mirrored"].any() else v5
        assert torch.equal(ref["tokens"], v6["tokens"].long())
        assert torch.equal(ref["policy"], v6["policy"])
        assert torch.equal(ref["legal_mask"], v6["legal"].float())
        assert torch.equal(ref["value"], v6["value"])
        assert torch.equal(ref["value_cp"], v6["value_cp"].float())
        assert torch.equal(ref["has_policy"], v6["has_policy"].float())
        assert torch.equal(ref["pv_idx"], v6["pv_idx"])
        assert torch.equal(ref["n_pv"], v6["n_pv"]) and torch.equal(ref["n_legal"], v6["n_legal"])
        stats["mirrored"] += int(v6["mirrored"].sum())
    stats["records"] += len(idx)
    stats["policy_rows"] += int(v6["has_policy"].sum())
    stats["next_sample"] += len(idx)


def test_s1_reader_parity_with_v5():
    """Bit-identical tokens, dense targets, legal masks and values vs v5's
    MultiPVDataset (+ its color_mirror), 100k seeded records: 50k frozen val +
    50k 90M train (random reads), mirror off and on."""
    on = BatchBuilder("v5_68", 0.5, seed=20260924, epsilon=0.05, temperature=None)
    off = BatchBuilder("v5_68", 0.0, seed=20260924, epsilon=0.05, temperature=None)
    stats = {"records": 0, "policy_rows": 0, "mirrored": 0, "next_sample": 0}
    for shard_dir, split, manifest, seed in [(FROZEN, "val", None, 1), (C90, "train", M90, 2)]:
        v5_ds = MultiPVDataset(shard_dir, split=split)
        ss = ShardSet(shard_dir, split, manifest)
        assert len(ss) == len(v5_ds)
        idx = np.random.default_rng([20260924, seed]).choice(len(ss), 50_000, replace=False)
        for chunk in np.array_split(idx, 50):
            _s1_compare(v5_ds, ss, chunk, on, off, stats)
        v5_ds.close()
        ss.close()
    print(f"\nS1: {stats['records']:,} records ({stats['policy_rows']:,} with policy), "
          f"{stats['mirrored']:,} mirrored rows compared; all fields bit-identical")
    assert stats["records"] == 100_000 and stats["mirrored"] > 40_000


# --------------------------------------------------------------------- S4

GROUPS = MixtureConfig(groups=[
    {"name": "policy", "where": {"label": "multipv"}, "share": 0.65},
    {"name": "endgame_vo", "where": {"label": "value_only", "bucket": ["le5", "6_14"]}, "share": 0.2},
])   # rest (all other value-only rows) gets 0.15


@pytest.fixture(scope="module")
def frozen_strata():
    return np.asarray(load_strata(FROZEN_STRATA, 452_405))


def test_s4_exact_shares_10000_batches(frozen_strata):
    mix = Mixture(GROUPS, 452_405, 512, seed=7, strata=frozen_strata)
    assert mix.names == ["policy", "endgame_vo", "rest"]
    member_of = np.full(452_405, -1)
    for g, where in enumerate([{"label": "multipv"},
                               {"label": "value_only", "bucket": ["le5", "6_14"]}]):
        member_of[match(frozen_strata, where)] = g
    member_of[member_of < 0] = 2
    cum = np.zeros(3, dtype=np.int64)
    h = hashlib.sha256()
    for k in range(10_000):
        rec, sample, gid = mix.batch(k)
        # the records' own strata put each one in the group it was drawn for
        assert np.array_equal(member_of[rec], gid)
        c = np.bincount(gid, minlength=3)
        assert c.sum() == 512
        assert np.array_equal(c, mix.counts(k))
        assert (np.abs(c - mix.shares * 512) < 1.0).all()           # floor or ceil
        cum += c
        assert (np.abs(cum - mix.shares * 512 * (k + 1)) < 1.0).all()  # long run exact
        assert np.array_equal(sample, k * 512 + np.arange(512))
        h.update(rec.tobytes())
    print(f"\nS4: 10,000 batches, final shares {np.round(cum / cum.sum(), 6)} vs "
          f"{np.round(mix.shares, 6)}; passes {mix.passes(10_000 * 512)}; stream {h.hexdigest()[:16]}")
    # identical stream from a fresh sampler
    mix2 = Mixture(GROUPS, 452_405, 512, seed=7, strata=frozen_strata)
    h2 = hashlib.sha256()
    for k in range(10_000):
        h2.update(mix2.batch(k)[0].tobytes())
    assert h2.hexdigest() == h.hexdigest()


def test_s4_each_pass_is_a_permutation_of_the_group(frozen_strata):
    mix = Mixture(GROUPS, 452_405, 512, seed=7, strata=frozen_strata)
    members = np.flatnonzero(match(frozen_strata, {"label": "value_only", "bucket": ["le5", "6_14"]}))
    n = len(members)
    drawn = []
    k = 0
    while sum(len(d) for d in drawn) < 2 * n + 512:
        rec, _, gid = mix.batch(k)
        drawn.append(rec[gid == 1])
        k += 1
    stream = np.concatenate(drawn)
    assert np.array_equal(np.sort(stream[:n]), members)            # pass 0
    assert np.array_equal(np.sort(stream[n:2 * n]), members)       # pass 1
    assert not np.array_equal(stream[:n], stream[n:2 * n])         # fresh order


def test_feistel_is_a_permutation():
    for n in (1, 2, 3, 17, 1000, 65_537, 271_876):
        p = feistel_perm(np.arange(n), n, key=12345)
        assert np.array_equal(np.sort(p), np.arange(n))
    p = feistel_perm(np.arange(100_000), 100_000, key=1)
    assert abs(np.corrcoef(np.arange(100_000), p)[0, 1]) < 0.02       # not the identity


def test_apportion_is_exact():
    s = np.array([0.65, 0.25, 0.1])
    for total in (0, 1, 511, 512, 10 ** 9 + 7):
        a = apportion(s, total)
        assert a.sum() == total and (np.abs(a - s * total) < 1).all()


def _stream_digest(ds, ks, workers):
    dl = DataLoader(ds, batch_size=None, sampler=ks, num_workers=workers,
                    persistent_workers=False)
    return [_digest(b["record_index"].numpy(), b["tokens"].numpy(), b["policy"].numpy(),
                    b["legal"].numpy(), b["value"].numpy(), b["mirrored"].numpy()) for b in dl]


def test_s4_stream_identical_across_runs_workers_and_resume(frozen_strata):
    """Same batches whatever the worker count, and a resumed stream (a new
    sampler started at micro-batch r) equals the tail of the uninterrupted one,
    including at the micro-batch where a group starts a new pass."""
    ss = ShardSet(FROZEN, "val")
    builder = BatchBuilder("v5_68", 0.5, seed=7, epsilon=0.05, temperature=None)

    def ds():
        return StreamDataset(ss, Mixture(GROUPS, 452_405, 512, seed=7, strata=frozen_strata),
                             builder, n_micro=10_000)

    # the micro-batch in which `endgame_vo` wraps into pass 1
    mix = Mixture(GROUPS, 452_405, 512, seed=7, strata=frozen_strata)
    n1 = mix.sizes[1]
    wrap = next(k for k in range(10_000) if mix.consumed(512 * (k + 1))[1] > n1)
    assert mix.consumed(512 * wrap)[1] <= n1 < mix.consumed(512 * (wrap + 1))[1]
    ks = list(range(wrap - 3, wrap + 3))
    full = _stream_digest(ds(), ks, workers=0)
    assert _stream_digest(ds(), ks, workers=0) == full                 # second run
    assert _stream_digest(ds(), ks, workers=2) == full                 # 2 worker processes
    assert _stream_digest(ds(), ks[3:], workers=0) == full[3:]         # resume at the wrap
    ss.close()


# --------------------------------------------------------------------- S5

def _s5_records():
    """1,000,000 v1 records: all 452,405 frozen val + 547,595 train records read
    as small contiguous blocks from seeded offsets. Yields (rec, synthetic hard_move)."""
    ss = ShardSet(FROZEN, "val")
    for s in range(0, len(ss), 4096):
        yield ss.read(np.arange(s, min(len(ss), s + 4096)))
    ss.close()
    tr = ShardSet(C90, "train", M90)
    rng = np.random.default_rng(20260924)
    left = 1_000_000 - 452_405
    while left:
        n = min(4096, left)
        start = int(rng.integers(0, len(tr) - n))
        yield tr.read(np.arange(start, start + n))
        left -= n
    tr.close()


def _with_hard_moves(rec):
    """v1 has no hard_move; give every row one (pv_idx[0] or a legal move) so
    the remap is exercised, and some -1s."""
    rec = rec.copy()
    pick = np.where(rec["n_pv"] > 0, rec["pv_idx"][:, 0], rec["legal_idx"][:, 0])
    pick = np.where(rec["n_legal"] > 0, pick, -1)
    pick[::7] = -1
    rec["hard_move"] = pick
    return rec


def test_s5_canonical_transform_1m():
    canon = BatchBuilder("canonical_65", 0.0, seed=0, epsilon=0.05, temperature=None)
    n = n_black = n_hard = 0
    for rec in _s5_records():
        rec = _with_hard_moves(rec)
        allm = np.ones(len(rec), dtype=bool)
        mrec = mirror_records(rec, allm)
        # remaps round-trip exactly
        assert mirror_records(mrec, allm).tobytes() == rec.tobytes()
        pol = dense_policy(rec, 0.05, None)
        assert np.array_equal(dense_policy(mrec, 0.05, None), pol[:, POLICY_PERM])
        assert np.array_equal(dense_legal(mrec), dense_legal(rec)[:, POLICY_PERM])
        # canonical(x) == canonical(mirror(x)) on every output
        a, b = canon(rec, np.arange(len(rec))), canon(mrec, np.arange(len(rec)))
        for k in ("tokens", "policy", "legal", "value", "value_cp", "hard_move", "has_policy"):
            assert torch.equal(a[k], b[k]), k
        # value sign: canonical value is side-to-move POV
        black = rec["tokens"][:, 64] == 14
        stm = np.where(black, -1.0, 1.0).astype(np.float32)
        assert np.array_equal(a["value"].numpy(), rec["value"].astype(np.float32) * stm)
        # hard_move: Black rows remapped, White rows untouched, -1 kept
        hm = rec["hard_move"].astype(np.int64)
        want = np.where(black & (hm >= 0), POLICY_PERM[np.clip(hm, 0, 4095)], hm)
        assert np.array_equal(a["hard_move"].numpy(), want)
        n += len(rec)
        n_black += int(black.sum())
        n_hard += int((hm >= 0).sum())
    print(f"\nS5: {n:,} records ({n_black:,} Black to move, {n_hard:,} hard moves): "
          f"canonical(x)==canonical(mirror(x)), remaps round-trip, value sign correct")
    assert n == 1_000_000


def test_s5_worker_path_matches_reference_tokenizer_100k():
    canon = BatchBuilder("canonical_65", 0.0, seed=0, epsilon=0.05, temperature=None)
    n = n_ep_token = n_ep_legal = n_castle = 0
    left = 100_000
    for rec in _s5_records():
        if not left:
            break
        rec = rec[:left] if len(rec) > left else rec
        toks = canon(rec, np.arange(len(rec)))["tokens"].numpy()
        for raw, got in zip(rec["tokens"], toks):
            board = board_from_v5_tokens(raw)
            assert np.array_equal(tokens_canonical_65(board), got), board.fen()
        n += len(rec)
        n_ep_token += int((rec["tokens"][:, 66] != 31).sum())
        n_ep_legal += int((toks == 15).any(1).sum())
        n_castle += int((toks == 13).any(1).sum())
        left -= len(rec)
    print(f"\nS5 FEN: {n:,} records match the reference tokenizer; {n_ep_token:,} carry an "
          f"ep file, {n_ep_legal:,} with a legal ep capture; {n_castle:,} with castling rooks")
    assert n == 100_000 and n_ep_legal > 0


def _record_from_board(board):
    from core.guofish_net.tokenizers import tokens_v5_68
    rec = np.zeros(1, dtype=V2_DTYPE)
    rec["tokens"] = tokens_v5_68(board)
    legal = [m.from_square * 64 + m.to_square for m in board.legal_moves]
    rec["n_legal"] = len(legal)
    rec["legal_idx"][0, :len(legal)] = legal
    rec["hard_move"] = -1
    return rec


@pytest.mark.parametrize("fen, ep_token", [
    ("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3", True),   # legal
    ("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1", False),                               # no pawn to capture
    ("8/8/8/K1pP3r/8/8/8/7k w - c6 0 2", False),                               # pinned: dxc6 exposes Ka5
    ("8/8/8/KPp4r/8/8/8/7k w - c6 0 2", False),                                # capturer pinned on the rank
    ("7k/8/8/1pP5/8/8/8/K7 w - b6 0 2", True),                                 # White to move, legal
])
def test_canonical_ep_worker_path_hand_positions(fen, ep_token):
    """The worker infers ep legality from the stored legal moves; real data
    only ever carries a legal ep square (S5: 144/144), so the illegal cases
    are pinned here against the python-chess reference."""
    import chess
    board = chess.Board(fen)
    assert board.has_legal_en_passant() == ep_token
    canon = BatchBuilder("canonical_65", 0.0, seed=0, epsilon=0.05, temperature=None)
    got = canon(_record_from_board(board), np.arange(1))["tokens"].numpy()[0]
    assert np.array_equal(got, tokens_canonical_65(board))
    assert (15 in got) == ep_token


# ------------------------------------------------- v2 reader + strata, synthetic

def test_v2_reader_and_strata_on_synthetic_shards(tmp_path):
    """Synthetic v2 shards built from real v1 records plus hard_move/origin/
    src_line, then build_strata over them against a synthetic Pass A index and
    90M manifest: depth_tier from max_depth at src_line, in_90m from the replay."""
    from pass_a_index import INDEX_DTYPE
    from pass_b_convert import build_selection
    from training.v6.data.strata import compute_strata
    src = ShardSet(FROZEN, "val").read(np.arange(20_000))
    rec = _with_hard_moves(src)
    rec["origin"] = np.array([1, 2, 0, 0, 0])[np.arange(20_000) % 5]
    rec["has_policy"] = np.where(rec["origin"] > 0, 0, rec["has_policy"])
    rec["value_depth"] = 26
    rec["src_line"] = np.arange(20_000, dtype=np.uint32) * 3
    shards = tmp_path / "c2"
    shards.mkdir()
    for i, part in enumerate(np.array_split(rec, 3)):
        part.tofile(shards / f"train_{i:04d}.bin")
    (shards / "manifest.json").write_text(json.dumps(
        {"record_dtype": dtype_descr(V2_DTYPE), "record_size_bytes": V2_DTYPE.itemsize}))
    ss = ShardSet(shards, "train")
    assert ss.format == "v2" and len(ss) == 20_000
    assert ss.read(np.arange(20_000)).tobytes() == rec.tobytes()
    with pytest.raises(ValueError, match="need the index"):
        compute_strata(rec[:10])

    rng = np.random.default_rng(0)
    ix = np.zeros(60_000, dtype=INDEX_DTYPE)
    ix["piece_count"] = rng.integers(3, 33, len(ix))
    ix["max_depth"] = rng.integers(24, 31, len(ix))
    ix["policy_depth"] = rng.integers(18, 23, len(ix))
    ix.tofile(tmp_path / "index.bin")
    rates = ({"<=5": 0.9, "6-14": 0.5, "15-27": 0.4, ">=28": 0.6},
             {"<=5": 0.3, "6-14": 0.2, "15-27": 0.3, ">=28": 0.1})
    sel90 = build_selection(tmp_path / "index.bin", {
        "value_min_depth": 26, "policy_min_depth": 20,
        "bucket_rates_policy": rates[0], "bucket_rates_value_only": rates[1]}, 5)
    (tmp_path / "m90.json").write_text(json.dumps({
        "seed": 5, "value_min_depth": 26, "policy_min_depth": 20,
        "bucket_sampling_rates_policy": rates[0], "bucket_sampling_rates_value_only": rates[1],
        "selected_lines": len(sel90)}))
    cmd = [sys.executable, "-m", "training.v6.tools.build_strata", "--shards", str(shards),
           "--split", "train", "--out-dir", str(tmp_path / "strata"), "--index",
           str(tmp_path / "index.bin"), "--m90", str(tmp_path / "m90.json")]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True)
    out = tmp_path / "strata" / "c2_train.strata2.npy"
    codes = np.asarray(load_strata(out, 20_000))
    label = field_of(codes, "label")
    has_pol = rec["has_policy"] > 0
    assert np.array_equal(label == 0, has_pol)
    assert np.array_equal(label == 1, ~has_pol & (rec["hard_move"] >= 0))
    assert np.array_equal(field_of(codes, "origin"), rec["origin"])
    assert (label == 1).sum() > 0 and (field_of(codes, "origin") == 2).sum() == 4_000
    ln = rec["src_line"].astype(np.int64)
    tier = field_of(codes, "depth_tier")
    assert np.array_equal(tier, np.where(ix["max_depth"][ln] >= 26, 0, 1))
    in90 = field_of(codes, "in_90m")
    assert np.array_equal(in90 == 1, np.isin(ln, sel90))
    assert set(tier.tolist()) == {0, 1} and 0 < in90.sum() < 20_000
    assert match(codes, {"depth_tier": "new"}).sum() == (tier == 1).sum()
    assert match(codes, {"in_90m": 1, "origin": ["ply1", "ply2"]}).sum() == (
        (in90 == 1) & (rec["origin"] > 0)).sum()
    again = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert again.returncode != 0 and "refusing to overwrite" in again.stderr


def test_strata_mirror_invariant_and_frozen_counts(frozen_strata):
    ss = ShardSet(FROZEN, "val")
    rec = ss.read(np.arange(0, 452_405, 11))
    from training.v6.data.strata import compute_strata
    assert np.array_equal(compute_strata(rec),
                          compute_strata(mirror_records(rec, np.ones(len(rec), dtype=bool))))
    assert np.array_equal(compute_strata(rec), frozen_strata[::11])
    assert int((field_of(frozen_strata, "label") == 0).sum()) == 271_876
    assert (field_of(frozen_strata, "depth_tier") == 2).all()             # v1 records
    assert (field_of(frozen_strata, "in_90m") == 1).all()
    assert (field_of(frozen_strata, "origin") == 0).all()
    ss.close()


def test_grouped_mixture_refuses_overlap(frozen_strata):
    cfg = MixtureConfig(groups=[{"name": "a", "where": {"label": "multipv"}, "share": 0.5},
                                {"name": "b", "where": {"bucket": "ge28"}, "share": 0.3}])
    with pytest.raises(ValueError, match="overlap"):
        Mixture(cfg, 452_405, 512, seed=0, strata=frozen_strata)
    cfg = MixtureConfig(groups=[{"name": "a", "where": {"label": "hard_only"}, "share": 0.5}])
    with pytest.raises(ValueError, match="no records"):
        Mixture(cfg, 452_405, 512, seed=0, strata=frozen_strata)


def test_config_mixture_loads():
    c = load_config(REPO / "training/v6/config/configs/base.yaml",
                    ["mixture.groups=[{name: policy, where: {label: multipv}, share: 0.75}]"])
    assert c.mixture.groups[0].share == 0.75 and c.mixture.rest_share == pytest.approx(0.25)
