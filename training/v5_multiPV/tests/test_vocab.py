"""The vocabulary guardrail: 43 is 41 active token IDs plus 2 reserved rows.

Runs under pytest, or standalone:
    python training/v5_multiPV/tests/test_vocab.py
    python training/v5_multiPV/tests/test_vocab.py --shards data/processed/multipv_90m

WHY THIS FILE EXISTS
====================
`vocab_size` is an `nn.Embedding` row count, and getting it wrong fails in two
opposite and equally quiet ways. Too small is loud on the first bad batch
(`IndexError: index out of range in self`). Too LARGE is silent forever - the
extra rows just never receive a gradient - so an inflated vocab survives a full
training run and only shows up as wasted parameters and a checkpoint whose
`config` misdescribes the encoding it was trained under.

The specific inflation this pins against is DOUBLE-COUNTING THE ACTIVE STATE
TOKENS: the encoding spends four of its 68 positions on non-square state (side
to move, castling, en passant, CLS), and it is easy to write a vocabulary sum
that charges those bands twice - once as "state tokens" and once inside their
own ranges - or to add the 64 square POSITIONS, which are sequence slots and
not vocabulary entries at all. Either mistake produces a plausible-looking
number. So the bands are reconstructed here from `data/pgn_parallel.py`'s own
constants and asserted PAIRWISE DISJOINT and CONTIGUOUS, which is the property
a double count breaks and a simple total does not.

The scheme, from data/pgn_parallel.py:

    0        empty square                            1
    1-12     white then black pieces (P..K, p..k)   12
    13-14    side to move                            2
    15-30    castling rights, 4-bit                 16
    31       no en passant                           1
    32-39    en passant file a-h                     8
    40       CLS                                     1
                                                   ---
                                                    41 active, 0..40
    41-42    reserved                                2
                                                   ---
                                                    43 = vocab_size

The reserved pair is DELIBERATE and documented at the top of pgn_parallel.py, so
this file asserts it stays exactly two rows rather than trimming it: every
shipped GuoFish checkpoint back to v2 has 43 embedding rows, and shrinking to 41
would make each of them unloadable for the sake of 768 parameters.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_V5 = _HERE.parent
_ROOT = _V5.parents[1]
_DATA = _ROOT / "data" / "multiPV"
for _p in (str(_V5), str(_DATA), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                      # noqa: E402

from data.pgn_parallel import (                                   # noqa: E402
    SEQ_LENGTH, TOKEN_BLACK_TO_MOVE, TOKEN_CASTLING_BASE, TOKEN_CLS,
    TOKEN_EP_BASE, TOKEN_EP_NONE, TOKEN_WHITE_TO_MOVE, VOCAB_SIZE,
    _board_to_tokens,
)
from mirror import VOCAB_MIRROR                                    # noqa: E402
from model_v5 import ModelConfig, build_model                      # noqa: E402

# The 68 sequence slots. Positions, not vocabulary - naming them keeps the two
# ideas apart in the assertions below.
POS_STM, POS_CASTLING, POS_EP, POS_CLS = 64, 65, 66, 67

N_RESERVED = 2


def vocabulary_bands() -> dict[str, range]:
    """The token-ID range each part of the encoding claims.

    Built from pgn_parallel's constants rather than restated, so a constant that
    moves there moves here too and the disjointness test does the work.
    """
    return {
        "empty": range(0, 1),
        "white_pieces": range(1, 7),
        "black_pieces": range(7, 13),
        "side_to_move": range(TOKEN_WHITE_TO_MOVE, TOKEN_BLACK_TO_MOVE + 1),
        "castling": range(TOKEN_CASTLING_BASE, TOKEN_CASTLING_BASE + 16),
        "ep_none": range(TOKEN_EP_NONE, TOKEN_EP_NONE + 1),
        "ep_file": range(TOKEN_EP_BASE, TOKEN_EP_BASE + 8),
        "cls": range(TOKEN_CLS, TOKEN_CLS + 1),
    }


# ---------------------------------------------------------------------------
# the guardrail proper
# ---------------------------------------------------------------------------
def test_bands_are_pairwise_disjoint():
    """No token ID is claimed by two bands. THIS is the double-count check.

    A vocabulary that counts the active state tokens twice shows up here as an
    overlap, and the message names both bands - which a bare total never could.
    """
    bands = vocabulary_bands()
    names = sorted(bands)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(bands[a]) & set(bands[b])
            assert not shared, (
                f"token IDs {sorted(shared)} are claimed by both '{a}' and "
                f"'{b}'; the vocabulary double-counts them")


def test_bands_are_contiguous_from_zero():
    """The active band is exactly 0..40 with no holes.

    A hole would mean an embedding row that can never be reached - the same
    wasted-capacity failure as an inflated vocab, arriving by a different route.
    """
    covered = sorted({t for r in vocabulary_bands().values() for t in r})
    assert covered == list(range(len(covered))), (
        f"active token IDs are not contiguous from 0: "
        f"missing {sorted(set(range(max(covered) + 1)) - set(covered))}")
    assert covered[-1] == TOKEN_CLS


def test_active_count_is_41_and_vocab_is_that_plus_reserved():
    bands = vocabulary_bands()
    per_band = {k: len(v) for k, v in bands.items()}
    assert per_band == {
        "empty": 1, "white_pieces": 6, "black_pieces": 6, "side_to_move": 2,
        "castling": 16, "ep_none": 1, "ep_file": 8, "cls": 1,
    }
    n_active = sum(per_band.values())
    assert n_active == 41, f"expected 41 active token IDs, counted {n_active}"
    assert VOCAB_SIZE == n_active + N_RESERVED == 43


def test_square_positions_are_not_vocabulary_entries():
    """64 squares are sequence SLOTS. Charging them to the vocabulary is the
    other way this number inflates, and it lands on a plausible 105."""
    assert SEQ_LENGTH == 64 + 4
    assert VOCAB_SIZE != SEQ_LENGTH - 64 + 41
    assert VOCAB_SIZE < 64, (
        "vocab_size has grown past the square count; something is charging "
        "sequence positions to the vocabulary")


def test_model_default_matches_the_encoding():
    assert ModelConfig().vocab_size == VOCAB_SIZE
    m = build_model()
    assert m.embedding.num_embeddings == VOCAB_SIZE
    assert m.embedding.weight.shape[0] == VOCAB_SIZE


def test_reserved_rows_are_exactly_two_and_unreachable():
    """41 and 42 exist, are documented, and no encoder path can emit them."""
    reserved = set(range(TOKEN_CLS + 1, VOCAB_SIZE))
    assert reserved == {41, 42}
    bands = {t for r in vocabulary_bands().values() for t in r}
    assert not (reserved & bands)


def test_mirror_lut_is_a_permutation_of_the_full_vocabulary():
    """The collate-time mirror indexes tokens through VOCAB_MIRROR, so a LUT
    shorter than the vocabulary would silently truncate and a non-permutation
    would break the involution. Both are cheap to pin here."""
    assert VOCAB_MIRROR.shape == (VOCAB_SIZE,)
    assert sorted(VOCAB_MIRROR.tolist()) == list(range(VOCAB_SIZE))
    assert np.array_equal(VOCAB_MIRROR[VOCAB_MIRROR], np.arange(VOCAB_SIZE))
    # Reserved rows are fixed points, so mirroring cannot manufacture one.
    assert VOCAB_MIRROR[41] == 41 and VOCAB_MIRROR[42] == 42


# ---------------------------------------------------------------------------
# against the encoder, and against the corpus the run will actually read
# ---------------------------------------------------------------------------
def test_encoder_emits_only_active_tokens_in_the_documented_slots():
    chess = pytest.importorskip("chess")
    boards = [chess.Board()]
    b = chess.Board()
    for uci in ("e2e4", "c7c5", "g1f3", "d7d6", "f1b5", "c8d7", "e1g1"):
        b.push_uci(uci)
        boards.append(b.copy())
    # A position with an en-passant target and one with no castling rights.
    boards.append(chess.Board(
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"))
    boards.append(chess.Board("8/5k2/8/8/8/8/5K2/8 w - - 0 1"))

    bands = vocabulary_bands()
    squares = set(bands["empty"]) | set(bands["white_pieces"]) | set(bands["black_pieces"])
    for board in boards:
        toks = _board_to_tokens(board)
        assert len(toks) == SEQ_LENGTH
        assert all(0 <= t <= TOKEN_CLS for t in toks), \
            f"token outside the active band for {board.fen()}"
        assert set(toks[:64]) <= squares
        assert toks[POS_STM] in bands["side_to_move"]
        assert toks[POS_CASTLING] in bands["castling"]
        assert toks[POS_EP] in (set(bands["ep_none"]) | set(bands["ep_file"]))
        assert toks[POS_CLS] == TOKEN_CLS


def _shard_dir() -> Path | None:
    for name in ("multipv_90m", "multipv"):
        p = _ROOT / "data" / "processed" / name
        if p.exists() and any(p.glob("train_*.bin")):
            return p
    return None


def test_corpus_tokens_stay_inside_the_active_band():
    """The empirical half: the shards this run trains on must agree with the
    scheme. If a converter ever emitted a token in the reserved band, the model
    would train on it happily and nothing else would notice."""
    shard_dir = _shard_dir()
    if shard_dir is None:
        pytest.skip("no converted shards on this box")

    from record_format import RECORD_DTYPE

    bands = vocabulary_bands()
    rng = np.random.default_rng(20260811)
    paths = sorted(shard_dir.glob("train_*.bin"))[:4]
    seen = set()
    for path in paths:
        m = np.memmap(path, dtype=RECORD_DTYPE, mode="r")
        try:
            pick = rng.choice(len(m), size=min(20_000, len(m)), replace=False)
            toks = m["tokens"][np.sort(pick)].astype(np.int64)
            assert toks.shape[1] == SEQ_LENGTH
            assert toks.min() >= 0 and toks.max() <= TOKEN_CLS, (
                f"{path.name}: tokens outside 0..{TOKEN_CLS} "
                f"(min {toks.min()}, max {toks.max()})")
            assert set(np.unique(toks[:, POS_STM]).tolist()) <= set(bands["side_to_move"])
            assert set(np.unique(toks[:, POS_CASTLING]).tolist()) <= set(bands["castling"])
            assert set(np.unique(toks[:, POS_EP]).tolist()) <= (
                set(bands["ep_none"]) | set(bands["ep_file"]))
            assert set(np.unique(toks[:, POS_CLS]).tolist()) == {TOKEN_CLS}
            seen |= set(np.unique(toks).tolist())
        finally:
            m._mmap.close()

    # The reserved rows must be genuinely unused by the corpus, not merely
    # unused by the encoder we can call in-process.
    assert not (seen & {41, 42}), \
        f"reserved token IDs appear in {shard_dir.name}: {sorted(seen & {41, 42})}"


def test_embedding_covers_every_token_the_corpus_can_hold():
    """int8 records: the widest value a shard can physically carry is 127, so
    the in-band assertion above is the only thing standing between the corpus
    and an IndexError. Confirm the model would in fact fault rather than wrap."""
    m = build_model()
    ok = torch.full((1, 68), TOKEN_CLS, dtype=torch.long)
    m.eval()
    with torch.no_grad():
        m(ok)                                   # highest active ID is fine
    with pytest.raises(IndexError):
        m(torch.full((1, 68), VOCAB_SIZE, dtype=torch.long))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
