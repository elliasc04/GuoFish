"""Colour-mirror correctness at the level training actually uses it.

Runs under pytest, or standalone:
    python training/v5_multiPV/tests/test_mirror.py

data/multiPV/test_pipeline.py already pins the LUTs against chess.Board.mirror().
What is re-pinned here is the involution as the training loop sees it - on a
collated BATCH, on the dense policy target, and on the policy INDICES - plus the
mirror-consistency diagnostic itself, which is only worth reporting if it
returns 0 for an equivariant model and something else for a model that is
leaning on absolute board orientation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_V5 = _HERE.parent
_ROOT = _V5.parents[1]
_DATA = _ROOT / "data" / "multiPV"
for _p in (str(_V5), str(_DATA), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.pgn_parallel import _board_to_tokens                   # noqa: E402
from dataset import color_mirror                                 # noqa: E402
from mirror import (POLICY_PERM, TOKEN_PERM, VOCAB_MIRROR,       # noqa: E402
                    mirror_dense_np, mirror_policy_index_np,
                    mirror_tokens_np)
from losses import masked_log_softmax                            # noqa: E402
from metrics import mirror_consistency                           # noqa: E402
from tests.test_losses import make_batch                         # noqa: E402

POLICY_SIZE = 4096

FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]


# ---------------------------------------------------------------------------
# the permutations themselves
# ---------------------------------------------------------------------------
def test_all_three_luts_are_involutions():
    assert np.array_equal(VOCAB_MIRROR[VOCAB_MIRROR],
                          np.arange(len(VOCAB_MIRROR)))
    assert np.array_equal(TOKEN_PERM[TOKEN_PERM], np.arange(len(TOKEN_PERM)))
    assert np.array_equal(POLICY_PERM[POLICY_PERM], np.arange(POLICY_SIZE))


def test_policy_index_formula():
    """The brief states the mapping explicitly: (from^56)*64 + (to^56)."""
    for frm in range(64):
        for to in range(64):
            i = frm * 64 + to
            assert POLICY_PERM[i] == (frm ^ 56) * 64 + (to ^ 56)
    idx = np.arange(POLICY_SIZE)
    assert np.array_equal(mirror_policy_index_np(mirror_policy_index_np(idx)), idx)


def test_token_mirror_involution_on_real_positions():
    for fen in FENS:
        t = np.asarray(_board_to_tokens(chess.Board(fen)), dtype=np.int64)
        assert np.array_equal(mirror_tokens_np(mirror_tokens_np(t)), t), fen


def test_dense_policy_mirror_moves_the_argmax_under_the_perm():
    rng = np.random.default_rng(0)
    dense = rng.random(POLICY_SIZE).astype(np.float32)
    m = mirror_dense_np(dense)
    assert np.array_equal(mirror_dense_np(m), dense)
    assert int(m.argmax()) == int(POLICY_PERM[int(dense.argmax())])


# ---------------------------------------------------------------------------
# the batch-level transform the collate actually calls
# ---------------------------------------------------------------------------
def _batch_with_pv(n=8, seed=30):
    b = make_batch(n=n, seed=seed)
    n_pv = 3
    pv_idx = torch.zeros(n, 8, dtype=torch.int64)
    for i in range(n):
        top = b["policy"][i].topk(n_pv).indices
        pv_idx[i, :n_pv] = top
    b["pv_idx"] = pv_idx
    b["n_pv"] = torch.full((n,), n_pv, dtype=torch.int64)
    return b


def test_color_mirror_is_an_involution_on_a_whole_batch():
    b = _batch_with_pv()
    all_true = torch.ones(b["tokens"].shape[0], dtype=torch.bool)
    once = color_mirror(b, all_true)
    twice = color_mirror(once, all_true)
    for k in b:
        assert torch.equal(twice[k], b[k]), f"field {k} did not round-trip"


def test_color_mirror_transforms_every_orientation_carrying_field():
    b = _batch_with_pv(n=6, seed=31)
    all_true = torch.ones(6, dtype=torch.bool)
    m = color_mirror(b, all_true)

    assert torch.equal(m["value"], -b["value"])
    assert torch.equal(m["value_cp"], -b["value_cp"])

    perm = torch.from_numpy(POLICY_PERM)
    assert torch.equal(m["policy"], b["policy"][:, perm])
    assert torch.equal(m["legal_mask"], b["legal_mask"][:, perm])
    assert torch.equal(m["pv_idx"][:, :3], perm[b["pv_idx"][:, :3]])

    tperm = torch.from_numpy(TOKEN_PERM)
    lut = torch.from_numpy(VOCAB_MIRROR)
    assert torch.equal(m["tokens"], lut[b["tokens"][:, tperm]])

    # The target's argmax must follow the permutation - this is what the
    # mirror-consistency policy check compares against.
    assert torch.equal(m["policy"].argmax(-1), perm[b["policy"].argmax(-1)])
    # The mirrored target still lives entirely on the mirrored legal mask.
    assert torch.all((m["policy"] > 0) <= (m["legal_mask"] > 0))


def test_partial_mask_leaves_unselected_rows_untouched():
    b = _batch_with_pv(n=6, seed=32)
    mask = torch.tensor([True, False, True, False, False, True])
    m = color_mirror(b, mask)
    for k in b:
        assert torch.equal(m[k][~mask], b[k][~mask]), f"{k} leaked across rows"
        if k in ("value", "value_cp"):
            assert torch.equal(m[k][mask], -b[k][mask])


# ---------------------------------------------------------------------------
# the diagnostic
# ---------------------------------------------------------------------------
class _EquivariantStub(nn.Module):
    """A model that is mirror-equivariant BY CONSTRUCTION, so the diagnostic
    must report exactly the ideal numbers on it.

        value   sum over squares of (+1 white piece, -1 black piece), squashed.
                The mirror swaps the colours, so the sum negates.
        policy  logits[from*64+to] = a*w[from] + b*w[to], with w[sq] read from a
                table indexed by (colour-blind piece type, FILE). Both indices
                are mirror-invariant - the mirror swaps colours and flips rank,
                and sq^56 preserves the file - so the mirror moves the token at
                sq^56 onto sq and therefore w'[sq] = w[sq^56], which makes
                logits' = logits[POLICY_PERM] exactly.

    The table is random and a != b so the logits carry no accidental ties; a
    degenerate stub would make the top-1 agreement a coin flip on ties rather
    than a real measurement.
    """

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.register_buffer("W", torch.randn(7, 8, generator=g))
        self.register_buffer("files", torch.arange(64) % 8)

    def forward(self, x, legal_move_mask=None):
        sq = x[:, :64]
        white = ((sq >= 1) & (sq <= 6)).float()
        black = ((sq >= 7) & (sq <= 12)).float()
        value = torch.tanh((white.sum(-1) - black.sum(-1)) * 0.1)

        colour_blind = torch.where(sq >= 7, sq - 6, sq).clamp(0, 6)
        w = self.W[colour_blind, self.files.expand_as(colour_blind)]
        logits = w.unsqueeze(2) + 1.61803398875 * w.unsqueeze(1)   # (B, 64, 64)
        logits = logits.reshape(x.shape[0], POLICY_SIZE)
        if legal_move_mask is not None:
            logits = logits.masked_fill(~legal_move_mask, float("-inf"))
        return logits, value


class _OrientedStub(nn.Module):
    """Deliberately leans on absolute orientation: it reads the raw square
    index, which the mirror moves. The diagnostic must NOT give it a pass."""

    def forward(self, x, legal_move_mask=None):
        sq = x[:, :64].float()
        rank = (torch.arange(64, device=x.device).float() // 8).unsqueeze(0)
        value = torch.tanh((sq * rank).mean(-1) * 0.05)
        w = sq * rank
        logits = (w.unsqueeze(2) + w.unsqueeze(1)).reshape(x.shape[0], POLICY_SIZE)
        if legal_move_mask is not None:
            logits = logits.masked_fill(~legal_move_mask, float("-inf"))
        return logits, value


def _real_token_batch(n=5):
    """Real positions, so the stub models see legal token distributions."""
    toks = [np.asarray(_board_to_tokens(chess.Board(f)), dtype=np.int64)
            for f in FENS[:n]]
    b = make_batch(n=len(toks), seed=33)
    b["tokens"] = torch.from_numpy(np.stack(toks))
    return b


def _drop_tied_argmax_rows(model, batch):
    """Rows whose masked argmax is not unique decide the top-1 comparison by
    tie-breaking rather than by equivariance, so they say nothing about the
    diagnostic.

    The ties are not a bug in the stub. The mirror identifies sq with sq^56, so
    ANY equivariant function must give two mirror-paired, mirror-equivalent
    squares (e.g. two empty squares on the same file) the same value - a
    genuinely tied argmax. A trained network hits this with probability zero.
    """
    logits, _ = model(batch["tokens"])
    lq, _ = masked_log_softmax(logits, batch["legal_mask"])
    top = lq.max(dim=-1, keepdim=True).values
    unique = (lq == top).sum(-1) == 1
    return {k: v[unique] for k, v in batch.items()}, int(unique.sum())


def test_equivariant_stub_logits_permute_exactly():
    """The strongest statement available, and the premise of the test below:
    logits(mirror(x)) == logits(x)[POLICY_PERM], bit for bit."""
    b = _real_token_batch()
    m = _EquivariantStub()
    all_true = torch.ones(b["tokens"].shape[0], dtype=torch.bool)
    lg, v = m(b["tokens"])
    lg_m, v_m = m(color_mirror(b, all_true)["tokens"])
    perm = torch.from_numpy(POLICY_PERM)
    assert torch.equal(lg_m, lg[:, perm]), "stub is not actually equivariant"
    assert torch.equal(v_m, -v)


def test_mirror_consistency_is_exact_for_an_equivariant_model():
    b = _real_token_batch()
    model = _EquivariantStub()
    b, kept = _drop_tied_argmax_rows(model, b)
    assert kept >= 3, f"only {kept} untied rows - test would be weak"

    out = mirror_consistency(model, [b], torch.device("cpu"), max_records=kept)
    assert out["mirror_n"] == kept
    assert out["mirror_value_abs_resid"] < 1e-6, out
    assert out["mirror_value_max_resid"] < 1e-6, out
    assert abs(out["mirror_pred_mean_balanced"]) < 1e-6, out
    assert out["mirror_policy_top1_agreement"] == 1.0, out
    assert abs(out["mirror_policy_kl"]) < 1e-5, out


def test_mirror_consistency_flags_an_orientation_dependent_model():
    b = _real_token_batch()
    out = mirror_consistency(_OrientedStub(), [b], torch.device("cpu"),
                             max_records=len(b["tokens"]))
    assert out["mirror_value_abs_resid"] > 1e-3, \
        f"diagnostic passed a model that reads absolute orientation: {out}"
    assert out["mirror_policy_top1_agreement"] < 1.0, out


def test_mirror_consistency_respects_max_records():
    b = _real_token_batch()
    out = mirror_consistency(_EquivariantStub(), [b, b], torch.device("cpu"),
                             max_records=3)
    assert out["mirror_n"] == 3


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
