"""Targets, symmetry transforms and collate (§5.4, §6.7, §7).

Everything here works on whole micro-batches of v2-dtype records in numpy,
inside the loader worker, so the main process only moves tensors.

    v5_68         rows are colour-mirrored when hash(seed, sample index) < mirror_prob.
    canonical_65  Black-to-move rows are mirrored (making the mover "White"),
                  then the 68 tokens become 65 canonical ones.

The mirror is applied to the RECORD (tokens, pv_idx, legal_idx, hard_move,
value, value_cp) before the dense targets are built. Building from permuted
indices puts the same float32 values, accumulated in the same order, at the
permuted positions, so this equals v5's dense-then-gather mirror bit for bit.
"""
from __future__ import annotations

import numpy as np
import torch

from core.guofish_net.model import POLICY_SIZE
from core.guofish_net.tokenizers import (
    C_CLS, C_EP_TARGET, C_OUR_CASTLE_ROOK, C_THEIR_CASTLE_ROOK,
)
from training.v6.data.formats import _MPV  # noqa: F401  (data/multiPV on sys.path)
from training.v6.data.mixture import mix_key, uniform01
from labels import MAX_LEGAL, MAX_PV, VALUE_MATE_MIN  # noqa: E402
from mirror import POLICY_PERM, TOKEN_PERM, VOCAB_MIRROR  # noqa: E402

MIRROR_STREAM = 2
_ARANGE_PV = np.arange(MAX_PV)
_ARANGE_LEGAL = np.arange(MAX_LEGAL)
# (castling bit, square, rook token that must be there, canonical token)
_CASTLE = ((8, 7, 4, C_OUR_CASTLE_ROOK), (4, 0, 4, C_OUR_CASTLE_ROOK),
           (2, 63, 10, C_THEIR_CASTLE_ROOK), (1, 56, 10, C_THEIR_CASTLE_ROOK))


def mirror_decisions(seed: int, sample_idx: np.ndarray, prob: float) -> np.ndarray:
    if prob <= 0.0:
        return np.zeros(len(sample_idx), dtype=bool)
    return uniform01(mix_key(seed, MIRROR_STREAM), sample_idx) < prob


def mirror_records(rec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Exact colour mirror of the selected rows. Returns a new array. Padded
    pv/legal slots keep their stored value, as v5's color_mirror does."""
    out = rec.copy()
    if not mask.any():
        return out
    m = rec[mask]
    m["tokens"] = VOCAB_MIRROR[m["tokens"][:, TOKEN_PERM].astype(np.int64)].astype(np.int8)
    pv_ok = _ARANGE_PV[None, :] < m["n_pv"][:, None]
    m["pv_idx"] = np.where(pv_ok, POLICY_PERM[np.clip(m["pv_idx"], 0, POLICY_SIZE - 1)],
                           m["pv_idx"]).astype(np.int16)
    lg_ok = _ARANGE_LEGAL[None, :] < m["n_legal"][:, None]
    m["legal_idx"] = np.where(lg_ok, POLICY_PERM[np.clip(m["legal_idx"], 0, POLICY_SIZE - 1)],
                              m["legal_idx"]).astype(np.int16)
    hm = m["hard_move"].astype(np.int64)
    m["hard_move"] = np.where(hm >= 0, POLICY_PERM[np.clip(hm, 0, POLICY_SIZE - 1)], -1)
    m["value"] = -m["value"]
    m["value_cp"] = -m["value_cp"]
    out[mask] = m
    return out


def canonical_tokens(rec: np.ndarray) -> np.ndarray:
    """White-to-move v5_68 records (Black rows already mirrored) -> (B, 65) int8.

    Castling rights ride on the rook token (the rook must be on its corner);
    the ep target gets its token only when the stored legal moves contain a
    pawn capture onto it from an adjacent file on rank 5."""
    tok = rec["tokens"]
    if (tok[:, 64] != 13).any():
        raise ValueError("canonical_tokens needs every row White-to-move")
    sq = tok[:, :64].astype(np.int8).copy()          # 1-6 ours, 7-12 theirs
    bits = tok[:, 65].astype(np.int64) - 15
    rows = np.arange(len(tok))
    for bit, square, rook, new in _CASTLE:
        has = (bits & bit) != 0
        if (sq[has, square] != rook).any():
            raise ValueError(f"castling right without a rook on square {square}")
        sq[has, square] = new

    ep = tok[:, 66].astype(np.int64)
    has_ep = ep != 31
    if has_ep.any():
        r = rows[has_ep]
        f = ep[has_ep] - 32
        target = 40 + f
        legal = rec["legal_idx"][has_ep].astype(np.int64)
        lg_ok = _ARANGE_LEGAL[None, :] < rec["n_legal"][has_ep][:, None].astype(np.int64)
        found = np.zeros(len(r), dtype=bool)
        for df in (-1, 1):
            fr = 32 + f + df
            ok = (f + df >= 0) & (f + df <= 7)
            frc = np.clip(fr, 32, 39)
            pawn = sq[r, frc] == 1
            mv = (frc * 64 + target)[:, None]
            found |= ok & pawn & ((legal == mv) & lg_ok).any(1)
        if (sq[r[found], target[found]] != 0).any():
            raise ValueError("ep target square is occupied")
        sq[r[found], target[found]] = C_EP_TARGET
    return np.concatenate([sq, np.full((len(tok), 1), C_CLS, dtype=np.int8)], axis=1)


def dense_legal(rec: np.ndarray) -> np.ndarray:
    n = rec["n_legal"].astype(np.int64)
    out = np.zeros((len(rec), POLICY_SIZE), dtype=bool)
    ok = _ARANGE_LEGAL[None, :] < n[:, None]
    out[np.repeat(np.arange(len(rec)), n), rec["legal_idx"][ok].astype(np.int64)] = True
    return out


def dense_policy(rec: np.ndarray, epsilon: float, temperature: float | None) -> np.ndarray:
    """v5's target: pv mass accumulated with +=, then eps/n_legal per legal
    entry with +=, both in float32 and in v5's order. Rows without a policy
    (or with no PV entries) stay all-zero. With a temperature the PV mass is
    rebuilt as (1-eps)*softmax(pv_score/T) in float64 first."""
    B = len(rec)
    out = np.zeros((B, POLICY_SIZE), dtype=np.float32)
    n_pv = rec["n_pv"].astype(np.int64)
    pol = (rec["has_policy"] > 0) & (n_pv > 0)
    pv_ok = (_ARANGE_PV[None, :] < n_pv[:, None]) & pol[:, None]
    if temperature is None:
        mass = rec["pv_prob"].astype(np.float32)
    else:
        s = np.where(pv_ok, rec["pv_score"].astype(np.float64) / temperature, -np.inf)
        s = s - np.where(pol, s.max(1), 0.0)[:, None]
        e = np.where(pv_ok, np.exp(s), 0.0)
        mass = ((1.0 - epsilon) * e / np.where(pol, e.sum(1), 1.0)[:, None]).astype(np.float32)
    rows = np.nonzero(pv_ok)[0]
    np.add.at(out, (rows, rec["pv_idx"][pv_ok].astype(np.int64)), mass[pv_ok])

    n_legal = rec["n_legal"].astype(np.int64)
    eps_rows = pol & (n_legal > 0)
    lg_ok = (_ARANGE_LEGAL[None, :] < n_legal[:, None]) & eps_rows[:, None]
    # float64 on purpose: v5's `np.add.at(policy, li, eps / n_legal)` passes a
    # Python float, so NumPy adds in float64 and rounds to float32 per add.
    # A float32 share differs by 1 ulp wherever entries accumulate (S1).
    share = epsilon / np.maximum(n_legal, 1).astype(np.float64)
    rows = np.nonzero(lg_ok)[0]
    np.add.at(out, (rows, rec["legal_idx"][lg_ok].astype(np.int64)), share[rows])
    return out


def value_stratum(value_cp: np.ndarray) -> np.ndarray:
    cp = value_cp.astype(np.int64)
    return np.where(cp == 0, 0, np.where(np.abs(cp) >= VALUE_MATE_MIN, 1, 2))


class BatchBuilder:
    """Records + sample indices -> a dict of CPU tensors ready for the step."""

    def __init__(self, token_scheme: str, mirror_prob: float, seed: int,
                 epsilon: float, temperature: float | None):
        if token_scheme not in ("v5_68", "canonical_65"):
            raise ValueError(token_scheme)
        if token_scheme == "canonical_65" and mirror_prob:
            raise ValueError("mirror augmentation is a no-op under canonical_65")
        self.token_scheme, self.mirror_prob, self.seed = token_scheme, mirror_prob, seed
        self.epsilon, self.temperature = epsilon, temperature

    def transform(self, rec: np.ndarray, sample_idx: np.ndarray):
        if self.token_scheme == "v5_68":
            flip = mirror_decisions(self.seed, sample_idx, self.mirror_prob)
            rec = mirror_records(rec, flip)
            return rec, rec["tokens"], flip
        flip = rec["tokens"][:, 64] == 14
        rec = mirror_records(rec, flip)
        return rec, canonical_tokens(rec), flip

    def __call__(self, rec: np.ndarray, sample_idx: np.ndarray) -> dict:
        rec, tokens, flip = self.transform(rec, sample_idx)
        t = torch.from_numpy
        return {
            "tokens": t(np.ascontiguousarray(tokens)),
            "policy": t(dense_policy(rec, self.epsilon, self.temperature)),
            "legal": t(dense_legal(rec)),
            "has_policy": t(rec["has_policy"] > 0),
            "value": t(rec["value"].astype(np.float32)),
            "value_cp": t(rec["value_cp"].astype(np.int32)),
            "value_stratum": t(value_stratum(rec["value_cp"])),
            "hard_move": t(rec["hard_move"].astype(np.int64)),
            "pv_idx": t(rec["pv_idx"].astype(np.int64)),
            "n_pv": t(rec["n_pv"].astype(np.int64)),
            "n_legal": t(rec["n_legal"].astype(np.int64)),
            "sample_index": t(np.asarray(sample_idx, dtype=np.int64)),
            "mirrored": t(flip),
        }


class StreamDataset(torch.utils.data.Dataset):
    """Map-style over micro-batch index k; item k is micro-batch k, collated.

    Use with DataLoader(batch_size=None, sampler=range(k0, k1)). The stream is
    a function of (seed, k) only, so worker count and resume point cannot
    change it."""

    def __init__(self, shards, mixture, builder: BatchBuilder, n_micro: int):
        self.shards, self.mixture, self.builder, self.n_micro = shards, mixture, builder, n_micro

    def __len__(self) -> int:
        return self.n_micro

    def __getitem__(self, k: int) -> dict:
        rec_idx, sample_idx, gid = self.mixture.batch(int(k))
        out = self.builder(self.shards.read(rec_idx), sample_idx)
        out["group"] = torch.from_numpy(gid)
        out["record_index"] = torch.from_numpy(rec_idx)
        return out


class EvalDataset(torch.utils.data.Dataset):
    """Sequential batches over a fixed index list, mirror off."""

    def __init__(self, shards, indices: np.ndarray, batch: int, builder: BatchBuilder):
        if builder.mirror_prob:
            raise ValueError("eval batches are never mirrored")
        self.shards, self.indices, self.batch, self.builder = shards, np.asarray(indices), batch, builder

    def __len__(self) -> int:
        return (len(self.indices) + self.batch - 1) // self.batch

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i * self.batch:(i + 1) * self.batch]
        out = self.builder(self.shards.read(idx), idx)
        out["record_index"] = torch.from_numpy(idx.astype(np.int64))
        return out
