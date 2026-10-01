"""S7 contract B — the engine's `canonical_65` path, design doc §11.2, tested by §11.3.

Contract B is three C++ changes, selected by the export's declared contract:
the canonical tokenizer (cpp/tokens.hpp `tokenize_canonical_into`), the policy
remap and the value sign (cpp/search.hpp `expand_from_live_row`). §11.3 defines
the tests, and each is here as written:

  B1  C++ canonical tokens equal the Python reference (`tokens_canonical_65`) on
      the 100k-FEN set of port chunk C2 (golden/tokens.npz), and
      tokens(x) = tokens(colour-mirror(x)) for every position.
  B2  f_sym(x) = 1/2 [f(x) + M(f(mirror(x)))] from the deployed v5 net, run under
      contract A and, through an adapter that feeds canonical tokens into f_sym,
      under contract B. At W=1, K=1 the trees must be identical node for node and
      the best moves identical on golden/c10_corpus.json.
  B3  Black to move with a legal ep capture, partial castling rights and a
      promotion: token output, remap and value sign checked by hand.

One reading had to be made in B2. The v5 net sees the RAW ep file (token 66); the
canonical row keeps an ep target only when the capture is legal. So f_sym on the
raw v5 row is not a function of the canonical tokens, and the two trees could
legitimately differ at every position after a double push. The contract-A arm
therefore clears token 66 when no legal ep capture exists before applying f_sym.
That is still a v5_68 network, and it is exactly colour-equivariant.

Per Amendment D there are no module-scope skips: B1's reference half needs torch
(the reference lives in `core.guofish_net`, whose package imports it), B2 needs
torch, CUDA and the v5 checkpoint, and each is marked individually.
"""
import json
from pathlib import Path

import chess
import numpy as np
import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent
C2_TOKENS = REPO_ROOT / "golden" / "tokens.npz"
CORPUS = REPO_ROOT / "golden" / "c10_corpus.json"
GATE2B_MANIFEST = REPO_ROOT / "golden" / "c10_gate2b_manifest.json"
V5_MODEL = REPO_ROOT / "models" / "guofish5_90M" / "v5_10.9M_best.pt"   # evaluator.SHIPPING_MODEL

WIDTH = guofish_core.CANONICAL_SEQ_LENGTH
MAX_REPORTED = 10


def _torch_unavailable() -> str | None:
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"torch is not importable ({type(exc).__name__}: {exc})"
    return None


def _b2_unavailable() -> str | None:
    reason = _torch_unavailable()
    if reason:
        return reason
    import torch
    if not torch.cuda.is_available():
        return ("CUDA is not available; on CPU the v5 loader quantises to int8 and the "
                "500-position sweep would take tens of minutes")
    if not V5_MODEL.exists():
        return f"the deployed v5 checkpoint is missing ({V5_MODEL})"
    return None


requires_torch = pytest.mark.skipif(_torch_unavailable() is not None,
                                    reason=str(_torch_unavailable()))
B2_UNAVAILABLE = _b2_unavailable()
requires_b2 = pytest.mark.skipif(B2_UNAVAILABLE is not None, reason=str(B2_UNAVAILABLE))


def _canonical(fen: str) -> np.ndarray:
    return guofish_core.eval_row(fen, "B")["tokens"]


@pytest.fixture(scope="module")
def c2_fens():
    with np.load(C2_TOKENS, allow_pickle=True) as archive:
        return [str(f) for f in archive["fens"]]


# ---------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------

@requires_torch
def test_b1_canonical_tokens_match_the_python_reference(c2_fens):
    from core.guofish_net.tokenizers import tokens_canonical_65

    bad, seen = [], set()
    for fen in c2_fens:
        cpp = _canonical(fen)[:WIDTH]
        ref = tokens_canonical_65(chess.Board(fen))
        seen.update(cpp.tolist())
        if not np.array_equal(cpp, ref) and len(bad) < MAX_REPORTED:
            bad.append(f"{fen}\n  cpp {cpp.tolist()}\n  ref {ref.tolist()}")
    assert not bad, f"{len(bad)}+ positions differ from tokens_canonical_65:\n" + "\n".join(bad)
    # the corpus reaches every canonical token, the castling rooks and ep target included
    assert seen == set(range(17)), sorted(set(range(17)) - seen)


def test_b1_canonical_tokens_are_blind_to_colour(c2_fens):
    """tokens(x) = tokens(mirror(x)); the row's side-to-move slot keeps their keys apart."""
    bad = []
    for fen in c2_fens:
        mirror = chess.Board(fen).mirror().fen(en_passant="fen")    # the raw ep square survives
        a, b = guofish_core.eval_row(fen, "B"), guofish_core.eval_row(mirror, "B")
        if (not np.array_equal(a["tokens"][:WIDTH], b["tokens"][:WIDTH])
                or a["nn_key"] == b["nn_key"]) and len(bad) < MAX_REPORTED:
            bad.append(f"{fen} vs {mirror}\n  {a['tokens'].tolist()}\n  {b['tokens'].tolist()}")
    assert not bad, "\n".join(bad)


# ---------------------------------------------------------------------------
# B3 — by hand
# ---------------------------------------------------------------------------

# Canonical-frame pictures, rank 8 first: the board flipped when Black moves,
# uppercase ours, lowercase theirs, C/c a rook still carrying a castling right,
# x the ep target. `move` is the one legal move whose logit the hand-made row
# marks, at `index`: (from ^ 56) * 64 + (to ^ 56), worked out by hand.
_PICTURE = {".": 0, "C": 13, "c": 14, "x": 15,
            **{p: i + 1 for i, p in enumerate("PNBRQK")},
            **{p: i + 7 for i, p in enumerate("pnbrqk")}}

B3 = {
    "legal ep": dict(
        fen="rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
        picture=["cnbqkbnc", "pppp.ppp", "....x...", "...Pp...",
                 "........", "........", "PPP.PPPP", "CNBQKBNC"],
        move="d4e3", index=35 * 64 + 44, collisions=1),      # d5e6
    "partial castling": dict(
        fen="r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1",
        picture=["r...k..c", "........", "........", "........",
                 "........", "........", "........", "C...K..R"],
        move="e8c8", index=4 * 64 + 2, collisions=1),         # e1c1
    "promotion": dict(
        fen="8/8/8/8/8/5k2/p7/5K2 b - - 0 1",
        picture=[".....k..", "P.......", ".....K..", "........",
                 "........", "........", "........", "........"],
        move="a2a1q", index=48 * 64 + 56, collisions=4),      # a7a8, all four promotions
}

LOGIT = 3.0
VALUE = 0.25      # the network's, side to move


def _row_of_picture(picture: list[str]) -> list[int]:
    out = [0] * 64
    for rank_from_top, squares in enumerate(picture):
        for file, ch in enumerate(squares):
            out[(7 - rank_from_top) * 8 + file] = _PICTURE[ch]
    return out + [16, 14, 0, 0]          # CLS, Black to move, two zeros


@pytest.mark.parametrize("name", list(B3))
def test_b3_tokens_by_hand(name):
    case = B3[name]
    assert _canonical(case["fen"]).tolist() == _row_of_picture(case["picture"])


class _HandNet:
    """A LiveEvaluator callback that answers every row with one hand-made row.

    A class so the evaluator -> callback -> evaluator cycle can be broken on
    purpose (see tests/test_c11b_temperature.py `_SyntheticNetwork`)."""

    def __init__(self, index: int):
        logits = np.zeros(guofish_core.POLICY_SIZE, dtype=np.float32)
        logits[index] = LOGIT
        self.row = (logits.view(np.uint32) >> 16).astype(np.uint16)     # exact in bf16
        self.evaluator = None

    def __call__(self, count):
        self.evaluator.policy_view()[:count] = self.row
        self.evaluator.value_view()[:count] = VALUE


def _root_after_one_simulation(fen: str, index: int, contract: str):
    net = _HandNet(index)
    net.evaluator = guofish_core.LiveEvaluator(8, net, 0.0, contract)
    search = guofish_core.ReplaySearchDouble(guofish_core.SearchConfig(arena_capacity=1 << 16))
    search.set_evaluator(net.evaluator)
    try:
        search.set_position(fen)
        search.search(1)
        arrays = search.dump_tree_arrays(0)
    finally:
        search.set_evaluator(None)
        net.evaluator = None
    priors = {guofish_core.move_to_uci(int(m)): float(p)
              for d, m, p in zip(arrays["depth"], arrays["move"], arrays["prior"]) if d == 1}
    return priors, int(arrays["visits"][0]), float(arrays["value_sum"][0])


@pytest.mark.parametrize("name", list(B3))
def test_b3_remap_and_value_sign_by_hand(name):
    case = B3[name]
    priors, visits, value_sum = _root_after_one_simulation(case["fen"], case["index"], "B")
    n, k = len(priors), case["collisions"]
    marked = np.exp(LOGIT) / (k * np.exp(LOGIT) + n - k)
    for uci, prior in priors.items():
        hit = uci[:4] == case["move"][:4]
        assert prior == pytest.approx(marked if hit else (1 - k * marked) / (n - k), rel=1e-6), uci
    assert sum(uci[:4] == case["move"][:4] for uci in priors) == k
    # Black to move: the side-to-move +0.25 is White-POV -0.25, which the root keeps
    # from the mover's side, i.e. as is.
    assert visits == 1 and value_sum == -VALUE

    # Control: read as contract A, the same row marks no legal move and keeps the sign.
    priors_a, _, value_a = _root_after_one_simulation(case["fen"], case["index"], "A")
    assert all(p == pytest.approx(1 / n, rel=1e-6) for p in priors_a.values())
    assert value_a == VALUE


def test_b3_under_the_parallel_dispatcher():
    """Contract B through search_parallel's batches, with more than one worker.

    Not the promotion position: a2a1q is mate in one, and search_parallel at W >= 2
    can hang on a root mate-in-one with any live evaluator, contract A included
    (seen on HEAD's C++ before this change; W=1 is unaffected)."""
    case = B3["legal ep"]
    serial, _, _ = _root_after_one_simulation(case["fen"], case["index"], "B")
    net = _HandNet(case["index"])
    net.evaluator = guofish_core.LiveEvaluator(8, net, 0.0, "B")
    search = guofish_core.ReplaySearchQ32(guofish_core.SearchConfig(arena_capacity=1 << 16))
    search.set_evaluator(net.evaluator)
    try:
        search.set_position(case["fen"])
        stats = search.search_parallel(200, guofish_core.ParallelConfig(workers=2, in_flight=4,
                                                                        max_batch=8))
        arrays = search.dump_tree_arrays(0)
    finally:
        search.set_evaluator(None)
        net.evaluator = None
    assert stats["best_move"] == case["move"]
    for d, m, p in zip(arrays["depth"], arrays["move"], arrays["prior"]):
        if d == 1:
            assert float(p) == serial[guofish_core.move_to_uci(int(m))]


# ---------------------------------------------------------------------------
# B2 — the symmetrised v5 net under both contracts
# ---------------------------------------------------------------------------

# canonical token -> v5_68 token on the canonical (White to move) board: castling
# rooks are rooks, the ep target square is empty, CLS is not a square
_DECANON = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 4, 10, 0, 0])


def _v5_of_canonical(row: np.ndarray) -> np.ndarray:
    """The adapter: a contract-B row back to the v5_68 tokens of the board it encodes."""
    sq = row[:64]
    ep = np.flatnonzero(sq == 15)
    castling = 8 * (sq[7] == 13) + 4 * (sq[0] == 13) + 2 * (sq[63] == 14) + (sq[56] == 14)
    return np.concatenate([_DECANON[sq], [13, 15 + castling, 32 + ep[0] % 8 if len(ep) else 31, 40]])


def _legal_ep(row: np.ndarray) -> np.ndarray:
    from core.guofish_net.tokenizers import board_from_v5_tokens

    if row[66] != 31 and not board_from_v5_tokens(row).has_legal_en_passant():
        row = row.copy()
        row[66] = 31
    return row


class _SymmetrisedV5:
    """f_sym on the deployed v5 net, as the callback of one contract's LiveEvaluator.

    f(x) and f(mirror(x)) are evaluated as one batch, the White-to-move board first,
    in both arms, so each arm sees the same bits for them. Both arms then add the
    same two terms, which makes f_sym exactly colour-equivariant (§11.3)."""

    def __init__(self, model, device, canonical: bool):
        from data.multiPV.mirror import POLICY_PERM

        self.model, self.device, self.canonical = model, device, canonical
        self.perm = POLICY_PERM
        self.evaluator = None

    def __call__(self, count):
        import torch
        from data.multiPV.mirror import mirror_tokens_np

        rows = np.asarray(self.evaluator.input_view()[:count], dtype=np.int64)
        x = np.stack([_v5_of_canonical(r) if self.canonical else _legal_ep(r) for r in rows])
        m = mirror_tokens_np(x)
        white = x[:, 64] == 13
        pair = np.concatenate([np.where(white[:, None], x, m), np.where(white[:, None], m, x)])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p, v = self.model(torch.from_numpy(pair).to(self.device))
        p, v = p.float().cpu(), v.float().cpu().reshape(-1)
        w = torch.from_numpy(white)
        px, pm = torch.where(w[:, None], p[:count], p[count:]), torch.where(w[:, None], p[count:], p[:count])
        vx, vm = torch.where(w, v[:count], v[count:]), torch.where(w, v[count:], v[:count])
        policy = (px + pm[:, self.perm]) * 0.5
        value = (vx - vm) * 0.5
        self.evaluator.policy_view()[:count] = (
            policy.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
        self.evaluator.value_view()[:count] = value.numpy()


# §11.3 names no budget. Identity is node for node, so every expanded node's
# priors and value are compared, and 16 is enough to reach both colours, the
# dispatcher and the cache on every position. It is ~16 ms a simulation with the
# screening queue on the GPU and a few ms without it.
B2_SIMS = 16
B2_CACHE_SLOTS = 400_000      # tests/test_c10_gate2b.py


def _b2_arm(model, device, canonical: bool, fens: list[str]):
    recorded = json.loads(GATE2B_MANIFEST.read_text(encoding="utf-8"))["search_config"]
    config = guofish_core.SearchConfig()
    for k in ("c_init", "c_base", "fpu_root", "fpu_tree", "virtual_loss", "max_tree_depth"):
        setattr(config, k, recorded[k])
    config.cache_slots = B2_CACHE_SLOTS
    net = _SymmetrisedV5(model, device, canonical)
    net.evaluator = guofish_core.LiveEvaluator(1, net, 0.0, "B" if canonical else "A")
    search = guofish_core.ReplaySearchDouble(config)
    search.set_evaluator(net.evaluator)
    parallel = guofish_core.ParallelConfig(workers=1, in_flight=1, max_batch=1)
    try:
        for fen in fens:
            search.set_position(fen)
            stats = search.search_parallel(B2_SIMS, parallel)
            yield stats["best_move"], search.dump_tree_arrays(0)
    finally:
        search.set_evaluator(None)
        net.evaluator = None


@requires_b2
def test_b2_symmetrised_v5_trees_are_identical_under_both_contracts():
    import torch
    from playing.v6 import evaluator as ev

    device = torch.device("cuda")
    model, _ = ev.load_default_model(V5_MODEL, device)
    fens = [p["fen"] for p in json.loads(CORPUS.read_text(encoding="utf-8"))["positions"]]
    arm_a = list(_b2_arm(model, device, False, fens))
    arm_b = list(_b2_arm(model, device, True, fens))
    differing = []
    for fen, (best_a, tree_a), (best_b, tree_b) in zip(fens, arm_a, arm_b):
        same = best_a == best_b and tree_a.keys() == tree_b.keys() and all(
            np.array_equal(tree_a[k], tree_b[k]) for k in tree_a)
        if not same:
            first = next((i for i in range(min(len(tree_a["move"]), len(tree_b["move"])))
                          if any(tree_a[k][i] != tree_b[k][i] for k in tree_a)), None)
            differing.append(f"{fen}: best {best_a} vs {best_b}, "
                             f"{len(tree_a['move'])} vs {len(tree_b['move'])} nodes, first differing node {first}")
    assert len(arm_a) == len(arm_b) == len(fens) == 500
    assert not differing, f"{len(differing)} of {len(fens)} trees differ:\n" + "\n".join(differing[:MAX_REPORTED])
