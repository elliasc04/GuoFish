"""C3b — regression tests that pin the three defects fixed in `core/mctsv4.py`.

Why this file exists
--------------------
Every C++ acceptance criterion from C5 onward is a golden file, and every golden
file is produced by running `core/mctsv4.py` (Global Rule 2). That makes the
Python reference the root of trust for the whole port: if a refactor silently
drops one of the fixes in commit `b43a7f0`, `tools/gen_gate1_golden.py` keeps
running, keeps writing internally consistent data, and permanently encodes the
defect into the acceptance criteria the C++ engine is judged against. The bug
would then be *required* rather than forbidden.

The three fixes were verified behaviourally in-session when they were made, but
the reproduction script was gitignored, so nothing in the repository held them
down. This file does.

Unlike the C5/C6 acceptance files, nothing here touches `guofish_core`. This is
a test OF the reference, not of the port.

The three defects, and what each test would have caught
------------------------------------------------------
1. `test_ep_cache_key` — the transposition cache was keyed on the raw Polyglot
   Zobrist hash. Polyglot folds the en-passant file in only when an enemy pawn
   stands ready to capture; `board_to_tokens` writes token 66 from
   `ep_square is not None`, unconditionally, because that is the rule the
   training shards were written with. So two positions the NETWORK sees
   differently could share a Zobrist key, and whichever was evaluated second was
   served the first one's policy and value. `make_cache_key` folds the raw ep
   square in and closes the gap. The twins are C3's adversarial corpus, read
   from `golden/keys_adversarial.jsonl` rather than restated here.

2. `test_terminal_guard` — a node found drawn by the fifty-move rule or by
   threefold repetition was marked `is_terminal = True` AND `is_expanded = True`
   with an empty `children` dict. Harmless inside a search (selection stops on
   the empty dict either way) and fatal outside it: those draws are only draws
   if someone CLAIMS them, so the host can hand that very position back as the
   one we must move from. `apply_move` promotes the node to root, `search()`
   sees `is_expanded` and skips `_expand_root`, finds no children, and returns
   None — `bestmove 0000`, a forfeit, with legal moves on the board.

3. `test_dirichlet_idempotence` — noise was mixed into `prior` in place. The
   tree persists across moves, so a node can be noised, promoted, and noised
   again; in-place mixing makes the applications COMPOSE, decaying the network's
   contribution to (1 - epsilon)^k = 0.75^k — 24% of its intended weight by the
   fifth application. The fix stores the untouched network prior in `base_prior`
   and derives noise from that, making the operation idempotent.

And one thing that is not a defect
----------------------------------
4. `test_equivalence_determinism` pins the CONFIGURATION that generates C5's and
   C6's golden data: one worker, tablebase off, Dirichlet off, `cache_size=1`,
   canonical child ordering. Two full 2,000-simulation searches must agree node
   for node with `repr()`-level float exactness. This has no pre-fix state to
   fail against — it is a pin, not a regression. Its teeth are shown by the fact
   that it compares EVERY node including unvisited children, and every float by
   its repr rather than by `==`.

Why `cache_size=1` and not "cache off"
--------------------------------------
There is no cache-off flag, and `cache_size=0` raises IndexError —
`TranspositionCache.put` indexes a zero-length ring. With the tablebase off the
cache is result-invariant (it stores raw logits; `expand()` softmaxes them
identically whether they arrived from the network or the ring), so a one-slot
ring is a genuine neutralisation rather than a workaround. Same reasoning as
`tools/gen_gate1_golden.py`, which is the configuration this file exists to pin.

Environment overrides
---------------------
``GUOFISH_GOLDEN_KEYS_ADVERSARIAL`` points the en-passant twins at a copy
elsewhere (Amendment B: mutation drills never write to ``golden/``).
``GUOFISH_C3B_MODEL`` and ``GUOFISH_C3B_DEVICE`` override the checkpoint and
device used by the determinism pin.
"""

import inspect
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

# Every OTHER file under tests/ deliberately imports neither `chess` nor
# `torch` — they test the C++ module and reach the reference only through
# golden files, so the Linux build boxes carry a minimal venv (numpy + pytest)
# and the C++ suite runs there unchanged. This file is the exception by
# definition: it is a test OF the Python reference, and the reference IS
# python-chess plus torch plus core.mctsv4. Skipping is the honest outcome on a
# box that cannot run the reference at all, and it costs nothing: Amendment A
# already pins the golden data to Windows / Python 3.13.7 / python-chess
# 1.11.2, so the reference is a Windows artifact and this is where it is
# checked. An ImportError at collection time would instead fail the whole suite
# on the Linux sanitizer builds for a reason that has nothing to do with them.
_NEEDS = "core.mctsv4 (the reference under test) requires it"
chess = pytest.importorskip("chess", reason=_NEEDS)
torch = pytest.importorskip("torch", reason=_NEEDS)

import chess.polyglot  # noqa: E402 — must follow the importorskip above

from core import mctsv4  # noqa: E402
from core.mctsv4 import (MCTSNode, ParallelMCTS, SearchParams,  # noqa: E402
                         TranspositionCache, board_to_tokens,
                         build_repetition_history, make_cache_key)

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ADVERSARIAL = REPO_ROOT / "golden" / "keys_adversarial.jsonl"
ADVERSARIAL_ENV = "GUOFISH_GOLDEN_KEYS_ADVERSARIAL"

DEFAULT_MODEL = REPO_ROOT / "models" / "guofish5_20M" / "v5_10.9M_best.pt"
MODEL_ENV = "GUOFISH_C3B_MODEL"
DEVICE_ENV = "GUOFISH_C3B_DEVICE"

# The en-passant token slot. board_to_tokens writes it from `ep_square is not
# None`; the whole point of defect 1 is that the Zobrist hash does not agree.
EP_TOKEN_INDEX = 66

# The determinism pin's position: index 0 of C5's accepted corpus, which is the
# same position tools/gen_gate1_golden.py re-runs for its own determinism
# self-check. Stated literally so this file does not depend on the manifest.
DETERMINISM_FEN = "r3kr2/1p1nnpp1/1bp1p1p1/p2pP1N1/P2P3P/BPP3P1/5PB1/R3K2R w KQ - 1 21"
DETERMINISM_SIMS = 2000

# The equivalence configuration's two virtual-loss magnitudes. 0.0 is the
# brief's isolation setting; 2.5 is the production magnitude the golden
# generator actually pins its own determinism check at, and the only place
# virtual loss's apply/repay ordering is exercised before concurrency lands.
EQUIVALENCE_VIRTUAL_LOSSES = (0.0, 2.5)


# ---------------------------------------------------------------------------
# A stub network
# ---------------------------------------------------------------------------
# Defects 2 and 3 are tree-logic defects: which flags a node carries, and which
# field noise is derived from. Neither depends on what the network says, so
# these tests run against a deterministic stand-in rather than the 10.9M
# checkpoint. That keeps them fast, hermetic and runnable without CUDA — and it
# means a failure here is provably in the tree logic, since the "network" is a
# closed-form function of the tokens.
#
# `require_v5_config` is duck-typed on `.config`, by deliberate design (see its
# docstring), so carrying the fields is all that is required.


class _StubConfig:
    seq_len = mctsv4.SEQ_LENGTH
    cls_index = mctsv4.CLS_INDEX
    policy_size = mctsv4.POLICY_SIZE
    d_model = 384
    activation = "gelu"
    final_norm = True


class _StubNet(torch.nn.Module):
    """A v5-shaped network whose output is a closed form of the token sum."""

    def __init__(self):
        super().__init__()
        self.config = _StubConfig()
        self.seq_length = mctsv4.SEQ_LENGTH

    def forward(self, tokens: torch.Tensor):
        t = tokens.to(torch.float32)
        position_sum = t.sum(dim=1, keepdim=True)
        index = torch.arange(mctsv4.POLICY_SIZE, dtype=torch.float32,
                             device=t.device).unsqueeze(0)
        policy = torch.sin(index * 0.001 + position_sum * 0.01)
        value = torch.tanh(position_sum * 0.0001 - 0.3).squeeze(1)
        return policy, value


@pytest.fixture
def stub_engine():
    """A one-worker, one-slot-cache, tablebase-free engine on the stub network.

    Constructed per test so no tree, cache entry or stat leaks between them.
    """
    engine = ParallelMCTS(
        model=_StubNet(),
        device=torch.device("cpu"),
        num_workers=1,
        cache_size=1,
        tablebase=None,
        params=SearchParams(),
    )
    try:
        yield engine
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _iter_tree(root: MCTSNode):
    """Yield ``(node, path)`` for every node in the subtree, path in UCI strings.

    Iterative so a deep endgame tree cannot hit the recursion limit, and in a
    fixed order (children in dict order, depth first) so two walks of two
    structurally identical trees produce comparable sequences.
    """
    stack = [(root, ())]
    while stack:
        node, path = stack.pop()
        yield node, path
        for move, child in reversed(list(node.children.items())):
            stack.append((child, path + (move.uci(),)))


def _board_at(root_board: chess.Board, path: tuple) -> chess.Board:
    board = root_board.copy(stack=False)
    for uci in path:
        board.push(chess.Move.from_uci(uci))
    return board


def _bits(x: float) -> str:
    """The exact 64-bit pattern of a float, as hex. Two values that print the
    same in decimal but differ by one ulp print differently here."""
    return struct.pack("<d", x).hex()


# ===========================================================================
# Defect 1 — the transposition cache key must partition en-passant twins
# ===========================================================================

@pytest.fixture(scope="session")
def ep_twins():
    """The `ep_twin` pairs from C3's adversarial corpus.

    Each pair is the SAME placement, castling rights and side to move, with the
    en-passant field set on one side and cleared on the other. That is exactly
    the axis `board_to_tokens` records at token 66 and the Zobrist hash may not.
    """
    path = Path(os.environ.get(ADVERSARIAL_ENV, DEFAULT_ADVERSARIAL))
    if not path.exists():
        pytest.fail(
            f"adversarial key golden missing: {path}\n"
            "Global Rule 2: regenerate with `python tools/gen_key_golden.py`, "
            "which runs the Python reference. Never produced from C++ output."
        )
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    twins = [r for r in rows if r["kind"] == "ep_twin"]
    assert twins, (
        f"{path} contains no `ep_twin` pairs. This test would pass vacuously; "
        "the corpus is wrong, not the cache key."
    )
    return twins


def test_ep_cache_key(ep_twins):
    """`make_cache_key` partitions the en-passant twins; the raw Zobrist hash
    does not.

    Three claims, in the order they matter:

    a) The twins are genuinely different POSITIONS as far as the network is
       concerned — they differ at token 66. Without this the collision would be
       harmless and there would be nothing to fix.
    b) `make_cache_key` separates all of them.
    c) The raw Polyglot Zobrist hash — the pre-fix key — collides on a non-empty
       subset of them. That subset is computed, not hardcoded: Polyglot folds
       the ep file in whenever an enemy pawn is ADJACENT (pins are not
       considered), so the pairs it collides on are the ones with no adjacent
       capturer at all.

    And the consequence is shown end to end on a real `TranspositionCache`: an
    entry stored under one twin must not be served to the other.
    """
    collide_under_zobrist = []
    failures = []

    for pair in ep_twins:
        a = chess.Board(pair["a"]["fen"])
        b = chess.Board(pair["b"]["fen"])
        pair_id = pair["pair_id"]

        # (a) the network tokenizes them differently.
        token_a = int(board_to_tokens(a)[EP_TOKEN_INDEX].item())
        token_b = int(board_to_tokens(b)[EP_TOKEN_INDEX].item())
        if token_a == token_b:
            failures.append(
                f"{pair_id}: both sides tokenize to {token_a} at token "
                f"{EP_TOKEN_INDEX}, so this pair proves nothing about the cache "
                f"key.\n  a: {pair['a']['fen']}\n  b: {pair['b']['fen']}")
            continue

        # (b) the fixed key separates them.
        key_a, key_b = make_cache_key(a), make_cache_key(b)
        if key_a == key_b:
            failures.append(
                f"{pair_id}: make_cache_key COLLIDES on a pair the network sees "
                f"differently (token {EP_TOKEN_INDEX}: {token_a} vs {token_b}). "
                f"key={key_a}\n  a: {pair['a']['fen']}\n  b: {pair['b']['fen']}")
            continue

        # (c) does the pre-fix key collide here?
        zob_a = chess.polyglot.zobrist_hash(a)
        zob_b = chess.polyglot.zobrist_hash(b)
        if zob_a == zob_b:
            collide_under_zobrist.append((pair_id, a, b, zob_a, token_a, token_b))

    assert not failures, (
        f"{len(failures)} en-passant twin(s) are not partitioned by the cache "
        f"key:\n" + "\n".join(failures))

    assert collide_under_zobrist, (
        "No en-passant twin in the corpus collides under the raw Polyglot "
        "Zobrist hash, so this test cannot show the pre-fix key was broken. "
        "Either the corpus lost its no-adjacent-capturer pairs or python-chess "
        "changed its Polyglot ep rule; both invalidate the test, and neither is "
        "fixed by relaxing it.")

    # The defect, played out on the real cache: store under one twin, ask for
    # the other. Pre-fix this returned the wrong policy and value.
    for pair_id, a, b, zob, token_a, token_b in collide_under_zobrist:
        assert chess.polyglot.zobrist_hash(a) == chess.polyglot.zobrist_hash(b)

        cache = TranspositionCache(max_size=16)
        policy_a = torch.full((mctsv4.POLICY_SIZE,), 1.0)
        cache.put(make_cache_key(a), policy_a, 0.75)

        assert cache.get(make_cache_key(a)) is not None, (
            f"{pair_id}: the entry cannot even be read back under its own key.")
        assert cache.get(make_cache_key(b)) is None, (
            f"{pair_id}: a cache entry stored for a position with ep token "
            f"{token_a} was served to a position with ep token {token_b}. Both "
            f"share Zobrist {zob:#018x}; the key must not be the hash alone.")


# ===========================================================================
# Defect 2 — a claimable-draw terminal node must stay recoverable
# ===========================================================================

# A rook endgame at halfmove clock 99: EVERY legal move is a rook or king move,
# so every child arrives at clock 100 and is drawn by the fifty-move rule. The
# position is not `is_game_over()` — that needs seventy-five — so a host is free
# to hand any of those children straight back to us as the position to move from.
FIFTY_MOVE_FEN = "8/8/4k3/8/8/4K3/8/6R1 w - - 99 100"

# Knight shuffles returning to the start position for the third time. After the
# stack below, the start position stands at three occurrences and 1.Nf3 reaches
# a position that has already occurred twice — a threefold on the child.
REPETITION_MOVES = ("g1f3", "g8f6", "f3g1", "f6g8",
                    "g1f3", "g8f6", "f3g1", "f6g8")
REPETITION_MOVE = "g1f3"


def _repetition_board() -> chess.Board:
    board = chess.Board()
    for uci in REPETITION_MOVES:
        board.push_uci(uci)
    return board


def _assert_claimable_draws_are_not_expanded(engine, root_board):
    """No node the search marked terminal on a CLAIMABLE draw may be expanded.

    "Claimable" is measured, not assumed: a terminal node whose own board is not
    `is_game_over()` can only have been marked by `_draw_by_rule` (fifty-move or
    threefold), because every other terminal path is gated on `is_game_over()`.
    Those are the nodes a host can force us to play from, and the ones that must
    stay expandable.
    """
    terminal_by_rule = 0
    offenders = []
    for node, path in _iter_tree(engine.root):
        if not node.is_terminal:
            continue
        board = _board_at(root_board, path)
        if board.is_game_over():
            continue  # checkmate/stalemate/75-move/5-fold: expanded is allowed
        terminal_by_rule += 1
        if node.is_expanded:
            offenders.append(
                f"  {'/'.join(path) or '<root>'}  is_expanded=True "
                f"children={len(node.children)}  {board.fen()}")
    assert terminal_by_rule, (
        "the search reached no claimable-draw terminal node, so this assertion "
        "is vacuous — the position or simulation count is wrong, not the engine")
    assert not offenders, (
        f"{len(offenders)} of {terminal_by_rule} claimable-draw terminal nodes "
        f"are marked is_expanded. Promoting one to root makes search() skip "
        f"_expand_root and return None — 'bestmove 0000' on a board with legal "
        f"moves:\n" + "\n".join(offenders))
    return terminal_by_rule


def _assert_promoted_terminal_plays_on(engine, board, move, label):
    """Promote `move`'s node to root and require a legal move out of it."""
    child = engine.root.children[move]
    assert child.is_terminal, (
        f"{label}: {move.uci()} was expected to be marked terminal by rule, but "
        f"is_terminal is False (visits={child.visit_count}). The scenario did "
        f"not reproduce; the guard below would pass vacuously.")
    assert child.visit_count > 0, (
        f"{label}: {move.uci()} was never visited, so its terminal mark says "
        f"nothing about what the search does.")
    assert child.terminal_value == 0.0, (
        f"{label}: a draw by rule must back up 0.0, got {child.terminal_value!r}")

    engine.apply_move(move)
    board.push(move)
    assert engine.root is child, (
        f"{label}: apply_move did not promote the terminal node to root.")
    assert not board.is_game_over(), (
        f"{label}: python-chess ends the game here, so a host could not force "
        f"us to move and the scenario is not the one under test: {board.fen()}")

    played = engine.search(board, num_simulations=64)
    assert played is not None, (
        f"{label}: search() returned None from {board.fen()}, which the UCI "
        f"layer emits as 'bestmove 0000' — a forfeit — while "
        f"{board.legal_moves.count()} legal moves exist. This is the defect.")
    assert played != chess.Move.null(), (
        f"{label}: search() returned the null move ({played.uci()}).")
    assert played in board.legal_moves, (
        f"{label}: search() returned {played.uci()}, which is not legal in "
        f"{board.fen()}")
    assert engine.root.children, (
        f"{label}: the promoted root was played from but still has no children.")
    assert not engine.root.is_terminal, (
        f"{label}: the promoted root has children and is still marked terminal. "
        f"No node may be both branching and terminal.")

    policy = engine.get_policy(board, num_simulations=64)
    assert policy, (
        f"{label}: get_policy() returned an empty distribution from "
        f"{board.fen()}; pick_engine_move's argmax produces no move at all.")
    assert set(policy) <= set(board.legal_moves), (
        f"{label}: get_policy() returned illegal moves: "
        f"{sorted(m.uci() for m in set(policy) - set(board.legal_moves))}")
    return played


def test_terminal_guard(stub_engine):
    """A node drawn by the fifty-move rule or by threefold repetition must not
    present as expanded-with-no-children, and must still yield a legal move when
    it is promoted to the search root.

    Run twice, once per rule, because the two reach `_draw_by_rule` by different
    conditions (`halfmove_clock >= 100` vs the repetition table) and a fix that
    only covered one would be worse than no fix at all.
    """
    assert chess.Move.null().uci() == "0000", (
        "the forfeit this test exists to prevent is spelled differently now")

    # --- fifty-move rule ------------------------------------------------
    board = chess.Board(FIFTY_MOVE_FEN)
    assert not board.is_game_over()
    assert board.halfmove_clock == 99

    best = stub_engine.search(board, num_simulations=64)
    assert best is not None and best in board.legal_moves
    drawn = _assert_claimable_draws_are_not_expanded(stub_engine, board)
    assert drawn == board.legal_moves.count(), (
        f"every one of the {board.legal_moves.count()} legal moves reaches "
        f"clock 100, so all of them should be drawn by rule; {drawn} were")

    _assert_promoted_terminal_plays_on(stub_engine, board, best, "fifty-move")

    # --- threefold repetition -------------------------------------------
    stub_engine.reset()
    stub_engine.clear_cache()

    board = _repetition_board()
    key = board._transposition_key()
    assert build_repetition_history(board).get(key) == 3, (
        "the repetition scenario did not set up: the root position should "
        "stand at three occurrences")
    assert not board.is_game_over(), (
        "python-chess must NOT end this game — a threefold is only a draw if "
        "claimed, which is the entire premise of the defect")

    stub_engine.search(board, num_simulations=400)
    _assert_claimable_draws_are_not_expanded(stub_engine, board)
    _assert_promoted_terminal_plays_on(
        stub_engine, board, chess.Move.from_uci(REPETITION_MOVE), "threefold")


# ===========================================================================
# Defect 3 — Dirichlet noise must not compound on a reused node
# ===========================================================================

def _noise_defaults():
    """Read alpha/epsilon off the function rather than restating them, so a
    retune of the reference does not turn this test into a lie."""
    params = inspect.signature(ParallelMCTS._add_dirichlet_noise).parameters
    return params["alpha"].default, params["epsilon"].default


def test_dirichlet_idempotence(stub_engine):
    """Applying Dirichlet noise twice must not compound it, and the mix must be
    taken from the stored network priors rather than from the live ones.

    The two halves are separable and both are checked:

    * IDEMPOTENCE — with the RNG reseeded identically, five applications must
      leave byte-identical priors. Pre-fix each one mixed into the previous
      result, so the network's contribution decayed as 0.75^k.
    * PROVENANCE — `base_prior` must still hold what `expand()` wrote, and each
      child's live prior must equal `(1 - eps) * base_prior + eps * n` to the
      bit, for the `n` the seeded draw produced. That is the difference between
      "the noise happens to be the same" and "the noise is a function of the
      untouched priors".

    Then the same claim through the public path — two `search(...,
    add_dirichlet_noise=True)` calls on a reused tree — because that is how the
    compounding actually happened in play.
    """
    alpha, epsilon = _noise_defaults()

    board = chess.Board()
    root = MCTSNode()
    stub_engine._expand_root(root, board)
    assert root.children, "the stub network expanded no children"

    at_expansion = {move: child.prior for move, child in root.children.items()}
    assert all(child.base_prior == at_expansion[move]
               for move, child in root.children.items()), (
        "expand() must seed base_prior from the network prior")

    snapshots = []
    for _ in range(5):
        np.random.seed(1234)
        stub_engine._add_dirichlet_noise(root)
        snapshots.append({m: c.prior for m, c in root.children.items()})

    first = snapshots[0]
    assert any(first[m] != at_expansion[m] for m in at_expansion), (
        "the noise changed nothing, so idempotence is vacuous here")

    for i, snap in enumerate(snapshots[1:], start=2):
        drifted = [
            f"  {m.uci()}: after 1 application {_bits(first[m])} "
            f"({first[m]!r}), after {i} {_bits(snap[m])} ({snap[m]!r}); "
            f"base_prior {root.children[m].base_prior!r}"
            for m in first if _bits(snap[m]) != _bits(first[m])
        ]
        assert not drifted, (
            f"Dirichlet noise COMPOUNDED: application {i} moved "
            f"{len(drifted)} of {len(first)} priors despite drawing identical "
            f"noise. In-place mixing decays the network's weight as "
            f"(1-{epsilon})^k:\n" + "\n".join(drifted[:10]))

    # Provenance: base_prior untouched, and the live prior is the exact mix.
    untouched = [
        f"  {m.uci()}: base_prior {c.base_prior!r} != expansion "
        f"{at_expansion[m]!r}"
        for m, c in root.children.items() if c.base_prior != at_expansion[m]
    ]
    assert not untouched, (
        "base_prior was mutated; the noise is no longer derived from the "
        "network's own priors:\n" + "\n".join(untouched[:10]))

    np.random.seed(1234)
    expected_noise = np.random.dirichlet([alpha] * len(root.children))
    mismatched = []
    for (move, child), n in zip(root.children.items(), expected_noise):
        expected = (1 - epsilon) * child.base_prior + epsilon * n
        if _bits(child.prior) != _bits(expected):
            mismatched.append(
                f"  {move.uci()}: prior {_bits(child.prior)} ({child.prior!r}) "
                f"!= (1-{epsilon})*{child.base_prior!r} + {epsilon}*{n!r} = "
                f"{_bits(expected)} ({expected!r})")
    assert not mismatched, (
        f"{len(mismatched)} of {len(root.children)} priors are not one "
        f"application of noise over the stored base_prior:\n"
        + "\n".join(mismatched[:10]))

    # --- and through the public path, on a genuinely reused tree ---------
    stub_engine.reset()
    stub_engine.clear_cache()

    board = chess.Board()
    np.random.seed(99)
    stub_engine.search(board, num_simulations=32, add_dirichlet_noise=True)
    reused_root = stub_engine.root
    after_first = {m: c.prior for m, c in reused_root.children.items()}

    np.random.seed(99)
    stub_engine.search(board, num_simulations=32, add_dirichlet_noise=True)
    assert stub_engine.root is reused_root, (
        "the second search built a new tree, so nothing was reused and the "
        "compounding scenario was never entered")
    after_second = {m: c.prior for m, c in reused_root.children.items()}

    compounded = [
        f"  {m.uci()}: {_bits(after_first[m])} -> {_bits(after_second[m])}"
        for m in after_first
        if _bits(after_first[m]) != _bits(after_second[m])
    ]
    assert not compounded, (
        f"a second search on the REUSED root re-mixed noise into already-noised "
        f"priors ({len(compounded)} of {len(after_first)} children moved "
        f"despite an identical seeded draw):\n" + "\n".join(compounded[:10]))


# ===========================================================================
# The equivalence configuration must be bit-deterministic
# ===========================================================================

@pytest.fixture(scope="session")
def reference_model():
    """The v5 checkpoint the golden data was generated with, on the same device.

    Session-scoped: loading it costs seconds and it is never mutated here.
    """
    path = Path(os.environ.get(MODEL_ENV, DEFAULT_MODEL))
    if not path.exists():
        pytest.skip(
            f"reference checkpoint not present: {path}. Set {MODEL_ENV} to "
            f"point at it. Every other test in this file runs without it.")
    from playing.v5.playv5 import load_model  # noqa: PLC0415 — heavy import

    device_name = os.environ.get(
        DEVICE_ENV, "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    return load_model(path, device), device


@pytest.fixture
def equivalence_globals():
    """Set the module globals the equivalence configuration pins, and put them
    back. `VIRTUAL_LOSS` and `GATE1_CANONICAL_ORDER` are module-level in
    `core.mctsv4` (nodes use __slots__; per-node config would cost more than the
    search), so a test that changes them must restore them or it silently
    reconfigures every test that follows in the session."""
    saved = (mctsv4.VIRTUAL_LOSS, mctsv4.GATE1_CANONICAL_ORDER)
    try:
        yield
    finally:
        mctsv4.VIRTUAL_LOSS, mctsv4.GATE1_CANONICAL_ORDER = saved


def _tree_fingerprint(root: MCTSNode) -> list[tuple]:
    """Every node of the tree, floats as `repr` strings.

    `repr` and not `==`: `float('nan') == float('nan')` is False, so an equality
    diff reports a spurious difference on a NaN that BOTH runs produced, and
    `-0.0 == 0.0` is True, so it hides a real sign difference. `repr` round-trips
    exactly for finite floats in CPython and distinguishes both cases. Unvisited
    children are included — they carry real priors in an order that is itself
    part of what the golden trees record.
    """
    return [
        (path, node.visit_count, repr(node.value_sum), repr(node.prior),
         repr(node.base_prior), node.is_expanded, node.is_terminal,
         repr(node.terminal_value), node.vloss_count)
        for node, path in _iter_tree(root)
    ]


def _describe_divergence(first: list[tuple], second: list[tuple]) -> str:
    if len(first) != len(second):
        return (f"the two runs built trees of different SIZE: {len(first)} "
                f"nodes vs {len(second)}")
    fields = ("visit_count", "value_sum", "prior", "base_prior", "is_expanded",
              "is_terminal", "terminal_value", "vloss_count")
    lines = []
    for a, b in zip(first, second):
        if a == b:
            continue
        if a[0] != b[0]:
            lines.append(f"  tree SHAPE diverges: {'/'.join(a[0]) or '<root>'} "
                         f"vs {'/'.join(b[0]) or '<root>'}")
        else:
            differing = [f"{name}: {x!r} vs {y!r}"
                         for name, x, y in zip(fields, a[1:], b[1:]) if x != y]
            lines.append(f"  {'/'.join(a[0]) or '<root>'}  "
                         + "; ".join(differing))
        if len(lines) >= 10:
            lines.append("  ... (truncated)")
            break
    return "\n".join(lines)


@pytest.mark.parametrize("virtual_loss", EQUIVALENCE_VIRTUAL_LOSSES)
def test_equivalence_determinism(reference_model, equivalence_globals,
                                 virtual_loss):
    """Two searches under the golden-data configuration must agree node for node.

    This is what makes `golden/gate1_*` a description of the REFERENCE rather
    than a description of one lucky run. If it fails, every C++ acceptance
    criterion downstream is comparing against a number that the reference itself
    would not reproduce.

    The configuration is `tools/gen_gate1_golden.py`'s, restated here rather than
    imported so the pin does not move when the generator does: one worker,
    tablebase off, Dirichlet off, `cache_size=1`, canonical child ordering, and
    the cache cleared before each run. Both virtual-loss magnitudes are covered
    — 0.0 is the isolation setting the brief names, 2.5 is the production
    magnitude and the only place the apply/repay ordering is exercised.
    """
    model, device = reference_model
    mctsv4.GATE1_CANONICAL_ORDER = True
    mctsv4.VIRTUAL_LOSS = virtual_loss

    engine = ParallelMCTS(
        model=model,
        device=device,
        num_workers=1,
        cache_size=1,
        tablebase=None,
        params=SearchParams(),
    )
    try:
        runs = []
        for _ in range(2):
            engine.reset()
            engine.clear_cache()
            board = chess.Board(DETERMINISM_FEN)
            move = engine.search(board, num_simulations=DETERMINISM_SIMS,
                                 add_dirichlet_noise=False)
            runs.append((move, _tree_fingerprint(engine.root)))
    finally:
        engine.shutdown()

    (move_a, tree_a), (move_b, tree_b) = runs

    assert len(tree_a) > 1000, (
        f"the reference tree has only {len(tree_a)} nodes at "
        f"{DETERMINISM_SIMS} simulations; a determinism claim over a tree this "
        f"small is not the claim C5 needs")

    assert move_a == move_b, (
        f"the equivalence configuration chose different moves on two identical "
        f"runs: {move_a} vs {move_b}")

    assert tree_a == tree_b, (
        f"the equivalence configuration (1 worker / VL {virtual_loss} / TB off "
        f"/ Dirichlet off / cache_size=1 / canonical order) is NOT "
        f"bit-deterministic. golden/gate1_* would describe one run rather than "
        f"the reference's behaviour, and every C++ acceptance criterion built "
        f"on it would be unreproducible.\n"
        + _describe_divergence(tree_a, tree_b))
