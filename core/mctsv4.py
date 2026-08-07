"""Parallel MCTS with Batched Neural Network Evaluation -- V5 students only.

Architecture:
- Multiple worker threads traverse the MCTS tree using PUCT
- Workers submit leaf nodes to a shared queue and wait for evaluation
- Single evaluator thread batches positions and runs them through the NN
- Virtual loss prevents workers from all exploring the same path

ARCHITECTURE SUPPORT: the v5 multi-PV student ONLY
(training/v5_multiPV/model_v5.ChessTransformerV5). Unlike core.mctsv3, which
accepts anything advertising seq_length=68 and assumes V2 when a model says
nothing at all, this file requires the model to carry the ModelConfig it was
built from and validates the three contracts the search depends on against it
(see require_v5_config): the 68-token encoding, the CLS slot the value head
reads, and the 4096 from*64+to policy layout. Legacy V2 nets (guofish2 ..
guofish4) carry no config and are rejected at construction -- run those on
core.mctsv3, which is unchanged and still supports them.

Being v5-exclusive is what lets the numerics here be stated rather than
assumed: the shards' encoding (data/multiPV/mirror.py), the policy index
(data/multiPV/labels.py:move_index) and the bf16 training precision
(training/v5_multiPV/configs/base.yaml) are all pinned below and checked
against the model, instead of being inherited V2 conventions that merely
happen to still hold.

Search tunables (exploration, first-play urgency, policy sharpening) live in a
SearchParams instance owned by ParallelMCTS and shared with every worker, not in
module globals -- see that class. The engine's UCI layer exposes all four as
options and mutates that instance in place between moves.

Usage:
    from core.mctsv4 import ParallelMCTS, SearchParams
    mcts = ParallelMCTS(model, device, num_workers=8)
    best_move = mcts.search(board, num_simulations=800)

    # retuned search
    mcts = ParallelMCTS(model, device,
                        params=SearchParams(c_init=1.25, fpu_tree=-0.2,
                                            policy_temperature=0.8))

    # or, architecture-routed, which is what the engine actually calls:
    from playing.v5.playv5 import build_mcts
"""

import math
import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Optional

import chess
import chess.polyglot
import chess.syzygy
import numpy as np
import torch

# === Measurement instrumentation (PORT RECON ONLY -- do not commit) ===
# Every probe below is guarded by `if _INSTR:`, which is False unless the
# environment variable GUOFISH_INSTR=1 is set at import time. With it unset the
# search executes exactly the instructions it did before: no timing calls, no
# counter updates, no allocations. This exists to measure the Python engine for
# the C++ port and is expected to be reverted afterwards.
#
# Counters live in a per-thread defaultdict so N workers never contend on a lock
# in the hot path (a shared lock would itself dominate the numbers we are trying
# to read). instr_merge() sums the per-thread buckets on the main thread.
_INSTR = os.environ.get("GUOFISH_INSTR") == "1"
_instr_local = threading.local()
_instr_buckets: list = []
_instr_buckets_lock = threading.Lock()
# Per-batch records for the evaluator breakdown (item 6). Appended only from the
# single evaluator thread, so no lock is needed.
_instr_batches: list = []


# Item 12/13 blast radius: what the cache key does NOT distinguish. Maps
# cache key -> (ep token 66, halfmove_clock, came_from_tablebase) as recorded
# at cache.put time, so a later cache *hit* can be compared against the position
# actually being asked about.
#
# Item 12 is now FIXED (make_cache_key folds ep_square into the key), so
# `cache_hit_ep_mismatch` should read exactly 0 -- the probe is kept as a
# regression check. `cache_hit_hmc_mismatch` (item 13) is still live: the key
# deliberately does not include the halfmove clock.
_instr_keymeta: dict = {}
_instr_keymeta_lock = threading.Lock()


def _instr_ep_token(board) -> int:
    if board.ep_square is not None:
        return TOKEN_EP_BASE + chess.square_file(board.ep_square)
    return TOKEN_EP_NONE


def _instr_bucket():
    """This thread's counter dict, registered for later merging."""
    b = getattr(_instr_local, "b", None)
    if b is None:
        b = defaultdict(float)
        _instr_local.b = b
        with _instr_buckets_lock:
            _instr_buckets.append(b)
    return b


def instr_reset() -> None:
    """Zero every thread's counters and drop recorded batches."""
    with _instr_buckets_lock:
        for b in _instr_buckets:
            b.clear()
    del _instr_batches[:]


def instr_merge() -> dict:
    """Sum all per-thread counters into one plain dict."""
    out: defaultdict = defaultdict(float)
    with _instr_buckets_lock:
        for b in _instr_buckets:
            for k, v in list(b.items()):
                out[k] += v
    return dict(out)


# === The v5 model contract ===
# These three are not local conventions: they are the format the training
# shards were written in, so a model whose ModelConfig disagrees with any of
# them would be fed inputs it was never trained on and would silently produce
# garbage. require_v5_config checks all three before the first forward pass.
#
# SEQ_LENGTH   board tokenization length (see board_to_tokens below); matches
#              data/multiPV/record_format.py's 68-wide `tokens` field.
# CLS_INDEX    slot the v5 value head pools from (ModelConfig.cls_index).
# POLICY_SIZE  flat policy width; the index is from_square*64 + to_square,
#              identical to data/multiPV/labels.py:move_index.
SEQ_LENGTH = 68
CLS_INDEX = 67
POLICY_SIZE = 4096

# Precision every forward pass runs in. v5 was trained under bf16 autocast
# (training/v5_multiPV/configs/base.yaml: `amp: bf16  # matches inference
# precision`), so evaluating in bf16 reproduces training numerics exactly.
# playv5.load_model already stores the weights in bf16 on Ampere+, which makes
# this autocast a no-op there; it still matters for fp32 weights (CPU-loaded
# checkpoints, tests) where it keeps the trained precision.
AUTOCAST_DTYPE = torch.bfloat16

# MCTS hyperparameters that are NOT per-search tunable. Both are read as module
# globals on the hot path by code that holds no SearchParams reference, so they
# stay live globals: `setattr(mctsv4, "VIRTUAL_LOSS", x)` takes effect on the
# next read. The three tunables (exploration, FPU, policy temperature) do NOT
# work that way -- see SearchParams.
VIRTUAL_LOSS = 2.5  # Penalize in-flight nodes to encourage exploration diversity
MAX_TREE_DEPTH = 80  # Maximum simulation depth to prevent endgame slowdowns

# === Search tunable defaults ===
# These seed SearchParams' field defaults at import time. Unlike the two globals
# above, rebinding them later does NOT change anything: dataclass defaults are
# captured when the class body executes. Override by constructing a SearchParams
# and handing it to ParallelMCTS, which is what the UCI layer does.
#
# C_PUCT_INIT / C_PUCT_BASE replace the former static C_PUCT = 2.00. The static
# constant is gone: exploration now scales with the parent's visit count via
# AlphaZero's log form (see SearchParams.c_puct). C_PUCT_INIT keeps the value
# the phase-3 sweep landed on (benchmarking/engine/logs/cpuct_phase3_finalists_
# results.md), so at low visit counts the search behaves as it did before; the
# log term only adds width as a node gets hot.
C_PUCT_INIT = 1.43
C_PUCT_BASE = 19652.0     # AlphaZero's value; larger => slower growth
# First Play Urgency: the Q an UNVISITED child is scored with, split by WHERE the
# selection happens. 0.0 everywhere is the original hardcoded behaviour
# (unvisited == drawn); FPU_TREE carries the 0.30 the single-knob version was
# tuned to, FPU_ROOT is pinned at 0.
#
# The two are separated because they are not the same problem. At the root every
# legal move gets visited within the first few dozen simulations whatever FPU
# says, so the constant there mostly reorders those opening visits -- and the
# root is the one node whose visit distribution IS the answer (bestmove is
# argmax visits), so biasing it distorts the output for no search benefit.
# Deeper in the tree the same constant decides whether a subtree's long shots
# are examined at all, which is a genuine width-vs-depth knob.
FPU_ROOT = 0.0
FPU_TREE = 0.30
# Inference-time policy sharpening exponent T, applied as P^(1/T) at expansion.
# 1.0 is a no-op.
POLICY_TEMPERATURE = 1.0

# === Gate 1 canonical child ordering (C5 port equivalence only) ===
# OFF by default, and the default path must stay bit-identical to the behaviour
# every benchmark in this repository was measured on. tools/gen_gate1_golden.py
# is the only thing that turns it on.
#
# WHY IT EXISTS. Python resolves exact PUCT ties by child insertion order, i.e.
# by python-chess's move-generation order (`select_child` iterates
# `self.children.values()`, and dicts preserve insertion order). ~1% of selection
# steps are exact ties -- mostly unvisited siblings at identical
# `fpu + c*P*sqrt(N)`, plus the four promotion moves that share one policy index
# and therefore one prior. Reproducing another library's generation order in C++
# is a fragile dependency on its internals, so scope 2.6 makes the C++ engine
# order children canonically by (from, to, promotion) over the normalised UCI
# string, and this flag makes the reference do the same for the Gate 1 run.
#
# WHY THE PERMUTE HAPPENS AFTER expand() AND NOT BY SORTING legal_moves FIRST.
# `torch.softmax` is not permutation-invariant: recon measured 109/200 random
# permutations bit-identical and a max delta of 3e-7 on the rest, and sorting
# before the softmax changed 26,569 of 51,927 nodes end to end. So a sorted
# `legal_moves` list would be a numerics change wearing an ordering change's
# clothes, and Gate 1 would be comparing two different searches. The discipline
# is therefore: gather and softmax in GENERATION order, then permute the
# resulting children into canonical order. Reduction order is never canonical;
# storage order always is.
#
# Sorting on `move.uci()` is the same key C1 established: python-chess's
# generators already emit standard (non-Chess960) castling moves, so `uci()` is
# the normalised destination -- e1g1, not e1h1 -- and byte order on that string
# is exactly the (from_file, from_rank, to_file, to_rank, promotion) tuple order.
GATE1_CANONICAL_ORDER = False


def _canonicalize_children(node: 'MCTSNode') -> None:
    """Permute `node.children` into canonical UCI order, in place.

    A no-op unless GATE1_CANONICAL_ORDER is set. Idempotent, so calling it on an
    already-expanded node (expand() returns early in that case) costs a sort and
    changes nothing.
    """
    if not GATE1_CANONICAL_ORDER or not node.children:
        return
    with node.lock:
        node.children = dict(sorted(node.children.items(),
                                    key=lambda item: item[0].uci()))


# === Mode 2: Syzygy tablebase leaf evaluation ===
# When a leaf has <= TABLEBASE_MAX_PIECES pieces (both kings counted), the
# tablebase knows the exact game-theoretic result, so we override the neural
# value head's output with the tablebase WDL. Policy logits still come from the
# network (tablebases don't supply a move-quality distribution). This is
# independent of Mode 1 (the UCI-layer bypass for <= 5-piece root positions);
# Mode 2 fires at *interior leaves* reached during a search of a larger root.
TABLEBASE_MAX_PIECES = 5


@dataclass
class SearchParams:
    """The knobs a caller can turn per ParallelMCTS instance.

    One object is created by ParallelMCTS and handed to every worker, so all
    threads read the same values, and mutating a field takes effect on the next
    node scored (the UCI layer relies on this for `setoption` between moves).
    Nodes never copy these -- MCTSNode uses __slots__ and there are millions of
    them, so per-node config would cost more memory than the search.

    SIGN CONVENTION for both FPU fields: every node's Q is from the perspective
    of the MOVER who reached it (see MCTSNode.backpropagate), so a child's Q is
    from the perspective of the player choosing at the parent. FPU is therefore
    "what we assume a move is worth before trying it", in that player's own
    terms: 0.0 == drawn (the original hardcoded behaviour), negative ==
    pessimistic, which narrows search onto already-visited moves, positive ==
    optimistic, which widens it.

    `fpu_root` applies when selecting among the SEARCH ROOT's children,
    `fpu_tree` everywhere below it. Which one applies is decided by the
    simulation's depth at selection time, not by a flag on the node: tree reuse
    promotes a child to root between moves (ParallelMCTS.apply_move), so any
    root-ness stored on a node would be stale one move later. Under pondering
    the "root" is the branch node that ponder search starts from, which is the
    consistent reading -- it is that search's argmax-visits node.

    Note this is ABSOLUTE FPU, not the parent-relative FPU-reduction some
    engines use. The reduction form would be `-parent.q_value - reduction`
    (negated because the parent's Q is stored from the opponent's side); if that
    variant is ever wanted it belongs here as a separate field rather than as a
    reinterpretation of these two.
    """

    c_init: float = C_PUCT_INIT
    c_base: float = C_PUCT_BASE
    fpu_root: float = FPU_ROOT
    fpu_tree: float = FPU_TREE
    policy_temperature: float = POLICY_TEMPERATURE

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Raise ValueError on a setting that would corrupt the search.

        Called on construction and again by callers that mutate fields in place
        (the UCI layer validates a candidate before assigning it, so a bad
        `setoption` leaves the running engine on its previous value).
        """
        if not self.c_base > 0:
            raise ValueError(f"c_base must be > 0 (log domain), got {self.c_base}")
        if not self.c_init >= 0:
            raise ValueError(f"c_init must be >= 0, got {self.c_init}")
        if not self.policy_temperature > 0:
            raise ValueError("policy_temperature must be > 0 (it divides the "
                             f"logits), got {self.policy_temperature}")
        for name in ("fpu_root", "fpu_tree"):
            value = getattr(self, name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [-1, 1], the range the "
                                 f"value head can produce, got {value}")

    def c_puct(self, parent_visits: float) -> float:
        """AlphaZero's visit-scaled exploration constant:

            c(N) = c_init + log((N + c_base + 1) / c_base)

        N is the PARENT's visit count, so every child of one parent shares a
        c_puct -- which is why select_child computes this once per selection
        step rather than once per child.

        The log term is ~0 while N << c_base and grows slowly after: it adds
        +0.04 at N=800, +0.29 at N=6k and +0.57 at N=15k, independent of c_init.
        So a cold node explores at essentially c_init -- the way the old static
        C_PUCT did -- and only a node the search keeps returning to widens out.
        """
        return self.c_init + math.log(
            (parent_visits + self.c_base + 1.0) / self.c_base)

    def fpu(self, at_root: bool) -> float:
        """The FPU that applies at this point in the tree. See the class
        docstring for why root and tree are separate constants."""
        return self.fpu_root if at_root else self.fpu_tree

    def describe(self) -> str:
        return (f"c_init={self.c_init:g} c_base={self.c_base:g} "
                f"fpu_root={self.fpu_root:+g} fpu_tree={self.fpu_tree:+g} "
                f"policy_temperature={self.policy_temperature:g}")


def count_pieces(board: chess.Board) -> int:
    """Total pieces on the board, both kings included. popcount on the occupied
    bitboard — cheaper than building piece_map(), which matters in the hot path."""
    return chess.popcount(board.occupied)


# Fields of training/v5_multiPV/model_v5.ModelConfig that must be present for a
# module to be recognised as a v5 student.
_V5_CONFIG_FIELDS = ("seq_len", "cls_index", "policy_size", "d_model",
                     "activation", "final_norm")


def require_v5_config(model: torch.nn.Module):
    """Return `model`'s ModelConfig, raising ValueError if it is not a v5 student.

    Checked structurally (duck-typed on `.config`) rather than with an
    isinstance against ChessTransformerV5, for two reasons: `core` keeps no
    import dependency on the training tree, and a torch.ao dynamically-quantized
    copy of the student still passes.

    The discriminator is sound because carrying a ModelConfig is exactly what
    separates the generations — the legacy V2 nets have no `config` attribute
    at all, and a V2 *checkpoint*'s `config` key holds training
    hyperparameters, which playv5.load_model never attaches to the module.

    Raises on any mismatch rather than warning: every contract below is an
    input-format assumption, so a violation means silently evaluating a
    position the network never saw in that form.
    """
    config = getattr(model, "config", None)
    missing = [f for f in _V5_CONFIG_FIELDS if not hasattr(config, f)]
    if config is None or missing:
        detail = ("has no `.config`" if config is None
                  else f"`.config` is missing {', '.join(missing)}")
        raise ValueError(
            f"core.mctsv4 supports only the v5 multi-PV student "
            f"(training/v5_multiPV/model_v5.ChessTransformerV5), which carries "
            f"its ModelConfig on the module. {type(model).__name__} {detail}. "
            f"Legacy V2 nets (guofish2..guofish4) are still supported by "
            f"core.mctsv3 — use playv5.build_mcts to route automatically."
        )

    for field, expected, why in (
        ("seq_len", SEQ_LENGTH, "board_to_tokens emits this many tokens"),
        ("cls_index", CLS_INDEX, "the value head pools from this slot"),
        ("policy_size", POLICY_SIZE, "priors are indexed from_square*64 + to_square"),
    ):
        actual = getattr(config, field)
        if actual != expected:
            raise ValueError(
                f"Unsupported v5 variant: config.{field}={actual}, expected "
                f"{expected} ({why}). core.mctsv4 is pinned to the shard format "
                f"in data/multiPV/record_format.py."
            )

    # The module's own advertised length must agree with its config; they are
    # set from the same place in model_v5, so a mismatch means the module was
    # patched after construction.
    model_seq_length = getattr(model, "seq_length", config.seq_len)
    if model_seq_length != config.seq_len:
        raise ValueError(
            f"Inconsistent model: seq_length={model_seq_length} but "
            f"config.seq_len={config.seq_len}."
        )

    return config


def wdl_to_value(wdl: int) -> float:
    """Map a Syzygy WDL result (side-to-move perspective) onto the value head's
    [-1, +1] scale:

        2 -> +1.0 (win)        1 -> +0.5 (cursed win)   0 -> 0.0 (draw)
       -1 -> -0.5 (blessed loss)            -2 -> -1.0 (loss)

    +1/-1 endpoints match the bounded range the value head produces, so a
    tablebase win/loss doesn't read as out-of-distribution to MCTS. The +/-0.5
    mappings treat 50-move-rule cursed/blessed results as half-decisive (the
    safer choice than counting them as full win/loss)."""
    return wdl / 2.0


def probe_tablebase_value(tablebase: chess.syzygy.Tablebase,
                          board: chess.Board) -> Optional[float]:
    """Probe WDL for `board` and return the value in ABSOLUTE (White) perspective,
    matching the neural value head's convention, or None if the position isn't
    covered by the loaded tables.

    probe_wdl reports from the side-to-move's perspective, so we negate when
    Black is to move to express it from White's perspective. Returning an
    absolute-perspective value lets the caller drop it straight into the same
    cache + mover-perspective conversion the NN value already flows through
    (see MCTSWorker._run_simulation), instead of re-deriving the sign at the
    backup site — which is exactly where tablebase perspective bugs hide."""
    try:
        wdl = tablebase.probe_wdl(board)
    except (chess.syzygy.MissingTableError, KeyError, ValueError):
        # MissingTableError subclasses KeyError; ValueError guards malformed
        # probes. Any miss => caller keeps the neural value.
        return None
    stm_value = wdl_to_value(wdl)
    return stm_value if board.turn == chess.WHITE else -stm_value


def build_repetition_history(board: chess.Board) -> dict:
    """Count the transposition keys of every prior position that could still
    repeat the current line, so simulations can detect threefold draws.

    Keyed by ``chess.Board._transposition_key()`` (the same key python-chess's
    own ``is_repetition`` uses: piece placement + side-to-move + castling + a
    *legal* en-passant file). The count includes the current (root) position.

    We only walk back to the halfmove-clock horizon: a position before the last
    zeroing move (pawn push / capture) has different material or pawn structure,
    so it can never share a key with a future position -- counting further is
    harmless but wasted. This mirrors is_repetition()'s own bound.

    Built ONCE per search from the root board, which retains the full game move
    stack. Simulations copy the board with ``stack=False`` for speed (no move
    history), so they cannot call is_repetition() themselves; they instead add
    their within-line repeats on top of this precomputed history (see
    MCTSWorker._draw_by_rule). The returned dict is read-only across workers.
    """
    _h0 = 0.0
    if _INSTR:
        _h0 = time.perf_counter()
    counter: dict = {}
    probe = board.copy()
    counter[probe._transposition_key()] = 1
    for _ in range(min(probe.halfmove_clock, len(probe.move_stack))):
        probe.pop()
        key = probe._transposition_key()
        counter[key] = counter.get(key, 0) + 1
    if _INSTR:
        _b = _instr_bucket()
        _b['rep_history_s'] += time.perf_counter() - _h0
        _b['rep_history_calls'] += 1
        _b['rep_history_plies'] += min(board.halfmove_clock, len(board.move_stack))
    return counter


CacheKey = tuple[int, Optional[int]]


def make_cache_key(board: chess.Board) -> CacheKey:
    """Transposition-cache key for `board`: Zobrist hash + raw en-passant square.

    The Zobrist hash alone is COARSER than the network's own input, so it cannot
    be the whole key. `chess.polyglot.zobrist_hash` follows the Polyglot rule and
    folds in the en-passant file only when an enemy pawn actually stands ready to
    capture; `board_to_tokens` writes token 66 from `board.ep_square is not None`
    -- unconditionally. That is the same rule the training shards were written
    with (data/pgn_parallel.py's ep branch), so the tokenization is the correct,
    trained contract and must not change. The consequence is that two positions
    the network tokenizes *differently* can share a Zobrist key, and whichever is
    evaluated second is served the first one's policy and value.

    Appending the raw ep square closes the gap without touching tokenization: any
    two boards differing at token 66 now differ in the key, because that token is
    a function of ep_square. The key is marginally finer than token 66 (which
    keeps only the file), but ep rank is implied by side-to-move, which the
    Zobrist hash already covers -- so the extra precision costs no real entries.
    """
    return (chess.polyglot.zobrist_hash(board), board.ep_square)


class TranspositionCache:
    """Thread-safe ring buffer cache for NN evaluations, keyed by make_cache_key.

    Uses a circular buffer instead of OrderedDict to avoid O(n) LRU operations.
    Stores (policy_logits, value) pairs to avoid re-evaluating positions
    reached via different move orders (transpositions).
    """

    def __init__(self, max_size: int = 500_000):
        self.max_size = max_size
        # Hash table for O(1) lookup
        self._cache: dict[CacheKey, tuple[torch.Tensor, float]] = {}
        # Ring buffer of keys for O(1) eviction (no ordering maintained).
        # None is the empty-slot sentinel; a real key is always a tuple.
        self._ring: list[Optional[CacheKey]] = [None] * max_size
        self._ring_idx = 0  # Next position to write
        self._lock = threading.Lock()

        # Stats
        self.hits = 0
        self.misses = 0

    def get(self, key: CacheKey) -> Optional[tuple[torch.Tensor, float]]:
        """Get cached (policy_logits, value) for key. Returns None on miss."""
        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                self.hits += 1
                return result
            self.misses += 1
            return None

    def put(self, key: CacheKey, policy: torch.Tensor, value: float):
        """Store (policy_logits, value) for key. Evicts oldest entry if at capacity."""
        with self._lock:
            if key in self._cache:
                # Update existing entry (no ring position change needed)
                self._cache[key] = (policy, value)
                return

            # Evict entry at current ring position if occupied
            old_key = self._ring[self._ring_idx]
            if old_key is not None:
                self._cache.pop(old_key, None)

            # Insert new entry
            self._cache[key] = (policy, value)
            self._ring[self._ring_idx] = key

            # Advance ring pointer
            self._ring_idx = (self._ring_idx + 1) % self.max_size

    def clear(self):
        """Clear all cached entries and reset stats."""
        with self._lock:
            self._cache.clear()
            self._ring = [None] * self.max_size
            self._ring_idx = 0
            self.hits = 0
            self.misses = 0

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        with self._lock:
            return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class EvalRequest:
    """Request from worker to evaluator."""
    node: 'MCTSNode'
    tokens: torch.Tensor  # Pre-tokenized by worker (offloads CPU work from evaluator)
    event: threading.Event = field(default_factory=threading.Event)
    policy: Optional[torch.Tensor] = None
    value: Optional[float] = None


class MCTSNode:
    """A node in the MCTS tree with virtual loss support."""

    __slots__ = ['parent', 'move', 'prior', 'base_prior', 'children',
                 'visit_count', 'value_sum', 'vloss_count', 'is_expanded',
                 'lock', 'is_terminal', 'terminal_value']

    def __init__(self, parent: Optional['MCTSNode'] = None,
                 move: Optional[chess.Move] = None, prior: float = 0.0):
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior  # P(s,a) actually used by PUCT (may include noise)
        # The network's own P(s,a), written once at expansion and never mutated
        # afterwards. Dirichlet noise is a function OF this value rather than an
        # edit to `prior`, so re-applying it on a reused tree recomputes instead
        # of compounding -- see ParallelMCTS._add_dirichlet_noise.
        self.base_prior = prior
        self.children: dict[chess.Move, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        # Integer COUNT of in-flight virtual losses through this node (how many
        # workers are currently descending through it), NOT the penalty magnitude.
        # The penalty (vloss_count * VIRTUAL_LOSS) is applied at read time in
        # effective_visits / q_value. Keeping the conserved quantity an integer
        # makes apply/revert exact inverses at ANY VIRTUAL_LOSS magnitude, so a
        # quiescent tree sums to exactly 0 with no floating-point residue.
        self.vloss_count = 0
        self.is_expanded = False
        # Cached terminal state — once a leaf is detected as game-over, store
        # the result so future sims that reach this node skip the (expensive)
        # board.is_game_over() / board.result() recomputation.
        self.is_terminal = False
        self.terminal_value = 0.0
        self.lock = threading.Lock()

    @property
    def effective_visits(self) -> float:
        """Visit count including in-flight virtual losses (vloss_count * VIRTUAL_LOSS)."""
        return self.visit_count + self.vloss_count * VIRTUAL_LOSS

    @property
    def q_value(self) -> float:
        """Mean action value Q(s,a), adjusted for virtual loss."""
        # Penalty is computed from the integer count at read time; same numeric
        # value as the old VL-unit accumulator, but the stored state stays exact.
        penalty = self.vloss_count * VIRTUAL_LOSS
        total_visits = self.visit_count + penalty
        if total_visits == 0:
            return 0.0
        # Each in-flight virtual loss counts as a loss (value contribution -VIRTUAL_LOSS).
        adjusted_value = self.value_sum - penalty
        return adjusted_value / total_visits

    def ucb_score(self, sqrt_parent_visits: float, c_puct: float,
                  fpu: float) -> float:
        """PUCT score for node selection.

        Takes the parent-dependent terms precomputed rather than a SearchParams:
        sqrt(N_parent) and c_puct(N_parent) are identical for every child of one
        parent, so select_child hoists both out of the loop. This is the hottest
        function in the search -- it runs once per child per selection step --
        and hoisting turns one sqrt and one log PER CHILD into one of each per
        step.

        FPU applies only to a node that is both unvisited AND has no worker
        descending through it. The `vloss_count == 0` half is not optional: an
        in-flight node must keep reading the virtual loss (q_value returns -1.0
        for it), or every worker would score it at `fpu` simultaneously and pile
        onto the same unexplored move, which is exactly what virtual loss exists
        to prevent.
        """
        if self.visit_count == 0 and self.vloss_count == 0:
            q = fpu
        else:
            q = self.q_value
        exploration = c_puct * self.prior * sqrt_parent_visits / (1 + self.effective_visits)
        return q + exploration

    def apply_virtual_loss(self):
        """Register one in-flight virtual loss (a worker is descending through this node)."""
        with self.lock:
            self.vloss_count += 1

    def revert_virtual_loss(self):
        """Remove one in-flight virtual loss after evaluation completes."""
        with self.lock:
            self.vloss_count -= 1

    def expand(self, policy: torch.Tensor, legal_moves: list[chess.Move],
               policy_temperature: float = 1.0):
        """Expand node with children based on policy network output.

        `policy_temperature` is the inference-time sharpening exponent T: priors
        become P^(1/T), renormalised over the legal moves. T < 1 sharpens
        (concentrates prior mass on the network's top moves), T > 1 flattens,
        T = 1 is a no-op and the default.

        It is applied to the LOGITS as `softmax(logits / T)`, which is not an
        approximation of P^(1/T) but the same function: with P = softmax(l),
        P_i^(1/T) = exp(l_i/T) / Z^(1/T), and the constant Z^(1/T) divides out
        in the renormalisation. Doing it on logits avoids raising already-tiny
        probabilities to a large power, where a bf16-derived prior would
        underflow to exactly 0.
        """
        with self.lock:
            if self.is_expanded:
                return

            # Extract logits ONLY for legal moves. The index must match the one
            # the policy targets were built with — data/multiPV/labels.py:
            # move_index is this same from_square*64 + to_square — or every
            # prior lands on the wrong move. require_v5_config pins the width.
            #
            # Promotions collide: a 64x64 head cannot distinguish e7e8q from
            # e7e8n, and the labels deliberately accumulate their mass onto the
            # shared index (move_index's docstring). So the four promotion moves
            # here are separate children that all start from the same prior, and
            # search alone separates them — which is the behaviour training
            # assumed, not a loss of information at play time.
            legal_indices = [m.from_square * 64 + m.to_square for m in legal_moves]

            if not legal_indices:
                self.is_expanded = True
                return

            # Slice out logits for legal moves only, then softmax.
            # Policy is already on CPU (evaluator does bulk D2H before distributing).
            # .float() handles the GPU bf16 case; on CPU it's a no-op cast.
            legal_logits = policy[legal_indices]
            if legal_logits.dtype != torch.float32:
                legal_logits = legal_logits.float()
            # Sharpening, as P^(1/T) over the legal moves. Guarded so the
            # default path is bit-identical to the pre-sharpening behaviour
            # rather than merely equal-to-rounding after a divide by 1.0.
            if policy_temperature != 1.0:
                legal_logits = legal_logits / policy_temperature
            _t0 = _t1 = 0.0
            if _INSTR:
                _t0 = time.perf_counter()
            legal_probs = torch.softmax(legal_logits, dim=0).tolist()
            if _INSTR:
                _t1 = time.perf_counter()

            # Map the properly scaled probabilities to the children
            for move, prior in zip(legal_moves, legal_probs):
                self.children[move] = MCTSNode(parent=self, move=move, prior=prior)

            if _INSTR:
                _b = _instr_bucket()
                _t2 = time.perf_counter()
                _b['expand_calls'] += 1
                _b['expand_softmax_s'] += _t1 - _t0
                _b['expand_children_s'] += _t2 - _t1
                _b['expand_children_n'] += len(legal_probs)

            self.is_expanded = True

    def select_child(self, params: SearchParams,
                     at_root: bool = False) -> 'MCTSNode | None':
        """Select child with highest UCB score.

        c_puct, sqrt(parent_visits) and the applicable FPU depend only on this
        node, so all three are resolved once here and passed down to every child
        (see ucb_score).

        `at_root` selects between params.fpu_root and params.fpu_tree. The
        caller supplies it from the simulation's current depth rather than the
        node knowing its own role -- see SearchParams for why root-ness cannot
        be cached on a node.
        """
        parent_visits = self.effective_visits
        c_puct = params.c_puct(parent_visits)
        sqrt_parent_visits = math.sqrt(parent_visits)
        fpu = params.fpu(at_root)
        best_score = float('-inf')
        best_child = None

        if _INSTR:
            # Item 16: how often does insertion-order tie-breaking actually
            # decide the selection? Collect every child's score, then compare
            # the top two. A separate pass so the measured loop below stays the
            # code the port has to reproduce.
            _scored = [(c.ucb_score(sqrt_parent_visits, c_puct, fpu), c)
                       for c in self.children.values()]
            _b = _instr_bucket()
            _b['select_steps'] += 1
            if len(_scored) >= 2:
                _ranked = sorted(_scored, key=lambda t: t[0], reverse=True)
                _s0, _c0 = _ranked[0]
                _s1, _c1 = _ranked[1]
                if _s0 - _s1 <= 1e-12:
                    _b['select_ties'] += 1
                    _m0, _m1 = _c0.move, _c1.move
                    if (_m0 is not None and _m1 is not None
                            and _m0.promotion is not None
                            and _m1.promotion is not None
                            and _m0.from_square == _m1.from_square
                            and _m0.to_square == _m1.to_square):
                        _b['select_ties_promo'] += 1
                    # How many children sit inside the tie band at all.
                    _b['select_tie_width'] += sum(
                        1 for s, _ in _ranked if _s0 - s <= 1e-12)

        for child in self.children.values():
            score = child.ucb_score(sqrt_parent_visits, c_puct, fpu)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def backpropagate(self, value: float):
        """Propagate evaluation result up the tree."""
        node = self
        while node is not None:
            with node.lock:
                node.visit_count += 1
                node.value_sum += value
            value = -value  # Flip perspective for opponent
            node = node.parent


class BatchedEvaluator:
    """Evaluator thread that batches NN requests for efficiency."""

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 max_batch_size: int = 128, min_batch_size: int = 8,
                 batch_timeout_ms: float = 50.0,
                 inline: bool = False):
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size  # Wait for at least this many
        self.batch_timeout = batch_timeout_ms / 1000.0  # Convert to seconds
        # When inline=True, workers call eval_inline() directly instead of going through
        # the queue. Used on CPU where batching is counterproductive and thread handoffs
        # dominate the per-sim cost.
        self.inline = inline

        self.request_queue: Queue[EvalRequest] = Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        # Stats lock — incremented by all workers in inline mode
        self._stats_lock = threading.Lock()

        # Stats
        self.total_evals = 0
        self.total_batches = 0
        # Per-batch size log for game-level diagnostics (avg / histogram /
        # saturation). Distinct from total_evals/total_batches, which callers
        # reset every move; this persists until reset_batch_history() so an
        # end-of-game summary can see the whole game. Guarded by _stats_lock
        # (appended from the evaluator thread, read from the main thread).
        self.batch_size_history: list[int] = []

    def reset_batch_history(self) -> None:
        """Drop the recorded per-batch sizes. Call at the start of a game so the
        end-of-game summary only covers that game."""
        with self._stats_lock:
            self.batch_size_history.clear()

    def format_batch_summary(self) -> str:
        """Build a multi-line, human-readable summary of the batch sizes recorded
        since the last reset_batch_history(): average size, an ASCII histogram
        (printed, never persisted), and the proportion of batches that saturated
        max_batch_size or bottomed out at min_batch_size (both configured above).

        Returns a single placeholder line when no batches were recorded (e.g.
        raw-policy play, CPU inline mode with batch size 1, or a game that ended
        before any search ran)."""
        with self._stats_lock:
            sizes = list(self.batch_size_history)

        if not sizes:
            return "Batch statistics: no batches recorded this game."

        arr = np.array(sizes)
        n = len(arr)
        total_evals = int(arr.sum())
        avg = arr.mean()
        at_max = int((arr >= self.max_batch_size).sum())
        at_min = int((arr <= self.min_batch_size).sum())

        lines = [
            "=" * 60,
            "Batch size summary (this game)",
            "=" * 60,
            f"  batches: {n}   evals: {total_evals}",
            f"  avg batch size: {avg:.1f}   (observed min={int(arr.min())}, "
            f"max={int(arr.max())})",
            f"  configured range: {self.min_batch_size}-{self.max_batch_size}",
            f"  at max batch ({self.max_batch_size}): {at_max} ({at_max / n:.1%})",
            f"  at min batch ({self.min_batch_size}): {at_min} ({at_min / n:.1%})",
            "",
            "  Histogram:",
        ]

        lo, hi = int(arr.min()), int(arr.max())
        if lo == hi:
            lines.append(f"    all {n} batches were size {lo}")
        else:
            nbins = min(20, hi - lo + 1)
            counts, edges = np.histogram(arr, bins=nbins, range=(lo, hi + 1))
            peak = int(counts.max()) or 1
            bar_w = 40
            for i, c in enumerate(counts):
                bar = "#" * int(round(bar_w * c / peak))
                lines.append(
                    f"    [{edges[i]:5.0f}-{edges[i + 1]:5.0f}) "
                    f"{int(c):6d} |{bar}"
                )
        lines.append("=" * 60)
        return "\n".join(lines)

    def eval_inline(self, tokens: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Synchronous single-position evaluation. Used in inline mode (CPU).

        Each worker calls this on its own thread; PyTorch releases the GIL during
        the forward pass, so N workers run N parallel forward passes on N cores.
        """
        with torch.no_grad():
            policy_logits, values = self.model(tokens.unsqueeze(0))
        with self._stats_lock:
            self.total_evals += 1
            self.total_batches += 1
            self.batch_size_history.append(1)
        # .clone() so the cached row doesn't pin the full forward-pass output buffer.
        return policy_logits[0].clone(), values[0].item()

    def start(self):
        """Start the evaluator thread."""
        if self.running:
            return  # Already running
        self.running = True
        self.thread = threading.Thread(target=self._eval_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the evaluator thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

    def submit(self, request: EvalRequest):
        """Submit an evaluation request."""
        self.request_queue.put(request)

    def _eval_loop(self):
        """Main evaluation loop - collect batches and evaluate."""
        while self.running:
            batch = self._collect_batch()
            if batch:
                self._evaluate_batch(batch)

    def _collect_batch(self) -> list[EvalRequest]:
        """Collect requests aggressively to maximize batch size for GPU efficiency."""
        batch = []
        _t_enter = _t_first = _t_drained = 0.0
        _hit_deadline = False
        if _INSTR:
            _t_enter = time.perf_counter()

        # First, block until we get at least one request (no timeout polling)
        while len(batch) == 0 and self.running:
            try:
                # Use blocking get() - avoids CPU spin from timeout polling
                request = self.request_queue.get(timeout=1.0)  # Long timeout for clean shutdown
                batch.append(request)
            except Empty:
                continue

        if _INSTR:
            _t_first = time.perf_counter()

        # Immediately drain everything currently in the queue (non-blocking)
        while len(batch) < self.max_batch_size:
            try:
                request = self.request_queue.get_nowait()
                batch.append(request)
            except Empty:
                break

        if _INSTR:
            _t_drained = time.perf_counter()
            _n_after_drain = len(batch)

        # If we don't have enough yet, wait a bit more for stragglers
        if len(batch) < self.min_batch_size:
            deadline = time.time() + self.batch_timeout
            while len(batch) < self.min_batch_size and self.running:
                remaining = deadline - time.time()
                if remaining <= 0:
                    _hit_deadline = True
                    break
                try:
                    request = self.request_queue.get(timeout=min(remaining, 0.01))
                    batch.append(request)
                except Empty:
                    continue

        if _INSTR:
            _t_end = time.perf_counter()
            _instr_batches.append({
                'wait_first_s': _t_first - _t_enter,
                'drain_s': _t_drained - _t_first,
                'straggler_s': _t_end - _t_drained,
                'hit_deadline': _hit_deadline,
                'n_after_drain': _n_after_drain,
                'batch_size': len(batch),
                'forward_s': None,   # filled in by _evaluate_batch
                'total_eval_s': None,
            })

        return batch

    def _evaluate_batch(self, batch: list[EvalRequest]):
        """Run batch through neural network and distribute results."""
        if not batch:
            return

        _e0 = _e1 = _e2 = _e3 = 0.0
        if _INSTR:
            _e0 = time.perf_counter()

        # Stack pre-tokenized tensors from workers (no CPU work here)
        tokens_batch = torch.stack([req.tokens for req in batch]).to(self.device)

        if _INSTR:
            _e1 = time.perf_counter()

        # Forward pass, in the precision v5 was trained under (AUTOCAST_DTYPE).
        # No legal_move_mask: masking happens in MCTSNode.expand, which slices
        # the legal logits out and softmaxes only those.
        with torch.no_grad():
            with torch.amp.autocast_mode.autocast(device_type='cuda', dtype=AUTOCAST_DTYPE,
                                                  enabled=self.device.type == 'cuda'):
                policy_logits, values = self.model(tokens_batch)

        if _INSTR:
            _e2 = time.perf_counter()

        # Bulk D2H transfer - move entire tensors to CPU before iterating
        # This avoids per-item synchronization overhead
        policy_cpu = policy_logits.cpu()
        values_cpu = values.cpu()

        if _INSTR:
            _e3 = time.perf_counter()

        # Distribute results back to workers (now from CPU tensors).
        # .clone() each row so cache entries don't pin the whole batch buffer
        # (policy_cpu[i] is a view sharing storage with the full batch tensor).
        for i, req in enumerate(batch):
            req.policy = policy_cpu[i].clone()
            req.value = values_cpu[i].item()
            req.event.set()  # Signal worker that result is ready

        if _INSTR:
            _e4 = time.perf_counter()
            if _instr_batches:
                _rec = _instr_batches[-1]
                _rec['h2d_s'] = _e1 - _e0
                _rec['fwd_launch_s'] = _e2 - _e1
                # The D2H is where the GPU work is actually awaited, so
                # fwd_launch + d2h is the real end-to-end forward cost.
                _rec['d2h_s'] = _e3 - _e2
                _rec['distribute_s'] = _e4 - _e3
                _rec['forward_s'] = _e3 - _e0
                _rec['total_eval_s'] = _e4 - _e0

        # Stats
        with self._stats_lock:
            self.total_batches += 1
            self.total_evals += len(batch)
            self.batch_size_history.append(len(batch))


class MCTSWorker:
    """Worker thread that traverses the tree and submits leaves for evaluation."""

    def __init__(self, worker_id: int, root: MCTSNode, root_board: chess.Board,
                 evaluator: BatchedEvaluator, cache: TranspositionCache,
                 stats: dict, target_sims: int, completion_event: threading.Event,
                 tablebase: Optional[chess.syzygy.Tablebase] = None,
                 repetition_history: Optional[dict] = None,
                 params: Optional[SearchParams] = None):
        # Shared with ParallelMCTS and every sibling worker (not copied), so a
        # setoption between moves is picked up without rebuilding the workers.
        self.params = params if params is not None else SearchParams()
        self.worker_id = worker_id
        self.root = root
        self.root_board = root_board
        self.evaluator = evaluator
        self.cache = cache
        self.stats = stats
        self.target_sims = target_sims
        self.completion_event = completion_event
        # Transposition-key -> occurrence count for the game history leading to
        # root_board (built by build_repetition_history). Lets simulations, which
        # run on stack-stripped board copies, detect threefold repetition by
        # adding their within-line repeats on top of it. Read-only; shared across
        # workers. None => no history (repetition still detected within a line).
        self.repetition_history = repetition_history if repetition_history is not None else {}
        # Syzygy tablebase for Mode 2 leaf evaluation. None => disabled.
        # Thread-safe to share read-only across workers because each worker
        # probes its own board copy (chess.syzygy guarantees this).
        self.tablebase = tablebase
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start the worker thread."""
        self.running = True
        self.thread = threading.Thread(target=self._work_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the worker thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _work_loop(self):
        """Main work loop - select, expand, evaluate, backprop."""
        try:
            while self.running:
                # External termination signal (e.g. ponder_stop). Must come
                # before the target_sims check so pondering — which targets an
                # effectively infinite sim count — can be interrupted.
                if self.completion_event.is_set():
                    break
                if self.stats['simulations'] >= self.target_sims:
                    self.completion_event.set()
                    break
                self._run_simulation()
        except Exception as e:
            # Log to stderr so it's visible in Cutechess debug output.
            # Set completion_event so the main thread doesn't hang forever.
            import sys, traceback
            print(f"[mcts worker {self.worker_id}] {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            self.completion_event.set()

    def _draw_by_rule(self, board: chess.Board, path_counts: dict) -> bool:
        """True if `board` is a draw by the fifty-move rule or threefold
        repetition, given the game history plus repeats within this simulation.

        `path_counts` maps transposition key -> times the key has appeared on the
        CURRENT simulation's path from the root; it is mutated here (the key for
        `board` is counted in) and is local to one _run_simulation call, so two
        sims exploring different lines never pollute each other's counts, and a
        tree transposition (two different lines reaching one position) is not
        mistaken for a single line repeating.

        The repetition value is intentionally never written to the Zobrist NN
        cache: whether a position is a draw is path-dependent (it depends on the
        prior game line), whereas the NN cache is keyed purely by position. We
        store the draw only on the tree node (path-specific) at the call site.
        """
        # Fifty-move rule: 100 half-moves without a pawn move or capture. The
        # halfmove clock is board state, preserved through stack=False copies.
        if board.halfmove_clock >= 100:
            return True
        # Threefold repetition: history occurrences + occurrences on this line.
        key = board._transposition_key()
        seen = path_counts.get(key, 0) + 1
        path_counts[key] = seen
        return self.repetition_history.get(key, 0) + seen >= 3

    def _run_simulation(self):
        """Run one MCTS simulation.

        Virtual-loss bookkeeping is centralized: every node this simulation
        descends through is registered in `applied` the instant its
        apply_virtual_loss() runs, and a single `repay()` reverts exactly that
        set. repay() is called explicitly right before each backprop (preserving
        the original revert-before-backprop ordering, so backprop always runs
        with zero in-flight loss on the path) and again in the `finally` as a
        safety net, so EVERY exit -- normal leaf, in-search terminal (repetition
        / fifty-move / checkmate / stalemate), max-depth cutoff, transposition
        cache hit, tablebase leaf override, AND exceptions -- repays precisely
        what it applied. Because each node's in-flight count is an integer,
        apply and revert are exact inverses regardless of the VIRTUAL_LOSS
        magnitude, so a quiescent tree conserves to exactly 0.
        """
        node = self.root
        # stack=False skips cloning move history (much faster, we don't need undo)
        board = self.root_board.copy(stack=False)
        depth = 0
        # Per-simulation tally of transposition keys along this path (excludes the
        # pre-root game history, which lives in self.repetition_history). Rebuilt
        # fresh every simulation so lines never share repeat counts.
        path_counts: dict = {}

        # Ledger of nodes this simulation has applied a virtual loss to and still
        # owes a revert. repay() drains it exactly once per node; the finally
        # below guarantees it runs even on an early return or an exception.
        applied: list[MCTSNode] = []

        def repay() -> None:
            while applied:
                applied.pop().revert_virtual_loss()

        try:
            # Register the root, then each node as we arrive at it during descent.
            node.apply_virtual_loss()
            applied.append(node)

            # === Selection: traverse to leaf ===
            while node.is_expanded and node.children:
                # depth == 0 exactly on the first iteration, i.e. while choosing
                # among the search root's children -- the one selection step
                # fpu_root governs. `self.root` is this worker's search root,
                # which under pondering is the branch node, as intended.
                node = node.select_child(self.params, at_root=(depth == 0))
                assert node is not None  # loop condition guarantees children exist
                board.push(node.move)
                depth += 1
                node.apply_virtual_loss()
                applied.append(node)

                # === Draw-by-rule detection (threefold repetition / fifty-move) ===
                # Checked on every step of the descent, not just the final leaf: with
                # tree reuse an already-expanded interior node can have *become* a
                # draw now that the game history is longer. Detecting it here stops
                # the line and backs a draw value (0.0) up to the move that enters
                # the repetition, so a winning engine sees entering it as worse than
                # its positive alternatives (and a losing engine sees it as better) --
                # no explicit "am I winning" logic needed; the value comparison does
                # it in both directions. The value is stored on the tree node, which
                # is path-specific, NOT in the position-keyed NN cache.
                if self._draw_by_rule(board, path_counts):
                    repay()
                    # is_expanded is deliberately LEFT ALONE (False) here.
                    #
                    # A draw by fifty-move / threefold is only a draw if someone
                    # CLAIMS it: python-chess's is_game_over() ends the game on
                    # the seventy-five-move and fivefold rules alone, and hosts
                    # differ -- cutechess claims both automatically, Lichess does
                    # not. So a node we mark here can still be handed to us as
                    # the position we must move from. Marking it expanded with an
                    # empty children dict made that unrecoverable: apply_move
                    # promotes it to root, search() sees is_expanded and skips
                    # _expand_root, then finds no children and returns None --
                    # 'bestmove 0000', a forfeit, with legal moves on the board.
                    #
                    # Leaving it unexpanded costs the search nothing: the
                    # selection loop above stops here either way (its condition
                    # needs `is_expanded AND children`), and the cached-terminal
                    # fast path below returns terminal_value before any expansion
                    # is attempted. The only code that can now expand this node is
                    # _expand_root, which runs exactly when it has become a search
                    # root -- i.e. when we are actually being forced to play on.
                    with node.lock:
                        node.is_terminal = True
                        node.terminal_value = 0.0
                    node.backpropagate(0.0)
                    self.stats['simulations'] += 1
                    return

                # Max depth cutoff to prevent endgame slowdowns
                if depth >= MAX_TREE_DEPTH:
                    # Treat as terminal with value 0 (draw-ish).
                    repay()
                    node.backpropagate(0.0)
                    self.stats['simulations'] += 1
                    return

            # === Cached terminal (fast path) ===
            # If this leaf was previously identified as terminal, just backprop the
            # cached mover-perspective value
            if node.is_terminal:
                repay()
                node.backpropagate(node.terminal_value)
                # Mate-in-one short-circuit: a root-child (depth==1) that is
                # terminal with value +1 from the mover's perspective is a
                # winning move for us. Signal search() to take it immediately.
                if depth == 1 and node.terminal_value == 1.0 and node.move is not None:
                    self.stats['mating_move'] = node.move
                    self.completion_event.set()
                self.stats['simulations'] += 1
                return

            # === Check terminal state (slow path, first visit only) ===
            if board.is_game_over():
                result = board.result()
                # Value must be from MOVER's perspective (who moved TO this node, i.e., opponent of board.turn)
                # This allows selection to use max(Q) directly without negation.
                if result == "1-0":
                    # White wins. If it's Black's turn (Black mated), mover was White, value = +1 for mover.
                    value = 1.0 if board.turn == chess.BLACK else -1.0
                elif result == "0-1":
                    # Black wins. If it's White's turn (White mated), mover was Black, value = +1 for mover.
                    value = 1.0 if board.turn == chess.WHITE else -1.0
                else:
                    value = 0.0

                # Cache result so subsequent visits hit the fast path above.
                # Setting is_expanded=True is harmless WITHIN the search:
                # selection bails on the empty children dict before reaching this
                # node again. Unlike the draw-by-rule path above, this branch is
                # gated on board.is_game_over(), so the position is over by a rule
                # that needs no claim -- checkmate, stalemate, insufficient
                # material, seventy-five-move, fivefold. A host can still decline
                # to end on the last three, so search()/get_policy() re-expand a
                # promoted root that arrives with no children rather than trusting
                # this flag; that recovery is what makes leaving it here safe.
                with node.lock:
                    node.is_terminal = True
                    node.terminal_value = value
                    node.is_expanded = True

                repay()
                node.backpropagate(value)
                if depth == 1 and value == 1.0 and node.move is not None:
                    self.stats['mating_move'] = node.move
                    self.completion_event.set()
                self.stats['simulations'] += 1
                return

            # === Expansion & Evaluation ===
            # Check transposition cache first. The key is (Zobrist, ep_square),
            # NOT the Zobrist hash alone -- see make_cache_key for why the hash
            # by itself is coarser than the network's own tokenization.
            cache_key = make_cache_key(board)
            cached = self.cache.get(cache_key)

            if _INSTR and cached is not None:
                # Items 12/13: the key matched, but did the POSITION match on
                # the axes the key ignores (ep token, halfmove clock)?
                with _instr_keymeta_lock:
                    _meta = _instr_keymeta.get(cache_key)
                if _meta is not None:
                    _b = _instr_bucket()
                    _b['cache_hit_checked'] += 1
                    _stored_ep, _stored_hmc, _stored_tb = _meta
                    if _stored_ep != _instr_ep_token(board):
                        _b['cache_hit_ep_mismatch'] += 1
                    if _stored_hmc != board.halfmove_clock:
                        _b['cache_hit_hmc_mismatch'] += 1
                        if _stored_tb:
                            _b['cache_hit_tb_hmc_mismatch'] += 1
                            # Only a 50-move-relevant difference matters: one
                            # side near the 100-ply limit, the other not.
                            if (_stored_hmc < 100) != (board.halfmove_clock < 100):
                                _b['cache_hit_tb_hmc_crossing'] += 1

            if cached is not None:
                # Cache hit - use cached policy and value
                policy, nn_value = cached
            else:
                tokens = board_to_tokens(board)

                if self.evaluator.inline:
                    # CPU path: run forward pass on this worker's thread (no queueing).
                    # PyTorch releases the GIL during the forward, so workers run in parallel.
                    policy, nn_value = self.evaluator.eval_inline(tokens)
                else:
                    # GPU path: submit to batched evaluator
                    request = EvalRequest(node=node, tokens=tokens)
                    self.evaluator.submit(request)
                    request.event.wait()  # Block until evaluated
                    policy = request.policy
                    nn_value = request.value if request.value is not None else 0.0

                # === Mode 2: tablebase value override ===
                # If this leaf is within tablebase range, replace the neural value
                # with the exact WDL result. Policy logits are left untouched (the
                # network still decides move ordering). The piece-count check is a
                # cheap popcount that filters out the overwhelming majority of
                # middlegame leaves before any probe cost. On a tablebase miss
                # (position not covered) we keep the neural value. We override
                # BEFORE caching so the WDL value is what gets stored — subsequent
                # transpositions to this position reuse it without re-probing, and
                # it flows through the same absolute->mover perspective conversion
                # below as the neural value. See probe_tablebase_value().
                _was_tb = False
                if self.tablebase is not None and count_pieces(board) <= TABLEBASE_MAX_PIECES:
                    tb_value = probe_tablebase_value(self.tablebase, board)
                    if tb_value is not None:
                        nn_value = tb_value
                        _was_tb = True

                # Store in cache (policy logits, not softmax'd)
                if policy is not None:
                    if _INSTR:
                        with _instr_keymeta_lock:
                            _instr_keymeta[cache_key] = (
                                _instr_ep_token(board), board.halfmove_clock, _was_tb)
                    self.cache.put(cache_key, policy, nn_value)

            # Expand node with policy LOGITS (Softmax, and any sharpening,
            # happen inside expand). The cache stores raw logits, so a cache hit
            # is sharpened at the same temperature as a fresh evaluation.
            legal_moves = list(board.legal_moves)
            node.expand(policy, legal_moves, self.params.policy_temperature)
            # Gate 1 only: permute the children just built into canonical order.
            # AFTER expand(), never by sorting `legal_moves` above -- see
            # _canonicalize_children for why that distinction is the whole point.
            _canonicalize_children(node)

            repay()

            # === Absolute to Mover's Perspective Conversion ===
            # The v5 value head is White-POV by construction (model_v5: tanh off
            # the CLS token, "White-POV"), so its output is Absolute: White
            # winning = +1.0, Black winning = -1.0, independent of side to move.
            # We need Mover's perspective (who moved TO this node = opponent of board.turn).
            # - If board.turn == BLACK, mover was WHITE, use NN value as-is.
            # - If board.turn == WHITE, mover was BLACK, negate NN value.
            mover_value = nn_value if board.turn == chess.BLACK else -nn_value

            # Backpropagate from mover's perspective
            node.backpropagate(mover_value)

            self.stats['simulations'] += 1
        finally:
            # Safety net: on a normal exit `applied` is already drained by repay()
            # above (no-op here); on an early return path that skipped it, or any
            # exception mid-simulation, this guarantees every applied virtual loss
            # is reverted so none is stranded in the tree.
            repay()


class ParallelMCTS:
    """Parallel MCTS with batched neural network evaluation and tree reuse."""

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 num_workers: Optional[int] = None, max_batch_size: Optional[int] = None,
                 min_batch_size: Optional[int] = None,
                 cache_size: int = 100_000,
                 tablebase: Optional[chess.syzygy.Tablebase] = None,
                 params: Optional[SearchParams] = None):
        self.model = model
        self.device = device
        # Exploration / FPU / policy-temperature settings. Held as one object and
        # shared (not copied) with every worker, so a caller holding the same
        # instance can retune between moves -- the UCI layer's `setoption` path.
        # Defaults reproduce the pre-tunable behaviour: static-equivalent c_puct
        # at low visit counts, FPU 0, no sharpening.
        self.params: SearchParams = params if params is not None else SearchParams()
        # Syzygy tablebase for Mode 2 leaf evaluation. Owned by the caller
        # (the UCI wrapper opens/closes it); we only read from it. Passed to
        # every worker at search time. None => Mode 2 disabled.
        self.tablebase = tablebase

        # V5 architecture only — validated up front, before any worker thread
        # exists, so a wrong model fails at construction with a readable error
        # instead of producing plausible-looking garbage evaluations. See the
        # module docstring for why this file refuses what mctsv3 accepts.
        self.config = require_v5_config(model)
        self.seq_length: int = SEQ_LENGTH

        # tanh scale the value head was fitted against, when the checkpoint
        # carried one (playv5.load_model attaches it). Search never needs it —
        # Q stays in the network's own [-1, 1] units throughout — but callers
        # converting Q to centipawns can read it off the engine rather than
        # reaching back to the model. None for a checkpoint without one.
        self.value_scale: Optional[float] = getattr(model, "value_scale", None)

        # Auto-tune workers based on hardware.
        # GPU: many workers feed the batched evaluator despite the GIL (forward pass
        #      releases it, so workers can build up requests in parallel).
        # CPU: one worker per core, each runs its own forward pass inline. Pin
        #      torch.set_num_threads(1) so N workers don't oversubscribe — true
        #      parallelism comes from N workers, not intra-op threading.
        if num_workers is None:
            num_workers = 32 if device.type == 'cuda' else (os.cpu_count() or 4)
        self.num_workers = num_workers

        if device.type == 'cpu':
            # Each worker's forward pass uses one thread; parallelism comes from
            # running num_workers forward passes concurrently. Without this, every
            # forward pass tries to fan out across all cores and they all collide.
            torch.set_num_threads(1)

        # Auto-tune batch size based on hardware.
        # On CPU, batching provides little benefit (forward pass scales linearly),
        # so keep batches small to avoid waiting.
        if max_batch_size is None:
            if device.type == 'cuda':
                max_batch_size = 1024  # GPU can handle large batches
            else:
                max_batch_size = 1  # CPU runs inline (no batching)

        # Min batch size - on CPU, don't wait for batches to fill.
        # On GPU, larger min batches improve utilization.
        # An explicit min_batch_size (e.g. playv5's --turbo) overrides the
        # hardware default; None falls back to auto-tuning below.
        if device.type == 'cuda':
            auto_min_batch_size = min(32, num_workers)
            batch_timeout_ms = 10.0
        else:
            auto_min_batch_size = 1  # Process immediately on CPU
            batch_timeout_ms = 5.0  # Short timeout - don't block workers
        if min_batch_size is None:
            min_batch_size = auto_min_batch_size

        self.evaluator = BatchedEvaluator(
            model=model,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            batch_timeout_ms=batch_timeout_ms,
            inline=(device.type == 'cpu'),
        )

        # Transposition cache for NN evaluations (persists across searches)
        self.cache = TranspositionCache(max_size=cache_size)

        # Persistent tree for reuse across searches
        self.root: Optional[MCTSNode] = None
        self.root_board: Optional[chess.Board] = None
        self._root_hash: Optional[int] = None  # Zobrist hash for position comparison

        self.model.eval()

        # Store last search stats for external access
        self.last_root_q = 0.0  # Q-value of root after search (from side-to-move perspective)
        self.last_best_child_q = 0.0  # Q-value of best move

        # Pondering state. A background thread runs MCTS simulations on a
        # predicted-opponent-reply subtree between searches. All tree-mutating
        # methods below call ponder_stop() first, so callers don't need to.
        self._pondering: bool = False
        self._ponder_thread: Optional[threading.Thread] = None
        self._ponder_stop_event: Optional[threading.Event] = None
        # Instrumentation: hold the predicted branches and a stats dict
        # across the ponder lifecycle so apply_move() can log hit/miss +
        # total sim count. A list because confidence-gated multi-PV may
        # ponder several candidate replies. Cleared on the next
        # ponder_start() (once the previous ponder has been consumed) or on
        # reset().
        self._ponder_predicted_moves: list[chess.Move] = []
        self._ponder_stats: Optional[defaultdict] = None

    def shutdown(self):
        """Stop the evaluator thread. Call when done with MCTS."""
        self.ponder_stop()
        self.evaluator.stop()

    def clear_cache(self):
        """Clear the transposition cache."""
        self.ponder_stop()
        self.cache.clear()

    def reset(self):
        """Clear the persistent tree. Call at start of a new game."""
        self.ponder_stop()
        self.root = None
        self.root_board = None
        self._root_hash = None
        self._ponder_predicted_moves = []
        self._ponder_stats = None

    def apply_move(self, move: chess.Move):
        """
        Advance the tree by the given move, preserving the relevant subtree.
        Call this after a move is played to enable tree reuse.
        """
        self.ponder_stop()

        # Log outcome of the just-completed ponder (if any): did the opponent
        # play any of the moves we predicted, and how many sims did ponder add?
        if self._ponder_predicted_moves:
            sims = self._ponder_stats['simulations'] if self._ponder_stats is not None else 0
            hit = move in self._ponder_predicted_moves
            def _san(m: chess.Move, b: Optional[chess.Board]) -> str:
                try:
                    return b.san(m) if b is not None else m.uci()
                except Exception:
                    return m.uci()
            preds = ",".join(_san(m, self.root_board) for m in self._ponder_predicted_moves)
            actual_str = _san(move, self.root_board)
            # Diagnostic: show the current visit_count on each pondered
            # sub-root (the node that will be promoted if the user played
            # that branch). If these are near-zero on a hit, ponder work
            # didn't transfer and the next search will be slow.
            branch_visits_parts: list[str] = []
            if self.root is not None:
                for m in self._ponder_predicted_moves:
                    child = self.root.children.get(m)
                    v = child.visit_count if child is not None else -1
                    branch_visits_parts.append(f"{_san(m, self.root_board)}:{v}v")
            # Also show the visit_count of the actually-played move's
            # child — this is what the next search will see as existing_visits.
            actual_child = self.root.children.get(move) if self.root is not None else None
            actual_visits = actual_child.visit_count if actual_child is not None else -1
            print(f"[ponder] end: predicted=[{preds}] actual={actual_str} "
                  f"hit={hit} sims={sims} "
                  f"branch_visits=[{', '.join(branch_visits_parts)}] "
                  f"actual_visits={actual_visits}",
                  file=sys.stderr, flush=True)
            self._ponder_predicted_moves = []
            self._ponder_stats = None

        if self.root is None or self.root_board is None or move not in self.root.children:
            # Nothing to reuse
            self.root = None
            self.root_board = None
            self._root_hash = None
            return

        # Advance to child node
        new_root = self.root.children[move]
        new_root.parent = None  # Detach so siblings can be garbage collected
        self.root = new_root
        self.root_board.push(move)
        self._root_hash = chess.polyglot.zobrist_hash(self.root_board)

    def _reset_virtual_loss(self, node: MCTSNode):
        """Recursively reset the in-flight virtual-loss count to 0 in the subtree (defensive)."""
        if _INSTR:
            _instr_bucket()['vloss_reset_nodes'] += 1
        node.vloss_count = 0
        for child in node.children.values():
            self._reset_virtual_loss(child)

    def _sum_virtual_loss(self, node: MCTSNode) -> int:
        """Recursively sum the in-flight virtual-loss counts across the subtree.

        Integer counts (not VL-scaled magnitudes), so a quiescent tree returns
        exactly 0 at any VIRTUAL_LOSS value -- no floating-point residue."""
        total = node.vloss_count
        for child in node.children.values():
            total += self._sum_virtual_loss(child)
        return total

    def _add_dirichlet_noise(self, root: MCTSNode, alpha: float = 0.3, epsilon: float = 0.25):
        """
        Add Dirichlet noise to root's children priors for exploration.
        Standard AlphaZero formula: P'(a) = (1 - epsilon) * P(a) + epsilon * Dir(alpha)

        P(a) is read from `base_prior` -- the untouched network prior -- and the
        result is written to `prior`. Reading and writing the same field would be
        wrong here, because this tree PERSISTS across moves: a node can be the
        root of one search, get promoted through apply_move, and be noised again.
        In-place mixing makes those applications compose, so after k of them the
        network's contribution has decayed to (1 - epsilon)^k = 0.75^k -- 24% of
        its intended weight by the fifth. Deriving from base_prior each time
        makes the operation idempotent: one application's worth of noise, always,
        no matter how many searches the node has already served.
        """
        if not root.children:
            return

        moves = list(root.children.keys())
        noise = np.random.dirichlet([alpha] * len(moves))

        for move, n in zip(moves, noise):
            child = root.children[move]
            child.prior = (1 - epsilon) * child.base_prior + epsilon * n

    def search(self, board: chess.Board, num_simulations: int = 800,
               time_limit: float = 0.0, add_dirichlet_noise: bool = False) -> Optional[chess.Move]:
        """
        Run MCTS search and return the best move.

        Args:
            board: Current board position
            num_simulations: Target total simulations (including prior visits if reusing tree)
            time_limit: Optional time limit in seconds (overrides num_simulations)
            add_dirichlet_noise: If True, add Dirichlet noise to root priors (for self-play)

        Returns:
            Best move according to MCTS, or None if no legal moves
        """
        self.ponder_stop()
        board_hash = chess.polyglot.zobrist_hash(board)

        # Check if we can reuse the existing tree
        if self.root is not None and self._root_hash == board_hash:
            # Reuse existing root
            root = self.root
            # Defensive: ensure no stale virtual losses from previous search
            if _INSTR:
                _rv0 = time.perf_counter()
                self._reset_virtual_loss(root)
                _b = _instr_bucket()
                _b['vloss_reset_s'] += time.perf_counter() - _rv0
                _b['vloss_reset_calls'] += 1
            else:
                self._reset_virtual_loss(root)
            # If this node was an unexplored leaf in the prior tree, it has no
            # children yet. Expand now so max(root.children) doesn't return None
            # (which would produce an illegal "0000" bestmove).
            #
            # `or not root.children` covers the other way a promoted node can
            # arrive childless: the board.is_game_over() path in _run_simulation
            # marks a terminal leaf is_expanded with an empty children dict. That
            # is correct for checkmate/stalemate (no legal moves to generate, and
            # _expand_root re-derives the same empty result), but the same flag
            # combination also lands on positions a host may refuse to end --
            # insufficient material, seventy-five-move, fivefold. Re-expanding
            # costs one forward pass on a position we play from at most once.
            if not root.is_expanded or not root.children:
                self._expand_root(root, board)
        else:
            # Create fresh root
            root = MCTSNode()
            self._expand_root(root, board)
            self.root = root
            self.root_board = board.copy()
            self._root_hash = board_hash

        if not root.children:
            # Last-resort net. With the two paths above this should now be
            # unreachable unless the position genuinely has no legal moves, but
            # returning None here means 'bestmove 0000' at the UCI layer, which
            # loses the game on the spot. Only concede when the board agrees.
            legal = list(board.legal_moves)
            if not legal:
                return None
            print(f"[mcts] WARNING: root has no children but {len(legal)} legal "
                  f"moves exist ({board.fen()}); playing {legal[0].uci()}",
                  file=sys.stderr, flush=True)
            self.last_root_q = 0.0
            self.last_best_child_q = 0.0
            return legal[0]

        # Add Dirichlet noise for exploration (per-search, not per-tree)
        if add_dirichlet_noise:
            self._add_dirichlet_noise(root)

        # Calculate how many new simulations to run
        # num_simulations is the target total, so subtract existing visits
        existing_visits = root.visit_count
        target_new_sims = max(0, num_simulations - existing_visits)

        if target_new_sims == 0:
            # Already have enough simulations (e.g. a ponder hit promoted a
            # subtree that already meets num_simulations). Use the SAME sign
            # convention as the normal-search exit below: each node's q_value is
            # from the MOVER's perspective, so root.q_value (opponent moved to
            # reach root) is negated and best_child.q_value (engine moved to
            # reach child) is used as-is. Previously these were inverted here,
            # which flipped the reported UCI score whenever this path was taken.
            best_move, best_child = max(root.children.items(), key=lambda x: x[1].visit_count)
            self.last_root_q = -root.q_value
            self.last_best_child_q = best_child.q_value
            return best_move

        stats = defaultdict(int)
        completion_event = threading.Event()

        # Start evaluator (keeps running across searches). No-op in inline mode.
        if not self.evaluator.inline:
            self.evaluator.start()

        # Precompute the game-history repetition counts once for this search.
        # `board` carries the full move stack (the UCI layer replays every game
        # move), so this sees repetitions that occurred before the current move.
        repetition_history = build_repetition_history(board)

        # Start workers
        workers = []
        for i in range(self.num_workers):
            worker = MCTSWorker(
                worker_id=i,
                root=root,
                root_board=board,
                evaluator=self.evaluator,
                cache=self.cache,
                stats=stats,
                target_sims=target_new_sims,
                completion_event=completion_event,
                tablebase=self.tablebase,
                repetition_history=repetition_history,
                params=self.params,
            )
            worker.start()
            workers.append(worker)

        # Wait for simulations to complete (event-based, no polling)
        if time_limit:
            completion_event.wait(timeout=time_limit)
        else:
            completion_event.wait()

        # Stop workers (evaluator keeps running for next search)
        for worker in workers:
            worker.stop()

        if _INSTR:
            # Item 9: what the workers actually did vs what was asked for.
            self.last_search_audit = {
                'existing_visits': existing_visits,
                'target_new_sims': target_new_sims,
                'stats_simulations': stats['simulations'],
                'root_visit_count': root.visit_count,
                'num_workers': self.num_workers,
            }

        # Mate-in-one short-circuit: a worker found a root-child that's
        # terminal with a winning value for us. Return it directly with a
        # mate-equivalent score so UCI cp reporting reflects the win.
        mating_move = stats.get('mating_move')
        if mating_move is not None and mating_move in root.children:
            self.last_root_q = 1.0
            self.last_best_child_q = 1.0
            return mating_move

        # Select best move (most visited)
        best_move, best_child = max(root.children.items(), key=lambda x: x[1].visit_count)

        # Store evaluation stats from SIDE-TO-MOVE-AT-ROOT's (engine's) perspective.
        # Convention: each node's q_value is from the MOVER's perspective (who moved TO that node).
        # - root.q_value is from the opponent's view (they moved to reach root) -> negate
        # - best_child.q_value is from engine's view (engine moves to reach child) -> use as-is
        self.last_root_q = -root.q_value
        self.last_best_child_q = best_child.q_value

        return best_move

    def _expand_root(self, root: MCTSNode, board: chess.Board):
        """Expand root node synchronously."""
        _xr0 = 0.0
        if _INSTR:
            _xr0 = time.perf_counter()
        tokens = board_to_tokens(board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast_mode.autocast(device_type='cuda', dtype=AUTOCAST_DTYPE,
                                                  enabled=self.device.type == 'cuda'):
                policy_logits, value = self.model(tokens)

        legal_moves = list(board.legal_moves)
        # Same sharpening as an interior node: the root's priors feed the same
        # PUCT comparison, and Dirichlet noise (added afterwards, when enabled)
        # is defined as a mix with the final priors.
        root.expand(policy_logits[0], legal_moves, self.params.policy_temperature)
        # Gate 1 only; see the interior-node call site in _run_simulation.
        # Pondering inherits this through _expand_root.
        _canonicalize_children(root)
        # If this node had been marked drawn-by-rule (fifty-move / threefold) it
        # was left unexpanded precisely so we could reach here. Now that it IS a
        # search root with real children, the mark is stale: the host declined to
        # claim the draw and we are playing on. Clearing it keeps the tree's
        # invariant honest (no node is both terminal and branching). Guarded on
        # children actually existing, so a genuine checkmate/stalemate leaf --
        # legal_moves empty, expand() returns without children -- keeps its mark
        # and its cached value.
        if root.children:
            root.is_terminal = False
        root.visit_count = 1
        # Seed value in MOVER's perspective (same convention as backpropagate).
        # The v5 value head is White-POV by construction (model_v5: "tanh,
        # reading the CLS token at position 67. White-POV"), so +1 means White
        # is winning regardless of side to move. Mover to root = opponent of
        # board.turn.
        nn_value = value[0].item()
        root.value_sum = nn_value if board.turn == chess.BLACK else -nn_value
        if _INSTR:
            _b = _instr_bucket()
            _b['expand_root_s'] += time.perf_counter() - _xr0
            _b['expand_root_calls'] += 1

    def get_policy(self, board: chess.Board, num_simulations: int = 800,
                   add_dirichlet_noise: bool = False) -> dict[chess.Move, float]:
        """
        Run MCTS and return visit count distribution over moves.
        Useful for training data generation. Benefits from tree reuse.

        Args:
            board: Current board position
            num_simulations: Target total simulations
            add_dirichlet_noise: If True, add Dirichlet noise to root priors

        Returns:
            Dictionary mapping moves to visit count proportions
        """
        self.ponder_stop()
        board_hash = chess.polyglot.zobrist_hash(board)

        # Check if we can reuse the existing tree
        if self.root is not None and self._root_hash == board_hash:
            root = self.root
            self._reset_virtual_loss(root)
            # If this node was an unexplored leaf in the prior tree, expand now
            # so root.children is populated before the search proceeds. The
            # `or not root.children` half is the same forced-to-play recovery
            # search() does -- see the comment there.
            if not root.is_expanded or not root.children:
                self._expand_root(root, board)
        else:
            root = MCTSNode()
            self._expand_root(root, board)
            self.root = root
            self.root_board = board.copy()
            self._root_hash = board_hash

        if not root.children:
            # Mirror of search()'s last-resort net: an empty policy makes callers
            # (pick_engine_move's argmax) produce no move at all. Only return
            # empty when the board really has nothing to play.
            legal = list(board.legal_moves)
            if not legal:
                return {}
            print(f"[mcts] WARNING: root has no children but {len(legal)} legal "
                  f"moves exist ({board.fen()}); returning uniform policy",
                  file=sys.stderr, flush=True)
            return {m: 1.0 / len(legal) for m in legal}

        if add_dirichlet_noise:
            self._add_dirichlet_noise(root)

        # Calculate new simulations needed
        existing_visits = root.visit_count
        target_new_sims = max(0, num_simulations - existing_visits)

        if target_new_sims > 0:
            stats = defaultdict(int)
            completion_event = threading.Event()

            if not self.evaluator.inline:
                self.evaluator.start()

            repetition_history = build_repetition_history(board)

            workers = []
            for i in range(self.num_workers):
                worker = MCTSWorker(
                    worker_id=i,
                    root=root,
                    root_board=board,
                    evaluator=self.evaluator,
                    cache=self.cache,
                    stats=stats,
                    target_sims=target_new_sims,
                    completion_event=completion_event,
                    tablebase=self.tablebase,
                    repetition_history=repetition_history,
                    params=self.params,
                )
                worker.start()
                workers.append(worker)

            completion_event.wait()

            for worker in workers:
                worker.stop()

            # Mate-in-one short-circuit: a worker found a winning terminal at
            # depth 1. The visit distribution at this point is unreliable
            # (mate detection fires completion_event after a single visit on
            # the mating child while other children have already accumulated
            # many visits from earlier sims). Return a one-hot policy on the
            # mating move so callers like pick_engine_move's argmax pick it.
            mating_move = stats.get('mating_move')
            if mating_move is not None and mating_move in root.children:
                self.last_root_q = 1.0
                self.last_best_child_q = 1.0
                return {mating_move: 1.0}

        # Return normalized visit counts
        total_visits = sum(child.visit_count for child in root.children.values())
        if total_visits == 0:
            return {}

        policy = {
            move: child.visit_count / total_visits
            for move, child in root.children.items()
        }

        # Update search stats so callers can read root/best-child Q-values
        # Both stored from SIDE-TO-MOVE-AT-ROOT's (engine's) perspective.
        best_child = max(root.children.values(), key=lambda c: c.visit_count)
        self.last_root_q = -root.q_value  # root.q is from opponent's view -> negate
        self.last_best_child_q = best_child.q_value  # child.q is from engine's view -> as-is

        return policy

    # === Pondering ===
    # Between turns, grow the subtree under the predicted opponent reply. If
    # the opponent plays the predicted move, the next search starts with
    # thousands of visits already in place (via apply_move promoting the
    # pondered child). Otherwise the transposition cache still carries over.

    def predict_opponent_move(self) -> Optional[chess.Move]:
        """Most-visited child of the current root — i.e. the opponent's most-likely reply.

        Intended to be called after apply_move(engine_move), when self.root is
        at an opponent-to-move position and its children are opponent replies.
        """
        if self.root is None or not self.root.children:
            return None
        return max(self.root.children.items(), key=lambda x: x[1].visit_count)[0]

    def ponder_start(self, board: chess.Board,
                     confidence_threshold: float = 0.7,
                     max_branches: int = 2):
        """Begin background MCTS on the predicted opponent reply(ies).

        Confidence-gated multi-PV: if the top child's share of root visits is
        at least `confidence_threshold`, ponder only that child (top-1). Else
        ponder up to `max_branches` children, with workers allocated
        proportionally to each child's root-visit share.

        Pass `max_branches=1` to force single-PV, or `confidence_threshold=0.0`
        to always split. No-op if already pondering or the tree isn't set up.
        """
        if self._pondering:
            return
        if self.root is None or not self.root.children:
            return

        # Rank children by root visits (same criterion as search's bestmove).
        children_sorted = sorted(self.root.children.items(),
                                 key=lambda x: x[1].visit_count, reverse=True)
        total_root_visits = sum(c.visit_count for _, c in children_sorted)
        if total_root_visits == 0:
            return  # tree too shallow to predict anything

        top_share = children_sorted[0][1].visit_count / total_root_visits

        # Confidence gate: high-confidence top-1 keeps all workers on one
        # branch (best ROI when prediction is right). Low-confidence splits
        # across top-K.
        if top_share >= confidence_threshold or max_branches <= 1:
            selected = children_sorted[:1]
        else:
            selected = children_sorted[:max_branches]

        # Allocate workers proportionally to selected branches' visit counts.
        # Each branch gets at least 1 worker. Rounding drift is absorbed by
        # the highest-visit branch.
        weights = [c.visit_count for _, c in selected]
        total_w = sum(weights) or 1
        worker_counts = [max(1, int(round(self.num_workers * w / total_w)))
                         for w in weights]
        drift = self.num_workers - sum(worker_counts)
        if drift != 0:
            biggest = max(range(len(weights)), key=lambda i: weights[i])
            worker_counts[biggest] = max(1, worker_counts[biggest] + drift)

        # Build the per-branch ponder setup: expand unexpanded leaves and
        # compute the board position each branch is rooted at.
        branches: list[tuple[chess.Move, MCTSNode, chess.Board, int]] = []
        for (move, node), n_workers in zip(selected, worker_counts):
            bboard = board.copy()
            bboard.push(move)
            if not node.is_expanded:
                self._expand_root(node, bboard)
            branches.append((move, node, bboard, n_workers))

        # Log the decision so the user can see the confidence gate in action.
        def _san(m: chess.Move, b: Optional[chess.Board]) -> str:
            try:
                return b.san(m) if b is not None else m.uci()
            except Exception:
                return m.uci()
        parts = [f"{_san(m, self.root_board)}({n}w,{c.visit_count}v)"
                 for (m, c), n in zip(selected, worker_counts)]
        mode = "single" if len(branches) == 1 else f"multi-{len(branches)}"
        print(f"[ponder] start {mode}: top_confidence={top_share:.0%} "
              f"branches=[{', '.join(parts)}]",
              file=sys.stderr, flush=True)

        # Keep predicted moves + stats dict alive past ponder_stop so that
        # apply_move() can read them to log the outcome.
        self._ponder_predicted_moves = [m for m, _, _, _ in branches]
        self._ponder_stats = defaultdict(int)

        self._ponder_stop_event = threading.Event()
        self._pondering = True
        self._ponder_thread = threading.Thread(
            target=self._ponder_run,
            args=(branches, self._ponder_stop_event, self._ponder_stats),
            daemon=True,
        )
        self._ponder_thread.start()

    def _ponder_run(self,
                    branches: list[tuple[chess.Move, MCTSNode, chess.Board, int]],
                    stop_event: threading.Event, stats: defaultdict):
        """Background thread body: spawn workers across `branches` until `stop_event` is set.

        Each branch is (move, sub_root, sub_board, num_workers_for_branch).
        Workers share the stop_event and stats dict; they share the
        transposition cache too, so transpositions across branches
        deduplicate automatically.
        """
        try:
            for _, node, _, _ in branches:
                self._reset_virtual_loss(node)

            if not self.evaluator.inline:
                self.evaluator.start()

            workers = []
            worker_id = 0
            for _, node, bboard, n_workers in branches:
                # Each branch simulates from its own board (root + predicted
                # reply), so it needs that board's history for repetition checks.
                branch_rep_history = build_repetition_history(bboard)
                for _ in range(n_workers):
                    worker = MCTSWorker(
                        worker_id=worker_id,
                        root=node,
                        root_board=bboard,
                        evaluator=self.evaluator,
                        cache=self.cache,
                        stats=stats,
                        # Cap pondering total sims to prevent unbounded tree
                        # growth
                        target_sims=60000 if self.device.type == 'cuda' else 5000,
                        completion_event=stop_event,
                        tablebase=self.tablebase,
                        repetition_history=branch_rep_history,
                        params=self.params,
                    )
                    worker.start()
                    workers.append(worker)
                    worker_id += 1

            stop_event.wait()

            for worker in workers:
                worker.stop()
        except Exception as e:
            import traceback
            print(f"[mcts ponder] {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)

    def ponder_stop(self):
        """Stop background pondering if active. Safe to call when not pondering."""
        if not self._pondering:
            return
        if self._ponder_stop_event is not None:
            self._ponder_stop_event.set()
        if self._ponder_thread is not None:
            self._ponder_thread.join()
        self._pondering = False
        self._ponder_thread = None
        self._ponder_stop_event = None


# === Board Tokenization ===
# 68 tokens: 64 squares + side + castling + ep + CLS.
#
# These IDs are the v5 training shards' encoding, not a choice made here: they
# must stay identical to data/multiPV/mirror.py (which mirrors them) and to the
# `tokens` field written by data/multiPV/record_format.py. The v5 student never
# saw any other mapping, so a divergence here is unrecoverable at play time and
# would not raise — it would just evaluate the wrong position.

TOKEN_WHITE_TO_MOVE = 13
TOKEN_BLACK_TO_MOVE = 14
TOKEN_CASTLING_BASE = 15
TOKEN_EP_NONE = 31
TOKEN_EP_BASE = 32
TOKEN_CLS = 40


def board_to_tokens(board: chess.Board) -> torch.Tensor:
    """68 tokens (64 squares + side + castling + ep + CLS), as v5 was trained.

    Iterates only occupied squares via piece_map() — empty squares stay zero from
    the preallocated tensor, avoiding 64 piece_at() calls per leaf.
    """
    _b2t0 = 0.0
    if _INSTR:
        _b2t0 = time.perf_counter()
    tokens = torch.zeros(SEQ_LENGTH, dtype=torch.long)

    # Positions 0-63: piece placement (only occupied squares; empties stay 0)
    for square, piece in board.piece_map().items():
        offset = 0 if piece.color else 6
        tokens[square] = piece.piece_type + offset

    # Position 64: side to move
    tokens[64] = TOKEN_WHITE_TO_MOVE if board.turn else TOKEN_BLACK_TO_MOVE

    # Position 65: castling rights (4-bit encoded)
    castling_bits = (
        (8 if board.has_kingside_castling_rights(chess.WHITE) else 0) |
        (4 if board.has_queenside_castling_rights(chess.WHITE) else 0) |
        (2 if board.has_kingside_castling_rights(chess.BLACK) else 0) |
        (1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
    )
    tokens[65] = TOKEN_CASTLING_BASE + castling_bits

    # Position 66: en passant target file
    if board.ep_square is not None:
        tokens[66] = TOKEN_EP_BASE + chess.square_file(board.ep_square)
    else:
        tokens[66] = TOKEN_EP_NONE

    # Position 67: CLS token
    tokens[67] = TOKEN_CLS

    if _INSTR:
        _b = _instr_bucket()
        _b['board_to_tokens_s'] += time.perf_counter() - _b2t0
        _b['board_to_tokens_calls'] += 1

    return tokens


# === Demo / Testing ===

def main():
    """Demo: run MCTS on starting position with tree reuse."""
    import sys
    from pathlib import Path
    _project_root = Path(__file__).resolve().parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    # Try to load the model
    from training.train import ChessTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check for trained model
    import glob
    model_files = glob.glob(str(_project_root / "models" / "guofish_*.pt"))

    if model_files:
        # Load latest model
        model_path = max(model_files, key=lambda x: x)
        print(f"Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model = ChessTransformer().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No trained model found. Using random weights for demo.")
        model = ChessTransformer().to(device)

    # Configure MCTS
    num_workers = 4 if device.type == 'cuda' else 2
    mcts = ParallelMCTS(model, device, num_workers=num_workers)

    # === First search: starting position ===
    board = chess.Board()
    print(f"\n{'='*60}")
    print(f"Search 1: Starting position with {num_workers} workers")
    print(f"{'='*60}")
    print(board)

    start_time = time.time()
    best_move = mcts.search(board, num_simulations=400)
    elapsed = time.time() - start_time

    print(f"\nBest move: {best_move}")
    print(f"Time: {elapsed:.2f}s | Sims/sec: {400/elapsed:.1f}")
    print(f"Root visits after search: {mcts.root.visit_count if mcts.root else 0}")
    print(f"Cache: {mcts.cache.hits} hits, {mcts.cache.misses} misses ({mcts.cache.hit_rate:.1%})")

    if best_move is None:
        print("No legal moves found!")
        return

    # === Apply the best move and demonstrate tree reuse ===
    print(f"\n{'='*60}")
    print(f"Applying move {best_move} and reusing subtree...")
    print(f"{'='*60}")

    # Check visits in the child we're about to promote
    if mcts.root and best_move in mcts.root.children:
        child_visits_before = mcts.root.children[best_move].visit_count
        print(f"Child '{best_move}' visits before apply_move: {child_visits_before}")

    mcts.apply_move(best_move)
    board.push(best_move)

    print(f"Root visits after apply_move: {mcts.root.visit_count if mcts.root else 0}")
    print(board)

    # === Second search: after e2e4 (or whatever move was played) ===
    print(f"\n{'='*60}")
    print(f"Search 2: Position after {best_move} (tree reuse active)")
    print(f"{'='*60}")

    # Request 600 total simulations - some already exist from first search
    existing = mcts.root.visit_count if mcts.root else 0
    print(f"Existing visits: {existing}, requesting 600 total")

    start_time = time.time()
    best_move_2 = mcts.search(board, num_simulations=600)
    elapsed = time.time() - start_time

    new_visits = (mcts.root.visit_count if mcts.root else 0) - existing
    print(f"\nBest move: {best_move_2}")
    print(f"Time: {elapsed:.2f}s | New sims: {new_visits}")
    print(f"Root visits after search: {mcts.root.visit_count if mcts.root else 0}")
    print(f"Cache: {mcts.cache.hits} hits, {mcts.cache.misses} misses ({mcts.cache.hit_rate:.1%})")

    # === Summary ===
    print(f"\n{'='*60}")
    print("Tree Reuse Summary:")
    print(f"  - First search built tree with ~400 visits")
    print(f"  - apply_move() preserved subtree under '{best_move}'")
    print(f"  - Second search only needed ~{new_visits} new simulations")
    print(f"  - Total evaluator batches: {mcts.evaluator.total_batches}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
