"""Parallel MCTS with Batched Neural Network Evaluation.

Architecture:
- Multiple worker threads traverse the MCTS tree using PUCT
- Workers submit leaf nodes to a shared queue and wait for evaluation
- Single evaluator thread batches positions and runs them through the NN
- Virtual loss prevents workers from all exploring the same path

Usage:
    from core.mcts import ParallelMCTS
    mcts = ParallelMCTS(model, device, num_workers=8)
    best_move = mcts.search(board, num_simulations=800)
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
import numpy as np
import torch

# MCTS hyperparameters
C_PUCT = 1.5  # Exploration constant
VIRTUAL_LOSS = 3  # Penalize in-flight nodes to encourage exploration diversity
MAX_TREE_DEPTH = 60  # Maximum simulation depth to prevent endgame slowdowns


class TranspositionCache:
    """Thread-safe ring buffer cache for NN evaluations, keyed by Zobrist hash.

    Uses a circular buffer instead of OrderedDict to avoid O(n) LRU operations.
    Stores (policy_logits, value) pairs to avoid re-evaluating positions
    reached via different move orders (transpositions).
    """

    def __init__(self, max_size: int = 500_000):
        self.max_size = max_size
        # Hash table for O(1) lookup
        self._cache: dict[int, tuple[torch.Tensor, float]] = {}
        # Ring buffer of keys for O(1) eviction (no ordering maintained)
        self._ring: list[Optional[int]] = [None] * max_size
        self._ring_idx = 0  # Next position to write
        self._lock = threading.Lock()

        # Stats
        self.hits = 0
        self.misses = 0

    def get(self, zobrist_hash: int) -> Optional[tuple[torch.Tensor, float]]:
        """Get cached (policy_logits, value) for hash. Returns None on miss."""
        with self._lock:
            result = self._cache.get(zobrist_hash)
            if result is not None:
                self.hits += 1
                return result
            self.misses += 1
            return None

    def put(self, zobrist_hash: int, policy: torch.Tensor, value: float):
        """Store (policy_logits, value) for hash. Evicts oldest entry if at capacity."""
        with self._lock:
            if zobrist_hash in self._cache:
                # Update existing entry (no ring position change needed)
                self._cache[zobrist_hash] = (policy, value)
                return

            # Evict entry at current ring position if occupied
            old_key = self._ring[self._ring_idx]
            if old_key is not None:
                self._cache.pop(old_key, None)

            # Insert new entry
            self._cache[zobrist_hash] = (policy, value)
            self._ring[self._ring_idx] = zobrist_hash

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

    __slots__ = ['parent', 'move', 'prior', 'children', 'visit_count',
                 'value_sum', 'virtual_loss', 'is_expanded', 'lock']

    def __init__(self, parent: Optional['MCTSNode'] = None,
                 move: Optional[chess.Move] = None, prior: float = 0.0):
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior  # P(s,a) from neural network
        self.children: dict[chess.Move, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0  # In-flight penalty
        self.is_expanded = False
        self.lock = threading.Lock()

    @property
    def effective_visits(self) -> int:
        """Visit count including virtual losses."""
        return self.visit_count + self.virtual_loss

    @property
    def q_value(self) -> float:
        """Mean action value Q(s,a), adjusted for virtual loss."""
        total_visits = self.visit_count + self.virtual_loss
        if total_visits == 0:
            return 0.0
        # Virtual losses count as losses (value = -1)
        adjusted_value = self.value_sum - self.virtual_loss
        return adjusted_value / total_visits

    def ucb_score(self, parent_visits: int) -> float:
        """PUCT score for node selection."""
        exploration = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.effective_visits)
        return self.q_value + exploration

    def apply_virtual_loss(self):
        """Apply virtual loss when worker starts traversing through this node."""
        with self.lock:
            self.virtual_loss += VIRTUAL_LOSS

    def revert_virtual_loss(self):
        """Remove virtual loss after evaluation completes."""
        with self.lock:
            self.virtual_loss -= VIRTUAL_LOSS

    def expand(self, policy: torch.Tensor, legal_moves: list[chess.Move]):
        """Expand node with children based on policy network output."""
        with self.lock:
            if self.is_expanded:
                return

            # Extract logits ONLY for legal moves
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
            legal_probs = torch.softmax(legal_logits, dim=0).tolist()

            # Map the properly scaled probabilities to the children
            for move, prior in zip(legal_moves, legal_probs):
                self.children[move] = MCTSNode(parent=self, move=move, prior=prior)

            self.is_expanded = True

    def select_child(self) -> 'MCTSNode':
        """Select child with highest UCB score."""
        parent_visits = self.effective_visits
        best_score = float('-inf')
        best_child = None

        for child in self.children.values():
            score = child.ucb_score(parent_visits)
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
                 batch_timeout_ms: float = 50.0, seq_length: int = 68,
                 inline: bool = False):
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size  # Wait for at least this many
        self.batch_timeout = batch_timeout_ms / 1000.0  # Convert to seconds
        self.seq_length = seq_length  # Tokenization scheme (65 or 68)
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
        return policy_logits[0], values[0].item()

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

        # First, block until we get at least one request (no timeout polling)
        while len(batch) == 0 and self.running:
            try:
                # Use blocking get() - avoids CPU spin from timeout polling
                request = self.request_queue.get(timeout=1.0)  # Long timeout for clean shutdown
                batch.append(request)
            except Empty:
                continue

        # Immediately drain everything currently in the queue (non-blocking)
        while len(batch) < self.max_batch_size:
            try:
                request = self.request_queue.get_nowait()
                batch.append(request)
            except Empty:
                break

        # If we don't have enough yet, wait a bit more for stragglers
        if len(batch) < self.min_batch_size:
            deadline = time.time() + self.batch_timeout
            while len(batch) < self.min_batch_size and self.running:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    request = self.request_queue.get(timeout=min(remaining, 0.01))
                    batch.append(request)
                except Empty:
                    continue

        return batch

    def _evaluate_batch(self, batch: list[EvalRequest]):
        """Run batch through neural network and distribute results."""
        if not batch:
            return

        # Stack pre-tokenized tensors from workers (no CPU work here)
        tokens_batch = torch.stack([req.tokens for req in batch]).to(self.device)

        # Forward pass
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
                policy_logits, values = self.model(tokens_batch)

        # Bulk D2H transfer - move entire tensors to CPU before iterating
        # This avoids per-item synchronization overhead
        policy_cpu = policy_logits.cpu()
        values_cpu = values.cpu()

        # Distribute results back to workers (now from CPU tensors)
        for i, req in enumerate(batch):
            req.policy = policy_cpu[i]
            req.value = values_cpu[i].item()
            req.event.set()  # Signal worker that result is ready

        # Stats
        self.total_batches += 1
        self.total_evals += len(batch)


class MCTSWorker:
    """Worker thread that traverses the tree and submits leaves for evaluation."""

    def __init__(self, worker_id: int, root: MCTSNode, root_board: chess.Board,
                 evaluator: BatchedEvaluator, cache: TranspositionCache,
                 stats: dict, target_sims: int, completion_event: threading.Event):
        self.worker_id = worker_id
        self.root = root
        self.root_board = root_board
        self.evaluator = evaluator
        self.cache = cache
        self.stats = stats
        self.target_sims = target_sims
        self.completion_event = completion_event
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

    def _run_simulation(self):
        """Run one MCTS simulation."""
        node = self.root
        # stack=False skips cloning move history (much faster, we don't need undo)
        board = self.root_board.copy(stack=False)
        path = [node]  # Track path for virtual loss
        depth = 0

        # === Selection: traverse to leaf ===
        while node.is_expanded and node.children:
            node.apply_virtual_loss()
            node = node.select_child()
            path.append(node)
            board.push(node.move)
            depth += 1

            # Max depth cutoff to prevent endgame slowdowns
            if depth >= MAX_TREE_DEPTH:
                # Treat as terminal with value 0 (draw-ish)
                for n in path:
                    n.revert_virtual_loss()
                node.backpropagate(0.0)
                self.stats['simulations'] += 1
                return

        node.apply_virtual_loss()

        # === Check terminal state ===
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

            # Revert virtual loss and backprop
            for n in path:
                n.revert_virtual_loss()
            node.backpropagate(value)
            return

        # === Expansion & Evaluation ===
        # Check transposition cache first (Zobrist hash includes side-to-move, castling, ep)
        zobrist_hash = chess.polyglot.zobrist_hash(board)
        cached = self.cache.get(zobrist_hash)

        if cached is not None:
            # Cache hit - use cached policy and value
            policy, nn_value = cached
        else:
            tokens = board_to_tokens(board, self.evaluator.seq_length)

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

            # Store in cache (policy logits, not softmax'd)
            if policy is not None:
                self.cache.put(zobrist_hash, policy, nn_value)

        # Expand node with policy LOGITS (Softmax happens inside expand now)
        legal_moves = list(board.legal_moves)
        node.expand(policy, legal_moves)

        # Revert virtual loss along path
        for n in path:
            n.revert_virtual_loss()

        # === Absolute to Mover's Perspective Conversion ===
        # The NN outputs Absolute value (White winning = +1.0, Black winning = -1.0).
        # We need Mover's perspective (who moved TO this node = opponent of board.turn).
        # - If board.turn == BLACK, mover was WHITE, use NN value as-is.
        # - If board.turn == WHITE, mover was BLACK, negate NN value.
        mover_value = nn_value if board.turn == chess.BLACK else -nn_value

        # Backpropagate from mover's perspective
        node.backpropagate(mover_value)

        self.stats['simulations'] += 1


class ParallelMCTS:
    """Parallel MCTS with batched neural network evaluation and tree reuse."""

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 num_workers: Optional[int] = None, max_batch_size: Optional[int] = None,
                 cache_size: int = 500_000):
        self.model = model
        self.device = device

        # Auto-detect tokenization scheme from model's seq_length attribute
        # (set by ChessTransformerV1/V2 in play.py)
        self.seq_length: int = getattr(model, 'seq_length', 68)

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
        if device.type == 'cuda':
            min_batch_size = max(4, num_workers)
            batch_timeout_ms = 100.0
        else:
            min_batch_size = 1  # Process immediately on CPU
            batch_timeout_ms = 5.0  # Short timeout - don't block workers

        self.evaluator = BatchedEvaluator(
            model=model,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            batch_timeout_ms=batch_timeout_ms,
            seq_length=self.seq_length,
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
        """Recursively reset virtual_loss to 0 in the subtree (defensive)."""
        node.virtual_loss = 0
        for child in node.children.values():
            self._reset_virtual_loss(child)

    def _add_dirichlet_noise(self, root: MCTSNode, alpha: float = 0.3, epsilon: float = 0.25):
        """
        Add Dirichlet noise to root's children priors for exploration.
        Standard AlphaZero formula: P'(a) = (1 - epsilon) * P(a) + epsilon * Dir(alpha)
        """
        if not root.children:
            return

        moves = list(root.children.keys())
        noise = np.random.dirichlet([alpha] * len(moves))

        for move, n in zip(moves, noise):
            child = root.children[move]
            child.prior = (1 - epsilon) * child.prior + epsilon * n

    def search(self, board: chess.Board, num_simulations: int = 800,
               time_limit: float = None, add_dirichlet_noise: bool = False) -> Optional[chess.Move]:
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
            self._reset_virtual_loss(root)
            # If this node was an unexplored leaf in the prior tree, it has no
            # children yet. Expand now so max(root.children) doesn't return None
            # (which would produce an illegal "0000" bestmove).
            if not root.is_expanded:
                self._expand_root(root, board)
        else:
            # Create fresh root
            root = MCTSNode()
            self._expand_root(root, board)
            self.root = root
            self.root_board = board.copy()
            self._root_hash = board_hash

        if not root.children:
            # No legal moves
            return None

        # Add Dirichlet noise for exploration (per-search, not per-tree)
        if add_dirichlet_noise:
            self._add_dirichlet_noise(root)

        # Calculate how many new simulations to run
        # num_simulations is the target total, so subtract existing visits
        existing_visits = root.visit_count
        target_new_sims = max(0, num_simulations - existing_visits)

        if target_new_sims == 0:
            # Already have enough simulations
            best_move, best_child = max(root.children.items(), key=lambda x: x[1].visit_count)
            self.last_root_q = root.q_value
            self.last_best_child_q = -best_child.q_value
            return best_move

        stats = defaultdict(int)
        completion_event = threading.Event()

        # Start evaluator (keeps running across searches). No-op in inline mode.
        if not self.evaluator.inline:
            self.evaluator.start()

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
                completion_event=completion_event
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
        tokens = board_to_tokens(board, self.seq_length).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
                policy_logits, value = self.model(tokens)

        legal_moves = list(board.legal_moves)
        root.expand(policy_logits[0], legal_moves)
        root.visit_count = 1
        # Seed value in MOVER's perspective (same convention as backpropagate).
        # NN outputs absolute value (White winning = +1). Mover to root = opponent of board.turn.
        nn_value = value[0].item()
        root.value_sum = nn_value if board.turn == chess.BLACK else -nn_value

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
            # so root.children is populated before the search proceeds.
            if not root.is_expanded:
                self._expand_root(root, board)
        else:
            root = MCTSNode()
            self._expand_root(root, board)
            self.root = root
            self.root_board = board.copy()
            self._root_hash = board_hash

        if not root.children:
            return {}

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
                    completion_event=completion_event
                )
                worker.start()
                workers.append(worker)

            completion_event.wait()

            for worker in workers:
                worker.stop()

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
                for _ in range(n_workers):
                    worker = MCTSWorker(
                        worker_id=worker_id,
                        root=node,
                        root_board=bboard,
                        evaluator=self.evaluator,
                        cache=self.cache,
                        stats=stats,
                        # Effectively unbounded — stop_event is the real terminator.
                        target_sims=10**18,
                        completion_event=stop_event,
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
# V1 (65 tokens): 64 squares + side-to-move
# V2 (68 tokens): 64 squares + side + castling + ep + CLS

TOKEN_WHITE_TO_MOVE = 13
TOKEN_BLACK_TO_MOVE = 14
TOKEN_CASTLING_BASE = 15
TOKEN_EP_NONE = 31
TOKEN_EP_BASE = 32
TOKEN_CLS = 40


def board_to_tokens_v1(board: chess.Board) -> torch.Tensor:
    """V1: 65 tokens (64 squares + side-to-move).

    Iterates only occupied squares via piece_map() — empty squares stay zero from
    the preallocated tensor, avoiding 64 piece_at() calls per leaf.
    """
    tokens = torch.zeros(65, dtype=torch.long)
    for square, piece in board.piece_map().items():
        offset = 0 if piece.color else 6
        tokens[square] = piece.piece_type + offset
    tokens[64] = 13 if board.turn else 14
    return tokens


def board_to_tokens_v2(board: chess.Board) -> torch.Tensor:
    """V2: 68 tokens (64 squares + side + castling + ep + CLS)."""
    tokens = torch.zeros(68, dtype=torch.long)

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

    return tokens


def board_to_tokens(board: chess.Board, seq_length: int = 68) -> torch.Tensor:
    """Convert board to tokens using appropriate scheme based on seq_length."""
    if seq_length == 65:
        return board_to_tokens_v1(board)
    else:
        return board_to_tokens_v2(board)


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
