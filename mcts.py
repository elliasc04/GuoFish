"""Parallel MCTS with Batched Neural Network Evaluation.

Architecture:
- Multiple worker threads traverse the MCTS tree using PUCT
- Workers submit leaf nodes to a shared queue and wait for evaluation
- Single evaluator thread batches positions and runs them through the NN
- Virtual loss prevents workers from all exploring the same path

Usage:
    from mcts import ParallelMCTS
    mcts = ParallelMCTS(model, device, num_workers=8)
    best_move = mcts.search(board, num_simulations=800)
"""

import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Optional

import chess
import torch

# MCTS hyperparameters
C_PUCT = 1.5  # Exploration constant
VIRTUAL_LOSS = 3  # Penalize in-flight nodes to encourage exploration diversity


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

            # Isolate, move to CPU, convert to float32, and Softmax
            legal_logits = policy[legal_indices].float().cpu()
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
                 batch_timeout_ms: float = 50.0):
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size  # Wait for at least this many
        self.batch_timeout = batch_timeout_ms / 1000.0  # Convert to seconds

        self.request_queue: Queue[EvalRequest] = Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Stats
        self.total_evals = 0
        self.total_batches = 0

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

        # First, block until we get at least one request
        while len(batch) == 0 and self.running:
            try:
                request = self.request_queue.get(timeout=0.005)
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

        # Distribute results back to workers
        for i, req in enumerate(batch):
            req.policy = policy_logits[i]
            req.value = values[i].item()
            req.event.set()  # Signal worker that result is ready

        # Stats
        self.total_batches += 1
        self.total_evals += len(batch)


class MCTSWorker:
    """Worker thread that traverses the tree and submits leaves for evaluation."""

    def __init__(self, worker_id: int, root: MCTSNode, root_board: chess.Board,
                 evaluator: BatchedEvaluator, stats: dict, target_sims: int,
                 completion_event: threading.Event):
        self.worker_id = worker_id
        self.root = root
        self.root_board = root_board
        self.evaluator = evaluator
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
        while self.running:
            # Check if we've hit the target
            if self.stats['simulations'] >= self.target_sims:
                self.completion_event.set()
                break
            self._run_simulation()

    def _run_simulation(self):
        """Run one MCTS simulation."""
        node = self.root
        board = self.root_board.copy()
        path = [node]  # Track path for virtual loss

        # === Selection: traverse to leaf ===
        while node.is_expanded and node.children:
            node.apply_virtual_loss()
            node = node.select_child()
            path.append(node)
            board.push(node.move)

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

        # === Expansion & Evaluation: submit to evaluator ===
        # Tokenize on worker thread to offload CPU work from evaluator
        tokens = board_to_tokens(board)
        request = EvalRequest(node=node, tokens=tokens)
        self.evaluator.submit(request)
        request.event.wait()  # Block until evaluated

        # Expand node with policy LOGITS (Softmax happens inside expand now)
        legal_moves = list(board.legal_moves)
        node.expand(request.policy, legal_moves)

        # Revert virtual loss along path
        for n in path:
            n.revert_virtual_loss()

        # === Absolute to Mover's Perspective Conversion ===
        # The NN outputs Absolute value (White winning = +1.0, Black winning = -1.0).
        # We need Mover's perspective (who moved TO this node = opponent of board.turn).
        # - If board.turn == BLACK, mover was WHITE, use NN value as-is.
        # - If board.turn == WHITE, mover was BLACK, negate NN value.
        nn_value = request.value if request.value is not None else 0.0
        mover_value = nn_value if board.turn == chess.BLACK else -nn_value

        # Backpropagate from mover's perspective
        node.backpropagate(mover_value)

        self.stats['simulations'] += 1


class ParallelMCTS:
    """Parallel MCTS with batched neural network evaluation."""

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 num_workers: Optional[int] = None, max_batch_size: Optional[int] = None):
        self.model = model
        self.device = device

        # Auto-tune workers: more workers helps fill batches despite GIL
        # Use many workers to keep the request queue full
        if num_workers is None:
            num_workers = 32 if device.type == 'cuda' else 8
        self.num_workers = num_workers

        # Auto-tune batch size based on hardware
        if max_batch_size is None:
            if device.type == 'cuda':
                max_batch_size = 1024  # GPU can handle large batches
            else:
                max_batch_size = 32  # CPU - keep it small

        # Min batch size - aim for good GPU utilization
        # Larger min = better GPU efficiency but more latency
        min_batch_size = max(16, num_workers)

        self.evaluator = BatchedEvaluator(
            model=model,
            device=device,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            batch_timeout_ms=100.0  # Wait up to 100ms to collect min_batch_size
        )

        self.model.eval()

        # Store last search stats for external access
        self.last_root_q = 0.0  # Q-value of root after search (from side-to-move perspective)
        self.last_best_child_q = 0.0  # Q-value of best move

    def shutdown(self):
        """Stop the evaluator thread. Call when done with MCTS."""
        self.evaluator.stop()

    def search(self, board: chess.Board, num_simulations: int = 800,
               time_limit: float = None) -> chess.Move:
        """
        Run MCTS search and return the best move.

        Args:
            board: Current board position
            num_simulations: Target number of simulations
            time_limit: Optional time limit in seconds (overrides num_simulations)

        Returns:
            Best move according to MCTS
        """
        root = MCTSNode()
        stats = defaultdict(int)
        completion_event = threading.Event()

        # Initial expansion of root
        self._expand_root(root, board)

        if not root.children:
            # No legal moves
            return None

        # Start evaluator (keeps running across searches)
        self.evaluator.start()

        # Start workers
        workers = []
        for i in range(self.num_workers):
            worker = MCTSWorker(
                worker_id=i,
                root=root,
                root_board=board,
                evaluator=self.evaluator,
                stats=stats,
                target_sims=num_simulations,
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

        # Store evaluation stats for external access
        # Q-values are from the perspective of the side that just moved TO this position
        # So we negate to get perspective of side-to-move at root
        self.last_root_q = root.q_value
        self.last_best_child_q = -best_child.q_value  # Negate: child Q is from opponent's view

        return best_move

    def _expand_root(self, root: MCTSNode, board: chess.Board):
        """Expand root node synchronously."""
        tokens = board_to_tokens(board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
                policy_logits, value = self.model(tokens)

        legal_moves = list(board.legal_moves)
        root.expand(policy_logits[0], legal_moves)
        root.visit_count = 1
        root.value_sum = value[0].item()

    def get_policy(self, board: chess.Board, num_simulations: int = 800) -> dict[chess.Move, float]:
        """
        Run MCTS and return visit count distribution over moves.
        Useful for training data generation.
        """
        root = MCTSNode()
        stats = defaultdict(int)
        completion_event = threading.Event()

        self._expand_root(root, board)

        if not root.children:
            return {}

        self.evaluator.start()

        workers = []
        for i in range(self.num_workers):
            worker = MCTSWorker(
                worker_id=i,
                root=root,
                root_board=board,
                evaluator=self.evaluator,
                stats=stats,
                target_sims=num_simulations,
                completion_event=completion_event
            )
            worker.start()
            workers.append(worker)

        completion_event.wait()

        for worker in workers:
            worker.stop()

        # Return normalized visit counts
        total_visits = sum(child.visit_count for child in root.children.values())
        policy = {
            move: child.visit_count / total_visits
            for move, child in root.children.items()
        }

        return policy


# === Board Tokenization (matches play.py) ===

def board_to_tokens(board: chess.Board) -> torch.Tensor:
    """Convert chess board to token tensor for the neural network."""
    tokens = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            tokens.append(0)
        else:
            # White pieces: 1-6, Black pieces: 7-12
            offset = 0 if piece.color else 6
            tokens.append(piece.piece_type + offset)
    # Side to move token: 13 for white, 14 for black
    tokens.append(13 if board.turn else 14)
    return torch.tensor(tokens, dtype=torch.long)


# === Demo / Testing ===

def main():
    """Demo: run MCTS on starting position."""
    import sys
    sys.path.insert(0, '.')

    # Try to load the model
    from train import ChessTransformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check for trained model
    import glob
    model_files = glob.glob("models/guofish_*.pt")

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

    # Run search on starting position
    board = chess.Board()
    print(f"\nSearching starting position with {num_workers} workers...")
    print(board)

    start_time = time.time()
    best_move = mcts.search(board, num_simulations=400)
    elapsed = time.time() - start_time

    print(f"\nBest move: {best_move}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Simulations: 400")
    print(f"Sims/sec: {400/elapsed:.1f}")
    print(f"Batches: {mcts.evaluator.total_batches}")
    print(f"Avg batch size: {mcts.evaluator.total_evals / max(1, mcts.evaluator.total_batches):.1f}")


if __name__ == "__main__":
    main()
