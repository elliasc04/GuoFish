"""Play chess against a ChessTransformer model.

Supports both regular and FP16-compressed checkpoints.
Optionally uses MCTS for stronger play.

Usage:
    python play.py                                           # uses default checkpoint
    python play.py models/chess_transformer_25.8M_50.5pct_fp16.pt
    python play.py --mcts --simulations 800                  # use MCTS search
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import chess
import chess.polyglot
import torch
import torch.nn as nn

# Make the project root importable when this file is run as a script
# (python playing/play.py) rather than as a module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# --- Model definitions ---

class ChessTransformerV1(nn.Module):
    """Original architecture: 65 tokens (64 squares + side-to-move), mean pooling."""

    def __init__(self, vocab_size=15, d_model=512, nhead=8, num_layers=8, dropout=0.1, head_dim=256):
        super().__init__()
        self.seq_length = 65
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.pos_encoder = nn.Parameter(torch.randn(1, 65, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1), nn.Tanh(),
        )
        self.head_dim = head_dim
        self.from_proj = nn.Linear(d_model, head_dim)
        self.to_proj = nn.Linear(d_model, head_dim)
        self.logit_scale = 1.0 / math.sqrt(head_dim)

    def forward(self, x, legal_move_mask=None):
        x = self.embedding(x) + self.pos_encoder
        x = self.emb_dropout(x)
        x = self.transformer(x)
        pooled_state = x.mean(dim=1)
        value = self.value_head(pooled_state).squeeze(-1)
        x_squares = x[:, :64, :]
        from_feats = self.from_proj(x_squares)
        to_feats = self.to_proj(x_squares)
        policy_logits = torch.bmm(from_feats, to_feats.transpose(1, 2)) * self.logit_scale
        policy_logits = policy_logits.view(x.size(0), 4096)
        if legal_move_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_move_mask, float('-inf'))
        return policy_logits, value


class ChessTransformerV2(nn.Module):
    """New architecture: 68 tokens (64 squares + side + castling + ep + CLS), CLS pooling."""

    def __init__(self, vocab_size=43, d_model=512, nhead=8, num_layers=8, dropout=0.1, head_dim=64):
        super().__init__()
        self.seq_length = 68
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.pos_encoder = nn.Parameter(torch.randn(1, 68, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1), nn.Tanh(),
        )
        self.head_dim = head_dim
        self.from_proj = nn.Linear(d_model, head_dim)
        self.to_proj = nn.Linear(d_model, head_dim)
        self.logit_scale = 1.0 / math.sqrt(head_dim)

    def forward(self, x, legal_move_mask=None):
        x = self.embedding(x) + self.pos_encoder
        x = self.emb_dropout(x)
        x = self.transformer(x)
        cls_state = x[:, 67, :]  # CLS token at position 67
        value = self.value_head(cls_state).squeeze(-1)
        x_squares = x[:, :64, :]
        from_feats = self.from_proj(x_squares)
        to_feats = self.to_proj(x_squares)
        policy_logits = torch.bmm(from_feats, to_feats.transpose(1, 2)) * self.logit_scale
        policy_logits = policy_logits.view(x.size(0), 4096)
        if legal_move_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_move_mask, float('-inf'))
        return policy_logits, value


# --- Loading utilities ---

def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model from checkpoint, auto-detecting architecture version."""
    print(f"Loading {checkpoint_path} on {device}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        if "val_acc" in ckpt:
            print(f"Model accuracy: {ckpt['val_acc']:.1f}%")
    else:
        state_dict = ckpt

    # Auto-detect architecture from pos_encoder shape
    pos_encoder_shape = state_dict["pos_encoder"].shape
    seq_length = pos_encoder_shape[1]

    if seq_length == 65:
        print("Detected V1 architecture (65 tokens, mean pooling)")
        ModelClass = ChessTransformerV1
    elif seq_length == 68:
        print("Detected V2 architecture (68 tokens, CLS pooling)")
        ModelClass = ChessTransformerV2
    else:
        raise ValueError(f"Unknown architecture: pos_encoder has {seq_length} positions")

    # Convert FP16 weights back to FP32 for CPU, keep FP16 for CUDA
    if device.type == "cuda":
        model = ModelClass().half().to(device)
        state_dict = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
    else:
        model = ModelClass().to(device)
        state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    if device.type == "cpu":
        # Dynamic INT8 quantization on all Linear layers (FFN, attention out_proj,
        # value/policy heads). Weights are stored as INT8 with per-batch activation
        # quantization. Typically 2-4x faster on CPU with negligible accuracy loss.
        seq_length = model.seq_length
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        model.seq_length = seq_length

        # After quantization, Linear.weight becomes a method (returns dequantized weight
        # on demand) instead of a tensor attribute. nn.TransformerEncoderLayer's fast-path
        # eligibility check iterates `tensor_args` and reads `.device.type`, which crashes
        # on the methodified weights. Trip an earlier short-circuit in the fast-path check
        # so the device probe never runs — the slow path uses self.activation (still the
        # real GELU), so this only affects the fast-path decision, not the math.
        for layer in model.transformer.layers:
            layer.activation_relu_or_gelu = False

    return model


# --- ANSI color codes for console output ---

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


# --- Token constants for V2 architecture ---

TOKEN_WHITE_TO_MOVE = 13
TOKEN_BLACK_TO_MOVE = 14
TOKEN_CASTLING_BASE = 15  # castling = base + (K*8 + Q*4 + k*2 + q*1)
TOKEN_EP_NONE = 31
TOKEN_EP_BASE = 32  # ep file a-h = base + file (0-7)
TOKEN_CLS = 40


# --- Encoding helpers ---

def board_to_tokens_v1(board: chess.Board) -> torch.Tensor:
    """V1: 65 tokens (64 squares + side-to-move)."""
    tokens = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            tokens.append(0)
        else:
            offset = 0 if piece.color else 6
            tokens.append(piece.piece_type + offset)
    tokens.append(13 if board.turn else 14)
    return torch.tensor(tokens, dtype=torch.long)


def board_to_tokens_v2(board: chess.Board) -> torch.Tensor:
    """V2: 68 tokens (64 squares + side + castling + ep + CLS)."""
    tokens = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            tokens.append(0)
        else:
            offset = 0 if piece.color else 6
            tokens.append(piece.piece_type + offset)

    # Position 64: side to move
    tokens.append(TOKEN_WHITE_TO_MOVE if board.turn else TOKEN_BLACK_TO_MOVE)

    # Position 65: castling rights (4-bit encoded)
    castling_bits = (
        (8 if board.has_kingside_castling_rights(chess.WHITE) else 0) |
        (4 if board.has_queenside_castling_rights(chess.WHITE) else 0) |
        (2 if board.has_kingside_castling_rights(chess.BLACK) else 0) |
        (1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
    )
    tokens.append(TOKEN_CASTLING_BASE + castling_bits)

    # Position 66: en passant target file
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        tokens.append(TOKEN_EP_BASE + ep_file)
    else:
        tokens.append(TOKEN_EP_NONE)

    # Position 67: CLS token
    tokens.append(TOKEN_CLS)

    return torch.tensor(tokens, dtype=torch.long)


def board_to_tokens(board: chess.Board, seq_length: int = 65) -> torch.Tensor:
    """Convert board to tokens using appropriate scheme based on seq_length."""
    if seq_length == 65:
        return board_to_tokens_v1(board)
    else:
        return board_to_tokens_v2(board)


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(4096, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move.from_square * 64 + move.to_square] = True
    return mask


def decode_move(index: int, board: chess.Board) -> chess.Move:
    from_sq = index // 64
    to_sq = index % 64
    piece = board.piece_at(from_sq)
    promotion = None
    if piece is not None and piece.piece_type == chess.PAWN:
        rank = chess.square_rank(to_sq)
        if (piece.color == chess.WHITE and rank == 7) or (piece.color == chess.BLACK and rank == 0):
            promotion = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion=promotion)


def pick_engine_move(model: nn.Module, board: chess.Board, device: torch.device,
                     mcts_engine=None, num_simulations: int = 800,
                     temperature: float = 0.0, avoid_repetition: bool = True) -> tuple[chess.Move | None, dict]:
    """Pick engine move using either raw policy or MCTS.

    Args:
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        avoid_repetition: If True and winning, avoid moves that repeat positions

    Returns:
        tuple: (move, stats_dict) where stats_dict contains timing and search info
    """
    import numpy as np
    start_time = time.time()
    stats = {}

    if mcts_engine is not None:
        # Use MCTS search - get policy distribution
        policy_dict = mcts_engine.get_policy(board, num_simulations=num_simulations)
        elapsed = time.time() - start_time

        if not policy_dict:
            return None, stats

        moves = list(policy_dict.keys())
        visit_probs = np.array([policy_dict[m] for m in moves])

        # Get eval from root Q-value.
        # last_root_q is from engine's (side-to-move) perspective.
        # Convert to absolute: positive = White winning, negative = Black winning.
        root_q = mcts_engine.last_root_q
        stats['eval'] = root_q if board.turn == chess.WHITE else -root_q

        # Anti-repetition: only trigger when we're winning AND opponent has already
        # caused a 2-fold repetition (one more repeat by us = 3-fold draw claim).
        # Among moves MCTS considers near-co-best (>= 90% of best visits), avoid
        # the ones that would complete the 3-fold repetition.
        if avoid_repetition and root_q > 0.15 and board.is_repetition(2):
            best_prob = np.max(visit_probs)
            threshold = best_prob * 0.9  # Only moves MCTS considers near-equal to best
            good_moves_mask = visit_probs >= threshold

            # Check which moves would trigger a 3-fold repetition draw
            repeats = []
            for move in moves:
                board.push(move)
                repeats.append(board.is_repetition(3))
                board.pop()

            # Only penalize if there's at least one near-co-best move that doesn't draw
            has_good_non_repeat = any(good_moves_mask[i] and not repeats[i]
                                       for i in range(len(moves)))

            if has_good_non_repeat:
                for i in range(len(moves)):
                    if good_moves_mask[i] and repeats[i]:
                        visit_probs[i] *= 0.01

                if visit_probs.sum() > 0:
                    visit_probs = visit_probs / visit_probs.sum()
                else:
                    visit_probs = np.array([policy_dict[m] for m in moves])

        # Apply temperature sampling
        if temperature > 0.001:
            # Transform visit probabilities with temperature
            visit_counts = visit_probs * num_simulations  # Approximate counts
            visit_counts = np.power(visit_counts, 1.0 / temperature)
            visit_probs = visit_counts / visit_counts.sum()
            move = np.random.choice(moves, p=visit_probs)
            stats['sampled'] = True
        else:
            # Deterministic: pick most visited
            move = moves[np.argmax(visit_probs)]
            stats['sampled'] = False

        # Get stats
        stats['time'] = elapsed
        stats['simulations'] = num_simulations
        stats['sims_per_sec'] = num_simulations / elapsed if elapsed > 0 else 0
        stats['batches'] = mcts_engine.evaluator.total_batches
        stats['avg_batch'] = (mcts_engine.evaluator.total_evals /
                              max(1, mcts_engine.evaluator.total_batches))

        # Reset evaluator stats for next move
        mcts_engine.evaluator.total_batches = 0
        mcts_engine.evaluator.total_evals = 0
    else:
        # Use raw policy (single forward pass)
        seq_length: int = model.seq_length  # type: ignore[assignment]
        tokens = board_to_tokens(board, seq_length).unsqueeze(0).to(device)
        mask = legal_move_mask(board).unsqueeze(0).to(device)

        with torch.no_grad():
            # Handle FP16 model on GPU
            if next(model.parameters()).dtype == torch.float16:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    policy_logits, value = model(tokens, legal_move_mask=mask)
            else:
                policy_logits, value = model(tokens, legal_move_mask=mask)

        best_index = int(torch.argmax(policy_logits, dim=1).item())
        move = decode_move(best_index, board)
        elapsed = time.time() - start_time

        # NN outputs absolute value already (White winning = +, Black winning = -).
        stats['time'] = elapsed
        stats['eval'] = value.item()
        stats['simulations'] = None
        stats['sampled'] = False

    return move, stats


def probe_opening_book(book_reader: Optional[chess.polyglot.MemoryMappedReader],
                       board: chess.Board) -> Optional[chess.Move]:
    """Look up current position in opening book. Returns weighted-random book move or None."""
    if book_reader is None:
        return None
    try:
        entry = book_reader.weighted_choice(board)
        return entry.move
    except IndexError:
        return None


def format_engine_stats(stats: dict) -> str:
    """Format engine stats for display."""
    parts = []

    if stats.get('eval') is not None:
        parts.append(f"eval: {stats['eval']:+.3f}")

    if stats.get('simulations') is not None:
        parts.append(f"sims: {stats['simulations']}")
        parts.append(f"{stats['sims_per_sec']:.0f} sims/s")
        if stats.get('avg_batch'):
            parts.append(f"batch: {stats['avg_batch']:.0f}")

    parts.append(f"time: {stats['time']*1000:.0f}ms")

    return " | ".join(parts)


def format_move_history(board: chess.Board, last_n: int = 8) -> str:
    """Return the last last_n plies of the game in PGN-style notation."""
    moves = list(board.move_stack)
    if not moves:
        return "(start of game)"

    # Replay from scratch to get SAN for each ply
    tmp = chess.Board()
    san_list = []
    for move in moves:
        san_list.append(tmp.san(move))
        tmp.push(move)

    start_ply = max(0, len(san_list) - last_n)
    recent = san_list[start_ply:]

    parts = []
    move_num = (start_ply // 2) + 1
    i = 0

    if start_ply % 2 == 1:
        # First shown ply is Black's move
        parts.append(f"{move_num}...{recent[0]}")
        i, move_num = 1, move_num + 1

    while i < len(recent):
        white = recent[i]
        if i + 1 < len(recent):
            parts.append(f"{move_num}.{white} {recent[i + 1]}")
            i += 2
        else:
            parts.append(f"{move_num}.{white}")
            i += 1
        move_num += 1

    return " ".join(parts)


class RestartGame(Exception):
    """Raised from any input prompt to abort the current game and start a new one."""


_RESTART_WORDS = {"new", "new game", "restart", "again", "play again"}


def prompt(message: str) -> str:
    """input() wrapper that raises RestartGame when the user types a restart keyword."""
    value = input(message).strip()
    if value.lower() in _RESTART_WORDS:
        raise RestartGame()
    return value


def ask_play_again() -> bool:
    while True:
        ans = input("Play again? [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", "quit", "exit"):
            return False
        print("Please enter 'y' or 'n'.")


def main():
    parser = argparse.ArgumentParser(description="Play chess against ChessTransformer")
    parser.add_argument("checkpoint", type=Path, nargs="?",
                        default=_PROJECT_ROOT / "models" / "guofish2_25.6M_54.8p.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--mcts", action="store_true",
                        help="Use MCTS search instead of raw policy")
    parser.add_argument("--simulations", type=int, default=800,
                        help="Number of MCTS simulations per move (default: 800)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of MCTS worker threads (default: auto)")
    parser.add_argument("--book", action="store_true",
                        help="Use opening book for early moves")
    parser.add_argument("--book-path", type=Path,
                        default=_PROJECT_ROOT / "assets" / "gm2001.bin",
                        help="Path to Polyglot opening book .bin file (default: assets/gm2001.bin)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: {args.checkpoint} not found")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    # Initialize MCTS if requested
    mcts_engine = None
    if args.mcts:
        from core.mcts import ParallelMCTS
        # Let ParallelMCTS auto-tune workers unless explicitly specified
        mcts_engine = ParallelMCTS(model, device, num_workers=args.workers)
        print(f"MCTS enabled: {args.simulations} simulations, {mcts_engine.num_workers} workers, "
              f"batch size {mcts_engine.evaluator.min_batch_size}-{mcts_engine.evaluator.max_batch_size}")

    # Initialize opening book if requested
    book_reader: Optional[chess.polyglot.MemoryMappedReader] = None
    if args.book:
        if args.book_path.exists():
            try:
                book_reader = chess.polyglot.open_reader(str(args.book_path))
                print(f"Opening book loaded: {args.book_path}")
            except Exception as e:
                print(f"Failed to load opening book {args.book_path}: {e}")
                book_reader = None
        else:
            print(f"Opening book not found at {args.book_path}, playing without book")

    # Outer restart loop — any input prompt can raise RestartGame to land back here.
    while True:
        try:
            # Ask the user which side to play
            while True:
                side_input = prompt("Play as white or black? [w/b]: ").lower()
                if side_input in ("w", "white"):
                    human_side = chess.WHITE
                    break
                if side_input in ("b", "black"):
                    human_side = chess.BLACK
                    break
                print("Please enter 'w' or 'b'.")

            board = chess.Board()
            print("\nStarting game. Enter moves in SAN (e.g. e4, Nf3, O-O).")
            print("To inject an engine move, enter two moves: 'e4 e5' (your move, then engine's).")
            print("Type 'undo' to rewind one full move, 'new' to start a new game, 'quit' to stop.\n")

            def mcts_apply(move: chess.Move) -> None:
                """Advance the MCTS tree to match the board. No-op if MCTS not in use."""
                if mcts_engine is not None:
                    mcts_engine.apply_move(move)

            def start_ponder() -> None:
                """Start background MCTS on the predicted user reply, if possible."""
                if mcts_engine is None or board.is_game_over():
                    return
                # ParallelMCTS.ponder_start auto-selects top-1 or top-K
                # branches based on root-visit confidence.
                mcts_engine.ponder_start(board)

            def stop_ponder() -> None:
                if mcts_engine is not None:
                    mcts_engine.ponder_stop()

            def play_engine_move() -> bool:
                """Probe book, else run engine search. Returns False if no legal moves."""
                book_move = probe_opening_book(book_reader, board)
                if book_move is not None:
                    print(f"{RED}Engine plays: {board.san(book_move)}{RESET}  [book]\n")
                    board.push(book_move)
                    mcts_apply(book_move)
                    start_ponder()
                    return True
                move, stats = pick_engine_move(model, board, device, mcts_engine, args.simulations)
                if move is None:
                    print("Engine has no legal moves!")
                    return False
                print(f"{RED}Engine plays: {board.san(move)}{RESET}  [{format_engine_stats(stats)}]\n")
                board.push(move)
                mcts_apply(move)
                start_ponder()
                return True

            def end_game() -> None:
                """Ask whether to play again. Returns via RestartGame if yes, else exits main()."""
                if not ask_play_again():
                    raise SystemExit(0)
                raise RestartGame()

            # If the engine plays white, make its opening move first (with injection option)
            if human_side == chess.BLACK:
                first_move_input = prompt("Engine's first move (press Enter for engine choice): ")
                if first_move_input:
                    try:
                        move = board.parse_san(first_move_input)
                        print(f"{RED}Engine plays: {board.san(move)}{RESET}  [injected]\n")
                        board.push(move)
                        mcts_apply(move)
                        start_ponder()
                    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError) as e:
                        print(f"Could not parse '{first_move_input}' ({type(e).__name__}). Engine will choose.")
                        if not play_engine_move():
                            end_game()
                else:
                    if not play_engine_move():
                        end_game()

            while True:
                raw = prompt(f"{GREEN}Your move: {RESET}")
                # Stop any pondering before mutating board/tree in response to input.
                # Ponder was running in the background while prompt() blocked.
                stop_ponder()
                if raw.lower() in ("quit", "exit"):
                    print("Quitting.")
                    return

                # Handle undo command
                if raw.lower() in ("undo", "back"):
                    if len(board.move_stack) >= 2:
                        # Undo both player and engine moves
                        undone_engine = board.pop()
                        undone_player = board.pop()
                        # Tree has no inverse of apply_move; rebuild from scratch next search.
                        if mcts_engine is not None:
                            mcts_engine.reset()
                        print(f"Undid: {undone_player} (you), {undone_engine} (engine)")
                        print(board)
                        print(f"History: {format_move_history(board)}")
                        print()
                    elif len(board.move_stack) == 1 and human_side == chess.BLACK:
                        # Undo just the engine's first move (playing as black)
                        undone_engine = board.pop()
                        if mcts_engine is not None:
                            mcts_engine.reset()
                        print(f"Undid engine's first move: {undone_engine}")
                        print(board)
                        print(f"History: {format_move_history(board)}")
                        print()
                        # Re-prompt for engine's first move
                        first_move_input = prompt("Engine's first move (press Enter for engine choice): ")
                        if first_move_input:
                            try:
                                move = board.parse_san(first_move_input)
                                print(f"{RED}Engine plays: {board.san(move)}{RESET}  [injected]\n")
                                board.push(move)
                                mcts_apply(move)
                                start_ponder()
                            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError) as e:
                                print(f"Could not parse '{first_move_input}' ({type(e).__name__}). Engine will choose.")
                                if not play_engine_move():
                                    end_game()
                        else:
                            if not play_engine_move():
                                end_game()
                    else:
                        print("Nothing to undo.")
                    continue

                # Split input to check for injected engine move
                parts = raw.split()
                if not parts:
                    continue

                # Parse user's move (first part)
                try:
                    human_move = board.parse_san(parts[0])
                except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError) as e:
                    print(f"Could not parse '{parts[0]}' as a legal move ({type(e).__name__}). Try again.")
                    continue

                board.push(human_move)
                mcts_apply(human_move)

                if board.is_game_over():
                    outcome = board.outcome()
                    print(f"Game over: {outcome.result() if outcome else 'unknown'}")
                    if board.is_checkmate():
                        print("You win by checkmate!")
                    end_game()

                # Check for injected engine move (second part)
                injected_move = None
                if len(parts) >= 2:
                    try:
                        injected_move = board.parse_san(parts[1])
                    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError) as e:
                        print(f"Could not parse injected move '{parts[1]}' ({type(e).__name__}). Engine will play normally.")
                        injected_move = None

                if injected_move is not None:
                    # Use injected move instead of engine search
                    print(f"{RED}Engine plays: {board.san(injected_move)}{RESET}  [injected]\n")
                    board.push(injected_move)
                    mcts_apply(injected_move)
                    start_ponder()
                else:
                    # Normal engine move (book first, then search)
                    if not play_engine_move():
                        end_game()

                if board.is_game_over():
                    outcome = board.outcome()
                    print(f"Game over: {outcome.result() if outcome else 'unknown'}")
                    if board.is_checkmate():
                        print("Engine wins by checkmate!")
                    end_game()
        except RestartGame:
            # Stop any in-flight pondering and drop the old game's tree before
            # starting fresh. RestartGame can be raised mid-prompt so ponder
            # may still be running at this point.
            if mcts_engine is not None:
                mcts_engine.reset()
            print("\n--- Starting new game ---\n")
            continue


if __name__ == "__main__":
    main()
