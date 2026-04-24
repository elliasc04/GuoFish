"""UCI (Universal Chess Interface) protocol wrapper for Guofish2 chess engine.

Usage:
    python uci_wrapper.py                                      # auto-detect latest checkpoint
    python uci_wrapper.py --model models/guofish2_25.6M_54.8p.pt

This allows the engine to be used with UCI-compatible GUIs and tournament
managers like Cutechess, Arena, or lichess-bot.
"""

import argparse
import contextlib
import glob
import io
import os
import sys
import traceback
from typing import Optional

import chess
import torch

from mcts import ParallelMCTS
from play import load_model, ChessTransformerV2


def err(msg: str):
    """Print to stderr (visible in Cutechess debug log; does not pollute UCI stdout)."""
    print(msg, file=sys.stderr, flush=True)


def log(msg: str):
    """Print to stdout and flush immediately (required for UCI)."""
    print(msg, flush=True)


def find_latest_checkpoint(models_dir: str = "models") -> Optional[str]:
    """Find the most recent checkpoint file by modification time."""
    patterns = [
        os.path.join(models_dir, "guofish2_*.pt"),
        os.path.join(models_dir, "guofish_*.pt"),
        os.path.join(models_dir, "chess_transformer_*.pt"),
    ]

    all_checkpoints = []
    for pattern in patterns:
        all_checkpoints.extend(glob.glob(pattern))

    # Exclude fp16 and zip files
    all_checkpoints = [
        f for f in all_checkpoints
        if not f.endswith("_fp16.pt") and not f.endswith(".zip")
    ]

    if not all_checkpoints:
        return None

    # Return most recently modified
    return max(all_checkpoints, key=os.path.getmtime)


class UCIEngine:
    """UCI protocol handler for Guofish2."""

    def __init__(self, model_path: Optional[str] = None, num_workers: int = 32,
                 sim_cap: int = 10000):
        # Optional explicit checkpoint path; if None, auto-detects latest.
        self.model_path: Optional[str] = model_path
        self.num_workers: int = num_workers
        self.sim_cap: int = sim_cap
        self.model: Optional[torch.nn.Module] = None
        self.mcts: Optional[ParallelMCTS] = None
        self.device: Optional[torch.device] = None
        self.board: chess.Board = chess.Board()

        # Track if we're continuing from the same game (for tree reuse)
        self._last_position_fen: Optional[str] = None
        self._last_moves: list[str] = []

    def handle_uci(self):
        """Respond to 'uci' command with engine identification."""
        log("id name Guofish2")
        log("id author Guo")
        log("uciok")

    def handle_isready(self):
        """Load model and initialize MCTS, then respond 'readyok'."""
        if self.mcts is None:
            self._initialize_engine()
        log("readyok")

    def _initialize_engine(self):
        """Load the model and set up MCTS."""
        # Force CUDA - UCI wrapper is GPU-only.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available - uci_wrapper.py requires a GPU.")
        self.device = torch.device("cuda")

        # Use explicit model path if provided, else auto-detect latest checkpoint
        if self.model_path is not None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
            checkpoint_path = self.model_path
        else:
            checkpoint_path = find_latest_checkpoint()

        err(f"[init] device={self.device} checkpoint={checkpoint_path}")

        # load_model prints to stdout (corrupts UCI stream). Redirect to stderr.
        buf = io.StringIO()
        if checkpoint_path is None:
            # Fall back to random weights if no checkpoint found
            self.model = ChessTransformerV2().to(self.device)
            self.model.eval()
        else:
            from pathlib import Path
            with contextlib.redirect_stdout(buf):
                self.model = load_model(Path(checkpoint_path), self.device)
        for line in buf.getvalue().splitlines():
            err(f"[load_model] {line}")

        # NOTE: torch.compile(mode="reduce-overhead") removed - it hangs on Windows
        # due to missing/flaky Triton support. Eager mode is fast enough on GPU.

        # Initialize parallel MCTS. num_workers is configurable so CuteChess can
        # scale it down when running many concurrent games (otherwise N games ×
        # 32 workers thrashes the GIL and cores).
        self.mcts = ParallelMCTS(
            model=self.model,
            device=self.device,
            num_workers=self.num_workers,
        )
        err(f"[init] MCTS ready (workers={self.num_workers}, sim_cap={self.sim_cap})")

    def handle_ucinewgame(self):
        """Handle 'ucinewgame' command - reset state for a new game."""
        self.board = chess.Board()
        self._last_position_fen = None
        self._last_moves = []
        if self.mcts is not None:
            self.mcts.reset()

    def handle_position(self, args: list[str]):
        """
        Parse 'position' command and set up board state.

        Formats:
            position startpos
            position startpos moves e2e4 e7e5 ...
            position fen <fen>
            position fen <fen> moves e2e4 e7e5 ...
        """
        if not args:
            return

        moves_start_idx = -1

        if args[0] == "startpos":
            base_fen = chess.STARTING_FEN
            moves_start_idx = 1
            if len(args) > 1 and args[1] == "moves":
                moves_start_idx = 2
        elif args[0] == "fen":
            # FEN is 6 space-separated fields
            fen_parts = []
            idx = 1
            while idx < len(args) and args[idx] != "moves":
                fen_parts.append(args[idx])
                idx += 1
            base_fen = " ".join(fen_parts)
            moves_start_idx = idx
            if idx < len(args) and args[idx] == "moves":
                moves_start_idx = idx + 1
        else:
            return

        # Extract moves list
        moves_list = args[moves_start_idx:] if moves_start_idx < len(args) else []

        # Check if we can use tree reuse (same base position, moves are an extension)
        can_reuse_tree = (
            self.mcts is not None
            and self._last_position_fen == base_fen
            and len(moves_list) >= len(self._last_moves)
            and moves_list[:len(self._last_moves)] == self._last_moves
        )

        if can_reuse_tree:
            # Apply only the new moves to existing board/tree
            new_moves = moves_list[len(self._last_moves):]
            for move_uci in new_moves:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                        self.mcts.apply_move(move)
                    else:
                        # Illegal move - reset everything
                        can_reuse_tree = False
                        break
                except ValueError:
                    can_reuse_tree = False
                    break

        if not can_reuse_tree:
            # Full reset - set up board from scratch
            try:
                self.board = chess.Board(base_fen)
            except ValueError:
                self.board = chess.Board()
                return

            if self.mcts is not None:
                self.mcts.reset()

            # Apply all moves
            for move_uci in moves_list:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                        if self.mcts is not None:
                            self.mcts.apply_move(move)
                    else:
                        break
                except ValueError:
                    break

        # Remember state for next position command
        self._last_position_fen = base_fen
        self._last_moves = moves_list.copy()

    def handle_go(self, args: list[str]):
        """
        Parse 'go' command and execute search.

        Supported parameters:
            nodes <n>  - Search for n simulations
            wtime <ms> - White time remaining (fallback to default sims)
            btime <ms> - Black time remaining (fallback to default sims)
            movetime <ms> - Time for this move
            infinite - Search until 'stop' (not implemented, uses default)
        """
        # Opening book disabled - uci_wrapper.py is eval-only, openings come from Cutechess.
        if self.mcts is None:
            self._initialize_engine()

        # Parse arguments
        num_simulations = 5000  # Default fallback

        i = 0
        while i < len(args):
            arg = args[i]

            if arg == "nodes" and i + 1 < len(args):
                try:
                    num_simulations = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
            elif arg in ("wtime", "btime", "winc", "binc", "movestogo", "movetime", "depth"):
                # Time control parameters - skip value, use default sims
                i += 2
            elif arg == "infinite":
                # Infinite search - just use default for now
                i += 1
            else:
                i += 1

        # Clamp simulations to reasonable range
        num_simulations = max(1, min(num_simulations, self.sim_cap))

        # Run search
        best_move = self.mcts.search(self.board, num_simulations=num_simulations)

        # Emit evaluation info for Cutechess adjudication.
        # last_best_child_q is in [-1.0, 1.0] from engine's perspective; scale to centipawns.
        q_value = self.mcts.last_best_child_q
        cp_score = int(q_value * 1000)
        log(f"info depth 1 score cp {cp_score}")

        if best_move is not None:
            log(f"bestmove {best_move.uci()}")
        else:
            # No legal moves - should not happen in normal play
            log("bestmove 0000")

    def handle_quit(self):
        """Clean shutdown."""
        if self.mcts is not None:
            self.mcts.shutdown()
        sys.exit(0)

    def run(self):
        """Main UCI loop - read commands from stdin and dispatch."""
        while True:
            try:
                line = input().strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                break

            if not line:
                continue

            parts = line.split()
            command = parts[0].lower()
            args = parts[1:]

            try:
                if command == "uci":
                    self.handle_uci()
                elif command == "isready":
                    self.handle_isready()
                elif command == "ucinewgame":
                    self.handle_ucinewgame()
                elif command == "position":
                    self.handle_position(args)
                elif command == "go":
                    self.handle_go(args)
                elif command == "quit":
                    self.handle_quit()
                elif command == "stop":
                    # Stop is typically used during pondering - just ignore for now
                    pass
                elif command == "setoption":
                    # Options not implemented yet - silently ignore
                    pass
                # Unknown commands are silently ignored (UCI spec compliant)
            except Exception as e:
                # Log errors to stderr (visible in Cutechess debug output).
                # Still emit a fallback bestmove for 'go' so Cutechess doesn't hang.
                err(f"[error] command={command} exception={type(e).__name__}: {e}")
                err(traceback.format_exc())
                if command == "go":
                    # Pick any legal move so the game can continue / be adjudicated
                    fallback = next(iter(self.board.legal_moves), None)
                    if fallback is not None:
                        log(f"bestmove {fallback.uci()}")
                    else:
                        log("bestmove 0000")


def main():
    parser = argparse.ArgumentParser(description="UCI wrapper for Guofish2")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint (default: auto-detect latest)")
    parser.add_argument("--workers", type=int, default=32,
                        help="MCTS worker threads per engine instance. Lower this when "
                             "running many concurrent games (default: 32)")
    parser.add_argument("--sim-cap", type=int, default=10000,
                        help="Upper bound on 'go nodes N' (default: 10000)")
    args = parser.parse_args()

    engine = UCIEngine(model_path=args.model, num_workers=args.workers,
                       sim_cap=args.sim_cap)
    engine.run()


if __name__ == "__main__":
    main()
