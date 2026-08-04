"""UCI protocol wrapper for the GuoFish V5 engine -- V5 checkpoints ONLY.

Usage:
    python uci_wrapper_v5.py                                       # auto-detect latest
    python uci_wrapper_v5.py --model models/guofish5_20M/v5_10.9M_best.pt

This wrapper serves the v5 multi-PV student
(training/v5_multiPV/model_v5.ChessTransformerV5) and nothing else. It loads
the model, refuses anything that is not a v5 student, and searches with
core.mctsv4, which is itself specialized to that architecture.

    Legacy v2 checkpoints (guofish2 .. guofish4) are served by the sibling
    playing/uci_wrapper.py, which is unchanged and still routes through
    core.mctsv3. Point that one at those, not this one.

Being v5-exclusive is what lets two things be exact rather than conditional:
inference runs in bf16, the precision v5 was trained in, and 'score cp' is the
true inverse of the value head's own tanh calibration (which travels in the
checkpoint) rather than an arbitrary linear rescale -- see q_to_centipawns for
the saturation ceiling Cutechess adjudication has to clear.

This allows the engine to be used with UCI-compatible GUIs and tournament
managers like Cutechess, Arena, or lichess-bot.

SEARCH OPTIONS. Four tunables are exposed both as UCI options (settable at any
time; they reach the running search without a reload) and as command-line flags,
so a Cutechess sweep can pin them per engine without a GUI:

    CPuctInit          --c-init              base exploration constant
    CPuctBase          --c-base              visit scale of the log term
    FPUTree            --fpu-tree            Q for unvisited moves below root
    FPURoot            --fpu-root            Q for unvisited moves AT the root
    PolicyTemperature  --policy-temperature  prior sharpening, P^(1/T)

All are declared `type string` because UCI spin options are integer-only and
every one of these is a float. See core.mctsv4.SearchParams for the semantics
and the exploration formula.
"""

import argparse
import contextlib
import glob
import io
import math
import os
import random
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Optional

import chess
import chess.polyglot
import chess.syzygy
import torch

# Make the project root importable when this file is run as a script
# (python playing/uci_wrapper_v5.py) rather than as a module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# core.mctsv4 is the only search here. It is v5-exclusive itself and validates
# the encoding/CLS/policy contracts against the model's own ModelConfig at
# construction, so it is both the search and the second half of the
# architecture gate. core.mctsv3 is deliberately NOT imported: routing between
# generations is playing/uci_wrapper.py's job, not this file's.
import core.mctsv4 as mctsv4
# load_model reads the architecture from the checkpoint; is_v5_model is the
# gate applied to whatever comes back. playv5's build_mcts is deliberately
# unused — it exists to route v2 to mctsv3, which this wrapper never does.
from playing.v5.playv5 import load_model, is_v5_model


MODELS_DIR = _PROJECT_ROOT / "models"
OPENING_BOOK_PATH = _PROJECT_ROOT / "assets" / "gm2001.bin"
# Syzygy endgame tablebase directory (WDL .rtbw + DTZ .rtbz files for the
# 3/4/5-piece sets). Used by the "Mode 1" bypass: positions with <= 5 pieces
# skip MCTS entirely and play the tablebase-optimal move. See
# benchmarking/scratch/verify_syzygy.py (setup check) and
# benchmarking/scratch/test_mode1.py (bypass behavior).
SYZYGY_PATH = _PROJECT_ROOT / "assets" / "syzygy"
# Total piece count (both kings included) at or below which we probe the
# tablebase instead of searching. The downloaded set covers up to 5 pieces.
TABLEBASE_MAX_PIECES = 5

# Largest |value| the v5 label pipeline ever produced (data/multiPV/labels.py:
# VALUE_CLAMP / VALUE_MATE = 0.995). Both mates and anything past the +-2000cp
# clip are pinned there, so the value head was never shown a target outside
# this band and the tanh inversion below saturates at it. Duplicated rather
# than imported so the engine stays independent of the data pipeline.
VALUE_CLAMP = 0.995

def err(msg: str):
    """Print to stderr (visible in Cutechess debug log; does not pollute UCI stdout)."""
    print(msg, file=sys.stderr, flush=True)


def log(msg: str):
    """Print to stdout and flush immediately (required for UCI)."""
    print(msg, flush=True)


def q_to_centipawns(q: float, value_scale: float) -> int:
    """Convert an MCTS Q in [-1, 1] to a calibrated centipawn score.

    v5's value target is tanh(cp / value_scale), so the exact inverse is
    cp = value_scale * atanh(q) -- not a linear rescale. `value_scale` is the
    per-checkpoint constant fitted in data/multiPV/fit_value_scale.py (290.6806
    for the shipped corpus) and travels in the checkpoint, so it is read off
    the loaded model rather than hardcoded here. It is required, not optional:
    a v5 checkpoint always carries one (train_v5.save_checkpoint writes it
    unconditionally), and guessing a calibration would silently corrupt every
    adjudication decision downstream.

    SATURATION. atanh diverges at +-1, and the labels stop at +-VALUE_CLAMP
    anyway: mates and everything past the +-2000cp clip were all trained to the
    same 0.995, so Q beyond that carries no magnitude information. Q is clamped
    there, which caps the reported score at value_scale * atanh(0.995) ~= +-870cp
    at the shipped scale. That is the honest ceiling of the calibration -- a
    proven mate reports ~+870, not +30000 -- so Cutechess `-resign score=` must
    sit below it to ever fire.
    """
    q = max(-VALUE_CLAMP, min(VALUE_CLAMP, q))
    return int(round(value_scale * math.atanh(q)))


def find_latest_checkpoint(models_dir: Optional[str] = None) -> Optional[str]:
    """Find the most recent checkpoint file by modification time.

    Recursive and name-agnostic: every generation writes to its own
    subdirectory (models/guofish4/, models/guofish5_10M/, ...) under a naming
    scheme of its own, so matching a fixed list of prefixes at the top level
    found nothing at all. load_model works out the architecture from whatever
    turns up, so there is no reason to filter by name here.

    Only a fallback -- every production launcher (playing/run_guofish.bat, the
    benchmarking scripts) passes --model explicitly.
    """
    if models_dir is None:
        models_dir = str(MODELS_DIR)

    all_checkpoints = glob.glob(os.path.join(models_dir, "**", "*.pt"), recursive=True)

    # Exclude fp16 exports (the fp32 original is alongside) and zip files
    all_checkpoints = [
        f for f in all_checkpoints
        if not f.endswith("_fp16.pt") and not f.endswith(".zip")
    ]

    if not all_checkpoints:
        return None

    # Return most recently modified
    return max(all_checkpoints, key=os.path.getmtime)


class UCIEngine:
    """UCI protocol handler for the GuoFish V5 student."""

    def __init__(self, model_path: Optional[str] = None, num_workers: int = 32,
                 sim_cap: int = 15000, ponder: bool = False,
                 fixed_sims: Optional[int] = None,
                 book_path: Optional[Path] = None,
                 book_seed: Optional[int] = None,
                 tablebase_path: Optional[Path] = None,
                 params: Optional[mctsv4.SearchParams] = None):
        # Optional explicit checkpoint path; if None, auto-detects latest.
        self.model_path: Optional[str] = model_path
        self.num_workers: int = num_workers
        self.sim_cap: int = sim_cap
        # When set, every 'go' uses exactly this many sims, ignoring whatever
        # 'nodes' / time-control args the GUI sends. Used for fixed-budget
        # tournaments (C_PUCT sweeps, ELO calibration) where any per-move
        # variation in sim count would corrupt the comparison.
        self.fixed_sims: Optional[int] = fixed_sims
        # Pondering is off by default. Enabling it during Cutechess ELO runs
        # makes concurrent games fight for the same GPU/CPU budget between
        # turns, hurting Stockfish's fixed-time results.
        self.ponder: bool = ponder
        self.model: Optional[torch.nn.Module] = None
        self.mcts: Optional[mctsv4.ParallelMCTS] = None
        # Search tunables (exploration, FPU, policy temperature). Built here
        # rather than in _initialize_engine because a GUI sends every
        # `setoption` BEFORE the first `isready`, which is what triggers the
        # load -- so this object must exist and be mutable before MCTS does.
        # The SAME instance is handed to ParallelMCTS, so a later setoption
        # reaches the running search without rebuilding anything.
        self.search_params: mctsv4.SearchParams = (
            params if params is not None else mctsv4.SearchParams())
        self.device: Optional[torch.device] = None
        self.board: chess.Board = chess.Board()
        # tanh scale the value head was trained against, read off the loaded
        # checkpoint in _initialize_engine, which refuses to start without one.
        # Drives the 'score cp' inversion. Only None before the model loads.
        self.value_scale: Optional[float] = None

        # Track if we're continuing from the same game (for tree reuse)
        self._last_position_fen: Optional[str] = None
        self._last_moves: list[str] = []

        # Polyglot opening book. Loaded lazily in _initialize_engine so a
        # missing book file fails loudly through the same code path as
        # other init errors. None book_path => book disabled.
        self.book_path: Optional[Path] = book_path
        self.book_reader: Optional[chess.polyglot.MemoryMappedReader] = None
        # Seedable RNG so book-driven tournaments stay reproducible when the
        # user passes --book-seed.
        self.book_rng: random.Random = random.Random(book_seed)

        # Syzygy endgame tablebase. Opened once in _initialize_engine (opening
        # is expensive; probing is sub-millisecond on a warm page cache) and
        # closed in handle_quit. None path => bypass disabled.
        self.tablebase_path: Optional[Path] = tablebase_path
        self.tablebase: Optional[chess.syzygy.Tablebase] = None

    def handle_uci(self):
        """Respond to 'uci' command with engine identification and options.

        All four search tunables are advertised as `type string` rather than
        `type spin`. UCI spin options are integer-only, and every one of these
        is a float -- CPuctInit 2.0, PolicyTemperature 0.8 and FPU -0.2 would
        all be truncated or rejected. Lc0 declares its float options the same
        way, and Cutechess/Arena pass strings through untouched.

        The advertised defaults are this process's CURRENT values, not the
        module defaults, so a GUI attached to an engine started with
        `--fpu -0.2` sees -0.2 as the default and a "reset to default" does not
        silently undo the command line.
        """
        log("id name Guofish2")
        log("id author Guo")
        for name, value in self._uci_option_values().items():
            log(f"option name {name} type string default {value:g}")
        log("uciok")

    # UCI option name -> SearchParams attribute. Lookup is case-insensitive
    # (see handle_setoption); the spec is case-sensitive but GUIs are not
    # consistent about it, and being lenient here costs nothing.
    _OPTION_TO_PARAM = {
        "cpuctinit": "c_init",
        "cpuctbase": "c_base",
        "fpuroot": "fpu_root",
        "fputree": "fpu_tree",
        "policytemperature": "policy_temperature",
    }
    # Display spelling for the same options, in advertisement order.
    _OPTION_NAMES = ("CPuctInit", "CPuctBase", "FPURoot", "FPUTree",
                     "PolicyTemperature")

    def _uci_option_values(self) -> dict[str, float]:
        """Current value of every advertised option, keyed by display name."""
        return {
            name: getattr(self.search_params,
                          self._OPTION_TO_PARAM[name.lower()])
            for name in self._OPTION_NAMES
        }

    def handle_setoption(self, args: list[str]):
        """Handle `setoption name <Name> value <Value>`.

        Applied to the live SearchParams, which ParallelMCTS shares, so a change
        between moves takes effect on the next search with no reload.

        A rejected value (unparseable, or outside what the search can use) is
        logged to stderr and DROPPED, leaving the previous value in place. That
        is deliberate: a tournament engine that silently retunes itself to a
        typo'd value produces results nobody can interpret, and one that exits
        forfeits the game. Unknown option names are ignored per the UCI spec.
        """
        # Parse positionally: an option name may contain spaces, so take
        # everything between 'name' and 'value' as the name and the rest as the
        # value, rather than assuming fixed token positions.
        lowered = [a.lower() for a in args]
        if "name" not in lowered:
            err(f"[setoption] malformed (no 'name'): {' '.join(args)}")
            return
        name_at = lowered.index("name")
        if "value" in lowered:
            value_at = lowered.index("value", name_at)
            name = " ".join(args[name_at + 1:value_at])
            raw = " ".join(args[value_at + 1:])
        else:
            name = " ".join(args[name_at + 1:])
            raw = ""

        attr = self._OPTION_TO_PARAM.get(name.strip().lower())
        if attr is None:
            err(f"[setoption] ignoring unknown option '{name}'")
            return

        try:
            value = float(raw)
        except ValueError:
            err(f"[setoption] {name}: '{raw}' is not a number; keeping "
                f"{getattr(self.search_params, attr):g}")
            return

        previous = getattr(self.search_params, attr)
        if value == previous:
            return

        # Validate on a copy first: SearchParams.validate() checks the whole
        # object, so mutating in place and rolling back on failure would leave a
        # window where a worker thread could read the bad value. `replace`
        # itself validates (SearchParams.__post_init__), which is why the
        # construction is inside the try and not just the explicit call.
        try:
            replace(self.search_params, **{attr: value}).validate()
        except ValueError as e:
            err(f"[setoption] {name}: rejected ({e}); keeping {previous:g}")
            return

        setattr(self.search_params, attr, value)
        err(f"[setoption] {name}: {previous:g} -> {value:g}")

        # Policy temperature is baked into a node's priors at expansion time, so
        # any tree built under the old value is stale in a way the other three
        # options are not (they are read fresh on every node scored). Drop the
        # tree so the new temperature applies uniformly. The transposition cache
        # survives: it holds raw logits, which are temperature-independent.
        if attr == "policy_temperature" and self.mcts is not None:
            self.mcts.reset()
            err("[setoption] PolicyTemperature changed: search tree reset "
                "(priors are baked in at expansion; NN cache kept)")

    def handle_isready(self):
        """Load model and initialize MCTS, then respond 'readyok'."""
        if self.mcts is None:
            self._initialize_engine()
        log("readyok")

    def _initialize_engine(self):
        """Load the model and set up MCTS."""
        # Force CUDA - UCI wrapper is GPU-only.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available - uci_wrapper_v5.py requires a GPU.")
        self.device = torch.device("cuda")

        # Use explicit model path if provided, else auto-detect latest checkpoint
        if self.model_path is not None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
            checkpoint_path = self.model_path
        else:
            checkpoint_path = find_latest_checkpoint()

        err(f"[init] device={self.device} checkpoint={checkpoint_path}")

        if checkpoint_path is None:
            # A v5-only engine with no v5 weights cannot do its job. Refuse to
            # start rather than fall back to random weights: this file exists to
            # play rated games, and a random net loses them silently instead of
            # visibly. Consistent with how the CUDA and missing-path checks
            # above already treat fatal init problems.
            raise FileNotFoundError(
                f"No checkpoint found under {MODELS_DIR}. Pass --model with a v5 "
                "checkpoint (e.g. models/guofish5_20M/v5_10.9M_best.pt)."
            )

        # load_model prints to stdout (corrupts UCI stream). Redirect to stderr.
        # It also reports the architecture it detected and the checkpoint's
        # provenance, so those land in the GUI's debug log.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.model = load_model(Path(checkpoint_path), self.device)
        for line in buf.getvalue().splitlines():
            err(f"[load_model] {line}")

        # === Architecture gate ===
        # load_model happily returns a legacy v2 net; this wrapper does not
        # serve one. Reject it here, before MCTS construction, so the error
        # names the file that does rather than surfacing as mctsv4's (correct
        # but less actionable) rejection two lines later.
        if not is_v5_model(self.model):
            raise ValueError(
                f"{checkpoint_path} is not a v5 checkpoint. uci_wrapper_v5.py "
                "serves only the v5 student; use playing/uci_wrapper.py for "
                "legacy v2 checkpoints (guofish2..guofish4)."
            )

        # Value calibration for 'score cp'. Required, not optional: every v5
        # checkpoint carries the tanh scale it was fitted with, and running
        # without one would mean reporting scores on an unknown scale to the
        # tournament's adjudicator. Report the resulting ceiling so -resign can
        # be set under it.
        value_scale = getattr(self.model, "value_scale", None)
        if value_scale is None:
            raise ValueError(
                f"{checkpoint_path} carries no value_scale, so 'score cp' has no "
                "calibration to invert. Every train_v5 checkpoint writes one; a "
                "checkpoint without it did not come from that pipeline."
            )
        self.value_scale = float(value_scale)
        err(f"[eval] score cp = {self.value_scale:.4f} * atanh(q), "
            f"saturating at +-{q_to_centipawns(1.0, self.value_scale)} cp "
            f"(|q| clamped to {VALUE_CLAMP})")

        # NOTE: torch.compile(mode="reduce-overhead") removed - it hangs on Windows
        # due to missing/flaky Triton support. Eager mode is fast enough on GPU.

        # Initialize parallel MCTS. core.mctsv4 re-validates the model against
        # the v5 encoding/CLS/policy contracts at construction, so the gate
        # above is the friendly error and this is the authoritative one.
        # num_workers is configurable so CuteChess can scale it down when
        # running many concurrent games (otherwise N games × 32 workers
        # thrashes the GIL and cores).
        # search_params is passed by reference, not copied, so every later
        # `setoption` mutates the object this search is already reading.
        self.mcts = mctsv4.ParallelMCTS(
            self.model,
            self.device,
            num_workers=self.num_workers,
            params=self.search_params,
        )
        err(f"[init] MCTS ready via {type(self.mcts).__module__} (v5 student), "
            f"workers={self.num_workers}, sim_cap={self.sim_cap}")
        err(f"[init] search params: {self.search_params.describe()}")

        # Open the Polyglot book once at init. Keep the reader open for the
        # process lifetime; chess.polyglot uses an mmap so the per-lookup
        # cost is just a binary search on the move-key table.
        if self.book_path is not None:
            if not self.book_path.exists():
                err(f"[book] WARN: opening book not found at {self.book_path}; book disabled.")
            else:
                try:
                    self.book_reader = chess.polyglot.open_reader(str(self.book_path))
                    err(f"[book] opened {self.book_path.name}")
                except Exception as e:
                    err(f"[book] WARN: failed to open {self.book_path}: {e}; book disabled.")
                    self.book_reader = None

        # Open the Syzygy tablebase once. Like the book, this is a one-time
        # cost; chess.syzygy mmaps the table files so per-probe overhead is
        # just a lookup into already-mapped (and, after warmup, cached) pages.
        if self.tablebase_path is not None:
            if not self.tablebase_path.is_dir():
                err(f"[syzygy] WARN: tablebase dir not found at {self.tablebase_path}; "
                    "endgame bypass disabled.")
            else:
                try:
                    self.tablebase = chess.syzygy.open_tablebase(str(self.tablebase_path))
                    err(f"[syzygy] opened tablebase at {self.tablebase_path} "
                        f"(bypass for <= {TABLEBASE_MAX_PIECES} pieces)")
                except Exception as e:
                    err(f"[syzygy] WARN: failed to open {self.tablebase_path}: {e}; "
                        "endgame bypass disabled.")
                    self.tablebase = None

        # Hand the same tablebase handle to MCTS for Mode 2 (leaf evaluation:
        # interior leaves with <= 5 pieces get the exact WDL value instead of
        # the neural value head's estimate). Shared read-only across workers;
        # chess.syzygy probing is thread-safe with per-worker board copies.
        self.mcts.tablebase = self.tablebase

    def _probe_tablebase(self) -> Optional[chess.Move]:
        """Return the tablebase-optimal move for self.board, or None on miss.

        Mode 1 bypass helper. Selects, among legal moves:
          1. the best WDL outcome from our perspective (win > cursed-win >
             draw > blessed-loss > loss), then
          2. within that outcome, the move that makes fastest progress toward
             a zeroing move (capture / pawn push that resets the 50-move
             counter): smallest DTZ when winning, largest DTZ when losing.

        python-chess's chess.syzygy.Tablebase has no probe_root, so we do the
        manual root probe described above. probe_dtz/probe_wdl report values
        from the side-to-move's perspective; after we push our move it is the
        opponent to move, so we negate to get our perspective.

        Returns None (caller falls back to MCTS) if the tablebase is closed or
        any required table is missing.
        """
        if self.tablebase is None:
            return None

        # Outcome rank (higher is better for us). Mate is handled separately
        # as strictly better than any tablebase win.
        best_move: Optional[chess.Move] = None
        best_key: Optional[tuple] = None

        try:
            for move in self.board.legal_moves:
                zeroing = self.board.is_zeroing(move)
                self.board.push(move)
                try:
                    if self.board.is_checkmate():
                        # Immediate mate: rank above everything, distance 0.
                        outcome, distance = 3, 0
                    elif self.board.is_stalemate() or self.board.is_insufficient_material():
                        # Drawn terminal position regardless of tablebase.
                        outcome, distance = 0, 0
                    else:
                        wdl_child = self.tablebase.probe_wdl(self.board)
                        dtz_child = self.tablebase.probe_dtz(self.board)
                        # Negate to our perspective.
                        our_wdl = -wdl_child   # 2 win .. -2 loss
                        # Map WDL to a coarse outcome rank (1 = win, 0 = draw,
                        # -1 = loss). Cursed/blessed (|wdl|==1) collapse with
                        # their decisive counterparts for ranking; DTZ breaks
                        # the tie within an outcome.
                        if our_wdl > 0:
                            outcome = 1
                        elif our_wdl < 0:
                            outcome = -1
                        else:
                            outcome = 0

                        if outcome > 0:
                            # Winning: make concrete progress. A zeroing move
                            # that keeps the win resets the counter and is the
                            # best kind of progress, so rank it distance 0.
                            # Otherwise prefer the smallest plies-to-zero
                            # (-dtz_child, positive since the opponent loses).
                            distance = 0 if zeroing else -dtz_child
                        elif outcome < 0:
                            # Losing: stall. Larger dtz_child = opponent needs
                            # more plies to zero us in. Negate so the sort key
                            # (which we minimize) prefers the largest delay.
                            distance = -dtz_child
                        else:
                            distance = 0
                finally:
                    self.board.pop()

                # Maximize outcome, then minimize distance.
                key = (outcome, -distance)
                if best_key is None or key > best_key:
                    best_key = key
                    best_move = move
        except chess.syzygy.MissingTableError as e:
            err(f"[syzygy] WARN: missing table while probing {self.board.fen()}: {e}; "
                "falling back to MCTS.")
            return None
        except Exception as e:
            err(f"[syzygy] WARN: probe failed for {self.board.fen()}: {e}; "
                "falling back to MCTS.")
            return None

        return best_move

    def _probe_book(self) -> Optional[chess.Move]:
        """Return a weighted-random book move for self.board, or None on miss.

        Misses on (a) book closed, (b) position not in book, (c) book entry
        whose UCI move is somehow not legal in the current position.
        """
        if self.book_reader is None:
            return None
        try:
            entry = self.book_reader.weighted_choice(self.board, random=self.book_rng)
        except IndexError:
            return None  # position not in book
        except Exception as e:
            err(f"[book] WARN: lookup failed for {self.board.fen()}: {e}")
            return None
        mv = entry.move
        if mv not in self.board.legal_moves:
            err(f"[book] WARN: book move {mv.uci()} not legal in {self.board.fen()}; skipping book.")
            return None
        return mv

    def handle_ucinewgame(self):
        """Handle 'ucinewgame' command - reset state for a new game."""
        self.board = chess.Board()
        self._last_position_fen = None
        self._last_moves = []
        if self.mcts is not None:
            self.mcts.reset()
            # Drop the transposition cache between games. reset() only clears
            # the tree; without this the cache (full 4096-wide policy tensors,
            # keyed by Zobrist hash) accumulates across every game in a
            # tournament until it hits its cap (~GBs/process). Cross-game
            # transposition reuse is near-worthless anyway since only shared
            # opening positions recur, so clearing here bounds RAM to a single
            # game with no real strength cost.
            self.mcts.clear_cache()

    def handle_position(self, args: list[str]):
        """
        Parse 'position' command and set up board state.

        Formats:
            position startpos
            position startpos moves e2e4 e7e5 ...
            position fen <fen>
            position fen <fen> moves e2e4 e7e5 ...
        """
        # Stop any in-flight pondering before mutating board/tree. Defensive:
        # ParallelMCTS.apply_move/reset also call ponder_stop internally, but
        # stopping up-front avoids racing on self.board.
        if self.mcts is not None:
            self.mcts.ponder_stop()

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
        if self.mcts is None:
            self._initialize_engine()
        assert self.mcts is not None  # narrowed — _initialize_engine always assigns it

        # === Opening book ===
        # Try the Polyglot book first. On a hit, return the book move
        # immediately and skip MCTS. weighted_choice samples proportional to
        # the entry weights (book-author's preference), so popular mainlines
        # are favored but sidelines still appear; this is the standard
        # Polyglot semantics and gives natural opening variety.
        book_move = self._probe_book()
        if book_move is not None:
            log(f"info depth 0 string book {book_move.uci()}")
            log(f"bestmove {book_move.uci()}")
            if self.ponder:
                self.board.push(book_move)
                self.mcts.apply_move(book_move)
                self._last_moves.append(book_move.uci())
                if not self.board.is_game_over():
                    self.mcts.ponder_start(self.board)
            return

        # === Mode 1: Syzygy endgame tablebase bypass ===
        # For positions with <= TABLEBASE_MAX_PIECES pieces (both kings
        # counted) the tablebase plays perfectly, so running MCTS is wasted
        # work and strictly weaker. Probe the tablebase and, on a hit, return
        # the tablebase-optimal move WITHOUT invoking MCTS. The network still
        # loads normally; it just isn't consulted for these positions.
        #
        # On any tablebase miss/error (table not present, position Syzygy treats
        # as drawn-by-rule, etc.) _probe_tablebase returns None and we fall
        # through to MCTS rather than crashing. See _probe_tablebase and the
        # validation scripts in benchmarking/scratch/ (verify_syzygy.py,
        # test_mode1.py). Mode 2 (probing at MCTS leaves) is a separate change.
        if self.tablebase is not None and len(self.board.piece_map()) <= TABLEBASE_MAX_PIECES:
            tb_move = self._probe_tablebase()
            if tb_move is not None:
                log(f"info depth 0 string syzygy {tb_move.uci()}")
                log(f"bestmove {tb_move.uci()}")
                return

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

        # --sims overrides any per-move sim count from the GUI. Tournaments
        # that compare engines on a fixed budget rely on this.
        if self.fixed_sims is not None:
            num_simulations = self.fixed_sims
        else:
            num_simulations = max(1, min(num_simulations, self.sim_cap))

        # Run search
        best_move = self.mcts.search(self.board, num_simulations=num_simulations)

        # Emit evaluation info for Cutechess adjudication.
        # last_best_child_q is in [-1.0, 1.0] from the engine's perspective;
        # q_to_centipawns inverts the value head's own tanh calibration, so the
        # number is real centipawns rather than an arbitrary rescale — see that
        # function for the saturation ceiling adjudication thresholds must clear.
        # `nodes` reports the resolved sim budget so the actual search size is
        # visible in the Cutechess PGN comments — this is how you confirm a
        # `go nodes N` actually propagated (and wasn't clamped by --sim-cap).
        q_value = self.mcts.last_best_child_q
        cp_score = q_to_centipawns(q_value, self.value_scale)
        log(f"info depth 1 nodes {num_simulations} score cp {cp_score}")

        if best_move is not None:
            log(f"bestmove {best_move.uci()}")
        else:
            # No legal moves - should not happen in normal play
            log("bestmove 0000")
            return

        # === Pondering (optional) ===
        # When enabled, commit our move to the tree, predict the opponent's
        # reply, and start background MCTS on that subtree. If the opponent
        # plays the predicted move, the next 'position' command's apply_move
        # promotes a subtree with thousands of ponder-added visits.
        #
        # Off by default: during concurrent Cutechess ELO runs, background
        # pondering across N games all contends for the same GPU/CPU and
        # distorts Stockfish's fixed-time results. Enable with --ponder for
        # single-game or interactive use.
        if self.ponder:
            self.board.push(best_move)
            self.mcts.apply_move(best_move)
            # Keep _last_moves in sync so the next 'position' command's
            # tree-reuse check (prefix match against incoming moves_list)
            # still holds.
            self._last_moves.append(best_move.uci())

            if not self.board.is_game_over():
                # ParallelMCTS.ponder_start auto-selects top-1 or top-K
                # branches based on root-visit confidence.
                self.mcts.ponder_start(self.board)

    def handle_quit(self):
        """Clean shutdown."""
        if self.mcts is not None:
            self.mcts.shutdown()
        if self.book_reader is not None:
            try:
                self.book_reader.close()
            except Exception:
                pass
        if self.tablebase is not None:
            try:
                self.tablebase.close()
            except Exception:
                pass
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
                    self.handle_setoption(args)
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
    parser = argparse.ArgumentParser(
        description="UCI wrapper for the GuoFish V5 student (v5 checkpoints only; "
                    "use playing/uci_wrapper.py for legacy v2 checkpoints)")
    parser.add_argument("--model", "--checkpoint", dest="model", type=str, default=None,
                        help="Path to model checkpoint (default: auto-detect latest). "
                             "--checkpoint is an alias.")
    parser.add_argument("--workers", type=int, default=32,
                        help="MCTS worker threads per engine instance. Lower this when "
                             "running many concurrent games (default: 32)")
    parser.add_argument("--sim-cap", type=int, default=15000,
                        help="Upper bound on 'go nodes N' (default: 15000). NOTE: a "
                             "'go nodes N' larger than this is silently clamped to it, "
                             "so keep this >= the nodes= you pass from Cutechess. "
                             "Ignored when --sims is given.")
    parser.add_argument("--sims", type=int, default=None,
                        help="Force every search to use exactly this many sims, "
                             "ignoring 'go nodes' / time-control args from the GUI. "
                             "Use for fixed-budget tournaments (default: off).")
    parser.add_argument("--c-init", type=float, default=mctsv4.C_PUCT_INIT,
                        help="Base PUCT exploration constant c_init (default: "
                             f"{mctsv4.C_PUCT_INIT}). Exploration is "
                             "c_init + log((N + c_base + 1) / c_base), so this "
                             "is what the old static C_PUCT was at low visit "
                             "counts. Also settable at runtime via "
                             "'setoption name CPuctInit value X'.")
    parser.add_argument("--c-base", type=float, default=mctsv4.C_PUCT_BASE,
                        help="PUCT visit scale c_base (default: "
                             f"{mctsv4.C_PUCT_BASE:g}). Larger => exploration "
                             "grows more slowly with visits. UCI: CPuctBase.")
    parser.add_argument("--c-puct", type=float, default=None,
                        help="DEPRECATED alias for --c-init, kept so existing "
                             "sweep scripts keep running. The static C_PUCT no "
                             "longer exists; this sets c_init only, leaving the "
                             "visit-scaled term active.")
    parser.add_argument("--fpu-tree", type=float, default=mctsv4.FPU_TREE,
                        help="First Play Urgency BELOW the root: the Q assigned "
                             "to an unvisited move during selection (default: "
                             f"{mctsv4.FPU_TREE}). Negative is pessimistic "
                             "(narrower search), positive optimistic (wider). "
                             "Must lie in [-1, 1]. UCI: FPUTree.")
    parser.add_argument("--fpu-root", type=float, default=mctsv4.FPU_ROOT,
                        help="First Play Urgency AT the root only (default: "
                             f"{mctsv4.FPU_ROOT}). Normally left at 0: every "
                             "root move gets visited early regardless, and the "
                             "root's visit distribution is the engine's answer, "
                             "so biasing it distorts the output. Exposed so "
                             "that assumption can be A/B'd. UCI: FPURoot.")
    parser.add_argument("--policy-temperature", type=float,
                        default=mctsv4.POLICY_TEMPERATURE,
                        help="Inference-time policy sharpening exponent T: "
                             "priors become P^(1/T) over the legal moves "
                             f"(default: {mctsv4.POLICY_TEMPERATURE} = off). "
                             "T < 1 sharpens onto the network's top moves, "
                             "T > 1 flattens. UCI: PolicyTemperature.")
    parser.add_argument("--virtual-loss", type=float, default=None,
                        help="Override the in-flight virtual-loss penalty (default: "
                             f"{mctsv4.VIRTUAL_LOSS}). Mutates "
                             "core.mctsv4.VIRTUAL_LOSS at startup. Used for "
                             "hyperparameter A/B runs.")
    parser.add_argument("--max-tree-depth", type=int, default=None,
                        help="Override the per-simulation depth cap (default: "
                             f"{mctsv4.MAX_TREE_DEPTH}). Mutates "
                             "core.mctsv4.MAX_TREE_DEPTH at startup. Used for "
                             "hyperparameter A/B runs.")
    parser.add_argument("--ponder", action="store_true",
                        help="Ponder on the predicted opponent reply between "
                             "turns. Off by default: concurrent Cutechess ELO "
                             "runs should leave this off so background work "
                             "from one game doesn't steal compute from another.")
    parser.add_argument("--book", type=str, default=str(OPENING_BOOK_PATH),
                        help=f"Polyglot opening book path (default: {OPENING_BOOK_PATH}). "
                             "On a book hit the engine returns the book move and skips MCTS.")
    parser.add_argument("--no-book", dest="use_book", action="store_false",
                        help="Disable the opening book; play every move with MCTS.")
    parser.add_argument("--book-seed", type=int, default=None,
                        help="Seed for the book's weighted-random move selection "
                             "(default: nondeterministic). Pin this for reproducible "
                             "book lines in tournaments.")
    parser.add_argument("--syzygy", type=str, default=str(SYZYGY_PATH),
                        help=f"Syzygy tablebase directory (default: {SYZYGY_PATH}). "
                             f"Positions with <= {TABLEBASE_MAX_PIECES} pieces skip "
                             "MCTS and play the tablebase-optimal move.")
    parser.add_argument("--no-syzygy", dest="use_syzygy", action="store_false",
                        help="Disable the endgame tablebase bypass; play every "
                             "position with MCTS.")
    parser.set_defaults(use_book=True, use_syzygy=True)
    args = parser.parse_args()

    # VIRTUAL_LOSS and MAX_TREE_DEPTH are still live module globals in
    # core.mctsv4 (read at call time by code that holds no SearchParams), so
    # rebinding them before any engine exists is sufficient. The exploration /
    # FPU / temperature settings are NOT globals any more -- they travel in the
    # SearchParams built below.
    for name, value in (("VIRTUAL_LOSS", args.virtual_loss),
                        ("MAX_TREE_DEPTH", args.max_tree_depth)):
        if value is None:
            continue
        setattr(mctsv4, name, value)
        err(f"[init] {name} overridden to {value} in core.mctsv4")

    c_init = args.c_init
    if args.c_puct is not None:
        c_init = args.c_puct
        err(f"[init] --c-puct is deprecated; treating {args.c_puct} as --c-init "
            f"(visit-scaled term stays on, c_base={args.c_base:g})")

    try:
        params = mctsv4.SearchParams(
            c_init=c_init,
            c_base=args.c_base,
            fpu_root=args.fpu_root,
            fpu_tree=args.fpu_tree,
            policy_temperature=args.policy_temperature,
        )
    except ValueError as e:
        # Fail at startup rather than mid-tournament: a bad flag here would
        # otherwise surface as a losing engine with a confusing log.
        parser.error(str(e))

    book_path = Path(args.book) if args.use_book else None
    tablebase_path = Path(args.syzygy) if args.use_syzygy else None
    engine = UCIEngine(model_path=args.model, num_workers=args.workers,
                       sim_cap=args.sim_cap, ponder=args.ponder,
                       fixed_sims=args.sims,
                       book_path=book_path, book_seed=args.book_seed,
                       tablebase_path=tablebase_path,
                       params=params)
    engine.run()


if __name__ == "__main__":
    main()
