"""Interactive human play against the v6 engine — playv5's loop on the C++ core.

    python -m playing.v6.playv6_interactive --sims 10000
    python -m playing.v6.playv6_interactive --pgn game.pgn --pgn-ply 24 --stats
    python -m playing.v6.playv6_interactive models/guofish4/guofish4_25.6M_policy_final.pt

WHAT THIS FILE IS FOR
=====================
`playv6.py` is a headless engine plus a one-shot analysis CLI: it analyses a
position and exits, or self-plays N plies and exits. `uci_wrapper_v6.py` is the
protocol adapter a GUI drives. Neither lets a human sit down and play a game at
a terminal, which `playing/v5/playv5.py` did and which is worth keeping.

This is that loop, and ONLY that loop. It owns no engine defaults, no search
parameters and no configuration of its own: `add_config_arguments` is imported
from `playv6` so the two entry points cannot disagree about a default, which is
the same discipline `uci_wrapper_v6.py` follows and the reason C11 exists. Every
flag `playv6.py` accepts is accepted here and means the same thing.

WHAT IT MIRRORS FROM playv5
===========================
The command surface is deliberately identical, so muscle memory carries over:

  * moves in SAN (`e4`, `Nf3`, `O-O`)
  * TWO moves on one line — `e4 e5` — plays your move and then INJECTS the
    second as the engine's reply instead of searching
  * a pasted PGN at the start prompt to begin from a position
  * `--pgn` / `--pgn-ply` walk-back: replay a file's mainline, then let the
    engine move at the loaded position
  * `undo` / `back`, `new` / `restart`, `quit` / `exit`, and a play-again prompt
  * `--stats` for the root search distribution and the principal variation

WHAT IS DIFFERENT, AND WHY
==========================
1. NO PONDERING. playv5 searched in the background while the human thought, via
   `mcts_engine.ponder_start/ponder_stop`. The v6 `Engine` exposes no such pair —
   pondering lives in `uci_wrapper_v6.py`, which owns the threads and the
   `ponderhit` conversion that make it answerable. Reproducing it here would mean
   running a search thread while the main thread blocks in `input()`, which is
   the exact configuration `playv6.preimport_torch` documents as a Windows
   deadlock hazard. `--ponder` is accepted (it is an `EngineConfig` field) and
   has no effect in this frontend; a game plays identically with or without it.

2. THE TREE SURVIVES YOUR MOVE. `Engine.set_position` extends the existing tree
   whenever the move list it is handed is the previous one plus new moves, so
   both your move and the engine's reply are applied incrementally and the search
   inherits the visits underneath them. `undo` shortens the list, which is not an
   extension, so the tree is rebuilt from scratch — correct, and the same
   fallback playv5 took by calling `mcts_engine.reset()`.

3. DELIVERED, NOT REQUESTED, SIMULATIONS ARE REPORTED. Because of the point
   above, a move that asked for 10,000 sims on a reused root may do far fewer.
   The bracket after each engine move shows what the core actually backed up
   (`SearchOutcome.delivered`), never the budget. See playv6's module docstring
   for the defect that accounting exists to close.

THE BOOK WILL ANSWER YOUR FIRST FEW MOVES
=========================================
Both the opening book and Syzygy default ON, as they do everywhere in v6. A book
or tablebase hit BYPASSES MCTS entirely, so those moves come back instantly with
no evaluation, no PV and zero delivered simulations. That is not the engine being
fast — it is the engine not having searched. The move is tagged `[book]` or
`[tablebase]`, `--stats` says so instead of printing an empty table, and the
end-of-game summary counts the three sources separately. Pass `--no-book
--no-syzygy` to play every move with the search.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chess          # noqa: E402
import chess.pgn      # noqa: E402

import guofish_core   # noqa: E402

from playing.v6.playv6 import (  # noqa: E402
    ConfigError,
    DEFAULT_MODEL,
    Engine,
    SearchOutcome,
    add_config_arguments,
    aggregate_sims_per_s,
    apply_switch_interval,
    config_from_args,
    err,
    preimport_torch,
)

# Set to "" by `--no-color` and whenever stdout is not a terminal, so a piped
# transcript does not carry escape sequences.
GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"

_RESTART_WORDS = {"new", "new game", "restart", "again", "play again"}
_QUIT_WORDS = {"quit", "exit"}
_UNDO_WORDS = {"undo", "back"}


class RestartGame(Exception):
    """Raised from any input prompt to abort the current game and start a new one."""


class QuitSession(Exception):
    """Raised from any input prompt to leave the program by the summary path.

    NOT `SystemExit`. The end-of-session summary and `Engine.close()` both have
    to run, and a bare exit from inside a nested prompt would skip the first and
    leave the second to the interpreter — which frees the CUDA graphs and the
    Fathom tables in whatever order the collector happens to pick.
    """


def prompt(message: str) -> str:
    """`input()` that turns the control words into exceptions.

    EOF (Ctrl-D, or a closed pipe) and Ctrl-C are treated as `quit` rather than
    as errors: both mean the human is done, and both otherwise surface as a
    traceback over a half-finished game.
    """
    try:
        value = input(message).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise QuitSession() from None
    lowered = value.lower()
    if lowered in _RESTART_WORDS:
        raise RestartGame()
    if lowered in _QUIT_WORDS:
        raise QuitSession()
    return value


def ask_play_again() -> bool:
    while True:
        try:
            ans = input("Play again? [y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", "quit", "exit"):
            return False
        print("Please enter 'y' or 'n'.")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _numbered(entries: list[tuple[int, bool, str]]) -> str:
    """PGN-style numbering from (fullmove_number, white_to_move, san) triples.

    Built from the triples rather than from a ply index because this frontend can
    start from an arbitrary FEN (`--fen`, or a pasted PGN), where "ply 0 is
    White's first move" is simply false. Each entry carries the number and the
    side it was actually played on, so a line that opens on Black renders as
    `14...Nf6` without the caller having to work out the parity.
    """
    parts: list[str] = []
    i = 0
    while i < len(entries):
        number, is_white, san = entries[i]
        if not is_white:
            parts.append(f"{number}...{san}")
            i += 1
            continue
        if i + 1 < len(entries) and not entries[i + 1][1]:
            parts.append(f"{number}.{san} {entries[i + 1][2]}")
            i += 2
        else:
            parts.append(f"{number}.{san}")
            i += 1
    return " ".join(parts)


def format_move_history(board: chess.Board, base_fen: str, last_n: int = 8) -> str:
    """The last `last_n` plies of the game, in SAN.

    Replayed from `base_fen` rather than from the standard start, because
    `board.move_stack` holds only what was pushed since the root this session
    began at and SAN for a move is only defined against the position before it.
    """
    if not board.move_stack:
        return "(start of game)"
    replay = chess.Board(base_fen)
    entries: list[tuple[int, bool, str]] = []
    for move in board.move_stack:
        entries.append((replay.fullmove_number, replay.turn == chess.WHITE,
                        replay.san(move)))
        replay.push(move)
    return _numbered(entries[-last_n:])


def white_relative(value: float, white_to_move: bool) -> float:
    """A mover-relative score flipped to the absolute, White-positive convention.

    THE ENGINE'S NUMBERS ARE MOVER-RELATIVE AND THIS SCREEN'S ARE NOT.
    `SearchOutcome.q` is the best child's backed-up value, and the backup negates
    on the way up, so it reads from the perspective of whoever is to move at the
    root — the engine, on the move this frontend is about to print. Displayed
    raw, the engine reports `+0.42` when it is winning as White and `+0.42` when
    it is winning as Black, which is not a convention a human reading a board
    can use.

    Every display in this file is therefore flipped to the board convention: `+`
    means White stands better, `-` means Black does, whichever side the engine
    happens to have. `playv6` itself is NOT changed and neither is
    `uci_wrapper_v6.py` — UCI's `info score cp` is REQUIRED to be relative to
    the side to move, and a GUI does this same flip on the way to its own eval
    bar. The mover-relative number is the protocol's; the White-relative one is
    the reader's.

    `-0.0` is normalised to `0.0` so a dead-level position does not print as
    `-0.00`.
    """
    flipped = value if white_to_move else -value
    return flipped if flipped else 0.0


def format_pawns(score_cp: int) -> str:
    """Centipawns as a Stockfish-style pawn score: `+0.51`, `-1.24`, `+0.00`.

    A pure display change — `score_cp` is already the calibrated number,
    `value_scale * atanh(q)` from `playv6.q_to_centipawns`, and this only moves
    the decimal point. THE CALLER IS RESPONSIBLE FOR THE SIGN: pass a
    White-relative value (see `white_relative`) and this prints the board
    convention; pass the engine's raw `score_cp` and it prints the mover's.

    IT SATURATES AT +-8.70, and that ceiling is real rather than a formatting
    artefact. `q_to_centipawns` clamps |q| to VALUE_CLAMP = 0.995 because atanh
    diverges at 1 and the v5 label pipeline trained everything past +-2000cp to
    that same 0.995 — so a completely won position and a forced mate both read
    as +8.70 on the shipped 290.6806 scale. A Stockfish `+15.3` has no
    counterpart here; treat anything at the rail as "winning", not as a
    magnitude.
    """
    return f"{score_cp / 100:+.2f}"


def format_engine_stats(outcome: SearchOutcome, white_to_move: bool) -> str:
    """The bracket after `Engine plays:`.

    A BYPASSED MOVE GETS A DIFFERENT LINE, for the same reason
    `SearchOutcome.telemetry` gives it one: printing `sims: 0 | 0 sims/s` reads
    as a search that did nothing rather than as a search that did not happen, and
    the eval would be a 0.00 nobody computed.
    """
    if outcome.bypassed:
        return f"{outcome.source} | time: {outcome.wall_s * 1000:.0f}ms"
    # BOTH FLIPPED TO THE BOARD CONVENTION, and both from the same flag, so the
    # two can never disagree about which side the sign belongs to. `score_cp` is
    # an int and negates exactly; `q` goes through `white_relative` for the -0.0
    # normalisation.
    parts = [
        # Pawns first, because it is the number a human reads. Q stays beside it
        # rather than being replaced by it: Q is the engine's native unit and the
        # `Q(white)` column of the --stats table is in it, so dropping it here
        # would leave that table with nothing to be read against.
        f"score: {format_pawns(outcome.score_cp if white_to_move else -outcome.score_cp)}",
        f"q: {white_relative(outcome.q, white_to_move):+.3f}",
        f"sims: {outcome.delivered}",
        f"{outcome.sims_per_s:,.0f} sims/s",
        f"depth: {outcome.depth}/{outcome.max_depth}",
        f"time: {outcome.wall_s * 1000:.0f}ms",
    ]
    if outcome.inherited:
        parts.append(f"reused: {outcome.inherited}")
    return " | ".join(parts)


def root_distribution(engine: Engine, board: chess.Board) -> list[dict]:
    """The root's children: visits, prior and backed-up Q, most-visited first.

    `dump_tree_arrays(1)` is the VISITED subtree, which is ~1 node per simulation
    rather than the ~30 per expansion the whole arena holds — the same choice
    `playv6._principal_variation` makes and for the same reason. The dump is DFS
    preorder with an ABSOLUTE depth column, so `depth == 1` selects the root's
    children exactly, wherever they fall in the traversal.

    Q is `value_sum / visits`, then flipped to the board convention. The backup
    negates on the way up, so a child's stored value arrives in the ROOT MOVER's
    perspective — every child on the same footing, which is what makes one
    negation for the whole table correct. playv5 displayed this raw and called
    the column `Q(stm)`; here it is `Q(white)`, so a losing position reads
    negative whichever colour the engine has. See `white_relative`.
    """
    arrays = engine.search.dump_tree_arrays(1)
    depth = arrays["depth"]
    if depth.size == 0:
        return []
    visits, value_sum = arrays["visits"], arrays["value_sum"]
    prior, packed = arrays["prior"], arrays["move"]
    white_to_move = board.turn == chess.WHITE

    rows: list[dict] = []
    for i in range(depth.size):
        if int(depth[i]) != 1:
            continue
        count = int(visits[i])
        uci = guofish_core.move_to_uci(int(packed[i]))
        try:
            san = board.san(chess.Move.from_uci(uci))
        except (ValueError, AssertionError):
            # A move the tree holds but this board does not accept means the two
            # have drifted apart; showing the raw UCI is more useful than
            # dropping the row and pretending the search was narrower.
            san = uci
        rows.append({"uci": uci, "san": san, "visits": count,
                     "prior": float(prior[i]),
                     "q": white_relative(
                         float(value_sum[i]) / count if count else 0.0,
                         white_to_move)})
    rows.sort(key=lambda r: r["visits"], reverse=True)
    return rows


def format_root_distribution(rows: list[dict], played: Optional[chess.Move],
                             top_n: int = 16) -> str:
    if not rows:
        return "(no search distribution - book/tablebase move, MCTS did not run)"
    total = sum(r["visits"] for r in rows) or 1
    played_uci = played.uci() if played is not None else None
    lines = [f"Search distribution: {total} root visits across {len(rows)} moves",
             f"  {'move':<8}{'visits':>9}{'share':>8}{'prior':>8}{'Q(white)':>10}"]
    for row in rows[:top_n]:
        mark = "  <-- played" if row["uci"] == played_uci else ""
        lines.append(f"  {row['san']:<8}{row['visits']:>9}"
                     f"{row['visits'] / total * 100:>7.1f}%"
                     f"{row['prior'] * 100:>7.1f}%{row['q']:>+10.3f}{mark}")
    rest = rows[top_n:]
    if rest:
        spent = sum(r["visits"] for r in rest)
        lines.append(f"  {('(+%d more)' % len(rest)):<8}{spent:>9}"
                     f"{spent / total * 100:>7.1f}%")
    return "\n".join(lines)


def format_pv(board: chess.Board, outcome: SearchOutcome) -> str:
    """The engine's principal variation in SAN, from the position before it moved.

    The walk stops at the first move that is not legal in the line so far rather
    than raising. A PV is a most-visited descent through a tree that reuse can
    leave holding stale children, so an illegal continuation is a real
    possibility and truncating is the honest response to one.
    """
    if outcome.bypassed or not outcome.pv:
        return "(no PV - book/tablebase move, MCTS did not run)"
    replay = board.copy()
    entries: list[tuple[int, bool, str]] = []
    for uci in outcome.pv:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in replay.legal_moves:
            break
        entries.append((replay.fullmove_number, replay.turn == chess.WHITE,
                        replay.san(move)))
        replay.push(move)
    if not entries:
        return "(no PV)"
    return f"Principal variation ({len(entries)} plies): {_numbered(entries)}"


# ---------------------------------------------------------------------------
# PGN input
# ---------------------------------------------------------------------------


def load_pgn_mainline(pgn_path: Path, max_ply: Optional[int] = None) -> list[chess.Move]:
    """The first game's mainline from a PGN file, optionally truncated.

    `max_ply` is `--pgn-ply`: set it so the side you want the engine to analyse
    is the one to move at the final position.
    """
    with open(pgn_path, encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise ValueError(f"no game found in {pgn_path}")
    moves = list(game.mainline_moves())
    if max_ply is not None:
        moves = moves[:max_ply]
    return moves


def apply_pgn_text(board: chess.Board, text: str) -> int:
    """Push a pasted PGN's mainline onto `board`. Returns the ply count.

    EVERY MOVE IS CHECKED FOR LEGALITY before it is pushed, and the first one
    that fails raises. python-chess parses a mainline against the standard start
    position, so a paste combined with `--fen` can silently describe a different
    game; failing loudly and letting the caller restore the board beats playing
    on from a position neither side asked for.
    """
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None:
        raise ValueError("no game found in the pasted text")
    plies = 0
    for move in game.mainline_moves():
        if move not in board.legal_moves:
            raise ValueError(f"{move.uci()} is not legal at ply {plies + 1}")
        board.push(move)
        plies += 1
    if plies == 0:
        raise ValueError("the pasted text contained no moves")
    return plies


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


class Session:
    """One process's worth of play: the engine, the board, and the loop.

    The board is this class's source of truth and the tree is derived from it —
    `Engine.set_position` is handed `base_fen` plus `[m.uci() for m in
    board.move_stack]` before every search, so there is no second move list that
    can drift out of step with what the human sees. Growing that list extends the
    tree; shortening it (an `undo`) rebuilds, which `set_position` decides for
    itself.
    """

    def __init__(self, engine: Engine, args: argparse.Namespace, budget: int):
        self.engine = engine
        self.args = args
        self.budget = budget
        self.base_fen = args.fen or chess.STARTING_FEN
        self.board = chess.Board(self.base_fen)
        self.human_side = chess.WHITE
        self.outcomes: list[SearchOutcome] = []

    # --- engine ---------------------------------------------------------

    def engine_move(self) -> bool:
        """Search (or bypass) and play. False when there is nothing to play.

        The bypass is NOT probed here. `Engine.search_move` does it internally,
        ahead of the slice loop, and returns an outcome whose `source` says which
        lookup answered — so this frontend cannot disagree with the engine about
        whether a move was searched, which is the whole point of that field.
        """
        if self.board.is_game_over():
            return False
        self.engine.set_position(self.base_fen,
                                 [m.uci() for m in self.board.move_stack])
        outcome = self.engine.search_move(budget=self.budget, nominal=self.budget,
                                          budget_source="nodes")
        if outcome.best_move is None:
            print("Engine has no move to play.")
            return False
        try:
            move = chess.Move.from_uci(outcome.best_move)
        except ValueError:
            print(f"Engine returned an unparseable move {outcome.best_move!r}.")
            return False
        if move not in self.board.legal_moves:
            print(f"Engine returned an illegal move {outcome.best_move} in "
                  f"{self.board.fen()}.")
            return False

        # SAN and the root dump must both be taken BEFORE the push: SAN is
        # defined against the position the move is made from, and the tree is
        # still rooted there until the next `set_position`.
        san = self.board.san(move)
        # Read BEFORE the push: `board.turn` is the side the engine is moving
        # for, which is the perspective every score in `outcome` is expressed
        # in. After the push it is the opponent's, and every sign would invert.
        white_to_move = self.board.turn == chess.WHITE
        # NOT dumped for a bypassed move. The tree is still rooted wherever the
        # last search left it, so a dump here would render the PREVIOUS
        # position's children against this board — a table that looks like a
        # search and describes one that never ran.
        want_stats = self.args.stats and not outcome.bypassed
        rows = root_distribution(self.engine, self.board) if want_stats else []
        pv = format_pv(self.board, outcome) if self.args.stats else ""

        print(f"{RED}Engine plays: {san}{RESET}  "
              f"[{format_engine_stats(outcome, white_to_move)}]")
        if self.args.stats:
            print()
            print(format_root_distribution(rows, move))
            print()
            print(pv)
        if self.args.telemetry:
            err(outcome.telemetry())
        print()

        self.board.push(move)
        self.outcomes.append(outcome)
        return True

    def inject(self, move: chess.Move, label: str = "injected") -> None:
        print(f"{RED}Engine plays: {self.board.san(move)}{RESET}  [{label}]\n")
        self.board.push(move)

    # --- display --------------------------------------------------------

    def show(self) -> None:
        print(self.board)
        print(f"{DIM}History: {format_move_history(self.board, self.base_fen)}{RESET}")
        print()

    def summary(self) -> None:
        """THIS GAME's counts, with the bypassed moves excluded from the rate.

        Per game, not per process: `outcomes` is cleared by `play()` and the
        engine's own `decision_counts` is cleared by `new_game()`, so both are
        already scoped that way and a cross-game total would have to be
        accumulated deliberately. Labelled `[game]` so the number is not read as
        a session average.

        `aggregate_sims_per_s` is imported rather than reimplemented because it
        is the one place that exclusion lives: a book move delivers zero
        simulations in nonzero wall time, so folding it into a mean would report
        a throughput that depends on which opening was played.
        """
        if not self.outcomes:
            return
        rate, counts = aggregate_sims_per_s(self.outcomes)
        print()
        print(f"{DIM}[game] engine moves={counts['moves']} "
              f"search={counts['search']} book={counts['book']} "
              f"tablebase={counts['tablebase']}{RESET}")
        if counts["searched"]:
            print(f"{DIM}[game] over the searched moves only: "
                  f"{rate:,.0f} delivered sims/s "
                  f"(bypassed {counts['bypassed']} moves, "
                  f"{counts['excluded_wall_s']:.2f}s excluded){RESET}")

    # --- game over ------------------------------------------------------

    def finished(self) -> bool:
        if not self.board.is_game_over():
            return False
        outcome = self.board.outcome()
        print(f"Game over: {outcome.result() if outcome else 'unknown'}")
        if self.board.is_checkmate():
            loser_is_human = self.board.turn == self.human_side
            print("You win by checkmate!" if not loser_is_human
                  else "Engine wins by checkmate!")
        elif self.board.is_stalemate():
            print("Stalemate.")
        elif self.board.is_insufficient_material():
            print("Draw by insufficient material.")
        return True

    def end_game(self) -> None:
        """Ask whether to play again; RestartGame if yes, QuitSession if not."""
        self.summary()
        if not ask_play_again():
            raise QuitSession()
        raise RestartGame()

    # --- setup ----------------------------------------------------------

    def ask_side(self) -> None:
        while True:
            answer = prompt("Play as white or black? [w/b]: ").lower()
            if answer in ("w", "white"):
                self.human_side = chess.WHITE
                return
            if answer in ("b", "black"):
                self.human_side = chess.BLACK
                return
            print("Please enter 'w' or 'b'.")

    def ask_pgn_injection(self) -> None:
        text = prompt("Paste PGN moves to start from a position "
                      "(or press Enter for a new game): ")
        if not text:
            return
        try:
            plies = apply_pgn_text(self.board, text)
        except Exception as exc:                              # noqa: BLE001
            print(f"Could not use that PGN ({exc}). Starting from the beginning.")
            self.board = chess.Board(self.base_fen)
            return
        print(f"Loaded {plies} plies.")
        self.show()

    def ask_first_move(self) -> None:
        """The engine is to move and the game is fresh: let the human pick for it.

        playv5 offered this so an opening could be forced without a book edit; it
        is kept because it is the only way to steer the first move now that the
        book answers it deterministically at `BookSeed = 0`.
        """
        text = prompt("Engine's first move (press Enter for the engine's choice): ")
        if not text:
            if not self.engine_move():
                self.end_game()
            return
        try:
            self.inject(self.board.parse_san(text))
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError,
                chess.AmbiguousMoveError) as exc:
            print(f"Could not parse {text!r} ({type(exc).__name__}). "
                  f"The engine will choose.")
            if not self.engine_move():
                self.end_game()

    # --- commands -------------------------------------------------------

    def undo(self) -> None:
        """Rewind one full move — the engine's reply and your move.

        The tree is not rewound, and does not need to be: the shortened move list
        is no longer an extension of the one the tree was built from, so the next
        `set_position` rebuilds it. That is the same fallback playv5 reached by
        calling `mcts_engine.reset()`, arrived at by the engine rather than
        asserted by the frontend.
        """
        if len(self.board.move_stack) >= 2:
            engine_move = self.board.pop()
            human_move = self.board.pop()
            print(f"Undid: {human_move.uci()} (you), {engine_move.uci()} (engine)")
            self.show()
        elif len(self.board.move_stack) == 1 and self.human_side == chess.BLACK:
            engine_move = self.board.pop()
            print(f"Undid the engine's first move: {engine_move.uci()}")
            self.show()
            self.ask_first_move()
        else:
            print("Nothing to undo.")

    # --- the loop -------------------------------------------------------

    def play(self, pgn_preload: Optional[list[chess.Move]]) -> None:
        """One game. Returns normally only via RestartGame or QuitSession."""
        self.engine.new_game()
        self.board = chess.Board(self.base_fen)
        self.outcomes = []

        if pgn_preload is not None:
            for move in pgn_preload:
                if move not in self.board.legal_moves:
                    raise ValueError(
                        f"the loaded PGN's move {move.uci()} is not legal at ply "
                        f"{len(self.board.move_stack) + 1}; --fen and --pgn "
                        f"describe different games")
                self.board.push(move)
            # The engine takes whichever side is to move at the loaded position,
            # which is what makes this walk-back analysis rather than a game.
            self.human_side = not self.board.turn
            print(f"\nLoaded {len(pgn_preload)} plies. Engine "
                  f"({'White' if self.board.turn == chess.WHITE else 'Black'}) "
                  f"to move.")
        else:
            self.ask_side()
            self.ask_pgn_injection()

        print("\nStarting game. Enter moves in SAN (e.g. e4, Nf3, O-O).")
        print("To inject the engine's reply, enter two moves: 'e4 e5'.")
        print("'undo' rewinds one full move, 'new' starts a new game, "
              "'quit' stops.\n")

        if pgn_preload is not None:
            self.show()
            if not self.engine_move():
                self.end_game()
        elif self.board.turn != self.human_side:
            if not self.board.move_stack:
                self.ask_first_move()
            elif not self.engine_move():
                self.end_game()

        while True:
            if self.finished():
                self.end_game()
            raw = prompt(f"{GREEN}Your move: {RESET}")
            if not raw:
                continue
            if raw.lower() in _UNDO_WORDS:
                self.undo()
                continue

            parts = raw.split()
            try:
                human_move = self.board.parse_san(parts[0])
            except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError,
                    chess.AmbiguousMoveError) as exc:
                print(f"Could not parse {parts[0]!r} as a legal move "
                      f"({type(exc).__name__}). Try again.")
                continue
            self.board.push(human_move)

            if self.finished():
                self.end_game()

            # A second token is the engine's reply, supplied by hand. Parsed only
            # after the human's move is on the board, because it is legal in the
            # position that move creates and in no other.
            injected = None
            if len(parts) >= 2:
                try:
                    injected = self.board.parse_san(parts[1])
                except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError,
                        chess.AmbiguousMoveError) as exc:
                    print(f"Could not parse the injected move {parts[1]!r} "
                          f"({type(exc).__name__}). The engine will play normally.")

            if injected is not None:
                self.inject(injected)
            elif not self.engine_move():
                self.end_game()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    # FIRST STATEMENT, before any thread exists — playv6.apply_switch_interval.
    before, after = apply_switch_interval()

    parser = argparse.ArgumentParser(
        description="Play chess against the v6 (C++ core) engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands during a game: SAN moves ('e4'), two moves to inject "
               "the engine's reply ('e4 e5'), 'undo', 'new', 'quit'.")
    parser.add_argument("checkpoint", type=Path, nargs="?", default=None,
                        help="checkpoint to play, as a positional (playv5's "
                             "ergonomics). Equivalent to --model, which wins if "
                             f"both are given. Default: {DEFAULT_MODEL}")
    parser.add_argument("--fen", type=str, default=None,
                        help="start every game from this FEN instead of the "
                             "standard start position")
    parser.add_argument("--pgn", type=Path, default=None,
                        help="walk-back analysis: replay this PGN's mainline "
                             "before play starts, so the FIRST game begins at "
                             "the loaded position with the engine to move")
    parser.add_argument("--pgn-ply", type=int, default=None,
                        help="with --pgn, replay only the first N plies. Set N "
                             "so the side you want analysed is to move")
    parser.add_argument("--stats", action="store_true",
                        help="after every searched move, print the root search "
                             "distribution (visits, share, prior, Q) and the "
                             "principal variation")
    parser.add_argument("--telemetry", action="store_true",
                        help="also emit the engine's own [search] line per move "
                             "on stderr — delivered vs nominal sims, inflation, "
                             "cache traffic")
    parser.add_argument("--no-color", dest="color", action="store_false",
                        default=True, help="suppress ANSI colour")
    add_config_arguments(parser)
    args = parser.parse_args(argv)

    global GREEN, RED, DIM, RESET
    if not args.color or not sys.stdout.isatty():
        GREEN = RED = DIM = RESET = ""

    # `--model` owns the field, so the positional only fills it when it is empty.
    if args.checkpoint is not None and args.model_path is None:
        args.model_path = args.checkpoint

    if args.switch_interval != after:
        sys.setswitchinterval(args.switch_interval)
    err(f"[init] switch_interval {before:g} -> {sys.getswitchinterval():g}")

    # BEFORE the first `input()`. See playv6.preimport_torch: importing torch
    # while another thread is blocked in `sys.stdin.readline()` deadlocks on
    # Windows, and this frontend spends its life one call away from that state.
    err(f"[init] torch imported in {preimport_torch():.1f}s")

    try:
        config = config_from_args(args)
    except ConfigError as exc:
        parser.error(str(exc))
        return 2  # unreachable; parser.error exits

    pgn_preload: Optional[list[chess.Move]] = None
    if args.pgn is not None:
        try:
            pgn_preload = load_pgn_mainline(args.pgn, args.pgn_ply)
        except (OSError, ValueError) as exc:
            parser.error(f"--pgn {args.pgn}: {exc}")
            return 2

    budget = config.fixed_sims or config.default_sims
    engine = Engine(config)
    try:
        engine.ensure_ready()
        err(f"[init] book={engine.book_state}")
        err(f"[init] syzygy={engine.syzygy_state}")
        print(f"\nGuoFish v6 - {budget:,} simulations per move.")
        if engine.book is not None:
            print("The opening book is ON: early moves are looked up, not "
                  "searched. Pass --no-book to search them.")

        session = Session(engine, args, budget)
        while True:                       # the restart loop
            try:
                session.play(pgn_preload)
            except RestartGame:
                pgn_preload = None        # only the first game uses the preload
                print("\n--- Starting a new game ---\n")
                continue
            except QuitSession:
                session.summary()
                print("Quitting.")
                return 0
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
