"""C11 — the UCI surface of the C++ engine. The production entry point.

    python playing/uci_wrapper_v6.py
    python playing/uci_wrapper_v6.py --virtual-loss 3.0 --c-puct-factor 1.2
    python playing/uci_wrapper_v6.py --fixed-sims 2000        # fixed-budget match

WHAT IS HERE AND WHAT IS NOT
============================
This file is a protocol adapter and holds NO defaults. Every knob, every
default, the resolved-configuration log and the simulation accounting live in
`playing/v6/playv6.py`; this module maps UCI option names onto that object's
fields and UCI commands onto its methods. That is the split C11 asks for, and
it is what makes "the anchor's configuration is invisible" unrepresentable: the
thing that is logged and the thing that is used are the same object.

EVERY FIELD IS SETTABLE, AND A VALUE THE CORE CANNOT HONOUR IS REFUSED
======================================================================
`OPTIONS` below is generated from `EngineConfig`'s field list, so a field cannot
be added to the engine and forgotten here. v5 advertised five options, quietly
ignored `setoption` for anything else, and took `--virtual-loss` only as a
command-line flag that rebound a module global — which is how a benchmark anchor
came to be running at a virtual loss its own logs never mentioned.

A rejected value is logged on stderr and DROPPED, leaving the previous value in
place, exactly as v5 did — but here the rejection is loud, names the reason, and
the surviving value is re-printed in the next configuration log, so the two can
be reconciled after the fact. `DirichletEpsilon` is advertised and refused at
anything but 0.0: the C++ arena stores one prior per child and preserves no
untouched network distribution for noise to be derived from. See
`playv6.UNSUPPORTED_IN_CORE`.

`PolicyTemperature` WAS in that set until C11b and is now a real knob — the
search core divides the logits by it at the root and at every interior node.
Changing it rebuilds the search, which drops the tree AND the transposition
cache, because both hold priors materialised at the old temperature.

THE BOOK AND SYZYGY OPTIONS (C11b)
==================================
`UseBook`, `BookPath`, `BookSeed`, `UseSyzygy` and `SyzygyPath`. Both features
default ON. None of the five is a `SearchConfig` field, so none rebuilds the
search or costs the tree; what they do is reopen a reader, which happens here at
the `setoption` that caused it rather than at the next `isready`.

A book or tablebase move BYPASSES MCTS. The engine says so on every such move —
`info depth 0 string book e2e4`, an `info string source=...` with the running
per-game tally, and a `[game] decided N moves: ...` line on `ucinewgame` — so a
benchmark that accidentally left the book on reports it in its own output
instead of quietly shifting the ELO.

FLOATS ARE `type string`
========================
UCI spin options are integer-only and `CPuctInit 1.43`, `VirtualLoss 2.5` and
`FPUTree 0.3` are all floats. Lc0 declares its float options the same way and
Cutechess/Arena pass strings through untouched. v5's convention, kept.

THREADS: THE SEARCH RUNS ON THE MAIN THREAD
===========================================
There are exactly two Python threads in this process during a search: the main
thread, which is inside `search_parallel` with the GIL RELEASED, and a stdin
reader, which is blocked in `readline` with the GIL released. That is scope
§2.1's discipline, and C10's acquire-wait histogram is a measurement of a
process shaped exactly like this one. The reader thread exists because UCI
requires `stop`, `ponderhit` and `isready` to be answered while a search is in
flight, and `input()` on the main thread cannot do that.

The search is cut into wall-clock slices (see `Engine.search_move`) and the flags
are read between them, because the C++ core has no stop flag and no clock — by
C9's design, and deliberately not changed here.
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
import traceback
from dataclasses import fields
from pathlib import Path
from typing import Callable, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chess  # noqa: E402
import guofish_core  # noqa: E402

from playing.v6.playv6 import (  # noqa: E402
    ConfigError, Engine, EngineConfig, SearchOutcome, add_config_arguments,
    apply_switch_interval, config_from_args, err, force_utf8_streams,
    format_root_branches, move_label, preimport_torch,
)

ENGINE_NAME = "GuoFish v6"
ENGINE_AUTHOR = "Guo"

# stdout is the UCI stream and is written from two threads (the main loop, and
# the reader thread answering `readyok` during a search). One lock so a line
# cannot be interleaved with another.
_STDOUT_LOCK = threading.Lock()


def log(msg: str) -> None:
    """One UCI line to stdout, flushed. The protocol is line-oriented."""
    with _STDOUT_LOCK:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# The option table
# ---------------------------------------------------------------------------


class Option:
    """One UCI option: its display name, its EngineConfig field, its coercion.

    `kind` is what goes in the `option name ... type ...` advertisement.
    `parse` turns the GUI's string into the field's type and raises ValueError
    on anything it cannot; `EngineConfig.validate` then decides whether the
    resulting value is one the engine will run with. The two are separate on
    purpose — "not a number" and "a number the core cannot honour" are different
    failures and get different messages.
    """

    __slots__ = ("name", "attr", "kind", "parse", "extra")

    def __init__(self, name: str, attr: str, kind: str,
                 parse: Callable[[str], object], extra: str = ""):
        self.name = name
        self.attr = attr
        self.kind = kind
        self.parse = parse
        self.extra = extra


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"{raw!r} is not a boolean (true/false)")


def _parse_optional_int(raw: str) -> Optional[int]:
    lowered = raw.strip().lower()
    if lowered in ("", "none", "off", "0"):
        return None
    return int(lowered)


def _parse_optional_float(raw: str) -> Optional[float]:
    """A float, or None for "unset". Mirrors `_parse_optional_int`'s spellings.

    "0" maps to None rather than to 0.0, which is safe because the only field
    using this is `value_scale` and `EngineConfig.validate` refuses a
    non-positive one anyway. It is also what makes the round trip work:
    `_format_option_value(None)` advertises `default 0`, and a GUI that echoes
    that default straight back must not thereby set a value.
    """
    lowered = raw.strip().lower()
    if lowered in ("", "none", "off", "0"):
        return None
    return float(lowered)


def _parse_optional_path(raw: str) -> Optional[Path]:
    stripped = raw.strip()
    return Path(stripped) if stripped and stripped.lower() != "none" else None


_AFFINITY_COMBO = " ".join(f"var {p}" for p in guofish_core.AFFINITY_POLICIES)

# Intermediate `info` lines are throttled to this many per second. A 50 ms slice
# would otherwise emit 20 lines a second per game, and a Cutechess run at
# concurrency 7 turns that into 140 lines a second of log nobody reads. The FINAL
# line of a move is never throttled — it carries the score adjudication uses.
INFO_INTERVAL_S = 0.25

# EVERY EngineConfig field appears here. The completeness of this table is
# asserted by tests/test_c11_uci.py rather than trusted, because "a field was
# added to the engine and not to the protocol" is defect 1 in its purest form.
OPTIONS: tuple[Option, ...] = (
    Option("ModelPath", "model_path", "string", _parse_optional_path),
    # `type string` for the same reason every float option here is: UCI spin is
    # integer-only. Read ONCE, when the checkpoint loads, so it only has an
    # effect if it is set before the first `isready`.
    #
    # Leaving it unset is the normal case for BOTH generations: a v5 checkpoint
    # carries its own calibration and the legacy guofish2..guofish4 nets fall
    # back to `playv6.LEGACY_VALUE_SCALE`, which is the same constant their value
    # head was trained against. This exists to override that, not to enable it.
    Option("ValueScale", "value_scale", "string", _parse_optional_float),
    Option("MaxBatch", "max_batch", "spin", int, "min 1 max 1024"),
    Option("Graphs", "graphs", "check", _parse_bool),
    # C12b. Like `Graphs`, this is read ONCE — at `ensure_ready`, when the
    # forward is compiled and captured — so a GUI must set it before the first
    # `isready` for it to have any effect. Default on: the engine ships the
    # Inductor forward. Turning it off selects tag GUOFISH_NUMERICS_BASELINE's
    # unfused eager numerics, which is what Gate 2 and Gate 2b certified and what
    # a regression is bisected against.
    Option("Compile", "compile", "check", _parse_bool),
    Option("Pin", "pin", "check", _parse_bool),
    Option("SwitchInterval", "switch_interval", "string", float),

    Option("Threads", "threads", "spin", int, "min 1 max 64"),
    Option("MaxOutstanding", "max_outstanding", "spin", int, "min 1 max 4096"),
    Option("Affinity", "affinity", "combo", str, _AFFINITY_COMBO),

    Option("CPuctInit", "c_puct_init", "string", float),
    Option("CPuctBase", "c_puct_base", "string", float),
    Option("CPuctFactor", "c_puct_factor", "string", float),
    Option("FPURoot", "fpu_root", "string", float),
    Option("FPUTree", "fpu_tree", "string", float),
    Option("VirtualLoss", "virtual_loss", "string", float),
    Option("MaxTreeDepth", "max_tree_depth", "spin", int, "min 1 max 512"),
    Option("PolicyTemperature", "policy_temperature", "string", float),
    Option("DirichletEpsilon", "dirichlet_epsilon", "string", float),
    Option("DirichletAlpha", "dirichlet_alpha", "string", float),

    Option("CacheEntries", "cache_entries", "spin", int, "min 0 max 100000000"),
    Option("CacheShards", "cache_shards", "spin", int, "min 0 max 65536"),
    # C11c. BOTH ARE OPTIONAL-VALUED NOW, and 0 is the UCI spelling of "unset"
    # that `_parse_optional_int` already uses for FixedSims. Unset means the
    # computed default — `60 x (sims_per_move + ponder_max_sims)` and
    # `sims_per_move / ponder_decay` respectively — so a GUI that raises
    # DefaultSims gets an arena sized for it without also having to know the
    # formula. The `min 0` is what makes "reset to default" expressible.
    Option("ArenaCapacity", "arena_capacity", "spin", _parse_optional_int,
           "min 0 max 200000000"),
    Option("PonderMaxSims", "ponder_max_sims", "spin", _parse_optional_int,
           "min 0 max 100000000"),
    Option("PonderDecay", "ponder_decay", "string", float),
    Option("VerifyCompaction", "verify_compaction", "check", _parse_bool),

    # C11b. Both features default ON — see playv6's module docstring for why,
    # and for the telemetry that keeps a bypassed move from silently
    # contaminating a benchmark.
    Option("UseBook", "use_book", "check", _parse_bool),
    Option("BookPath", "book_path", "string", _parse_optional_path),
    Option("BookSeed", "book_seed", "spin", int, "min 0 max 2147483647"),
    Option("UseSyzygy", "use_syzygy", "check", _parse_bool),
    Option("SyzygyPath", "syzygy_path", "string", _parse_optional_path),

    Option("DefaultSims", "default_sims", "spin", int, "min 1 max 100000000"),
    Option("SimCap", "sim_cap", "spin", int, "min 1 max 100000000"),
    Option("FixedSims", "fixed_sims", "spin", _parse_optional_int,
           "min 0 max 100000000"),
    Option("SliceSeconds", "slice_seconds", "string", float),
    Option("MinSliceSims", "min_slice_sims", "spin", int, "min 1 max 1000000"),
    Option("MoveOverhead", "move_overhead_ms", "spin", int, "min 0 max 10000"),

    Option("Ponder", "ponder", "check", _parse_bool),
)

_BY_LOWER_NAME = {option.name.lower(): option for option in OPTIONS}

# C11b. Options that reopen a READER rather than rebuilding the SearchConfig.
#
# None of these is a `SearchConfig` field, so `Engine.reconfigure` does nothing
# for them and the tree survives — which is right: changing which book is open
# does not invalidate a single prior in the tree. What it does invalidate is the
# open file handle, so it is reopened here, at the command that caused it,
# rather than at the next `isready` where a failure would have no cause
# attached.
#
# Split by reader so a `BookSeed` change does not make Fathom re-read 290 table
# headers to answer a question about a random number generator.
_BOOK_OPTIONS = frozenset({"use_book", "book_path", "book_seed"})
_SYZYGY_OPTIONS = frozenset({"use_syzygy", "syzygy_path"})


def missing_options() -> set[str]:
    """EngineConfig fields with no UCI option. Must be empty; the test asserts it."""
    covered = {option.attr for option in OPTIONS}
    return {f.name for f in fields(EngineConfig)} - covered


# ---------------------------------------------------------------------------
# `go` arguments
# ---------------------------------------------------------------------------


class GoParams:
    """A parsed `go` line.

    Unknown tokens are ignored per the spec; a token whose value will not parse
    is reported on stderr and dropped rather than defaulted, so a GUI sending
    `movetime abc` does not silently get the node budget instead.
    """

    _INT_KEYS = ("wtime", "btime", "winc", "binc", "movestogo", "movetime",
                 "depth", "nodes", "mate")

    def __init__(self, args: list[str]):
        self.values: dict[str, int] = {}
        self.infinite = False
        self.ponder = False
        self.searchmoves: list[str] = []

        i = 0
        while i < len(args):
            token = args[i].lower()
            if token == "infinite":
                self.infinite = True
                i += 1
            elif token == "ponder":
                self.ponder = True
                i += 1
            elif token == "searchmoves":
                self.searchmoves = args[i + 1:]
                break
            elif token in self._INT_KEYS:
                if i + 1 >= len(args):
                    err(f"[go] '{token}' with no value; ignored")
                    i += 1
                    continue
                try:
                    self.values[token] = int(args[i + 1])
                except ValueError:
                    err(f"[go] {token}={args[i + 1]!r} is not an integer; ignored")
                i += 2
            else:
                err(f"[go] ignoring unknown token {args[i]!r}")
                i += 1

        if self.searchmoves:
            # Honest about a real limitation rather than silently searching
            # everything: neither the core nor this layer can restrict the root
            # move list, so a GUI that asks for it must be told it did not get it.
            err(f"[go] searchmoves is NOT implemented; searching all root moves "
                f"(asked for: {' '.join(self.searchmoves)})")

    def get(self, key: str) -> Optional[int]:
        return self.values.get(key)

    def has_clock(self) -> bool:
        return any(k in self.values for k in ("wtime", "btime", "movetime"))


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class UCIEngine:
    """The protocol loop. Owns the board, the config, and one `Engine`."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.engine = Engine(config)
        self.board = chess.Board()
        self._base_fen = chess.STARTING_FEN
        self._moves: list[str] = []

        # Set by the reader thread, read by the search loop between slices.
        self._stop = threading.Event()
        self._ponderhit = threading.Event()
        self._quit = threading.Event()
        self._searching = threading.Event()
        self._commands: "queue.Queue[str]" = queue.Queue()
        self._last_info = 0.0
        self._last_config_kv: Optional[str] = None

        # --- C11c: the ponder state machine ----------------------------------
        #
        # `_pondering` IS THE STATE. Set on the main thread for the whole of a
        # `go ponder` — the search phase AND the idle wait that follows it — and
        # read by the reader thread, which is what makes `ponderhit` with no
        # ponder in flight distinguishable from a real one, and what tells the
        # reader that a `position` arriving now is a discontinuity to stop on
        # rather than an ordinary command to queue.
        self._pondering = threading.Event()
        # Set by the reader when ANY of stop / ponderhit / quit arrives. It is
        # what the idle wait blocks on, so a ponder that has already spent its
        # simulation ceiling costs nothing while it waits for the GUI to say
        # which way the prediction went — and answers in microseconds when it
        # does, rather than on a polling interval.
        self._ponder_verdict = threading.Event()
        # THE `bestmove` LEDGER. Exactly one per `go`, which the protocol
        # requires and which the error path below would otherwise break: an
        # exception thrown after the move was already sent used to produce a
        # second one. Reset at the top of every `go`.
        self._bestmove_sent = False

    # --- identification and options ---------------------------------------

    def handle_uci(self) -> None:
        log(f"id name {ENGINE_NAME}")
        log(f"id author {ENGINE_AUTHOR}")
        for option in OPTIONS:
            value = getattr(self.config, option.attr)
            # The advertised default is THIS PROCESS'S current value, not the
            # dataclass default, so a GUI attached to an engine started with
            # `--virtual-loss 3.0` sees 3.0 and its "reset to default" does not
            # silently undo the command line. v5 did this right; kept.
            shown = _format_option_value(value)
            extra = f" {option.extra}" if option.extra else ""
            log(f"option name {option.name} type {option.kind} "
                f"default {shown}{extra}")
        log("uciok")

    def handle_setoption(self, args: list[str]) -> None:
        """`setoption name <Name> value <Value>`.

        Parsed positionally because an option name may contain spaces. Applied
        to a VALIDATED COPY of the configuration: `EngineConfig.replace` runs
        `__post_init__` and therefore `validate`, so a rejected value never
        reaches the live object and there is no window in which a search could
        read it.
        """
        lowered = [a.lower() for a in args]
        if "name" not in lowered:
            err(f"[setoption] malformed (no 'name'): {' '.join(args)}")
            return
        name_at = lowered.index("name")
        if "value" in lowered[name_at:]:
            value_at = lowered.index("value", name_at)
            name = " ".join(args[name_at + 1:value_at])
            raw = " ".join(args[value_at + 1:])
        else:
            name, raw = " ".join(args[name_at + 1:]), ""

        option = _BY_LOWER_NAME.get(name.strip().lower())
        if option is None:
            err(f"[setoption] REFUSED: unknown option {name!r}. Known options: "
                f"{', '.join(o.name for o in OPTIONS)}")
            return

        previous = getattr(self.config, option.attr)
        try:
            value = option.parse(raw)
        except (ValueError, TypeError) as exc:
            err(f"[setoption] REFUSED {option.name}={raw!r}: {exc}; keeping "
                f"{_format_option_value(previous)}")
            return

        if value == previous:
            err(f"[setoption] {option.name} already {_format_option_value(value)}; "
                f"no change")
            return

        # The evaluator's buffers are allocated at `max_batch` rows and the core
        # refuses a drain wider than them, so raising MaxBatch after `isready`
        # would surface as an invalid_argument out of the first search. Say so
        # now, at the command that caused it.
        if (option.attr == "max_batch" and self.engine.ready
                and int(value) > self.engine.evaluator.max_batch):
            err(f"[setoption] REFUSED MaxBatch={value}: the evaluator's buffers "
                f"hold {self.engine.evaluator.max_batch} rows and were allocated "
                f"at 'isready'. Restart the engine with --max-batch, or set this "
                f"before the first 'isready'. Keeping {previous}.")
            return

        try:
            updated = self.config.replace(**{option.attr: value})
        except ConfigError as exc:
            err(f"[setoption] REFUSED {option.name}={raw!r}: {exc}; keeping "
                f"{_format_option_value(previous)}")
            return

        self.config = updated
        self.engine.reconfigure(updated)
        # C11b. Reopen the affected reader, but only once the engine is loaded:
        # before `isready` there is nothing to reopen and `ensure_ready` will
        # open it from the final config anyway, which is the whole reason the
        # readers are opened there rather than in `Engine.__init__`.
        if self.engine.ready:
            if option.attr in _BOOK_OPTIONS:
                self.engine.reopen_book()
            elif option.attr in _SYZYGY_OPTIONS:
                self.engine.reopen_syzygy()
        err(f"[setoption] {option.name}: {_format_option_value(previous)} -> "
            f"{_format_option_value(value)}")

    def handle_isready(self, *, log_config: bool = True) -> None:
        """Load on the first call, then answer `readyok` and log the config.

        THE CONFIGURATION LOG LIVES HERE because `isready` is the last thing a
        GUI sends before it starts caring about answers, and because it is sent
        again before every search by well-behaved GUIs. Every search is therefore
        preceded by a stderr record of the configuration that will run it — which
        is C11's telemetry requirement, and the thing v5 never emitted at all.

        THE FULL CONFIGURATION ON EVERY CALL, but not the same way every time.
        `describe()`'s first line is `as_kv()` — every field, resolved, on one
        line — and that one is emitted unconditionally, so the record before
        every search is complete. The grouped human-readable block that follows
        it is emitted only when something CHANGED, because a GUI sends `isready`
        before every move and eleven identical lines per move is 660 lines for
        two games (measured). Nothing is dropped: the kv line carries the same
        fields the block does.
        """
        if not self.engine.ready:
            self.engine.ensure_ready()   # prints the whole block itself, once
            self._last_config_kv = self.config.as_kv()
            log("readyok")
            return

        if log_config:
            current = self.config.as_kv()
            err(f"[config] {current}")
            if current != self._last_config_kv:
                for line in self.config.describe()[1:]:
                    err(line)
                self._last_config_kv = current
        log("readyok")

    # --- position ---------------------------------------------------------

    def handle_ucinewgame(self) -> None:
        # C11b. The game that is ENDING is reported before its counters are
        # reset, because this is the last moment they exist. A run whose book
        # was on by accident says so here, in its own log, per game.
        if self.engine.ready:
            counts = self.engine.decision_counts
            total = sum(counts.get(s, 0) for s in ("search", "book", "tablebase"))
            if total:
                err(f"[game] decided {total} moves: search={counts.get('search', 0)} "
                    f"book={counts.get('book', 0)} "
                    f"tablebase={counts.get('tablebase', 0)}  "
                    f"[{self.engine.book_state}] [{self.engine.syzygy_state}]")
        self.board = chess.Board()
        self._base_fen = chess.STARTING_FEN
        self._moves = []
        self.engine.new_game()

    def handle_position(self, args: list[str]) -> None:
        if not args:
            err("[position] empty; ignored")
            return

        if args[0].lower() == "startpos":
            base_fen = chess.STARTING_FEN
            rest = args[1:]
        elif args[0].lower() == "fen":
            parts = []
            i = 1
            while i < len(args) and args[i].lower() != "moves":
                parts.append(args[i])
                i += 1
            base_fen = " ".join(parts)
            rest = args[i:]
        else:
            err(f"[position] unrecognised form {args[0]!r}; ignored")
            return

        moves = rest[1:] if rest and rest[0].lower() == "moves" else []

        try:
            board = chess.Board(base_fen)
        except ValueError as exc:
            err(f"[position] REFUSED: {base_fen!r} is not a FEN ({exc}); "
                f"keeping the previous position")
            return

        applied: list[str] = []
        for uci in moves:
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                err(f"[position] {uci!r} is not a UCI move; stopping the "
                    f"move list here")
                break
            if move not in board.legal_moves:
                err(f"[position] {uci} is illegal in {board.fen()}; stopping "
                    f"the move list here")
                break
            board.push(move)
            applied.append(uci)

        self.board = board
        self._base_fen = base_fen
        self._moves = applied
        self.engine.set_position(base_fen, applied)

    # --- go ---------------------------------------------------------------

    def handle_go(self, params: GoParams) -> None:
        # `_stop` is NOT cleared here. The reader thread clears it when it sees
        # the `go` line, which is the only place that preserves stdin order: a
        # `stop` sent immediately after `go` arrives while this method is still
        # being dispatched, and clearing here would swallow it and search the
        # full budget. See _reader.
        if self.board.is_game_over(claim_draw=False):
            # `bestmove 0000` is the protocol's null move and is one of the four
            # things the smoke run counts. It is emitted here and nowhere else,
            # and only for a position with no legal move at all.
            #
            # C11c: even for a `go ponder`. A finished position cannot be
            # pondered and hanging until the GUI notices is the failure mode
            # this whole chunk is tested through a pipe to prevent.
            err(f"[go] {self.board.fen()} is over ({self.board.result()}); "
                f"there is no move to make")
            err(f"[bestmove] bestmove 0000 (game already over) "
                f"fen={self.board.fen()}")
            self._send_bestmove("bestmove 0000")
            return

        # A `go` with no preceding `position` is legal UCI: the engine is
        # expected to search whatever position it holds, which is the start
        # position until told otherwise.
        self.engine.set_position(self._base_fen, self._moves)

        if params.ponder:
            self.handle_go_ponder(params)
            return

        budget, deadline, nominal, source, plan = self._plan(params)
        err(f"[go] {plan}")

        self._last_info = 0.0
        self._searching.set()
        try:
            outcome = self.engine.search_move(
                budget=budget,
                deadline=deadline,
                nominal=nominal,
                budget_source=source,
                should_stop=self._should_stop,
                on_slice=self._emit_info,
            )
        finally:
            self._searching.clear()

        err(outcome.telemetry())
        self._emit_info(outcome, final=True)
        self._emit_bestmove(outcome)

    # --- C11c: pondering ---------------------------------------------------

    def handle_go_ponder(self, params: GoParams) -> None:
        """`go ponder`: search the opponent's clock, answer stop or ponderhit.

        THE ONE PROTOCOL RULE THAT SHAPES ALL OF THIS: an engine that is
        pondering MUST NOT send `bestmove` until the GUI has said `ponderhit` or
        `stop`. Not when the simulation ceiling is reached, not when the arena
        fills, not when a mate is found. So the method is two phases and an idle
        wait between them, rather than one search whose end is the move's end.

          phase 1   search the given position — which ALREADY INCLUDES the
                    predicted reply, per standard UCI — with no clock and a
                    ceiling of `ponder_max_sims`. No `info` is emitted.
          wait      block on `_ponder_verdict` until the GUI speaks. Costs
                    nothing and answers in microseconds; a ponder that spent its
                    ceiling in 200 ms of a 30 s opponent think sits here for the
                    other 29.8 s.
          phase 2   ONLY on `ponderhit`: continue on the SAME TREE with the real
                    clock, as a second `search_move` against a higher target.
                    The tree is not rebuilt, nothing is promoted, and no visit
                    is decayed — this is the same position it has been searching
                    all along and the GUI has just confirmed it.

        WHY PHASE 2 IS A SECOND CALL rather than a deadline dropped into the
        first one. Budget accounting has to start at the ponderhit instant: the
        node ceiling for the real move is `ponder_max_sims` further on, and the
        wall clock the sims/s figure divides by is the post-hit one. Two calls
        make both of those structural instead of remembered, and they are what
        put `ponder_sims` and `search_sims` in the outcome as separate numbers.
        The TREE is what carries across, which is the entire point of pondering,
        and it carries because `search_move` never touches it — it just calls
        `search_parallel` at a higher absolute target, exactly as the slice loop
        inside one call already does.

        A MISS discards nothing here. `stop` is answered with a bestmove the GUI
        throws away, and the discontinuity arrives as the GUI's next `position`,
        which `Engine.set_position` resolves the ordinary way — extending the
        tree if the new line is this one plus moves, rebuilding it otherwise.
        Salvaging the pondered subtree on a miss was considered and rejected;
        see DECISIONS.md, C11c.
        """
        # `_pondering` was set by the READER when it saw the `go ponder` line,
        # not here — see `_reader`, and the ponderhit-immediately-after-go race
        # it exists to close. This method only clears it.
        ponder_outcome: Optional[SearchOutcome] = None
        bypass_uci: Optional[str] = None
        bypass_source = "search"
        try:
            # Requirement 6. The ponder position already includes the predicted
            # move, so if the book or the tablebase answers it the move is
            # ALREADY DETERMINED and searching it is pure waste — the whole of
            # the opponent's clock spent computing something a lookup will
            # overrule. Cheap, and it composes with C11b's decision source
            # because it is the same probe the real search would do.
            bypass_uci, bypass_source = self.engine.bypass_move(self.board)
            if bypass_uci is not None:
                err(f"[ponder] NOT pondering {self.board.fen()}: "
                    f"{bypass_source} answers it with {bypass_uci}. The move is "
                    f"already determined, so the opponent's clock buys nothing.")
            else:
                budget, _, nominal, source, plan = self._plan(params)
                err(f"[go] {plan}")
                # THE PREDICTION, NAMED. Standard UCI gives the engine the
                # position INCLUDING the predicted reply, so the prediction is
                # not something this engine chose — it is the last move of the
                # position the GUI handed us, which is the `ponder` move our own
                # previous `bestmove` suggested (see `_emit_bestmove`). Logging
                # it is what lets a reader of the stderr alone reconstruct what
                # was being pondered, which the reference printed as
                # `predicted=[...]` from its own top-K.
                err(f"[ponder] start: predicted={self._predicted_san()} "
                    f"root_visits={int(self.engine.search.root_visits)} "
                    f"(inherited from the last search) "
                    f"ceiling={self.config.ponder_max_sims_resolved} fresh sims, "
                    f"no clock")
                self._last_info = 0.0
                self._searching.set()
                try:
                    ponder_outcome = self.engine.search_move(
                        budget=budget,
                        deadline=None,
                        nominal=nominal,
                        budget_source=source,
                        should_stop=self._should_stop_pondering,
                        # Requirement 5. NO `info` DURING PONDER. Formatting an
                        # info line is pure-Python work holding the GIL, and
                        # C10c measured what that costs at the wrong switch
                        # interval — 4 sims/s, an engine that has stopped. The
                        # interval fix is already mandatory so this is defence
                        # in depth rather than correctness, but GUIs discard
                        # ponder info and it buys nothing.
                        on_slice=None,
                        # Already probed above; probing again would charge the
                        # same lookup to a search that will not use it.
                        allow_bypass=False,
                        # A ponder decides no move. Tallying it would
                        # double-count every pondered move in the per-game
                        # figures a benchmark artifact carries.
                        count_decision=False,
                    )
                finally:
                    self._searching.clear()
                err(f"[ponder] {ponder_outcome.telemetry()}")

            # THE IDLE WAIT. Phase 1 can end on the ceiling, on the arena, on a
            # mate or on the GUI; only the last of those is permission to speak.
            if not self._ponder_verdict.is_set():
                err("[ponder] search phase over; waiting for ponderhit/stop "
                    "(the protocol forbids a bestmove before the GUI speaks)")
                self._ponder_verdict.wait()
        finally:
            # PHASE 2 IS NOT A PONDER. Cleared before the hit is even read, so
            # the timed search below is an ordinary search as far as the reader
            # is concerned: `ponderhit` arriving during it is the protocol error
            # it is, and `position` arriving during it is not a ponder
            # discontinuity.
            self._pondering.clear()

        hit = self._ponderhit.is_set() and not self._quit.is_set()
        self._ponderhit.clear()

        # ONE CANONICAL VERDICT LINE PER RESOLVED PONDER, and exactly one.
        #
        # The ponder-hit rate is the number that decides whether the rejected
        # salvage-on-miss design is ever worth revisiting (see DECISIONS.md,
        # C11c), so it has to be recoverable from an ordinary match log rather
        # than from an instrumented run nobody will do again. Machine-readable
        # and greppable: `tools/bench_c11c_ponder.py` counts these.
        # NEW KEYS GO ON THE END. `tools/bench_c11c_ponder.py`'s `_VERDICT`
        # regex matches the prefix through `bypassed=` and is not anchored, so
        # appending is safe and reordering is not.
        carried = int(self.engine.search.root_visits) if self.engine.ready else 0
        err(f"[ponder] verdict={'hit' if hit else 'miss'} "
            f"ponder_sims={ponder_outcome.delivered if ponder_outcome else 0} "
            f"ponder_wall_ms={(ponder_outcome.wall_s * 1000) if ponder_outcome else 0.0:.1f} "
            f"bypassed={bypass_source if bypass_uci else 'no'} "
            f"quit={self._quit.is_set()} "
            f"predicted={self._predicted_san()} "
            # WHAT THE PONDER IS HANDING OVER, which is the whole point of
            # having pondered and the number the reference buried as
            # `actual_visits=`. On a hit this becomes `inherited` on the very
            # next `[search]` line and the two can be reconciled; on a miss it
            # is what the GUI's next `position` will discard.
            f"root_visits_carried={carried}")
        if ponder_outcome is not None and not self._quit.is_set():
            # The branch table, read while the pondered tree is still rooted
            # where the ponder left it. On a hit the GUI has confirmed the
            # prediction, so the whole root IS the branch that transfers and
            # these are the continuations under it — one ply deeper than the
            # interactive frontend's table, because UCI pondering is rooted a
            # ply further in.
            for line in format_root_branches(
                    self.engine.root_branches(), board=self.board,
                    prefix=f"[ponder] {'hit' if hit else 'miss'}"):
                err(line)

        if not hit:
            # A MISS, or a `stop`, or `quit`. The GUI discards this bestmove; it
            # is sent because the protocol owes exactly one per `go` and a
            # silent `go` is a game lost on time with no diagnosis.
            err(f"[ponder] MISS/stop: answering the go with the move the ponder "
                f"had. quit={self._quit.is_set()}")
            if bypass_uci is not None:
                self._send_bestmove(f"bestmove {bypass_uci}")
                return
            if ponder_outcome is None:
                # Nothing ran at all — a `stop` that beat the search's first
                # slice. Answer from the position rather than not at all.
                fallback = next(iter(self.board.legal_moves), None)
                self._send_bestmove(
                    f"bestmove {fallback.uci()}" if fallback else "bestmove 0000")
                return
            self._emit_info(ponder_outcome, final=True)
            self._emit_bestmove(ponder_outcome)
            return

        # --- ponderhit: the prediction held ---------------------------------
        if bypass_uci is not None:
            # The position we were told to ponder is a book or tablebase hit and
            # the opponent played into it. The move is decided; there is nothing
            # to time.
            self.engine.decision_counts[bypass_source] = (
                self.engine.decision_counts.get(bypass_source, 0) + 1)
            err(f"[ponderhit] {bypass_source} answers the position directly; "
                f"no search needed")
            self._send_bestmove(f"bestmove {bypass_uci}")
            return

        budget, deadline, nominal, source, plan = self._plan_after_ponderhit(params)
        err(f"[ponderhit] {plan}")
        self._last_info = 0.0
        self._searching.set()
        try:
            outcome = self.engine.search_move(
                budget=budget,
                deadline=deadline,
                nominal=nominal,
                budget_source=source,
                should_stop=self._should_stop,
                on_slice=self._emit_info,
            )
        finally:
            self._searching.clear()

        # THE TWO SIM COUNTS, KEPT APART. `delivered` on this outcome is the
        # post-hit work and nothing else; the ponder's contribution rides
        # alongside it. Merging them would make the engine look 2-3x faster than
        # it is on a hit and would make its throughput incomparable with the
        # ponder-off arm Gate 5 is measured against.
        if ponder_outcome is not None:
            outcome.ponder_sims = ponder_outcome.delivered
            outcome.ponder_wall_s = ponder_outcome.wall_s
            outcome.arena_exhausted = (outcome.arena_exhausted
                                       or ponder_outcome.arena_exhausted)
            outcome.arena_exhausted_at = (outcome.arena_exhausted_at
                                          or ponder_outcome.arena_exhausted_at)
        err(outcome.telemetry())
        self._emit_info(outcome, final=True)
        self._emit_bestmove(outcome)

    def _predicted_san(self) -> str:
        """The GUI's ponder move in SAN, for the ponder log lines.

        Standard UCI hands the engine the position ALREADY INCLUDING the
        predicted reply, so the prediction is the last move of `self.board` —
        which means SAN for it is defined against the position BEFORE that move,
        not against the one being pondered. Hence the copy and the pop; using
        `self.board` directly would either raise or, worse, name a different
        legal move that happens to share the coordinates' destination.
        """
        if not self._moves:
            return "unknown"
        parent = self.board.copy()
        try:
            parent.pop()
        except (IndexError, AssertionError):
            # A `position fen ... moves ...` whose stack this board does not
            # carry. Coordinates are still true.
            return self._moves[-1]
        return move_label(parent, self._moves[-1])

    def _plan(self, params: GoParams
              ) -> tuple[int, Optional[float], Optional[int], str, str]:
        """(root-visit target, deadline, requested sims, budget source, why).

        `requested sims` is None whenever the GUI asked for a duration rather
        than a count — see SearchOutcome on why a ceiling must not be reported
        as a request.

        The budget is an ABSOLUTE root-visit target because that is what the core
        takes and what the reference took (`target_new_sims = num_simulations -
        existing`). For a `go nodes N` that is exactly N — a reused root that
        already holds N visits correctly does no work. For a TIMED search the
        target is `current + sim_cap` instead: a fixed ceiling would let a deeply
        reused root exceed it and return instantly with the clock untouched.
        """
        cfg = self.config
        current = int(self.engine.search.root_visits) if self.engine.ready else 0

        if cfg.fixed_sims is not None and not params.ponder:
            # Fixed-budget tournaments compare engines on simulation count, so
            # the clock is deliberately not consulted. v5's --sims contract.
            #
            # `not params.ponder` IS LOAD-BEARING AND WAS A BUG WITHOUT IT.
            # A ponder has no clock, so "ignore the clock" says nothing about
            # it, and this branch returns an ABSOLUTE budget — which a ponder
            # must never take, because it runs on a tree that has already been
            # searched. With FixedSims=4000 and a root already holding 4,000
            # visits from the previous move, `current >= budget` on the first
            # pass of the slice loop and the ponder delivers ZERO simulations,
            # then sits in the idle wait until the GUI speaks. Pondering
            # silently does nothing, and the only symptom is ponder_sims=0 on a
            # verdict line nobody reads.
            #
            # It was unreachable until the --sims/--fixed-sims collapse, since
            # nothing set FixedSims by default; the collapse put it behind the
            # flag every fixed-budget run types. The ponder branch below is the
            # correct one for this case and needs no special-casing: its
            # ceiling is `ponder_max_sims_resolved`, which is DERIVED from
            # `sims_per_move` and therefore from `fixed_sims` when one is
            # pinned. A fixed-budget ponder is bounded by the fixed budget
            # already — as `current + fixed_sims/decay` fresh sims, which is
            # what a ponder is supposed to get.
            return (cfg.fixed_sims, None, cfg.fixed_sims, "fixed",
                    f"fixed budget {cfg.fixed_sims} sims (clock ignored; "
                    f"FixedSims is set)")

        deadline: Optional[float] = None
        clock_note = "no clock"
        if not params.ponder and not params.infinite:
            allotted = self._allot(params)
            if allotted is not None:
                deadline = time.monotonic() + allotted
                clock_note = f"{allotted * 1000:.0f} ms allotted"

        nodes = params.get("nodes")
        if nodes is not None and not params.ponder:
            # `not params.ponder` FOR THE SAME REASON THE FIXED BRANCH CARRIES
            # IT, and this is the door that is actually open in deployment.
            #
            # python-chess puts `nodes` on the ponder line: `_go` appends it
            # whenever the Limit carries one and `ponder=True` only prepends the
            # keyword, while lichess-bot ponders with `copy.copy(limit)`. So a
            # config with `go_commands.nodes: N` and `ponder: true` sends
            # `go ponder ... nodes N`, and without this guard the ponder would
            # take an ABSOLUTE budget of N instead of its own ceiling — capped
            # at N TOTAL root visits rather than given `ponder_max_sims`
            # FRESH ones, and delivering nothing at all once the pondered root
            # already holds N.
            #
            # A `nodes` count is the GUI's budget for A MOVE. A ponder is not a
            # move; it is speculative work on the opponent's clock, and the
            # quantity that bounds it is `ponder_max_sims_resolved`.
            budget = max(1, min(nodes, cfg.sim_cap))
            note = f"go nodes {nodes} -> budget {budget}"
            if nodes > cfg.sim_cap:
                err(f"[go] nodes {nodes} exceeds SimCap {cfg.sim_cap}; clamped")
            return budget, deadline, budget, "nodes", f"{note}, {clock_note}"

        if params.ponder:
            # C11c. THE PONDER CEILING IS `ponder_max_sims`, NOT `sim_cap`.
            #
            # A ponder has no clock of its own — it runs for as long as the
            # opponent thinks — so the only thing bounding it is this. `sim_cap`
            # is the wrong bound in both directions: at 60,000 it is twelve
            # times the default budget, which is how the Python reference came
            # to hand a 60,000-visit tree to a 2,000-simulation search and
            # leave it unable to redistribute (scope E6's over-commit defect).
            #
            # `ponder_max_sims` is derived from the decay for exactly that
            # reason — see EngineConfig.ponder_max_sims_resolved — and the
            # coupling it satisfies is printed on the startup line beside it.
            cap = cfg.ponder_max_sims_resolved
            source_note = "PonderMaxSims"

            # THE GUI'S OWN BUDGET FOR THIS MOVE, WHEN IT STATED ONE, BEATS THE
            # STARTUP ESTIMATE. `ponder_max_sims_resolved` derives from
            # `sims_per_move`, which is `fixed_sims or sim_cap` — a guess made
            # before any `go` arrived, and justified on the reasoning that "every
            # timed search runs to current + sim_cap and stops on the clock".
            # That reasoning holds for a clock-governed tournament and fails for
            # a harness that sends `nodes` every move: lichess-bot with
            # `go_commands.nodes: 25000` makes the real per-move budget 25,000
            # while `sim_cap` sits at its 60,000 default and never binds.
            #
            # Left alone, the ponder would spend 60,000 simulations to feed a
            # 25,000-simulation move. At `ponder_decay=1.0` the inherited visits
            # carry FULL weight, so the fresh search cannot outvote a verdict
            # formed by a tree 2.4x its size — scope E6's over-commit exactly,
            # and invisible because `coupling_holds` compares against `sim_cap`
            # and cheerfully reports OK.
            #
            # python-chess puts `nodes` on the ponder line (`_go` appends it
            # whenever the Limit carries one), so the number is right here and
            # needs no remembering across moves. Taking the MINIMUM keeps
            # `PonderMaxSims` an upper bound an operator can still lower.
            # DIVIDED BY THE DECAY, exactly as `ponder_max_sims_resolved` is.
            # The quantity a ponder must not outweigh is not the move budget but
            # the move budget the FRESH search can outvote, and that is what
            # `ponder_decay` sets: C8 decays inherited visits on a hit, so at
            # decay 0.5 a 100,000-visit pondered tree arrives weighing 50,000
            # and a 50,000-simulation fresh search can still overturn it. Capping
            # at the raw budget would make `PonderDecay` unable to buy the deeper
            # ponder it exists to permit.
            #
            # At the default decay of 1.0 this is the raw budget, which is the
            # case that matters most: nothing is decayed, so a ponder larger than
            # the move it feeds hands the fresh search a tree it cannot outvote.
            nodes = params.get("nodes")
            if nodes is not None:
                move_budget = max(1, min(nodes, cfg.sim_cap))
                allowed = max(1, int(math.ceil(move_budget / cfg.ponder_decay)))
                if allowed < cap:
                    cap = allowed
                    source_note = (f"go nodes {nodes} / decay {cfg.ponder_decay:g}"
                                   f", the GUI's budget for this move")

            budget = current + cap
            coupling = ("" if cfg.coupling_holds else
                        " WARNING: decay x ponder_max_sims exceeds sims_per_move; "
                        "a hit hands the next search a tree it cannot outvote")
            return (budget, None, None, "ponder",
                    f"ponder: up to {cap} new sims (root at {current}) "
                    f"[{source_note}], no clock, "
                    f"ending on ponderhit/stop.{coupling}")

        if params.infinite:
            budget = current + cfg.sim_cap
            return (budget, None, None, "infinite",
                    f"infinite: up to {cfg.sim_cap} new sims, ending on stop")

        if deadline is not None:
            budget = current + cfg.sim_cap
            return (budget, deadline, None, "time",
                    f"timed: {clock_note}, ceiling {cfg.sim_cap} new sims")

        return (current + cfg.default_sims, None, cfg.default_sims, "default",
                f"no clock and no nodes: DefaultSims {cfg.default_sims} new sims")

    def _plan_after_ponderhit(self, params: GoParams
                              ) -> tuple[int, Optional[float], Optional[int], str, str]:
        """The budget for phase 2, measured FROM THE PONDERHIT INSTANT.

        Both halves of that are requirement 2 of the brief and both are
        structural here rather than remembered:

          the clock   `_allot` is called NOW, so the allotted time runs from
                      this instant. The `wtime`/`btime` on the original `go
                      ponder` line are the numbers the GUI had when it sent it,
                      which is the best information available and is what every
                      engine uses.
          the nodes   the target is `root_visits_now + sim_cap`, so the ponder's
                      simulations are a BONUS and not a draw against this move's
                      allocation. Taking `sim_cap` as an absolute ceiling
                      instead would let a productive ponder arrive with the
                      budget already spent and return instantly with zero fresh
                      work — the reference's `existing_visits >= num_simulations`
                      early return, which is the defect this port is fixing.
        """
        cfg = self.config
        current = int(self.engine.search.root_visits)

        if cfg.fixed_sims is not None:
            # A fixed-budget arm ignores the clock everywhere else and must
            # ignore it here too, or a pondered move would be the one move in
            # the match decided on time.
            return (current + cfg.fixed_sims, None, cfg.fixed_sims, "fixed",
                    f"fixed budget {cfg.fixed_sims} FRESH sims on the pondered "
                    f"tree (root at {current}; clock ignored, FixedSims is set)")

        nodes = params.get("nodes")
        if nodes is not None:
            # THE GUI ASKED FOR A NODE BUDGET AND STILL MEANS IT. Without this,
            # a `go ponder ... nodes N` that HITS fell through to the clock
            # below, so a pondered move was timed while every other move in the
            # same game was node-bounded — the one budget model the operator did
            # not choose, applied to exactly the moves the prediction got right.
            #
            # `current + N` rather than an absolute N, for the reason this whole
            # method exists: the ponder's simulations ran on the OPPONENT's
            # clock and are a bonus, not a draw against this move's allocation.
            # An absolute N would let a productive ponder arrive with the budget
            # already spent and return instantly with zero fresh work.
            budget = max(1, min(nodes, cfg.sim_cap))
            return (current + budget, None, budget, "ponderhit",
                    f"ponderhit: {budget} FRESH sims on the pondered tree "
                    f"(root at {current}; go nodes {nodes}, clock not consulted "
                    f"because the GUI asked in nodes)")

        allotted = self._allot(params)
        if allotted is None:
            # A `go ponder` with no clock on it. Legal, and it means the GUI
            # gave the engine nothing to time against, so the node ceiling is
            # all there is. Said out loud because a move that ends on nodes when
            # the operator expected a clock is otherwise invisible.
            err("[ponderhit] the original 'go ponder' carried no clock; "
                "running to the node ceiling instead of a deadline")
            return (current + cfg.sim_cap, None, None, "ponderhit",
                    f"ponderhit with no clock: up to {cfg.sim_cap} fresh sims "
                    f"on the pondered tree (root at {current})")

        deadline = time.monotonic() + allotted
        return (current + cfg.sim_cap, deadline, None, "ponderhit",
                f"ponderhit: {allotted * 1000:.0f} ms FROM NOW, ceiling "
                f"{cfg.sim_cap} fresh sims on the pondered tree "
                f"(root at {current}, {current} inherited from the ponder)")

    def _allot(self, params: GoParams) -> Optional[float]:
        """Seconds to spend on this move, or None when there is no clock.

        `movetime` is taken literally minus the move overhead. Otherwise the
        classic split: an equal share of the remaining time over `movestogo`
        (30 when the GUI does not say), plus most of the increment, capped at a
        fraction of what is left so a long think cannot flag the game.
        """
        cfg = self.config
        overhead = cfg.move_overhead_ms / 1000.0

        movetime = params.get("movetime")
        if movetime is not None:
            return max(0.001, movetime / 1000.0 - overhead)

        white = self.board.turn == chess.WHITE
        remaining = params.get("wtime" if white else "btime")
        if remaining is None:
            return None
        increment = params.get("winc" if white else "binc") or 0
        movestogo = params.get("movestogo") or 30

        seconds = remaining / 1000.0
        budget = seconds / max(1, movestogo) + 0.8 * (increment / 1000.0)
        # Never commit more than 40% of what is on the clock to one move: with
        # `movestogo` absent the divisor is a guess, and this is the guard that
        # keeps a wrong guess from being fatal.
        budget = min(budget, 0.4 * seconds)
        return max(0.001, budget - overhead)

    def _should_stop(self) -> bool:
        """Polled between slices of an ORDINARY search, phase 2 included.

        `ponderhit` is deliberately not consulted. By the time phase 2 runs the
        hit has already happened and has already been converted into this
        search's deadline; a second `ponderhit` is a GUI protocol error, and the
        reader logs it rather than letting it shorten a search that is now
        running on a real clock.
        """
        return self._stop.is_set() or self._quit.is_set()

    def _should_stop_pondering(self) -> bool:
        """Polled between slices of PHASE 1. Ends on anything the GUI says.

        All three verdicts end the search and none of them decides what happens
        next — `handle_go_ponder` reads the flags afterwards and chooses. Keeping
        the decision out of the callback is what makes "ponderhit continues on
        the same tree" true by construction: this returns, `search_move` returns,
        and the tree is simply still there.
        """
        return (self._stop.is_set() or self._quit.is_set()
                or self._ponderhit.is_set())

    # --- output -----------------------------------------------------------

    def _emit_info(self, outcome: SearchOutcome, *, final: bool = False) -> None:
        """One `info` line. Runs on the main thread with no search in flight.

        `nodes` is DELIVERED simulations, not the budget. That is the reported
        number C11 exists to correct: v5 printed `nodes {num_simulations}`, so
        every `nps` a GUI or a log parser computed from its output was the
        nominal rate and, under tree reuse, 2.3-2.9x too high.

        The intermediate lines carry the root's own Q; the final one carries the
        best CHILD's Q, which is the reference's `last_best_child_q` and the
        number Cutechess adjudicates on. See Engine._outcome.
        """
        now = time.monotonic()
        if not final and now - self._last_info < INFO_INTERVAL_S:
            return
        self._last_info = now

        if outcome.bypassed:
            # C11b. A bypassed move has no search to describe: no depth, no
            # nodes, no score the engine computed. Emitting the ordinary line
            # with zeros would tell a GUI the engine searched and found nothing,
            # which is a different and false statement. `depth 0` plus a string
            # naming the source is v5's convention and what a log parser can
            # filter on.
            counts = self.engine.decision_counts
            log(f"info depth 0 string {outcome.source} {outcome.best_move}")
            log(f"info string source={outcome.source} delivered=0 "
                f"bypass=true "
                f"game_counts search={counts.get('search', 0)} "
                f"book={counts.get('book', 0)} "
                f"tablebase={counts.get('tablebase', 0)}")
            return

        parts = [
            f"info depth {max(1, outcome.depth)}",
            f"seldepth {max(1, outcome.max_depth)}",
            f"time {int(outcome.wall_s * 1000)}",
            f"nodes {outcome.delivered}",
            f"nps {int(outcome.sims_per_s)}",
            f"hashfull {outcome.hashfull}",
            f"score cp {outcome.score_cp}",
        ]
        if final and outcome.pv:
            parts.append("pv " + " ".join(outcome.pv))
        log(" ".join(parts))
        if final:
            # The nominal rate, once, as a string info. Present so a log reader
            # can see BOTH numbers and check the correction rather than trust it;
            # `nps` above stays the delivered one. `n/a` where the caller asked
            # for a duration and there is no requested count to compare against.
            nominal = "n/a" if outcome.nominal is None else str(outcome.nominal)
            nominal_nps = ("n/a" if outcome.nominal_sims_per_s is None
                           else str(int(outcome.nominal_sims_per_s)))
            inflation = ("n/a" if outcome.inflation is None
                         else f"{outcome.inflation:.2f}")
            counts = self.engine.decision_counts
            # C11c. THE DEGRADATION NOTICE, on stdout and on its own line.
            #
            # "Emit it as `info string arena exhausted at N nodes`" is the
            # brief's wording and its reason is worth keeping next to the code:
            # a silent degradation that only ever surfaces as unexplained
            # weakness is worse than a crash, because nobody investigates a
            # crash that did not happen. stderr already carries it on the
            # `[search]` line; this is the half a GUI or a log parser that never
            # opens stderr still sees.
            if outcome.arena_exhausted:
                # ASCII ONLY, and it is not a style rule. stdout is the UCI
                # stream: a GUI reads it as a byte-oriented line protocol, and
                # a cutechess/lichess-bot harness reading the pipe under a
                # cp1252 console turns a stray em-dash into U+FFFD and then
                # crashes on printing it. Found exactly that way — the first
                # run of tools/bench_c11c_ponder.py died with a
                # UnicodeEncodeError on this line rather than on anything to do
                # with the arena.
                log(f"info string arena exhausted at "
                    f"{outcome.arena_exhausted_at} nodes - this move delivered "
                    f"{outcome.delivered} simulations of its budget and is NOT "
                    f"admissible as a benchmark row")
            log(f"info string source={outcome.source} "
                f"delivered={outcome.delivered} "
                f"nominal={nominal} inherited={outcome.inherited} "
                f"delivered_nps={int(outcome.sims_per_s)} "
                f"nominal_nps={nominal_nps} "
                f"inflation={inflation} "
                # C11c. Always both, never merged. See SearchOutcome.
                f"ponder_sims={outcome.ponder_sims} "
                f"search_sims={outcome.search_sims} "
                f"total_sims={outcome.total_sims} "
                f"arena_exhausted={str(outcome.arena_exhausted).lower()} "
                f"arena_util={outcome.arena_utilisation:.3f} "
                f"budget_source={outcome.budget_source} "
                f"reason={outcome.reason} "
                f"game_counts search={counts.get('search', 0)} "
                f"book={counts.get('book', 0)} "
                f"tablebase={counts.get('tablebase', 0)}")

    def _send_bestmove(self, line: str) -> None:
        """The ONE place `bestmove` reaches stdout. C11c.

        Exactly one per `go` is a protocol requirement and was previously only a
        property of the control flow: the error handler in `run()` could not
        tell whether the move had already gone out, so an exception raised after
        the emit produced a second one. A GUI reading two `bestmove` lines for
        one `go` desynchronises for the rest of the game.

        Cleared at the top of every `go`, not here, so a duplicate is DROPPED
        and reported rather than silently sent.
        """
        if self._bestmove_sent:
            err(f"[bestmove] SUPPRESSED duplicate for this go: {line!r}. "
                f"Exactly one bestmove per go; the first one has already been "
                f"sent. This is a bug in the engine, not in the GUI.")
            return
        self._bestmove_sent = True
        log(line)

    def _emit_bestmove(self, outcome: SearchOutcome) -> None:
        """Emit the move, and mirror it to stderr.

        The mirror exists so the smoke run can count null bestmoves without
        cutechess's `-debug` transcript: this build of cutechess-cli 1.4 rejects
        `-debug` outright ("Empty value for option") and there is then no record
        of what the engine sent. The engine's own stderr is forwarded by
        cutechess into the match log either way, so `[bestmove] ...` is the
        record. See tools/smoke_c11.py.
        """
        if outcome.best_move is None:
            err(f"[go] the search returned no move for {self.board.fen()}; "
                f"falling back to the first legal move")
            fallback = next(iter(self.board.legal_moves), None)
            line = f"bestmove {fallback.uci()}" if fallback else "bestmove 0000"
        elif self.config.ponder and len(outcome.pv) >= 2:
            line = f"bestmove {outcome.best_move} ponder {outcome.pv[1]}"
        else:
            line = f"bestmove {outcome.best_move}"
        err(f"[bestmove] {line} fen={self.board.fen()}")
        self._send_bestmove(line)

    # --- the loop ---------------------------------------------------------

    def _reader(self) -> None:
        """Read stdin forever. Answer what cannot wait; queue the rest.

        Blocked in `readline` almost always, which releases the GIL — this thread
        is why `stop` is answerable at all, and it costs the search nothing
        because it is not running Python while the search is.

        `readline` rather than `for line in sys.stdin`: the iterator form reads
        ahead into a buffer, which on a pipe can hold a `stop` back until the
        next line arrives — and the next line is the one the `stop` was meant to
        prevent.
        """
        while True:
            raw = sys.stdin.readline()
            if raw == "":            # EOF: the GUI closed the pipe
                break
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            command = tokens[0].lower()

            if command == "go":
                # Cleared HERE, on the reader thread, so that a `stop` arriving
                # after this `go` is never wiped by the main thread's dispatch.
                # stdin order is preserved because both flags are touched by
                # this thread alone.
                self._stop.clear()
                self._ponderhit.clear()
                self._ponder_verdict.clear()
                # C11c. `_pondering` IS SET HERE, BY THE READER, AND NOT BY THE
                # MAIN THREAD WHEN IT GETS ROUND TO THE SEARCH.
                #
                # The main thread is an unbounded distance behind stdin — it may
                # be finishing the previous move — so a `ponderhit` sent
                # immediately after `go ponder` can arrive before
                # `handle_go_ponder` has begun. Setting the flag on the thread
                # that reads the lines is what puts the two in stdin order, and
                # it is the same argument that already governs `_stop` above.
                # C11's conformance run tests `stop` IMMEDIATELY after `go` for
                # exactly this reason; pondering adds a second flag that has to
                # survive the same ordering.
                if "ponder" in (t.lower() for t in tokens[1:]):
                    self._pondering.set()
                else:
                    self._pondering.clear()
                self._commands.put(line)
                continue
            if command == "stop":
                self._stop.set()
                # C11c. THE MISS CRITICAL PATH. Before this, `stop` could not
                # reach a search already inside `search_parallel`, so its
                # latency was bounded below by `slice_seconds`. On the ponder
                # miss path the opponent has already moved and the engine is
                # burning the engine's OWN clock until it answers, so the
                # running slice is aborted rather than waited out. See
                # Engine.interrupt_slice.
                self.engine.interrupt_slice()
                self._ponder_verdict.set()
                continue
            if command == "ponderhit":
                if not self._pondering.is_set():
                    # Requirement 4's first illegal transition. ANSWERED, not
                    # hung and not acted on: converting a search that is not a
                    # ponder into a timed one would apply a clock the GUI never
                    # offered, and silently dropping it would hide a GUI bug.
                    err("[ponderhit] no ponder in flight; ignored. A ponderhit "
                        "outside a `go ponder` is a GUI protocol error.")
                    continue
                self._ponderhit.set()
                self.engine.interrupt_slice()
                self._ponder_verdict.set()
                continue
            if command == "quit":
                self._quit.set()
                self._stop.set()
                self.engine.interrupt_slice()
                # Releases the ponder idle wait, which would otherwise block the
                # main thread forever and turn `quit` during a ponder into the
                # hang this chunk is piped-tested to prevent.
                self._ponder_verdict.set()
                self._commands.put("quit")
                return
            if command in ("position", "ucinewgame") and self._pondering.is_set():
                # Requirement 8. THE POSITION DISCONTINUITY. The GUI has moved
                # on; whatever is being pondered is about to stop being the
                # position of interest, and holding the opponent's clock to
                # finish searching it is both useless and — because the main
                # thread cannot dequeue this command until the ponder ends — a
                # hang of up to `ponder_max_sims` simulations.
                #
                # Stopped rather than dropped: the `go ponder` is still owed
                # exactly one `bestmove`, which the GUI discards, and the
                # command is queued behind it so the main thread applies it in
                # stdin order.
                err(f"[{command}] arrived mid-ponder; stopping the ponder and "
                    f"discarding it (the GUI has moved on)")
                self._stop.set()
                self.engine.interrupt_slice()
                self._ponder_verdict.set()
                self._commands.put(line)
                continue
            if command == "isready" and self._searching.is_set():
                # Answered from here because the main thread is inside a search.
                # The configuration dump is skipped: it is a dozen stderr writes
                # and this is the one moment in the process when Python running
                # competes with the dispatcher for the GIL.
                log("readyok")
                continue
            self._commands.put(line)

        # EOF. Same three releases as `quit`, for the same reason: a pipe that
        # closes while the engine is pondering must not leave the main thread
        # blocked in the idle wait.
        self._quit.set()
        self._stop.set()
        self.engine.interrupt_slice()
        self._ponder_verdict.set()
        self._commands.put("quit")

    def run(self) -> int:
        reader = threading.Thread(target=self._reader, name="uci-stdin",
                                  daemon=True)
        reader.start()

        while True:
            line = self._commands.get()
            parts = line.split()
            command, args = parts[0].lower(), parts[1:]

            if command == "quit":
                break
            try:
                if command == "uci":
                    self.handle_uci()
                elif command == "isready":
                    self.handle_isready()
                elif command == "ucinewgame":
                    self.handle_ucinewgame()
                elif command == "setoption":
                    self.handle_setoption(args)
                elif command == "position":
                    self.handle_position(args)
                elif command == "go":
                    # THE LEDGER, reset here and nowhere else: one `bestmove`
                    # per `go`, counted from the moment the `go` is dispatched.
                    self._bestmove_sent = False
                    try:
                        self.handle_go(GoParams(args))
                    finally:
                        # THE BACKSTOP for the ponder state, covering the paths
                        # `handle_go_ponder` never reaches — a `go ponder` on a
                        # finished position, and any exception out of the
                        # search. A `_pondering` left set would make the reader
                        # treat the GUI's next ordinary `position` as a
                        # discontinuity to stop a ponder that is not running.
                        self._pondering.clear()
                elif command in ("stop", "ponderhit"):
                    # Reached only when they arrive with no search in flight;
                    # the reader thread handles the in-flight case. The spec says
                    # to ignore them, and ignoring them loudly is how a protocol
                    # bug in the GUI becomes visible.
                    err(f"[{command}] no search in flight; ignored")
                elif command == "debug":
                    err(f"[debug] {' '.join(args)} (this engine always logs to stderr)")
                else:
                    err(f"[uci] ignoring unknown command {line!r}")
            except Exception as exc:                      # noqa: BLE001
                err(f"[error] command={command} {type(exc).__name__}: {exc}")
                err(traceback.format_exc())
                if command == "go":
                    # Never leave a `go` unanswered: Cutechess waits forever and
                    # scores it as a loss on time with no diagnosis in the PGN.
                    #
                    # ANNOUNCED ON BOTH STREAMS, and that is the point of this
                    # block rather than an afterthought to it. The move goes to
                    # stdout because the protocol demands one; the fact that it
                    # was NOT SEARCHED goes to stderr, because nothing else can
                    # carry it. A first-legal-move answer is a real move in a
                    # legal position and Cutechess records it beside a plausible
                    # score, so the PGN, the result and the ELO are all silent
                    # about it — a fixed-node T4 arm scored 647 of these and
                    # published an ELO-per-doubling with the sign flipped. The
                    # only reason that was diagnosable at all is that the
                    # exception itself was logged; the move was not, so no
                    # `[bestmove]` line existed to grep and the engine's own
                    # artifact disagreed with the PGN about what it had played.
                    #
                    # `info string` as well as `[bestmove]`: stderr is per-arm
                    # and easy to lose, and a GUI or a log parser that never
                    # opens it still sees the disclaimer inline.
                    #
                    # C11c: routed through `_send_bestmove`, so an exception
                    # raised AFTER the move already went out — which pondering
                    # makes reachable, since `handle_go_ponder` emits from
                    # several places — does not produce a second `bestmove` and
                    # desynchronise the GUI for the rest of the game.
                    fallback = next(iter(self.board.legal_moves), None)
                    uci = fallback.uci() if fallback else "0000"
                    err(f"[bestmove] bestmove {uci} NOT SEARCHED — first legal "
                        f"move, answering a {type(exc).__name__} from `go` "
                        f"(see the traceback above) fen={self.board.fen()}")
                    if not self._bestmove_sent:
                        log(f"info depth 0 string UNSEARCHED fallback={uci} "
                            f"reason={type(exc).__name__}")
                    self._send_bestmove(f"bestmove {uci}")

        self.engine.close()
        return 0


def _format_option_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "0"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    # C11c. BEFORE ANY LINE IS WRITTEN. Under Cutechess the streams are pipes
    # and Windows picks cp1252 for them, while every reader in this repo opens
    # the captured file as UTF-8 — which silently corrupted the `[book] opened`
    # line the provenance check reads, and cost a sound 20-game run its verdict.
    # See playv6.force_utf8_streams.
    stream_encodings = force_utf8_streams()

    # Then, before the reader thread and before the evaluator exists. Scope
    # §2.1's mitigation, measured in C10 and C10b. See playv6.apply_switch_interval.
    before, after = apply_switch_interval()

    parser = argparse.ArgumentParser(
        description="UCI wrapper for the GuoFish v6 C++ engine (C11).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # C11c: `--ponder` moved INTO `add_config_arguments`. It is an
    # `EngineConfig` field, so defining it here was the one place this file
    # owned a flag — the exact split C11 exists to have removed — and it left
    # `playv6_interactive` without one. See playv6.add_config_arguments.
    add_config_arguments(parser)
    args = parser.parse_args(argv)

    if args.switch_interval != after:
        sys.setswitchinterval(args.switch_interval)
    err(f"[init] switch_interval {before:g} -> {sys.getswitchinterval():g} "
        f"(set before any thread was started)")
    err(f"[init] streams {stream_encodings} -> "
        f"stdout={sys.stdout.encoding} stderr={sys.stderr.encoding} "
        f"(C11c: the log a provenance check reads must be UTF-8)")

    unmapped = missing_options()
    if unmapped:
        # A hard failure, not a warning: an EngineConfig field with no UCI
        # option is exactly the invisible-configuration defect C11 closes, and
        # shipping one silently would defeat the chunk.
        parser.error(f"EngineConfig fields with no UCI option: {sorted(unmapped)}. "
                     f"Add them to OPTIONS in {Path(__file__).name}.")

    try:
        config = config_from_args(args)
    except ConfigError as exc:
        parser.error(str(exc))
        return 2  # unreachable

    # BEFORE UCIEngine.run() starts the reader thread, and there is no choice
    # about it: `import torch` deadlocks against a thread blocked in
    # `sys.stdin.readline()` when stdin is a pipe. See playv6.preimport_torch for
    # the minimal reproduction. Doing it here rather than inside the module's
    # import block keeps it on the main thread and makes the cost visible.
    err(f"[init] torch imported in {preimport_torch():.1f}s "
        f"(before any thread; see playv6.preimport_torch)")

    return UCIEngine(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
