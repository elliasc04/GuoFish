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
    apply_switch_interval, config_from_args, err, preimport_torch,
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
    Option("ArenaCapacity", "arena_capacity", "spin", int, "min 1024 max 200000000"),
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
        # Set on the main thread when a `go ponder` starts; read by _should_stop.
        self._pending_ponder_params: Optional[GoParams] = None
        self._ponder_deadline: Optional[float] = None

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
            err(f"[go] {self.board.fen()} is over ({self.board.result()}); "
                f"there is no move to make")
            err(f"[bestmove] bestmove 0000 (game already over) "
                f"fen={self.board.fen()}")
            log("bestmove 0000")
            return

        # A `go` with no preceding `position` is legal UCI: the engine is
        # expected to search whatever position it holds, which is the start
        # position until told otherwise.
        self.engine.set_position(self._base_fen, self._moves)

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

        # A ponder search that was converted by `ponderhit` has already had its
        # deadline applied inside `_should_stop`; one that ended on `stop`
        # reports a move anyway, which is what the GUI expects.
        err(outcome.telemetry())
        self._emit_info(outcome, final=True)
        self._emit_bestmove(outcome)

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

        if cfg.fixed_sims is not None:
            # Fixed-budget tournaments compare engines on simulation count, so
            # the clock is deliberately not consulted. v5's --sims contract.
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
        if nodes is not None:
            budget = max(1, min(nodes, cfg.sim_cap))
            note = f"go nodes {nodes} -> budget {budget}"
            if nodes > cfg.sim_cap:
                err(f"[go] nodes {nodes} exceeds SimCap {cfg.sim_cap}; clamped")
            return budget, deadline, budget, "nodes", f"{note}, {clock_note}"

        if params.ponder or params.infinite:
            budget = current + cfg.sim_cap
            kind = "ponder" if params.ponder else "infinite"
            return (budget, None, None, kind,
                    f"{kind}: up to {cfg.sim_cap} new sims, ending on "
                    f"{'ponderhit/stop' if params.ponder else 'stop'}")

        if deadline is not None:
            budget = current + cfg.sim_cap
            return (budget, deadline, None, "time",
                    f"timed: {clock_note}, ceiling {cfg.sim_cap} new sims")

        return (current + cfg.default_sims, None, cfg.default_sims, "default",
                f"no clock and no nodes: DefaultSims {cfg.default_sims} new sims")

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
        """Polled between slices. Also converts a ponder into a timed search.

        `ponderhit` means the opponent played the move we were pondering on, so
        the search continues on the same tree and the clock starts NOW. The
        conversion happens here rather than in `handle_go` because that method is
        inside `search_move` at the time and this callback is the only code that
        runs between slices.
        """
        if self._stop.is_set() or self._quit.is_set():
            return True
        if self._ponderhit.is_set():
            self._ponderhit.clear()
            allotted = self._allot(self._pending_ponder_params) \
                if self._pending_ponder_params is not None else None
            if allotted is None:
                err("[ponderhit] no clock in the original 'go'; continuing to "
                    "the node ceiling")
                self._ponder_deadline = None
            else:
                self._ponder_deadline = time.monotonic() + allotted
                err(f"[ponderhit] converting to a timed search: "
                    f"{allotted * 1000:.0f} ms from now")
        if self._ponder_deadline is not None and time.monotonic() >= self._ponder_deadline:
            err("[ponderhit] allotted time reached")
            return True
        return False

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
            log(f"info string source={outcome.source} "
                f"delivered={outcome.delivered} "
                f"nominal={nominal} inherited={outcome.inherited} "
                f"delivered_nps={int(outcome.sims_per_s)} "
                f"nominal_nps={nominal_nps} "
                f"inflation={inflation} "
                f"budget_source={outcome.budget_source} "
                f"reason={outcome.reason} "
                f"game_counts search={counts.get('search', 0)} "
                f"book={counts.get('book', 0)} "
                f"tablebase={counts.get('tablebase', 0)}")

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
        log(line)

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
            command = line.split()[0].lower()

            if command == "go":
                # Cleared HERE, on the reader thread, so that a `stop` arriving
                # after this `go` is never wiped by the main thread's dispatch.
                # stdin order is preserved because both flags are touched by
                # this thread alone.
                self._stop.clear()
                self._ponderhit.clear()
                self._commands.put(line)
                continue
            if command == "stop":
                self._stop.set()
                continue
            if command == "ponderhit":
                self._ponderhit.set()
                continue
            if command == "quit":
                self._quit.set()
                self._stop.set()
                self._commands.put("quit")
                return
            if command == "isready" and self._searching.is_set():
                # Answered from here because the main thread is inside a search.
                # The configuration dump is skipped: it is a dozen stderr writes
                # and this is the one moment in the process when Python running
                # competes with the dispatcher for the GIL.
                log("readyok")
                continue
            self._commands.put(line)

        self._quit.set()
        self._stop.set()
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
                    params = GoParams(args)
                    self._pending_ponder_params = params if params.ponder else None
                    self._ponder_deadline = None
                    self.handle_go(params)
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
                    fallback = next(iter(self.board.legal_moves), None)
                    uci = fallback.uci() if fallback else "0000"
                    err(f"[bestmove] bestmove {uci} NOT SEARCHED — first legal "
                        f"move, answering a {type(exc).__name__} from `go` "
                        f"(see the traceback above) fen={self.board.fen()}")
                    log(f"info depth 0 string UNSEARCHED fallback={uci} "
                        f"reason={type(exc).__name__}")
                    log(f"bestmove {uci}")

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
    # FIRST, before the reader thread and before the evaluator exists. Scope
    # §2.1's mitigation, measured in C10 and C10b. See playv6.apply_switch_interval.
    before, after = apply_switch_interval()

    parser = argparse.ArgumentParser(
        description="UCI wrapper for the GuoFish v6 C++ engine (C11).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--ponder", action="store_true", default=False,
                        help="advertise and honour the UCI ponder handshake")
    args = parser.parse_args(argv)

    if args.switch_interval != after:
        sys.setswitchinterval(args.switch_interval)
    err(f"[init] switch_interval {before:g} -> {sys.getswitchinterval():g} "
        f"(set before any thread was started)")

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
