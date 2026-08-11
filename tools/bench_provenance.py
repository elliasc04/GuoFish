#!/usr/bin/env python
"""C11b — the book/Syzygy state every benchmark artifact has to carry.

WHY THIS IS A MODULE AND NOT A PARAGRAPH IN A README
====================================================
C11b turns the opening book and the Syzygy tablebase ON by default. That is the
right call for deployment — both are free strength — and it is wrong for
measurement in a specific, quiet way:

    A book or tablebase move BYPASSES MCTS. On such a move the C++ port and the
    Python reference are identical by construction — same book file, same
    tables, same lookup — so it carries none of the signal a strength gate
    exists to measure. And it delivers ZERO simulations in non-zero wall time,
    so folding it into a throughput mean drags the mean down by however many
    moves the opening happened to cover.

Neither failure announces itself. A run with the book accidentally on produces a
perfectly plausible ELO and a perfectly plausible sims/s, both wrong, and
nothing in the artifact says which run it was.

The brief's answer is a convention rather than a flag: **a harness refuses to
emit a strength or throughput table without the resolved state in its header.**
`tools/bench_c10.py` already established the shape of that refusal when it
declined to publish from a sanitizer build — the check is cheap, it runs before
the table, and it exits non-zero rather than printing a caveat nobody reads.
This module is that check, factored out so a harness cannot forget to apply it
by forgetting to copy it.

"NOT APPLICABLE" IS A RESOLVED STATE
===================================
`tools/bench_c10.py` and `tools/bench_c10b.py` drive `guofish_core` directly.
They never construct an `EngineConfig`, never open a book, and could not take a
bypass if one were offered. That is not a reason to omit the line — it is the
strongest possible version of it, and `not_applicable()` is how such a harness
records it. A reader of BENCH.md should never have to work out from the tool's
name whether a number could have included book moves.
"""

from __future__ import annotations

import re

# What a compliant header contains. Checked as substrings rather than parsed,
# because the check has to survive a harness formatting its header differently
# and must fail only when the INFORMATION is absent.
REQUIRED_MARKERS = ("book=", "syzygy=")


class UnrecordedState(RuntimeError):
    """A harness tried to publish a table without the resolved feature state.

    Raised rather than warned. See the module docstring: a warning printed above
    a table is read by nobody and survives into no artifact.
    """


def not_applicable(reason: str) -> dict:
    """The resolved state for a harness that cannot take a bypass at all."""
    return {
        "applicable": False,
        "book": f"n/a — {reason}",
        "syzygy": f"n/a — {reason}",
        "book_seed": None,
        "hits": {"search": 0, "book": 0, "tablebase": 0},
    }


def from_config(config, engine=None) -> dict:
    """The resolved state of a `playv6.EngineConfig`, and of an `Engine` if given.

    The CONFIG says what was asked for; the ENGINE says what actually opened. A
    header that recorded only the first would describe a run with a typo'd
    `SyzygyPath` as a run with tablebases on, which is exactly the confusion the
    warning in `Engine._open_syzygy` exists to prevent one layer down.
    """
    state = {
        "applicable": True,
        "book": (f"requested {config.book_target}" if config.use_book
                 else "off (UseBook=false)"),
        "syzygy": (f"requested {config.syzygy_target}" if config.use_syzygy
                   else "off (UseSyzygy=false)"),
        "book_seed": config.book_seed,
        "hits": {"search": 0, "book": 0, "tablebase": 0},
    }
    if engine is not None:
        state["book"] = engine.book_state
        state["syzygy"] = engine.syzygy_state
        state["hits"] = dict(engine.decision_counts)
    return state


# `[book] opened <path> — <how>` / `[book] WARNING: ...` and the Syzygy pair, as
# `playv6.Engine` writes them.
_BOOK_OPEN = re.compile(r"^\[book\] opened (?P<path>.+?) — (?P<how>.+)$", re.M)
_BOOK_WARN = re.compile(r"^\[book\] WARNING: (?P<why>.+)$", re.M)
_SYZYGY_OPEN = re.compile(r"^\[syzygy\] opened (?P<path>.+?) — (?P<how>.+)$", re.M)
_SYZYGY_WARN = re.compile(r"^\[syzygy\] WARNING: (?P<why>.+)$", re.M)
_KV = re.compile(r"\buse_book=(?P<use_book>\w+)\b.*?\bbook_seed=(?P<seed>-?\d+)\b"
                 r".*?\buse_syzygy=(?P<use_syzygy>\w+)\b")
# `info string source=... game_counts search=N book=N tablebase=N`, and the
# per-game summary `[game] decided N moves: search=N book=N tablebase=N`.
_GAME_COUNTS = re.compile(r"\[game\] decided \d+ moves: search=(?P<search>\d+) "
                          r"book=(?P<book>\d+) tablebase=(?P<tablebase>\d+)")
_MOVE_SOURCE = re.compile(r"\bsource=(?P<source>search|book|tablebase)\b")


def from_engine_log(text: str) -> dict:
    """The resolved state and the realised hit counts, read back out of a match.

    THIS IS THE ONE THAT MATTERS FOR A GAMES TABLE. A strength run drives the
    engine as a subprocess under Cutechess, so the harness never holds the
    `Engine` object and cannot ask it anything — the only record of what the
    engine actually did is what it wrote to stderr, which Cutechess captures per
    engine. That is by design: it makes the state a property of the ARTIFACT
    rather than of the harness's memory of the run.

    `hits` prefers the per-game `[game] decided` summaries when they are present
    (they are emitted on `ucinewgame`, so a 20-game match has 19 of them plus
    whatever the last game leaves unreported) and falls back to counting
    `source=` markers on individual moves. Both are counted and both are
    returned, because they answer slightly different questions and a large
    disagreement between them is itself worth seeing.
    """
    book_open = _BOOK_OPEN.search(text)
    book_warn = _BOOK_WARN.search(text)
    syzygy_open = _SYZYGY_OPEN.search(text)
    syzygy_warn = _SYZYGY_WARN.search(text)
    kv = _KV.search(text)

    if book_open:
        book = f"open {book_open.group('path')} [{book_open.group('how')}]"
    elif book_warn:
        book = f"DISABLED — {book_warn.group('why')}"
    elif kv and kv.group("use_book").lower() == "false":
        book = "off (UseBook=false)"
    else:
        book = None

    if syzygy_open:
        syzygy = f"open {syzygy_open.group('path')} [{syzygy_open.group('how')}]"
    elif syzygy_warn:
        syzygy = f"DISABLED — {syzygy_warn.group('why')}"
    elif kv and kv.group("use_syzygy").lower() == "false":
        syzygy = "off (UseSyzygy=false)"
    else:
        syzygy = None

    per_game = {"search": 0, "book": 0, "tablebase": 0}
    for match in _GAME_COUNTS.finditer(text):
        for key in per_game:
            per_game[key] += int(match.group(key))
    per_move = {"search": 0, "book": 0, "tablebase": 0}
    for match in _MOVE_SOURCE.finditer(text):
        per_move[match.group("source")] += 1

    return {
        "applicable": True,
        "book": book,
        "syzygy": syzygy,
        "book_seed": int(kv.group("seed")) if kv else None,
        "hits": per_game if sum(per_game.values()) else per_move,
        "hits_from_game_summaries": per_game,
        "hits_from_move_markers": per_move,
    }


def header_lines(state: dict) -> list[str]:
    """The lines a table's header must carry. Two, always, in this order."""
    hits = state.get("hits") or {}
    seed = state.get("book_seed")
    seed_note = ("" if seed is None else
                 f" seed={seed}"
                 f"{' (deterministic: highest-weight entry)' if seed == 0 else ''}")
    return [
        f"book={state.get('book')}{seed_note}",
        f"syzygy={state.get('syzygy')}",
        f"decisions: search={hits.get('search', 0)} book={hits.get('book', 0)} "
        f"tablebase={hits.get('tablebase', 0)}",
    ]


def require_recorded_state(state: dict) -> list[str]:
    """Return the header lines, or raise if the state was never resolved.

    THE REFUSAL. `book` or `syzygy` being None means the harness could not
    establish what the engine ran with — most often because the engine's stderr
    was not captured, which is a one-character mistake in a Cutechess command
    line and produces a match log that looks completely normal.

    Publishing a strength or throughput table from such a run is exactly the
    failure this module exists to prevent, so it raises rather than filling in a
    plausible default.
    """
    missing = [key for key in ("book", "syzygy") if not state.get(key)]
    if missing:
        raise UnrecordedState(
            f"refusing to publish: the resolved {' and '.join(missing)} state "
            f"could not be determined for this run.\n"
            f"  A book or tablebase move bypasses MCTS entirely, so a table that "
            f"does not say whether either was on describes a measurement nobody "
            f"can interpret — and both default to ON.\n"
            f"  The state is read from the engine's own stderr. Check that the "
            f"harness captured it (cutechess: `stderr=<path>` on the engine "
            f"line) and that the engine reached `isready`.")
    lines = header_lines(state)
    text = "\n".join(lines)
    absent = [marker for marker in REQUIRED_MARKERS if marker not in text]
    if absent:
        raise UnrecordedState(
            f"the assembled header is missing {absent}; header_lines() and "
            f"REQUIRED_MARKERS have drifted apart")
    return lines


__all__ = [
    "REQUIRED_MARKERS",
    "UnrecordedState",
    "from_config",
    "from_engine_log",
    "header_lines",
    "not_applicable",
    "require_recorded_state",
]
