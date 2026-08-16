"""C11 — the production Python surface over the C++ core.

    python -m playing.v6.playv6 --fen <FEN> --sims 20000
    python -m playing.v6.playv6 --selfplay 40 --sims 2000

WHAT THIS FILE IS FOR
=====================
Everything the engine is configured by, and everything it reports about itself,
lives here. `playing/uci_wrapper_v6.py` is a protocol adapter on top of it and
owns no defaults of its own; this module owns the whole parameter surface, the
resolved-configuration log, and the simulation accounting. That split is
deliberate — C11 exists because the v5 wrapper owned both and lost track of both.

THE TWO v5 DEFECTS THIS CHUNK CLOSES
====================================
1. SILENTLY DROPPED OPTIONS. `uci_wrapper_v5.py` advertised five UCI options and
   accepted `--virtual-loss` and `--max-tree-depth` on the command line by
   rebinding module globals in `core.mctsv4`. Nothing ever printed the resolved
   result, so an anchor's true virtual loss was invisible in its own logs, and a
   `setoption` for a name outside `_OPTION_TO_PARAM` was answered with one line
   on stderr that nobody read. Here every field of `EngineConfig` is settable,
   every field is printed on `isready` (`describe()` / `as_kv()`), and a value
   the core cannot honour is REFUSED rather than dropped — see
   `EngineConfig.validate` and `UNSUPPORTED_IN_CORE` below.

2. THROUGHPUT REPORTED AGAINST THE WRONG NUMERATOR. v5's `info` line carried
   `nodes {num_simulations}` — the REQUESTED budget — and every sims/s figure
   derived from it. With tree reuse the root arrives already carrying visits, so
   the work actually done is `budget - inherited`, and dividing the budget by the
   wall clock inflated the rate by the reuse ratio (2.3-2.9x in the v5 logs;
   1.17x measured over the C11 smoke run's 538 moves at 800 sims, and 9-20x on
   individual deep-reuse moves). `SearchOutcome` carries `delivered` (what the
   core backed up into the root, counted by the thread that did it) and
   `nominal` (what the caller asked for, or None if it asked in seconds) as
   separate fields, and `sims_per_s` is delivered/wall. The nominal rate is
   available as `nominal_sims_per_s` and is never the headline.

WHAT THE CORE DOES NOT IMPLEMENT, AND IS THEREFORE REFUSED
==========================================================
C11 refused BOTH `policy_temperature` and Dirichlet root noise, because neither
had a counterpart in `cpp/` and C11's scope forbade changing the search core.

C11b was given that mandate and used it for one of the two. `policy_temperature`
is now a real knob: `SearchConfig::policy_temperature` divides the logits inside
`gather_softmax_canonical`, at the root and at every interior node alike, and
this module passes it through like any other search field. It is no longer in
`UNSUPPORTED_IN_CORE`.

Dirichlet noise stays refused, and stays refused for the reason C11 gave rather
than for want of a mandate: the arena stores ONE prior per child and nothing
preserves the untouched network distribution that the reference's C3b fix
requires noise to be derived from. Mixing noise into the stored priors would
compound on every reused root — the exact defect C3b fixed over there. Adding a
`base_prior` field to the arena is a chunk of its own and nothing currently
planned needs it. The consequence, stated plainly so it is not discovered later:
**the C++ engine cannot generate self-play training data.** Play strength,
benchmarking and tournament use are unaffected; training-data generation runs on
the Python reference until that chunk happens.

THE OPENING BOOK AND SYZYGY, AND WHY THEY DEFAULT ON
====================================================
C11b re-introduces both, as v5 had them. Both default to **ON**, because both
are free strength and the engine should play its best out of the box.

That is right for deployment and wrong for measurement, and the discipline that
has to come with it is not "default them off" — it is that a bypassed move is
VISIBLE IN THE ENGINE'S OWN OUTPUT. A book or tablebase move skips MCTS
entirely, so it is a move where this port and the Python reference are identical
by construction, and folding it into a strength or throughput figure dilutes
exactly the signal that figure exists to carry.

Three mechanisms, all here:

  * `SearchOutcome.source` is `search`, `book` or `tablebase`, on every move.
  * A bypassed move delivers ZERO simulations, so `sims_per_s` on it is not a
    slow move — it is a meaningless one. `SearchOutcome.bypassed` marks it and
    every aggregate in this file and in `tools/` excludes it rather than
    dividing by it or averaging it in.
  * `Engine.decision_counts` tallies the three per game, and `describe()` /
    `as_kv()` carry the RESOLVED reader state — the path that was opened, or the
    path that was tried and missing. A run that accidentally had the book on
    reports book hits in its own output instead of quietly shifting the ELO.

`BookSeed = 0` means "always play the highest-weight move". A non-zero seed
picks weighted-randomly from a seeded RNG and the RNG is NOT reset between
games, because varied play across a session is the whole point of asking for a
seed. Zero gives benchmarking a fully reproducible book without needing the book
disabled, which is the cleanest resolution of the contamination problem above.

Missing files WARN AND DISABLE, naming the path they tried, and never fail: a
typo'd `SyzygyPath` and an intentionally absent one must not be
indistinguishable.

SWITCH INTERVAL
===============
`sys.setswitchinterval(0.0005)` is scope §2.1's GIL mitigation, measured in C10
and re-measured in C10b. `apply_switch_interval()` applies it and returns what
it replaced, so the config log can show both; every entry point calls it as its
first statement, before any thread exists. `guofish_core.LiveEvaluator`'s
constructor sets it too, which is a second belt on the same trousers — the
constructor runs after argument parsing, and a crash between the two would leave
the process on the 5 ms default with nothing saying so.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import math
import random
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

DEFAULT_MODEL = REPO_ROOT / "models" / "guofish5_90M" / "v5_10.9M_best.pt"
DEFAULT_SWITCH_INTERVAL = guofish_core.DEFAULT_SWITCH_INTERVAL

# The two host-side asset locations, restated from uci_wrapper_v5.py rather than
# imported so the v6 surface does not depend on the v5 one. Neither is a build
# dependency: a missing book or an absent tablebase directory disables that
# feature with a warning naming the path, and the engine plays on.
DEFAULT_BOOK = REPO_ROOT / "assets" / "gm2001.bin"
DEFAULT_SYZYGY = REPO_ROOT / "assets" / "syzygy"

# What decided a move. `SearchOutcome.source` is one of these and nothing else.
DECISION_SOURCES = ("search", "book", "tablebase")

# Largest |value| the v5 label pipeline ever produced (data/multiPV/labels.py:
# VALUE_CLAMP / VALUE_MATE = 0.995), restated from uci_wrapper_v5.py rather than
# imported so the v6 surface does not depend on the v5 one. See
# `q_to_centipawns` for what it does to the reported ceiling.
VALUE_CLAMP = 0.995

# The centipawn calibration BOTH generations were trained against, used when a
# checkpoint carries none of its own.
#
# NOT A GUESS, AND NOT v5's NUMBER BORROWED FOR v4. It is one project-wide
# constant that predates both, and it is written down in four places:
#
#   data/csv_parallel.py:29        CP_SCALE, the v2-era pipeline — the one that
#                                  fed data/eval_data_processing.py and
#                                  training/train.py, i.e. guofish2..guofish4
#   data/multiPV/labels.py:44      VALUE_SCALE, the v5 multi-PV pipeline
#   data/multiPV/fit_value_scale.py  re-examined against our own data and
#                                  explicitly LOCKED here on 2026-08-02
#   benchmarking/player/acpl_elo_estimator.py:49  Q_TO_CP_SCALE, which inverts
#                                  it with atanh exactly as q_to_centipawns does
#
# Both generations' value heads were therefore trained on tanh(cp / 290.6806)
# with the same +-2000cp clip; it is the Lc0 WDL calibration, chosen so that the
# value approximates expected score rather than being a fit on GuoFish data. The
# legacy guofish2..guofish4 checkpoints omit `value_scale` only because the v2-era
# checkpoint writer predates the field — NOT because they used a different scale.
#
# So falling back to it is recovering a documented constant, not inventing one,
# and it is what puts a v4-vs-v5 match on ONE reported scale. `_resolve_value_scale`
# still announces every use of the fallback, because "the checkpoint said so" and
# "the engine assumed the project default" are different claims.
LEGACY_VALUE_SCALE = 290.6806


def err(msg: str) -> None:
    """stderr, flushed. Never stdout: stdout is the UCI stream."""
    print(msg, file=sys.stderr, flush=True)


def force_utf8_streams() -> str:
    """Pin stdout and stderr to UTF-8. Returns what they were, for the log.

    C11c, AND IT IS A CORRECTNESS FIX RATHER THAN TIDINESS.

    On Windows, a Python process whose streams are PIPES picks its encoding from
    the locale — cp1252 here — while every reader in this repo opens the
    resulting file as UTF-8. The engine's own configuration lines have carried
    em-dashes since C11b (`[book] opened <path> - <how>`), and cp1252 encodes
    one as the single byte 0x97, which UTF-8 then decodes as U+FFFD.

    That is not cosmetic. `tools/bench_provenance.py` matches those lines with a
    regex containing a literal em-dash, and it is what
    `require_recorded_state()` uses to decide whether a games table may be
    PUBLISHED AT ALL. The C11c 20-game ponder smoke passed every one of its four
    acceptance criteria and then failed its provenance check for this reason:
    the run was sound and its artifact said the book state could not be
    determined.

    Called first thing in every entry point, before the reader thread and before
    anything writes a line. stdout is separately kept ASCII-only by protocol —
    UCI is a byte-oriented line protocol and a GUI is not obliged to decode
    anything else — so this changes nothing there and fixes the stream that
    carries prose.
    """
    before = f"stdout={getattr(sys.stdout, 'encoding', '?')} " \
             f"stderr={getattr(sys.stderr, 'encoding', '?')}"
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):        # pragma: no cover
            # A stream that is not a TextIOWrapper — a test harness's capture
            # object, most likely. Nothing to do and nothing worth failing over.
            pass
    return before


def apply_switch_interval(interval: float = DEFAULT_SWITCH_INTERVAL) -> tuple[float, float]:
    """Set the interpreter's thread switch interval. Returns (before, after).

    MUST run before any thread is spawned — scope §2.1, and C10's measurement is
    of a process that did. Called first thing in both entry points' `main()`, and
    reported in the configuration log so a run cannot claim the mitigation
    without having applied it.
    """
    before = sys.getswitchinterval()
    sys.setswitchinterval(interval)
    return before, sys.getswitchinterval()


def preimport_torch() -> float:
    """Import torch NOW, on the main thread, before any other thread exists.

    NOT an optimisation. `import torch` DEADLOCKS on Windows when another Python
    thread is blocked in `sys.stdin.readline()`, which is exactly the state the
    UCI wrapper's reader thread spends its life in. Reproduced minimally in C11:

        thread = threading.Thread(target=lambda: sys.stdin.readline())
        thread.start()          # parent holds the stdin pipe open
        import torch            # never returns

    The same child with the reader thread replaced by `time.sleep(3600)`, or by a
    thread reading an ordinary FILE, imports in 1.4 s; with the import moved
    ahead of the thread, everything after it — CUDA init, the checkpoint load and
    the CUDA graph capture — runs normally with the reader blocked. So the rule
    is narrow and mechanical: torch first, threads second.

    It bites only through a pipe, which is to say only under Cutechess,
    lichess-bot and `tools/uci_conform_c11.py`, and never at an interactive
    terminal — the configuration nobody tests interactively. It cost this chunk
    an afternoon; hence the size of this comment.

    Returns the seconds it took, for the startup log.
    """
    started = time.perf_counter()
    import torch  # noqa: F401
    return time.perf_counter() - started


def q_to_centipawns(q: float, value_scale: float) -> int:
    """An MCTS Q in [-1, 1] to a calibrated centipawn score.

    v5's value target is tanh(cp / value_scale), so the exact inverse is
    cp = value_scale * atanh(q), not a linear rescale. `value_scale` travels in
    the checkpoint (train_v5.save_checkpoint writes it unconditionally) and is
    read off the loaded model, never guessed.

    Q is clamped to +-VALUE_CLAMP because atanh diverges at +-1 and the labels
    stopped there anyway: mates and everything past the +-2000cp clip trained to
    the same 0.995, so Q beyond it carries no magnitude. The reported score
    therefore ceilings at ~+-870cp on the shipped scale, and a Cutechess
    `-resign score=` must sit below that to ever fire.
    """
    q = max(-VALUE_CLAMP, min(VALUE_CLAMP, q))
    return int(round(value_scale * math.atanh(q)))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fields the C++ core has no mechanism for, mapped to the ONLY value this
# module will accept for them and the sentence explaining why. Refusing is the
# whole point: a value the core cannot honour, accepted silently, is a benchmark
# row that describes a search nobody ran.
#
# C11b REMOVED `policy_temperature` FROM THIS SET. It is implemented now — see
# SearchConfig::policy_temperature and cpp/evaluator.hpp — so refusing it would
# have become the mirror-image defect: a knob the engine has, declared broken.
# Dirichlet noise stays, and the module docstring says what that costs.
UNSUPPORTED_IN_CORE: dict[str, tuple[float, str]] = {
    "dirichlet_epsilon": (
        0.0,
        "cpp/ has no root-noise mechanism and no per-node base_prior: the arena "
        "stores one prior per child and nothing preserves the untouched network "
        "distribution the C3b fix requires noise to be derived from. Mixing "
        "noise into the stored priors instead would compound on every reused "
        "root, which is the exact defect C3b fixed in the reference. Only 0.0 "
        "(noise off) is accepted.",
    ),
}


class ConfigError(ValueError):
    """A configuration value the engine will not run with.

    Raised rather than logged-and-ignored. A tournament engine that silently
    retunes itself to a value nobody asked for produces results nobody can
    interpret; the UCI layer catches this per-`setoption` and keeps the previous
    value, and the command line lets it reach `parser.error`.
    """


@dataclass
class EngineConfig:
    """Every knob, in one object, with the shipping defaults.

    The defaults are BENCH.md C10b-3g's selection — W=1, K=24 (so
    `max_outstanding` 24), `max_batch` 128, affinity `none` — plus C10's cache
    and arena sizing. They are here and nowhere else: `uci_wrapper_v6.py`
    constructs one of these and never invents a number.
    """

    # --- model and evaluator ---------------------------------------------
    model_path: Optional[Path] = None
    max_batch: int = 128
    graphs: bool = True
    pin: bool = True
    switch_interval: float = DEFAULT_SWITCH_INTERVAL

    # C12b. THE ADOPTION HAPPENS HERE, which is why this default is True while
    # `TorchEvaluator`'s is False. The library default keeps C10/C10b's
    # acceptance tests comparing the forward they certified; the ENGINE — the
    # thing that plays games and the thing Gate 4 and Gate 5 measure — ships
    # Inductor, because ~1.19x on fresh-root searches is worth having and the
    # numerics it costs are re-certified by Gate 2' rather than lost.
    #
    # `compile=False` is a real fallback and stays one: it reproduces tag
    # GUOFISH_NUMERICS_BASELINE bit-exactly (asserted in
    # tests/test_c12b_gate2prime.py), so a regression can be bisected against it
    # rather than argued about, in the same spirit as `graphs=` and `pin=`.
    compile: bool = True

    # An explicit centipawn calibration for `q_to_centipawns` to invert. None
    # means "resolve it normally" — the checkpoint's own value if it has one,
    # else LEGACY_VALUE_SCALE. See `Engine._resolve_value_scale`.
    #
    # Setting this OVERRIDES a checkpoint that carries its own, which is the only
    # reason it is a knob at all: the two automatic sources already agree at
    # 290.6806 for everything shipped, so the flag exists for recalibration
    # experiments and for a checkpoint whose recorded scale is not trusted.
    value_scale: Optional[float] = None

    # --- parallelism (C9/C10b) -------------------------------------------
    # W. `in_flight` (K) is derived from max_outstanding rather than set
    # directly, because scope §2.2 makes W*K the governing quantity and C9c
    # measured the two independently: at a fixed outstanding count the concurrent
    # rows are no flatter at the root than the serial control, so the knob a
    # tuner should turn is the product.
    threads: int = 1
    max_outstanding: int = 24
    affinity: str = "none"

    # --- search (SearchConfig) -------------------------------------------
    c_puct_init: float = 1.43
    c_puct_base: float = 19652.0
    c_puct_factor: float = 1.0
    fpu_root: float = 0.0
    fpu_tree: float = 0.30
    virtual_loss: float = 2.5
    max_tree_depth: int = 80
    cache_entries: int = 400_000
    cache_shards: int = 0          # 0 => the core's kDefaultCacheShards

    # C11c. NODES PER ARENA, and `None` means "compute it from the budget".
    #
    # It was a hard 1,200,000 until C11c, which is a number that was right for
    # the budget it was measured at and silently wrong for any other. Pondering
    # makes that unacceptable rather than merely untidy: an unbounded ponder
    # searches for as long as the opponent thinks, so the arena has to be sized
    # against `sims_per_move + ponder_max_sims` and not against `sims_per_move`
    # alone. See `arena_nodes` for the formula and the measured curve it fits.
    #
    # Still an explicit field, because no formula is provably right for every
    # position and an operator must be able to overrule it. Setting it pins the
    # value; leaving it None tracks the budget, which is what a
    # `--sims 15000` run wants without also having to remember to raise this.
    arena_capacity: Optional[int] = None

    # C11c. The ceiling on a PONDER search's simulations, and `None` means
    # "compute it from the budget and the decay". See `ponder_max_sims_resolved`.
    #
    # THE PRIMARY MECHANISM, with graceful arena exhaustion as the backstop
    # rather than the other way round: a ponder has no budget of its own, so on
    # a slow opponent clock it would otherwise fill any arena that exists.
    ponder_max_sims: Optional[int] = None

    ponder_decay: float = 1.0
    verify_compaction: bool = False

    # C11b. T in softmax(logits / T). 1.0 is the identity, skips the divide, and
    # is what every figure recorded before C11b was measured at. A SearchConfig
    # field, so changing it rebuilds the search — which is how the tree and the
    # cache get dropped together. See `reconfigure` and `_SEARCH_CONFIG_FIELDS`.
    policy_temperature: float = 1.0

    # --- declared, refused unless identity (see UNSUPPORTED_IN_CORE) ------
    dirichlet_epsilon: float = 0.0
    dirichlet_alpha: float = 0.3

    # --- budgets ----------------------------------------------------------
    # `fixed_sims` IS THE PER-MOVE BUDGET and is what `--sims` sets. `sim_cap`
    # bounds whatever `go nodes` asks for and sizes the arena; `default_sims` is
    # the floor under a `go` that carried neither a clock nor nodes.
    #
    # NO LONGER A COMMAND-LINE FLAG (the `--sims`/`--fixed-sims` collapse; see
    # `add_config_arguments`). It stays a FIELD, with its `DefaultSims` UCI
    # option, because `_plan`'s last branch needs a number and `fixed_sims` may
    # not have one: `fixed_sims=None` is what "defer to the GUI" is spelled as,
    # and `_plan` tests it first and returns deadline=None, so a non-None
    # default here would make the engine ignore every clock in every game.
    #
    # In practice it is unreachable in a real game — lichess-bot and Cutechess
    # both always send a clock or nodes — and it exists for a bare `go` at a
    # terminal and as the CLI/interactive fallback (`fixed_sims or
    # default_sims`).
    default_sims: int = 5_000
    sim_cap: int = 60_000
    fixed_sims: Optional[int] = None
    # Wall-clock slice the search is cut into so `stop` and the clock are
    # answerable. See Engine.search.
    slice_seconds: float = 0.05
    min_slice_sims: int = 32

    # --- host-side features (C11b) ----------------------------------------
    # BOTH DEFAULT ON. See the module docstring for why, and for the three
    # mechanisms that keep a bypassed move from silently contaminating a
    # measurement.
    #
    # The paths are `None` for "the shipped default", exactly as `model_path`
    # is, and `book_target`/`syzygy_target` resolve them. What gets logged is
    # the RESOLVED path and whether the reader actually opened, because "the
    # config said assets/gm2001.bin" and "the engine is playing book moves" are
    # different claims and only the second one matters.
    use_book: bool = True
    book_path: Optional[Path] = None
    # 0 IS NOT "unseeded". It means "always play the highest-weight entry",
    # which is what gives a benchmark a fully reproducible book without having
    # to turn the book off. Any other value seeds a `random.Random` for
    # weighted selection. See `Engine.probe_book`.
    book_seed: int = 0
    use_syzygy: bool = True
    syzygy_path: Optional[Path] = None

    ponder: bool = False
    move_overhead_ms: int = 100

    def __post_init__(self) -> None:
        for name in ("model_path", "book_path", "syzygy_path"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, Path(value))
        self.validate()

    # --- derived ----------------------------------------------------------

    @property
    def book_target(self) -> Path:
        """The Polyglot file this configuration would open. Always a path.

        Resolution and USE are separate questions: this answers the first even
        when `use_book` is False, so a log can say which book was declined.
        """
        return self.book_path or DEFAULT_BOOK

    @property
    def syzygy_target(self) -> Path:
        """The tablebase directory this configuration would open."""
        return self.syzygy_path or DEFAULT_SYZYGY

    # --- C11c: the budget, the ponder cap and the arena, in that order ------
    #
    # These three are ONE derivation and are written next to each other for that
    # reason. Setting them independently is exactly how the Python reference
    # reached 60,000 inherited ponder simulations against 2,000 fresh ones —
    # scope E6's over-commit defect — and `describe()` prints all three on one
    # line so a mismatched pair is visible before a game rather than after one.

    @property
    def sims_per_move(self) -> int:
        """The MOST simulations one move can spend. The other two derive from it.

        `fixed_sims` when a fixed-budget tournament has pinned one, else
        `sim_cap`.

        `sim_cap` RATHER THAN `default_sims`, AND THE DIFFERENCE WAS MEASURED
        RATHER THAN ARGUED. `default_sims` (5,000) is what a move spends only
        when the GUI supplies neither a clock nor a node count, which in a
        tournament never happens; every timed search runs to `current +
        sim_cap` and stops on the clock. Sizing off `default_sims` therefore
        sizes for the one case that does not occur.

        The measurement, from the first pipe drive of the C11c ponder machine
        at `go ponder wtime 30000` + `ponderhit`, shipping defaults:

            ponder phase      4,999 sims -> 158,504 nodes   31.7 nodes/sim
            ponderhit phase  18,881 sims -> 599,995 nodes   31.8 nodes/sim

        600,000 is exactly `60 x (default_sims + ponder_max_sims)`, and the
        second row is it, to five digits: ONE ORDINARY PONDERHIT MOVE ON A 30 s
        CLOCK EXHAUSTED THE ARENA at the shipping defaults. That is the
        graceful-degradation path working, and it is not a state to ship in —
        the backstop is for the position that beats the estimate, not for the
        median move.

        The cost of the wider base is address space and nothing else; see
        `arena_bytes`, which the startup line prints so the trade is visible
        before a game rather than after one. Recorded in DECISIONS.md, C11c,
        because the brief specifies the formula and not which quantity feeds it.
        """
        return self.fixed_sims if self.fixed_sims is not None else self.sim_cap

    @property
    def ponder_max_sims_resolved(self) -> int:
        """The ponder simulation ceiling: `ponder_max_sims`, or the computed one.

            ponder_max_sims = sims_per_move / ponder_decay

        TIED TO THE DECAY, because the decay is the knob it interacts with.
        C8's inheritance decay exists so that a fresh search can overturn a
        verdict the ponder inherited, and that only works while the decayed
        inherited weight is comparable to the fresh budget:

            ponder_decay x ponder_max_sims  <=  sims_per_move

        which rearranges to the line above and holds with equality at the
        computed default. At the C8 default decay of 1.0 that is 1x
        `sims_per_move`; at the 0.25 the knob exists to be set to, 4x.

        The alternative — a hardcoded multiplier — cannot be tuned, cannot
        follow the deployment budget, and above all does not say WHY its
        multiplier is the one it is. `coupling_holds` asserts the inequality and
        `describe()` prints both sides.
        """
        if self.ponder_max_sims is not None:
            return self.ponder_max_sims
        return max(1, int(math.ceil(self.sims_per_move / self.ponder_decay)))

    @property
    def coupling_holds(self) -> bool:
        """Does `ponder_decay x ponder_max_sims <= sims_per_move` still hold?

        True by construction at the computed default. False is legal — an
        operator may pin `ponder_max_sims` above it deliberately — and is
        reported loudly rather than refused, because the consequence is a
        weaker search rather than a broken one, and refusing would make the
        knob unusable for the experiment that would justify changing it.
        """
        # A tolerance of one simulation, because the computed default rounds up.
        return (self.ponder_decay * self.ponder_max_sims_resolved
                <= self.sims_per_move + 1)

    @property
    def arena_nodes(self) -> int:
        """Nodes per arena: `arena_capacity`, or the computed default.

            arena_nodes = 60 x (sims_per_move + ponder_max_sims)

        60 NODES PER SIMULATION, from the measured curve and deliberately from
        its CONSERVATIVE end. Three points, mildly sublinear because more
        simulations revisit existing nodes rather than creating new ones:

            C8            2,000 sims ->    77,800 nodes   38.9 nodes/sim
            C8            8,000 sims ->   317,600 nodes   39.7 nodes/sim
            playtesting  37,000 sims -> <=1,200,000 nodes  <=32.4 nodes/sim

        Fitting the low-sim 40 and multiplying by 1.5 for branching variance
        gives 60, which predicts 1.48M at 37k sims against the 1.2M that was
        observed to be sufficient — 23% headroom where it was measured, and
        growing at larger budgets because of the sublinearity.

        DELIBERATELY NOT TUNED TIGHTLY. The asymmetry is total: over-provisioning
        costs tens of megabytes of address space, under-provisioning costs a
        game. Graceful degradation (`SearchOutcome.arena_exhausted`) is the
        backstop for the position that beats the estimate anyway, and
        `tools/bench_c11c_arena.py` reports the measured high-water against this
        prediction rather than trusting it.

        THE FLOOR IS `sim_cap`, NOT `sims_per_move`, AND THAT IS DELIBERATE.
        `sims_per_move` collapses to `fixed_sims` when one is pinned, so once
        `--sims` became `fixed_sims` (see `add_config_arguments`) a plain
        `--sims 4000` would have sized the arena at 60 x 8,000 = 480k nodes
        where it previously got `sim_cap`'s 7.2M. That is ample for ONE search
        — ~15.4 nodes/sim measured, so ~62k nodes at 4,000 sims — and is the
        wrong budget for an ACCUMULATING tree. DECISIONS.md's C12b entry is the
        record of exactly that mistake: a reuse arm running seven searches of
        20,000 new simulations on one tree needed 1,436,625 nodes, exhausted a
        1.2M arena sized off one move's allowance, and delivered 6,666 of
        20,000. `playv6 --selfplay` accumulates the same way.

        Taking the max keeps the sizing where it has always been for every
        invocation that does not pin `arena_capacity`, so the flag collapse
        changes no run's memory profile. The cost is over-provisioning a
        fixed-budget run — 536 MB at the shipping `sim_cap` regardless of the
        budget — which is a deliberate trade: at the concurrency this engine
        runs at, that is affordable, and the failure it buys off is a degraded
        game rather than a slow one.

        WHY THIS AND NOT `sims_per_move` ITSELF. `sims_per_move` also feeds
        `ponder_max_sims_resolved`, and pinning THAT to `sim_cap` would let a
        ponder in a `--sims 2000` run spend 60,000 simulations against the
        2,000 the next move gets — scope E6's over-commit defect, restored
        verbatim. The arena is the only consumer that wants the wider number,
        so the widening lives here and nowhere else: BOTH terms are recomputed
        as if no fixed budget were pinned, while the ceilings the SEARCH runs
        under stay tied to the budget actually asked for.
        """
        if self.arena_capacity is not None:
            return self.arena_capacity
        budget, ponder = self.arena_sizing_terms
        return 60 * (budget + ponder)

    @property
    def arena_sizing_terms(self) -> tuple[int, int]:
        """(budget, ponder ceiling) the arena is sized from. See `arena_nodes`.

        Separate from `sims_per_move` / `ponder_max_sims_resolved` — which are
        what the SEARCH is bounded by — and returned as a pair so `describe()`
        prints the formula it actually used rather than a second copy of it
        that can drift.

        An explicitly pinned `ponder_max_sims` is honoured: an operator who
        named that number meant it, and it is the ponder ceiling the search
        will really run under.
        """
        budget = max(self.sims_per_move, self.sim_cap)
        if self.ponder_max_sims is not None:
            return budget, self.ponder_max_sims
        return budget, max(1, int(math.ceil(budget / self.ponder_decay)))

    @property
    def arena_bytes(self) -> int:
        """Bytes BOTH ping-pong arenas reserve at `arena_nodes`.

        Both, because both are live: `apply_move` compacts the surviving subtree
        into the standby arena and swaps, so the peak occupancy is the moment
        the two overlap. Reporting one arena would understate the footprint by
        exactly a factor of two, and the startup line exists to be read by
        somebody deciding whether a number is absurd.

        The per-node width comes from the core rather than from a constant here:
        the SoA arena's field widths are its business and a duplicated 40 would
        be wrong the first time an accumulator changed.
        """
        per_node = (guofish_core.NodeArenaQ32.bytes_per_node
                    + 4    # parent_, one uint32 per node
                    + 2)   # raw_move_, one uint16 per node
        return 2 * per_node * self.arena_nodes

    @property
    def in_flight(self) -> int:
        """K = max_outstanding / threads, at least 1.

        Integer division with a floor rather than a round: K*W must not EXCEED
        the outstanding count the caller asked for, because that count is the
        virtual-loss exposure and the batch ceiling at once. `effective_outstanding`
        reports what the division actually produced so the log can show both.
        """
        return max(1, self.max_outstanding // max(1, self.threads))

    @property
    def effective_outstanding(self) -> int:
        return self.threads * self.in_flight

    # --- validation -------------------------------------------------------

    def validate(self) -> "EngineConfig":
        """Refuse anything the engine cannot run with. Returns self."""
        for name, (only, why) in UNSUPPORTED_IN_CORE.items():
            value = getattr(self, name)
            if float(value) != only:
                raise ConfigError(
                    f"{name}={value!r} is not implemented by the C++ core "
                    f"(only {only!r} is accepted). {why}")

        if self.threads < 1:
            raise ConfigError(f"threads must be >= 1, got {self.threads}")
        if self.max_outstanding < 1:
            raise ConfigError(f"max_outstanding must be >= 1, got {self.max_outstanding}")
        if self.max_outstanding < self.threads:
            raise ConfigError(
                f"max_outstanding ({self.max_outstanding}) is below threads "
                f"({self.threads}); every search thread needs at least one "
                f"in-flight path, so W*K could not reach the count asked for.")
        if self.max_batch < 1:
            raise ConfigError(f"max_batch must be >= 1, got {self.max_batch}")
        # atanh's argument is bounded; the SCALE multiplying it is not, but it
        # must be a positive, finite number or every reported score is nonsense.
        # 0 would report every position as 0.00 regardless of Q; a negative would
        # report a won position as lost, which is worse than either.
        if self.value_scale is not None and not self.value_scale > 0.0:
            raise ConfigError(
                f"value_scale must be > 0 (score cp = value_scale * atanh(q)), "
                f"got {self.value_scale}")
        if self.affinity not in guofish_core.AFFINITY_POLICIES:
            raise ConfigError(
                f"affinity={self.affinity!r} is not one of "
                f"{list(guofish_core.AFFINITY_POLICIES)}")
        if self.max_tree_depth < 1:
            raise ConfigError(f"max_tree_depth must be >= 1, got {self.max_tree_depth}")
        if self.cache_entries < 0:
            raise ConfigError(f"cache_entries must be >= 0, got {self.cache_entries}")
        # C11c. `None` is "compute it" and is the default; a pinned value is
        # still refused below 1024. The floor is not arbitrary: a position has
        # at most 218 legal moves, and `expand_root` is the ONE expansion that
        # may not degrade gracefully — a root with no children cannot produce a
        # move at all — so the arena has to be able to hold the root plus its
        # widest possible child block with room to spare.
        if self.arena_capacity is not None and self.arena_capacity < 1024:
            raise ConfigError(
                f"arena_capacity must be >= 1024, got {self.arena_capacity}")
        if self.ponder_max_sims is not None and self.ponder_max_sims < 1:
            raise ConfigError(
                f"ponder_max_sims must be >= 1, got {self.ponder_max_sims}")
        if not (0.0 < self.ponder_decay <= 1.0):
            raise ConfigError(
                f"ponder_decay must lie in (0, 1], got {self.ponder_decay}")
        # The reference validates FPU the same way: it is a Q, and a Q outside
        # [-1, 1] is not a value the backup can ever produce.
        for name in ("fpu_root", "fpu_tree"):
            value = getattr(self, name)
            if not (-1.0 <= value <= 1.0):
                raise ConfigError(f"{name} must lie in [-1, 1], got {value}")
        if self.c_puct_base <= 0.0:
            raise ConfigError(f"c_puct_base must be > 0, got {self.c_puct_base}")
        if self.c_puct_init < 0.0:
            raise ConfigError(f"c_puct_init must be >= 0, got {self.c_puct_init}")
        if self.c_puct_factor <= 0.0:
            raise ConfigError(f"c_puct_factor must be > 0, got {self.c_puct_factor}")
        if self.virtual_loss < 0.0:
            raise ConfigError(f"virtual_loss must be >= 0, got {self.virtual_loss}")
        # It DIVIDES the logits, so 0 is a division by zero and a negative is an
        # inverted policy. The reference validates it with the same
        # parenthetical (SearchParams.__post_init__) and so does the C++
        # ReplaySearch constructor; this is the third of three, and it is the
        # one that turns the failure into a `setoption` refusal rather than an
        # exception out of the first search.
        if not self.policy_temperature > 0.0:
            raise ConfigError(
                f"policy_temperature must be > 0 (it divides the logits), got "
                f"{self.policy_temperature}")
        if self.default_sims < 1:
            raise ConfigError(f"default_sims must be >= 1, got {self.default_sims}")
        if self.sim_cap < 1:
            raise ConfigError(f"sim_cap must be >= 1, got {self.sim_cap}")
        if self.fixed_sims is not None and self.fixed_sims < 1:
            raise ConfigError(f"fixed_sims must be >= 1, got {self.fixed_sims}")
        if self.slice_seconds <= 0.0:
            raise ConfigError(f"slice_seconds must be > 0, got {self.slice_seconds}")
        if self.min_slice_sims < 1:
            raise ConfigError(f"min_slice_sims must be >= 1, got {self.min_slice_sims}")
        if self.move_overhead_ms < 0:
            raise ConfigError(
                f"move_overhead_ms must be >= 0, got {self.move_overhead_ms}")
        # A negative seed is not "0 but signed": 0 has a specific meaning here
        # (deterministic, highest weight) and anything else is an RNG seed, so
        # the only value worth refusing is one that reads as a typo.
        if self.book_seed < 0:
            raise ConfigError(
                f"book_seed must be >= 0 (0 means 'always the highest-weight "
                f"book move'), got {self.book_seed}")
        return self

    # --- the core objects -------------------------------------------------

    def to_search_config(self) -> "guofish_core.SearchConfig":
        config = guofish_core.SearchConfig(
            c_init=self.c_puct_init,
            c_base=self.c_puct_base,
            c_factor=self.c_puct_factor,
            fpu_root=self.fpu_root,
            fpu_tree=self.fpu_tree,
            virtual_loss=self.virtual_loss,
            max_tree_depth=self.max_tree_depth,
            # C11c. The RESOLVED value, not the field: `None` means "compute it
            # from the budget" and the core takes a number. See `arena_nodes`.
            arena_capacity=self.arena_nodes,
            cache_slots=self.cache_entries,
            ponder_decay=self.ponder_decay,
            verify_compaction=self.verify_compaction,
            policy_temperature=self.policy_temperature,
        )
        # 0 means "whatever the core's default is". Setting it explicitly to 0
        # would be a cache with no shards, which the core refuses.
        if self.cache_shards:
            config.cache_shards = self.cache_shards
        return config

    def to_parallel_config(self, *, collect_histograms: bool = False):
        """`collect_histograms` defaults OFF here and ON in the benchmarks.

        It is a few thousand int64s per search and it is measurement, not
        engine behaviour; a game does not read it.
        """
        return guofish_core.ParallelConfig(
            workers=self.threads,
            in_flight=self.in_flight,
            max_batch=self.max_batch,
            affinity=self.affinity,
            collect_histograms=collect_histograms,
        )

    # --- reporting --------------------------------------------------------

    def as_dict(self) -> dict:
        """Every declared field plus the derived ones, JSON-ish and ordered."""
        out: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = str(value) if isinstance(value, Path) else value
        out["in_flight"] = self.in_flight
        out["effective_outstanding"] = self.effective_outstanding
        # C11c. THE RESOLVED VALUES BESIDE THE DECLARED ONES. `arena_capacity=None`
        # and `ponder_max_sims=None` say what was ASKED FOR; these say what the
        # engine will actually run with, which is the only one of the two a
        # benchmark artifact can be reconciled against. Same discipline as
        # `effective_outstanding` above.
        out["sims_per_move"] = self.sims_per_move
        out["ponder_max_sims_resolved"] = self.ponder_max_sims_resolved
        out["arena_nodes"] = self.arena_nodes
        out["arena_mb"] = round(self.arena_bytes / (1024 * 1024), 1)
        out["coupling_holds"] = self.coupling_holds
        return out

    def as_kv(self) -> str:
        """The whole resolved configuration on ONE line, `key=value` separated.

        This is the machine-readable half of the config log and the thing C11's
        validation greps: a value that reached the engine appears here, and a
        value that did not, does not. Kept on one line on purpose — a smoke run
        produces one of these per `isready` and they have to be diffable.
        """
        parts = []
        for key, value in self.as_dict().items():
            if isinstance(value, float):
                parts.append(f"{key}={value:g}")
            elif isinstance(value, bool):
                parts.append(f"{key}={'true' if value else 'false'}")
            else:
                parts.append(f"{key}={value}")
        return " ".join(parts)

    def describe(self) -> list[str]:
        """The same thing grouped, for a human reading a Cutechess debug log."""
        return [
            f"[config] {self.as_kv()}",
            f"[config] model      : {self.model_path or DEFAULT_MODEL} "
            f"value_scale="
            f"{f'checkpoint, else {LEGACY_VALUE_SCALE:.4f}' if self.value_scale is None else f'{self.value_scale:.4f} (OVERRIDE)'}",
            f"[config] evaluator  : max_batch={self.max_batch} graphs={self.graphs} "
            f"compile={self.compile} pin={self.pin} "
            f"switch_interval={self.switch_interval:g}"
            f"{'' if self.compile else ' (EAGER NUMERICS BASELINE)'}",
            f"[config] parallel   : threads(W)={self.threads} in_flight(K)={self.in_flight} "
            f"max_outstanding={self.max_outstanding} "
            f"(effective W*K={self.effective_outstanding}) affinity={self.affinity}",
            f"[config] exploration: c_puct_init={self.c_puct_init:g} "
            f"c_puct_base={self.c_puct_base:g} c_puct_factor={self.c_puct_factor:g} "
            f"fpu_root={self.fpu_root:g} fpu_tree={self.fpu_tree:g}",
            f"[config] search     : virtual_loss={self.virtual_loss:g} "
            f"max_tree_depth={self.max_tree_depth} "
            f"policy_temperature={self.policy_temperature:g}"
            f"{'' if self.policy_temperature == 1.0 else ' (NON-IDENTITY: priors are '
               'being sharpened/flattened at the root and at every interior node)'} "
            f"dirichlet_epsilon={self.dirichlet_epsilon:g} "
            f"dirichlet_alpha={self.dirichlet_alpha:g}",
            # C11c. THE LINE THE BRIEF REQUIRES TO BE READABLE AT A GLANCE, and
            # the reason it carries the whole derivation rather than just the
            # answer: `arena: 2 x 1,500k nodes (111.6 MB)` is checkable by
            # somebody who knows neither the formula nor the budget, and a
            # number that is absurd or that is tiny should be catchable by
            # reading the launch line rather than by watching a game go wrong.
            f"[config] memory     : cache_entries={self.cache_entries} "
            f"cache_shards={self.cache_shards or 'default'} "
            f"arena: 2 x {self.arena_nodes:,} nodes "
            f"({self.arena_bytes / (1024 * 1024):.1f} MB total) "
            f"[{'explicit' if self.arena_capacity is not None else f'computed 60 x ({self.arena_sizing_terms[0]:,} + {self.arena_sizing_terms[1]:,})'}]"
            # The arena COMMITS rather than reserves — measured at 268 MB RSS
            # for one 7.2M-node arena — so a footprint this size is real memory
            # and not address space. Warned rather than refused: it is a legal
            # configuration and the operator may have meant it, but a
            # PonderDecay of 0.25 quadruples the ponder cap and therefore this
            # number, which is not obvious from the option that caused it.
            + (f"  *** WARNING: {self.arena_bytes / (1024 ** 3):.2f} GB is a "
               f"lot of committed memory. It is real RSS, not reserved address "
               f"space. Lower SimCap, raise PonderDecay, or pin ArenaCapacity "
               f"explicitly. ***" if self.arena_bytes > (1 << 30) else ""),
            # C11c. BOTH VALUES ON ONE LINE, WITH THE INEQUALITY THEY HAVE TO
            # SATISFY. Setting the ponder cap and the decay independently is how
            # the reference reached 60,000 inherited simulations against 2,000
            # fresh ones — scope E6's over-commit defect, which this port is
            # deliberately fixing rather than reproducing — and the defence
            # against re-introducing it is that a mismatched pair is visible
            # before a game rather than inferred after one.
            f"[config] ponder     : ponder={self.ponder} "
            f"sims_per_move={self.sims_per_move:,} "
            f"ponder_max_sims={self.ponder_max_sims_resolved:,} "
            f"[{'explicit' if self.ponder_max_sims is not None else 'computed sims_per_move/decay'}] "
            f"ponder_decay={self.ponder_decay:g} "
            f"coupling: decay x ponder_max_sims = "
            f"{self.ponder_decay * self.ponder_max_sims_resolved:,.0f} "
            f"{'<=' if self.coupling_holds else '>'} sims_per_move "
            f"{self.sims_per_move:,} "
            f"{'OK' if self.coupling_holds else 'VIOLATED: a ponder can out-weigh the fresh search it hands its tree to'}"
            # C12c. WHAT THIS LINE CANNOT KNOW, said on the line itself.
            #
            # Every number above derives from `sims_per_move`, i.e. from
            # `fixed_sims or sim_cap` — a value fixed before any `go` arrived.
            # A GUI that sends `go nodes N` every move (lichess-bot with
            # `go_commands.nodes`, cutechess with `nodes=`) is stating a smaller
            # per-move budget than `sim_cap`, and the UCI layer re-derives the
            # ponder ceiling from it per move as `N / ponder_decay`.
            #
            # So this line is the CEILING and the `[go] ponder:` line is what
            # was actually spent. Saying so here is the difference between a
            # startup record that is wrong and one that is bounded — and it is
            # the same discipline as reporting delivered rather than requested
            # simulations one layer down.
            + f"  (a `go nodes N` lowers this at runtime to N/decay = "
              f"N/{self.ponder_decay:g}; the [go] ponder: line reports what "
              f"was used)",
            f"[config] budget     : default_sims={self.default_sims} "
            f"sim_cap={self.sim_cap} fixed_sims={self.fixed_sims} "
            f"slice_seconds={self.slice_seconds:g} "
            f"move_overhead_ms={self.move_overhead_ms}",
            # `ponder` and `ponder_decay` moved to the `[config] ponder` line
            # above at C11c, where they sit beside the two quantities they are
            # coupled to. They are not repeated here: two lines carrying the
            # same field is how the two stop agreeing.
            f"[config] host       : verify_compaction={self.verify_compaction}",
            # THE LINE A BENCHMARK ARTIFACT HAS TO CARRY. Both features bypass
            # MCTS, so a strength or throughput number measured with either on
            # is measuring something other than this port's search — and the
            # only defence against discovering that afterwards is that the
            # resolved state is in the run's own log, every time.
            f"[config] book       : use_book={self.use_book} "
            f"path={self.book_target} "
            f"seed={self.book_seed}"
            f"{' (DETERMINISTIC: always the highest-weight entry)' if self.book_seed == 0 else ' (weighted-random)'}",
            f"[config] syzygy     : use_syzygy={self.use_syzygy} "
            f"path={self.syzygy_target}",
            "[config] NOT IMPLEMENTED BY THE CORE, accepted only at the value shown: "
            + ", ".join(f"{name}={only:g}" for name, (only, _) in
                        sorted(UNSUPPORTED_IN_CORE.items())),
        ]

    def replace(self, **changes) -> "EngineConfig":
        """A validated copy. `dataclasses.replace` runs __post_init__, so a
        rejected change raises here rather than producing a bad object."""
        return dataclasses.replace(self, **changes)


# ---------------------------------------------------------------------------
# One move's telemetry
# ---------------------------------------------------------------------------


@dataclass
class SearchOutcome:
    """What one `Engine.search` did, with the two sim counts kept apart.

    `delivered` is what the core actually backed up into the root, summed over
    the slices, and it is what `sims_per_s` divides.

    `nominal` is the SIMULATION COUNT THE CALLER ASKED FOR — `go nodes N`'s N,
    the number v5 printed as `nodes` and divided by the wall clock — and it is
    None when the caller did not ask in simulations at all. On a fresh root the
    two agree to within the root's own seed visit; after a tree reuse promoted
    3,564 of the 4,000 visits the engine does 436 simulations and v5 would still
    have reported 4,000, a 9x overstatement. `inflation` is that ratio, printed
    on every move so it can never again be inferred after the fact.

    NONE FOR A TIMED SEARCH, and that is the point of the field being optional.
    A `go wtime ... btime ...` asks for a duration, not a count; the node budget
    the engine runs under is a CEILING it chose (`root_visits + sim_cap`), and
    dividing that ceiling by the wall clock produces a 20x "inflation" that
    describes nothing. Reporting `n/a` there is the same discipline as reporting
    delivered sims everywhere else: a number nobody asked for is not a
    measurement.
    """

    best_move: Optional[str]
    mating_move: Optional[str]
    nominal: Optional[int]
    inherited: int
    delivered: int
    wall_s: float
    slices: int
    root_visits: int
    score_cp: int
    q: float
    pv: list[str] = field(default_factory=list)
    depth: int = 0
    max_depth: int = 0
    nodes: int = 0
    hashfull: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    eval_rows: int = 0
    eval_batches: int = 0
    stopped: bool = False
    reason: str = ""
    # What set the budget: "nodes", "fixed", "default", "time", "infinite" or
    # "ponder". Logged so a reader knows without inference why `nominal` is a
    # number in one line and `n/a` in the next.
    budget_source: str = "nodes"

    # C11b. WHAT DECIDED THE MOVE: "search", "book" or "tablebase".
    #
    # The other two mean MCTS never ran. That is not a detail of provenance —
    # it is the difference between a move this port computed and a move it
    # looked up, and a bypassed move is one where this engine and the Python
    # reference are identical by construction. Gate 5 exists to measure where
    # they are not, so a bypassed move dilutes precisely the signal it wants.
    # Carrying the source on every outcome is what makes an accidentally-on
    # book self-revealing in the run's own output rather than a mystery in the
    # ELO afterwards.
    source: str = "search"

    # --- C11c: pondering ----------------------------------------------------
    #
    # `delivered` KEEPS ITS MEANING — what THIS search_move call backed up — and
    # `ponder_sims` is what a preceding ponder phase contributed to the same
    # tree. They are never merged into one figure, which is the delivered-sims
    # discipline that has held since C11 applied to the one new way of splitting
    # the work.
    #
    # WHY THEY MUST STAY APART. Budget accounting starts at the `ponderhit`
    # instant: ponder simulations ran on the OPPONENT's clock and are a bonus,
    # not a draw against this move's allocation. A merged count would make the
    # engine look 2-3x faster than it is on a hit and would make its sims/s
    # incomparable with a ponder-off run — which is exactly the arm Gate 5 has
    # to be measured against, since the 2687.7 anchor ran with ponder off.
    ponder_sims: int = 0
    ponder_wall_s: float = 0.0

    # --- C11c: graceful arena exhaustion ------------------------------------
    #
    # `arena_exhausted` is what makes `delivered < requested` a legitimate
    # outcome rather than a broken row. A benchmark harness must consult it and
    # REJECT the row; a game log must print it loudly, because a silent
    # degradation that only shows up as unexplained weakness is worse than a
    # crash — nobody investigates a crash's absence.
    arena_exhausted: bool = False
    arena_exhausted_at: int = 0
    arena_high_water: int = 0
    arena_capacity: int = 0
    # Whether the C++ mutable deadline, rather than the Python slice loop,
    # ended the last slice. On a ponderhit-converted search this is the normal
    # ending and its absence is the interesting case.
    deadline_hit: bool = False

    @property
    def search_sims(self) -> int:
        """Simulations delivered SINCE the ponderhit. An alias for `delivered`.

        Named because the brief names it, and because `delivered` beside
        `ponder_sims` reads ambiguously in a report while `search_sims` beside
        `ponder_sims` does not.
        """
        return self.delivered

    @property
    def total_sims(self) -> int:
        """Every simulation in the tree this move was decided from.

        Reported ALONGSIDE the two, never instead of them: it is the right
        number for "how much search backs this move" and the wrong one for
        every throughput figure.
        """
        return self.ponder_sims + self.delivered

    @property
    def arena_utilisation(self) -> float:
        """Peak arena occupancy as a fraction of capacity. 0.0 when unknown.

        The number the C11c sizing verification reports: the formula predicts
        `arena_nodes` and this says what was actually needed, so a safety factor
        nobody has seen engage stops being an assumption.
        """
        if self.arena_capacity <= 0:
            return 0.0
        return self.arena_high_water / self.arena_capacity

    @property
    def bypassed(self) -> bool:
        """True when MCTS did not run. `delivered` is 0 and `sims_per_s` is 0/0.

        Every aggregate over a set of moves must filter on this rather than
        averaging a zero-simulation move in — see `aggregate_sims_per_s`.
        """
        return self.source != "search"

    @property
    def sims_per_s(self) -> float:
        """DELIVERED simulations per second. The honest rate (C11's objective).

        Zero on a bypassed move, and that zero is not a slow move: it is the
        absence of a measurement. `bypassed` is how a caller tells the two
        apart.
        """
        return self.delivered / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def nominal_sims_per_s(self) -> Optional[float]:
        """The requested count divided by the wall clock — v5's number.

        Kept for comparison and never the headline. Equals `sims_per_s` only
        when nothing was inherited. None when nothing was requested in
        simulations.
        """
        if self.nominal is None:
            return None
        return self.nominal / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def inflation(self) -> Optional[float]:
        """How much the requested count overstates the delivered one.

        1.0 on a fresh root; v5's per-game average sat at 2.3-2.9 here without
        saying so, and an individual deep-reuse move can be far higher. A move
        that delivered NOTHING — the reused root already met the budget —
        reports `inf` rather than 1.0, because 1.0 would say the two agreed when
        in fact the request was reported in full against zero work. None when
        the caller asked for a duration rather than a count.
        """
        if self.nominal is None:
            return None
        return self.nominal / self.delivered if self.delivered else float("inf")

    def telemetry(self) -> str:
        """One line, stderr, per move. The provenance record for Gate 4/5."""
        nominal = "n/a" if self.nominal is None else str(self.nominal)
        nominal_rate = ("n/a" if self.nominal_sims_per_s is None
                        else f"{self.nominal_sims_per_s:,.0f}")
        inflation = ("n/a" if self.inflation is None
                     else f"{self.inflation:.2f}x")
        if self.bypassed:
            # A bypassed move has no delivered sims, no inflation, no PV depth
            # and no cache traffic worth reporting, and printing the search
            # fields as zeros would read as a search that did nothing rather
            # than as a search that did not happen. Different line, same prefix.
            return (f"[search] source={self.source} BYPASS (MCTS did not run) "
                    f"best={self.best_move} wall={self.wall_s * 1000:.1f}ms "
                    f"delivered=0 nominal={nominal} "
                    f"budget_source={self.budget_source} reason={self.reason}")
        # C11c. The two sim counts are always both present and are never added
        # together. `ponder_sims=0` on a move that was not pondered is a zero
        # that means what it says.
        ponder = (f"ponder_sims={self.ponder_sims} search_sims={self.search_sims} "
                  f"total_sims={self.total_sims} "
                  f"ponder_wall={self.ponder_wall_s * 1000:.1f}ms ")
        # LOUD, and only when it fired. Scope: a degraded move must be
        # investigable from the run's own log without anybody having gone
        # looking for it first.
        degraded = ("" if not self.arena_exhausted else
                    f"ARENA_EXHAUSTED at={self.arena_exhausted_at} "
                    f"(delivered {self.delivered} of the budget; this move is "
                    f"NOT admissible as a benchmark row) ")
        return (f"[search] source={self.source} delivered={self.delivered} "
                f"nominal={nominal} "
                f"inherited={self.inherited} slices={self.slices} "
                f"wall={self.wall_s * 1000:.1f}ms "
                f"{ponder}{degraded}"
                f"delivered_sims_per_s={self.sims_per_s:,.0f} "
                f"nominal_sims_per_s={nominal_rate} "
                f"inflation={inflation} "
                f"root_visits={self.root_visits} nodes={self.nodes} "
                f"arena_hw={self.arena_high_water}/{self.arena_capacity} "
                f"({self.arena_utilisation:.1%}) "
                f"eval_rows={self.eval_rows} eval_batches={self.eval_batches} "
                f"cache_hits={self.cache_hits} cache_misses={self.cache_misses} "
                f"best={self.best_move} score_cp={self.score_cp} "
                f"budget_source={self.budget_source} "
                f"deadline_hit={self.deadline_hit} "
                f"stopped={self.stopped} reason={self.reason}")


def aggregate_sims_per_s(outcomes: Iterable[SearchOutcome]) -> tuple[float, dict]:
    """(delivered sims/s over the SEARCHED moves, a per-source tally).

    THE ONE PLACE THE EXCLUSION IS IMPLEMENTED, so that no caller has to
    remember it. A book or tablebase move delivers zero simulations in some
    nonzero wall time, so folding it into a throughput mean drags the mean down
    by however many moves the book happened to cover — a number that has
    nothing to do with the engine's speed and everything to do with which
    opening was played.

    Both halves are returned together on purpose: a rate without the counts
    beside it is a number whose denominator nobody can check.
    """
    outcomes = list(outcomes)
    counts = {source: 0 for source in DECISION_SOURCES}
    for outcome in outcomes:
        counts[outcome.source] = counts.get(outcome.source, 0) + 1
    searched = [o for o in outcomes if not o.bypassed]
    delivered = sum(o.delivered for o in searched)
    wall = sum(o.wall_s for o in searched)
    counts["moves"] = len(outcomes)
    counts["searched"] = len(searched)
    counts["bypassed"] = len(outcomes) - len(searched)
    counts["excluded_wall_s"] = sum(o.wall_s for o in outcomes if o.bypassed)
    return (delivered / wall if wall > 0 else 0.0), counts


# Replies below this share of the root's child visits are pruned from the ponder
# tables. DISPLAY ONLY — every arithmetic caller (`inherited_by_next_search`,
# `gained`, the transfer rate) reads the unpruned list, because the move a human
# or a GUI actually plays is frequently a long-shot and its inherited visits are
# exactly what the next search starts from.
#
# 0.20 is the operator's number. It is also self-limiting in a useful way: at
# most five replies can hold a 20% share, so the table cannot run long and the
# `top_n` cap effectively never binds at this threshold.
PONDER_MIN_SHARE = 0.20


def move_label(board, uci: Optional[str]) -> str:
    """SAN for `uci` in `board`, falling back to the UCI string.

    SAN BECAUSE THESE LINES ARE READ BY A PERSON. `g1f3` and `Nf3` name the same
    move and only one of them can be recognised at a glance in a list of six.
    The reference did the same thing and for the same reason (core/mctsv4.py's
    `_san` helper, used in every ponder line it printed).

    Total, never raising: the caller is emitting a diagnostic, and a diagnostic
    that can throw is worse than one that degrades to coordinates. A board that
    is None, a move that is illegal in it, or a python-chess that is not
    importable all fall back rather than fail.
    """
    if uci is None:
        return "none"
    if board is None:
        return uci
    try:
        import chess
        return board.san(chess.Move.from_uci(uci))
    except Exception:                                            # noqa: BLE001
        return uci


def format_root_branches(branches: list[tuple[str, int, float]], *,
                         predicted: Optional[str] = None,
                         played: Optional[str] = None,
                         before: Optional[dict] = None,
                         board=None,
                         top_n: int = 5,
                         min_share: float = PONDER_MIN_SHARE,
                         prefix: str = "[ponder]") -> list[str]:
    """The branch table, one line per reply. See `Engine.root_branches`.

    A TABLE RATHER THAN THE REFERENCE'S INLINE LIST, and that is the whole of
    the change. core/mctsv4.py printed
    `branches=[Nf3(32w,1640v)]` and `branch_visits=[Nf3:2140v, e5:830v]` beside
    a separate `actual_visits=2140`: three encodings of the same quantity, two
    of them packing two numbers into one parenthesis behind single-letter
    suffixes, and the one number that matters — what the NEXT search inherits —
    repeated at the end where it reads as a fourth unrelated field. Columns with
    headings, one row per reply, and the two roles marked in place instead.

    `predicted` is the reply the engine expected; `played` is the one that
    actually arrived. They are marked separately because a hit is exactly the
    case where they coincide, and a table that could only show one of them
    would be unable to say so.

    `before` is `{uci: visits}` from the same call taken earlier, and adding it
    turns on a `gained` column. THAT COLUMN IS THE ONE THAT ANSWERS THE
    QUESTION. A branch holding 2,140 visits has told you nothing about the
    ponder until you know whether 1,890 of them or 40 of them arrived during
    it; the reference printed only the total and left the reader to remember
    what it had been.
    """
    if not branches:
        return [f"{prefix}   (no visited replies - the ponder built nothing)"]
    # MATCHING STAYS ON UCI, DISPLAY BECOMES SAN. `predicted` and `played` are
    # coordinates because that is what the tree and the callers hold; rendering
    # is the last step, so an unlabellable move degrades to coordinates in one
    # column instead of silently failing to match.
    # PRUNE AND TRUNCATE FOR DISPLAY ONLY, and never drop the played move. A
    # reply below the threshold is common — a human plays one often — and its
    # inherited visits are exactly what the next search starts from, so if it
    # was pruned it is pulled back on rather than being silently reported as
    # unvisited. That is the bug a real game caught, where a branch holding 207
    # visits at a 0% share was called absent while the engine printed
    # `reused: 207` on the very next line.
    kept = [row for row in branches if row[2] >= min_share]
    if not kept:
        # A flat distribution where nothing clears the bar. An empty table is
        # strictly less useful than the leader, so the leader always shows.
        kept = branches[:1]
    shown = kept[:top_n]
    elided = len(branches) - len(shown)
    if played is not None and all(uci != played for uci, _, _ in shown):
        pulled = [row for row in branches if row[0] == played]
        if pulled:
            shown = shown + pulled
            elided -= 1

    labels = {uci: move_label(board, uci) for uci, _, _ in shown}
    # At least as wide as the heading, or the header row runs a character long
    # and every column below it is off by one.
    width = max(5, max(len(label) for label in labels.values()))
    header = f"{prefix}   {'reply':<{width}}  {'visits':>9}  {'share':>6}"
    if before is not None:
        header += f"  {'gained':>8}"
    lines = [header]
    for uci, visits, share in shown:
        marks = []
        if predicted is not None and uci == predicted:
            marks.append("predicted")
        if played is not None and uci == played:
            marks.append("PLAYED -> inherited by the next search")
        note = f"   <- {', '.join(marks)}" if marks else ""
        row = (f"{prefix}   {labels[uci]:<{width}}  {visits:>9,}  {share:>6.0%}")
        if before is not None:
            row += f"  {visits - before.get(uci, 0):>+8,}"
        lines.append(row + note)
    if elided > 0:
        why = (f" below the {min_share:.0%} share threshold" if min_share > 0
               else "")
        lines.append(f"{prefix}   ... and {elided:,} more visited "
                     f"{'reply' if elided == 1 else 'replies'}{why}")
    # Tested against the FULL list, not the displayed one. This is the
    # reference's near-zero case and it has to mean what it says: the played
    # move genuinely received no visits, so the next search starts cold.
    if played is not None and all(uci != played for uci, _, _ in branches):
        lines.append(f"{prefix}   {move_label(board, played)} received NO "
                     f"visits: the ponder built nothing on the line that was "
                     f"played, and the next search starts from an unvisited "
                     f"branch")
    return lines


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


class Engine:
    """`guofish_core.ReplaySearchQ32` plus the live evaluator, driven in slices.

    WHY THE SEARCH IS SLICED. `search_parallel(N)` runs to N ROOT VISITS and
    returns; there is no stop flag in the core (`aborted_` is set by
    `record_error` and by nothing else) and no clock in the dispatcher, by C9's
    design. A UCI engine must answer `stop` and must respect a move time, so the
    budget is cut into slices and the flags are read between them. Each slice is
    a fresh `search_parallel` at a HIGHER absolute target, which the core already
    handles: `target_ = num_simulations - existing`, the same arithmetic tree
    reuse goes through. The cost is W+1 thread spawns per slice, which at the
    shipping W=1 and a 50 ms slice is two threads every 50 ms.

    The tree, the arena and the transposition cache all survive slicing
    untouched, so a sliced search and a single-call search of the same budget
    differ only in where the dispatcher's drain boundaries fall.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.evaluator = None
        self.search = None
        self.value_scale: Optional[float] = None
        # Which of `_resolve_value_scale`'s three sources supplied it: "config",
        # "checkpoint" or "legacy-default". On the [init] line so a run's own log
        # says whether its scale was recorded or assumed.
        self.value_scale_source: str = "unresolved"
        self.model = None
        self.device = None
        self._ready = False
        # The position the tree is rooted at, as the UCI layer described it.
        self._base_fen: Optional[str] = None
        self._moves: list[str] = []
        # A python-chess mirror of the rooted position, maintained by
        # `set_position`. The bypasses need a board — Polyglot keys it and
        # `tablebase_root_move` takes its FEN — and reconstructing one per move
        # from `_base_fen` + `_moves` would replay the whole game every ply.
        self._board = None

        # --- C11b: the two host-side readers -------------------------------
        # Opened ONCE, in `ensure_ready`. Not in `__init__`, so a GUI can send
        # every `setoption` first; not in `new_game`, because the Polyglot
        # reader is memory-mapped and reopening it per game is pure cost for no
        # benefit — see `new_game`.
        self.book = None
        self.book_rng: Optional[random.Random] = None
        self.tablebase = None
        # What actually happened when they were opened, as a short string for
        # the config log. "off", "open <path>", "missing <path>" or
        # "error <path>: ...". This is the RESOLVED state a benchmark artifact
        # has to record; the config alone only says what was asked for.
        self.book_state = "off"
        self.syzygy_state = "off"
        # Per-game tally of what decided each move. Reset by `new_game`.
        self.decision_counts = {source: 0 for source in DECISION_SOURCES}

    # --- lifecycle --------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    def ensure_ready(self) -> None:
        """Load the checkpoint, capture the graphs, build the search. Idempotent.

        Everything expensive happens here rather than in `__init__` so that a GUI
        can send every `setoption` before the first `isready` — which is what
        GUIs do, and what makes the configuration logged here the RESOLVED one
        rather than the command line's.
        """
        if self._ready:
            return

        import torch
        from playing.v6 import evaluator as live_evaluator

        cfg = self.config
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. The v6 surface runs the graphed CUDA "
                "evaluator; there is no CPU path that has been measured.")

        build = guofish_core.build_info()
        if build["asan"] or build["ubsan"]:
            err(f"[init] WARNING: instrumented build {build}. Any timing this "
                f"process reports is a sanitizer's timing, not the engine's.")

        # load_model prints to stdout, which is the UCI stream. Capture it.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.model, self.device = live_evaluator.load_default_model(
                cfg.model_path or DEFAULT_MODEL)
        for line in buf.getvalue().splitlines():
            err(f"[load_model] {line}")

        self.value_scale = self._resolve_value_scale()

        self.evaluator = live_evaluator.TorchEvaluator(
            self.model, self.device, cfg.max_batch,
            switch_interval=cfg.switch_interval, pin=cfg.pin, graphs=cfg.graphs,
            compile=cfg.compile)
        self.search = guofish_core.ReplaySearchQ32(cfg.to_search_config())
        self.search.set_evaluator(self.evaluator.core)
        self._ready = True

        # C11b. Both readers open here and nowhere else. `open_readers` is
        # idempotent and is also what a path change calls, so there is one
        # implementation of "open what the config asks for and say what
        # happened".
        self.open_readers()

        for line in cfg.describe():
            err(line)
        err(f"[init] build={build} device={torch.cuda.get_device_name(0)} "
            f"torch={torch.__version__}")
        err(f"[init] switch_interval now {sys.getswitchinterval():g} "
            f"(evaluator reports before={self.evaluator.switch_interval_before:g} "
            f"after={self.evaluator.switch_interval:g})")
        if self.evaluator.graph_report is not None:
            err(f"[init] capture={self.evaluator.graph_report.describe()}")
        else:
            err("[init] capture=eager (graphs disabled)")
        err(f"[init] pinned={self.evaluator.pinned} "
            f"topology={guofish_core.ReplaySearchQ32.topology()['source']}")
        # WHICH GENERATION ACTUALLY LOADED, named rather than inferred. The
        # architecture is chosen from the checkpoint's own metadata, so the only
        # way a reader knows whether a run was the v5 student or a legacy
        # guofish2..guofish4 net is if the engine says so — and a v4-vs-v5 match
        # is exactly the comparison where getting that backwards would invalidate
        # the result. `seq_length` is on the line because it is the one contract
        # both generations advertise and `require_engine_contract` checks.
        parameters = sum(p.numel() for p in self.model.parameters())
        err(f"[init] architecture={type(self.model).__name__} "
            f"params={parameters / 1e6:.1f}M seq_length={self.model.seq_length} "
            f"policy_size={guofish_core.POLICY_SIZE} "
            f"value_scale={self.value_scale:.4f} source={self.value_scale_source}")
        err(f"[eval] score cp = {self.value_scale:.4f} * atanh(q), saturating at "
            f"+-{q_to_centipawns(1.0, self.value_scale)} cp "
            f"(|q| clamped to {VALUE_CLAMP})")

    def _resolve_value_scale(self) -> float:
        """The constant `q_to_centipawns` inverts, and where it came from.

        Three sources, in this order:

          1. `EngineConfig.value_scale`, if set. AN EXPLICIT INSTRUCTION WINS
             OVER THE CHECKPOINT, because the case it exists for is a checkpoint
             whose own number is not trusted, and a flag that silently lost to
             the file would be a flag that does nothing on exactly the models
             somebody bothered to set it for. Overriding a checkpoint that HAD a
             value is logged with both numbers, because a stale flag left over
             from one run is otherwise invisible in the next.
          2. the checkpoint's own `value_scale`, attached by
             `chess_transformer_v2.load_model` when the file carries it. Every
             train_v5 checkpoint writes one unconditionally.
          3. LEGACY_VALUE_SCALE — for the guofish2..guofish4 checkpoints, whose
             v2-era writer predates the field. See that constant for why this is
             recovering a documented project-wide calibration rather than
             assuming one: the v2 pipeline trained against the identical number.

        EVERY PATH IS ANNOUNCED, including the ordinary one, because
        `score cp = value_scale * atanh(q)` is what Cutechess `-resign score=`
        and every adjudication threshold are read against. Which of the three
        supplied it is exactly the kind of thing that has to be in the run's own
        log rather than reconstructed afterwards.
        """
        from_checkpoint = getattr(self.model, "value_scale", None)
        override = self.config.value_scale
        model_name = type(self.model).__name__

        if override is not None:
            if from_checkpoint is not None:
                err(f"[eval] value_scale OVERRIDDEN: {override:.4f} replaces the "
                    f"{float(from_checkpoint):.4f} carried by "
                    f"{self.config.model_path or DEFAULT_MODEL}. Every reported "
                    f"'score cp' is on the override's scale.")
            else:
                err(f"[eval] value_scale {override:.4f} supplied by configuration "
                    f"(source=config); {model_name} carries none of its own.")
            self.value_scale_source = "config"
            return float(override)

        if from_checkpoint is not None:
            err(f"[eval] value_scale {float(from_checkpoint):.4f} from the "
                f"checkpoint (source=checkpoint).")
            self.value_scale_source = "checkpoint"
            return float(from_checkpoint)

        err(f"[eval] value_scale {LEGACY_VALUE_SCALE:.4f} assumed "
            f"(source=legacy-default): {model_name} is a legacy "
            f"guofish2..guofish4 net whose v2-era checkpoint writer predates the "
            f"field. This is the SAME Lc0 WDL calibration its value head was "
            f"trained against (data/csv_parallel.py CP_SCALE) and the same one "
            f"the v5 pipeline uses (data/multiPV/labels.py VALUE_SCALE), so v4 "
            f"and v5 report on one scale. Pass --value-scale to override.")
        self.value_scale_source = "legacy-default"
        return float(LEGACY_VALUE_SCALE)

    def reconfigure(self, config: EngineConfig) -> None:
        """Adopt a new configuration.

        Fields that live in `SearchConfig` are baked into the search object at
        construction, so changing one after `isready` means rebuilding the
        search — and that throws the tree away.

        THE REBUILD DROPS BOTH THE TREE AND THE TRANSPOSITION CACHE, and for
        `policy_temperature` that is a correctness requirement rather than a
        side effect (C11b requirement 4). Both stores hold priors that were
        materialised at a temperature: the arena holds one per child from
        expansion time, the cache holds whole post-softmax vectors. Flushing
        only the cache — which is the obvious half, because the cache is
        visibly full of priors — would leave a retained tree serving children
        sharpened at the old T alongside fresh expansions at the new one.
        Constructing a new `ReplaySearchQ32` makes that unrepresentable: the
        arena and the cache are both members of the object being replaced.

        `_base_fen`/`_moves` are cleared too, so the next `position` rebuilds
        from scratch instead of trying to extend a tree that no longer exists.

        The UCI layer applies options to `self.config` and calls this per
        `setoption`; the reader can see which changes cost a rebuild in the
        stderr line below.
        """
        rebuild = self._ready and _search_config_differs(self.config, config)
        self.config = config
        if not rebuild:
            return
        # `_rebuild_search` re-points the new search at the OPEN prober. That is
        # not optional bookkeeping: without it, any `setoption CPuctInit`
        # mid-game would turn mode 2 off for the rest of the game while the
        # configuration log went on reporting Syzygy as open.
        self._rebuild_search("a SearchConfig field changed; both the tree and "
                             "the cache hold values materialised under the "
                             "previous configuration")
        for line in config.describe():
            err(line)

    # --- C11b: the opening book and the tablebase -------------------------

    def open_readers(self) -> None:
        """Open both readers per the current config. Idempotent.

        Called from `ensure_ready`, and never from `new_game`.

        MISSING FILES WARN AND DISABLE, NAMING THE PATH. A typo'd `SyzygyPath`
        and an intentionally absent one must not produce the same output, so the
        warning always carries the path that was tried; and neither is a reason
        to refuse to play chess, so neither raises.
        """
        self.reopen_book()
        self.reopen_syzygy()

    def reopen_book(self) -> None:
        """Close and reopen the book alone, for a `UseBook`/`BookPath`/`BookSeed`
        change. Kept separate from the tablebase so a seed change does not make
        Fathom re-read 290 table headers for nothing."""
        if self.book is not None:
            self.book.close()
        self.book = None
        self.book_rng = None
        self.book_state = "off"
        self._open_book()

    def reopen_syzygy(self) -> None:
        """Close and reopen the tablebase alone, for a `UseSyzygy`/`SyzygyPath`
        change.

        REPLACING THE PROBER MEANS REPLACING THE SEARCH OBJECT, and that is not
        over-caution — it is the only thing that works. Two facts collide:

          1. `FathomProber` is ONE PER PROCESS. Fathom keeps its state in file
             scope (`tb_init`/`tb_free` take no handle), so constructing a
             second one while the first is alive raises rather than silently
             sharing and double-freeing its tables.

          2. `set_tablebase` carries pybind11's `keep_alive<1, 2>`: the SEARCH
             holds a hard reference to the prober for the search's whole
             lifetime, so that a caller who drops their only reference cannot
             leave a dangling pointer in the leaf path. `set_tablebase(None)`
             clears the pointer — `tablebase_backend` goes to None — but it does
             NOT release that reference. Measured, not assumed.

        So detaching is not freeing, and a `setoption name SyzygyPath value
        <other>` would find the old tables still open, fail to construct the new
        prober, and disable Syzygy entirely while the config log went on saying
        it was on. Destroying the search is what releases the reference.

        The tree and the cache go with it. That is a real cost and it is the
        right one at the moment it is paid: a tablebase path change is a
        configuration event, not a per-move one, and it normally arrives before
        the first `isready` — where `ensure_ready` opens the readers against a
        search that never had a prober and none of this runs at all.
        """
        had_prober = self.tablebase is not None
        # Ours first, so the rebuild's `reattach_tablebase` finds nothing to
        # reattach and the old prober is not carried onto the new search.
        self.tablebase = None
        if had_prober and self.search is not None:
            self._rebuild_search(
                "the Syzygy configuration changed and the previous prober is held "
                "by the search object (pybind11 keep_alive); Fathom allows only "
                "one open tablebase per process, so the search is rebuilt to "
                "release it")
        self.syzygy_state = "off"
        self._open_syzygy()

    def _rebuild_search(self, why: str) -> None:
        """Replace the C++ search object. THE TREE AND THE CACHE ARE BOTH LOST.

        One implementation, two callers — `reconfigure` when a `SearchConfig`
        field changed, and `reopen_syzygy` when the prober has to be released —
        so that "what a rebuild entails" is written down once. Both stores hold
        state that a rebuild is the correct way to discard: the arena holds
        priors materialised at expansion time, the cache holds post-softmax
        prior vectors, and neither can be selectively invalidated.
        """
        err(f"[config] rebuilding the search — {why}. The TREE and the "
            f"TRANSPOSITION CACHE are both discarded; the checkpoint and the "
            f"graphs are not.")
        self.search.set_tablebase(None)
        self.search.set_evaluator(None)
        self.search = guofish_core.ReplaySearchQ32(self.config.to_search_config())
        self.search.set_evaluator(self.evaluator.core)
        self.reattach_tablebase()
        self._base_fen = None
        self._moves = []
        self._board = None

    def _open_book(self) -> None:
        import chess.polyglot

        cfg = self.config
        path = cfg.book_target
        if not cfg.use_book:
            self.book_state = f"off (UseBook=false; would have used {path})"
            return
        if not path.is_file():
            self.book_state = f"missing {path}"
            err(f"[book] WARNING: no Polyglot book at {path}; book DISABLED for "
                f"this run. Set BookPath, or UseBook=false to stop asking.")
            return
        try:
            self.book = chess.polyglot.open_reader(str(path))
        except Exception as exc:                              # noqa: BLE001
            self.book_state = f"error {path}: {type(exc).__name__}: {exc}"
            err(f"[book] WARNING: {path} could not be opened "
                f"({type(exc).__name__}: {exc}); book DISABLED for this run.")
            self.book = None
            return
        # Seeded once, here, and NOT reset by `new_game`: varied play across a
        # session is the entire point of asking for a non-zero seed, and
        # re-seeding per game would make every game of a match open identically
        # while still costing the reproducibility that seed 0 gives for free.
        self.book_rng = random.Random(cfg.book_seed) if cfg.book_seed else None
        how = ("highest-weight entry (deterministic)" if cfg.book_seed == 0
               else f"weighted-random, seed {cfg.book_seed}")
        self.book_state = f"open {path} [{how}]"
        err(f"[book] opened {path} — {how}")

    def _open_syzygy(self) -> None:
        cfg = self.config
        path = cfg.syzygy_target
        if not cfg.use_syzygy:
            self.syzygy_state = f"off (UseSyzygy=false; would have used {path})"
            return
        if not path.is_dir():
            self.syzygy_state = f"missing {path}"
            err(f"[syzygy] WARNING: no tablebase directory at {path}; Syzygy "
                f"DISABLED for this run. Set SyzygyPath, or UseSyzygy=false to "
                f"stop asking.")
            return
        try:
            # ONE FathomProber PER PROCESS — Fathom keeps its state in file
            # scope, so a second live instance raises rather than silently
            # sharing and double-freeing. Releasing the previous one is NOT as
            # simple as dropping this reference; see `reopen_syzygy`, which is
            # the only caller allowed to reach here with a prober already
            # attached to a live search.
            self.tablebase = guofish_core.FathomProber(str(path))
        except Exception as exc:                              # noqa: BLE001
            self.syzygy_state = f"error {path}: {type(exc).__name__}: {exc}"
            err(f"[syzygy] WARNING: {path} could not be opened "
                f"({type(exc).__name__}: {exc}); Syzygy DISABLED for this run.")
            self.tablebase = None
            return
        # Mode 2 — leaf overrides on the search path. Mode 1, the root bypass,
        # is `probe_tablebase_root` below and needs no wiring. C7's four strong
        # value types make cache poisoning by a tablebase result uncompilable,
        # so this needs no runtime guard: the network's value is what reaches
        # the cache, and the tablebase value is what reaches the backup.
        self.search.set_tablebase(self.tablebase)
        self.syzygy_state = f"open {path} [{self.tablebase.largest}-man]"
        err(f"[syzygy] opened {path} — largest table {self.tablebase.largest} men "
            f"(mode 1 root bypass + mode 2 leaf overrides)")

    def _close_readers(self) -> None:
        """Teardown for `close()`, where the SEARCH is about to go too.

        That last clause is load-bearing and is why this is not the same as
        `open_readers`'s per-reader path. `set_tablebase(None)` detaches the
        pointer but does not release the prober — pybind11's `keep_alive<1, 2>`
        ties it to the search object's lifetime — so what actually frees
        Fathom's tables is `close()` dropping `self.search` immediately after
        this returns. Calling this WITHOUT then dropping the search leaves the
        prober alive and the next `FathomProber(...)` raises. `reopen_syzygy`
        exists precisely because of that, and does not call this.
        """
        if self.search is not None:
            self.search.set_tablebase(None)
        if self.book is not None:
            self.book.close()
        self.book = None
        self.book_rng = None
        self.tablebase = None
        self.book_state = "off"
        self.syzygy_state = "off"

    def reattach_tablebase(self) -> None:
        """Point the CURRENT search object at the open prober.

        `reconfigure` replaces the search object, and the new one starts with no
        tablebase. Without this, a `setoption` for any SearchConfig field would
        silently turn mode 2 off for the rest of the game while the config log
        still said Syzygy was open — a config-says-one-thing/engine-does-another
        defect of exactly the kind C11 exists to have removed.
        """
        if self.search is not None and self.tablebase is not None:
            self.search.set_tablebase(self.tablebase)

    def probe_book(self, board) -> Optional[str]:
        """A book move for `board` as UCI, or None on any miss.

        Misses on: the book closed, the position absent, a lookup error, or an
        entry whose move is not legal in this position (a corrupt or
        wrong-variant book). Every one of those falls through to MCTS, which is
        the reference's behaviour and the only safe one.

        `BookSeed = 0` TAKES THE HIGHEST-WEIGHT ENTRY rather than sampling. That
        is what makes a benchmark's opening distribution fixed across two
        engines without having to disable the book — and disabling it is the
        alternative that changes what is being measured.
        """
        if self.book is None:
            return None
        try:
            if self.book_rng is None:
                entry = max(self.book.find_all(board), key=lambda e: e.weight,
                            default=None)
                if entry is None:
                    return None
            else:
                entry = self.book.weighted_choice(board, random=self.book_rng)
        except IndexError:
            return None                      # the position is not in the book
        except Exception as exc:             # noqa: BLE001
            err(f"[book] WARNING: lookup failed for {board.fen()} "
                f"({type(exc).__name__}: {exc}); falling through to search")
            return None
        move = entry.move
        if move not in board.legal_moves:
            err(f"[book] WARNING: book move {move.uci()} is not legal in "
                f"{board.fen()}; falling through to search")
            return None
        return move.uci()

    def probe_tablebase_root(self, board) -> Optional[str]:
        """Mode 1: the tablebase-optimal move for `board`, or None on any miss.

        Delegates to `guofish_core.tablebase_root_move`, which is C7's tested
        implementation running on a `SearchBoard` — so the position handed to
        Fathom carries the RAW en-passant square rather than chess-library's
        filtered one. Nothing is re-implemented here.

        None means "fall through to MCTS" and covers out-of-range positions
        (more men than the loaded tables), missing tables, and anything the
        backend declines to answer.
        """
        if self.tablebase is None:
            return None
        try:
            return guofish_core.tablebase_root_move(board.fen(en_passant="fen"),
                                                    self.tablebase)
        except Exception as exc:                              # noqa: BLE001
            err(f"[syzygy] WARNING: root probe failed for {board.fen()} "
                f"({type(exc).__name__}: {exc}); falling through to search")
            return None

    def bypass_move(self, board) -> tuple[Optional[str], str]:
        """(uci, source) if a bypass fires, else (None, "search").

        BOOK FIRST, THEN TABLEBASE, which is v5's order and the right one: they
        cannot both fire (a book covers openings, a 5-man table covers
        endgames), and checking the cheaper memory-mapped lookup first costs
        nothing when it misses.
        """
        uci = self.probe_book(board)
        if uci is not None:
            return uci, "book"
        uci = self.probe_tablebase_root(board)
        if uci is not None:
            return uci, "tablebase"
        return None, "search"

    def close(self) -> None:
        self._close_readers()
        if self.search is not None:
            self.search.set_evaluator(None)
            self.search = None
        if self.evaluator is not None:
            self.evaluator.close()
            self.evaluator = None
        self._ready = False

    # --- position ---------------------------------------------------------

    def set_position(self, base_fen: str, moves: Iterable[str]) -> bool:
        """Root the tree at `base_fen` + `moves`, reusing the tree where possible.

        Returns True if the existing tree was extended (the UCI `position` line
        is the previous one plus new moves), False if it was rebuilt from
        scratch. A rebuild is the reference's `move not in root.children` branch
        widened to the whole position: cheap, correct and always available.
        """
        import chess

        self.ensure_ready()
        moves = list(moves)
        extendable = (self._base_fen == base_fen
                      and self._board is not None
                      and len(moves) >= len(self._moves)
                      and moves[:len(self._moves)] == self._moves)

        if extendable:
            board = self._board.copy()
            for uci in moves[len(self._moves):]:
                try:
                    self.search.apply_move(uci)
                except (ValueError, RuntimeError) as exc:
                    err(f"[position] apply_move({uci}) refused ({exc}); rebuilding "
                        f"the tree from {base_fen}")
                    extendable = False
                    break
                board.push(chess.Move.from_uci(uci))
            if extendable:
                self._moves = moves
                # Advanced in step with the tree, so the bypass probes and the
                # search always see the same position. A copy is pushed onto and
                # only adopted on success, so a refused apply_move leaves the
                # mirror untouched rather than half-advanced.
                self._board = board
                return True

        self._rebuild(base_fen, moves)
        return False

    def _rebuild(self, base_fen: str, moves: list[str]) -> None:
        """set_position on the final position, with its repetition history.

        The history is the FENs `build_repetition_history` walks: the positions
        BEFORE each of the last min(halfmove_clock, plies) moves, most recent
        first, WITHOUT the root — the core seeds the root's own count itself
        (see ReplaySearch::set_position). `fen(en_passant="fen")` throughout,
        because python-chess's default omits the ep square when no capture is
        legal and that moves token 66 and therefore the nn_key. DECISIONS.md, C2.
        """
        import chess

        board = chess.Board(base_fen)
        for uci in moves:
            board.push(chess.Move.from_uci(uci))

        probe = board.copy()
        history: list[str] = []
        for _ in range(min(probe.halfmove_clock, len(probe.move_stack))):
            probe.pop()
            history.append(probe.fen(en_passant="fen"))

        self.search.set_position(board.fen(en_passant="fen"), history)
        self._base_fen = base_fen
        self._moves = list(moves)
        self._board = board

    def new_game(self) -> None:
        """Between games: drop the tree AND the transposition cache.

        The cache survives `set_position` by design (C7), which is right within
        a game and wrong across one: cross-game transpositions are near-worthless
        and the entries are full prior vectors, so a tournament process would
        grow to its cap and stay there.

        THE READERS ARE NOT REOPENED HERE, and that is a deliberate omission
        rather than a gap. `chess.polyglot.open_reader` memory-maps the file and
        the Fathom prober re-reads every table header, so reopening per game
        costs real time at the start of every game and buys nothing: neither
        file changes while the process runs, and a path change goes through
        `open_readers` from the UCI layer.

        THE BOOK RNG IS NOT RESET EITHER, when the seed is non-zero. A caller
        that asked for a seeded book asked for varied play across a session;
        re-seeding per game would open every game of a match identically, which
        is what `BookSeed = 0` is for and is not what a non-zero seed means.

        What IS reset is the per-game decision tally, because it is per game.
        """
        if not self._ready:
            return
        self.search.clear_cache()
        self._base_fen = None
        self._moves = []
        self._board = None
        self.decision_counts = {source: 0 for source in DECISION_SOURCES}

    # --- the search -------------------------------------------------------

    def interrupt_slice(self) -> None:
        """Abort the RUNNING slice, from another thread. C11c.

        THE ONLY ENGINE METHOD MEANT TO BE CALLED WHILE A SEARCH IS IN FLIGHT,
        and it does exactly one thing: arms the C++ mutable deadline in the
        past, which the workers observe at their abort point within a single
        simulation.

        It does not decide anything. The slice loop still re-reads
        `should_stop()` and the clock afterwards and is the only thing that
        chooses whether to continue — so an interrupt that races a slice
        boundary costs at most one nearly-empty slice, never a wrong decision.

        WHY IT EXISTS. Before C11c the wrapper's `stop` latency was bounded
        below by `slice_seconds` (50 ms), because nothing could reach a search
        already inside `search_parallel`. That was tolerable for `stop` on a
        clock the engine had budgeted for; it is not tolerable on the ponder
        miss path, where a `stop` sits on the critical path of the opponent
        having already moved. It is also how `ponderhit` gets the running
        search to hand control back so the timed phase can start ON the tree
        the ponder built.

        Safe from any thread: `set_deadline_in` is one atomic store, takes no
        lock, and does not release the GIL. Harmless when no search is running
        — the next `search_move` clears the deadline before its first slice.
        """
        if self.search is not None:
            self.search.set_deadline_in(0.0)

    def search_move(self, *, budget: int, deadline: Optional[float] = None,
                    nominal: Optional[int] = None, budget_source: str = "nodes",
                    should_stop: Optional[Callable[[], bool]] = None,
                    on_slice: Optional[Callable[["SearchOutcome"], None]] = None,
                    allow_bypass: bool = True, count_decision: bool = True
                    ) -> SearchOutcome:
        """Run to `budget` root visits, in slices, honouring clock and stop flag.

        `budget` is an ABSOLUTE root-visit target, matching the core and the
        reference (`target_new_sims = num_simulations - existing`). What the
        engine actually does is `budget - inherited`, and that is what the
        returned `delivered` counts.

        `nominal` is what the CALLER asked for in simulations, or None when it
        asked for a duration instead — see SearchOutcome. It is deliberately not
        defaulted to `budget`: for a timed search the budget is a ceiling this
        engine picked, and reporting a ceiling as a request is the reporting
        defect one layer over.

        `deadline` is a `time.monotonic()` stamp or None for "until the budget
        runs out". `should_stop` is polled between slices. `on_slice` gets a
        partial outcome after each slice, for `info` emission — it runs on this
        thread with no search in flight, which is the only moment Python may run
        without competing with the dispatcher for the GIL.

        C11c ADDS TWO FLAGS AND ONE CLOCK.

        `allow_bypass=False` skips the book/tablebase probe. The PONDER phase
        passes it, because the caller has already probed the predicted position
        and declined to ponder if either answered — probing twice would be the
        same lookup charged to a search that is not going to use it.

        `count_decision=False` keeps this call out of `decision_counts`. A
        ponder decides no move, so tallying it would double-count every
        pondered move against the per-game figures a benchmark artifact carries.

        THE CLOCK: `deadline` is now armed on the C++ side as well, so it aborts
        the slice that is RUNNING rather than only preventing the next one. The
        Python deadline still owns the decision and the reporting; see
        `interrupt_slice` and `ReplaySearch::set_deadline_in`. It is cleared in
        a `finally`, because the C++ deadline deliberately outlives the
        `search_parallel` call that observed it and a stale one would leave the
        next search pre-expired.
        """
        self.ensure_ready()
        if self._base_fen is None:
            raise RuntimeError("Engine.search_move: no position set")

        # C11b. THE BYPASS, BEFORE ANY SIMULATION IS BUDGETED.
        #
        # A book or tablebase hit answers the move outright and MCTS never
        # runs, so this sits ahead of the slice loop rather than inside it —
        # there is no partial search to abandon and no `info` line to emit
        # about a search that did not happen. `delivered` is 0 by construction,
        # `source` says which lookup answered, and `bypassed` is what every
        # aggregate filters on.
        if allow_bypass and self._board is not None:
            started = time.perf_counter()
            uci, source = self.bypass_move(self._board)
            if uci is not None:
                self.decision_counts[source] = self.decision_counts.get(source, 0) + 1
                return self._bypass_outcome(uci, source, nominal, budget_source,
                                            time.perf_counter() - started)

        cfg = self.config
        parallel = cfg.to_parallel_config()
        inherited = int(self.search.root_visits)
        delivered = 0
        slices = 0
        # `eval_stats()` is reset at the start of every `search_parallel`, so a
        # sliced search must sum them or it reports the last slice as if it were
        # the move. Same failure mode as the delivered-sims one, one layer down.
        eval_rows = 0
        eval_batches = 0
        reason = "budget"
        stopped = False
        # C11c. ORed across the slices: "this MOVE degraded" is the statement a
        # game log and a benchmark row both need, and the core clears its own
        # flag at the top of every `search_parallel`.
        exhausted = False
        exhausted_at = 0
        deadline_hit = False
        started = time.perf_counter()
        # Seeded from the shipping configuration's measured rate so the first
        # slice is already about the right size; every slice after it uses this
        # search's own measurement.
        rate = 14_000.0

        # C11c. Arm the C++ mutable deadline from the Python one, so the clock
        # can end the slice that is RUNNING. Cleared unconditionally first: the
        # deadline outlives the call that observed it by design, so a previous
        # move's expired deadline would otherwise abort this one's first slice.
        self.search.clear_deadline()
        if deadline is not None:
            self.search.set_deadline_in(deadline - time.monotonic())

        try:
            while True:
                current = int(self.search.root_visits)
                if current >= budget:
                    reason = "budget"
                    break
                if should_stop is not None and should_stop():
                    reason, stopped = "stop", True
                    break
                now = time.perf_counter()
                if deadline is not None and now >= deadline:
                    reason = "time"
                    break

                chunk = max(cfg.min_slice_sims, int(rate * cfg.slice_seconds))
                if deadline is not None:
                    # Do not start a slice that cannot finish inside the
                    # deadline; shrink it instead. `min_slice_sims` is the
                    # floor, so a nearly expired clock still runs one small
                    # slice rather than spinning.
                    affordable = int(rate * max(0.0, deadline - now))
                    chunk = max(cfg.min_slice_sims, min(chunk, affordable))
                target = min(budget, current + chunk)

                self.search.search_parallel(target, parallel)
                par = self.search.parallel_stats()
                evals = self.search.eval_stats()
                delivered += int(par["delivered"])
                eval_rows += int(evals["rows"])
                eval_batches += int(evals["batches"])
                slices += 1
                if par["arena_exhausted"]:
                    exhausted = True
                    exhausted_at = exhausted_at or int(par["arena_exhausted_at"])
                deadline_hit = bool(par["deadline_hit"])
                elapsed = time.perf_counter() - started
                if delivered > 0 and elapsed > 0:
                    rate = delivered / elapsed

                if on_slice is not None:
                    on_slice(self._outcome(nominal, budget_source, inherited,
                                           delivered, elapsed, slices, eval_rows,
                                           eval_batches, stopped=False,
                                           reason="running", with_pv=False,
                                           exhausted=exhausted,
                                           exhausted_at=exhausted_at,
                                           deadline_hit=deadline_hit))

                # The depth-1 mate short-circuit ends the search wherever it
                # fired, exactly as the reference's completion_event does.
                # Without this the next slice would clear `mating_move` and keep
                # going, and the root would never reach the target because the
                # hack keeps re-firing.
                if self.search.mating_move is not None:
                    reason = "mate"
                    break
                # C11c. AHEAD OF THE `stalled` CHECK, and that ordering is the
                # whole of it. An exhausted arena delivers a short slice and
                # then a zero one, so a loop that tested `stalled` first would
                # report "stalled" for a condition that has a name, a cause and
                # a remedy — and `arena_exhausted` would reach the outcome
                # attached to the wrong reason string.
                if exhausted:
                    reason = "arena_exhausted"
                    break
                if deadline_hit:
                    # The clock ended the slice. The loop's own deadline test
                    # above will agree on the next pass; breaking here just
                    # avoids one more nearly-empty slice.
                    reason = "time"
                    break
                if int(par["delivered"]) == 0 and int(par["requested"]) > 0:
                    # Nothing moved: a root with no legal continuation left, or
                    # a target the core declined. Reporting it beats spinning.
                    reason = "stalled"
                    break
        finally:
            # ALWAYS. See the docstring: the C++ deadline is deliberately not
            # reset by `search_parallel`, so whoever armed it has to disarm it
            # or the next search starts pre-expired. `interrupt_slice` may have
            # armed it from another thread, which is exactly the case a
            # `finally` covers and an `if deadline is not None` would not.
            self.search.clear_deadline()

        if count_decision:
            self.decision_counts["search"] = self.decision_counts.get("search", 0) + 1
        return self._outcome(nominal, budget_source, inherited, delivered,
                             time.perf_counter() - started, slices,
                             eval_rows, eval_batches,
                             stopped=stopped, reason=reason, with_pv=True,
                             exhausted=exhausted, exhausted_at=exhausted_at,
                             deadline_hit=deadline_hit)

    # --- reporting internals ---------------------------------------------

    def _bypass_outcome(self, uci: str, source: str, nominal: Optional[int],
                        budget_source: str, wall: float) -> SearchOutcome:
        """The outcome for a move MCTS did not compute.

        Every search field is zero and is MEANT to read as zero — no delivered
        simulations, no inherited visits, no slices, no PV beyond the move
        itself, no score. A book move carries no evaluation, so reporting one
        would be inventing it; `score_cp` stays 0 and the `info` line the UCI
        layer emits says `depth 0`, which is what v5 did and what a GUI
        expects.

        `root_visits` and `nodes` are read off the live tree rather than
        zeroed: the tree still exists and still holds whatever the previous
        move built, and reporting 0 there would say the engine had thrown it
        away.
        """
        return SearchOutcome(
            best_move=uci,
            mating_move=None,
            nominal=nominal,
            inherited=0,
            delivered=0,
            wall_s=wall,
            slices=0,
            root_visits=int(self.search.root_visits),
            score_cp=0,
            q=0.0,
            pv=[uci],
            depth=1,
            max_depth=0,
            nodes=int(self.search.nodes),
            hashfull=0,
            stopped=False,
            reason=source,
            budget_source=budget_source,
            source=source,
        )

    def _outcome(self, nominal: Optional[int], budget_source: str,
                 inherited: int, delivered: int, wall: float, slices: int,
                 eval_rows: int, eval_batches: int, *,
                 stopped: bool, reason: str, with_pv: bool,
                 exhausted: bool = False, exhausted_at: int = 0,
                 deadline_hit: bool = False) -> SearchOutcome:
        best = self.search.best_move
        mating = self.search.mating_move
        cache = self.search.cache_stats()
        root_visits = int(self.search.root_visits)

        q, pv, depth, max_depth = 0.0, [], 0, 0
        if with_pv:
            q, pv, depth, max_depth = self._principal_variation(best)
        else:
            # O(1) stand-in for the per-slice `info` line: the root's own Q,
            # negated because a node's value_sum reads from the perspective of
            # the player who moved TO it and that is the opponent at the root.
            # The end-of-move line uses the best CHILD's Q instead, which is the
            # reference's `last_best_child_q` and the number adjudication sees.
            if root_visits > 0:
                q = -float(self.search.root_value_sum) / root_visits

        capacity = int(cache["capacity"]) or 1
        return SearchOutcome(
            best_move=best,
            mating_move=mating,
            nominal=nominal,
            inherited=inherited,
            delivered=delivered,
            wall_s=wall,
            slices=slices,
            root_visits=root_visits,
            score_cp=q_to_centipawns(q, self.value_scale or 1.0),
            q=q,
            pv=pv,
            depth=depth,
            max_depth=max_depth,
            nodes=int(self.search.nodes),
            hashfull=min(1000, int(1000 * int(cache["size"]) / capacity)),
            cache_hits=int(cache["hits"]),
            cache_misses=int(cache["misses"]),
            eval_rows=eval_rows,
            eval_batches=eval_batches,
            stopped=stopped,
            reason=reason,
            budget_source=budget_source,
            arena_exhausted=exhausted,
            arena_exhausted_at=exhausted_at,
            # Read off the LIVE tree rather than off the last slice's stats:
            # `high_water` survives resets and covers both ping-pong arenas, so
            # this is the peak the memory budget is actually about.
            arena_high_water=int(self.search.arena_high_water),
            arena_capacity=int(self.search.arena_capacity),
            deadline_hit=deadline_hit,
        )

    def root_branches(self, top_n: Optional[int] = None
                      ) -> list[tuple[str, int, float]]:
        """[(uci, visits, share)] for the root's children, most-visited first.

        ALL OF THEM BY DEFAULT, and truncation is the display layer's job.
        Returning the top 5 here was a bug caught by a real game: the caller
        looks the PLAYED move up in this list to report what the next search
        inherits, and a reply outside the top 5 was reported as having inherited
        nothing. The engine then printed `reused: 208` on the very next move and
        contradicted it. `format_root_branches(top_n=...)` shortens the table;
        every arithmetic caller sees the whole list.

        THE PONDER DIAGNOSTIC, and the one number that says whether pondering
        did anything useful. The reference printed the same thing from
        `root.children[m].visit_count` (core/mctsv4.py, `apply_move`'s
        `branch_visits=[...]`) with the comment that says why it exists: if the
        branch the opponent actually played is near-zero on a hit, the ponder
        work did not transfer and the next search starts from nothing.

        `share` is of the CHILDREN's total, not of `root_visits`, so the figures
        sum to 1.0 and are read as "how the search divided its attention". The
        root carries its own seed visit and including it would make a
        two-reply position look like it had spent 3% of its effort somewhere
        invisible.

        Reads `dump_tree_arrays(1)` — the VISITED subtree, one node per
        simulation rather than the ~30 per expansion the arena holds — and takes
        the depth-1 rows, which are exactly the root's children. Unvisited
        children are absent by construction, which is correct: a reply with no
        visits is not a branch the ponder built.
        """
        if self.search is None:
            return []
        arrays = self.search.dump_tree_arrays(1)
        depth = arrays["depth"]
        if depth.size == 0:
            return []
        visits, packed = arrays["visits"], arrays["move"]
        rows = [(int(visits[i]), int(packed[i]))
                for i in range(int(depth.size)) if int(depth[i]) == 1]
        if not rows:
            return []
        rows.sort(key=lambda row: row[0], reverse=True)
        total = sum(count for count, _ in rows) or 1
        return [(guofish_core.move_to_uci(move), count, count / total)
                for count, move in (rows if top_n is None else rows[:top_n])]

    def _principal_variation(self, best: Optional[str], max_plies: int = 12
                             ) -> tuple[float, list[str], int, int]:
        """(best child Q, PV, PV length, deepest visited ply).

        Walks `dump_tree_arrays(1)` — the VISITED subtree, which is ~1 node per
        simulation rather than the ~30 per expansion the whole arena holds, so a
        20,000-sim tree dumps ~20,000 rows and not ~600,000. The dump is DFS
        preorder with a depth column, so a child of the node at index i is the
        next entry with depth == depth[i] + 1 before the first entry with
        depth <= depth[i]; the walk follows the most-visited child at each step,
        which is what the reference's PV does.

        `dump_tree_arrays` rather than `dump_tree` because the array form carries
        the depth column and the tuple form does not, and because it is one
        traversal: the packed moves are turned into UCI one at a time, for the
        dozen nodes the PV actually visits.
        """
        arrays = self.search.dump_tree_arrays(1)
        depth = arrays["depth"]
        if depth.size == 0:
            return 0.0, [], 0, 0

        visits = arrays["visits"]
        value_sum = arrays["value_sum"]
        packed = arrays["move"]
        max_depth = int(depth.max())

        pv: list[str] = []
        q = 0.0
        i = 0
        while len(pv) < max_plies:
            here = int(depth[i])
            best_index = -1
            best_visits = 0
            j = i + 1
            while j < depth.size and int(depth[j]) > here:
                if int(depth[j]) == here + 1 and int(visits[j]) > best_visits:
                    best_visits, best_index = int(visits[j]), j
                j += 1
            if best_index < 0:
                break
            if not pv:
                # The engine's own move: its value_sum is already in the
                # engine's perspective (the backup negates on the way up), so it
                # is used as-is. Same convention as the reference's
                # last_best_child_q.
                q = float(value_sum[best_index]) / max(1, best_visits)
            pv.append(guofish_core.move_to_uci(int(packed[best_index])))
            i = best_index

        # `best_move` is the core's answer and the one that gets played; if the
        # PV walk disagrees the PV is wrong, not the move, so say so rather than
        # emitting a PV whose first move is not the one played.
        if best is not None and pv and pv[0] != best:
            err(f"[pv] WARNING: PV head {pv[0]} != best_move {best}; "
                f"reporting the best move alone")
            pv = [best]
        if best is not None and not pv:
            pv = [best]
        return q, pv, len(pv), max_depth


# Which fields force a rebuild of the C++ search object when they change.
# Everything in SearchConfig, plus the two that size the arena and the cache;
# `max_batch`/`threads`/`max_outstanding`/`affinity` live in ParallelConfig,
# which is rebuilt per search and therefore free.
#
# `policy_temperature` IS IN THIS LIST AND IT IS THE ENTRY MOST WORTH EXPLAINING
# (C11b requirement 4). Everything else here changes how the NEXT selection is
# scored; temperature changes what is already STORED. Priors are materialised
# into the arena at expansion time and into the transposition cache as
# post-softmax probabilities, so after a `setoption PolicyTemperature` between
# moves — no `ucinewgame`, which is the ordinary Cutechess sequence — a retained
# tree would keep serving children sharpened at the old T while every fresh
# expansion used the new one. One tree, two temperatures, and nothing anywhere
# saying so.
#
# Being in this tuple is what prevents that, and it drops BOTH stores at once:
# `reconfigure` constructs a new ReplaySearchQ32, whose arena and whose cache are
# both new. That is why the mechanism is a rebuild rather than a `clear_cache()`
# call — flushing the cache alone would leave the tree, and the tree is the half
# that survives a move.
_SEARCH_CONFIG_FIELDS = (
    "c_puct_init", "c_puct_base", "c_puct_factor", "fpu_root", "fpu_tree",
    "virtual_loss", "max_tree_depth", "cache_entries", "cache_shards",
    "arena_capacity", "ponder_decay", "verify_compaction", "policy_temperature",
)


def _search_config_differs(a: EngineConfig, b: EngineConfig) -> bool:
    return any(getattr(a, name) != getattr(b, name) for name in _SEARCH_CONFIG_FIELDS)


# ---------------------------------------------------------------------------
# CLI — analysis and a headless self-play smoke
# ---------------------------------------------------------------------------


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """The full EngineConfig surface as command-line flags.

    Shared with `playing/uci_wrapper_v6.py` so the two entry points cannot
    disagree about a default or omit a flag from one another — which is how
    `--virtual-loss` came to be settable in v5 and invisible in its logs.
    """
    g = parser.add_argument_group("model / evaluator")
    g.add_argument("--model", "--checkpoint", dest="model_path", type=Path, default=None,
                   help="v5 student or legacy guofish2..guofish4 checkpoint; the "
                        "architecture is read off the file, not from a flag "
                        f"(default: {DEFAULT_MODEL})")
    g.add_argument("--value-scale", type=float, default=None,
                   help="centipawn calibration inverted by score cp = "
                        "value_scale * atanh(q), OVERRIDING the checkpoint's own. "
                        "Omit for the normal resolution: the checkpoint's value "
                        "if it has one, else the project-wide Lc0 WDL constant "
                        "290.681 that both the v2-era and v5 label "
                        "pipelines trained against. The source is logged at "
                        "[eval] and on the [init] architecture line")
    g.add_argument("--max-batch", type=int, default=EngineConfig.max_batch,
                   help="dispatcher batch ceiling and the evaluator's buffer rows "
                        f"(default: {EngineConfig.max_batch}, the measured knee)")
    g.add_argument("--no-graphs", dest="graphs", action="store_false", default=True,
                   help="run the eager forward instead of the captured CUDA graph")
    g.add_argument("--no-compile", dest="compile", action="store_false", default=True,
                   help="capture the UNFUSED eager ATen forward instead of the "
                        "Inductor-codegen'd one — i.e. run tag "
                        "GUOFISH_NUMERICS_BASELINE's numerics (C12b). Slower on "
                        "fresh roots; bit-exact against the frozen baseline")
    g.add_argument("--no-pin", dest="pin", action="store_false", default=True,
                   help="do not page-lock the boundary buffers")
    g.add_argument("--switch-interval", type=float, default=DEFAULT_SWITCH_INTERVAL,
                   help=f"sys.setswitchinterval (default: {DEFAULT_SWITCH_INTERVAL})")

    g = parser.add_argument_group("parallelism")
    g.add_argument("--threads", type=int, default=EngineConfig.threads,
                   help=f"W, search threads (default: {EngineConfig.threads})")
    g.add_argument("--max-outstanding", type=int, default=EngineConfig.max_outstanding,
                   help="W*K, the in-flight leaf count the engine throttles on "
                        f"(default: {EngineConfig.max_outstanding}); K is derived")
    g.add_argument("--affinity", type=str, default=EngineConfig.affinity,
                   choices=list(guofish_core.AFFINITY_POLICIES),
                   help=f"thread pinning policy (default: {EngineConfig.affinity})")

    g = parser.add_argument_group("search")
    g.add_argument("--c-puct-init", "--c-init", dest="c_puct_init", type=float,
                   default=EngineConfig.c_puct_init)
    g.add_argument("--c-puct-base", "--c-base", dest="c_puct_base", type=float,
                   default=EngineConfig.c_puct_base)
    g.add_argument("--c-puct-factor", dest="c_puct_factor", type=float,
                   default=EngineConfig.c_puct_factor,
                   help="multiplies c(N); 1.0 leaves the value untouched")
    g.add_argument("--fpu-root", type=float, default=EngineConfig.fpu_root)
    g.add_argument("--fpu-tree", type=float, default=EngineConfig.fpu_tree)
    g.add_argument("--virtual-loss", type=float, default=EngineConfig.virtual_loss,
                   help="in-flight penalty, as an integer COUNT scaled by this "
                        f"magnitude (default: {EngineConfig.virtual_loss})")
    g.add_argument("--max-tree-depth", type=int, default=EngineConfig.max_tree_depth)
    g.add_argument("--policy-temperature", type=float,
                   default=EngineConfig.policy_temperature,
                   help="T in softmax(logits / T), applied at the root and at "
                        "every interior node. <1 sharpens, >1 flattens, 1.0 is "
                        "the identity and skips the divide entirely")
    g.add_argument("--dirichlet-epsilon", type=float,
                   default=EngineConfig.dirichlet_epsilon,
                   help="NOT IMPLEMENTED BY THE CORE; only 0.0 is accepted")
    g.add_argument("--dirichlet-alpha", type=float, default=EngineConfig.dirichlet_alpha)

    g = parser.add_argument_group("memory")
    g.add_argument("--cache-entries", type=int, default=EngineConfig.cache_entries,
                   help="transposition-cache slots; 0 disables the cache")
    g.add_argument("--cache-shards", type=int, default=EngineConfig.cache_shards,
                   help="0 uses the core's default shard count")
    g.add_argument("--arena-capacity", type=int, default=None,
                   help="nodes per arena, BOTH halves of the ping-pong pair "
                        "being reserved. Omit for the computed default "
                        "60 x (sims_per_move + ponder_max_sims), which tracks "
                        "the budget instead of having to be remembered "
                        "alongside it; the resolved value and its MB footprint "
                        "are on the [config] memory line")
    # C11c. HERE RATHER THAN IN uci_wrapper_v6.main(), which is where it lived
    # until this chunk. `ponder` is an `EngineConfig` field and this function's
    # contract is "the full EngineConfig surface as command-line flags", so the
    # wrapper defining its own was always the anomaly — and it left
    # `playv6_interactive` with no `--ponder` at all, which was invisible while
    # the flag did nothing there.
    #
    # It stopped being invisible the moment `--ponder-max-sims` was added:
    # argparse's prefix matching turned a bare `--ponder` into "ambiguous
    # option: could match --ponder-max-sims, --ponder-decay". Defining it
    # exactly, once, resolves the prefix and gives all three entry points the
    # same flag.
    g.add_argument("--ponder", action="store_true", default=EngineConfig.ponder,
                   help="advertise and honour pondering. Over UCI that is the "
                        "`go ponder`/`ponderhit` handshake; in the interactive "
                        "frontend it searches the CURRENT position while you "
                        "think. OFF by default: pondering changes the time "
                        "model rather than move selection, and Gate 5 must "
                        "match the ponder-off anchor")
    g.add_argument("--ponder-max-sims", type=int, default=None,
                   help="ceiling on a ponder search's simulations. Omit for the "
                        "computed default sims_per_move / ponder_decay, which "
                        "is the value that keeps a pondered tree from "
                        "out-weighing the fresh search it is handed to")
    g.add_argument("--ponder-decay", type=float, default=EngineConfig.ponder_decay)
    g.add_argument("--verify-compaction", action="store_true", default=False)

    g = parser.add_argument_group("budget")
    # ONE FLAG FOR THE PER-MOVE BUDGET, and it is `fixed_sims`.
    #
    # It was two until now — `--sims` bound to `default_sims` and `--fixed-sims`
    # to `fixed_sims` — and that was a trap rather than a choice. In
    # uci_wrapper.py and uci_wrapper_v5.py `--sims` IS the fixed budget
    # (`fixed_sims=args.sims`, "force every search to use exactly this many
    # sims, ignoring 'go nodes' / time-control args from the GUI"), and every
    # ELO run this project has published used it that way: the 2878 @ 10k
    # sweeps, run_15k_ab.ps1, tune_cpuct.bat. v6 silently rebound the same
    # spelling to the LOWEST-precedence branch of `_plan` — the fallback for a
    # `go` carrying neither a clock nor nodes — so the flag that used to
    # override the GUI became the one the GUI overrides. Same name, inverted
    # precedence, no error.
    #
    # In the CLI and interactive frontends the two were redundant anyway:
    # `budget = fixed_sims or default_sims` is "whichever of these you set",
    # and there is no GUI there for `--fixed-sims` to override.
    #
    # `default_sims` SURVIVES AS A FIELD, because the UCI fallback branch needs
    # a number and `fixed_sims=None` is load-bearing: `_plan` tests
    # `fixed_sims is not None` FIRST and returns deadline=None, so giving it a
    # non-None default would make the engine ignore every clock in every game.
    # It keeps its `DefaultSims` UCI option; it is simply no longer a flag
    # anybody types.
    g.add_argument("--sims", "--fixed-sims", dest="fixed_sims", type=int,
                   default=None,
                   help="THE per-move simulation budget: force every move to "
                        "this many sims and ignore the clock. This is v5's "
                        "--sims contract, restored. Omit to let the GUI decide "
                        "(go nodes / the clock), which is what a rated game "
                        "wants; set it for fixed-budget benchmarking. In the "
                        "CLI and interactive frontends there is no GUI, so this "
                        "is simply the budget (default: DefaultSims, 5000)")
    g.add_argument("--sim-cap", type=int, default=EngineConfig.sim_cap,
                   help="ceiling on whatever 'go nodes N' asks for, AND the "
                        "quantity the arena is sized from (see arena_nodes)")
    g.add_argument("--slice-seconds", type=float, default=EngineConfig.slice_seconds,
                   help="wall-clock granularity at which 'stop' and the clock "
                        "are answered")
    g.add_argument("--min-slice-sims", type=int, default=EngineConfig.min_slice_sims,
                   help="floor on a slice's simulation count, so a nearly "
                        "expired clock still makes progress instead of spinning")
    g.add_argument("--move-overhead-ms", type=int, default=EngineConfig.move_overhead_ms,
                   help="subtracted from every allotted move time")

    g = parser.add_argument_group("opening book / Syzygy (C11b; both default ON)")
    g.add_argument("--no-book", dest="use_book", action="store_false", default=True,
                   help="disable the Polyglot opening book. NOTE: a book move "
                        "bypasses MCTS entirely, so leaving it on dilutes any "
                        "measurement of the search itself — see --book-seed")
    g.add_argument("--book-path", type=Path, default=None,
                   help=f"Polyglot book (default: {DEFAULT_BOOK})")
    g.add_argument("--book-seed", type=int, default=EngineConfig.book_seed,
                   help="0 (the default) always plays the highest-weight entry, "
                        "which makes the book fully reproducible; any other "
                        "value seeds weighted-random selection")
    g.add_argument("--no-syzygy", dest="use_syzygy", action="store_false", default=True,
                   help="disable the Syzygy tablebase (root bypass and leaf "
                        "overrides both)")
    g.add_argument("--syzygy-path", type=Path, default=None,
                   help=f"Syzygy directory (default: {DEFAULT_SYZYGY})")


def config_from_args(args: argparse.Namespace) -> EngineConfig:
    """Build an EngineConfig from a namespace `add_config_arguments` filled."""
    names = {f.name for f in fields(EngineConfig)}
    kwargs = {k: v for k, v in vars(args).items() if k in names}
    return EngineConfig(**kwargs)


def _analyse(engine: Engine, fen: str, moves: list[str], budget: int) -> SearchOutcome:
    engine.set_position(fen, moves)
    outcome = engine.search_move(budget=budget, nominal=budget,
                                 budget_source="nodes")
    err(outcome.telemetry())
    print(f"bestmove {outcome.best_move} score cp {outcome.score_cp} "
          f"pv {' '.join(outcome.pv)}", flush=True)
    return outcome


def main(argv: Optional[list[str]] = None) -> int:
    # FIRST STATEMENT, before any thread exists. See apply_switch_interval.
    # C11c. Before any line is written: the streams are pipes under a
    # harness and Windows picks cp1252 for them, while every reader in
    # this repo opens the captured log as UTF-8. See
    # playv6.force_utf8_streams for the run this cost a verdict.
    force_utf8_streams()
    before, after = apply_switch_interval()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fen", type=str, default=None,
                        help="position to analyse (default: the start position)")
    parser.add_argument("--moves", type=str, nargs="*", default=[],
                        help="UCI moves applied to --fen")
    parser.add_argument("--selfplay", type=int, default=0, metavar="PLIES",
                        help="play PLIES moves against itself instead of analysing "
                             "one position; exercises the tree-reuse path the "
                             "delivered-sims accounting exists for")
    add_config_arguments(parser)
    args = parser.parse_args(argv)

    # The switch interval is a config field too, so honour a flag that changed it.
    if args.switch_interval != after:
        sys.setswitchinterval(args.switch_interval)
    err(f"[init] switch_interval {before:g} -> {sys.getswitchinterval():g}")

    try:
        config = config_from_args(args)
    except ConfigError as exc:
        parser.error(str(exc))
        return 2  # unreachable; parser.error exits

    import chess

    engine = Engine(config)
    try:
        engine.ensure_ready()
        fen = args.fen or chess.STARTING_FEN
        budget = config.fixed_sims or config.default_sims

        if args.selfplay <= 0:
            _analyse(engine, fen, list(args.moves), budget)
            return 0

        board = chess.Board(fen)
        for uci in args.moves:
            board.push(chess.Move.from_uci(uci))
        played = list(args.moves)
        outcomes: list[SearchOutcome] = []
        for ply in range(args.selfplay):
            if board.is_game_over(claim_draw=True):
                err(f"[selfplay] game over after {ply} plies: {board.result()}")
                break
            engine.set_position(fen, played)
            outcome = engine.search_move(budget=budget, nominal=budget,
                                         budget_source="nodes")
            err(f"[selfplay] ply {ply + 1}: {outcome.telemetry()}")
            outcomes.append(outcome)
            if outcome.best_move is None:
                err("[selfplay] no move returned; stopping")
                break
            board.push(chess.Move.from_uci(outcome.best_move))
            played.append(outcome.best_move)

        # BYPASSED MOVES ARE EXCLUDED FROM THE RATE, not folded into it. A book
        # move delivers zero simulations in some non-zero wall time, so including
        # it would report a throughput that depends on which opening was played.
        # The counts are printed beside the rate so the denominator is visible.
        rate, counts = aggregate_sims_per_s(outcomes)
        searched = [o for o in outcomes if not o.bypassed]
        delivered = sum(o.delivered for o in searched)
        nominal = sum(o.nominal or 0 for o in searched)
        wall = sum(o.wall_s for o in searched) or 1e-9
        err(f"[selfplay] TOTAL moves={counts['moves']} "
            f"search={counts['search']} book={counts['book']} "
            f"tablebase={counts['tablebase']} "
            f"(bypassed {counts['bypassed']}, excluded {counts['excluded_wall_s']:.2f}s)")
        err(f"[selfplay] OVER THE SEARCHED MOVES ONLY: delivered={delivered} "
            f"nominal={nominal} wall={wall:.2f}s "
            f"delivered_sims_per_s={rate:,.0f} "
            f"nominal_sims_per_s={nominal / wall:,.0f} "
            f"inflation={nominal / max(1, delivered):.2f}x")
        print(" ".join(played), flush=True)
        return 0
    finally:
        engine.close()


__all__ = [
    "DECISION_SOURCES",
    "DEFAULT_BOOK",
    "DEFAULT_MODEL",
    "DEFAULT_SWITCH_INTERVAL",
    "DEFAULT_SYZYGY",
    "ConfigError",
    "aggregate_sims_per_s",
    "Engine",
    "EngineConfig",
    "SearchOutcome",
    "UNSUPPORTED_IN_CORE",
    "add_config_arguments",
    "apply_switch_interval",
    "config_from_args",
    "format_root_branches",
    "move_label",
    "preimport_torch",
    "q_to_centipawns",
]


if __name__ == "__main__":
    raise SystemExit(main())
