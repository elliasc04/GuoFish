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
`policy_temperature` and Dirichlet root noise exist in the Python reference
(`core.mctsv4.POLICY_TEMPERATURE`, `ParallelMCTS._add_dirichlet_noise` and the
`MCTSNode.base_prior` field the C3b fix added) and have NO counterpart anywhere
in `cpp/`. C11's scope forbids changing the C++ search core, so this module
cannot make them work. It refuses them instead: both are advertised, both accept
only their identity value, and any other value raises with a message naming the
missing mechanism. Accepting them and doing nothing is precisely defect 1.

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
import sys
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

DEFAULT_MODEL = REPO_ROOT / "models" / "guofish5_20M" / "v5_10.9M_best.pt"
DEFAULT_SWITCH_INTERVAL = guofish_core.DEFAULT_SWITCH_INTERVAL

# Largest |value| the v5 label pipeline ever produced (data/multiPV/labels.py:
# VALUE_CLAMP / VALUE_MATE = 0.995), restated from uci_wrapper_v5.py rather than
# imported so the v6 surface does not depend on the v5 one. See
# `q_to_centipawns` for what it does to the reported ceiling.
VALUE_CLAMP = 0.995


def err(msg: str) -> None:
    """stderr, flushed. Never stdout: stdout is the UCI stream."""
    print(msg, file=sys.stderr, flush=True)


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
# whole point: `policy_temperature = 0.8` on a core that cannot sharpen priors
# is a benchmark row that describes a search nobody ran.
UNSUPPORTED_IN_CORE: dict[str, tuple[float, str]] = {
    "policy_temperature": (
        1.0,
        "cpp/ has no prior-sharpening step: SearchConfig carries no temperature "
        "field and gather_softmax_canonical raises the logits to no power. The "
        "reference's core.mctsv4.POLICY_TEMPERATURE was not ported and C11's "
        "scope forbids changing the search core, so only the identity value 1.0 "
        "is accepted.",
    ),
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
    arena_capacity: int = 1_200_000
    ponder_decay: float = 1.0
    verify_compaction: bool = False

    # --- declared, refused unless identity (see UNSUPPORTED_IN_CORE) ------
    policy_temperature: float = 1.0
    dirichlet_epsilon: float = 0.0
    dirichlet_alpha: float = 0.3

    # --- budgets ----------------------------------------------------------
    # The per-move simulation budget when the GUI sends no `nodes`. `sim_cap`
    # bounds whatever it does send; `fixed_sims`, when set, overrides both and is
    # what a fixed-budget tournament pins.
    default_sims: int = 5_000
    sim_cap: int = 60_000
    fixed_sims: Optional[int] = None
    # Wall-clock slice the search is cut into so `stop` and the clock are
    # answerable. See Engine.search.
    slice_seconds: float = 0.05
    min_slice_sims: int = 32

    # --- host-side features ----------------------------------------------
    # NO opening book and NO Syzygy bypass. v5's wrapper carried both; neither
    # is in C11's implementation scope, and a config field that is logged but
    # does nothing is the defect this chunk exists to remove. Cutechess supplies
    # openings with `-openings file=...`, which needs nothing from the engine,
    # and the core's tablebase backend is attached through `set_tablebase` by
    # whichever chunk decides to ship one.
    ponder: bool = False
    move_overhead_ms: int = 100

    def __post_init__(self) -> None:
        if self.model_path is not None:
            self.model_path = Path(self.model_path)
        self.validate()

    # --- derived ----------------------------------------------------------

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
        if self.affinity not in guofish_core.AFFINITY_POLICIES:
            raise ConfigError(
                f"affinity={self.affinity!r} is not one of "
                f"{list(guofish_core.AFFINITY_POLICIES)}")
        if self.max_tree_depth < 1:
            raise ConfigError(f"max_tree_depth must be >= 1, got {self.max_tree_depth}")
        if self.cache_entries < 0:
            raise ConfigError(f"cache_entries must be >= 0, got {self.cache_entries}")
        if self.arena_capacity < 1024:
            raise ConfigError(
                f"arena_capacity must be >= 1024, got {self.arena_capacity}")
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
            arena_capacity=self.arena_capacity,
            cache_slots=self.cache_entries,
            ponder_decay=self.ponder_decay,
            verify_compaction=self.verify_compaction,
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
            f"[config] model      : {self.model_path or DEFAULT_MODEL}",
            f"[config] evaluator  : max_batch={self.max_batch} graphs={self.graphs} "
            f"pin={self.pin} switch_interval={self.switch_interval:g}",
            f"[config] parallel   : threads(W)={self.threads} in_flight(K)={self.in_flight} "
            f"max_outstanding={self.max_outstanding} "
            f"(effective W*K={self.effective_outstanding}) affinity={self.affinity}",
            f"[config] exploration: c_puct_init={self.c_puct_init:g} "
            f"c_puct_base={self.c_puct_base:g} c_puct_factor={self.c_puct_factor:g} "
            f"fpu_root={self.fpu_root:g} fpu_tree={self.fpu_tree:g}",
            f"[config] search     : virtual_loss={self.virtual_loss:g} "
            f"max_tree_depth={self.max_tree_depth} "
            f"policy_temperature={self.policy_temperature:g} "
            f"dirichlet_epsilon={self.dirichlet_epsilon:g} "
            f"dirichlet_alpha={self.dirichlet_alpha:g}",
            f"[config] memory     : cache_entries={self.cache_entries} "
            f"cache_shards={self.cache_shards or 'default'} "
            f"arena_capacity={self.arena_capacity}",
            f"[config] budget     : default_sims={self.default_sims} "
            f"sim_cap={self.sim_cap} fixed_sims={self.fixed_sims} "
            f"slice_seconds={self.slice_seconds:g} "
            f"move_overhead_ms={self.move_overhead_ms}",
            f"[config] host       : ponder={self.ponder} "
            f"ponder_decay={self.ponder_decay:g} "
            f"verify_compaction={self.verify_compaction} "
            f"(no opening book, no Syzygy bypass — neither is wired in v6)",
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

    @property
    def sims_per_s(self) -> float:
        """DELIVERED simulations per second. The honest rate (C11's objective)."""
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
        return (f"[search] delivered={self.delivered} nominal={nominal} "
                f"inherited={self.inherited} slices={self.slices} "
                f"wall={self.wall_s * 1000:.1f}ms "
                f"delivered_sims_per_s={self.sims_per_s:,.0f} "
                f"nominal_sims_per_s={nominal_rate} "
                f"inflation={inflation} "
                f"root_visits={self.root_visits} nodes={self.nodes} "
                f"eval_rows={self.eval_rows} eval_batches={self.eval_batches} "
                f"cache_hits={self.cache_hits} cache_misses={self.cache_misses} "
                f"best={self.best_move} score_cp={self.score_cp} "
                f"budget_source={self.budget_source} "
                f"stopped={self.stopped} reason={self.reason}")


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
        self.model = None
        self.device = None
        self._ready = False
        # The position the tree is rooted at, as the UCI layer described it.
        self._base_fen: Optional[str] = None
        self._moves: list[str] = []

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

        value_scale = getattr(self.model, "value_scale", None)
        if value_scale is None:
            raise RuntimeError(
                f"{cfg.model_path or DEFAULT_MODEL} carries no value_scale, so "
                f"'score cp' has no calibration to invert. Every train_v5 "
                f"checkpoint writes one; a checkpoint without it did not come "
                f"from that pipeline.")
        self.value_scale = float(value_scale)

        self.evaluator = live_evaluator.TorchEvaluator(
            self.model, self.device, cfg.max_batch,
            switch_interval=cfg.switch_interval, pin=cfg.pin, graphs=cfg.graphs)
        self.search = guofish_core.ReplaySearchQ32(cfg.to_search_config())
        self.search.set_evaluator(self.evaluator.core)
        self._ready = True

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
        err(f"[eval] score cp = {self.value_scale:.4f} * atanh(q), saturating at "
            f"+-{q_to_centipawns(1.0, self.value_scale)} cp "
            f"(|q| clamped to {VALUE_CLAMP})")

    def reconfigure(self, config: EngineConfig) -> None:
        """Adopt a new configuration.

        Fields that live in `SearchConfig` are baked into the search object at
        construction, so changing one after `isready` means rebuilding the
        search — and that throws the tree away. The UCI layer therefore applies
        options to `self.config` and calls this only between games; a `setoption`
        arriving mid-game is applied to the config and takes effect on the next
        rebuild, which is reported rather than assumed.
        """
        rebuild = self._ready and _search_config_differs(self.config, config)
        self.config = config
        if not rebuild:
            return
        err("[config] a SearchConfig field changed; rebuilding the search "
            "(the tree is discarded, the checkpoint and the graphs are not)")
        self.search.set_evaluator(None)
        self.search = guofish_core.ReplaySearchQ32(config.to_search_config())
        self.search.set_evaluator(self.evaluator.core)
        self._base_fen = None
        self._moves = []
        for line in config.describe():
            err(line)

    def close(self) -> None:
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
        self.ensure_ready()
        moves = list(moves)
        extendable = (self._base_fen == base_fen
                      and len(moves) >= len(self._moves)
                      and moves[:len(self._moves)] == self._moves)

        if extendable:
            for uci in moves[len(self._moves):]:
                try:
                    self.search.apply_move(uci)
                except (ValueError, RuntimeError) as exc:
                    err(f"[position] apply_move({uci}) refused ({exc}); rebuilding "
                        f"the tree from {base_fen}")
                    extendable = False
                    break
            if extendable:
                self._moves = moves
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

    def new_game(self) -> None:
        """Between games: drop the tree AND the transposition cache.

        The cache survives `set_position` by design (C7), which is right within
        a game and wrong across one: cross-game transpositions are near-worthless
        and the entries are full prior vectors, so a tournament process would
        grow to its cap and stay there.
        """
        if not self._ready:
            return
        self.search.clear_cache()
        self._base_fen = None
        self._moves = []

    # --- the search -------------------------------------------------------

    def search_move(self, *, budget: int, deadline: Optional[float] = None,
                    nominal: Optional[int] = None, budget_source: str = "nodes",
                    should_stop: Optional[Callable[[], bool]] = None,
                    on_slice: Optional[Callable[["SearchOutcome"], None]] = None
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
        """
        self.ensure_ready()
        if self._base_fen is None:
            raise RuntimeError("Engine.search_move: no position set")

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
        started = time.perf_counter()
        # Seeded from the shipping configuration's measured rate so the first
        # slice is already about the right size; every slice after it uses this
        # search's own measurement.
        rate = 14_000.0

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
                # Do not start a slice that cannot finish inside the deadline;
                # shrink it instead. `min_slice_sims` is the floor, so a nearly
                # expired clock still runs one small slice rather than spinning.
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
            elapsed = time.perf_counter() - started
            if delivered > 0 and elapsed > 0:
                rate = delivered / elapsed

            if on_slice is not None:
                on_slice(self._outcome(nominal, budget_source, inherited,
                                       delivered, elapsed, slices, eval_rows,
                                       eval_batches, stopped=False,
                                       reason="running", with_pv=False))

            # The depth-1 mate short-circuit ends the search wherever it fired,
            # exactly as the reference's completion_event does. Without this the
            # next slice would clear `mating_move` and keep going, and the root
            # would never reach the target because the hack keeps re-firing.
            if self.search.mating_move is not None:
                reason = "mate"
                break
            if int(par["delivered"]) == 0 and int(par["requested"]) > 0:
                # Nothing moved: a root with no legal continuation left, or a
                # target the core declined. Reporting it beats spinning.
                reason = "stalled"
                break

        return self._outcome(nominal, budget_source, inherited, delivered,
                             time.perf_counter() - started, slices,
                             eval_rows, eval_batches,
                             stopped=stopped, reason=reason, with_pv=True)

    # --- reporting internals ---------------------------------------------

    def _outcome(self, nominal: Optional[int], budget_source: str,
                 inherited: int, delivered: int, wall: float, slices: int,
                 eval_rows: int, eval_batches: int, *,
                 stopped: bool, reason: str, with_pv: bool) -> SearchOutcome:
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
        )

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
_SEARCH_CONFIG_FIELDS = (
    "c_puct_init", "c_puct_base", "c_puct_factor", "fpu_root", "fpu_tree",
    "virtual_loss", "max_tree_depth", "cache_entries", "cache_shards",
    "arena_capacity", "ponder_decay", "verify_compaction",
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
                   help=f"v5 checkpoint (default: {DEFAULT_MODEL})")
    g.add_argument("--max-batch", type=int, default=EngineConfig.max_batch,
                   help="dispatcher batch ceiling and the evaluator's buffer rows "
                        f"(default: {EngineConfig.max_batch}, the measured knee)")
    g.add_argument("--no-graphs", dest="graphs", action="store_false", default=True,
                   help="run the eager forward instead of the captured CUDA graph")
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
                   help="NOT IMPLEMENTED BY THE CORE; only 1.0 is accepted")
    g.add_argument("--dirichlet-epsilon", type=float,
                   default=EngineConfig.dirichlet_epsilon,
                   help="NOT IMPLEMENTED BY THE CORE; only 0.0 is accepted")
    g.add_argument("--dirichlet-alpha", type=float, default=EngineConfig.dirichlet_alpha)

    g = parser.add_argument_group("memory")
    g.add_argument("--cache-entries", type=int, default=EngineConfig.cache_entries,
                   help="transposition-cache slots; 0 disables the cache")
    g.add_argument("--cache-shards", type=int, default=EngineConfig.cache_shards,
                   help="0 uses the core's default shard count")
    g.add_argument("--arena-capacity", type=int, default=EngineConfig.arena_capacity)
    g.add_argument("--ponder-decay", type=float, default=EngineConfig.ponder_decay)
    g.add_argument("--verify-compaction", action="store_true", default=False)

    g = parser.add_argument_group("budget")
    g.add_argument("--sims", dest="default_sims", type=int,
                   default=EngineConfig.default_sims,
                   help="per-move budget when the GUI sends no 'nodes'")
    g.add_argument("--sim-cap", type=int, default=EngineConfig.sim_cap,
                   help="ceiling on whatever 'go nodes N' asks for")
    g.add_argument("--fixed-sims", type=int, default=None,
                   help="force every move to this budget, ignoring the GUI")
    g.add_argument("--slice-seconds", type=float, default=EngineConfig.slice_seconds,
                   help="wall-clock granularity at which 'stop' and the clock "
                        "are answered")
    g.add_argument("--min-slice-sims", type=int, default=EngineConfig.min_slice_sims,
                   help="floor on a slice's simulation count, so a nearly "
                        "expired clock still makes progress instead of spinning")
    g.add_argument("--move-overhead-ms", type=int, default=EngineConfig.move_overhead_ms,
                   help="subtracted from every allotted move time")


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
        totals = {"delivered": 0, "nominal": 0, "wall": 0.0}
        for ply in range(args.selfplay):
            if board.is_game_over(claim_draw=True):
                err(f"[selfplay] game over after {ply} plies: {board.result()}")
                break
            engine.set_position(fen, played)
            outcome = engine.search_move(budget=budget, nominal=budget,
                                         budget_source="nodes")
            err(f"[selfplay] ply {ply + 1}: {outcome.telemetry()}")
            totals["delivered"] += outcome.delivered
            totals["nominal"] += outcome.nominal or 0
            totals["wall"] += outcome.wall_s
            if outcome.best_move is None:
                err("[selfplay] no move returned; stopping")
                break
            board.push(chess.Move.from_uci(outcome.best_move))
            played.append(outcome.best_move)
        wall = totals["wall"] or 1e-9
        err(f"[selfplay] TOTAL delivered={totals['delivered']} "
            f"nominal={totals['nominal']} wall={wall:.2f}s "
            f"delivered_sims_per_s={totals['delivered'] / wall:,.0f} "
            f"nominal_sims_per_s={totals['nominal'] / wall:,.0f} "
            f"inflation={totals['nominal'] / max(1, totals['delivered']):.2f}x")
        print(" ".join(played), flush=True)
        return 0
    finally:
        engine.close()


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SWITCH_INTERVAL",
    "ConfigError",
    "Engine",
    "EngineConfig",
    "SearchOutcome",
    "UNSUPPORTED_IN_CORE",
    "add_config_arguments",
    "apply_switch_interval",
    "config_from_args",
    "preimport_torch",
    "q_to_centipawns",
]


if __name__ == "__main__":
    raise SystemExit(main())
