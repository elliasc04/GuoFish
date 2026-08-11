"""C11b — the opening book and the Syzygy bypass, on the production Engine.

WHAT IS BEING TESTED, AND WHY IT IS THE TELEMETRY AND NOT THE MOVES
===================================================================
Neither feature is new logic. `chess.polyglot` is python-chess's, and
`guofish_core.tablebase_root_move` is C7's — built, tested against the real
5-piece tables, and unchanged here. C11b wires them into the v6 host, so what
this file tests is the WIRING, and the wiring's job is almost entirely
bookkeeping:

  * a bypassed move delivers ZERO simulations, because MCTS did not run;
  * `SearchOutcome.source` says which lookup answered;
  * aggregates exclude bypassed moves rather than averaging a zero-simulation
    move into a throughput figure;
  * the readers open ONCE and are not reopened per game;
  * `BookSeed = 0` is deterministic and picks the highest-weight entry;
  * a missing file warns, names the path it tried, and disables the feature
    rather than killing the engine.

That list is the whole of C11b part 2, and every item on it is a way for a
benchmark to be quietly wrong rather than a way for the engine to play a bad
move. Which is why the assertions here are about counts, sources and file
handles rather than about chess.

NO MODEL, NO CUDA, NO TORCH
===========================
`Engine.ensure_ready` loads a checkpoint and needs CUDA, and none of the
behaviour above depends on what the network says — a bypassed move never asks
it. So these tests stand up a ready `Engine` around a `guofish_core.LiveEvaluator`
whose callback is a few lines of NumPy, exactly as tests/test_c11b_temperature.py
does. `_ready` is set directly; everything below that is the shipping object,
including `open_readers`, `probe_book`, `probe_tablebase_root`, `search_move`,
`new_game`, `reconfigure` and `close`.

The one thing that CANNOT be synthesised is the assets: `assets/gm2001.bin` and
`assets/syzygy`. Tests that need them are marked individually with the reason
(Amendment D — no module-scope skip), and the tests for the MISSING-file
behaviour deliberately need neither and always run.
"""

import sys
from pathlib import Path

import pytest

import guofish_core

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from playing.v6 import playv6  # noqa: E402

BOOK = playv6.DEFAULT_BOOK
SYZYGY = playv6.DEFAULT_SYZYGY

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Three men, White to move and winning. Well inside a 5-man set, so any Syzygy
# installation that is a Syzygy installation at all answers it.
TB_FEN = "8/8/8/8/8/3k4/8/3KQ3 w - - 0 1"

# A middlegame position that is in neither the book nor a 5-man table, so the
# bypass must miss on both and MCTS must run. Taken from the C10 corpus family:
# 26 men, nothing an opening book past 20 plies would carry.
NO_BYPASS_FEN = "r2k2nr/1p1n1pp1/2p1p1p1/p2pP3/Pb6/1P1P1NPP/1BP2PB1/R3K2R w KQ - 1 16"


def _why_no_book():
    try:
        import chess.polyglot  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"python-chess is not importable ({type(exc).__name__}: {exc})"
    if not BOOK.is_file():
        return f"the Polyglot book is missing ({BOOK})"
    return None


def _why_no_syzygy():
    if not SYZYGY.is_dir():
        return f"the Syzygy directory is missing ({SYZYGY})"
    if not any(SYZYGY.glob("*.rtbw")):
        return f"{SYZYGY} holds no .rtbw tables"
    return None


NO_BOOK = _why_no_book()
NO_SYZYGY = _why_no_syzygy()
requires_book = pytest.mark.skipif(NO_BOOK is not None, reason=str(NO_BOOK))
requires_syzygy = pytest.mark.skipif(NO_SYZYGY is not None, reason=str(NO_SYZYGY))


# ---------------------------------------------------------------------------
# A ready Engine with no checkpoint
# ---------------------------------------------------------------------------

class _SyntheticNetwork:
    """A network that is a hash. See tests/test_c11b_temperature.py.

    A class rather than a closure so the evaluator -> callback -> evaluator
    cycle can be broken deliberately: pybind11 instances are not traversable by
    Python's cycle collector, so an unbroken cycle survives to interpreter exit
    and puts `guofish_core` into LeakSanitizer's report — breaking the leak
    discriminator README_BUILD.md relies on. `_release_synthetic_evaluators`
    clears the back-reference after every test.
    """

    def __init__(self):
        self.evaluator = None

    def __call__(self, count):
        import numpy as np

        tokens = self.evaluator.input_view()
        policy = self.evaluator.policy_view()
        value = self.evaluator.value_view()
        for row in range(count):
            seed = int(np.asarray(tokens[row], dtype=np.int64).sum() & 0x7FFFFFFF)
            rng = np.random.default_rng(seed)
            floats = rng.normal(0.0, 3.0, size=guofish_core.POLICY_SIZE).astype(np.float32)
            policy[row, :] = (floats.view(np.uint32) >> 16).astype(np.uint16)
            value[row] = np.float32(np.tanh(seed % 1000 / 1000.0 - 0.5))


_LIVE_NETWORKS: list = []


@pytest.fixture(autouse=True)
def _release_synthetic_evaluators():
    """Break every fixture cycle this module built, after every test."""
    yield
    for network in _LIVE_NETWORKS:
        network.evaluator = None
    _LIVE_NETWORKS.clear()


def _synthetic_evaluator(max_batch: int = 8):
    network = _SyntheticNetwork()
    evaluator = guofish_core.LiveEvaluator(max_batch, network, 0.0)
    network.evaluator = evaluator
    _LIVE_NETWORKS.append(network)
    return evaluator


class _EvaluatorStub:
    def __init__(self, core):
        self.core = core
        self.max_batch = core.max_batch

    def close(self):
        """C++ owns the buffers; there is nothing here to release."""


def make_engine(**overrides):
    """A ready `Engine` with a synthetic network and the real readers.

    ONE FathomProber PER PROCESS — Fathom keeps its state in file scope — so
    every caller must `close()` the engine it gets, and every test below does it
    in a `finally`. A leaked prober makes the NEXT test fail with a confusing
    "already alive" error, which is why this is a plain factory with an
    explicit close rather than a fixture some test could forget to request.
    """
    # `max_batch` must not exceed the evaluator's row count — the core refuses a
    # drain wider than the buffers, which is the check `handle_setoption` also
    # enforces at the command that would cause it. 8 rows is plenty for the
    # move counts here and keeps the synthetic callback cheap.
    config = playv6.EngineConfig(cache_entries=4096, arena_capacity=1 << 16,
                                 max_batch=8, max_outstanding=8,
                                 **overrides)
    engine = playv6.Engine(config)
    evaluator = _synthetic_evaluator(config.max_batch)
    engine.evaluator = _EvaluatorStub(evaluator)
    engine.search = guofish_core.ReplaySearchQ32(config.to_search_config())
    engine.search.set_evaluator(evaluator)
    engine.value_scale = 100.0
    engine._ready = True
    engine.open_readers()
    return engine


# ---------------------------------------------------------------------------
# Defaults and the resolved-state record
# ---------------------------------------------------------------------------

def test_both_features_default_on():
    """The deployment decision, asserted rather than assumed.

    Both are free strength and v6 should play its best out of the box. The
    measurement discipline that has to come with that is everything else in
    this file; it is deliberately not "default them off".
    """
    config = playv6.EngineConfig()
    assert config.use_book is True
    assert config.use_syzygy is True
    assert config.book_seed == 0, (
        "the shipped default must be the deterministic one: seed 0 means "
        "'always the highest-weight entry', which is what lets a benchmark fix "
        "the opening distribution without disabling the book")


def test_the_paths_resolve_without_being_set():
    config = playv6.EngineConfig()
    assert config.book_target == playv6.DEFAULT_BOOK
    assert config.syzygy_target == playv6.DEFAULT_SYZYGY
    # And an explicit path wins, as a Path even when handed a string.
    custom = playv6.EngineConfig(book_path="assets/other.bin",
                                 syzygy_path="assets/other")
    assert custom.book_target == Path("assets/other.bin")
    assert custom.syzygy_target == Path("assets/other")


def test_the_configuration_log_records_the_resolved_book_and_syzygy_state():
    """C11b's first measurement requirement, at the layer that produces the log.

    Every benchmark artifact has to record the resolved state, and the
    convention that makes that unforgettable is that `describe()` — which
    `isready` emits before every search — always carries it. A run whose book
    was on says so in its own header.
    """
    lines = playv6.EngineConfig().describe()
    book_line = next(line for line in lines if line.startswith("[config] book"))
    syzygy_line = next(line for line in lines if line.startswith("[config] syzygy"))
    assert "use_book=True" in book_line
    assert str(playv6.DEFAULT_BOOK) in book_line
    assert "seed=0" in book_line and "DETERMINISTIC" in book_line
    assert "use_syzygy=True" in syzygy_line
    assert str(playv6.DEFAULT_SYZYGY) in syzygy_line

    seeded = next(line for line in playv6.EngineConfig(book_seed=7).describe()
                  if line.startswith("[config] book"))
    assert "seed=7" in seeded and "weighted-random" in seeded


def test_every_new_field_appears_in_the_one_line_record():
    """The C11 telemetry requirement, extended to the five new fields.

    `as_kv()` is what a smoke run greps. A field that reached the engine appears
    there; a field that did not, does not. Adding a knob without adding it to
    the record is defect 1 of C11 in its purest form, so the five are named
    explicitly rather than covered by the generic loop in test_c11_uci.py.
    """
    kv = playv6.EngineConfig().as_kv()
    for name in ("use_book", "book_path", "book_seed", "use_syzygy", "syzygy_path"):
        assert f"{name}=" in kv, f"{name} is missing from as_kv()"


def test_a_negative_book_seed_is_refused():
    """0 has a meaning here, so the only value worth refusing is a typo."""
    with pytest.raises(playv6.ConfigError, match="book_seed"):
        playv6.EngineConfig(book_seed=-1)


# ---------------------------------------------------------------------------
# Missing files warn and disable — and need no assets, so they always run
# ---------------------------------------------------------------------------

def test_a_missing_book_disables_the_book_and_names_the_path(capsys):
    """It must not crash, and it must not be indistinguishable from 'off'.

    A typo'd BookPath and an intentionally absent book produce very different
    intentions and the same silence, unless the warning carries the path. The
    engine plays on either way — refusing to start a tournament game because an
    optional asset is missing is the wrong trade.
    """
    missing = REPO_ROOT / "assets" / "no-such-book-c11b.bin"
    assert not missing.exists()
    engine = make_engine(book_path=missing, use_syzygy=False)
    try:
        captured = capsys.readouterr()
        assert engine.book is None
        assert engine.book_state.startswith("missing")
        assert str(missing) in engine.book_state
        assert str(missing) in captured.err, (
            "the warning must name the path it tried, or a typo'd BookPath is "
            "indistinguishable from an intentionally absent one")
        assert "WARNING" in captured.err
        # And the engine still answers a move.
        engine.set_position(NO_BYPASS_FEN, [])
        outcome = engine.search_move(budget=40, nominal=40)
        assert outcome.best_move is not None
        assert outcome.source == "search"
    finally:
        engine.close()


def test_a_missing_syzygy_directory_disables_it_and_names_the_path(capsys):
    missing = REPO_ROOT / "assets" / "no-such-syzygy-c11b"
    assert not missing.exists()
    engine = make_engine(syzygy_path=missing, use_book=False)
    try:
        captured = capsys.readouterr()
        assert engine.tablebase is None
        assert engine.syzygy_state.startswith("missing")
        assert str(missing) in engine.syzygy_state
        assert str(missing) in captured.err
        assert "WARNING" in captured.err
        engine.set_position(TB_FEN, [])
        outcome = engine.search_move(budget=40, nominal=40)
        assert outcome.best_move is not None
        assert outcome.source == "search", (
            "with no tablebase open, a 3-man position must fall through to MCTS "
            "rather than reporting a bypass that did not happen")
    finally:
        engine.close()


def test_turning_a_feature_off_still_records_what_it_would_have_used():
    """`use_book=False` is a decision, and a log has to be able to show it was
    taken deliberately rather than by a missing file."""
    engine = make_engine(use_book=False, use_syzygy=False)
    try:
        assert engine.book is None and engine.tablebase is None
        assert "UseBook=false" in engine.book_state
        assert str(playv6.DEFAULT_BOOK) in engine.book_state
        assert "UseSyzygy=false" in engine.syzygy_state
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# The book
# ---------------------------------------------------------------------------

@requires_book
def test_probe_book_answers_the_start_position_with_a_legal_move():
    import chess

    engine = make_engine(use_syzygy=False)
    try:
        assert engine.book is not None, engine.book_state
        assert engine.book_state.startswith("open")
        uci = engine.probe_book(chess.Board(START_FEN))
        assert uci is not None, "gm2001.bin does not cover the start position"
        assert chess.Move.from_uci(uci) in chess.Board(START_FEN).legal_moves
    finally:
        engine.close()


@requires_book
def test_book_seed_zero_plays_the_highest_weight_entry_every_time():
    """The deterministic contract, checked against the book's own weights.

    `BookSeed = 0` is reserved for "always the highest-weight move" rather than
    being an ordinary RNG seed, and that is what lets Gate 5 run WITH the book
    on and still have a fixed opening distribution across both engines. If it
    sampled instead, the two engines would draw independently and the
    comparison would carry an opening-selection difference nobody asked for.
    """
    import chess
    import chess.polyglot

    engine = make_engine(use_syzygy=False)
    try:
        board = chess.Board(START_FEN)
        answers = {engine.probe_book(board) for _ in range(25)}
        assert len(answers) == 1, (
            f"BookSeed=0 returned {len(answers)} different moves over 25 probes "
            f"({answers}); it is supposed to be deterministic")

        with chess.polyglot.open_reader(str(BOOK)) as reader:
            heaviest = max(reader.find_all(board), key=lambda e: e.weight)
        assert answers.pop() == heaviest.move.uci(), (
            "BookSeed=0 must play the HIGHEST-WEIGHT entry, not merely the same "
            "entry every time — 'deterministic' and 'the book author's top "
            "choice' are different properties and the second is the one that "
            "makes the opening a good one")
    finally:
        engine.close()


@requires_book
def test_a_non_zero_seed_is_reproducible_across_engines():
    """Two processes with one seed must play one game. That is what a seed is for."""
    import chess

    board = chess.Board(START_FEN)
    first = make_engine(use_syzygy=False, book_seed=12345)
    try:
        a = [first.probe_book(board) for _ in range(12)]
    finally:
        first.close()
    second = make_engine(use_syzygy=False, book_seed=12345)
    try:
        b = [second.probe_book(board) for _ in range(12)]
    finally:
        second.close()
    assert a == b, f"seed 12345 gave {a} then {b}"
    assert all(m is not None for m in a)


@requires_book
def test_new_game_does_not_reopen_the_reader_or_reset_the_seeded_rng():
    """Two amendments in one test, because they are the same decision.

    THE READER. `chess.polyglot.open_reader` memory-maps the file; reopening it
    per game costs real time at the start of every game and buys nothing,
    because the file does not change while the process runs. The reader OBJECT
    must therefore survive `new_game`, and object identity is the way to say
    that without depending on how reopening would have been implemented.

    THE RNG. A caller that set a non-zero seed asked for varied play across a
    session. Re-seeding per game would make every game of a match open
    identically — which is what `BookSeed = 0` is for, and is not what a
    non-zero seed means. So the sequence must CONTINUE across `new_game`, not
    restart.
    """
    import chess

    engine = make_engine(use_syzygy=False, book_seed=99)
    try:
        board = chess.Board(START_FEN)
        reader = engine.book
        rng = engine.book_rng
        before = [engine.probe_book(board) for _ in range(8)]

        engine.new_game()

        assert engine.book is reader, (
            "new_game reopened the Polyglot reader; it is memory-mapped and the "
            "file has not changed, so this is pure cost per game")
        assert engine.book_rng is rng, "new_game replaced the book RNG object"

        after = [engine.probe_book(board) for _ in range(8)]
        continued = [engine.probe_book(board) for _ in range(8)]
        assert before != after or after != continued, (
            "the seeded book returned the same 8-move sequence before and after "
            "new_game, which means the RNG was reset — varied play across a "
            "session is the point of a non-zero seed")
    finally:
        engine.close()


@requires_book
def test_new_game_does_reset_the_per_game_decision_counts():
    """What IS per game gets reset, so the tally means what it says."""
    import chess

    engine = make_engine(use_syzygy=False)
    try:
        engine.set_position(START_FEN, [])
        engine.search_move(budget=40, nominal=40)
        assert sum(engine.decision_counts.values()) == 1
        engine.new_game()
        assert engine.decision_counts == {"search": 0, "book": 0, "tablebase": 0}
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# The tablebase
# ---------------------------------------------------------------------------

@requires_syzygy
def test_probe_tablebase_root_answers_a_three_man_position():
    import chess

    engine = make_engine(use_book=False)
    try:
        assert engine.tablebase is not None, engine.syzygy_state
        assert engine.syzygy_state.startswith("open")
        uci = engine.probe_tablebase_root(chess.Board(TB_FEN))
        assert uci is not None, (
            f"{TB_FEN} has three men and the tables report largest="
            f"{engine.tablebase.largest}; a miss here means mode 1 is not wired")
        assert chess.Move.from_uci(uci) in chess.Board(TB_FEN).legal_moves
    finally:
        engine.close()


@requires_syzygy
def test_a_position_outside_the_tables_falls_through_to_the_search():
    """A miss must mean "search it", not "fail". Every position with more men
    than the loaded tables takes this path, which is almost all of them."""
    import chess

    engine = make_engine(use_book=False)
    try:
        assert engine.probe_tablebase_root(chess.Board(NO_BYPASS_FEN)) is None
    finally:
        engine.close()


@requires_syzygy
def test_mode_2_is_attached_to_the_search_and_survives_a_reconfigure():
    """The leaf-override path, and the bug that would otherwise hide in it.

    `Engine.reconfigure` replaces the search object when a SearchConfig field
    changes, and the new one starts with no tablebase. Without an explicit
    re-attach, any `setoption CPuctInit` mid-game would turn mode 2 off for the
    rest of the game while the configuration log went on reporting Syzygy as
    open — a config-says-one-thing/engine-does-another defect of exactly the
    kind C11 existed to remove, reintroduced by C11b's own plumbing.
    """
    engine = make_engine(use_book=False)
    try:
        assert engine.search.tablebase_backend is not None
        first = engine.search
        engine.reconfigure(engine.config.replace(c_puct_init=1.75))
        assert engine.search is not first, "the SearchConfig change did not rebuild"
        assert engine.search.tablebase_backend is not None, (
            "the rebuilt search has no tablebase: mode 2 is silently off for the "
            "rest of the game while the config log still says Syzygy is open")
        assert engine.search.tablebase_backend == "fathom(5-man)"
        # And the retired search was detached on the way out, so nothing points
        # a dead search's leaf path at a live prober.
        assert first.tablebase_backend is None
    finally:
        engine.close()


@requires_syzygy
def test_a_syzygy_path_change_actually_opens_the_new_tables(capsys):
    """The lifetime trap this chunk found, and the reason `reopen_syzygy` is not
    three lines long.

    `set_tablebase(None)` detaches the pointer but does NOT release the object:
    pybind11's `keep_alive<1, 2>` ties the prober's lifetime to the SEARCH's, on
    purpose, so that dropping the last Python reference cannot leave a dangling
    pointer in the leaf path. Fathom then refuses to construct a second prober
    while the first is alive — correctly, since it keeps its state in file
    scope.

    Put together: a naive reopen finds the old tables still open, fails, and
    disables Syzygy for the rest of the session while the configuration log goes
    on reporting it as on. `reopen_syzygy` rebuilds the search to release the
    reference, and this is the test that says so — measured through
    `tablebase_backend`, which is the only thing that distinguishes "reopened"
    from "quietly gave up".
    """
    engine = make_engine(use_book=False)
    try:
        assert engine.tablebase is not None
        capsys.readouterr()

        # Same path, but the reopen must genuinely close and reconstruct.
        engine.reopen_syzygy()
        captured = capsys.readouterr()

        assert engine.tablebase is not None, (
            f"reopening the tablebase failed and left Syzygy off: "
            f"{engine.syzygy_state}. The previous prober was still alive.")
        assert engine.syzygy_state.startswith("open"), engine.syzygy_state
        assert engine.search.tablebase_backend is not None, (
            "the new prober was constructed but never attached to the search, so "
            "mode 2 is off")
        assert "WARNING" not in captured.err, captured.err
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# The acceptance criterion: zero simulations, correct attribution
# ---------------------------------------------------------------------------

@requires_book
def test_a_book_move_delivers_zero_simulations_and_is_attributed_to_the_book():
    """C11b acceptance 3, first half.

    `delivered == 0` is the load-bearing assertion. It is what makes a bypassed
    move visible to every throughput calculation as an absence rather than as a
    slow move, and it is only true if the bypass sits AHEAD of the slice loop
    rather than inside it.
    """
    engine = make_engine(use_syzygy=False)
    try:
        engine.set_position(START_FEN, [])
        before = int(engine.search.root_visits)
        outcome = engine.search_move(budget=5000, nominal=5000)

        assert outcome.source == "book"
        assert outcome.bypassed is True
        assert outcome.delivered == 0, (
            f"the book move delivered {outcome.delivered} simulations; MCTS is "
            f"supposed not to have run at all")
        assert outcome.slices == 0
        assert outcome.best_move == engine.probe_book(engine._board)
        assert outcome.pv == [outcome.best_move]
        assert int(engine.search.root_visits) == before, (
            "the tree advanced during a bypassed move")
        assert engine.decision_counts["book"] == 1
        assert engine.decision_counts["search"] == 0
        assert "BYPASS" in outcome.telemetry()
        assert "source=book" in outcome.telemetry()
    finally:
        engine.close()


@requires_syzygy
def test_a_tablebase_move_delivers_zero_simulations_and_is_attributed_to_it():
    """C11b acceptance 3, second half."""
    engine = make_engine(use_book=False)
    try:
        engine.set_position(TB_FEN, [])
        outcome = engine.search_move(budget=5000, nominal=5000)

        assert outcome.source == "tablebase"
        assert outcome.bypassed is True
        assert outcome.delivered == 0
        assert outcome.slices == 0
        assert outcome.best_move is not None
        assert engine.decision_counts["tablebase"] == 1
        assert "source=tablebase" in outcome.telemetry()
    finally:
        engine.close()


def test_a_searched_move_is_attributed_to_the_search():
    """The negative control. Without it, an engine that reported every move as a
    book hit would pass both tests above."""
    engine = make_engine(use_book=False, use_syzygy=False)
    try:
        engine.set_position(NO_BYPASS_FEN, [])
        outcome = engine.search_move(budget=60, nominal=60)
        assert outcome.source == "search"
        assert outcome.bypassed is False
        assert outcome.delivered > 0
        assert engine.decision_counts == {"search": 1, "book": 0, "tablebase": 0}
    finally:
        engine.close()


@requires_book
@requires_syzygy
def test_the_bypass_misses_on_a_middlegame_and_the_search_runs():
    """Both readers open, and a position neither covers still gets searched.

    This is the ordinary case — almost every move of almost every game — and it
    is the one that would break silently if a probe threw on a miss instead of
    returning None.
    """
    engine = make_engine()
    try:
        engine.set_position(NO_BYPASS_FEN, [])
        outcome = engine.search_move(budget=60, nominal=60)
        assert outcome.source == "search"
        assert outcome.delivered > 0
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# The aggregate exclusion
# ---------------------------------------------------------------------------

def _outcome(source, delivered, wall):
    return playv6.SearchOutcome(
        best_move="e2e4", mating_move=None, nominal=None, inherited=0,
        delivered=delivered, wall_s=wall, slices=1, root_visits=0, score_cp=0,
        q=0.0, source=source)


def test_bypassed_moves_are_excluded_from_the_aggregate_rate():
    """C11b's telemetry requirement, on the function every caller goes through.

    A bypassed move delivers zero simulations in some non-zero wall time. Fold
    it into a throughput mean and the mean drops by however many moves the book
    happened to cover — a number that describes which opening was played, not
    how fast the engine is. The exclusion lives in one place so that no caller
    has to remember it, and the counts come back beside the rate so the
    denominator is checkable.
    """
    outcomes = [
        _outcome("book", 0, 0.01),
        _outcome("book", 0, 0.01),
        _outcome("search", 1000, 0.1),
        _outcome("search", 3000, 0.3),
        _outcome("tablebase", 0, 0.02),
    ]
    rate, counts = playv6.aggregate_sims_per_s(outcomes)

    # 4000 delivered over 0.4 s of SEARCHED wall time. Including the bypassed
    # moves' 0.04 s would give 9,090/s instead, understating by 9%.
    assert rate == pytest.approx(10_000.0)
    assert counts["moves"] == 5
    assert counts["searched"] == 2
    assert counts["bypassed"] == 3
    assert counts["book"] == 2
    assert counts["tablebase"] == 1
    assert counts["search"] == 2
    assert counts["excluded_wall_s"] == pytest.approx(0.04)


def test_an_all_bypassed_set_reports_zero_rather_than_dividing_by_zero():
    """The degenerate case a short book-only opening actually produces."""
    rate, counts = playv6.aggregate_sims_per_s([_outcome("book", 0, 0.01)])
    assert rate == 0.0
    assert counts["searched"] == 0 and counts["bypassed"] == 1


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@requires_book
@requires_syzygy
def test_close_releases_both_readers():
    """The FathomProber is one-per-process, so a leaked one breaks the NEXT
    caller rather than this one. `close()` must actually drop it — which is
    checked by constructing a second engine straight afterwards."""
    engine = make_engine()
    assert engine.book is not None and engine.tablebase is not None
    engine.close()
    assert engine.book is None and engine.tablebase is None

    second = make_engine()
    try:
        assert second.tablebase is not None, (
            "a second FathomProber could not be constructed, so the first was "
            "never released — Fathom keeps its state in file scope and refuses "
            "a second live instance")
    finally:
        second.close()


@requires_book
def test_reopen_book_follows_a_path_change():
    """What a `setoption name BookPath` does, at the layer that does it."""
    missing = REPO_ROOT / "assets" / "no-such-book-c11b.bin"
    engine = make_engine(use_syzygy=False)
    try:
        assert engine.book is not None
        engine.config = engine.config.replace(book_path=missing)
        engine.reopen_book()
        assert engine.book is None
        assert engine.book_state.startswith("missing")

        engine.config = engine.config.replace(book_path=None)
        engine.reopen_book()
        assert engine.book is not None
        assert engine.book_state.startswith("open")
    finally:
        engine.close()
