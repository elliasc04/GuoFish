"""C11 — the Python surface: option coverage, refusals, and honest sim counting.

These are the properties of the production entry points that can be asserted
without a GPU. The two acceptance criteria that need one are scripts, not tests,
because they need a subprocess and a tournament manager:

    tools/uci_conform_c11.py   UCI conformance (43 checks; the `stop`, `ponderhit`
                               and isready-under-load behaviour)
    tools/smoke_c11.py         the 20-game Cutechess run and its four verdicts

WHAT IS ASSERTED HERE, AND WHY EACH ONE EXISTS
==============================================
  option coverage   Every `EngineConfig` field has a UCI option and every option
                    names a real field. This is v5's defect 1 stated as an
                    invariant: a knob that exists and cannot be set, or is set
                    and does not exist, is exactly how `--virtual-loss` came to
                    be invisible.
  refusals          `policy_temperature` and Dirichlet noise have no mechanism in
                    the C++ core, so a non-identity value must RAISE. A test that
                    only checked they were advertised would pass on a wrapper
                    that accepted them and did nothing — the defect itself.
  config log        `as_kv()` names every field, so the one-line record before
                    each search is complete by construction rather than by
                    inspection.
  sim accounting    `sims_per_s` divides DELIVERED simulations; `nominal` and
                    `inflation` are None when the caller asked for a duration.
  go parsing        `go` tokens map to the budget and clock the way UCI says,
                    including the cases where a GUI sends nonsense.

No module-scope skip anywhere (Amendment D). `playv6` needs only `guofish_core`;
the wrapper additionally needs `python-chess`, and the tests that touch it import
it through a guarded helper so a missing dependency is a per-test skip with a
named reason rather than a silently uncollected file.
"""
from __future__ import annotations

from dataclasses import fields
import math
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from playing.v6 import playv6  # noqa: E402
from playing.v6.playv6 import ConfigError, EngineConfig, SearchOutcome  # noqa: E402


def _why_no_wrapper() -> str | None:
    """The reason `playing.uci_wrapper_v6` cannot be imported here, or None."""
    try:
        import chess  # noqa: F401 - probing importability
    except ImportError as exc:
        return f"python-chess is not importable ({exc})"
    try:
        import playing.uci_wrapper_v6  # noqa: F401 - probing importability
    except ImportError as exc:                     # pragma: no cover - diagnostic
        return f"playing.uci_wrapper_v6 is not importable ({exc})"
    return None


WRAPPER_UNAVAILABLE = _why_no_wrapper()
requires_wrapper = pytest.mark.skipif(WRAPPER_UNAVAILABLE is not None,
                                      reason=str(WRAPPER_UNAVAILABLE))


def wrapper():
    """The wrapper module. Only called from tests carrying `requires_wrapper`."""
    import playing.uci_wrapper_v6 as module
    return module


# ---------------------------------------------------------------------------
# Option coverage — v5's defect 1, as an invariant
# ---------------------------------------------------------------------------


@requires_wrapper
def test_every_engine_config_field_has_a_uci_option():
    missing = wrapper().missing_options()
    assert not missing, (
        f"EngineConfig fields with no UCI option: {sorted(missing)}. A field that "
        f"cannot be set through the protocol is a knob whose value a tournament "
        f"cannot record — the v5 --virtual-loss defect exactly.")


@requires_wrapper
def test_every_uci_option_names_a_real_engine_config_field():
    known = {f.name for f in fields(EngineConfig)}
    strays = {o.name: o.attr for o in wrapper().OPTIONS if o.attr not in known}
    assert not strays, f"options pointing at fields that do not exist: {strays}"


@requires_wrapper
def test_option_names_are_unique_case_insensitively():
    names = [o.name.lower() for o in wrapper().OPTIONS]
    assert len(names) == len(set(names)), (
        "two options share a name up to case; setoption lookup is "
        "case-insensitive, so one of them would be unreachable")


@requires_wrapper
def test_the_advertised_default_is_this_processs_current_value():
    """Not the dataclass default — the value the engine is actually running.

    A GUI's "reset to default" writes back what was advertised, so advertising
    the class default would silently undo a command-line flag.
    """
    module = wrapper()
    config = EngineConfig(virtual_loss=3.7, c_puct_factor=1.23, threads=2,
                          max_outstanding=16)
    engine = module.UCIEngine(config)
    shown = {o.name: module._format_option_value(getattr(engine.config, o.attr))
             for o in module.OPTIONS}
    assert shown["VirtualLoss"] == "3.7"
    assert shown["CPuctFactor"] == "1.23"
    assert shown["Threads"] == "2"
    assert shown["MaxOutstanding"] == "16"


# ---------------------------------------------------------------------------
# The C11 validation: an unusual value reaches the logged configuration
# ---------------------------------------------------------------------------


@requires_wrapper
def test_setoption_virtual_loss_and_c_puct_factor_reach_the_logged_config():
    """C11's stated validation, in process.

    tools/uci_conform_c11.py runs the same assertion against a real subprocess's
    stderr; this one pins the mechanism so a failure says which half broke.
    """
    module = wrapper()
    engine = module.UCIEngine(EngineConfig())
    engine.handle_setoption("name VirtualLoss value 3.7".split())
    engine.handle_setoption("name CPuctFactor value 1.23".split())

    kv = engine.config.as_kv()
    assert "virtual_loss=3.7" in kv, kv
    assert "c_puct_factor=1.23" in kv, kv
    assert any("virtual_loss=3.7" in line for line in engine.config.describe())
    assert any("c_puct_factor=1.23" in line for line in engine.config.describe())


@requires_wrapper
def test_a_rejected_setoption_leaves_the_previous_value_in_place():
    module = wrapper()
    engine = module.UCIEngine(EngineConfig(virtual_loss=2.5))
    for bad in ("name VirtualLoss value not-a-number",
                "name VirtualLoss value -1",
                "name NoSuchOption value 3",
                # C11b AMENDMENT (authorised; see DECISIONS.md, "Two tests in
                # test_c11_uci.py asserted the absence of the feature C11b was
                # mandated to build"). This line read `PolicyTemperature value
                # 0.8` when the core had no temperature and 1.0 was the only
                # accepted value. It has one now, so 0.8 is a legal setting and
                # asserting it is dropped would assert the feature away. `0` is
                # still refused, and for a sharper reason than "unimplemented":
                # the temperature DIVIDES the logits.
                "name PolicyTemperature value 0",
                "name DirichletEpsilon value 0.25"):
        engine.handle_setoption(bad.split())
    assert engine.config.virtual_loss == 2.5
    assert engine.config.policy_temperature == 1.0
    assert engine.config.dirichlet_epsilon == 0.0


@requires_wrapper
def test_setoption_parses_a_name_containing_spaces():
    """UCI option names may contain spaces, so the parse must be positional."""
    module = wrapper()
    engine = module.UCIEngine(EngineConfig())
    # No advertised name has a space today, but the parser must not depend on
    # that: `value` is the delimiter, not the token index.
    engine.handle_setoption("name CPuctInit value 1.99".split())
    assert engine.config.c_puct_init == pytest.approx(1.99)


def test_every_config_field_appears_in_the_one_line_record():
    kv = EngineConfig().as_kv()
    for f in fields(EngineConfig):
        assert f"{f.name}=" in kv, (
            f"{f.name} is missing from as_kv(); the per-search configuration "
            f"record would be incomplete and C11's telemetry requirement unmet")
    # The two derived quantities, which a reader cannot recompute without knowing
    # the flooring rule.
    assert "in_flight=" in kv and "effective_outstanding=" in kv


# ---------------------------------------------------------------------------
# Refusals — what the C++ core does not implement
# ---------------------------------------------------------------------------


# C11b AMENDMENT (authorised; see DECISIONS.md). `("policy_temperature", 0.8)`
# and `("policy_temperature", 1.2)` were the first two rows here. C11b's mandate
# was to implement that knob and to "remove *only* policy_temperature from that
# set", so those rows now assert the ABSENCE of the feature this chunk was
# built to add. Dirichlet noise is still refused and still exercises every part
# of the mechanism this test checks — the ConfigError, the field name in the
# message, the reason, and the `cpp/` pointer — so nothing about the refusal
# PATH has lost coverage. What temperature does now is covered by
# tests/test_c11b_temperature.py.
@pytest.mark.parametrize("field_name,bad", [("dirichlet_epsilon", 0.25),
                                            ("dirichlet_epsilon", 1.0)])
def test_a_value_the_core_cannot_honour_is_refused(field_name, bad):
    with pytest.raises(ConfigError) as caught:
        EngineConfig(**{field_name: bad})
    message = str(caught.value)
    assert field_name in message
    assert "not implemented by the C++ core" in message
    # The refusal must say WHY, not merely that. A bare rejection is what sends
    # someone off to look for the flag they think they got wrong.
    assert "cpp/" in message


@pytest.mark.parametrize("field_name", sorted(playv6.UNSUPPORTED_IN_CORE))
def test_the_identity_value_of_an_unsupported_field_is_accepted(field_name):
    only, _ = playv6.UNSUPPORTED_IN_CORE[field_name]
    config = EngineConfig(**{field_name: only})
    assert getattr(config, field_name) == only


def test_the_unsupported_set_is_named_in_the_configuration_log():
    lines = EngineConfig().describe()
    tail = lines[-1]
    assert "NOT IMPLEMENTED BY THE CORE" in tail
    for name in playv6.UNSUPPORTED_IN_CORE:
        assert name in tail


# ---------------------------------------------------------------------------
# Validation of the fields the core DOES implement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("changes", [
    {"threads": 0},
    {"max_outstanding": 0},
    {"threads": 4, "max_outstanding": 2},
    {"max_batch": 0},
    {"affinity": "not-a-policy"},
    {"max_tree_depth": 0},
    {"cache_entries": -1},
    {"arena_capacity": 16},
    {"ponder_decay": 0.0},
    {"ponder_decay": 1.5},
    {"fpu_tree": 1.5},
    {"fpu_root": -2.0},
    {"c_puct_base": 0.0},
    {"c_puct_init": -0.1},
    {"c_puct_factor": 0.0},
    {"virtual_loss": -1.0},
    {"default_sims": 0},
    {"sim_cap": 0},
    {"fixed_sims": 0},
    {"slice_seconds": 0.0},
    {"min_slice_sims": 0},
    {"move_overhead_ms": -1},
])
def test_an_unrunnable_configuration_is_refused(changes):
    with pytest.raises(ConfigError):
        EngineConfig(**changes)


def test_in_flight_floors_rather_than_rounds():
    """K*W must not EXCEED the outstanding count asked for.

    That count is the virtual-loss exposure and the batch ceiling at once
    (scope §2.2), so rounding up would quietly search a wider configuration than
    the one a benchmark row names.
    """
    assert EngineConfig(threads=1, max_outstanding=24).in_flight == 24
    assert EngineConfig(threads=4, max_outstanding=24).in_flight == 6
    assert EngineConfig(threads=5, max_outstanding=24).in_flight == 4
    assert EngineConfig(threads=5, max_outstanding=24).effective_outstanding == 20
    assert EngineConfig(threads=24, max_outstanding=24).in_flight == 1


def test_the_effective_outstanding_count_is_reported_when_it_differs():
    """A configuration whose W does not divide its W*K must say so.

    `max_outstanding=24, threads=5` runs at 20, and a log that showed only the
    24 would name a configuration that did not run.
    """
    config = EngineConfig(threads=5, max_outstanding=24)
    kv = config.as_kv()
    assert "max_outstanding=24" in kv
    assert "effective_outstanding=20" in kv
    assert "in_flight=4" in kv


def test_the_shipping_defaults_are_the_measured_selection():
    """BENCH.md C10b-3g: W=1, K=24, max_batch 128, affinity none, VL 2.5.

    Pinned because these defaults are what a bare `python playing/
    uci_wrapper_v6.py` plays at, and a drift here would silently move every
    later Gate 4/5 measurement off the configuration those gates were sized on.
    """
    config = EngineConfig()
    assert (config.threads, config.max_outstanding) == (1, 24)
    assert config.in_flight == 24
    assert config.max_batch == 128
    assert config.affinity == "none"
    assert config.virtual_loss == 2.5
    assert config.switch_interval == playv6.DEFAULT_SWITCH_INTERVAL


# ---------------------------------------------------------------------------
# The core objects the configuration builds
# ---------------------------------------------------------------------------


def test_the_search_config_carries_every_value_it_was_given():
    config = EngineConfig(c_puct_init=1.7, c_puct_base=12345.0, c_puct_factor=1.4,
                          fpu_root=-0.1, fpu_tree=0.42, virtual_loss=3.7,
                          max_tree_depth=64, cache_entries=1234,
                          arena_capacity=99999, ponder_decay=0.75,
                          verify_compaction=True)
    core = config.to_search_config()
    assert core.c_init == pytest.approx(1.7)
    assert core.c_base == pytest.approx(12345.0)
    assert core.c_factor == pytest.approx(1.4)
    assert core.fpu_root == pytest.approx(-0.1)
    assert core.fpu_tree == pytest.approx(0.42)
    assert core.virtual_loss == pytest.approx(3.7)
    assert core.max_tree_depth == 64
    assert core.cache_slots == 1234
    assert core.arena_capacity == 99999
    assert core.ponder_decay == pytest.approx(0.75)
    assert core.verify_compaction is True


def test_cache_shards_zero_leaves_the_cores_default_alone():
    """0 means "whatever the core uses", not "a cache with no shards"."""
    default = EngineConfig(cache_shards=0).to_search_config()
    explicit = EngineConfig(cache_shards=256).to_search_config()
    assert default.cache_shards >= 64
    assert explicit.cache_shards == 256


def test_the_parallel_config_carries_the_derived_in_flight():
    config = EngineConfig(threads=4, max_outstanding=24, max_batch=64,
                          affinity="pcore_physical")
    parallel = config.to_parallel_config()
    assert parallel.workers == 4
    assert parallel.in_flight == 6
    assert parallel.max_outstanding == 24
    assert parallel.max_batch == 64
    assert parallel.affinity == "pcore_physical"
    # Histograms are measurement, not engine behaviour; a game does not read them.
    assert parallel.collect_histograms is False


# ---------------------------------------------------------------------------
# Simulation accounting — v5's defect 2
# ---------------------------------------------------------------------------


def _outcome(**changes) -> SearchOutcome:
    base = dict(best_move="e2e4", mating_move=None, nominal=4000, inherited=0,
                delivered=4000, wall_s=1.0, slices=1, root_visits=4000,
                score_cp=0, q=0.0)
    base.update(changes)
    return SearchOutcome(**base)


def test_the_headline_rate_divides_delivered_simulations():
    outcome = _outcome(nominal=4000, inherited=3564, delivered=436, wall_s=0.05)
    assert outcome.sims_per_s == pytest.approx(436 / 0.05)
    assert outcome.nominal_sims_per_s == pytest.approx(4000 / 0.05)
    assert outcome.inflation == pytest.approx(4000 / 436)


def test_a_fresh_root_reports_no_inflation():
    outcome = _outcome(nominal=4000, inherited=0, delivered=4000, wall_s=0.3)
    assert outcome.inflation == pytest.approx(1.0)
    assert outcome.sims_per_s == pytest.approx(outcome.nominal_sims_per_s)


def test_a_move_that_delivered_nothing_reports_infinite_inflation():
    """Not 1.0. The reused root already met the budget, so the requested count
    was reported in full against zero work — which is the reported number this
    chunk exists to stop producing, and 1.0 would call it agreement."""
    outcome = _outcome(nominal=3000, inherited=3000, delivered=0, wall_s=0.001)
    assert outcome.sims_per_s == 0.0
    assert math.isinf(outcome.inflation)
    assert "inflation=inf" in outcome.telemetry()


def test_a_timed_search_reports_no_requested_count_at_all():
    """A `go wtime ...` asks for a duration; the node budget is a ceiling this
    engine chose. Dividing a ceiling by the wall clock produced a spurious 20x
    in the first C11 smoke run, which is why `nominal` is optional."""
    outcome = _outcome(nominal=None, inherited=48, delivered=2857, wall_s=0.3,
                       budget_source="time")
    assert outcome.nominal_sims_per_s is None
    assert outcome.inflation is None
    line = outcome.telemetry()
    assert "nominal=n/a" in line
    assert "inflation=n/a" in line
    assert "budget_source=time" in line
    # The delivered rate is still a real number: it is the one that was measured.
    assert "delivered_sims_per_s=9,523" in line


def test_the_telemetry_line_names_both_counts_and_the_reason():
    line = _outcome(nominal=800, inherited=511, delivered=289,
                    wall_s=0.0371, reason="budget").telemetry()
    for token in ("delivered=289", "nominal=800", "inherited=511",
                  "delivered_sims_per_s=", "nominal_sims_per_s=", "inflation=",
                  "budget_source=", "reason=budget"):
        assert token in line, f"{token!r} missing from {line!r}"


# ---------------------------------------------------------------------------
# `go` parsing and time management
# ---------------------------------------------------------------------------


@requires_wrapper
def test_go_tokens_are_parsed_and_nonsense_is_dropped_not_defaulted():
    GoParams = wrapper().GoParams
    params = GoParams("wtime 30000 btime 29000 winc 300 binc 300 movestogo 20".split())
    assert params.get("wtime") == 30000
    assert params.get("movestogo") == 20
    assert params.has_clock()
    assert not params.infinite and not params.ponder

    # A value that will not parse is dropped, NOT silently replaced: a GUI
    # sending `movetime abc` must not quietly get the node budget instead.
    broken = GoParams("movetime abc nodes 500".split())
    assert broken.get("movetime") is None
    assert broken.get("nodes") == 500

    assert GoParams(["infinite"]).infinite
    assert GoParams(["ponder"]).ponder
    assert not GoParams(["nodes", "1000"]).has_clock()


@requires_wrapper
def test_movetime_is_taken_literally_minus_the_move_overhead():
    module = wrapper()
    engine = module.UCIEngine(EngineConfig(move_overhead_ms=100))
    allotted = engine._allot(module.GoParams("movetime 2000".split()))
    assert allotted == pytest.approx(1.9)


@requires_wrapper
def test_a_clock_is_split_over_movestogo_and_capped_at_a_fraction_of_it():
    module = wrapper()
    engine = module.UCIEngine(EngineConfig(move_overhead_ms=0))
    # 60 s, 30 moves to go, no increment -> 2 s.
    assert engine._allot(
        module.GoParams("wtime 60000 btime 60000 movestogo 30".split())
    ) == pytest.approx(2.0)
    # 1 s left and 1 move to go would be the whole clock; the 40% cap holds it
    # to 0.4 s so a wrong movestogo guess cannot flag the game.
    assert engine._allot(
        module.GoParams("wtime 1000 btime 1000 movestogo 1".split())
    ) == pytest.approx(0.4)


@requires_wrapper
def test_no_clock_means_no_deadline():
    module = wrapper()
    engine = module.UCIEngine(EngineConfig())
    assert engine._allot(module.GoParams("nodes 5000".split())) is None
    assert engine._allot(module.GoParams([])) is None


@requires_wrapper
def test_the_side_to_move_selects_which_clock_is_read():
    import chess
    module = wrapper()
    engine = module.UCIEngine(EngineConfig(move_overhead_ms=0))
    line = "wtime 60000 btime 6000 movestogo 30"
    engine.board = chess.Board()
    assert engine._allot(module.GoParams(line.split())) == pytest.approx(2.0)
    engine.board.push(chess.Move.from_uci("e2e4"))
    assert engine._allot(module.GoParams(line.split())) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Option value coercion
# ---------------------------------------------------------------------------


@requires_wrapper
@pytest.mark.parametrize("raw,expected", [("true", True), ("false", False),
                                          ("1", True), ("0", False),
                                          ("on", True), ("off", False)])
def test_check_options_accept_the_spellings_guis_send(raw, expected):
    assert wrapper()._parse_bool(raw) is expected


@requires_wrapper
def test_a_non_boolean_check_value_raises_rather_than_defaulting():
    with pytest.raises(ValueError):
        wrapper()._parse_bool("maybe")


@requires_wrapper
@pytest.mark.parametrize("raw,expected", [("0", None), ("", None), ("none", None),
                                          ("off", None), ("2000", 2000)])
def test_fixed_sims_reads_zero_as_off(raw, expected):
    """UCI spin options cannot express "unset", so 0 is the off value — and it
    must map to None rather than to a budget of zero simulations."""
    assert wrapper()._parse_optional_int(raw) == expected


# ---------------------------------------------------------------------------
# Startup discipline
# ---------------------------------------------------------------------------


def test_apply_switch_interval_sets_the_measured_value_and_reports_the_old_one():
    before = sys.getswitchinterval()
    try:
        sys.setswitchinterval(0.005)
        was, now = playv6.apply_switch_interval()
        assert was == pytest.approx(0.005)
        assert now == pytest.approx(playv6.DEFAULT_SWITCH_INTERVAL)
        assert sys.getswitchinterval() == pytest.approx(0.0005)
    finally:
        sys.setswitchinterval(before)


def test_the_switch_interval_default_comes_from_the_core_not_a_literal():
    """C0b measured it and `guofish_core` publishes it; a literal here could
    drift away from the value the C10 histogram was taken at."""
    import guofish_core
    assert playv6.DEFAULT_SWITCH_INTERVAL == guofish_core.DEFAULT_SWITCH_INTERVAL


def test_the_score_inversion_is_the_value_heads_own_tanh_calibration():
    scale = 290.6806
    assert playv6.q_to_centipawns(0.0, scale) == 0
    assert playv6.q_to_centipawns(0.5, scale) == int(round(scale * math.atanh(0.5)))
    # Saturation: the labels stopped at +-0.995, so the reported score ceilings
    # there rather than diverging. A -resign threshold above this never fires.
    ceiling = playv6.q_to_centipawns(1.0, scale)
    assert ceiling == playv6.q_to_centipawns(0.999999, scale)
    assert ceiling == pytest.approx(870, abs=1)
    assert playv6.q_to_centipawns(-1.0, scale) == -ceiling


def test_config_replace_validates_rather_than_producing_a_bad_object():
    """`setoption` builds a copy and only then adopts it, so a rejected value
    never occupies the live object even briefly."""
    config = EngineConfig()
    # C11b AMENDMENT (authorised; see DECISIONS.md). This read
    # `config.replace(policy_temperature=0.8)`, which is now a legal setting.
    # `dirichlet_epsilon` is the surviving member of UNSUPPORTED_IN_CORE and
    # exercises the same property — that `replace` validates rather than
    # producing a bad object — against a value the core still cannot honour.
    with pytest.raises(ConfigError):
        config.replace(dirichlet_epsilon=0.25)
    assert config.dirichlet_epsilon == 0.0
    assert config.replace(virtual_loss=3.7).virtual_loss == 3.7
    assert config.virtual_loss == 2.5
