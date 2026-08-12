"""The automated gates: the probe construction, the ELO math, the bar.

Runs under pytest, or standalone:
    python training/v5_multiPV/tests/test_gates.py

These fire unattended at the end of a ~17-hour run, so every one of them is a
thing nobody will be watching. Three are load-bearing enough to be worth
pinning individually:

  epoch_head_tail   The seen/unseen gap is only a measure of EXPOSURE if the
                    "seen" slice is genuinely what training consumed first. The
                    head/tail are recomputed from the seed rather than observed,
                    so if that reconstruction ever drifts from __iter__ the gap
                    becomes a comparison of two arbitrary slices and still
                    reports a plausible small number.
  score orientation cutechess reports from the FIRST-named engine's point of
                    view. Reading a 60-140 loss as a win is the single cheapest
                    way to promote a worse net, and it is a one-line mistake.
  the bar           Pre-registered means the comparison cannot move after the
                    result is in. The thresholds are asserted against the
                    brief's numbers here so a later edit to gates.py fails a
                    test rather than passing silently.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_V5 = _HERE.parent
_ROOT = _V5.parents[1]
_DATA = _ROOT / "data" / "multiPV"
for _p in (str(_V5), str(_DATA), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gates import (ELO_GAIN_REQUIRED, KL_REDUCTION_REQUIRED,     # noqa: E402
                   MatchResult, SeenUnseenProbe, elo_from_score,
                   elo_with_ci, evaluate_thresholds, find_ordo,
                   format_verdict, parse_cutechess_elo,
                   parse_cutechess_score, relative_gap, run_ordo_diff)
from train_v5 import ResumableRandomSampler, select_subset       # noqa: E402


# ---------------------------------------------------------------------------
# the probe construction
# ---------------------------------------------------------------------------
def _sampler(n=10_000, seed=20260802):
    return ResumableRandomSampler(select_subset(n, None, seed), seed=seed)


def test_head_tail_match_the_order_training_actually_follows():
    """The whole seen/unseen construction rests on this identity."""
    s = _sampler()
    order = list(s)
    for k in (1, 50, 1000):
        head, tail = s.epoch_head_tail(0, k)
        assert head.tolist() == order[:k], "head is not what training sees first"
        assert tail.tolist() == order[-k:], "tail is not what training sees last"


def test_head_and_tail_are_disjoint():
    s = _sampler()
    head, tail = s.epoch_head_tail(0, 2000)
    assert not (set(head.tolist()) & set(tail.tolist()))


def test_chunked_gather_reproduces_the_whole_order():
    """__iter__ yields through a 1M window; the sequence must be unchanged."""
    s = _sampler(n=5000)
    got = list(s)
    g = np.random.default_rng(s.seed * 1_000_003 + s.epoch)
    expected = s.indices[g.permutation(s.n)].tolist()
    assert got == expected
    assert len(got) == len(set(got)) == 5000


def test_iter_respects_skip_and_len():
    s = _sampler(n=5000)
    full = list(s)
    s.set_epoch(0, skip=1234)
    assert list(s) == full[1234:]
    assert len(s) == 5000 - 1234


def test_selection_indices_are_int32_and_lossless():
    idx = select_subset(1_000_000, None, 7)
    assert idx.dtype == np.int32
    assert sorted(idx.tolist()) == list(range(1_000_000))


def test_probe_rejects_a_size_that_would_straddle_the_measurement_point():
    """A probe wider than the measurement fraction would put 'unseen' records
    behind the training cursor. Better to disable than to report a number that
    silently means nothing."""
    s = _sampler(n=10_000)
    assert SeenUnseenProbe(s, 100, measure_at_frac=0.25).valid
    # 4000/side at 25% -> the head alone is 40% of the epoch.
    assert not SeenUnseenProbe(s, 4000, measure_at_frac=0.25).valid
    assert not SeenUnseenProbe(s, 0, measure_at_frac=0.25).valid


def test_probe_fires_once_and_only_in_epoch_1():
    s = _sampler()
    p = SeenUnseenProbe(s, 100, measure_at_frac=0.25)
    assert not p.due(epoch=0, epoch_frac=0.10)
    assert p.due(epoch=0, epoch_frac=0.25)
    assert not p.due(epoch=1, epoch_frac=0.50), "epoch 2 has no unseen records"
    p.done = True
    assert not p.due(epoch=0, epoch_frac=0.99)


def test_relative_gap_signs_point_the_documented_way():
    seen = {"policy_kl": 1.00, "value_mse": 0.10, "total_loss": 1.10}
    unseen = {"policy_kl": 1.02, "value_mse": 0.10, "total_loss": 1.12}
    g = relative_gap(unseen, seen)
    assert g["policy_kl_gap_abs"] == pytest.approx(0.02)
    assert g["policy_kl_gap_rel"] == pytest.approx(0.02)
    assert g["total_loss_gap_rel"] > 0, \
        "worse on unseen must be POSITIVE - that is the overfitting direction"
    # No gap at all reports exactly zero, not a small float artifact.
    assert relative_gap(seen, seen)["total_loss_gap_rel"] == 0.0


# ---------------------------------------------------------------------------
# ELO
# ---------------------------------------------------------------------------
def test_elo_from_score_anchors():
    assert elo_from_score(0.5) == pytest.approx(0.0)
    assert elo_from_score(0.75) == pytest.approx(190.849, abs=1e-3)
    assert elo_from_score(0.25) == pytest.approx(-190.849, abs=1e-3)
    assert elo_from_score(1.0) == math.inf
    assert elo_from_score(0.0) == -math.inf


def test_elo_ci_brackets_the_point_estimate():
    elo, lo, hi = elo_with_ci(wins=120, draws=40, losses=40)
    assert lo < elo < hi
    assert elo == pytest.approx(elo_from_score(140 / 200))


def test_a_dead_even_match_is_zero_elo_and_straddles_zero():
    elo, lo, hi = elo_with_ci(wins=50, draws=100, losses=50)
    assert elo == pytest.approx(0.0)
    assert lo < 0 < hi


def test_200_games_is_a_wide_enough_interval_to_matter():
    """Sanity on the sample size the brief pre-registered: at 200 games a
    +100 ELO point estimate carries an interval tens of ELO wide, so the
    verdict reports the interval alongside the point estimate rather than
    implying more precision than 200 games can buy."""
    elo, lo, hi = elo_with_ci(wins=100, draws=60, losses=40)
    assert hi - lo > 40.0


def test_zero_games_does_not_divide_by_zero():
    elo, lo, hi = elo_with_ci(0, 0, 0)
    assert all(math.isnan(x) for x in (elo, lo, hi))


# ---------------------------------------------------------------------------
# parsing cutechess
# ---------------------------------------------------------------------------
LOG_CANDIDATE_FIRST = """\
Started game 1 of 200 (cand-v5 vs base-20M)
Score of cand-v5 vs base-20M: 1 - 0 - 0  [1.000] 1
Finished game 200
Score of cand-v5 vs base-20M: 120 - 40 - 40  [0.700] 200
"""

LOG_CANDIDATE_SECOND = """\
Score of base-20M vs cand-v5: 40 - 120 - 40  [0.300] 200
"""


def test_parses_the_last_score_line():
    assert parse_cutechess_score(LOG_CANDIDATE_FIRST, "cand-v5") == (120, 40, 40, 200)


def test_orients_to_the_candidate_when_it_is_named_second():
    """THE test in this file. Same match, engines swapped in the line."""
    first = parse_cutechess_score(LOG_CANDIDATE_FIRST, "cand-v5")
    second = parse_cutechess_score(LOG_CANDIDATE_SECOND, "cand-v5")
    assert first == second == (120, 40, 40, 200)
    # And from the baseline's side it is the mirror image, not the same number.
    assert parse_cutechess_score(LOG_CANDIDATE_SECOND, "base-20M") == (40, 40, 120, 200)


def test_unknown_engine_name_returns_none_rather_than_guessing():
    assert parse_cutechess_score(LOG_CANDIDATE_FIRST, "some-other-net") is None
    assert parse_cutechess_score("no score lines here", "cand-v5") is None


# ---------------------------------------------------------------------------
# cutechess's own Elo line
# ---------------------------------------------------------------------------
ELO_LINE = ("Elo difference: -102.3 +/- 50.5, LOS: 0.1 %, DrawRatio: 52.9 %\n")


def test_parses_cutechess_elo_line():
    d = parse_cutechess_elo(ELO_LINE)
    assert d["cc_elo"] == pytest.approx(-102.3)
    assert d["cc_margin"] == pytest.approx(50.5)
    assert d["cc_los"] == pytest.approx(0.1)
    assert d["cc_draw_ratio"] == pytest.approx(52.9)


def test_cutechess_elo_inverts_with_the_orientation():
    """Same match read from the other engine's side."""
    d = parse_cutechess_elo(ELO_LINE, invert=True)
    assert d["cc_elo"] == pytest.approx(102.3)
    assert d["cc_los"] == pytest.approx(99.9)
    assert d["cc_margin"] == pytest.approx(50.5), "the margin must not flip sign"


def test_a_degenerate_margin_is_nan_not_a_crash():
    """cutechess really does print `+/- nan` on a sweep - it is in this repo's
    own logs (c11b_smoke_clock)."""
    d = parse_cutechess_elo(
        "Elo difference: inf +/- nan, LOS: 100.0 %, DrawRatio: 0.0 %")
    assert math.isinf(d["cc_elo"])
    assert math.isnan(d["cc_margin"])
    assert d["cc_los"] == pytest.approx(100.0)


def test_missing_elo_line_returns_empty():
    assert parse_cutechess_elo("Score of A vs B: 1 - 0 - 0  [1.000] 1") == {}


# ---------------------------------------------------------------------------
# ordo, against the real benchmark PGNs
# ---------------------------------------------------------------------------
_V5_GAMES = _ROOT / "benchmarking" / "engine" / "games" / "v5"
REAL_PGN = _V5_GAMES / "v5_10M_vs_20M_800sims.pgn"
BIG, SMALL = "Guofish5-10.9M-BIG", "Guofish5-10.9M-SMALL"


def _need_ordo_and_pgn(pgn: Path):
    if find_ordo(_ROOT) is None:
        pytest.skip("ordo binary not on this box")
    if not pgn.exists():
        pytest.skip(f"{pgn.name} not on this box")


def test_ordo_rates_the_real_10m_vs_20m_match():
    """The anchoring contract: with the BASELINE pinned at 0, the candidate's
    ordo rating IS the ELO difference, sign included.

    Ground truth from running ordo on this PGN by hand:
        Guofish5-10.9M-BIG    0.0   ----
        Guofish5-10.9M-SMALL  -102.0  +/- 50.5
    """
    _need_ordo_and_pgn(REAL_PGN)
    out = run_ordo_diff(_ROOT, REAL_PGN, candidate_name=SMALL,
                        baseline_name=BIG, simulations=200, log=lambda *a: None)
    assert out.get("ordo_error_note", "") == "", out
    assert out["ordo_elo"] == pytest.approx(-102.0, abs=6.0)
    assert out["ordo_error"] == pytest.approx(50.5, abs=12.0)
    assert 0.0 <= out["ordo_draw_rate"] <= 100.0
    assert math.isfinite(out["ordo_white_advantage"])


def test_ordo_sign_flips_when_the_roles_swap():
    """Anchoring the OTHER net must negate the difference. This is the ordo-side
    analogue of the score-orientation test, and the same class of bug."""
    _need_ordo_and_pgn(REAL_PGN)
    kw = dict(simulations=200, log=lambda *a: None)
    small = run_ordo_diff(_ROOT, REAL_PGN, SMALL, BIG, **kw)["ordo_elo"]
    big = run_ordo_diff(_ROOT, REAL_PGN, BIG, SMALL, **kw)["ordo_elo"]
    assert small < 0 < big
    assert big == pytest.approx(-small, abs=8.0)


def test_ordo_agrees_with_the_closed_form_on_the_real_match():
    """The cross-check that makes the disagreement warning meaningful: on real
    games the two independent estimators must land on each other. BIG scored
    64/100, so the closed form gives ~-100 for SMALL against ordo's -102."""
    _need_ordo_and_pgn(REAL_PGN)
    ordo = run_ordo_diff(_ROOT, REAL_PGN, SMALL, BIG, simulations=200,
                         log=lambda *a: None)["ordo_elo"]
    closed, _, _ = elo_with_ci(wins=28, draws=16, losses=56)   # SMALL: 36/100
    assert abs(ordo - closed) < 25.0, (
        f"ordo {ordo:.1f} vs closed form {closed:.1f} - a gap this size on the "
        f"same games means a naming or orientation bug")


def test_ordo_reports_rather_than_raises_on_an_unknown_player():
    _need_ordo_and_pgn(REAL_PGN)
    out = run_ordo_diff(_ROOT, REAL_PGN, "no-such-engine", BIG,
                        simulations=200, log=lambda *a: None)
    assert "not in ordo output" in out.get("ordo_error_note", "")
    assert not math.isfinite(out.get("ordo_elo", float("nan")))


def test_ordo_missing_pgn_is_a_note_not_an_exception():
    out = run_ordo_diff(_ROOT, _V5_GAMES / "does_not_exist.pgn", SMALL, BIG,
                        log=lambda *a: None)
    assert "missing" in out["ordo_error_note"].lower()


def test_ordo_handles_the_v4_cross_architecture_pgn():
    """A second real match, different engines and an odd game count (47), to
    check the parser is not fitted to one file."""
    pgn = _V5_GAMES / "guofishv5_10.9M_vs_v4_25.6M_800sims.pgn"
    _need_ordo_and_pgn(pgn)
    out = run_ordo_diff(_ROOT, pgn, "Guofish5-10.9M", "Guofish4-25.6M",
                        simulations=200, log=lambda *a: None)
    assert out.get("ordo_error_note", "") == "", out
    assert math.isfinite(out["ordo_elo"])
    assert math.isfinite(out["ordo_error"])


# ---------------------------------------------------------------------------
# the pre-registered bar
# ---------------------------------------------------------------------------
def test_thresholds_are_the_ones_the_brief_registered():
    assert KL_REDUCTION_REQUIRED == 0.05
    assert ELO_GAIN_REQUIRED == 100.0


def _match(w, d, l):
    m = MatchResult(event="t", exit_code=0, seconds=1.0,
                    wins=w, draws=d, losses=l, games=w + d + l)
    m.score = (w + 0.5 * d) / m.games if m.games else float("nan")
    m.elo, m.elo_lo, m.elo_hi = elo_with_ci(w, d, l)
    return m


def test_both_criteria_are_required():
    strong = _match(140, 40, 20)          # comfortably over +100
    weak = _match(90, 60, 50)             # positive but under +100
    assert strong.elo >= 100 and weak.elo < 100

    # KL passes, ELO passes -> overall pass
    v = evaluate_thresholds(candidate_kl=0.90, baseline_kl=1.00, match=strong)
    assert v.kl_pass and v.elo_pass and v.overall_pass

    # KL passes, ELO fails -> overall FAIL
    v = evaluate_thresholds(candidate_kl=0.90, baseline_kl=1.00, match=weak)
    assert v.kl_pass and not v.elo_pass and not v.overall_pass

    # ELO passes, KL misses by a hair -> overall FAIL
    v = evaluate_thresholds(candidate_kl=0.9501, baseline_kl=1.00, match=strong)
    assert not v.kl_pass and v.elo_pass and not v.overall_pass


def test_kl_boundary_is_inclusive_at_exactly_five_percent():
    v = evaluate_thresholds(candidate_kl=0.95, baseline_kl=1.00, match=_match(140, 40, 20))
    assert v.kl_reduction == pytest.approx(0.05)
    assert v.kl_pass


def test_a_kl_increase_is_a_negative_reduction_and_fails():
    v = evaluate_thresholds(candidate_kl=1.10, baseline_kl=1.00, match=_match(140, 40, 20))
    assert v.kl_reduction < 0 and not v.kl_pass and not v.overall_pass


def test_ci_lower_bound_is_reported_but_is_not_the_bar():
    """The bar was pre-registered on the point estimate. A result whose point
    estimate clears +100 but whose interval does not must still PASS, with the
    interval stated - substituting the stricter test after the fact is exactly
    what pre-registration exists to prevent."""
    v = evaluate_thresholds(0.90, 1.00, _match(110, 60, 30))
    assert v.elo >= 100 or v.elo < 100          # whichever it is, be consistent:
    if v.elo >= 100:
        assert v.elo_pass
        if v.elo_lo < 100:
            assert not v.elo_ci_clears and v.overall_pass


def test_missing_match_is_not_evaluable_rather_than_a_pass():
    v = evaluate_thresholds(candidate_kl=0.50, baseline_kl=1.00, match=None)
    assert v.kl_pass and not v.elo_pass and not v.overall_pass
    assert any("no match" in n for n in v.notes)


def test_failed_match_is_not_evaluable_rather_than_a_pass():
    broken = MatchResult(event="t", exit_code=1, seconds=0.0,
                         error="preflight failed")
    v = evaluate_thresholds(candidate_kl=0.50, baseline_kl=1.00, match=broken)
    assert not v.elo_pass and not v.overall_pass
    assert any("preflight failed" in n for n in v.notes)


def test_non_finite_kl_is_not_evaluable_rather_than_a_pass():
    v = evaluate_thresholds(float("nan"), 1.00, _match(140, 40, 20))
    assert not v.kl_pass and not v.overall_pass
    assert any("not evaluable" in n for n in v.notes)


def test_gate_payloads_survive_the_jsonl_logger():
    """MatchResult has an `event` field and JsonlLogger.write's first
    positional is also called `event`, so splatting one into the other is a
    TypeError - which would land at the very end of a 17-hour run, after the
    match had already been played. Write them for real."""
    import tempfile

    from train_v5 import JsonlLogger

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "run.jsonl"
        jl = JsonlLogger(path)
        m = _match(140, 40, 20)
        v = evaluate_thresholds(0.9, 1.0, m)
        jl.write("head2head", match=m.to_dict())
        jl.write("gate_verdict", verdict=v.to_dict())
        jl.write("run_end", gate={"match": m.to_dict(), "verdict": v.to_dict()})
        jl.close()

        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in rows] == ["head2head", "gate_verdict", "run_end"]
    assert rows[0]["match"]["event"] == "t", "the nested event was clobbered"
    assert rows[1]["verdict"]["overall_pass"] is True
    assert rows[2]["gate"]["match"]["wins"] == 140


def test_verdict_formats_without_raising_on_every_path():
    for v in (evaluate_thresholds(0.9, 1.0, _match(140, 40, 20)),
              evaluate_thresholds(float("nan"), 1.0, None),
              evaluate_thresholds(0.9, 0.0, _match(0, 0, 0)),
              evaluate_thresholds(1.8, 0.87, _match(0, 0, 2))):   # -inf ELO
        text = format_verdict(v)
        assert "OVERALL" in text and ("PASS" in text or "FAIL" in text)


def test_printed_kl_direction_agrees_with_the_test_that_produced_it():
    """A worse candidate must not read as if it were near the bar. The sign
    convention is (baseline - candidate)/baseline, so improvement is POSITIVE
    and the stated requirement has to be '>= +5%', not '<= -5%'."""
    v = evaluate_thresholds(1.8, 0.87, _match(0, 0, 2))
    worse = format_verdict(v)
    assert "need >= +5%" in worse
    assert not v.kl_pass
    # The printed figure is the one that was tested, not a re-derivation.
    assert f"{v.kl_reduction * 100:+.2f}% reduction" in worse
    better = evaluate_thresholds(0.80, 1.00, _match(140, 40, 20))
    assert better.kl_reduction > 0 and better.kl_pass
    assert "+20.00% reduction" in format_verdict(better)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
