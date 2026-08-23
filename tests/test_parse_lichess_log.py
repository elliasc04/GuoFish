"""L0's parser, against synthetic logs in both dialects.

WHY SYNTHETIC AND NOT THE REAL LOG. The real logs are 33 MB and live outside this
repository, so a test that read them would be a test of the operator's disk. The
fixtures below are verbatim-shaped excerpts — real line prefixes, real `info
string` grammar, real timestamps — trimmed to the smallest case that pins each
behaviour.

WHAT IS PINNED HERE, and each of these was a real hazard while writing it:

  * the move's clock starts at `ponderhit`, not at `go ponder`;
  * `delivered`, `inherited` and `nominal` come off the `info string` and not off
    the `nodes` field, which is the same number only sometimes;
  * the gap decomposes into a pre-search leg and a tail leg with a small
    residual, which is the whole of R4;
  * the INFO-only dialect degrades to the fields it actually has rather than
    inventing the rest;
  * a game that ended on time is recognised.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from telemetry.parse_lichess_log import (  # noqa: E402
    LogParser, _parse_go, _parse_info, _parse_info_string, _si, pct,
)

WIRE = """\
2026-08-17 01:33:58,605 lib.lichess_bot (lichess_bot.py:684) INFO +++ https://lichess.org/ktKNqw1a/black Rapid vs BOT LegoTechnicControlPl (2833) (ktKNqw1a)
2026-08-17 01:34:09,255 chess.engine (engine.py:950) DEBUG <UciProtocol (pid=31612)>: << go wtime 600000 btime 600000 winc 3000 binc 3000 nodes 200000
2026-08-17 01:34:09,555 chess.engine (engine.py:976) DEBUG <UciProtocol (pid=31612)>: >> info depth 1 seldepth 1 time 250 nodes 4000 nps 16000 hashfull 10 score cp 12
2026-08-17 01:34:21,555 chess.engine (engine.py:976) DEBUG <UciProtocol (pid=31612)>: >> info depth 1 seldepth 1 time 12250 nodes 196000 nps 16000 hashfull 90 score cp 15
2026-08-17 01:34:22,055 chess.engine (engine.py:976) DEBUG <UciProtocol (pid=31612)>: >> info depth 12 seldepth 27 time 12500 nodes 199999 nps 15999 hashfull 92 score cp 16 pv e2e4 e7e5
2026-08-17 01:34:22,056 chess.engine (engine.py:976) DEBUG <UciProtocol (pid=31612)>: >> info string source=search delivered=199999 nominal=200000 inherited=0 delivered_nps=15999 nominal_nps=16000 inflation=1.00 ponder_sims=0 search_sims=199999 total_sims=199999 arena_exhausted=false arena_util=0.132 budget_source=nodes reason=budget game_counts search=1 book=3 tablebase=0
2026-08-17 01:34:22,057 chess.engine (engine.py:976) DEBUG <UciProtocol (pid=31612)>: >> bestmove e2e4 ponder e7e5
2026-08-17 01:34:22,700 urllib3.connectionpool (connectionpool.py:544) DEBUG https://lichess.org:443 "POST /api/bot/game/ktKNqw1a/move/e2e4?offeringDraw=false HTTP/1.1" 200 11
2026-08-17 01:34:22,800 chess.engine (engine.py:950) DEBUG <UciProtocol (pid=31612)>: << go ponder wtime 597000 btime 600000 winc 3000 binc 3000 nodes 200000
2026-08-17 01:34:29,000 chess.engine (engine.py:950) DEBUG <UciProtocol (pid=31612)>: << ponderhit
2026-08-17 01:34:29,900 chess.engine (engine.py:976) DEBUG <UciProtocol (pid=31612)>: >> info depth 1 seldepth 1 time 500 nodes 8000 nps 16000 hashfull 200 score cp 14
2026-08-17 01:34:41,400 chess.engine (engine.py:976) DEBUG <UciProtocol (pid=31612)>: >> info depth 1 seldepth 1 time 12000 nodes 192000 nps 16000 hashfull 300 score cp 14
2026-08-17 01:34:42,900 chess.engine (engine.py:976) DEBUG <UciProtocol (pid=31612)>: >> info depth 11 seldepth 30 time 12500 nodes 200000 nps 16000 hashfull 310 score cp 13 pv g1f3
2026-08-17 01:34:42,901 chess.engine (engine.py:976) DEBUG <UciProtocol (pid=31612)>: >> info string source=search delivered=200000 nominal=200000 inherited=194824 delivered_nps=16000 nominal_nps=16000 inflation=1.00 ponder_sims=6219 search_sims=200000 total_sims=206219 arena_exhausted=false arena_util=0.264 budget_source=ponderhit reason=budget game_counts search=2 book=3 tablebase=0
2026-08-17 01:34:42,902 chess.engine (engine.py:976) DEBUG <UciProtocol (pid=31612)>: >> bestmove g1f3 ponder b8c6
2026-08-17 01:34:43,500 urllib3.connectionpool (connectionpool.py:544) DEBUG https://lichess.org:443 "POST /api/bot/game/ktKNqw1a/move/g1f3?offeringDraw=false HTTP/1.1" 200 11
2026-08-17 01:34:44,000 chess.engine (engine.py:950) DEBUG <UciProtocol (pid=31612)>: << go ponder wtime 590000 btime 600000 winc 3000 binc 3000 nodes 200000
2026-08-17 01:34:46,000 chess.engine (engine.py:950) DEBUG <UciProtocol (pid=31612)>: << stop
2026-08-17 01:34:46,100 chess.engine (engine.py:976) DEBUG <UciProtocol (pid=31612)>: >> bestmove b1c3 ponder d7d5
2026-08-17 01:35:00,000 lib.lichess_bot (lichess_bot.py:456) DEBUG Event: {'type': 'gameFinish', 'game': {'gameId': 'ktKNqw1a', 'color': 'black', 'status': {'id': 35, 'name': 'outoftime'}, 'winner': 'white'}}
2026-08-17 01:35:01,000 lib.lichess_bot (lichess_bot.py:923) INFO --- https://lichess.org/ktKNqw1a/black Game over
"""

INFO_ONLY = """\
2026-08-15 21:20:20,897 lib.lichess_bot (lichess_bot.py:684) INFO +++ https://lichess.org/e4Ge15yh/white Rapid vs BOT scipio-bot (1508) (e4Ge15yh)
2026-08-15 21:20:35,183 lib.lichess_bot (lichess_bot.py:838) INFO move: 2
2026-08-15 21:20:35,183 lib.engine_wrapper (engine_wrapper.py:726) INFO Searching for wtime 598000 btime 600000 for game e4Ge15yh
2026-08-15 21:20:44,456 lib.engine_wrapper (engine_wrapper.py:334) INFO Source: Engine
2026-08-15 21:20:44,456 lib.engine_wrapper (engine_wrapper.py:334) INFO Evaluation: 1.07
2026-08-15 21:20:44,456 lib.engine_wrapper (engine_wrapper.py:334) INFO Depth: 12
2026-08-15 21:20:44,456 lib.engine_wrapper (engine_wrapper.py:334) INFO Nodes: 200.0K
2026-08-15 21:20:44,456 lib.engine_wrapper (engine_wrapper.py:334) INFO Speed: 21.9Knps
2026-08-15 21:20:44,457 lib.engine_wrapper (engine_wrapper.py:334) INFO Pv: 2. d4 c6 3. c4 d6
2026-08-15 21:20:50,000 lib.lichess_bot (lichess_bot.py:923) INFO --- https://lichess.org/e4Ge15yh/white Game over
"""


@pytest.fixture
def wire_games(tmp_path):
    path = tmp_path / "wire.log"
    path.write_text(WIRE, encoding="utf-8")
    return LogParser(path).parse()


@pytest.fixture
def info_games(tmp_path):
    path = tmp_path / "info.txt"
    path.write_text(INFO_ONLY, encoding="utf-8")
    return LogParser(path).parse()


# --- the token grammars ----------------------------------------------------


def test_go_parsing_keeps_ponder_apart_from_the_numbers():
    assert _parse_go("go ponder wtime 1 btime 2 winc 3 binc 4 nodes 5") == {
        "ponder": True, "infinite": False,
        "wtime": 1, "btime": 2, "winc": 3, "binc": 4, "nodes": 5}
    assert _parse_go("go nodes 200000 movetime 10000")["ponder"] is False


def test_info_parsing_swallows_the_pv_tail():
    got = _parse_info("info depth 12 seldepth 27 time 12500 nodes 199999 nps "
                      "15999 hashfull 92 score cp 16 pv e2e4 e7e5 g1f3")
    assert got["time"] == 12500 and got["nodes"] == 199999
    assert got["score"] == 16 and got["score_kind"] == "cp"
    assert got["pv"] == ["e2e4", "e7e5", "g1f3"]


def test_info_string_turns_n_a_into_none_rather_than_a_string():
    """An aggregate must not be able to sum the string `n/a`."""
    got = _parse_info_string("source=search delivered=47516 nominal=n/a "
                             "inherited=29533 nominal_nps=n/a inflation=n/a "
                             "arena_exhausted=false arena_util=1.000 "
                             "budget_source=time reason=stop "
                             "game_counts search=9 book=0 tablebase=2")
    assert got["nominal"] is None and got["nominal_nps"] is None
    assert got["delivered"] == 47516 and got["inherited"] == 29533
    assert got["arena_exhausted"] is False and got["arena_util"] == 1.0
    assert got["search"] == 9 and got["tablebase"] == 2


@pytest.mark.parametrize("text,want", [
    ("200.0K", 200000), ("1.2M", 1200000), ("815", 815), ("21.9", 22),
    ("nonsense", None),
])
def test_si_suffixes(text, want):
    assert _si(text) == want


def test_percentiles_interpolate_and_survive_a_single_sample():
    assert pct([5.0], 50) == 5.0
    assert pct([0.0, 10.0], 50) == 5.0
    assert pct([0.0, 1.0, 2.0, 3.0], 100) == 3.0


# --- the wire dialect ------------------------------------------------------


def test_wire_dialect_finds_the_game_and_its_moves(wire_games):
    assert len(wire_games) == 1
    game = wire_games[0]
    assert game.game_id == "ktKNqw1a"
    assert game.our_color == "black"
    assert game.opponent == "LegoTechnicControlPl"
    assert game.opponent_rating == 2833
    # Two moves, and the `stop`-answered ponder is NOT one of them: the GUI
    # throws that bestmove away.
    assert len(game.moves) == 2
    assert [m.played for m in game.moves] == ["e2e4", "g1f3"]
    assert game.ponder_hits == 1
    assert game.ponder_misses == 1


def test_a_pondered_moves_clock_starts_at_ponderhit_not_at_go_ponder(wire_games):
    """The single most consequential decision in the parser.

    Everything between `go ponder` and `ponderhit` ran on the OPPONENT's clock.
    Timing the move from the `go ponder` would charge us 6.2 s we never spent
    and would make every pondered move look catastrophically slow.
    """
    move = wire_games[0].moves[1]
    assert move.go_kind == "ponderhit"
    assert move.was_pondering is True
    assert move.ponder_verdict == "hit"
    # `go ponder` at 22.800, `ponderhit` at 29.000, `bestmove` at 42.902.
    assert move.ponder_window_ms == pytest.approx(6200, abs=1)
    assert move.bestmove_ms == pytest.approx(13902, abs=1)


def test_the_gap_decomposes_into_two_legs_with_a_small_residual(wire_games):
    """R4, as an arithmetic identity rather than a narrative."""
    fresh, pondered = wire_games[0].moves

    # Fresh: `go` 09.255, first info 09.555 reporting time=250 -> 50 ms before
    # the first simulation. Final info 22.055 at time=12500, penultimate 21.555
    # at time=12250: 500 ms of wall for 250 ms of search -> a 250 ms PV walk.
    assert fresh.search_wall_ms == 12500
    assert fresh.bestmove_ms == pytest.approx(12802, abs=1)
    assert fresh.ponder_stop_ms_derived == pytest.approx(50, abs=1)
    assert fresh.tail_ms_derived == pytest.approx(250, abs=1)
    assert fresh.gap_ms_derived == pytest.approx(302, abs=1)
    assert abs(fresh.residual_ms_derived) < 5

    # Pondered: `ponderhit` 29.000, first info 29.900 at time=500 -> a 400 ms
    # pre-search leg, which is the ponder exit plus the branch-table dump.
    assert pondered.ponder_stop_ms_derived == pytest.approx(400, abs=1)
    assert pondered.tail_ms_derived == pytest.approx(1000, abs=1)
    assert abs(pondered.residual_ms_derived) < 5


def test_the_counts_come_off_the_info_string_not_off_the_nodes_field(wire_games):
    fresh, pondered = wire_games[0].moves
    assert (fresh.delivered, fresh.inherited, fresh.nominal) == (199999, 0, 200000)
    assert fresh.identity_holds_derived is False   # 199,999 + 0 != 200,000
    assert fresh.budget_source == "nodes"

    assert (pondered.delivered, pondered.inherited) == (200000, 194824)
    assert pondered.budget_source == "ponderhit"
    assert pondered.ponder_sims == 6219
    # `delivered + inherited == nominal` cannot hold on a ponderhit budget.
    assert pondered.identity_holds_derived is False
    assert pondered.reuse_ratio_derived == pytest.approx(194824 / 200000)
    assert pondered.tree_size_at_go_derived == 194824


def test_the_move_ack_leg_is_measured_from_the_post(wire_games):
    fresh = wire_games[0].moves[0]
    # bestmove 22.057 -> POST returned 22.700
    assert fresh.ack_ms == pytest.approx(643, abs=1)


def test_a_game_that_ended_on_time_is_recognised(wire_games):
    game = wire_games[0]
    assert game.result == "outoftime"
    assert game.ended_on_time is True
    assert game.winner == "white"        # we were black; we flagged


# --- the INFO-only dialect -------------------------------------------------


def test_info_only_dialect_degrades_to_the_fields_it_has(info_games):
    assert len(info_games) == 1
    game = info_games[0]
    assert game.our_color == "white"
    assert len(game.moves) == 1
    move = game.moves[0]
    assert move.dialect == "info"
    assert move.move_number == 2
    assert move.delivered == 200000
    assert move.reported_nps == 21900
    assert move.score_cp == 107
    # `Searching for` 35.183 -> `Source:` 44.456
    assert move.bestmove_ms == pytest.approx(9273, abs=1)
    # No `info` lines exist in this dialect, so search wall is DERIVED from
    # delivered / nps and the wire-only legs are absent rather than invented.
    assert move.search_wall_ms == pytest.approx(1000 * 200000 / 21900, abs=1)
    assert move.ponder_stop_ms_derived is None
    assert move.tail_ms_derived is None
    assert move.ack_ms is None
    assert move.inherited is None


def test_the_two_dialects_do_not_contaminate_each_other(tmp_path):
    """A wire log also contains the INFO lines; the wire path must win.

    lichess-bot writes both `Searching for ...` and the UCI traffic in the same
    DEBUG file. A parser that let the INFO handler open a second move record
    would double-count every move in every wire log.
    """
    path = tmp_path / "both.log"
    path.write_text(
        WIRE.replace(
            "2026-08-17 01:34:09,255 chess.engine",
            "2026-08-17 01:34:09,100 lib.lichess_bot (lichess_bot.py:838) INFO move: 7\n"
            "2026-08-17 01:34:09,150 lib.engine_wrapper (engine_wrapper.py:726) "
            "INFO Searching for wtime 600000 btime 600000 for game ktKNqw1a\n"
            "2026-08-17 01:34:09,255 chess.engine", 1),
        encoding="utf-8")
    games = LogParser(path).parse()
    assert len(games[0].moves) == 2
    assert all(m.dialect == "wire" for m in games[0].moves)
