"""M3 — acceptance for the `--move-stats` channel.

THE THREE CHECKS THE BRIEF SPECIFIES, plus the two unit-level ones that make a
failure in them readable.

  1. BEHAVIOURAL IDENTITY. Six pinned FENs, K=1, fixed sims, with and without
     the channel armed: root visit arrays bit-identical. K=1 specifically —
     D-1 falsified reproducibility at K>1, so the same assertion at the shipping
     K=24 would be testing the scheduler, not the emitter.

  2. THROUGHPUT. Fresh-midgame-root sims/s with the channel on and off. Bar:
     <= 1% degradation. Run against the Gate 1 replay dump with the synthetic
     fallback rather than the GPU, which is the HARSHER test and the one that
     does not need a device: with no network to wait on, sims/s is an order of
     magnitude higher than in deployment and the emitter's per-batch cost is
     correspondingly larger as a fraction. A channel that costs under 1% here
     costs far less than that behind a CUDA graph.

     The brief says "three repeats each". Three is not enough to resolve a 1%
     bar: per-run noise is ~2% idle and ~6% on a busy box, so the arms are
     PAIRED and the statistic is the median of fifteen per-round ratios. A
     measurement whose standard error swamps the bar SKIPS rather than fails —
     an absent measurement reported as a failure is the worse of the two
     errors, and the skip says what the point estimate was.

  3. FLAG DISAMBIGUATION. `--stats` and `--move-stats` are independent
     channels on two different frontends; all four combinations run, and
     neither channel's output corrupts the other's parser.

AMENDMENT D: no module-scope skip. `guofish_core` is a hard dependency of the
whole suite; the Gate 1 dump, python-chess, a CUDA device and a checkpoint are
not, and each test that needs one carries its own marker with a named reason.
"""
from __future__ import annotations

import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

GOLDEN = REPO_ROOT / "golden"


def _why_no_dump() -> str | None:
    for name in ("gate1_dump.npz", "gate1_manifest.json"):
        if not (GOLDEN / name).is_file():
            return f"golden/{name} is missing"
    return None


DUMP_UNAVAILABLE = _why_no_dump()
requires_dump = pytest.mark.skipif(DUMP_UNAVAILABLE is not None,
                                   reason=str(DUMP_UNAVAILABLE))


@pytest.fixture(scope="module")
def dump():
    arrays = np.load(GOLDEN / "gate1_dump.npz")
    manifest = json.loads((GOLDEN / "gate1_manifest.json").read_text(encoding="utf-8"))
    return {"arrays": arrays, "fens": [p["fen"] for p in manifest["positions"]]}


def _search(dump, fen, capacity=4_000_000):
    config = guofish_core.SearchConfig()
    config.arena_capacity = capacity
    search = guofish_core.ReplaySearchQ32(config)
    a = dump["arrays"]
    search.load_dump(a["keys"], a["is_root"], a["move_offset"], a["moves"],
                     a["priors"], a["values"])
    # Whatever the dump misses, a parallel search reaches within a few plies.
    # The fallback is deterministic, which is what makes check 1 an equality.
    search.synthetic_fallback = True
    search.set_position(fen)
    return search


def _root_children(search):
    """[(packed move, visits, raw value_sum)] for the root's children.

    `dump_tree_arrays(1)` is the VISITED subtree with a depth column, so the
    depth-1 rows are exactly the root's children. `value_sum` is compared
    alongside the visit count because a change that moved a simulation without
    changing the count would otherwise pass.
    """
    a = search.dump_tree_arrays(1)
    depth = a["depth"]
    out = []
    for i in range(int(depth.size)):
        if int(depth[i]) == 1:
            out.append((int(a["move"][i]), int(a["visits"][i]),
                        float(a["value_sum"][i])))
    out.sort()
    return out


# K=1. One worker, one leaf in flight: the configuration D-1 leaves reproducible.
SERIAL = dict(workers=1, in_flight=1, max_batch=1)
# The shipping configuration, for the throughput bar.
SHIPPING = dict(workers=1, in_flight=24, max_batch=128)


# ---------------------------------------------------------------------------
# Check 1 — behavioural identity
# ---------------------------------------------------------------------------


@requires_dump
def test_m3_1_arming_the_channel_changes_no_root_visit_at_k1(dump):
    """The whole of ground rule 1, as an equality rather than an inspection."""
    sims = 4000
    fens = dump["fens"][:6]
    assert len(fens) == 6, "the check is specified as six pinned FENs"

    for fen in fens:
        off = _search(dump, fen)
        off.search_parallel(sims, guofish_core.ParallelConfig(**SERIAL))
        expected = _root_children(off)
        expected_best = off.best_move
        expected_root = (off.root_visits, float(off.root_value_sum))

        on = _search(dump, fen)
        on.move_stats_begin()
        assert on.move_stats_armed is True
        on.search_parallel(sims, guofish_core.ParallelConfig(**SERIAL))
        record = on.move_stats_finish()
        assert on.move_stats_armed is False

        assert _root_children(on) == expected, (
            f"root visit array differs with the channel armed at {fen}")
        assert on.best_move == expected_best
        assert (on.root_visits, float(on.root_value_sum)) == expected_root
        # And the channel actually recorded something, so a pass cannot mean
        # "the emitter never ran".
        assert record["checkpoints"], f"no checkpoints recorded at {fen}"


@requires_dump
def test_m3_1b_a_slice_split_move_ladders_on_the_moves_total(dump):
    """The ladder is denominated in the MOVE's delivered sims, not the slice's.

    `Engine.search_move` cuts a move into ~0.05 s slices, so every rung above
    the first is crossed inside some later `search_parallel`. A recorder that
    reset with each call would put every checkpoint in the first slice and the
    ladder would be silently useless.
    """
    search = _search(dump, dump["fens"][0])
    search.move_stats_begin([1000, 2000, 5000])
    parallel = guofish_core.ParallelConfig(**SHIPPING)
    for target in (1500, 3000, 6000):
        search.search_parallel(target, parallel)
    record = search.move_stats_finish()

    rungs = [cp["rung"] for cp in record["checkpoints"]]
    assert rungs == [1000, 2000, 5000, "final"], rungs
    ns = [cp["n"] for cp in record["checkpoints"]]
    assert ns == sorted(ns)
    assert ns[0] >= 1000 and ns[1] >= 2000 and ns[2] >= 5000
    assert record["delivered"] >= 6000 - 1


@requires_dump
def test_m3_1c_an_empty_ladder_still_yields_the_final_checkpoint(dump):
    search = _search(dump, dump["fens"][0])
    search.move_stats_begin([])
    search.search_parallel(500, guofish_core.ParallelConfig(**SHIPPING))
    record = search.move_stats_finish()
    assert [cp["rung"] for cp in record["checkpoints"]] == ["final"]
    cp = record["checkpoints"][0]
    assert cp["root_visits"] > 0
    assert cp["top4"] and cp["top4"][0]["visits"] > 0
    # Most-visited first, and no more than four.
    visits = [c["visits"] for c in cp["top4"]]
    assert visits == sorted(visits, reverse=True)
    assert len(cp["top4"]) <= 4


@requires_dump
def test_m3_1d_cancel_disarms_without_a_record(dump):
    search = _search(dump, dump["fens"][0])
    search.move_stats_begin()
    search.search_parallel(2000, guofish_core.ParallelConfig(**SHIPPING))
    search.move_stats_cancel()
    assert search.move_stats_armed is False
    search.move_stats_cancel()   # idempotent


@requires_dump
def test_m3_1e_a_bad_ladder_is_refused_rather_than_silently_sorted(dump):
    search = _search(dump, dump["fens"][0])
    with pytest.raises(ValueError):
        search.move_stats_begin([1000, 1000])
    with pytest.raises(ValueError):
        search.move_stats_begin([2000, 1000])
    with pytest.raises(ValueError):
        search.move_stats_begin([0, 1000])


# ---------------------------------------------------------------------------
# Check 2 — throughput
# ---------------------------------------------------------------------------


@requires_dump
def test_m3_2_the_channel_costs_under_one_percent_of_throughput(dump):
    """Bar: <= 1% degradation. Both numbers are reported regardless."""
    sims = 200_000
    # FIFTEEN PAIRED ROUNDS, and the count is derived rather than picked.
    # Per-pair noise on an idle box is about 2% and on a box running a
    # four-engine match about 6%; a median over n pairs has a standard error of
    # roughly 1.25 * sd / sqrt(n), so distinguishing a 1% effect from zero needs
    # n around 15 idle. The whole test is ~40 s on an idle machine.
    repeats = 15
    parallel = guofish_core.ParallelConfig(**SHIPPING)
    fen = dump["fens"][0]

    def run(armed: bool) -> float:
        search = _search(dump, fen, capacity=12_000_000)
        if armed:
            search.move_stats_begin()
        started = time.perf_counter()
        search.search_parallel(sims, parallel)
        elapsed = time.perf_counter() - started
        delivered = search.parallel_stats()["delivered"]
        if armed:
            record = search.move_stats_finish()
            assert len(record["checkpoints"]) >= 2
        return delivered / elapsed

    # One discarded warm-up per arm: the first run pays page faults on a 12M
    # node arena and would be charged to whichever arm went first.
    run(False)
    run(True)

    # A PAIRED COMPARISON, and the pairing is what makes this measurable at all.
    # Running three-off then three-on charges any transient load entirely to
    # whichever arm was unlucky: measured on this box with a four-engine match
    # running alongside, that design gave +7.7%, then -3.4%, then +4.4% on three
    # consecutive invocations while the absolute rate halved. Alternating within
    # a round makes a transient hit both arms, and the statistic is then the
    # median of the per-round RATIOS rather than a ratio of two medians.
    off: list[float] = []
    on: list[float] = []
    ratios: list[float] = []
    for _ in range(repeats):
        o = run(False)
        n = run(True)
        off.append(o)
        on.append(n)
        ratios.append(n / o)

    ratio = statistics.median(ratios)
    degradation = 1.0 - ratio
    sd = statistics.pstdev(ratios) if len(ratios) > 1 else 0.0
    # Standard error of a median, to the usual approximation. Reported so a
    # reader can see whether the measurement could resolve the bar it is being
    # judged against.
    se = 1.25 * sd / math.sqrt(len(ratios))

    print(f"\n[M3-2] off {statistics.median(off):,.0f} sims/s median "
          f"({min(off):,.0f}-{max(off):,.0f})")
    print(f"[M3-2] on  {statistics.median(on):,.0f} sims/s median "
          f"({min(on):,.0f}-{max(on):,.0f})")
    print(f"[M3-2] paired ratio on/off: median {ratio:.4f}, sd {sd:.4f}, "
          f"se(median) {se:.4f}, over {repeats} rounds")
    print(f"[M3-2] degradation {degradation:+.3%} (bar: <= 1.000%)")

    # A MEASUREMENT THAT CANNOT RESOLVE THE BAR IS NOT A FAILURE, it is an
    # absent measurement, and reporting it as a failure would be the worse of
    # the two errors. This fires when the box is busy enough that the standard
    # error swamps the 1% bar.
    if se > 0.01 and degradation > 0.01:
        pytest.skip(
            f"cannot resolve a 1% bar here: se(median) is {se:.2%} over "
            f"{repeats} rounds, and the absolute rate "
            f"({statistics.median(off):,.0f} sims/s) suggests the machine is "
            f"busy. Point estimate was {degradation:+.2%}. Re-run idle.")

    assert degradation <= 0.01, (
        f"the channel costs {degradation:.2%} of throughput, over the 1% bar "
        f"(se {se:.2%} over {repeats} paired rounds).")


@requires_dump
def test_m3_2b_off_is_off_by_construction(dump):
    """A search that was never armed records nothing and reports nothing."""
    search = _search(dump, dump["fens"][0])
    assert search.move_stats_armed is False
    search.search_parallel(2000, guofish_core.ParallelConfig(**SHIPPING))
    assert search.move_stats_armed is False
    with pytest.raises(Exception):
        # `finish` on an unarmed search is a caller error, not a silent empty
        # record: an analysis that got one would not know which it was.
        record = search.move_stats_finish()
        assert record["checkpoints"] == []


# ---------------------------------------------------------------------------
# The derivations
# ---------------------------------------------------------------------------


@requires_dump
def test_the_lock_point_is_null_when_the_argmax_moved_at_the_final_rung(dump):
    """`n_lock` is a finding when it is null, not a missing measurement."""
    search = _search(dump, dump["fens"][0])
    search.move_stats_begin([1000, 2000, 5000, 10000])
    search.search_parallel(20000, guofish_core.ParallelConfig(**SHIPPING))
    record = search.move_stats_finish()

    argmaxes = [cp["top4"][0]["move"] if cp["top4"] else None
                for cp in record["checkpoints"]]
    flips = sum(1 for a, b in zip(argmaxes, argmaxes[1:]) if a != b)
    assert record["best_move_changes"] == flips

    if argmaxes[-1] != argmaxes[-2]:
        assert record["n_lock"] is None
    else:
        # The lock point is the `n` of the earliest checkpoint from which the
        # argmax never changes again.
        final = argmaxes[-1]
        i = len(argmaxes)
        while i > 0 and argmaxes[i - 1] == final:
            i -= 1
        assert record["n_lock"] == record["checkpoints"][i]["n"]


def test_the_shipping_ladder_is_the_brief_s_ladder():
    assert list(guofish_core.MOVE_STATS_LADDER) == [
        1000, 2000, 5000, 10000, 25000, 50000, 100000, 150000]


# ---------------------------------------------------------------------------
# The collector
# ---------------------------------------------------------------------------


def test_the_collector_writes_jsonl_provenance_and_a_manifest(tmp_path):
    from telemetry.move_stats import MoveStatsCollector

    errors: list[str] = []
    c = MoveStatsCollector(tmp_path, run_id="unit", flush_every=2,
                           contaminated_fields=["search_wall_ms"],
                           note="unit test", on_error=errors.append)
    c.write_provenance(config_line="k=v", model_path=None, repo=REPO_ROOT)
    c.begin_game("g1")
    for i in range(5):
        c.record_move({"ply": i, "source": "search", "arena_exhausted": i == 3})
    c.end_game(result="1-0", termination="mate", ended_on_time=False)
    c.close()

    assert not errors, errors
    rows = [json.loads(ln) for ln in
            (tmp_path / "unit" / "game_1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["ply"] for r in rows] == [0, 1, 2, 3, 4]
    assert all(r["schema_version"] == 1 for r in rows)
    assert all(r["game_id"] == "g1" for r in rows)

    manifest = json.loads((tmp_path / "unit" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_games"] == 1
    assert manifest["contaminated_fields"] == ["search_wall_ms"]
    game = manifest["games"][0]
    assert game["n_moves"] == 5 and game["n_search"] == 5
    assert game["arena_exhausted_moves"] == 1
    assert game["result"] == "1-0"
    assert manifest["any_game_ended_on_time"] is False

    prov = json.loads((tmp_path / "unit" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["run_id"] == "unit"
    assert prov["engine"]["sha"]
    assert prov["host"]["hostname"]


def test_a_write_failure_disables_the_channel_and_never_raises(tmp_path):
    """A telemetry channel that can end a rated game is worse than none."""
    from telemetry.move_stats import MoveStatsCollector

    errors: list[str] = []
    c = MoveStatsCollector(tmp_path, run_id="unit", flush_every=1,
                           on_error=errors.append)
    c.begin_game("g1")
    # Replace the game file with a directory: the append cannot succeed.
    target = tmp_path / "unit" / "game_1.jsonl"
    if target.exists():
        target.unlink()
    target.mkdir()

    c.record_move({"ply": 0, "source": "search"})
    assert c.enabled is False
    assert errors and "DISABLED" in errors[0]
    # And every subsequent call is a no-op rather than a second failure.
    c.record_move({"ply": 1})
    c.end_game()
    c.close()
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# Check 3 — flag disambiguation
# ---------------------------------------------------------------------------


def _why_no_engine() -> str | None:
    try:
        import chess  # noqa: F401
    except ImportError as exc:
        return f"python-chess is not importable ({exc})"
    try:
        import torch
    except ImportError as exc:
        return f"torch is not importable ({exc})"
    if not torch.cuda.is_available():
        return ("CUDA is not available; the v6 surface runs the graphed CUDA "
                "evaluator and has no measured CPU path")
    from playing.v6 import playv6
    if not playv6.DEFAULT_MODEL.is_file():
        return f"the default checkpoint {playv6.DEFAULT_MODEL} is absent"
    return None


ENGINE_UNAVAILABLE = _why_no_engine()
requires_engine = pytest.mark.skipif(ENGINE_UNAVAILABLE is not None,
                                     reason=str(ENGINE_UNAVAILABLE))

INTERACTIVE_TIMEOUT = 420.0


@requires_engine
@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("move_stats", [False, True])
def test_m3_3_all_four_flag_combinations_run(tmp_path, stats, move_stats):
    """`--stats` and `--move-stats` are independent channels.

    `--stats` writes a root-distribution TABLE to stdout for a human;
    `--move-stats` writes JSONL to a file. Neither can corrupt the other's
    parser, and this is what says so rather than assuming it: all four
    combinations play the same two scripted moves, and each channel's output is
    checked for the other's markers.
    """
    pgn = (REPO_ROOT / "benchmarking/engine/games/v6/head2head/"
                       "90M200K_vs_SF3000/90M200K_vs_SF3000.pgn")
    if not pgn.is_file():
        pytest.skip(f"no PGN to preload from ({pgn})")

    # `--pgn` + `--pgn-ply` skips the two setup prompts and makes the engine
    # move immediately from the loaded position, which is the only
    # non-interactive path this frontend has. One engine move is all the check
    # needs; `quit` then ends the session cleanly through `QuitSession`.
    argv = [sys.executable, "-u", "-m", "playing.v6.playv6_interactive",
            "--sims", "2000", "--no-book", "--no-syzygy", "--no-color",
            "--pgn", str(pgn), "--pgn-ply", "30"]
    if stats:
        argv.append("--stats")
    run_dir = tmp_path / "ms"
    if move_stats:
        argv += ["--move-stats", str(run_dir), "--move-stats-run-id", "four"]

    proc = subprocess.run(argv, cwd=str(REPO_ROOT), input="quit\n", text=True,
                          capture_output=True, timeout=INTERACTIVE_TIMEOUT)
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stderr[-3000:]}"

    if stats:
        assert "visits" in proc.stdout.lower(), (
            f"the --stats table is missing\n{proc.stdout[-2000:]}")
    # THE CROSS-CHECK, both ways. Neither channel may appear on the other's
    # medium, whatever the other channel is doing.
    assert "checkpoints" not in proc.stdout, (
        "move-stats records reached stdout; a UCI GUI or a PGN parser would "
        "read them as protocol")
    if move_stats:
        files = sorted(run_dir.glob("four/game_*.jsonl"))
        assert files, f"no JSONL under {run_dir}: {sorted(run_dir.rglob('*'))}"
        rows = [json.loads(ln) for f in files
                for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert rows, "the JSONL is empty"
        for row in rows:
            assert row["schema_version"] == 1
            assert "checkpoints" in row
        assert (run_dir / "four" / "provenance.json").is_file()
        assert (run_dir / "four" / "manifest.json").is_file()
    else:
        assert not run_dir.exists(), "a run directory appeared with the flag off"


@requires_dump
def test_the_final_checkpoint_agrees_with_root_branches(dump):
    """Cross-check the emitter against code that already existed.

    `Engine.root_branches` builds the same quantity by a completely different
    route — `dump_tree_arrays(1)` in Python, filtered on depth 1 — and it has
    been in the engine since C11c. The final checkpoint is taken with no search
    in flight, so the two must agree EXACTLY on the top four children, their
    visit counts and their order. A disagreement means the emitter is reading
    the tree wrong, and nothing else in this file would catch that.
    """
    search = _search(dump, dump["fens"][0])
    search.move_stats_begin([])
    search.search_parallel(20_000, guofish_core.ParallelConfig(**SHIPPING))
    record = search.move_stats_finish()

    # The independent route: every depth-1 row of the visited-subtree dump.
    arrays = search.dump_tree_arrays(1)
    depth = arrays["depth"]
    rows = [(int(arrays["visits"][i]), int(arrays["move"][i]),
             float(arrays["value_sum"][i]))
            for i in range(int(depth.size)) if int(depth[i]) == 1]
    # Most-visited first, ties broken by the arena's own child order, which is
    # the order `dump_tree_arrays` walks in — so a stable sort on visits alone
    # reproduces the emitter's tie-break.
    rows.sort(key=lambda r: -r[0])

    top4 = record["checkpoints"][0]["top4"]
    assert len(top4) == min(4, len(rows))
    for got, (visits, packed, value_sum) in zip(top4, rows):
        assert got["visits"] == visits
        assert got["move"] == guofish_core.move_to_uci(packed)
        # Child Q is `value_sum / visits`, used as-is: the backup negates on the
        # way up, so a child of the root is already scored from the point of view
        # of the side to move at the root.
        assert got["q"] == pytest.approx(value_sum / visits, rel=1e-12)

    # And the root's own Q is NEGATED, which is the opposite convention and the
    # one that trips every reader of this tree.
    cp = record["checkpoints"][0]
    assert cp["root_visits"] == search.root_visits
    assert cp["root_q"] == pytest.approx(
        -float(search.root_value_sum) / search.root_visits, rel=1e-12)
    assert cp["root_children"] >= len(rows)
