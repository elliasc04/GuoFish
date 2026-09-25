"""M6 on synthetic data: corpus v2 builder (S9 checks) and the frozen-val
re-materialisation (S6), end to end on a generated dump and index.

The "90M" v1 build is produced in-process with pass_b_convert's own
build_selection + convert_one, never through its CLI (whose defaults point at
the live corpus, H1).
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter

import chess
import numpy as np
import pytest
import zstandard

from core.guofish_net.tokenizers import board_from_v5_tokens
from training.v6.data.formats import REPO, V1_DTYPE, dtype_descr
from training.v6.data.reader import ShardSet

sys.path.insert(0, str(REPO / "data/multiPV"))
from extract_frozen_v2 import verify  # noqa: E402
from pass_a_index import INDEX_DTYPE, iter_lines, scan_line  # noqa: E402
from pass_b_convert import _splitmix64, build_selection, convert_one  # noqa: E402
from pass_b_v2 import position_key  # noqa: E402

SEED = 20260802
VAL_PERMILLE = 150
RATES90 = ({"<=5": 0.8, "6-14": 0.7, "15-27": 0.7, ">=28": 0.7},
           {"<=5": 0.5, "6-14": 0.4, "15-27": 0.4, ">=28": 0.5})


def _random_board(rng):
    while True:
        b = chess.Board()
        for _ in range(rng.randint(0, 70)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if not b.is_game_over():
            return b


def _pv(board, rng, first, length):
    b, line = board.copy(), [first]
    b.push(first)
    for _ in range(length - 1):
        moves = list(b.legal_moves)
        if not moves or b.is_game_over():
            break
        m = rng.choice(moves)
        line.append(m)
        b.push(m)
    return " ".join(m.uci() for m in line)


def _entry(board, rng, planted, i):
    """One dump line with 1-3 eval blocks; pvs sorted so rel[0] is the max."""
    moves = list(board.legal_moves)
    stm = 1 if board.turn == chess.WHITE else -1
    evals = []
    for depth in rng.sample([18, 20, 22, 24, 25, 26, 28, 32, 36], rng.randint(1, 3)):
        k = min(len(moves), rng.randint(1, 3))
        firsts = rng.sample(moves, k)
        if rng.random() < 0.08:
            m = rng.randint(1, 6) * stm                  # side to move mates: rel[0] is the max
            scores = [{"mate": m}] + [{"cp": int(-stm * rng.randint(0, 900))} for _ in firsts[1:]]
        else:
            rel = sorted((rng.randint(-400, 400) for _ in firsts), reverse=True)
            scores = [{"cp": r * stm} for r in rel]
        pvs = [{**s, "line": _pv(board, rng, f, rng.randint(1, 6))} for s, f in zip(scores, firsts)]
        evals.append({"depth": depth, "pvs": pvs, "knodes": 1000})
    fen = " ".join(board.fen().split()[:4])
    if i in planted["missing_line"]:
        for e in evals:
            e["pvs"][0]["line"] = ""
    elif i in planted["unparseable"]:
        for e in evals:
            e["pvs"][0]["line"] = "zz99 " + e["pvs"][0]["line"]
    elif i in planted["illegal"]:
        empty = next(s for s in chess.SQUARES if board.piece_at(s) is None)
        for e in evals:
            e["pvs"][0]["line"] = f"{chess.square_name(empty)}{chess.square_name(empty ^ 1)}"
    return {"fen": fen, "evals": evals}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("m6")
    rng = random.Random(6)
    n = 3000
    idx = list(range(n))
    rng.shuffle(idx)
    planted = {"missing_line": set(idx[:6]), "unparseable": set(idx[6:12]), "illegal": set(idx[12:18])}
    rows = []
    for i in range(n):
        rows.append(_entry(_random_board(rng), rng, planted, i))
    # duplicates: a root equal to another line's position after its first PV move
    # (-> derived/root collision), and exact repeats (-> derived/derived collision)
    for i in range(0, 60, 3):
        src = rows[i]
        b = chess.Board(src["fen"] + " 0 1")
        deep = max(src["evals"], key=lambda e: e["depth"])
        first = deep["pvs"][0]["line"].split()
        if first and first[0] != "zz99":
            try:
                b.push_uci(first[0])
            except ValueError:
                continue
            if not b.is_game_over():
                rows.append(_entry(b, rng, {k: set() for k in planted}, -1))
        rows.append(json.loads(json.dumps(src)))
    raw = "\n".join(json.dumps(r, separators=(",", ":")) for r in rows).encode()
    dump = tmp / "dump.jsonl.zst"
    dump.write_bytes(zstandard.ZstdCompressor().compress(raw))
    ix = np.array([scan_line(off, line) for off, line in iter_lines(dump)], dtype=INDEX_DTYPE)
    assert len(ix) == len(rows)
    ix.tofile(tmp / "index.bin")

    # the synthetic "90M" v1 build
    cfg90 = {"value_min_depth": 26, "policy_min_depth": 20,
             "bucket_rates_policy": RATES90[0], "bucket_rates_value_only": RATES90[1]}
    sel90 = build_selection(tmp / "index.bin", cfg90, SEED)
    cfg_v1 = dict(value_min_depth=26, policy_min_depth=20, cp_clamp=10_000, temperature=30.0,
                  epsilon=0.05, value_scale=290.6806, val_permille=VAL_PERMILLE)
    n_val = 2
    val_shards = [[] for _ in range(n_val)]
    stats = Counter()
    sel = set(sel90.tolist())
    for ln, (_o, line) in enumerate(iter_lines(dump)):
        if ln in sel:
            out, is_val = convert_one(line, cfg_v1, stats)
            if out is not None and is_val is True:
                val_shards[_splitmix64(ln ^ SEED) % n_val].append(out)
    fz = tmp / "frozen_v1"
    fz.mkdir()
    for j, recs in enumerate(val_shards):
        (np.concatenate(recs) if recs else np.zeros(0, V1_DTYPE)).tofile(fz / f"val_{j:04d}.bin")
    (fz / "manifest.json").write_text(json.dumps(
        {"record_dtype": dtype_descr(V1_DTYPE), "record_size_bytes": V1_DTYPE.itemsize}))
    m90 = {"seed": SEED, "value_min_depth": 26, "policy_min_depth": 20,
           "bucket_sampling_rates_policy": RATES90[0], "bucket_sampling_rates_value_only": RATES90[1],
           "selected_lines": int(len(sel90)), "n_val_shards": n_val}
    (tmp / "m90.json").write_text(json.dumps(m90))

    v2 = tmp / "v2"
    cmd = [sys.executable, str(REPO / "data/multiPV/pass_b_v2.py"), "--out-dir", str(v2),
           "--source", str(dump), "--index", str(tmp / "index.bin"), "--floor-manifest",
           str(tmp / "m90.json"), "--target", "1500", "--val-permille", str(VAL_PERMILLE),
           "--n-train-shards", "3", "--n-val-shards", "2", "--n-valderived-shards", "2",
           "--workers", "1", "--derived-rate", "1.0", "--max-ply", "2"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return {"tmp": tmp, "v2": v2, "cmd": cmd, "run": r, "planted": planted, "sel90": sel90,
            "n_frozen": sum(len(s) for s in val_shards), "m90": m90}


def test_builder_s9(synth):
    r = synth["run"]
    man = json.loads((synth["v2"] / "manifest.json").read_text())
    cov = man["hard_move_coverage"]
    assert r.returncode == (0 if cov >= 0.99 else 3), r.stderr[-3000:]
    assert man["record_format"] == "v2" and man["value_min_depth"] == 24
    assert man["nesting"]["prior_rows_dropped"] == 0                        # every 90M line again
    for k in ("json_error", "no_evals"):
        assert man["rejection_histogram"].get(f"reject_{k}", 0) == 0
    for b in RATES90[0]:                                                    # floored rates
        assert man["bucket_sampling_rates_policy"][b] >= RATES90[0][b]
        assert man["bucket_sampling_rates_value_only"][b] >= RATES90[1][b]

    selected = set(build_selection(synth["tmp"] / "index.bin", {
        "value_min_depth": 24, "policy_min_depth": 20,
        "bucket_rates_policy": man["bucket_sampling_rates_policy"],
        "bucket_rates_value_only": man["bucket_sampling_rates_value_only"]}, SEED).tolist())
    assert set(synth["sel90"].tolist()) <= selected
    tr, va = ShardSet(synth["v2"], "train"), ShardSet(synth["v2"], "val")
    vd = ShardSet(synth["v2"], "valderived", synth["v2"] / "manifest.json")
    rt, rv, rd = tr.read(np.arange(len(tr))), va.read(np.arange(len(va))), vd.read(np.arange(len(vd)))
    roots = np.concatenate([rt[rt["origin"] == 0], rv])
    derived = np.concatenate([rt[rt["origin"] > 0], rd])
    assert (rv["origin"] == 0).all() and (rd["origin"] > 0).all()           # derived never in val_*
    assert man["roots"] == len(roots) and len(derived) > 100

    # planted hard-move failures are counted by reason, on the selected lines only
    fails = man["hard_move_failures"]
    for reason, lines in synth["planted"].items():
        assert fails.get(reason, 0) == len(lines & {int(x) for x in roots["src_line"]}), reason
    src_depth = {int(r["src_line"]): int(r["value_depth"]) for r in roots}
    src_cp = {int(r["src_line"]): int(r["value_cp"]) for r in roots}
    root_keys = {position_key(board_from_v5_tokens(t)) for t in roots["tokens"]}
    keys = set()
    for d in derived:
        board = board_from_v5_tokens(d["tokens"])
        mv = int(d["hard_move"])
        assert mv >= 0 and d["has_policy"] == 0 and d["n_pv"] == 0
        assert any(m.from_square * 64 + m.to_square == mv for m in board.legal_moves)
        root = int(d["src_line"])
        assert int(d["value_depth"]) == src_depth[root] - int(d["origin"]) >= 20
        if abs(src_cp[root]) < 29_000:
            assert int(d["value_cp"]) == src_cp[root]                        # cp: the root's value
        else:
            assert np.sign(d["value_cp"]) == np.sign(src_cp[root])
            assert abs(int(d["value_cp"])) >= abs(src_cp[root])             # mate got closer
        k = position_key(board)
        assert k not in root_keys and k not in keys                          # dedup post-pass
        keys.add(k)
    c = man["counters"]
    assert c["derived_dropped_root_duplicate"] > 0 and c["derived_dropped_derived_duplicate"] > 0
    assert c["derived_written_train"] + c["derived_written_valderived"] == len(derived)
    for s in (tr, va, vd):
        s.close()
    print(f"\nS9 synthetic: {len(roots):,} roots, {len(derived):,} derived; hard_move coverage "
          f"{cov:.2%}; failures {fails}; dedup dropped {c['derived_dropped_root_duplicate']} "
          f"root-dup + {c['derived_dropped_derived_duplicate']} derived-dup; nesting 0 dropped")


def test_builder_refuses_existing_out_dir(synth):
    r = subprocess.run(synth["cmd"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0 and "refusing (H1)" in r.stderr


def test_frozen_rematerialisation_s6(synth):
    out = synth["tmp"] / "frozen_v2"
    r = subprocess.run([sys.executable, str(REPO / "data/multiPV/extract_frozen_v2.py"),
                        "--corpus", str(synth["v2"]), "--out", str(out), "--frozen-v1",
                        str(synth["tmp"] / "frozen_v1"), "--index", str(synth["tmp"] / "index.bin"),
                        "--m90", str(synth["tmp"] / "m90.json"), "--expect", str(synth["n_frozen"])],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rep = json.loads((out / "manifest.json").read_text())["s6"]
    assert rep["passed"] and rep["count_v2"] == synth["n_frozen"] > 50
    assert not any(rep["field_mismatches"].values())
    print(f"\nS6 synthetic: {rep['count_v2']} frozen records re-materialised; all shared fields identical")
    # negative control: one flipped value bit is caught
    shards = [ShardSet(out, "val").read_raw(np.arange(len(ShardSet(out, "val"))))]
    lens = [x["records"] for x in json.loads((out / "manifest.json").read_text())["shards"]]
    parts = np.split(shards[0], np.cumsum(lens)[:-1])
    parts[0] = parts[0].copy()
    parts[0]["value_cp"][0] += 1
    bad = verify(parts, synth["tmp"] / "frozen_v1")
    assert not bad["passed"] and bad["field_mismatches"]["value_cp"] == 1
