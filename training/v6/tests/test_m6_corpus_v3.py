"""Corpus v3 brief §2.4 on synthetic data: the t20 tier, nesting in corpus v2 as well as
the 90M build, the exact-counts file, the generalized gates, strata definition v3, and
the R2 tooling (sha256 list, resumable push). Reuses test_m6_corpus_v2's generated dump,
index, "90M" build and corpus v2 build (its `synth` fixture), so v3 nests in a real v2.
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from training.v6.data.formats import REPO, V2_DTYPE
from training.v6.data.reader import ShardSet
from training.v6.data.strata import DEFINITION, compute_strata, field_of
from training.v6.tests.test_m6_corpus_v2 import (  # noqa: F401  (synth is a fixture)
    BUCKETS, PLAN, SEED, VAL_PERMILLE, synth)

sys.path.insert(0, str(REPO / "data/multiPV"))
from pass_a_index import INDEX_DTYPE  # noqa: E402

PB = str(REPO / "data/multiPV/pass_b_v2.py")
# every v2 (PLAN) cell raised or kept; t20 policy only, as rate_plan_v3.json
PLAN3 = {"old": {"policy": {"<=5": 0.8, "6-14": 1.0, "15-27": 1.0, ">=28": 1.0},
                 "value_only": {"<=5": 0.5, "6-14": 1.0, "15-27": 1.0, ">=28": 1.0}},
         "new": {"policy": {"<=5": 0.3, "6-14": 1.0, "15-27": 1.0, ">=28": 1.0},
                 "value_only": {"<=5": 0.0, "6-14": 1.0, "15-27": 1.0, ">=28": 1.0}},
         "t20": {"policy": {"<=5": 0.0, "6-14": 1.0, "15-27": 1.0, ">=28": 1.0},
                 "value_only": {"<=5": 0.0, "6-14": 0.0, "15-27": 0.0, ">=28": 0.0}}}
TIERS = {"old": (26, 10**6), "new": (24, 25), "t20": (20, 23)}


def expected3(index_path, plan):
    """The three-tier selection written out directly, not with the builder's code."""
    ix = np.fromfile(index_path, dtype=INDEX_DTYPE)
    u = np.random.default_rng(SEED).random(len(ix))
    keep = np.zeros(len(ix), dtype=bool)
    counts = {t: {lab: {} for lab in ("policy", "value_only")} for t in plan}
    pol = ix["policy_depth"] >= 20
    for t in plan:
        lo_d, hi_d = TIERS[t]
        in_t = (ix["max_depth"] >= lo_d) & (ix["max_depth"] <= hi_d) & (ix["piece_count"] <= 32)
        for b, lo, hi in BUCKETS:
            for lab, m in (("policy", pol), ("value_only", ~pol)):
                k = in_t & m & (ix["piece_count"] >= lo) & (ix["piece_count"] <= hi) & (u < plan[t][lab][b])
                counts[t][lab][b] = int(k.sum())
                keep |= k
    return np.flatnonzero(keep), counts


def _args(synth, plan, *extra, vmd=20):
    t = synth["tmp"]
    p = t / f"plan3_{abs(hash(json.dumps(plan, sort_keys=True)))}.json"
    p.write_text(json.dumps(plan))
    return [sys.executable, PB, "--source", str(t / "dump.jsonl.zst"), "--index", str(t / "index.bin"),
            "--floor-manifest", str(t / "m90.json"), "--rate-plan", str(p), "--value-min-depth", str(vmd),
            "--nest-manifest", str(synth["v2"] / "manifest.json"), *extra]


def _run(cmd):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


@pytest.fixture(scope="module")
def v3(synth):
    assert synth["run"].returncode in (0, 3), synth["run"].stderr[-2000:]
    t = synth["tmp"]
    counts = t / "v3_expected_counts.json"
    dry = _run(_args(synth, PLAN3, "--dry-run", "--write-counts", str(counts)))
    out = t / "v3"
    build = _run(_args(synth, PLAN3, "--out-dir", str(out), "--expect-counts", str(counts),
                       "--val-permille", str(VAL_PERMILLE), "--n-train-shards", "3", "--n-val-shards", "2",
                       "--n-valderived-shards", "1", "--workers", "1", "--derived-rate", "0"))
    return {"dry": dry, "counts": counts, "build": build, "out": out}


def test_dry_run_t20_counts_and_both_replays(synth, v3):
    assert v3["dry"].returncode == 0, v3["dry"].stderr[-2000:]
    out = json.loads(v3["dry"].stdout)
    sel, want = expected3(synth["tmp"] / "index.bin", PLAN3)
    assert out["selected_counts"] == want and out["selected_lines"] == len(sel)
    assert sum(want["t20"]["policy"].values()) > 0                         # the fixture has t20 rows
    v2 = json.loads((synth["v2"] / "manifest.json").read_text())
    assert out["nesting"]["floor_rows_dropped"] == 0 == out["nesting"]["nest_rows_dropped"]
    assert out["nesting"]["nest_rows_replayed"] == v2["selected_lines"]
    assert out["nesting"]["floor_rows_replayed"] == len(synth["sel90"])
    doc = json.loads(v3["counts"].read_text())
    assert doc["selected_counts"] == want and doc["totals"]["t20"]["value_only"] == 0
    # the v2 plan, rebuilt by the new code, selects exactly what corpus v2 did
    r = _run([sys.executable, PB, "--dry-run", "--source", str(synth["tmp"] / "dump.jsonl.zst"),
              "--index", str(synth["tmp"] / "index.bin"), "--floor-manifest", str(synth["tmp"] / "m90.json"),
              "--rate-plan", str(synth["tmp"] / "plan.json")])
    assert r.returncode == 0 and json.loads(r.stdout)["selected_counts"] == v2["selected_counts"]


def test_expect_counts_mismatch_exits_3(synth, v3):
    bad = synth["tmp"] / "bad_counts.json"
    doc = json.loads(v3["counts"].read_text())
    doc["selected_counts"]["t20"]["policy"]["15-27"] += 1
    bad.write_text(json.dumps(doc))
    r = _run(_args(synth, PLAN3, "--dry-run", "--expect-counts", str(bad)))
    assert r.returncode == 3 and "COUNT MISMATCH" in r.stderr
    r = _run(_args(synth, PLAN3, "--dry-run", "--expect-counts", str(v3["counts"])))
    assert r.returncode == 0 and "reproduces" in r.stderr
    r = _run(_args(synth, PLAN3, "--dry-run", "--write-counts", str(v3["counts"])))   # exists
    assert r.returncode != 0 and "--write-counts" in r.stderr


def test_refuses_plans_that_break_nesting(synth):
    low = json.loads(json.dumps(PLAN3))                    # below corpus v2's cell, above the 90M floor
    low["new"]["policy"]["6-14"] = PLAN["new"]["policy"]["6-14"] - 1e-9
    r = _run(_args(synth, low, "--dry-run"))
    assert r.returncode != 0 and "nest manifest's break nesting" in r.stderr and "'new', 'policy', '6-14'" in r.stderr
    gone = json.loads(json.dumps(PLAN3))                   # a v2 tier missing from the plan
    del gone["new"]
    r = _run(_args(synth, gone, "--dry-run"))
    assert r.returncode != 0 and "rate plan must be" in r.stderr
    floor = json.loads(json.dumps(PLAN3))                  # still refused below the 90M floor
    floor["old"]["value_only"]["<=5"] = 0.39
    r = _run(_args(synth, floor, "--dry-run"))
    assert r.returncode != 0 and "floor manifest's break nesting" in r.stderr
    r = _run(_args(synth, PLAN3, "--dry-run", vmd=24))    # t20 needs value_min_depth 20
    assert r.returncode != 0 and "lowest tier" in r.stderr


def test_build_gates_and_v2_roots_kept(synth, v3):
    r = v3["build"]
    man = json.loads((v3["out"] / "manifest.json").read_text())
    assert r.returncode == (0 if man["hard_move_coverage"] >= 0.99 else 3), r.stderr[-3000:]
    assert man["tier_min_depth"] == {"old": 26, "new": 24, "t20": 20} and man["records_valderived"] == 0
    g = _run([sys.executable, str(REPO / "data/multiPV/corpus_v2_gates.py"), "s9", "--corpus", str(v3["out"]),
              "--manifest-copy", "none", "--expect-counts", str(v3["counts"]),
              "--floor-replayed", str(len(synth["sel90"]))])
    res = json.loads(g.stdout)
    ok_but_coverage = {k: v for k, v in res["checks"].items() if k != "hard_move_coverage_ge_99pct"}
    assert all(v is not False for v in ok_but_coverage.values()), res   # synthetic coverage is < 99% by design
    assert res["checks"]["second_run_refused_H1"] and res["checks"]["selection_equals_expected_counts"]
    assert res["checks"]["nesting_nest_0_dropped"] is True
    assert sorted(res["not_applicable"]) == ["manifest_copy_identical"]

    # every corpus v2 root is in v3, byte for byte
    def roots(d):
        tr, va = ShardSet(d, "train"), ShardSet(d, "val")
        rt, rv = tr.read_raw(np.arange(len(tr))), va.read_raw(np.arange(len(va)))
        tr.close(), va.close()
        rr = np.concatenate([rt[rt["origin"] == 0], rv])
        return {int(x["src_line"]): x.tobytes() for x in rr}
    r2, r3 = roots(synth["v2"]), roots(v3["out"])
    assert set(r2) <= set(r3) and all(r3[k] == b for k, b in r2.items())
    # t20 roots: index-policy rows, value from a block at depth 20-23. A synthetic row whose
    # PVs are all duplicates converts without a policy (has_policy 0); real v2 data had none
    # in its policy-only new tier (CORPUS_V2_REPORT: new value-only roots 0).
    md = np.fromfile(synth["tmp"] / "index.bin", dtype=INDEX_DTYPE)["max_depth"]
    t20 = [k for k in r3 if 20 <= md[k] < 24]
    assert t20
    rec = np.frombuffer(b"".join(r3[k] for k in t20), dtype=V2_DTYPE)
    assert ((rec["value_depth"] >= 20) & (rec["value_depth"] < 24)).all()
    assert (rec["has_policy"] == 1).mean() >= 0.95

    # frozen val re-extracted from v3 equals the v1 frozen set
    fz = synth["tmp"] / "frozen_from_v3"
    r = _run([sys.executable, str(REPO / "data/multiPV/extract_frozen_v2.py"), "--corpus", str(v3["out"]),
              "--manifest", str(v3["out"] / "manifest.json"), "--out", str(fz),
              "--frozen-v1", str(synth["tmp"] / "frozen_v1"), "--index", str(synth["tmp"] / "index.bin"),
              "--m90", str(synth["tmp"] / "m90.json"), "--expect", str(synth["n_frozen"])])
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]

    # strata v3: t20 roots get depth_tier t20
    so = synth["tmp"] / "strata3"
    r = _run([sys.executable, "-m", "training.v6.tools.build_strata", "--shards", str(v3["out"]),
              "--split", "train", "val", "--out-dir", str(so), "--index", str(synth["tmp"] / "index.bin"),
              "--m90", str(synth["tmp"] / "m90.json")])
    assert r.returncode == 0, r.stderr[-2000:]
    meta = json.loads((so / "v3_train.strata2.json").read_text())
    n_t20 = sum(json.loads((so / f"v3_{s}.strata2.json").read_text())["counts"]["depth_tier"]["t20"]
                for s in ("train", "val"))
    assert meta["definition"]["version"] == 3 and n_t20 == len(t20)


def test_strata_v3_codes():
    assert DEFINITION["version"] == 3 and DEFINITION["tier_min_depth"] == {"old": 26, "new": 24, "t20": 20}
    rec = np.zeros(6, dtype=V2_DTYPE)
    rec["value_depth"] = 22
    md = np.array([30, 26, 25, 24, 23, 20])
    tier = field_of(compute_strata(rec, md, np.zeros(6, bool)), "depth_tier")
    assert tier.tolist() == [0, 0, 1, 1, 3, 3]                 # old, new unchanged; t20 = 3, after v1 = 2
    with pytest.raises(ValueError, match="below every depth tier"):
        compute_strata(rec[:1], np.array([19]), np.zeros(1, bool))


def test_sha256_list_and_resumable_push(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / "tools"))
    from make_sha256_list import make_list
    from r2_push import push
    from training.v6.r2 import DirStore, parse_sha_list, pull
    root = tmp_path / "data"
    (root / "processed/a").mkdir(parents=True)
    (root / "processed/a/x.bin").write_bytes(b"x" * 1000)
    (root / "processed/a/sub").mkdir()
    (root / "processed/a/sub/y.json").write_text("{}")
    (root / "processed/z.npy").write_bytes(b"z" * 7)
    (root / "processed/skip.txt").write_text("not listed")
    text = make_list(root, ["processed/a", "processed/z.npy"])
    ents = parse_sha_list(text)
    assert [e[2] for e in ents] == ["processed/a/sub/y.json", "processed/a/x.bin", "processed/z.npy"]
    assert [e[1] for e in ents] == [2, 1000, 7]
    with pytest.raises(SystemExit, match="not under"):
        make_list(root, ["../outside"])

    store = DirStore(tmp_path / "bucket")
    logs = []
    s = push(store, root, text, "data/", 2, publish=True, marker="upload_ok.json", log=logs.append)
    assert s["sent_files"] == 3 and store.get_bytes("data/sha256.txt") == text.encode()
    assert json.loads(store.get_bytes("data/upload_ok.json"))["remote_sizes_verified"] == 3
    s = push(store, root, text, "data/", 2, publish=True, marker=None, log=logs.append)
    assert s["sent_files"] == 0                                             # resumed: nothing re-sent
    store.put_bytes(b"short", "data/processed/z.npy")                      # wrong size remotely: re-sent
    assert push(store, root, text, "data/", 1, False, None, logs.append)["sent_files"] == 1
    pull(store, "data/", tmp_path / "pulled", 2)                           # the harness's pull accepts it
    assert (tmp_path / "pulled/processed/a/x.bin").read_bytes() == b"x" * 1000
    (root / "processed/z.npy").write_bytes(b"changed!")                    # a stale list is refused
    with pytest.raises(SystemExit, match="stale"):
        push(store, root, text, "data/", 1, False, None, logs.append)
