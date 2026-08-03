"""Subset selection: nested, representative, reproducible.

Runs under pytest, or standalone:
    python training/v5_multiPV/tests/test_subset.py

Why this is not a nicety. Source order in the Pass B shards is NOT
exchangeable: every shard is one sweep from opening-like positions to endgames,
so a record's within-shard position predicts almost everything about it -
measured across offsets, mean piece count runs 23.6 -> 16.2, has_policy 0.75 ->
0.19, and the mate share nearly triples. A prefix of source order is therefore
representative only when it happens to land on a whole number of shards. The
first 10M records (21.6 shards) match the corpus to 2.5%; the first 50k (a
fragment of one shard) carry 0.380 of their mass in >=28-piece positions against
a corpus figure of 0.236.

Taking the prefix of a seeded global permutation instead makes that hold by
construction, and makes subsets NESTED so a 5M/10M/20M scaling study varies
quantity with composition fixed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_V5 = _HERE.parent
_ROOT = _V5.parents[1]
_DATA = _ROOT / "data" / "multiPV"
for _p in (str(_V5), str(_DATA), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_v5 import select_subset                       # noqa: E402

TOTAL = 1_000_000
SEED = 20260802


def test_subsets_are_nested():
    """The scaling-study property: 5M subset-of 10M subset-of 20M, exactly.
    Without it, a 10M->20M delta confounds data quantity with the composition
    of whatever the second tranche happened to contain."""
    a = select_subset(TOTAL, 50_000, SEED)
    b = select_subset(TOTAL, 200_000, SEED)
    c = select_subset(TOTAL, 800_000, SEED)
    assert np.array_equal(a, b[:len(a)])
    assert np.array_equal(b, c[:len(b)])
    assert np.array_equal(a, c[:len(a)])


def test_selection_is_a_valid_index_set():
    idx = select_subset(TOTAL, 200_000, SEED)
    assert len(idx) == 200_000
    assert len(np.unique(idx)) == len(idx), "duplicate records selected"
    assert idx.min() >= 0 and idx.max() < TOTAL


def test_subset_none_is_a_full_permutation():
    idx = select_subset(TOTAL, None, SEED)
    assert len(idx) == TOTAL
    assert np.array_equal(np.sort(idx), np.arange(TOTAL))


def test_subset_larger_than_corpus_clamps():
    idx = select_subset(TOTAL, TOTAL * 10, SEED)
    assert len(idx) == TOTAL


def test_reproducible_and_seed_sensitive():
    a = select_subset(TOTAL, 100_000, SEED)
    assert np.array_equal(a, select_subset(TOTAL, 100_000, SEED)), \
        "same seed must reproduce the same selection"
    assert not np.array_equal(a, select_subset(TOTAL, 100_000, SEED + 1))


def test_selection_is_spread_across_the_whole_corpus():
    """The actual point: a small subset must not concentrate in one region.

    A source-order prefix of 20k records lives entirely inside the first 2% of
    the corpus; the permuted selection is spread uniformly, which is what makes
    its composition match. Checked as uniformity of the selection across ten
    equal bins.
    """
    n = 20_000
    idx = select_subset(TOTAL, n, SEED)
    counts, _ = np.histogram(idx, bins=10, range=(0, TOTAL))
    expected = n / 10
    assert counts.min() > 0.9 * expected, f"selection is clumped: {counts}"
    assert counts.max() < 1.1 * expected, f"selection is clumped: {counts}"

    # The negative control: the prefix this replaces fails the same check hard.
    prefix = np.arange(n)
    pc, _ = np.histogram(prefix, bins=10, range=(0, TOTAL))
    assert pc[0] == n and pc[1:].sum() == 0, \
        "test is vacuous - a source-order prefix should be entirely in bin 0"


def _main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            import traceback
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
