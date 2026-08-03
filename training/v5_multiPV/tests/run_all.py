"""Run every v5 test module without pytest (there is none installed here).

    python training/v5_multiPV/tests/run_all.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_V5 = _HERE.parent
_ROOT = _V5.parents[1]
_DATA = _ROOT / "data" / "multiPV"
for _p in (str(_V5), str(_DATA), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODULES = ["tests.test_losses", "tests.test_accum",
           "tests.test_mirror", "tests.test_model", "tests.test_subset"]


def main() -> int:
    total = passed = 0
    failed = []
    for name in MODULES:
        print(f"\n=== {name} ===")
        mod = importlib.import_module(name)
        tests = [(k, v) for k, v in sorted(vars(mod).items())
                 if k.startswith("test_") and callable(v)]
        for tname, fn in tests:
            total += 1
            try:
                fn()
                passed += 1
                print(f"  PASS  {tname}")
            except Exception as exc:
                import traceback
                failed.append(f"{name}::{tname}")
                print(f"  FAIL  {tname}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
    print(f"\n{'=' * 60}\n{passed}/{total} passed")
    if failed:
        print("failed:\n  " + "\n  ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
