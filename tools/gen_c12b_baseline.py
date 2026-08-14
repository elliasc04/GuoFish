#!/usr/bin/env python
"""C12b Stage 1 — freeze the eager engine's outputs, so Gate 2' has something fixed to compare against.

    python tools/gen_c12b_baseline.py --sections forward
    python tools/gen_c12b_baseline.py --sections search --sims 1600
    python tools/gen_c12b_baseline.py --force

WHY THIS WRITES TO `baseline/` AND NOT TO `golden/`
===================================================
Global Rule 2: *golden data is produced by the Python reference only, by scripts
under tools/. You may not produce golden data from C++ output under any
circumstance.* Everything this file writes is C++ and torch output. It is
therefore **not golden data and must not live where golden data lives**, and the
separate directory is the mechanism that keeps that true rather than remembered —
a future reader who finds `baseline/` cannot mistake it for a reference dump, and
a future tool that globs `golden/` cannot pick it up.

What it is instead is the artifact half of the C12b brief's certification chain:

    Python reference ──Gate 1 (bit-exact, replay)───► C++ tree logic
                     ──Gate 2 (<=1e-6, 0 inversions)► C++ eager forward
                     ──Gate 2b (497/500 = 99.4%)────► C++ eager engine   <── THIS FILE
                                                            │
                                                       Gate 2' ▼
                                                       C++ Inductor engine

The eager C++ engine is a *certified derivative* of the reference, and this
records what that certified derivative actually computes so that Gate 2' is a
comparison against a fixed artifact rather than against whatever eager happens to
do on the day. **A baseline you have to rebuild to compare against is a baseline
that will drift** — which is the brief's requirement 2, and the reason the
manifest carries a sha256 of every array file it wrote.

Gate 1 and Gate 2's own golden files are untouched by all of this. Gate 1 runs
the replay evaluator with no network in the loop, so no forward change can reach
it; that is the fact that makes the whole re-baselining safe rather than a
loosening, and it is why this tool never opens `golden/gate1_dump.npz`.

WHAT IS RECORDED, AND WHY EACH PIECE
====================================
520 positions: the 500 of `golden/c10_corpus.json` (Gate 2b's corpus) followed by
the 20 of `golden/gate1_manifest.json` (the Gate 1 position set), which the brief
names as Gate 2''s corpus. They are disjoint — checked, not assumed.

  forward   Raw bf16 policy words and values straight out of the production
            callback, per captured shape, plus the gathered priors in canonical
            order that `guofish_core.gather_softmax` produces from them.

            THE RAW WORDS ARE THE POINT. Priors are a lossy view of the logits —
            a softmax over ~30 of 4096 columns — so a change that moved a logit
            the gather never reads would be invisible in them. Acceptance
            criterion 5 ("`compile=False` still reproduces the frozen baseline
            bit-exactly") is asserted on the words; the priors are what Gate 2'
            reports L-inf and L1 distributions over, because priors are what PUCT
            actually reads.

            Recorded at every captured shape, not one, because shape changes the
            cuBLAS tiling and therefore the logits by up to an ulp (BENCH.md
            C10b-1c) — a baseline at one shape could not certify the others. Each
            shape covers `(520 // shape) * shape` positions, so every block is
            full and no row is padding.

  search    Best move and the whole root child visit vector per position, from
            the C++ engine at W=1/K=1 — the configuration Gate 2b ran and the one
            the brief's ">= 99% move agreement" criterion names. The full vector
            rather than a margin, so Gate 2' can say what the baseline thought of
            the move Inductor chose and not only what it thought of its own.

`--sections forward` is seconds and `--sections search` is about an hour; they
are separable for that reason.

EVERY NUMBER HERE IS PRODUCED WITH `compile=False`
==================================================
Asserted, not intended: `TorchEvaluator.compiled` is checked before anything is
written. A baseline accidentally generated through Inductor would make Gate 2' a
comparison of Inductor against itself, which passes trivially and certifies
nothing — the single worst failure mode available to this chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

CORPUS = REPO_ROOT / "golden" / "c10_corpus.json"
GATE1_MANIFEST = REPO_ROOT / "golden" / "gate1_manifest.json"
GATE2B_MANIFEST = REPO_ROOT / "golden" / "c10_gate2b_manifest.json"

# THE 90M-CORPUS CHECKPOINT — the one the engine ships and the one the tuning
# phase will bench, not the 20M one C10's goldens are anchored to. See
# `playing/v6/evaluator.py` for why the two constants differ and why repointing
# the library default instead would break C10 and C10b for an unrelated reason.
# Same v5 architecture, so the captured ladder and every shape here is unchanged.
MODEL = REPO_ROOT / "models" / "guofish5_90M" / "v5_10.9M_best.pt"

BASELINE_DIR = REPO_ROOT / "baseline"
FORWARD_OUT = BASELINE_DIR / "c12b_eager_forward.npz"
SEARCH_OUT = BASELINE_DIR / "c12b_eager_search.json"
MANIFEST_OUT = BASELINE_DIR / "c12b_eager_manifest.json"

# The tag the brief requires the baseline to be pinned to. Recorded AND checked:
# the manifest carries the sha the tag resolves to, and `--allow-dirty` is what a
# caller must pass to write a baseline from a tree that is not that commit.
BASELINE_TAG = "GUOFISH_NUMERICS_BASELINE"

# Gate 2b's sizing, restated so the two runs are comparable. Large enough that
# nothing is evicted over a 520-position sweep, so no answer depends on the order
# the corpus was visited in.
CACHE_SLOTS = 400_000

# `max_batch` for the forward section. It fixes the captured ladder — with
# `DEFAULT_CAPTURE_SIZES` this yields (1, 8, 16, 24) — and 24 is the shipping
# outstanding-leaf count, so the production shape is in the set. Shape 1 is in it
# too, and shape 1 is what the W=1/K=1 search below evaluates every batch at.
FORWARD_MAX_BATCH = 24


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(("git", "-C", str(REPO_ROOT)) + args,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance, never fatal to the run
        return ""


def load_positions() -> list[dict]:
    """The 500-position corpus followed by the 20 Gate 1 positions. Disjoint.

    The disjointness is checked rather than assumed because a duplicated FEN
    would be counted twice in the agreement rate, which would misreport the gate
    in whichever direction the duplicate happened to fall.
    """
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))["positions"]
    gate1 = json.loads(GATE1_MANIFEST.read_text(encoding="utf-8"))["positions"]
    positions = ([{"fen": p["fen"], "source": "c10_corpus"} for p in corpus] +
                 [{"fen": p["fen"], "source": "gate1"} for p in gate1])
    seen = {}
    for index, entry in enumerate(positions):
        if entry["fen"] in seen:
            raise SystemExit(
                f"position {index} ({entry['fen']}) also appears at index "
                f"{seen[entry['fen']]}. Gate 2''s corpus is the union of two sets "
                f"that are supposed to be disjoint; a duplicate would be counted "
                f"twice in the agreement rate.")
        seen[entry["fen"]] = index
    return positions


def top_two_margin(visits: dict[str, int]) -> float:
    """(v1 - v2) / total over the root's children. 1.0 when there is only one.

    Normalised by the visits actually distributed rather than by the budget, for
    the reason `tools/gen_c10_gate2b_golden.py` gives: a search that spent visits
    on terminal fast paths has fewer to distribute, and a raw difference would
    read as more decisive than it was.
    """
    counts = sorted(visits.values(), reverse=True)
    total = sum(counts)
    if total == 0:
        return 0.0
    if len(counts) == 1:
        return 1.0
    return (counts[0] - counts[1]) / total


def build_eager_evaluator(max_batch: int):
    """A `compile=False` evaluator, with the flag verified on the object.

    `compile=False` is the constructor's default, so this could pass nothing —
    and it passes it explicitly and then asserts the result, because the default
    is exactly the kind of thing a later chunk flips. See the module docstring:
    a baseline generated through Inductor makes Gate 2' self-comparing.
    """
    from playing.v6 import evaluator as live_evaluator

    if MODEL != live_evaluator.SHIPPING_MODEL:
        raise SystemExit(f"MODEL ({MODEL}) is not evaluator.SHIPPING_MODEL "
                         f"({live_evaluator.SHIPPING_MODEL}); the baseline must be "
                         f"the checkpoint the engine ships")
    built = live_evaluator.build(max_batch, model_path=MODEL, compile=False)
    if built.compiled:
        raise SystemExit("the evaluator reports compiled=True; this baseline must "
                         "be the UNFUSED eager forward")
    if built.graph is None:
        raise SystemExit("no graph was captured; the baseline must come from the "
                         "shipped captured path, not the ungraphed fallback")
    if type(built.graph).__name__ != "GraphedForward":
        raise SystemExit(f"the graph is a {type(built.graph).__name__}; the "
                         f"baseline must come from the plain manual capture")
    return built


# --- sections ---------------------------------------------------------------


def section_forward(positions: list[dict], payload: dict) -> dict:
    """Raw policy words, values and gathered priors, per captured shape."""
    evaluator = build_eager_evaluator(FORWARD_MAX_BATCH)
    try:
        fens = [entry["fen"] for entry in positions]
        tokens = np.stack([guofish_core.tokens(fen) for fen in fens])
        arrays: dict[str, np.ndarray] = {}
        covered = {}

        for shape in evaluator.graph_sizes:
            rows = (len(fens) // shape) * shape
            policy = np.zeros((rows, guofish_core.POLICY_SIZE), dtype=np.uint16)
            value = np.zeros(rows, dtype=np.float32)
            for start in range(0, rows, shape):
                # Through `_evaluate`, i.e. through the production callback, so
                # what is frozen is what the dispatcher would have seen — the pad
                # discipline and the narrowing included — rather than a
                # convenient re-spelling of the forward.
                evaluator._input_np[:shape] = tokens[start:start + shape]
                evaluator._evaluate(shape)
                policy[start:start + shape] = evaluator._policy_np[:shape]
                value[start:start + shape] = evaluator._value_np[:shape]
            arrays[f"policy_{shape}"] = policy
            arrays[f"value_{shape}"] = value
            covered[shape] = rows
            print(f"  shape {shape:>3}: {rows} positions, "
                  f"{policy.size:,} policy words", flush=True)

        # The gathered priors, from the SHAPE 1 pass, because shape 1 is what the
        # W=1/K=1 search in `section_search` evaluates every batch at. Gate 2'
        # compares priors that the two engines' searches actually saw.
        shape1 = arrays["policy_1"]
        moves: list[int] = []
        priors: list[np.ndarray] = []
        offsets = [0]
        for index, fen in enumerate(fens):
            uci, row = guofish_core.gather_softmax(fen, shape1[index])
            moves.extend(guofish_core.pack_uci(m) for m in uci)
            priors.append(row)
            offsets.append(offsets[-1] + len(uci))
        arrays["move_offset"] = np.array(offsets, dtype=np.uint64)
        arrays["moves"] = np.array(moves, dtype=np.uint16)
        arrays["priors"] = np.concatenate(priors).astype(np.float32)
        arrays["fens"] = np.array(fens)
        arrays["source"] = np.array([entry["source"] for entry in positions])

        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(FORWARD_OUT, **arrays)
        print(f"  priors: {arrays['priors'].size:,} over {len(fens)} positions")
        print(f"wrote {FORWARD_OUT} ({FORWARD_OUT.stat().st_size / 2**20:.1f} MiB)")
        payload["forward"] = {
            "shapes": {str(s): covered[s] for s in evaluator.graph_sizes},
            "prior_count": int(arrays["priors"].size),
            "capture": evaluator.graph_report.describe(),
        }
        return payload["forward"]
    finally:
        evaluator.close()


def section_search(positions: list[dict], payload: dict, sims: int,
                   out: Path = SEARCH_OUT) -> dict:
    """Best move and the root visit vector per position, at W=1/K=1.

    `out` exists so `--limit` can write a pilot somewhere that is not the
    baseline. A partial baseline is worse than no baseline: it would look like
    the real thing to every consumer and would quietly reduce Gate 2''s corpus.
    """
    recorded = json.loads(GATE2B_MANIFEST.read_text(encoding="utf-8"))["search_config"]
    if recorded["policy_temperature"] != 1.0:
        raise SystemExit(f"the Gate 2b manifest records policy_temperature="
                         f"{recorded['policy_temperature']}; the C++ engine has no "
                         f"knob for it and 1.0 is the only value it reproduces")

    config = guofish_core.SearchConfig()
    config.c_init = recorded["c_init"]
    config.c_base = recorded["c_base"]
    config.fpu_root = recorded["fpu_root"]
    config.fpu_tree = recorded["fpu_tree"]
    config.virtual_loss = recorded["virtual_loss"]
    config.max_tree_depth = recorded["max_tree_depth"]
    config.cache_slots = CACHE_SLOTS

    evaluator = build_eager_evaluator(1)
    search = guofish_core.ReplaySearchDouble(config)
    search.set_evaluator(evaluator.core)
    parallel = guofish_core.ParallelConfig(workers=1, in_flight=1, max_batch=1)

    records = []
    started = time.perf_counter()
    try:
        for index, entry in enumerate(positions):
            search.set_position(entry["fen"])
            stats = search.search_parallel(sims, parallel)
            arrays = search.dump_tree_arrays(0)
            visits = {guofish_core.move_to_uci(int(move)): int(count)
                      for depth, move, count in zip(arrays["depth"], arrays["move"],
                                                    arrays["visits"]) if depth == 1}
            records.append({
                "fen": entry["fen"],
                "source": entry["source"],
                "best_move": stats["best_move"],
                "root_visits": int(stats["root_visits"]),
                "mating_move": stats["mating_move"],
                "visits": visits,
                "margin": top_two_margin(visits),
            })
            if (index + 1) % 25 == 0 or index + 1 == len(positions):
                rate = (index + 1) / (time.perf_counter() - started)
                print(f"  {index + 1}/{len(positions)}  {rate:.2f} pos/s  "
                      f"eta {(len(positions) - index - 1) / rate:.0f}s", flush=True)
    finally:
        search.set_evaluator(None)
        evaluator.close()

    elapsed = time.perf_counter() - started
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"sims": sims, "workers": 1, "in_flight": 1, "max_batch": 1,
                    "search_config": recorded, "cache_slots": CACHE_SLOTS,
                    "records": records}, indent=1) + "\n",
        encoding="utf-8", newline="\n")
    margins = sorted(r["margin"] for r in records)
    print(f"wrote {out}")
    print(f"  {len(records)} positions at {sims} sims in {elapsed:.0f}s")
    print(f"  top-two margin: min {margins[0]:.4f}  "
          f"p10 {margins[len(margins) // 10]:.4f}  median {margins[len(margins) // 2]:.4f}")
    print(f"  under 2%: {sum(1 for m in margins if m < 0.02)}/{len(margins)}")
    payload["search"] = {"sims": sims, "positions": len(records), "seconds": elapsed,
                         "near_ties": sum(1 for m in margins if m < 0.02)}
    return payload["search"]


# --- provenance -------------------------------------------------------------


def write_manifest(payload: dict, positions: list[dict], allow_dirty: bool) -> None:
    """The Amendment A header, over whichever artifacts exist on disk.

    Amendment A pins golden data to an interpreter and a python-chess version.
    Nothing here runs python-chess — the corpus arrives as FEN strings and the
    tokenizer is `guofish_core` — so the fields that matter for THIS artifact are
    the ones that can move its bits: the interpreter, torch, the driver and GPU,
    the checkpoint, and the commit. All of them are recorded, and the ones that
    can be checked cheaply are checked by `tests/test_c12b_gate2prime.py`.
    """
    import torch

    # A two-step generation (`--sections forward`, then `--sections search`) must
    # not leave the manifest describing only the second step. The artifacts on
    # disk are the authority — every one of them is re-digested below whether or
    # not this invocation produced it — so the only thing worth carrying forward
    # is the per-section summary, and it is carried rather than dropped.
    if MANIFEST_OUT.exists():
        try:
            previous = json.loads(MANIFEST_OUT.read_text(encoding="utf-8"))
            payload = {**previous.get("sections", {}), **payload}
        except (OSError, ValueError):
            pass

    head = git("rev-parse", "HEAD")
    tag_sha = git("rev-list", "-n", "1", BASELINE_TAG)
    dirty = git("status", "--porcelain")
    if not tag_sha:
        raise SystemExit(
            f"tag {BASELINE_TAG} does not exist. The C12b brief requires the "
            f"baseline to be pinned to a COMMIT, not to whatever eager does "
            f"today; create the tag before generating the baseline.")
    if head != tag_sha and not allow_dirty:
        raise SystemExit(
            f"HEAD ({head[:12]}) is not {BASELINE_TAG} ({tag_sha[:12]}). The "
            f"baseline must be generated from the tagged eager engine; pass "
            f"--allow-dirty to record it anyway, and expect the manifest to say "
            f"so.")

    manifest = {
        "generator": "tools/gen_c12b_baseline.py",
        "what": ("the eager (unfused ATen) C++ engine's outputs, frozen so Gate 2' "
                 "has a fixed artifact to compare the Inductor engine against"),
        "not_golden_data": ("C++/torch output, NOT produced by the Python reference. "
                            "Global Rule 2 keeps golden/ for the reference alone; "
                            "this lives in baseline/ for that reason."),
        "baseline_tag": BASELINE_TAG,
        "baseline_tag_sha": tag_sha,
        "head_sha": head,
        "head_is_tag": head == tag_sha,
        "worktree_dirty": bool(dirty),
        "dirty_paths": dirty.splitlines() if dirty else [],
        "compile": False,
        "provenance": {
            "python": platform.python_version(),
            "python_build": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_device": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None),
            "cuda_capability": (list(torch.cuda.get_device_capability(0))
                                if torch.cuda.is_available() else None),
            "build_info": guofish_core.build_info(),
        },
        "model": {"path": str(MODEL.relative_to(REPO_ROOT)), "sha256": sha256_of(MODEL)},
        "corpus": {
            "positions": len(positions),
            "c10_corpus": {"path": str(CORPUS.relative_to(REPO_ROOT)),
                           "sha256": sha256_of(CORPUS),
                           "positions": sum(1 for p in positions
                                            if p["source"] == "c10_corpus")},
            "gate1": {"path": str(GATE1_MANIFEST.relative_to(REPO_ROOT)),
                      "sha256": sha256_of(GATE1_MANIFEST),
                      "positions": sum(1 for p in positions if p["source"] == "gate1")},
        },
        "sections": payload,
        "artifacts": {},
    }
    for name, path in (("forward", FORWARD_OUT), ("search", SEARCH_OUT)):
        if path.exists():
            manifest["artifacts"][name] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_of(path),
                "bytes": path.stat().st_size,
            }
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=1) + "\n",
                            encoding="utf-8", newline="\n")
    print(f"wrote {MANIFEST_OUT}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", nargs="+", default=["forward", "search"],
                        choices=["forward", "search"])
    parser.add_argument("--sims", type=int, default=1600,
                        help="Gate 2b's budget, so Gate 2' is comparable to it")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing baseline")
    parser.add_argument("--allow-dirty", action="store_true",
                        help=f"record a baseline from a tree that is not {BASELINE_TAG}")
    parser.add_argument("--limit", type=int, default=0,
                        help="pilot runs only: first N positions, written beside the "
                             "baseline as *.pilot.json and with no manifest")
    args = parser.parse_args()

    if args.limit and args.sections != ["search"]:
        print("--limit is only meaningful for --sections search", file=sys.stderr)
        return 2

    outputs = {"forward": FORWARD_OUT, "search": SEARCH_OUT}
    if not args.limit:
        for name in args.sections:
            if outputs[name].exists() and not args.force:
                print(f"refusing to overwrite {outputs[name]} without --force",
                      file=sys.stderr)
                return 2

    positions = load_positions()
    print(f"{len(positions)} positions "
          f"({sum(1 for p in positions if p['source'] == 'c10_corpus')} corpus + "
          f"{sum(1 for p in positions if p['source'] == 'gate1')} gate1)")
    if args.limit:
        positions = positions[:args.limit]
        print(f"PILOT: first {len(positions)}; no baseline and no manifest is written")

    payload: dict = {}
    for name in args.sections:
        print(f"\n== {name} ==")
        if name == "forward":
            section_forward(positions, payload)
        else:
            out = (SEARCH_OUT.with_suffix(".pilot.json") if args.limit else SEARCH_OUT)
            section_search(positions, payload, args.sims, out)

    if args.limit:
        return 0
    write_manifest(payload, positions, args.allow_dirty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
