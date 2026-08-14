#!/usr/bin/env python
"""C12b — a digest of the forward over Gate 2''s corpus, from a fresh process.

    python tools/c12b_forward_digest.py --compile
    python tools/c12b_forward_digest.py --no-compile --json-out runs/c12b/eager.json

WHY A SEPARATE PROCESS IS THE POINT
===================================
The determinism criterion the C12b brief sets is *"same code + same GPU + same
driver ⇒ bit-identical priors across RUNS"*, and a run is a process. Two captures
inside one interpreter share the Inductor compile cache, the loaded Triton
binaries, the cuBLAS handles and the caching allocator's arenas — so agreeing
with itself is a much weaker claim than agreeing with a fresh process, and it is
the weaker claim that a naive in-test loop would be making.

So this is a tool rather than a test helper: `tests/test_c12b_gate2prime.py`
invokes it with `subprocess` and compares its digest against one taken in-process,
and `tools/drill_c12b.py` invokes it under a mutated tree. Both get the same
numbers because both run the same code.

**It also has to be runnable under a DIFFERENT checkout**, which is why it imports
nothing from the rest of `tools/` and takes the position list apart itself. The
provenance drill checks the C12b working tree's `compile=False` forward against
tag GUOFISH_NUMERICS_BASELINE's, and the tag's checkout has no
`gen_c12b_baseline.py` in it to import from.

WHAT IS DIGESTED
================
The raw bf16 policy words and the float32 values, per captured shape, exactly as
the production callback writes them into the C++-owned buffers — not the priors.
Priors are a softmax over the ~30 legal columns of a 4096-wide row, so a logit
the gather never reads could move without moving any prior; digesting the words
is what makes "bit-identical" mean bit-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import guofish_core  # noqa: E402

CORPUS = REPO_ROOT / "golden" / "c10_corpus.json"
GATE1_MANIFEST = REPO_ROOT / "golden" / "gate1_manifest.json"

# The 90M-corpus checkpoint the engine ships, spelled out as a literal rather
# than imported from `evaluator.SHIPPING_MODEL`, because this tool has to run
# under a checkout that predates that constant. Same v5 architecture as the 20M
# net, so the captured ladder is identical and only the weights differ.
MODEL = REPO_ROOT / "models" / "guofish5_90M" / "v5_10.9M_best.pt"

# Matches tools/gen_c12b_baseline.py's FORWARD_MAX_BATCH, so the shapes digested
# here are the shapes the baseline recorded: (1, 8, 16, 24) off the default
# ladder. 24 is the shipping outstanding-leaf count and 1 is what the W=1/K=1
# gate searches evaluate every batch at.
DEFAULT_MAX_BATCH = 24


def corpus_fens() -> list[str]:
    """Gate 2''s 520: the Gate 2b corpus, then the Gate 1 position set."""
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))["positions"]
    gate1 = json.loads(GATE1_MANIFEST.read_text(encoding="utf-8"))["positions"]
    return [p["fen"] for p in corpus] + [p["fen"] for p in gate1]


def digest(compiled: bool, max_batch: int = DEFAULT_MAX_BATCH,
           model: Path = MODEL) -> dict:
    """Run the corpus through the production callback and hash what came back."""
    import torch

    from playing.v6 import evaluator as live_evaluator

    fens = corpus_fens()
    tokens = np.stack([guofish_core.tokens(fen) for fen in fens])

    # `compile=` reached this module only in C12b. A checkout that predates it —
    # which is exactly what the provenance drill runs this under — has no such
    # keyword, and there `compile=False` is the only behaviour there is.
    # THE CHECKPOINT IS PASSED EXPLICITLY, NEVER DEFAULTED. The provenance drill
    # runs this tool under a checkout of tag GUOFISH_NUMERICS_BASELINE, whose
    # `evaluator.DEFAULT_MODEL` is the 20M-corpus net; defaulting would compare a
    # 20M forward against a 90M one and report a difference that is entirely the
    # checkpoint. `compile=` is the other thing that arrived in C12b, and there
    # the eager forward is the only behaviour there is.
    try:
        evaluator = live_evaluator.build(max_batch, model_path=model, compile=compiled)
    except TypeError:
        if compiled:
            raise
        evaluator = live_evaluator.build(max_batch, model_path=model)

    try:
        shapes = {}
        for shape in evaluator.graph_sizes:
            rows = (len(fens) // shape) * shape
            policy = np.zeros((rows, guofish_core.POLICY_SIZE), dtype=np.uint16)
            value = np.zeros(rows, dtype=np.float32)
            for start in range(0, rows, shape):
                evaluator._input_np[:shape] = tokens[start:start + shape]
                evaluator._evaluate(shape)
                policy[start:start + shape] = evaluator._policy_np[:shape]
                value[start:start + shape] = evaluator._value_np[:shape]
            shapes[str(shape)] = {
                "rows": int(rows),
                "policy_sha256": hashlib.sha256(policy.tobytes()).hexdigest(),
                "value_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        # Asserted here rather than left to the caller: a digest taken from a run
        # that recompiled mid-sweep is a digest of two different forwards, and it
        # would compare equal or unequal for reasons nobody could attribute.
        evaluator.assert_no_recompilation("while digesting the corpus")
        return {
            "compile": bool(getattr(evaluator, "compiled", False)),
            "max_batch": int(max_batch),
            "model": str(model),
            "model_sha256": hashlib.sha256(Path(model).read_bytes()).hexdigest(),
            "positions": len(fens),
            "capture": evaluator.graph_report.describe() if evaluator.graph_report else None,
            "shapes": shapes,
            "provenance": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_capability": list(torch.cuda.get_device_capability(0)),
                "platform": platform.platform(),
            },
        }
    finally:
        evaluator.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--compile", dest="compile", action="store_true", default=False)
    group.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--max-batch", type=int, default=DEFAULT_MAX_BATCH)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    result = digest(args.compile, args.max_batch, args.model)
    text = json.dumps(result, indent=1)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8", newline="\n")
    # The marker is what lets a caller find the payload in a stream that also
    # carries `load_model`'s banner and Inductor's warnings.
    print("C12B_FORWARD_DIGEST " + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
