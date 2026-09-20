"""Play chess against the v5 student, on a laptop CPU.

    python playing/v5.1/playv51.py --mcts --simulations 800

WHAT THIS FILE IS FOR
=====================
playing/v5/playv5.py is the reference front-end and it already does the hard
parts of running on a CPU: it keeps the weights in fp32 rather than round-
tripping them through fp16, and it dynamically quantizes every Linear to INT8
(1.7x on this chip, which has AVX512-VNNI). What it does not do is anything
about HOW the forward is dispatched, and on CPU that turns out to be where the
time is.

Measured on this laptop (i7-1165G7, 4C/8T, 12 MB L3, 15 W), with the search
instrumented via GUOFISH_INSTR=1 and the forward wrapped:

    NN forward                                     95.0% of wall
    board_to_tokens                                 1.4%
    select + backup + board copy + terminal         2.7%
    expand children (31.4 avg) + policy softmax      0.5%

That is the exact inverse of the GPU engine's problem. C12 spent its effort
hiding host latency behind a busy device; here the host IS the device, the tree
code is free, and the only lever that moves anything is the forward itself.

THIS FILE OWNS ONE CHANGE: the forward is JIT-traced and frozen, so it crosses
into C++ once per evaluation instead of ~200 times. That is worth 1.4-1.5x end
to end (interleaved, median of 3), and it is bit-identical -- 0 differing policy words over a 250-position
corpus at every batch width. Unlike the Inductor adoption on the GPU side
(C12b, which moved 61-65% of policy words at up to 0.047 of a logit and forced
Gate 2' to be re-based), nothing here needs re-certifying, because nothing here
changed a number.

Everything else -- the board, the book, syzygy, pondering, the PV printer, the
interactive loop -- is playv5's, called directly rather than copied. This file
substitutes two functions into that module and hands over:

    load_model  -> playv5's, plus the JIT wrapper on CPU
    build_mcts  -> core.mctsv5 (the CPU-tuned search) instead of core.mctsv4

WHAT WAS TRIED AND LOST
=======================
Recorded here because the losing arms are the reason the winning one is small
and boring. Full numbers in NOTES.md.

* CPU batching. The obvious idea, and it is wrong on this hardware: batch 4 is
  the peak at 1.24x and batch 32 REGRESSES to 0.89x, because per-batch
  activation requantization grows faster than the weight reuse saves. v4's
  `max_batch_size = 1` comment was right and stays.
* Intra-op threads. batch-1 with 8 torch threads is 2.7x SLOWER than with 1.
  `torch.set_num_threads(1)` stays.
* torch.compile / Inductor. Cannot be used here: it shells out to `cl`, which
  on this machine only exists inside vcvars64.bat (README_BUILD.md). Making an
  interactive play script depend on an MSVC environment to start is a worse
  trade than the speed is worth. Untested rather than rejected on merit.
* optimize_for_inference on top of freeze. Measured 12% SLOWER than freeze
  alone. Not used.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import chess
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import core.mctsv5 as mctsv5
from playing.v5 import playv5

# Bound at import, BEFORE main() substitutes this module's versions into playv5.
# The wrappers below delegate to these, not to the module attributes -- reading
# `playv5.load_model` at call time would find the substitute and recurse.
_playv5_load_model = playv5.load_model
_playv5_build_mcts = playv5.build_mcts

# The checkpoint that actually exists in models/. playv5's own default points at
# models/guofish5_20M/v5_10.9M_best.pt, a layout this tree does not have, so
# running it with no argument fails on a missing file. Substituted below only
# when the caller named no checkpoint AND playv5's default is absent.
DEFAULT_CHECKPOINT = _PROJECT_ROOT / "models" / "v5_10.9M_best_fp16.pt"

# Forward passes to run before the first search. TorchScript's profiling
# executor specializes on the first couple of calls, so without this the first
# move of the game pays the specialization and looks like a stall. Cheap
# insurance: ~10 forwards is well under a second.
JIT_WARMUP_ITERS = 10


class JitForward(nn.Module):
    """The v5 student with its hot-path forward served by TorchScript.

    Holds BOTH modules, and which one runs is decided by the call, not by a
    flag:

        forward(x)                        -> traced      (the search; 100% of
                                             the hot path, batch always 1 in
                                             inline mode)
        forward(x, legal_move_mask=mask)  -> eager       (playv5's raw-policy
                                             path, at most once per move)

    That split is not a nicety, it is forced. `torch.jit.trace` records the
    tensor ops it actually saw, and it saw `legal_move_mask=None`, so the mask
    branch is not in the traced graph and the traced module will not accept the
    keyword at all. Routing on the argument means the mask path keeps running
    the real module instead of silently losing its masking.

    Keeping the eager module also restores what `torch.jit.freeze` takes away:
    freeze inlines parameters as constants, so a frozen module reports ZERO
    parameters. Three separate things break on that, and all three are fixed by
    having a real module underneath:

      * `next(model.parameters()).dtype` (playv5.py:570) raises StopIteration.
      * `playv5.is_v5_model` duck-types on `.config`, which a ScriptModule does
        not carry -- so build_mcts would silently route to core.mctsv3, the
        LEGACY V2 search, and evaluate a v5 net through V2 assumptions.
      * `core.mctsv5.require_v5_config` rejects it outright.

    The cost is that both copies are resident, ~22 MB total for the 10.9M
    student. On a machine with 12 MB of L3 that sounds worse than it is: the
    eager weights are never touched during a search, so they occupy RAM, not
    cache.
    """

    def __init__(self, eager: nn.Module, traced: torch.jit.ScriptModule):
        super().__init__()
        self.eager = eager
        self.traced = traced

    def forward(self, x: torch.Tensor, legal_move_mask: Optional[torch.Tensor] = None):
        if legal_move_mask is None:
            return self.traced(x)
        return self.eager(x, legal_move_mask=legal_move_mask)


def jit_wrap(model: nn.Module, device: torch.device,
             warmup: int = JIT_WARMUP_ITERS) -> nn.Module:
    """Trace and freeze `model`'s forward, returning a JitForward around it.

    Returns `model` untouched, with a printed reason, if anything goes wrong or
    if this is not a CPU run. Failing to accelerate is not a reason to fail to
    play, and on CUDA the eager module is already the measured configuration --
    the GPU engine's own answer to this question is playing/v6, which is a C++
    core, not a traced module.
    """
    if device.type != "cpu":
        return model

    # Match the search's threading before warming up, so the warmup measures
    # (and specializes against) the configuration the workers will actually run
    # in. ParallelMCTS sets this too, but not until construction, which is after
    # this function has already run.
    torch.set_num_threads(1)

    try:
        example = mctsv5.board_to_tokens(chess.Board()).unsqueeze(0)
        with torch.no_grad():
            # check_trace=False because the tracer's own re-run check compares
            # against an eager call, and this module is verified far more
            # strictly than that by bench51.py --verify, which checks a
            # 250-position corpus for bit-equality at four batch widths.
            traced = torch.jit.trace(model, (example,), check_trace=False)
            traced = torch.jit.freeze(traced)
            for _ in range(warmup):
                traced(example)
    except Exception as exc:
        print(f"JIT trace unavailable ({type(exc).__name__}: {exc}); "
              f"running the eager forward.")
        return model

    wrapped = JitForward(model, traced)
    for name in playv5._CARRIED_ATTRS:
        if hasattr(model, name):
            setattr(wrapped, name, getattr(model, name))
    wrapped.eval()
    print(f"Forward: JIT traced + frozen (warmup {warmup}), "
          f"eager retained for the raw-policy path")
    return wrapped


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """playv5.load_model, plus the traced forward on CPU."""
    model = _playv5_load_model(checkpoint_path, device)
    return jit_wrap(model, device)


def build_mcts(model: nn.Module, device: torch.device, **kwargs):
    """Route to core.mctsv5, the CPU-tuned search.

    playv5.build_mcts routes between mctsv4 (v5 students) and mctsv3 (legacy V2)
    by inspecting the model. That routing is kept -- a legacy net still has to go
    to mctsv3, and mctsv5 would refuse it -- but v5 students come here instead of
    to mctsv4. See core/mctsv5.py's docstring for what differs; it is two things
    and neither touches the search.
    """
    if not playv5.is_v5_model(model):
        return _playv5_build_mcts(model, device, **kwargs)
    return mctsv5.ParallelMCTS(model, device, **kwargs)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-jit", action="store_true",
                        help="Run the eager forward (the v5 behaviour). Use this "
                             "to A/B, or if a torch version traces badly.")
    args, passthrough = parser.parse_known_args()

    # `-h` is deliberately left in passthrough so playv5's parser prints the real
    # option list; announce the one flag that parser has never heard of.
    if any(a in ("-h", "--help") for a in passthrough):
        print("v5.1 adds one option on top of everything below:\n"
              "  --no-jit    Run the eager forward instead of the JIT-traced one\n"
              "              (the playv5 behaviour). For A/B, or if a torch\n"
              "              version traces badly.\n")

    if args.no_jit:
        print("JIT disabled (--no-jit): eager forward.")
    else:
        playv5.load_model = load_model
    playv5.build_mcts = build_mcts

    # Supply a checkpoint only when the caller named none and playv5's default is
    # missing. The discriminator is the `.pt` suffix, not "does not start with
    # '-'": playv5's only positional IS the checkpoint, but option VALUES are
    # bare words too, so `--simulations 60` would otherwise read as a checkpoint
    # named 60 and suppress the substitution.
    gave_checkpoint = any(a.endswith(".pt") for a in passthrough)
    if not gave_checkpoint:
        playv5_default = (_PROJECT_ROOT / "models" / "guofish5_20M" /
                          "v5_10.9M_best.pt")
        if not playv5_default.exists() and DEFAULT_CHECKPOINT.exists():
            print(f"No checkpoint given and playv5's default is absent; "
                  f"using {DEFAULT_CHECKPOINT.name}")
            passthrough.insert(0, str(DEFAULT_CHECKPOINT))

    sys.argv = [sys.argv[0]] + passthrough
    playv5.main()


if __name__ == "__main__":
    main()
