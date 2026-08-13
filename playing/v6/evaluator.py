"""C10 — the Python half of the evaluation boundary. C10b — through a CUDA graph.

WHAT THIS FILE IS ALLOWED TO DO
===============================
Five torch calls on pre-allocated tensors, and nothing else. `_evaluate` runs on
a C++-created dispatcher thread that has just acquired the GIL, so every
microsecond spent here is a microsecond no other Python thread can run and — more
to the point — a microsecond the dispatcher is not launching kernels. Scope §2.1:
"keep the callback minimal; a handful of torch calls on pre-allocated tensors, no
Python loops."

C10b MADE THAT LITERAL. C9a found the callback's ~4 ms was ~110 host ATen
submissions, not GPU work, so the forward is now captured as a CUDA graph
(playing/v6/graphs.py) and the callback is four calls with no allocation:

    H2D copy of `count` pinned int32 rows  ->  one graph replay  ->  two D2H copies

The GIL-held window drops with it, from the ~4 ms of kernel launching C9a
predicted to the copies plus one `cudaGraphLaunch`; BENCH.md C10b-4 re-runs
C10d's histogram to say by how much. The model, the autocast context and the
buffers are unchanged — `graphs=False` restores the eager callback exactly, and
Gate 2 is re-run on the graphed one because capture can change kernel selection.

The measured thing it replaces is the reference's `BatchedEvaluator._evaluate_batch`,
whose per-item `policy_cpu[i].clone()` + `req.event.set()` is 3.84 µs of real work
that inflated to 926 µs under GIL contention at 32 workers. There is no per-item
work here at all: C++ writes a `[n, 68]` block, torch reads it as one tensor, and
the results come back as two block copies.

WHAT CROSSES, AND IN WHICH DIRECTION
====================================
Three C++-owned buffers, exposed as NumPy views and wrapped once, at construction,
as torch tensors over the same memory. Nothing is allocated per batch:

    input_buffer   [max_batch, 68]    int32    C++ writes, this reads
    policy_buffer  [max_batch, 4096]  bf16     this writes, C++ reads
    value_buffer   [max_batch]        float32  this writes, C++ reads

`policy_buffer` MUST STAY bf16. The reference's `policy_logits.cpu()` is a bf16
tensor and `MCTSNode.expand` gathers from it *before* `.float()`, so bf16 is what
the reference's gather sees. Handing C++ float32 would round the logits through a
wider type on a different schedule and Gate 2's claim — that the two gathers see
the same bits — would stop being true. NumPy has no bfloat16, so the view arrives
as uint16 and is reinterpreted here; the bit patterns are identical and the
reinterpretation is free.

WHY THE MODEL IS NOT ASKED FOR A LEGAL-MOVE MASK
================================================
Because masking is not the evaluator's job on either side. C++ gathers the ~26-38
legal logits it needs directly out of the 4096-wide row (scope §2.1) and softmaxes
those; materialising a mask, or a masked row, per item is precisely the marshalling
cost the boundary was designed to delete.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

import guofish_core

# THE FORMAT CONTRACT, TAKEN FROM THE C++ CORE RATHER THAN FROM `core.mctsv4`.
#
# It used to be imported from the reference, on the reasoning that whoever owns
# the tokenizer owns the contract. That reasoning is now inverted: `cpp/tokens.hpp`
# owns the tokenizer the engine actually runs (`kSeqLength`, `kIdxCls`), and
# `cpp/evaluator.hpp` owns the policy gather (`kPolicySize`), and both are
# compile-time constants baked into the extension. `guofish_core` re-exports them,
# so reading them from there means this module cannot disagree with the buffers it
# is wrapping — which is what the disagreement would actually corrupt.
#
# `core.mctsv4` is being retired with the Python search; nothing here imports it
# any more, and nothing here imports a Python MCTS at all.
SEQ_LENGTH = guofish_core.SEQ_LENGTH
POLICY_SIZE = guofish_core.POLICY_SIZE

# The precision v5 was trained under and every forward here runs in. Restated
# rather than imported for the same reason as above; `playing/v6/graphs.py` pins
# the identical value and tests/test_c10b_graphs.py compares the two paths
# bit-for-bit, so a drift between them cannot go unnoticed.
AUTOCAST_DTYPE = torch.bfloat16

# The smallest embedding table that can answer every token id the C++ tokenizer
# emits. `cpp/tokens.hpp` tops out at `kTokenCls = 40`, so 41 rows is the floor
# and both shipped generations carry 43.
#
# Checked because the failure mode is silent: `nn.Embedding` with an index past
# the table does an out-of-bounds DEVICE-SIDE read, which returns garbage priors
# (or trips an async IMA thousands of launches later) rather than raising
# anything a caller could attribute.
MIN_VOCAB_SIZE = 41

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "guofish5_20M" / "v5_10.9M_best.pt"

# scope §2.1, settled by C0b's measurement rather than by projection. See
# guofish_core.LiveEvaluator's docstring for the numbers and for why the Windows
# margin is an artifact that must not be spent.
DEFAULT_SWITCH_INTERVAL = guofish_core.DEFAULT_SWITCH_INTERVAL

# A bf16 signalling-NaN-adjacent pattern: 0x7FC0 is bf16 quiet NaN. Written into
# the policy rows a batch did not use when `poison_unused_rows` is on, so that a
# leak shows up as a NaN prior rather than as a plausible number nobody notices.
POISON_BITS = 0x7FC0


def require_engine_contract(model: torch.nn.Module) -> int:
    """Refuse any model whose I/O does not match the buffers C++ will hand it.

    REPLACES `core.mctsv4.require_v5_config`, and it is deliberately a WEAKER
    check against a DIFFERENT thing.

    The old gate asked "is this the v5 multi-PV student?", answered by duck-typing
    on a `ModelConfig` attached to the module. That question was right for
    `core.mctsv4`, which is specialised to that student — but it is the wrong
    question here, and it was refusing models this engine can actually run. The
    C++ core never sees the model: it writes 68 int32s, reads back 4096 bf16
    logits and one float, and is indifferent to what happened in between. So the
    only thing worth refusing is a model that breaks THAT, and carrying a
    ModelConfig is not it.

    Concretely, the legacy `ChessTransformerV2` (guofish2..guofish4) satisfies
    every equality below and has no `.config` at all, so the old gate rejected a
    net whose tensors are shape-for-shape identical to the one it accepted.

    What is checked, and why each one is silent if it is not:

      seq_length   the module's own advertised token count. Both generations set
                   it (`ChessTransformerV5.__init__` calls the spelling
                   "load-bearing" for exactly this reason). A mismatch means C++
                   fills 68 columns of a row the model reads as some other width.
      vocab_size   see MIN_VOCAB_SIZE — an out-of-range embedding index is a
                   device-side garbage read, not an exception.
      cls_index    CHECKED ONLY ON v5, because only v5 exposes it. V2 hardcodes
        policy_size 67 and 4096 as literals inside `forward`, so there is no
                   attribute to disagree with; its shapes are pinned by the class
                   instead, and `load_state_dict(strict=True)` is what enforces
                   them.

    Returns the model's `seq_length`. Raises ValueError on any mismatch — never
    warns: every equality here is an input-format assumption, and a violated one
    means silently evaluating a position the network never saw in that form.
    """
    name = type(model).__name__

    seq_length = getattr(model, "seq_length", None)
    if seq_length is None:
        raise ValueError(
            f"{name} does not advertise `seq_length`, so there is nothing to "
            f"check the {SEQ_LENGTH}-token board encoding against. Every module "
            f"`playing.v6.chess_transformer_v2.load_model` returns sets it; a "
            f"module that does not was built some other way.")
    if seq_length != SEQ_LENGTH:
        raise ValueError(
            f"{name}.seq_length={seq_length}, but the C++ tokenizer emits "
            f"{SEQ_LENGTH} tokens per position (guofish_core.SEQ_LENGTH, from "
            f"cpp/tokens.hpp kSeqLength). The input buffer is [max_batch, "
            f"{SEQ_LENGTH}] and cannot be reshaped for this model.")

    embedding = getattr(model, "embedding", None)
    vocab_size = getattr(embedding, "num_embeddings", None)
    if vocab_size is not None and vocab_size < MIN_VOCAB_SIZE:
        raise ValueError(
            f"{name}'s embedding table has {vocab_size} rows, but the C++ "
            f"tokenizer emits ids up to {MIN_VOCAB_SIZE - 1} (cpp/tokens.hpp "
            f"kTokenCls). Indexing past the table is an out-of-bounds device "
            f"read that returns garbage priors instead of raising.")

    # v5 only: it publishes these, so checking them is free. `getattr` rather
    # than a v5 test, so a future generation that publishes them is checked too.
    for attr, expected, why in (
        ("cls_index", guofish_core.SEQ_LENGTH - 1,
         "the value head pools from this slot; cpp/tokens.hpp writes kTokenCls there"),
        ("policy_size", POLICY_SIZE,
         "priors are indexed from_square*64 + to_square"),
    ):
        actual = getattr(model, attr, None)
        if actual is not None and actual != expected:
            raise ValueError(
                f"{name}.{attr}={actual}, expected {expected} ({why}).")

    return int(seq_length)


def hoist_norm_parameter_casts(model: torch.nn.Module) -> int:
    """Store every LayerNorm's affine parameters in fp32. Returns how many.

    C12. **This changes no bit of any output, and that is checked rather than
    argued** — `tests/test_c10b_graphs.py` compares the whole Gate 2 corpus's
    raw bf16 policy words at every captured shape, and BENCH.md C12-6 records
    the before/after as bit-identical at shapes 24 and 128.

    WHY IT IS FREE. `torch.amp.autocast` keeps `layer_norm` on its fp32 list, so
    every call widens the input AND both affine parameters to fp32 before the
    kernel runs — Nsight Compute confirms the kernel is
    `vectorized_layer_norm_kernel<float, float>`. The input's cast is real work
    on a [count x 68, 384] activation and cannot be avoided without changing the
    numerics. The parameters' casts are 26 launches of 384 elements each, one
    per parameter per forward, recomputing a constant. bf16 -> fp32 is an exact
    widening, so precomputing it stores exactly the bits the runtime cast would
    have produced, and autocast then finds the tensor already in its target dtype
    and does nothing.

    WHAT IT BUYS. 26 of the forward's ~151 kernel launches, each a grid of ONE
    block — i.e. 1/48th of this device's SMs — measured at 9.5% of device time at
    shape 24 (BENCH.md C12-2). Removing them is 1.024x on the graphed forward's
    device time at shape 24 and 1.009x at 128; the gap between the two is the
    point, because shape 24 is what the engine ships at.

    CALLED BEFORE CAPTURE, from `TorchEvaluator.__init__`. A graph records the
    kernels it was captured with, so hoisting afterwards would leave the casts
    baked into the graph and change nothing at all.
    """
    hoisted = 0
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm) and module.elementwise_affine:
            if module.weight.dtype != torch.float32:
                module.weight.data = module.weight.data.float()
                module.bias.data = module.bias.data.float()
                hoisted += 1
    return hoisted


def load_default_model(model_path: Path | None = None,
                       device: torch.device | None = None) -> tuple[torch.nn.Module,
                                                                    torch.device]:
    """Either supported generation on the best available device.

    Routed through `playing.v6.chess_transformer_v2.load_model`, which picks the
    architecture off the checkpoint's own metadata — v5 students
    (training/v5_multiPV) and the legacy ChessTransformerV2 that guofish2..
    guofish4 are. There is no flag to get wrong.

    That loader used to be `playing.v5.playv5.load_model`, which is the same code
    but reached through a module whose import executes `import core.mctsv3` and
    `import core.mctsv4` so that ITS `build_mcts` can route between two Python
    MCTS implementations. v6 drives the C++ core and uses neither, so the whole
    legacy search tree was arriving as a side effect of asking for a loader.

    `log=` is left at its default `print` here; `Engine.ensure_ready` wraps this
    call in `contextlib.redirect_stdout` and re-emits each line on stderr, which
    keeps the UCI stream clean without this layer needing to know it is under a
    protocol.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from playing.v6.chess_transformer_v2 import load_model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(model_path) if model_path else DEFAULT_MODEL, device)
    require_engine_contract(model)
    return model, device


class TorchEvaluator:
    """Hosts a v5 checkpoint behind `guofish_core.LiveEvaluator`'s callback.

    Use it as a context manager, or call `close()`. Both matter when the buffers
    are page-locked: the unregister has to happen before C++ frees the memory,
    and `close()` is what guarantees the order.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, max_batch: int,
                 *, switch_interval: float = DEFAULT_SWITCH_INTERVAL, pin: bool = True,
                 graphs: bool = True, graph_method: str = "cudagraph",
                 graph_sizes=None, hoist_norm_casts: bool = True):
        if max_batch < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch}")
        require_engine_contract(model)

        # C12, and BEFORE capture — see hoist_norm_parameter_casts. The flag
        # exists for the same reason `graphs=` and `pin=` do: so the A/B that
        # justified it stays reproducible, and so a future regression can be
        # bisected against it rather than reasoned about.
        self.hoisted_norm_casts = (
            hoist_norm_parameter_casts(model) if hoist_norm_casts else 0)

        self.model = model
        self.device = device
        self.max_batch = int(max_batch)
        self._closed = False

        # Constructing this sets sys.setswitchinterval, before any search thread
        # exists. That is the whole reason the setting lives in a constructor —
        # see guofish_core.LiveEvaluator.
        self.core = guofish_core.LiveEvaluator(self.max_batch, self._evaluate, switch_interval)
        self.switch_interval = self.core.switch_interval
        self.switch_interval_before = self.core.switch_interval_before

        # NumPy views onto C++-owned memory. `.base` is the capsule, not an owner:
        # these are aliases, not copies, and writing them writes what C++ reads.
        self._input_np: np.ndarray = self.core.input_view()
        self._policy_np: np.ndarray = self.core.policy_view()
        self._value_np: np.ndarray = self.core.value_view()

        # Wrapped ONCE. `torch.from_numpy` shares storage, and `.view(bfloat16)`
        # reinterprets the uint16 block in place — same pointer, same bytes, no
        # conversion. Doing either per call would put an allocation and a type
        # check inside the GIL-held region.
        self._input_t = torch.from_numpy(self._input_np)
        self._policy_t = torch.from_numpy(self._policy_np).view(torch.bfloat16)
        self._value_t = torch.from_numpy(self._value_np)
        assert self._policy_t.data_ptr() == torch.from_numpy(self._policy_np).data_ptr()

        # C12. The same argument as GraphedForward._build_view_cache, on the host
        # side: `self._input_t[:count]` and the two output slices are three
        # Python objects and three dispatcher round trips per batch, inside the
        # GIL-held callback, for views that alias storage which by construction
        # never moves. There are only `max_batch` of each, so they are built here
        # instead of on every crossing.
        self._row_views = {
            count: (self._input_t[:count], self._policy_t[:count],
                    self._value_t[:count])
            for count in range(1, self.max_batch + 1)}

        self._use_autocast = device.type == "cuda"

        # C10b. The graphed forward, or None for the eager callback. Built
        # BEFORE the buffers are page-locked and before any search thread
        # exists: capture must own the device while it runs, and it allocates
        # (each shape's private pool), which is exactly what the callback may
        # not do afterwards.
        self.graph = None
        if graphs and device.type == "cuda":
            from playing.v6 import graphs as graph_module

            self.graph = graph_module.build(graph_method, model, device, self.max_batch,
                                            sizes=graph_sizes)
            self.graph_report = self.graph.report
            self.graph_sizes = self.graph.sizes
        else:
            self.graph_report = None
            self.graph_sizes = ()
            # The eager path's device-side token block, int64 because
            # nn.Embedding indexes with int64. The graphed path has no
            # equivalent: it stages int32 and the widening is inside the graph.
            self._device_tokens = torch.empty((self.max_batch, SEQ_LENGTH), dtype=torch.long,
                                              device=device)

        # Test-only, and it stamps BOTH sides of the boundary because a padded
        # row can leak in two independent ways:
        #
        #   host   rows [count, max_batch) of the policy buffer get bf16 NaN,
        #          which catches C++ reading a row it was not given;
        #   device rows [count, padded) of the graph's own output get NaN,
        #          which catches THIS side narrowing wrongly and handing a
        #          padding row's priors out as if they belonged to a leaf.
        #
        # The second was added because the C10b drill's `pad-leak` mutation
        # survived a suite that only had the first. Off in production: together
        # they are a 2 MiB store per batch at max_batch 256.
        self.poison_unused_rows = False

        self.pinned = self._pin_buffers() if (pin and device.type == "cuda") else False

    # --- the callback ----------------------------------------------------

    def _evaluate(self, count: int) -> None:
        """Rows [0, count) of the input buffer -> the policy and value buffers.

        Called by the C++ dispatcher, once per batch, with the GIL held.

        FOUR TORCH CALLS AND NO ALLOCATION on the graphed path. `graph.run`
        copies the pinned int32 rows into the one static input, restores any
        padding rows a larger previous batch dirtied, and replays the graph
        captured for `pad_to(count)`; it hands back views already narrowed to
        `count`, so the padded rows' outputs have no name here to be copied out
        by accident.
        """
        rows_in, policy_out, value_out = self._row_views[count]
        if self.graph is not None:
            policy, value = self.graph.run(count, rows_in,
                                           poison=self.poison_unused_rows)
        else:
            tokens = self._device_tokens[:count]
            tokens.copy_(rows_in, non_blocking=self.pinned)
            with torch.no_grad():
                with torch.amp.autocast_mode.autocast(device_type="cuda",
                                                      dtype=AUTOCAST_DTYPE,
                                                      enabled=self._use_autocast):
                    policy, value = self.model(tokens)
        if self.poison_unused_rows:
            self._policy_np[count:] = POISON_BITS
        # C12. THE VALUE COPY IS ISSUED FIRST AND ASYNCHRONOUSLY, AND THE ORDER
        # IS THE OPTIMISATION. Both copies run on the same stream, so the policy
        # copy below — which is blocking — also completes this one. Two blocking
        # copies cost two `cudaStreamSynchronize` round trips, and on WDDM a
        # round trip is ~50 us against 96 bytes of actual payload at 24 rows
        # (BENCH.md C12-3 measures the `value` phase at 59 us per crossing).
        # Reordering makes it one round trip and copies exactly the same bytes.
        #
        # bf16 -> float32 is an exact widening: the reference's
        # `values_cpu[i].item()` produces the same number through the same
        # promotion, just one element at a time.
        value_out.copy_(value, non_blocking=True)
        # bf16 -> bf16, exactly the reference's `policy_logits.cpu()`. Blocking,
        # and therefore the D2H sync point for BOTH copies — nothing below needs
        # an explicit synchronize, and nothing may be added below that assumes
        # otherwise.
        policy_out.copy_(policy)

    def padded_batch(self, count: int) -> int:
        """The shape `count` rows are actually evaluated at. `count` if ungraphed."""
        return count if self.graph is None else self.graph.pad_to(count)

    def graph_counters_reset(self) -> None:
        """Zero the replay/rows/padding counters. No-op on the eager path."""
        if self.graph is not None:
            self.graph.reset_counters()

    def graph_pad_ratio(self) -> float:
        """Padded rows divided by real rows since the last reset.

        1.0 is a batch that always landed exactly on a captured shape; 2.7x is
        what the reuse regime pays for evaluating 3-row batches at shape 8. The
        eager path reports 1.0 because it has no padding, which keeps the
        benchmark's column meaningful in both configurations rather than blank
        in one.
        """
        if self.graph is None or self.graph.rows == 0:
            return 1.0
        return self.graph.padded_rows / self.graph.rows

    # --- page-locking ----------------------------------------------------

    def _pin_buffers(self) -> bool:
        """Page-lock the three C++-owned buffers, via torch's own cudart.

        Scope §2.1 specifies the input buffer as pinned. Doing it from C++ would
        make the CUDA runtime a build dependency of an extension that is
        deliberately torch-free (Global Rule 7), and torch already exposes
        `cudaHostRegister`, so the registration happens here and the matching
        unregister is handed back to C++ through `set_teardown` — which runs
        immediately before the allocations are freed. Registering without that
        hand-back would leave CUDA holding a mapping onto memory the allocator
        had taken back.

        The policy buffer is registered too, and it is the one that pays: at
        batch 256 it takes a 2 MiB device-to-host copy per batch against the
        input's 68 KiB. Failure is not fatal — an unpinned copy is correct and
        merely synchronous — so a refusal is reported, not raised.
        """
        cudart = torch.cuda.cudart()
        spans = self.core.buffer_spans()
        registered: list[tuple[int, int]] = []
        try:
            for address, nbytes in spans.values():
                status = cudart.cudaHostRegister(address, nbytes, 0)
                if int(status) != 0:
                    raise RuntimeError(f"cudaHostRegister returned {int(status)}")
                registered.append((address, nbytes))
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            for address, _ in registered:
                cudart.cudaHostUnregister(address)
            self.pin_error = str(exc)
            return False

        def _unregister() -> None:
            for address, _ in registered:
                cudart.cudaHostUnregister(address)

        self.core.set_teardown(_unregister)
        self.pin_error = None
        return True

    # --- lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Drop this object's references to the C++ evaluator and its views.

        Idempotent, and deliberately NOT a free. `Search.set_evaluator` is bound
        with pybind11's `keep_alive<1, 2>`, so a search that has been given an
        evaluator keeps it alive for its own lifetime whatever this method does.
        That is the property that makes the page-locking safe: the buffers cannot
        be released while anything still holds a pointer to them, so the teardown
        hook — and therefore `cudaHostUnregister` — cannot run early.

        What it does do is drop the last references this object holds, so that
        once the search is gone too the C++ destructor runs promptly rather than
        at interpreter shutdown, when running Python from a destructor is at its
        most fragile. Callers should still clear the search first
        (`search.set_evaluator(None)`) for readability; correctness does not
        depend on the order.
        """
        if self._closed:
            return
        self._closed = True
        # Before the buffers, because the captured graphs hold device memory
        # whose pools are only returned when the last graph referencing them
        # dies. Dropping it here rather than at interpreter shutdown is what
        # lets a bench build a second evaluator without capturing on top of the
        # first one's ~560 MiB.
        self.graph = None
        # C12. BEFORE the three tensors, and it is not cosmetic: every entry
        # holds a view onto the same C++-owned memory, so leaving the cache
        # populated would keep the buffers alive past `close()` and defer the
        # `cudaHostUnregister` this method exists to sequence.
        self._row_views = {}
        self._input_t = None
        self._policy_t = None
        self._value_t = None
        self._input_np = None
        self._policy_np = None
        self._value_np = None
        self.core = None

    def __enter__(self) -> "TorchEvaluator":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def build(max_batch: int, *, model_path: Path | None = None,
          device: torch.device | None = None,
          switch_interval: float = DEFAULT_SWITCH_INTERVAL,
          pin: bool = True, graphs: bool = True, graph_method: str = "cudagraph",
          graph_sizes=None) -> TorchEvaluator:
    """Load the default checkpoint and wrap it. The one-liner every caller wants."""
    model, resolved = load_default_model(model_path, device)
    return TorchEvaluator(model, resolved, max_batch, switch_interval=switch_interval,
                          pin=pin, graphs=graphs, graph_method=graph_method,
                          graph_sizes=graph_sizes)


__all__ = [
    "AUTOCAST_DTYPE",
    "DEFAULT_MODEL",
    "DEFAULT_SWITCH_INTERVAL",
    "MIN_VOCAB_SIZE",
    "POISON_BITS",
    "POLICY_SIZE",
    "SEQ_LENGTH",
    "TorchEvaluator",
    "build",
    "load_default_model",
    "require_engine_contract",
]
