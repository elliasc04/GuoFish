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

# The v5 model contract, imported from the reference rather than restated. It is
# three equalities — seq_len 68, cls_index 67, policy_size 4096 — and every one of
# them is an input-format assumption whose violation means silently evaluating a
# position the network never saw in that form. `core.mctsv4` owns the check
# because it owns the tokenizer that produces the format.
#
# This is the ONLY thing this module takes from the reference, and it deliberately
# does not import the reference's search.
from core.mctsv4 import AUTOCAST_DTYPE, POLICY_SIZE, SEQ_LENGTH, require_v5_config

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


def load_default_model(model_path: Path | None = None,
                       device: torch.device | None = None) -> tuple[torch.nn.Module,
                                                                    torch.device]:
    """The v5 student on the best available device, through playv5's loader.

    Routed through `playing.v5.playv5.load_model` rather than a second loader so
    the port cannot end up evaluating a checkpoint the reference would have
    loaded differently — the dtype selection, the value_scale attachment and the
    CPU int8 path all live there.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from playing.v5.playv5 import load_model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(model_path) if model_path else DEFAULT_MODEL, device)
    require_v5_config(model)
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
                 graph_sizes=None):
        if max_batch < 1:
            raise ValueError(f"max_batch must be >= 1, got {max_batch}")
        require_v5_config(model)

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
        if self.graph is not None:
            policy, value = self.graph.run(count, self._input_t[:count],
                                           poison=self.poison_unused_rows)
        else:
            tokens = self._device_tokens[:count]
            tokens.copy_(self._input_t[:count], non_blocking=self.pinned)
            with torch.no_grad():
                with torch.amp.autocast_mode.autocast(device_type="cuda",
                                                      dtype=AUTOCAST_DTYPE,
                                                      enabled=self._use_autocast):
                    policy, value = self.model(tokens)
        if self.poison_unused_rows:
            self._policy_np[count:] = POISON_BITS
        # bf16 -> bf16, exactly the reference's `policy_logits.cpu()`. The copy is
        # the D2H sync point, so nothing below needs an explicit synchronize.
        self._policy_t[:count].copy_(policy)
        # bf16 -> float32, which is an exact widening: the reference's
        # `values_cpu[i].item()` produces the same number through the same
        # promotion, just one element at a time.
        self._value_t[:count].copy_(value)

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
    "DEFAULT_MODEL",
    "DEFAULT_SWITCH_INTERVAL",
    "POISON_BITS",
    "POLICY_SIZE",
    "SEQ_LENGTH",
    "TorchEvaluator",
    "build",
    "load_default_model",
]
