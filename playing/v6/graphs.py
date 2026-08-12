"""C10b Stage 1 — the forward pass as a replayable CUDA graph.

WHY THIS FILE EXISTS
====================
C9a measured the small-batch cost and found it on the HOST: ~110 ATen op
submissions at 17-34 us each, so a 6-layer v5 forward costs ~4 ms of CPU
regardless of batch size, with `launch ~ total` and `drain ~ 0`. The GPU was
idle-waiting on the dispatcher. Nothing downstream of that — bigger batches,
double buffering, a second stream — can help, because there is no GPU time to
hide the host work behind.

A CUDA graph deletes the submissions rather than overlapping them. The ~110
kernel launches are recorded once, at construction, and every subsequent batch
is ONE `cudaGraphLaunch`. That is the whole of Stage 1, and C10b's ordering
(graphs first, pipeline second) follows from C9a's measurement rather than from
a preference.

THE THREE PROPERTIES A CAPTURED FORWARD HAS TO HAVE
===================================================
1. FIXED SHAPES. A graph records the kernels for one input shape. So a small set
   of batch sizes is captured and every real batch is padded up to the next one.
   `sizes` is that set; `pad_to` is the rounding; the padded rows' outputs are
   never read (see `GraphedForward.run`, which returns views narrowed to
   `count`, and tests/test_c10b_graphs.py, which fails if a padded row's prior
   can reach an expansion).

2. ALLOCATION STABILITY. Every tensor the graph touches must live at the same
   address on every replay. The input is one static int32 staging buffer; the
   outputs are the tensors the capture itself allocated, out of the graph's own
   private memory pool. The callback therefore allocates nothing: it copies into
   a static buffer, replays, and copies out of a static buffer.

3. NUMERICAL IDENTITY. Capture can change kernel selection, so the graphed
   forward is not automatically the eager one. Here it is — verified bit-for-bit
   over the 500-position Gate 2 corpus at every captured shape — and Gate 2 is
   re-run on it regardless, because "we checked" is the only reason to believe
   it (C10b acceptance 1).

WHY PRIVATE MEMORY POOLS AND NOT ONE SHARED POOL
================================================
Graphs may share a memory pool only if they are always replayed in the order
they were captured. This dispatcher replays whichever size the next batch rounds
up to, in whatever order the search produces — so the precondition does not
hold, and the shared-pool version would be a correctness bug that shows up as
occasional wrong priors under load rather than as a crash. Each graph therefore
owns its pool. The cost is memory (measured: ~560 MiB for {8, 32, 128, 256} on
the 10.9M student) and it is reported by `capture_report` so a host can see it.

WHY THE INT32 -> INT64 WIDENING IS INSIDE THE GRAPH
===================================================
C++ writes int32 tokens; `nn.Embedding` indexes with int64. Converting in the
callback means a cross-device, cross-dtype `copy_`, which allocates a temporary
on every batch — exactly what property 2 forbids. Capturing the widening puts
the conversion kernel inside the graph and leaves the callback with one pure
pinned-int32 H2D copy.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch

import guofish_core

# The token width and the autocast dtype are the boundary's contract, taken from
# the same places playing/v6/evaluator.py takes them and for the same reason: a
# disagreement here is a silently mis-shaped input. `SEQ_LENGTH` comes from the
# C++ constant that actually governs (cpp/tokens.hpp kSeqLength, re-exported by
# guofish_core) rather than from `core.mctsv4`, which is retired along with the
# Python search.
SEQ_LENGTH = guofish_core.SEQ_LENGTH
AUTOCAST_DTYPE = torch.bfloat16

# The position padding rows are filled with. Any legal position works — the rows
# are never read — but it must be FIXED, so that a padded batch costs the same
# every time and so that nothing about the pad depends on what the previous
# batch happened to contain. The start position is the obvious constant, and
# `guofish_core.tokens` is the production tokenizer, which keeps python-chess
# out of this module.
PAD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# THE LADDER, AND WHY IT IS DENSER THAN THE BRIEF PROPOSED
# --------------------------------------------------------
# C10b Stage 1 proposed {32, 128}. That set was built, measured and rejected. A
# captured shape is what a batch PAYS FOR, and one graphed forward costs about
# 0.55 ms fixed plus ~44 us per row (BENCH.md C10b-3a), so a 42-row batch
# evaluated at shape 128 spends 6.2 ms where 2.4 ms would do. The first live
# W x K grid, run on {1, 8, 32, 128}, shows exactly that: `pad waste` swings
# between 1.09x and 2.87x across cells and throughput tracks it inversely —
# W=1/K=48 measured 6,888 sims/s on that ladder and 14,828 on this one, a 2.15x
# difference from padding alone. Padding waste was the largest single term in
# the grid, larger than W and K together.
#
# So the ladder is chosen to BOUND the waste rather than to be small: no gap
# costs more than ~1.4x, and the measured spread across every cell falls to
# 1.08-1.29x. Two entries are there for reasons the ladder's shape does not
# explain:
#
#   8  because C10h measured 3.2 rows per crossing in the reuse-heavy endgame
#      regime — 92% of leaves answered by the transposition cache — and padding
#      a 3-row batch to 32 would spend 1.9 ms where 0.85 ms does.
#
#   1  because W=1/K=1 is an ACCEPTANCE configuration, not a production one:
#      Gate 2b's 500-position differential and the grid's serial reference row
#      both run `max_batch=1`, so every batch there is exactly one row. Padding
#      it to 8 would evaluate that row under a different cuBLAS tiling, and
#      shape changes bf16 logits by up to 1 ulp — a property of the library, not
#      of graphs (eager torch does the same; BENCH.md C10b-1c). Capturing 1
#      keeps a batch of one evaluating at shape one, which is what makes C10's
#      Gate 2b evidence still evidence.
#
# The cost is device memory: ~1.9 MiB per captured row, so ~730 MiB at
# max_batch 128 and ~1.2 GiB at 256. `CaptureReport` publishes it rather than
# leaving a host to discover it, and `sizes=` overrides the whole thing.
DEFAULT_CAPTURE_SIZES = (1, 8, 16, 24, 32, 48, 64, 96, 128)


@dataclass
class CaptureReport:
    """What capture cost, so a host can see it rather than infer it."""

    method: str
    sizes: tuple[int, ...] = ()
    seconds: float = 0.0
    reserved_before: int = 0
    reserved_after: int = 0
    warmup_iters: int = 0

    @property
    def reserved_delta(self) -> int:
        return self.reserved_after - self.reserved_before

    def describe(self) -> str:
        return (f"{self.method}: sizes {list(self.sizes)}, {self.seconds:.2f} s, "
                f"+{self.reserved_delta / 2**20:.0f} MiB reserved")


def resolve_sizes(max_batch: int, sizes=None) -> tuple[int, ...]:
    """The captured shapes, ascending, with `max_batch` always the last one.

    `max_batch` has to be in the set: it is the largest batch the dispatcher can
    hand over, so without it some batch would have no shape to round up to. The
    rest are dropped if they are at or above it, which is what makes
    `GraphedForward(max_batch=8)` a legal single-shape configuration rather than
    an error.
    """
    if max_batch < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch}")
    proposed = DEFAULT_CAPTURE_SIZES if sizes is None else tuple(int(s) for s in sizes)
    for size in proposed:
        if size < 1:
            raise ValueError(f"capture sizes must be >= 1, got {size}")
    kept = sorted({s for s in proposed if s < max_batch} | {int(max_batch)})
    return tuple(kept)


class GraphedForward:
    """A v5 forward captured as one CUDA graph per shape in `sizes`.

    The public surface is three things: `sizes`, `pad_to(count)` and
    `run(count)`. Everything else is capture machinery that runs once.

    `run(count)` returns `(policy, value)` views NARROWED TO `count` rows. The
    narrowing is the contract, not a convenience — the rows beyond `count` are
    the padding's outputs and reading them would be reading the answer to a
    question nobody asked.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, max_batch: int, *,
                 sizes=None, warmup_iters: int = 3, pad_fen: str = PAD_FEN):
        if device.type != "cuda":
            raise ValueError("GraphedForward requires a CUDA device; the graph IS the "
                             "point and there is no CPU equivalent")
        self.model = model
        self.device = device
        self.max_batch = int(max_batch)
        self.sizes = resolve_sizes(self.max_batch, sizes)
        self.warmup_iters = int(warmup_iters)

        # THE ONE STATIC INPUT. int32 because that is what C++ writes, so the
        # H2D copy in the callback is a straight byte move from pinned memory
        # with no conversion and no temporary.
        self.tokens = torch.zeros((self.max_batch, SEQ_LENGTH), dtype=torch.int32,
                                  device=device)

        import guofish_core

        self._pad_row = torch.from_numpy(guofish_core.tokens(pad_fen)).to(device)
        # Every row starts as padding, so a first batch smaller than its captured
        # size never sees an uninitialised token id — which `nn.Embedding` would
        # answer with an out-of-bounds device-side read rather than an exception.
        self.tokens[:] = self._pad_row
        # Rows [0, _dirty) may hold real tokens; the rest are known padding. The
        # refill in `run` is skipped whenever the tail is already clean, which is
        # every batch that is not smaller than its predecessor.
        self._dirty = 0

        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._policy: dict[int, torch.Tensor] = {}
        self._value: dict[int, torch.Tensor] = {}
        self.report = CaptureReport(method="cudagraph", sizes=self.sizes,
                                    warmup_iters=self.warmup_iters)
        # Counted rather than timed, and the three counters answer three
        # different questions. `replays` must equal the dispatcher's crossing
        # count or a batch went somewhere other than the graph. `rows` against
        # `padded_rows` is the padding waste C10b Stage 3.4 asks for — under
        # deep reuse a 3-row batch runs at shape 8, and the ratio is how much
        # of the GPU's work was answering questions nobody asked. `by_size`
        # keeps the shape histogram, which is what says whether a captured size
        # earns its memory.
        #
        # Three integer updates per batch, inside the GIL-held window. Measured
        # at well under a microsecond against a ~2 ms replay; the alternative
        # was a C++-side histogram, which is more machinery for a number only
        # the bench reads.
        self.replays = 0
        self.rows = 0
        self.padded_rows = 0
        self.by_size = {size: 0 for size in self.sizes}
        self._capture()

    # --- capture ----------------------------------------------------------

    def _eager(self, tokens_int32: torch.Tensor):
        """The forward exactly as playing/v6/evaluator.py ran it before graphs.

        Under `no_grad` and inside the autocast context, which is where the
        capture has to happen: the brief allows either capturing inside autocast
        or exporting an already-bf16 module, and this is the first — it keeps
        one spelling of the forward for the eager path, the capture and the
        Gate 2 golden generator.
        """
        with torch.no_grad():
            with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE,
                                                  enabled=True):
                return self.model(tokens_int32.to(torch.int64))

    def _capture(self) -> None:
        started = time.perf_counter()
        self.report.reserved_before = torch.cuda.memory_reserved(self.device)

        # Global warmup on a side stream, per the CUDA graph capture protocol:
        # lazy one-time initialisation (cuBLAS handles, autotuning, the caching
        # allocator's first blocks) must be complete before capture begins, or
        # it gets recorded into the graph or fails the capture outright.
        self._warm(self.sizes[0])

        for size in self.sizes:
            self._warm(size)
            graph = torch.cuda.CUDAGraph()
            # No `pool=`: a private pool per graph. See the module docstring —
            # sharing requires a replay order this dispatcher does not have.
            with torch.cuda.graph(graph):
                policy, value = self._eager(self.tokens[:size])
            self._graphs[size] = graph
            self._policy[size] = policy
            self._value[size] = value

        torch.cuda.synchronize(self.device)
        self.report.reserved_after = torch.cuda.memory_reserved(self.device)
        self.report.seconds = time.perf_counter() - started

    def _warm(self, size: int) -> None:
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(self.warmup_iters):
                self._eager(self.tokens[:size])
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)

    # --- the shape contract -----------------------------------------------

    def pad_to(self, count: int) -> int:
        """The captured shape `count` rows will be evaluated at."""
        if count < 1 or count > self.max_batch:
            raise ValueError(f"count must be in [1, {self.max_batch}], got {count}")
        for size in self.sizes:
            if size >= count:
                return size
        # Unreachable: `sizes` always ends at max_batch and count <= max_batch.
        raise AssertionError(f"no captured size covers {count}; sizes={self.sizes}")

    def pad_rows(self, count: int) -> int:
        return self.pad_to(count) - count

    # --- the hot path -----------------------------------------------------

    def stage(self, count: int, source: torch.Tensor) -> int:
        """Copy `count` rows of host int32 tokens in and clean the pad tail.

        Returns the captured size the batch will run at. Split out from `run`
        so a caller can time the two halves apart, and so the pad discipline is
        one readable place rather than a line inside a hot function.
        """
        padded = self.pad_to(count)
        self.tokens[:count].copy_(source, non_blocking=True)
        if self._dirty > count:
            # Only the rows a PREVIOUS, larger batch wrote need restoring; the
            # rest are still the pad position from construction. This is why the
            # common case — batches of non-decreasing size — costs nothing.
            self.tokens[count:self._dirty] = self._pad_row
        self._dirty = count
        return padded

    def replay(self, padded: int) -> None:
        self._graphs[padded].replay()
        self.replays += 1

    def outputs(self, count: int, padded: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The captured shape's outputs, narrowed to the rows that were asked for."""
        return self._policy[padded][:count], self._value[padded][:count]

    def poison_pad(self, count: int, padded: int) -> None:
        """Stamp the padded rows' OUTPUTS with NaN. Test-only; see `run`.

        The host-side poison in playing/v6/evaluator.py catches one kind of leak
        — C++ reading a policy row it was not given. It cannot catch the other,
        which is THIS side handing out the wrong rows, because whatever
        `outputs` selects is copied into host rows [0, count) and then looks
        entirely legitimate. The C10b drill's `pad-leak` mutation is exactly
        that second kind and it survived the suite until this existed.

        Stamped after the replay, because the replay is what writes these rows.
        """
        if padded > count:
            self._policy[padded][count:padded].fill_(float("nan"))
            self._value[padded][count:padded].fill_(float("nan"))

    def run(self, count: int, source: torch.Tensor,
            poison: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        padded = self.stage(count, source)
        self.replay(padded)
        if poison:
            self.poison_pad(count, padded)
        self.rows += count
        self.padded_rows += padded
        self.by_size[padded] += 1
        return self.outputs(count, padded)

    def reset_counters(self) -> None:
        self.replays = 0
        self.rows = 0
        self.padded_rows = 0
        self.by_size = {size: 0 for size in self.sizes}

    # --- what a test needs to see -----------------------------------------

    def pad_view(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The padded rows' outputs — the ones production must never read.

        Exists so tests/test_c10b_graphs.py can assert those rows hold the
        padding's answer and that the answer never reaches an expansion. There
        is no caller in the search.
        """
        padded = self.pad_to(count)
        return self._policy[padded][count:padded], self._value[padded][count:padded]

    def eager(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The un-captured forward over the same static input, for comparison."""
        return self._eager(self.tokens[:count])


class CompiledForward:
    """`torch.compile(model, mode="reduce-overhead")` behind `GraphedForward`'s API.

    The C10b brief asks for this to be TRIED FIRST and measured against manual
    capture, keeping whichever wins *and is shape-stable*. It is kept — as a
    selectable method, not as the default — so that comparison stays
    reproducible rather than becoming a sentence in DECISIONS.md. What it
    measured here is in BENCH.md (C10b-1b): faster than manual capture on every
    shape, and it fails Gate 2 by four orders of magnitude, because Inductor
    fuses the epilogues and the bf16 logits come out different. Speed does not
    buy that back; the brief says a graphed forward that fails Gate 2 is
    rejected regardless of it.

    Two further frictions, recorded because they are reasons and not grumbles:
    it needs `torch._inductor.config.use_static_cuda_launcher = False` on this
    machine or every launch raises `OverflowError` from the static launcher, and
    warmup costs ~35 s of compilation at engine start against ~2 s of capture.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, max_batch: int, *,
                 sizes=None, warmup_iters: int = 3, pad_fen: str = PAD_FEN,
                 disable_static_launcher: bool = True):
        if device.type != "cuda":
            raise ValueError("CompiledForward requires a CUDA device")
        self.model = model
        self.device = device
        self.max_batch = int(max_batch)
        self.sizes = resolve_sizes(self.max_batch, sizes)
        self.warmup_iters = int(warmup_iters)

        if disable_static_launcher:
            import torch._inductor.config as inductor_config

            inductor_config.use_static_cuda_launcher = False

        self.tokens = torch.zeros((self.max_batch, SEQ_LENGTH), dtype=torch.int32,
                                  device=device)
        import guofish_core

        self._pad_row = torch.from_numpy(guofish_core.tokens(pad_fen)).to(device)
        self.tokens[:] = self._pad_row
        self._dirty = 0
        self.replays = 0
        self.rows = 0
        self.padded_rows = 0
        self.by_size = {size: 0 for size in self.sizes}

        self._compiled = torch.compile(model, mode="reduce-overhead", dynamic=False)
        self.report = CaptureReport(method="torch.compile/reduce-overhead", sizes=self.sizes,
                                    warmup_iters=self.warmup_iters)
        self._last = (None, None)
        self._warm()

    def _forward(self, tokens_int32: torch.Tensor):
        with torch.no_grad():
            with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE,
                                                  enabled=True):
                return self._compiled(tokens_int32.to(torch.int64))

    def _warm(self) -> None:
        started = time.perf_counter()
        self.report.reserved_before = torch.cuda.memory_reserved(self.device)
        for size in self.sizes:
            for _ in range(max(self.warmup_iters, 3)):
                self._forward(self.tokens[:size])
        torch.cuda.synchronize(self.device)
        self.report.reserved_after = torch.cuda.memory_reserved(self.device)
        self.report.seconds = time.perf_counter() - started
        self.frames_after_warmup = self.frame_count()

    @staticmethod
    def frame_count() -> int:
        """Dynamo's compiled-frame counter — the shape-stability guard.

        `torch.compile` can silently recompile per shape, and a recompilation
        mid-search is a multi-second stall that looks like a hung engine. The
        counter is sampled after warmup and re-read by
        `assert_no_recompilation`; a change means a shape reached the model that
        warmup did not cover.
        """
        from torch._dynamo.utils import counters

        return int(dict(counters["frames"]).get("total", 0))

    def assert_no_recompilation(self) -> None:
        now = self.frame_count()
        if now != self.frames_after_warmup:
            raise AssertionError(
                f"torch.compile recompiled during the run: {self.frames_after_warmup} "
                f"frames after warmup, {now} now. A shape reached the model that the "
                f"warmup did not capture, and the graph is not shape-stable.")

    def pad_to(self, count: int) -> int:
        if count < 1 or count > self.max_batch:
            raise ValueError(f"count must be in [1, {self.max_batch}], got {count}")
        for size in self.sizes:
            if size >= count:
                return size
        raise AssertionError(f"no captured size covers {count}; sizes={self.sizes}")

    def pad_rows(self, count: int) -> int:
        return self.pad_to(count) - count

    def stage(self, count: int, source: torch.Tensor) -> int:
        padded = self.pad_to(count)
        self.tokens[:count].copy_(source, non_blocking=True)
        if self._dirty > count:
            self.tokens[count:self._dirty] = self._pad_row
        self._dirty = count
        return padded

    def replay(self, padded: int) -> None:
        self._last = self._forward(self.tokens[:padded])
        self.replays += 1

    def outputs(self, count: int, padded: int) -> tuple[torch.Tensor, torch.Tensor]:
        del padded
        policy, value = self._last
        return policy[:count], value[:count]

    def poison_pad(self, count: int, padded: int) -> None:
        del count, padded   # compile's outputs are its own; nothing to stamp

    def run(self, count: int, source: torch.Tensor,
            poison: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        del poison
        padded = self.stage(count, source)
        self.replay(padded)
        self.rows += count
        self.padded_rows += padded
        self.by_size[padded] += 1
        return self.outputs(count, padded)

    def reset_counters(self) -> None:
        self.replays = 0
        self.rows = 0
        self.padded_rows = 0
        self.by_size = {size: 0 for size in self.sizes}

    def eager(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=AUTOCAST_DTYPE,
                                                  enabled=True):
                return self.model(self.tokens[:count].to(torch.int64))


METHODS = {"cudagraph": GraphedForward, "compile": CompiledForward}


def build(method: str, model, device, max_batch: int, **kwargs):
    try:
        cls = METHODS[method]
    except KeyError:
        raise ValueError(f"unknown graph method {method!r}; "
                         f"expected one of {sorted(METHODS)}") from None
    return cls(model, device, max_batch, **kwargs)


__all__ = [
    "CaptureReport",
    "CompiledForward",
    "DEFAULT_CAPTURE_SIZES",
    "GraphedForward",
    "METHODS",
    "PAD_FEN",
    "build",
    "resolve_sizes",
]
