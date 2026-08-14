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

# C12b. Inductor compiles on first call at each shape, so a 3-iteration warmup
# spends most of itself inside the compiler and measures nothing about the
# steady state. Six is enough that the last few are compiled-and-settled, and
# `_warm_to_convergence` re-runs whole rounds until the frame counter stops
# moving regardless, so this is a starting point rather than a bound.
INDUCTOR_WARMUP_ITERS = 6

# Spare specialisations above the captured ladder, when C12b raises dynamo's
# `recompile_limit`. Nothing should use them — `pad_to` admits no shape outside
# `sizes` — so the headroom exists only so that the "exactly one frame per shape"
# assertion is what reports a surprise, rather than dynamo's silent eager
# fallback getting there first. See `InductorGraphedForward._raise_recompile_limit`.
RECOMPILE_HEADROOM = 8


def dynamo_frame_count() -> int:
    """`torch._dynamo`'s cumulative compiled-frame counter.

    THE INSTRUMENT THE "NO RECOMPILATION" ACCEPTANCE CRITERION IS MEASURED WITH.
    It is process-global and monotonic, which is what makes it usable: sample it
    when warmup settles, sample it again later, and any difference is a frame
    dynamo compiled in between. There is no per-module counter to prefer — a
    recompilation anywhere in the process is a recompilation that could have
    landed inside a capture.

    Imported inside the function because `torch._dynamo` is a heavy private
    import and the eager path must not pay for it.
    """
    from torch._dynamo.utils import counters

    return int(dict(counters["frames"]).get("total", 0))


def configure_inductor() -> dict:
    """Set the two Inductor options C12b ships, and return what was set.

    Returned rather than merely applied so `CaptureReport` and DECISIONS.md can
    state the configuration the numbers were taken under instead of describing
    it. Both settings are corrections to defaults that are wrong for this engine,
    and neither is a tuning knob.

    `use_static_cuda_launcher = False`
        REQUIRED FOR CORRECTNESS ON THIS MACHINE, not preferred. With torch
        2.8.0+cu129 / triton 3.4.0 on this RTX 5070, the static launcher raises
        `OverflowError: Python int too large to convert to C long` from
        `_StaticCudaLauncher._launch_kernel` on the very first Inductor kernel.
        C10b hit the same thing and recorded it; it is restated here because
        C10b's `CompiledForward` was never shipped and this is.

    `triton.autotune_pointwise = False`
        REQUIRED FOR DETERMINISM, and it is the whole answer to the brief's
        "pin or ship the autotune cache". Left on, Inductor benchmarks several
        Triton configs per pointwise/reduction kernel at first call and caches
        the winner in a `.best_config` file; the winner is chosen from measured
        times, so it varies with GPU clock and machine load. Measured: **17 of
        28 `.best_config` files differ between two cold compiles of this model**
        (XBLOCK 128 against 256, XBLOCK 1024/4 warps against 512/8 warps), and a
        reduction kernel's block size changes the accumulation order, which
        changes the bits. Turned off, the heuristics emit ONE config per kernel,
        **zero `.best_config` files are written**, and three independent cold
        compiles produce bit-identical priors at every shape.

        Disabling it is strictly better than pinning a cache: there is no
        artifact to ship, nothing keyed on the GPU, driver or torch version, and
        nothing to go stale — and the cost is not measurable. On the graphed
        forward's device time at shape 24 it is 964.6 us against 970.3 us with
        autotuning on, i.e. inside the run-to-run spread, and capture is ~15 s
        faster because the benchmarking is what was slow. BENCH.md C12b-2.
    """
    import torch._inductor.config as inductor_config

    settings = {"use_static_cuda_launcher": False, "triton.autotune_pointwise": False}
    inductor_config.use_static_cuda_launcher = False
    inductor_config.triton.autotune_pointwise = False
    return settings


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
        self._build_view_cache()

    # --- the view cache (C12) ---------------------------------------------

    def _build_view_cache(self) -> None:
        """One tensor view per (count) and per (shape, count), built once.

        C12 PROFILING FINDING. `self.tokens[:count]` and `self._policy[p][:count]`
        are cheap in the sense that they copy nothing — and they are not cheap in
        the sense that matters, because each one constructs a Python object and
        runs a slice through the ATen dispatcher, inside the GIL-held callback,
        once per boundary crossing. BENCH.md C12-3 measures the whole callback at
        ~205 us of host time against a 1,282 us graph, and `stage` alone at 55 us
        of which the `cudaMemcpyAsync` is 17.8. There are only `max_batch`
        possible counts and nine possible shapes, so every one of these views can
        exist before the search starts.

        The views alias the same storage as the tensors they are cut from —
        `tokens` and the captured outputs never change identity, which is exactly
        the property CUDA graph capture already requires of them (docstring
        property 2). So a cached view cannot go stale: if it could, the graph
        would already be replaying against the wrong addresses.

        Memory is negligible: `max_batch` + `len(sizes) x max_batch` view objects,
        no storage.

        WHY THE CACHE DOES NOT SHORT-CIRCUIT `pad_to` OR `replay`. Both are
        mutation targets in tools/drill_c10b_graphs.py (`round-down`,
        `stale-replay`), and a `run` that inlined them would leave those
        mutations unexercised — the drill would keep reporting that the suite
        catches them while the suite no longer ran them. The cache is keyed on
        what `pad_to` returns rather than on what it returned at construction.
        """
        self._token_views = {count: self.tokens[:count]
                             for count in range(1, self.max_batch + 1)}
        self._output_views = {
            (size, count): (self._policy[size][:count], self._value[size][:count])
            for size in self.sizes for count in range(1, size + 1)}

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
        self._token_views[count].copy_(source, non_blocking=True)
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
        cached = self._output_views.get((padded, count))
        if cached is not None:
            return cached
        # `padded < count` is unreachable through `pad_to`, and reachable through
        # a mutation of it (drill_c10b_graphs.py `round-down`). Slice rather than
        # KeyError so the mutation still produces the SHAPE MISMATCH the caller
        # would have seen before the cache existed, and fails the test it is
        # aimed at rather than a different one.
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
    """DEPRECATED (C12b). `torch.compile(model, mode="reduce-overhead")` behind `GraphedForward`'s API.

    **SUPERSEDED BY `InductorGraphedForward`. DO NOT SHIP THIS, AND PREFER IT FOR
    NOTHING.** It is retained only so that C10b-1b's measurement stays
    reproducible; `graph_method="compile"` still selects it and `TorchEvaluator`
    refuses to combine it with `compile=True`.

    C12b rejected this shape of the idea explicitly. `mode="reduce-overhead"`
    brings Inductor's OWN cudagraph trees, and stacking those on the manual
    capture replaces `pad_to`, the static input buffer, the pad-tail restore and
    the narrowed output views — i.e. every property `tools/drill_c10b_graphs.py`
    mutates and every shape and staleness drill built on them. Orphaning those
    drills costs more than the launch overhead the mode saves, which the manual
    capture was already saving. The shipped path takes Inductor's codegen and
    keeps C10b's machinery around it, which is `InductorGraphedForward`.

    Its other recorded liability is now a known bug rather than a mystery: this
    class needs `use_static_cuda_launcher = False` on this machine or every
    launch raises `OverflowError`, and `configure_inductor()` sets exactly that
    for the shipped path (DECISIONS.md, C12b).

    THE HISTORICAL MEASUREMENT, unchanged. The C10b brief asked for this to be
    TRIED FIRST and measured against manual capture, keeping whichever wins *and
    is shape-stable*. BENCH.md C10b-1b: faster than manual capture on every
    shape, and it fails Gate 2 by four orders of magnitude, because Inductor
    fuses the epilogues and the bf16 logits come out different. Speed did not buy
    that back, and C10b rejected it. **C12b's answer to the same numbers was the
    opposite one** — adopt the fusion and re-base the gate onto the frozen eager
    engine (Gate 2') — so what dates this class is the cudagraph stacking, not
    the numerics.

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


class InductorGraphedForward(GraphedForward):
    """C12b — the SAME manual capture, over an Inductor-codegen'd module.

    WHY THIS SUBCLASSES `GraphedForward` RATHER THAN `CompiledForward`
    =================================================================
    `CompiledForward` above is `mode="reduce-overhead"`, which brings Inductor's
    OWN cudagraph trees. The C12b brief forbids stacking those on top of the
    manual capture, and the reason is not tidiness: `pad_to`, the static input
    buffer, the pad-tail restore and the narrowed output views are the machinery
    every C10b shape and staleness drill bites on, and a compile-owned cudagraph
    replaces all of it with something the drills cannot see. Orphaning
    `tools/drill_c10b_graphs.py` is a bigger loss than the launch overhead
    `reduce-overhead` saves, which the manual capture is already saving anyway.

    So this class changes exactly one thing about the shipped path: the module
    that `_eager` calls is `torch.compile(model)` in DEFAULT mode — Inductor
    codegen, no cudagraphs — and everything else is inherited verbatim. `stage`,
    `replay`, `outputs`, `poison_pad`, `pad_to` and the view cache are the base
    class's, unmodified, so every property C10b established about them still
    holds and every drill still runs against the code it was written for.

    **The capture works, and that was not a foregone conclusion.** The brief asks
    for it to be reported as a finding if default mode could not be captured
    manually. It can: all nine shipped shapes capture, the replay is bit-identical
    to the un-captured compiled call (0 differing policy words at every shape),
    and no dynamo frame is compiled during capture. BENCH.md C12b-1 has the table.

    WHAT THIS COSTS THE NUMERICS, WHICH IS THE WHOLE POINT OF THE CHUNK
    ==================================================================
    Inductor changes no precision — it stays bf16 and fuses bias/GELU/LayerNorm
    into the matmul epilogue — but it changes the ACCUMULATION ORDER, and a bf16
    logit moved by one ulp moves a prior by ~1e-3, three orders over Gate 2's
    1e-6 bound. Measured here: ~63% of policy words differ from the eager forward
    at every shape, at a max |dlogit| of 0.03125. That is why C12b replaces Gate 2
    with Gate 2' (`tests/test_c12b_gate2prime.py`) rather than re-running it, and
    why `compile=False` stays shippable and stays certified.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, max_batch: int, *,
                 sizes=None, warmup_iters: int = INDUCTOR_WARMUP_ITERS,
                 pad_fen: str = PAD_FEN, max_warmup_rounds: int = 8,
                 configure: bool = True):
        if device.type != "cuda":
            raise ValueError("InductorGraphedForward requires a CUDA device; the "
                             "capture IS the shipped path and there is no CPU "
                             "equivalent to fall back to")
        self.inductor_settings = configure_inductor() if configure else {}
        # Kept because `TorchEvaluator` hands the base class whatever `self.model`
        # ends up being, and a caller that wants the unwrapped module back (a
        # bench doing an A/B, a test hoisting norms) should not have to know that
        # `torch.compile` returns a wrapper.
        self.eager_model = model
        self.max_warmup_rounds = int(max_warmup_rounds)
        self.warmup_rounds: dict[int, int] = {}
        self.frames_before_warmup = dynamo_frame_count()
        self.frames_after_warmup = self.frames_before_warmup
        self.recompile_limit = None
        # `dynamic=False` so a new batch width is a new compiled frame rather than
        # a silent fall back to a dynamic kernel. That is the opposite of what a
        # serving stack usually wants and exactly what a CAPTURED forward wants:
        # every shape this thing runs is a shape it captured, so a dynamic kernel
        # would trade specialisation away for a flexibility nothing here uses.
        super().__init__(torch.compile(model, dynamic=False), device, max_batch,
                         sizes=sizes, warmup_iters=warmup_iters, pad_fen=pad_fen)
        self.report.method = "inductor+cudagraph"

    # --- warmup and the recompilation guard --------------------------------

    def _warm_to_convergence(self, size: int) -> int:
        """Warm `size` until dynamo stops compiling. Returns the rounds it took.

        HAZARD 2 OF THE BRIEF, AND IT IS A CAPTURE HAZARD RATHER THAN A SPEED ONE.
        Inductor compiles per shape. A recompilation triggered *during* capture
        would be recorded into the graph — or would fail the capture outright —
        and it would be silent either way, so every shape has to be compiled and
        settled before any capture begins.

        "Settled" is asserted rather than assumed: a round is `warmup_iters`
        forwards, and the loop exits only when a whole round adds no compiled
        frame. The first round compiles, the second confirms, so two is the floor
        and anything more means a shape needed guards re-specialising. It raises
        rather than proceeding if it never converges, because the alternative is
        capturing a graph whose contents nobody can name.
        """
        for round_index in range(1, self.max_warmup_rounds + 1):
            before = dynamo_frame_count()
            self._warm(size)
            if dynamo_frame_count() == before:
                return round_index
        raise RuntimeError(
            f"torch.compile never stopped recompiling at shape {size}: "
            f"{self.max_warmup_rounds} warmup rounds of {self.warmup_iters} "
            f"iterations each still added dynamo frames. Capturing now would "
            f"record a graph whose kernels are not the ones a steady-state batch "
            f"would run.")

    def assert_no_recompilation(self, when: str = "since warmup") -> None:
        """Raise if dynamo has compiled anything since warmup settled.

        Public because it is an acceptance criterion, not an internal check:
        `tests/test_c12b_gate2prime.py` calls it after driving every captured
        shape through the production callback, and `tools/bench_c12b.py` calls it
        after each measured search. A recompilation mid-search is a multi-second
        stall that reads as a hung engine, and — captured — is worse than a stall.
        """
        now = dynamo_frame_count()
        if now != self.frames_after_warmup:
            raise AssertionError(
                f"torch.compile recompiled {when}: {self.frames_after_warmup} "
                f"dynamo frames when warmup settled, {now} now. A shape reached "
                f"the model that the per-shape warmup did not cover, so the "
                f"captured graphs are not the ones being replayed.")

    # --- capture ------------------------------------------------------------

    def _raise_recompile_limit(self) -> None:
        """Let dynamo specialise once per captured shape. THIS IS NOT A TUNING KNOB.

        **THE BUG THIS EXISTS TO PREVENT SHIPPED SILENTLY.** `torch._dynamo`'s
        `recompile_limit` defaults to **8**. The shipping ladder is
        `DEFAULT_CAPTURE_SIZES` — nine shapes — and `dynamic=False` makes each one
        its own specialised frame, so the ninth blows the limit. Dynamo's response
        is not to raise: it logs a warning and **falls back to running that shape
        EAGER, forever**. The engine ships `max_batch=128`, so the shape that fell
        back was 128, and the failure is invisible from every direction that
        matters — the capture succeeds, the priors are correct, the recompilation
        counter is stable (nothing is being compiled any more, which is the
        problem), and only a throughput table shows it: shape 128 measured exactly
        1.000x against eager with 0 of 524,288 policy words differing, i.e. it WAS
        eager.

        So the limit is raised to cover the ladder with headroom, and
        `_capture` asserts afterwards that exactly `len(sizes)` frames compiled —
        the assertion is the real guard, and this is just making room for it to
        pass. Raising it can only permit specialisations the ladder already
        implies; it cannot make an unbounded number of shapes appear, because
        `pad_to` admits no shape outside `sizes`.
        """
        import torch._dynamo.config as dynamo_config

        wanted = len(self.sizes) + RECOMPILE_HEADROOM
        # `recompile_limit` is the current spelling and `cache_size_limit` the
        # historical one; both are set when present so this does not quietly stop
        # working on a torch that renames it back or forward.
        for name in ("recompile_limit", "cache_size_limit"):
            if hasattr(dynamo_config, name) and getattr(dynamo_config, name) < wanted:
                setattr(dynamo_config, name, wanted)
        self.recompile_limit = getattr(dynamo_config, "recompile_limit", None)

    def assert_every_shape_is_fused(self) -> None:
        """Every captured shape must actually be running Inductor's code. MEASURED.

        THE COUNTER CANNOT ANSWER THIS AND IT TOOK A SHIPPED-CONFIG FAILURE TO
        NOTICE. The obvious guard is "warmup compiled one dynamo frame per shape",
        and it is wrong twice over: a shape that has hit the recompile limit stops
        compiling and therefore *converges immediately*, looking exactly like a
        shape that finished; and a second `torch.compile` of the same module in one
        process reuses dynamo's cache, so a perfectly healthy second evaluator
        compiles ZERO new frames. Both readings are indistinguishable from the
        failure by counting alone.

        So the question is asked semantically instead, of each shape, in the terms
        that actually matter: **does this shape's output differ from the unfused
        module's?** Inductor fuses the bias/GELU/LayerNorm epilogues and that moves
        ~60-65% of the bf16 policy words at every shape this engine captures. A
        shape that fell back to eager returns bit-identical words — which is
        precisely how the failure was found, shape 128 measuring exactly 1.000x
        with 0 of 524,288 words differing.

        If a future torch ever fuses these epilogues bit-exactly, this raises and
        it should: that would mean the fusion no longer re-bases the numerics, and
        the right response is to retire Gate 2' and go back to Gate 2, not to
        weaken this check.
        """
        identical = []
        for size in self.sizes:
            fused, _ = self._eager(self.tokens[:size])
            unfused, _ = self.eager_unfused(size)
            torch.cuda.synchronize(self.device)
            if torch.equal(fused.view(torch.uint16), unfused.view(torch.uint16)):
                identical.append(size)
        if identical:
            raise RuntimeError(
                f"shapes {identical} produced output bit-identical to the UNFUSED "
                f"module, so Inductor is not running there — dynamo hit its "
                f"recompile limit (now {self.recompile_limit}, ladder is "
                f"{list(self.sizes)}) and silently fell back to eager. Capturing "
                f"now would record a correct but unfused graph that no counter "
                f"would ever complain about again.")

    def _capture(self) -> None:
        """Warm EVERY shape to convergence, then capture every shape.

        The base class interleaves them — warm one size, capture it, warm the
        next. That is correct for eager ATen, where warmup only has to get cuBLAS
        handles and the caching allocator's first blocks out of the way, and it is
        not safe here: compiling shape N+1 while shape N's graph exists is fine,
        but the ordering leaves no single point at which "nothing more will be
        compiled" is true, which is the fact `assert_no_recompilation` needs to
        rest on. So the two phases are separated and the frame counter is sampled
        between them.
        """
        started = time.perf_counter()
        self.report.reserved_before = torch.cuda.memory_reserved(self.device)
        self._raise_recompile_limit()

        for size in self.sizes:
            self.warmup_rounds[size] = self._warm_to_convergence(size)
        self.frames_after_warmup = dynamo_frame_count()

        self.assert_every_shape_is_fused()

        for size in self.sizes:
            graph = torch.cuda.CUDAGraph()
            # No `pool=`, for the base class's reason: sharing a pool requires
            # replaying in capture order, which this dispatcher does not do.
            with torch.cuda.graph(graph):
                policy, value = self._eager(self.tokens[:size])
            self._graphs[size] = graph
            self._policy[size] = policy
            self._value[size] = value
            self.assert_no_recompilation(f"while capturing shape {size}")

        torch.cuda.synchronize(self.device)
        self.report.reserved_after = torch.cuda.memory_reserved(self.device)
        self.report.seconds = time.perf_counter() - started

    def eager(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The un-captured COMPILED forward — i.e. the capture term, not the fusion term.

        Overridden only to say so. `GraphedForward.eager` means "the same maths
        without the graph", and that is what this still is: `self.model` is the
        compiled module, so this measures whether CAPTURE changed anything, which
        is the comparison `tests/test_c12b_gate2prime.py` wants and the one that
        must come out at zero. The fusion term — Inductor against unfused ATen —
        is a comparison against the frozen baseline, and it is large by design.
        """
        return super().eager(count)

    def eager_unfused(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The ORIGINAL module's forward, bypassing Inductor entirely.

        The other half of the pair above, and the one that makes the
        re-baselining measurable from inside a single process: same input rows,
        same shape, same autocast context, unfused ATen. `tools/bench_c12b.py`
        reports the word-difference count and max |dlogit| from exactly this.
        """
        with torch.no_grad():
            with torch.amp.autocast_mode.autocast(device_type="cuda",
                                                  dtype=AUTOCAST_DTYPE, enabled=True):
                return self.eager_model(self.tokens[:count].to(torch.int64))


METHODS = {"cudagraph": GraphedForward, "compile": CompiledForward,
           "inductor": InductorGraphedForward}


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
    "INDUCTOR_WARMUP_ITERS",
    "InductorGraphedForward",
    "METHODS",
    "PAD_FEN",
    "build",
    "configure_inductor",
    "dynamo_frame_count",
    "resolve_sizes",
]
