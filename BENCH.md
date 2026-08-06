# BENCH

Accumulated performance numbers, one section per chunk.

Machine for all numbers below: **Intel Core i5-12600K** (10 cores / 16 threads),
Windows 11 Pro 26200. Linux figures are from WSL2 Ubuntu 24.04 on the same host.

Reproduce with `python tools/bench_c0.py`, `python tools/bench_c0b.py` and
`python tools/bench_c2.py` (see `README_BUILD.md`).

---

## C0 — GIL round-trip latency

What is measured: one iteration of the dispatcher loop in scope 2.1 — release
the GIL, write a row of `int32[68]` into a C++-owned buffer, **acquire the GIL,
call a Python callback, return**. The bolded segment is what the table reports.

20,000 iterations per batch size. The harness asserts the callback ran exactly
20,000 times, so these are not timings of a loop that skipped the Python call.

> **These tables were re-measured in C0b.** C0 originally passed *no data* to the
> callback, and reported a flat 0.1 µs at every batch size. The callback now
> receives the live zero-copy `rows x 68` view and sums it, which is why the
> numbers below are ~30x larger and scale with `rows`. The original C0 table is
> not reproduced: it measured a dispatcher that transfers nothing.

### Windows / MSVC 19.51, Release, Python 3.13.7

| rows | median (µs) | p99 (µs) | mean (µs) | max (µs) | p99 (ms) |
|-----:|------------:|---------:|----------:|---------:|---------:|
| 32 | 2.600 | 3.200 | 2.610 | 28.000 | 0.0032 |
| 64 | 3.900 | 6.000 | 3.989 | 187.400 | 0.0060 |
| 128 | 6.500 | 8.500 | 6.645 | 169.000 | 0.0085 |
| 256 | 9.600 | 15.000 | 9.792 | 143.500 | 0.0150 |

### Linux / Clang 18.1.3, Release, Python 3.12.3 (WSL2)

| rows | median (µs) | p99 (µs) | mean (µs) | max (µs) | p99 (ms) |
|-----:|------------:|---------:|----------:|---------:|---------:|
| 32 | 1.373 | 2.591 | 1.511 | 231.370 | 0.0026 |
| 64 | 1.823 | 3.533 | 1.997 | 169.834 | 0.0035 |
| 128 | 2.671 | 4.830 | 2.848 | 68.612 | 0.0048 |
| 256 | 4.335 | 8.689 | 4.588 | 168.766 | 0.0087 |

### Gate verdict

> C0 brief: *"If p99 round-trip exceeds ~2 ms at batch 256, the dispatcher design
> in scope 2.1 needs revisiting before C5, not after C10."*

**PASS.** Worst p99 at batch 256 is 0.0150 ms (Windows) / 0.0087 ms (Linux)
against a 2 ms budget — a margin of ~130x. Note this is 130x, not the ~10,000x
the pre-C0b table claimed; the difference is the cost of actually moving data.

Cost now scales roughly linearly with `rows`, as it should: the added work is
`arr.sum()` over `rows x 68` int32, which is O(rows). The GIL mechanism itself
still contributes only ~0.1 µs of that.

### Caveat — this benchmark is uncontended

The GIL is uncontended here: single thread, nothing competing. C0b measures what
happens when something is. **Read the C0b verdict before using these numbers to
size anything.**

---

## C0b — Contended GIL acquisition

What is measured: the same dispatcher loop, but split into three separately
timed phases, with a second Python thread competing for the interpreter.

```
  t0 -> t1   acquire_wait_us   GIL requested, not yet held
  t1 -> t2   call_us           GIL held, callback(view) runs
  t2 -> t3   release_us        GIL handed back at scope exit
```

`clock::now()` needs no GIL, so `t0` and `t3` are sampled from outside the
acquire scope without perturbing what they measure. A test asserts the three
phases sum to no more than the measured wall time, so `acquire_wait_us` cannot
be quietly measuring something else.

### The three configurations

| | background thread | `sys.setswitchinterval` |
|---|---|---|
| **A** | none (control) | 0.005 (default) |
| **B** | UCI `info` formatter | 0.005 (default) |
| **C** | UCI `info` formatter | 0.0005 (mitigation) |

The background thread is **pure Python**: it formats a full `info depth ... pv
<24 moves>` line with f-strings and `" ".join`, continuously. A C extension
would hold the GIL inside one long call and never yield at bytecode boundaries,
which would make `setswitchinterval` irrelevant and the experiment meaningless.

The thread is verified to be running before anything is timed, and its
throughput was measured against its solo rate: it runs at **90–97% of solo
throughput** in every configuration, i.e. it really is holding the GIL almost
all the time. Config C is not fast because the competitor is starved.

### Why there is a GIL-free gap, and why it is not zero

`contention_bench` takes a `work_us` argument: the GIL-free interval between one
callback and the next, standing in for the time the real dispatcher waits for
search threads to fill the next batch. All tables below use **200 µs**.

This is load-bearing. At `work_us = 0` the loop re-requests the GIL within
~100 ns of releasing it and usually wins the re-acquire before the competing
thread is even scheduled — so ~90% of iterations never contend, and config B's
*median* looks identical to config A's while the cost hides in the tail. The
`B-nogap` row below is that regime: p50 1.2 µs, p99 15.8 ms. **This is the trap
C0's benchmark fell into**, and it understates the median by four orders of
magnitude.

The size of the effect depends on how fast the build is — under ASan the
instrumented row write is itself a gap of several microseconds and the no-gap
p50 stays in the milliseconds — so the test asserts only the direction (a real
gap is never *less* adversarial) and reports the magnitude here.

### Configurations A / B / C — `acquire_wait_us`

Gap 200 µs. Iteration counts differ per config because the configs differ in
cost by four orders of magnitude; `iters` is stated because **`max` is not a
scale-free statistic** — the max of 8,000 samples is not comparable to the max
of 800.

#### Windows / MSVC 19.51, Release, Python 3.13.7 — *authoritative*

| config | rows | iters | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| A | 32 | 6000 | 0.100 | 0.800 | 2.900 | 29.200 |
| A | 256 | 6000 | 0.100 | 0.700 | 2.500 | 30.700 |
| B | 32 | 800 | 15229.600 | 15858.500 | 16228.400 | 19162.600 |
| B | 256 | 800 | 15249.500 | 15882.900 | 16096.200 | 16250.500 |
| B-nogap | 256 | 800 | 1.200 | 15164.400 | 15792.600 | 16129.600 |
| C | 32 | 8000 | 3.300 | 6.300 | 61.800 | 673.400 |
| C | 256 | 8000 | 3.300 | 8.400 | **78.000** | **316.800** |

#### Linux / Clang 18.1.3, Release, Python 3.12.3 (WSL2) — *sanity check only*

| config | rows | iters | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| A | 32 | 6000 | 0.136 | 0.674 | 3.961 | 113.577 |
| A | 256 | 6000 | 0.156 | 0.627 | 4.058 | 119.264 |
| B | 32 | 800 | 5080.206 | 5159.324 | 5223.257 | 5294.719 |
| B | 256 | 800 | 5079.091 | 5143.863 | 5215.707 | 5317.141 |
| B-nogap | 256 | 800 | 0.172 | 1.280 | 5095.647 | 5183.424 |
| C | 32 | 8000 | 547.547 | 604.410 | 667.515 | 1481.528 |
| C | 256 | 8000 | 548.537 | 607.953 | **668.384** | **2597.713** |

### `call_us` and `release_us`

| | | Windows | | Linux | |
|---|---|---|---|---|---|
| config | rows | call p50 | call p99 | call p50 | call p99 |
| A | 32 | 2.900 | 35.600 | 2.413 | 36.179 |
| A | 256 | 9.900 | 49.200 | 5.730 | 44.943 |
| B | 32 | 71.100 | 15745.900 | 42.023 | 160.072 |
| B | 256 | 15300.200 | 16626.200 | 47.126 | 5199.052 |
| C | 32 | 6.600 | 52.400 | 14.169 | 575.205 |
| C | 256 | 13.900 | 87.500 | 29.248 | 638.936 |

`release_us` is negligible everywhere: p50 ≤ 0.5 µs, p99 ≤ 34 µs in the worst
configuration. Releasing the GIL is not a cost worth designing around.

`call_us` in config B is inflated because **NumPy releases the GIL inside
`arr.sum()`** and must re-acquire it, so a handoff wait lands inside the
callback rather than before it. That is realistic — a real evaluator calling
PyTorch will do the same — but it means `call_us` under contention is not a
measure of callback compute.

### Why the mitigation works, and how differently on each platform

Sweeping `sys.setswitchinterval` at batch 256 (`tools/bench_c0b.py --sweep`)
shows two completely different mechanisms:

| interval (s) | → ms | **Windows** p50 | p99 | **Linux** p50 | p99 |
|---|---:|---:|---:|---:|---:|
| 0.0001 | 0 | 3.2 | 28.4 | 155.8 | 347.1 |
| 0.0005 | 0 | 3.2 | 19.9 | 545.4 | 760.4 |
| 0.0009 | 0 | 3.3 | 24.8 | 929.7 | 1150.5 |
| 0.0011 | 1 | 15319.7 | 16123.1 | 1122.5 | 1332.1 |
| 0.0015 | 1 | 15277.0 | 16215.8 | 1567.8 | 1689.8 |
| 0.002 | 2 | 15304.9 | 16054.5 | 2075.5 | 2197.8 |
| 0.003 | 3 | 15305.8 | 16024.3 | 3070.4 | 3215.6 |
| 0.005 | 5 | 15316.3 | 16074.3 | 5085.8 | 5184.7 |
| 0.010 | 10 | 15286.1 | 15958.4 | 10084.8 | 10162.5 |

**Windows is a step function with a cliff at exactly 1 ms.** CPython's Windows
condition variable is `SleepConditionVariableSRW`, which takes a `DWORD` of
milliseconds, computed as `microseconds / 1000` by integer division. Any switch
interval below 1 ms truncates to a **0 ms** timeout: the waiting thread times out
instantly, sets `gil_drop_request` immediately, and the holder drops at its next
eval-breaker check ~3 µs later. At 1 ms and above, the wait quantises to the
Windows system timer tick (~15.6 ms) — which is why p50 is ~15.3 ms for *every*
interval from 1.1 ms to 10 ms, and why p95 is ~15.9 ms throughout.

**Linux is a straight line.** `pthread_cond_timedwait` takes a nanosecond
deadline, so p50 tracks the requested interval almost exactly (156 µs → 10.1 ms)
with no cliff and no quantisation.

The practical consequence is that the *same* `setswitchinterval(0.0005)` call
buys a 190x improvement on Windows and a 9x improvement on Linux, and the
Windows margin comes from an implementation artifact rather than from the number
being small. See the C10 note below.

### Gate stability

`max` is one sample out of thousands, so a single run says little about it.
10 runs × 2,000 iterations = 20,000 samples per configuration
(`tools/bench_c0b.py --repeat 10 --scale 1`):

| build | p99 range (µs) | max median (µs) | worst max (µs) | runs p99 ≥ 1 ms | runs max ≥ 2 ms |
|---|---|---|---|---|---|
| Windows Release *(authoritative)* | 20.3 – 63.8 | 242.6 | 2270.9 | 0/10 | 1/10 |
| Windows ASan | 26.4 – 62.7 | 244.7 | 2035.6 | 0/10 | 1/10 |
| WSL2 Release | 644.9 – 691.0 | 824.1 | 1987.2 | 0/10 | 0/10 |
| WSL2 ASan | 684.6 – 733.4 | 1228.3 | 3313.6 | 0/10 | 3/10 |

A further 3 × 20,000 samples on Windows Release gave max-of-max 305.0, 205.2 and
217.9 µs with 0/30 runs over 2 ms, so the single 2270.9 µs excursion above is
external machine activity rather than a property of the dispatcher: **1 run in
40 on Windows Release**. Config A — which has *no contention at all* — shows the
same class of tail at lower amplitude (max up to 235 µs), confirming the
baseline is the OS descheduling the measuring thread.

**p99 never came within 25% of its gate in any of those 80,000 samples.**

### Gate verdict

> C0b brief: *"In Configuration C, the p99 `acquire_wait_us` must be < 1 ms, and
> the max must be < 2 ms at batch 256."*
> *"Treat Windows numbers as authoritative. WSL2 is only a sanity check."*

**PASS on the authoritative platform.**

| criterion | Windows Release | budget | margin |
|---|---|---|---|
| p99 @ 256 | 78.0 µs | 1000 µs | **12.8x** |
| max @ 256 | 316.8 µs | 2000 µs | **6.3x** |

On WSL2 the p99 criterion passes (668.4 µs, 1.5x margin) and the max criterion
fails (2597.7 µs at 8,000 samples). Both WSL2 results are consistent with the
brief's own warning that WSL2 tails are not representative; the failure is a
scheduling excursion, not GIL behaviour, and p99 — the statistic that is stable
under resampling — passes on every build tested.

### Interpretation — the answer the chunk was run to get

The C0b brief asks for one of three explicit conclusions.

**B is bad and C is good, so `sys.setswitchinterval` is sufficient.** The
experiment is valid on its own terms: config B degrades against config A by
**152,000x** at the median on Windows (0.1 µs → 15.25 ms) and **33,000x** on
Linux (0.156 µs → 5.08 ms), so the "A, B and C indistinguishable → FAIL" branch
does not apply. Config C then recovers essentially all of it.

Concretely, with the default switch interval a dispatcher that needs the GIL
while UCI output is being formatted in Python waits a **median of 15 ms on
Windows and 5 ms on Linux**, with a p99 of 16 ms and 5.2 ms. At 256-position
batches that is a hard ceiling of roughly 60 batches/second regardless of how
fast the evaluator is. It would have been misdiagnosed as a slow neural net.

**Recommendation for C10: call `sys.setswitchinterval(0.0005)` at engine
startup, before any search thread starts.** Three caveats to carry forward:

1. **The Windows margin is an artifact, not a design property.** It comes from a
   millisecond truncation in CPython's Windows condvar shim. If CPython ever
   moves to a higher-resolution wait, Windows would regress to Linux-like
   behaviour — p50 ~500 µs, p99 ~760 µs. That still passes the p99 gate, but the
   margin goes from 12.8x to 1.3x. Do not spend the current headroom.

2. **Linux has no cliff, so the interval is a real dial there.** `0.0005` is
   about the largest value that keeps p99 under 1 ms on Linux (`0.0009` already
   gives p99 1150 µs). If Linux ever becomes a production target, either drop to
   `0.0001` (p99 347 µs) or do (3).

3. **C++ stdout emission remains the durable fix and should stay on the table
   for C10.** It is not *mandated* by this data — the gate passes — but it is the
   only option that removes the dependence on interpreter internals entirely.
   The cost of deferring it is now quantified rather than guessed.

Also worth carrying into C10: a shorter switch interval increases context-switch
frequency interpreter-wide. This benchmark did not measure that overhead, and it
should be checked against real search throughput before shipping.

### Clock resolution

Every microsecond figure above depends on this, so it is measured rather than
assumed (`guofish_core.clock_info()` reports the smallest non-zero delta between
200,000 back-to-back `steady_clock::now()` calls):

| platform | nominal tick | measured tick | back-to-back reads that were identical |
|---|---|---|---|
| Windows | 1 ns | **100 ns** | 83% |
| Linux | 1 ns | **15–16 ns** | 0% |

`steady_clock::period` claims 1 ns on both, which is false on Windows: MSVC backs
it with QPC at 10 MHz. The 83% figure means most consecutive reads land in the
same tick — the clock is saturated at that granularity.

This matters for reading the tables: **Windows config A and config C p50 values
(0.1 µs and 3.3 µs) are 1 and 33 ticks respectively.** The 0.1 µs is at the
resolution floor and the true value is somewhere at or below 100 ns; the 3.3 µs
is comfortably resolved. Every number the gate depends on is ≥ 78 µs, i.e. ≥ 780
ticks, so the gate verdict is not resolution-limited. Linux resolves everything
with room to spare.

---

## C2 — Tokenization throughput

What is measured: FEN string in, 68 `int32` tokens out — the encoding in
`cpp/tokens.hpp`, which is the C++ replica of `core.mctsv4.board_to_tokens`.

Corpus: all 100,000 FENs from `golden/tokens.npz`, best of 5 passes. Reproduce
with `python tools/bench_c2.py [--python]`.

Three rows, and they answer different questions:

| row | what it includes | who pays it |
|---|---|---|
| `encoder` | the encoder only — FENs are copied into C++ strings before the timer starts and the timed region touches no Python object | the C5 search, where the positions are never Python objects to begin with |
| `fill()` | `TokenBatch.fill(fens)` end to end: materialising the iterable, borrowing each `str`'s UTF-8 buffer, then encoding | a caller still driving the search from Python |
| `python` | the reference, `board_to_tokens(chess.Board(fen))` | today's engine |

### Windows / MSVC 19.51, Release, Python 3.13.7

| path | positions/s | ns/position |
|---|---:|---:|
| encoder (`tokenize_bench`) | **3,678,645** | 271.8 |
| batch (`TokenBatch.fill`) | 3,431,179 | 291.4 |
| python (`board_to_tokens`) | 8,841 | 113,109.0 |

### Linux / Clang 18.1.3, Release, Python 3.12.3 (WSL2)

| path | positions/s | ns/position |
|---|---:|---:|
| encoder (`tokenize_bench`) | **4,227,148** | 236.6 |
| batch (`TokenBatch.fill`) | 4,127,002 | 242.3 |

The Python reference was not re-measured under WSL2; the ratio below uses the
Windows figure, which is the production platform.

### Gate verdict

The C2 brief sets the target at **≥ 100x** the Python reference's ~10,200 pos/s,
i.e. ≥ 1,020,000 pos/s.

| baseline | encoder speedup | `fill()` speedup |
|---|---:|---:|
| 10,200 pos/s (quoted in the brief) | **360.7x** | 336.4x |
| 8,841 pos/s (measured here, Windows) | **416.1x** | 388.1x |

**PASS**, by 3.6x over the requirement against the brief's own baseline.

Both baselines are given because they disagree, and the direction matters: this
machine measures the reference *slower* than the brief quotes, which inflates
the speedup. The 360.7x figure — the conservative one — is the number to carry
forward.

### Why the two C++ rows differ, and why the gap will close

`fill()` runs 7% slower than the encoder on Windows and 2% slower on Linux. The
difference is not tokenization; it is `PySequence_List` plus one
`PyUnicode_AsUTF8AndSize` per element, paid once per batch at the language
boundary. Two things follow:

* It is a **fixed cost per FEN crossing the boundary**, not per token, so it does
  not grow with the encoding.
* By C5 it disappears entirely. The search's leaves are `chess::Board` objects in
  C++; nothing round-trips through a Python `str`. The `encoder` row is the
  figure that predicts the shipped engine, and `fill()` is what the transitional
  Python-driven path sees.

### What this buys the dispatcher

At 3.7M pos/s a batch of 256 positions is tokenized in **70 µs**. Read that next
to C0b: the *contended* GIL acquire wait alone is 78 µs p99 on Windows at the
recommended switch interval. Tokenization is therefore not a term in the
dispatcher's budget — it is already below the noise floor of the boundary
crossing that surrounds it, and no further optimisation of it is worth spending.

The relevant consequence for C5 is the reverse of the usual one: **the encoder is
fast enough that it does not need to be cached.** The Python engine memoises
token arrays per node; the C++ search can re-encode from the board every time it
needs a batch row, which removes a cache, its invalidation, and the en-passant
key collision class recorded against the Python side.

### Measurement notes

* Benchmarked on the **Release** build. `tools/bench_c2.py` prints `asan=` in its
  header and warns loudly on a sanitizer build (README_BUILD.md, Benchmarks).
* The harness asserts the CLS column is 40 across the filled buffer afterwards, so
  a `fill()` that silently encoded nothing cannot post a good number.
* `tokenize_bench` sums the whole scratch buffer inside the timed scope's tail and
  returns the checksum, so the encoder cannot be optimised away as dead stores.
* Each position is written to its own row rather than reusing one, so the working
  set is 100,000 x 272 B rather than a single cache line.
