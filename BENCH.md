# BENCH

Accumulated performance numbers, one section per chunk.

Machine for all numbers below: **Intel Core i5-12600K** (10 cores / 16 threads),
Windows 11 Pro 26200. Linux figures are from WSL2 Ubuntu 24.04 on the same host.

Reproduce with `python tools/bench_c0.py`, `python tools/bench_c0b.py`,
`python tools/bench_c2.py`, `python tools/bench_c4.py` and `python tools/bench_c8.py`
(see `README_BUILD.md`).

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

---

## C4 — Sibling scan, Q32 vs double accumulator

**This table is a design decision, not a measurement.** `value_sum` is a Q32
fixed-point `atomic<int64>` in production, so every read of it during PUCT
selection has to be converted back to a double. That conversion sits in the
hottest loop in the engine — the scan of all ~32 siblings, once per node on
every simulation's path. The chunk brief requires it decided here rather than in
C12, because the fallback (a separate `float` array of Q values maintained on
backup) is a second array to keep coherent, and retrofitting it after C9 is
expensive.

Reproduce with `python tools/bench_c4.py --trials 5 --sweep`.

### What the loop does

Reads `visit_count`, `value_sum` and `prior` for a contiguous block of 32
siblings and reduces them to an argmax of `value_sum / visits + prior`. Both
accumulators are filled with identical logical contents by the same LCG, both
are warmed with an untimed pass, and both are scanned in the same call.

It is deliberately **not** the PUCT formula — no cpuct, no `sqrt(parent visits)`,
no virtual loss — because that arithmetic is out of C4's scope. The consequence
is conservative in the right direction: the real loop does strictly more work per
child, so the conversion's *share* of it is smaller than this table shows.

Hot working set is 16 B/node (`atomic<int32>` + `atomic<int64>` + `float`), and
it is **identical for both accumulators** — `atomic<double>` is also 8 bytes — so
the two columns differ only in ALU work, never in memory traffic.

### Windows / MSVC 19.51, Release, Python 3.13.7

| working set | nodes | hot set | Q32 ns/scan | double ns/scan | Q32 ns/child | double ns/child | Q32 / double |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 - one block | 32 | 512 B | 40.20 | 36.82 | 1.256 | 1.151 | 1.092x |
| L1 - 2,048 nodes | 2,048 | 32 KB | 40.49 | 37.45 | 1.265 | 1.170 | 1.081x |
| L2/L3 - 131k nodes | 131,072 | 2 MB | 84.93 | 80.44 | 2.654 | 2.514 | 1.056x |
| RAM - 2.1M nodes | 2,097,152 | 32 MB | 89.35 | 85.22 | 2.792 | 2.663 | 1.048x |

### Linux / Clang 18.1.3, Release, Python 3.12.3 (WSL2)

| working set | nodes | hot set | Q32 ns/scan | double ns/scan | Q32 ns/child | double ns/child | Q32 / double |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 - one block | 32 | 512 B | 35.70 | 34.47 | 1.116 | 1.077 | 1.036x |
| L1 - 2,048 nodes | 2,048 | 32 KB | 35.48 | 34.70 | 1.109 | 1.084 | 1.022x |
| L2/L3 - 131k nodes | 131,072 | 2 MB | 36.19 | 35.05 | 1.131 | 1.095 | 1.032x |
| RAM - 2.1M nodes | 2,097,152 | 32 MB | 40.23 | 38.27 | 1.257 | 1.196 | 1.051x |

### Verdict — Q32 stays in the hot path

> C4 brief: *"if [the conversion] does not vanish into the loop, the fallback is
> a separate float array updated on backup. Decide this here, not in C12."*

**It effectively vanishes. Q32 stays; no separate Q array is built.**

The row that decides it is the last one: at the scope's 2-3M node budget the scan
is memory-bound and the conversion costs **4.8% on Windows and 5.1% on Linux** —
one `cvtsi2sd` and one `mulsd` per child, mostly hidden behind the load latency
the loop is already waiting on. In the L1-resident regime, where there is nothing
to hide behind, it is still only 9.2% / 3.6%.

Scale that against what it buys: a single `lock xadd` instead of a CAS retry loop
whose retry rate climbs with thread count, on an accumulator every backup
touches; and **associativity**, which makes multithreaded search bit-reproducible
— a property the Python engine never had and which the original brief wrote off
as permanently unreachable. A ~5% cost on one loop is not close to the price of
giving that up.

The rejected alternative is worth stating precisely, because it looks free: a
parallel `float q[]` array updated on backup would remove the conversion but add
a fourth stream to the scan (+4 B/node hot set, ~25% more memory traffic on the
row that is already memory-bound), a second value that can disagree with
`value_sum`, and a write on the backup path that has no atomic story. It is
slower *and* less safe on the configuration that matters.

### Where the two toolchains disagree, and why it does not change the answer

MSVC's memory-bound row is 89 ns/scan against Clang's 40 ns — a 2.2x codegen gap
that has nothing to do with the accumulator, since both columns move together.
It is recorded because C12 will want to know that the scan has headroom on
Windows specifically, and because a reader comparing the two tables would
otherwise assume one of them is wrong. The *ratio*, which is what this section
decides, agrees to within 4 points on every row.

### Q32 conversion accuracy

Not a benchmark, but it is the other half of the same decision — the conversion
is only affordable if it is also exact enough to be uninteresting.

Exhaustive sweep of **every IEEE-754 float bit pattern in [-1, 1], both signs**:
2,130,706,434 values, 9.7 s on MSVC Release, 7.0 s on Clang Release.

| property | result |
|---|---|
| max absolute round-trip error | 1.1641532e-10 = exactly 2^-33, half the Q32 tick |
| largest value that is *not* bit-exact | 0.001953124884, i.e. the float just below 2^-9 |
| Q32 -> double -> Q32 mismatches | 0 |
| asymmetric roundings (`to_q32(-v) != -to_q32(v)`) | 0 |

Every float at or above 2^-9 round-trips **bit-identically**, because at that
magnitude a float's ulp is already 2^-32 or coarser. Below it the quantizer
rounds, and never by more than half a tick. Against a network that emits bf16
(8 mantissa bits) this is seven orders of magnitude finer than its own input.

Overflow needs 2^31 = 2.1e9 visits at |v| = 1, against a 15k-sim budget.

### Measurement notes

* Benchmarked on **Release** builds. `tools/bench_c4.py` prints the compiler,
  sanitizer and assert status in its header and warns loudly on a non-production
  build.
* `sibling_scan_bench` accumulates each block's argmax index and returns the
  total as `checksum`; the harness rejects a zero checksum, so a scan whose reads
  were optimised away cannot post a good number. The tool additionally requires
  the checksum to be identical across trials.
* Each configuration is run 5 times and the **median** reported.
* One untimed warm pass per arena before timing, so the comparison is not a
  measurement of which allocation happened to be paged in first.
* Both arenas are filled by the same deterministic LCG (not `<random>`, whose
  distributions are not pinned across implementations), so the two columns scan
  identical logical trees.

---

## C5 — Search throughput on the replay evaluator

**Read this as a headroom check, not as a Gate 4 projection.** What is measured
is the traverse loop and only the traverse loop: PUCT selection over ~35 siblings
per descent step, virtual-loss apply and repay, tokenization and `nn_key` at
every leaf, a hash lookup, expansion of ~35 children, and backup to the root.
Everything Gate 1 compares, and nothing else.

Two things that dominate the real engine are absent, and quoting this number as
a throughput projection would be flattering in exactly the wrong direction:

* **No network.** The replay evaluator is an `unordered_map` lookup. In
  production a leaf costs a batched GPU forward — 4.86 ms at batch 64, ~13.2k
  evals/s — which is the actual ceiling. This measures the CPU spent *around*
  that ceiling.
* **No cache, no tree reuse, no concurrency.** C7, C8 and C9.

Reproduce with `python tools/bench_c5.py --trials 5 --markdown`.

### Windows / MSVC 19.51, Release, Python 3.13.7 (i5-12600K)

20 quiet positions x 4,999 simulations each = 99,980 delivered simulations per
trial, median of 5 trials, single-threaded.

| virtual loss | positions | sims | median s | sims/s | us/sim | vs Python CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 20 | 99,980 | 0.562 | 177,761 | 5.63 | 42x |
| 2.5 | 20 | 99,980 | 0.644 | 155,321 | 6.44 | 37x |

Dump load (108,966 entries / 3,989,554 moves, from NumPy arrays into two hash
tables and two vectors): **0.01–0.02 s**.

### What the ratio means

Scope 2.2 sizes C9's worker count on the claim that *CPU descent capacity is not
the constraint*, and it projected "a conservative 10x for C++" against the
Python engine's measured 236 us/sim of real CPU work (479 ms per 2,030
simulations, recon). **The measured factor is 37–42x**, so the projection holds
with a wide margin:

* At 5.63 us/sim, one thread delivers ~178k sims/s against the ~16.5k sims/s
  that batch-64 GPU throughput supports at a ~20% cache hit rate. That is ~11x
  headroom on a single thread, before any of C9's eight.
* The C9 default of **W = 8, K = 8** therefore stays a *selection-quality*
  decision rather than a throughput one, which is what scope 2.2 argued and this
  is the first measurement that supports it.

The VL 2.5 column costs **14% more per simulation** than VL 0.0. That is not
overhead in the virtual-loss bookkeeping — the apply/repay path is identical, an
integer increment either way — it is the search doing different work: a nonzero
virtual loss widens selection, so descents run marginally longer and touch more
distinct nodes. Worth recording because C9 will re-measure it under contention,
where the same 14% could easily be mistaken for a synchronisation cost.

### Measurement notes

* Benchmarked on a **Release** build; `tools/bench_c5.py` prints the compiler,
  sanitizer and assert status in its header. The ASan build runs the same suite
  roughly 7x slower (256 s vs 34 s for the full test run) and is not what these
  numbers come from.
* The timed region excludes `set_position` and `load_dump`, and excludes the
  NumPy comparison the test suite does afterwards. It is `search()` only.
* One untimed warm pass before timing, so the first position is not paying for
  cold arena pages and a cold hash table.
* `simulations` as reported by `search()` is used as the denominator, not the
  requested budget — the reference seeds the root with one visit, so a 5,000-sim
  search delivers 4,999. This is the "report delivered, not nominal" discipline
  scope 3 requires of `playv6.py`, applied to the bench that establishes the
  baseline.
* The arena is sized from the golden trees (218,821 nodes for the largest run)
  and recycled between positions by `set_position`, so no run is timed against a
  fresh 2M-node allocation.

---

## C6 — What terminal handling costs

**The C5 numbers above were measured before terminal handling existed, and they
are no longer what this build does.** C6 added work to every descent step and to
every leaf, and the honest way to price it is to re-run the same corpus on the
same machine under the new code. That is the `quiet` row below; the C5 table is
left in place as the before.

Reproduce with `python tools/bench_c6.py --trials 9 --markdown`.

### Windows / MSVC 19.51, Release, Python 3.13.7 (i5-12600K)

| corpus | virtual loss | positions | sims | median s | sims/s | us/sim | vs Python CPU | sims ending in a claimable draw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| quiet | 0.0 | 20 | 99,980 | 0.825 | 121,256 | 8.25 | 29x | 0% |
| quiet | 2.5 | 20 | 99,980 | 0.862 | 115,977 | 8.62 | 27x | 0% |
| terminal | 0.0 | 25 | 97,981 | 0.346 | 283,510 | 3.53 | 67x | 18% |
| terminal | 2.5 | 25 | 97,981 | 0.344 | 285,226 | 3.51 | 67x | 18% |

Run-to-run spread over five separate invocations was **7.6–8.3 us/sim** on the
quiet corpus and **3.1–3.8** on the terminal one, i.e. roughly ±5%. Quote the
band, not the third decimal.

### The tax on a quiet position: 5.63 -> ~8 us/sim, about 40%

None of C6's machinery *fires* on a quiet position — no draw is ever claimed and
no terminal is ever found, which is exactly what the C5 corpus was selected for —
so the whole of that 40% is work done to establish that nothing happened:

* a `ParsedFen` rebuild per descent step (64 squares plus nine bitboards),
* FNV-1a over ~90 bytes for `rep_key`, per descent step,
* a linear scan of the path's repetition tally, per descent step,
* a halfmove-clock update and two move classifiers, per descent step,
* at every leaf, `inCheck()` plus the legal-move count fed to `outcome_of`.

The `ParsedFen` rebuild is the expensive one and it is already shared three ways:
the descent threads a single one through, so it serves the move classifiers for
the next step, `rep_key` for this one, and `nn_key` at the leaf. Rebuilding per
consumer would have been three per step instead of one.

**This is the minimum the rules need at this stage, not the minimum possible.**
The repetition key is a function of the whole position, so *something* has to
walk the board once per step. What would remove most of the cost is making it
incremental — updating a key from the move rather than rebuilding from the
board — and that is a change with its own correctness surface (C3's entire
argument for FNV over a serialisation rather than Zobrist is that an incremental
update is where the en-passant rules silently diverge). It is not a C6 change.

### Headroom is unaffected, which is the number that matters

Scope 2.2 sizes C9's worker count on the claim that CPU descent capacity is not
the constraint, projecting "a conservative 10x for C++" against the Python
engine's 236 us/sim of real CPU work. At 8.25 us/sim the measured factor is
**29x**, down from 37–42x and still nearly 3x the projection. One thread delivers
~121k sims/s against the ~16.5k sims/s that batch-64 GPU throughput supports at a
20% cache hit rate — **~7x headroom on a single thread**, before any of C9's
eight. The C9 default of W = 8, K = 8 stays a selection-quality decision rather
than a throughput one.

### The terminal corpus is FASTER, and that is not a paradox

3.5 us/sim against the quiet corpus's 8.25. A simulation that ends in a claimable
draw at ply four is *cheaper* than a normal one, not dearer: it never reaches the
leaf, so it never tokenizes, never computes an `nn_key`, never looks anything up
in the dump, and never expands ~35 children. 18% of the terminal corpus's
simulations end that way, and on the two blocked-pawn specs it is over 90%.

So this number is not a cost measurement and must not be read as one. What it is
good for is the opposite of a headroom check: it is the number that would
collapse if the per-path repetition tally were accidentally quadratic in the path
length, or if the fivefold walk-back ran on every leaf instead of behind its
occupancy pre-check. It is a regression tripwire on the two pieces of C6 whose
cost is not obviously bounded.

Note also that the VL 0.0 / VL 2.5 gap, a consistent 14% in C5, has closed to
nothing on the terminal corpus and to ~4% on the quiet one. The C5 reading of
that gap was that virtual loss widens selection so descents run longer and touch
more distinct nodes; the per-step cost has roughly doubled, so the same extra
steps are now a smaller share of a bigger total. C9 will re-measure it under
contention, where a residual gap could otherwise be mistaken for a
synchronisation cost.

### Measurement notes

Everything in the C5 section's measurement notes applies unchanged. Two
additions:

* The terminal corpus runs four positions at a **lowered `max_tree_depth`** (4 to
  7 — see DECISIONS.md, C6), so `tools/bench_c6.py` keeps one engine per
  (virtual loss, cap) pair rather than one per virtual loss. Those four runs are
  at 2,000 simulations rather than 5,000, which is why the terminal corpus
  delivers 97,981 simulations from 25 positions where the quiet one delivers
  99,980 from 20.
* `set_position` is passed each position's recorded repetition history, as the
  test suite does. Building it is outside the timed region; consulting it is
  inside, and it is one `unordered_map` lookup per descent step.

---

## C8 — Tree reuse: arena high-water, and what a compaction costs

Two questions. The acceptance criterion asks only the first.

Reproduce with `python tools/bench_c8.py` against a **Release** build.

### Windows / MSVC 19.51, Release, Python 3.13.7 (i5-12600K)

Arena capacity 524,288 nodes **per arena**, ping-pong pair, cache on (2^20
slots). "apply" is one `apply_move`: the compacting copy, the structural diff,
the swap and the repetition-history rebuild.

| game | plies | sims/move | peak nodes | nodes/sim | kept % | reuse x | apply p50 | apply max | p50 no diff | max no diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gate1-20 | 20 | 5000 | 189,236 | 37.8 | 76.5 | 3.71 | 12.700 | 20.202 | 4.587 | 10.380 |
| quiet-80 | 80 | 2000 | 71,845 | 35.9 | 69.7 | 3.38 | 2.879 | 6.821 | 1.068 | 4.221 |
| quiet-vl | 40 | 2000 | 89,353 | 44.7 | 69.4 | 3.17 | 5.352 | 12.060 | 1.800 | 6.188 |
| threefold-walk | 30 | 2000 | 34,857 | 17.4 | 50.9 | 1.98 | 0.875 | 4.442 | 0.420 | 4.599 |
| fifty-walk | 20 | 2000 | 23,022 | 11.5 | 11.9 | 1.26 | 0.207 | 4.507 | 0.183 | 5.066 |

`nodes/sim` divides the peak by the **move budget**, not by the simulations
actually delivered. With reuse most of a move's budget is already in the tree —
the `reuse x` column is scope §3's tree-reuse factor, and at 3.2–3.7x it is
above the 2.3–2.9x Python measured — so dividing by the ~600 new simulations
would report a rate three times too high for a tree of the same size.

`kept %` is the fraction of the arena the compaction copies. It is a property of
the position, not of the code: at ply 0 of `gate1-20` the best move already holds
97% of the tree, while in `fifty-walk` almost every simulation ends in a
fifty-move draw at ply one or two, so 88% of the arena is discarded every move.

### Criterion 3: the arena high-water over an 80-ply game

**71,845 nodes — 3.6% of the 2M budget floor.** The reserved footprint is
**39.0 MiB** for the ping-pong pair at capacity 524,288, and the standby arena is
allocated lazily, so a search that never applies a move reserves half of that.

`arena_high_water` is the peak of **either** arena. That matters: during a
compaction the source subtree and its copy are alive at the same instant, so the
moment of greatest occupancy is the moment the two overlap, and a counter that
tracked only the active arena would miss its own worst case.

The peak equals the largest single post-search tree in every game, which is the
shape the ping-pong is supposed to produce: the arena does not accumulate across
plies. `test_apply_move_actually_frees_the_dead_branches` asserts the same thing
from the other direction — over 80 plies the arena's occupancy is not monotone.

**MEASURED at 2,000 and 5,000 simulations per move. DERIVED at 15,000.** The
worst rate in the corpus is 44.7 peak nodes per simulation of move budget
(`quiet-vl`); at 15,000 sims/move that implies **~670,000 peak nodes**, inside
scope §2.3's 2–3M budget with 3x margin. That figure is an extrapolation and is
not asserted anywhere — the reference corpus this replays was generated at 2,000
and 5,000 sims because 15,000 x 80 plies is most of a day of Python. Scope
§2.3's own budget was derived the same way (~40 nodes/sim x 15k sims), and the
measured 35.9–44.7 nodes/sim brackets that 40 rather than contradicting it.

The one thing an extrapolation cannot see is whether the reuse factor holds at a
larger budget. It should get *better*, not worse — a bigger search concentrates
more of its visits under the eventual best move — but that is an argument, and
Phase 5 runs at the real budget and can replace it with a number.

### What the ping-pong costs, and where scope §2.3's estimate lands

Scope §2.3 prices the compaction at "~19 MB memcpy with pointer fixup at 600k
nodes — a few ms". **Measured, at 184,272 nodes copied (the largest compaction
in the corpus, `gate1-20`): 4.6 ms p50 for the copy alone, 12.7 ms with the
structural diff.**

Scaled linearly to the 600k nodes the estimate was written for, that is ~15 ms
for the copy and ~41 ms with the diff — so **the "few ms" estimate is optimistic
by roughly 3–5x**. It does not change the conclusion it was supporting: at
15,000 sims a move takes on the order of a second, and 15–41 ms is 1.5–4% of it,
paid once per move. But the estimate should not be quoted as measured, and the
number to plan Phase 5 around is this one.

**The structural diff costs more than the copy it checks** — +8.1 ms against
4.6 ms at 184k nodes. That is not surprising: the copy is nine sequential
streams into fresh memory, while the diff walks two arenas in DFS order
comparing ten fields per node, plus a visited bitmap. It is enabled here in
every acceptance run (`SearchConfig.verify_compaction`) and unconditionally in
any build with asserts. Whether to keep it on in production is a real choice and
this is the number to make it with; scope §7 lists it as the mitigation for the
fixup-bug risk, so the default for the shipped engine should be "on until Phase
5 says the move budget cannot afford it".

The `max` columns are dominated by the first compaction of each game, which pays
for the lazy allocation of the standby arena (~20 MB of value-initialised
storage at this capacity). `fifty-walk` shows it most clearly: p50 0.2 ms,
max 4.5 ms, on a game whose largest compaction is 2,600 nodes.

### Measurement notes

* Every row is one replay of the C8 corpus against `golden/c8_reuse_dump.npz`,
  i.e. the same deterministic work the acceptance suite does. The "no diff"
  columns are a second replay of the identical game with
  `verify_compaction=False`, so the diff's cost is a difference between two runs
  rather than an estimate.
* `apply_move` timing is measured from Python with `time.perf_counter()`, so it
  includes one pybind11 crossing (~1 µs, C0) — irrelevant at these magnitudes
  but not zero.
* The ASan build is roughly 8x slower here (the full suite runs in 15 min
  against 110 s) and its numbers do not belong in this table.

---

## C9 — The GPU knee, the W × K grid, and what the root distribution costs

Reproduce with `python tools/bench_c9_knee.py --markdown` (GPU, needs torch) and
`python tools/bench_c9.py --markdown` (CPU only, replay evaluator).

### C9a — Where the evaluation curve actually knees

The chunk brief required this measured before any W × K was locked, because two
prior measurements gave *opposite* advice. They are now both explained.

Sweep of batch ∈ {8…256} on the 10.9M v5 student, RTX 5070, torch 2.8.0+cu129,
bf16 autocast, real board tokens from the Pass B shards.
`torch.cuda.synchronize()` around **every** timed iteration.

Three paths, because they are three different claims: `forward` is
device-resident tokens and forward only; `reference` is
`core.mctsv4._evaluate_batch` verbatim (H2D int64 → forward → full 4096-wide
`.cpu()`); `gathered` is scope §2.1/§2.5's production path (pinned int32 H2D →
forward → 64-wide gather → narrow D2H).

**`gathered` — the curve C10 will run on:**

| batch | isolated ms | saturated ms | best ms | best pos/s | SM MHz |
|------:|------------:|-------------:|--------:|-----------:|-------:|
| 8 | 5.766 | 5.663 | 4.101 | 1,951 | 2955 |
| 16 | 4.927 | 5.469 | 3.763 | 4,252 | 2932 |
| 32 | 5.528 | 5.692 | 3.821 | 8,374 | 2932 |
| 64 | 5.019 | 5.484 | 4.009 | 15,965 | 2917 |
| 128 | 7.462 | 7.826 | 7.046 | 18,166 | 2895 |
| 256 | 14.320 | 14.525 | 13.946 | 18,356 | 2910 |

`forward` peaks at 19,420 pos/s and `reference` at 17,950; the ordering across
paths is stable and the shape is identical.

**The knee is at batch 128, and the cost below it is on the HOST, not the GPU.**

Per-batch time is flat at ~3.8–4.0 ms from batch 8 to batch 64 and only then
rises linearly. That flat segment is not GPU work. Measured directly:

| probe | host launch time | drain after |
|---|---:|---:|
| one v5 forward, batch 8 | 4.700 ms | 0.035 ms |
| one v5 forward, batch 64 | 5.673 ms | 0.061 ms |
| one v5 forward, batch 256 | 13.147 ms | 0.814 ms |
| single 64×384 @ 384×384 matmul | 17.0 µs | — |
| single `nn.Linear(384,384)` | 34.4 µs | — |
| one `TransformerEncoderLayer` | 0.611 ms | — |

`launch ≈ total` and `drain ≈ 0` below batch 128: the GPU finishes faster than
the host can enqueue. At ~17–34 µs per ATen op on this machine and ~110 ops per
forward, a 6-layer v5 forward costs ~4 ms of **CPU** regardless of batch size.

This reconciles both prior measurements rather than choosing between them:

* the recon-era model (**B ≈ 4.86 ms fixed, flat to batch 64**) was *right about
  the shape* and wrong about the cause — it is host dispatch, not launch
  latency, and it does not vanish under pipelining;
* the 2026-08 model (**B ≈ 0, I ≈ 54 µs, linear**) sampled only 128 and 256,
  which are both in the GPU-bound regime, and mistook the linear tail for the
  whole curve.

**Two things follow that C10 and C12 inherit.** Small batches are actively
expensive here — batch 32 buys only ~8.4k evals/s — so the "minimise outstanding
outright" reading of the newer curve does not hold. And the dispatcher will hold
the GIL for ~4 ms per batch while launching kernels, which is four orders above
the ~60 ns boundary cost C0b measured; CUDA graphs are the obvious lever and
belong to C12.

A methodological note worth keeping: the first run of this sweep produced a
completely different curve (median 4–6 ms with p10 3.6 / p90 8.9 at every batch
size) because a synchronized per-call measurement lets the GPU idle between
iterations and an RTX 5070 drops to ~307 MHz of its 3090 MHz within that gap.
Every row above reports the SM clock it was taken at, for exactly that reason.

### C9b — The W × K grid, with W=1/K=1 as the reference row

8 quiet positions × 6 repeats, 2,000 sims, virtual loss 2.5, `max_batch` 128,
P-core pinning, Q32 accumulator. `TV vs serial` is total-variation distance
between this cell's root visit distribution and the W=1/K=1 row's on the same
position; `run-to-run TV` is the worst distance between two repeats of the same
cell.

| W | K | outstanding | sims/s | mean batch | collisions/sim | TV vs serial | run-to-run TV (mean / worst) | top share | stand-in% |
|--:|--:|------------:|-------:|-----------:|---------------:|-------------:|----------------------------:|----------:|----------:|
| 1 | 1 | 1 | 55,646 | 1.0 | 0.0000 | 0.0% | 0.0% / 0.0% | 78.4% | 0.0% |
| 2 | 4 | 8 | 156,588 | 6.9 | 0.0313 | 14.5% | 2.6% / 7.2% | 77.8% | 9.2% |
| 4 | 4 | 16 | 195,790 | 12.2 | 0.0808 | 16.3% | 5.1% / 26.7% | 73.2% | 11.3% |
| 6 | 4 | 24 | 211,086 | 17.1 | 0.1184 | 19.3% | 4.4% / 17.5% | 69.3% | 13.9% |
| 8 | 4 | 32 | 185,929 | 22.6 | 0.1019 | 29.3% | 11.3% / 62.8% | 60.8% | 25.0% |
| 2 | 8 | 16 | 136,890 | 12.8 | 0.0282 | 16.4% | 4.2% / 20.9% | 72.7% | 11.3% |
| **4** | **8** | **32** | **190,152** | **21.0** | **0.0792** | **28.6%** | **11.3% / 59.9%** | **61.0%** | **24.1%** |
| 6 | 8 | 48 | 216,233 | 27.6 | 0.1154 | 32.0% | 5.2% / 9.4% | 59.3% | 26.6% |
| 8 | 8 | 64 | 200,062 | 36.2 | 0.1071 | 35.3% | 7.2% / 16.4% | 53.2% | 29.3% |
| 2 | 16 | 32 | 158,072 | 24.2 | 0.0236 | 29.6% | 8.6% / 40.8% | 59.8% | 24.8% |
| 4 | 16 | 64 | 197,443 | 34.6 | 0.0623 | 34.8% | 7.7% / 24.4% | 54.3% | 28.3% |
| 6 | 16 | 96 | 216,060 | 41.5 | 0.0974 | 36.7% | 7.7% / 20.0% | 50.6% | 31.3% |
| 8 | 16 | 128 | 213,157 | 53.8 | 0.0981 | 39.4% | 5.0% / 10.5% | 45.6% | 36.9% |

**Do not read the `run-to-run TV` column as this engine's reproducibility.** It
is an upper bound contaminated by the stand-in evaluator — see the caveat under
C9c, and DECISIONS.md C9 for the measurement showing the column tracking the
stand-in's *share* rather than the simulation count. The acceptance criterion
built on it is a ratio against `TV vs serial` (both equally contaminated), not an
absolute bound; the absolute version is deferred to C10, where the real evaluator
removes the confound. `test_layer3_root_visit_distribution_is_stable_across_runs`
measures 0.15 against a 0.75 tolerance.

### C9c — The control row, and the finding it produced

The same measurement at **W=1** and K equal to each cell's W×K. These hold the
same number of leaves in flight — therefore the same virtual-loss exposure — but
have no concurrency at all and are deterministic.

| W | K | outstanding | sims/s | TV vs serial | top share | stand-in% |
|--:|--:|------------:|-------:|-------------:|----------:|----------:|
| 1 | 1 | 1 | 55,646 | 0.0% | 78.4% | 0.0% |
| 1 | 8 | 8 | 102,013 | 14.7% | 77.3% | 8.9% |
| 1 | 16 | 16 | 118,228 | 22.4% | 66.8% | 14.2% |
| 1 | 24 | 24 | 126,425 | 31.3% | 61.4% | 25.5% |
| 1 | 32 | 32 | 129,582 | 34.3% | 55.8% | 28.0% |
| 1 | 48 | 48 | 137,934 | 39.3% | 48.2% | 34.1% |
| 1 | 64 | 64 | 137,640 | 40.7% | 44.5% | 38.7% |
| 1 | 96 | 96 | 136,798 | 43.3% | 43.6% | 44.6% |
| 1 | 128 | 128 | 134,426 | 44.9% | 41.0% | 47.3% |

**Root flattening is a function of the outstanding-leaf count and of nothing
else.** Compare like for like at equal in-flight counts:

| outstanding | deterministic (W=1) | concurrent | excess |
|---:|---:|---|---:|
| 16 | 22.4% | 16.3% (W=4/K=4), 16.4% (W=2/K=8) | **−6.1 pp** |
| 32 | 34.3% | 28.6% (W=4/K=8), 29.3% (W=8/K=4), 29.6% (W=2/K=16) | **−5.2 pp** |
| 64 | 40.7% | 35.3% (W=8/K=8), 34.8% (W=4/K=16) | **−5.6 pp** |
| 128 | 44.9% | 39.4% (W=8/K=16) | **−5.5 pp** |

The concurrent configurations are consistently *less* flattened than the
deterministic control holding the same leaves in flight, never more. **The new
parallelism model carries no concurrency tax of its own** — the entire cost is
virtual-loss exposure, exactly as scope §2.2 predicted when it made
`max_outstanding` the governing knob in place of the worker count. This is the
diagnostic the C9 brief asked for and it is the chunk's main measured result.

The effect size matches Python's landmark closely: top-move share falls
78.4% (serial) → 77.3% (8) → 66.8% (16) → 55.8% (32) → 44.5% (64) → 41.0% (128),
against Python's serial 58% → 37% at 32 outstanding.

The reproduction is `python tools/bench_c9.py --markdown`, whose `--control`
rows are on by default for this reason, and the property is pinned as a test:
`test_root_flattening_tracks_outstanding_leaves_not_worker_count`.

**Caveat, stated because it bounds every number in the two tables above.** C9
runs on the replay evaluator, whose dump holds exactly the positions the *serial*
Python reference evaluated. A parallel search leaves that set within about five
plies, so 24–37% of expansions at production settings are answered by the
stand-in evaluator (`stand-in%` column). Absolute `TV vs serial` figures are
therefore an upper bound: the serial row has 0% stand-in coverage and every other
row does not. The *comparisons between rows at equal outstanding count* — which
is where the finding lives — are sound, because those rows carry near-identical
stand-in shares (e.g. 28.0% vs 24.1% at 32 outstanding).

### C9d — Thread affinity: measured, and only visible at a realistic budget

K=8, 6 positions × 5 repeats. **20,000 sims per run**, which matters: the same
sweep at 2,000 sims shows no effect at all, because a 10 ms search does not give
Windows' Thread Director time to move anything.

| configuration | outstanding | sims/s @ 20k | sims/s @ 2k |
|---|---:|---:|---:|
| W=4, one thread per P-core | 32 | **184,907** | 196,663 |
| W=4, unpinned | 32 | 160,869 | 195,928 |
| W=6, one thread per P-core | 48 | **225,825** | 209,989 |
| W=6, unpinned | 48 | 181,357 | 214,160 |
| W=12, both SMT siblings | 96 | **240,330** | 234,295 |
| W=12, unpinned | 96 | 230,478 | 227,922 |
| W=12, all logical (E-cores included) | 96 | 237,861 | 228,998 |

Pinning to P-cores buys **+14.9% at W=4** and **+24.5% at W=6**, and only +4.3%
at W=12 where the threads fill the P-core complex anyway. The brief's reasoning
holds: a worker parked on an E-core holds the root's contended atomics longer,
and the root is on every descent path.

W=6 vs W=12 answers as **W=6, one thread per P-core**. W=12 is 6.4% faster in raw
sims/s, but it does that at 96 outstanding leaves against 48 — twice the
virtual-loss exposure, worth ~5 points of top-move share — and throughput is not
the binding constraint (see C9e). SMT does help descent slightly, which is
consistent with it being memory-latency-bound; it is not worth the leaves.

### C9e — Why throughput does not decide this

The CPU numbers above are on the replay evaluator, i.e. the descent rate with a
free evaluator. The GPU ceiling from C9a is ~18.4k evals/s at batch 128; at the
measured 24.7% cache hit rate (C7) that is ~24.4k sims/s. Every cell in the grid
is 5–10× that. **The CPU is not the constraint and cannot be made one by
choosing W** — so W and K are chosen on root-distribution quality, and the only
question throughput answers is which configurations are ruled out. None are.

Sizing against Gate 4's floor, using C9a's `gathered` column:

| outstanding | evals/s | ⇒ sims/s at 24.7% hits | vs Gate 4 floor (8k) | top share |
|---:|---:|---:|---|---:|
| 16 | 4,252 | 5,647 | **short** | 73% |
| 32 | 8,374 | 11,121 | 1.39× | 61% |
| 64 | 15,965 | 21,201 | 2.65× (clears the 15k stretch) | 53% |
| 128 | 18,166 | 24,124 | 3.02× | 46% |

The sims/s column is `evals/s ÷ (1 − 0.247)` and is therefore a **floor**: scope
§6.2 reconciles `sims = evals + cache hits + terminal visits`, and terminal
visits need no evaluation either, so the true rate is above these. It is also a
GPU-side ceiling only — it assumes the dispatcher can keep the device fed, which
C9's quiescence drain trigger does not fully do (workers idle during a batch).
C10 measures the real thing.

**Shipping default: W=4, K=8, `max_batch` 128, P-core physical pinning.** 32
outstanding clears Gate 4's floor with 1.39× margin at 61% top-move share. Going
to 64 outstanding would clear the 15k stretch target but costs 8 points of top
share, and the brief is explicit that a configuration faster and visibly flatter
at the root is the wrong trade. If C12 finds Gate 4 short, W=4/K=16 or W=8/K=8 is
the lever, and it is a measured trade rather than a guess.

### C9f — Dispatcher GIL acquisition: the prediction held

Scope §2.1's prediction was specific — acquisition stays near the uncontended
floor, and any excursion toward milliseconds means Python bytecode is running
during a search that should not be. W=4/K=8, dispatcher acquires once per batch.

| run | batches | median | p90 | p99 | max | total | share of wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2,000 sims | 107 | 1.00 µs | 2.10 µs | 2.70 µs | 9.90 µs | 0.133 ms | 1.64% |
| 20,000 sims | 906 | 1.30 µs | 2.40 µs | 3.60 µs | 12.20 µs | 1.409 ms | 1.46% |
| in-suite | 236 | 1.60 µs | — | 3.70 µs | 17.20 µs | 0.425 ms | 1.60% |

**Prediction held.** p99 is 2.7–3.7 µs against C10's 200 µs trigger for C++-side
`info` emission — nearly two orders of margin — and the worst single acquisition
across every run is 17 µs. Nothing competes for the interpreter during a search,
which is what §2.1's discipline predicted and what C10 depends on.

The median sits above C0b's ~60 ns floor because this is a *foreign* thread
acquiring for real (C0b's 60 ns was a recursive no-op on the calling thread), and
because `std::chrono::steady_clock` on Windows is QPC at a 100 ns tick.

This measurement is taken with the replay evaluator, so it is the floor and not
the answer: C10 puts torch in the callback, and C9a says that callback will hold
the GIL for ~4 ms per batch. The number to watch there is the dispatcher's
*acquire wait*, which is what this measures, not the call duration.

### C9g — ThreadSanitizer: the mandatory acceptance run

TSan does not exist on MSVC, which is the reason Global Rule 8 requires this
codebase to build on Linux at all. Build: Clang 18.1.3, Debug, `-fsanitize=thread`,
asserts live, `TSAN_OPTIONS=halt_on_error=0:history_size=7:second_deadlock_stack=1`
so a run reports *every* race rather than stopping at the first.

```
$ LD_PRELOAD="$(clang -print-file-name=libclang_rt.tsan-x86_64.so)" \
  TSAN_OPTIONS=halt_on_error=0:history_size=7:second_deadlock_stack=1 \
  PYTHONPATH="$HOME/build/gf-c9-tsan" python -m pytest tests/ -q

1229 passed, 49 skipped in 651.03s (0:10:51)
pytest exit=0

ThreadSanitizer warnings: 0
```

That covers acceptance layers 2 and 3 and the whole W × K sweep — every
`search_parallel` test in `tests/test_c9_concurrency.py` runs under it, including
W=8/K=16 at 128 outstanding leaves.

**The sanitizer is demonstrably able to fail.** A clean run is evidence only if
"TSan reported nothing" can be distinguished from "TSan was not looking", and the
two produce identical logs. `guofish_core.race_probe()` increments a plain `int`
from four threads with no synchronisation:

```
WARNING: ThreadSanitizer: data race (pid=32190)
  Read of size 4 at 0x7ffc61fbfacc by thread T2:
    #0 ...::race_probe...::operator()() const cpp/bindings.cpp:2750:39
  Previous write of size 4 at 0x7ffc61fbfacc by thread T1:
    #0 ...::race_probe...::operator()() const cpp/bindings.cpp:2750:37
  Location is stack of main thread.
```

`test_thread_sanitizer_can_actually_fail` asserts this: under a TSan build the
probe must produce a `data race` report and a non-zero exit; under any other
build it must run to completion with no report. `guofish_core.TSAN` reports which
build is loaded, so the acceptance claim rests on a checked fact rather than on a
log file's name.

**What a clean TSan run does not prove.** Both concurrency bugs found during this
chunk — a missed wakeup on the abort path, and a worker spinning instead of
sleeping because the drain epoch was read outside its retry loop — are *not* data
races, and TSan was silent on both. They were found by acceptance layer 2 and by a
test suite that took 600 s instead of 3 s. TSan proves the absence of races; it
does not prove the absence of concurrency bugs, and the layered acceptance is what
covers the rest.

### C9h — AddressSanitizer + UndefinedBehaviorSanitizer

Same source, Clang 18.1.3, Debug, `-fsanitize=address,undefined`:

```
1229 passed, 49 skipped in 401.00s (0:06:40)

leaks whose stack mentions guofish_core : 0
UBSan runtime errors                    : 0
ASan hard errors (heap/stack/global/UAF): 0
SUMMARY: AddressSanitizer: 1349986 byte(s) leaked in 1254 allocation(s)
```

The 1.35 MB is CPython and numpy holding interpreter-lifetime allocations, which
`README_BUILD.md` documents at ~1.4 MB in ~1300 allocations. The discriminating
check is the first line, not the total: **zero** leaked allocations run through
`guofish_core`. That property was briefly lost in this chunk and restored — see
DECISIONS.md, C9, on why `py::enum_` was removed from the Python surface.

### Measurement notes

* `sims/s` is always **delivered** simulations, counted by `fetch_add` on the
  thread that performed the backup — never the requested budget. Every cell also
  asserts `delivered == requested`, `vloss_total == 0` and zero conservation
  failures before it is allowed into a table; a cell whose numbers are not
  conserved is not a measurement.
* The W=1/K=1 row's 55.6k sims/s is far below C5's 178k single-threaded figure
  and that is expected, not a regression: at one in-flight path every simulation
  round-trips through the dispatcher handshake. K=1 is an acceptance
  configuration, not a production one.
* `mean batch` tracks the outstanding count rather than the worker count, which
  is the property scope §2.2 asked for. At virtual loss 0 with K=8 it collapses
  to ~1 — the second descent re-selects the first descent's leaf and collides —
  and that is the positive evidence that the dispatcher has no minimum-batch
  floor.
* The knee sweep needs a GPU and torch; the grid does not. Everything in C9b–C9f
  runs on the replay evaluator and belongs in CI.

---

## C10 — The live evaluation boundary

Reproduce with `python tools/bench_c10.py` (needs torch, a CUDA device and the v5
checkpoint). The Gate 2 numbers come from `pytest tests/test_c10_gate2.py -s`,
which prints the full distribution rather than the maximum alone; the corpus
figures come from `python tools/gen_c10_corpus.py`.

Model for every C10 number: `models/guofish5_20M/v5_10.9M_best.pt` — the 10.9M v5
student, 6 layers, d_model 384, bf16 autocast — on an RTX 5070, torch 2.8.0+cu129.

### C10a — Gate 2: the C++ gather+softmax against ATen

500 game-realistic positions, 15,036 priors. The input to both sides is one set
of numbers: the model's full 4096-wide **bf16** policy row per position, recorded
once by the Python reference and read by both. So this is two gathers of one
tensor, not two forward passes.

| comparison | max abs delta | p99.9 | p99 | p90 | p50 | mean | exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| C++ vs reference interior (ATen CPU, python-chess order) | 2.384e-07 | 1.192e-07 | 5.960e-08 | 7.451e-09 | 1.164e-10 | 2.707e-09 | 5,315 |
| C++ vs reference root (ATen CUDA, python-chess order) | 2.384e-07 | 1.192e-07 | 4.470e-08 | 7.451e-09 | 1.164e-10 | 2.781e-09 | 4,291 |
| C++ vs ATen CPU in **chess-library** order | 2.682e-07 | 8.941e-08 | 4.470e-08 | 7.451e-09 | 1.164e-10 | 2.734e-09 | 5,262 |
| *permutation alone* (both sides ATen, two orders) | 3.576e-07 | 8.941e-08 | 2.980e-08 | 3.725e-09 | 0 | 1.772e-09 | 7,941 |
| *reference root vs reference interior* | 2.384e-07 | 5.960e-08 | 2.980e-08 | 3.725e-09 | 1.164e-10 | 1.914e-09 | 5,658 |

**Gate: max <= 1e-6 and zero prior-ordering inversions. Both met, with ~4 orders
of margin on the first.** Priors over 1e-6: **0 of 15,036** in every row.
Inversions: **0** against all three reference columns. Collapsed pairs (reference
separates, C++ ties): **0**.

Read the italic rows first, because they are what makes the others
interpretable. Neither involves C++:

* **Permutation alone is 3.576e-07** — larger than any C++-vs-ATen figure in the
  table. `torch.softmax` is not permutation-invariant, python-chess and
  chess-library generate in different orders, and that difference is unavoidable
  for a correct implementation. This is the measurement that retires the <= 4 ULP
  bound the chunk table originally carried; scope §2.6's "up to 3e-7" is
  confirmed on a corpus it did not measure.
* **The reference disagrees with itself by 2.384e-07 across 9,378 of 15,036
  priors** — the root/interior device split. The brief records C5's
  single-position figure of "6 of 37 priors, max 1.9e-9"; over 500 positions it
  is two orders of magnitude larger and touches 62% of priors. C10 unifies on one
  path, so this is the divergence it accepts at roots, and Gate 2b is what
  arbitrates it.

The C++ row against the *same reduction order* (2.682e-07) is not smaller than
the rows against permuted ones, so the softmax-implementation term and the
permutation term are the same size. Neither dominates and both are noise against
the gate.

**Identical on both toolchains.** Every cell above reproduces bit-for-bit on
Windows/MSVC 19.51 and Linux/Clang 18.1.3, down to the per-row `exact` counts.
`std::exp` and the accumulation order agree across the two, so the C++ side of
Gate 2 is one number rather than a platform's number.

### C10b — The corpus, and one number that changed the mutation drill

`golden/c10_corpus.json`: 500 positions, seeded, sampled several plies per game
across three benchmark PGNs with a per-file quota.

| property | value |
|---|---|
| positions | 500 (167 / 167 / 166 across the three PGNs) |
| ply | min 8, median 49, max 238 |
| legal moves | min 2, mean 30.1, max 63 |
| priors recorded | 15,036 |
| **smallest non-zero gap between two priors** | **1.927e-06** |

That last row is the finding, and it is above the 1e-6 Gate 2 bound. bf16 logits
carry 8 mantissa bits, so the priors they produce are coarse: on this corpus,
every pair the ordering criterion could catch is far enough apart that the
magnitude criterion catches it too. The mutation drill reports it and constructs
its way around it — see DECISIONS.md, C10, and `tools/drill_c10_gate2.py`'s
`invert-inside-tolerance`.

### C10c — The switch interval, against a real search

`python tools/bench_c10.py --sims 4000`. Release, `asan=False` — the tool now
*refuses* to print these tables from a sanitizer build, because it published one
set of instrumented numbers before that check existed (DECISIONS.md, C10).

W=1, K=32, max_batch 256, on a quiet midgame; 294 boundary crossings at 11.3
rows/batch in every row, so the four cells differ only in the interpreter
setting and in whether a pure-Python thread is competing. The competitor is
`UciFormatterLoad` from `tests/test_c0b_contention.py` — imported, not
reimplemented, so the published numbers and the asserted ones cannot drift.

| interval | competing Python thread | delivered | wall (s) | **delivered sims/s** |
|---:|:--|---:|---:|---:|
| 0.005 | no | 3999 | 1.67 | 2,401 |
| 0.005 | **yes** | 3999 | 971.14 | **4** |
| 0.0005 | no | 3999 | 1.31 | 3,060 |
| 0.0005 | **yes** | 3999 | 2.23 | **1,797** |

**436x.** That is the whole argument for `sys.setswitchinterval(0.0005)` being
mandatory rather than advisory, on a real search rather than on C0b's synthetic
loop. Four simulations per second is not a slow engine; it is an engine that has
stopped. And the competing thread is not a stress test — it is a UCI `info`
formatter, which is what the host does while the engine thinks.

### C10d — Dispatcher GIL acquire wait, per batch (microseconds)

The histogram, never a mean. Sampled from *outside* the `gil_scoped_acquire`
scope, so it contains waiting and nothing else.

| interval | competing | batches | p50 | p90 | p99 | max | total (ms) | share of wall | C10 trigger |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|:--|
| 0.005 | no | 294 | 2.30 | 3.10 | 5.90 | 140.40 | 0.82 | 0.049% | clear |
| 0.005 | yes | 294 | 15,376.00 | 15,915.10 | 16,393.10 | 17,671.00 | 4,352.36 | 0.448% | **FIRED** |
| 0.0005 | no | 294 | 2.20 | 3.70 | 5.70 | 6.10 | 0.71 | 0.055% | clear |
| **0.0005** | **yes** | 294 | 5.40 | 6.90 | **20.70** | 70.30 | 1.77 | **0.079%** | **clear** |

**The C10 contingency does not fire, and this is the row it is decided on.** The
brief makes C++-side `info` emission mandatory if the acquire-wait p99 exceeds
**200 µs** or if the total exceeds **1% of the move's wall time**. In the
shipping configuration *with the adversarial competitor running*, p99 is
**20.7 µs (9.7x under)** and the share is **0.079% (12.6x under)**. C++ stdout
emission is therefore recorded and deferred, per the brief, with a measurement
behind the deferral rather than an assumption.

The FIRED row is not a shipping configuration — nothing sets 0.005 — but it is
kept because it shows the trigger works. A threshold no measurement ever crosses
is not a threshold.

Note the `max` column at 0.005/no-competitor: **140 µs against a p99 of 5.9 µs**.
One excursion, 24x the p99, on an idle interpreter. That is the tail C9's brief
insisted be kept as a histogram, and a mean of 2.79 µs would have hidden it
completely.

### C10e — Callback duration, and why it is NOT the contention metric

| interval | competing | p50 | p90 | p99 | max |
|---:|:--|---:|---:|---:|---:|
| 0.005 | no | 5,036 | 7,401 | 8,836 | 10,409 |
| 0.005 | yes | **3,266,800** | 3,713,389 | 4,300,416 | 4,673,716 |
| 0.0005 | no | 3,903 | 5,166 | 7,795 | 7,870 |
| 0.0005 | yes | 6,457 | 8,331 | 10,563 | 11,189 |

Scope §2.1 warns that torch and numpy re-acquire the GIL mid-call, so a handoff
wait can land *inside* the callback. Row two is that warning as a number: the
callback "takes" **3.27 seconds** at p50 while the acquire wait that preceded it
was 15 ms. The model did not get slower by a factor of 650; the callback spent
its time handing the interpreter back and forth with the competing thread. Anyone
optimising against `call_us` here would be optimising the wrong thing, which is
exactly why `run()` returns two numbers.

### C10f — What the shorter interval costs on a quiet interpreter

C0b flagged this as unmeasured: the shorter interval raises context-switch
frequency interpreter-wide, and nobody had checked what that costs when there is
nothing to contend with. Five repeats per interval at 20,000 simulations:

| interval | n | median sims/s | min | max | spread |
|---:|---:|---:|---:|---:|---:|
| 0.005 | 5 | 3,257 | 2,622 | 3,353 | 22.4% |
| 0.0005 | 5 | 2,829 | 2,355 | 3,468 | 39.4% |

**No measurable cost.** The medians differ by 13% and the run-to-run spread is
22-39%, so the difference is inside the noise — and the sign is not even stable:
the single-shot 4,000-simulation run in C10c had 0.0005 *faster* by 27%. The
honest statement is that on this host the setting's own overhead cannot be
resolved against scheduling variance, while the contention it prevents is a
factor of 436. Nothing about that trade is close.

This is also a caution about C10c's quiet rows: at 4,000 simulations they run for
1.3-1.7 seconds and should be read as "both around 2,400-3,000 sims/s", not as a
comparison.

### C10g — Gate 2b, and what it costs to run

500 positions x 1,600 simulations, W=1/K=1, both engines on the live evaluator.

| side | build | wall | rate |
|---|---|---:|---:|
| Python reference (`tools/gen_c10_gate2b_golden.py`) | CPython 3.13.7 | 3,157 s (53 min) | 0.16 pos/s |
| C++ engine (`tests/test_c10_gate2b.py`) | RelWithDebInfo + **ASan** + asserts | 12,531 s (3 h 29 m) | 0.04 pos/s |

**This file is the dominant term in the suite's runtime**, and the C++ figure is
a sanitizer figure — see DECISIONS.md on the build that reported `BUILD_OK` and
relinked nothing. It is reported rather than re-run on Release because that run
*is* the Global Rule 5 acceptance evidence: 800,000 simulations of the live path
under AddressSanitizer with debug asserts live, no ASan error, no assert.

Two things drive the cost and neither is the network:

* **W=1/K=1 means every forward is a batch of one.** C9 measured host ATen
  dispatch at ~4 ms flat to batch 64, so a batch-of-one search is dispatch-bound.
  The configuration is required — it is what removes leaf parallelism from the
  comparison — so this is a cost of the *test*, not of the engine. The same
  position at W=1/K=32 runs 7,000 simulations in 3.6 s (C10h).
* **Virtual loss 2.5 widens the tree**, so more distinct positions are expanded
  and fewer probes hit the cache. The pilots that sized this test were run at VL
  0 and were ~2.5x faster per position; that is why the original runtime estimate
  was wrong by a factor of three.

### C10h — The smoke position, both configurations

`8/2R5/4R2p/5pp1/2N1k3/6Pb/r4r1P/6K1 b - - 11 46`, 7,000 simulations, live
evaluator, sanitizer build.

| config | wall | best move | root visits on the answer |
|---|---:|---|---:|
| W=1, K=1 | 27.6 s | **Kf3** (`e4f3`) | 6,499 / 7,000 |
| W=1, K=32 | 3.6 s | **Kf3** (`e4f3`) | — |

The K=32 run crossed the boundary **145 times for 460 rows** (3.2 rows/batch)
while the transposition cache answered **5,617** leaves — 92% of the 6,077 leaves
that needed an answer never reached the network. That is `prepare_live_batch`
doing its job: probing before the crossing rather than inside the expansion loop,
so a cached leaf costs no row of the batch.

The 7.7x wall-clock difference between the two rows at identical simulation
counts is the batch-of-one cost stated above, isolated.

### Measurement notes

* Every C10 table is **delivered** simulations, asserted equal to the requested
  budget before the row is allowed into the table.
* `tools/bench_c10.py` prints `asan=` in its header and **refuses to run** on a
  sanitizer build without `--allow-instrumented`. That check exists because the
  first version of C10c was measured on an instrumented build and looked
  plausible; the throughput columns were 2-4x low and the acquire-wait columns
  4-8x high.
* The quiet-interpreter rows are short (1.3-1.7 s) and are not a comparison; C10f
  is. The contended rows are long enough that their difference is not in doubt.

---

## C10b — The forward as a CUDA graph, and the decision surface re-measured

Reproduce with `python tools/bench_c10b_knee.py --markdown` (the GPU curve) and
`python tools/bench_c10b.py --markdown` (the live grid, Gate 4, padding and the
acquire-wait histogram). Both need torch, a CUDA device and the v5 checkpoint,
and `bench_c10b.py` refuses to publish from a sanitizer build.

Same machine and model as C10 throughout: `models/guofish5_20M/v5_10.9M_best.pt`
(10.9M v5 student, 6 layers, d_model 384, bf16 autocast), RTX 5070, torch
2.8.0+cu129, MSVC 19.51 Release, `asan=False`.

**Every absolute figure in C9a and C9b is superseded here.** C9a measured an
ungraphed forward — i.e. ~4 ms of host dispatch per batch — and C9b ran on the
replay evaluator with a stand-in answering 24-37% of expansions. Both confounds
are gone.

### C10b-1a — Capture fidelity: the graphed forward is the eager forward

The claim every other number rests on. For each captured shape, the whole
500-position Gate 2 corpus is run through the graph and through the un-captured
forward **at the same shape**, and the raw bf16 policy bit patterns are
compared. From `pytest tests/test_c10b_graphs.py -s`:

| shape | policy words compared | differing | values differing |
|---:|---:|---:|---:|
| 1 | 2,048,000 | **0** | 0 |
| 8 | 2,031,616 | **0** | 0 |
| 16 | 2,031,616 | **0** | 0 |
| 24 | 1,966,080 | **0** | 0 |
| 32 | 1,966,080 | **0** | 0 |
| 48 | 1,966,080 | **0** | 0 |
| 64 | 1,835,008 | **0** | 0 |
| 96 | 1,966,080 | **0** | 0 |
| 128 | 1,572,864 | **0** | 0 |

Graph capture did not change kernel selection on this model. That is not a
property to assume — it is exactly what the C10b brief required be checked — and
it is what makes Gate 2's re-pass below reproduce C10a cell for cell.

### C10b-1b — Why `torch.compile(mode="reduce-overhead")` is rejected

The brief asks for it to be tried first and measured against manual capture.
Both were built.

| batch | manual capture | `torch.compile` | compile speedup |
|---:|---:|---:|---:|
| 8 | 0.747 ms | 0.507 ms | 1.47x |
| 32 | 1.814 ms | 1.478 ms | 1.23x |
| 128 | 6.350 ms | 4.931 ms | 1.29x |
| 256 | 13.269 ms | 10.474 ms | 1.27x |

| | policy words differing from the eager forward |
|---|---:|
| manual capture | **0** of 2,048,000 |
| `torch.compile` | **1,773,671** of ~2,621,440 (68%) |

Inductor fuses the epilogues, which is where the speed comes from and why the
numbers move: a bf16 logit shifted by one ulp moves a prior by ~1e-3, three
orders over Gate 2's 1e-6 bound. The brief's own rule settles it — *"a graphed
forward that fails Gate 2 is rejected regardless of speed."* Two further costs,
recorded in DECISIONS.md: it needs
`torch._inductor.config.use_static_cuda_launcher = False` on this machine or
every launch raises `OverflowError`, and warmup is ~35 s of compilation against
~0.6 s of capture.

### C10b-1c — The shape term: padding is not free, and it is not the graph's

Same rows, two batch widths, **eager torch only — no graph anywhere near it.**
Sweeps over the Gate 2 corpus, comparing width n against the width it would pad
up to:

| n | padded to | words differing | max abs delta on a logit | max abs delta on a prior | prior-ordering inversions |
|---:|---:|---:|---:|---:|---:|
| 1, 2, 3, 7, 8 | 1 / 8 | **0%** | 0 | 0 | 0 |
| 4 | 8 | 66.3% | 0.0625 | 6.2e-03 | 21 |
| 5 | 8 | 66.5% | 0.0625 | 5.8e-03 | 31 |
| 6 | 8 | 66.1% | 0.0625 | 7.8e-03 | 28 |
| 9-31 | 32 | **0%** | 0 | 0 | 0 |
| 33-127 | 128 | **0%** | 0 | 0 | 0 |
| 129-255 | 256 | **0%** | 0 | 0 | 0 |

cuBLAS selects a different kernel at widths 4-6 and nowhere else on this device.
The effect is one bf16 ulp, and eager torch had it before C10b existed — the
pre-C10b engine evaluated at whatever width the dispatcher happened to drain, so
it saw up to `max_batch` distinct widths. C10b reduces that to nine. **Padding
makes the engine more shape-stable, not less**, and the residual cost is
confined to three batch widths.

### C10b-2 — Gate 2, re-run on the graphed forward

Same corpus, same criterion, same C++ gather — logits from the graph instead of
from the eager forward. 384 positions (the ones the golden generator evaluated
at a captured shape), 11,481 priors.

| comparison | max abs delta | p99 | exact | over 1e-6 | inversions |
|---|---:|---:|---:|---:|---:|
| graphed vs reference interior (ATen CPU, python-chess order) | 2.384e-07 | 5.364e-08 | 3,986 | **0** | **0** |
| graphed vs reference root (ATen CUDA, python-chess order) | 2.384e-07 | 4.470e-08 | 3,341 | **0** | **0** |
| graphed vs ATen CPU in chess-library order | 2.682e-07 | 5.960e-08 | 3,865 | **0** | **0** |

**Gate 2 re-passed: max 2.682e-07 against the 1e-6 bound, zero inversions.**
Every maximum reproduces C10a's exactly, which is the expected consequence of
C10b-1a and the reason the table is worth printing rather than asserting.

### C10b-3a — The knee, graphed

`python tools/bench_c10b_knee.py --markdown`. C9a's methodology imported rather
than restated — `torch.cuda.synchronize()` around every timed iteration, SM
clock reported per row. `best pos/s` is the minimum saturated per-batch time, as
in C9a. Every swept batch size is captured, so this table has no padding in it.

**`graph-gathered`** — C9a's `gathered` path with the forward graphed, and the
row acceptance criterion 3 is denominated in:

| batch | isolated ms | best ms | best pos/s | C9a `gathered` | speedup | SM MHz |
|------:|------------:|--------:|-----------:|---------------:|--------:|-------:|
| 8 | 0.851 | 0.848 | 9,430 | 1,951 | 4.83x | 2917 |
| 16 | 1.193 | 1.130 | 14,164 | 4,252 | 3.33x | 2880 |
| 32 | 1.957 | 1.902 | **16,821** | 8,374 | **2.01x** | 2887 |
| 64 | 3.576 | 3.374 | 18,971 | 15,965 | 1.19x | 2880 |
| 128 | 6.440 | 6.187 | 20,689 | 18,166 | 1.14x | 2857 |
| 256 | 14.247 | 13.394 | 19,113 | 18,356 | 1.04x | 2880 |

**`graph-production`** — what `playing/v6/evaluator.py` actually does. C10 put
the gather in C++, so the full 4096-wide bf16 row crosses the bus, not scope
§2.5's 64-wide one. Quoted separately because `gathered` is the comparable row
and this is the true one:

| batch | isolated ms | best ms | best pos/s |
|------:|------------:|--------:|-----------:|
| 8 | 0.862 | 0.836 | 9,568 |
| 16 | 1.178 | 1.150 | 13,917 |
| 32 | 1.956 | 1.921 | **16,662** |
| 64 | 3.528 | 3.451 | 18,546 |
| 128 | 6.490 | 6.254 | 20,468 |
| 256 | 14.103 | 13.101 | 19,541 |

The 4096-wide D2H costs 1% at batch 32 and nothing at all at 128. It is not
worth moving the gather to the GPU.

**`graph-forward`**, and the ungraphed paths re-run in the same session so the
comparison is not across two days of driver state:

| batch | graph-forward | forward (eager) | reference (eager) | gathered (eager) |
|------:|--------------:|----------------:|------------------:|-----------------:|
| 8 | 11,223 | 2,234 | 2,099 | 2,076 |
| 16 | 15,901 | 4,454 | 4,175 | 4,080 |
| 32 | 18,407 | 9,433 | 8,996 | 9,322 |
| 64 | 19,998 | 17,719 | 15,383 | 16,246 |
| 128 | **21,018** | 19,564 | 17,907 | 18,630 |
| 256 | 19,736 | 19,100 | 18,016 | 18,333 |

**Acceptance criterion 3: MET.** Batch-32 `gathered` is 16,821 pos/s against
C9a's published 8,374 — **2.01x**, target 16,748. Stated honestly alongside it:
the same-session re-measurement of the ungraphed path is 9,322 rather than
8,374, so the in-session speedup is **1.80x**. The criterion names the published
figure and is met against it; the smaller number is the one to believe about
this machine today, and it is still within 8% of the brief's prediction.

**The prediction landed.** The brief predicted "post-graph batch-32 forward ≈ 2
ms → ~16k pos/s, batch 64 ≈ 17k". Measured: 1.90 ms and 16,821 at batch 32,
18,971 at batch 64.

**The knee is still 128** — `graph-forward` peaks there at 21,018 pos/s and
falls at 256 — but the shape below it has changed completely. Batch 32 is now
80% of peak throughput against 46% before, because the flat ~4 ms host-dispatch
segment C9a found from batch 8 to 64 is gone. Per-batch cost is now ~0.55 ms
fixed plus ~44 us per row, and the fixed term is kernel *execution* overhead
inside the graph, not submission.

### C10b-3b — The W x K grid, live and graphed

`python tools/bench_c10b.py --sections grid`. 8 quiet positions x 4 repeats,
2,000 sims, virtual loss 2.5, `max_batch` 128, Q32 accumulator. Every cell
asserts `delivered == requested`, `vloss_total == 0`, zero conservation failures
and — new here — `synthetic_evaluations == 0`, so the stand-in that contaminated
C9b cannot contribute.

`GPU share` is `call_ns / wall_ns`: the dispatcher is blocked inside the
callback for the whole GPU wait, so this is the fraction of the cycle already on
the device, and `pipeline ceiling` is its reciprocal — the most Stage 2 could
ever buy.

| W | K | outstanding | sims/s | mean batch | rows/crossing | pad waste | GPU share | pipeline ceiling | collisions/sim | TV vs serial | run-to-run TV (mean / worst) | top share |
|--:|--:|------------:|-------:|-----------:|--------------:|----------:|----------:|-----------------:|---------------:|-------------:|----------------------------:|----------:|
| 1 | 1 | 1 | 1,827 | 1.0 | 1.0 | 1.00x | 92% | 1.08x | 0.0000 | 0.0% | 0.0% / 0.0% | 78.4% |
| 2 | 4 | 8 | 4,506 | 4.5 | 4.1 | 1.96x | 96% | 1.05x | 0.0063 | 14.5% | 1.1% / 2.9% | 78.8% |
| 4 | 4 | 16 | 8,526 | 10.9 | 9.9 | 1.30x | 93% | 1.08x | 0.0517 | 17.3% | 1.5% / 4.5% | 73.2% |
| 6 | 4 | 24 | 10,172 | 16.3 | 14.9 | 1.23x | 90% | 1.11x | 0.0810 | 20.8% | 2.3% / 6.5% | 68.6% |
| 8 | 4 | 32 | 9,874 | 18.8 | 17.2 | 1.19x | 82% | 1.21x | 0.0610 | 23.3% | 2.9% / 6.0% | 65.7% |
| 2 | 8 | 16 | 8,181 | 8.9 | 8.1 | 1.11x | 93% | 1.07x | 0.0074 | 18.4% | 2.4% / 5.2% | 71.8% |
| 4 | 8 | 32 | 11,859 | 19.9 | 18.3 | 1.15x | 92% | 1.09x | 0.0545 | 24.7% | 2.8% / 6.5% | 64.3% |
| 6 | 8 | 48 | 12,854 | 29.0 | 26.5 | 1.18x | 92% | 1.09x | 0.0867 | 26.9% | 2.9% / 7.4% | 62.3% |
| 8 | 8 | 64 | 13,532 | 37.9 | 34.5 | 1.19x | 91% | 1.10x | 0.0867 | 29.3% | 2.7% / 5.6% | 59.1% |
| 2 | 16 | 32 | 12,108 | 17.7 | 16.3 | 1.10x | 94% | 1.06x | 0.0073 | 25.9% | 3.0% / 8.2% | 62.3% |
| 4 | 16 | 64 | 13,366 | 34.4 | 31.3 | 1.18x | 92% | 1.09x | 0.0509 | 29.5% | 3.2% / 7.8% | 59.4% |
| 6 | 16 | 96 | 13,805 | 44.5 | 40.4 | 1.19x | 91% | 1.10x | 0.0798 | 31.0% | 4.0% / 6.7% | 57.4% |
| 8 | 16 | 128 | 14,774 | 56.9 | 51.5 | 1.19x | 92% | 1.09x | 0.0800 | 34.1% | 3.9% / 7.6% | 53.5% |

**`stand-in%` is gone and `run-to-run TV` is now the engine's own number.** That
is C9's deferred acceptance debt, closed: see C10b-3g.

### C10b-3c — The control: W=1 at the same outstanding-leaf counts

Same virtual-loss exposure, no concurrency, deterministic by construction.

| W | K | outstanding | sims/s | mean batch | rows/crossing | pad waste | GPU share | collisions/sim | TV vs serial | run-to-run TV | top share |
|--:|--:|------------:|-------:|-----------:|--------------:|----------:|----------:|---------------:|-------------:|--------------:|----------:|
| 1 | 1 | 1 | 1,827 | 1.0 | 1.0 | 1.00x | 92% | 0.0000 | 0.0% | 0.0% | 78.4% |
| 1 | 8 | 8 | 8,203 | 8.0 | 7.3 | 1.09x | 91% | 0.0001 | 15.3% | 0.0% | 78.0% |
| 1 | 16 | 16 | 11,900 | 15.9 | 14.5 | 1.09x | 90% | 0.0003 | 19.9% | 0.0% | 70.3% |
| **1** | **24** | **24** | **13,944** | **23.8** | **21.7** | **1.08x** | **89%** | **0.0004** | **24.7%** | **0.0%** | **64.2%** |
| 1 | 32 | 32 | 14,413 | 31.4 | 28.6 | 1.08x | 89% | 0.0004 | 26.6% | 0.0% | 61.8% |
| 1 | 48 | 48 | 15,813 | 46.8 | 42.3 | 1.09x | 89% | 0.0006 | 31.1% | 0.0% | 57.5% |
| 1 | 64 | 64 | 15,847 | 61.0 | 55.0 | 1.08x | 89% | 0.0009 | 33.7% | 0.0% | 53.9% |
| 1 | 96 | 96 | 16,392 | 88.4 | 78.8 | 1.09x | 89% | 0.0010 | 37.3% | 0.0% | 47.8% |
| 1 | 128 | 128 | 17,640 | 113.5 | 100.3 | 1.08x | 89% | 0.0011 | 40.4% | 0.0% | 44.1% |

**C9c's finding survives the change of evaluator, and is now the reason the
selection moves.** Root flattening still tracks the outstanding-leaf count and
nothing else — compare at equal counts: 24 outstanding gives 64.2% (W=1/K=24)
against 68.6% (W=6/K=4); 32 gives 61.8% (W=1) against 64.3-65.7% concurrent; 64
gives 53.9% against 59.1-59.4%. The concurrent rows remain slightly *sharper*
at equal exposure, exactly as C9c measured.

What changed is throughput. On the replay evaluator W=1 was the slow control
row; on the live graphed evaluator it is the fastest row at every outstanding
count, because the GPU is the bottleneck and W=1 produces whole batches instead
of fragments (mean batch 23.8 against 19.9 at the same 24-32 leaves in flight).

### C10b-3d — Crossings, rows per crossing, and what padding costs

One move at W=1/K=24. `pad waste` is padded rows divided by real rows.

| regime | budget | crossings | rows | rows/crossing | cache-answered | pad waste | shape 1 | 8 | 16 | 24 | 32+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh midgame root | 20,000 | 835 | 17,740 | 21.25 | 2,260 | 1.12x | 1 | 1 | 6 | 827 | 0 |
| endgame root (the C10h regime) | 7,000 | 248 | 1,811 | 7.30 | 4,284 | 1.46x | 13 | 160 | 57 | 18 | 0 |
| endgame after 6 plies of reuse | 20,000 | 1 | 2 | 2.00 | 11 | 4.00x | 0 | 1 | 0 | 0 | 0 |

The brief asked whether padding waste dominates in the reuse regime and whether
a batch-8 graph is needed. **It is, and it is captured** — 160 of 248 crossings
in the endgame land on shape 8, and without it they would land on 32 and pay
1.9 ms each instead of 0.85 ms.

The third row is where the network stops mattering: after six plies of reuse,
20,000 simulations produce **one** boundary crossing of two rows, because the
tree is fully expanded and the transposition cache and terminal fast paths
answer everything else. Its 4.00x pad waste is 2 rows evaluated at shape 8, i.e.
6 wasted rows in a whole move — the ratio is alarming and the absolute cost is
0.4 ms. Ratios need denominators.

### C10b-3e — Gate 4 margin, both regimes

20,000 simulations, `max_batch` 128, 5 repeats (median), **delivered** sims
asserted equal to the budget. The reuse-heavy rows play 6 plies of the engine's
own moves first, so the tree, the arena and the cache are in the state a real
game leaves them in.

| config | regime | delivered sims/s | min | max | rows/crossing | pad waste | cache hit | GPU share | vs floor (8k) | vs stretch (15k) | vs Python (838) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **W=1, K=24** | **fresh root** | **13,912** | 13,769 | 14,138 | 21.2 | 1.12x | 11.3% | 87% | **1.74x** | 0.93x | **17x** |
| **W=1, K=24** | **reuse-heavy** | **92,639** | 88,531 | 94,901 | 2.0 | 4.00x | 84.6% | 48% | **11.58x** | 6.18x | **111x** |
| W=1, K=32 | fresh root | 14,372 | 14,180 | 14,535 | 28.1 | 1.13x | 12.1% | 87% | 1.80x | 0.96x | 17x |
| W=1, K=32 | reuse-heavy | 86,839 | 63,717 | 92,762 | 4.0 | 2.00x | 69.2% | 51% | 10.85x | 5.79x | 104x |
| W=1, K=64 | fresh root | 15,945 | 15,789 | 16,034 | 56.1 | 1.14x | 11.9% | 87% | 1.99x | 1.06x | 19x |
| W=1, K=64 | reuse-heavy | 90,399 | 87,254 | 96,472 | 6.0 | 1.33x | 79.3% | 48% | 11.30x | 6.03x | 108x |
| W=4, K=8 | fresh root | 13,123 | 13,007 | 13,186 | 17.2 | 1.17x | 11.2% | 93% | 1.64x | 0.87x | 16x |
| W=4, K=8 | reuse-heavy | 104,230 | 87,739 | 169,765 | 2.0 | 2.87x | 86.1% | 40% | 13.03x | 6.95x | 124x |
| W=8, K=4 | fresh root | 13,820 | 13,763 | 14,346 | 21.1 | 1.16x | 11.2% | 92% | 1.73x | 0.92x | 16x |
| W=8, K=4 | reuse-heavy | 65,859 | 22,924 | 177,657 | 1.3 | 1.73x | 80.4% | 26% | 8.23x | 4.39x | 79x |
| W=8, K=16 | fresh root | 15,767 | 15,677 | 16,494 | 57.4 | 1.19x | 12.2% | 91% | 1.97x | 1.05x | 19x |
| W=8, K=16 | reuse-heavy | 70,156 | 60,954 | 86,851 | 3.0 | 1.91x | 79.7% | 48% | 8.77x | 4.68x | 84x |

**Gate 4 clears at every candidate**, on a fresh midgame root, at 1.64-1.99x the
8,000 floor. The stretch target of 15,000 is reached only at 64+ outstanding
leaves, which costs ~10 points of top-move share; see the selection below.

The multi-worker configurations are visibly less stable in the reuse regime —
W=8/K=4 spans 22,924 to 177,657 across five repeats — because eight threads
descending a fully-expanded endgame tree collide constantly and produce batches
of 1.3 rows. W=1 spans 88,531 to 94,901 on the same position.

### C10b-3f — Pinning, at W=1

W=1/K=24, 20,000 simulations, median of 4.

| affinity | delivered sims/s | min | max |
|---|---:|---:|---:|
| **none** | **13,923** | 13,828 | 14,089 |
| one thread per P-core | 12,661 | 11,318 | 13,287 |
| both SMT siblings | 12,160 | 10,810 | 13,337 |

**Pinning is a 9% LOSS at W=1**, inverting C9d's +14.9%/+24.5% at W=4/W=6 — and
by C9d's own mechanism. Pinning helps because a worker on an E-core holds the
root's contended atomics longer; at one worker there is no contention on the
root, and all pinning does is put the single worker on a core the dispatcher
also wants.

### C10b-3g — THE SELECTION, and C9's deferred root-stability tolerance

**Shipping: W = 1, K = 24, `max_batch` 128, affinity `none`, capture ladder
{1, 8, 16, 24, 32, 48, 64, 96, 128}. No longer provisional.**

| | W=4/K=8 (C9's pick) | **W=1/K=24** | W=1/K=32 |
|---|---:|---:|---:|
| delivered sims/s, 20k fresh root | 13,123 | **13,912** | 14,372 |
| top-move share | 64.3% | **64.2%** | 61.8% |
| TV vs serial | 24.7% | **24.7%** | 26.6% |
| run-to-run TV (mean / worst) | 2.8% / 6.5% | **0.0% / 0.0%** | 0.0% / 0.0% |
| select collisions per simulation | 0.0545 | **0.0004** | 0.0004 |
| Gate 4 margin | 1.64x | **1.74x** | 1.80x |

W=1/K=24 matches C9's provisional pick on both axes and is deterministic.
W=1/K=32 is 3.3% faster for 2.4 points of top-move share and is declined on
C9's own principle: throughput is not the binding constraint at a 1.74x Gate 4
margin, and a configuration faster and visibly flatter at the root is the wrong
trade.

**C9's deferred absolute root-stability tolerance, now measurable and now
measured.** With the live evaluator answering every leaf, `run-to-run TV` is the
engine's own variation rather than the stand-in's share:

* at the shipping configuration: **0.0%** — reproducible run to run;
* across every concurrent cell measured: **<= 4.0% mean, <= 8.2% worst** of 8
  positions x 4 repeats, against C9b's contaminated 5-11% / 62.8%.

**The absolute tolerance is set at 10% run-to-run TV** for any concurrent
configuration: above every measured worst case with room for scheduling
variance, and far enough below the 24.7% that separates the shipping
configuration from the serial reference that the two cannot be confused.

### C10b-4 — Dispatcher GIL acquire wait, re-run (microseconds)

W=1/K=24, 20,000 simulations, `sys.setswitchinterval(0.0005)`. Sampled from
outside the `gil_scoped_acquire` scope, so it contains waiting and nothing else.
The competitor is `UciFormatterLoad` from `tests/test_c0b_contention.py` —
imported, not reimplemented.

| forward | competing | batches | rows/batch | sims/s | p50 | p90 | p99 | max | total (ms) | share of wall | C10 trigger |
|:--|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| graphed | no | 736 | 21.0 | 15,502 | 1.50 | 2.40 | 3.70 | 95.30 | 1.38 | 0.107% | clear |
| **graphed** | **yes** | 736 | 21.0 | **15,126** | 5.00 | 6.40 | **61.90** | 106.10 | 4.90 | **0.371%** | **clear** |
| eager | no | 736 | 21.0 | 5,383 | 2.10 | 3.00 | 3.90 | 75.90 | 1.83 | 0.049% | clear |
| eager | yes | 736 | 21.0 | 3,509 | 5.40 | 6.60 | 11.90 | 89.30 | 4.34 | 0.076% | clear |

**Acceptance criterion 4: the trigger stays clear with the adversarial formatter
running.** p99 is 61.9 us against the 200 us threshold (3.2x under) and the
share of wall is 0.371% against 1% (2.7x under). A second run of the same cell
measured p99 21.8 us and 0.371% — the p99 of a 736-sample tail moves by a factor
of three between runs on this host, and both are clear, which is the honest way
to state it.

The row that is not about the trigger: **graphed is 4.3x faster than eager with
the competitor running** (15,126 against 3,509) and 2.9x without. The eager
callback loses more to contention because it holds the interpreter across ~110
op submissions; the graphed one has four calls to be interrupted between.

### C10b-4b — Callback duration, and what actually shrank

Not a contention metric (C10e), but it is where the GIL-held window lives.

| forward | competing | p50 | p90 | p99 | max |
|:--|:--|---:|---:|---:|---:|
| graphed | no | 1,496 | 1,689 | 1,940 | 2,077 |
| graphed | yes | 1,533 | 1,647 | 1,822 | 3,474 |
| eager | no | 4,331 | 6,970 | 7,638 | 8,398 |
| eager | yes | 7,230 | 9,005 | 10,297 | 12,666 |

**The brief predicted the held window would fall from ~4 ms to "the copy plus
replay-launch (~tens of us)". It fell to ~1.5 ms, and the prediction was wrong
for an instructive reason.** The callback does not return when the launch is
submitted; it returns when the D2H copy has completed, so it contains the GPU
wait — 1.90 ms of graph replay at shape 24 is most of the 1.5 ms p50 here at
21 rows. What the brief was actually predicting is the *GIL-held* window, and
that did collapse: torch releases the interpreter across the blocking copy,
which is why the competing thread's acquire wait in C10b-4 is 5-62 us rather
than 1,500. The two columns measure different things and C10e already said so;
this is the case that makes the distinction load-bearing.

Note also that the graphed callback is nearly **immune to contention** — p50
1,496 -> 1,533 us with the formatter running, against eager's 4,331 -> 7,230.
Fewer torch calls means fewer places for a handoff to land inside one.

### Measurement notes

* Every C10b table reports **delivered** simulations, asserted equal to the
  requested budget before the row is allowed in. Grid cells additionally assert
  `vloss_total == 0`, zero conservation failures and `synthetic_evaluations ==
  0` — the last one is what makes this grid's `run-to-run TV` mean something
  C9b's could not.
* `tools/bench_c10b.py` refuses to run on a sanitizer build without
  `--allow-instrumented`, for the reason recorded in DECISIONS.md, C10.
* The first live grid was run on the brief's proposed {1, 8, 32, 128} ladder and
  is not published, because it measures padding rather than W and K: `pad waste`
  swings 1.09x-2.87x across cells and W=1/K=48 comes out at 6,888 sims/s against
  15,813 on the shipping ladder. It is described in DECISIONS.md because
  discarding a measurement is itself a decision.
* GPU memory: the shipping ladder reserves ~730 MiB of graph pools at
  `max_batch` 128 and ~1.2 GiB at 256 (Gate 2b's fixture). ~1.9 MiB per captured
  row, reported by `CaptureReport.describe()` at construction.
* Capture costs ~0.6 s at engine start and is one-time.

## C11c — Pondering: stop latency, arena high-water, and the ponder-hit rate

Produced by `tools/bench_c11c_ponder.py` and `tools/smoke_c11.py --ponder`.
Windows/MSVC Release, RTX 5070, shipping defaults (W=1, K=24, max_batch 128,
VL 2.5, `PolicyTemperature` 1.0, `PonderDecay` 1.0).

**None of these is a strength measurement, and Gate 5 must not use pondering
at all** — the 2687.7 anchor was measured with ponder off, so the strength gate
has to match it. `tools/smoke_c11.py --ponder` refuses `--mode nodes` for a
related reason: fixed-node play sends `go nodes N` with `tc=inf`, so there is no
opponent clock to ponder on and the run would measure nothing while looking as
if it had.

### Resolved sizing at the shipping defaults

| quantity | value | where from |
|---|---:|---|
| `sims_per_move` | 60,000 | `fixed_sims or sim_cap` — see DECISIONS.md, C11c |
| `ponder_max_sims` | 60,000 | `sims_per_move / ponder_decay`, decay 1.0 |
| `arena_nodes` | 7,200,000 | `60 x (sims_per_move + ponder_max_sims)` |
| arena footprint | 268 MB per arena, **536 MB for the ping-pong pair** | measured RSS, not reserved address space |
| coupling | `1.0 x 60,000 <= 60,000` OK | printed on the `[config] ponder` line every `isready` |

### Stop latency during ponder — a histogram, not a mean

30 samples, measured from the `stop` WRITE to the `bestmove` READ over a real
pipe, walking a game so `apply_move` compacts between samples.

| bucket (ms) | count |
|---|---:|
| [5, 10) | 1 |
| [10, 20) | 18 |
| [20, 50) | 11 |
| [50, inf) | 0 |

| | ms |
|---|---:|
| min | 7.8 |
| p50 | 17.2 |
| p90 | 28.2 |
| p95 | 29.8 |
| p99 | 30.9 |
| max | 30.9 |

Root visits at the moment of the stop: min 6,393, p50 19,763, max 40,245.

**C11's baseline for `stop` under `go infinite` was 7-109 ms.** The tail is now
inside it, which is `Engine.interrupt_slice` arming the C++ mutable deadline in
the past rather than the loop waiting out a 50 ms slice. This sits on the ponder
MISS critical path, where the opponent has already moved and the engine is
spending its own clock until it answers.

A first version of this measurement re-pondered ONE position and produced 7.6 ms
rising monotonically to 227.5 ms over 25 samples. That was the tree growing
without ever advancing, and the O(visited nodes) principal-variation walk growing
with it — a state no game reaches. See DECISIONS.md, C11c.

### Arena high-water against the predicted `arena_nodes`

| run | plies | clock | hit rate | peak nodes | of 7,200,000 |
|---|---:|---|---|---:|---:|
| self-play, forced hits | 40 | 400 ms/move | 100% | 5,993,742 | 83.2% |
| self-play, forced hits | 70 | 400 ms/move | 100% | 4,580,052 | 63.6% |
| **Cutechess vs Stockfish** | **488 moves / 20 games** | **10+0.1** | **36.8%** | **1,589,189** | **22.1%** |

The self-play rows force a 100% hit rate — the driver knows both moves — which
is the worst case for the arena because every ponder's nodes are carried into
the timed search. The match row is the shipping configuration against a real
opponent. **Zero `ARENA_EXHAUSTED` lines across 976 telemetry lines in the
match.**

Occupancy plateaus rather than ramping: 44% by ply 10 in the self-play runs,
then slowly, because `apply_move` compacts every move.

Nodes per simulation measured this chunk: **31.7-31.8** at 5,000-19,000
simulations. That extends the brief's curve at its cheap end (C8: 38.9 at 2,000
and 39.7 at 8,000; playtesting: <=32.4 at ~37,000) and confirms that fitting the
observed ratio rather than the conservative 40 would have halved the arena.

### Graceful degradation, made to fire

`--arena-capacity 32768`, `go nodes 4000`, through a pipe:

| | |
|---|---|
| bestmove | `b5c6` — legal |
| delivered | 1,055 of 4,000 |
| `arena_exhausted` | true |
| exhausted at | 32,755 nodes |
| stdout notice | `info string arena exhausted at 32755 nodes - this move delivered 1055 simulations of its budget and is NOT admissible as a benchmark row` |
| Python traceback | none |
| next `go` | answered normally |

Library-level, against the Gate 1 dump and the stand-in evaluator, at three
capacities and a 4,000-simulation budget:

| arena | delivered / requested | exhausted at | best move | `vloss_total` | conservation failures |
|---:|---|---:|---|---:|---:|
| 4,096 | 119 / 3,999 | 4,089 | g5h7 | 0 | 0 |
| 40,000 | 1,206 / 3,999 | 39,982 | g2f1 | 0 | 0 |
| 1,200,000 | 3,999 / 3,999 | — | h4h5 | 0 | 0 |

The last two columns are the point. A degraded search leaves the tree exactly as
it found it, which is what makes "the search returned a slightly weaker move" a
true description rather than a hopeful one.

### Ponder-hit rate — 20 games, 10+0.1, vs Stockfish UCI_Elo 1800

| | |
|---|---:|
| ponders resolved | 486 |
| hits | 179 |
| misses | 307 |
| **hit rate** | **36.8%** |
| skipped (book/tablebase answered the position) | 1 |

Simulations carried into a hit: p50 3,209, p90 11,740, p95 13,797, p99 20,003,
max 25,500, mean 4,478.

This is the number that decides whether salvage-on-miss is worth revisiting. It
is low, and that is not by itself an argument for salvage: a miss costs nothing
that was otherwise being used, so the case needs Gate 5-class evidence that the
lost work matters as well. See DECISIONS.md, C11c.

### 20-game Cutechess smoke, `ponder=true`, 10+0.1, concurrency 1

13/13 verdicts passed, including all four the brief names:

| verdict | result |
|---|---|
| illegal moves | 0 (cutechess, and an independent python-chess replay of all 20 PGNs) |
| null bestmoves | 0 over 820 mirrored bestmoves |
| engine crashes / stalls | 0, and 0 Python tracebacks |
| timeouts | 0 |
| abnormal PGN terminations | 0 |
| resolved book/Syzygy state recorded | book open, Syzygy open (5-man) |

Decisions: search 465, book 23, tablebase 0 — 23 of 488 moves bypassed MCTS.

**Concurrency 1 is not incidental.** Pondering multiplies GPU contention by the
number of games in flight: four simultaneous games pondering means four searches
against one RTX 5070, each getting a fraction of the throughput while believing
it has the whole device, and per-game strength falls rather than rises. Either
cap concurrency at one game when pondering is enabled, or disable pondering when
concurrency exceeds one.

### Measurement notes

* `SearchOutcome.ponder_sims` and `search_sims` are never summed into one
  figure. `sims_per_s` divides post-hit work by post-hit wall time; folding the
  ponder into either would inflate the rate by whatever the opponent spent
  thinking, and would make a ponder-on run incomparable with the ponder-off arm
  Gate 5 is measured against.
* `arena_exhausted` makes `delivered < requested` a legitimate outcome for the
  first time since C10. A benchmark harness must consult it and REJECT the row
  rather than publish a short one;
  `test_a_benchmark_harness_rejects_a_degraded_row_rather_than_publishing_it`
  asserts the rule rather than a paraphrase of it.
* The stop-latency and arena figures come from a Release build. Nothing in this
  section was measured on a sanitizer build.

---

## C12 — Where the time actually goes, and what could be done about it

Reproduce with:

```
python tools/profile_c12.py search --regime fresh --sims 20000     # the workload
nsys profile --trace=cuda,nvtx --cuda-graph-trace=graph ...        # C12-1
powershell -File tools/run_ncu_c12.ps1                             # C12-2 (needs admin)
python tools/nsys_report_c12.py runs/c12/*.sqlite --markdown
python tools/ncu_report_c12.py runs/c12/*.ncu-rep --markdown
python tools/bench_c12.py --markdown                               # C12-3, C12-5, C12-7
```

Same machine and model as C10/C10b throughout: `models/guofish5_20M/v5_10.9M_best.pt`
(10.9M v5 student, 6 layers, d_model 384, bf16 autocast), RTX 5070 (48 SMs, **WDDM**),
torch 2.8.0+cu129, MSVC 19.51 Release, `asan=False`, `verify_compaction=False`.

**Read C12-4 before quoting any reuse-heavy figure from C10b.** One of this chunk's
findings is that C10b-3e's reuse-heavy Gate 4 rows measure a 220-simulation search.

### C12-1 — Nsight Systems: the GPU busy fraction, and the number it corrects

One 20,000-simulation search on the fresh midgame root at the shipping W=1/K=24, with
the setup, the capture and the warmup outside the NVTX region every figure is
restricted to. Profiling overhead is not assumed small: the same search runs at
14,022 sims/s under nsys against 14,003 unprofiled, i.e. **0.1%**.

`GPU busy` is the **union** of every device interval — graph executions and memcpys —
not their sum. Summing lets a 90%-busy device report 130% when streams overlap.

| | pre-C12 code | **shipped C12 code** |
|---|---:|---:|
| region wall | 1,426.2 ms | 1,313.2 ms |
| GPU busy | 1,049.3 ms | 1,015.5 ms |
| **GPU busy fraction** | **73.6%** | **77.3%** |
| GPU idle | 377.0 ms (26.4%) | 297.7 ms (22.7%) |
| graph launches | 835 | 835 |
| graph execution, p50 | 1,253.1 µs | 1,208.9 µs |
| gap between launches, p50 | 441.8 µs | 342.8 µs |
| gaps, total | 381.8 ms = 26.8% | 302.8 ms = 23.1% |

**The GPU is 73.6% busy, not 92%.** C10b's grid reported `GPU share` as
`call_ns / wall_ns` — the fraction of the cycle the dispatcher spends *blocked inside
the callback* — and read a pipelining ceiling of 1.08–1.11x off it. That number is
correct for what it measures and wrong for what it was used for: the callback contains
its own host work, so being inside it is not the same as the device being busy. The
measured ceiling is **1/0.736 = 1.36x**, three times the payoff C10b's proxy implied.
This is exactly the disambiguation the brief said Step 1 would produce, and it changes
the answer. C12-8 then decomposes it and declines it anyway, for a different reason.

**Copies are not a term.** Over the whole search: 835 H2D totalling 4.6 MiB in 0.94 ms,
and 1,670 D2H totalling 138.6 MiB in 4.33 ms. All of it is 0.4% of wall. The 4096-wide
policy row crossing the bus instead of scope §2.5's 64-wide gather costs nothing
measurable, confirming C10b-3a's conclusion from the other side.

**Host-side CUDA API cost, and why it is the shape of the gap.** From the same trace,
the host duration of each CUDA call (pre-C12 code):

| CUDA runtime API | calls | total ms | p50 µs | p90 µs | max µs |
|---|---:|---:|---:|---:|---:|
| `cudaStreamSynchronize` | 1,670 | 1,036.9 | 308.5 | 1,252.9 | 1,407.5 |
| `cudaMemcpyAsync` | 2,645 | 54.2 | 17.8 | 39.8 | 175.5 |
| `cudaGraphLaunch` | 835 | 48.2 | **53.6** | 72.8 | 199.6 |
| `cudaLaunchKernel` | 201 | 5.8 | 26.3 | 38.1 | 174.3 |
| `cudaStreamIsCapturing` | 835 | 0.9 | 0.7 | 1.2 | 26.8 |

`cudaStreamSynchronize` is the GPU wait and belongs to the device. The rest is 103.3 ms
= **7.2% of wall spent submitting work**, and the line that matters is the third:
**a graph launch costs 53.6 µs of host time on this machine.** That is a WDDM property —
submission goes through the OS scheduler — and it is why the callback's `replay` phase
in C12-3 is ~40–50 µs for a call that enqueues one node and returns. It is also why
`cudaMemcpyAsync` costs 17.8 µs to move 96 bytes.

**The reuse-heavy regime is not GPU-bound at all.** Six plies into the endgame the same
20,000-simulation budget produces **one** boundary crossing of two rows: GPU busy 26.3%
of a 2.5 ms region. Whatever is true of the network is irrelevant there, and every
optimisation below is a fresh-root optimisation. See C12-4 for why that search is 2.5 ms
long in the first place.

### C12-2 — Nsight Compute: the kernel breakdown, and what is actually limiting

`--set full`, `--graph-profiling node`, restricted by `--nvtx-include` to one
steady-state forward. Collection needs administrator rights on this machine
(`ERR_NVGPUCTRPERM`; `RmProfilingAdminOnly` is unset, i.e. the default of 1), so
`tools/run_ncu_c12.ps1` exists and was run elevated; parsing needs nothing.

**One forward is 151 kernel launches of 20 distinct kernels.**

| class | launches | share @ shape 24 | share @ shape 128 | achieved occupancy @ 24 | @ 128 | compute SoL @ 24 | DRAM SoL @ 24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GEMM (cutlass bf16 tensorop) | 29 | **48.0%** | **53.2%** | 12.9% | 13.6% | 36.0% | 12.7% |
| elementwise / copy | 103 | **36.3%** | **31.4%** | 69.8% | 86.1% | 33.1% | 37.9% |
| attention (`fmha_cutlassF`) | 6 | 8.6% | 8.6% | 27.3% | 31.8% | 35.0% | 20.8% |
| normalisation (`vectorized_layer_norm_kernel<float,float>`) | 13 | 7.2% | 6.9% | 68.2% | 73.4% | 51.5% | 36.5% |

Device time is 1,947 µs at shape 24 and 8,310 µs at 128 **under ncu**, which serialises
and replays every kernel; the uninstrumented figures are 1,209 µs and ~6,200 µs (C12-1,
C10b-3a). Use ncu for the shape of the distribution and nsys for its magnitude.

The largest kernels at shape 24, and the one column that decides item (a):

| kernel | calls | device µs | share | grid | block | achieved / **theoretical** occupancy | compute SoL |
|---|---:|---:|---:|---|---|---:|---:|
| `cutlass::Kernel2<...64x256...>` | 12 | 386.6 | 19.9% | (56,1,1) | (128,1,1) | 8.3% / **8.3%** | 37.1% |
| `cutlass::Kernel2<...256x128...>` | 6 | 318.0 | 16.3% | (104,1,1) | (256,1,1) | 16.6% / **16.7%** | 33.5% |
| `cutlass::Kernel2<...64x64...>` | 6 | 206.2 | 10.6% | (208,3,1) | (128,1,1) | 15.6% / **16.7%** | 39.6% |
| `at::unrolled_elementwise_kernel<direct_copy...>` | 39 | 185.5 | 9.5% | **(1,1,1)** | (128,1,1) | 49.0% / 100.0% | 19.5% |
| `at::elementwise_kernel` | 19 | 178.3 | 9.2% | (1224,1,1) | (128,1,1) | 82.3% / 100.0% | 40.8% |
| `fmha_cutlassF_bf16_aligned_64x64_rf_sm80` | 6 | 166.7 | 8.6% | (2,6,24) | (32,4,1) | 27.3% / 33.3% | 35.0% |

**Achieved occupancy equals theoretical in every GEMM row, so occupancy is not the
lever.** 8.3% of 48 warps per SM is 4 warps, which is exactly one 128-thread CTA: the
cutlass tiles are limited to one block per SM by their own register and shared-memory
footprint, by design, trading occupancy for register-resident accumulators. At shape 128
the grid is 544 blocks over 48 SMs — 11 full waves, no quantisation excuse — and compute
Speed-of-Light is still 38.9%. The forward is **not** occupancy-starved and **not**
bandwidth-starved (DRAM SoL 8.9–17.7% on the GEMMs); it is a small-K problem, K = 384
over 1,632 tokens, where each tile does few MMA steps and spends its time on prologue
and epilogue.

**The brief's roofline diagnosis is confirmed and made specific: 52.0% of device time at
shape 24 (46.8% at 128) is not a GEMM at all.** That is the activation materialisation.
Of it, `aten::_to_copy` is 54 launches / 137.7 µs, and `aten::clone` + `aten::contiguous`
are 30 launches / ~174 µs — `nn.MultiheadAttention` reshaping q/k/v around SDPA.

**Item (a)(2) is banked, not available.** Attention already routes through
`scaled_dot_product_attention`: the kernel is `fmha_cutlassF_bf16_aligned_64x64_rf_sm80`,
i.e. the memory-efficient SDPA backend, reached via
`aten::_scaled_dot_product_efficient_attention`. There is no matmul+softmax to replace.

### C12-3 — Every millisecond of a fresh-root move, named

One search: 19,999 delivered simulations, 835 crossings, 21.2 rows each, W=1/K=24, on
the shipped code. Phases timed from inside a `TorchEvaluator` subclass so they are
measured together rather than reconciled afterwards.

| line | total ms | share of wall | per crossing µs |
|---|---:|---:|---:|
| graph replay wait (the `policy` D2H blocks until the forward is done) | 1,043.2 | **76.1%** | 1,249.4 |
| C++ search outside the callback (descent, expand, backup, tokenize, probe) | 179.8 | **13.1%** | 215.4 |
| H2D of the token rows + pad-tail restore (`stage`) | 52.9 | 3.9% | 63.4 |
| value D2H (`value`) | 48.9 | 3.6% | 58.6 |
| `cudaGraphLaunch` (`replay`, submission only) | 38.6 | 2.8% | 46.2 |
| Python interpreter overhead inside the callback | 6.4 | 0.5% | 7.7 |
| dispatcher GIL acquire wait | 1.5 | 0.1% | 1.8 |
| **wall** | **1,371.4** | **100%** | 1,642.4 |

**The residual is zero by construction and is not evidence.** The C++-search line is
computed as `wall - call_ns - acquire_ns` and the Python line as
`call_ns - (the four phases)`, so the column sums to the wall no matter what is true.
What makes the table checkable is C12-1, which measures the same quantities from the
CUDA trace instead of from these timers: **76.1% here against 77.3% GPU busy there**, and
7.2% of wall in CUDA API submission there against 10.3% in `stage`+`replay`+`value` here
(the difference being the Python and torch-dispatch time those three phases also
contain). The two independent decompositions agree to about a point.

The GIL is not a term — 1.5 ms of 1,371, p50 1.8 µs per crossing — which closes out the
C10 contingency for the third time and the last.

### C12-4 — The reuse-heavy Gate 4 rows in C10b measure 220 simulations

`search_parallel(N)` runs to **N root visits**, not N new simulations:
`target_ = num_simulations - existing` (`cpp/search.hpp`), and `ParallelStats::requested`
is set to that *remainder*. So `delivered == requested` — the assertion every C10b table
is gated on — is satisfied by a search that did almost nothing, and cannot detect it.

Replaying C10b's construction exactly (six plies at an absolute 20,000, then a measured
search at an absolute 20,000):

| | inherited root visits | delivered | wall | reported sims/s |
|---|---:|---:|---:|---:|
| C10b-3e's spelling, reproduced | 19,780 | **220** | 2.0 ms | 108,911 |
| production's spelling (`root_visits + 20,000`) | 80,510 | **20,000** | 59.8 ms | 334,315 |

**It is also not how the engine plays.** `playing/uci_wrapper_v6.py` sets
`budget = current + cfg.sim_cap` for both timed and infinite searches; only a literal
`go nodes N` uses an absolute target. C10b's published 92,639 sims/s is therefore 220
simulations divided by a wall clock that is mostly fixed per-call cost, and its
five-repeat spread of 22,924–177,657 at W=8/K=4 is what a 220-sample looks like.

`tools/bench_c12.py` asserts `delivered > 0` as well as `delivered == requested`, which
is the check that would have caught this. The first version of C12's own table published
a **zero** in this cell for the mirror-image reason — it built the tree with one spelling
and measured it with the other — which is recorded in DECISIONS.md rather than quietly
fixed.

### C12-5 — Gate 4, both regimes, delivered simulations

W=1/K=24, `max_batch` 128, virtual loss 2.5, `verify_compaction` **off**, 5 repeats
(median), shipped code. No opening book and no tablebase are attached on the library
path, so **no move is bypassed and there is nothing to exclude from the rate**.

| regime | inherited | delivered | wall | **delivered sims/s** | min | max | rows/crossing | cache hit | vs floor (8k) | vs stretch (15k) | vs Python (838) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **fresh midgame root** | 1 | 19,999 | 1,303.5 ms | **15,342** | 15,207 | 15,484 | 21.2 | 11.3% | **1.92x** | **1.02x** | **18x** |
| **reuse-heavy endgame** | 80,510 | 20,000 | 59.8 ms | **334,315** | 323,557 | 340,134 | 7.9 | 66.8% | **41.79x** | 22.29x | **399x** |
| *reuse-heavy, C10b's spelling (220 sims — not a rate)* | 19,780 | 220 | 2.0 ms | *108,911* | *76,301* | *112,867* | 2.0 | 84.6% | — | — | — |

**Gate 4 PASSES in both regimes, and the stretch target is now met on a fresh root** —
15,342 against 15,000, where C10b measured 13,912. The floor is cleared by 1.92x.

### C12-6 — What shipped, and the proof it changed no bit

Three changes, all in Python, all on the host side of the boundary, **all bit-exact by
construction and verified so**.

1. **`hoist_norm_parameter_casts`** (`playing/v6/evaluator.py`). Autocast keeps
   `layer_norm` on its fp32 list, so every forward widens both affine parameters of all
   13 LayerNorms to fp32 — 26 kernel launches of 384 elements each, at grid **(1,1,1)**,
   recomputing a constant. bf16 to fp32 is an exact widening, so storing the result once
   produces exactly the bits the runtime cast would have. Called before capture, because
   a graph records the kernels it was captured with.
2. **Cached tensor views** (`GraphedForward._build_view_cache`, `TorchEvaluator._row_views`).
   `tokens[:count]`, `_policy[shape][:count]`, `_input_t[:count]` and the two output
   slices are five Python objects and five dispatcher round trips per crossing, for views
   onto storage that CUDA graph capture already requires never to move. There are only
   `max_batch` of each; they are built once.
3. **The two D2H copies reordered.** The value copy is issued first and asynchronously;
   the policy copy is blocking and, being on the same stream, completes both. Two
   blocking copies cost two `cudaStreamSynchronize` round trips, and C12-1 measures a
   round trip at ~50 µs against 96 bytes of payload.

| | delivered sims/s (median of 15) | range | paired |
|---|---:|---|---:|
| pre-C12 | 14,479 | 14,263 – 14,741 | — |
| **shipped** | **15,420** | 15,203 – 15,597 | **1.065x**, 15/15 wins |

Interleaved, not two blocks, because the run-to-run spread is ~4% and drifts with GPU
clock; two blocks of 15 would compare two machine states. Isolated, on the graphed
forward's *device* time via CUDA events: change 1 alone is **1.024x at shape 24** and
1.009x at 128 — the gap between the two is the point, because shape 24 is what the
engine ships at.

**Root flattening effect: none, and it is measured rather than argued.** The C12
acceptance criterion asks every optimisation to report it, so:

* the root visit vector is **bit-identical** pre- and post-change on every A/B pair;
* `tests/test_c10b_graphs.py` finds **0 differing policy words** of 2,048,000 at every
  captured shape, against the eager forward;
* the entire top-move-share and TV column of C12-7 reproduces digit for digit between
  the pre-C12 and shipped runs (78.7 / 75.1 / 71.2 / 70.3 / 65.9 / 60.8 / 53.9 / 46.9).

A throughput gain that changed the search would show up in all three. None of them moves.

### C12-7 — Item (b): outstanding leaves, and why the tolerance does not settle it

W=1, 20,000 **new** simulations on the fresh midgame root, 5 repeats, shipped code.

| K | delivered sims/s | vs K=24 | rows/crossing | pad waste | **top-move share** | TV vs K=8 | run-to-run TV |
|--:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 9,637 | 0.62x | 7.1 | 1.12x | 78.7% | 0.0% | 0.0% |
| 16 | 13,101 | 0.85x | 14.2 | 1.12x | 75.1% | 3.6% | 0.0% |
| **24** | **15,453** | **1.00x** | 21.2 | 1.12x | **71.2%** | 7.5% | 0.0% |
| 32 | 15,657 | 1.01x | 28.1 | 1.13x | 70.3% | 8.6% | 0.0% |
| 48 | 16,822 | 1.09x | 42.0 | 1.14x | 65.9% | 12.8% | 0.0% |
| 64 | 16,844 | 1.09x | 56.1 | 1.14x | 60.8% | 17.9% | 0.0% |
| 96 | 17,414 | 1.13x | 83.8 | 1.14x | 53.9% | 24.8% | 0.0% |
| 128 | 18,037 | 1.17x | 110.1 | 1.15x | 46.9% | 31.8% | 0.0% |

**The brief says item (b) is "blocked on C10b's absolute root-stability tolerance", and
C10b set that tolerance at 10% run-to-run TV. Every row above passes it, and that
settles nothing.** At W=1 the search is deterministic, so run-to-run TV is 0.0% at every
K by construction — the tolerance is a *reproducibility* bound, and the quantity that
actually moves is *flattening*, which is a different column. Item (b) is not unblocked;
it was blocked on the wrong unknown.

The real exchange rate is in this table: **1.09x throughput costs 10.4 points of
top-move share (K=24 to K=64), and 1.17x costs 24.3 points.** C10b-3g declined K=32 for
2.4 points at a 1.74x Gate 4 margin. At a 1.92x margin and with the stretch target met,
the same principle declines K=64 by a wider margin than it declined K=32.

**Recommendation: K stays 24.** What would change it is Gate 5-class evidence — a
time-controlled match at K=24 against K=64 — and the brief's own text says that decision
belongs to a match, not to a throughput table. It is handed to C13.

### C12-8 — Item (c): the pipelining ceiling, decomposed, and declined

C12-1 raised the ceiling from C10b's 1.09x to 1.36x. Decomposing the 22.7% idle on the
shipped code says what would actually be recoverable:

| the idle is | share of wall | recoverable by |
|---|---:|---|
| C++ search — descent, expansion, backup, tokenize, cache probe | 13.1% | **raising outstanding leaves**, i.e. item (b) |
| CUDA submission — `stage` + `replay` + `value` | 10.3% | pipelining the dispatcher stages |
| Python interpreter + GIL wait | 0.6% | neither |

**Items (b) and (c) are not independent, and this is the finding that decides item (c).**
At W=1/K=24 the single worker has all 24 of its leaves outstanding, so it has nothing to
descend while the GPU runs; the only way to overlap that 13.1% is to give it more leaves
in flight, which is item (b) and costs root sharpness. Pipelining the dispatcher stages —
what the brief actually asks for, and correctly separates from raising the budget —
reaches the other 10.3%, so **its true ceiling is 1/(1 - 0.103) = 1.11x**, not 1.36x.

**Declined.** 1.11x is an upper bound assuming perfect overlap, on a C++ change to the
dispatcher in `search.hpp` with determinism and TSan consequences, against an engine that
already clears the stretch target. Its stated prerequisite is already met and cost
nothing: `TorchEvaluator._pin_buffers` page-locks **all three** buffers through
`buffer_spans()`, not just the input, so the pageable-D2H hazard the brief flags does not
exist here. The brief's own escape clause — "if the increased virtual loss staleness
costs more sharpness than the throughput is worth, ship without pipelining" — points the
same way, and no measurement of staleness was needed to get there.

### C12-9 — Item (a): blocked by Gate 2, and the ruling that would unblock it

**`torch.compile` is not in the shipped path**, so nothing is banked from it: C10b built
it, measured it 1.23–1.47x faster, and **rejected it because it fails Gate 2 by four
orders of magnitude** — 1,773,671 of ~2,621,440 policy words differing (C10b-1b). The
shipped forward is a manual capture of the eager ATen ops, unfused.

The C12 brief's premise for item (a) is that *"fusing the existing bf16 ops preserves the
gate; changing precision is a deliberate re-baselining."* **That premise is false on this
model, and C10b already measured it false.** Inductor changes no precision — it keeps
bf16 throughout and fuses bias/GELU/LayerNorm into the matmul epilogue — and a bf16 logit
moved by one ulp moves a prior by ~1e-3, three orders over the 1e-6 bound. Fusion *is*
re-baselining here, so (a)(1) and (a)(3) are blocked for the same reason (a)'s precision
clause blocks fp16. (a)(2) is banked (C12-2). (a)(4) has no dominant kernel to name — the
largest is 19.9% — only a dominant *class*.

Per the brief's instruction, this is **halted pending a ruling** and scoped in
`docs/c12_kernel_fusion_scope.md`: Option A do nothing (recommended), Option B adopt
Inductor and re-baseline Gate 2 (~1.19x delivered sims/s, ~2 points, costs the gate's
meaning and Global Rule 2's golden provenance), Option C bit-exact data-movement Triton
(~1.05x, ~8 points).

### Gate 2, re-passed on the changed forward

`pytest tests/test_c10b_graphs.py -q -s` — 14 passed. Every cell reproduces C10b-2
exactly, which is the expected consequence of the change being bit-exact:

| comparison | max abs delta | p99 | exact | over 1e-6 | inversions |
|---|---:|---:|---:|---:|---:|
| graphed vs reference interior (ATen CPU, python-chess order) | 2.384e-07 | 5.364e-08 | 3,986 | **0** | **0** |
| graphed vs reference root (ATen CUDA, python-chess order) | 2.384e-07 | 4.470e-08 | 3,341 | **0** | **0** |
| graphed vs ATen CPU in chess-library order | 2.682e-07 | 5.960e-08 | 3,865 | **0** | **0** |

Graphed vs eager: **0 differing policy words** of 2,048,000 at shape 1 and of ~1.6–2.0M
at each of shapes 8, 16, 24, 32, 48, 64, 96, 128.

### Measurement notes

* `tools/profile_c12.py` and `tools/bench_c12.py` both **refuse to run on an instrumented
  build**, for the reason recorded in DECISIONS.md, C10.
* The NVTX region excludes evaluator construction, graph capture, the reuse plies and a
  throwaway warmup search, so no figure here includes first-touch on the arena's pages or
  a cold transposition cache.
* Per-batch NVTX markers are **off by default** (`--nvtx-batches` turns them on). They sit
  inside the GIL-held callback; the per-batch structure in C12-1 is recovered from the
  CUDA trace's own kernel timestamps instead, which costs the callback nothing.
* `--cuda-graph-trace=node` costs 5.4% (13,262 sims/s against 14,022) and is used only for
  the kernel breakdown. Every timing figure comes from a `graph`-granularity capture.
* ncu figures are collected with kernel serialisation and replay, so device times there
  are ~1.6x the uninstrumented ones. Shares, occupancy and Speed-of-Light are what that
  report is quoted for.
* Gate 4 rows report **delivered** simulations and assert `delivered == requested`,
  `delivered > 0`, `vloss_total == 0`, zero conservation failures and
  `arena_exhausted == false` before the row is admitted.

## C12b — Inductor adopted, Gate 2' re-based, and the gate that did not pass

Every figure here is on **`models/guofish5_90M/v5_10.9M_best.pt`** — the checkpoint the
engine ships (`playv6.DEFAULT_MODEL`), not the 20M-corpus net the golden gates are anchored
to. Same v5 architecture (10,887,681 parameters, 68 tokens, `d_model` 384 x 6), so the
captured ladder is unchanged and only the weights differ. README_BUILD.md's Golden data
section carries the full anchoring table.

**Gate 2' passes as ruled, having been reported failed as briefed.** The throughput,
determinism and no-recompilation criteria are met outright. The move-agreement criterion
missed the briefed 99% at 98.65%; the owner ruled 98% sufficient after adjudicating the
decisive disagreements against Stockfish. C12b-6 has the numbers, the measurement is
unchanged, and DECISIONS.md has the sequencing.

### C12b-1 — Default-mode `torch.compile` can be captured by the C10b machinery

The brief asks for this to be reported as a finding if it could not be. It can.

`torch.compile(model, dynamic=False)` in **default mode** — Inductor codegen, no
compile-owned cudagraphs — captured by the existing `pad_to` / static-buffer / `replay`
machinery, at every one of the nine shipped shapes. `mode="reduce-overhead"` is not used
and must not be: it brings Inductor's own cudagraph trees, which would replace the padding
and staleness machinery that `tools/drill_c10b_graphs.py` bites on.

| property | result |
|---|---|
| shapes captured | 9 of 9 (`1, 8, 16, 24, 32, 48, 64, 96, 128`) |
| captured replay vs un-captured compiled call | **0 differing policy words** at every shape |
| dynamo frames compiled during capture | **0** |
| warmup rounds to convergence | **2 per shape** — one compiles, one confirms |

Two settings are required rather than preferred, and both are set by `configure_inductor()`:

* **`use_static_cuda_launcher = False`** — with torch 2.8.0+cu129 / triton 3.4.0 on this
  RTX 5070 the static launcher raises `OverflowError: Python int too large to convert to C
  long` on the first Inductor kernel. C10b hit the same thing.
* **`triton.autotune_pointwise = False`** — see C12b-2.

### C12b-2 — Determinism, and why the autotune cache is removed rather than pinned

The brief requires bit-identical priors across runs and asks for the autotune cache to be
pinned or shipped. Measured, the cache is the hazard, so it is deleted instead of pinned.

| configuration | `.best_config` files written | two cold compiles agree? |
|---|---:|:--|
| `triton.autotune_pointwise = True` (torch default) | 28 | **no — 17 of 28 configs differ** |
| `triton.autotune_pointwise = False` (**shipped**) | **0** | **yes, 3 of 3 cold compiles bit-identical** |

With autotuning on, Inductor benchmarks several Triton configs per pointwise/reduction
kernel at first call and caches the winner it *timed* fastest. The differing picks are real
schedule changes — XBLOCK 128 against 256, XBLOCK 1024 / 4 warps against 512 / 8 warps —
and a reduction kernel's block size changes the accumulation order, hence the bits. A warm
default cache and a fresh one disagreed at shape 24 in exactly this way.

**It costs nothing to turn off**, which is what makes the choice easy:

| | device time, shape 24 | capture |
|---|---:|---:|
| autotuning on | 970.3 us | ~15 s slower |
| autotuning off | **964.6 us** | — |

Determinism as the acceptance criterion states it — *across runs*, i.e. across processes:

| check | result |
|---|---|
| two captures in one process | bit-identical policy words |
| **a fresh process** (`tools/c12b_forward_digest.py`) | **bit-identical at all four digested shapes** |
| 520-position corpus, every captured shape | identical sha256 per shape |

Shipping a pinned cache directory was rejected on three counts: it is an artifact that can
drift from the code, it is keyed on device/driver/torch, and on Windows the Triton cache
manager hits `MAX_PATH` and fails outright when the cache directory is more than a few
dozen characters deep — hit in practice while measuring this.

### C12b-3 — The graphed forward's device time, and the fusion term

CUDA events, 200 timed replays after 20 warm ones, `max_batch` 128. Device time is where
the change is; C12b-4's sims/s is this after the dispatcher, the C++ descent and the
padding have diluted it.

| shape | eager us | inductor us | speedup | policy words differing | max abs dlogit |
|--:|---:|---:|---:|---:|---:|
| 1 | 267.7 | 205.9 | 1.300x | 2,656 of 4,096 (64.8%) | 0.04688 |
| 8 | 633.9 | 471.9 | 1.343x | 20,848 of 32,768 (63.6%) | 0.03906 |
| 16 | 943.9 | 1,010.1 | **0.934x** | 39,872 of 65,536 (60.8%) | 0.04688 |
| **24** | 1,232.1 | 963.8 | **1.278x** | 59,808 of 98,304 (60.8%) | 0.04688 |
| 32 | 1,658.6 | 1,400.9 | 1.184x | 79,648 of 131,072 (60.8%) | 0.04688 |
| 48 | 2,317.2 | 1,833.3 | 1.264x | 119,616 of 196,608 (60.8%) | 0.04688 |
| 64 | 3,121.4 | 2,479.9 | 1.259x | 159,488 of 262,144 (60.8%) | 0.04688 |
| 96 | 4,547.9 | 3,498.6 | 1.300x | 239,232 of 393,216 (60.8%) | 0.04688 |
| 128 | 5,903.8 | 4,596.8 | 1.284x | 318,976 of 524,288 (60.8%) | 0.04688 |

**Shape 16 regresses and is reported rather than tuned.** It is reproducible across every
run and every `max_batch`, and it is not an eager fallback — 39,872 words differ, so
Inductor is running; it is simply a worse schedule at that width. K=24 rounds to shape 24,
so the shipping configuration rarely lands there. The ladder is C10b's and re-tuning it is
not this chunk's remit.

**The fusion term is the whole point of the chunk**: ~61-65% of the bf16 policy words move
at every shape, at up to 0.047 of a logit. That is what re-bases the numerics and what
Gate 2' exists to re-certify.

#### The bug this table found: dynamo's recompile limit silently ran shape 128 eager

`torch._dynamo`'s `recompile_limit` defaults to **8**. The shipping ladder has **nine**
shapes and `dynamic=False` gives each its own specialised frame, so the ninth blew the
limit — and dynamo does not raise, it logs a warning and falls back to running that shape
**eager, permanently**. The engine ships `max_batch=128`, so the shape that fell back was
the largest one. Before the fix:

    shape 128   eager 5,912.6 us   inductor 5,912.8 us   1.000x   0 of 524,288 words differ

The capture succeeded, the priors were correct and the recompilation counter was *stable*
(nothing was being compiled any more, which is the problem). Only the throughput row
showed it. `_raise_recompile_limit` now raises the limit to `len(sizes) + 8`, and
`assert_every_shape_is_fused()` checks the property semantically — every shape's output
must differ from the unfused module's — because counting compiled frames cannot express it.

### C12b-4 — Gate 4 on the Inductor forward, both regimes, REPORTED SEPARATELY

W=1, K=24, `max_batch` 128, virtual loss 2.5, `verify_compaction` off, 5 repeats (median),
arms interleaved repeat by repeat so the ~4% GPU-clock drift divides out of the paired
ratio instead of adding to it. `delivered > 0` and no-recompilation are asserted before a
row is admitted.

| regime | arm | inherited | delivered | wall | **delivered sims/s** | min | max | rows/crossing | pad waste | cache hit | paired speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh midgame root | `compile=False` | 1 | 19,999 | 1,326.3 ms | **15,079** | 14,912 | 15,200 | 21.6 | 1.11x | 9.9% | — |
| fresh midgame root | `compile=True` | 1 | 19,999 | 1,108.8 ms | **18,036** | 15,501 | 18,406 | 21.6 | 1.11x | 9.9% | **1.189x**, 5/5 wins |
| reuse-heavy endgame | `compile=False` | 57,726 | 20,000 | 1,306.4 ms | **15,310** | 12,518 | 15,451 | 18.2 | 1.23x | 24.1% | — |
| reuse-heavy endgame | `compile=True` | 55,080 | 20,000 | 1,142.9 ms | **17,499** | 15,727 | 17,808 | 18.1 | 1.23x | 24.6% | **1.151x**, 5/5 wins |

**There is deliberately no combined number**, per the brief's Part 4. Gate 4's floor is
8,000 delivered sims/s and its stretch target 15,000; the Python reference measured 838.
Both regimes clear the stretch target on both arms.

**1.189x on a fresh root matches the brief's ~1.19x prediction.** What does not match is the
reuse-heavy regime.

#### The reuse-heavy regime is not the one C12 measured, and the checkpoint is why

C12-5 reported the reuse-heavy endgame at **334,315** sims/s with a **66.8%** cache hit
rate — the cheap regime where the GPU is 26.3% busy and Inductor should buy ~nothing. On
the 90M checkpoint that regime does not reproduce: same positions, same budget, **24.1%**
cache hit and 15,310 sims/s eager. So a `vs C12-5` ratio is not a comparison and is not
quoted.

The cause is the tree, measured over the six reuse plies at 20,000 NEW simulations each:

| ply | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| 20M nodes | 261,111 | 390,075 | 415,635 | 203,807 | 156,615 | 156,126 |
| 90M nodes | 308,188 | 530,925 | 646,246 | 654,081 | 864,057 | 1,104,746 |

The 20M net's reuse tree peaks at ply 3 and then *shrinks* as the endgame simplifies. The
90M net's grows every ply and reaches **1,436,625** nodes by the measured search. So on the
shipped net the two regimes are much closer together than the brief anticipated, and
Inductor buys 1.151x under reuse rather than nothing.

This also exposed a harness bug: `bench_c10b.ARENA_CAPACITY = 1,200,000` predates C11c and
coincides with `60 x 20,000`, i.e. C11c's rule for **one** search — but the reuse arm runs
seven searches on one accumulating tree. At 1.2M the measured search exhausts the arena and
delivers 6,666 of 20,000, which `timed_search` correctly refuses to publish.
`tools/bench_c12b.py` now derives the arena from the rule, `60 x sims x (reuse_plies + 1)`.
**The engine's own sizing is not implicated**: per search the 90M net builds 308,188 nodes
for 20,000 simulations — **15.4 nodes/sim against the 60 the rule assumes**, ~4x headroom.

### C12b-5 — What adoption costs at engine start

| arm | method | capture | construct | device memory | warmup rounds per shape |
|---|---|---:|---:|---:|---|
| `compile=False` | `cudagraph` | 0.6 s | 0.9 s | +730 MiB | — |
| `compile=True` | `inductor+cudagraph` | 7.7 s | 9.6 s | **+326 MiB** | 2 at every shape |

`ensure_ready` pays this once per process, and the figures are for a **warm** Inductor
cache; a cold cache (new machine, new GPU, new torch) costs roughly 30 s more. The device
memory is the surprise and it goes the right way: Inductor's captured graphs reserve **less
than half** what the eager captures do, because the fused epilogues materialise fewer
intermediate activations into each graph's private pool.

### C12b-6 — Gate 2': the numbers, and the two criteria it does not meet

520 positions (Gate 2b's 500 plus the 20 Gate 1 positions, disjoint), 1,600 simulations,
W=1/K=1, against the eager baseline frozen at tag `GUOFISH_NUMERICS_BASELINE`.

**The prior shift — reported, not gated.** This is the re-baselining, measured:

| quantity | value |
|---|---|
| priors compared | 15,764 over 520 positions |
| bit-identical | **0 (0.0%)** |
| max abs dprior | **9.116e-03** — 9,116x Gate 2's 1e-06 bound |
| p99 / median abs dprior | 2.598e-03 / 2.216e-05 |
| per-position Linf: max / p99 / median | 9.116e-03 / 6.254e-03 / 1.577e-03 |
| per-position L1: max / p99 / median | 1.830e-02 / 1.448e-02 / 5.768e-03 |

**Inversions — measured, not gated**, exactly as the brief directs:

| quantity | value |
|---|---|
| inverted pairs | **170 of 269,257 comparable (0.063%)** |
| baseline gap at an inverted pair: max / p99 / median / min | 2.570e-03 / 1.746e-03 / 3.699e-05 / 4.499e-06 |
| widest | 2.570e-03 at `5bk1/6pp/3pbp2/8/3B4/6P1/2p1PP1P/R5K1 b - - 0 34` |

Every inversion sits far inside the `2 x max|dprior| = 1.823e-02` a shift of this size can
cross, so the measurement is self-consistent. The minimum inter-prior gap on this corpus is
1.927e-06 (C10b), so a ~1e-3 shift inverting near-ties is arithmetic, not a defect.

**The move-agreement criteria — as briefed, as measured, and as RULED:**

| criterion | as briefed | measured | as ruled |
|---|---|---|:--|
| move agreement | >= 99% | **98.65%** (513/520) | **>= 98% — PASSES** |
| disagreements outside a 2% margin | 0 | **5 of 7** | **reported, not gated** |

The ruling (owner, 2026-08-13) followed a hand adjudication of the five decisive
disagreements below against Stockfish at high depth: **every Inductor move was equal to or
better than the baseline's.** The criteria were held, the chunk was reported blocked, and
the budget escape was tested and refuted before the ruling was made — see DECISIONS.md.

    c10_corpus   494/500 = 98.80%
    gate1         19/20  = 95.00%

Only 20 of 520 positions (3.85%) were decided by the baseline within 2%; 2 of those 20
disagreed, and **5 disagreements fall outside that set entirely**. The five decisive ones:

| position | baseline | inductor |
|---|---|---|
| `r5k1/5pp1/1p2p1b1/6Np/PP1p3P/Q7/2q2PP1/R5K1 w - - 0 36` | a4a5 (27.0%) | a1c1 (16.2%) |
| `r3k2r/1b1n1ppp/pq2p3/3nP1B1/BP6/8/P3QPPP/3R1RK1 w kq - 3 21` | e2g4 (48.8%) | e2h5 (6.9%) |
| `r1b1kb1r/pp1p1ppp/1q2pn2/8/1n1P1B2/1Q2PN2/PP3PPP/RN2KB1R b KQkq - 2 8` | a7a5 (46.6%) | b4d5 (51.5%) |
| `rn1qkb1r/pp3ppp/2b1p1n1/4P3/3p4/1P1B1N2/P2P1PPP/RNBQR1K1 w kq - 0 10` | c1b2 (45.5%) | g2g3 (40.7%) |
| `r4bk1/pp1b1pp1/2n2q1p/3p4/5n2/1NPB1N1P/PP3PPB/R2Q2K1 w - - 5 18` | d3c2 (2.4%) | h2f4 (7.1%) |

#### The near-tie criterion has never been met, at any numerical distance

`tests/test_c10_gate2b.py::test_every_disagreement_is_a_near_tie` has been **red since C10**
— it is the single failure in C11's whole-suite figures, and C11b's record shows the owner
dropped it from scope rather than fixing it. Re-running the (normally deselected)
differential on this tree reproduces C10 exactly:

| | C10 recorded | this run |
|---|---|---|
| agreement | 497/500 = 99.4% | **497/500 = 99.4%** |
| decisive disagreements | 2 | **2, the same positions, the same margins** |

So the Python-to-C++ differential has not moved, and **the near-tie bound fails between two
engines that agree to 1e-6**. It is a property of a corpus containing chaotic positions — a
perturbation anywhere flips an early selection and both trees then converge confidently
elsewhere — not a bound Inductor broke. Gate 2 could hold 1e-6 on the *priors*; nothing has
ever held a near-tie bound on the *moves*.

Only the **>= 99% agreement** criterion is one Inductor is the first to miss, and it misses
it by seven positions.

#### The budget hypothesis, tested on the whole corpus and REFUTED

Sampling four decisive positions across budgets suggested 1,600 was simply not converged —
the eager arm **alone**, with no Inductor anywhere, changes its own answer on three of four:

| position | 1,600 | 3,200 | 6,400 | 12,800 |
|---|---|---|---|---|
| `rn1q1rk1/...R2QK2R w KQ - 0 13` | e4d5 | e3b6 | e3b6 | e3b6 |
| `3r2k1/...R1R3K1 b - - 12 23` | e5g4 | h7h6 | h7h6 | h7h6 |
| `rnbqk2r/...R2QKB1R b KQkq - 5 5` | e8g8 | e8g8 | e8g8 | c7c5 |
| `2R5/...2r5 w - - 1 49` | c5b4 | c5b4 | c5b4 | c5b4 |

**The sample did not generalise.** Both arms over the whole 520-position corpus at 6,400
simulations (`tools/sweep_c12b_budget.py`, 2,127 s) makes the gate *worse*:

| simulations | agreement | >= 99%? | disagreements | of which decisive | all near-ties? |
|---:|---:|:--|---:|---:|:--|
| 1,600 | 98.65% (513/520) | **NO** | 7 | 5 | **NO** |
| **6,400** | **97.88%** (509/520) | **NO** | **11** | **8** | **NO** |

**The divergence grows with search depth rather than converging away.** In hindsight that
is the expected direction: more simulations mean more selections for a ~1e-3 prior
perturbation to flip, and a deeper tree amplifies an early divergence instead of washing it
out. Individual positions can converge; the corpus does not. There is therefore no version
of "keep the criterion, move the budget" that rescues Gate 2', and **the criterion was not
moved** to fit the result (Global Rule 10).

The fourth sampled row is `2R5/7p/P3k3/1PK2p1p/5P1P/8/8/2r5 w - - 1 49` — **one of C10's
two known decisive positions**, where the Python reference plays c5b6 and the C++ engine
c5b4. C12b's two arms landed on the two sides of a divergence that predates this chunk.

### C12b-7 — Rule 3, the suite, Amendment D

`pytest tests/ -v -p no:randomly` on Windows, Release build, **nothing deselected** —
including the four Gate 2b differential tests that C12 deselects by convention:

    3 failed, 1456 passed, 49 skipped in 752.43s (12m32s)

| failure | status |
|---|---|
| `test_c10_gate2b.py::test_every_disagreement_is_a_near_tie` | **pre-existing since C10**, owner-acknowledged, reproduces C10's two positions exactly |
| `test_c12b_gate2prime.py::..._on_99_percent_of_moves` | **resolved by the ruling** — 98.65% against a floor now set at 98%. Renamed `test_move_agreement_against_the_baseline_meets_the_ruled_floor`, since the name was the last thing still asserting 99 |
| `test_c12b_gate2prime.py::test_every_disagreement_is_a_near_tie` | **resolved by the ruling** — converted to a report and renamed `test_the_decisive_disagreements_are_listed_for_adjudication` |

**That count is the pre-ruling run, left as measured.** Post-ruling the suite stands at
**1 failed / 1,458 passed / 49 skipped** — and that figure is derived rather than
re-measured, so here is the derivation. Nothing outside `tests/test_c12b_gate2prime.py`
was touched by the ruling; inside it only `MIN_AGREEMENT` moved and one gate became a
report. All 18 of its tests have since been executed against the changed file — 14 in the
drill's baseline run and the 4 differential ones separately (4m37s) — and all 18 pass, with
the differential reproducing 513/520 and the same five decisive positions. The search is
deterministic at W=1/K=1, so re-running the other 1,440 tests would restate them.

The one remaining red is `test_c10_gate2b.py::test_every_disagreement_is_a_near_tie`, which
predates this chunk by two chunks.

The 49 skips match C12's count exactly and are the same set. **The runtime figure corrects
C12's record**: the whole suite including the four deselected tests is 12m32s, and the
differential's C++ arm alone is 306 s. C12's "3 h 29 m" is an ASan build and its "53
minutes" is the Python reference arm, which lives in the golden generator rather than in
the test.

**The Linux arm certifies nothing in this chunk.** It has no CUDA, Inductor emits Triton,
and every test in `test_c12b_gate2prime.py` is individually `skipif`-marked with a reason
naming CUDA (Amendment D — no module-scope skips). A green Linux run here means 18 tests
skipped, not that Inductor was checked. C12's Amendment D note applies unchanged.

### C12b-8 — Mutation drill (Amendment B)

`python tools/drill_c12b.py` — **5/5 mutations caught, `golden/` and `baseline/` unchanged.**

| mutation | what it breaks | caught by |
|---|---|---|
| `no-inductor` | `compile=True` builds an `InductorGraphedForward` that captures the UNFUSED module, with the constructor's own guard disabled — the self-comparison trap | `test_the_prior_shift_is_reported` |
| `autotune-on` | restores `triton.autotune_pointwise`, the setting whose picks differ between cold compiles | `test_autotuning_is_off_because_that_is_what_pins_the_kernels` |
| `recompile-limit` | puts dynamo's limit below the captured ladder, so shapes fall back to eager silently | `test_warmup_converged_before_anything_was_captured` |
| `stale-replay` | skips the graph launch and re-reads the previous batch's outputs | `test_the_manual_capture_did_not_change_the_inductor_forward` |
| `corrupt-baseline` | one bf16 word of `policy_1` changed in a scratch COPY of the frozen baseline | `test_compile_false_reproduces_the_frozen_baseline_bit_exactly` |

**`no-inductor` is the one the drill exists for.** Every other test in the file passes if
`compile=True` silently yields the eager forward — capture fidelity compares eager against
eager, both determinism checks agree trivially, and Gate 2' reports 100% move agreement
against a baseline it merely reproduced. The assertion that closes it is a single line in
`test_the_prior_shift_is_reported`: a genuinely fused forward moves ~61% of the bf16 policy
words, so a max `|dprior|` of exactly zero means Inductor never ran.

**The drill caught a defect in itself first, and that is worth recording.** The initial
`corrupt-baseline` flipped bytes at a fixed offset near the end of the `.npz`.
`np.savez_compressed` writes members in insertion order, so the tail of that archive holds
`fens` and `source` — two arrays the gate never reads. The mutation corrupted the file,
changed nothing under test, and was correctly reported as **MISSED**. It now re-saves every
array unchanged except one word of `policy_1`. A mutation has to land on the quantity under
test to say anything about it.

Four tests are deselected from the drill: the ones sharing the 4.5-minute `differential`
fixture. Two of them are red for the reason C12b-6 reports, and a drill whose target does
not pass proves nothing; the other two are excluded on cost, since no mutation here is
aimed at search output. Both reasons are recorded in the tool so that the list is revisited
rather than inherited.
