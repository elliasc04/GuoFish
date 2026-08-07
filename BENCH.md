# BENCH

Accumulated performance numbers, one section per chunk.

Machine for all numbers below: **Intel Core i5-12600K** (10 cores / 16 threads),
Windows 11 Pro 26200. Linux figures are from WSL2 Ubuntu 24.04 on the same host.

Reproduce with `python tools/bench_c0.py`, `python tools/bench_c0b.py`,
`python tools/bench_c2.py` and `python tools/bench_c4.py` (see `README_BUILD.md`).

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
