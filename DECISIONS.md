# DECISIONS

Judgment calls the chunk briefs did not specify. Each entry: what was chosen,
what else was on the table, and why. Newest chunk last.

---

# C0 — Toolchain spike (2026-08-05)

## Defects found in the pre-existing draft

`cpp/bindings.cpp` and `tests/test_c0_toolchain.py` already existed from an
earlier session. Both were rewritten. Two of the changes were bug fixes, not
preferences, and are worth stating plainly because the old code *passed its own
tests*:

**1. Nothing was actually 64-byte aligned.** The draft declared
`alignas(64) std::vector<int32_t> data;`. That aligns the *vector object* — the
three pointers making up its header — and says nothing about the heap block the
vector allocates, which came from plain `operator new` at 16-byte alignment. The
C0 acceptance criterion "buffer memory must be 64-byte aligned" was not met, and
the draft test never checked an address, so it went unnoticed.

**2. The zero-copy base object was a double free.** The draft passed
`py::cast(g_buffer.get())` as the NumPy base. `py::cast` on a raw pointer
defaults to `return_value_policy::automatic`, which for pointers means
`take_ownership` — so Python would `delete` the `AlignedBuffer` when the array
died, while the module-level `unique_ptr` still owned the same object and would
delete it again at shutdown. See "Buffer ownership" below for the replacement.

## Memory

**64-byte alignment via C++17 over-aligned `operator new`.**
Chosen: `::operator new(n, std::align_val_t(64))` / matching sized-aligned
`::operator delete`.
Alternatives: `_aligned_malloc`/`_aligned_free` (MSVC-only), `posix_memalign`
(POSIX-only), `std::aligned_alloc` (C11; **MSVC does not provide it**, because
Windows' `free` cannot release it).
Why: it is the only spelling that compiles unmodified on both toolchains, so
Global Rule 8's "portable fallback" requirement is satisfied by having no
platform branch at all. It also has no size constraint, whereas
`std::aligned_alloc` requires the size to be a multiple of the alignment — which
matters here because the natural row width is 68 int32 = 272 bytes, not a
multiple of 64.

**Buffer ownership: `py::capsule` holding a `shared_ptr` copy.**
Chosen: the module keeps `std::shared_ptr<AlignedBuffer> g_buffer`, and each
NumPy view gets a capsule owning its own `shared_ptr` copy.
Alternatives: raw pointer + `py::cast` (the draft's double free); making
`AlignedBuffer` a `py::class_` and returning it as base (works, but exposes an
internal type to Python for no benefit); never freeing (leak).
Why: the allocation then outlives whichever reference dies first, in both
directions. Python can `del` the array while `g_buffer` still points at the
memory, and a later `make_buffer()` call can reseat `g_buffer` while an older
view is still alive. `test_old_view_survives_reallocation` covers exactly this,
and is the test the draft implementation would have crashed on under ASan.

## API shape

**No fifth binding.** The brief says the module exposes "exactly these
functions": `ping`, `make_buffer`, `buffer_checksum`, `roundtrip_bench`. The
obvious way to test alignment would be a `buffer_address()` accessor, but that
would add a function the brief did not ask for. The tests instead read
`arr.__array_interface__["data"][0]`, which is the address NumPy would hand a C
consumer — a strictly better thing to assert on anyway, since it verifies the
pointer *Python sees* is aligned rather than trusting a number C++ reports about
itself.

**`roundtrip_bench` calls `callback()` with no arguments**, per the brief. The
draft called `callback(rows)` and its test used `def dummy_callback(batch_size)`.
These are incompatible; the brief wins. This is the one place the rewritten test
is not backward compatible with the draft test.

**`roundtrip_bench` allocates its own scratch buffer** of `rows x 68` rather
than reusing `g_buffer`. Why: benchmarking must not disturb the buffer the
zero-copy tests are inspecting. 68 is the tokenizer width the port targets
(scope C2, `tokens(fen) -> int32[68]`), so the simulated work has the shape of a
real dispatcher batch rather than being an arbitrary memset.

**Dead-store defence.** The simulated C++ work writes rows that nothing reads,
which an optimiser is entitled to delete outright — leaving a benchmark of an
empty loop. The buffer is summed after the loop and the result returned as
`work_checksum` so the writes are observable. The value is not otherwise
meaningful.

## Measurement

**Timed segment excludes the GIL release.** The brief specifies the
"acquire -> call -> return" segment, so the primary `median_us`/`p99_us` sample
is taken *inside* the `gil_scoped_acquire` scope, immediately after `callback()`
returns. The release at scope exit is real cost the dispatcher pays, so it is
reported separately as `scope_median_us`/`scope_p99_us` rather than silently
folded in or silently dropped.

**Percentiles by nearest rank, no interpolation.** `p99` is the smallest sample
at or above position `ceil(0.99n)`. Why: every reported number is then a latency
that actually occurred, rather than an average of two that did not.

**`steady_clock`, not `high_resolution_clock`.** The latter is permitted to be
non-monotonic and is an alias for `system_clock` on some standard libraries;
`steady_clock` is the correct choice for durations.

**The Windows measurement is resolution-limited.** MSVC's `steady_clock` ticks
at 100 ns (QPC, 10 MHz). Median 0.100 µs is one tick and p99 0.200 µs is two, so
the harness cannot resolve Windows latency further. Recorded in BENCH.md rather
than presented as a precise figure. The Linux clock (~1 ns) puts the real value
at 58–59 ns median, consistent to within one Windows tick.

**The gate number is a floor, not a prediction.** The GIL is uncontended in this
harness — single thread, nothing competing. Multi-threading is explicitly out of
C0's scope, so contention was not measured. Flagged prominently in BENCH.md so
the 10,000x margin is not mistaken for 10,000x of headroom under load. **p99 at
batch 256 is 0.0002 ms against the ~2 ms budget, so no `DECISIONS.md` alarm is
raised**, but C9 must re-measure with real contention.

## Build system

**`chess-library` pinned to commit `53e6a84` (v0.9.4), not `master`.**
The draft tracked `master`. C1 is a *movegen parity* chunk: with a floating
dependency, the reference behaviour can change under a passing test suite with
nothing in the repo changing, and the failure would surface as an inexplicable
parity break weeks later. pybind11 was already pinned at `v2.12.0`; kept.

**`SOURCE_SUBDIR do-not-configure` on the chess-library fetch.** The library is
header-only, but its repository ships a `CMakeLists.txt` with test and benchmark
targets. Pointing `SOURCE_SUBDIR` at a directory containing no `CMakeLists.txt`
makes `FetchContent` fetch the sources without `add_subdirectory()`-ing those
targets into our build. Alternative was `FetchContent_Populate`, which is
deprecated as of CMake 3.30.

**chess-library included as a `SYSTEM` include directory.** This keeps
third-party headers from tripping `/W4` and `-Wall -Wextra`. It is not a Rule 4
violation: no `-Wno-*` flag and no `#pragma` is involved, and our own translation
units remain fully strict. (In the event, chess.hpp is warning-clean on both
toolchains at the current pin, so this is insurance rather than a live
suppression.)

**ASan is opt-in via `-DGUOFISH_ASAN=ON`, as a separate build directory.**
Rule 5 requires the *test* build to be sanitized; making it the default would
put a ~2x tax on every build including the benchmark, which would corrupt the
BENCH.md numbers. Both builds are run against the full suite.

Two MSVC-specific consequences, handled in CMake rather than left to the caller:
`/RTC1` (present in CMake's stock MSVC Debug flags) is rejected outright
alongside `/fsanitize=address`, so it is stripped; and incremental linking is
incompatible with ASan, so `/INCREMENTAL:NO` is forced. `NDEBUG` is also stripped
from release configs under `GUOFISH_ASAN` so Rule 5's "all debug asserts" holds
regardless of which config someone sanitizes.

**The MSVC ASan runtime DLL is copied next to the module by the build.** Since
Python 3.8 the interpreter no longer searches `%PATH%` when resolving an
extension module's dependencies, so the usual "add it to PATH" advice silently
fails with a missing-DLL error at import. Copying it beside the `.pyd` is the
only thing that works without editing the test harness.

**Linux builds write the module to the build directory, not the repo root.**
Forced, not preferred: the checkout is on `/mnt/c` (DrvFS), pybind11 strips
`Release`/`MinSizeRel` builds automatically, and `llvm-strip`'s in-place rewrite
fails on DrvFS with `Operation not permitted` — so a Linux Release build writing
into the repo *fails to link*. The `GUOFISH_MODULE_OUTPUT_DIR` cache variable
exists for this; Linux callers set it to an ext4 path and add that to
`PYTHONPATH`. This also avoids DrvFS's very poor small-file performance.

Consequence worth knowing: `python -m pytest` prepends the working directory to
`sys.path`, so a stale `.so` left in the repo root will shadow `PYTHONPATH`.
Noted in README_BUILD.md.

**Stale-output hazard, documented rather than engineered away.** Two build
directories writing the same module path interact badly with Ninja: after the
release build writes the `.pyd`, building the ASan directory reports
`ninja: no work to do`, because the output is newer than its inputs. The
sanitized module is never produced and the suite silently re-tests the release
build — *this bit during C0's own final verification and produced a false green
until the run time (0.11 s vs 0.24 s) gave it away.* The fix considered was
giving every Windows configuration its own `GUOFISH_MODULE_OUTPUT_DIR`, but the
repo root then has no module and the convenient `import guofish_core` from the
repo breaks. Chosen: keep the repo root as the single install location, and
document "delete the module before switching build directories" plus two
positive signals that the relink really happened. Revisit if C9/C12 need
several configurations resident at once.

## Leak checking

**No LSan suppression file.** CPython and numpy leak ~1.4 MB in ~1300
interpreter-lifetime allocations by design, so the raw report is always red. The
tempting fix — `leak:/usr/bin/python3.12` — is wrong here: every allocation
`guofish_core` makes is reached through Python frames, so that suppression would
hide our leaks along with theirs, and would do so silently and permanently.

Two discriminating checks are used instead:
* no leaked allocation's stack mentions `guofish_core` (grep over the report);
* the leak total does not grow with allocation count — a 1-buffer run and a
  500-buffer run (500 x 272 KB = ~136 MB if they leaked) report a **byte-identical**
  941,456 bytes in 858 allocations.

MSVC has no LeakSanitizer at all (LSan is not implemented on Windows), so leak
coverage is Linux-only. Windows ASan still covers memory errors.

## Test suite

**`tests/test_c0_toolchain.py` was overwritten, with explicit authorisation.**
Global Rule 1 forbids modifying anything under `tests/`. The file that existed
was a prior session's draft, not authored acceptance criteria, and it did not
test the C0 criteria: no alignment assertion, no `.base`/`OWNDATA` assertion, no
median/p99. It also asserted an `avg_ms` key and a one-argument callback, both of
which conflict with the brief. This was raised before any code was written and
the replacement was approved. **Recording it here because it is the one place
this chunk crossed a global rule, and it should not become precedent.**

**Mutation-checked, per the "How to tell whether a chunk actually passed"
procedure.** C0 has no golden data to corrupt, so the implementation was mutated
instead:
* alignment 64 -> 16: 5 tests fail. Note `test_buffer_is_64_byte_aligned[1-1]`
  *passed anyway* — a small allocation landed on a 64-byte boundary by luck,
  which is precisely why `test_alignment_holds_across_repeated_allocation` does
  100 consecutive allocations.
* dropping the `base` argument (making pybind11 copy): 8 tests fail — the
  zero-copy assertion, every write-visibility assertion, and the alignment
  assertions, since NumPy's own allocation is not 64-aligned.

Both mutations were reverted and the suite re-run.

---

# C0b — Contended GIL acquisition (2026-08-05)

## What C0 got wrong, and how C0b found it

Two defects in C0's own measurement, both of which made the result look better
than reality. Neither was caught by C0's tests, because C0's tests asserted the
gate and the gate passed.

**1. `roundtrip_bench` transferred no data.** The callback took no arguments and
the buffer was never handed to Python, so the "cost of crossing the boundary"
excluded the boundary's payload. With the live `rows x 68` view passed and
summed, the p99 at batch 256 goes from 0.0002 ms to 0.0150 ms — 75x — and starts
scaling with `rows` as it should. The C0 gate still passes (130x margin instead
of the claimed 10,000x), but the original margin was fiction.

**2. The uncontended number was not a floor, it was a different regime.** C0
recorded "the GIL is uncontended here, these numbers are a floor". That framing
turned out to understate the problem: under contention the median goes from
0.1 µs to 15,250 µs on Windows. A reader treating 0.1 µs as a floor to build on
would have been wrong by five orders of magnitude, not by a safety factor.

## The contention harness

**The GIL-free gap (`work_us`) is a parameter, and it is not zero.**
Chosen: `contention_bench(rows, iters, callback, work_us=0.0)`, with all
published configurations using 200 µs.
Alternatives: no gap at all (what C0 did); a `sleep` instead of a spin; making
the gap implicit and fixed.
Why: this was the single biggest finding of the chunk. With no gap the loop
re-requests the GIL ~100 ns after releasing it and usually wins the re-acquire
before the competing thread is scheduled, so ~90% of iterations never contend
and config B's *median* is indistinguishable from config A's. The first version
of this chunk's test asserted on p50 and reported "the contention simulation is
broken" — it was not broken, it was measuring a real but unrepresentative
regime. A real dispatcher waits for search threads to fill a batch, so a gap is
both more realistic and strictly more adversarial. 200 µs is the order of
magnitude of assembling 256 positions across a few search threads; the sweep in
BENCH.md shows 10 µs to 1000 µs all behave the same, so the exact value does not
matter — only that it is not zero.

A spin rather than a sleep because the search threads it stands in for occupy a
core. Sleeping would hand the core back to the OS and change the scheduling
picture the gap exists to reproduce.

**The background load is pure Python, and its throughput is verified.**
The brief requires pure Python (so the thread yields only at bytecode
boundaries, making `setswitchinterval` meaningful). Beyond that, the thread's
format-pass rate was measured against its solo rate and is 90–97% in every
configuration. This matters because "config C has a tiny acquire wait" has two
possible explanations — the mitigation works, or the competitor is starved — and
they have opposite implications. Recorded in BENCH.md.

**`UciFormatterLoad` publishes its iteration count continuously, not at exit.**
So the context manager can assert the thread is actually running *before*
anything is timed. A background thread that died at startup would make config B
look identical to config A, which is precisely the false green the brief calls
out as the worst outcome.

**Phases are asserted to partition the loop.** `acquire_wait_us + call_us +
release_us` must not exceed the measured wall time. Without this,
`acquire_wait_us` could be measuring something other than the wait and the gate
would pass for the wrong reason.

## Interpreting the result

**The Windows mitigation works for a reason nobody would predict.** CPython's
Windows condition variable is `SleepConditionVariableSRW`, which takes a `DWORD`
of milliseconds computed as `microseconds / 1000`. `setswitchinterval(0.0005)`
therefore becomes a **0 ms** timeout — the waiter times out instantly and
requests the drop immediately. The improvement is a cliff at exactly 1 ms, not a
slope: every interval from 1.1 ms to 10 ms gives the same ~15.3 ms p50, and
every interval below 1 ms gives the same ~3.2 µs.

This was found by sweeping the interval rather than by reading CPython's source,
and the sweep is kept in `tools/bench_c0b.py --sweep` because the conclusion
depends on it. It is recorded prominently because it makes the Windows margin
**fragile**: it is a property of an integer division in a platform shim, not of
the number 0.0005. Linux, whose `pthread_cond_timedwait` takes a nanosecond
deadline, shows the honest linear behaviour and only ~1.3x margin at the same
setting.

**The recommendation is `setswitchinterval(0.0005)`, not C++ stdout emission.**
The brief offers three verdicts; this is the "B bad, C good" branch and the data
supports it unambiguously (B degrades against A by 152,000x at the median;
C recovers essentially all of it). C++ stdout emission is recorded in BENCH.md
as the durable fix that remains available, but the gate passes and mandating a
redesign that the evidence does not require would be inventing work.

## Where the `max` criterion is enforced

**This is the one place C0b interprets the brief rather than following it, and
it is the entry to read most sceptically.**

The gate has two halves. `p99 < 1 ms` is asserted unconditionally on every
platform and every build, and it passed with at least 25% margin in all 80,000
samples measured. `max < 2 ms at batch 256` is asserted only on Windows without
sanitizers, and elsewhere is reported (as a Python warning, and in BENCH.md)
alongside a relative assertion that config C's max must still beat config B's.

Chosen because `max` is a single worst sample out of thousands and therefore
measures the OS scheduler's tail, not GIL behaviour. Evidence, 20,000 samples
per configuration:

| build | worst max | runs ≥ 2 ms |
|---|---|---|
| Windows Release | 2270.9 µs | 1/10 (1/40 over four sets) |
| Windows ASan | 2035.6 µs | 1/10 |
| WSL2 Release | 1987.2 µs | 0/10 |
| WSL2 ASan | 3313.6 µs | 3/10 |

Two facts make this a scheduler artifact rather than a real result: config A,
which has *no contention whatsoever*, shows the same class of tail at lower
amplitude (max up to 235 µs); and `max` grows with sample count, so the WSL2
"failure" at 8,000 samples (2597.7 µs) and its pass at 2,000 samples are the
same underlying distribution.

The brief already designates Windows authoritative and WSL2 a sanity check, so
scoping the WSL2 side is following it. **Extending that to sanitized Windows
builds is mine**, and the justification is Global Rule 3: the suite must pass on
every build, and a 1-in-10 flake trains people to re-run until green, which is a
worse failure mode than a documented scope.

What was deliberately *not* done: the criterion was not moved from 2 ms to a
larger number, `max` was not replaced with p99.9, and the sample budget was not
reduced to make excursions less likely. On the authoritative configuration the
assertion is exactly what the brief specifies, and it is expected to fail
roughly 1 run in 40 from external machine load. If that proves annoying in
practice the right fix is a quieter machine, not a looser gate.

## New bindings

Three functions were added beyond C0's "exactly these functions" list. C0b
mandates `contention_bench`; the other two are mine.

**`clock_info(probes)`** — the brief requires the clock resolution to be logged
so floor values are not mistaken for measurements. Exposed as a function rather
than hardcoded in the bench script because the answer differs per platform
(100 ns on Windows, 15 ns on Linux) and is measured, not declared:
`steady_clock::period` claims 1 ns on both, and is wrong on Windows.

**`build_info()`** — reports compiler, sanitizer status and whether `NDEBUG` is
set. Added because the gate assertion above needs to know whether it is looking
at a production build, and because BENCH.md needs provenance. It also gives
Global Rule 5 a direct test: if ASan is on, `asserts` must be true, which proves
CMake's `NDEBUG`-stripping actually worked in the module under test rather than
in the CMake cache.

**`/Zc:__cplusplus` added to the MSVC flags.** Not a preference — MSVC reports
`__cplusplus` as `199711L` regardless of `/std:` unless this switch is passed,
so anything feature-testing the language version concludes it is compiling C++98
while being fed C++17. The C0b buffer relies on C++17 over-aligned `operator
new`. `build_info()` additionally reads `_MSVC_LANG` first, which is correct even
without the flag, so the reported value stays truthful if the flag is ever
dropped. This is a conformance switch, not a warning suppression (Rule 4).

## Measurement details

**Nearest-rank p50, not the interpolated median.** `summarize()` reports p50/p95/
p99 all by nearest rank so every key in the dict is a latency that actually
occurred. `roundtrip_bench` keeps C0's interpolated `median_us` unchanged, so the
C0 table stays comparable with itself.

**The benchmark tool imports the test module rather than reimplementing it.**
`tools/bench_c0b.py` calls `run_configurations()` from
`tests/test_c0b_contention.py`. The alternative — duplicating the load generator
and the config matrix in the tool — means the published numbers and the asserted
numbers can drift apart silently, which for a chunk whose entire output is a
table is the most likely way for it to become wrong. Direction of the dependency
is tool → test, so the tests never depend on `tools/`.

**The view passed to the callback is built once, not per iteration.** The buffer
address is stable, so a production dispatcher would reuse the view too;
rebuilding it every iteration would time NumPy object construction rather than
the GIL round trip. A test asserts the callback receives the same object every
time, so this cannot silently change.

**`call_us` under contention is not a measure of callback compute.** NumPy
releases the GIL inside `arr.sum()` and must re-acquire it, so in config B a
handoff wait lands *inside* the callback: `call_us` p50 is 15.3 ms at batch 256
on Windows. This is realistic — a real evaluator calling PyTorch behaves the same
way — but it means only `acquire_wait_us` is a clean measure of contention.
Noted in BENCH.md.

## Mutation check

C0b has no golden data to corrupt, so the implementation was mutated instead,
per the same procedure C0 used.

* **`t1` sampled before the acquire instead of after** (`acquire_wait_us` is then
  always ~0): **6 tests fail** — both `config_b_visibly_degrades` cases, both
  `shorter_switch_interval_improves` cases, the dispatch-gap test, and the
  report's collapse check. Note the *gate test itself passed*, since a
  permanently-zero acquire wait trivially satisfies `p99 < 1 ms`. This is the
  whole reason the validity tests exist and it is the failure mode the brief
  warns about most loudly.
* **`work_us` ignored** (the GIL-free gap never happens): **5 tests fail** —
  `work_us_actually_consumes_the_requested_time` plus the same four contention
  comparisons.

Both mutations were reverted and the suite re-run on all four build
configurations.

## Python versioning

C0b's numbers come from **Python 3.13.7 on Windows and Python 3.12.3 on Linux
(WSL2)**, which is fine here because C0b produces benchmarks, not golden data —
nothing under `golden/` was written or read by this chunk.

**From C1 onward this must not continue.** Golden data must be generated once on
a single pinned interpreter version and consumed unchanged on every platform. Two
interpreters generating golden data independently would let a divergence in, for
example, dict ordering or float repr appear as a parity failure in C++ code that
is actually correct. The pin should be recorded here when C1 sets it.

## Global Rule 1

`tests/test_c0_toolchain.py` was modified. The change is the callback signature:
`roundtrip_bench` now passes the live view, so `def callback():` became
`def callback(arr):` in five places, and the gate callback additionally calls
`arr.sum()` to touch the payload. This is mandated by the C0b brief's
implementation scope ("update the test-side callback to explicitly read/touch
the data"), and is possible only because the brief marks C0/C0b as the last
chunks where agent-authored tests are unavoidable.

**No assertion was weakened or removed.** Three tests were added to that file
covering the new payload (that it is a stable, zero-copy, aligned, correctly
sized view, and that it survives past the call). From C1 onward this rule is
hard and this must not be treated as precedent.

# C1 — Movegen parity (2026-08-05)

`legal_moves(fen) -> list[str]` is exposed from `cpp/bindings.cpp`; the
generation and formatting live in `cpp/movegen.hpp` so that C4/C5 can include
them without pulling in the benchmark machinery. Acceptance is
`tests/test_c1_movegen.py` against `golden/movegen.jsonl` — 100,000 positions,
0 mismatches.

## Castling normalisation

chess-library encodes castling as **king-takes-rook**: for White kingside,
`from()` is e1 and `to()` is **h1**, the rook's square, not the king's
destination. That is the UCI_Chess960 convention and it is what the library's
own `makeMove()` expects back, but it is not what a standard UCI GUI or
python-chess emits.

`uci_destination()` rewrites it: on a `CASTLING` move it returns file **G** when
the rook is to the king's right and file **C** when it is to the left, on the
king's own rank. The left/right test is `move.to() > move.from()` rather than a
hard-coded e1/h1, so it is still correct on a shuffled back rank.

Three alternatives were considered:

* **Call `chess::uci::moveToUci(move, false)`**, which does the same rewrite.
  Rejected: its `chess960` parameter *defaults to false*, so a caller who
  forgets the argument gets silently wrong output on a 960 board, and the brief
  asks for this normalisation to be ours and auditable. Our version reads
  `board.chess960()` itself instead of taking a defaultable flag, so it cannot
  be called wrongly.
* **Normalise at the FEN/parse layer** — not possible; the encoding is a
  property of the generated move, not of the position.
* **Let the search carry king-takes-rook internally and normalise only at the
  UCI boundary.** Deferred, not rejected: it is the right answer for C5, but C1
  is judged on strings and carrying two representations in one chunk would make
  the parity result harder to trust, not easier.

**Chess960 is passed through untouched** when `board.chess960()` is set, because
there king-takes-rook *is* the correct UCI. This mirrors python-chess's
`Board.uci()`, which normalises only when `self.chess960` is false. Every FEN in
the golden file is standard, so this branch is not covered by the parity run; it
exists so the helper is not silently wrong the first time a 960 board reaches
it. `legal_moves(fen)` cannot construct one — there is no 960 flag on the API —
so it is currently unreachable from Python.

The failure mode if the normalisation is skipped is quiet, which is why it gets
this much attention: `e1h1` is a well-formed UCI string that simply never
appears in a reference list, so it reads as "one move missing, one move extra"
rather than as a formatting bug. `format_mismatch()` in the test therefore names
it explicitly when it sees one.

## Canonical order

**Byte-wise lexicographic sort of the UCI strings**, via `std::sort` on
`std::vector<std::string>`.

Every string is `[file][rank][file][rank][promo?]` over a fixed ASCII alphabet,
so this is exactly the `(from, to, promotion)` tuple order the brief asks for,
and it is the order `tools/gen_movegen_golden.py` writes. Python's `str` sort
compares code points and `std::char_traits<char>::compare` compares as unsigned
bytes; both are ASCII here, so the two agree, including on the shorter-first
rule for prefixes.

**It is deliberately not a sort on chess-library's square index.** Index order
is rank-major (a1, b1, … h1, a2, …); UCI string order is file-major (a1, a2, …
a8, b1, …). They disagree on almost every position. Because PUCT resolves ties
by child order, picking the wrong one would not fail loudly — C++ and Python
would just explore different moves at equal priors, which is precisely the class
of divergence this chunk exists to prevent. `test_output_is_in_canonical_order`
asserts sortedness directly, independently of the golden comparison, so an
ordering that merely happens to match the file is not what is being certified.

A 4-character and a 5-character move can never share a from/to pair — a pawn
reaching the back rank must promote — so the prefix case never arises between
two real moves. `legal_moves_uci()` asserts the sorted list has no adjacent
duplicates, which is how a normalisation that collided with a real king move
would surface.

## Rejecting a FEN early enough

**The king count is checked before `setFen()` is called, not after.** This
started as the obvious "parse, then validate" and it was wrong.

`Board::setFen()` does not merely accept a kingless position. On its way out it
builds the castling paths, and that loop calls `kingSq()` for **both colours
unconditionally, before it consults the castling rights** (chess.hpp:3471).
`kingSq()` is `assert(pieces(KING, color) != 0ull)` followed by `lsb()` on the
king bitboard — an assertion failure in a debug build, an out-of-range square in
a release one. Validating afterwards is too late by one function call.

This passed the MSVC Release suite and aborted the interpreter the moment the
suite ran against the ASan/debug module (`Assertion failed: Expression:
pieces(PieceType::King, color) != 0ull`). It is a good advertisement for Global
Rule 5: the release build reported 78 passed.

`count_kings()` therefore scans the FEN's piece-placement field as text, before
a `Board` exists, mirroring chess-library's own leading-space trim and
split-on-first-space so it reads the same field the parser will. Exactly one `K`
and one `k` are required. Two kings of one colour are refused for the same
reason: `lsb()` would silently pick one.

Rejecting rather than tolerating is a judgement call. python-chess is equally
undefined on a kingless board in practice, and nothing in `golden/movegen.jsonl`
is affected either way, but `legal_moves` is reachable from Python and a
caller's typo should not be able to reach UB. Both rejections raise
`std::invalid_argument`, which pybind11's stock translator turns into
`ValueError` with no custom registration.

## Where the test could have fooled itself

`test_golden_corpus_covers_the_edge_cases` exists because a parity run over 100k
quiet middlegames would be green with the castling normalisation deleted. It
counts, from the *reference* lists, how many positions offer castling, a
promotion, an en-passant target square, and no move at all, and fails if any
bucket is thin.

The first version identified castling by looking for the literal strings `e1g1 /
e1c1 / e8g8 / e8c8`. **That is wrong**, and the mutation drill below found it by
landing its "de-normalise a castle" mutation on
`1N6/2p2K2/8/5p1P/2k2b2/2P5/8/4q3 b - - 1 49` — a position with a **Black queen
on e1**, where `e1g1`, `e1c1`, `e1a1` and `e1h1` are all ordinary queen moves.
The e-file back-rank square is not always the king's.

`has_legal_castling()` now looks the side-to-move's king square up out of the
FEN and asks whether any move starts there and crosses two files on the same
rank, which is a castle and nothing else.
`test_castling_detector_is_anchored_on_the_king_square` pins both lookalikes,
including one where the castling field is non-empty so the `-` shortcut is not
what saves it.

The same strings are still used in `format_mismatch()`, but only as a *hint* in
a report that has already failed for another reason, and the comment there says
so.

## Mutation check

The brief asks for a golden value to be corrupted, the diagnostic inspected, and
the file restored. Global Rules 1 and 2 say `golden/` is never written to. Those
are reconcilable and the rules win: the parity test reads an optional
`GUOFISH_GOLDEN_MOVEGEN` environment variable, so the drill ran against a
**corrupted copy in the scratch directory** and `golden/movegen.jsonl` was never
opened for writing. Its SHA-256 was `1754e3aa…de6151a2` before the drill and
`1754e3aa…de6151a2` after.

Four mutations in one file, one per failure mode the report must tell apart. All
four produced the required diagnostic, and the other 99,996 positions stayed
green:

| mutation | what the report said |
|---|---|
| dropped `c3a5` from line 1 | `extra in C++ : ['c3a5']` |
| invented `z9z9` on line 2 | `missing from C++ : ['z9z9']` |
| swapped two moves on line 3 | `ORDER ONLY … first difference at index 0: C++ 'a7a3' vs python 'a7a4'` |
| `e1g1` → `e1h1` on line 1112 | `missing from C++ : ['e1h1']`, `extra in C++ : ['e1g1']` |

Because the corruption is in the *reference*, the fourth case does not trigger
the `CASTLING NOT NORMALISED` hint — that fires on the real direction, C++
emitting `e1h1`. Four `test_diagnostic_*` tests exercise the formatter directly
so the requirement is covered by the suite and not only by a one-off drill.

## Python versioning

C0b asked that C1 record the interpreter pin for golden data. What can be
verified from this repo: `golden/movegen.jsonl` agrees exactly with
**python-chess 1.11.2 on Python 3.13.7** (Windows), which is the interpreter
`tools/gen_movegen_golden.py` runs under here. The file was generated by the
project lead and the generating interpreter is not recorded inside it, so that
is a checked agreement, not a provenance claim. **Recording it inside the file —
a header line carrying the interpreter, python-chess version, seed and
arguments — is the fix, and it belongs to whoever next regenerates it.**

The consuming side needs no pin at all: `tests/test_c1_movegen.py` does not
import `chess`. The reference's answers reach it only through the golden file,
so the parity test cannot drift into re-deriving its expectation from the same
library that produced it, and it gives the same verdict on 3.12 (Linux) and 3.13
(Windows).

## Not done

* **No GIL release around the generation.** `legal_moves` is a validation
  helper, not a search path; the full 100k-position sweep is 0.65 s on the
  Release build, and `py::call_guard` would add a release/acquire pair per call
  for no benefit. C5's search will not call this function.
* **No batch entry point.** Per-call pybind overhead is not the bottleneck at
  this scale and the brief specifies the single-FEN signature.
* **Strings, not packed integers.** Explicitly what the brief asks for; the
  production core will carry the 16-bit `chess::Move` and format only at the UCI
  boundary.

## Global Rule 1

No existing test file was modified. `tests/test_c1_movegen.py` is new; `git
status` over `tests/` shows one untracked file and nothing else.

---

# C2 — Tokenizer parity (2026-08-05)

`tokens(fen) -> numpy.int32[68]` and the `TokenBatch` batch buffer are exposed
from `cpp/bindings.cpp`; the encoding itself lives in `cpp/tokens.hpp` so C4/C5
can include it without the benchmark machinery, exactly as C1 did for movegen.
Acceptance is `tests/test_c2_tokens.py` against `golden/tokens.npz` — 100,000
positions, 0 mismatches, through both entry points.

## Not building this on `chess::Board`

**The decision that determined everything else in this chunk.** The obvious
implementation is `board.setFen(fen)` followed by reading the pieces, side,
castling rights and ep square off the `Board`. It is wrong, and it fails
silently.

chess-library validates the en-passant square on the way in. From
`setFenCommon()` in `chess.hpp`:

```cpp
if (ep_sq_ != Square::NO_SQ) {
    valid = movegen::isEpSquareValid<...>(*this, ep_sq_);
    if (!valid) ep_sq_ = Square::NO_SQ;
}
```

That is defensible for a search library — an ep square no pawn can act on does
not change the position — but it is not the FEN's fourth field, and it is not
what `board.ep_square` reports in python-chess, which stores the field verbatim
and offers the legality question separately as `has_legal_en_passant()`.
`board_to_tokens` reads the attribute.

The gap is neither hypothetical nor rare. Of the **3,610** positions in the
corpus carrying an ep target, **3,203 have no legal ep capture available**. A
`Board`-based tokenizer would emit 31 where the reference emits 32..39 on 3.2%
of the corpus — and on precisely the positions that follow a double pawn push,
which is to say constantly, in every game.

**Alternatives considered:**

* *Use `Board` and patch the ep square back in from the FEN text.* Rejected: it
  still pays `setFen`'s zobrist hashing, castling-path construction and a
  movegen call on a hot path that needs none of them, and it leaves a `Board`
  in the code whose `enpassantSq()` disagrees with the token that was emitted —
  a trap laid for whoever touches this next.
* *Parse the FEN directly.* Chosen. It is the only version where the ep rule is
  a single line that says what it means, and it is also the fast one: 272 ns per
  position on Windows, 361x the reference (BENCH.md).

The cost is that `cpp/tokens.hpp` now contains a second FEN parser, independent
of the one inside chess-library. That duplication is real and is the price of
the parity guarantee; the header says so at the top.

## Castling rights: transcribing python-chess rather than reading the letters

`board_to_tokens` does not read the FEN's castling field. It calls
`has_kingside_castling_rights()` / `has_queenside_castling_rights()`, and those
answer a *narrower* question: python-chess cleans the claimed rights against
where the king and rooks actually stand. A FEN claiming `KQkq` with the White
king on d1 yields no White rights at all.

**Every FEN in `golden/tokens.npz` is self-consistent** — the field's letters and
the cleaned rights agree on all 100,000 positions, and the token-65 histogram
matches the castling-field histogram exactly. So the naive implementation
(`'K' in field ? 8 : 0`, etc.) also passes the acceptance test, and it is
shorter.

It was rejected anyway. The corpus is not the contract: the first inconsistent
FEN reaching a search built on the naive rule would be mis-encoded with nothing
to catch it, and FENs arriving over UCI from a GUI are not guaranteed to be
self-consistent. `parse_castling_field` / `clean_castling_rights` /
`has_castling_right` in `cpp/tokens.hpp` are a line-by-line transcription of
`_set_castling_fen`, `clean_castling_rights` and `has_*_castling_rights` for the
standard-chess (`chess960=False`) branch, which is the mode
`tools/gen_token_golden.py` runs in.

`test_castling_rights_are_cleaned_against_the_position` is what fails if this is
ever simplified back: two FENs claiming `KQkq`, one with the king off e1 and one
with the a1 rook missing, whose reference tokens are 18 and 26 rather than 30.

Three details are transcribed rather than rewritten, and all three are
deliberate:

* **`lsb()`/`msb()` return -1 on an empty bitboard.** python-chess's castling
  parser depends on it: `msb(0) == -1` is what pushes the "no rook on the back
  rank" case into the `else` branch instead of indexing a bitboard.
* **`rook > king_mask` compares whole 64-bit masks**, not square indices. With
  the one king per side that real positions have these are the same test, but
  python-chess permits two kings and `tokens()` therefore has to answer for
  them; keeping the comparison in its original shape means both languages land
  on the same side of it.
* **The X-FEN `~` promoted-piece suffix is parsed.** No FEN in the corpus uses
  it, but python-chess excludes promoted kings from every castling test
  (`kings & ~promoted`), and — more immediately — treating `~` as an ordinary
  character would place every subsequent piece on that rank one square off.

Shredder-FEN file letters (`a`..`h`) in the castling field are also handled, for
the same reason: `FEN_CASTLING_REGEX` admits them, so silently dropping a letter
the reference honoured would be a parity hole.

## Where this parser is knowingly laxer than the reference

`tokens()` does not parse the halfmove clock or the fullmove number. python-chess
validates both as integers and raises on garbage; this accepts it.

That is the only divergence, and it is safe in the only direction that matters:
**no FEN the reference accepts is refused here, and no accepted FEN produces a
different token.** Neither counter reaches the encoding —
`test_move_counters_do_not_reach_the_encoding` pins that — so chasing `int()`'s
full accepted syntax (leading `+`, underscores, surrounding whitespace) would be
code with no observable behaviour.

Everything the encoding *does* depend on is refused exactly where the reference
refuses it: rank and file counts, adjacent digits, `~` placement, piece letters,
side-to-move, the castling regex, the ep square name, and a seventh field.
Trailing fields are defaulted rather than required, because python-chess defaults
them and `tokens()` and `legal_moves()` are fed from the same call sites — two
different ideas of what parses would be a bug visible in only one of them.

## Kingless positions are encoded, not refused

C1's `legal_moves` refuses them: chess-library's `setFen` calls `kingSq()` while
building castling paths and asserts on an empty king bitboard, so the guard is
load-bearing there (see "C1 / rejecting a FEN early enough").

`tokens()` does not touch chess-library and python-chess encodes a kingless board
without complaint, so refusing here would be a divergence rather than a safety
check. The two functions therefore accept different sets of FENs *on purpose*,
and both are documented at their binding.

## The batch interface

The brief asks for "a batch/buffer interface that writes tokenized sequences
directly into a C++-owned 2D array without intermediate memory copies".

**Chosen shape:** a `TokenBatch` class owning one 64-byte-aligned
`AlignedBuffer` of `[capacity, 68]` int32, with `view()` returning a zero-copy
NumPy alias and `fill(fens, row_offset=0)` encoding into rows in place.

**Alternatives considered:**

* *`tokens_batch(fens) -> ndarray` returning a fresh array each call.* Rejected:
  it reallocates per batch and gives the caller no way to hand the *same* array
  to the network every iteration, which is what the evaluator in scope 2.1 does.
* *Reuse C0's module-level `make_buffer` / `g_buffer` singleton.* Rejected: a
  single global buffer cannot serve more than one dispatcher, and C9 will have
  more than one thread.

`row_offset` exists so several `fill()` calls can pack one network batch. The
parity sweep in `test_c2_tokens.py` deliberately alternates the offset between 0
and mid-buffer, because a sweep that always started at row 0 would never
exercise the argument and would not notice a stale row being read back.

### Releasing the GIL over Python-owned string memory

`fill()` encodes with the GIL released. That is only safe because of a specific
sequence, and it is worth stating since it is the sort of thing that looks like a
bug later:

1. `PySequence_List` materialises the iterable into a list this call owns — so a
   generator is consumed exactly once, here, and never re-entered from inside
   the GIL-free region.
2. `PyUnicode_AsUTF8AndSize` caches the encoded form *inside each str object*,
   so the returned pointer lives as long as the object does, and the list holds
   every object alive on this frame.
3. Python strings are immutable, so nothing can move or rewrite those buffers
   under a reader.
4. The encode loop makes no Python API call at all. It reads bytes and writes
   int32s.

`py::gil_scoped_release`'s destructor reacquires on scope exit *including during
unwinding*, so a `std::invalid_argument` thrown from a bad FEN mid-batch is
translated to `ValueError` with the GIL correctly held.

**Documented behaviour on a bad FEN mid-batch:** rows written before it keep
their new contents; the rest keep their old ones. Rolling back was considered and
rejected — the caller's batch is wrong either way, and quietly restoring stale
rows for the network to read is worse than a partially updated buffer plus an
exception. `test_a_rejected_fen_inside_a_batch_raises` pins it so a change is a
deliberate one.

## Two benchmark numbers, not one

`tokenize_bench` copies the FENs into C++ strings before starting the timer, so
the timed region touches no Python object; `TokenBatch.fill` is measured
separately end to end. Both are in BENCH.md.

Publishing only the faster one would overstate what a Python-driven dispatcher
sees today (by 7% on Windows). Publishing only the slower one would understate
what C5 will see, where the leaves are `chess::Board` objects and no FEN string
is constructed at all. The gate is judged on the encoder figure — 3,678,645
pos/s, **360.7x** the brief's 10,200 pos/s baseline against a 100x requirement.

`tools/bench_c2.py --python` also re-measures the reference on this machine. It
comes out at 8,841 pos/s, *slower* than the 10,200 the brief quotes, which would
inflate the speedup to 416x. BENCH.md reports both and carries the conservative
number forward.

## Mutation check

Same reconciliation as C1: the brief asks for a golden value to be corrupted and
the file restored; Global Rules 1 and 2 say `golden/` is never written to, and
the rules win. `tests/test_c2_tokens.py` reads an optional
`GUOFISH_GOLDEN_TOKENS` environment variable, so the drill ran against a
**corrupted copy in the scratch directory** and `golden/tokens.npz` was never
opened for writing. Its SHA-256 was `ea9bf8df…413ce562` before the drill and
`ea9bf8df…413ce562` after.

Four mutations in one file, one per field the report has to tell apart. All four
produced the required diagnostic — the FEN, both 68-element arrays in full, and
every differing index with both values — through **both** the single-position and
the batched path, and the other 99,996 positions stayed green:

| mutation | what the report said |
|---|---|
| row 5, index 4: rook on e1 → 0 | `[4] square e1  expected 0  got 4` |
| row 6, index 65: castling 29 → 22 | `[65] castling rights  expected 22  got 29`, plus `CASTLING MASK: expected 0111, got 1110 (bits are WK WQ BK BQ, high to low)` |
| row 184, index 66: ep 37 → 31 | `[66] en-passant file  expected 31  got 37` |
| row 99999, index 67: CLS 40 → 0 | `[67] CLS  expected 0  got 40` |

Two notes on the report:

* Indices are named, not numbered: `[4] square e1`, `[66] en-passant file`. A
  bare index list is not much use when the failure is on square 41.
* Both arrays print via `.tolist()`, not NumPy's repr, which elides the middle of
  a 68-element array with `...` — and the elided region is squares 24..47, most
  of the board. `test_diagnostic_reports_every_differing_index` asserts the
  report contains no `...`.

The `EN PASSANT DISCARDED` hint did not fire on the row-184 mutation, and that is
correct: it fires on the real-bug direction (C++ emitting 31 where the reference
has a file), not on a corrupted reference. Five `test_diagnostic_*` tests
exercise the formatter directly so the requirement is covered by the suite and
not only by a one-off drill.

## Where the test could have fooled itself

Three ways this suite could have been green while the encoder was wrong, and what
closes each:

* **A corpus with no hard cases.** `test_corpus_covers_en_passant_without_a_capturer`
  asserts at least 1,000 ep positions with no pawn placed to capture (there are
  3,203), and `test_corpus_covers_every_token_value` asserts all 16 castling
  masks, all 9 ep values and all 13 square values appear. Without the first, a
  `Board`-based tokenizer passes outright.
* **Judging the ep token against the golden file only.**
  `test_en_passant_is_emitted_without_a_capturer` re-derives the expected file
  from the FEN string itself, so it still fails if the reference and the C++ side
  were somehow wrong together.
* **Testing only the path the search will not use.** `tokens()` and
  `TokenBatch.fill()` are compared against the golden file *separately*. They
  reach the same encoder through different plumbing — fresh NumPy array vs. a row
  of a C++ buffer, GIL held vs. GIL released — and a plumbing bug shows up in
  exactly one.

Nothing in `tests/test_c2_tokens.py` imports `chess`. The reference's answers
reach it only through the golden file and through targeted literals that were
cross-checked against `core.mctsv4.board_to_tokens` on python-chess 1.11.2, so
the suite cannot drift into re-deriving its expectation from the library that
produced the file.

## Sanitizers

Suite green on four builds: MSVC Release, MSVC Debug + ASan, Clang Release,
Clang Debug + ASan + UBSan. 143 tests each (62 of them C2's), no ASan or UBSan
runtime errors.

`TokenBatch`'s allocation was leak-checked with the quantitative method
README_BUILD.md prescribes rather than by reading the summary: 1 vs 500
`TokenBatch(1000)` allocations (272 KB each) give a **byte-identical** total of
941,456 bytes in 858 allocations. Had the buffer leaked, the second run would
have been ~136 MB heavier. No leaked allocation's stack mentions `guofish_core`.

## Not done

* **No tokenization from a `chess::Board`.** C5 will want `tokenize_into(const
  chess::Board&, int32_t*)` so search leaves never round-trip through a FEN
  string. It is deliberately not in C2: there is no golden data for it, so it
  would ship unverified. The square/castling/ep logic is already factored to be
  reusable when that chunk arrives — and note the ep rule will need the *board's
  own* ep square, which is where the trap at the top of this entry reappears.
* **No multi-threaded fill.** `TokenBatch.fill` is single-threaded. At 272 ns per
  position a 256-row batch takes 70 µs, which is below the C0b GIL acquire wait
  that surrounds it (78 µs p99); parallelising it would optimise a term that is
  already noise. Revisit only if profiling in C5 says otherwise.
* **No `int16` or packed encoding.** The network's input is `int32[68]` and the
  buffer matches it, so no conversion happens on the way in. A narrower buffer
  would save memory the dispatcher does not lack and add a widening step it
  currently does not pay.

## Global Rule 1

No existing test file was modified. `tests/test_c2_tokens.py` is new; `git
status` over `tests/` shows one untracked file and nothing else. `golden/` was
not written to — see "Mutation check" for the before/after hash.

---

# C3 — Two keys, never mixed (2026-08-06) — golden data only

This entry covers `tools/gen_key_golden.py` and the two files it writes,
`golden/keys.jsonl` (100,000 positions, the shared C1 corpus) and
`golden/keys_adversarial.jsonl` (22 constructed pairs). The C++ side of C3 is
not written yet; the decisions here are the ones the C++ must conform to.

## The one that needed a ruling: what `nn_key` actually is

The chunk brief and the scope disagree, and the disagreement is not cosmetic.

* **Chunk brief (C3):** "`nn_key(fen) -> uint64` — must reproduce the
  **post-fix** Python key including the en-passant correction." The post-fix
  Python key is `make_cache_key()`, which returns the *tuple*
  `(chess.polyglot.zobrist_hash(board), board.ep_square)`.
* **Scope §2.4:** `nn_key` is the "**hash of the 68-token sequence itself**",
  and says so emphatically — "the key cannot be coarser than the network's
  input if it *is* the network's input" — while explicitly noting this is "a
  third convention distinct from both python-chess's polyglot hash and its
  transposition key".

**Chosen: scope §2.4, the token hash.** Reasons, in order of weight:

1. The brief's own requirement is `-> uint64`, and `make_cache_key` is a tuple
   with more entropy than 64 bits (a 64-bit hash plus a 65-valued ep field).
   Any uint64 rendering of it is a *lossy* packing, invented here and recorded
   nowhere else, that C++ would have to reproduce bit-for-bit from the golden
   numbers alone. The token hash is uint64-native with nothing to invent.
2. The brief's stated intent — "including the en-passant correction" — is
   satisfied strictly *better* by the token hash. `make_cache_key` patches the
   defect by appending the ep square to a hash that is coarser than the
   network's input; hashing the tokens makes the defect unrepresentable, which
   is the phrasing scope §2.4 uses and the outcome the brief is asking for.
3. Both C3 acceptance criteria then hold by construction rather than by
   argument. Clock-twins share an `nn_key` because the halfmove clock is not a
   token; ep-twins never share one because token 66 is written from
   `board.ep_square is not None` unconditionally.

**Resolved 2026-08-06: there was no conflict.** The scope is right about the
mechanism, the brief is right about the constraint; they only read as
incompatible because the brief stated the constraint as *value* equality when
what the cache depends on is *partition* equality. The two keys induce the same
equivalence classes, so they are interchangeable for cache-hit behaviour and
Gate 1 cannot tell them apart. The brief has been amended to say so, and
`check_partition_matches_python()` now asserts it at generation time.

The equivalence, since the argument decides the chunk:

* **tokens equal ⟹ Python key equal.** Exact, no assumptions. Equal tokens fix
  placement, side to move, the four castling flags and the ep *file*; the ep
  *rank* is implied by side to move, so the raw ep square matches too — and
  every input to the Zobrist is then equal.
* **Python key equal ⟹ tokens equal.** Holds unless two distinct positions
  collide in the 64-bit Zobrist.

The asymmetry is worth knowing: only the Python key can be coarser, never ours.
That is the same failure mode the ep correction was introduced to fix,
surviving as a hash collision rather than as a rule. If this assertion ever
trips, suspect Python's key, not the port's.

Measured over the 100k corpus: 96,068 classes on each side, partitions
identical.

## The three en-passant rules

The reason this chunk is dangerous, in one table. All three coexist, and each
is correct for its own consumer:

| rule | who uses it | ep square counts when |
|---|---|---|
| raw FEN field | token 66, therefore `nn_key` | a double push just happened, always |
| legal capture | `_transposition_key()`, therefore `rep_key` | a *legal* ep capture exists |
| pseudo-legal adjacency | `chess.polyglot.zobrist_hash` | a pawn stands beside it — "legality [...] is irrelevant", its own comment |

The corpus holds 3,610 positions with an ep target, 3,203 of which have no
legal capture, and **3** where all three rules give three different answers —
the horizontal-pin positions carried as literals in the generator
(`EP_PIN_1..3`). They are in the adversarial set for that reason.

Consequences for the C++ side, both read off `chess.hpp` at the pinned revision:

* `nn_key` must **not** be built on `chess::Board`. `setFenCommon()` runs
  `isEpSquareValid()` and silently clears an ep square with no legal capture —
  see the C2 entry, which is why `cpp/tokens.hpp` parses the FEN directly. Hash
  C2's token buffer and this is already right.
* `rep_key` **may** use `Board::enpassantSq()`, because `isEpSquareValid()` is a
  full legality test — `checkMask`, both pin masks, and `generateEPMove`, which
  handles the horizontal double-pawn pin explicitly (`chess.hpp` ~4085, with
  the `7k/4p3/8/2KP3r/8/8/8/8 b - - 0 1` case in its comment). That it
  coincides with `has_legal_en_passant()` is a fact about this pinned revision,
  not a guarantee; the three `ep_pinned_*` pairs exist to catch it changing.

## FNV-1a-64 over an explicit byte serialisation

Chosen for both keys: offset basis `0xcbf29ce484222325`, prime
`0x100000001b3`, over a payload that begins with a domain tag.

*Alternatives:* reuse `chess.polyglot.zobrist_hash` for one or both (it is
already there, and chess-library ships a Zobrist too); `std::hash` (not stable
across implementations, so unusable for golden data); xxhash/wyhash (a new
third-party dependency, Global Rule 7).

*Why:* a Zobrist realisation of either rule has to **re-derive** the ep handling
inside a 781-entry random table, which is precisely the silent divergence this
chunk exists to prevent — and for `nn_key` it would have to be a Zobrist over
tokens, a table nobody has. Hashing a byte string puts the rule in one readable
place. FNV-1a is ~6 lines of dependency-free C++ and needs no shared table. It
is not a strong hash, but it is not being asked to be one: no adversary, and 0
collisions over 100,000 positions.

`nn_key`'s payload is the byte image of the `int32[68]` buffer C2 already
produces, so C++ hashes the buffer it is holding with no repacking.

**The domain tags (`guofish/nn_key/v1\0`, `guofish/rep_key/v1\0`) are
load-bearing.** They guarantee `nn_key(p) != rep_key(p)` for every position, so
a swapped key can never coincidentally compare equal. That is the runtime
complement to the strong typedefs the brief requires at compile time: the
typedefs stop the mistake being written, the tags stop it being survivable if it
ever is. Cost is 18 bytes per hash, once.

## Two files, and a pair per line

`golden/keys.jsonl` is `{fen, nn_key, rep_key}` per position. The adversarial
set is a separate file with a *pair* per line, carrying `expect_nn` and
`expect_rep` as `"same"`/`"differ"` alongside both members.

*Alternative:* one flat file of positions with a `desc` string, as the previous
draft declared. *Why not:* the pairs are the unit of meaning — the brief calls
them "position *pairs* [that] will not occur adjacently in any corpus" — and a
flat file forces `tests/test_c3_keys.py` to re-pair records by parsing `desc`
and to hardcode which relation each pair is meant to prove. Both belong in the
data. A test reading this file cannot mis-pair records and cannot disagree with
the generator about what a pair means.

## The generator refuses to emit data it cannot satisfy

`--self-check` (default on) blocks the write unless, for every emitted record:
the FEN is canonical; **both keys are recomputable from the stored FEN alone**;
`nn_key != rep_key`; every declared pair relation actually holds; no FEN
anywhere in either file maps to two different key values; and no two distinct
payloads share a key.

This is not defensive habit, it is the specific defect the previous draft had
(below). A golden generator is the one program with no downstream check on it,
so its self-consistency has to be its own responsibility.

The last check distinguishes two things that look identical in the output:
3,932 corpus positions share an `nn_key` with another, and every one of them is
the *same board with a different clock* (the corpus is deduplicated on the full
FEN, clocks included). Those must share a key. A genuine FNV collision would
look the same in the file, so the check keeps a blake2b digest of each payload
and aborts if two different payloads ever land on one key.

## What the previous draft got wrong

`tools/gen_key_golden.py` already existed and was rewritten, as in C0. Recorded
because the draft's output was not merely imprecise — it was unsatisfiable, and
it would have failed in a way that reads as a C++ bug.

1. **It wrote `board.fen()`.** python-chess defaults to `en_passant="legal"`,
   which omits the ep square unless a capture is available, while the keys were
   computed from the raw `board.ep_square`. 3 of its 8 records did not survive a
   FEN round trip, and two *pairs* collapsed onto identical FEN strings carrying
   different `nn_key` values. No implementation can satisfy that file. It also
   erased the ep-twin distinction that is the entire point of the pair.
2. **`get_rep_key` returned a raw polyglot Zobrist**, with a docstring claiming
   "Polyglot Zobrist natively follows the 'only if capturer present' rule".
   Polyglot's rule is pseudo-legal adjacency; `_transposition_key()`'s is legal
   capture. Wrong on all three horizontal-pin positions in the corpus.
3. **No corpus.** 8 records against an acceptance criterion of 100k FENs, with a
   comment saying the corpus was skipped "for brevity".
4. **4 pairs, not ~20**, and the pair labelled "transposition" was not one:
   `1.d4 Nf6 2.c4` and `1.c4 Nf6 2.d4` reach the same placement with *different*
   ep files (c3 vs d3). It is a good adversarial case and is kept — as
   `near_miss_ep_file`, with the relation it actually has.

## Mutation check

Per "How to tell whether a chunk actually passed", against the guards rather
than an implementation, since there is no C++ yet:

* `fen_of` reverted to python-chess's default (defect 1 above): caught on the
  first pair — `ep_no_capturer.a: keys do not survive a FEN round trip`.
* `nn_key` blinded to token 66 — the original ep cache-key defect, reintroduced:
  the partition check catches it, but **only at full corpus size**. It passes
  on the first 20,000 rows and fails somewhere before 50,000. The corpus holds
  just 32 groups of positions that share placement, side and castling while
  differing in the raw ep square, and a smoke run under `--limit` can easily
  contain none of them. Two consequences: `--limit` runs do not certify this
  property, and the constructed ep-twins are not redundant with the corpus —
  they are the only coverage that does not depend on a 32-in-100,000 accident.
* FNV truncated to 16 bits: the collision guard fires after 478 positions; to 24
  bits, after 4,327. At 32 bits it does not fire in 60,000 positions, which is
  correct — ~0.4 collisions are expected there. The guard tracks the birthday
  bound rather than firing on anything unusual-looking.

Independently validated by a checker that re-derives FNV, both payloads and all
three ep rules from the scope text without importing the generator: 100,000 rows
reproduced, corpus identical to `golden/movegen.jsonl` FEN-for-FEN and in order,
and `nn_key` cross-checked against the committed `golden/tokens.npz` —
100,000/100,000 agree, which ties the C2 and C3 artifacts to each other.

## Not done

* No `tests/test_c3_keys.py`. Writing the acceptance test is part of the C++
  chunk; this was golden data only.
* `rep_key` is a hash of `_transposition_key()`'s fields, not the tuple itself,
  so it inherits a collision probability the Python engine does not have. At
  100k entries that is ~2.7e-10, and the corpus shows none. If repetition
  detection is ever seen misfiring, this is the first place to look.

## Global Rule 1

Nothing under `tests/` was touched. `golden/movegen.jsonl` and
`golden/tokens.npz` were read and not modified; `golden/keys.jsonl` and
`golden/keys_adversarial.jsonl` are new files, which is what this chunk's
golden-data step is for. The generator refuses to overwrite either output unless
`--force` is passed, so a later accidental re-run cannot regenerate committed
golden data.

---

# C3 â€” Two keys, never mixed (2026-08-06) â€” the C++ side

The entry above covers the golden data. This one covers `cpp/keys.hpp`, the
bindings, `tests/test_c3_keys.py`, and the one change to `cpp/tokens.hpp`.

## Acceptance criterion 1: the audit of `tools/gen_key_golden.py`

Re-run independently rather than read. A checker was written that imports
neither the generator nor anything from `cpp/`, re-deriving FNV-1a, both
payloads and all three en-passant rules from the scope text and python-chess
1.11.2 alone, so agreement is evidence rather than a tautology. It confirmed:

| claim | result |
|---|---|
| `keys.jsonl` is the shared C1 corpus, same FENs in the same order | 100,000/100,000, identical to `golden/movegen.jsonl` |
| every `nn_key` recomputes from `core.mctsv4.board_to_tokens` | 100,000/100,000 |
| every `rep_key` recomputes from `_transposition_key()`'s fields | 100,000/100,000 |
| `nn_key` partitions the corpus exactly as `make_cache_key` does | identical, 96,068 classes each â€” the figure the chunks doc quotes |
| `nn_key != rep_key` everywhere (domain tags) | 100,000/100,000 |
| every FEN is canonical (`en_passant="fen"`) and round-trips to its keys | clean, corpus and pairs |
| no FEN maps to two different key pairs anywhere in the data | clean |
| adversarial pairs | 22: 7 ep_twin, 5 clock_twin, 5 transposition, 5 near_miss |
| every pair's declared relation actually holds | clean |

En-passant census, which is the number that matters for this chunk: 3,610
corpus positions carry an ep square under the **raw** rule, 407 under the
**legal-capture** rule, 410 under **Polyglot's pseudo-legal adjacency**. The
last two disagree on exactly 3 positions, and they are the three carried as
`ep_pinned_1..3` in the adversarial file. Nothing in the generator needed
changing and nothing was changed.

The brief's `nn_key` wording ("derived from Polyglot Zobrist hashing plus raw
en-passant square state") and the generator's mechanism (FNV-1a over the 68
tokens) still read as a conflict in isolation. `docs/guofish_port_chunks.md`
Â§C3 settles it explicitly â€” "**Mechanism:** scope Â§2.4 â€” a hash of the 68-token
sequence itself â€¦ **Constraint:** *partition* equality with the post-fix Python
`make_cache_key`, not value equality" â€” and that is what the generator does and
what the audit verified. See the previous entry for the equivalence argument.

## Strong key types

`NNKey` and `RepKey` are distinct structs with one `std::uint64_t` member and an
`explicit` constructor. Not `using NNKey = std::uint64_t;`, which the brief
warns against and which is worth restating: an alias is the *same type*, so a
repetition check handed a cache key compiles silently and is wrong only at
runtime, which is the entire defect class this chunk exists to close.

Rejected alternative: a single `template <class Tag> struct Key`. It gives the
same separation in less code, and was not used because the two keys are not two
instances of one idea â€” they answer different questions, and C6/C7 will want to
hang different operations off them. Two named structs cost eight lines.

What is asserted at compile time in `cpp/keys.hpp`, so that a build in which any
of it is false does not produce a module:

* `is_invocable_v<decltype(void(RepKey)), NNKey>` is false, and the same the
  other way. This is acceptance criterion 3 stated in its own words.
* neither converts to, is constructible from, or is assignable from the other;
* neither converts to or from a bare `std::uint64_t` in either direction;
* `nn == rep` does not compile â€” there is no heterogeneous `operator==`, so a
  swapped key cannot even be *tested* â€” while `nn == nn` does;
* `sizeof` is 8 and both are trivially copyable, i.e. the safety is free.

`guofish_core.key_type_separation()` reports those same `<type_traits>` answers
to Python so `tests/test_c3_keys.py` can assert them where someone is looking,
rather than in a build log nobody re-reads. A compile-fail test that shells out
to the compiler was considered and dropped: it would need the chess-library
include path resolved at test time on both platforms, and it would prove nothing
the static asserts do not already prove at every build.

Deliberately **not** provided: a default constructor. A default-constructed key
is a valid-looking key for no position. C7 will need one for hash-table slots
and should add it then, with an explicit empty sentinel, rather than inherit a
zero that means "the position whose payload hashed to the offset basis".

## One parse, two keys (the `cpp/tokens.hpp` change)

`tokenize_into()` used to walk the FEN itself. It now takes a `ParsedFen` â€” the
placement bitboards, side to move, cleaned castling rights and raw ep square â€”
produced by a new `parse_fen()`, and the old `string_view` overload is a
one-line wrapper, so C2's behaviour is unchanged. `Placement` gained the four
missing piece-type bitboards (`pawns`, `knights`, `bishops`, `queens`), filled
in the same pass that was already filling `rooks` and `kings`.

The alternative was a second parser in `keys.hpp` for `rep_key`'s fields. That
is precisely the failure this chunk is about: two parsers can drift on a
castling edge case and produce two internally consistent keys for two different
boards, and every test here would still pass because each key would be
self-consistent. There is now exactly one place where a FEN character becomes a
bit. `rep_key`'s castling field is python-chess's `clean_castling_rights()`
bitboard, which C2 already had to implement correctly for token 65.

## The legal-en-passant rule: implemented, not delegated

`rep_key` needs `has_legal_en_passant()`. Three ways to get it:

1. **`Board::setFen()` then `enpassantSq() != NO_SQ`.** chess-library validates
   the ep square on the way in via `isEpSquareValid`, which is a real legality
   test â€” `checkMask`, both pin masks, and `generateEPMove`, which handles the
   horizontal double-pawn pin explicitly and cites `7k/4p3/8/2KP3r/8/8/8/8` in a
   comment. Five lines, and very nearly right.
2. **Transcribe python-chess's decomposition** â€” `pin_mask`, `_ep_skewered`,
   `_generate_evasions`. Faithful by construction, and a second movegen.
3. **Compute the predicate the decomposition stands for** (chosen): apply
   python-chess's three pseudo-legal conditions to bitboards, then for each
   capturer build the post-capture occupancy and ask whether the mover's king is
   attacked.

(3) was chosen over (1) for two reasons. `isEpSquareValid` checks the horizontal
skewer but not the **diagonal** one; python-chess checks both, with a comment
saying the diagonal case "is not actually possible in a real game, because if
the latest double pawn move covers a diagonal attack, then the other side would
have been in check already". That is true of positions reached by legal play and
not true of an arbitrary FEN handed to a public entry point, and `rep_key(fen)`
is one. And (1) costs a full `setFen` â€” castling paths, Zobrist, ep validation â€”
for a question that is a handful of bitboard operations, on a path C6 calls once
per node; it also inherits `setFen`'s refusal of kingless boards, which C2 went
out of its way to keep accepting.

(3) over (2) because one occupancy-and-attack test *is* the predicate the pin
masks and skewers decompose. Pins in every direction, both skewers, and the
already-in-check case all fall out of it, in about fifteen lines, with nothing
to keep in sync with python-chess's internal structure.

Only `chess::attacks::bishop/rook/knight/king/pawn` are borrowed â€” pure
functions of (square, occupancy) with no board state and no opinion about en
passant. `chess::attacks::initAttacks()` is called from the module initialiser:
it already runs at load time via a dynamic initialiser inside `chess.hpp`, so
this is redundant today, and it is there so the dependency is stated by the code
that has it rather than inherited from an unrelated inline variable.

**Known boundary, stated because it is real.** On a FEN where the ep square is
set but no enemy pawn stands on the square behind it, python-chess's answer is
an artifact of its decomposition rather than a rule anyone wrote down, and this
implementation may differ from it. No such FEN can arise from legal play, none
is in the corpus, and both C6 and C7 feed this from positions reached by making
moves. Reproducing python-chess's behaviour on inputs it has no defined
behaviour for was judged not worth a second movegen. If `rep_key` is ever
suspected on a hand-written FEN, this is the first thing to check.

Verified rather than argued: all 100,000 corpus `rep_key`s match the reference
value-exactly, including the 3,610 raw-ep positions, the 3,203 with no legal
capture, and the 3 where Polyglot's rule disagrees with the legal one.

## Byte order

Both payloads are serialised a byte at a time, little-endian, rather than
`memcpy`-ing the object representation of the token buffer or the bitboards. The
golden payloads are defined as little-endian byte strings (`struct.pack("<...")`,
`astype("<i4")`), and writing the byte order out removes the host's from the
answer. FNV-1a is a byte-at-a-time loop regardless, so this costs nothing.

## Python surface

`nn_key(fen)` and `rep_key(fen)` return plain ints; `keys(fen)` returns both from
one parse; `nn_keys`/`rep_keys`/`key_pairs` are the batched forms, which
materialise the FENs and borrow their UTF-8 buffers under the GIL and then run
the sweep with it released, exactly as `TokenBatch.fill()` does.

The strong typing cannot cross the boundary â€” Python has no way to reject
`cache[rep_key(fen)]` â€” and no attempt was made to fake it with wrapper classes,
which would add a per-key allocation to buy a guarantee Python cannot enforce
anyway. What protects the Python side is the domain tags: the two keys for one
position are never equal, which `test_the_two_keys_never_coincide` checks over
all 100k. After C7 owns the cache, no Python code holds either key at all.

## Mutation check

Golden mutations, run against copies in a scratch directory via the
`GUOFISH_GOLDEN_KEYS` override so nothing under `golden/` is ever written
(sha256 of `golden/keys.jsonl` confirmed identical before and after):

* one `nn_key` bit flipped on a unique row â†’ `test_values_match_golden_exactly`
  fails, naming row 7, its FEN, and both keys in hex.
* one `nn_key` bit flipped on a row that *shares* its key with an earlier one â†’
  the partition test fails too, and says which row the reference groups it with,
  both FENs, and that the reference now has 96,069 classes where C++ has 96,068.
* one `rep_key` bit flipped â†’ same, on the `rep` side.

Implementation mutations â€” the historical defects, reintroduced on purpose,
rebuilt and run:

* **`nn_key` blinded to en-passant squares with no legal capture** (the original
  cache defect, a key coarser than the network's own input): 7 tests fail,
  including the partition test and the coverage test whose message names it.
  C2's tokenizer tests stay green, correctly â€” the tokenizer was not touched.
* **`rep_key` on the raw ep square** (nn_key's rule, misapplied): 7 tests fail.
* **`rep_key` on Polyglot's pseudo-legal adjacency** (the third rule): 4 tests
  fail â€” and **the partition test is not one of them**.

That last result is the important one, and it changed what this suite asserts.
The two rules disagree on 3 corpus positions, all of which are singletons in the
partition, so relabelling them leaves the equivalence classes identical.
**Partition equivalence over the corpus â€” the criterion the chunk brief names â€”
is blind to a `rep_key` built on Polyglot's rule.** What catches it is the
`ep_pinned_*` adversarial pairs and the value comparison. Two consequences worth
carrying forward: the constructed pairs are not redundant with the 100k sweep,
they are the only coverage of the defect the chunk is named after; and
`test_values_match_golden_exactly` is kept even though the brief says value
equality is "neither required nor sufficient", because it is the cheapest thing
that fails when the partition test cannot. Its docstring says what to conclude
if it is ever the only failure.

## Build and sanitizers

Warning-clean at `/W4` (MSVC 19.51) and `-Wall -Wextra` (Clang 18.1.3), on a
forced full rebuild of the translation unit on both, with no pragmas and no
`-Wno-*`. Full suite, 172 tests, green in four configurations:

| toolchain | config | result |
|---|---|---|
| MSVC | Release | 172 passed in 15.42s |
| MSVC | Debug + `/fsanitize=address`, asserts live | 172 passed in 97.81s |
| Clang | Debug | 172 passed in 13.01s |
| Clang | Debug + `-fsanitize=address,undefined`, asserts live | 172 passed in 22.52s |

Under Clang ASan+UBSan: no non-leak sanitizer errors, no UBSan runtime errors,
and no leaked allocation whose stack mentions `guofish_core` (the ~1.3 MB LSan
reports is CPython's and numpy's interpreter-exit leakage, unchanged from C2 â€”
see `build/asan-leakcheck.sh` for why it is answered this way rather than
suppressed).

`chess::Square(int)` and `generateEPMove` both assert on their preconditions, so
the ASan/debug run is a real test of the ep path: an ep square on the wrong rank
would abort rather than return a wrong answer. It cannot reach them â€” the
capturer-rank filter returns first â€” but the assert in `has_legal_en_passant`
that pins that reasoning is live in that build.

## Not done

* No compile-fail test that invokes the compiler; the static asserts cover the
  same ground at every build. See above.
* `rep_key` remains a 64-bit hash of `_transposition_key()`'s fields rather than
  the tuple itself, so it carries a collision probability the Python engine does
  not (~2.7e-10 at 100k entries; none in the corpus). Unchanged from the
  golden-data entry, and still the first place to look if repetition detection
  is ever seen misfiring.
* Neither key is incremental. C6 and C7 call them per position from a FEN or a
  `ParsedFen`. If that shows up in the C12 profile, an incremental update is
  possible for `rep_key` â€” and would reintroduce exactly the re-derived-ep risk
  this chunk was written to eliminate, so it should not be attempted without
  re-running this suite against it.

## Global Rule 1

Nothing under `tests/` was modified: `tests/test_c3_keys.py` is a new file and
`git status` over `tests/` shows one untracked file and nothing else. Nothing
under `golden/` was written â€” the mutation check ran against copies in a scratch
directory through an environment override, and the sha256 of
`golden/keys.jsonl` is unchanged
(`a901750f28aa37490ac96c2d1a321e80ad50a175f1eaa5010820b519642ca504`).
`tools/gen_key_golden.py` was audited and not edited.

---

# C4 — Arena and node storage (2026-08-06)

`cpp/arena.hpp` holds the layout, the Q32 helpers, the packed-move encoding and
the `NodeArena<Accumulator>` template; `cpp/bindings.cpp` binds it for
`tests/test_c4_arena.py`. There is no golden data for this chunk and there could
not be — what C4 delivers is a shape, not an answer — so the acceptance is a set
of structural properties plus one exhaustive numeric sweep.

## The decision the chunk was run to make: Q32 stays in the hot path

The brief says to decide here, not in C12, whether the Q32 -> double conversion
is affordable inside the sibling scan. **It is: 4.8% on Windows and 5.1% on
Linux at the 2.1M-node working set the scope budgets, 9.2% / 3.6% L1-resident.**
Full tables and the rejected alternative are in BENCH.md; the short version is
that a parallel `float q[]` array would remove one `cvtsi2sd`+`mulsd` per child
and pay for it with a fourth stream in the scan (+25% memory traffic on the row
that is already memory-bound), a second value that can disagree with
`value_sum`, and a write on the backup path with no atomic story. It is slower
*and* less safe on the configuration that matters.

Two things about the benchmark that decide how much to trust it:

* **The loop is deliberately cheaper than real PUCT** — no cpuct, no
  `sqrt(parent visits)`, no virtual loss, because that arithmetic is C5's. The
  real loop does strictly more work per child, so the conversion's share of it
  is *smaller* than the table shows. Conservative in the right direction.
* **The hot working set is byte-identical between the two accumulators**
  (`atomic<double>` is also 8 bytes), so the columns differ only in ALU work.
  Had the double arena been narrower, the comparison would have been measuring
  cache behaviour and calling it conversion cost.

MSVC's memory-bound row is 2.2x slower than Clang's (89 vs 40 ns/scan) with both
columns moving together, so it is a codegen difference rather than an
accumulator one. Recorded in BENCH.md rather than chased: it does not change the
ratio, and C12 is where scan codegen becomes worth attacking.

## What Q32 actually is, and why the scale is a power of two

`v * 2^32`, rounded, in an `int64`. The scale is not tuning:

* `q * 2^-32` only changes a double's exponent, so it is **exact** for every
  |q| <= 2^53. The int -> double -> int round trip is therefore exact by
  construction over the whole code range, not merely measured to be.
* Resolution 2^-32 = 2.3283e-10, against a network emitting bf16 with 8 mantissa
  bits — seven orders finer than its own input.
* Overflow at 2^31 = 2.1e9 visits of magnitude 1, against a 15k-sim budget.

**Rounding is `std::llround`, i.e. half away from zero.** Chosen over truncation
and over `rint`'s half-to-even because the standard pins it, both toolchains
implement it identically, and it is symmetric about zero — an asymmetric
quantizer would bias every backup in one direction by half a tick. Ties are
reachable, not theoretical: a float in [2^-10, 2^-9) has an ulp of 2^-33, so
`v * 2^32` lands exactly on k + 0.5 for half of them.

The obvious faster spelling — `(int64)(v * scale + copysign(0.5, v))` — is
exactly equivalent on this domain (|scaled| <= 2^32, so `x + 0.5` is exact) and
was not used. `to_q32` is called once per backup, not once per child in the
scan, so it is not on the path the benchmark above measures, and correctness
that is obvious beats correctness that requires the preceding sentence.

### "Exhaustive" means every float, and that needed defining

The brief asks for "exhaustive round-trip tests across the representable range
[-1.0, 1.0]". Two ranges could be meant and they are not the same size:

* **every `float` in [-1, 1]** — 2,130,706,434 bit patterns, denormals included.
  This is the domain a network value actually comes from, and it is what
  `q32_roundtrip_sweep(1)` walks. 9.7 s on MSVC Release, 7.0 s on Clang, 52 s
  under MSVC ASan/Debug. Run in full in every configuration.
* **every Q32 code in [-2^32, 2^32]** — 8,589,934,593 of them, which is not
  testable in a suite anyone will run. `q32_code_sweep` walks it at a stride and
  pins every boundary code unconditionally.

The second is not weaker than it looks, and the reason is worth stating because
it is the only place this chunk substitutes an argument for a measurement:
`from_q32` multiplies by a power of two, which is exact in double for
|q| <= 2^53, so `to_q32` is handed the integer `q` back unchanged. The sweep
exists to catch that premise breaking, not to establish the property. Both
`q32_roundtrip_sweep`'s `code_mismatches` counter and `q32_code_sweep` report 0.

The exhaustive sweep's results are exact, not approximate:

| property | result |
|---|---|
| max absolute round-trip error | 2^-33, exactly half the tick |
| largest value that is not bit-exact | the float just below 2^-9 |
| Q32 -> double -> Q32 mismatches | 0 |
| asymmetric roundings | 0 |

**Every float at or above 2^-9 round-trips bit-identically**, because at that
magnitude a float's ulp is already 2^-32. That boundary is asserted directly
rather than the inexact *count*, because a count is a number nobody can check
and a boundary is a claim that either holds or does not.

## Both accumulators are compiled in every build

The brief allows "templated or `#ifdef`-switched". Templated, and both
instantiations are bound to Python in **every** build as `NodeArenaQ32` and
`NodeArenaDouble`; `GUOFISH_VALUE_SUM=double` only selects which one the
`guofish_core.NodeArena` alias and `guofish::DefaultArena` name.

*Alternative:* a real `#ifdef` on the storage type, so one build has the Q32
arena and the other has the double one. *Why not:* the brief requires "both must
be exercised by tests", and under an `#ifdef` each test run exercises exactly
one. The equivalence build is a rarely-flipped switch, which is precisely the
configuration that rots — a compile error in the `atomic<double>` branch would
be found at Gate 1, months after it was introduced. Templating costs nothing
here (the two policies are eight lines each) and lets
`test_q32_accumulation_is_order_independent_and_double_is_not` compare the two
in one process, which is the test that states *why* there are two.

Both accumulator settings were still built and run through the full suite on
their own, because "the code builds under both" is a separate claim from "both
types compile".

**`DoubleAccumulator::add` is a CAS loop**, not `fetch_add`: C++17 has no
floating-point `fetch_add` (it arrived in C++20). That is not a workaround, it
is the cost the scope names — the retry rate climbs with thread count on an
accumulator every backup touches, and it is one of the two reasons production
uses Q32. Single-threaded, which is what Gate 1 runs, it never retries.

## Terminal-ness: refused at the door, not asserted afterwards

The scope requires terminal to be a distinct bit from expanded so the
`bestmove 0000` defect class is unrepresentable. Implemented as: lifecycle
(`UNEXPANDED`/`PENDING`/`EXPANDED`) in the low two bits of the state byte,
`TERMINAL` at bit 7, and four rejections on the way in:

| refused | why it is a real state, not a hypothetical |
|---|---|
| `set_children(count = 0)` | the defect itself — an expanded node the move selector finds nothing in |
| `set_children` on a terminal node | a node where the game ends has no moves; allowing it reaches "expanded, zero children" by a second route |
| `set_children` on an expanded node | double expansion orphans the first child block, and its visits leave the tree silently |
| `mark_terminal` on a node with children | the same invariant from the other side |

Checked on the way in rather than by an invariant-checking pass, because a pass
has to be called and the call is what gets forgotten. `mark_terminal` ORs the
bit into whatever lifecycle is there rather than replacing the byte, so
`PENDING` + terminal is representable — which it has to be, since a leaf is
claimed on selection but its terminality is only discovered when the position is
examined.

Rejections are typed so Python can tell them apart: `ValueError` for a bad
argument (count of zero), `IndexError` for a range outside the arena,
`RuntimeError` (via `std::logic_error`) for a structural violation.

## Fixed capacity, and a bump pointer that clears what it hands out

**The arena never reallocates.** Growth would move every base pointer while
other threads are mid-scan, and making that safe costs an indirection on the
hottest read path in the engine. The scope sizes this from measurement (~40
nodes/sim, 15k sims, game-long reuse, 2-3M nodes peak), so the allocation
happens once and the bump pointer runs out instead.

`try_allocate` returns `kNoNode` on exhaustion and `allocate` throws; C5's
expansion path uses the former, because a full arena is a condition to handle,
not a bug. The bump is a CAS loop on an `atomic<uint32>` even though expansion
is single-threaded by construction (scope 2.2) — it costs nothing uncontended
and removes a thing to remember in C9.

**`try_allocate` clears the block it returns; `reset()` clears nothing.** The
alternative — clear on reset — makes `reset()` a 19 MB memset for a 2M-node
arena and clears slots that will never be handed out. The chosen split means
C8's ping-pong recycle is O(1) and a recycled node cannot arrive carrying the
previous search's visit count. That failure mode is why it is worth stating: a
stale visit count does not crash, it biases selection toward a node that was
never visited in this search, and it is invisible in any single position.

Consequence: **indices are bounded by `size()`, not `capacity()`.** An
unallocated slot is memory, not a node, and reading one would read whatever the
last search left there.

## Packed moves, and the sort key that is not the packing

`move` is `from << 10 | to << 4 | promo`, squares numbered as chess-library
numbers them (rank * 8 + file).

**A raw integer sort of this packing is not canonical order**, and that is the
trap this section exists to mark. Square indices are rank-major (a1, b1, ... h1,
a2); C1's canonical order is the UCI string's, which is file-major (a1, a2, ...
a8, b1). They disagree on almost every position — measured at more than half the
corpus in `test_sorting_packed_moves_directly_is_the_wrong_order` — and because
PUCT resolves ties by child order, sorting on the wrong one does not fail
loudly: C++ and Python simply explore different moves at equal priors. C1's
DECISIONS entry raised exactly this hazard; C4 is where the field that would
embody it gets defined.

`canonical_move_key(packed)` returns
`from_file << 13 | from_rank << 10 | to_file << 7 | to_rank << 4 | promo`, whose
unsigned order is exactly the byte order of the UCI string.
`test_canonical_move_key_reproduces_c1_ordering_on_the_golden_corpus` drives it
against 4,000 positions of `golden/movegen.jsonl` — C1's own reference, read
only — so the claim is checked against the artifact C1 was judged on rather than
against an argument.

*Alternative considered:* make the packing itself file-major so a raw sort is
canonical. Rejected: packing and unpacking would then need a bit shuffle on
every use, including on the selection path that has to reconstruct a
`chess::Move`, to save a key function that is called once per expansion.

**Promotion codes are alphabetical by UCI letter** — none 0, bishop 1, knight 2,
queen 3, rook 4 — not by piece value. This looks wrong and is what makes the
promotion field already canonical: UCI compares `b < n < q < r`, and a
four-character move sorts before any promotion sharing its from/to because the
absent letter is less than every letter.

`kNoMove` is 0, which is `a1a1` — a legal *pattern* that is not a legal *move*.
There is no spare bit pattern (all 65,536 are well-formed triples), so the
root's move slot holds this and nothing should read it as a move.

## The `is_lock_free()` accessor that Clang deleted

An early version exposed a per-object `atomic<T>::is_lock_free()` to Python,
reasoning that `is_always_lock_free` is a claim about the *type* while the
arrays are about specific objects at specific addresses.

**The Clang build failed to import**: `undefined symbol: __atomic_is_lock_free`.
On libstdc++ the runtime query is an out-of-line call into libatomic, which is
not linked by default. It could have been fixed with `-latomic`, and was not,
because the accessor reports nothing new: the standard says `is_always_lock_free`
is true only if *every* object of the type is lock-free, and the static_asserts
already require it. The property is now asserted as the two things that compose
it — `arena_layout()`'s compile-time flags plus the alignment entries — which is
what actually establishes it.

Recorded because it is Global Rule 8 earning its keep on the first chunk that
touches atomics, and because the MSVC build linked it without complaint.

## Alignment: reported against constants, not against the request

Each array is a separate over-aligned allocation
(`::operator new(n, align_val_t(64))`, C0's spelling, for C0's portability
reasons) at a **full cache line** rather than at `alignof(T)`. Natural alignment
is what correctness requires; the cache line is so that a 32-sibling scan reads
the minimum number of lines rather than one extra at each end.

`assert()` is compiled out under NDEBUG, so the constructor's checks are live in
exactly the builds that do not ship. `array_info()` therefore reports every base
address to Python and the tests assert on it in whatever build is running.

**The mutation drill found a defect in this reporter**, and it is the most useful
thing the drill did. `is_aligned()` originally asked
`address % alignment() == 0` — against the alignment that had been *requested*.
Under the mutation that cut `alignment()` to `alignof(T)`, it dutifully reported
"aligned", and `test_alignment_holds_across_repeated_allocation` passed while the
arrays were no longer cache-line aligned at all. A check that answers its own
question is not a check. It now asks against `kCacheLine` and `alignof(T)`
directly, and the test additionally asserts `requested_align >= CACHE_LINE` so
the request itself is pinned. Post-fix, that mutation fails 8 tests instead of 6.

## Mutation check

C4 has no golden data, so the implementation was mutated, per the procedure C0
and C0b used. Amendment B: the drill ran against the working tree with pristine
copies held in a scratch directory and restored after every mutation; nothing
under `golden/` was written, and the sha256 of all four golden files is
unchanged (`movegen.jsonl` `1754e3aa...de6151a2`).

Twelve mutations, one per property the suite claims. All twelve were caught:

| mutation | result |
|---|---|
| M1 array alignment cut to `alignof(T)` | 8 failed — both cache-line tests and both repeated-allocation tests |
| M2 `set_children` accepts zero children | 2 failed — `test_a_node_cannot_be_expanded_with_no_children` |
| M3 `set_children` no longer refuses a terminal node | 2 failed |
| M4 `mark_terminal` no longer refuses a node with children | 2 failed |
| M5 `to_q32` truncates instead of rounding to nearest | 3 failed — including the exhaustive sweep, whose max error doubles |
| M6 Q32 scale is 2^31 | 12 failed |
| M7 `canonical_move_key` is the identity | 2 failed — both golden-corpus ordering tests |
| M8 `allocate()` hands out blocks without clearing them | 2 failed — the recycle test |
| M9 `child(i, k)` ignores `children_offset` | 6 failed |
| M10 `set_children` does not validate the child range | 2 failed |
| M11 `try_claim_pending` is a load-then-store, not a CAS | 2 failed |
| M12 `mark_terminal` clobbers the lifecycle instead of ORing the bit | 2 failed — the PENDING-then-terminal test |

M1's original result (6 failures) is what exposed the reporter defect above; the
8 in the table is post-fix. Every mutation was reverted and the suite re-run.

## Build and sanitizers

Warning-clean at `/W4` (MSVC 19.51) and `-Wall -Wextra` (Clang 18.1.3) on a
forced full rebuild of the translation unit on both, no pragmas and no `-Wno-*`.
Clang caught one unused helper (`bits_from_float`) that MSVC did not; it was
deleted rather than suppressed.

Full suite, 294 tests (122 of them C4's), green in seven configurations:

| toolchain | config | accumulator | result |
|---|---|---|---|
| MSVC | Release | q32 | 294 passed in 29.12s |
| MSVC | Release | double | 294 passed in 29.11s |
| MSVC | Debug + `/fsanitize=address`, asserts live | q32 | 294 passed in 238.54s |
| MSVC | Debug + `/fsanitize=address`, asserts live | double | 294 passed in 226.64s |
| Clang | Debug | q32 | 294 passed in 33.36s |
| Clang | Debug | double | 294 passed in 33.11s |
| Clang | Debug + `-fsanitize=address,undefined`, asserts live | q32 | 294 passed in 55.24s |

The one pytest warning in the Clang runs is C0b's documented
`max`-on-a-non-authoritative-platform notice, unrelated to this chunk.

No ASan or UBSan runtime errors. The arena's nine allocations per instance were
leak-checked with the quantitative method README_BUILD.md prescribes rather than
by reading the summary: 1 vs 500 `(NodeArenaQ32(20000), NodeArenaDouble(20000))`
pairs give a **byte-identical** 941,456 bytes in 858 allocations — the same
CPython/numpy baseline C0 and C2 measured — where a leak would have added
~660 MB. No leaked allocation's stack mentions `guofish_core`, in that check or
in the full ASan/UBSan suite run (whose 1,348,338 bytes in 1,252 allocations is
the same interpreter-exit leakage C3 recorded).

The ASan/Debug runs are a real test of the Q32 domain, not only of memory:
`to_q32` asserts `-1 <= v <= 1` and rejects NaN, so a value outside the network's
range reaching the accumulator aborts rather than converting to an unspecified
integer.

## Not done

* **No concurrency test.** Every atomic here is exercised single-threaded.
  `try_claim_pending`'s CAS is verified to be a CAS (the second claim fails), but
  nothing runs two threads at it — pybind11 holds the GIL across these calls, so
  a Python-level thread test would prove nothing it did not already know. Real
  contention is C9's, and C9 should not treat C4's green suite as evidence about
  it.
* **No false-sharing padding.** Two threads backing up through adjacent siblings
  write `visit_count` entries 4 bytes apart, i.e. on the same line. That is
  inherent to SoA and is the trade the layout makes — the scan wants them
  adjacent. If C9's scaling curve flattens early this is the first suspect, and
  the fix (padding, or per-thread shards) is a change to this file. Flagged, not
  pre-optimised.
* **No `chess::Move` interop.** `pack_move` takes integers. C5 will want
  `pack_move(chess::Move)` including the king-takes-rook castling normalisation
  C1 established; it is not here because there is no golden data for it in this
  chunk and it would ship unverified.
* **No ping-pong compaction.** C8's, explicitly out of scope. The pieces it will
  need are in place: offsets not pointers throughout, and `set_children`
  validates every remapped range against the live region, which is where a fixup
  off-by-one surfaces.
* **`set_move` does not validate its argument.** It is a raw `uint16` field
  setter; `pack_move` is the validating constructor. A promotion code above 4 can
  be stored and would decode to a nonsense promotion. Deliberate — the setter is
  on the expansion path — but it means `move` is only as canonical as its writer.

## Global Rule 1

Nothing under `tests/` was modified: `tests/test_c4_arena.py` is a new file and
`git status` over `tests/` shows one untracked file and nothing else. Nothing
under `golden/` was written — see "Mutation check" for the before/after hashes.
`golden/movegen.jsonl` is read by the canonical-ordering tests and never opened
for writing.
