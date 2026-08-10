# Building `guofish_core`

The C++ core builds as a Python extension module named `guofish_core`. Two
toolchains are supported and both are exercised on every chunk: **Windows/MSVC**
is production, **Linux/Clang** exists because ThreadSanitizer does not exist on
MSVC and C9 will need it.

Dependencies (`pybind11`, `Disservin/chess-library`, `jdart1/Fathom`) are fetched
by CMake at configure time and pinned to immutable revisions. Nothing needs
installing by hand; the first configure needs network access.

Fathom (C7) is the Syzygy tablebase prober and is the only C in the build — one
translation unit, compiled into a static `fathom` target and linked into the
module. Two things about it are worth knowing before you touch the CMake:

* **Warning flags are per-target, not directory-wide.** `/W4` and
  `-Wall -Wextra` are applied with `target_compile_options(guofish_core …)`, so
  our code stays exactly as strict while Fathom's warnings are not ours to
  suppress (Global Rule 4 forbids `-Wno-*` and pragmas). If you add a target,
  add `${GUOFISH_STRICT_WARNINGS}` to it deliberately.
* **MSVC needs `/experimental:c11atomics`** to compile `<stdatomic.h>`, which
  Fathom uses for the synchronisation that makes `tb_probe_wdl` thread-safe.
  Defining `TB_NO_THREADS` instead would compile everywhere with no flag and
  would remove exactly the property C9 needs. Clang needs nothing.

The tablebase files themselves are **not** a build dependency. `assets/syzygy`
holds a 5-man set; without it the tablebase tests skip and the engine runs with
tablebases off, which is the default.

The test suite additionally needs `numpy`, `pytest` and — for the C3b reference
tests and C7's tablebase oracle — `python-chess` (pinned by the golden data at
1.11.2). Install it in the Linux venv too, or C7's tablebase section silently
skips there.

## CMake options

| Option | Default | Meaning |
|---|---|---|
| `GUOFISH_ASAN` | `OFF` | AddressSanitizer; also UndefinedBehaviorSanitizer on Clang. Strips `/RTC1` and disables incremental linking on MSVC. Keeps `assert()` live even in release configs. |
| `GUOFISH_TSAN` | `OFF` | ThreadSanitizer. **Linux/Clang only** — TSan has no MSVC implementation, and CMake fails the configure rather than producing a build that silently checks nothing. Mutually exclusive with `GUOFISH_ASAN`. C9's acceptance requires a clean run; see "Linux, ThreadSanitizer" below. |
| `GUOFISH_MODULE_OUTPUT_DIR` | repo root | Where the built `.pyd`/`.so` is written. |
| `GUOFISH_VALUE_SUM` | `q32` | Which accumulator `guofish::DefaultArena` and `guofish_core.NodeArena` name: `q32` (production) or `double` (Gate 1 equivalence). Both `NodeArena` types are compiled and bound in *every* build; this only selects the default. |
| `GUOFISH_DEBUG_VL` | `ON` for `Debug`, else `OFF` | Compile `ReplaySearch.debug_total_vloss()`, C8's read-only full-tree virtual-loss audit. Its absence from a Release build is the point — C8 forbids a production equivalent of the reference's defensive `_reset_virtual_loss` walk — so the flag is reported as `guofish_core.DEBUG_VL` and `tests/test_c8_reuse.py` asserts both halves: the invariant where the audit exists, the absence where it should not. |

Windows and Linux artifacts have different suffixes
(`guofish_core.cp313-win_amd64.pyd` vs `guofish_core.cpython-312-x86_64-linux-gnu.so`)
so they can coexist in the same directory. Two builds for the *same* platform
(e.g. Release and ASan) cannot — they overwrite each other.

> **Delete the module before switching between two build directories that share
> an output path.** Ninja decides an output is up to date by comparing its mtime
> against its inputs. After `build/msvc-release` writes the `.pyd`, a
> `cmake --build build/msvc-asan` sees an output newer than its inputs and
> reports `ninja: no work to do` — leaving the *release* module in place while
> you believe you are testing the sanitized one. This fails silently and it will
> tell you the sanitizer found nothing.
>
> ```bat
> del guofish_core.cp313-win_amd64.pyd
> cmake --build build/msvc-asan
> ```
>
> Confirm you got what you asked for: the ASan build logs
> `Staging ASan runtime next to guofish_core` when it actually relinks, and the
> suite runs visibly slower under ASan (~0.24 s vs ~0.11 s here).

---

## Windows (MSVC)

Requires Visual Studio with the C++ workload. CMake and Ninja ship with it; the
paths below are for a VS 18 Community install, adjust the `VS` variable if yours
differs.

Open a plain `cmd.exe` and run:

```bat
set "VS=C:\Program Files\Microsoft Visual Studio\18\Community"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat"
set "PATH=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

cd /d "C:\Users\Ethan Guo\Github\GuoFish"

cmake -S . -B build/msvc-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/msvc-release

python -m pytest tests/ -v
```

Or, from any shell (it runs `vcvars64.bat` for you):

```bat
build\win.bat build/msvc-release Release
build\win.bat build/msvc-asan Debug asan
build\win.bat build/msvc-release-double Release - double
```

The fourth argument is `GUOFISH_VALUE_SUM`; `-` in the third slot means "no
ASan". Since both arena types are compiled either way, this switch only needs
exercising when you want to confirm the *default alias* is wired correctly —
`guofish_core.DEFAULT_ACCUMULATOR` reports what you got.

Arguments are positional because `cmd` splits argv on `=`, so a
`-DCMAKE_BUILD_TYPE=Release` forwarded through `%1` arrives at CMake as two
separate arguments.

### Windows, AddressSanitizer

```bat
cmake -S . -B build/msvc-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DGUOFISH_ASAN=ON
cmake --build build/msvc-asan

python -m pytest tests/ -v
```

The build stages `clang_rt.asan_dynamic-x86_64.dll` next to the module
automatically. This is required, not a convenience: since Python 3.8 the
interpreter no longer searches `%PATH%` when resolving an extension module's
dependencies, so putting the ASan runtime on `PATH` does **not** work — it has to
sit beside the `.pyd`.

MSVC has no LeakSanitizer (LSan is not implemented on Windows), so the Windows
ASan run catches memory errors but says nothing about leaks. Leaks are covered on
the Linux side below.

To go back to a normal build afterwards, rebuild the release configuration and
delete the staged runtime:

```bat
cmake --build build/msvc-release
del clang_rt.asan_dynamic-x86_64.dll
```

---

## Linux (Clang)

Production is Windows; this is a WSL2 Ubuntu 24.04 distro. One-time setup:

```bash
sudo apt update
sudo apt install -y clang cmake ninja-build python3-dev python3-venv git
python3 -m venv ~/.venvs/guofish
~/.venvs/guofish/bin/pip install numpy pytest
```

`python3-dev` is not optional — pybind11 needs `Python.h`. Ubuntu 24.04 enforces
PEP 668, so the venv is mandatory rather than merely advisable.

### Why the build output does not go in the repo

The checkout lives on `/mnt/c`, which WSL serves over DrvFS. Two things follow:

1. `llvm-strip` fails with `Operation not permitted` on DrvFS, and pybind11
   strips `Release`/`MinSizeRel` builds automatically. A Release build writing
   into the repo therefore *fails to link*.
2. DrvFS is slow for the many small files a build produces.

So Linux builds write the module into the build directory on ext4 and put that
directory on `PYTHONPATH`. Keep the sources on `/mnt/c` — one checkout, editable
from Windows.

```bash
SRC="/mnt/c/Users/Ethan Guo/Github/GuoFish"
VENV="$HOME/.venvs/guofish"

cmake -S "$SRC" -B ~/build/gf-release -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DPython_EXECUTABLE="$VENV/bin/python" \
    -DGUOFISH_MODULE_OUTPUT_DIR="$HOME/build/gf-release"
cmake --build ~/build/gf-release

cd "$SRC"
PYTHONPATH="$HOME/build/gf-release" "$VENV/bin/python" -m pytest tests/ -v
```

Note `python -m pytest` prepends the working directory to `sys.path`, so a stale
`.so` sitting in the repo root will shadow `PYTHONPATH`. If you ever built with
the default output directory, delete
`guofish_core.cpython-*-linux-gnu.so` from the repo root.

### Linux, AddressSanitizer + UndefinedBehaviorSanitizer

```bash
cmake -S "$SRC" -B ~/build/gf-asan -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DPython_EXECUTABLE="$VENV/bin/python" \
    -DGUOFISH_MODULE_OUTPUT_DIR="$HOME/build/gf-asan" \
    -DGUOFISH_ASAN=ON
cmake --build ~/build/gf-asan

cd "$SRC"
LD_PRELOAD="$(clang -print-file-name=libclang_rt.asan-x86_64.so)" \
ASAN_OPTIONS=detect_leaks=1 \
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
PYTHONPATH="$HOME/build/gf-asan" \
    "$VENV/bin/python" -m pytest tests/ -v
```

The `LD_PRELOAD` is required. Python itself is not ASan-instrumented, so without
it the import aborts with *"ASan runtime does not come first in initial library
list"*.

### Linux, ThreadSanitizer — C9 acceptance

**TSan does not exist on MSVC.** That is the reason Global Rule 8 requires this
codebase to build on Linux at all: production is Windows, but the only tool that
can prove the C9 descent is race-free runs somewhere else. A clean TSan run over
acceptance layers 2 and 3 is part of C9's acceptance, not an optional extra.

`GUOFISH_TSAN` and `GUOFISH_ASAN` are mutually exclusive — they use incompatible
shadow-memory layouts — and CMake raises a `FATAL_ERROR` rather than leaving it
to a linker message nobody reads. Use two build directories.

```bash
cmake -S "$SRC" -B ~/build/gf-tsan -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DPython_EXECUTABLE="$VENV/bin/python" \
    -DGUOFISH_MODULE_OUTPUT_DIR="$HOME/build/gf-tsan" \
    -DGUOFISH_TSAN=ON
cmake --build ~/build/gf-tsan

cd "$SRC"
LD_PRELOAD="$(clang -print-file-name=libclang_rt.tsan-x86_64.so)" \
TSAN_OPTIONS=halt_on_error=0:history_size=7:second_deadlock_stack=1 \
PYTHONPATH="$HOME/build/gf-tsan" \
    "$VENV/bin/python" -m pytest tests/ -q
```

The `LD_PRELOAD` is required for the same reason it is under ASan: Python itself
is not instrumented, so without it the import fails with `undefined symbol:
__tsan_write_range`.

`halt_on_error=0` so that a run reports *every* race rather than stopping at the
first. Grep the output for `WARNING: ThreadSanitizer`; the count must be zero.

**Verify the sanitizer has teeth before believing a clean run.**
`guofish_core.race_probe()` is a deliberate four-thread data race on a plain
`int`, and `test_thread_sanitizer_can_actually_fail` asserts that it produces a
report on a TSan build and runs to completion on any other. A clean run without
this check is indistinguishable from a sanitizer that was not instrumenting the
module. `guofish_core.TSAN` reports which build you are on.

TSan costs roughly 13x here — the full suite runs in ~12 min against 54 s.

### Reading the leak report

LeakSanitizer will always report roughly 1.4 MB leaked in ~1300 allocations.
**This is CPython and numpy, not us** — both leak interpreter-lifetime
allocations by design.

Do not "fix" this with a suppression file. Every allocation `guofish_core` makes
is reached through Python frames, so a `leak:/usr/bin/python3.12` suppression
would hide our leaks along with theirs. Two checks that actually discriminate:

* No leaked allocation's stack should mention `guofish_core`:
  `grep guofish_core <log>` over the ASan output should find nothing.
* The leak total should not grow with the number of allocations made. Comparing
  a 1-buffer run against a 500-buffer run (500 x 272 KB = ~136 MB) gives a
  byte-identical total when the buffers are being freed properly. C4 applies
  the same test to the arena: 1 vs 500 `(NodeArenaQ32(20000), NodeArenaDouble(20000))`
  pairs, where a leak would add ~660 MB.

Helper scripts for all of the above live in `build/` (gitignored).

---

## Benchmarks

None of these scripts write to `golden/`; they are benchmarks, not reference
implementations (Global Rule 2). Results are transcribed into `BENCH.md`.

| command | what it produces |
|---|---|
| `python tools/bench_c0.py` | C0 table: uncontended GIL round-trip latency, with the live payload |
| `python tools/bench_c0b.py --scale 4` | C0b tables: configurations A/B/C × batch {32, 256} |
| `python tools/bench_c0b.py --sweep` | acquire wait vs `sys.setswitchinterval` — locates the regime change |
| `python tools/bench_c0b.py --repeat 10` | gate stability: 10 × 2,000 samples of config C at batch 256 |
| `python tools/bench_c2.py` | C2 table: tokenization throughput, encoder vs `TokenBatch.fill` |
| `python tools/bench_c2.py --python` | as above, plus the `board_to_tokens` reference on this machine (imports torch) |
| `python tools/bench_c4.py --trials 5` | C4 table: sibling scan under both accumulators, four working sets |
| `python tools/bench_c4.py --sweep` | as above, plus the exhaustive Q32 round-trip sweep over all 2.13e9 floats in [-1, 1] (~10 s on Release, ~52 s under ASan) |
| `python tools/bench_c5.py --markdown` | C5 table: search throughput on the replay evaluator |
| `python tools/bench_c6.py --markdown` | C6 table: what terminal handling costs, both corpora |
| `python tools/bench_c8.py` | C8: arena high-water over whole games, compaction cost |
| `python tools/bench_c9_knee.py --markdown` | **C9a: the GPU evaluation knee.** Needs torch and a CUDA device. Synchronizes around every timed iteration — see BENCH.md on why an un-synchronized measurement produces a different and wrong curve |
| `python tools/bench_c9.py --markdown` | C9b–C9f: the W × K grid with its W=1 control rows, the affinity comparison, and the dispatcher GIL histogram. Replay evaluator only, no GPU |
| `python tools/bench_c9.py --affinity-only --sims 20000` | the affinity table alone, at a budget long enough for the effect to appear — at 2,000 sims pinning measures as noise |
| `python tools/bench_c10.py` | **C10: the live evaluation boundary.** Needs torch, a CUDA device and the v5 checkpoint. Real search throughput at `sys.setswitchinterval` 0.005 vs 0.0005, with and without a competing pure-Python thread, plus the dispatcher's acquire-wait histogram against C10's 200 µs / 1%-of-wall triggers |

On Linux, prefix with the build directory as usual:

```bash
PYTHONPATH="$HOME/build/gf-release" "$HOME/.venvs/guofish/bin/python" tools/bench_c0b.py --sweep
```

**Benchmark on the Release build, not the ASan build.** Instrumentation roughly
doubles the callback cost and materially widens the latency tail — enough to
change how often the C0b gate's `max` criterion is met (see BENCH.md, "Gate
stability"). `tools/bench_c0b.py` and `tools/bench_c2.py` print `asan=True/False`
in their headers so a mistake is visible in the output rather than silently
transcribed; `bench_c2.py` and `bench_c4.py` additionally print a warning banner.

`tools/bench_c0b.py` imports its experiment from `tests/test_c0b_contention.py`
rather than reimplementing it, so the published numbers and the asserted numbers
cannot drift apart. The dependency runs tool → test only.

### These benchmarks are timing-sensitive

They measure OS scheduling tails, so a loaded machine changes the answer. Close
other work before generating numbers for `BENCH.md`. The `p99` figures are stable
under resampling; the `max` figures are not, and `max` is not comparable across
different iteration counts — always read it next to the `iters` column.

---

## Golden data

`golden/` holds the reference answers every parity test is judged against. It is
produced by the Python reference only, by scripts under `tools/`, and it is never
regenerated to make a test pass (Global Rules 1 and 2).

| file | generator | size |
|---|---|---:|
| `movegen.jsonl` | `tools/gen_movegen_golden.py` | 27 MB |
| `tokens.npz` | `tools/gen_token_golden.py` | 3.2 MB |
| `keys.jsonl`, `keys_adversarial.jsonl` | `tools/gen_key_golden.py` | 13 MB |
| `gate1_dump.npz`, `gate1_trees.npz`, `gate1_manifest.json` | `tools/gen_gate1_golden.py` | 19 MB |
| `gate1_terminal_dump.npz`, `gate1_terminal_trees.npz`, `gate1_terminal_manifest.json` | `tools/gen_gate1_golden.py --corpus terminal` | 9 MB |
| `c10_corpus.json` | `tools/gen_c10_corpus.py` | 90 KB |
| `c10_gate2.npz`, `c10_gate2_manifest.json` | `tools/gen_c10_gate2_golden.py` | 3 MB |
| `c10_gate2b.json`, `c10_gate2b_manifest.json` | `tools/gen_c10_gate2b_golden.py --sims 1600` | 1 MB |

The Gate 1 files are the only ones whose generation needs a GPU and the v5
checkpoint. The quiet corpus takes ~35 minutes, the terminal corpus ~25:

```bat
python tools/gen_gate1_golden.py --force
python tools/gen_gate1_golden.py --corpus terminal --force
```

Both run the Python MCTS at the Gate 1 equivalence configuration (1 worker,
`cache_size=1`, tablebase off, Dirichlet off, canonical-ordering patch on), and
they differ in what they do with what they find.

**Quiet (C5)** samples midgame positions from the benchmark PGNs and **rejects
any position whose reference search touches a terminal, draw or depth-cap
path** — that audit is what licensed C5's omission of terminal handling, so a run
that rejects nothing is a run to be suspicious of.

**Terminal (C6)** runs a hand-specified corpus that exists to touch exactly that
machinery, and **records** rather than rejects: every terminal, draw-by-rule hit,
depth-cap hit and early exit is counted into the manifest. It refuses to write
anything if a class never fired, so `gate1_terminal_manifest.json`'s `coverage`
block is a measurement rather than a claim. Useful flags while iterating on a
spec: `--only <name>`, `--limit N`, `--max-sims N` (all of which produce data that
is *not* acceptance-grade, and say so).

Both manifests record the full provenance: interpreter, library versions,
checkpoint SHA-256, the SHA-256 of `core/mctsv4.py`, seed, arguments, every
position and — for the quiet corpus — every rejection with its reason.

The terminal manifest additionally records, per position, the `base_fen` and the
legal `moves` played onto it, plus the `history` those moves produce. The history
is the pre-root game the repetition rule is evaluated against, and it is passed
to the C++ side through `set_position(fen, history)`; without it a threefold is
essentially unreachable inside search range.

### The C10 files: one corpus, two gates

`c10_corpus.json` is 500 game-realistic positions sampled from three benchmark
PGNs with a seeded generator, several plies per game, quota-balanced across the
files. Both C10 gates read it and both manifests record its SHA-256, because a
Gate 2b disagreement whose position Gate 2 never gathered is a disagreement with
nowhere to look. Regenerating the corpus invalidates both goldens, and the tests
say so rather than comparing across a mismatch:

```bat
python tools/gen_c10_corpus.py --force
python tools/gen_c10_gate2_golden.py --force
python tools/gen_c10_gate2b_golden.py --sims 1600 --force
```

**Gate 2 (`c10_gate2.npz`)** takes ~1 minute and needs a GPU: it records the
model's full 4096-wide bf16 policy row per position, plus ATen's priors computed
three different ways — the reference's interior path (CPU softmax, python-chess
generation order), its root path (CUDA softmax, same order), and ATen's CPU
softmax over *chess-library's* generation order. The third column is what
separates the softmax-implementation difference from the permutation variance;
see DECISIONS.md, C10.

**Gate 2b (`c10_gate2b.json`)** takes ~65 minutes and is the reference actually
playing: one worker, 1,600 simulations, cache on, tablebase off, Dirichlet off,
and **no canonical-ordering patch**. That last one is the point — Gate 1 ran the
reference patched so both sides gathered in the same order, and Gate 2b exists to
establish that the ordering choice does not move search behaviour, which it can
only do against unpatched Python. The generator refuses to run with
`mctsv4.GATE1_CANONICAL_ORDER` set, and the test asserts the manifest says so.

Useful while iterating: `--limit N --out <scratch> --manifest <scratch>` runs a
pilot without touching `golden/`.

### The mutation drills

Amendment B: drills never touch `golden/`. Each parity test reads an optional
`GUOFISH_GOLDEN_*` override, so a corrupted copy goes in a scratch directory and
the real file's SHA-256 is recorded unchanged before and after.

```bat
python tools/drill_c5_gate1.py
python tools/drill_c6_gate1.py
python tools/drill_c8_reuse.py
python tools/drill_c10_gate2.py
```

The C5 drill corrupts the quiet Gate 1 data four ways; the C6 drill corrupts the
terminal data seven ways, including the three fields C5 had no equivalent of (the
terminal bit, the cached terminal value, and the per-run `max_tree_depth`) and the
recorded repetition history. Both require the suite to fail each time **with the
divergent node's path from the root** (a bare "trees differ" is not acceptable),
and print the before/after hashes as proof `golden/` was not written to.

**The C8 drill is a different shape, and it needs a working MSVC toolchain.** Its
subject is the C++ compaction rather than a comparison, so five of its eight
mutations are applied to a **copy of the source**: it duplicates `cpp/` and
`CMakeLists.txt` into a scratch directory, changes one line (a `children_offset`
one slot low, a dropped terminal bit, an un-advanced repetition history, an
`_expand_root` that accumulates instead of assigning), builds a separate module
into `<scratch>/module`, and runs the acceptance suite's own comparison helpers
against it. Configure re-uses the already-fetched dependency sources under
`build/msvc-release/_deps`, so it needs no network — but it does need
`build/msvc-release` to exist first. The remaining three mutations are the
classic golden-copy form and need no rebuild. Run it against a Release build;
each source mutation costs one ~40 s compile.

**The C10 drill is the classic golden-copy form and takes seconds** — it needs no
GPU, no torch and no rebuild, because Gate 2 compares a pure function of a FEN
and a recorded logit row. Its five mutations exist in two pairs plus a control:
`swap-closest` and `swap-adjacent` must be caught by the ordering check,
`nudge-over` by the magnitude check *and not* by the ordering one, and
`invert-inside-tolerance` by the ordering check *and not* by the magnitude one.
The last pair is what establishes that the two halves of Gate 2 are independent;
without it, "zero prior-ordering inversions" is a criterion the corpus cannot
distinguish from the bound beside it. See DECISIONS.md, C10, for why the corpus
forced that construction.

---

## Benchmarks

One script per chunk, all of them read-only with respect to `golden/`:

```bat
python tools/bench_c2.py --markdown     REM tokenization throughput
python tools/bench_c4.py --markdown     REM sibling scan, Q32 vs double
python tools/bench_c5.py --markdown     REM search throughput on the replay evaluator
python tools/bench_c6.py --markdown     REM what terminal handling costs, both corpora
python tools/bench_c8.py                REM arena high-water over whole games, compaction cost
```

Run them against a **Release** build. Each prints the compiler, sanitizer and
assert status in its header; numbers from an ASan build are roughly 7x slower and
do not belong in `BENCH.md`.
