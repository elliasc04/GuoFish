# Building `guofish_core`

The C++ core builds as a Python extension module named `guofish_core`. Two
toolchains are supported and both are exercised on every chunk: **Windows/MSVC**
is production, **Linux/Clang** exists because ThreadSanitizer does not exist on
MSVC and C9 will need it.

Dependencies (`pybind11`, `Disservin/chess-library`) are fetched by CMake at
configure time and pinned to immutable revisions. Nothing needs installing by
hand; the first configure needs network access.

## CMake options

| Option | Default | Meaning |
|---|---|---|
| `GUOFISH_ASAN` | `OFF` | AddressSanitizer; also UndefinedBehaviorSanitizer on Clang. Strips `/RTC1` and disables incremental linking on MSVC. Keeps `assert()` live even in release configs. |
| `GUOFISH_MODULE_OUTPUT_DIR` | repo root | Where the built `.pyd`/`.so` is written. |
| `GUOFISH_VALUE_SUM` | `q32` | Which accumulator `guofish::DefaultArena` and `guofish_core.NodeArena` name: `q32` (production) or `double` (Gate 1 equivalence). Both `NodeArena` types are compiled and bound in *every* build; this only selects the default. |

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
