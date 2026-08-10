cmake_minimum_required(VERSION 3.20)
# C is enabled for Fathom, which is one C translation unit (C7). Nothing of ours
# is C.
project(guofish_core LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# 0. Options
# ---------------------------------------------------------------------------
option(GUOFISH_ASAN "Build with AddressSanitizer (plus UBSan on Clang/GCC)" OFF)
# C9. Linux/Clang only; see section 2b.
option(GUOFISH_TSAN "Build with ThreadSanitizer (Linux/Clang only)" OFF)

# C8. Compile the full-tree virtual-loss AUDIT (`ReplaySearch::debug_total_vloss`).
#
# The reference walks the whole tree writing `vloss_count = 0` before every
# search, defensively — 3.4 ms at 2k sims, 937 ms over a game (scope §2.3). C8's
# brief forbids a production equivalent: repayment here is scope-guaranteed by
# RAII, so a reset would hide a bug rather than fix one. What is allowed, and
# only "strictly behind a debug flag", is a READ of the same quantity so a test
# can assert the invariant. This is that flag.
#
# Default ON for Debug, OFF otherwise, so the sanitized run — the one Global
# Rule 5 requires the suite to pass under — carries the audit while the Release
# module does not contain the symbol at all. tests/test_c8_reuse.py checks both
# halves of that: the invariant where the function exists, and its ABSENCE where
# it should not.
if (NOT DEFINED GUOFISH_DEBUG_VL)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(GUOFISH_DEBUG_VL ON)
    else ()
        set(GUOFISH_DEBUG_VL OFF)
    endif ()
endif ()
option(GUOFISH_DEBUG_VL "Compile the debug-only full-tree virtual-loss audit" ${GUOFISH_DEBUG_VL})

# Which accumulator `guofish::DefaultArena` names — the one the engine will use.
#
# Both NodeArena<Q32Accumulator> and NodeArena<DoubleAccumulator> are compiled
# and bound in EVERY build, so this switch cannot let one of them rot: it only
# selects the default alias and the `guofish_core.NodeArena` name. `q32` is
# production (associative, bit-reproducible under threads); `double` is the
# Gate 1 equivalence build, which has to reproduce Python's float arithmetic.
set(GUOFISH_VALUE_SUM "q32" CACHE STRING
        "Accumulator for value_sum: q32 (production) or double (Gate 1 equivalence)")
set_property(CACHE GUOFISH_VALUE_SUM PROPERTY STRINGS q32 double)
if (NOT GUOFISH_VALUE_SUM STREQUAL "q32" AND NOT GUOFISH_VALUE_SUM STREQUAL "double")
    message(FATAL_ERROR "GUOFISH_VALUE_SUM must be 'q32' or 'double', got '${GUOFISH_VALUE_SUM}'")
endif ()

# Where the importable extension lands. Defaults to the repo root so that
# `import guofish_core` works with `pythonpath = .` from pytest.ini. Override it
# to keep an out-of-tree build (see README_BUILD.md, WSL section).
set(GUOFISH_MODULE_OUTPUT_DIR "${CMAKE_SOURCE_DIR}" CACHE PATH
        "Directory the built Python extension is written to")

# ---------------------------------------------------------------------------
# 1. Warnings — Global Rule 4: clean at /W4 and -Wall -Wextra, no suppressions.
#
# These are applied PER TARGET (see section 4), not with add_compile_options,
# and that changed in C7 when Fathom arrived. Fathom is third-party C that we
# compile rather than merely include, so a directory-wide /W4 would put its
# warnings in our build with only two ways out: a -Wno-* or a pragma, both of
# which Global Rule 4 forbids. Scoping the strictness to our own target instead
# means our translation units stay exactly as strict as they were, and no
# suppression exists anywhere. It is the same reasoning that made chess-library
# a SYSTEM include in C0, applied to a target instead of an include path.
#
# GUOFISH_STRICT_WARNINGS is applied to guofish_core and to nothing else.
# ---------------------------------------------------------------------------
if (MSVC)
    set(GUOFISH_STRICT_WARNINGS /W4)
    # MSVC reports __cplusplus as 199711L regardless of /std: unless this is
    # passed. Anything that feature-tests the language version — ours or a
    # dependency's headers — otherwise concludes it is compiling C++98 while
    # actually being fed C++17. This is a conformance switch, not a warning
    # suppression (Global Rule 4), so it stays directory-wide.
    add_compile_options(/Zc:__cplusplus)
else ()
    set(GUOFISH_STRICT_WARNINGS -Wall -Wextra)
endif ()

# ---------------------------------------------------------------------------
# 2. Sanitizers — Global Rule 5.
#
# Enabled with -DGUOFISH_ASAN=ON. This is a separate configure, not the default,
# so the production build is not paying the ~2x ASan tax. The test suite is run
# against BOTH builds; see README_BUILD.md.
# ---------------------------------------------------------------------------
if (GUOFISH_ASAN)
    if (MSVC)
        add_compile_options(/fsanitize=address)
        # /RTC1 is in CMake's stock MSVC Debug flags and MSVC rejects it outright
        # alongside ASan ("/RTC1 and /fsanitize=address are incompatible").
        # Strip it here rather than making every caller edit the cache by hand.
        foreach (flag_var CMAKE_CXX_FLAGS_DEBUG CMAKE_C_FLAGS_DEBUG)
            string(REGEX REPLACE "/RTC[1csu]+" "" ${flag_var} "${${flag_var}}")
        endforeach ()
        # ASan cannot coexist with incremental linking.
        add_link_options(/INCREMENTAL:NO)
    else ()
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    endif ()

    # Global Rule 5 also requires debug asserts to be live. The Debug config has
    # them on already; this makes the sanitizer build correct even if someone
    # configures it as RelWithDebInfo.
    foreach (flag_var CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_MINSIZEREL)
        string(REGEX REPLACE "[-/]DNDEBUG" "" ${flag_var} "${${flag_var}}")
    endforeach ()
endif ()

# ---------------------------------------------------------------------------
# 2b. ThreadSanitizer — C9 acceptance, and Linux/Clang only.
#
# TSan DOES NOT EXIST ON MSVC. That is the whole reason Global Rule 8 requires
# this codebase to build on Linux at all: production is Windows, but the only
# tool that can prove the C9 descent is race-free runs somewhere else. A clean
# TSan run over acceptance layers 2 and 3 is part of C9's acceptance, not an
# optional extra.
#
# Mutually exclusive with GUOFISH_ASAN — the two instrument the same shadow
# memory and cannot be combined — and the error is raised here rather than left
# to a linker message nobody reads.
if (GUOFISH_TSAN)
    if (GUOFISH_ASAN)
        message(FATAL_ERROR
                "GUOFISH_TSAN and GUOFISH_ASAN cannot both be ON: ThreadSanitizer and "
                "AddressSanitizer use incompatible shadow-memory layouts. Configure two "
                "build directories instead.")
    endif ()
    if (MSVC)
        message(FATAL_ERROR
                "GUOFISH_TSAN is not available under MSVC — ThreadSanitizer has no MSVC "
                "implementation. Build it on Linux/Clang; see README_BUILD.md.")
    endif ()
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=thread)
    foreach (flag_var CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_MINSIZEREL)
        string(REGEX REPLACE "[-/]DNDEBUG" "" ${flag_var} "${${flag_var}}")
    endforeach ()
endif ()

# ---------------------------------------------------------------------------
# 3. Third-party dependencies — Global Rule 7 allows pybind11,
#    Disservin/chess-library and (from C7, on explicit authorisation)
#    jdart1/Fathom. All three are pinned to an immutable revision: C1 is a
#    movegen *parity* chunk, and a floating `master` would let the reference
#    behaviour change under a passing test suite without anything in the repo
#    changing. See DECISIONS.md.
# ---------------------------------------------------------------------------
include(FetchContent)

set(PYBIND11_FINDPYTHON ON)

FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# The pinned revision, named once. C5 asserts en-passant behaviour that lives in
# chess-library's makeMove() — which is a *property of this revision*, not of the
# library in general — so the pin is passed into the build as a string and
# tests/test_c5_ep_pin.py checks it. A re-pin then fails a named test instead of
# silently moving token 66 on ~3% of positions.
set(GUOFISH_CHESS_LIBRARY_PIN "53e6a841dcda7059a2af363d85f785ef1817304a")

FetchContent_Declare(
        chess_library
        GIT_REPOSITORY https://github.com/Disservin/chess-library.git
        GIT_TAG        ${GUOFISH_CHESS_LIBRARY_PIN}  # v0.9.4
        # chess-library is header-only. Pointing SOURCE_SUBDIR at a directory
        # with no CMakeLists.txt makes FetchContent fetch the sources without
        # add_subdirectory()-ing its test/benchmark targets into our build.
        SOURCE_SUBDIR  do-not-configure
)
FetchContent_MakeAvailable(chess_library)

# --- Fathom (Syzygy tablebase probing) -------------------------------------
#
# Authorised explicitly for C7; before that the allowed set was pybind11 +
# chess-library + the standard library, and Global Rule 7 required asking.
#
# Pinned for the same reason the other two are, and with one extra: a tablebase
# probe is a lookup whose ANSWER is the thing under test. A floating revision
# could change the WDL a position reports — through an ep convention, a
# 50-move-rule adjustment, a DTZ sign — with nothing in this repository
# changing, and the failure would surface as an engine playing a different
# endgame move, not as a build error.
set(GUOFISH_FATHOM_PIN "c9c6fef0dddc05d2e242c183acf5833149ab676d")

FetchContent_Declare(
        fathom
        GIT_REPOSITORY https://github.com/jdart1/Fathom.git
        GIT_TAG        ${GUOFISH_FATHOM_PIN}
        # Fathom ships a plain Makefile, not a CMakeLists.txt, so there is
        # nothing to add_subdirectory() — but naming a non-existent SOURCE_SUBDIR
        # keeps the behaviour explicit and identical to the chess-library entry
        # above rather than relying on the absence of a file.
        SOURCE_SUBDIR  do-not-configure
)
FetchContent_MakeAvailable(fathom)

# ONE translation unit: tbprobe.c `#include`s tbchess.c (Fathom's own Makefile
# does the same). Built as a static library of its own so that
# GUOFISH_STRICT_WARNINGS never reaches it — see section 1.
add_library(fathom STATIC "${fathom_SOURCE_DIR}/src/tbprobe.c")
target_include_directories(fathom PUBLIC "${fathom_SOURCE_DIR}/src")
set_target_properties(fathom PROPERTIES
        # C11 for <stdatomic.h>, which Fathom uses for its lock-free table
        # initialisation unless TB_NO_THREADS is defined. We do NOT define it:
        # C9 will probe from several search threads, and the atomics are what
        # makes tb_probe_wdl thread-safe.
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        # The static library is linked into a shared module (the .pyd / .so), so
        # its objects must be position-independent on Linux.
        POSITION_INDEPENDENT_CODE ON)

if (MSVC)
    # MSVC ships <stdatomic.h> but refuses to compile it without this switch
    # ("C atomic support is not enabled"). It is a language-conformance switch,
    # not a warning suppression (Global Rule 4) and not an opt-in to unfinished
    # behaviour — the name is historical; C11 atomics have been supported since
    # VS 2022 17.5.
    #
    # The alternative is Fathom's TB_NO_THREADS, which compiles everywhere with
    # no flag and is the wrong trade: it removes the synchronisation that makes
    # tb_probe_wdl safe to call from several search threads, which is precisely
    # what C9 will do. Clang and GCC need nothing here.
    target_compile_options(fathom PRIVATE /experimental:c11atomics)

    # Fathom defines its own `min` and `max` macros (tbprobe.c:196) after having
    # included <stdlib.h> and <windows.h>, both of which define them on MSVC.
    # That is two C4005 macro-redefinition warnings at MSVC's DEFAULT level, so
    # they fire at any warning level — and they are not spurious: Fathom's
    # `#define max(a,b) a > b ? a : b` has no parentheses, so which definition
    # wins genuinely matters.
    #
    # Both are turned off AT THE POINT OF CONFLICT rather than by silencing the
    # report, which is the difference between a fix and a suppression (Global
    # Rule 4). Fathom's own definitions then stand unopposed and are the ones
    # its code was written against:
    #
    #   NOMINMAX                    stops <windows.h>/minwindef.h defining them.
    #   _CRT_DECLARE_NONSTDC_NAMES  stops <stdlib.h> defining them; they live in
    #                               its "Non-ANSI Names for Compatibility"
    #                               section, behind exactly this macro. It also
    #                               withdraws the other non-standard names —
    #                               `open`, `close`, `strdup` — and Fathom uses
    #                               `open`/`close` only inside its `#ifndef
    #                               _WIN32` branch, which this build does not
    #                               compile. Checked, not assumed.
    target_compile_definitions(fathom PRIVATE NOMINMAX _CRT_DECLARE_NONSTDC_NAMES=0)
endif ()

# ---------------------------------------------------------------------------
# 4. The extension module
# ---------------------------------------------------------------------------
pybind11_add_module(guofish_core
        cpp/bindings.cpp
        cpp/arena.hpp
        cpp/cache.hpp
        cpp/evaluator.hpp
        cpp/fathom.hpp
        cpp/keys.hpp
        cpp/movegen.hpp
        cpp/search.hpp
        cpp/tablebase.hpp
        cpp/terminal.hpp
        cpp/tokens.hpp
        cpp/values.hpp
)

target_include_directories(guofish_core PRIVATE cpp/)

# Global Rule 4, scoped to our own code — see section 1.
target_compile_options(guofish_core PRIVATE ${GUOFISH_STRICT_WARNINGS})

target_link_libraries(guofish_core PRIVATE fathom)

target_compile_definitions(guofish_core PRIVATE
        GUOFISH_CHESS_LIBRARY_PIN="${GUOFISH_CHESS_LIBRARY_PIN}"
        GUOFISH_FATHOM_PIN="${GUOFISH_FATHOM_PIN}")

if (GUOFISH_VALUE_SUM STREQUAL "double")
    target_compile_definitions(guofish_core PRIVATE GUOFISH_VALUE_SUM_DOUBLE)
endif ()

if (GUOFISH_DEBUG_VL)
    target_compile_definitions(guofish_core PRIVATE GUOFISH_DEBUG_VL)
endif ()

# chess-library is third-party: SYSTEM keeps its headers from tripping /W4 and
# -Wall -Wextra. This is not a warning suppression on our own code (Rule 4) —
# no -Wno-* and no pragmas are involved, and our translation units stay strict.
target_include_directories(guofish_core SYSTEM PRIVATE
        "${chess_library_SOURCE_DIR}/include")

# Fathom's headers likewise. `target_link_libraries` already propagates its
# include directory, but not as SYSTEM, and tbprobe.h is included from our own
# strictly-compiled translation unit.
target_include_directories(guofish_core SYSTEM PRIVATE "${fathom_SOURCE_DIR}/src")

# ---------------------------------------------------------------------------
# 5. Output location
# ---------------------------------------------------------------------------
set_target_properties(guofish_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${GUOFISH_MODULE_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${GUOFISH_MODULE_OUTPUT_DIR}"
        # Keep debug databases in the build tree instead of the repo root.
        PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Multi-config generators (Visual Studio) append the config name unless the
# per-config properties are set explicitly.
foreach (cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(guofish_core PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY_${cfg} "${GUOFISH_MODULE_OUTPUT_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_${cfg} "${GUOFISH_MODULE_OUTPUT_DIR}"
            PDB_OUTPUT_DIRECTORY_${cfg} "${CMAKE_BINARY_DIR}"
            COMPILE_PDB_OUTPUT_DIRECTORY_${cfg} "${CMAKE_BINARY_DIR}")
endforeach ()

# MSVC's ASan runtime is a DLL that must sit next to the module: since Python 3.8
# the interpreter no longer searches %PATH% when resolving an extension's
# dependencies, so a plain "add it to PATH" does not work here.
if (GUOFISH_ASAN AND MSVC)
    get_filename_component(_msvc_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(_asan_dll "${_msvc_bin}/clang_rt.asan_dynamic-x86_64.dll")
    if (EXISTS "${_asan_dll}")
        add_custom_command(TARGET guofish_core POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_asan_dll}" "${GUOFISH_MODULE_OUTPUT_DIR}"
                COMMENT "Staging ASan runtime next to guofish_core")
    else ()
        message(WARNING "ASan runtime not found at ${_asan_dll}; "
                "importing guofish_core will fail with a missing-DLL error.")
    endif ()
endif ()
