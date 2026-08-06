cmake_minimum_required(VERSION 3.20)
project(guofish_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# 0. Options
# ---------------------------------------------------------------------------
option(GUOFISH_ASAN "Build with AddressSanitizer (plus UBSan on Clang/GCC)" OFF)

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
# ---------------------------------------------------------------------------
if (MSVC)
    add_compile_options(/W4)
    # MSVC reports __cplusplus as 199711L regardless of /std: unless this is
    # passed. Anything that feature-tests the language version — ours or a
    # dependency's headers — otherwise concludes it is compiling C++98 while
    # actually being fed C++17. This is a conformance switch, not a warning
    # suppression (Global Rule 4).
    add_compile_options(/Zc:__cplusplus)
else ()
    add_compile_options(-Wall -Wextra)
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
# 3. Third-party dependencies — Global Rule 7 allows exactly pybind11 and
#    Disservin/chess-library. Both are pinned to an immutable revision: C1 is a
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

FetchContent_Declare(
        chess_library
        GIT_REPOSITORY https://github.com/Disservin/chess-library.git
        GIT_TAG        53e6a841dcda7059a2af363d85f785ef1817304a  # v0.9.4
        # chess-library is header-only. Pointing SOURCE_SUBDIR at a directory
        # with no CMakeLists.txt makes FetchContent fetch the sources without
        # add_subdirectory()-ing its test/benchmark targets into our build.
        SOURCE_SUBDIR  do-not-configure
)
FetchContent_MakeAvailable(chess_library)

# ---------------------------------------------------------------------------
# 4. The extension module
# ---------------------------------------------------------------------------
pybind11_add_module(guofish_core
        cpp/bindings.cpp
        cpp/arena.hpp
        cpp/keys.hpp
        cpp/movegen.hpp
        cpp/tokens.hpp
)

target_include_directories(guofish_core PRIVATE cpp/)

if (GUOFISH_VALUE_SUM STREQUAL "double")
    target_compile_definitions(guofish_core PRIVATE GUOFISH_VALUE_SUM_DOUBLE)
endif ()

# chess-library is third-party: SYSTEM keeps its headers from tripping /W4 and
# -Wall -Wextra. This is not a warning suppression on our own code (Rule 4) —
# no -Wno-* and no pragmas are involved, and our translation units stay strict.
target_include_directories(guofish_core SYSTEM PRIVATE
        "${chess_library_SOURCE_DIR}/include")

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
