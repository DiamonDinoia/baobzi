cmake_minimum_required(VERSION 3.25)
project(baobzi VERSION 0.0.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# User-facing options
# ---------------------------------------------------------------------------
option(BAOBZI_BUILD_EXAMPLES         "Build C++ examples"                       ON)
option(BAOBZI_BUILD_TESTS            "Build tests"                              ON)

# Quality / tooling
option(BAOBZI_STRICT_WARNINGS        "Enable baobzi's strict warning profile"   ON)
option(BAOBZI_WARNINGS_AS_ERRORS     "Treat warnings as errors"                 ON)
option(BAOBZI_ENABLE_SANITIZERS      "Enable ASan/UBSan"                        OFF)
option(BAOBZI_ENABLE_STATIC_ANALYSIS "Run clang-tidy/cppcheck"                  OFF)
option(BAOBZI_ENABLE_COVERAGE        "Build with coverage instrumentation"      OFF)

# ---------------------------------------------------------------------------
# Global project settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type specified. Defaulting to Release.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Valid options: Debug, RelWithDebInfo, Release" FORCE)
endif()

include(GNUInstallDirs)
include(CheckIPOSupported)
check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_error)
# gcc 14's LTO chokes (ICE / "invalid tree code") on the heavy template
# instantiations in the timing examples. Skip IPO on gcc until that's
# fixed upstream — clang's LTO handles the same code without trouble.
if(_ipo_supported AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif(NOT _ipo_supported)
  message(WARNING "IPO is not supported: ${_ipo_error}")
endif()

# Performance-critical flags for ND polyfit codegen.
#
# `-march=native` is critical: without it the compiler targets the x86-64
# baseline ISA, which has no FMA — every Horner step then becomes a libm
# fma() call instead of a single `vfmadd231sd` / `vfmadd231pd`. Profiling
# the multi-leaf hot path showed ~50% of runtime in libm's __fma_fma3
# before this was enabled.
#
# Override at configure time with `-DBAOBZI_ARCH=...` to target a portable
# baseline (e.g. `x86-64-v3` for AVX2+FMA without per-machine tuning).
set(BAOBZI_ARCH "native" CACHE STRING
    "CPU target for -march (e.g. native, x86-64-v3)")
# gcc rejects `-mtune=x86-64-v{2,3,4}` (only `-march=` accepts those microarch
# levels), so we derive a tune value: pass through anything else, but fall
# back to `generic` for the x86-64-v* baselines.
if(BAOBZI_ARCH MATCHES "^x86-64-v[0-9]+$")
    set(_baobzi_tune "generic")
else()
    set(_baobzi_tune "${BAOBZI_ARCH}")
endif()
set(BAOBZI_TUNE "${_baobzi_tune}" CACHE STRING
    "CPU target for -mtune (defaults to BAOBZI_ARCH, or generic for x86-64-v*)")
# Apple clang on AArch64 rejects `-march=apple-*` / `-mtune=` — it expects
# `-mcpu=` for that microarch family. Use the right spelling per platform.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(_baobzi_arch_flags -mcpu=${BAOBZI_ARCH})
else()
    set(_baobzi_arch_flags -march=${BAOBZI_ARCH} -mtune=${BAOBZI_TUNE})
endif()
# Global compile flags. We deliberately stay conservative here:
#   - Arch flags are needed for FMA codegen at all.
#   - `-ffp-contract=fast` permits FMA contraction (a single `fma` is
#     *more* accurate than a separate mul+add, so it is monotonic for
#     fit accuracy).
# We do *not* enable `-funroll-loops` or `-funsafe-math-optimizations`
# globally: the eval hot path already unrolls the per-axis loops via
# `poet::static_for` (compile-time unrolling, no need for the optimiser
# to guess), and the Horner kernel is written so reassociation would
# only hurt accuracy. Enabling them would also reorder the FP arithmetic
# in the fit's convergence test and flip outcomes on borderline panels.
add_compile_options(${_baobzi_arch_flags} -ffp-contract=fast)
add_link_options(${_baobzi_arch_flags} -ffp-contract=fast)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Wiring — each module's internals live in its own cmake/*.cmake file.
# ---------------------------------------------------------------------------
include(dev_helpers)       # warnings / sanitizers / static analysis helpers
include(baobzi_deps)       # fetch polyfit, POET, Catch2
include(baobzi_c_api)      # C ABI (libbaobzi_c) — built before examples/tests
include(baobzi_examples)   # C++ examples
include(baobzi_tests)      # Catch2 suites
include(baobzi_bindings)   # optional Python / Julia / MATLAB wrappers (default OFF)
include(baobzi_install)    # install rules
