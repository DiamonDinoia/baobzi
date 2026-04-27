cmake_minimum_required(VERSION 3.25)
project(baobzi LANGUAGES CXX)

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
if(_ipo_supported)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
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
    "CPU target for -march/-mtune (e.g. native, x86-64-v3)")
add_compile_options(-march=${BAOBZI_ARCH} -mtune=${BAOBZI_ARCH}
                    -funroll-loops -funsafe-math-optimizations
                    -ffp-contract=fast)
add_link_options(-march=${BAOBZI_ARCH} -mtune=${BAOBZI_ARCH}
                 -funroll-loops -funsafe-math-optimizations
                 -ffp-contract=fast)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Wiring — each module's internals live in its own cmake/*.cmake file.
# ---------------------------------------------------------------------------
include(dev_helpers)       # warnings / sanitizers / static analysis helpers
include(baobzi_deps)       # fetch polyfit, POET, Catch2
include(baobzi_examples)   # C++ examples
include(baobzi_tests)      # Catch2 suites
include(baobzi_install)    # install rules
