# baobzi_deps.cmake — fetch header-only dependencies and mark their include
# trees as system headers so consumer warnings (-Werror) don't fire on them.

include_guard(GLOBAL)
include(FetchContent)

# Polyfit fetches xsimd internally via CPM. To redirect that fetch onto
# our fork (DiamonDinoia/xsimd:feat/dynamic-masks, which carries the
# dynamic-mask / masked-load primitives the eval pipeline needs for
# clean partial loads), we set CPM's per-package source override before
# polyfit's CPMAddPackage(NAME xsimd ...) runs. The CPM_xsimd_SOURCE
# variable points CPM at a local checkout; we maintain that checkout
# under build-new/_deps_external/xsimd-fork so it persists across
# CMake re-configures.
set(_baobzi_xsimd_fork "${PROJECT_BINARY_DIR}/_deps_external/xsimd-fork")
if(NOT EXISTS "${_baobzi_xsimd_fork}/.git")
  message(STATUS "baobzi: cloning DiamonDinoia/xsimd:feat/dynamic-masks "
                 "→ ${_baobzi_xsimd_fork}")
  execute_process(
    COMMAND git clone --depth=1 --branch feat/dynamic-masks
            https://github.com/DiamonDinoia/xsimd.git
            "${_baobzi_xsimd_fork}"
    RESULT_VARIABLE _baobzi_xsimd_clone_rc)
  if(NOT _baobzi_xsimd_clone_rc EQUAL 0)
    message(FATAL_ERROR "baobzi: failed to clone xsimd fork (rc=${_baobzi_xsimd_clone_rc})")
  endif()
endif()
# Tell CPM (used inside polyfit) to source xsimd from our local fork.
set(CPM_xsimd_SOURCE "${_baobzi_xsimd_fork}" CACHE PATH "" FORCE)

# Polyfit's `feat/parametric-block-size` branch extends the `estrin`-branch
# ScalarKernel selector with an opt-in `HybridK<K>` block-size override plus
# a consteval `optimal_block_size<NCOEFFS,SIMD_W,NREG,Policy>` picker and an
# `EvalPolicy { Latency, Throughput, Balanced }` enum. Baobzi threads
# `EvalPolicy` through `Function` and routes the scalar kernel and K choice
# per policy (see include/baobzi/detail/eval_policy.hpp); today's default
# `Balanced` keeps the scalar `Horner` mapping to avoid regressing the
# 1.3-1.6x scalar slowdown reported on Core Ultra 7 at xsimd lane_w=4 until
# the K-sweep measurement campaign retunes the formula.
FetchContent_Declare(
    polyfit
    GIT_REPOSITORY https://github.com/DiamonDinoia/polyfit.git
    GIT_TAG        828582f2523678d206b2f76281088237427ff5b3
    SYSTEM
)
set(POLYFIT_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(POLYFIT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(polyfit)

FetchContent_Declare(
    poet
    GIT_REPOSITORY https://github.com/DiamonDinoia/POET.git
    GIT_TAG        b55580fd1f17df500abb8afb16cd4983e66a27ac
    SYSTEM
)
set(POET_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(POET_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(POET_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(POET_BUILD_DOCS       OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(poet)

# Treat transitive CPM-fetched dependencies (xsimd, mdspan, …) as system
# headers too so their warnings don't gate our -Werror build.
function(_baobzi_make_system target)
  if(NOT TARGET "${target}")
    return()
  endif()
  get_target_property(_aliased "${target}" ALIASED_TARGET)
  if(_aliased)
    set(target "${_aliased}")
  endif()
  get_target_property(_incl "${target}" INTERFACE_INCLUDE_DIRECTORIES)
  if(_incl)
    set_target_properties("${target}" PROPERTIES
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_incl}")
  endif()
endfunction()
foreach(_t IN ITEMS polyfit xsimd mdspan std::mdspan poet)
  _baobzi_make_system("${_t}")
endforeach()

if(BAOBZI_BUILD_EXAMPLES)
  # nanobench: header-only microbench harness with proper warmup,
  # MdAPE-based stability checks, and TSC-frequency calibration.
  # We fetch the source archive and expose the include directory only —
  # consumers get the impl by defining ANKERL_NANOBENCH_IMPLEMENT in
  # exactly one TU (done in baobzi_microbench.cpp).
  FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG        v4.3.11
    SYSTEM
  )
  FetchContent_MakeAvailable(nanobench)
  _baobzi_make_system(nanobench)
endif()

if(BAOBZI_BUILD_TESTS)
  FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.9.0
    SYSTEM
  )
  FetchContent_MakeAvailable(catch2)
endif()
