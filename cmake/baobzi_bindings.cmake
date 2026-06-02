# baobzi_bindings.cmake — optional host-language wrappers (Python / Julia /
# MATLAB) over the C ABI (libbaobzi_c).
#
# Each language is guarded by its own option, all default OFF, so a plain
# `cmake ..` never pays for a toolchain the user doesn't have. When an option
# is ON we *discover* the toolchain and register CTest entries where possible;
# a missing toolchain degrades to a STATUS message and a skipped test rather
# than a hard configure error.
#
# The wrappers all reuse the already-instantiated C ABI:
#   * Python links the baobzi_c_static archive into a nanobind extension;
#   * Julia dlopen()s the baobzi_c shared library at runtime;
#   * MATLAB links baobzi_c_static into a MEX gateway.
# So nothing here re-instantiates the heavy template shapes.

include_guard(GLOBAL)

option(BAOBZI_BUILD_PYTHON "Build the Python (nanobind) bindings + register pytest" OFF)
option(BAOBZI_BUILD_JULIA  "Register the Julia binding test suite"                   OFF)
option(BAOBZI_BUILD_MATLAB "Build the MATLAB (MEX) bindings + register the test"     OFF)

if(NOT (BAOBZI_BUILD_PYTHON OR BAOBZI_BUILD_JULIA OR BAOBZI_BUILD_MATLAB))
  return()
endif()

if(NOT TARGET baobzi_c_static OR NOT TARGET baobzi_c)
  message(WARNING "baobzi: bindings requested but the C ABI targets are absent "
                  "(set BAOBZI_BUILD_C_API=ON). Skipping all bindings.")
  return()
endif()

enable_testing()
set(_baobzi_bind_dir "${PROJECT_SOURCE_DIR}/bindings")

# ---------------------------------------------------------------------------
# Python — nanobind extension linking baobzi_c_static + a pytest CTest.
# ---------------------------------------------------------------------------
if(BAOBZI_BUILD_PYTHON)
  find_package(Python 3.9 QUIET COMPONENTS Interpreter Development.Module)
  if(NOT Python_FOUND)
    message(STATUS "baobzi[python]: Python 3.9+ (Development.Module) not found — skipping.")
  else()
    # Shared module definition (also used by the pip wheel build) — locates
    # nanobind (interpreter or FetchContent) and defines the `_baobzi` target.
    include("${_baobzi_bind_dir}/python/baobzi_python.cmake")
    baobzi_add_python_module(baobzi::baobzi_c_static)

    # Assemble an importable package *in the build tree* — never write the .so
    # (or copy .py) into the source tree. Layout: <build>/python-pkg/baobzi/.
    set(_baobzi_pypkg "${CMAKE_BINARY_DIR}/python-pkg")
    set_target_properties(_baobzi PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${_baobzi_pypkg}/baobzi")
    # Stage the pure-Python sources next to the extension (re-copied on
    # reconfigure if they change, via the configure_file input dependency).
    file(GLOB _baobzi_py "${_baobzi_bind_dir}/python/baobzi/*.py")
    foreach(_f IN LISTS _baobzi_py)
      get_filename_component(_n "${_f}" NAME)
      configure_file("${_f}" "${_baobzi_pypkg}/baobzi/${_n}" COPYONLY)
    endforeach()

    add_test(NAME python_baobzi
      COMMAND "${Python_EXECUTABLE}" -m pytest -q
              "${_baobzi_bind_dir}/python/tests")
    set_tests_properties(python_baobzi PROPERTIES
      ENVIRONMENT "PYTHONPATH=${_baobzi_pypkg}")
    message(STATUS "baobzi[python]: extension + pytest 'python_baobzi' registered "
                   "(build-tree package at ${_baobzi_pypkg}).")
  endif()
endif()

# ---------------------------------------------------------------------------
# Julia — no build step; the package dlopen()s libbaobzi_c at test time. We
# just point it at the freshly-built shared library via LIBBAOBZI_C.
# ---------------------------------------------------------------------------
if(BAOBZI_BUILD_JULIA)
  find_program(BAOBZI_JULIA_EXECUTABLE julia)
  if(NOT BAOBZI_JULIA_EXECUTABLE)
    message(STATUS "baobzi[julia]: julia not on PATH — skipping.")
  else()
    add_test(NAME julia_baobzi
      COMMAND "${BAOBZI_JULIA_EXECUTABLE}"
              "--project=${_baobzi_bind_dir}/julia/Baobzi"
              "${_baobzi_bind_dir}/julia/Baobzi/test/runtests.jl")
    set_tests_properties(julia_baobzi PROPERTIES
      ENVIRONMENT "LIBBAOBZI_C=$<TARGET_FILE:baobzi_c>")
    message(STATUS "baobzi[julia]: test 'julia_baobzi' registered.")
  endif()
endif()

# ---------------------------------------------------------------------------
# MATLAB — mwrap is the real generator. baobzi.mw is the source of truth; the
# generated gateway (baobzi_mex_gen.cpp) + bz_*.m stubs are *committed*, so a
# plain build needs neither mwrap nor a network. We only (re)generate when
# baobzi.mw is newer than the committed gateway, discovering mwrap on PATH or
# cloning + building the official mwrap (DiamonDinoia/mwrap is the patchable
# fallback). The mex is compiled (-DR2008OO + static libstdc++/libgcc +
# version-script) and the suite run via a bash-free cmake -P driver.
# ---------------------------------------------------------------------------
if(BAOBZI_BUILD_MATLAB)
  find_program(BAOBZI_MATLAB_EXECUTABLE matlab)
  if(NOT BAOBZI_MATLAB_EXECUTABLE)
    message(STATUS "baobzi[matlab]: matlab not on PATH — skipping.")
  else()
    set(_baobzi_ml_dir "${_baobzi_bind_dir}/matlab")
    set(_baobzi_ml_gen "${_baobzi_ml_dir}/baobzi_mex_gen.cpp")
    set(_baobzi_ml_spec "${_baobzi_ml_dir}/baobzi.mw")

    # MATLAB_ROOT = two levels up from the real .../bin/matlab (the PATH entry
    # is usually a symlink), needed for mex.h + the version script.
    get_filename_component(_baobzi_mlreal "${BAOBZI_MATLAB_EXECUTABLE}" REALPATH)
    get_filename_component(_baobzi_mlbin "${_baobzi_mlreal}" DIRECTORY)
    get_filename_component(_baobzi_mlroot "${_baobzi_mlbin}" DIRECTORY)

    # Locate mwrap: PATH first, else clone+build the official one. Building
    # mwrap from a release needs only a C/C++ compiler + make — flex/bison are
    # required *only* to modify the grammar (the lexer/parser sources ship
    # generated), so we don't gate on them.
    find_program(BAOBZI_MWRAP_EXECUTABLE mwrap)
    function(_baobzi_ensure_mwrap)
      if(BAOBZI_MWRAP_EXECUTABLE)
        return()
      endif()
      set(_mw "${PROJECT_BINARY_DIR}/_deps_external/mwrap")
      if(NOT EXISTS "${_mw}/.git")
        message(STATUS "baobzi[matlab]: cloning zgimbutas/mwrap → ${_mw}")
        execute_process(
          COMMAND git clone --depth=1 https://github.com/zgimbutas/mwrap.git "${_mw}"
          RESULT_VARIABLE _rc)
        if(NOT _rc EQUAL 0)
          message(STATUS "baobzi[matlab]: mwrap clone failed (rc=${_rc}).")
          return()
        endif()
      endif()
      if(NOT EXISTS "${_mw}/mwrap")
        message(STATUS "baobzi[matlab]: building mwrap (make) in ${_mw}")
        execute_process(COMMAND make WORKING_DIRECTORY "${_mw}" RESULT_VARIABLE _rc)
        if(NOT _rc EQUAL 0 OR NOT EXISTS "${_mw}/mwrap")
          message(STATUS "baobzi[matlab]: mwrap build failed (rc=${_rc}).")
          return()
        endif()
      endif()
      set(BAOBZI_MWRAP_EXECUTABLE "${_mw}/mwrap" CACHE FILEPATH "mwrap binary" FORCE)
    endfunction()

    # Regenerate the committed gateway only when the spec is newer.
    if("${_baobzi_ml_spec}" IS_NEWER_THAN "${_baobzi_ml_gen}")
      _baobzi_ensure_mwrap()
      if(BAOBZI_MWRAP_EXECUTABLE)
        message(STATUS "baobzi[matlab]: baobzi.mw changed — regenerating gateway with "
                       "${BAOBZI_MWRAP_EXECUTABLE}.")
        execute_process(
          COMMAND "${BAOBZI_MWRAP_EXECUTABLE}" -mex baobzi_mex
                  -c baobzi_mex_gen.cpp -mb baobzi.mw
          WORKING_DIRECTORY "${_baobzi_ml_dir}" RESULT_VARIABLE _rc)
        if(NOT _rc EQUAL 0)
          message(WARNING "baobzi[matlab]: mwrap regeneration failed (rc=${_rc}); "
                          "using the committed gateway.")
        endif()
      else()
        message(STATUS "baobzi[matlab]: baobzi.mw is newer than the committed "
                       "gateway but mwrap is unavailable — using the committed gateway.")
      endif()
    endif()

    # Bash-free CTest driver: builds the mex (committed gateway) then runs the
    # MATLAB suite. $<TARGET_FILE_DIR:baobzi_c_static> is the dir with the .a.
    add_test(NAME matlab_baobzi
      COMMAND ${CMAKE_COMMAND}
              "-DMATLAB_ROOT=${_baobzi_mlroot}"
              "-DMATLAB_EXE=${BAOBZI_MATLAB_EXECUTABLE}"
              "-DBAOBZI_BUILD=$<TARGET_FILE_DIR:baobzi_c_static>"
              "-DSRC_DIR=${_baobzi_ml_dir}"
              -P "${_baobzi_ml_dir}/run_ctest.cmake")

    # Developer convenience: force a regeneration regardless of timestamps.
    if(BAOBZI_MWRAP_EXECUTABLE)
      add_custom_target(regenerate-matlab
        COMMAND "${BAOBZI_MWRAP_EXECUTABLE}" -mex baobzi_mex
                -c baobzi_mex_gen.cpp -mb baobzi.mw
        WORKING_DIRECTORY "${_baobzi_ml_dir}"
        COMMENT "Regenerating baobzi_mex_gen.cpp + bz_*.m from baobzi.mw")
    endif()

    message(STATUS "baobzi[matlab]: test 'matlab_baobzi' registered (MATLAB_ROOT=${_baobzi_mlroot}).")
  endif()
endif()
