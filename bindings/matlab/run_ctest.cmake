# run_ctest.cmake — bash-free driver for the `matlab_baobzi` CTest.
#
# Builds the MEX gateway from the committed sources via the Makefile, then runs
# the MATLAB smoke/parity suite headless. Invoked as:
#   cmake -DMATLAB_ROOT=.. -DMATLAB_EXE=.. -DBAOBZI_BUILD=.. -DSRC_DIR=.. -P run_ctest.cmake
#
# Required -D variables:
#   MATLAB_ROOT  — MATLAB install root (for mex.h + the version script)
#   MATLAB_EXE   — the matlab launcher
#   BAOBZI_BUILD — directory containing libbaobzi_c.a
#   SRC_DIR      — bindings/matlab (holds the Makefile, gateway, *.m, test)

foreach(_v MATLAB_ROOT MATLAB_EXE BAOBZI_BUILD SRC_DIR)
  if(NOT DEFINED ${_v})
    message(FATAL_ERROR "run_ctest.cmake: -D${_v} is required")
  endif()
endforeach()

# The committed (or configure-time regenerated) gateway is the build input;
# bump its mtime so `make` never tries to invoke mwrap here at test time.
if(EXISTS "${SRC_DIR}/baobzi_mex_gen.cpp")
  file(TOUCH_NOCREATE "${SRC_DIR}/baobzi_mex_gen.cpp")
endif()

execute_process(
  COMMAND make -C "${SRC_DIR}"
          "MATLAB_ROOT=${MATLAB_ROOT}" "BAOBZI_BUILD=${BAOBZI_BUILD}"
  RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "matlab_baobzi: MEX build failed (rc=${_rc})")
endif()

execute_process(
  COMMAND "${MATLAB_EXE}" -batch "run('test_baobzi.m')"
  WORKING_DIRECTORY "${SRC_DIR}"
  RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "matlab_baobzi: test_baobzi.m failed (rc=${_rc})")
endif()

message(STATUS "matlab_baobzi: passed")
