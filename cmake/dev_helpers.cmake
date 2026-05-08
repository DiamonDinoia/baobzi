include_guard(GLOBAL)

# ==============================================================================
# BAOBZI Development Helpers
# ==============================================================================
# This file provides CMake helper functions and targets for BAOBZI development:
# - Compiler warnings configuration (baobzi_enable_warnings)
# - Sanitizers setup (baobzi_enable_sanitizers)
# - Static analysis tools (baobzi_configure_static_analysis)
# - Documentation generation (doxygen, sphinx, docs targets)
# - Code coverage reporting (coverage target)
#
# These helpers are only used for development/testing builds and are not
# required when using BAOBZI as a header-only library.
# ==============================================================================

# Prepare CPM (CMake Package Manager) for fetching dependencies. Only fetched
# when a feature that needs it (sanitizers) is actually enabled, so the default
# baobzi build does not touch the network.
include(FetchContent)

function(_baobzi_ensure_cpm)
  get_property(_done GLOBAL PROPERTY _baobzi_cpm_ready SET)
  if(_done)
    return()
  endif()

  FetchContent_Declare(
    CPM
    URL https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.42.0/CPM.cmake
    URL_HASH SHA256=2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a
    DOWNLOAD_NO_EXTRACT TRUE
  )
  FetchContent_GetProperties(CPM)
  if(NOT CPM_POPULATED)
    FetchContent_MakeAvailable(CPM)
  endif()
  include(${cpm_SOURCE_DIR}/CPM.cmake)

  set_property(GLOBAL PROPERTY _baobzi_cpm_ready TRUE)
endfunction()

# -------------------------
# Warnings helper (from PoetWarnings.cmake)
# -------------------------
if(NOT DEFINED BAOBZI_WARNINGS_AS_ERRORS)
  option(BAOBZI_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
endif()

# Enable comprehensive compiler warnings for a target
# Supports GCC, Clang, AppleClang, and MSVC compilers
# Applies warnings with PRIVATE scope for regular targets, INTERFACE scope for interface libraries
function(baobzi_enable_warnings target)
  if (NOT TARGET "${target}")
    message(FATAL_ERROR "baobzi_enable_warnings called with non-existent target '${target}'")
  endif()

  # Determine the appropriate scope for applying warnings
  # INTERFACE scope for interface libraries (warnings propagate to consumers)
  # PRIVATE scope for other targets (warnings only apply to this target's sources)
  get_target_property(_target_type "${target}" TYPE)
  if(_target_type STREQUAL "INTERFACE_LIBRARY")
    set(_scope INTERFACE)
  else()
    set(_scope PRIVATE)
  endif()

  # Generator expressions for compiler detection (evaluated at build time)
  set(_clang_like $<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>)
  set(_gnu $<CXX_COMPILER_ID:GNU>)
  set(_gnu_or_clang $<OR:${_gnu},${_clang_like}>)
  set(_msvc $<CXX_COMPILER_ID:MSVC>)
  set(_lang_is_cxx $<COMPILE_LANGUAGE:CXX>)

  set(_warnings_clang_like
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wdouble-promotion
    -Wold-style-cast
    -Wnon-virtual-dtor
    -Wnull-dereference
    -Woverloaded-virtual
    -Wcast-align
    -Wunused
    -Wimplicit-fallthrough
    -Wformat=2
  )

  # Additional curated warnings that are checked for compiler support before enabling
  # These are only added if the compiler supports them (GCC-specific flags)
  set(_additional_warnings
    -Wduplicated-cond
    -Wlogical-op
    -Wuseless-cast
    -Winit-self
    -Wmissing-include-dirs
    -Wredundant-decls
  )

  # Check which additional warnings are supported by the current compiler and add them
  include(CheckCXXCompilerFlag)
  foreach(_f IN LISTS _additional_warnings)
    check_cxx_compiler_flag("${_f}" _flag_supported)
    if(_flag_supported)
      list(APPEND _warnings_clang_like ${_f})
    endif()
  endforeach()

  set(_warnings_gnu_only
    -Wmisleading-indentation
    -Wsuggest-override
  )

  set(_warnings_msvc
    /W4
    /permissive-
    /bigobj
    /w14242
    /w14254
    /w14263
    /w14265
    /w14287
    /we4289
    /w14296
    /w14311
    /w14545
    /w14546
    /w14547
    /w14549
    /w14555
    /w14619
    /w14640
    /w14826
    /w14905
    /w14906
    /w14928
    # MSVC's C4702 ("unreachable code") fires inside heavily-templated
    # `if constexpr` ladders in polyfit / poet / our numerics — it's a
    # known false-positive class with no clean source-level fix. Disable
    # it explicitly so /WX doesn't reject those harmless paths.
    /wd4702
  )

  # Build compiler-specific warning flags using generator expressions
  # Flags are only applied when compiling C++ code with the matching compiler
  set(_compile_options)
  foreach(flag IN LISTS _warnings_clang_like)
    list(APPEND _compile_options $<$<AND:${_lang_is_cxx},${_gnu_or_clang}>:${flag}>)
  endforeach()
  foreach(flag IN LISTS _warnings_gnu_only)
    list(APPEND _compile_options $<$<AND:${_lang_is_cxx},${_gnu}>:${flag}>)
  endforeach()
  foreach(flag IN LISTS _warnings_msvc)
    list(APPEND _compile_options $<$<AND:${_lang_is_cxx},${_msvc}>:${flag}>)
  endforeach()

  # Add -Werror / /WX if treating warnings as errors
  if(BAOBZI_WARNINGS_AS_ERRORS)
    list(APPEND _compile_options $<$<AND:${_lang_is_cxx},${_gnu_or_clang}>:-Werror>)
    list(APPEND _compile_options $<$<AND:${_lang_is_cxx},${_msvc}>:/WX>)
  endif()

  target_compile_options(${target} ${_scope} ${_compile_options})
endfunction()

# -------------------------
# Sanitizers helper (from PoetSanitizers.cmake)
# -------------------------
option(BAOBZI_ENABLE_SANITIZERS "Master switch to enable all sanitizers at once" OFF)

option(BAOBZI_ENABLE_ASAN "Enable AddressSanitizer (memory error detection)" ${BAOBZI_ENABLE_SANITIZERS})
option(BAOBZI_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer (undefined behavior detection)" ${BAOBZI_ENABLE_SANITIZERS})

# Internal: Check if any sanitizer is enabled
function(_baobzi_any_sanitizer_enabled out_var)
  if(BAOBZI_ENABLE_ASAN OR BAOBZI_ENABLE_UBSAN)
    set(${out_var} TRUE PARENT_SCOPE)
  else()
    set(${out_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

# Internal: Download and configure sanitizers-cmake module (cached to avoid re-downloading)
# Sets out_var to TRUE if successful, FALSE otherwise
function(_baobzi_prepare_sanitizers out_var)
  # Check if we've already prepared sanitizers (cached in global property)
  get_property(_prepared GLOBAL PROPERTY _baobzi_sanitizers_prepared SET)
  if(_prepared)
    get_property(_prepared_value GLOBAL PROPERTY _baobzi_sanitizers_prepared)
    set(${out_var} ${_prepared_value} PARENT_SCOPE)
    return()
  endif()

  set(_status TRUE)

  _baobzi_ensure_cpm()

  # Download sanitizers-cmake from GitHub using CPM
  CPMAddPackage(
    NAME SanitizersCMake
    GITHUB_REPOSITORY arsenm/sanitizers-cmake
    GIT_TAG 0573e2ea8651b9bb3083f193c41eb086497cc80a
    DOWNLOAD_ONLY YES
    OPTIONS "CMAKE_POLICY_VERSION_MINIMUM 3.10"
  )

  if(NOT SanitizersCMake_SOURCE_DIR)
    set(_status FALSE)
  endif()

  if(_status)
    # Add sanitizers-cmake to module path
    list(APPEND CMAKE_MODULE_PATH "${SanitizersCMake_SOURCE_DIR}/cmake")

    # Configure sanitizers-cmake options based on BAOBZI settings
    set(SANITIZE_ADDRESS ${BAOBZI_ENABLE_ASAN} CACHE BOOL
      "Enable AddressSanitizer for sanitized targets." FORCE)
    mark_as_advanced(SANITIZE_ADDRESS)

    set(SANITIZE_UNDEFINED ${BAOBZI_ENABLE_UBSAN} CACHE BOOL
      "Enable UndefinedBehaviorSanitizer for sanitized targets." FORCE)
    mark_as_advanced(SANITIZE_UNDEFINED)

    # Load the sanitizers module
    find_package(Sanitizers REQUIRED QUIET)
  endif()

  # Cache the preparation result
  set_property(GLOBAL PROPERTY _baobzi_sanitizers_prepared ${_status})
  set(${out_var} ${_status} PARENT_SCOPE)
endfunction()

# Enable sanitizers for a target based on BAOBZI_ENABLE_ASAN and BAOBZI_ENABLE_UBSAN options
# For interface libraries: manually applies sanitizer flags to compile/link options
# For regular targets: uses the add_sanitizers() function from sanitizers-cmake
function(baobzi_enable_sanitizers target)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR "baobzi_enable_sanitizers called with non-existent target '${target}'")
  endif()

  _baobzi_any_sanitizer_enabled(_baobzi_any_enabled)
  if(NOT _baobzi_any_enabled)
    return()
  endif()

  _baobzi_prepare_sanitizers(_baobzi_sanitizers_ready)
  if(NOT _baobzi_sanitizers_ready)
    return()
  endif()

  get_target_property(_target_type "${target}" TYPE)

  if(_target_type STREQUAL "INTERFACE_LIBRARY")
    # Special handling for interface libraries (like header-only libraries)
    # add_sanitizers() doesn't work with interface libraries, so we manually add flags
    if(NOT CMAKE_CXX_COMPILER_ID)
      message(FATAL_ERROR "baobzi_enable_sanitizers requires a C++ compiler when sanitizers are enabled")
    endif()

    # Build list of requested sanitizers
    set(_requested_sanitizers)
    if(BAOBZI_ENABLE_ASAN)
      list(APPEND _requested_sanitizers ASan)
    endif()
    if(BAOBZI_ENABLE_UBSAN)
      list(APPEND _requested_sanitizers UBSan)
    endif()

    # For each sanitizer, retrieve compiler-specific flags and apply them
    foreach(_sanitizer IN LISTS _requested_sanitizers)
      # Flag variable names come from sanitizers-cmake (e.g., ASan_GNU_FLAGS, UBSan_Clang_FLAGS)
      set(_flag_var "${_sanitizer}_${CMAKE_CXX_COMPILER_ID}_FLAGS")
      if(NOT DEFINED ${_flag_var} OR "${${_flag_var}}" STREQUAL "")
        message(FATAL_ERROR
          "${_sanitizer} is not supported for compiler '${CMAKE_CXX_COMPILER_ID}' in the current toolchain")
      endif()

      # Convert flag string to list and apply to both compile and link options
      separate_arguments(_sanitizer_flag_list UNIX_COMMAND "${${_flag_var}}")
      if(_sanitizer_flag_list)
        target_compile_options(${target} INTERFACE ${_sanitizer_flag_list})
        target_link_options(${target} INTERFACE ${_sanitizer_flag_list})
      endif()
    endforeach()
  else()
    # For regular targets (non-interface), use sanitizers-cmake's add_sanitizers() function
    add_sanitizers(${target})
  endif()
endfunction()

# -------------------------
# Static analysis helper (from PoetStaticAnalysis.cmake)
# -------------------------
option(BAOBZI_ENABLE_CLANG_TIDY "Enable clang-tidy static analysis" ${BAOBZI_ENABLE_STATIC_ANALYSIS})
option(BAOBZI_CLANG_TIDY_CHECKS "Override default clang-tidy checks (leave empty for default checks)" "")
option(BAOBZI_CLANG_TIDY_WARNINGS_AS_ERRORS "Treat clang-tidy warnings as errors" OFF)
option(BAOBZI_ENABLE_CPPCHECK "Enable cppcheck static analysis" ${BAOBZI_ENABLE_STATIC_ANALYSIS})
option(BAOBZI_CPPCHECK_OPTIONS "Additional cppcheck options" "--enable=warning,style,performance,portability")

# Configure static analysis tools (clang-tidy and/or cppcheck) for a target
# Tools are only enabled if found on PATH, otherwise a warning is issued
function(baobzi_configure_static_analysis target)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR "baobzi_configure_static_analysis called with non-existent target '${target}'")
  endif()

  if(BAOBZI_ENABLE_CLANG_TIDY)
    find_program(_clang_tidy_exe NAMES clang-tidy clang-tidy-17 clang-tidy-16)
    if(_clang_tidy_exe)
      set(_clang_tidy_command "${_clang_tidy_exe}")
      # When BAOBZI_CLANG_TIDY_CHECKS is set, use those checks; otherwise let
      # clang-tidy pick up the .clang-tidy config file from the source tree.
      if(BAOBZI_CLANG_TIDY_CHECKS)
        set(_clang_tidy_command "${_clang_tidy_command};-checks=${BAOBZI_CLANG_TIDY_CHECKS}")
      endif()
      # Only analyze headers in the project's include/baobzi and src directories (not external dependencies)
      set(_clang_tidy_command "${_clang_tidy_command};-header-filter=^${PROJECT_SOURCE_DIR}/(include/baobzi|src)")
      # Speed up analysis by only checking syntax, not generating code
      set(_clang_tidy_command "${_clang_tidy_command};--extra-arg=-fsyntax-only")
      if(BAOBZI_CLANG_TIDY_WARNINGS_AS_ERRORS)
        set(_clang_tidy_command "${_clang_tidy_command};-warnings-as-errors=*")
      endif()
      set_property(TARGET ${target} PROPERTY CXX_CLANG_TIDY "${_clang_tidy_command}")
    else()
      message(WARNING "BAOBZI_ENABLE_CLANG_TIDY is ON but clang-tidy was not found on PATH")
    endif()
  endif()

  if(BAOBZI_ENABLE_CPPCHECK)
    find_program(_cppcheck_exe NAMES cppcheck)
    if(_cppcheck_exe)
      set(_cppcheck_command "${_cppcheck_exe};--inline-suppr;${BAOBZI_CPPCHECK_OPTIONS}")
      set_property(TARGET ${target} PROPERTY CXX_CPPCHECK "${_cppcheck_command}")
    else()
      message(WARNING "BAOBZI_ENABLE_CPPCHECK is ON but cppcheck was not found on PATH")
    endif()
  endif()
endfunction()

# -------------------------
# Docs helper (from PoetDocs.cmake)
# -------------------------
option(BAOBZI_GENERATE_DOCS "Generate documentation using Doxygen + Sphinx pipeline" OFF)

if(BAOBZI_GENERATE_DOCS)
    # Require Doxygen for API documentation extraction
    find_package(Doxygen REQUIRED)
    # Require Sphinx for generating HTML documentation
    find_program(SPHINX_BUILD_EXECUTABLE NAMES sphinx-build REQUIRED)
    # Require Python for Sphinx and its extensions
    find_package(Python COMPONENTS Interpreter REQUIRED)

    # Check if required Python packages (breathe, exhale) are installed
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import breathe, exhale"
        RESULT_VARIABLE DOCS_DEPS_CHECK_RESULT
        OUTPUT_QUIET ERROR_QUIET
    )

    if(NOT DOCS_DEPS_CHECK_RESULT EQUAL 0)
        message(WARNING "Python packages 'breathe' and 'exhale' not found. Docs generation may fail. Please run 'pip install -r docs/requirements.txt'.")
    endif()

    # Generate Doxyfile from template
    configure_file(${CMAKE_SOURCE_DIR}/docs/Doxyfile.in ${CMAKE_BINARY_DIR}/docs/Doxyfile @ONLY)

    # Target: Generate Doxygen XML output from source code
    add_custom_target(doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/docs/Doxyfile
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/docs
        COMMENT "Generating API documentation with Doxygen"
    )

    # Target: Generate HTML documentation from Doxygen XML using Sphinx
    add_custom_target(sphinx
        DEPENDS doxygen
        COMMAND ${CMAKE_COMMAND} -E env DOXYGEN_XML_OUTPUT=${CMAKE_BINARY_DIR}/docs/xml
                ${SPHINX_BUILD_EXECUTABLE} -b html
                ${CMAKE_SOURCE_DIR}/docs ${CMAKE_BINARY_DIR}/docs/_build/html
        COMMENT "Generating HTML documentation with Sphinx"
    )

    # Target: Complete documentation build (alias for sphinx target)
    add_custom_target(docs DEPENDS sphinx)
    message(STATUS "BAOBZI: Documentation targets enabled (doxygen, sphinx, docs)")
endif()

# -------------------------
# Coverage target (moved from top-level)
# -------------------------
# Creates a `coverage` custom target that:
# 1. Builds all test executables
# 2. Runs the test suite using CTest
# 3. Collects code coverage data
# 4. Generates an HTML coverage report
#
# Prefers lcov+genhtml (more robust), falls back to gcovr if unavailable
find_program(GCOVR_EXECUTABLE gcovr)
find_program(LCOV_EXECUTABLE lcov)
find_program(GENHTML_EXECUTABLE genhtml)

# Prefer lcov+genhtml when available (generally more robust and handles complex build trees better)
# Falls back to gcovr if lcov/genhtml are not found
if(LCOV_EXECUTABLE AND GENHTML_EXECUTABLE)
  set(LCOV_INFO ${CMAKE_BINARY_DIR}/coverage.info)
  set(LCOV_FILTERED ${CMAKE_BINARY_DIR}/coverage.filtered.info)
  set(COVERAGE_DIR ${CMAKE_BINARY_DIR}/coverage)

  add_custom_target(coverage
    DEPENDS baobzi_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR} --output-on-failure
    COMMAND ${LCOV_EXECUTABLE} --capture --directory ${CMAKE_BINARY_DIR} --output-file ${LCOV_INFO} --ignore-errors inconsistent,unused
    # Remove system headers (/usr/*) and CMake FetchContent dependencies (*/_deps/*)
    # to focus coverage reports on project sources only
    COMMAND ${LCOV_EXECUTABLE} --remove ${LCOV_INFO} "/usr/*" "*/_deps/*" --output-file ${LCOV_FILTERED} --ignore-errors inconsistent,unused
    COMMAND ${GENHTML_EXECUTABLE} -o ${COVERAGE_DIR} ${LCOV_FILTERED}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running tests and generating coverage report (lcov+genhtml) -> ${COVERAGE_DIR}/index.html"
    VERBATIM
  )
elseif(GCOVR_EXECUTABLE)
  # Fallback to gcovr if lcov/genhtml aren't available
  add_custom_target(coverage
    DEPENDS baobzi_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR} --output-on-failure
    # Filter to project sources (include/baobzi and tests), excluding external dependencies and system headers
    COMMAND ${GCOVR_EXECUTABLE} -r ${CMAKE_SOURCE_DIR} --filter "include/baobzi/|tests/" --exclude ".*/_deps/.*" --exclude "/usr/.*" --gcov-ignore-errors=no_working_dir_found --html --html-details -o ${CMAKE_BINARY_DIR}/coverage-report.html
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running tests and generating coverage report (gcovr) -> ${CMAKE_BINARY_DIR}/coverage-report.html"
    VERBATIM
  )
else()
  message(STATUS "Coverage tools not found: install 'lcov'+'genhtml' or 'gcovr' to enable the 'coverage' target.")
  add_custom_target(coverage
    COMMAND ${CMAKE_COMMAND} -E echo "Coverage tools missing. Install 'lcov'+'genhtml' or 'gcovr' and re-run CMake to enable coverage generation."
  )
endif()

# Ensure coverage target builds all test executables before running CTest
# This adds dependencies on all C++ standard-specific test targets (C++17, C++20, C++23)
if(TARGET coverage)
  set(BAOBZI_TEST_STANDARDS 23 20 17)
  foreach(BAOBZI_STD IN LISTS BAOBZI_TEST_STANDARDS)
    if(TARGET baobzi_tests_std${BAOBZI_STD})
      add_dependencies(coverage baobzi_tests_std${BAOBZI_STD})
    endif()
  endforeach()
  # Also add dependency on the main test target if it exists
  if(TARGET baobzi_tests)
    add_dependencies(coverage baobzi_tests)
  endif()
endif()
