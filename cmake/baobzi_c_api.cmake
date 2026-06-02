# baobzi_c_api.cmake — C ABI (libbaobzi_c) for the new fit/options surface.
#
# Builds a shared and a static library from src/capi/*.cpp (each per-(dtype,
# input_dim) dispatch TU compiles in parallel) plus the extern "C" shim, and
# wires the C smoke example. The global -march/-ffp-contract flags from the
# top CMakeLists already apply; nothing ISA-specific is added here.

include_guard(GLOBAL)

# ---------------------------------------------------------------------------
# Header-only C++ template API.
#
# `baobzi::baobzi` carries the `include/` tree plus the transitive polyfit /
# POET headers it instantiates against. Those deps are FetchContent-only (not
# separately installable), so this target is for in-tree consumers
# (add_subdirectory / FetchContent) — it is deliberately NOT part of the
# installed `find_package(baobzi)` export set. The installed package ships the
# self-contained C ABI (`baobzi::baobzi_c`) instead. See baobzi_install.cmake.
# ---------------------------------------------------------------------------
add_library(baobzi_headers INTERFACE)
add_library(baobzi::baobzi ALIAS baobzi_headers)
target_include_directories(baobzi_headers INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(baobzi_headers INTERFACE polyfit::polyfit poet::poet)
target_compile_features(baobzi_headers INTERFACE cxx_std_20)

option(BAOBZI_BUILD_C_API "Build the C ABI (libbaobzi_c) + C example" ON)
if(NOT BAOBZI_BUILD_C_API)
  return()
endif()

set(_baobzi_c_sources
  src/capi/baobzi.cpp
  src/capi/dispatch_f64_dim1.cpp
  src/capi/dispatch_f64_dim2.cpp
  src/capi/dispatch_f64_dim3.cpp
  src/capi/dispatch_f32_dim1.cpp
  src/capi/dispatch_f32_dim2.cpp
  src/capi/dispatch_f32_dim3.cpp)

function(_baobzi_configure_c_target tgt)
  target_include_directories(${tgt} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  # polyfit/POET are header-only and only needed while compiling these TUs.
  # Guard with BUILD_INTERFACE so the dependency is absent from the exported
  # usage requirements — critical for the STATIC archive, whose PRIVATE deps
  # would otherwise leak as LINK_ONLY into baobziTargets.cmake and make
  # install(EXPORT) demand polyfit/POET (FetchContent-only) be exported too.
  target_link_libraries(${tgt} PRIVATE
    $<BUILD_INTERFACE:polyfit::polyfit> $<BUILD_INTERFACE:poet::poet>)
  set_property(TARGET ${tgt} PROPERTY POSITION_INDEPENDENT_CODE ON)
  baobzi_enable_warnings(${tgt})
endfunction()

add_library(baobzi_c SHARED ${_baobzi_c_sources})
add_library(baobzi::baobzi_c ALIAS baobzi_c)
_baobzi_configure_c_target(baobzi_c)
set_property(GLOBAL APPEND PROPERTY BAOBZI_INSTALL_TARGETS baobzi_c)

add_library(baobzi_c_static STATIC ${_baobzi_c_sources})
add_library(baobzi::baobzi_c_static ALIAS baobzi_c_static)
_baobzi_configure_c_target(baobzi_c_static)
set_target_properties(baobzi_c_static PROPERTIES OUTPUT_NAME baobzi_c)
set_property(GLOBAL APPEND PROPERTY BAOBZI_INSTALL_TARGETS baobzi_c_static)

# The C examples and the pure-C conformance test are compiled with a C (not
# C++) compiler against the installed-shape header, proving the surface is
# C-clean and that nothing throws across the boundary. `add_test` below needs
# CTest enabled; this include runs before baobzi_tests.cmake (the C lib must
# exist before test_c links it), so enable testing here. enable_testing() is
# idempotent with the later include(CTest) in baobzi_tests.cmake.
if(BAOBZI_BUILD_EXAMPLES OR BAOBZI_BUILD_TESTS)
  enable_language(C)
  enable_testing()
  # C TUs call exp()/fabs()/etc.: C, unlike C++, needs libm linked explicitly.
  find_library(BAOBZI_LIBM m)

  # Build a C11 executable linking baobzi_c (+ libm). Deliberately no
  # baobzi_enable_warnings: that profile is C++-oriented (-Wold-style-cast,
  # etc.) and these are C translation units.
  function(_baobzi_add_c_program name source)
    add_executable(${name} ${source})
    set_target_properties(${name} PROPERTIES C_STANDARD 11
                                             C_STANDARD_REQUIRED ON)
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(${name} PRIVATE baobzi_c)
    if(BAOBZI_LIBM)
      target_link_libraries(${name} PRIVATE ${BAOBZI_LIBM})
    endif()
  endfunction()
endif()

# C examples: each self-checks and returns EXIT_FAILURE on a bad result, so it
# doubles as a ctest.
if(BAOBZI_BUILD_EXAMPLES)
  foreach(_ex simple simple2d simple3d vector_output sorted with_options float32)
    _baobzi_add_c_program(baobzi_c_${_ex} examples/C/${_ex}.c)
    add_test(NAME c_example_${_ex} COMMAND baobzi_c_${_ex})
  endforeach()
endif()

# Pure-C ABI conformance test (+ an optional valgrind run proving baobzi_free
# is leak-clean, registered only when valgrind is available).
if(BAOBZI_BUILD_TESTS)
  _baobzi_add_c_program(test_c_abi tests/test_c_abi.c)
  add_test(NAME test_c_abi COMMAND test_c_abi)

  find_program(BAOBZI_VALGRIND valgrind)
  if(BAOBZI_VALGRIND)
    add_test(NAME test_c_abi_valgrind
             COMMAND ${BAOBZI_VALGRIND} --error-exitcode=1 --leak-check=full
                     --errors-for-leak-kinds=definite $<TARGET_FILE:test_c_abi>)
  endif()
endif()
