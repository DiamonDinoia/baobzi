cmake_minimum_required(VERSION 3.25)
project(baobzi_python LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Python discovery (nanobind is located by the shared fragment below).
# ---------------------------------------------------------------------------
find_package(Python 3.9
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# ---------------------------------------------------------------------------
# Pull in the parent project so we get baobzi::baobzi_c_static.
# Guard so that an add_subdirectory(bindings/python) from the parent doesn't
# re-include the parent (which would recurse).
# ---------------------------------------------------------------------------
if(NOT TARGET baobzi_c_static)
  # Disable optional targets that take a long time (or need extra deps).
  set(BAOBZI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(BAOBZI_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
  # A stray warning on the wheel-builder's compiler must never break
  # `pip install` — never promote warnings to errors here.
  set(BAOBZI_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)

  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."   # repo root
    "${CMAKE_BINARY_DIR}/baobzi_root"    # dedicated binary subtree
  )
endif()

# ---------------------------------------------------------------------------
# nanobind extension (shared definition).
# ---------------------------------------------------------------------------
include("${CMAKE_CURRENT_SOURCE_DIR}/baobzi_python.cmake")
baobzi_add_python_module(baobzi::baobzi_c_static)

# Install the shared object into the baobzi/ package directory so that
# `pip install .` (scikit-build-core) puts it alongside __init__.py.
install(TARGETS _baobzi LIBRARY DESTINATION baobzi)
