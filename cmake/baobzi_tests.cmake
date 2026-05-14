# baobzi_tests.cmake — Catch2 test suite wiring.

include_guard(GLOBAL)

if(NOT BAOBZI_BUILD_TESTS)
  return()
endif()

include(CTest)
include(Catch)

function(_baobzi_add_cpp_test name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(${name} PRIVATE
    Catch2::Catch2WithMain polyfit::polyfit poet::poet)
  baobzi_enable_warnings(${name})
  # Catch2's TEST_CASE expands to a use of __COUNTER__, which bleeding-edge
  # clang flags as a C2y extension; the warning is harmless and out of our
  # control (lives in Catch2 macros), so silence it on test targets only.
  target_compile_options(${name} PRIVATE
    $<$<AND:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>,$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,19>>:-Wno-c2y-extensions>)
  catch_discover_tests(${name})
endfunction()

_baobzi_add_cpp_test(test_template tests/test_template.cpp)
_baobzi_add_cpp_test(test_cpp      tests/test_cpp.cpp)
_baobzi_add_cpp_test(test_greens   tests/test_greens.cpp)
_baobzi_add_cpp_test(test_threadsafe tests/test_threadsafe.cpp)
_baobzi_add_cpp_test(test_scratch    tests/test_scratch.cpp)
find_package(Threads REQUIRED)
target_link_libraries(test_threadsafe PRIVATE Threads::Threads)
