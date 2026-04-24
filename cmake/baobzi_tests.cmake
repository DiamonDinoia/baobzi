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
  catch_discover_tests(${name})
endfunction()

_baobzi_add_cpp_test(test_template tests/test_template.cpp)
_baobzi_add_cpp_test(test_cpp      tests/test_cpp.cpp)
