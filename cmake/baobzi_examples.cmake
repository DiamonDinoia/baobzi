# baobzi_examples.cmake — C++ examples.

include_guard(GLOBAL)

if(NOT BAOBZI_BUILD_EXAMPLES)
  return()
endif()

function(_baobzi_add_cpp_example name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(${name} PRIVATE polyfit::polyfit poet::poet)
  baobzi_enable_warnings(${name})
endfunction()

_baobzi_add_cpp_example(baobzi_timing_cpp  examples/c++/baobzi_timing.cpp)
_baobzi_add_cpp_example(baobzi_microbench  examples/c++/baobzi_microbench.cpp)
target_link_libraries(baobzi_microbench PRIVATE nanobench)
_baobzi_add_cpp_example(baobzi_bench_pack_scatter
                        examples/c++/baobzi_bench_pack_scatter.cpp)
target_link_libraries(baobzi_bench_pack_scatter PRIVATE nanobench)
_baobzi_add_cpp_example(baobzi_bench_sorted
                        examples/c++/baobzi_bench_sorted.cpp)
target_link_libraries(baobzi_bench_sorted PRIVATE nanobench)
_baobzi_add_cpp_example(bench_scratch_gate examples/c++/bench_scratch_gate.cpp)
target_link_libraries(bench_scratch_gate PRIVATE nanobench)
_baobzi_add_cpp_example(baobzi_perf_driver examples/c++/baobzi_perf_driver.cpp)
_baobzi_add_cpp_example(simple1d           examples/c++/simple1d.cpp)
_baobzi_add_cpp_example(simple2d           examples/c++/simple2d.cpp)
_baobzi_add_cpp_example(simple3d           examples/c++/simple3d.cpp)
_baobzi_add_cpp_example(with_options       examples/c++/with_options.cpp)
_baobzi_add_cpp_example(vector_output      examples/c++/vector_output.cpp)
