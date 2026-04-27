# baobzi_deps.cmake — fetch header-only dependencies and mark their include
# trees as system headers so consumer warnings (-Werror) don't fire on them.

include_guard(GLOBAL)
include(FetchContent)

FetchContent_Declare(
    polyfit
    GIT_REPOSITORY https://github.com/DiamonDinoia/polyfit.git
    GIT_TAG        c70d92334742d476892b7ce4b4028868302cdfe8
    SYSTEM
)
set(POLYFIT_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(POLYFIT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(polyfit)

FetchContent_Declare(
    poet
    GIT_REPOSITORY https://github.com/DiamonDinoia/POET.git
    GIT_TAG        af0d3a482f4d35da8277f73ef63858037d636441
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

if(BAOBZI_BUILD_TESTS)
  FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.9.0
    SYSTEM
  )
  FetchContent_MakeAvailable(catch2)
endif()
