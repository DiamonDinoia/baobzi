# baobzi_install.cmake — install rules + a relocatable find_package(baobzi)
# package for the C ABI.
#
# The installed package ships the C ABI (`baobzi::baobzi_c` /
# `baobzi::baobzi_c_static`): it is self-contained (only needs the installed
# `baobzi.h`, links polyfit/POET privately). The header-only C++ template API
# (`baobzi::baobzi`) is intentionally NOT installed — it instantiates against
# polyfit/POET headers that are FetchContent-only, so it is consumed in-tree
# via add_subdirectory / FetchContent (where those deps resolve). See
# baobzi_c_api.cmake and the README.

include_guard(GLOBAL)
include(CMakePackageConfigHelpers)

set(_baobzi_cmakedir "${CMAKE_INSTALL_LIBDIR}/cmake/baobzi")

get_property(_install_targets GLOBAL PROPERTY BAOBZI_INSTALL_TARGETS)
if(_install_targets)
  install(TARGETS ${_install_targets}
    EXPORT baobziTargets
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR})

  install(EXPORT baobziTargets
    NAMESPACE baobzi::
    DESTINATION ${_baobzi_cmakedir})

  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/baobziConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/baobziConfig.cmake"
    INSTALL_DESTINATION ${_baobzi_cmakedir})

  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/baobziConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  install(FILES
    "${PROJECT_BINARY_DIR}/baobziConfig.cmake"
    "${PROJECT_BINARY_DIR}/baobziConfigVersion.cmake"
    DESTINATION ${_baobzi_cmakedir})
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Relative DESTINATION so `cmake --install --prefix <dir>` is honored (an
# absolute ${CMAKE_INSTALL_PREFIX} here would bake in the configure-time
# prefix and ignore a later --prefix override).
install(FILES ${PROJECT_SOURCE_DIR}/LICENSE
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses/Baobzi)
