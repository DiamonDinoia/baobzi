# baobzi_install.cmake — install rules for whatever libraries were built.

include_guard(GLOBAL)

get_property(_install_targets GLOBAL PROPERTY BAOBZI_INSTALL_TARGETS)
if(_install_targets)
  install(TARGETS ${_install_targets})
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${PROJECT_SOURCE_DIR}/LICENSE
  DESTINATION ${CMAKE_INSTALL_PREFIX}/share/licenses/Baobzi)
