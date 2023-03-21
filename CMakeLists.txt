set (CMAKE_FIND_NO_INSTALL_PREFIX TRUE FORCE)
cmake_minimum_required (VERSION 3.13)
project(treelite LANGUAGES CXX C VERSION 3.2.0)

# check MSVC version
if(MSVC)
  if(MSVC_VERSION LESS 1910)
    message(FATAL_ERROR "Need Visual Studio 2017 or newer to compile Treelite")
  endif()
endif()

option(TEST_COVERAGE "C++ test coverage" OFF)
option(USE_OPENMP "Use OpenMP" ON)
option(BUILD_DOXYGEN "Build documentation for C/C++ functions using Doxygen." OFF)
option(BUILD_CPP_TEST "Build C++ tests" OFF)
option(BUILD_STATIC_LIBS "Build static libs, in addition to dynamic libs" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(BUILD_JVM_RUNTIME "Build Treelite runtime for JVM" OFF)
option(ENABLE_ALL_WARNINGS "Enable all compiler warnings. Only effective for GCC/Clang" OFF)
option(USE_SANITIZER "Use sanitizer flags" OFF)
SET(ENABLED_SANITIZERS "address" "leak" "undefined" CACHE STRING
        "Semicolon separated list of sanitizer names. E.g 'address;leak'.")

if(USE_SANITIZER)
  include(cmake/Sanitizer.cmake)
  enable_sanitizers("${ENABLED_SANITIZERS}")
endif()

if(ENABLE_ALL_WARNINGS)
  if((NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang") AND (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(SEND_ERROR "ENABLE_ALL_WARNINGS is only available for Clang and GCC.")
  endif()
endif()

if(BUILD_DOXYGEN)
  include(cmake/Doxygen.cmake)
  run_doxygen()
endif()

# When installing dependencies, use Conda environment if available
if(DETECT_CONDA_ENV)
  if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
    message(STATUS "Detected Conda environment, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
      message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
      set(CMAKE_INSTALL_PREFIX $ENV{CONDA_PREFIX})
    endif()
  else()
    message(STATUS "No Conda environment detected")
  endif()
endif()

include(cmake/ExternalLibs.cmake)
include(cmake/Utils.cmake)
include(cmake/Version.cmake)

add_subdirectory(src)
if(BUILD_JVM_RUNTIME)
  add_subdirectory(runtime/java)
endif()
if(BUILD_CPP_TEST)
  add_subdirectory(tests/cpp)
endif()

add_library(treelite SHARED)
add_library(treelite_runtime SHARED)
target_link_libraries(treelite PRIVATE objtreelite objtreelite_common)
target_link_libraries(treelite_runtime PRIVATE objtreelite_runtime objtreelite_common)

set(TREELITE_TARGETS treelite treelite_runtime)

if(BUILD_STATIC_LIBS)
  add_library(treelite_static STATIC)
  add_library(treelite_runtime_static STATIC)
  target_link_libraries(treelite_static PRIVATE objtreelite objtreelite_common)
  target_link_libraries(treelite_runtime_static PRIVATE objtreelite_runtime objtreelite_common)
  list(APPEND TREELITE_TARGETS treelite_static treelite_runtime_static)
endif()

foreach(lib ${TREELITE_TARGETS})
  set_output_directory(${lib} ${PROJECT_BINARY_DIR})
endforeach()

# Export install targets
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(INSTALL_TARGETS ${TREELITE_TARGETS}
                    objtreelite objtreelite_common objtreelite_runtime rapidjson)
if(NOT FMTLIB_FROM_SYSTEM_ROOT)
  list(APPEND INSTALL_TARGETS fmt-header-only)
endif()
install(TARGETS ${INSTALL_TARGETS}
    EXPORT TreeliteTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION include)
install(DIRECTORY include/treelite ${PROJECT_BINARY_DIR}/include/treelite
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT TreeliteTargets
    FILE TreeliteTargets.cmake
    NAMESPACE treelite::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/treelite)
configure_package_config_file(
    cmake/TreeliteConfig.cmake.in
    ${PROJECT_BINARY_DIR}/cmake/TreeliteConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/treelite)
write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/cmake/TreeliteConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${PROJECT_BINARY_DIR}/cmake/TreeliteConfig.cmake
    ${PROJECT_BINARY_DIR}/cmake/TreeliteConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/treelite)

write_version()

set_default_configuration_release()
msvc_use_static_runtime()
