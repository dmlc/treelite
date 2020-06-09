cmake_minimum_required (VERSION 3.16)
project(treelite LANGUAGES CXX C VERSION 0.92)

# check MSVC version
if(MSVC)
  if(MSVC_VERSION LESS 1910)
    message(FATAL_ERROR "Need Visual Studio 2017 or newer to compile Treelite")
  endif()
endif()

option(ENABLE_PROTOBUF "Enable Protobuf" OFF)
option(TEST_COVERAGE "C++ test coverage" OFF)
option(USE_OPENMP "Use OpenMP" ON)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)

# When installing dependencies, use Conda environment if available
if(DETECT_CONDA_ENV)
  if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
    message(STATUS "Detected Conda environment, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
      message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}")
      set (CMAKE_INSTALL_PREFIX $ENV{CONDA_PREFIX})
    endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  else()
    message(STATUS "No Conda environment detected")
  endif()
endif()

include(cmake/ExternalLibs.cmake)
include(cmake/Utils.cmake)
include(cmake/Version.cmake)

add_subdirectory(src)

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(objtreelite PUBLIC OpenMP::OpenMP_CXX)
  target_link_libraries(objtreelite_runtime PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(objtreelite PRIVATE -DTREELITE_OPENMP_SUPPORT)
  target_compile_definitions(objtreelite_runtime PRIVATE -DTREELITE_OPENMP_SUPPORT)
else()
  message(STATUS "Disabling OpenMP")
endif()

if(TEST_COVERAGE)
  if(MSVC)
    message(FATAL_ERROR "Test coverage not available on Windows")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
endif()

add_library(treelite SHARED)
target_link_libraries(treelite PUBLIC objtreelite)
add_library(treelite_static STATIC)
target_link_libraries(treelite_static PUBLIC objtreelite)

add_library(treelite_runtime SHARED)
target_link_libraries(treelite_runtime PUBLIC objtreelite_runtime)
add_library(treelite_runtime_static STATIC)
target_link_libraries(treelite_runtime_static PUBLIC objtreelite_runtime)

write_version()

set_default_configuration_release()
msvc_use_static_runtime()
