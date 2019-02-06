cmake_minimum_required (VERSION 3.1)
project(treelite_runtime)
include(cmake/Utils.cmake)
find_package(OpenMP)

# enable custom logging facility in dmlc-core
add_definitions(-DDMLC_LOG_CUSTOMIZE)

# check MSVC version
if(MSVC)
  if(MSVC_VERSION LESS 1900)
    message(FATAL_ERROR "Need Visual Studio 2015 or newer to compile treelite")
  endif()
endif()

set_default_configuration_release()
msvc_use_static_runtime()

option(ENABLE_S3 "Build with S3 support" OFF)
option(TEST_COVERAGE "C++ test coverage" OFF)

# Compiler flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  # Multithreaded compilation
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
  # Performance
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
endif()
if(TEST_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fprofile-arcs -ftest-coverage")
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

if(EXISTS ${PROJECT_SOURCE_DIR}/FILELIST)
  copy_files(FILELIST RUNTIME_FILES)
endif()

file(GLOB_RECURSE SOURCES
    src/*.cc
    src/*.h
    include/*.h
)

# -ldl for UNIX-like systems
if (UNIX)
  list(APPEND RUNTIME_LINK_LIBRARIES dl)
endif (UNIX)

add_library(objtreelite_runtime OBJECT ${SOURCES})

get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)
get_filename_component(GRANDPARENT_DIR ${PARENT_DIR} DIRECTORY)

# dmlc-core
if (NOT TARGET dmlc)
  set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS}")  # Save BUILD_SHARED_LIBS
  set(BUILD_SHARED_LIBS OFF)  # dmlc lib must always be static
  # dmlc-core options
  include(${GRANDPARENT_DIR}/dmlc-core/cmake/Utils.cmake)
  if (ENABLE_S3)
    dmlccore_option(USE_S3 "Build with S3 support" ON)
  endif()
  # OpenMP check
  if(NOT OPENMP_FOUND)
    dmlccore_option(USE_OPENMP "Build with OpenMP" OFF)
  endif()
  add_subdirectory(${GRANDPARENT_DIR}/dmlc-core ${GRANDPARENT_DIR}/dmlc-core/build)
  include_directories(${GRANDPARENT_DIR}/dmlc-core/include)
  set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}")  # Restore BUILD_SHARED_LIBS
  if (TARGET dmlc_unit_tests)
    set_target_properties(dmlc_unit_tests PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
  endif()
endif()
list(APPEND RUNTIME_LINK_LIBRARIES dmlc)

# Shared library
add_library(treelite_runtime SHARED $<TARGET_OBJECTS:objtreelite_runtime>)
target_link_libraries(treelite_runtime ${RUNTIME_LINK_LIBRARIES})
set_output_directory(treelite_runtime ${PROJECT_SOURCE_DIR}/lib)
if(MINGW)
  # remove the 'lib' prefix to conform to windows convention for shared library names
  set_target_properties(treelite_runtime PROPERTIES PREFIX "")
endif()

if(WIN32)
  install(TARGETS treelite_runtime
          RUNTIME DESTINATION lib)
else()
  install(TARGETS treelite_runtime
          LIBRARY DESTINATION lib)
endif()

install(DIRECTORY include/treelite DESTINATION include
        FILES_MATCHING PATTERN "*.h")
