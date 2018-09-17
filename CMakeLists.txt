cmake_minimum_required (VERSION 3.1)
project(treelite)
include(cmake/Utils.cmake)
find_package(OpenMP)

# Use RPATH on Mac OS X as flexible mechanism for locating dependencies
# See https://blog.kitware.com/upcoming-in-cmake-2-8-12-osx-rpath-support/
set(CMAKE_MACOSX_RPATH TRUE)

# Set BUILD_SHARED_LIBS as option. By default, build shared libraries;
# User can set this to OFF to build static libraries instead.
option(BUILD_SHARED_LIBS "Build shared library" ON)
option(ENABLE_PROTOBUF "Enable Protobuf" OFF)
option(ENABLE_S3 "Build with S3 support" OFF)
option(TEST_COVERAGE "C++ test coverage" OFF)

# enable custom logging facility in dmlc-core
add_definitions(-DDMLC_LOG_CUSTOMIZE)

# Protobuf
if (ENABLE_PROTOBUF)
  set(protobuf_BUILD_TESTS OFF CACHE BOOL "enable tests for protobuf" FORCE)
  set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "enable shared libs for protobuf" FORCE)
  add_subdirectory(3rdparty/protobuf/cmake)
  set(PROTOBUF_PROTOC_EXECUTABLE protoc)
  set(PROTOBUF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/3rdparty/protobuf/src")
  set(PROTOBUF_LIBRARIES libprotobuf)
  add_definitions(-DTREELITE_PROTOBUF_SUPPORT)
else()
  set(PROTOBUF_LIBRARIES "")
endif()

# check MSVC version
if(MSVC)
  if(MSVC_VERSION LESS 1900)
    message(FATAL_ERROR "Need Visual Studio 2015 or newer to compile treelite")
  endif()
endif()

set_default_configuration_release()
msvc_use_static_runtime()

# dmlc-core option
include(dmlc-core/cmake/Utils.cmake)
if (ENABLE_S3)
  dmlccore_option(USE_S3 "Build with S3 support" ON)
endif()

# OpenMP check
if(OPENMP_FOUND)
  add_definitions(-DTREELITE_OPENMP_SUPPORT)
else(OPENMP_FOUND)
  dmlccore_option(USE_OPENMP "Build with OpenMP" OFF)
endif(OPENMP_FOUND)

# Compiler flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif(OPENMP_FOUND)
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

include_directories (
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/dmlc-core/include
    ${PROJECT_SOURCE_DIR}/runtime/native/include
    ${PROJECT_SOURCE_DIR}/3rdparty/fmt   # fmtlib
)

# use fmtlib as header-only
add_definitions(-DFMT_HEADER_ONLY)
add_subdirectory(3rdparty/fmt EXCLUDE_FROM_ALL)

file(GLOB_RECURSE SOURCES
    src/*.cc
    src/*.h
    include/*.h
)

# Protobuf library
if (ENABLE_PROTOBUF)
  include_directories(${PROTOBUF_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
  PROTOBUF_GENERATE_CPP(TREE_PROTO_SRCS TREE_PROTO_HDRS src/tree.proto)
  PROTOBUF_GENERATE_JAVA(tree_proto_java src/tree.proto)
  PROTOBUF_GENERATE_PYTHON(tree_proto_python src/tree.proto)
  list(INSERT SOURCES 0 ${TREE_PROTO_SRCS} ${TREE_PROTO_HDRS})
endif()

# dmlc-core
set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS}")  # Save BUILD_SHARED_LIBS
set(BUILD_SHARED_LIBS OFF)  # dmlc lib must always be static
add_subdirectory(dmlc-core)
set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}")  # Restore BUILD_SHARED_LIBS
if (TARGET dmlc_unit_tests)
  set_target_properties(dmlc_unit_tests PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
endif()

set(LINK_LIBRARIES dmlc)

# Protobuf library
list(APPEND LINK_LIBRARIES ${PROTOBUF_LIBRARIES})

add_library(objtreelite OBJECT ${SOURCES})

# Native runtime
add_subdirectory(runtime/native)

# JNI runtime
find_package(JNI)
if(JNI_FOUND)
  add_subdirectory(runtime/java)
endif()

# Shared library
add_library(treelite $<TARGET_OBJECTS:objtreelite>)
target_link_libraries(treelite ${LINK_LIBRARIES})
set_output_directory(treelite ${PROJECT_SOURCE_DIR}/lib)
if(MINGW)
  # remove the 'lib' prefix to conform to windows convention for shared library names
  set_target_properties(treelite PROPERTIES PREFIX "")
endif()

# Group sources
auto_source_group("${SOURCES}")
