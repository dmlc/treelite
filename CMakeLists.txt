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

# enable custom logging facility in dmlc-core
add_definitions(-DDMLC_LOG_CUSTOMIZE)

# Protobuf
set(protobuf_BUILD_TESTS OFF CACHE BOOL "enable tests for protobuf" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "enable shared libs for protobuf" FORCE)
add_subdirectory(3rdparty/protobuf/cmake)
set(PROTOBUF_PROTOC_EXECUTABLE protoc)
set(PROTOBUF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/3rdparty/protobuf/src")
set(PROTOBUF_LIBRARIES libprotobuf)

# check MSVC version
if(MSVC)
  if(MSVC_VERSION LESS 1900)
    message(FATAL_ERROR "Need Visual Studio 2015 or newer to compile treelite")
  endif()
endif()

set_default_configuration_release()
msvc_use_static_runtime()

# OpenMP check
if(OPENMP_FOUND)
  add_definitions(-DTREELITE_OPENMP_SUPPORT)
else(OPENMP_FOUND)
  include(dmlc-core/cmake/Utils.cmake)
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

include_directories (
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/dmlc-core/include
    ${PROJECT_SOURCE_DIR}/3rdparty/fmt   # fmtlib
)

# use fmtlib as header-only
add_definitions(-DFMT_HEADER_ONLY)
add_subdirectory(3rdparty/fmt EXCLUDE_FROM_ALL)

# Copy files for runtime deployment
read_filelist(runtime/FILELIST RUNTIME_FILES)
deploy_file(runtime/CMakeLists.txt ".")
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/runtime/build)
foreach(f ${RUNTIME_FILES})
  deploy_file(${f} runtime)
endforeach()

file(GLOB_RECURSE SOURCES
    src/*.cc
    src/*.h
    include/*.h
)

# SOURCES: sources included only in the main package
# RUNTIME_SOURCES: sources included only in the runtime
# COMMON_SOURCES: sources included both in the main package and in the runtime
set(COMMON_SOURCES include/treelite/c_api_common.h include/treelite/logging.h include/treelite/omp.h src/c_api/c_api_common.cc src/c_api/c_api_error.cc src/c_api/c_api_error.h src/common/math.h src/logging.cc)
set(RUNTIME_SOURCES include/treelite/c_api_runtime.h include/treelite/predictor.h src/c_api/c_api_runtime.cc src/predictor.cc)

# Only add main function for executable target
# Only add runtime functions for runtime library
set(COMMON_SOURCES_WITH_PREFIX "")
set(RUNTIME_SOURCES_WITH_PREFIX "")
foreach(f ${COMMON_SOURCES})
  list(APPEND COMMON_SOURCES_WITH_PREFIX ${PROJECT_SOURCE_DIR}/${f})
endforeach(f)
foreach(g ${RUNTIME_SOURCES})
  list(APPEND RUNTIME_SOURCES_WITH_PREFIX ${PROJECT_SOURCE_DIR}/${g})
endforeach(g)
list(REMOVE_ITEM SOURCES ${PROJECT_SOURCE_DIR}/src/cli_main.cc ${COMMON_SOURCES_WITH_PREFIX} ${RUNTIME_SOURCES_WITH_PREFIX})

# Protobuf library
set_source_files_properties(src/frontend/protobuf.cc PROPERTIES COMPILE_DEFINITIONS PROTOBUF_SUPPORT)
include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS src/tree.proto)
PROTOBUF_GENERATE_JAVA(tree_proto_java src/tree.proto)
list(INSERT SOURCES 0 ${PROTO_SRCS} ${PROTO_HDRS})

# dmlc-core
set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS}")  # Save BUILD_SHARED_LIBS
set(BUILD_SHARED_LIBS OFF)  # dmlc lib must always be static
add_subdirectory(dmlc-core)
set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}")  # Restore BUILD_SHARED_LIBS
if (TARGET dmlc_unit_tests)
  set_target_properties(dmlc_unit_tests PROPERTIES EXCLUDE_FROM_ALL 1 EXCLUDE_FROM_DEFAULT_BUILD 1)
endif()

# separate sets of dependencies for main package and runtime
set(LINK_LIBRARIES dmlc)
set(RUNTIME_LINK_LIBRARIES "")

# -ldl for UNIX-like systems
if (UNIX)
  list(APPEND RUNTIME_LINK_LIBRARIES dl)
endif (UNIX)

# Protobuf library
list(APPEND LINK_LIBRARIES ${PROTOBUF_LIBRARIES})

add_library(objtreelite OBJECT ${SOURCES})
add_library(objtreelite_runtime OBJECT ${RUNTIME_SOURCES})
add_library(objtreelite_common OBJECT ${COMMON_SOURCES})

# Executable
add_executable(runtreelite $<TARGET_OBJECTS:objtreelite> $<TARGET_OBJECTS:objtreelite_runtime> $<TARGET_OBJECTS:objtreelite_common> src/cli_main.cc)
set_target_properties(runtreelite PROPERTIES
  OUTPUT_NAME treelite
)
set_output_directory(runtreelite ${PROJECT_SOURCE_DIR})
target_link_libraries(runtreelite ${LINK_LIBRARIES} ${RUNTIME_LINK_LIBRARIES})

# Shared library
add_library(treelite $<TARGET_OBJECTS:objtreelite> $<TARGET_OBJECTS:objtreelite_common>)
add_library(treelite_runtime $<TARGET_OBJECTS:objtreelite_runtime> $<TARGET_OBJECTS:objtreelite_common>)
target_link_libraries(treelite ${LINK_LIBRARIES})
target_link_libraries(treelite_runtime ${RUNTIME_LINK_LIBRARIES})
set_output_directory(treelite ${PROJECT_SOURCE_DIR}/lib)
set_output_directory(treelite_runtime ${PROJECT_SOURCE_DIR}/lib)
if(MINGW)
  # remove the 'lib' prefix to conform to windows convention for shared library names
  set_target_properties(treelite PROPERTIES PREFIX "")
  set_target_properties(treelite_runtime PROPERTIES PREFIX "")
endif()

# Ensure these two targets do not build simultaneously, as they produce outputs with conflicting names
add_dependencies(treelite runtreelite)

# Group sources
auto_source_group("${SOURCES}")
auto_source_group("${RUNTIME_SOURCES}")
auto_source_group("${COMMON_SOURCES}")
