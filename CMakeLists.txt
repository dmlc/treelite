cmake_minimum_required (VERSION 3.1)
project(treelite)
include(cmake/Utils.cmake)
find_package(OpenMP)
find_package(Protobuf)

# enable custom logging facility in dmlc-core
add_definitions(-DDMLC_LOG_CUSTOMIZE)

# check MSVC version
if(MSVC)
  if(MSVC_VERSION LESS 1900)
    message(FATAL_ERROR "Need Visual Studio 2015 or newer to compile tree-lite")
  endif()
endif()

# define CMake info macros
msvc_locate_vcvarsall_bat()
set_source_files_properties(src/c_api/c_api_common.cc PROPERTIES COMPILE_DEFINITIONS TREELITE_MSVC_VARSALL_BAT="${VCVARSALL_BAT}")

set_default_configuration_release()
msvc_use_static_runtime()

# OpenMP check
if(OpenMP_FOUND)
  set_source_files_properties(src/c_api/c_api_common.cc PROPERTIES COMPILE_DEFINITIONS OPENMP_SUPPORT)
else()
  include(dmlc-core/cmake/Utils.cmake)
  dmlccore_option(USE_OPENMP "Build with OpenMP" OFF)
endif()

# Compiler flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
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
)

file(GLOB_RECURSE SOURCES
    src/*.cc
    src/*.h
    include/*.h
)

# SOURCES: sources included only in the main package
# RUNTIME_SOURCES: sources included only in the runtime
# COMMON_SOURCES: sources included both in the main pakcage and in the runtime
set(COMMON_SOURCES src/data.cc src/logging.cc src/c_api/c_api_common.h src/c_api/c_api_common.cc src/c_api/c_api_error.h src/c_api/c_api_error.cc src/common/math.h include/treelite/data.h include/treelite/c_api_common.h include/treelite/logging.h)
set(RUNTIME_SOURCES src/c_api/c_api_runtime.cc src/predictor.cc include/treelite/predictor.h include/treelite/c_api_runtime.h)

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
if(Protobuf_FOUND)
  set_source_files_properties(src/frontend/protobuf.cc PROPERTIES COMPILE_DEFINITIONS PROTOBUF_SUPPORT)
  include_directories(${PROTOBUF_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
  protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS src/tree.proto)
  list(INSERT SOURCES 0 ${PROTO_SRCS} ${PROTO_HDRS})
endif()

# dmlc-core
add_subdirectory(dmlc-core)
# separate sets of dependencies for main package and runtime
set(LINK_LIBRARIES "")
set(RUNTIME_LINK_LIBRARIES "")
set(COMMON_LINK_LIBRARIES dmlc)  # dmlc is needed by both runtime and main package

# -ldl for UNIX-like systems
if (UNIX)
  list(APPEND RUNTIME_LINK_LIBRARIES dl)
endif (UNIX)

# Protobuf library
if (Protobuf_FOUND)
  list(APPEND LINK_LIBRARIES ${PROTOBUF_LIBRARIES})
endif()

add_library(objtreelite OBJECT ${SOURCES})
add_library(objtreelite_runtime OBJECT ${RUNTIME_SOURCES})
add_library(objtreelite_common OBJECT ${COMMON_SOURCES})

# Executable
add_executable(runtreelite $<TARGET_OBJECTS:objtreelite> $<TARGET_OBJECTS:objtreelite_runtime> $<TARGET_OBJECTS:objtreelite_common> src/cli_main.cc)
set_target_properties(runtreelite PROPERTIES
  OUTPUT_NAME treelite
)
set_output_directory(runtreelite ${PROJECT_SOURCE_DIR})
target_link_libraries(runtreelite ${LINK_LIBRARIES} ${RUNTIME_LINK_LIBRARIES} ${COMMON_LINK_LIBRARIES})

# Shared library
add_library(treelite SHARED $<TARGET_OBJECTS:objtreelite> $<TARGET_OBJECTS:objtreelite_common>)
add_library(treelite_runtime SHARED $<TARGET_OBJECTS:objtreelite_runtime> $<TARGET_OBJECTS:objtreelite_common>)
target_link_libraries(treelite ${LINK_LIBRARIES} ${COMMON_LINK_LIBRARIES})
target_link_libraries(treelite_runtime ${RUNTIME_LINK_LIBRARIES} ${COMMON_LINK_LIBRARIES})
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
