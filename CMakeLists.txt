cmake_minimum_required (VERSION 3.1)
project(treelite)
include(cmake/Utils.cmake)
find_package(OpenMP)
find_package(Protobuf)

set_default_configuration_release()
msvc_use_static_runtime()

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
# Only add main function for executable target
list(REMOVE_ITEM SOURCES ${PROJECT_SOURCE_DIR}/src/cli_main.cc)

# Protobuf library

if (Protobuf_FOUND)
  set_source_files_properties(src/frontend/protobuf.cc PROPERTIES COMPILE_DEFINITIONS PROTOBUF_SUPPORT)
  include_directories(${PROTOBUF_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
  protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS src/tree.proto)
  list(INSERT SOURCES 0 ${PROTO_SRCS} ${PROTO_HDRS})
endif()

# dmlc-core
add_subdirectory(dmlc-core)
set(LINK_LIBRARIES dmlc)

# -ldl for UNIX-like systems
if (UNIX)
  list(APPEND LINK_LIBRARIES dl)
endif (UNIX)

# Protobuf library
if (Protobuf_FOUND)
  list(APPEND LINK_LIBRARIES ${PROTOBUF_LIBRARIES})
endif()

add_library(objtreelite OBJECT ${SOURCES})

# Executable
add_executable(runtreelite $<TARGET_OBJECTS:objtreelite> src/cli_main.cc)
set_target_properties(runtreelite PROPERTIES
  OUTPUT_NAME treelite
)
set_output_directory(runtreelite ${PROJECT_SOURCE_DIR})
target_link_libraries(runtreelite ${LINK_LIBRARIES})

# Shared library
add_library(treelite SHARED $<TARGET_OBJECTS:objtreelite>)
target_link_libraries(treelite ${LINK_LIBRARIES})
set_output_directory(treelite ${PROJECT_SOURCE_DIR}/lib)
if(MINGW)
  # remove the 'lib' prefix to conform to windows convention for shared library names
  set_target_properties(treelite PROPERTIES PREFIX "")
endif()

# Ensure these two targets do not build simultaneously, as they produce outputs with conflicting names
add_dependencies(treelite runtreelite)

# Group sources
auto_source_group("${SOURCES}")
