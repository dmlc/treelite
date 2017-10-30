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
    dmlc-core/include/*.h
)

# -ldl for UNIX-like systems
if (UNIX)
  list(APPEND RUNTIME_LINK_LIBRARIES dl)
endif (UNIX)

add_library(objtreelite_runtime OBJECT ${SOURCES})

# Shared library
add_library(treelite_runtime SHARED $<TARGET_OBJECTS:objtreelite_runtime>)
target_link_libraries(treelite_runtime ${RUNTIME_LINK_LIBRARIES})
set_output_directory(treelite_runtime ${PROJECT_SOURCE_DIR}/lib)
if(MINGW)
  # remove the 'lib' prefix to conform to windows convention for shared library names
  set_target_properties(treelite PROPERTIES PREFIX "")
  set_target_properties(treelite_runtime PROPERTIES PREFIX "")
endif()
