cmake_minimum_required (VERSION 3.1)
project(treelite4j)
find_package(JNI REQUIRED)
include(../native/cmake/Utils.cmake)

# check MSVC version
if(MSVC)
  if(MSVC_VERSION LESS 1900)
    message(FATAL_ERROR "Need Visual Studio 2015 or newer to compile treelite4j")
  endif()
endif()

set_default_configuration_release()
msvc_use_static_runtime()

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

# -ldl for UNIX-like systems
if (UNIX)
  list(APPEND LINK_LIBRARIES dl)
endif (UNIX)

get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)
if (NOT TARGET objtreelite_runtime)
  add_subdirectory(${PARENT_DIR}/native ${PARENT_DIR}/native/build)
endif()

include_directories(${JNI_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/treelite4j/src/native ${PARENT_DIR}/native/include)
add_library(treelite4j SHARED $<TARGET_OBJECTS:objtreelite_runtime> treelite4j/src/native/treelite4j.cpp)
set_output_directory(treelite4j ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(treelite4j ${LINK_LIBRARIES} ${JAVA_JVM_LIBRARY})
set_output_directory(treelite4j ${PROJECT_SOURCE_DIR}/lib)
if(MINGW)
  # remove the 'lib' prefix to conform to windows convention for shared library names
  set_target_properties(treelite4j PROPERTIES PREFIX "")
endif()
