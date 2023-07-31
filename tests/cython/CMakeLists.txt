cmake_minimum_required(VERSION 3.18)

project(treelite_serializer_ext LANGUAGES CXX)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

find_program(CYTHON "cython")

find_package(Treelite REQUIRED)

add_custom_command(
  OUTPUT serializer.cpp
  DEPENDS serializer.pyx
  VERBATIM
  COMMAND "${CYTHON}" "${PROJECT_SOURCE_DIR}/serializer.pyx" --output-file
          "${PROJECT_BINARY_DIR}/serializer.cpp")

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

python_add_library(serializer MODULE "${PROJECT_BINARY_DIR}/serializer.cpp" WITH_SOABI)
target_include_directories(serializer PRIVATE
  "${PROJECT_SOURCE_DIR}/../../include"
  "${PROJECT_SOURCE_DIR}/../../build/include")
target_link_libraries(serializer PRIVATE treelite::treelite)
set_target_properties(serializer
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)

install(TARGETS serializer DESTINATION "${PROJECT_SOURCE_DIR}")
