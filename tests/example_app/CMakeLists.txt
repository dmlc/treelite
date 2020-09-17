cmake_minimum_required(VERSION 3.16)
project(example_app LANGUAGES C CXX)

if(DEFINED ENV{CONDA_PREFIX})
  set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
  message(STATUS "Detected Conda environment, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
else()
  message(STATUS "No Conda environment detected")
endif()

find_package(Treelite REQUIRED)

add_executable(example example.cc)
target_link_libraries(example PRIVATE treelite::treelite_static treelite::treelite_runtime_static)

set_target_properties(example PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)
