cmake_minimum_required(VERSION 3.16)
project(example_app LANGUAGES C CXX)

if(DEFINED ENV{CONDA_PREFIX})
  set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
  message(STATUS "Detected Conda environment, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
else()
  message(STATUS "No Conda environment detected")
endif()

find_package(Treelite REQUIRED)

add_executable(cpp_example example.cc)
target_link_libraries(cpp_example PRIVATE treelite::treelite treelite::treelite_runtime)

add_executable(c_example example.c)
target_link_libraries(c_example PRIVATE treelite::treelite treelite::treelite_runtime)

set_target_properties(cpp_example PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)

set_target_properties(c_example PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED YES
)
