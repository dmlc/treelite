cmake_minimum_required(VERSION 3.1)
project(mushroom_example)

# Compiler flags
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(mushroom SHARED mushroom.c)
