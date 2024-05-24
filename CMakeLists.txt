cmake_policy(SET CMP0025 NEW)
cmake_policy(SET CMP0091 NEW)
set(CMAKE_FIND_NO_INSTALL_PREFIX TRUE FORCE)
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(treelite LANGUAGES CXX C VERSION 4.3.0)

# Check compiler versions
# Use latest compilers to ensure that std::filesystem is available
if(MSVC)
  if(MSVC_VERSION LESS 1930)
    message(FATAL_ERROR "Need Visual Studio 2022 or newer to build Treelite")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.1")
    message(FATAL_ERROR "Need GCC 8.1 or newer to build Treelite")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0")
    message(FATAL_ERROR "Need Xcode 11.0 (AppleClang 11.0) or newer to build Treelite")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    message(FATAL_ERROR "Need Clang 9.0 or newer to build Treelite")
  endif()
endif()

# Options useful for downstream projects
option(Treelite_BUILD_STATIC_LIBS "Build static libs, in addition to dynamic libs" OFF)
option(Treelite_USE_DYNAMIC_MSVC_RUNTIME
    "Whether to dynamically link MSVC runtime. ON: Use dynamic link (-MD), OFF: Use static link (-MT)"
    OFF)

# Developer options
option(TEST_COVERAGE "C++ test coverage" OFF)
option(USE_OPENMP "Use OpenMP" ON)
option(BUILD_DOXYGEN "Build documentation for C/C++ functions using Doxygen." OFF)
option(BUILD_CPP_TEST "Build C++ tests" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(HIDE_CXX_SYMBOLS "Hide all C++ symbols. Useful when building Pip package" OFF)
option(ENABLE_ALL_WARNINGS "Enable all compiler warnings. Only effective for GCC/Clang" OFF)
option(USE_SANITIZER "Use sanitizer flags" OFF)
SET(ENABLED_SANITIZERS "address" "leak" "undefined" CACHE STRING
    "Semicolon separated list of sanitizer names. E.g 'address;leak'.")

if(MSVC)
  if(Treelite_USE_DYNAMIC_MSVC_RUNTIME)
    message(STATUS "Using dynamically linked MSVC runtime...")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  else()
    message(STATUS "Using statically linked MSVC runtime...")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
endif()

if(USE_SANITIZER)
  include(cmake/Sanitizer.cmake)
  enable_sanitizers("${ENABLED_SANITIZERS}")
endif()

if(ENABLE_ALL_WARNINGS)
  if((NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang") AND (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(SEND_ERROR "ENABLE_ALL_WARNINGS is only available for Clang and GCC.")
  endif()
endif()

if(BUILD_DOXYGEN)
  include(cmake/Doxygen.cmake)
  run_doxygen()
endif()

# When installing dependencies, use Conda environment if available
if(DETECT_CONDA_ENV)
  if(DEFINED ENV{CONDA_PREFIX})
    string(REPLACE "\\" "/" CONDA_PREFIX "$ENV{CONDA_PREFIX}")
    set(CMAKE_PREFIX_PATH "${CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
    message(STATUS "Detected Conda environment, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR USING_CONDA_INSTALL_PREFIX)
      message(STATUS "No CMAKE_INSTALL_PREFIX argument detected, setting to: ${CONDA_PREFIX}")
      set(CMAKE_INSTALL_PREFIX "${CONDA_PREFIX}")
      set(USING_CONDA_INSTALL_PREFIX ON CACHE BOOL "Installing into Conda prefix")
    endif()
  else()
    message(STATUS "No Conda environment detected")
  endif()
endif()

include(cmake/ExternalLibs.cmake)
include(cmake/Utils.cmake)
include(cmake/Version.cmake)

add_subdirectory(src)

if(BUILD_CPP_TEST)
  add_subdirectory(tests/cpp)
endif()

add_library(treelite SHARED)
target_link_libraries(treelite PRIVATE objtreelite)

set(TREELITE_TARGETS treelite)

if(Treelite_BUILD_STATIC_LIBS)
  add_library(treelite_static STATIC)
  target_link_libraries(treelite_static PRIVATE objtreelite)
  list(APPEND TREELITE_TARGETS treelite_static)
endif()

foreach(lib ${TREELITE_TARGETS})
  set_output_directory(${lib} ${PROJECT_BINARY_DIR})
endforeach()

# Export install targets
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Include CPack only if the current project is top level.
  include(CPack)
endif()
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(INSTALL_TARGETS ${TREELITE_TARGETS} objtreelite rapidjson mdspan)
install(TARGETS ${INSTALL_TARGETS}
    EXPORT TreeliteTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION include)
install(DIRECTORY include/treelite "${PROJECT_BINARY_DIR}/include/treelite"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT TreeliteTargets
    FILE TreeliteTargets.cmake
    NAMESPACE treelite::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/treelite")
configure_package_config_file(
    cmake/TreeliteConfig.cmake.in
    "${PROJECT_BINARY_DIR}/cmake/TreeliteConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/treelite")
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/cmake/TreeliteConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(FILES
    "${PROJECT_BINARY_DIR}/cmake/TreeliteConfig.cmake"
    "${PROJECT_BINARY_DIR}/cmake/TreeliteConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/treelite")

write_version()
set_default_configuration_release()
