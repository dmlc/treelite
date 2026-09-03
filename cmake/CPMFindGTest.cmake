# Description:
#
#   Find / build GTest. Prefers a system install if one is found,
#   otherwise downloads sources with CPM (https://github.com/cpm-cmake/CPM.cmake)
#   and builds GTest.
#
# Targets:
#
#   GTest::gtest
#   Gtest::mock
#
# Usage:
#
#   include(CPMFindGTest.cmake)
#

function(_treelite_find_gtest gtest_version)

  # Prefer a system-installed GTest if one is found
  find_package(GTest ${gtest_version})
  if(GTEST_FOUND)
    return()
  endif()

  # not found, use CPM to download sources and build it
  message(STATUS "GTest not found, fetching GTest via CPM.")

  # populate local CPM cache with googletest source (if not already populated), build targets
  include(${treelite_SOURCE_DIR}/cmake/CPMSetup.cmake)

  # prefer statically linking GTest, so test executables are relocatable
  CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    GIT_TAG v${gtest_version}
    VERSION ${gtest_version}
    EXCLUDE_FROM_ALL YES
    OPTIONS
      "BUILD_GMOCK ON"
      "BUILD_SHARED_LIBS OFF"
      "CMAKE_POSITION_INDEPENDENT_CODE ON"
      "INSTALL_GTEST OFF"
  )
endfunction()

_treelite_find_gtest(1.18.0)
