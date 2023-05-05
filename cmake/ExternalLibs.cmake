include(FetchContent)
include(cmake/FetchContentMakeAvailable.cmake)

# fmtlib
find_package(fmt)
if(fmt_FOUND)
  get_target_property(fmt_loc fmt::fmt INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "Found fmtlib at ${fmt_loc}")
  set(FMTLIB_FROM_SYSTEM_ROOT TRUE)
else()
  message(STATUS "Did not find fmtlib in the system root. Fetching fmtlib now...")
  FetchContent_Declare(
    fmtlib
    GIT_REPOSITORY  https://github.com/fmtlib/fmt.git
    GIT_TAG         7.0.3
  )
  FetchContent_MakeAvailable(fmtlib)
  set_target_properties(fmt PROPERTIES EXCLUDE_FROM_ALL TRUE)
  set(FMTLIB_FROM_SYSTEM_ROOT FALSE)
endif()

# RapidJSON (header-only library)
add_library(rapidjson INTERFACE)
find_package(RapidJSON)
if(RapidJSON_FOUND)
  target_include_directories(rapidjson INTERFACE ${RAPIDJSON_INCLUDE_DIRS})
else()
  message(STATUS "Did not find RapidJSON in the system root. Fetching RapidJSON now...")
  FetchContent_Declare(
    RapidJSON
    GIT_REPOSITORY      https://github.com/Tencent/rapidjson
    GIT_TAG             v1.1.0
  )
  FetchContent_Populate(RapidJSON)
  message(STATUS "RapidJSON was downloaded at ${rapidjson_SOURCE_DIR}.")
  target_include_directories(rapidjson INTERFACE $<BUILD_INTERFACE:${rapidjson_SOURCE_DIR}/include>)
endif()
add_library(RapidJSON::rapidjson ALIAS rapidjson)

# Google C++ tests
if(BUILD_CPP_TEST)
  find_package(GTest 1.11.0 CONFIG)
  if(NOT GTEST_FOUND)
    message(STATUS "Did not find Google Test in the system root. Fetching Google Test now...")
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        release-1.11.0
    )
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest ALIAS gtest)
    add_library(GTest::gmock ALIAS gmock)
    if(IS_DIRECTORY "${googletest_SOURCE_DIR}")
      # Do not install gtest
      set_property(DIRECTORY ${googletest_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL YES)
    endif()
  endif()
endif()
