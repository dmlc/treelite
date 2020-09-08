include(FetchContent)

FetchContent_Declare(
  dmlccore
  GIT_REPOSITORY  https://github.com/dmlc/dmlc-core
  GIT_TAG         v0.4
)
FetchContent_MakeAvailable(dmlccore)

FetchContent_Declare(
  fmtlib
  GIT_REPOSITORY  https://github.com/fmtlib/fmt.git
  GIT_TAG         6.2.1
)
FetchContent_MakeAvailable(fmtlib)
set_target_properties(fmt PROPERTIES EXCLUDE_FROM_ALL TRUE)

# RapidJSON (header-only library)
add_library(rapidjson INTERFACE)
find_package(RapidJSON)
if(RapidJSON_FOUND)
  target_include_directories(rapidjson INTERFACE ${RAPIDJSON_INCLUDE_DIRS})
else()
  message(STATUS "Did not found RapidJSON in the system root. Fetching RapidJSON now...")
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
  find_package(GTest)
  if(NOT GTEST_FOUND)
    message(STATUS "Did not found Google Test in the system root. Fetching Google Test now...")
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        release-1.10.0
    )
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest ALIAS gtest)
  endif()
endif()
