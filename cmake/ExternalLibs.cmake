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

# Protobuf
if(ENABLE_PROTOBUF)
  set(Protobuf_USE_STATIC_LIBS ON)
  find_package(Protobuf)
  if(NOT Protobuf_FOUND)
    message(STATUS "Did not found Protobuf in the system root. Fetching Protobuf now...")
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests for protobuf" FORCE)
    set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "enable shared libs for protobuf" FORCE)
    FetchContent_Populate(protobuf
      GIT_REPOSITORY  https://github.com/protocolbuffers/protobuf
      GIT_TAG         v3.12.3
      SOURCE_SUBDIR   cmake
    )
    add_subdirectory(${protobuf_SOURCE_DIR}/cmake ${protobuf_BINARY_DIR})
    set(Protobuf_PROTOC_EXECUTABLE protobuf::protoc)
    set(Protobuf_INCLUDE_DIRS ${protobuf_SOURCE_DIR}/src)
    set(Protobuf_LIBRARIES protobuf::libprotobuf)
    set_target_properties(libprotobuf PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()
else()
  set(Protobuf_LIBRARIES "")
endif()

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
