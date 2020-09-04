include(FetchContent)

FetchContent_Declare(
  dmlccore
  GIT_REPOSITORY  https://github.com/dmlc/dmlc-core
  GIT_TAG         v0.4
)
FetchContent_MakeAvailable(dmlccore)
target_compile_options(dmlc PRIVATE
                       -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
if (TARGET dmlc_unit_tests)
  target_compile_options(dmlc_unit_tests PRIVATE
                         -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
endif (TARGET dmlc_unit_tests)

FetchContent_Declare(
  fmtlib
  GIT_REPOSITORY  https://github.com/fmtlib/fmt.git
  GIT_TAG         6.2.1
)
FetchContent_MakeAvailable(fmtlib)
set_target_properties(fmt PROPERTIES EXCLUDE_FROM_ALL TRUE)

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
