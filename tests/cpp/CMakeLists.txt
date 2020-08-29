add_executable(treelite_cpp_test)
set_target_properties(treelite_cpp_test
    PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)
target_link_libraries(treelite_cpp_test
    PRIVATE objtreelite objtreelite_runtime objtreelite_common GTest::GTest)
set_output_directory(treelite_cpp_test ${PROJECT_BINARY_DIR})

if(TEST_COVERAGE)
  if(MSVC)
    message(FATAL_ERROR "Test coverage not available on Windows")
  endif()
  target_compile_options(treelite_cpp_test PUBLIC -g -O0 --coverage)
  target_link_options(treelite_cpp_test PUBLIC --coverage)
endif()

target_sources(treelite_cpp_test
  PRIVATE  test_main.cc
           #test_serializer.cc
)

msvc_use_static_runtime()
