add_executable(treelite_cpp_test)
set_target_properties(treelite_cpp_test
    PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON)
target_link_libraries(treelite_cpp_test
    PRIVATE objtreelite objtreelite_runtime objtreelite_common GTest::GTest)
set_output_directory(treelite_cpp_test ${PROJECT_BINARY_DIR})

target_sources(treelite_cpp_test
  PRIVATE  test_main.cc
)
