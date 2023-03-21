function (run_doxygen)
  find_package(Doxygen REQUIRED dot)

  configure_file(
    ${treelite_SOURCE_DIR}/docs/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc_doxygen ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generate documentation for C/C++ functions"
    VERBATIM)
endfunction (run_doxygen)
