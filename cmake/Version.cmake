function (write_version)
  message(STATUS "Treelite VERSION: ${PROJECT_VERSION}")
  configure_file(
      ${PROJECT_SOURCE_DIR}/cmake/version.h.in
      include/treelite/version.h)
endfunction (write_version)
