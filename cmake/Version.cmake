function (write_version)
    message(STATUS "Treelite VERSION: ${PROJECT_VERSION}")
    configure_file(
            ${PROJECT_SOURCE_DIR}/cmake/Python_version.in
            ${PROJECT_SOURCE_DIR}/python/treelite/VERSION @ONLY)
    configure_file(
            ${PROJECT_SOURCE_DIR}/cmake/Python_version.in
            ${PROJECT_SOURCE_DIR}/runtime/python/treelite_runtime/VERSION @ONLY)
endfunction (write_version)
