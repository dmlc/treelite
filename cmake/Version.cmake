function (write_version)
    message(STATUS "Treelite VERSION: ${PROJECT_VERSION}")
    configure_file(
            ${PROJECT_SOURCE_DIR}/cmake/Python_version.in
            ${PROJECT_SOURCE_DIR}/python/treelite/VERSION @ONLY)
    configure_file(
            ${PROJECT_SOURCE_DIR}/cmake/version.h.in
            include/treelite/version.h)
endfunction (write_version)
