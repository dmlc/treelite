cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

# scikit-build-core entry point for the Treelite Python wheel

project(treelite_python LANGUAGES NONE)

# resolve _TL_CPP_SRC
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp_src/CMakeLists.txt")
    set(_TL_CPP_SRC "${CMAKE_CURRENT_SOURCE_DIR}/cpp_src")
else()
    set(_TL_CPP_SRC "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()
get_filename_component(_TL_CPP_SRC "${_TL_CPP_SRC}" ABSOLUTE)

if(WIN32)
    set(_TL_LIBNAME "treelite.dll")
elseif(APPLE)
    set(_TL_LIBNAME "libtreelite.dylib")
else()
    set(_TL_LIBNAME "libtreelite.so")
endif()

option(TREELITE_USE_SYSTEM_LIBTREELITE
        "Don't bundle pre-existing libtreelite into the wheel; the runtime loader will find it on the host"
         OFF)
set(TREELITE_SYSTEM_LIBTREELITE_DIR "" CACHE PATH
        "Directory where pre-existing libtreelite library exists")
# ensure valid configuration for pre-existing lib
if(TREELITE_SYSTEM_LIBTREELITE_DIR AND NOT TREELITE_USE_SYSTEM_LIBTREELITE)
    message(FATAL_ERROR
        "'TREELITE_SYSTEM_LIBTREELITE_DIR' was set but 'TREELITE_USE_SYSTEM_LIBTREELITE' was not;
        'TREELITE_USE_SYSTEM_LIBTREELITE' must be set if using 'TREELITE_SYSTEM_LIBTREELITE_DIR'")
endif()
set(TREELITE_PREBUILT_LIB_DIR "${_TL_CPP_SRC}/build" CACHE PATH
    "Directory holding prebuild libtreelite to place in wheel")

if(TREELITE_USE_SYSTEM_LIBTREELITE)
    # if using the systems libtreelite library at a specified path then configure the path at build time
    if(TREELITE_SYSTEM_LIBTREELITE_DIR)
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/path_config.py.in"
            "${CMAKE_CURRENT_BINARY_DIR}/treelite/path_config.py"
            @ONLY
        )
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/treelite/path_config.py"
                DESTINATION treelite
                COMPONENT TreelitePython)
    endif()
    message(STATUS
        "treelite-python: TREELITE_USE_SYSTEM_LIBTREELITE=ON, not bundling libtreelite; "
        "runtime will use the specified path if set, else fall back to sys.base_prefix/lib"
    )
    return()
endif()

if (EXISTS "${TREELITE_PREBUILT_LIB_DIR}/${_TL_LIBNAME}")
    # lib pre-existing, rename resolved file path and install
    file(REAL_PATH "${TREELITE_PREBUILT_LIB_DIR}/${_TL_LIBNAME}" _TL_LIB_RESOLVED)
    message(STATUS
        "treelite-python: bundling pre-built ${_TL_LIB_RESOLVED} into the wheel "
        "as treelite/lib/${_TL_LIBNAME}")
    install(FILES "${_TL_LIB_RESOLVED}"
            DESTINATION "treelite/lib"
            RENAME "${_TL_LIBNAME}"
            COMPONENT "TreelitePython")
else()
    # lib doesn't exist, build from source
    message(STATUS "treelite-python: building libtreelite from sources at ${_TL_CPP_SRC}")

    set(_TREELITE_FOR_PYTHON_WHEEL ON CACHE BOOL "" FORCE)
    add_subdirectory("${_TL_CPP_SRC}" treelite_cpp_build)
    install(FILES $<TARGET_FILE:treelite>
            DESTINATION treelite/lib
            RENAME "${_TL_LIBNAME}"
            COMPONENT TreelitePython)
endif()
