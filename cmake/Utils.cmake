# Force static runtime for MSVC
function(msvc_use_static_runtime)
  if(MSVC)
    set(variables
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    foreach(variable ${variables})
      if(${variable} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${variable} "${${variable}}")
        set(${variable} "${${variable}}"  PARENT_SCOPE)
      endif()
    endforeach()
  endif()
endfunction(msvc_use_static_runtime)

# Set a default build type to release if none was specified
function(set_default_configuration_release)
  if(CMAKE_CONFIGURATION_TYPES STREQUAL "Debug;Release;MinSizeRel;RelWithDebInfo") # multiconfig generator?
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
  elseif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE )
  endif()
endfunction(set_default_configuration_release)

#=============================================================================
# Copyright 2009 Kitware, Inc.
# Copyright 2009-2011 Philip Lowman <philip@yhbt.com>
# Copyright 2008 Esben Mose Hansen, Ange Optimization ApS
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)
function(PROTOBUF_GENERATE_CPP SRCS HDRS)
  if(NOT ARGN)
    message(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif()

  # Create an include path for each file specified
  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(ABS_PATH ${ABS_FIL} PATH)
    list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
    if(${_contains_already} EQUAL -1)
      list(APPEND _protobuf_include_path -I ${ABS_PATH})
    endif()
  endforeach()

  if(DEFINED Protobuf_IMPORT_DIRS)
    foreach(DIR ${Protobuf_IMPORT_DIRS})
      get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  endif()

  set(PROTOC_DEPENDENCY ${Protobuf_PROTOC_EXECUTABLE})

  set(${SRCS})
  set(${HDRS})
  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc"
           "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h"
      COMMAND ${Protobuf_PROTOC_EXECUTABLE}
      ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} ${_protobuf_include_path} ${ABS_FIL}
      DEPENDS ${ABS_FIL} ${PROTOC_DEPENDENCY}
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM
    )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

function(PROTOBUF_GENERATE_JAVA TARGET_NAME PROTO_FILE)
  get_filename_component(ABS_FIL ${PROTO_FILE} ABSOLUTE)
  get_filename_component(ABS_PATH ${ABS_FIL} PATH)
  get_filename_component(FIL_WE ${PROTO_FILE} NAME_WE)
  list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
  if(${_contains_already} EQUAL -1)
    list(APPEND _protobuf_include_path -I ${ABS_PATH})
  endif()

  if(DEFINED Protobuf_IMPORT_DIRS)
    foreach(DIR ${Protobuf_IMPORT_DIRS})
      get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  endif()

  set(PROTOC_DEPENDENCY ${Protobuf_PROTOC_EXECUTABLE})

  add_custom_target(${TARGET_NAME} ALL
    ${Protobuf_PROTOC_EXECUTABLE} --java_out ${CMAKE_CURRENT_BINARY_DIR} ${_protobuf_include_path} ${ABS_FIL}
    DEPENDS ${ABS_FIL} ${PROTOC_DEPENDENCY}
    COMMENT "Running Java protocol buffer compiler on ${PROTO_FILE}"
    VERBATIM
  )
endfunction()

function(PROTOBUF_GENERATE_PYTHON TARGET_NAME PROTO_FILE)
  get_filename_component(ABS_FIL ${PROTO_FILE} ABSOLUTE)
  get_filename_component(ABS_PATH ${ABS_FIL} PATH)
  get_filename_component(FIL_WE ${PROTO_FILE} NAME_WE)
  list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
  if(${_contains_already} EQUAL -1)
    list(APPEND _protobuf_include_path -I ${ABS_PATH})
  endif()

  if(DEFINED Protobuf_IMPORT_DIRS)
    foreach(DIR ${Protobuf_IMPORT_DIRS})
      get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  endif()

  set(PROTOC_DEPENDENCY ${Protobuf_PROTOC_EXECUTABLE})

  add_custom_target(${TARGET_NAME} ALL
    ${Protobuf_PROTOC_EXECUTABLE} --python_out ${CMAKE_CURRENT_BINARY_DIR} ${_protobuf_include_path} ${ABS_FIL}
    DEPENDS ${ABS_FIL} ${PROTOC_DEPENDENCY}
    COMMENT "Running Python protocol buffer compiler on ${PROTO_FILE}"
    VERBATIM
  )
endfunction()
