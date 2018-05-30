function(copy_files FILENAME)
  file(READ ${FILENAME} CONTENTS)
  # Convert file contents into a CMake list (where each element in the list
  # is one line of the file)
  string(REGEX REPLACE ";" "\\\\;" CONTENTS "${CONTENTS}")
  string(REGEX REPLACE "\n" ";" CONTENTS "${CONTENTS}")
  # Ignore all lines starting with # (comments)
  set(FILELIST "")
  foreach(f ${CONTENTS})
    if(NOT "${f}" MATCHES "^#.*")
      # Split each non-comment line with delimier '->' 
      string(REGEX REPLACE "[ ]*->[ ]*" ";" RESULT "${f}")
      list(GET RESULT 0 SRC)
      list(GET RESULT 1 DEST)
      get_filename_component(FILENAME "${SRC}" NAME)
      file(COPY "${PROJECT_SOURCE_DIR}/${SRC}" DESTINATION "${PROJECT_SOURCE_DIR}/${DEST}")
      list(APPEND FILELIST "${PROJECT_SOURCE_DIR}/${DEST}/${FILENAME}")
    endif()
  endforeach()
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${FILELIST}")
endfunction(copy_files)

# Automatically set source group based on folder
function(auto_source_group SOURCES)

  foreach(FILE ${SOURCES})
    get_filename_component(PARENT_DIR "${FILE}" PATH)

    # skip src or include and changes /'s to \\'s
    string(REPLACE "${CMAKE_CURRENT_LIST_DIR}" "" GROUP "${PARENT_DIR}")
    string(REPLACE "/" "\\\\" GROUP "${GROUP}")
    string(REGEX REPLACE "^\\\\" "" GROUP "${GROUP}")

    source_group("${GROUP}" FILES "${FILE}")
  endforeach()
endfunction(auto_source_group)

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

# Set output directory of target, ignoring debug or release
function(set_output_directory target dir)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${dir}              # for executable
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${dir}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${dir}
    LIBRARY_OUTPUT_DIRECTORY ${dir}              # for shared library
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${dir}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${dir}
    ARCHIVE_OUTPUT_DIRECTORY ${dir}              # for static library
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${dir}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${dir}
  )
endfunction(set_output_directory)

# Set a default build type to release if none was specified
function(set_default_configuration_release)
  if(CMAKE_CONFIGURATION_TYPES STREQUAL "Debug;Release;MinSizeRel;RelWithDebInfo") # multiconfig generator?
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)
  elseif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE )
  endif()
endfunction(set_default_configuration_release)

function(format_gencode_flags flags out)
  foreach(ver ${flags})
    set(${out} "${${out}}-gencode arch=compute_${ver},code=sm_${ver};")
  endforeach()
  set(${out} "${${out}}" PARENT_SCOPE)
endfunction(format_gencode_flags flags)
