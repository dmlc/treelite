
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
		RUNTIME_OUTPUT_DIRECTORY ${dir}
		RUNTIME_OUTPUT_DIRECTORY_DEBUG ${dir}
		RUNTIME_OUTPUT_DIRECTORY_RELEASE ${dir}
		LIBRARY_OUTPUT_DIRECTORY ${dir}
		LIBRARY_OUTPUT_DIRECTORY_DEBUG ${dir}
		LIBRARY_OUTPUT_DIRECTORY_RELEASE ${dir}
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

# if MSVC, locate vcvarsall.bat that sets all necessary environment variables 
function(msvc_locate_vcvarsall_bat)
  if(MSVC)
    if(DEFINED ENV{VCVARSALL_PATH})
      if (NOT EXISTS $ENV{VCVARSALL_PATH})
        message(FATAL_ERROR "vcvarsall.bat not found at $ENV{VCVARSALL_PATH}")
      endif()
      set(VCVARSALL_BAT $ENV{VCVARSALL_PATH})
    else()
      get_filename_component(CMAKE_CXX_COMPILER_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
      string(REGEX MATCH "C:/Program Files \\(x86\\)/Microsoft Visual Studio[ 0-9\\.]*" MSVC_PREFIX ${CMAKE_CXX_COMPILER_DIR})
      file(GLOB_RECURSE VCVARSALL_BAT
        ${MSVC_PREFIX}/*/vcvarsall.bat
      )
      list(LENGTH VCVARSALL_BAT NUM_VCVARSALL_RESULT)
      if (${NUM_VCVARSALL_RESULT} LESS 1)
        message(FATAL_ERROR "vcvarsall.bat not found; please set environment variable VCVARSALL_PATH and try again.")
      elseif(${NUM_VCVARSALL_RESULT} GREATER 1)
        list(GET VCVARSALL_BAT 0 TMP)
        set(VCVARSALL_BAT ${TMP})
      endif()
    endif()
    message("-- Found vcvarsall.bat: ${VCVARSALL_BAT}")
    set(VCVARSALL_BAT ${VCVARSALL_BAT} PARENT_SCOPE)
  else()
    set(VCVARSALL_BAT "" PARENT_SCOPE)
  endif()
endfunction(msvc_locate_vcvarsall_bat)