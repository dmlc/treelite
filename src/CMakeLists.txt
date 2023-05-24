# Object library

add_library(objtreelite OBJECT)
target_link_libraries(objtreelite PRIVATE fmt::fmt-header-only RapidJSON::rapidjson)

if(USE_OPENMP)
  if(APPLE)
    # Require CMake 3.16+ on Mac OSX, as previous versions of CMake had trouble locating
    # OpenMP on Mac. See https://github.com/dmlc/xgboost/pull/5146#issuecomment-568312706
    cmake_minimum_required(VERSION 3.16)
    find_package(OpenMP)
    if (NOT OpenMP_FOUND)
      # Try again with extra path info; required for libomp 15+ from Homebrew
      message(STATUS "OpenMP not found; attempting to locate libomp from Homebrew...")
      execute_process(COMMAND brew --prefix libomp
          OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
          OUTPUT_STRIP_TRAILING_WHITESPACE)
      set(OpenMP_C_FLAGS
          "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
      set(OpenMP_CXX_FLAGS
          "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
      set(OpenMP_C_LIB_NAMES omp)
      set(OpenMP_CXX_LIB_NAMES omp)
      set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
      find_package(OpenMP REQUIRED)
    endif()
  else()
    find_package(OpenMP REQUIRED)
  endif()
else()
  message(STATUS "Disabling OpenMP")
endif()

if(ENABLE_ALL_WARNINGS)
  target_compile_options(objtreelite PRIVATE -Wall -Wextra)
endif()

target_include_directories(objtreelite PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)
if(MSVC)
  target_compile_options(objtreelite PRIVATE /MP)
  target_compile_definitions(objtreelite PRIVATE -DNOMINMAX)
  target_compile_options(objtreelite PRIVATE /utf-8 -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
else()
  target_compile_options(objtreelite PRIVATE -funroll-loops)
endif()
if(USE_OPENMP)
  target_link_libraries(objtreelite PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(objtreelite PUBLIC -DTREELITE_OPENMP_SUPPORT)
endif()
if(TEST_COVERAGE)
  if(MSVC)
    message(FATAL_ERROR "Test coverage not available on Windows")
  endif()
  target_compile_options(objtreelite PUBLIC -g -O0 --coverage)
  target_link_options(objtreelite PUBLIC --coverage)
endif()

set_target_properties(objtreelite
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)

if(HIDE_CXX_SYMBOLS)
  set_target_properties(objtreelite
    PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden)
endif(HIDE_CXX_SYMBOLS)

target_sources(objtreelite
    PRIVATE
    c_api/c_api_error.cc
    c_api/c_api.cc
    frontend/builder.cc
    frontend/json_importer.cc
    frontend/lightgbm.cc
    frontend/sklearn.cc
    frontend/xgboost.cc
    frontend/xgboost_json.cc
    frontend/xgboost_util.cc
    gtil/predict.cc
    gtil/config.cc
    gtil/pred_transform.h
    gtil/pred_transform.cc
    threading_utils/parallel_for.h
    data.cc
    serializer.cc
    json_serializer.cc
    model_concat.cc
    logging.cc
    typeinfo.cc
    ${PROJECT_SOURCE_DIR}/include/treelite/base.h
    ${PROJECT_SOURCE_DIR}/include/treelite/c_api.h
    ${PROJECT_SOURCE_DIR}/include/treelite/c_api_error.h
    ${PROJECT_SOURCE_DIR}/include/treelite/contiguous_array.h
    ${PROJECT_SOURCE_DIR}/include/treelite/data.h
    ${PROJECT_SOURCE_DIR}/include/treelite/frontend.h
    ${PROJECT_SOURCE_DIR}/include/treelite/frontend_impl.h
    ${PROJECT_SOURCE_DIR}/include/treelite/gtil.h
    ${PROJECT_SOURCE_DIR}/include/treelite/omp.h
    ${PROJECT_SOURCE_DIR}/include/treelite/task_type.h
    ${PROJECT_SOURCE_DIR}/include/treelite/thread_local.h
    ${PROJECT_SOURCE_DIR}/include/treelite/tree.h
    ${PROJECT_SOURCE_DIR}/include/treelite/tree_impl.h
    ${PROJECT_SOURCE_DIR}/include/treelite/error.h
    ${PROJECT_SOURCE_DIR}/include/treelite/logging.h
    ${PROJECT_SOURCE_DIR}/include/treelite/math.h
    ${PROJECT_SOURCE_DIR}/include/treelite/omp_exception.h
    ${PROJECT_SOURCE_DIR}/include/treelite/typeinfo.h
    ${PROJECT_SOURCE_DIR}/include/treelite/detail/contiguous_array.h
)

# In MSVC solution, group sources according to directories
file(GLOB_RECURSE ALL_SOURCES *.cc *.h)
auto_source_group("${ALL_SOURCES}")
