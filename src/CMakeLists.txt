# Object library

add_library(objtreelite OBJECT)
target_link_libraries(objtreelite PRIVATE RapidJSON::rapidjson std::mdspan)

if(USE_OPENMP)
  if(APPLE)
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
target_compile_features(objtreelite PUBLIC cxx_std_17)

if(HIDE_CXX_SYMBOLS)
  set_target_properties(objtreelite
    PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden)
endif(HIDE_CXX_SYMBOLS)

target_sources(objtreelite
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include/treelite/c_api.h
    ${PROJECT_SOURCE_DIR}/include/treelite/c_api_error.h
    ${PROJECT_SOURCE_DIR}/include/treelite/contiguous_array.h
    ${PROJECT_SOURCE_DIR}/include/treelite/error.h
    ${PROJECT_SOURCE_DIR}/include/treelite/gtil.h
    ${PROJECT_SOURCE_DIR}/include/treelite/logging.h
    ${PROJECT_SOURCE_DIR}/include/treelite/model_builder.h
    ${PROJECT_SOURCE_DIR}/include/treelite/model_loader.h
    ${PROJECT_SOURCE_DIR}/include/treelite/pybuffer_frame.h
    ${PROJECT_SOURCE_DIR}/include/treelite/thread_local.h
    ${PROJECT_SOURCE_DIR}/include/treelite/tree.h
    ${PROJECT_SOURCE_DIR}/include/treelite/detail/contiguous_array.h
    ${PROJECT_SOURCE_DIR}/include/treelite/detail/file_utils.h
    ${PROJECT_SOURCE_DIR}/include/treelite/detail/omp_exception.h
    ${PROJECT_SOURCE_DIR}/include/treelite/detail/serializer.h
    ${PROJECT_SOURCE_DIR}/include/treelite/detail/serializer_mixins.h
    ${PROJECT_SOURCE_DIR}/include/treelite/detail/threading_utils.h
    ${PROJECT_SOURCE_DIR}/include/treelite/detail/tree.h
    ${PROJECT_SOURCE_DIR}/include/treelite/enum/operator.h
    ${PROJECT_SOURCE_DIR}/include/treelite/enum/task_type.h
    ${PROJECT_SOURCE_DIR}/include/treelite/enum/tree_node_type.h
    ${PROJECT_SOURCE_DIR}/include/treelite/enum/typeinfo.h
    json_serializer.cc
    logging.cc
    model_concat.cc
    serializer.cc
    c_api/c_api_error.cc
    c_api/c_api_utils.h
    c_api/gtil.cc
    c_api/logging.cc
    c_api/model.cc
    c_api/model_builder.cc
    c_api/model_loader.cc
    c_api/serializer.cc
    c_api/sklearn.cc
    enum/operator.cc
    enum/task_type.cc
    enum/tree_node_type.cc
    enum/typeinfo.cc
    gtil/config.cc
    gtil/output_shape.cc
    gtil/postprocessor.cc
    gtil/postprocessor.h
    gtil/predict.cc
    model_builder/metadata.cc
    model_builder/model_builder.cc
    model_builder/detail/json_parsing.h
    model_loader/lightgbm.cc
    model_loader/sklearn.cc
    model_loader/xgboost_json.cc
    model_loader/xgboost_legacy.cc
    model_loader/detail/lightgbm.h
    model_loader/detail/string_utils.h
    model_loader/detail/xgboost.cc
    model_loader/detail/xgboost.h
    model_loader/detail/xgboost_json.h
)
