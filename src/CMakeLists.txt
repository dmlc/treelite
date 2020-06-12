# Object library

add_library(objtreelite OBJECT)
target_include_directories(objtreelite PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(objtreelite PUBLIC dmlc PRIVATE fmt::fmt-header-only)
target_compile_definitions(objtreelite PRIVATE -DDMLC_LOG_CUSTOMIZE)

add_library(objtreelite_runtime OBJECT)
target_include_directories(objtreelite_runtime PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(objtreelite_runtime PUBLIC dmlc)
# -ldl for UNIX-like systems
if (UNIX)
  target_link_libraries(objtreelite_runtime PUBLIC dl)
endif (UNIX)
target_compile_definitions(objtreelite_runtime PRIVATE -DDMLC_LOG_CUSTOMIZE)

if(MSVC)
  target_compile_options(objtreelite PRIVATE /MP)
  target_compile_options(objtreelite_runtime PRIVATE /MP)
  target_compile_definitions(objtreelite PRIVATE -DNOMINMAX)
  target_compile_definitions(objtreelite_runtime PRIVATE -DNOMINMAX)
else()
  target_compile_options(objtreelite PRIVATE -funroll-loops)
  target_compile_options(objtreelite_runtime PRIVATE -funroll-loops)
endif()

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(objtreelite PUBLIC OpenMP::OpenMP_CXX)
  target_link_libraries(objtreelite_runtime PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(objtreelite PRIVATE -DTREELITE_OPENMP_SUPPORT)
  target_compile_definitions(objtreelite_runtime PRIVATE -DTREELITE_OPENMP_SUPPORT)
else()
  message(STATUS "Disabling OpenMP")
endif()

if(TEST_COVERAGE)
  if(MSVC)
    message(FATAL_ERROR "Test coverage not available on Windows")
  endif()
  target_compile_options(objtreelite PUBLIC -g -O0 --coverage)
  target_compile_options(objtreelite_runtime PUBLIC -g -O0 --coverage)
  target_link_options(objtreelite PUBLIC --coverage)
  target_link_options(objtreelite_runtime PUBLIC --coverage)
endif()

set_target_properties(objtreelite objtreelite_runtime
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON)

# Use Protobuf optionally
if(ENABLE_PROTOBUF)
  target_include_directories(objtreelite PRIVATE ${Protobuf_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
  target_compile_definitions(objtreelite PRIVATE -DTREELITE_PROTOBUF_SUPPORT)
  target_link_libraries(objtreelite PRIVATE ${Protobuf_LIBRARIES})
  PROTOBUF_GENERATE_CPP(TREE_PROTO_SRCS TREE_PROTO_HDRS tree.proto)
  PROTOBUF_GENERATE_JAVA(tree_proto_java tree.proto)
  PROTOBUF_GENERATE_PYTHON(tree_proto_python tree.proto)
else()
  set(TREE_PROTO_SRCS "")
  set(TREE_PROTO_HDRS "")
endif()

target_sources(objtreelite
  PRIVATE   c_api/c_api.cc
            c_api/c_api_common.cc
            c_api/c_api_error.cc
            c_api/c_api_error.h
            compiler/ast/ast.h
            compiler/ast/build.cc
            compiler/ast/builder.h
            compiler/ast/dump.cc
            compiler/ast/fold_code.cc
            compiler/ast/is_categorical_array.cc
            compiler/ast/load_data_counts.cc
            compiler/ast/quantize.cc
            compiler/ast/split.cc
            compiler/common/categorical_bitmap.h
            compiler/common/code_folding_util.h
            compiler/common/format_util.h
            compiler/elf/elf_formatter.cc
            compiler/elf/elf_formatter.h
            compiler/native/code_folder_template.h
            compiler/native/header_template.h
            compiler/native/main_template.h
            compiler/native/pred_transform.h
            compiler/native/qnode_template.h
            compiler/ast_native.cc
            compiler/compiler.cc
            compiler/failsafe.cc
            compiler/pred_transform.cc
            compiler/pred_transform.h
            frontend/builder.cc
            frontend/lightgbm.cc
            frontend/xgboost.cc
            frontend/protobuf.cc
            annotator.cc
            data.cc
            filesystem.cc
            logging.cc
            optable.cc
            reference_serializer.cc
            ${TREE_PROTO_SRCS}
            ${TREE_PROTO_HDRS}
  PUBLIC    ${PROJECT_SOURCE_DIR}/include/treelite/annotator.h
            ${PROJECT_SOURCE_DIR}/include/treelite/base.h
            ${PROJECT_SOURCE_DIR}/include/treelite/c_api.h
            ${PROJECT_SOURCE_DIR}/include/treelite/c_api_common.h
            ${PROJECT_SOURCE_DIR}/include/treelite/compiler.h
            ${PROJECT_SOURCE_DIR}/include/treelite/compiler_param.h
            ${PROJECT_SOURCE_DIR}/include/treelite/data.h
            ${PROJECT_SOURCE_DIR}/include/treelite/filesystem.h
            ${PROJECT_SOURCE_DIR}/include/treelite/frontend.h
            ${PROJECT_SOURCE_DIR}/include/treelite/logging.h
            ${PROJECT_SOURCE_DIR}/include/treelite/math.h
            ${PROJECT_SOURCE_DIR}/include/treelite/omp.h
            ${PROJECT_SOURCE_DIR}/include/treelite/tree.h
            ${PROJECT_SOURCE_DIR}/include/treelite/tree_impl.h
)

target_sources(objtreelite_runtime
  PRIVATE   c_api/c_api_common.cc
            c_api/c_api_error.cc
            c_api/c_api_error.h
            c_api/c_api_runtime.cc
            predictor/thread_pool/spsc_queue.h
            predictor/thread_pool/thread_pool.h
            predictor/predictor.cc
            logging.cc
  PUBLIC    ${PROJECT_SOURCE_DIR}/include/treelite/c_api_common.h
            ${PROJECT_SOURCE_DIR}/include/treelite/c_api_runtime.h
            ${PROJECT_SOURCE_DIR}/include/treelite/entry.h
            ${PROJECT_SOURCE_DIR}/include/treelite/logging.h
            ${PROJECT_SOURCE_DIR}/include/treelite/math.h
            ${PROJECT_SOURCE_DIR}/include/treelite/predictor.h
)
