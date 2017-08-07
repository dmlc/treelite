/*!
 * Copyright (c) 2017 by Contributors
 * \file c_api.h
 * \author Philip Cho
 * \brief C API of tree-lite, used for interfacing with other languages
 */
#ifndef TREELITE_C_API_H_
#define TREELITE_C_API_H_

#ifdef __cplusplus
#define TREELITE_EXTERN_C extern "C"
#include <cstdio>
#include <cstdint>
#else
#define TREELITE_EXTERN_C
#include <stdio.h>
#include <stdint.h>
#endif

/* Note: Make sure to use slash-asterisk form of comments in this file
   (like this one). Do not use double-slash (//). */

/* special symbols for DLL library on Windows */
#if defined(_MSC_VER) || defined(_WIN32)
#define TREELITE_DLL TREELITE_EXTERN_C __declspec(dllexport)
#else
#define TREELITE_DLL TREELITE_EXTERN_C
#endif

/* opaque handles */
typedef void* ModelHandle;
typedef void* ModelBufferHandle;
typedef void* AnnotationHandle;
typedef void* CompilerHandle;
typedef void* PredictorHandle;
typedef void* DMatrixHandle;

/* display last error; can be called by different threads */
TREELITE_DLL const char* TreeliteGetLastError();

/* data matrix interface */
TREELITE_DLL int TreeliteDMatrixCreateFromFile(const char* path,
                                               int nthread,
                                               int verbose,
                                               DMatrixHandle* out);
TREELITE_DLL int TreeliteDMatrixCreateFromCSR(const float* data,
                                              const unsigned* col_ind,
                                              const size_t* row_ptr,
                                              size_t num_row,
                                              size_t num_col,
                                              size_t nnz,
                                              int nthread,
                                              int verbose,
                                              DMatrixHandle* out);
TREELITE_DLL int TreeliteDMatrixCreateFromCSC(const float* data,
                                              const unsigned* row_ind,
                                              const size_t* col_ptr,
                                              size_t num_row,
                                              size_t num_col,
                                              size_t nnz,
                                              int nthread,
                                              int verbose,
                                              DMatrixHandle* out);
TREELITE_DLL int TreeliteDMatrixCreateFromMat(const float* data,
                                              size_t num_row,
                                              size_t num_col,
                                              float missing_value,
                                              int nthread,
                                              int verbose,
                                              DMatrixHandle* out);
TREELITE_DLL int TreeliteDMatrixFree(DMatrixHandle handle);

/* branch annotation */
TREELITE_DLL int TreeliteAnnotateBranch(ModelHandle model,
                                        DMatrixHandle dmat,
                                        int nthread,
                                        int verbose,
                                        AnnotationHandle* out);
TREELITE_DLL int TreeliteAnnotationSave(AnnotationHandle handle,
                                        const char* path);
TREELITE_DLL int TreeliteAnnotationLoad(const char* path,
                                        AnnotationHandle* out);
TREELITE_DLL int TreeliteAnnotationFree(AnnotationHandle handle);

/* compiler interface */
TREELITE_DLL int TreeliteCompilerCreate(const char* name,
                                        CompilerHandle* out);
TREELITE_DLL int TreeliteCompilerSetParam(CompilerHandle handle,
                                          const char* name,
                                          const char* value);
TREELITE_DLL int TreeliteCompilerGenerateCode(CompilerHandle compiler,
                                              ModelHandle model,
                                              int verbose,
                                              const char* path);
TREELITE_DLL int TreeliteCompilerFree(CompilerHandle handle);

/* predictor interface */
TREELITE_DLL int TreelitePredictorLoad(const char* library_path,
                                       PredictorHandle* out);
TREELITE_DLL int TreelitePredictorPredict(PredictorHandle handle,
                                          DMatrixHandle dmat,
                                          int nthread,
                                          int verbose,
                                          const float** out_result);
TREELITE_DLL int TreelitePredictorFree(PredictorHandle handle);

/* file interface: read from the disk */
TREELITE_DLL int TreeliteLoadLightGBMModel(const char* filename,
                                           ModelHandle* out);
TREELITE_DLL int TreeliteLoadXGBoostModel(const char* filename,
                                          ModelHandle* out);
TREELITE_DLL int TreeliteLoadProtobufModel(const char* filename,
                                           ModelHandle* out);
TREELITE_DLL int TreeliteFreeModel(ModelHandle handle);

/* interactive interface: build trees incrementally */
TREELITE_DLL int TreeliteCreateModelBuffer(int num_features,
                                           ModelBufferHandle* out);
TREELITE_DLL int TreeliteDeleteModelBuffer(ModelBufferHandle handle);
TREELITE_DLL int TreeliteCreateTree(ModelBufferHandle handle,
                                    int index);
TREELITE_DLL int TreeliteDeleteTree(ModelBufferHandle handle,
                                    int index);
TREELITE_DLL int TreeliteCreateNode(ModelBufferHandle handle,
                                    int tree_index, int node_key);
TREELITE_DLL int TreeliteDeleteNode(ModelBufferHandle handle,
                                    int tree_index, int node_key);
TREELITE_DLL int TreeliteSetRootNode(ModelBufferHandle handle,
                                     int tree_index, int node_key);
TREELITE_DLL int TreeliteSetTestNode(ModelBufferHandle handle,
                                     int tree_index, int node_key,
                                     unsigned feature_id, const char* opname,
                                     float threshold, int default_left,
                                     int left_child_key, int right_child_key);
TREELITE_DLL int TreeliteSetLeafNode(ModelBufferHandle handle,
                                     int tree_index, int node_key,
                                     float leaf_value);
TREELITE_DLL int TreeliteCommitModel(ModelBufferHandle handle,
                                     ModelHandle* out);

#endif  /* TREELITE_C_API_H_ */
