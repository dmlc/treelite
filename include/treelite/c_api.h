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
typedef void* ModelBuilderHandle;
typedef void* AnnotationHandle;
typedef void* CompilerHandle;
typedef void* PredictorHandle;
typedef void* DMatrixHandle;

/*!
 * \brief display last error; can be called by different threads
 * \return error string
 */
TREELITE_DLL const char* TreeliteGetLastError();

/***************************************************************************
 * Part 1: data matrix interface                                           *
 ***************************************************************************/
/*!
 * \brief create DMatrix from a file
 * \param path file path
 * \param format file format
 * \param nthread number of threads to use
 * \param verbose whether to produce extra messages
 * \param out the created DMatrix
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixCreateFromFile(const char* path,
                                               const char* format,
                                               int nthread,
                                               int verbose,
                                               DMatrixHandle* out);
/*!
 * \brief create DMatrix from a (in-memory) CSR matrix
 * \param data feature values
 * \param col_ind feature indices
 * \param row_ptr pointer to row headers
 * \param num_row number of rows
 * \param num_col number of columns
 * \param out the created DMatrix
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixCreateFromCSR(const float* data,
                                              const unsigned* col_ind,
                                              const size_t* row_ptr,
                                              size_t num_row,
                                              size_t num_col,
                                              DMatrixHandle* out);
/*!
 * \brief create DMatrix from a (in-memory) dense matrix
 * \param data feature values
 * \param num_row number of rows
 * \param num_col number of columns
 * \param missing_value value to represent missing value
 * \param out the created DMatrix
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixCreateFromMat(const float* data,
                                              size_t num_row,
                                              size_t num_col,
                                              float missing_value,
                                              DMatrixHandle* out);
/*!
 * \brief get dimensions of a DMatrix
 * \param handle handle to DMatrix
 * \param out_num_row used to set number of rows
 * \param out_num_col used to set number of columns
 * \param out_nelem used to set number of nonzero entries
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixGetDimension(DMatrixHandle handle,
                                             size_t* out_num_row,
                                             size_t* out_num_col,
                                             size_t* out_nelem);
/*!
 * \brief delete DMatrix from memory
 * \param handle handle to DMatrix
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixFree(DMatrixHandle handle);

/***************************************************************************
 * Part 2: branch annotator interface
 ***************************************************************************/
/*!
 * \brief annotate branches in a given model using frequency patterns in the
 *        training data.
 * \param model model to annotate
 * \param dmat training data matrix
 * \param nthread number of threads to use
 * \param verbose whether to produce extra messages
 * \param out used to save handle for the created annotation
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteAnnotateBranch(ModelHandle model,
                                        DMatrixHandle dmat,
                                        int nthread,
                                        int verbose,
                                        AnnotationHandle* out);
/*!
 * \brief load branch annotation from a JSON file
 * \param path path to JSON file
 * \param out used to save handle for the loaded annotation
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteAnnotationLoad(const char* path,
                                        AnnotationHandle* out);
/*!
 * \brief save branch annotation to a JSON file
 * \param handle annotation to save
 * \param path path to JSON file
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteAnnotationSave(AnnotationHandle handle,
                                        const char* path);
/*!
 * \brief delete branch annotation from memory
 * \param handle annotation to remove
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteAnnotationFree(AnnotationHandle handle);

/***************************************************************************
 * Part 3: compiler interface
 ***************************************************************************/
/*!
 * \brief create a compiler with a given name
 * \param name name of compiler
 * \param out created compiler
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteCompilerCreate(const char* name,
                                        CompilerHandle* out);
/*!
 * \brief set a parameter for a compiler
 * \param handle compiler
 * \param name name of parameter
 * \param value value of parameter
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteCompilerSetParam(CompilerHandle handle,
                                          const char* name,
                                          const char* value);
/*!
 * \brief generate prediction code from a tree ensemble model. The code will
 *        be C99 compliant. One header file (.h) will be generated, along with
 *        one or more source files (.c).
 * Usage example:
 * \code
 *   TreeliteCompilerGenerateCode(compiler, model, 1, "./my/model");
 *   // files to generate: ./my/model.h, ./my/model.c
 *   // if parallel compilation is enabled:
 *   // ./my/model.h, ./my/model0.c, ./my/model1.c, ./my/model2.c, and so forth
 * \endcode
 * \param handle compiler
 * \param model tree ensemble model
 * \param verbose whether to produce extra messages
 * \param path_prefix path prefix for header and source files
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteCompilerGenerateCode(CompilerHandle compiler,
                                              ModelHandle model,
                                              int verbose,
                                              const char* path_prefix);
/*!
 * \brief delete compiler from memory
 * \param handle compiler to remove
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteCompilerFree(CompilerHandle handle);

/***************************************************************************
 * Part 4: predictor interface
 ***************************************************************************/
/*!
 * \brief load prediction code into memory.
 * This function assumes that the prediction code has been already compiled into
 * a dynamic shared library object (.so/.dll/.dylib).
 * \param library_path path to library object file containing prediction code
 * \param out handle to predictor
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorLoad(const char* library_path,
                                       PredictorHandle* out);
/*!
 * \brief make predictions on a dataset
 * \param handle predictor
 * \param dmat data matrix
 * \param nthread number of threads to use
 * \param verbose whether to produce extra messages
 * \param out_result used to store result of prediction
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorPredict(PredictorHandle handle,
                                          DMatrixHandle dmat,
                                          int nthread,
                                          int verbose,
                                          float* out_result);
/*!
 * \brief delete predictor from memory
 * \param handle predictor to remove
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorFree(PredictorHandle handle);

/***************************************************************************
 * Part 5. model loader interface: read trees from the disk
 ***************************************************************************/
/*!
 * \brief load a model file generated by LightGBM (Microsoft/LightGBM). The
 *        model file must contain a decision tree ensemble.
 * \param filename name of model file
 * \param out loaded model
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteLoadLightGBMModel(const char* filename,
                                           ModelHandle* out);
/*!
 * \brief load a model file generated by XGBoost (dmlc/xgboost). The model file
 *        must contain a decision tree ensemble.
 * \param filename name of model file
 * \param out loaded model
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteLoadXGBoostModel(const char* filename,
                                          ModelHandle* out);
/*!
 * \brief load a model in Protocol Buffers format. Protocol Buffers
 *        (google/protobuf) is a language- and platform-neutral mechanism for
 *        serializing structured data. See tree.proto for format spec.
 * \param filename name of model file
 * \param out loaded model
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteLoadProtobufModel(const char* filename,
                                           ModelHandle* out);
/*!
 * \brief delete model from memory
 * \param handle model to remove
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteFreeModel(ModelHandle handle);

/***************************************************************************
 * Part 6. model builder interface: build trees incrementally
 ***************************************************************************/
/*!
 * \brief Create a new model builder
 * \param num_features number of features used in model being built. We assume
 *                     that all feature indices are between 0 and
 *                     (num_features - 1).
 * \param out newly created model builder
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteCreateModelBuilder(int num_features,
                                            ModelBuilderHandle* out);
/*!
 * \brief Delete a model builder from memory
 * \param handle model builder to remove
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteDeleteModelBuilder(ModelBuilderHandle handle);
/*!
 * \brief Create a new tree
 * \param handle model builder
 * \param index location within the ensemble at which the new tree
 *              would be placed; use -1 to insert at the end
 * \return index of the new tree within the ensemble; -1 for failure
 */
TREELITE_DLL int TreeliteCreateTree(ModelBuilderHandle handle,
                                    int index);
/*!
 * \brief Remove a tree from the ensemble
 * \param handle model builder
 * \param index index of the tree that would be removed
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteDeleteTree(ModelBuilderHandle handle,
                                    int index);
/*!
 * \brief Create an empty node within a tree
 * \param handle model builder
 * \param tree_index index of the tree into which the new node will be placed
 * \param node_key unique integer key to identify the new node
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteCreateNode(ModelBuilderHandle handle,
                                    int tree_index, int node_key);
/*!
 * \brief Remove a node from a tree
 * \param handle model builder
 * \param tree_index index of the tree from which a node will be removed
 * \param node_key unique integer key to identify the node to be removed
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteDeleteNode(ModelBuilderHandle handle,
                                    int tree_index, int node_key);
/*!
 * \brief Set a node as the root of a tree
 * \param handle model builder
 * \param tree_index index of the tree whose root is being set
 * \param node_key unique integer key to identify the root node
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteSetRootNode(ModelBuilderHandle handle,
                                     int tree_index, int node_key);
/*!
 * \brief Turn an empty node into a test (non-leaf) node; the test is in the
 *        form [feature value] OP [threshold]. Depending on the result of the
 *        test, either left or right child would be taken.
 * \param handle model builder
 * \param tree_index index of the tree containing the node being modified
 * \param node_key unique integer key to identify the node being modified;
 *                 this node needs to be empty
 * \param feature_id id of feature
 * \param op_name binary operator to use in the test
 * \param threshold threshold value
 * \param default_left default direction for missing values
 * \param left_child_key unique integer key to identify the left child node
 * \param right_child_key unique integer key to identify the right child node
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteSetTestNode(ModelBuilderHandle handle,
                                     int tree_index, int node_key,
                                     unsigned feature_id, const char* opname,
                                     float threshold, int default_left,
                                     int left_child_key, int right_child_key);
/*!
 * \brief Turn an empty node into a leaf node
 * \param handle model builder
 * \param tree_index index of the tree containing the node being modified
 * \param node_key unique integer key to identify the node being modified;
 *                 this node needs to be empty
 * \param leaf_value leaf value (weight) of the leaf node
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteSetLeafNode(ModelBuilderHandle handle,
                                     int tree_index, int node_key,
                                     float leaf_value);
/*!
 * \brief finalize the model and produce the in-memory representation
 * \param handle model builder
 * \param out_model place to store in-memory representation of the finished
 *                  model
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteCommitModel(ModelBuilderHandle handle,
                                     ModelHandle* out);

#endif  /* TREELITE_C_API_H_ */
