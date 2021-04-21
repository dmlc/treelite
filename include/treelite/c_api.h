/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file c_api.h
 * \author Hyunsu Cho
 * \brief C API of Treelite, used for interfacing with other languages
 *        This header is excluded from the runtime
 */

/* Note: Make sure to use slash-asterisk form of comments in this file
   (like this one). Do not use double-slash (//). */

#ifndef TREELITE_C_API_H_
#define TREELITE_C_API_H_

#include "c_api_common.h"

/*!
 * \addtogroup opaque_handles
 * opaque handles
 * \{
 */
/*! \brief handle to a decision tree ensemble model */
typedef void* ModelHandle;
/*! \brief handle to tree builder class */
typedef void* TreeBuilderHandle;
/*! \brief handle to ensemble builder class */
typedef void* ModelBuilderHandle;
/*! \brief handle to branch annotation data */
typedef void* AnnotationHandle;
/*! \brief handle to compiler class */
typedef void* CompilerHandle;
/*! \brief handle to a polymorphic value type, used in the model builder API */
typedef void* ValueHandle;
/*! \} */

/*!
 * \defgroup annotator
 * Branch annotator interface
 * \{
 */
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
TREELITE_DLL int TreeliteAnnotateBranch(
    ModelHandle model, DMatrixHandle dmat, int nthread, int verbose, AnnotationHandle* out);
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
/*! \} */

/*!
 * \defgroup compiler
 * Compiler interface
 * \{
 */
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
 *
 * Usage example:
 * \code
 *   TreeliteCompilerGenerateCode(compiler, model, 1, "./my/model");
 *   // files to generate: ./my/model/header.h, ./my/model/main.c
 *   // if parallel compilation is enabled:
 *   // ./my/model/header.h, ./my/model/main.c, ./my/model/tu0.c,
 *   // ./my/model/tu1.c, and so forth
 * \endcode
 * \param compiler handle for compiler
 * \param model handle for tree ensemble model
 * \param verbose whether to produce extra messages
 * \param dirpath directory to store header and source files
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteCompilerGenerateCode(CompilerHandle compiler,
                                              ModelHandle model,
                                              int verbose,
                                              const char* dirpath);
/*!
 * \brief delete compiler from memory
 * \param handle compiler to remove
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteCompilerFree(CompilerHandle handle);
/*! \} */

/*!
 * \defgroup model_loader
 * Model loader interface: read trees from the disk
 * \{
 */
/*!
 * \brief load a model file generated by LightGBM (Microsoft/LightGBM). The
 *        model file must contain a decision tree ensemble.
 * \param filename name of model file
 * \param out loaded model
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteLoadLightGBMModel(const char* filename, ModelHandle* out);
/*!
 * \brief load a model file generated by XGBoost (dmlc/xgboost). The model file
 *        must contain a decision tree ensemble.
 * \param filename name of model file
 * \param out loaded model
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteLoadXGBoostModel(const char* filename, ModelHandle* out);
/*!
 * \brief load a json model file generated by XGBoost (dmlc/xgboost). The model
 *        file must contain a decision tree ensemble.
 * \param filename name of model file
 * \param out loaded model
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteLoadXGBoostJSON(const char* filename,
                                         ModelHandle* out);
/*!
 * \brief load a model stored as JSON stringby XGBoost (dmlc/xgboost). The model
 *        json must contain a decision tree ensemble.
 * \param json_str the string containing the JSON model specification
 * \param length the length of the JSON string
 * \param out loaded model
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteLoadXGBoostJSONString(const char* json_str,
                                               size_t length,
                                               ModelHandle* out);
/*!
 * \brief load an XGBoost model from a memory buffer.
 * \param buf memory buffer
 * \param len size of memory buffer
 * \param out loaded model
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteLoadXGBoostModelFromMemoryBuffer(const void* buf,
                                                          size_t len,
                                                          ModelHandle* out);

TREELITE_DLL int TreeliteLoadSKLearnRandomForestRegressor(
    int n_estimators, int n_features, const int64_t* node_count, const int64_t** children_left,
    const int64_t** children_right, const int64_t** feature, const double** threshold,
    const double** value, const int64_t** n_node_samples, const double** impurity,
    ModelHandle* out);
TREELITE_DLL int TreeliteLoadSKLearnRandomForestClassifier(
    int n_estimators, int n_features, int n_classes, const int64_t* node_count,
    const int64_t** children_left, const int64_t** children_right, const int64_t** feature,
    const double** threshold, const double** value, const int64_t** n_node_samples,
    const double** impurity, ModelHandle* out);

/*!
 * \brief Query the number of trees in the model
 * \param handle model to query
 * \param out number of trees
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteQueryNumTree(ModelHandle handle, size_t* out);
/*!
 * \brief Query the number of features used in the model
 * \param handle model to query
 * \param out number of features
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteQueryNumFeature(ModelHandle handle, size_t* out);
/*!
 * \brief Query the number of classes of the model. (1 if the model is binary classifier or
 *        regressor)
 * \param handle model to query
 * \param out number of output groups
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteQueryNumClass(ModelHandle handle, size_t* out);

/*!
 * \brief keep first N trees of model, limit must smaller than number of trees.
 * \param handle model
 * \param limit number of trees to keep
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteSetTreeLimit(ModelHandle handle, size_t limit);

/*!
 * \brief delete model from memory
 * \param handle model to remove
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteFreeModel(ModelHandle handle);
/*! \} */

/*!
 * \defgroup model_builder
 * Model builder interface: build trees incrementally
 * \{
 */
/*!
 * \brief Create a new Value object. Some model builder API functions accept this Value type to
 *        accommodate values of multiple types.
 * \param init_value pointer to the value to be stored
 * \param type Type of the value to be stored
 * \param out newly created Value object
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderCreateValue(const void* init_value, const char* type,
                                                ValueHandle* out);
/*!
 * \brief Delete a Value object from memory
 * \param handle pointer to the Value object to be deleted
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderDeleteValue(ValueHandle handle);
/*!
 * \brief Create a new tree builder
 * \param threshold_type Type of thresholds in numerical splits. All thresholds in a given model
 *                       must have the same type.
 * \param leaf_output_type Type of leaf outputs. All leaf outputs in a given model must have the
 *                         same type.
 * \param out newly created tree builder
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteCreateTreeBuilder(const char* threshold_type, const char* leaf_output_type,
                                           TreeBuilderHandle* out);
/*!
 * \brief Delete a tree builder from memory
 * \param handle tree builder to remove
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteDeleteTreeBuilder(TreeBuilderHandle handle);
/*!
 * \brief Create an empty node within a tree
 * \param handle tree builder
 * \param node_key unique integer key to identify the new node
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderCreateNode(TreeBuilderHandle handle, int node_key);
/*!
 * \brief Remove a node from a tree
 * \param handle tree builder
 * \param node_key unique integer key to identify the node to be removed
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderDeleteNode(TreeBuilderHandle handle, int node_key);
/*!
 * \brief Set a node as the root of a tree
 * \param handle tree builder
 * \param node_key unique integer key to identify the root node
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderSetRootNode(TreeBuilderHandle handle, int node_key);
/*!
 * \brief Turn an empty node into a test node with numerical split.
 * The test is in the form [feature value] OP [threshold]. Depending on the
 * result of the test, either left or right child would be taken.
 * \param handle tree builder
 * \param node_key unique integer key to identify the node being modified;
 *                 this node needs to be empty
 * \param feature_id id of feature
 * \param opname binary operator to use in the test
 * \param threshold threshold value
 * \param default_left default direction for missing values
 * \param left_child_key unique integer key to identify the left child node
 * \param right_child_key unique integer key to identify the right child node
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderSetNumericalTestNode(
    TreeBuilderHandle handle, int node_key, unsigned feature_id, const char* opname,
    ValueHandle threshold, int default_left, int left_child_key, int right_child_key);
/*!
 * \brief Turn an empty node into a test node with categorical split.
 * A list defines all categories that would be classified as the left side.
 * Categories are integers ranging from 0 to (n-1), where n is the number of
 * categories in that particular feature. Let's assume n <= 64.
 * \param handle tree builder
 * \param node_key unique integer key to identify the node being modified;
 *                 this node needs to be empty
 * \param feature_id id of feature
 * \param left_categories list of categories belonging to the left child
 * \param left_categories_len length of left_cateogries
 * \param default_left default direction for missing values
 * \param left_child_key unique integer key to identify the left child node
 * \param right_child_key unique integer key to identify the right child node
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderSetCategoricalTestNode(
    TreeBuilderHandle handle, int node_key, unsigned feature_id,
    const unsigned int* left_categories, size_t left_categories_len, int default_left,
    int left_child_key, int right_child_key);
/*!
 * \brief Turn an empty node into a leaf node
 * \param handle tree builder
 * \param node_key unique integer key to identify the node being modified;
 *                 this node needs to be empty
 * \param leaf_value leaf value (weight) of the leaf node
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderSetLeafNode(
    TreeBuilderHandle handle, int node_key, ValueHandle leaf_value);
/*!
 * \brief Turn an empty node into a leaf vector node
 * The leaf vector (collection of multiple leaf weights per leaf node) is
 * useful for multi-class random forest classifier.
 * \param handle tree builder
 * \param node_key unique integer key to identify the node being modified;
 *                 this node needs to be empty
 * \param leaf_vector leaf vector of the leaf node
 * \param leaf_vector_len length of leaf_vector
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderSetLeafVectorNode(
    TreeBuilderHandle handle, int node_key, const ValueHandle* leaf_vector, size_t leaf_vector_len);
/*!
 * \brief Create a new model builder
 * \param num_feature number of features used in model being built. We assume that all feature
 *                    indices are between 0 and (num_feature - 1).
 * \param num_class number of output groups. Set to 1 for binary classification and
 *                  regression; >1 for multiclass classification
 * \param average_tree_output whether the outputs from the trees should be averaged
 *                            (!=0 yes, =0 no)
 * \param threshold_type Type of thresholds in numerical splits. All thresholds in a given model
 *                       must have the same type.
 * \param leaf_output_type Type of leaf outputs. All leaf outputs in a given model must have the
 *                         same type.
 * \param out newly created model builder
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteCreateModelBuilder(
    int num_feature, int num_class, int average_tree_output, const char* threshold_type,
    const char* leaf_output_type, ModelBuilderHandle* out);
/*!
 * \brief Set a model parameter
 * \param handle model builder
 * \param name name of parameter
 * \param value value of parameter
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteModelBuilderSetModelParam(ModelBuilderHandle handle,
                                                   const char* name,
                                                   const char* value);
/*!
 * \brief Delete a model builder from memory
 * \param handle model builder to remove
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteDeleteModelBuilder(ModelBuilderHandle handle);
/*!
 * \brief Insert a tree at specified location
 * \param handle model builder
 * \param tree_builder builder for the tree to be inserted. The tree must not
 *                     be part of any other existing tree ensemble. Note:
 *                     The tree_builder argument will become unusuable after
 *                     the tree insertion. Should you want to modify the
 *                     tree afterwards, use GetTree(*) method to get a fresh
 *                     handle to the tree.
 * \param index index of the element before which to insert the tree;
 *              use -1 to insert at the end
 * \return index of the new tree within the ensemble; -1 for failure
 */
TREELITE_DLL int TreeliteModelBuilderInsertTree(ModelBuilderHandle handle,
                                                TreeBuilderHandle tree_builder,
                                                int index);
/*!
 * \brief Get a reference to a tree in the ensemble
 * \param handle model builder
 * \param index index of the tree in the ensemble
 * \param out used to save reference to the tree
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteModelBuilderGetTree(ModelBuilderHandle handle,
                                             int index,
                                             TreeBuilderHandle *out);
/*!
 * \brief Remove a tree from the ensemble
 * \param handle model builder
 * \param index index of the tree that would be removed
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteModelBuilderDeleteTree(ModelBuilderHandle handle,
                                                int index);
/*!
 * \brief finalize the model and produce the in-memory representation
 * \param handle model builder
 * \param out used to save handle to in-memory representation of the finished
 *            model
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteModelBuilderCommitModel(ModelBuilderHandle handle,
                                                 ModelHandle* out);
/*! \} */

#endif  /* TREELITE_C_API_H_ */
