/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file frontend.h
 * \brief Collection of front-end methods to load or construct ensemble model
 * \author Hyunsu Cho
 */
#ifndef TREELITE_FRONTEND_H_
#define TREELITE_FRONTEND_H_

#include <treelite/base.h>
#include <memory>
#include <vector>
#include <cstdint>

namespace treelite {

struct Model;  // forward declaration

namespace frontend {

//--------------------------------------------------------------------------
// model loader interface: read from the disk
//--------------------------------------------------------------------------
/*!
 * \brief load a model file generated by LightGBM (Microsoft/LightGBM). The
 *        model file must contain a decision tree ensemble.
 * \param filename name of model file
 * \param out reference to loaded model
 */
void LoadLightGBMModel(const char *filename, Model* out);
/*!
 * \brief load a model file generated by XGBoost (dmlc/xgboost). The model file
 *        must contain a decision tree ensemble.
 * \param filename name of model file
 * \param out reference to loaded model
 */
void LoadXGBoostModel(const char* filename, Model* out);
/*!
 * \brief load an XGBoost model from a memory buffer.
 * \param buf memory buffer
 * \param len size of memory buffer
 * \param out reference to loaded model
 */
void LoadXGBoostModel(const void* buf, size_t len, Model* out);

//--------------------------------------------------------------------------
// model builder interface: build trees incrementally
//--------------------------------------------------------------------------

/* forward declarations */
struct TreeBuilderImpl;
struct ModelBuilderImpl;
class ModelBuilder;

class Value {
 private:
  std::shared_ptr<void> handle_;
  TypeInfo type_;
 public:
  Value();
  ~Value() = default;
  Value(const Value&) = default;
  Value(Value&&) noexcept = default;
  Value& operator=(const Value&) = default;
  Value& operator=(Value&&) noexcept = default;
  template <typename T>
  static Value Create(T init_value);
  static Value Create(const void* init_value, TypeInfo type);
  template <typename T>
  T& Get();
  template <typename T>
  const T& Get() const;
  template <typename Func>
  inline auto Dispatch(Func func);
  template <typename Func>
  inline auto Dispatch(Func func) const;
  TypeInfo GetValueType() const;
};

/*! \brief tree builder class */
class TreeBuilder {
 public:
  /*!
   * \brief Constructor
   * \param threshold_type Type of thresholds in numerical splits. All thresholds in a given model
   *                       must have the same type.
   * \param leaf_output_type Type of leaf outputs. All leaf outputs in a given model must have the
   *                         same type.
   */
  TreeBuilder(TypeInfo threshold_type, TypeInfo leaf_output_type);  // constructor
  ~TreeBuilder();  // destructor
  // this class is only move-constructible and move-assignable
  TreeBuilder(const TreeBuilder&) = delete;
  TreeBuilder(TreeBuilder&&) = default;
  TreeBuilder& operator=(const TreeBuilder&) = delete;
  TreeBuilder& operator=(TreeBuilder&&) = default;
  /*!
   * \brief Create an empty node within a tree
   * \param node_key unique integer key to identify the new node
   */
  void CreateNode(int node_key);
  /*!
   * \brief Remove a node from a tree
   * \param node_key unique integer key to identify the node to be removed
   */
  void DeleteNode(int node_key);
  /*!
   * \brief Set a node as the root of a tree
   * \param node_key unique integer key to identify the root node
   */
  void SetRootNode(int node_key);
  /*!
   * \brief Turn an empty node into a numerical test node; the test is in the
   *        form [feature value] OP [threshold]. Depending on the result of the
   *        test, either left or right child would be taken.
   * \param node_key unique integer key to identify the node being modified;
   *                 this node needs to be empty
   * \param feature_id id of feature
   * \param op binary operator to use in the test
   * \param threshold threshold value
   * \param default_left default direction for missing values
   * \param left_child_key unique integer key to identify the left child node
   * \param right_child_key unique integer key to identify the right child node
   */
  void SetNumericalTestNode(int node_key, unsigned feature_id, const char* op, Value threshold,
                            bool default_left, int left_child_key, int right_child_key);
  void SetNumericalTestNode(int node_key, unsigned feature_id, Operator op, Value threshold,
                            bool default_left, int left_child_key, int right_child_key);
  /*!
   * \brief Turn an empty node into a categorical test node.
   * A list defines all categories that would be classified as the left side.
   * Categories are integers ranging from 0 to (n-1), where n is the number of
   * categories in that particular feature. Let's assume n <= 64.
   * \param node_key unique integer key to identify the node being modified;
   *                 this node needs to be empty
   * \param feature_id id of feature
   * \param left_categories list of categories belonging to the left child
   * \param default_left default direction for missing values
   * \param left_child_key unique integer key to identify the left child node
   * \param right_child_key unique integer key to identify the right child node
   */
  void SetCategoricalTestNode(int node_key, unsigned feature_id,
                              const std::vector<uint32_t>& left_categories, bool default_left,
                              int left_child_key, int right_child_key);
  /*!
   * \brief Turn an empty node into a leaf node
   * \param node_key unique integer key to identify the node being modified;
   *                 this node needs to be empty
   * \param leaf_value leaf value (weight) of the leaf node
   */
  void SetLeafNode(int node_key, Value leaf_value);
  /*!
  * \brief Turn an empty node into a leaf vector node
  * The leaf vector (collection of multiple leaf weights per leaf node) is
  * useful for multi-class random forest classifier.
  * \param node_key unique integer key to identify the node being modified;
  *                 this node needs to be empty
  * \param leaf_vector leaf vector of the leaf node
  */
  void SetLeafVectorNode(int node_key, const std::vector<Value>& leaf_vector);

 private:
  std::unique_ptr<TreeBuilderImpl> pimpl_;  // Pimpl pattern
  ModelBuilder* ensemble_id_;  // id of ensemble (nullptr if not part of any)
  friend class ModelBuilder;
  friend struct ModelBuilderImpl;
};

/*! \brief model builder class */
class ModelBuilder {
 public:
  /*!
   * \brief Constructor
   * \param num_feature number of features used in model being built. We assume
   *                    that all feature indices are between 0 and
   *                    (num_feature - 1).
   * \param num_output_group number of output groups. Set to 1 for binary
   *                         classification and regression; >1 for multiclass
   *                         classification
   * \param random_forest_flag whether the model is a random forest (true) or
   *                           gradient boosted trees (false)
   * \param threshold_type Type of thresholds in numerical splits. All thresholds in a given model
   *                       must have the same type.
   * \param leaf_output_type Type of leaf outputs. All leaf outputs in a given model must have the
   *                         same type.
   */
  ModelBuilder(int num_feature, int num_output_group, bool random_forest_flag,
               TypeInfo threshold_type, TypeInfo leaf_output_type);
  ~ModelBuilder();  // destructor
  /*!
   * \brief Set a model parameter
   * \param name name of parameter
   * \param value value of parameter
   */
  void SetModelParam(const char* name, const char* value);
  /*!
   * \brief Insert a tree at specified location
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
  int InsertTree(TreeBuilder* tree_builder, int index = -1);
  /*!
   * \brief Get a reference to a tree in the ensemble
   * \param index index of the tree in the ensemble
   * \return reference to the tree
   */
  TreeBuilder* GetTree(int index);
  const TreeBuilder* GetTree(int index) const;
  /*!
   * \brief Remove a tree from the ensemble
   * \param index index of the tree that would be removed
   */
  void DeleteTree(int index);
  /*!
   * \brief finalize the model and produce the in-memory representation
   * \param out_model place to store in-memory representation of the finished model
   */
  void CommitModel(Model* out_model);

 private:
  std::unique_ptr<ModelBuilderImpl> pimpl_;  // Pimpl pattern
};

}  // namespace frontend
}  // namespace treelite

#include "frontend_impl.h"

#endif  // TREELITE_FRONTEND_H_
