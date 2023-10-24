/*!
 * Copyright (c) 2020-2023 by Contributors
 * \file tree.h
 * \brief Implementation for treelite/tree.h
 * \author Hyunsu Cho
 */
#ifndef TREELITE_DETAIL_TREE_H_
#define TREELITE_DETAIL_TREE_H_

#include <treelite/error.h>
#include <treelite/logging.h>
#include <treelite/version.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

namespace treelite {

template <typename ThresholdType, typename LeafOutputType>
inline Tree<ThresholdType, LeafOutputType> Tree<ThresholdType, LeafOutputType>::Clone() const {
  Tree<ThresholdType, LeafOutputType> tree;

  tree.num_nodes = num_nodes;

  tree.node_type_ = node_type_.Clone();
  tree.cleft_ = cleft_.Clone();
  tree.cright_ = cright_.Clone();
  tree.split_index_ = split_index_.Clone();
  tree.default_left_ = default_left_.Clone();
  tree.leaf_value_ = leaf_value_.Clone();
  tree.threshold_ = threshold_.Clone();
  tree.cmp_ = cmp_.Clone();
  tree.category_list_right_child_ = category_list_right_child_.Clone();

  tree.leaf_vector_ = leaf_vector_.Clone();
  tree.leaf_vector_begin_ = leaf_vector_begin_.Clone();
  tree.leaf_vector_end_ = leaf_vector_end_.Clone();
  tree.category_list_ = category_list_.Clone();
  tree.category_list_begin_ = category_list_begin_.Clone();
  tree.category_list_end_ = category_list_end_.Clone();

  tree.data_count_ = data_count_.Clone();
  tree.sum_hess_ = sum_hess_.Clone();
  tree.gain_ = gain_.Clone();
  tree.data_count_present_ = data_count_present_.Clone();
  tree.sum_hess_present_ = sum_hess_present_.Clone();
  tree.gain_present_ = gain_present_.Clone();

  tree.has_categorical_split_ = has_categorical_split_;
  tree.num_opt_field_per_tree_ = num_opt_field_per_tree_;
  tree.num_opt_field_per_node_ = num_opt_field_per_node_;

  return tree;
}

template <typename ThresholdType, typename LeafOutputType>
inline int Tree<ThresholdType, LeafOutputType>::AllocNode() {
  node_type_.PushBack(TreeNodeType::kLeafNode);
  cleft_.PushBack(-1);
  cright_.PushBack(-1);
  split_index_.PushBack(-1);
  default_left_.PushBack(false);
  leaf_value_.PushBack(static_cast<LeafOutputType>(0));
  threshold_.PushBack(static_cast<ThresholdType>(0));
  cmp_.PushBack(Operator::kNone);
  category_list_right_child_.PushBack(false);

  leaf_vector_begin_.PushBack(leaf_vector_.Size());
  leaf_vector_end_.PushBack(leaf_vector_.Size());
  category_list_begin_.PushBack(category_list_.Size());
  category_list_end_.PushBack(category_list_.Size());

  // Invariant: node stat array must either be empty or have exact length of [num_nodes]
  if (!data_count_present_.Empty()) {
    data_count_.PushBack(0);
    data_count_present_.PushBack(false);
  }
  if (!sum_hess_present_.Empty()) {
    sum_hess_.PushBack(0);
    sum_hess_present_.PushBack(false);
  }
  if (!gain_present_.Empty()) {
    gain_.PushBack(0);
    gain_present_.PushBack(false);
  }

  return num_nodes++;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::Init() {
  node_type_.Clear();
  cleft_.Clear();
  cright_.Clear();
  split_index_.Clear();
  default_left_.Clear();
  leaf_value_.Clear();
  threshold_.Clear();
  cmp_.Clear();
  category_list_right_child_.Clear();

  num_nodes = 0;
  has_categorical_split_ = false;

  leaf_vector_.Clear();
  leaf_vector_begin_.Clear();
  leaf_vector_end_.Clear();
  category_list_.Clear();
  category_list_begin_.Clear();
  category_list_end_.Clear();
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetNumericalTest(
    int nid, std::int32_t split_index, ThresholdType threshold, bool default_left, Operator cmp) {
  split_index_.at(nid) = split_index;
  threshold_.at(nid) = threshold;
  default_left_.at(nid) = default_left;
  cmp_.at(nid) = cmp;
  node_type_.at(nid) = TreeNodeType::kNumericalTestNode;
  category_list_right_child_.at(nid) = false;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetCategoricalTest(int nid,
    std::int32_t split_index, bool default_left, std::vector<std::uint32_t> const& category_list,
    bool category_list_right_child) {
  TREELITE_CHECK(CategoryList(nid).empty()) << "Cannot set categorical test twice for same node";

  std::size_t const begin = category_list_.Size();
  std::size_t const end = begin + category_list.size();
  category_list_.Extend(category_list);
  category_list_begin_.at(nid) = begin;
  category_list_end_.at(nid) = end;

  split_index_.at(nid) = split_index;
  default_left_.at(nid) = default_left;
  node_type_.at(nid) = TreeNodeType::kCategoricalTestNode;
  category_list_right_child_.at(nid) = category_list_right_child;

  has_categorical_split_ = true;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetLeaf(int nid, LeafOutputType value) {
  leaf_value_.at(nid) = value;
  cleft_.at(nid) = -1;
  cright_.at(nid) = -1;
  node_type_.at(nid) = TreeNodeType::kLeafNode;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetLeafVector(
    int nid, std::vector<LeafOutputType> const& node_leaf_vector) {
  TREELITE_CHECK(!HasLeafVector(nid)) << "Cannot set leaf vector twice for same node";
  std::size_t begin = leaf_vector_.Size();
  std::size_t end = begin + node_leaf_vector.size();
  leaf_vector_.Extend(node_leaf_vector);
  leaf_vector_begin_.at(nid) = begin;
  leaf_vector_end_.at(nid) = end;

  split_index_.at(nid) = -1;
  cleft_.at(nid) = -1;
  cright_.at(nid) = -1;
  node_type_.at(nid) = TreeNodeType::kLeafNode;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetSumHess(int nid, double sum_hess) {
  if (sum_hess_present_.Empty()) {
    sum_hess_present_.Resize(num_nodes, false);
    sum_hess_.Resize(num_nodes);
  }
  sum_hess_.at(nid) = sum_hess;
  sum_hess_present_.at(nid) = true;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetDataCount(int nid, std::uint64_t data_count) {
  if (data_count_present_.Empty()) {
    data_count_present_.Resize(num_nodes, false);
    data_count_.Resize(num_nodes);
  }
  data_count_.at(nid) = data_count;
  data_count_present_.at(nid) = true;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetGain(int nid, double gain) {
  if (gain_present_.Empty()) {
    gain_present_.Resize(num_nodes, false);
    gain_.Resize(num_nodes);
  }
  gain_.at(nid) = gain;
  gain_present_.at(nid) = true;
}

template <typename ThresholdType, typename LeafOutputType>
inline std::unique_ptr<Model> Model::Create() {
  std::unique_ptr<Model> model = std::make_unique<Model>();
  model->variant_ = ModelPreset<ThresholdType, LeafOutputType>();
  return model;
}

inline std::unique_ptr<Model> Model::Create(TypeInfo threshold_type, TypeInfo leaf_output_type) {
  std::unique_ptr<Model> model = std::make_unique<Model>();
  TREELITE_CHECK(threshold_type == TypeInfo::kFloat32 || threshold_type == TypeInfo::kFloat64)
      << "threshold_type must be either float32 or float64";
  TREELITE_CHECK(leaf_output_type == threshold_type)
      << "threshold_type must be identical to leaf_output_type";
  int const target_variant_index = threshold_type == TypeInfo::kFloat64;
  model->variant_ = SetModelPresetVariant<0>(target_variant_index);
  return model;
}

}  // namespace treelite
#endif  // TREELITE_DETAIL_TREE_H_
