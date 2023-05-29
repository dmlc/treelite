/*!
 * Copyright (c) 2020-2021 by Contributors
 * \file tree_impl.h
 * \brief Implementation for tree.h
 * \author Hyunsu Cho
 */
#ifndef TREELITE_TREE_IMPL_H_
#define TREELITE_TREE_IMPL_H_

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

namespace treelite::detail {

template <typename T>
inline std::string GetString(T x) {
  return std::to_string(x);
}

template <>
inline std::string GetString<float>(float x) {
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<float>::max_digits10) << x;
  return oss.str();
}

template <>
inline std::string GetString<double>(double x) {
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<double>::max_digits10) << x;
  return oss.str();
}

}  // namespace treelite::detail

namespace treelite {

template <typename Container>
inline std::vector<std::pair<std::string, std::string>> ModelParam::InitAllowUnknown(
    Container const& kwargs) {
  std::vector<std::pair<std::string, std::string>> unknowns;
  for (auto const& e : kwargs) {
    if (e.first == "pred_transform") {
      std::strncpy(this->pred_transform, e.second.c_str(), TREELITE_MAX_PRED_TRANSFORM_LENGTH - 1);
      this->pred_transform[TREELITE_MAX_PRED_TRANSFORM_LENGTH - 1] = '\0';
    } else if (e.first == "sigmoid_alpha") {
      this->sigmoid_alpha = std::stof(e.second, nullptr);
    } else if (e.first == "ratio_c") {
      this->ratio_c = std::stof(e.second, nullptr);
    } else if (e.first == "global_bias") {
      this->global_bias = std::stof(e.second, nullptr);
    }
  }
  return unknowns;
}

inline std::map<std::string, std::string> ModelParam::__DICT__() const {
  std::map<std::string, std::string> ret;
  ret.emplace("pred_transform", std::string(this->pred_transform));
  ret.emplace("sigmoid_alpha", detail::GetString(this->sigmoid_alpha));
  ret.emplace("ratio_c", detail::GetString(this->ratio_c));
  ret.emplace("global_bias", detail::GetString(this->global_bias));
  return ret;
}

template <typename ThresholdType, typename LeafOutputType>
inline Tree<ThresholdType, LeafOutputType> Tree<ThresholdType, LeafOutputType>::Clone() const {
  Tree<ThresholdType, LeafOutputType> tree;
  tree.num_nodes = num_nodes;
  tree.nodes_ = nodes_.Clone();
  tree.leaf_vector_ = leaf_vector_.Clone();
  tree.leaf_vector_begin_ = leaf_vector_begin_.Clone();
  tree.leaf_vector_end_ = leaf_vector_end_.Clone();
  tree.matching_categories_ = matching_categories_.Clone();
  tree.matching_categories_offset_ = matching_categories_offset_.Clone();
  return tree;
}

template <typename ThresholdType, typename LeafOutputType>
inline char const* Tree<ThresholdType, LeafOutputType>::GetFormatStringForNode() {
  if (std::is_same<ThresholdType, float>::value) {
    return "T{=l=l=L=f=Q=d=d=b=b=?=?=?=?xx}";
  } else {
    return "T{=l=l=Lxxxx=d=Q=d=d=b=b=?=?=?=?xx}";
  }
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::Node::Init() {
  std::memset(this, 0, sizeof(Node));
  cleft_ = cright_ = -1;
  sindex_ = 0;
  info_.leaf_value = static_cast<LeafOutputType>(0);
  info_.threshold = static_cast<ThresholdType>(0);
  data_count_ = 0;
  sum_hess_ = gain_ = 0.0;
  data_count_present_ = sum_hess_present_ = gain_present_ = false;
  categories_list_right_child_ = false;
  split_type_ = SplitFeatureType::kNone;
  cmp_ = Operator::kNone;
}

template <typename ThresholdType, typename LeafOutputType>
inline int Tree<ThresholdType, LeafOutputType>::AllocNode() {
  int nd = num_nodes++;
  if (nodes_.Size() != static_cast<std::size_t>(nd)) {
    throw Error("Invariant violated: nodes_ contains incorrect number of nodes");
  }
  for (int nid = nd; nid < num_nodes; ++nid) {
    leaf_vector_begin_.PushBack(0);
    leaf_vector_end_.PushBack(0);
    matching_categories_offset_.PushBack(matching_categories_offset_.Back());
    nodes_.Resize(nodes_.Size() + 1);
    nodes_.Back().Init();
  }
  return nd;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::Init() {
  num_nodes = 1;
  has_categorical_split_ = false;
  leaf_vector_.Clear();
  leaf_vector_begin_.Resize(1, {});
  leaf_vector_end_.Resize(1, {});
  matching_categories_.Clear();
  matching_categories_offset_.Resize(2, 0);
  nodes_.Resize(1);
  nodes_.at(0).Init();
  SetLeaf(0, static_cast<LeafOutputType>(0));
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::AddChilds(int nid) {
  int const cleft = this->AllocNode();
  int const cright = this->AllocNode();
  nodes_.at(nid).cleft_ = cleft;
  nodes_.at(nid).cright_ = cright;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetNumericalSplit(
    int nid, unsigned split_index, ThresholdType threshold, bool default_left, Operator cmp) {
  Node& node = nodes_.at(nid);
  if (split_index >= ((1U << 31U) - 1)) {
    throw Error("split_index too big");
  }
  if (default_left) {
    split_index |= (1U << 31U);
  }
  node.sindex_ = split_index;
  (node.info_).threshold = threshold;
  node.cmp_ = cmp;
  node.split_type_ = SplitFeatureType::kNumerical;
  node.categories_list_right_child_ = false;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetCategoricalSplit(int nid, unsigned split_index,
    bool default_left, std::vector<uint32_t> const& categories_list,
    bool categories_list_right_child) {
  if (split_index >= ((1U << 31U) - 1)) {
    throw Error("split_index too big");
  }

  const std::size_t end_oft = matching_categories_offset_.Back();
  const std::size_t new_end_oft = end_oft + categories_list.size();
  if (end_oft != matching_categories_.Size()) {
    throw Error("Invariant violated");
  }
  if (!std::all_of(&matching_categories_offset_.at(nid + 1), matching_categories_offset_.End(),
          [end_oft](std::size_t x) { return (x == end_oft); })) {
    throw Error("Invariant violated");
  }
  // Hopefully we won't have to move any element as we add node_matching_categories for node nid
  matching_categories_.Extend(categories_list);
  if (new_end_oft != matching_categories_.Size()) {
    throw Error("Invariant violated");
  }
  std::for_each(&matching_categories_offset_.at(nid + 1), matching_categories_offset_.End(),
      [new_end_oft](std::size_t& x) { x = new_end_oft; });
  if (!matching_categories_.Empty()) {
    std::sort(&matching_categories_.at(end_oft), matching_categories_.End());
  }

  Node& node = nodes_.at(nid);
  if (default_left) {
    split_index |= (1U << 31U);
  }
  node.sindex_ = split_index;
  node.split_type_ = SplitFeatureType::kCategorical;
  node.categories_list_right_child_ = categories_list_right_child;

  has_categorical_split_ = true;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetLeaf(int nid, LeafOutputType value) {
  Node& node = nodes_.at(nid);
  (node.info_).leaf_value = value;
  node.cleft_ = -1;
  node.cright_ = -1;
  node.split_type_ = SplitFeatureType::kNone;
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::SetLeafVector(
    int nid, std::vector<LeafOutputType> const& node_leaf_vector) {
  std::size_t begin = leaf_vector_.Size();
  std::size_t end = begin + node_leaf_vector.size();
  leaf_vector_.Extend(node_leaf_vector);
  leaf_vector_begin_[nid] = begin;
  leaf_vector_end_[nid] = end;
  Node& node = nodes_.at(nid);
  node.cleft_ = -1;
  node.cright_ = -1;
  node.split_type_ = SplitFeatureType::kNone;
}

template <typename ThresholdType, typename LeafOutputType>
inline std::unique_ptr<Model> Model::Create() {
  std::unique_ptr<Model> model = std::make_unique<ModelImpl<ThresholdType, LeafOutputType>>();
  model->threshold_type_ = TypeToInfo<ThresholdType>();
  model->leaf_output_type_ = TypeToInfo<LeafOutputType>();
  return model;
}

template <typename ThresholdType, typename LeafOutputType>
class ModelCreateImpl {
 public:
  inline static std::unique_ptr<Model> Dispatch() {
    return Model::Create<ThresholdType, LeafOutputType>();
  }
};

inline std::unique_ptr<Model> Model::Create(TypeInfo threshold_type, TypeInfo leaf_output_type) {
  return DispatchWithModelTypes<ModelCreateImpl>(threshold_type, leaf_output_type);
}

template <typename ThresholdType, typename LeafOutputType>
class ModelDispatchImpl {
 public:
  template <typename Func>
  inline static auto Dispatch(Model* model, Func func) {
    return func(*dynamic_cast<ModelImpl<ThresholdType, LeafOutputType>*>(model));
  }

  template <typename Func>
  inline static auto Dispatch(Model const* model, Func func) {
    return func(*dynamic_cast<ModelImpl<ThresholdType, LeafOutputType> const*>(model));
  }
};

template <typename Func>
inline auto Model::Dispatch(Func func) {
  return DispatchWithModelTypes<ModelDispatchImpl>(threshold_type_, leaf_output_type_, this, func);
}

template <typename Func>
inline auto Model::Dispatch(Func func) const {
  return DispatchWithModelTypes<ModelDispatchImpl>(threshold_type_, leaf_output_type_, this, func);
}

inline void InitParamAndCheck(
    ModelParam* param, std::vector<std::pair<std::string, std::string>> const& cfg) {
  auto unknown = param->InitAllowUnknown(cfg);
  if (!unknown.empty()) {
    std::ostringstream oss;
    for (auto const& kv : unknown) {
      oss << kv.first << ", ";
    }
    std::cerr << "\033[1;31mWarning: Unknown parameters found; "
              << "they have been ignored\u001B[0m: " << oss.str() << std::endl;
  }
}

}  // namespace treelite
#endif  // TREELITE_TREE_IMPL_H_
