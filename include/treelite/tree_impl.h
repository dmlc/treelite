/*!
 * Copyright (c) 2020 by Contributors
 * \file tree_impl.h
 * \brief Implementation for tree.h
 * \author Hyunsu Cho
 */
#ifndef TREELITE_TREE_IMPL_H_
#define TREELITE_TREE_IMPL_H_

#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <iostream>

namespace {

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

}  // anonymous namespace

namespace treelite {

template <typename T>
ContiguousArray<T>::ContiguousArray()
    : buffer_(nullptr), size_(0), capacity_(0), owned_buffer_(true) {}

template <typename T>
ContiguousArray<T>::~ContiguousArray() {
  if (buffer_ && owned_buffer_) {
    std::free(buffer_);
  }
}

template <typename T>
ContiguousArray<T>::ContiguousArray(ContiguousArray&& other) noexcept
    : buffer_(other.buffer_), size_(other.size_), capacity_(other.capacity_),
      owned_buffer_(other.owned_buffer_) {
  other.buffer_ = nullptr;
  other.size_ = other.capacity_ = 0;
}

template <typename T>
ContiguousArray<T>&
ContiguousArray<T>::operator=(ContiguousArray&& other) noexcept {
  if (buffer_ && owned_buffer_) {
    std::free(buffer_);
  }
  buffer_ = other.buffer_;
  size_ = other.size_;
  capacity_ = other.capacity_;
  owned_buffer_ = other.owned_buffer_;
  other.buffer_ = nullptr;
  other.size_ = other.capacity_ = 0;
  return *this;
}

template <typename T>
inline ContiguousArray<T>
ContiguousArray<T>::Clone() const {
  ContiguousArray clone;
  clone.buffer_ = static_cast<T*>(std::malloc(sizeof(T) * capacity_));
  if (!clone.buffer_) {
    throw std::runtime_error("Could not allocate memory for the clone");
  }
  std::memcpy(clone.buffer_, buffer_, sizeof(T) * size_);
  clone.size_ = size_;
  clone.capacity_ = capacity_;
  clone.owned_buffer_ = true;
  return clone;
}

template <typename T>
inline void
ContiguousArray<T>::UseForeignBuffer(void* prealloc_buf, size_t size) {
  if (buffer_ && owned_buffer_) {
    std::free(buffer_);
  }
  buffer_ = static_cast<T*>(prealloc_buf);
  size_ = size;
  capacity_ = size;
  owned_buffer_ = false;
}

template <typename T>
inline T*
ContiguousArray<T>::Data() {
  return buffer_;
}

template <typename T>
inline const T*
ContiguousArray<T>::Data() const {
  return buffer_;
}

template <typename T>
inline T*
ContiguousArray<T>::End() {
  return &buffer_[Size()];
}

template <typename T>
inline const T*
ContiguousArray<T>::End() const {
  return &buffer_[Size()];
}

template <typename T>
inline T&
ContiguousArray<T>::Back() {
  return buffer_[Size() - 1];
}

template <typename T>
inline const T&
ContiguousArray<T>::Back() const {
  return buffer_[Size() - 1];
}

template <typename T>
inline size_t
ContiguousArray<T>::Size() const {
  return size_;
}

template <typename T>
inline void
ContiguousArray<T>::Reserve(size_t newsize) {
  if (!owned_buffer_) {
    throw std::runtime_error("Cannot resize when using a foreign buffer; clone first");
  }
  T* newbuf = static_cast<T*>(std::realloc(static_cast<void*>(buffer_), sizeof(T) * newsize));
  if (!newbuf) {
    throw std::runtime_error("Could not expand buffer");
  }
  buffer_ = newbuf;
  capacity_ = newsize;
}

template <typename T>
inline void
ContiguousArray<T>::Resize(size_t newsize) {
  if (!owned_buffer_) {
    throw std::runtime_error("Cannot resize when using a foreign buffer; clone first");
  }
  if (newsize > capacity_) {
    size_t newcapacity = capacity_;
    if (newcapacity == 0) {
      newcapacity = 1;
    }
    while (newcapacity <= newsize) {
      newcapacity *= 2;
    }
    Reserve(newcapacity);
  }
  size_ = newsize;
}

template <typename T>
inline void
ContiguousArray<T>::Resize(size_t newsize, T t) {
  if (!owned_buffer_) {
    throw std::runtime_error("Cannot resize when using a foreign buffer; clone first");
  }
  size_t oldsize = Size();
  Resize(newsize);
  for (size_t i = oldsize; i < newsize; ++i) {
    buffer_[i] = t;
  }
}

template <typename T>
inline void
ContiguousArray<T>::Clear() {
  if (!owned_buffer_) {
    throw std::runtime_error("Cannot clear when using a foreign buffer; clone first");
  }
  Resize(0);
}

template <typename T>
inline void
ContiguousArray<T>::PushBack(T t) {
  if (!owned_buffer_) {
    throw std::runtime_error("Cannot add element when using a foreign buffer; clone first");
  }
  if (size_ == capacity_) {
    Reserve(capacity_ * 2);
  }
  buffer_[size_++] = t;
}

template <typename T>
inline void
ContiguousArray<T>::Extend(const std::vector<T>& other) {
  if (!owned_buffer_) {
    throw std::runtime_error("Cannot add elements when using a foreign buffer; clone first");
  }
  size_t newsize = size_ + other.size();
  if (newsize > capacity_) {
    size_t newcapacity = capacity_;
    if (newcapacity == 0) {
      newcapacity = 1;
    }
    while (newcapacity <= newsize) {
      newcapacity *= 2;
    }
    Reserve(newcapacity);
  }
  std::memcpy(&buffer_[size_], static_cast<const void*>(other.data()), sizeof(T) * other.size());
  size_ = newsize;
}

template <typename T>
inline T&
ContiguousArray<T>::operator[](size_t idx) {
  return buffer_[idx];
}

template <typename T>
inline const T&
ContiguousArray<T>::operator[](size_t idx) const {
  return buffer_[idx];
}

template<typename Container>
inline std::vector<std::pair<std::string, std::string> >
ModelParam::InitAllowUnknown(const Container& kwargs) {
  std::vector<std::pair<std::string, std::string>> unknowns;
  for (const auto& e : kwargs) {
    if (e.first == "pred_transform") {
      std::strncpy(this->pred_transform, e.second.c_str(),
                   TREELITE_MAX_PRED_TRANSFORM_LENGTH - 1);
      this->pred_transform[TREELITE_MAX_PRED_TRANSFORM_LENGTH - 1] = '\0';
    } else if (e.first == "sigmoid_alpha") {
      this->sigmoid_alpha = dmlc::stof(e.second, nullptr);
    } else if (e.first == "global_bias") {
      this->global_bias = dmlc::stof(e.second, nullptr);
    }
  }
  return unknowns;
}

inline std::map<std::string, std::string>
ModelParam::__DICT__() const {
  std::map<std::string, std::string> ret;
  ret.emplace("pred_transform", std::string(this->pred_transform));
  ret.emplace("sigmoid_alpha", GetString(this->sigmoid_alpha));
  ret.emplace("global_bias", GetString(this->global_bias));
  return ret;
}

inline PyBufferFrame GetPyBufferFromArray(void* data, const char* format,
                                          size_t itemsize, size_t nitem) {
  return PyBufferFrame{data, const_cast<char*>(format), itemsize, nitem};
}

// Infer format string from data type
template <typename T>
inline const char* InferFormatString() {
  switch (sizeof(T)) {
  case 1:
    return (std::is_unsigned<T>::value ? "=B" : "=b");
  case 2:
    return (std::is_unsigned<T>::value ? "=H" : "=h");
  case 4:
    if (std::is_integral<T>::value) {
      return (std::is_unsigned<T>::value ? "=L" : "=l");
    } else {
      if (!std::is_floating_point<T>::value) {
        throw std::runtime_error("Could not infer format string");
      }
      return "=f";
    }
  case 8:
    if (std::is_integral<T>::value) {
      return (std::is_unsigned<T>::value ? "=Q" : "=q");
    } else {
      if (!std::is_floating_point<T>::value) {
        throw std::runtime_error("Could not infer format string");
      }
      return "=d";
    }
  default:
    throw std::runtime_error("Unrecognized type");
  }
  return nullptr;
}

template <typename T>
inline PyBufferFrame GetPyBufferFromArray(ContiguousArray<T>* vec, const char* format) {
  return GetPyBufferFromArray(static_cast<void*>(vec->Data()), format, sizeof(T), vec->Size());
}

template <typename T>
inline PyBufferFrame GetPyBufferFromArray(ContiguousArray<T>* vec) {
  static_assert(std::is_arithmetic<T>::value,
      "Use GetPyBufferFromArray(vec, format) for composite types; specify format string manually");
  return GetPyBufferFromArray(vec, InferFormatString<T>());
}

inline PyBufferFrame GetPyBufferFromScalar(void* data, const char* format, size_t itemsize) {
  return GetPyBufferFromArray(data, format, itemsize, 1);
}

template <typename T>
inline PyBufferFrame GetPyBufferFromScalar(T* scalar, const char* format) {
  return GetPyBufferFromScalar(static_cast<void*>(scalar), format, sizeof(T));
}

template <typename T>
inline PyBufferFrame GetPyBufferFromScalar(T* scalar) {
  static_assert(std::is_arithmetic<T>::value,
                "Use GetPyBufferFromScalar(scalar, format) for composite types; "
                "specify format string manually");
  return GetPyBufferFromScalar(scalar, InferFormatString<T>());
}

template <typename T>
inline void InitArrayFromPyBuffer(ContiguousArray<T>* vec, PyBufferFrame buffer) {
  if (sizeof(T) != buffer.itemsize) {
    throw std::runtime_error("Incorrect itemsize");
  }
  vec->UseForeignBuffer(buffer.buf, buffer.nitem);
}

template <typename T>
inline void InitScalarFromPyBuffer(T* scalar, PyBufferFrame buffer) {
  if (sizeof(T) != buffer.itemsize) {
    throw std::runtime_error("Incorrect itemsize");
  }
  if (buffer.nitem != 1) {
    throw std::runtime_error("nitem must be 1 for a scalar");
  }
  T* t = static_cast<T*>(buffer.buf);
  *scalar = *t;
}

template <typename ThresholdType, typename LeafOutputType>
inline const char*
Tree<ThresholdType, LeafOutputType>::GetFormatStringForNode() {
  if (std::is_same<ThresholdType, float>::value) {
    return "T{=l=l=L=f=Q=d=d=b=b=?=?=?=?xx}";
  } else {
    return "T{=l=l=Lxxxx=d=Q=d=d=b=b=?=?=?=?xx}";
  }
}

constexpr size_t kNumFramePerTree = 6;

template <typename ThresholdType, typename LeafOutputType>
inline std::vector<PyBufferFrame>
Tree<ThresholdType, LeafOutputType>::GetPyBuffer() {
  return {
      GetPyBufferFromScalar(&num_nodes),
      GetPyBufferFromArray(&nodes_, GetFormatStringForNode()),
      GetPyBufferFromArray(&leaf_vector_),
      GetPyBufferFromArray(&leaf_vector_offset_),
      GetPyBufferFromArray(&left_categories_),
      GetPyBufferFromArray(&left_categories_offset_)
  };
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::InitFromPyBuffer(std::vector<PyBufferFrame> frames) {
  size_t frame_id = 0;
  InitScalarFromPyBuffer(&num_nodes, frames[frame_id++]);
  InitArrayFromPyBuffer(&nodes_, frames[frame_id++]);
  if (num_nodes != nodes_.Size()) {
    throw std::runtime_error("Could not load the correct number of nodes");
  }
  InitArrayFromPyBuffer(&leaf_vector_, frames[frame_id++]);
  InitArrayFromPyBuffer(&leaf_vector_offset_, frames[frame_id++]);
  InitArrayFromPyBuffer(&left_categories_, frames[frame_id++]);
  InitArrayFromPyBuffer(&left_categories_offset_, frames[frame_id++]);
  if (frame_id != kNumFramePerTree) {
    throw std::runtime_error("Wrong number of frames loaded");
  }
}

template <typename ThresholdType, typename LeafOutputType>
inline std::vector<PyBufferFrame>
ModelImpl<ThresholdType, LeafOutputType>::GetPyBuffer() {
  /* Header */
  std::vector<PyBufferFrame> frames{
      GetPyBufferFromScalar(&num_feature),
      GetPyBufferFromScalar(&num_output_group),
      GetPyBufferFromScalar(&random_forest_flag),
      GetPyBufferFromScalar(&param, "T{" _TREELITE_STR(TREELITE_MAX_PRED_TRANSFORM_LENGTH) "s=f=f}")
  };

  /* Body */
  for (Tree<ThresholdType, LeafOutputType>& tree : trees) {
    auto tree_frames = tree.GetPyBuffer();
    frames.insert(frames.end(), tree_frames.begin(), tree_frames.end());
  }
  return frames;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
ModelImpl<ThresholdType, LeafOutputType>::InitFromPyBuffer(std::vector<PyBufferFrame> frames) {
  /* Header */
  size_t frame_id = 0;
  InitScalarFromPyBuffer(&num_feature, frames[frame_id++]);
  InitScalarFromPyBuffer(&num_output_group, frames[frame_id++]);
  InitScalarFromPyBuffer(&random_forest_flag, frames[frame_id++]);
  InitScalarFromPyBuffer(&param, frames[frame_id++]);
  /* Body */
  const size_t num_frame = frames.size();
  if ((num_frame - frame_id) % kNumFramePerTree != 0) {
    throw std::runtime_error("Wrong number of frames");
  }
  trees.clear();
  for (; frame_id < num_frame; frame_id += kNumFramePerTree) {
    std::vector<PyBufferFrame> tree_frames(frames.begin() + frame_id,
                                           frames.begin() + frame_id + kNumFramePerTree);
    trees.emplace_back();
    trees.back().InitFromPyBuffer(tree_frames);
  }
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::Node::Init() {
  cleft_ = cright_ = -1;
  sindex_ = 0;
  info_.leaf_value = static_cast<LeafOutputType>(0);
  info_.threshold = static_cast<ThresholdType>(0);
  data_count_ = 0;
  sum_hess_ = gain_ = 0.0;
  missing_category_to_zero_ = false;
  data_count_present_ = sum_hess_present_ = gain_present_ = false;
  split_type_ = SplitFeatureType::kNone;
  cmp_ = Operator::kNone;
}

template <typename ThresholdType, typename LeafOutputType>
inline int
Tree<ThresholdType, LeafOutputType>::AllocNode() {
  int nd = num_nodes++;
  if (nodes_.Size() != static_cast<size_t>(nd)) {
    throw std::runtime_error("Invariant violated: nodes_ contains incorrect number of nodes");
  }
  for (int nid = nd; nid < num_nodes; ++nid) {
    leaf_vector_offset_.PushBack(leaf_vector_offset_.Back());
    left_categories_offset_.PushBack(left_categories_offset_.Back());
    nodes_.Resize(nodes_.Size() + 1);
    nodes_.Back().Init();
  }
  return nd;
}

template <typename ThresholdType, typename LeafOutputType>
inline Tree<ThresholdType, LeafOutputType>
Tree<ThresholdType, LeafOutputType>::Clone() const {
  Tree<ThresholdType, LeafOutputType> tree;
  tree.num_nodes = num_nodes;
  tree.nodes_ = nodes_.Clone();
  tree.leaf_vector_ = leaf_vector_.Clone();
  tree.leaf_vector_offset_ = leaf_vector_offset_.Clone();
  tree.left_categories_ = left_categories_.Clone();
  tree.left_categories_offset_ = left_categories_offset_.Clone();
  return tree;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::Init() {
  num_nodes = 1;
  leaf_vector_.Clear();
  leaf_vector_offset_.Resize(2, 0);
  left_categories_.Clear();
  left_categories_offset_.Resize(2, 0);
  nodes_.Resize(1);
  nodes_[0].Init();
  SetLeaf(0, 0.0f);
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::AddChilds(int nid) {
  const int cleft = this->AllocNode();
  const int cright = this->AllocNode();
  nodes_[nid].cleft_ = cleft;
  nodes_[nid].cright_ = cright;
}

template <typename ThresholdType, typename LeafOutputType>
inline std::vector<unsigned>
Tree<ThresholdType, LeafOutputType>::GetCategoricalFeatures() const {
  std::unordered_map<unsigned, bool> tmp;
  for (int nid = 0; nid < num_nodes; ++nid) {
    const SplitFeatureType type = SplitType(nid);
    if (type != SplitFeatureType::kNone) {
      const bool flag = (type == SplitFeatureType::kCategorical);
      const uint32_t split_index = SplitIndex(nid);
      if (tmp.count(split_index) == 0) {
        tmp[split_index] = flag;
      } else {
        if (tmp[split_index] != flag) {
          throw std::runtime_error("Feature " + std::to_string(split_index) +
                                   " cannot be simultaneously be categorical and numerical.");
        }
      }
    }
  }
  std::vector<unsigned> result;
  for (const auto& kv : tmp) {
    if (kv.second) {
      result.push_back(kv.first);
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

template <typename ThresholdType, typename LeafOutputType>
inline int
Tree<ThresholdType, LeafOutputType>::LeftChild(int nid) const {
  return nodes_[nid].cleft_;
}

template <typename ThresholdType, typename LeafOutputType>
inline int
Tree<ThresholdType, LeafOutputType>::RightChild(int nid) const {
  return nodes_[nid].cright_;
}

template <typename ThresholdType, typename LeafOutputType>
inline int
Tree<ThresholdType, LeafOutputType>::DefaultChild(int nid) const {
  return DefaultLeft(nid) ? LeftChild(nid) : RightChild(nid);
}

template <typename ThresholdType, typename LeafOutputType>
inline uint32_t
Tree<ThresholdType, LeafOutputType>::SplitIndex(int nid) const {
  return (nodes_[nid].sindex_ & ((1U << 31U) - 1U));
}

template <typename ThresholdType, typename LeafOutputType>
inline bool
Tree<ThresholdType, LeafOutputType>::DefaultLeft(int nid) const {
  return (nodes_[nid].sindex_ >> 31U) != 0;
}

template <typename ThresholdType, typename LeafOutputType>
inline bool
Tree<ThresholdType, LeafOutputType>::IsLeaf(int nid) const {
  return nodes_[nid].cleft_ == -1;
}

template <typename ThresholdType, typename LeafOutputType>
inline LeafOutputType
Tree<ThresholdType, LeafOutputType>::LeafValue(int nid) const {
  return (nodes_[nid].info_).leaf_value;
}

template <typename ThresholdType, typename LeafOutputType>
inline std::vector<LeafOutputType>
Tree<ThresholdType, LeafOutputType>::LeafVector(int nid) const {
  if (nid > leaf_vector_offset_.Size()) {
    throw std::runtime_error("nid too large");
  }
  return std::vector<LeafOutputType>(&leaf_vector_[leaf_vector_offset_[nid]],
                                     &leaf_vector_[leaf_vector_offset_[nid + 1]]);
}

template <typename ThresholdType, typename LeafOutputType>
inline bool
Tree<ThresholdType, LeafOutputType>::HasLeafVector(int nid) const {
  if (nid > leaf_vector_offset_.Size()) {
    throw std::runtime_error("nid too large");
  }
  return leaf_vector_offset_[nid] != leaf_vector_offset_[nid + 1];
}

template <typename ThresholdType, typename LeafOutputType>
inline ThresholdType
Tree<ThresholdType, LeafOutputType>::Threshold(int nid) const {
  return (nodes_[nid].info_).threshold;
}

template <typename ThresholdType, typename LeafOutputType>
inline Operator
Tree<ThresholdType, LeafOutputType>::ComparisonOp(int nid) const {
  return nodes_[nid].cmp_;
}

template <typename ThresholdType, typename LeafOutputType>
inline std::vector<uint32_t>
Tree<ThresholdType, LeafOutputType>::LeftCategories(int nid) const {
  if (nid > left_categories_offset_.Size()) {
    throw std::runtime_error("nid too large");
  }
  return std::vector<uint32_t>(&left_categories_[left_categories_offset_[nid]],
                               &left_categories_[left_categories_offset_[nid + 1]]);
}

template <typename ThresholdType, typename LeafOutputType>
inline SplitFeatureType
Tree<ThresholdType, LeafOutputType>::SplitType(int nid) const {
  return nodes_[nid].split_type_;
}

template <typename ThresholdType, typename LeafOutputType>
inline bool
Tree<ThresholdType, LeafOutputType>::HasDataCount(int nid) const {
  return nodes_[nid].data_count_present_;
}

template <typename ThresholdType, typename LeafOutputType>
inline uint64_t
Tree<ThresholdType, LeafOutputType>::DataCount(int nid) const {
  return nodes_[nid].data_count_;
}

template <typename ThresholdType, typename LeafOutputType>
inline bool
Tree<ThresholdType, LeafOutputType>::HasSumHess(int nid) const {
  return nodes_[nid].sum_hess_present_;
}

template <typename ThresholdType, typename LeafOutputType>
inline double
Tree<ThresholdType, LeafOutputType>::SumHess(int nid) const {
  return nodes_[nid].sum_hess_;
}

template <typename ThresholdType, typename LeafOutputType>
inline bool
Tree<ThresholdType, LeafOutputType>::HasGain(int nid) const {
  return nodes_[nid].gain_present_;
}

template <typename ThresholdType, typename LeafOutputType>
inline double
Tree<ThresholdType, LeafOutputType>::Gain(int nid) const {
  return nodes_[nid].gain_;
}

template <typename ThresholdType, typename LeafOutputType>
inline bool
Tree<ThresholdType, LeafOutputType>::MissingCategoryToZero(int nid) const {
  return nodes_[nid].missing_category_to_zero_;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::SetNumericalSplit(
    int nid, unsigned split_index, ThresholdType threshold, bool default_left, Operator cmp) {
  Node& node = nodes_[nid];
  if (split_index >= ((1U << 31U) - 1)) {
    throw std::runtime_error("split_index too big");
  }
  if (default_left) split_index |= (1U << 31U);
  node.sindex_ = split_index;
  (node.info_).threshold = threshold;
  node.cmp_ = cmp;
  node.split_type_ = SplitFeatureType::kNumerical;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::SetCategoricalSplit(
    int nid, unsigned split_index, bool default_left, bool missing_category_to_zero,
    const std::vector<uint32_t>& node_left_categories) {
  if (split_index >= ((1U << 31U) - 1)) {
    throw std::runtime_error("split_index too big");
  }

  const size_t end_oft = left_categories_offset_.Back();
  const size_t new_end_oft = end_oft + node_left_categories.size();
  if (end_oft != left_categories_.Size()) {
    throw std::runtime_error("Invariant violated");
  }
  if (!std::all_of(&left_categories_offset_[nid + 1], left_categories_offset_.End(),
                   [end_oft](size_t x) { return (x == end_oft); })) {
    throw std::runtime_error("Invariant violated");
  }
  // Hopefully we won't have to move any element as we add node_left_categories for node nid
  left_categories_.Extend(node_left_categories);
  if (new_end_oft != left_categories_.Size()) {
    throw std::runtime_error("Invariant violated");
  }
  std::for_each(&left_categories_offset_[nid + 1], left_categories_offset_.End(),
                [new_end_oft](size_t& x) { x = new_end_oft; });
  std::sort(&left_categories_[end_oft], left_categories_.End());

  Node& node = nodes_[nid];
  if (default_left) split_index |= (1U << 31U);
  node.sindex_ = split_index;
  node.split_type_ = SplitFeatureType::kCategorical;
  node.missing_category_to_zero_ = missing_category_to_zero;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::SetLeaf(int nid, LeafOutputType value) {
  Node& node = nodes_[nid];
  (node.info_).leaf_value = value;
  node.cleft_ = -1;
  node.cright_ = -1;
  node.split_type_ = SplitFeatureType::kNone;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::SetLeafVector(
    int nid, const std::vector<ThresholdType>& node_leaf_vector) {
  const size_t end_oft = leaf_vector_offset_.Back();
  const size_t new_end_oft = end_oft + node_leaf_vector.size();
  if (end_oft != leaf_vector_.Size()) {
    throw std::runtime_error("Invariant violated");
  }
  if (!std::all_of(&leaf_vector_offset_[nid + 1], leaf_vector_offset_.End(),
                   [end_oft](size_t x) { return (x == end_oft); })) {
    throw std::runtime_error("Invariant violated");
  }
  // Hopefully we won't have to move any element as we add leaf vector elements for node nid
  leaf_vector_.Extend(node_leaf_vector);
  if (new_end_oft != leaf_vector_.Size()) {
    throw std::runtime_error("Invariant violated");
  }
  std::for_each(&leaf_vector_offset_[nid + 1], leaf_vector_offset_.End(),
                [new_end_oft](size_t& x) { x = new_end_oft; });

  Node& node = nodes_[nid];
  node.cleft_ = -1;
  node.cright_ = -1;
  node.split_type_ = SplitFeatureType::kNone;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::SetSumHess(int nid, double sum_hess) {
  Node& node = nodes_[nid];
  node.sum_hess_ = sum_hess;
  node.sum_hess_present_ = true;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::SetDataCount(int nid, uint64_t data_count) {
  Node& node = nodes_[nid];
  node.data_count_ = data_count;
  node.data_count_present_ = true;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::SetGain(int nid, double gain) {
  Node& node = nodes_[nid];
  node.gain_ = gain;
  node.gain_present_ = true;
}

template <typename ThresholdType, typename LeafOutputType>
inline ModelImpl<ThresholdType, LeafOutputType>
ModelImpl<ThresholdType, LeafOutputType>::Clone() const {
  ModelImpl<ThresholdType, LeafOutputType> model;
  for (const Tree<ThresholdType, LeafOutputType>& t : trees) {
    model.trees.push_back(t.Clone());
  }
  model.num_feature = num_feature;
  model.num_output_group = num_output_group;
  model.random_forest_flag = random_forest_flag;
  model.param = param;
  return model;
}

template <typename ThresholdType, typename LeafOutputType>
inline ModelImpl<ThresholdType, LeafOutputType>&
Model::GetImpl() {
  return *static_cast<ModelImpl<ThresholdType, LeafOutputType>*>(handle_.get());
}

template <typename ThresholdType, typename LeafOutputType>
inline const ModelImpl<ThresholdType, LeafOutputType>&
Model::GetImpl() const {
  return *static_cast<const ModelImpl<ThresholdType, LeafOutputType>*>(handle_.get());
}

inline ModelType
Model::GetModelType() const {
  return type_;
}

template <typename ThresholdType, typename LeafOutputType>
inline Model
Model::Create() {
  Model model;
  model.handle_.reset(new ModelImpl<ThresholdType, LeafOutputType>());
  model.type_ = ModelType::kInvalid;

  const char* error_msg = "Unsupported combination of ThresholdType and LeafOutputType";
  static_assert(std::is_same<ThresholdType, float>::value
                || std::is_same<ThresholdType, double>::value,
                "ThresholdType should be either float32 or float64");
  static_assert(std::is_same<LeafOutputType, uint32_t>::value
                || std::is_same<LeafOutputType, float>::value
                || std::is_same<LeafOutputType, double>::value,
                "LeafOutputType should be uint32, float32 or float64");
  if (std::is_same<ThresholdType, float>::value) {
    if (std::is_same<LeafOutputType, uint32_t>::value) {
      model.type_ = ModelType::kFloat32ThresholdUInt32LeafOutput;
    } else if (std::is_same<LeafOutputType, float>::value) {
      model.type_ = ModelType::kFloat32ThresholdFloat32LeafOutput;
    } else {
      throw std::runtime_error(error_msg);
    }
  } else if (std::is_same<ThresholdType, double>::value) {
    if (std::is_same<LeafOutputType, uint32_t>::value) {
      model.type_ = ModelType::kFloat64ThresholdUint32LeafOutput;
    } else if (std::is_same<LeafOutputType, double>::value) {
      model.type_ = ModelType::kFloat64ThresholdFloat64LeafOutput;
    } else {
      throw std::runtime_error(error_msg);
    }
  } else {
    throw std::runtime_error(error_msg);
  }
  return model;
}

template <typename Func>
inline auto
Model::Dispatch(Func func) const {
  switch(type_) {
  case ModelType::kFloat32ThresholdUInt32LeafOutput:
    return func(GetImpl<float, uint32_t>());
  case ModelType::kFloat32ThresholdFloat32LeafOutput:
    return func(GetImpl<float, float>());
  case ModelType::kFloat64ThresholdUint32LeafOutput:
    return func(GetImpl<double, uint32_t>());
  case ModelType::kFloat64ThresholdFloat64LeafOutput:
    return func(GetImpl<double, double>());
  default:
    throw std::runtime_error("Unknown type name");
    return func(GetImpl<double, double>());  // avoid "missing return" warning
  }
}

template <typename Func>
inline auto
Model::Dispatch(Func func) {
  switch(type_) {
  case ModelType::kFloat32ThresholdUInt32LeafOutput:
    return func(GetImpl<float, uint32_t>());
  case ModelType::kFloat32ThresholdFloat32LeafOutput:
    return func(GetImpl<float, float>());
  case ModelType::kFloat64ThresholdUint32LeafOutput:
    return func(GetImpl<double, uint32_t>());
  case ModelType::kFloat64ThresholdFloat64LeafOutput:
    return func(GetImpl<double, double>());
  default:
    throw std::runtime_error("Unknown type name");
    return func(GetImpl<double, double>());  // avoid "missing return" warning
  }
}

inline ModelParam
Model::GetParam() const {
  return Dispatch([](const auto& handle) { return handle.param; });
}

inline int
Model::GetNumFeature() const {
  return Dispatch([](const auto& handle) { return handle.num_feature; });
}

inline int
Model::GetNumOutputGroup() const {
  return Dispatch([](const auto& handle) { return handle.num_output_group; });
}

inline bool
Model::GetRandomForestFlag() const {
  return Dispatch([](const auto& handle) { return handle.random_forest_flag; });
}

inline size_t
Model::GetNumTree() const {
  return Dispatch([](const auto& handle) { return handle.trees.size(); });
}

inline void
Model::SetTreeLimit(size_t limit) {
  Dispatch([limit](auto& handle) { handle.trees.resize(limit); });
}

inline void
Model::ReferenceSerialize(dmlc::Stream* fo) const {
  Dispatch([fo](const auto& handle) { handle.ReferenceSerialize(fo); });
}

inline std::vector<PyBufferFrame>
Model::GetPyBuffer() {
  return Dispatch([](auto& handle) { return handle.GetPyBuffer(); });
}

inline void
Model::InitFromPyBuffer(std::vector<PyBufferFrame> frames) {
  Dispatch([&frames](auto& handle) { handle.InitFromPyBuffer(frames); });
}

inline void InitParamAndCheck(ModelParam* param,
                              const std::vector<std::pair<std::string, std::string>>& cfg) {
  auto unknown = param->InitAllowUnknown(cfg);
  if (!unknown.empty()) {
    std::ostringstream oss;
    for (const auto& kv : unknown) {
      oss << kv.first << ", ";
    }
    std::cerr << "\033[1;31mWarning: Unknown parameters found; "
              << "they have been ignored\u001B[0m: " << oss.str() << std::endl;
  }
}

}  // namespace treelite
#endif  // TREELITE_TREE_IMPL_H_
