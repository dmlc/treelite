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

constexpr size_t kNumFramePerTree = 6;

inline std::vector<PyBufferFrame>
Tree::GetPyBuffer() {
  return {
      GetPyBufferFromScalar(&num_nodes),
      GetPyBufferFromArray(&nodes_, "T{=l=l=L=f=Q=d=d=b=b=?=?=?=?=H}"),
      GetPyBufferFromArray(&leaf_vector_),
      GetPyBufferFromArray(&leaf_vector_offset_),
      GetPyBufferFromArray(&left_categories_),
      GetPyBufferFromArray(&left_categories_offset_)
  };
}

inline void
Tree::InitFromPyBuffer(std::vector<PyBufferFrame> frames) {
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

inline std::vector<PyBufferFrame>
Model::GetPyBuffer() {
  /* Header */
  std::vector<PyBufferFrame> frames{
      GetPyBufferFromScalar(&num_feature),
      GetPyBufferFromScalar(&num_output_group),
      GetPyBufferFromScalar(&random_forest_flag),
      GetPyBufferFromScalar(&param, "T{" _TREELITE_STR(TREELITE_MAX_PRED_TRANSFORM_LENGTH) "s=f=f}")
  };

  /* Body */
  for (auto& tree : trees) {
    auto tree_frames = tree.GetPyBuffer();
    frames.insert(frames.end(), tree_frames.begin(), tree_frames.end());
  }
  return frames;
}

inline void
Model::InitFromPyBuffer(std::vector<PyBufferFrame> frames) {
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

inline void Tree::Node::Init() {
  cleft_ = cright_ = -1;
  sindex_ = 0;
  info_.leaf_value = 0.0f;
  info_.threshold = 0.0f;
  data_count_ = 0;
  sum_hess_ = gain_ = 0.0;
  missing_category_to_zero_ = false;
  data_count_present_ = sum_hess_present_ = gain_present_ = false;
  split_type_ = SplitFeatureType::kNone;
  cmp_ = Operator::kNone;
}

inline int
Tree::AllocNode() {
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

inline Tree
Tree::Clone() const {
  Tree tree;
  tree.num_nodes = num_nodes;
  tree.nodes_ = nodes_.Clone();
  tree.leaf_vector_ = leaf_vector_.Clone();
  tree.leaf_vector_offset_ = leaf_vector_offset_.Clone();
  tree.left_categories_ = left_categories_.Clone();
  tree.left_categories_offset_ = left_categories_offset_.Clone();
  return tree;
}

inline void
Tree::Init() {
  num_nodes = 1;
  leaf_vector_.Clear();
  leaf_vector_offset_.Resize(2, 0);
  left_categories_.Clear();
  left_categories_offset_.Resize(2, 0);
  nodes_.Resize(1);
  nodes_[0].Init();
  SetLeaf(0, 0.0f);
}

inline void
Tree::AddChilds(int nid) {
  const int cleft = this->AllocNode();
  const int cright = this->AllocNode();
  nodes_[nid].cleft_ = cleft;
  nodes_[nid].cright_ = cright;
}

inline std::vector<unsigned>
Tree::GetCategoricalFeatures() const {
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

inline int
Tree::LeftChild(int nid) const {
  return nodes_[nid].cleft_;
}

inline int
Tree::RightChild(int nid) const {
  return nodes_[nid].cright_;
}

inline int
Tree::DefaultChild(int nid) const {
  return DefaultLeft(nid) ? LeftChild(nid) : RightChild(nid);
}

inline uint32_t
Tree::SplitIndex(int nid) const {
  return (nodes_[nid].sindex_ & ((1U << 31U) - 1U));
}

inline bool
Tree::DefaultLeft(int nid) const {
  return (nodes_[nid].sindex_ >> 31U) != 0;
}

inline bool
Tree::IsLeaf(int nid) const {
  return nodes_[nid].cleft_ == -1;
}

inline tl_float
Tree::LeafValue(int nid) const {
  return (nodes_[nid].info_).leaf_value;
}

inline std::vector<tl_float>
Tree::LeafVector(int nid) const {
  if (nid > leaf_vector_offset_.Size()) {
    throw std::runtime_error("nid too large");
  }
  return std::vector<tl_float>(&leaf_vector_[leaf_vector_offset_[nid]],
                               &leaf_vector_[leaf_vector_offset_[nid + 1]]);
}

inline bool
Tree::HasLeafVector(int nid) const {
  if (nid > leaf_vector_offset_.Size()) {
    throw std::runtime_error("nid too large");
  }
  return leaf_vector_offset_[nid] != leaf_vector_offset_[nid + 1];
}

inline tl_float
Tree::Threshold(int nid) const {
  return (nodes_[nid].info_).threshold;
}

inline Operator
Tree::ComparisonOp(int nid) const {
  return nodes_[nid].cmp_;
}

inline std::vector<uint32_t>
Tree::LeftCategories(int nid) const {
  if (nid > left_categories_offset_.Size()) {
    throw std::runtime_error("nid too large");
  }
  return std::vector<uint32_t>(&left_categories_[left_categories_offset_[nid]],
                               &left_categories_[left_categories_offset_[nid + 1]]);
}

inline SplitFeatureType
Tree::SplitType(int nid) const {
  return nodes_[nid].split_type_;
}

inline bool
Tree::HasDataCount(int nid) const {
  return nodes_[nid].data_count_present_;
}

inline uint64_t
Tree::DataCount(int nid) const {
  return nodes_[nid].data_count_;
}

inline bool
Tree::HasSumHess(int nid) const {
  return nodes_[nid].sum_hess_present_;
}

inline double
Tree::SumHess(int nid) const {
  return nodes_[nid].sum_hess_;
}

inline bool
Tree::HasGain(int nid) const {
  return nodes_[nid].gain_present_;
}

inline double
Tree::Gain(int nid) const {
  return nodes_[nid].gain_;
}

inline bool
Tree::MissingCategoryToZero(int nid) const {
  return nodes_[nid].missing_category_to_zero_;
}

inline void
Tree::SetNumericalSplit(int nid, unsigned split_index, tl_float threshold,
                        bool default_left, Operator cmp) {
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

inline void
Tree::SetCategoricalSplit(int nid, unsigned split_index, bool default_left,
                          bool missing_category_to_zero,
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

inline void
Tree::SetLeaf(int nid, tl_float value) {
  Node& node = nodes_[nid];
  (node.info_).leaf_value = value;
  node.cleft_ = -1;
  node.cright_ = -1;
  node.split_type_ = SplitFeatureType::kNone;
}

inline void
Tree::SetLeafVector(int nid, const std::vector<tl_float>& node_leaf_vector) {
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

inline void
Tree::SetSumHess(int nid, double sum_hess) {
  Node& node = nodes_[nid];
  node.sum_hess_ = sum_hess;
  node.sum_hess_present_ = true;
}

inline void
Tree::SetDataCount(int nid, uint64_t data_count) {
  Node& node = nodes_[nid];
  node.data_count_ = data_count;
  node.data_count_present_ = true;
}

inline void
Tree::SetGain(int nid, double gain) {
  Node& node = nodes_[nid];
  node.gain_ = gain;
  node.gain_present_ = true;
}

inline Model
Model::Clone() const {
  Model model;
  for (const Tree& t : trees) {
    model.trees.push_back(t.Clone());
  }
  model.num_feature = num_feature;
  model.num_output_group = num_output_group;
  model.random_forest_flag = random_forest_flag;
  model.param = param;
  return model;
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
