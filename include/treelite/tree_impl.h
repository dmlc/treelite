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
#include <memory>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <typeinfo>
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

inline PyBufferFrame GetPyBufferFromScalar(TypeInfo* scalar) {
  using T = std::underlying_type<TypeInfo>::type;
  return GetPyBufferFromScalar(reinterpret_cast<T*>(scalar), InferFormatString<T>());
}

inline PyBufferFrame GetPyBufferFromScalar(TaskType* scalar) {
  using T = std::underlying_type<TaskType>::type;
  return GetPyBufferFromScalar(reinterpret_cast<T*>(scalar), InferFormatString<T>());
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

inline void InitScalarFromPyBuffer(TypeInfo* scalar, PyBufferFrame buffer) {
  using T = std::underlying_type<TypeInfo>::type;
  if (sizeof(T) != buffer.itemsize) {
    throw std::runtime_error("Incorrect itemsize");
  }
  if (buffer.nitem != 1) {
    throw std::runtime_error("nitem must be 1 for a scalar");
  }
  T* t = static_cast<T*>(buffer.buf);
  *scalar = static_cast<TypeInfo>(*t);
}

inline void InitScalarFromPyBuffer(TaskType* scalar, PyBufferFrame buffer) {
  using T = std::underlying_type<TaskType>::type;
  if (sizeof(T) != buffer.itemsize) {
    throw std::runtime_error("Incorrect itemsize");
  }
  if (buffer.nitem != 1) {
    throw std::runtime_error("nitem must be 1 for a scalar");
  }
  T* t = static_cast<T*>(buffer.buf);
  *scalar = static_cast<TaskType>(*t);
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
inline void
Tree<ThresholdType, LeafOutputType>::GetPyBuffer(std::vector<PyBufferFrame>* dest) {
  dest->push_back(GetPyBufferFromScalar(&num_nodes));
  dest->push_back(GetPyBufferFromArray(&nodes_, GetFormatStringForNode()));
  dest->push_back(GetPyBufferFromArray(&leaf_vector_));
  dest->push_back(GetPyBufferFromArray(&leaf_vector_offset_));
  dest->push_back(GetPyBufferFromArray(&matching_categories_));
  dest->push_back(GetPyBufferFromArray(&matching_categories_offset_));
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::InitFromPyBuffer(
    std::vector<PyBufferFrame>::iterator begin, std::vector<PyBufferFrame>::iterator end) {
  if (std::distance(begin, end) != kNumFramePerTree) {
    throw std::runtime_error("Wrong number of frames specified");
  }
  InitScalarFromPyBuffer(&num_nodes, *begin++);
  InitArrayFromPyBuffer(&nodes_, *begin++);
  if (num_nodes != nodes_.Size()) {
    throw std::runtime_error("Could not load the correct number of nodes");
  }
  InitArrayFromPyBuffer(&leaf_vector_, *begin++);
  InitArrayFromPyBuffer(&leaf_vector_offset_, *begin++);
  InitArrayFromPyBuffer(&matching_categories_, *begin++);
  InitArrayFromPyBuffer(&matching_categories_offset_, *begin++);
}

template <typename ThresholdType, typename LeafOutputType>
inline void Tree<ThresholdType, LeafOutputType>::Node::Init() {
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
inline int
Tree<ThresholdType, LeafOutputType>::AllocNode() {
  int nd = num_nodes++;
  if (nodes_.Size() != static_cast<size_t>(nd)) {
    throw std::runtime_error("Invariant violated: nodes_ contains incorrect number of nodes");
  }
  for (int nid = nd; nid < num_nodes; ++nid) {
    leaf_vector_offset_.PushBack(leaf_vector_offset_.Back());
    matching_categories_offset_.PushBack(matching_categories_offset_.Back());
    nodes_.Resize(nodes_.Size() + 1);
    nodes_.Back().Init();
  }
  return nd;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::Init() {
  num_nodes = 1;
  leaf_vector_.Clear();
  leaf_vector_offset_.Resize(2, 0);
  matching_categories_.Clear();
  matching_categories_offset_.Resize(2, 0);
  nodes_.Resize(1);
  nodes_[0].Init();
  SetLeaf(0, static_cast<LeafOutputType>(0));
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
  node.categories_list_right_child_ = false;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::SetCategoricalSplit(
    int nid, unsigned split_index, bool default_left,
    const std::vector<uint32_t>& categories_list, bool categories_list_right_child) {
  if (split_index >= ((1U << 31U) - 1)) {
    throw std::runtime_error("split_index too big");
  }

  const size_t end_oft = matching_categories_offset_.Back();
  const size_t new_end_oft = end_oft + categories_list.size();
  if (end_oft != matching_categories_.Size()) {
    throw std::runtime_error("Invariant violated");
  }
  if (!std::all_of(&matching_categories_offset_[nid + 1], matching_categories_offset_.End(),
                   [end_oft](size_t x) { return (x == end_oft); })) {
    throw std::runtime_error("Invariant violated");
  }
  // Hopefully we won't have to move any element as we add node_matching_categories for node nid
  matching_categories_.Extend(categories_list);
  if (new_end_oft != matching_categories_.Size()) {
    throw std::runtime_error("Invariant violated");
  }
  std::for_each(&matching_categories_offset_[nid + 1], matching_categories_offset_.End(),
                [new_end_oft](size_t& x) { x = new_end_oft; });
  std::sort(&matching_categories_[end_oft], matching_categories_.End());

  Node& node = nodes_[nid];
  if (default_left) split_index |= (1U << 31U);
  node.sindex_ = split_index;
  node.split_type_ = SplitFeatureType::kCategorical;
  node.categories_list_right_child_ = categories_list_right_child;
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
    int nid, const std::vector<LeafOutputType>& node_leaf_vector) {
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
inline std::unique_ptr<Model>
Model::Create() {
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

inline std::unique_ptr<Model>
Model::Create(TypeInfo threshold_type, TypeInfo leaf_output_type) {
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
  inline static auto Dispatch(const Model* model, Func func) {
    return func(*dynamic_cast<const ModelImpl<ThresholdType, LeafOutputType>*>(model));
  }
};

template <typename Func>
inline auto
Model::Dispatch(Func func) {
  return DispatchWithModelTypes<ModelDispatchImpl>(threshold_type_, leaf_output_type_, this, func);
}

template <typename Func>
inline auto
Model::Dispatch(Func func) const {
  return DispatchWithModelTypes<ModelDispatchImpl>(threshold_type_, leaf_output_type_, this, func);
}

inline std::vector<PyBufferFrame>
Model::GetPyBuffer() {
  std::vector<PyBufferFrame> buffer;
  buffer.push_back(GetPyBufferFromScalar(&threshold_type_));
  buffer.push_back(GetPyBufferFromScalar(&leaf_output_type_));
  this->GetPyBuffer(&buffer);
  return buffer;
}

inline std::unique_ptr<Model>
Model::CreateFromPyBuffer(std::vector<PyBufferFrame> frames) {
  using TypeInfoInt = std::underlying_type<TypeInfo>::type;
  TypeInfo threshold_type, leaf_output_type;
  if (frames.size() < 2) {
    throw std::runtime_error("Insufficient number of frames: there must be at least two");
  }
  InitScalarFromPyBuffer(&threshold_type, frames[0]);
  InitScalarFromPyBuffer(&leaf_output_type, frames[1]);

  std::unique_ptr<Model> model = Model::Create(threshold_type, leaf_output_type);
  model->InitFromPyBuffer(frames.begin() + 2, frames.end());
  return model;
}


template <typename ThresholdType, typename LeafOutputType>
inline void
ModelImpl<ThresholdType, LeafOutputType>::GetPyBuffer(std::vector<PyBufferFrame>* dest) {
  /* Header */
  dest->push_back(GetPyBufferFromScalar(&num_feature));
  dest->push_back(GetPyBufferFromScalar(&task_type));
  dest->push_back(GetPyBufferFromScalar(&average_tree_output));
  dest->push_back(GetPyBufferFromScalar(&task_param, "T{=B=?xx=I=I}"));
  dest->push_back(GetPyBufferFromScalar(
      &param, "T{" _TREELITE_STR(TREELITE_MAX_PRED_TRANSFORM_LENGTH) "s=f=f}"));

  /* Body */
  for (Tree<ThresholdType, LeafOutputType>& tree : trees) {
    tree.GetPyBuffer(dest);
  }
}

template <typename ThresholdType, typename LeafOutputType>
inline void
ModelImpl<ThresholdType, LeafOutputType>::InitFromPyBuffer(
    std::vector<PyBufferFrame>::iterator begin, std::vector<PyBufferFrame>::iterator end) {
  const size_t num_frame = std::distance(begin, end);
  /* Header */
  constexpr size_t kNumFrameInHeader = 5;
  if (num_frame < kNumFrameInHeader) {
    throw std::runtime_error("Wrong number of frames");
  }
  InitScalarFromPyBuffer(&num_feature, *begin++);
  InitScalarFromPyBuffer(&task_type, *begin++);
  InitScalarFromPyBuffer(&average_tree_output, *begin++);
  InitScalarFromPyBuffer(&task_param, *begin++);
  InitScalarFromPyBuffer(&param, *begin++);
  /* Body */
  if ((num_frame - kNumFrameInHeader) % kNumFramePerTree != 0) {
    throw std::runtime_error("Wrong number of frames");
  }
  trees.clear();
  for (; begin < end; begin += kNumFramePerTree) {
    trees.emplace_back();
    trees.back().InitFromPyBuffer(begin, begin + kNumFramePerTree);
  }
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
