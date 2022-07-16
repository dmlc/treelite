/*!
 * Copyright (c) 2020-2021 by Contributors
 * \file tree_impl.h
 * \brief Implementation for tree.h
 * \author Hyunsu Cho
 */
#ifndef TREELITE_TREE_IMPL_H_
#define TREELITE_TREE_IMPL_H_

#include <treelite/error.h>
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
#include <cstddef>

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
    throw Error("Could not allocate memory for the clone");
  }
  std::memcpy(clone.buffer_, buffer_, sizeof(T) * size_);
  clone.size_ = size_;
  clone.capacity_ = capacity_;
  clone.owned_buffer_ = true;
  return clone;
}

template <typename T>
inline void
ContiguousArray<T>::UseForeignBuffer(void* prealloc_buf, std::size_t size) {
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
inline std::size_t
ContiguousArray<T>::Size() const {
  return size_;
}

template <typename T>
inline bool
ContiguousArray<T>::Empty() const {
  return (Size() == 0);
}

template <typename T>
inline void
ContiguousArray<T>::Reserve(std::size_t newsize) {
  if (!owned_buffer_) {
    throw Error("Cannot resize when using a foreign buffer; clone first");
  }
  T* newbuf = static_cast<T*>(std::realloc(static_cast<void*>(buffer_), sizeof(T) * newsize));
  if (!newbuf) {
    throw Error("Could not expand buffer");
  }
  buffer_ = newbuf;
  capacity_ = newsize;
}

template <typename T>
inline void
ContiguousArray<T>::Resize(std::size_t newsize) {
  if (!owned_buffer_) {
    throw Error("Cannot resize when using a foreign buffer; clone first");
  }
  if (newsize > capacity_) {
    std::size_t newcapacity = capacity_;
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
ContiguousArray<T>::Resize(std::size_t newsize, T t) {
  if (!owned_buffer_) {
    throw Error("Cannot resize when using a foreign buffer; clone first");
  }
  std::size_t oldsize = Size();
  Resize(newsize);
  for (std::size_t i = oldsize; i < newsize; ++i) {
    buffer_[i] = t;
  }
}

template <typename T>
inline void
ContiguousArray<T>::Clear() {
  if (!owned_buffer_) {
    throw Error("Cannot clear when using a foreign buffer; clone first");
  }
  Resize(0);
}

template <typename T>
inline void
ContiguousArray<T>::PushBack(T t) {
  if (!owned_buffer_) {
    throw Error("Cannot add element when using a foreign buffer; clone first");
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
    throw Error("Cannot add elements when using a foreign buffer; clone first");
  }
  if (other.empty()) {
    return;  // appending an empty vector is a no-op
  }
  std::size_t newsize = size_ + other.size();
  if (newsize > capacity_) {
    std::size_t newcapacity = capacity_;
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
ContiguousArray<T>::operator[](std::size_t idx) {
  return buffer_[idx];
}

template <typename T>
inline const T&
ContiguousArray<T>::operator[](std::size_t idx) const {
  return buffer_[idx];
}

template <typename T>
inline T&
ContiguousArray<T>::at(std::size_t idx) {
  if (idx >= Size()) {
    throw Error("nid out of range");
  }
  return buffer_[idx];
}

template <typename T>
inline const T&
ContiguousArray<T>::at(std::size_t idx) const {
  if (idx >= Size()) {
    throw Error("nid out of range");
  }
  return buffer_[idx];
}

template <typename T>
inline T&
ContiguousArray<T>::at(int idx) {
  if (idx < 0 || static_cast<std::size_t>(idx) >= Size()) {
    throw Error("nid out of range");
  }
  return buffer_[static_cast<std::size_t>(idx)];
}

template <typename T>
inline const T&
ContiguousArray<T>::at(int idx) const {
  if (idx < 0 || static_cast<std::size_t>(idx) >= Size()) {
    throw Error("nid out of range");
  }
  return buffer_[static_cast<std::size_t>(idx)];
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
      this->sigmoid_alpha = std::stof(e.second, nullptr);
    } else if (e.first == "ratio_c") {
      this->ratio_c = std::stof(e.second, nullptr);
    } else if (e.first == "global_bias") {
      this->global_bias = std::stof(e.second, nullptr);
    }
  }
  return unknowns;
}

inline std::map<std::string, std::string>
ModelParam::__DICT__() const {
  std::map<std::string, std::string> ret;
  ret.emplace("pred_transform", std::string(this->pred_transform));
  ret.emplace("sigmoid_alpha", GetString(this->sigmoid_alpha));
  ret.emplace("ratio_c", GetString(this->ratio_c));
  ret.emplace("global_bias", GetString(this->global_bias));
  return ret;
}

inline PyBufferFrame GetPyBufferFromArray(void* data, const char* format,
                                          std::size_t itemsize, std::size_t nitem) {
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
        throw Error("Could not infer format string");
      }
      return "=f";
    }
  case 8:
    if (std::is_integral<T>::value) {
      return (std::is_unsigned<T>::value ? "=Q" : "=q");
    } else {
      if (!std::is_floating_point<T>::value) {
        throw Error("Could not infer format string");
      }
      return "=d";
    }
  default:
    throw Error("Unrecognized type");
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

inline PyBufferFrame GetPyBufferFromScalar(void* data, const char* format, std::size_t itemsize) {
  return GetPyBufferFromArray(data, format, itemsize, 1);
}

template <typename T>
inline PyBufferFrame GetPyBufferFromScalar(T* scalar, const char* format) {
  static_assert(std::is_standard_layout<T>::value, "T must be in the standard layout");
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
inline void InitArrayFromPyBuffer(ContiguousArray<T>* vec, PyBufferFrame frame) {
  if (sizeof(T) != frame.itemsize) {
    throw Error("Incorrect itemsize");
  }
  vec->UseForeignBuffer(frame.buf, frame.nitem);
}

inline void InitScalarFromPyBuffer(TypeInfo* scalar, PyBufferFrame buffer) {
  using T = std::underlying_type<TypeInfo>::type;
  if (sizeof(T) != buffer.itemsize) {
    throw Error("Incorrect itemsize");
  }
  if (buffer.nitem != 1) {
    throw Error("nitem must be 1 for a scalar");
  }
  T* t = static_cast<T*>(buffer.buf);
  *scalar = static_cast<TypeInfo>(*t);
}

inline void InitScalarFromPyBuffer(TaskType* scalar, PyBufferFrame buffer) {
  using T = std::underlying_type<TaskType>::type;
  if (sizeof(T) != buffer.itemsize) {
    throw Error("Incorrect itemsize");
  }
  if (buffer.nitem != 1) {
    throw Error("nitem must be 1 for a scalar");
  }
  T* t = static_cast<T*>(buffer.buf);
  *scalar = static_cast<TaskType>(*t);
}

template <typename T>
inline void InitScalarFromPyBuffer(T* scalar, PyBufferFrame buffer) {
  static_assert(std::is_standard_layout<T>::value, "T must be in the standard layout");
  if (sizeof(T) != buffer.itemsize) {
    throw Error("Incorrect itemsize");
  }
  if (buffer.nitem != 1) {
    throw Error("nitem must be 1 for a scalar");
  }
  T* t = static_cast<T*>(buffer.buf);
  *scalar = *t;
}

template <typename T>
inline void ReadScalarFromFile(T* scalar, FILE* fp) {
  static_assert(std::is_standard_layout<T>::value, "T must be in the standard layout");
  if (std::fread(scalar, sizeof(T), 1, fp) < 1) {
    throw Error("Could not read a scalar");
  }
}

template <typename T>
inline void WriteScalarToFile(T* scalar, FILE* fp) {
  static_assert(std::is_standard_layout<T>::value, "T must be in the standard layout");
  if (std::fwrite(scalar, sizeof(T), 1, fp) < 1) {
    throw Error("Could not write a scalar");
  }
}

template <typename T>
inline void ReadArrayFromFile(ContiguousArray<T>* vec, FILE* fp) {
  uint64_t nelem;
  if (std::fread(&nelem, sizeof(nelem), 1, fp) < 1) {
    throw Error("Could not read the number of elements");
  }
  vec->Clear();
  vec->Resize(nelem);
  if (nelem == 0) {
    return;  // handle empty arrays
  }
  const auto nelem_size_t = static_cast<std::size_t>(nelem);
  if (std::fread(vec->Data(), sizeof(T), nelem_size_t, fp) < nelem_size_t) {
    throw Error("Could not read an array");
  }
}

template <typename T>
inline void WriteArrayToFile(ContiguousArray<T>* vec, FILE* fp) {
  static_assert(sizeof(uint64_t) >= sizeof(size_t), "size_t too large");
  const auto nelem = static_cast<uint64_t>(vec->Size());
  if (std::fwrite(&nelem, sizeof(nelem), 1, fp) < 1) {
    throw Error("Could not write the number of elements");
  }
  if (nelem == 0) {
    return;  // handle empty arrays
  }
  const auto nelem_size_t = vec->Size();
  if (std::fwrite(vec->Data(), sizeof(T), nelem_size_t, fp) < nelem_size_t) {
    throw Error("Could not write an array");
  }
}

template <typename ThresholdType, typename LeafOutputType>
inline Tree<ThresholdType, LeafOutputType>
Tree<ThresholdType, LeafOutputType>::Clone() const {
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
inline const char*
Tree<ThresholdType, LeafOutputType>::GetFormatStringForNode() {
  if (std::is_same<ThresholdType, float>::value) {
    return "T{=l=l=L=f=Q=d=d=b=b=?=?=?=?xx}";
  } else {
    return "T{=l=l=Lxxxx=d=Q=d=d=b=b=?=?=?=?xx}";
  }
}

constexpr std::size_t kNumFramePerTree = 8;

template <typename ThresholdType, typename LeafOutputType>
template <typename ScalarHandler, typename PrimitiveArrayHandler, typename CompositeArrayHandler>
inline void
Tree<ThresholdType, LeafOutputType>::SerializeTemplate(
    ScalarHandler scalar_handler, PrimitiveArrayHandler primitive_array_handler,
    CompositeArrayHandler composite_array_handler) {
  scalar_handler(&num_nodes);
  scalar_handler(&has_categorical_split_);
  composite_array_handler(&nodes_, GetFormatStringForNode());
  primitive_array_handler(&leaf_vector_);
  primitive_array_handler(&leaf_vector_begin_);
  primitive_array_handler(&leaf_vector_end_);
  primitive_array_handler(&matching_categories_);
  primitive_array_handler(&matching_categories_offset_);
}

template <typename ThresholdType, typename LeafOutputType>
template <typename ScalarHandler, typename ArrayHandler>
inline void
Tree<ThresholdType, LeafOutputType>::DeserializeTemplate(
    ScalarHandler scalar_handler, ArrayHandler array_handler) {
  scalar_handler(&num_nodes);
  scalar_handler(&has_categorical_split_);
  array_handler(&nodes_);
  if (static_cast<std::size_t>(num_nodes) != nodes_.Size()) {
    throw Error("Could not load the correct number of nodes");
  }
  array_handler(&leaf_vector_);
  array_handler(&leaf_vector_begin_);
  array_handler(&leaf_vector_end_);
  array_handler(&matching_categories_);
  array_handler(&matching_categories_offset_);
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::GetPyBuffer(std::vector<PyBufferFrame>* dest) {
  auto scalar_handler = [dest](auto* field) {
    dest->push_back(GetPyBufferFromScalar(field));
  };
  auto primitive_array_handler = [dest](auto* field) {
    dest->push_back(GetPyBufferFromArray(field));
  };
  auto composite_array_handler = [dest](auto* field, const char* format) {
    dest->push_back(GetPyBufferFromArray(field, format));
  };
  SerializeTemplate(scalar_handler, primitive_array_handler, composite_array_handler);
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::SerializeToFile(FILE* dest_fp) {
  auto scalar_handler = [dest_fp](auto* field) {
    WriteScalarToFile(field, dest_fp);
  };
  auto primitive_array_handler = [dest_fp](auto* field) {
    WriteArrayToFile(field, dest_fp);
  };
  auto composite_array_handler = [dest_fp](auto* field, const char* format) {
    WriteArrayToFile(field, dest_fp);
  };
  SerializeTemplate(scalar_handler, primitive_array_handler, composite_array_handler);
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::InitFromPyBuffer(std::vector<PyBufferFrame>::iterator begin,
                                                      std::vector<PyBufferFrame>::iterator end) {
  if (std::distance(begin, end) != kNumFramePerTree) {
    throw Error("Wrong number of frames specified");
  }
  auto scalar_handler = [&begin](auto* field) {
    InitScalarFromPyBuffer(field, *begin++);
  };
  auto array_handler = [&begin](auto* field) {
    InitArrayFromPyBuffer(field, *begin++);
  };
  DeserializeTemplate(scalar_handler, array_handler);
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::DeserializeFromFile(FILE* src_fp) {
  auto scalar_handler = [src_fp](auto* field) {
    ReadScalarFromFile(field, src_fp);
  };
  auto array_handler = [src_fp](auto* field) {
    ReadArrayFromFile(field, src_fp);
  };
  DeserializeTemplate(scalar_handler, array_handler);
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
inline int
Tree<ThresholdType, LeafOutputType>::AllocNode() {
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
inline void
Tree<ThresholdType, LeafOutputType>::Init() {
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
inline void
Tree<ThresholdType, LeafOutputType>::AddChilds(int nid) {
  const int cleft = this->AllocNode();
  const int cright = this->AllocNode();
  nodes_.at(nid).cleft_ = cleft;
  nodes_.at(nid).cright_ = cright;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::SetNumericalSplit(
    int nid, unsigned split_index, ThresholdType threshold, bool default_left, Operator cmp) {
  Node& node = nodes_.at(nid);
  if (split_index >= ((1U << 31U) - 1)) {
    throw Error("split_index too big");
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
  if (default_left) split_index |= (1U << 31U);
  node.sindex_ = split_index;
  node.split_type_ = SplitFeatureType::kCategorical;
  node.categories_list_right_child_ = categories_list_right_child;

  has_categorical_split_ = true;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::SetLeaf(int nid, LeafOutputType value) {
  Node& node = nodes_.at(nid);
  (node.info_).leaf_value = value;
  node.cleft_ = -1;
  node.cright_ = -1;
  node.split_type_ = SplitFeatureType::kNone;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::SetLeafVector(
    int nid, const std::vector<LeafOutputType>& node_leaf_vector) {
  std::size_t begin = leaf_vector_.Size();
  std::size_t end = begin + node_leaf_vector.size();
  leaf_vector_.Extend(node_leaf_vector);
  leaf_vector_begin_[nid] = begin;
  leaf_vector_end_[nid] = end;
  Node &node = nodes_.at(nid);
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

template <typename HeaderPrimitiveFieldHandlerFunc>
inline void
Model::SerializeTemplate(HeaderPrimitiveFieldHandlerFunc header_primitive_field_handler) {
  header_primitive_field_handler(&major_ver_);
  header_primitive_field_handler(&minor_ver_);
  header_primitive_field_handler(&patch_ver_);
  header_primitive_field_handler(&threshold_type_);
  header_primitive_field_handler(&leaf_output_type_);
}

template <typename HeaderPrimitiveFieldHandlerFunc>
inline void
Model::DeserializeTemplate(HeaderPrimitiveFieldHandlerFunc header_primitive_field_handler,
                           TypeInfo& threshold_type, TypeInfo& leaf_output_type) {
  int major_ver, minor_ver, patch_ver;
  header_primitive_field_handler(&major_ver);
  header_primitive_field_handler(&minor_ver);
  header_primitive_field_handler(&patch_ver);
  if (major_ver != TREELITE_VER_MAJOR || minor_ver != TREELITE_VER_MINOR) {
    std::ostringstream oss;
    oss << "Cannot deserialize model from a different version of Treelite." << std::endl
        << "Currently running Treelite version " << TREELITE_VER_MAJOR << "."
        << TREELITE_VER_MINOR << "." << TREELITE_VER_PATCH << std::endl
        << "The model checkpoint was generated from Treelite version " << major_ver << "."
        << minor_ver << "." << patch_ver;
    throw Error(oss.str());
  }
  header_primitive_field_handler(&threshold_type);
  header_primitive_field_handler(&leaf_output_type);
}

template <typename ThresholdType, typename LeafOutputType>
template <typename HeaderPrimitiveFieldHandlerFunc, typename HeaderCompositeFieldHandlerFunc,
    typename TreeHandlerFunc>
inline void
ModelImpl<ThresholdType, LeafOutputType>::SerializeTemplate(
    HeaderPrimitiveFieldHandlerFunc header_primitive_field_handler,
    HeaderCompositeFieldHandlerFunc header_composite_field_handler,
    TreeHandlerFunc tree_handler) {
  /* Header */
  header_primitive_field_handler(&num_feature);
  header_primitive_field_handler(&task_type);
  header_primitive_field_handler(&average_tree_output);
  header_composite_field_handler(&task_param, "T{=B=?xx=I=I}");
  header_composite_field_handler(
      &param, "T{" _TREELITE_STR(TREELITE_MAX_PRED_TRANSFORM_LENGTH) "s=f=f=f}");

  /* Body */
  for (Tree<ThresholdType, LeafOutputType>& tree : trees) {
    tree_handler(tree);
  }
}

template <typename ThresholdType, typename LeafOutputType>
template <typename HeaderFieldHandlerFunc, typename TreeHandlerFunc>
inline void
ModelImpl<ThresholdType, LeafOutputType>::DeserializeTemplate(
    std::size_t num_tree,
    HeaderFieldHandlerFunc header_field_handler,
    TreeHandlerFunc tree_handler) {
  /* Header */
  header_field_handler(&num_feature);
  header_field_handler(&task_type);
  header_field_handler(&average_tree_output);
  header_field_handler(&task_param);
  header_field_handler(&param);
  /* Body */
  trees.clear();
  for (std::size_t i = 0; i < num_tree; ++i) {
    trees.emplace_back();
    tree_handler(trees.back());
  }
}

template <typename ThresholdType, typename LeafOutputType>
inline void
ModelImpl<ThresholdType, LeafOutputType>::GetPyBuffer(std::vector<PyBufferFrame>* dest) {
  auto header_primitive_field_handler = [dest](auto* field) {
    dest->push_back(GetPyBufferFromScalar(field));
  };
  auto header_composite_field_handler = [dest](auto* field, const char* format) {
    dest->push_back(GetPyBufferFromScalar(field, format));
  };
  auto tree_handler = [dest](Tree<ThresholdType, LeafOutputType>& tree) {
    tree.GetPyBuffer(dest);
  };
  SerializeTemplate(header_primitive_field_handler, header_composite_field_handler, tree_handler);
}

template <typename ThresholdType, typename LeafOutputType>
inline void
ModelImpl<ThresholdType, LeafOutputType>::SerializeToFileImpl(FILE* dest_fp) {
  const auto num_tree = static_cast<uint64_t>(this->trees.size());
  WriteScalarToFile(&num_tree, dest_fp);
  auto header_primitive_field_handler = [dest_fp](auto* field) {
    WriteScalarToFile(field, dest_fp);
  };
  auto header_composite_field_handler = [dest_fp](auto* field, const char* format) {
    WriteScalarToFile(field, dest_fp);
  };
  auto tree_handler = [dest_fp](Tree<ThresholdType, LeafOutputType>& tree) {
    tree.SerializeToFile(dest_fp);
  };
  SerializeTemplate(header_primitive_field_handler, header_composite_field_handler, tree_handler);
}

template <typename ThresholdType, typename LeafOutputType>
inline void
ModelImpl<ThresholdType, LeafOutputType>::InitFromPyBuffer(
    std::vector<PyBufferFrame>::iterator begin, std::vector<PyBufferFrame>::iterator end) {
  const std::size_t num_frame = std::distance(begin, end);
  constexpr std::size_t kNumFrameInHeader = 5;
  if (num_frame < kNumFrameInHeader || (num_frame - kNumFrameInHeader) % kNumFramePerTree != 0) {
    throw Error("Wrong number of frames");
  }
  const std::size_t num_tree = (num_frame - kNumFrameInHeader) / kNumFramePerTree;

  auto header_field_handler = [&begin](auto* field) {
    InitScalarFromPyBuffer(field, *begin++);
  };

  auto tree_handler = [&begin](Tree<ThresholdType, LeafOutputType>& tree) {
    // Read the frames in the range [begin, begin + kNumFramePerTree) into the tree
    tree.InitFromPyBuffer(begin, begin + kNumFramePerTree);
    begin += kNumFramePerTree;
      // Advance the iterator so that the next tree reads the next kNumFramePerTree frames
  };

  DeserializeTemplate(num_tree, header_field_handler, tree_handler);
}

template <typename ThresholdType, typename LeafOutputType>
inline void
ModelImpl<ThresholdType, LeafOutputType>::DeserializeFromFileImpl(FILE* src_fp) {
  uint64_t num_tree;
  ReadScalarFromFile(&num_tree, src_fp);

  auto header_field_handler = [src_fp](auto* field) {
    ReadScalarFromFile(field, src_fp);
  };

  auto tree_handler = [src_fp](Tree<ThresholdType, LeafOutputType>& tree) {
    tree.DeserializeFromFile(src_fp);
  };

  DeserializeTemplate(num_tree, header_field_handler, tree_handler);
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
