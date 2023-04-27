/*!
 * Copyright (c) 2020-2021 by Contributors
 * \file tree_impl.h
 * \brief Implementation for tree.h
 * \author Hyunsu Cho
 */
#ifndef TREELITE_TREE_IMPL_H_
#define TREELITE_TREE_IMPL_H_

#include <treelite/error.h>
#include <treelite/version.h>
#include <treelite/logging.h>
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
#include <cstdint>

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
  if (buffer_) {
    clone.buffer_ = static_cast<T*>(std::malloc(sizeof(T) * capacity_));
    if (!clone.buffer_) {
      throw Error("Could not allocate memory for the clone");
    }
    std::memcpy(clone.buffer_, buffer_, sizeof(T) * size_);
  } else {
    TREELITE_CHECK_EQ(size_, 0);
    TREELITE_CHECK_EQ(capacity_, 0);
    clone.buffer_ = nullptr;
  }
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
inline void ReadScalarFromStream(T* scalar, std::istream& is) {
  static_assert(std::is_standard_layout<T>::value, "T must be in the standard layout");
  is.read(reinterpret_cast<char*>(scalar), sizeof(T));
}

template <typename T>
inline void WriteScalarToStream(T* scalar, std::ostream& os) {
  static_assert(std::is_standard_layout<T>::value, "T must be in the standard layout");
  os.write(reinterpret_cast<const char*>(scalar), sizeof(T));
}

template <typename T>
inline void ReadArrayFromStream(ContiguousArray<T>* vec, std::istream& is) {
  std::uint64_t nelem;
  is.read(reinterpret_cast<char*>(&nelem), sizeof(nelem));
  vec->Clear();
  vec->Resize(nelem);
  if (nelem == 0) {
    return;  // handle empty arrays
  }
  is.read(reinterpret_cast<char*>(vec->Data()), sizeof(T) * nelem);
}

template <typename T>
inline void WriteArrayToStream(ContiguousArray<T>* vec, std::ostream& os) {
  static_assert(sizeof(std::uint64_t) >= sizeof(std::size_t), "size_t too large");
  const auto nelem = static_cast<std::uint64_t>(vec->Size());
  os.write(reinterpret_cast<const char*>(&nelem), sizeof(nelem));
  if (nelem == 0) {
    return;  // handle empty arrays
  }
  os.write(reinterpret_cast<const char*>(vec->Data()), sizeof(T) * vec->Size());
}

inline void SkipOptFieldInStream(std::istream& is) {
  std::uint16_t elem_size;
  std::uint64_t nelem;
  ReadScalarFromStream(&elem_size, is);
  ReadScalarFromStream(&nelem, is);

  const std::uint64_t nbytes = elem_size * nelem;
  TREELITE_CHECK_LE(nbytes, std::numeric_limits<std::streamoff>::max());  // NOLINT
  is.seekg(static_cast<std::streamoff>(nbytes), std::ios::cur);
}

template <typename ThresholdType, typename LeafOutputType>
Tree<ThresholdType, LeafOutputType>::Tree(bool use_opt_field)
  : use_opt_field_(use_opt_field)
{}

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

  /* Extension slot 2: Per-tree optional fields -- to be added later */
  num_opt_field_per_tree_ = 0;
  scalar_handler(&num_opt_field_per_tree_);

  /* Extension slot 3: Per-node optional fields -- to be added later */
  num_opt_field_per_node_ = 0;
  scalar_handler(&num_opt_field_per_node_);
}

template <typename ThresholdType, typename LeafOutputType>
template <typename ScalarHandler, typename ArrayHandler, typename SkipOptFieldHandlerFunc>
inline void
Tree<ThresholdType, LeafOutputType>::DeserializeTemplate(
    ScalarHandler scalar_handler,
    ArrayHandler array_handler,
    SkipOptFieldHandlerFunc skip_opt_field_handler) {
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

  if (use_opt_field_) {
    /* Extension slot 2: Per-tree optional fields -- to be added later */
    scalar_handler(&num_opt_field_per_tree_);
    // Ignore extra fields; the input is likely from a later version of Treelite
    for (int32_t i = 0; i < num_opt_field_per_tree_; ++i) {
      skip_opt_field_handler();
    }

    /* Extension slot 3: Per-node optional fields -- to be added later */
    scalar_handler(&num_opt_field_per_node_);
    // Ignore extra fields; the input is likely from a later version of Treelite
    for (int32_t i = 0; i < num_opt_field_per_node_; ++i) {
      skip_opt_field_handler();
    }
  } else {
    // Legacy (version 2.4)
    num_opt_field_per_tree_ = 0;
    num_opt_field_per_node_ = 0;
  }
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
Tree<ThresholdType, LeafOutputType>::SerializeToStream(std::ostream& os) {
  auto scalar_handler = [&os](auto* field) {
    WriteScalarToStream(field, os);
  };
  auto primitive_array_handler = [&os](auto* field) {
    WriteArrayToStream(field, os);
  };
  auto composite_array_handler = [&os](auto* field, const char* format) {
    WriteArrayToStream(field, os);
  };
  SerializeTemplate(scalar_handler, primitive_array_handler, composite_array_handler);
}

template <typename ThresholdType, typename LeafOutputType>
inline std::vector<PyBufferFrame>::iterator
Tree<ThresholdType, LeafOutputType>::InitFromPyBuffer(std::vector<PyBufferFrame>::iterator it) {
  std::vector<PyBufferFrame>::iterator new_it = it;
  auto scalar_handler = [&new_it](auto* field) {
    InitScalarFromPyBuffer(field, *(new_it++));
  };
  auto array_handler = [&new_it](auto* field) {
    InitArrayFromPyBuffer(field, *(new_it++));
  };
  auto skip_opt_field_handler = [&new_it]() {
    ++new_it;
  };
  DeserializeTemplate(scalar_handler, array_handler, skip_opt_field_handler);
  return new_it;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
Tree<ThresholdType, LeafOutputType>::DeserializeFromStream(std::istream& is) {
  auto scalar_handler = [&is](auto* field) {
    ReadScalarFromStream(field, is);
  };
  auto array_handler = [&is](auto* field) {
    ReadArrayFromStream(field, is);
  };
  auto skip_opt_field_handler = [&is]() {
    SkipOptFieldInStream(is);
  };
  DeserializeTemplate(scalar_handler, array_handler, skip_opt_field_handler);
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
  major_ver_ = TREELITE_VER_MAJOR;
  minor_ver_ = TREELITE_VER_MINOR;
  patch_ver_ = TREELITE_VER_PATCH;
  header_primitive_field_handler(&major_ver_);
  header_primitive_field_handler(&minor_ver_);
  header_primitive_field_handler(&patch_ver_);
  header_primitive_field_handler(&threshold_type_);
  header_primitive_field_handler(&leaf_output_type_);
}

template <typename HeaderPrimitiveFieldHandlerFunc>
inline void
Model::DeserializeTemplate(HeaderPrimitiveFieldHandlerFunc header_primitive_field_handler,
                           int32_t& major_ver, int32_t& minor_ver, int32_t& patch_ver,
                           TypeInfo& threshold_type, TypeInfo& leaf_output_type) {
  header_primitive_field_handler(&major_ver);
  header_primitive_field_handler(&minor_ver);
  header_primitive_field_handler(&patch_ver);
  if (major_ver != TREELITE_VER_MAJOR && !(major_ver == 2 && minor_ver == 4)) {
    TREELITE_LOG(FATAL)
        << "Cannot load model from a different major Treelite version or "
        << "a version before 2.4.0." << std::endl
        << "Currently running Treelite version " << TREELITE_VER_MAJOR << "."
        << TREELITE_VER_MINOR << "." << TREELITE_VER_PATCH << std::endl
        << "The model checkpoint was generated from Treelite version " << major_ver << "."
        << minor_ver << "." << patch_ver;
  } else if (major_ver == TREELITE_VER_MAJOR && minor_ver > TREELITE_VER_MINOR) {
    TREELITE_LOG(WARNING)
        << "The model you are loading originated from a newer Treelite version; some "
        << "functionalities may be unavailable." << std::endl
        << "Currently running Treelite version " << TREELITE_VER_MAJOR << "."
        << TREELITE_VER_MINOR << "." << TREELITE_VER_PATCH << std::endl
        << "The model checkpoint was generated from Treelite version " << major_ver << "."
        << minor_ver << "." << patch_ver;
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

  /* Extension Slot 1: Per-model optional fields -- to be added later */
  num_opt_field_per_model_ = 0;
  header_primitive_field_handler(&num_opt_field_per_model_);

  /* Body */
  for (Tree<ThresholdType, LeafOutputType>& tree : trees) {
    tree_handler(tree);
  }
}

template <typename ThresholdType, typename LeafOutputType>
template <typename HeaderFieldHandlerFunc, typename TreeHandlerFunc,
    typename SkipOptFieldHandlerFunc>
inline void
ModelImpl<ThresholdType, LeafOutputType>::DeserializeTemplate(
    std::size_t num_tree,
    HeaderFieldHandlerFunc header_field_handler,
    TreeHandlerFunc tree_handler,
    SkipOptFieldHandlerFunc skip_opt_field_handler) {
  /* Header */
  header_field_handler(&num_feature);
  header_field_handler(&task_type);
  header_field_handler(&average_tree_output);
  header_field_handler(&task_param);
  header_field_handler(&param);

  /* Extension Slot 1: Per-model optional fields -- to be added later */
  const bool use_opt_field = (major_ver_ >= 3);
  if (use_opt_field) {
    header_field_handler(&num_opt_field_per_model_);
    // Ignore extra fields; the input is likely from a later version of Treelite
    for (int32_t i = 0; i < num_opt_field_per_model_; ++i) {
      skip_opt_field_handler();
    }
  } else {
    // Legacy (version 2.4)
    num_opt_field_per_model_ = 0;
  }

  /* Body */
  trees.clear();
  for (std::size_t i = 0; i < num_tree; ++i) {
    trees.emplace_back(use_opt_field);
    tree_handler(trees.back());
  }
}

template <typename ThresholdType, typename LeafOutputType>
inline void
ModelImpl<ThresholdType, LeafOutputType>::GetPyBuffer(std::vector<PyBufferFrame>* dest) {
  num_tree_ = static_cast<uint64_t>(this->trees.size());
  auto header_primitive_field_handler = [dest](auto* field) {
    dest->push_back(GetPyBufferFromScalar(field));
  };
  auto header_composite_field_handler = [dest](auto* field, const char* format) {
    dest->push_back(GetPyBufferFromScalar(field, format));
  };
  auto tree_handler = [dest](Tree<ThresholdType, LeafOutputType>& tree) {
    tree.GetPyBuffer(dest);
  };
  header_primitive_field_handler(&num_tree_);
  SerializeTemplate(header_primitive_field_handler, header_composite_field_handler, tree_handler);
}

template <typename ThresholdType, typename LeafOutputType>
inline void
ModelImpl<ThresholdType, LeafOutputType>::SerializeToStreamImpl(std::ostream& os) {
  num_tree_ = static_cast<std::uint64_t>(this->trees.size());
  auto header_primitive_field_handler = [&os](auto* field) {
    WriteScalarToStream(field, os);
  };
  auto header_composite_field_handler = [&os](auto* field, const char* format) {
    WriteScalarToStream(field, os);
  };
  auto tree_handler = [&os](Tree<ThresholdType, LeafOutputType>& tree) {
    tree.SerializeToStream(os);
  };
  header_primitive_field_handler(&num_tree_);
  SerializeTemplate(header_primitive_field_handler, header_composite_field_handler, tree_handler);
}

template <typename ThresholdType, typename LeafOutputType>
inline std::vector<PyBufferFrame>::iterator
ModelImpl<ThresholdType, LeafOutputType>::InitFromPyBuffer(
    std::vector<PyBufferFrame>::iterator it, std::size_t num_frame) {
  std::vector<PyBufferFrame>::iterator new_it = it;
  auto header_field_handler = [&new_it](auto* field) {
    InitScalarFromPyBuffer(field, *(new_it++));
  };

  auto skip_opt_field_handler = [&new_it]() {
    ++new_it;
  };

  auto tree_handler = [&new_it](Tree<ThresholdType, LeafOutputType>& tree) {
    new_it = tree.InitFromPyBuffer(new_it);
  };

  if (major_ver_ == 2) {
    // From version 2.4 (legacy)
    // num_tree has to be inferred from the number of frames received
    constexpr std::size_t kNumFrameInHeader = 5;
    constexpr std::size_t kNumFramePerTree = 8;
    num_tree_ = (num_frame - kNumFrameInHeader) / kNumFramePerTree;
  } else {
    // From version 3.x
    // num_tree is now explicitly stored
    header_field_handler(&num_tree_);
  }

  DeserializeTemplate(num_tree_, header_field_handler, tree_handler, skip_opt_field_handler);
  TREELITE_CHECK_EQ(num_tree_, this->trees.size());

  return new_it;
}

template <typename ThresholdType, typename LeafOutputType>
inline void
ModelImpl<ThresholdType, LeafOutputType>::DeserializeFromStreamImpl(std::istream& is) {
  ReadScalarFromStream(&num_tree_, is);

  auto header_field_handler = [&is](auto* field) {
    ReadScalarFromStream(field, is);
  };

  auto skip_opt_field_handler = [&is]() {
    SkipOptFieldInStream(is);
  };

  auto tree_handler = [&is](Tree<ThresholdType, LeafOutputType>& tree) {
    tree.DeserializeFromStream(is);
  };

  DeserializeTemplate(num_tree_, header_field_handler, tree_handler, skip_opt_field_handler);
  TREELITE_CHECK_EQ(num_tree_, this->trees.size());
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
