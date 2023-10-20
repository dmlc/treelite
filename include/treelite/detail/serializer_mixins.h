/*!
 * Copyright (c) 2023 by Contributors
 * \file serializer_mixins.h
 * \brief Mix-in classes for serializers
 * \author Hyunsu Cho
 */

#ifndef TREELITE_DETAIL_SERIALIZER_MIXINS_H_
#define TREELITE_DETAIL_SERIALIZER_MIXINS_H_

#include <treelite/contiguous_array.h>
#include <treelite/detail/serializer.h>
#include <treelite/pybuffer_frame.h>

#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace treelite::detail::serializer {

class StreamSerializerMixIn {
 public:
  explicit StreamSerializerMixIn(std::ostream& os) : os_(os) {}

  template <typename T>
  void SerializeScalar(T* field) {
    WriteScalarToStream(field, os_);
  }

  void SerializeString(std::string* field) {
    WriteStringToStream(field, os_);
  }

  template <typename T>
  void SerializeArray(ContiguousArray<T>* field) {
    WriteArrayToStream(field, os_);
  }

 private:
  std::ostream& os_;
};

class StreamDeserializerMixIn {
 public:
  explicit StreamDeserializerMixIn(std::istream& is) : is_(is) {}

  template <typename T>
  void DeserializeScalar(T* field) {
    ReadScalarFromStream(field, is_);
  }

  void DeserializeString(std::string* field) {
    ReadStringFromStream(field, is_);
  }

  template <typename T>
  void DeserializeArray(ContiguousArray<T>* field) {
    ReadArrayFromStream(field, is_);
  }

  void SkipOptionalField() {
    SkipOptionalFieldInStream(is_);
  }

 private:
  std::istream& is_;
};

class PyBufferSerializerMixIn {
 public:
  PyBufferSerializerMixIn() = default;

  template <typename T>
  void SerializeScalar(T* field) {
    frames_.push_back(GetPyBufferFromScalar(field));
  }

  void SerializeString(std::string* field) {
    frames_.push_back(GetPyBufferFromString(field));
  }

  template <typename T>
  void SerializeArray(ContiguousArray<T>* field) {
    frames_.push_back(GetPyBufferFromArray(field));
  }

  std::vector<PyBufferFrame> GetFrames() {
    return frames_;
  }

 private:
  std::vector<PyBufferFrame> frames_;
};

class PyBufferDeserializerMixIn {
 public:
  explicit PyBufferDeserializerMixIn(std::vector<PyBufferFrame> const& frames)
      : frames_(frames), cur_idx_(0) {}

  template <typename T>
  void DeserializeScalar(T* field) {
    InitScalarFromPyBuffer(field, frames_[cur_idx_++]);
  }

  void DeserializeString(std::string* field) {
    InitStringFromPyBuffer(field, frames_[cur_idx_++]);
  }

  template <typename T>
  void DeserializeArray(ContiguousArray<T>* field) {
    InitArrayFromPyBuffer(field, frames_[cur_idx_++]);
  }

  void SkipOptionalField() {
    cur_idx_ += 2;  // field name + content
  }

 private:
  std::vector<PyBufferFrame> const& frames_;
  std::size_t cur_idx_;
};

}  // namespace treelite::detail::serializer

#endif  // TREELITE_DETAIL_SERIALIZER_MIXINS_H_
