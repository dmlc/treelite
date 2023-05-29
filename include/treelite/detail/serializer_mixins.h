/*!
 * Copyright (c) 2023 by Contributors
 * \file serializer_mixins.h
 * \brief Mix-in classes for serializers
 * \author Hyunsu Cho
 */

#ifndef TREELITE_DETAIL_SERIALIZER_MIXINS_H_
#define TREELITE_DETAIL_SERIALIZER_MIXINS_H_

#include <treelite/detail/serializer_impl.h>
#include <treelite/pybuffer_frame.h>

#include <istream>
#include <ostream>
#include <vector>

namespace treelite::detail::serializer {

class StreamSerializerMixIn {
 public:
  explicit StreamSerializerMixIn(std::ostream& os) : os_(os) {}

  template <typename T>
  void SerializePrimitiveField(T* field) {
    WriteScalarToStream(field, os_);
  }

  template <typename T>
  void SerializeCompositeField(T* field, char const*) {
    WriteScalarToStream(field, os_);
  }

  template <typename T>
  void SerializePrimitiveArray(T* field) {
    WriteArrayToStream(field, os_);
  }

  template <typename T>
  void SerializeCompositeArray(T* field, char const*) {
    WriteArrayToStream(field, os_);
  }

 private:
  std::ostream& os_;
};

class StreamDeserializerMixIn {
 public:
  explicit StreamDeserializerMixIn(std::istream& is) : is_(is) {}

  template <typename T>
  void DeserializePrimitiveField(T* field) {
    ReadScalarFromStream(field, is_);
  }

  template <typename T>
  void DeserializeCompositeField(T* field) {
    ReadScalarFromStream(field, is_);
  }

  template <typename T>
  void DeserializePrimitiveArray(T* field) {
    ReadArrayFromStream(field, is_);
  }

  template <typename T>
  void DeserializeCompositeArray(T* field) {
    ReadArrayFromStream(field, is_);
  }

  void SkipOptionalField() {
    SkipOptFieldInStream(is_);
  }

 private:
  std::istream& is_;
};

class PyBufferSerializerMixIn {
 public:
  PyBufferSerializerMixIn() = default;

  template <typename T>
  void SerializePrimitiveField(T* field) {
    frames_.push_back(GetPyBufferFromScalar(field));
  }

  template <typename T>
  void SerializeCompositeField(T* field, char const* format) {
    frames_.push_back(GetPyBufferFromScalar(field, format));
  }

  template <typename T>
  void SerializePrimitiveArray(T* field) {
    frames_.push_back(GetPyBufferFromArray(field));
  }

  template <typename T>
  void SerializeCompositeArray(T* field, char const* format) {
    frames_.push_back(GetPyBufferFromArray(field, format));
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
      : it_(frames.cbegin()) {}

  template <typename T>
  void DeserializePrimitiveField(T* field) {
    InitScalarFromPyBuffer(field, *it_++);
  }

  template <typename T>
  void DeserializeCompositeField(T* field) {
    InitScalarFromPyBuffer(field, *it_++);
  }

  template <typename T>
  void DeserializePrimitiveArray(T* field) {
    InitArrayFromPyBuffer(field, *it_++);
  }

  template <typename T>
  void DeserializeCompositeArray(T* field) {
    InitArrayFromPyBuffer(field, *it_++);
  }

  void SkipOptionalField() {
    ++it_;
  }

 private:
  std::vector<PyBufferFrame>::const_iterator it_;
};

}  // namespace treelite::detail::serializer

#endif  // TREELITE_DETAIL_SERIALIZER_MIXINS_H_
