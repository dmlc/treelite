/*!
 * Copyright (c) 2023 by Contributors
 * \file serializer_mixins.h
 * \brief Mix-in classes for serializers
 * \author Hyunsu Cho
 */

#ifndef TREELITE_DETAIL_SERIALIZER_MIXINS_H_
#define TREELITE_DETAIL_SERIALIZER_MIXINS_H_

#include <cstring>
#include <istream>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include <treelite/contiguous_array.h>
#include <treelite/detail/serializer.h>
#include <treelite/pybuffer_frame.h>

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

/*!
 * \brief Mixin that calculates the total serialized size without writing.
 *
 * This is used to pre-allocate a buffer of the exact size needed,
 * avoiding costly reallocations during serialization.
 */
class SizeCalculatorMixIn {
 public:
  SizeCalculatorMixIn() : total_size_(0) {}

  template <typename T>
  void SerializeScalar(T* /*field*/) {
    total_size_ += sizeof(T);
  }

  void SerializeString(std::string* field) {
    total_size_ += sizeof(std::uint64_t);  // length prefix
    total_size_ += field->length();
  }

  template <typename T>
  void SerializeArray(ContiguousArray<T>* field) {
    total_size_ += sizeof(std::uint64_t);  // length prefix
    total_size_ += sizeof(T) * field->Size();
  }

  std::size_t GetTotalSize() const {
    return total_size_;
  }

 private:
  std::size_t total_size_;
};

/*!
 * \brief Mixin that writes directly to a pre-allocated buffer.
 *
 * This avoids the overhead of std::ostringstream which has poor performance
 * due to many small allocations and a final string copy.
 */
class BufferSerializerMixIn {
 public:
  explicit BufferSerializerMixIn(std::vector<char>& buffer) : buffer_(buffer), offset_(0) {}

  template <typename T>
  void SerializeScalar(T* field) {
    static_assert(std::is_standard_layout_v<T>, "T must be in the standard layout");
    std::memcpy(buffer_.data() + offset_, field, sizeof(T));
    offset_ += sizeof(T);
  }

  void SerializeString(std::string* field) {
    auto const str_len = static_cast<std::uint64_t>(field->length());
    std::memcpy(buffer_.data() + offset_, &str_len, sizeof(str_len));
    offset_ += sizeof(str_len);
    if (str_len > 0) {
      std::memcpy(buffer_.data() + offset_, field->data(), str_len);
      offset_ += str_len;
    }
  }

  template <typename T>
  void SerializeArray(ContiguousArray<T>* field) {
    auto const nelem = static_cast<std::uint64_t>(field->Size());
    std::memcpy(buffer_.data() + offset_, &nelem, sizeof(nelem));
    offset_ += sizeof(nelem);
    if (nelem > 0) {
      std::size_t const nbytes = sizeof(T) * field->Size();
      std::memcpy(buffer_.data() + offset_, field->Data(), nbytes);
      offset_ += nbytes;
    }
  }

  std::size_t GetOffset() const {
    return offset_;
  }

 private:
  std::vector<char>& buffer_;
  std::size_t offset_;
};

}  // namespace treelite::detail::serializer

#endif  // TREELITE_DETAIL_SERIALIZER_MIXINS_H_
