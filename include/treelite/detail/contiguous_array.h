/*!
 * Copyright (c) 2023 by Contributors
 * \file contiguous_array.h
 * \brief Implementation for ContiguousArray
 * \author Hyunsu Cho
 */
#ifndef TREELITE_DETAIL_CONTIGUOUS_ARRAY_H_
#define TREELITE_DETAIL_CONTIGUOUS_ARRAY_H_

#include <treelite/logging.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

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
ContiguousArray<T>::ContiguousArray(std::vector<T> const& other) {
  buffer_ = static_cast<T*>(std::malloc(sizeof(T) * other.capacity()));
  TREELITE_CHECK(buffer_) << "Could not allocate buffer";
  std::memcpy(buffer_, other.data(), sizeof(T) * other.size());
  size_ = other.size();
  capacity_ = other.capacity();
  owned_buffer_ = true;
}

template <typename T>
ContiguousArray<T>& ContiguousArray<T>::operator=(std::vector<T> const& other) {
  if (buffer_ && owned_buffer_) {
    std::free(buffer_);
  }
  buffer_ = static_cast<T*>(std::malloc(sizeof(T) * other.capacity()));
  TREELITE_CHECK(buffer_) << "Could not allocate buffer";
  std::memcpy(buffer_, other.data(), sizeof(T) * other.size());
  size_ = other.size();
  capacity_ = other.capacity();
  owned_buffer_ = true;
  return *this;
}

template <typename T>
ContiguousArray<T>::ContiguousArray(ContiguousArray&& other) noexcept
    : buffer_(other.buffer_),
      size_(other.size_),
      capacity_(other.capacity_),
      owned_buffer_(other.owned_buffer_) {
  other.buffer_ = nullptr;
  other.size_ = other.capacity_ = 0;
}

template <typename T>
ContiguousArray<T>& ContiguousArray<T>::operator=(ContiguousArray&& other) noexcept {
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
inline ContiguousArray<T> ContiguousArray<T>::Clone() const {
  ContiguousArray clone;
  if (buffer_) {
    clone.buffer_ = static_cast<T*>(std::malloc(sizeof(T) * capacity_));
    TREELITE_CHECK(clone.buffer_) << "Could not allocate memory for the clone";
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
inline void ContiguousArray<T>::UseForeignBuffer(void* prealloc_buf, std::size_t size) {
  if (buffer_ && owned_buffer_) {
    std::free(buffer_);
  }
  buffer_ = static_cast<T*>(prealloc_buf);
  size_ = size;
  capacity_ = size;
  owned_buffer_ = false;
}

template <typename T>
inline T* ContiguousArray<T>::Data() {
  return buffer_;
}

template <typename T>
inline T const* ContiguousArray<T>::Data() const {
  return buffer_;
}

template <typename T>
inline T* ContiguousArray<T>::End() {
  return &buffer_[Size()];
}

template <typename T>
inline T const* ContiguousArray<T>::End() const {
  return &buffer_[Size()];
}

template <typename T>
inline T& ContiguousArray<T>::Back() {
  return buffer_[Size() - 1];
}

template <typename T>
inline T const& ContiguousArray<T>::Back() const {
  return buffer_[Size() - 1];
}

template <typename T>
inline std::size_t ContiguousArray<T>::Size() const {
  return size_;
}

template <typename T>
inline bool ContiguousArray<T>::Empty() const {
  return (Size() == 0);
}

template <typename T>
inline void ContiguousArray<T>::Reserve(std::size_t newsize) {
  TREELITE_CHECK(owned_buffer_) << "Cannot resize when using a foreign buffer; clone first";
  T* newbuf = static_cast<T*>(std::realloc(static_cast<void*>(buffer_), sizeof(T) * newsize));
  TREELITE_CHECK(newbuf) << "Could not expand buffer";
  buffer_ = newbuf;
  capacity_ = newsize;
}

template <typename T>
inline void ContiguousArray<T>::Resize(std::size_t newsize) {
  Resize(newsize, T{});
}

template <typename T>
inline void ContiguousArray<T>::Resize(std::size_t newsize, T t) {
  TREELITE_CHECK(owned_buffer_) << "Cannot resize when using a foreign buffer; clone first";
  std::size_t oldsize = Size();
  if (newsize > capacity_) {
    std::size_t newcapacity = capacity_;
    if (newcapacity == 0) {
      newcapacity = 1;
    }
    while (newcapacity < newsize) {
      newcapacity *= 2;
    }
    Reserve(newcapacity);
  }
  for (std::size_t i = oldsize; i < newsize; ++i) {
    buffer_[i] = t;
  }
  size_ = newsize;
}

template <typename T>
inline void ContiguousArray<T>::Clear() {
  TREELITE_CHECK(owned_buffer_) << "Cannot clear when using a foreign buffer; clone first";
  Resize(0);
}

template <typename T>
inline void ContiguousArray<T>::PushBack(T t) {
  TREELITE_CHECK(owned_buffer_) << "Cannot add element when using a foreign buffer; clone first";
  if (size_ == capacity_) {
    if (capacity_ == 0) {
      Reserve(1);
    } else {
      Reserve(capacity_ * 2);
    }
  }
  buffer_[size_++] = t;
}

template <typename T>
inline void ContiguousArray<T>::Extend(std::vector<T> const& other) {
  TREELITE_CHECK(owned_buffer_) << "Cannot add elements when using a foreign buffer; clone first";
  if (other.empty()) {
    return;  // Appending an empty vector is a no-op
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
  std::memcpy(&buffer_[size_], static_cast<void const*>(other.data()), sizeof(T) * other.size());
  size_ = newsize;
}

template <typename T>
inline void ContiguousArray<T>::Extend(ContiguousArray const& other) {
  TREELITE_CHECK(owned_buffer_) << "Cannot add elements when using a foreign buffer; clone first";
  if (other.Empty()) {
    return;  // Appending an empty vector is a no-op
  }
  std::size_t newsize = size_ + other.Size();
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
  std::memcpy(&buffer_[size_], static_cast<void const*>(other.Data()), sizeof(T) * other.Size());
  size_ = newsize;
}

template <typename T>
inline std::vector<T> ContiguousArray<T>::AsVector() const {
  auto const size = Size();
  std::vector<T> vec(size);
  std::copy(buffer_, buffer_ + size, vec.begin());
  return vec;
}

template <typename T>
inline bool ContiguousArray<T>::operator==(ContiguousArray const& other) {
  if (Size() != other.Size()) {
    return false;
  }
  for (std::size_t i = 0; i < Size(); ++i) {
    if (buffer_[i] != other.buffer_[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
inline T& ContiguousArray<T>::operator[](std::size_t idx) {
  return buffer_[idx];
}

template <typename T>
inline T const& ContiguousArray<T>::operator[](std::size_t idx) const {
  return buffer_[idx];
}

template <typename T>
inline T& ContiguousArray<T>::at(std::size_t idx) {
  TREELITE_CHECK_LT(idx, Size()) << "nid out of range";
  return buffer_[idx];
}

template <typename T>
inline T const& ContiguousArray<T>::at(std::size_t idx) const {
  TREELITE_CHECK_LT(idx, Size()) << "nid out of range";
  return buffer_[idx];
}

template <typename T>
inline T& ContiguousArray<T>::at(int idx) {
  if (idx < 0 || static_cast<std::size_t>(idx) >= Size()) {
    TREELITE_LOG(FATAL) << "nid out of range";
  }
  return buffer_[static_cast<std::size_t>(idx)];
}

template <typename T>
inline T const& ContiguousArray<T>::at(int idx) const {
  if (idx < 0 || static_cast<std::size_t>(idx) >= Size()) {
    TREELITE_LOG(FATAL) << "nid out of range";
  }
  return buffer_[static_cast<std::size_t>(idx)];
}

}  // namespace treelite

#endif  // TREELITE_DETAIL_CONTIGUOUS_ARRAY_H_
