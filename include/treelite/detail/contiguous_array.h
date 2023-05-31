/*!
 * Copyright (c) 2023 by Contributors
 * \file contiguous_array.h
 * \brief Implementation for ContiguousArray
 * \author Hyunsu Cho
 */
#ifndef TREELITE_DETAIL_CONTIGUOUS_ARRAY_H_
#define TREELITE_DETAIL_CONTIGUOUS_ARRAY_H_

#include <treelite/logging.h>

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
inline void ContiguousArray<T>::Resize(std::size_t newsize) {
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
inline void ContiguousArray<T>::Resize(std::size_t newsize, T t) {
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
inline void ContiguousArray<T>::Clear() {
  if (!owned_buffer_) {
    throw Error("Cannot clear when using a foreign buffer; clone first");
  }
  Resize(0);
}

template <typename T>
inline void ContiguousArray<T>::PushBack(T t) {
  if (!owned_buffer_) {
    throw Error("Cannot add element when using a foreign buffer; clone first");
  }
  if (size_ == capacity_) {
    Reserve(capacity_ * 2);
  }
  buffer_[size_++] = t;
}

template <typename T>
inline void ContiguousArray<T>::Extend(std::vector<T> const& other) {
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
  std::memcpy(&buffer_[size_], static_cast<void const*>(other.data()), sizeof(T) * other.size());
  size_ = newsize;
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
  if (idx >= Size()) {
    throw Error("nid out of range");
  }
  return buffer_[idx];
}

template <typename T>
inline T const& ContiguousArray<T>::at(std::size_t idx) const {
  if (idx >= Size()) {
    throw Error("nid out of range");
  }
  return buffer_[idx];
}

template <typename T>
inline T& ContiguousArray<T>::at(int idx) {
  if (idx < 0 || static_cast<std::size_t>(idx) >= Size()) {
    throw Error("nid out of range");
  }
  return buffer_[static_cast<std::size_t>(idx)];
}

template <typename T>
inline T const& ContiguousArray<T>::at(int idx) const {
  if (idx < 0 || static_cast<std::size_t>(idx) >= Size()) {
    throw Error("nid out of range");
  }
  return buffer_[static_cast<std::size_t>(idx)];
}

}  // namespace treelite

#endif  // TREELITE_DETAIL_CONTIGUOUS_ARRAY_H_
