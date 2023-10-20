/*!
 * Copyright (c) 2023 by Contributors
 * \file contiguous_array.h
 * \brief A simple array container, with owned or non-owned (externally allocated) buffer
 * \author Hyunsu Cho
 */

#ifndef TREELITE_CONTIGUOUS_ARRAY_H_
#define TREELITE_CONTIGUOUS_ARRAY_H_

#include <cstddef>
#include <vector>

namespace treelite {

template <typename T>
class ContiguousArray {
 public:
  ContiguousArray();
  ~ContiguousArray();
  // NOTE: use Clone to make deep copy; copy constructors disabled
  ContiguousArray(ContiguousArray const&) = delete;
  ContiguousArray& operator=(ContiguousArray const&) = delete;
  explicit ContiguousArray(std::vector<T> const& other);
  ContiguousArray& operator=(std::vector<T> const& other);
  ContiguousArray(ContiguousArray&& other) noexcept;
  ContiguousArray& operator=(ContiguousArray&& other) noexcept;
  inline ContiguousArray Clone() const;
  inline void UseForeignBuffer(void* prealloc_buf, std::size_t size);
  inline T* Data();
  inline T const* Data() const;
  inline T* End();
  inline T const* End() const;
  inline T& Back();
  inline T const& Back() const;
  inline std::size_t Size() const;
  inline bool Empty() const;
  inline void Reserve(std::size_t newsize);
  inline void Resize(std::size_t newsize);
  inline void Resize(std::size_t newsize, T t);
  inline void Clear();
  inline void PushBack(T t);
  inline void Extend(std::vector<T> const& other);
  inline void Extend(ContiguousArray const& other);
  inline std::vector<T> AsVector() const;
  inline bool operator==(ContiguousArray const& other);
  /* Unsafe access, no bounds checking */
  inline T& operator[](std::size_t idx);
  inline T const& operator[](std::size_t idx) const;
  /* Safe access, with bounds checking */
  inline T& at(std::size_t idx);
  inline T const& at(std::size_t idx) const;
  /* Safe access, with bounds checking + check against non-existent node (<0) */
  inline T& at(int idx);
  inline T const& at(int idx) const;
  static_assert(std::is_pod<T>::value, "T must be POD");

 private:
  T* buffer_;
  std::size_t size_;
  std::size_t capacity_;
  bool owned_buffer_;
};

}  // namespace treelite

#include <treelite/detail/contiguous_array.h>

#endif  // TREELITE_CONTIGUOUS_ARRAY_H_
