/*!
 * Copyright (c) 2023 by Contributors
 * \file serializer.h
 * \brief Building blocks for serializers
 * \author Hyunsu Cho
 */

#ifndef TREELITE_DETAIL_SERIALIZER_H_
#define TREELITE_DETAIL_SERIALIZER_H_

#include <treelite/contiguous_array.h>
#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/tree_node_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/logging.h>
#include <treelite/pybuffer_frame.h>

#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

namespace treelite::detail::serializer {

inline PyBufferFrame GetPyBufferFromArray(
    void* data, char const* format, std::size_t itemsize, std::size_t nitem) {
  return PyBufferFrame{data, const_cast<char*>(format), itemsize, nitem};
}

// Infer format string from data type
template <typename T>
inline char const* InferFormatString() {
  switch (sizeof(T)) {
  case 1:
    return (std::is_unsigned_v<T> ? "=B" : "=b");
  case 2:
    return (std::is_unsigned_v<T> ? "=H" : "=h");
  case 4:
    if (std::is_integral_v<T>) {
      return (std::is_unsigned_v<T> ? "=L" : "=l");
    } else {
      TREELITE_CHECK(std::is_floating_point_v<T>) << "Could not infer format string";
      return "=f";
    }
  case 8:
    if (std::is_integral_v<T>) {
      return (std::is_unsigned_v<T> ? "=Q" : "=q");
    } else {
      TREELITE_CHECK(std::is_floating_point_v<T>) << "Could not infer format string";
      return "=d";
    }
  default:
    TREELITE_LOG(FATAL) << "Unrecognized type";
  }
  return nullptr;
}

template <typename T>
inline PyBufferFrame GetPyBufferFromArray(ContiguousArray<T>* vec, char const* format) {
  return GetPyBufferFromArray(static_cast<void*>(vec->Data()), format, sizeof(T), vec->Size());
}

template <typename T>
inline PyBufferFrame GetPyBufferFromArray(ContiguousArray<T>* vec) {
  static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>,
      "Use GetPyBufferFromArray(vec, format) for composite types; specify format string manually");
  return GetPyBufferFromArray(vec, InferFormatString<T>());
}

inline PyBufferFrame GetPyBufferFromScalar(void* data, char const* format, std::size_t itemsize) {
  return GetPyBufferFromArray(data, format, itemsize, 1);
}

inline PyBufferFrame GetPyBufferFromString(std::string* str) {
  return GetPyBufferFromArray(str->data(), "=c", 1, str->length());
}

template <typename T>
inline PyBufferFrame GetPyBufferFromScalar(T* scalar, char const* format) {
  static_assert(std::is_standard_layout_v<T>, "T must be in the standard layout");
  return GetPyBufferFromScalar(static_cast<void*>(scalar), format, sizeof(T));
}

inline PyBufferFrame GetPyBufferFromScalar(TypeInfo* scalar) {
  using T = std::underlying_type_t<TypeInfo>;
  return GetPyBufferFromScalar(reinterpret_cast<T*>(scalar), InferFormatString<T>());
}

inline PyBufferFrame GetPyBufferFromScalar(TaskType* scalar) {
  using T = std::underlying_type_t<TaskType>;
  return GetPyBufferFromScalar(reinterpret_cast<T*>(scalar), InferFormatString<T>());
}

inline PyBufferFrame GetPyBufferFromScalar(TreeNodeType* scalar) {
  using T = std::underlying_type_t<TreeNodeType>;
  return GetPyBufferFromScalar(reinterpret_cast<T*>(scalar), InferFormatString<T>());
}

template <typename T>
inline PyBufferFrame GetPyBufferFromScalar(T* scalar) {
  static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>,
      "Use GetPyBufferFromScalar(scalar, format) for composite types; "
      "specify format string manually");
  return GetPyBufferFromScalar(scalar, InferFormatString<T>());
}

template <typename T>
inline void InitArrayFromPyBuffer(ContiguousArray<T>* vec, PyBufferFrame frame) {
  TREELITE_CHECK_EQ(sizeof(T), frame.itemsize) << "Incorrect itemsize";
  vec->UseForeignBuffer(frame.buf, frame.nitem);
}

template <typename T>
inline void InitArrayFromPyBufferWithCopy(ContiguousArray<T>* vec, PyBufferFrame frame) {
  TREELITE_CHECK_EQ(sizeof(T), frame.itemsize) << "Incorrect itemsize";
  ContiguousArray<T> new_vec;
  new_vec.UseForeignBuffer(frame.buf, frame.nitem);
  *vec = std::move(new_vec);
}

inline void InitStringFromPyBuffer(std::string* str, PyBufferFrame frame) {
  TREELITE_CHECK_EQ(sizeof(char), frame.itemsize) << "Incorrect itemsize";
  *str = std::string(static_cast<char*>(frame.buf), frame.nitem);
}

inline void InitScalarFromPyBuffer(TypeInfo* scalar, PyBufferFrame frame) {
  using T = std::underlying_type_t<TypeInfo>;
  TREELITE_CHECK_EQ(sizeof(T), frame.itemsize) << "Incorrect itemsize";
  TREELITE_CHECK_EQ(frame.nitem, 1) << "nitem must be 1 for a scalar";
  T* t = static_cast<T*>(frame.buf);
  *scalar = static_cast<TypeInfo>(*t);
}

inline void InitScalarFromPyBuffer(TaskType* scalar, PyBufferFrame frame) {
  using T = std::underlying_type_t<TaskType>;
  TREELITE_CHECK_EQ(sizeof(T), frame.itemsize) << "Incorrect itemsize";
  TREELITE_CHECK_EQ(frame.nitem, 1) << "nitem must be 1 for a scalar";
  T* t = static_cast<T*>(frame.buf);
  *scalar = static_cast<TaskType>(*t);
}

template <typename T>
inline void InitScalarFromPyBuffer(T* scalar, PyBufferFrame frame) {
  static_assert(std::is_standard_layout_v<T>, "T must be in the standard layout");
  TREELITE_CHECK_EQ(sizeof(T), frame.itemsize) << "Incorrect itemsize";
  TREELITE_CHECK_EQ(frame.nitem, 1) << "nitem must be 1 for a scalar";
  T* t = static_cast<T*>(frame.buf);
  *scalar = *t;
}

template <typename T>
inline void ReadScalarFromStream(T* scalar, std::istream& is) {
  static_assert(std::is_standard_layout_v<T>, "T must be in the standard layout");
  is.read(reinterpret_cast<char*>(scalar), sizeof(T));
}

template <typename T>
inline void WriteScalarToStream(T* scalar, std::ostream& os) {
  static_assert(std::is_standard_layout_v<T>, "T must be in the standard layout");
  os.write(reinterpret_cast<char const*>(scalar), sizeof(T));
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
  auto const nelem = static_cast<std::uint64_t>(vec->Size());
  os.write(reinterpret_cast<char const*>(&nelem), sizeof(nelem));
  if (nelem == 0) {
    return;  // handle empty arrays
  }
  os.write(reinterpret_cast<char const*>(vec->Data()), sizeof(T) * vec->Size());
}

inline void ReadStringFromStream(std::string* str, std::istream& is) {
  std::uint64_t str_len;
  is.read(reinterpret_cast<char*>(&str_len), sizeof(str_len));
  if (str_len == 0) {
    return;  // handle empty string
  }
  *str = std::string(str_len, '\0');
  is.read(str->data(), sizeof(char) * str_len);
}

inline void WriteStringToStream(std::string* str, std::ostream& os) {
  static_assert(sizeof(std::uint64_t) >= sizeof(std::size_t), "size_t too large");
  auto const str_len = static_cast<std::uint64_t>(str->length());
  os.write(reinterpret_cast<char const*>(&str_len), sizeof(str_len));
  if (str_len == 0) {
    return;  // handle empty string
  }
  os.write(str->data(), sizeof(char) * str->length());
}

inline void SkipOptionalFieldInStream(std::istream& is) {
  std::string field_name;
  ReadStringFromStream(&field_name, is);

  std::uint64_t elem_size, nelem;
  ReadScalarFromStream(&elem_size, is);
  ReadScalarFromStream(&nelem, is);

  std::uint64_t const nbytes = elem_size * nelem;
  TREELITE_CHECK_LE(nbytes, std::numeric_limits<std::streamoff>::max());  // NOLINT
  is.seekg(static_cast<std::streamoff>(nbytes), std::ios::cur);
}

}  // namespace treelite::detail::serializer

#endif  // TREELITE_DETAIL_SERIALIZER_H_
