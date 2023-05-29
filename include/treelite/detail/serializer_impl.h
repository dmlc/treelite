/*!
 * Copyright (c) 2023 by Contributors
 * \file serializer_impl.h
 * \brief Building blocks for serializers
 * \author Hyunsu Cho
 */

#ifndef TREELITE_DETAIL_SERIALIZER_IMPL_H_
#define TREELITE_DETAIL_SERIALIZER_IMPL_H_

#include <treelite/contiguous_array.h>
#include <treelite/logging.h>
#include <treelite/pybuffer_frame.h>
#include <treelite/task_type.h>
#include <treelite/typeinfo.h>

#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <type_traits>

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
    return (std::is_unsigned<T>::value ? "=B" : "=b");
  case 2:
    return (std::is_unsigned<T>::value ? "=H" : "=h");
  case 4:
    if (std::is_integral<T>::value) {
      return (std::is_unsigned<T>::value ? "=L" : "=l");
    } else {
      TREELITE_CHECK(std::is_floating_point<T>::value) << "Could not infer format string";
      return "=f";
    }
  case 8:
    if (std::is_integral<T>::value) {
      return (std::is_unsigned<T>::value ? "=Q" : "=q");
    } else {
      TREELITE_CHECK(std::is_floating_point<T>::value) << "Could not infer format string";
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
  static_assert(std::is_arithmetic<T>::value,
      "Use GetPyBufferFromArray(vec, format) for composite types; specify format string manually");
  return GetPyBufferFromArray(vec, InferFormatString<T>());
}

inline PyBufferFrame GetPyBufferFromScalar(void* data, char const* format, std::size_t itemsize) {
  return GetPyBufferFromArray(data, format, itemsize, 1);
}

template <typename T>
inline PyBufferFrame GetPyBufferFromScalar(T* scalar, char const* format) {
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

inline void SkipOptFieldInStream(std::istream& is) {
  std::uint16_t elem_size;
  std::uint64_t nelem;
  ReadScalarFromStream(&elem_size, is);
  ReadScalarFromStream(&nelem, is);

  const std::uint64_t nbytes = elem_size * nelem;
  TREELITE_CHECK_LE(nbytes, std::numeric_limits<std::streamoff>::max());  // NOLINT
  is.seekg(static_cast<std::streamoff>(nbytes), std::ios::cur);
}

}  // namespace treelite::detail::serializer

#endif  // TREELITE_DETAIL_SERIALIZER_IMPL_H_
