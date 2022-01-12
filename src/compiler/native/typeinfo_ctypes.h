/*!
 * Copyright (c) 2020 by Contributors
 * \file typeinfo_ctypes.h
 * \author Hyunsu Cho
 * \brief Look up C symbols corresponding to TypeInfo
 */


#ifndef TREELITE_COMPILER_NATIVE_TYPEINFO_CTYPES_H_
#define TREELITE_COMPILER_NATIVE_TYPEINFO_CTYPES_H_

#include <treelite/base.h>
#include <string>

namespace treelite {
namespace compiler {
namespace native {

/*!
 * \brief Get string representation of the C type that's equivalent to the given type info
 * \param info a type info
 * \return string representation
 */
inline std::string TypeInfoToCTypeString(TypeInfo type) {
  switch (type) {
  case TypeInfo::kInvalid:
    throw std::runtime_error("Invalid type");
    return "";
  case TypeInfo::kUInt32:
    return "uint32_t";
  case TypeInfo::kFloat32:
    return "float";
  case TypeInfo::kFloat64:
    return "double";
  default:
    throw std::runtime_error(std::string("Unrecognized type: ")
                             + std::to_string(static_cast<int>(type)));
    return "";
  }
}

/*!
 * \brief Look up the correct variant of exp() in C that should be used with a given type
 * \param info a type info
 * \return string representation
 */
inline std::string CExpForTypeInfo(TypeInfo type) {
  switch (type) {
  case TypeInfo::kInvalid:
  case TypeInfo::kUInt32:
    throw std::runtime_error(std::string("Invalid type: ") + TypeInfoToString(type));
    return "";
  case TypeInfo::kFloat32:
    return "expf";
  case TypeInfo::kFloat64:
    return "exp";
  default:
    throw std::runtime_error(std::string("Unrecognized type: ")
                             + std::to_string(static_cast<int>(type)));
    return "";
  }
}

/*!
 * \brief Look up the correct variant of exp2() in C that should be used with a given type
 * \param info a type info
 * \return string representation
 */
inline std::string CExp2ForTypeInfo(TypeInfo type) {
  switch (type) {
  case TypeInfo::kInvalid:
  case TypeInfo::kUInt32:
    throw std::runtime_error(std::string("Invalid type: ") + TypeInfoToString(type));
    return "";
  case TypeInfo::kFloat32:
    return "exp2f";
  case TypeInfo::kFloat64:
    return "exp2";
  default:
    throw std::runtime_error(std::string("Unrecognized type: ")
                             + std::to_string(static_cast<int>(type)));
    return "";
  }
}

/*!
 * \brief Look up the correct variant of copysign() in C that should be used with a given type
 * \param info a type info
 * \return string representation
 */
inline std::string CCopySignForTypeInfo(TypeInfo type) {
  switch (type) {
  case TypeInfo::kInvalid:
  case TypeInfo::kUInt32:
    throw std::runtime_error(std::string("Invalid type: ") + TypeInfoToString(type));
    return "";
  case TypeInfo::kFloat32:
    return "copysignf";
  case TypeInfo::kFloat64:
    return "copysign";
  default:
    throw std::runtime_error(std::string("Unrecognized type: ")
                             + std::to_string(static_cast<int>(type)));
    return "";
  }
}

/*!
 * \brief Look up the correct variant of log1p() in C that should be used with a given type
 * \param info a type info
 * \return string representation
 */
inline std::string CLog1PForTypeInfo(TypeInfo type) {
  switch (type) {
  case TypeInfo::kInvalid:
  case TypeInfo::kUInt32:
    throw std::runtime_error(std::string("Invalid type: ") + TypeInfoToString(type));
    return "";
  case TypeInfo::kFloat32:
    return "log1pf";
  case TypeInfo::kFloat64:
    return "log1p";
  default:
    throw std::runtime_error(std::string("Unrecognized type: ")
                             + std::to_string(static_cast<int>(type)));
    return "";
  }
}

}  // namespace native
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_NATIVE_TYPEINFO_CTYPES_H_
