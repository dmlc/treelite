/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file typeinfo.h
 * \brief Defines TypeInfo class and utilities
 * \author Hyunsu Cho
 */

#ifndef TREELITE_TYPEINFO_H_
#define TREELITE_TYPEINFO_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <type_traits>

namespace treelite {

/*! \brief Types used by thresholds and leaf outputs */
enum class TypeInfo : uint8_t {
  kInvalid = 0,
  kUInt32 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};
static_assert(std::is_same<std::underlying_type<TypeInfo>::type, uint8_t>::value,
              "TypeInfo must use uint8_t as underlying type");

/*! \brief conversion table from string to TypeInfo, defined in tables.cc */
extern const std::unordered_map<std::string, TypeInfo> typeinfo_table;

/*!
 * \brief Get string representation of type info
 * \param info a type info
 * \return string representation
 */
inline std::string TypeInfoToString(treelite::TypeInfo type) {
  switch (type) {
  case treelite::TypeInfo::kInvalid:
    return "invalid";
  case treelite::TypeInfo::kUInt32:
    return "uint32";
  case treelite::TypeInfo::kFloat32:
    return "float32";
  case treelite::TypeInfo::kFloat64:
    return "float64";
  default:
    throw std::runtime_error("Unrecognized type");
    return "";
  }
}

/*!
 * \brief Convert a template type into a type info
 * \tparam template type to be converted
 * \return TypeInfo corresponding to the template type arg
 */
template <typename T>
inline TypeInfo InferTypeInfoOf() {
  if (std::is_same<T, uint32_t>::value) {
    return TypeInfo::kUInt32;
  } else if (std::is_same<T, float>::value) {
    return TypeInfo::kFloat32;
  } else if (std::is_same<T, double>::value) {
    return TypeInfo::kFloat64;
  } else {
    throw std::runtime_error(std::string("Unrecognized Value type") + typeid(T).name());
    return TypeInfo::kInvalid;
  }
}

}  // namespace treelite

#endif  // TREELITE_TYPEINFO_H_
