/*!
 * Copyright (c) 2017-2023 by Contributors
 * \file typeinfo.h
 * \brief Defines enum type TypeInfo
 * \author Hyunsu Cho
 */

#ifndef TREELITE_ENUM_TYPEINFO_H_
#define TREELITE_ENUM_TYPEINFO_H_

#include <treelite/logging.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <typeinfo>

namespace treelite {

/*! \brief Types used by thresholds and leaf outputs */
enum class TypeInfo : std::uint8_t { kInvalid = 0, kUInt32 = 1, kFloat32 = 2, kFloat64 = 3 };

/*! \brief Get string representation of TypeInfo */
std::string TypeInfoToString(treelite::TypeInfo info);

/*! \brief Get TypeInfo from string */
TypeInfo TypeInfoFromString(std::string const& str);

/*!
 * \brief Convert a template type into a type info
 * \tparam template type to be converted
 * \return TypeInfo corresponding to the template type arg
 */
template <typename T>
inline TypeInfo TypeInfoFromType() {
  if (std::is_same_v<T, std::uint32_t>) {
    return TypeInfo::kUInt32;
  } else if (std::is_same_v<T, float>) {
    return TypeInfo::kFloat32;
  } else if (std::is_same_v<T, double>) {
    return TypeInfo::kFloat64;
  } else {
    return TypeInfo::kInvalid;
  }
}

}  // namespace treelite

#endif  // TREELITE_ENUM_TYPEINFO_H_
