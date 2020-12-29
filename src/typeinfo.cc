/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file typeinfo.cc
 * \author Hyunsu Cho
 * \brief Conversion tables to obtain TypeInfo from string
 */

// Do not include other Treelite headers here, to minimize cross-header dependencies

#include <treelite/typeinfo.h>
#include <string>
#include <unordered_map>

namespace treelite {

TypeInfo GetTypeInfoByName(const std::string& str) {
  if (str == "uint32") {
    return TypeInfo::kUInt32;
  } else if (str == "float32") {
    return TypeInfo::kFloat32;
  } else if (str == "float64") {
    return TypeInfo::kFloat64;
  } else {
    throw std::runtime_error("Unrecognized type");
    return TypeInfo::kInvalid;
  }
}

}  // namespace treelite
