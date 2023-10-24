/*!
 * Copyright (c) 2017-2023 by Contributors
 * \file typeinfo.cc
 * \author Hyunsu Cho
 * \brief Utilities for TypeInfo enum
 */

#include <treelite/enum/typeinfo.h>
#include <treelite/logging.h>

#include <string>

namespace treelite {

std::string TypeInfoToString(treelite::TypeInfo info) {
  switch (info) {
  case treelite::TypeInfo::kInvalid:
    return "invalid";
  case treelite::TypeInfo::kUInt32:
    return "uint32";
  case treelite::TypeInfo::kFloat32:
    return "float32";
  case treelite::TypeInfo::kFloat64:
    return "float64";
  default:
    TREELITE_LOG(FATAL) << "Unrecognized type";
    return "";
  }
}

TypeInfo TypeInfoFromString(std::string const& str) {
  if (str == "uint32") {
    return TypeInfo::kUInt32;
  } else if (str == "float32") {
    return TypeInfo::kFloat32;
  } else if (str == "float64") {
    return TypeInfo::kFloat64;
  } else {
    TREELITE_LOG(FATAL) << "Unrecognized type: " << str;
    return TypeInfo::kInvalid;
  }
}

}  // namespace treelite
