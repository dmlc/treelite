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

const std::unordered_map<std::string, TypeInfo> typeinfo_table{
  {"uint32", TypeInfo::kUInt32},
  {"float32", TypeInfo::kFloat32},
  {"float64", TypeInfo::kFloat64}
};

}  // namespace treelite
