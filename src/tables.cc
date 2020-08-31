/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file tables.cc
 * \author Hyunsu Cho
 * \brief Conversion tables to obtain Operator and TypeInfo from strings
 */

#include <treelite/base.h>

namespace treelite {

const std::unordered_map<std::string, Operator> optable{
  {"==", Operator::kEQ},
  {"<",  Operator::kLT},
  {"<=", Operator::kLE},
  {">",  Operator::kGT},
  {">=", Operator::kGE}
};

const std::unordered_map<std::string, TypeInfo> typeinfo_table{
  {"uint32", TypeInfo::kUInt32},
  {"float32", TypeInfo::kFloat32},
  {"float64", TypeInfo::kFloat64}
};

}  // namespace treelite
