/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file optable.cc
 * \author Hyunsu Cho
 * \brief Conversion tables to obtain Operator from string
 */

#include <treelite/base.h>
#include <string>
#include <unordered_map>

namespace treelite {

const std::unordered_map<std::string, Operator> optable{
  {"==", Operator::kEQ},
  {"<",  Operator::kLT},
  {"<=", Operator::kLE},
  {">",  Operator::kGT},
  {">=", Operator::kGE}
};

}  // namespace treelite
