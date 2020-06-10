/*!
 * Copyright (c) 2017 by Contributors
 * \file optable.cc
 * \author Philip Cho
 * \brief Conversion table from string to Operator
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

}  // namespace treelite
