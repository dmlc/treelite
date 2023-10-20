/*!
 * Copyright (c) 2023 by Contributors
 * \file operator.cc
 * \author Hyunsu Cho
 * \brief Utilities for Operator enum
 */

#include <treelite/enum/operator.h>
#include <treelite/logging.h>

#include <string>

namespace treelite {

/*! \brief Get string representation of Operator */
std::string OperatorToString(Operator op) {
  switch (op) {
  case Operator::kEQ:
    return "==";
  case Operator::kLT:
    return "<";
  case Operator::kLE:
    return "<=";
  case Operator::kGT:
    return ">";
  case Operator::kGE:
    return ">=";
  default:
    return "";
  }
}

/*! \brief Get Operator from string */
Operator OperatorFromString(std::string const& name) {
  if (name == "==") {
    return Operator::kEQ;
  } else if (name == "<") {
    return Operator::kLT;
  } else if (name == "<=") {
    return Operator::kLE;
  } else if (name == ">") {
    return Operator::kGT;
  } else if (name == ">=") {
    return Operator::kGE;
  } else {
    TREELITE_LOG(FATAL) << "Unknown operator: " << name;
    return Operator::kNone;
  }
}

}  // namespace treelite
