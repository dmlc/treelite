/*!
 * Copyright 2017 by Contributors
 * \file base.h
 * \brief defines configuration macros of Treelite
 * \author Hyunsu Cho
 */
#ifndef TREELITE_BASE_H_
#define TREELITE_BASE_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace treelite {

/*! \brief float type to be used internally */
typedef double tl_float;
/*! \brief feature split type */
enum class SplitFeatureType : int8_t {
  kNone, kNumerical, kCategorical
};
/*! \brief comparison operators */
enum class Operator : int8_t {
  kNone,
  kEQ,  /*!< operator == */
  kLT,  /*!< operator <  */
  kLE,  /*!< operator <= */
  kGT,  /*!< operator >  */
  kGE,  /*!< operator >= */
};
/*! \brief conversion table from string to operator, defined in optable.cc */
extern const std::unordered_map<std::string, Operator> optable;

/*!
 * \brief get string representation of comparsion operator
 * \param op comparison operator
 * \return string representation
 */
inline std::string OpName(Operator op) {
  switch (op) {
    case Operator::kEQ: return "==";
    case Operator::kLT: return "<";
    case Operator::kLE: return "<=";
    case Operator::kGT: return ">";
    case Operator::kGE: return ">=";
    default: return "";
  }
}

/*!
 * \brief perform comparison between two float's using a comparsion operator
 * The comparison will be in the form [lhs] [op] [rhs].
 * \param lhs float on the left hand side
 * \param op comparison operator
 * \param rhs float on the right hand side
 * \return whether [lhs] [op] [rhs] is true or not
 */
inline bool CompareWithOp(tl_float lhs, Operator op, tl_float rhs) {
  switch (op) {
    case Operator::kEQ: return lhs == rhs;
    case Operator::kLT: return lhs <  rhs;
    case Operator::kLE: return lhs <= rhs;
    case Operator::kGT: return lhs >  rhs;
    case Operator::kGE: return lhs >= rhs;
    default:
      throw std::runtime_error("operator undefined");
      return false;
  }
}

}  // namespace treelite

#endif  // TREELITE_BASE_H_
