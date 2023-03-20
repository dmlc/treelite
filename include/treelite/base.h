/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file base.h
 * \brief defines configuration macros of Treelite
 * \author Hyunsu Cho
 */
#ifndef TREELITE_BASE_H_
#define TREELITE_BASE_H_

#include <treelite/logging.h>
#include <treelite/error.h>
#include <treelite/typeinfo.h>
#include <cstdint>
#include <typeinfo>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace treelite {

/*! \brief float type to be used internally */
typedef float tl_float;
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

/*!
 * \brief get string representation of comparison operator
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
 * \brief Build operator enum from a string representation
 * \param name Name of operator
 * \return Operator enum
 */
inline Operator LookupOperatorByName(const std::string& name) {
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

/*!
 * \brief Get string representation of split type
 * \param type Type of a split
 * \return String representation
 */
inline std::string SplitFeatureTypeName(SplitFeatureType type) {
  switch (type) {
    case SplitFeatureType::kNone: return "none";
    case SplitFeatureType::kNumerical: return "numerical";
    case SplitFeatureType::kCategorical: return "categorical";
    default: return "";
  }
}

/*!
 * \brief Build split type from a string representation
 * \param name Name of split type
 * \return Split type enum
 */
inline SplitFeatureType LookupSplitFeatureTypeByName(const std::string& name) {
  if (name == "none") {
    return SplitFeatureType::kNone;
  } else if (name == "numerical") {
    return SplitFeatureType::kNumerical;
  } else if (name == "categorical") {
    return SplitFeatureType::kCategorical;
  } else {
    TREELITE_LOG(FATAL) << "Unknown split type: " << name;
    return SplitFeatureType::kNone;
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
template <typename ElementType, typename ThresholdType>
inline bool CompareWithOp(ElementType lhs, Operator op, ThresholdType rhs) {
  switch (op) {
    case Operator::kEQ: return lhs == rhs;
    case Operator::kLT: return lhs <  rhs;
    case Operator::kLE: return lhs <= rhs;
    case Operator::kGT: return lhs >  rhs;
    case Operator::kGE: return lhs >= rhs;
    default:
      TREELITE_LOG(FATAL) << "operator undefined: " << static_cast<int>(op);
      return false;
  }
}

}  // namespace treelite

#endif  // TREELITE_BASE_H_
