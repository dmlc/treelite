/*!
 * Copyright (c) 2017-2020 by Contributors
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

/*! \brief Types used by thresholds and leaf outputs */
enum class TypeInfo : uint8_t {
  kInvalid = 0,
  kUInt32 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};
static_assert(std::is_same<std::underlying_type<TypeInfo>::type, uint8_t>::value,
              "TypeInfo must use uint8_t as underlying type");

/*! \brief conversion table from string to operator, defined in optable.cc */
extern const std::unordered_map<std::string, Operator> optable;

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
 * \brief get string representation of type info
 * \param info a type info
 * \return string representation
 */
inline std::string TypeInfoToString(treelite::TypeInfo type) {
  switch (type) {
  case treelite::TypeInfo::kInvalid:
    return "invalid";
  case treelite::TypeInfo::kUInt32:
    return "uint32";
  case treelite::TypeInfo::kFloat32:
    return "float32";
  case treelite::TypeInfo::kFloat64:
    return "float64";
  default:
    throw std::runtime_error("Unrecognized type");
    return "";
  }
}

/*!
 * \brief Convert a template type into a type info
 * \tparam template type to be converted
 * \return TypeInfo corresponding to the template type arg
 */
template <typename T>
inline TypeInfo InferTypeInfoOf() {
  if (std::is_same<T, uint32_t>::value) {
    return TypeInfo::kUInt32;
  } else if (std::is_same<T, float>::value) {
    return TypeInfo::kFloat32;
  } else if (std::is_same<T, double>::value) {
    return TypeInfo::kFloat64;
  } else {
    throw std::runtime_error(std::string("Unrecognized Value type") + typeid(T).name());
    return TypeInfo::kInvalid;
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
template <typename ThresholdType>
inline bool CompareWithOp(ThresholdType lhs, Operator op, ThresholdType rhs) {
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
