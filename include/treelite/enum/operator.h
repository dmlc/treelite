/*!
 * Copyright (c) 2023 by Contributors
 * \file operator.h
 * \brief Define enum type Operator
 * \author Hyunsu Cho
 */

#ifndef TREELITE_ENUM_OPERATOR_H_
#define TREELITE_ENUM_OPERATOR_H_

#include <cstdint>
#include <string>

namespace treelite {

/*! \brief Type of comparison operators used in numerical test nodes */
enum class Operator : std::int8_t {
  kNone,
  kEQ, /*!< operator == */
  kLT, /*!< operator <  */
  kLE, /*!< operator <= */
  kGT, /*!< operator >  */
  kGE, /*!< operator >= */
};

/*! \brief Get string representation of Operator */
std::string OperatorToString(Operator type);

/*! \brief Get Operator from string */
Operator OperatorFromString(std::string const& name);

}  // namespace treelite

#endif  // TREELITE_ENUM_OPERATOR_H_
