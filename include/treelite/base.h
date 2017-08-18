/*!
 * Copyright 2017 by Contributors
 * \file base.h
 * \brief defines configuration macros of treelite
 * \author Philip Cho
 */
#ifndef TREELITE_BASE_H_
#define TREELITE_BASE_H_

#include <unordered_map>
#include <cstdint>

namespace treelite {

/*! \brief float type to be used internally */
typedef float tl_float;
/*! \brief feature split type */
enum class SplitFeatureType : int8_t {
  kNone, kNumerical, kCategorical
};
/*! \brief comparison operators */
enum class Operator : int8_t {
  kEQ,  /*!< operator == */
  kLT,  /*!< operator <  */
  kLE,  /*!< operator <= */
  kGT,  /*!< operator >  */
  kGE   /*!< operator >= */
};
/*! \brief conversion table from string to operator, defined in optable.cc */
extern const std::unordered_map<std::string, Operator> optable;

}  // namespace treelite

#endif  // TREELITE_BASE_H_
