/*!
* Copyright by 2017 Contributors
* \file common.h
* \brief Some useful math utilities
* \author Philip Cho
*/
#ifndef TREELITE_COMMON_MATH_H_
#define TREELITE_COMMON_MATH_H_

#include <cfloat>
#include <cmath>

namespace treelite {
namespace common {
namespace math {

/*!
 * \brief check for NaN (Not a Number)
 * \param value value to check
 * \return whether the given value is NaN or not
 * \tparam type of value (should be a floating-point value)
 */
template <typename T>
inline bool CheckNAN(T value) {
#ifdef _MSC_VER
  return (_isnan(value) != 0);
#else
  return std::isnan(value);
#endif
}

}  // namespace math
}  // namespace common
}  // namespace treelite

#endif  // TREELITE_COMMON_MATH_H_
