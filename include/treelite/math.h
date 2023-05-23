/*!
 * Copyright (c) 2020 by Contributors
 * \file math.h
 * \brief Some useful math utilities
 * \author Hyunsu Cho
 */
#ifndef TREELITE_MATH_H_
#define TREELITE_MATH_H_

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace treelite {
namespace math {

/*!
 * \brief perform binary search on the range [begin, end).
 * \param begin beginning of the search range
 * \param end end of the search range
 * \param val value being searched
 * \return iterator pointing to the value if found; end if value not found.
 * \tparam Iter type of iterator
 * \tparam T type of elements
 */
template <class Iter, class T>
Iter binary_search(Iter begin, Iter end, T const& val) {
  Iter i = std::lower_bound(begin, end, val);
  if (i != end && val == *i) {
    return i;  // found
  } else {
    return end;  // not found
  }
}

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
}  // namespace treelite

#endif  // TREELITE_MATH_H_
