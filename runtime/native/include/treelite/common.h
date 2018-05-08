/*!
 * Copyright by 2018 Contributors
 * \file common.h
 * \brief Some useful utilities
 * \author Philip Cho
 */
#ifndef TREELITE_COMMON_H_
#define TREELITE_COMMON_H_

#include <memory>

namespace treelite {
namespace common {

/*!
 * \brief construct a new object of type T and wraps it with a std::unique_ptr.
 *        This is support legacy compiles (e.g. g++ 4.8.x) that do not yet
 *        support std::make_unique<T>.
 * \param args list of arguments with which an instance of T will be constructed
 * \return unique_ptr wrapping the newly created object
 * \tparam T type of object to be constructed
 * \tparam Args variadic template for forwarded arguments
 */
template<typename T, typename ...Args>
std::unique_ptr<T> make_unique(Args&& ...args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace common
}  // namespace treelite
#endif  // TREELITE_COMMON_H_
