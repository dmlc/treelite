/*!
 * Copyright (c) 2020 by Contributors
 * \file frontend_impl.h
 * \brief Implementation for frontend.h
 * \author Hyunsu Cho
 */

#ifndef TREELITE_FRONTEND_IMPL_H_
#define TREELITE_FRONTEND_IMPL_H_

#include <treelite/error.h>

#include <string>

namespace treelite {
namespace frontend {

template <typename Func>
inline auto Value::Dispatch(Func func) {
  switch (type_) {
  case TypeInfo::kUInt32:
    return func(Get<uint32_t>());
  case TypeInfo::kFloat32:
    return func(Get<float>());
  case TypeInfo::kFloat64:
    return func(Get<double>());
  default:
    throw Error(
        std::string("Unknown value type detected: ") + std::to_string(static_cast<int>(type_)));
    return func(Get<double>());  // avoid "missing return" warning
  }
}

template <typename Func>
inline auto Value::Dispatch(Func func) const {
  switch (type_) {
  case TypeInfo::kUInt32:
    return func(Get<uint32_t>());
  case TypeInfo::kFloat32:
    return func(Get<float>());
  case TypeInfo::kFloat64:
    return func(Get<double>());
  default:
    throw Error(
        std::string("Unknown value type detected: ") + std::to_string(static_cast<int>(type_)));
    return func(Get<double>());  // avoid "missing return" warning
  }
}

}  // namespace frontend
}  // namespace treelite

#endif  // TREELITE_FRONTEND_IMPL_H_
