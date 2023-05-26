/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file typeinfo.h
 * \brief Defines TypeInfo class and utilities
 * \author Hyunsu Cho
 */

#ifndef TREELITE_TYPEINFO_H_
#define TREELITE_TYPEINFO_H_

#include <treelite/error.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <utility>

namespace treelite {

/*! \brief Types used by thresholds and leaf outputs */
enum class TypeInfo : uint8_t { kInvalid = 0, kUInt32 = 1, kFloat32 = 2, kFloat64 = 3 };
static_assert(std::is_same<std::underlying_type<TypeInfo>::type, uint8_t>::value,
    "TypeInfo must use uint8_t as underlying type");

/*! \brief conversion table from string to TypeInfo, defined in tables.cc */
TypeInfo GetTypeInfoByName(std::string const& str);

/*!
 * \brief Get string representation of type info
 * \param type Type info
 * \return String representation
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
    throw Error("Unrecognized type");
    return "";
  }
}

/*!
 * \brief Convert a template type into a type info
 * \tparam template type to be converted
 * \return TypeInfo corresponding to the template type arg
 */
template <typename T>
inline TypeInfo TypeToInfo() {
  if (std::is_same<T, std::uint32_t>::value) {
    return TypeInfo::kUInt32;
  } else if (std::is_same<T, float>::value) {
    return TypeInfo::kFloat32;
  } else if (std::is_same<T, double>::value) {
    return TypeInfo::kFloat64;
  } else {
    throw Error(std::string("Unrecognized Value type") + typeid(T).name());
    return TypeInfo::kInvalid;
  }
}

/*!
 * \brief Given a TypeInfo, dispatch a function with the corresponding template arg. More precisely,
 *        we shall call Dispatcher<T>::Dispatch() where the template arg T corresponds to the
 *        `type` parameter.
 * \tparam Dispatcher Function object that takes in one template arg.
 *         It must have a Dispatch() static function.
 * \tparam Parameter pack, to forward an arbitrary number of args to Dispatcher::Dispatch()
 * \param type TypeInfo corresponding to the template arg T with which
 *             Dispatcher<T>::Dispatch() is called.
 * \param args Other extra parameters to pass to Dispatcher::Dispatch()
 * \return Whatever that's returned by the dispatcher
 */
template <template <class> class Dispatcher, typename... Args>
inline auto DispatchWithTypeInfo(TypeInfo type, Args&&... args) {
  switch (type) {
  case TypeInfo::kUInt32:
    return Dispatcher<uint32_t>::Dispatch(std::forward<Args>(args)...);
  case TypeInfo::kFloat32:
    return Dispatcher<float>::Dispatch(std::forward<Args>(args)...);
  case TypeInfo::kFloat64:
    return Dispatcher<double>::Dispatch(std::forward<Args>(args)...);
  case TypeInfo::kInvalid:
  default:
    throw Error(std::string("Invalid type: ") + TypeInfoToString(type));
  }
  return Dispatcher<double>::Dispatch(std::forward<Args>(args)...);  // avoid missing return error
}

}  // namespace treelite

#endif  // TREELITE_TYPEINFO_H_
