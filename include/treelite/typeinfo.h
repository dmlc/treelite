/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file typeinfo.h
 * \brief Defines TypeInfo class and utilities
 * \author Hyunsu Cho
 */

#ifndef TREELITE_TYPEINFO_H_
#define TREELITE_TYPEINFO_H_

#include <cstdint>
#include <typeinfo>
#include <string>
#include <unordered_map>
#include <sstream>
#include <utility>
#include <type_traits>

namespace treelite {

/*! \brief Types used by thresholds and leaf outputs */
enum class TypeInfo : std::uint8_t {
  kInvalid = 0,
  kUInt32 = 1,
  kFloat32 = 2,
  kFloat64 = 3
};
static_assert(std::is_same<std::underlying_type<TypeInfo>::type, std::uint8_t>::value,
              "TypeInfo must use uint8_t as underlying type");

/*! \brief conversion table from string to TypeInfo, defined in tables.cc */
TypeInfo GetTypeInfoByName(const std::string& str);

/*!
 * \brief Get string representation of type info
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
inline TypeInfo TypeToInfo() {
  if (std::is_same<T, std::uint32_t>::value) {
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
template <template<class> class Dispatcher, typename ...Args>
inline auto DispatchWithTypeInfo(TypeInfo type, Args&& ...args) {
  switch (type) {
  case TypeInfo::kUInt32:
    return Dispatcher<std::uint32_t>::Dispatch(std::forward<Args>(args)...);
  case TypeInfo::kFloat32:
    return Dispatcher<float>::Dispatch(std::forward<Args>(args)...);
  case TypeInfo::kFloat64:
    return Dispatcher<double>::Dispatch(std::forward<Args>(args)...);
  case TypeInfo::kInvalid:
  default:
    throw std::runtime_error(std::string("Invalid type: ") + TypeInfoToString(type));
  }
  return Dispatcher<double>::Dispatch(std::forward<Args>(args)...);  // avoid missing return error
}

/*!
 * \brief Given the types for thresholds and leaf outputs, validate that they consist of a valid
 *        combination for a model and then dispatch a function with the corresponding template args.
 *        More precisely, we shall call Dispatcher<ThresholdType, LeafOutputType>::Dispatch() where
 *        the template args ThresholdType and LeafOutputType correspond to the parameters
 *        `threshold_type` and `leaf_output_type`, respectively.
 * \tparam Dispatcher Function object that takes in two template args.
 *         It must have a Dispatch() static function.
 * \tparam Parameter pack, to forward an arbitrary number of args to Dispatcher::Dispatch()
 * \param threshold_type TypeInfo indicating the type of thresholds
 * \param leaf_output_type TypeInfo indicating the type of leaf outputs
 * \param args Other extra parameters to pass to Dispatcher::Dispatch()
 * \return Whatever that's returned by the dispatcher
 */
template <template<class, class> class Dispatcher, typename ...Args>
inline auto DispatchWithModelTypes(
    TypeInfo threshold_type, TypeInfo leaf_output_type, Args&& ...args) {
  auto error_threshold_type = [threshold_type]() {
    std::ostringstream oss;
    oss << "Invalid threshold type: " << treelite::TypeInfoToString(threshold_type);
    return oss.str();
  };
  auto error_leaf_output_type = [threshold_type, leaf_output_type]() {
    std::ostringstream oss;
    oss << "Cannot use leaf output type " << treelite::TypeInfoToString(leaf_output_type)
        << " with threshold type " << treelite::TypeInfoToString(threshold_type);
    return oss.str();
  };
  switch (threshold_type) {
  case treelite::TypeInfo::kFloat32:
    switch (leaf_output_type) {
    case treelite::TypeInfo::kUInt32:
      return Dispatcher<float, std::uint32_t>::Dispatch(std::forward<Args>(args)...);
    case treelite::TypeInfo::kFloat32:
      return Dispatcher<float, float>::Dispatch(std::forward<Args>(args)...);
    default:
      throw std::runtime_error(error_leaf_output_type());
      break;
    }
    break;
  case treelite::TypeInfo::kFloat64:
    switch (leaf_output_type) {
    case treelite::TypeInfo::kUInt32:
      return Dispatcher<double, std::uint32_t>::Dispatch(std::forward<Args>(args)...);
    case treelite::TypeInfo::kFloat64:
      return Dispatcher<double, double>::Dispatch(std::forward<Args>(args)...);
    default:
      throw std::runtime_error(error_leaf_output_type());
      break;
    }
    break;
  default:
    throw std::runtime_error(error_threshold_type());
    break;
  }
  return Dispatcher<double, double>::Dispatch(std::forward<Args>(args)...);
    // avoid missing return value warning
}

}  // namespace treelite

#endif  // TREELITE_TYPEINFO_H_
