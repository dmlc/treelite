/*!
 * Copyright (c) 2026 by Contributors
 * \file categorical.h
 * \brief Define enum type CategoricalKind
 * \author Hyunsu Cho
 */

#pragma once

#include <cstdint>
#include <string>

namespace treelite {

/*! \brief Type of categorical features */
enum class CategoricalKind : std::int8_t {
  kNone = 0,
  kInteger = 1,
  kString = 2,
};

/*! \brief Get string representation of CategoricalKind */
std::string CategoricalKindToString(CategoricalKind kind);

/*! \brief Get CategoricalKind from string */
CategoricalKind CategoricalKindFromString(std::string const& name);

}  // namespace treelite
