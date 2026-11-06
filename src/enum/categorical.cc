/*!
 * Copyright (c) 2026 by Contributors
 * \file categorical.cc
 * \author Hyunsu Cho
 * \brief Utilities for CategoricalKind enum
 */

#include <string>

#include <treelite/enum/categorical.h>
#include <treelite/logging.h>

namespace treelite {

std::string CategoricalKindToString(CategoricalKind kind) {
  switch (kind) {
  case CategoricalKind::kNone:
    return "none";
  case CategoricalKind::kInteger:
    return "integer";
  case CategoricalKind::kString:
    return "string";
  default:
    return "";
  }
}

CategoricalKind CategoricalKindFromString(std::string const& name) {
  if (name == "none") {
    return CategoricalKind::kNone;
  }
  if (name == "integer") {
    return CategoricalKind::kInteger;
  }
  if (name == "string") {
    return CategoricalKind::kString;
  }
  TREELITE_LOG(FATAL) << "Unknown operator: " << name;
  return CategoricalKind::kNone;
}

}  // namespace treelite
