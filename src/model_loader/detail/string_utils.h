/*!
 * Copyright (c) 2023 by Contributors
 * \file string_utils.h
 * \brief Helper functions for manipulating strings
 * \author Hyunsu Cho
 */

#ifndef SRC_MODEL_LOADER_DETAIL_STRING_UTILS_H_
#define SRC_MODEL_LOADER_DETAIL_STRING_UTILS_H_

#include <algorithm>
#include <string>

namespace treelite::model_loader::detail {

inline bool StringStartsWith(std::string const& str, std::string const& prefix) {
  return str.rfind(prefix, 0) == 0;
}

inline void StringTrimFromEnd(std::string& s) {
  s.erase(std::find_if(
              s.rbegin(), s.rend(), [](char ch) { return ch != '\n' && ch != '\r' && ch != ' '; })
              .base(),
      s.end());
}

}  // namespace treelite::model_loader::detail

#endif  // SRC_MODEL_LOADER_DETAIL_STRING_UTILS_H_
