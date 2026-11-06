/*!
 * Copyright (c) 2026 by Contributors
 * \file categorical.cc
 * \brief Implementation for categorical encoding
 * \author Hyunsu Cho
 */

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <treelite/tree.h>

namespace treelite {

std::variant<std::monostate, std::vector<std::int64_t>, std::vector<std::string>>
Model::GetCategories(std::int32_t feature_id) {
  if (int_cat_feature_segments[feature_id] < int_cat_feature_segments[feature_id + 1]) {
    // Categorical feature with integer categories
    return std::vector<std::int64_t>(&int_cat_values[int_cat_feature_segments[feature_id]],
        &int_cat_values[int_cat_feature_segments[feature_id + 1]]);
  }
  if (str_cat_feature_segments[feature_id] < str_cat_feature_segments[feature_id + 1]) {
    // Categorical feature with string categories
    std::vector<std::string> result;
    for (std::int32_t i = str_cat_feature_segments[feature_id];
        i < str_cat_feature_segments[feature_id + 1]; i++) {
      result.emplace_back(
          &str_cat_values[str_cat_offsets[i]], &str_cat_values[str_cat_offsets[i + 1]]);
    }
    return result;
  }
  return std::monostate{};
}

std::optional<std::vector<std::int32_t>> Model::GetCategorySortedIdx(std::int32_t feature_id) {
  if (int_cat_feature_segments[feature_id] < int_cat_feature_segments[feature_id + 1]) {
    // Categorical feature with integer categories
    return std::vector<std::int32_t>(&int_sorted_idx[int_cat_feature_segments[feature_id]],
        &int_sorted_idx[int_cat_feature_segments[feature_id + 1]]);
  }
  if (str_cat_feature_segments[feature_id] < str_cat_feature_segments[feature_id + 1]) {
    // Categorical feature with string categories
    return std::vector<std::int32_t>(&str_sorted_idx[str_cat_feature_segments[feature_id]],
        &str_sorted_idx[str_cat_feature_segments[feature_id + 1]]);
  }
  // Not categorical feature
  return std::nullopt;
}

}  // namespace treelite
