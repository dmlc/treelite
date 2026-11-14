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

#include <treelite/enum/categorical.h>
#include <treelite/tree.h>

namespace treelite {

std::variant<std::monostate, std::vector<std::int64_t>, std::vector<std::string>>
Model::GetCategoryValues(std::int32_t feature_id) const {
  TREELITE_CHECK(feature_id >= 0 && feature_id < num_feature) << "feature_id out of range";

  if (int_cat_feature_segments_.Empty() || str_cat_feature_segments_.Empty()) {
    return std::monostate{};
  }
  if (int_cat_feature_segments_[feature_id] < int_cat_feature_segments_[feature_id + 1]) {
    // Categorical feature with integer categories
    return std::vector<std::int64_t>(&int_cat_values_[int_cat_feature_segments_[feature_id]],
        &int_cat_values_[int_cat_feature_segments_[feature_id + 1]]);
  }
  if (str_cat_feature_segments_[feature_id] < str_cat_feature_segments_[feature_id + 1]) {
    // Categorical feature with string categories
    std::vector<std::string> result;
    for (std::int32_t i = str_cat_feature_segments_[feature_id];
        i < str_cat_feature_segments_[feature_id + 1]; i++) {
      result.emplace_back(
          &str_cat_values_[str_cat_offsets_[i]], &str_cat_values_[str_cat_offsets_[i + 1]]);
    }
    return result;
  }
  return std::monostate{};
}

std::optional<std::vector<std::int32_t>> Model::GetCategorySortedIdx(
    std::int32_t feature_id) const {
  TREELITE_CHECK(feature_id >= 0 && feature_id < num_feature) << "feature_id out of range";

  if (int_cat_feature_segments_.Empty() || str_cat_feature_segments_.Empty()) {
    return std::nullopt;
  }
  if (int_cat_feature_segments_[feature_id] < int_cat_feature_segments_[feature_id + 1]) {
    // Categorical feature with integer categories
    return std::vector<std::int32_t>(&int_sorted_idx_[int_cat_feature_segments_[feature_id]],
        &int_sorted_idx_[int_cat_feature_segments_[feature_id + 1]]);
  }
  if (str_cat_feature_segments_[feature_id] < str_cat_feature_segments_[feature_id + 1]) {
    // Categorical feature with string categories
    return std::vector<std::int32_t>(&str_sorted_idx_[str_cat_feature_segments_[feature_id]],
        &str_sorted_idx_[str_cat_feature_segments_[feature_id + 1]]);
  }
  // Not categorical feature
  return std::nullopt;
}

}  // namespace treelite
