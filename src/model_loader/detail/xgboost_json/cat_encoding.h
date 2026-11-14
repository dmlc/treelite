/*!
 * Copyright (c) 2026 by Contributors
 * \file cat_encoding.h
 * \brief Utility function to parse categorical encoding information from XGBoost
 * \author Hyunsu Cho
 */

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace treelite {

class Model;

}  // namespace treelite

namespace treelite::model_loader::detail::xgboost {

using ParsedCategoryValuesArray
    = std::variant<std::monostate, std::vector<std::int8_t>, std::vector<std::int64_t>>;

struct ParsedCategoryInfo {
  std::optional<std::int64_t> type{};
  std::optional<std::vector<std::int32_t>> offsets{};
  ParsedCategoryValuesArray values{std::monostate{}};
};

struct ParsedCategoryContainer {
  std::vector<ParsedCategoryInfo> enc;
  std::vector<std::int32_t> feature_segments;
  std::vector<std::int32_t> sorted_idx;
};

void SetCategoricalEncoding(Model* model, ParsedCategoryContainer const& cat_container);

}  // namespace treelite::model_loader::detail::xgboost
