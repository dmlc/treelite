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

// XGBoost 3.1-3.4 stores string characters as signed 8-bit integers
// XGBoost 3.5+ stores string characters as unsigned 8-bit integers
using ParsedStringCategoryValues
    = std::variant<std::monostate, std::vector<std::uint8_t>, std::vector<std::int8_t>>;
using ParsedIntegerCategoryValues = std::optional<std::vector<std::int64_t>>;

struct ParsedCategoryInfo {
  std::optional<std::int64_t> type{};
  std::optional<std::vector<std::int32_t>> offsets{};
  ParsedStringCategoryValues str_values{std::monostate{}};
  ParsedIntegerCategoryValues int_values{std::nullopt};
};

struct ParsedCategoryContainer {
  std::vector<ParsedCategoryInfo> enc;
  std::vector<std::int32_t> feature_segments;
  std::vector<std::int32_t> sorted_idx;
};

void SetCategoricalEncoding(Model* model, ParsedCategoryContainer const& cat_container);

}  // namespace treelite::model_loader::detail::xgboost
