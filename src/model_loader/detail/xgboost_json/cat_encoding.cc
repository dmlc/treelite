/*!
 * Copyright (c) 2026 by Contributors
 * \file cat_encoding.cc
 * \brief Utility function to parse categorical encoding information from XGBoost
 * \author Hyunsu Cho
 */

#include "./cat_encoding.h"

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include <treelite/tree.h>

namespace treelite::model_loader::detail::xgboost {

void SetCategoricalEncoding(Model* model, ParsedCategoryContainer const& cat_container) {
  if (cat_container.enc.empty() && cat_container.feature_segments.empty()
      && cat_container.sorted_idx.empty()) {
    return;
  }
  TREELITE_CHECK_EQ(cat_container.enc.size(), static_cast<std::size_t>(model->num_feature));
  TREELITE_CHECK_EQ(
      cat_container.feature_segments.size(), static_cast<std::size_t>(model->num_feature + 1));
  TREELITE_CHECK_EQ(cat_container.sorted_idx.size(),
      static_cast<std::size_t>(cat_container.feature_segments.back()));

  model->int_cat_feature_segments_ = std::vector<std::int32_t>{0};
  model->str_cat_feature_segments_ = std::vector<std::int32_t>{0};
  model->str_cat_offsets_ = std::vector<std::int32_t>{0};
  std::size_t fidx = 0;
  for (ParsedCategoryInfo const& enc_item : cat_container.enc) {
    if (enc_item.type.has_value()) {
      // Integer categories
      TREELITE_CHECK(enc_item.int_values.has_value());
      auto const& int_cat_values = enc_item.int_values.value();
      TREELITE_CHECK(!int_cat_values.empty());
      model->cat_kind_.PushBack(CategoricalKind::kInteger);
      model->int_cat_values_.Extend(int_cat_values);
      model->int_cat_feature_segments_.PushBack(
          static_cast<std::int32_t>(model->int_cat_values_.Size()));
      model->int_sorted_idx_.Extend(std::vector<std::int32_t>(
          cat_container.sorted_idx.begin() + cat_container.feature_segments[fidx],
          cat_container.sorted_idx.begin() + cat_container.feature_segments[fidx + 1]));
      TREELITE_CHECK_EQ(model->int_sorted_idx_.Size(), model->int_cat_feature_segments_.Back());

      model->str_cat_feature_segments_.PushBack(model->str_cat_feature_segments_.Back());
    } else if (enc_item.offsets.has_value() && !enc_item.offsets->empty()) {
      // String categories
      model->cat_kind_.PushBack(CategoricalKind::kString);
      if (std::holds_alternative<std::vector<std::int8_t>>(enc_item.str_values)) {
        // XGBoost 3.1-3.4 uses int8_t to store string characters
        auto const& str_cat_values = std::get<std::vector<std::int8_t>>(enc_item.str_values);
        std::vector<std::uint8_t> unsigned_str(str_cat_values.size());
        std::transform(str_cat_values.begin(), str_cat_values.end(), unsigned_str.begin(),
            [](std::int8_t e) { return static_cast<std::uint8_t>(e); });
        model->str_cat_values_.Extend(unsigned_str);
      } else if (std::holds_alternative<std::vector<std::uint8_t>>(enc_item.str_values)) {
        // XGBoost 3.5+ uses uint8_t to store string characters
        auto const& str_cat_values = std::get<std::vector<std::uint8_t>>(enc_item.str_values);
        model->str_cat_values_.Extend(str_cat_values);
      } else {
        TREELITE_LOG(FATAL)
            << "The `offsets` field is given but no valid string categories are provided";
      }
      auto const& str_cat_offsets = enc_item.offsets.value();
      TREELITE_CHECK_GT(str_cat_offsets.size(), 1UL);
      auto last_offset = model->str_cat_offsets_.Back();
      for (std::size_t i = 1; i < str_cat_offsets.size(); ++i) {
        model->str_cat_offsets_.PushBack(last_offset + str_cat_offsets[i]);
      }
      model->str_cat_feature_segments_.PushBack(
          static_cast<std::int32_t>(model->str_cat_offsets_.Size() - 1));
      model->str_sorted_idx_.Extend(std::vector<std::int32_t>(
          cat_container.sorted_idx.begin() + cat_container.feature_segments[fidx],
          cat_container.sorted_idx.begin() + cat_container.feature_segments[fidx + 1]));
      TREELITE_CHECK_EQ(model->str_sorted_idx_.Size(), model->str_cat_feature_segments_.Back());

      model->int_cat_feature_segments_.PushBack(model->int_cat_feature_segments_.Back());
    } else {
      model->cat_kind_.PushBack(CategoricalKind::kNone);
      model->int_cat_feature_segments_.PushBack(model->int_cat_feature_segments_.Back());
      model->str_cat_feature_segments_.PushBack(model->str_cat_feature_segments_.Back());
    }
    fidx++;
  }
  TREELITE_CHECK_EQ(
      model->int_cat_feature_segments_.Size(), static_cast<std::size_t>(model->num_feature + 1));
  TREELITE_CHECK_EQ(
      model->str_cat_feature_segments_.Size(), static_cast<std::size_t>(model->num_feature + 1));
}

}  // namespace treelite::model_loader::detail::xgboost
