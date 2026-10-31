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

  model->int_cat_feature_segments = std::vector<std::int32_t>{0};
  model->str_cat_feature_segments = std::vector<std::int32_t>{0};
  model->str_cat_offsets = std::vector<std::int32_t>{0};
  std::size_t fidx = 0;
  for (ParsedCategoryInfo const& enc_item : cat_container.enc) {
    if (enc_item.type.has_value()) {
      // Integer categories
      auto const& int_cat_values = std::get<std::vector<std::int64_t>>(enc_item.values);
      TREELITE_CHECK(!int_cat_values.empty());
      model->int_cat_values.Extend(int_cat_values);
      model->int_cat_feature_segments.PushBack(
          static_cast<std::int32_t>(model->int_cat_values.Size()));
      model->int_sorted_idx.Extend(std::vector<std::int32_t>(
          cat_container.sorted_idx.begin() + cat_container.feature_segments[fidx],
          cat_container.sorted_idx.begin() + cat_container.feature_segments[fidx + 1]));
      TREELITE_CHECK_EQ(model->int_sorted_idx.Size(), model->int_cat_feature_segments.Back());

      model->str_cat_feature_segments.PushBack(model->str_cat_feature_segments.Back());
    } else if (enc_item.offsets.has_value() && !enc_item.offsets->empty()) {
      // String categories
      auto const& str_cat_values = std::get<std::vector<std::int8_t>>(enc_item.values);
      model->str_cat_values.Extend(str_cat_values);
      auto const& str_cat_offsets = enc_item.offsets.value();
      TREELITE_CHECK_GT(str_cat_offsets.size(), 1UL);
      auto last_offset = model->str_cat_offsets.Back();
      for (std::size_t i = 1; i < str_cat_offsets.size(); ++i) {
        model->str_cat_offsets.PushBack(last_offset + str_cat_offsets[i]);
      }
      model->str_cat_feature_segments.PushBack(
          static_cast<std::int32_t>(model->str_cat_offsets.Size() - 1));
      model->str_sorted_idx.Extend(std::vector<std::int32_t>(
          cat_container.sorted_idx.begin() + cat_container.feature_segments[fidx],
          cat_container.sorted_idx.begin() + cat_container.feature_segments[fidx + 1]));
      TREELITE_CHECK_EQ(model->str_sorted_idx.Size(), model->str_cat_feature_segments.Back());

      model->int_cat_feature_segments.PushBack(model->int_cat_feature_segments.Back());
    } else {
      model->int_cat_feature_segments.PushBack(model->int_cat_feature_segments.Back());
      model->str_cat_feature_segments.PushBack(model->str_cat_feature_segments.Back());
    }
    fidx++;
  }
  TREELITE_CHECK_EQ(
      model->int_cat_feature_segments.Size(), static_cast<std::size_t>(model->num_feature + 1));
  TREELITE_CHECK_EQ(
      model->str_cat_feature_segments.Size(), static_cast<std::size_t>(model->num_feature + 1));
}

}  // namespace treelite::model_loader::detail::xgboost
