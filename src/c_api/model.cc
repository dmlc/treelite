/*!
 * Copyright (c) 2023 by Contributors
 * \file model.cc
 * \author Hyunsu Cho
 * \brief C API for functions to query and modify model objects
 */

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <variant>

#include <treelite/c_api.h>
#include <treelite/c_api_error.h>
#include <treelite/tree.h>

#include "./c_api_utils.h"

int TreeliteDumpAsJSON(TreeliteModelHandle handle, int pretty_print, char const** out_json_str) {
  API_BEGIN();
  auto* model_ = static_cast<treelite::Model*>(handle);
  std::string& ret_str = treelite::c_api::ReturnValueStore::Get()->ret_str;
  ret_str = model_->DumpAsJSON(pretty_print != 0);
  *out_json_str = ret_str.c_str();
  API_END();
}

int TreeliteGetInputType(TreeliteModelHandle model, char const** out_str) {
  API_BEGIN();
  auto const* model_ = static_cast<treelite::Model const*>(model);
  auto& type_str = treelite::c_api::ReturnValueStore::Get()->ret_str;
  type_str = treelite::TypeInfoToString(model_->GetThresholdType());
  *out_str = type_str.c_str();
  API_END();
}

int TreeliteGetOutputType(TreeliteModelHandle model, char const** out_str) {
  API_BEGIN();
  auto const* model_ = static_cast<treelite::Model const*>(model);
  auto& type_str = treelite::c_api::ReturnValueStore::Get()->ret_str;
  type_str = treelite::TypeInfoToString(model_->GetLeafOutputType());
  *out_str = type_str.c_str();
  API_END();
}

int TreeliteQueryNumTree(TreeliteModelHandle model, std::size_t* out) {
  API_BEGIN();
  auto const* model_ = static_cast<treelite::Model const*>(model);
  *out = model_->GetNumTree();
  API_END();
}

int TreeliteQueryNumFeature(TreeliteModelHandle model, int* out) {
  API_BEGIN();
  auto const* model_ = static_cast<treelite::Model const*>(model);
  *out = model_->num_feature;
  API_END();
}

int TreeliteConcatenateModelObjects(
    TreeliteModelHandle const* objs, std::size_t len, TreeliteModelHandle* out) {
  API_BEGIN();
  std::vector<treelite::Model const*> model_objs(len, nullptr);
  std::transform(objs, objs + len, model_objs.begin(),
      [](TreeliteModelHandle e) { return static_cast<treelite::Model const*>(e); });
  auto concatenated_model = ConcatenateModelObjects(model_objs);
  *out = static_cast<TreeliteModelHandle>(concatenated_model.release());
  API_END();
}

int TreeliteFreeModel(TreeliteModelHandle handle) {
  API_BEGIN();
  delete static_cast<treelite::Model*>(handle);
  API_END();
}

int TreeliteGetTreeDepth(TreeliteModelHandle model, std::uint32_t** out, std::size_t* out_len) {
  API_BEGIN();
  auto* model_ = static_cast<treelite::Model*>(model);
  auto& ret_depth = treelite::c_api::ReturnValueStore::Get()->ret_uint32_vec;
  ret_depth = model_->GetTreeDepth();
  *out = ret_depth.data();
  *out_len = ret_depth.size();
  API_END();
}

int TreeliteGetCategoryKind(TreeliteModelHandle model, std::int32_t feature_id, std::int8_t* out) {
  API_BEGIN();
  auto const* model_ = static_cast<treelite::Model const*>(model);
  *out = static_cast<std::int8_t>(model_->GetCategoricalKind(feature_id));
  API_END();
}

int TreeliteGetCategoryValuesInteger(TreeliteModelHandle model, std::int32_t feature_id,
    std::size_t* out_len, std::int64_t** out_arr) {
  API_BEGIN();
  auto const* model_ = static_cast<treelite::Model const*>(model);
  TREELITE_CHECK(model_->GetCategoricalKind(feature_id) == treelite::CategoricalKind::kInteger)
      << "Feature " << feature_id << " is not a categorical feature with integer values";
  auto& cat_values = treelite::c_api::ReturnValueStore::Get()->ret_int64_vec;
  cat_values = std::get<std::vector<std::int64_t>>(model_->GetCategoryValues(feature_id));
  *out_len = cat_values.size();
  *out_arr = cat_values.data();
  API_END();
}

int TreeliteGetCategoryValuesString(TreeliteModelHandle model, std::int32_t feature_id,
    std::size_t* out_len, char const*** out_arr) {
  API_BEGIN();
  auto const* model_ = static_cast<treelite::Model const*>(model);
  TREELITE_CHECK(model_->GetCategoricalKind(feature_id) == treelite::CategoricalKind::kString)
      << "Feature " << feature_id << " is not a categorical feature with string values";

  auto& cat_values = treelite::c_api::ReturnValueStore::Get()->ret_vec_str;
  cat_values = std::get<std::vector<std::string>>(model_->GetCategoryValues(feature_id));

  auto& vec_charp = treelite::c_api::ReturnValueStore::Get()->ret_vec_charp;
  vec_charp.resize(cat_values.size());
  std::transform(cat_values.begin(), cat_values.end(), vec_charp.begin(),
      [](std::string const& e) { return e.c_str(); });
  *out_len = cat_values.size();
  *out_arr = vec_charp.data();
  API_END();
}

int TreeliteGetCategorySortedIdx(
    TreeliteModelHandle model, int32_t feature_id, size_t* out_len, int32_t** out_arr) {
  API_BEGIN();
  auto const* model_ = static_cast<treelite::Model const*>(model);
  auto& sorted_idx = treelite::c_api::ReturnValueStore::Get()->ret_int32_vec;

  std::optional<std::vector<std::int32_t>> sorted_idx_vals
      = model_->GetCategorySortedIdx(feature_id);
  TREELITE_CHECK(model_->GetCategoricalKind(feature_id) != treelite::CategoricalKind::kNone
                 && sorted_idx_vals.has_value())
      << "Feature " << feature_id << " is not a categorical feature";
  sorted_idx = sorted_idx_vals.value();
  *out_len = sorted_idx.size();
  *out_arr = sorted_idx.data();
  API_END();
}
