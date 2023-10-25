/*!
 * Copyright (c) 2023 by Contributors
 * \file sklearn.cc
 * \author Hyunsu Cho
 * \brief C API for scikit-learn loader functions
 */

#include <treelite/c_api.h>
#include <treelite/c_api_error.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

#include <cstdint>

int TreeliteLoadSKLearnRandomForestRegressor(int n_estimators, int n_features, int n_targets,
    std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity, TreeliteModelHandle* out) {
  API_BEGIN();
  auto model = treelite::model_loader::sklearn::LoadRandomForestRegressor(n_estimators, n_features,
      n_targets, node_count, children_left, children_right, feature, threshold, value,
      n_node_samples, weighted_n_node_samples, impurity);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnIsolationForest(int n_estimators, int n_features,
    std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity, double ratio_c,
    TreeliteModelHandle* out) {
  API_BEGIN();
  auto model = treelite::model_loader::sklearn::LoadIsolationForest(n_estimators, n_features,
      node_count, children_left, children_right, feature, threshold, value, n_node_samples,
      weighted_n_node_samples, impurity, ratio_c);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnRandomForestClassifier(int n_estimators, int n_features, int n_targets,
    std::int32_t const* n_classes, std::int64_t const* node_count,
    std::int64_t const** children_left, std::int64_t const** children_right,
    std::int64_t const** feature, double const** threshold, double const** value,
    std::int64_t const** n_node_samples, double const** weighted_n_node_samples,
    double const** impurity, TreeliteModelHandle* out) {
  API_BEGIN();
  auto model = treelite::model_loader::sklearn::LoadRandomForestClassifier(n_estimators, n_features,
      n_targets, n_classes, node_count, children_left, children_right, feature, threshold, value,
      n_node_samples, weighted_n_node_samples, impurity);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnGradientBoostingRegressor(int n_iter, int n_features,
    std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity, double const* base_scores,
    TreeliteModelHandle* out) {
  API_BEGIN();
  auto model = treelite::model_loader::sklearn::LoadGradientBoostingRegressor(n_iter, n_features,
      node_count, children_left, children_right, feature, threshold, value, n_node_samples,
      weighted_n_node_samples, impurity, base_scores);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnGradientBoostingClassifier(int n_iter, int n_features, int n_classes,
    std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity, double const* base_scores,
    TreeliteModelHandle* out) {
  API_BEGIN();
  auto model = treelite::model_loader::sklearn::LoadGradientBoostingClassifier(n_iter, n_features,
      n_classes, node_count, children_left, children_right, feature, threshold, value,
      n_node_samples, weighted_n_node_samples, impurity, base_scores);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnHistGradientBoostingRegressor(int n_iter, int n_features,
    std::int64_t const* node_count, void const** nodes, int expected_sizeof_node_struct,
    std::uint32_t n_categorical_splits, std::uint32_t const** raw_left_cat_bitsets,
    std::uint32_t const* known_cat_bitsets, std::uint32_t const* known_cat_bitsets_offset_map,
    double const* base_scores, TreeliteModelHandle* out) {
  API_BEGIN();
  auto model = treelite::model_loader::sklearn::LoadHistGradientBoostingRegressor(n_iter,
      n_features, node_count, nodes, expected_sizeof_node_struct, n_categorical_splits,
      raw_left_cat_bitsets, known_cat_bitsets, known_cat_bitsets_offset_map, base_scores);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadSKLearnHistGradientBoostingClassifier(int n_iter, int n_features, int n_classes,
    std::int64_t const* node_count, void const** nodes, int expected_sizeof_node_struct,
    std::uint32_t n_categorical_splits, std::uint32_t const** raw_left_cat_bitsets,
    std::uint32_t const* known_cat_bitsets, std::uint32_t const* known_cat_bitsets_offset_map,
    double const* base_scores, TreeliteModelHandle* out) {
  API_BEGIN();
  auto model = treelite::model_loader::sklearn::LoadHistGradientBoostingClassifier(n_iter,
      n_features, n_classes, node_count, nodes, expected_sizeof_node_struct, n_categorical_splits,
      raw_left_cat_bitsets, known_cat_bitsets, known_cat_bitsets_offset_map, base_scores);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}
