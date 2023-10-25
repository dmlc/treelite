/*!
 * Copyright (c) 2023 by Contributors
 * \file model_builder.cc
 * \author Hyunsu Cho
 * \brief C API for frontend functions
 */

#include <treelite/c_api.h>
#include <treelite/c_api_error.h>
#include <treelite/enum/operator.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <cstddef>
#include <cstdint>

int TreeliteGetModelBuilder(char const* json_str, TreeliteModelBuilderHandle* out) {
  API_BEGIN();
  auto builder = treelite::model_builder::GetModelBuilder(json_str);
  *out = static_cast<TreeliteModelBuilderHandle>(builder.release());
  API_END();
}

int TreeliteDeleteModelBuilder(TreeliteModelBuilderHandle model_builder) {
  API_BEGIN();
  delete static_cast<treelite::model_builder::ModelBuilder*>(model_builder);
  API_END();
}

int TreeliteModelBuilderStartTree(TreeliteModelBuilderHandle model_builder) {
  API_BEGIN();
  auto* builder_ = static_cast<treelite::model_builder::ModelBuilder*>(model_builder);
  builder_->StartTree();
  API_END();
}

int TreeliteModelBuilderEndTree(TreeliteModelBuilderHandle model_builder) {
  API_BEGIN();
  auto* builder_ = static_cast<treelite::model_builder::ModelBuilder*>(model_builder);
  builder_->EndTree();
  API_END();
}

int TreeliteModelBuilderStartNode(TreeliteModelBuilderHandle model_builder, int node_key) {
  API_BEGIN();
  auto* builder_ = static_cast<treelite::model_builder::ModelBuilder*>(model_builder);
  builder_->StartNode(node_key);
  API_END();
}

int TreeliteModelBuilderEndNode(TreeliteModelBuilderHandle model_builder) {
  API_BEGIN();
  auto* builder_ = static_cast<treelite::model_builder::ModelBuilder*>(model_builder);
  builder_->EndNode();
  API_END();
}

int TreeliteModelBuilderNumericalTest(TreeliteModelBuilderHandle model_builder,
    std::int32_t split_index, double threshold, int default_left, char const* cmp,
    int left_child_key, int right_child_key) {
  API_BEGIN();
  auto* builder_ = static_cast<treelite::model_builder::ModelBuilder*>(model_builder);
  treelite::Operator cmp_ = treelite::OperatorFromString(cmp);
  builder_->NumericalTest(
      split_index, threshold, default_left, cmp_, left_child_key, right_child_key);
  API_END();
}

int TreeliteModelBuilderCategoricalTest(TreeliteModelBuilderHandle model_builder,
    std::int32_t split_index, int default_left, std::uint32_t const* category_list,
    std::size_t category_list_len, int category_list_right_child, int left_child_key,
    int right_child_key) {
  API_BEGIN();
  auto* builder_ = static_cast<treelite::model_builder::ModelBuilder*>(model_builder);
  std::vector<std::uint32_t> category_list_(category_list, category_list + category_list_len);
  builder_->CategoricalTest(split_index, default_left, category_list_,
      category_list_right_child == 1, left_child_key, right_child_key);
  API_END();
}

int TreeliteModelBuilderLeafScalar(TreeliteModelBuilderHandle model_builder, double leaf_value) {
  API_BEGIN();
  auto* builder_ = static_cast<treelite::model_builder::ModelBuilder*>(model_builder);
  builder_->LeafScalar(leaf_value);
  API_END();
}

int TreeliteModelBuilderLeafVectorFloat32(TreeliteModelBuilderHandle model_builder,
    float const* leaf_vector, std::size_t leaf_vector_len) {
  API_BEGIN();
  auto* builder_ = static_cast<treelite::model_builder::ModelBuilder*>(model_builder);
  builder_->LeafVector(std::vector<float>(leaf_vector, leaf_vector + leaf_vector_len));
  API_END();
}

int TreeliteModelBuilderLeafVectorFloat64(TreeliteModelBuilderHandle model_builder,
    double const* leaf_vector, std::size_t leaf_vector_len) {
  API_BEGIN();
  auto* builder_ = static_cast<treelite::model_builder::ModelBuilder*>(model_builder);
  builder_->LeafVector(std::vector<double>(leaf_vector, leaf_vector + leaf_vector_len));
  API_END();
}

int TreeliteModelBuilderGain(TreeliteModelBuilderHandle model_builder, double gain) {
  API_BEGIN();
  auto* builder_ = static_cast<treelite::model_builder::ModelBuilder*>(model_builder);
  builder_->Gain(gain);
  API_END();
}

int TreeliteModelBuilderDataCount(
    TreeliteModelBuilderHandle model_builder, std::uint64_t data_count) {
  API_BEGIN();
  auto* builder_ = static_cast<treelite::model_builder::ModelBuilder*>(model_builder);
  builder_->DataCount(data_count);
  API_END();
}

int TreeliteModelBuilderSumHess(TreeliteModelBuilderHandle model_builder, double sum_hess) {
  API_BEGIN();
  auto* builder_ = static_cast<treelite::model_builder::ModelBuilder*>(model_builder);
  builder_->SumHess(sum_hess);
  API_END();
}

int TreeliteModelBuilderCommitModel(
    TreeliteModelBuilderHandle model_builder, TreeliteModelHandle* out) {
  API_BEGIN();
  auto* builder_ = static_cast<treelite::model_builder::ModelBuilder*>(model_builder);
  auto model = builder_->CommitModel();
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}
