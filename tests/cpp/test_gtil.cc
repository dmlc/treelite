/*!
 * Copyright (c) 2023 by Contributors
 * \file test_gtil.cc
 * \author Hyunsu Cho
 * \brief C++ tests for GTIL
 */

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/gtil.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <experimental/mdspan>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace treelite {

class ParametrizedTestSuite : public testing::TestWithParam<std::string> {};

TEST_P(ParametrizedTestSuite, MulticlassClfGrovePerClass) {
  std::unique_ptr<model_builder::ModelBuilder> builder = model_builder::GetModelBuilder(R"(
    {
      "threshold_type": "float32",
      "leaf_output_type": "float32",
      "metadata": {
        "num_feature": 1,
        "task_type": "kMultiClf",
        "average_tree_output": false,
        "num_target": 1,
        "num_class": [3],
        "leaf_vector_shape": [1, 1]
      },
      "tree_annotation": {
        "num_tree": 6,
        "target_id": [0, 0, 0, 0, 0, 0],
        "class_id": [0, 1, 2, 0, 1, 2]
      },
      "postprocessor": {
        "name": "softmax"
      },
      "base_scores": [0.3, 0.2, 0.5]
    }
  )");
  auto make_tree_stump = [&](float left_child_val, float right_child_val) {
    builder->StartTree();
    builder->StartNode(0);
    builder->NumericalTest(0, 0.0, false, Operator::kLT, 1, 2);
    builder->EndNode();
    builder->StartNode(1);
    builder->LeafScalar(left_child_val);
    builder->EndNode();
    builder->StartNode(2);
    builder->LeafScalar(right_child_val);
    builder->EndNode();
    builder->EndTree();
  };
  make_tree_stump(-1.0f, 1.0f);
  make_tree_stump(1.0f, -1.0f);
  make_tree_stump(0.5f, 0.5f);
  make_tree_stump(-1.0f, 0.0f);
  make_tree_stump(0.0f, -1.0f);
  make_tree_stump(0.5f, 1.5f);

  auto const predict_kind = GetParam();

  std::unique_ptr<Model> model = builder->CommitModel();
  gtil::Configuration config(fmt::format(R"({{
     "predict_type": "{}",
     "nthread": 1
  }})",
      predict_kind));

  std::vector<std::uint64_t> expected_output_shape;
  std::vector<std::vector<float>> expected_output;
  if (predict_kind == "raw") {
    expected_output_shape = {1, 1, 3};
    expected_output = {{1.3f, -1.8f, 2.5f}, {-1.7f, 1.2f, 1.5f}};
  } else if (predict_kind == "default") {
    expected_output_shape = {1, 1, 3};
    auto softmax = [](float a, float b, float c) {
      float const max = std::max({a, b, c});
      a -= max;
      b -= max;
      c -= max;
      float const sum = std::exp(a) + std::exp(b) + std::exp(c);
      return std::vector<float>{std::exp(a) / sum, std::exp(b) / sum, std::exp(c) / sum};
    };
    expected_output = {softmax(1.3f, -1.8f, 2.5f), softmax(-1.7f, 1.2f, 1.5f)};
  } else if (predict_kind == "leaf_id") {
    expected_output_shape = {1, 6};
    expected_output = {{2, 2, 2, 2, 2, 2}, {1, 1, 1, 1, 1, 1}};
  }
  auto output_shape = gtil::GetOutputShape(*model, 1, config);
  EXPECT_EQ(output_shape, expected_output_shape);

  std::vector<float> output(std::accumulate(
      output_shape.begin(), output_shape.end(), std::uint64_t(1), std::multiplies<>()));
  {
    std::vector<float> input{1.0f};
    gtil::Predict(*model, input.data(), 1, output.data(), config);
    EXPECT_EQ(output, expected_output[0]);
  }
  {
    std::vector<float> input{-1.0f};
    gtil::Predict(*model, input.data(), 1, output.data(), config);
    EXPECT_EQ(output, expected_output[1]);
  }
}

TEST_P(ParametrizedTestSuite, LeafVectorRF) {
  model_builder::Metadata metadata{1, TaskType::kMultiClf, true, 1, {3}, {1, 3}};
  model_builder::TreeAnnotation tree_annotation{2, {0, 0}, {-1, -1}};
  model_builder::PostProcessorFunc postprocessor{"identity_multiclass"};
  std::vector<double> base_scores{100.0, 200.0, 300.0};
  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::GetModelBuilder(TypeInfo::kFloat64, TypeInfo::kFloat64, metadata,
          tree_annotation, postprocessor, base_scores);
  auto make_tree_stump
      = [&](std::vector<double> const& left_child_val, std::vector<double> const& right_child_val) {
          builder->StartTree();
          builder->StartNode(0);
          builder->NumericalTest(0, 0.0, false, Operator::kLT, 1, 2);
          builder->EndNode();
          builder->StartNode(1);
          builder->LeafVector(left_child_val);
          builder->EndNode();
          builder->StartNode(2);
          builder->LeafVector(right_child_val);
          builder->EndNode();
          builder->EndTree();
        };
  make_tree_stump({1.0, 0.0, 0.0}, {0.0, 0.5, 0.5});
  make_tree_stump({1.0, 0.0, 0.0}, {0.0, 0.5, 0.5});

  auto const predict_kind = GetParam();

  std::unique_ptr<Model> model = builder->CommitModel();
  gtil::Configuration config(fmt::format(R"({{
     "predict_type": "{}",
     "nthread": 1
  }})",
      predict_kind));

  std::vector<std::uint64_t> expected_output_shape;
  std::vector<float> expected_output_left_child;
  std::vector<double> expected_output_right_child;
  if (predict_kind == "raw" || predict_kind == "default") {
    expected_output_shape = {1, 1, 3};
    expected_output_left_child = {100.0f, 200.5f, 300.5f};
    expected_output_right_child = {101.0, 200.0, 300.0};
  } else if (predict_kind == "leaf_id") {
    expected_output_shape = {1, 2};
    expected_output_left_child = {2, 2};
    expected_output_right_child = {1, 1};
  }
  auto output_shape = gtil::GetOutputShape(*model, 1, config);
  EXPECT_EQ(output_shape, expected_output_shape);

  auto output_size = std::accumulate(
      output_shape.begin(), output_shape.end(), std::uint64_t(1), std::multiplies<>());
  {
    std::vector<float> input{1.0f};
    std::vector<float> output(output_size);
    gtil::Predict(*model, input.data(), 1, output.data(), config);
    EXPECT_EQ(output, expected_output_left_child);
  }
  {
    std::vector<double> input{-1.0};
    std::vector<double> output(output_size);
    gtil::Predict(*model, input.data(), 1, output.data(), config);
    EXPECT_EQ(output, expected_output_right_child);
  }
}

INSTANTIATE_TEST_SUITE_P(GTIL, ParametrizedTestSuite, testing::Values("raw", "default", "leaf_id"));

TEST(GTIL, InvalidCategoricalInput) {
  model_builder::Metadata metadata{2, TaskType::kRegressor, false, 1, {1}, {1, 1}};
  model_builder::TreeAnnotation tree_annotation{1, {0}, {0}};
  model_builder::PostProcessorFunc postprocessor{"identity"};
  std::vector<double> base_scores{0.0};
  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::GetModelBuilder(TypeInfo::kFloat64, TypeInfo::kFloat64, metadata,
          tree_annotation, postprocessor, base_scores);

  builder->StartTree();
  builder->StartNode(0);
  builder->CategoricalTest(1, true, {0}, false, 1, 2);
  builder->EndNode();
  builder->StartNode(1);
  builder->LeafScalar(-1.0);
  builder->EndNode();
  builder->StartNode(2);
  builder->LeafScalar(1.0);
  builder->EndNode();
  builder->EndTree();
  auto model = builder->CommitModel();

  std::vector<double> categorical_column{-1.0, -0.6, -0.5, 0.0, 0.3, 0.7, 1.0,
      std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), 1e10,
      -1e10};
  std::size_t const n_rows = categorical_column.size();
  std::vector<double> elems(n_rows * 2);
  namespace stdex = std::experimental;
  using Array2DView = stdex::mdspan<double, stdex::dextents<std::size_t, 2>, stdex::layout_right>;
  auto dmat = Array2DView(elems.data(), n_rows, 2);
  for (std::size_t i = 0; i < n_rows; ++i) {
    dmat(i, 0) = 0.0;
    dmat(i, 1) = categorical_column[i];
  }

  gtil::Configuration config;
  config.nthread = 0;
  config.pred_kind = gtil::PredictKind::kPredictLeafID;

  auto output_shape = gtil::GetOutputShape(*model, n_rows, config);
  std::vector<double> output(std::accumulate(
      output_shape.begin(), output_shape.end(), std::uint64_t(1), std::multiplies<>()));
  gtil::Predict(*model, elems.data(), n_rows, output.data(), config);

  // Negative inputs are mapped to the right child node
  // 0.3 and 0.7 are mapped to the left child node, since they get rounded toward the zero.
  // Missing value gets mapped to the left child node, since default_left=True
  // inf, 1e10, and -1e10 don't match any element of left_categories, so they get mapped to the
  // right child.
  std::vector<double> expected_output{2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2};
  EXPECT_EQ(output, expected_output);
}

}  // namespace treelite
