/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file test_model_concat.cc
 * \author Hyunsu Cho
 * \brief C++ tests for model concatenation
 */
#include <gtest/gtest.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <variant>
#include <vector>

namespace {

inline void TestRoundTrip(treelite::Model* model) {
  // Test round trip with in-memory serialization
  std::ostringstream oss;
  oss.exceptions(std::ios::failbit | std::ios::badbit);
  model->SerializeToStream(oss);

  std::istringstream iss(oss.str());
  iss.exceptions(std::ios::failbit | std::ios::badbit);
  std::unique_ptr<treelite::Model> received_model = treelite::Model::DeserializeFromStream(iss);

  // Use ASSERT_TRUE, since ASSERT_EQ will dump all the raw bytes into a string, potentially
  // causing an OOM error
  ASSERT_TRUE(model->DumpAsJSON(false) == received_model->DumpAsJSON(false));
}

}  // anonymous namespace

namespace treelite {

TEST(ModelConcatenation, TreeStump) {
  std::vector<std::unique_ptr<Model>> model_objs;
  constexpr int kNumModelObjs = 5;

  for (int i = 0; i < kNumModelObjs; ++i) {
    auto builder = model_builder::GetModelBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32,
        model_builder::Metadata{2, TaskType::kRegressor, false, 1, {1}, {1, 1}},
        model_builder::TreeAnnotation{1, {0}, {0}}, model_builder::PostProcessorFunc{"identity"},
        {0.0});
    builder->StartTree();
    builder->StartNode(0);
    builder->NumericalTest(0, 0.0, true, Operator::kLT, 1, 2);
    builder->EndNode();
    builder->StartNode(1);
    builder->LeafScalar(1.0);
    builder->EndNode();
    builder->StartNode(2);
    builder->LeafScalar(2.0);
    builder->EndNode();
    builder->EndTree();
    model_objs.push_back(builder->CommitModel());
  }

  std::vector<Model const*> model_obj_refs;
  std::transform(model_objs.begin(), model_objs.end(), std::back_inserter(model_obj_refs),
      [](auto const& obj) { return obj.get(); });

  std::unique_ptr<Model> concatenated_model = ConcatenateModelObjects(model_obj_refs);
  ASSERT_EQ(concatenated_model->GetNumTree(), kNumModelObjs);
  EXPECT_EQ(concatenated_model->GetThresholdType(), TypeInfo::kFloat32);
  EXPECT_EQ(concatenated_model->GetLeafOutputType(), TypeInfo::kFloat32);
  EXPECT_TRUE(concatenated_model->target_id
              == ContiguousArray<std::int32_t>(std::vector<std::int32_t>(kNumModelObjs, 0)));
  EXPECT_TRUE(concatenated_model->class_id
              == ContiguousArray<std::int32_t>(std::vector<std::int32_t>(kNumModelObjs, 0)));
  TestRoundTrip(concatenated_model.get());
  auto& trees = std::get<ModelPreset<float, float>>(concatenated_model->variant_).trees;
  for (int i = 0; i < kNumModelObjs; ++i) {
    auto const& tree = trees[i];
    EXPECT_FALSE(tree.IsLeaf(0));
    EXPECT_TRUE(tree.IsLeaf(1));
    EXPECT_TRUE(tree.IsLeaf(2));
    EXPECT_EQ(tree.NodeType(0), TreeNodeType::kNumericalTestNode);
    EXPECT_EQ(tree.SplitIndex(0), 0);
    EXPECT_EQ(tree.Threshold(0), 0.0f);
    EXPECT_EQ(tree.LeftChild(0), 1);
    EXPECT_EQ(tree.RightChild(0), 2);
    EXPECT_EQ(tree.LeafValue(1), 1.0f);
    EXPECT_EQ(tree.LeafValue(2), 2.0f);
  }
}

TEST(ModelConcatenation, MismatchedTreeType) {
  std::vector<std::unique_ptr<Model>> model_objs;

  auto builder = model_builder::GetModelBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32,
      model_builder::Metadata{2, TaskType::kRegressor, false, 1, {1}, {1, 1}},
      model_builder::TreeAnnotation{1, {0}, {0}}, model_builder::PostProcessorFunc{"identity"},
      {0.0});
  builder->StartTree();
  builder->StartNode(0);
  builder->LeafScalar(1.0);
  builder->EndNode();
  builder->EndTree();
  model_objs.push_back(builder->CommitModel());

  auto builder2 = model_builder::GetModelBuilder(TypeInfo::kFloat64, TypeInfo::kFloat64,
      model_builder::Metadata{2, TaskType::kRegressor, false, 1, {1}, {1, 1}},
      model_builder::TreeAnnotation{1, {0}, {0}}, model_builder::PostProcessorFunc{"identity"},
      {0.0});
  builder2->StartTree();
  builder2->StartNode(0);
  builder2->LeafScalar(1.0);
  builder2->EndNode();
  builder2->EndTree();
  model_objs.push_back(builder2->CommitModel());

  std::vector<Model const*> model_obj_refs;
  std::transform(model_objs.begin(), model_objs.end(), std::back_inserter(model_obj_refs),
      [](auto const& obj) { return obj.get(); });
  ASSERT_THROW(ConcatenateModelObjects(model_obj_refs), treelite::Error);
}

}  // namespace treelite
