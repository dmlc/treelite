/*!
 * Copyright (c) 2021 by Contributors
 * \file test_model_concat.cc
 * \author Hyunsu Cho
 * \brief C++ tests for model concatenation
 */
#include <gtest/gtest.h>
#include <treelite/frontend.h>
#include <treelite/tree.h>

#include <algorithm>
#include <memory>
#include <sstream>
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
    std::unique_ptr<frontend::ModelBuilder> builder{
        new frontend::ModelBuilder(2, 1, false, TypeInfo::kFloat32, TypeInfo::kFloat32)};
    std::unique_ptr<frontend::TreeBuilder> tree{
        new frontend::TreeBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32)};
    tree->CreateNode(0);
    tree->CreateNode(1);
    tree->CreateNode(2);
    tree->SetNumericalTestNode(0, 0, "<", frontend::Value::Create<float>(0.0f), true, 1, 2);
    tree->SetRootNode(0);
    tree->SetLeafNode(1, frontend::Value::Create<float>(1.0f));
    tree->SetLeafNode(2, frontend::Value::Create<float>(2.0f));
    builder->InsertTree(tree.get());

    model_objs.push_back(builder->CommitModel());
  }

  std::vector<Model const*> model_obj_refs;
  std::transform(model_objs.begin(), model_objs.end(), std::back_inserter(model_obj_refs),
      [](auto const& obj) { return obj.get(); });

  std::unique_ptr<Model> concatenated_model = ConcatenateModelObjects(model_obj_refs);
  ASSERT_EQ(concatenated_model->GetNumTree(), kNumModelObjs);
  TestRoundTrip(concatenated_model.get());
  auto& trees = std::get<ModelPreset<float, float>>(concatenated_model->variant_).trees;
  for (int i = 0; i < kNumModelObjs; ++i) {
    auto const& tree = trees[i];
    ASSERT_FALSE(tree.IsLeaf(0));
    ASSERT_TRUE(tree.IsLeaf(1));
    ASSERT_TRUE(tree.IsLeaf(2));
    ASSERT_EQ(tree.SplitType(0), SplitFeatureType::kNumerical);
    ASSERT_EQ(tree.SplitIndex(0), 0);
    ASSERT_EQ(tree.Threshold(0), 0.0f);
    ASSERT_EQ(tree.LeftChild(0), 1);
    ASSERT_EQ(tree.RightChild(0), 2);
    ASSERT_EQ(tree.LeafValue(1), 1.0f);
    ASSERT_EQ(tree.LeafValue(2), 2.0f);
  }
}

TEST(ModelConcatenation, MismatchedTreeType) {
  std::vector<std::unique_ptr<Model>> model_objs;

  std::unique_ptr<frontend::ModelBuilder> builder{
      new frontend::ModelBuilder(2, 1, false, TypeInfo::kFloat32, TypeInfo::kFloat32)};
  std::unique_ptr<frontend::TreeBuilder> tree{
      new frontend::TreeBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32)};
  tree->CreateNode(0);
  tree->SetRootNode(0);
  tree->SetLeafNode(0, frontend::Value::Create<float>(1.0f));
  builder->InsertTree(tree.get());
  model_objs.push_back(builder->CommitModel());

  std::unique_ptr<frontend::ModelBuilder> builder2{
      new frontend::ModelBuilder(2, 1, false, TypeInfo::kFloat64, TypeInfo::kFloat64)};
  std::unique_ptr<frontend::TreeBuilder> tree2{
      new frontend::TreeBuilder(TypeInfo::kFloat64, TypeInfo::kFloat64)};
  tree2->CreateNode(0);
  tree2->SetRootNode(0);
  tree2->SetLeafNode(0, frontend::Value::Create<double>(1.0));
  builder2->InsertTree(tree2.get());
  model_objs.push_back(builder2->CommitModel());

  std::vector<Model const*> model_obj_refs;
  std::transform(model_objs.begin(), model_objs.end(), std::back_inserter(model_obj_refs),
      [](auto const& obj) { return obj.get(); });
  ASSERT_THROW(ConcatenateModelObjects(model_obj_refs), treelite::Error);
}

}  // namespace treelite
