/*!
 * Copyright (c) 2020 by Contributors
 * \file test_serializer.cc
 * \author Hyunsu Cho
 * \brief C++ tests for model serializer
 */
#include <gtest/gtest.h>
#include <treelite/tree.h>
#include <treelite/frontend.h>
#include <dmlc/memory_io.h>
#include <string>
#include <memory>
#include <stdexcept>

namespace {

inline std::string TreeliteToBytes(treelite::Model* model) {
  std::string s;
  std::unique_ptr<dmlc::Stream> mstrm{new dmlc::MemoryStringStream(&s)};
  model->ReferenceSerialize(mstrm.get());
  mstrm.reset();
  return s;
}

inline void TestRoundTrip(treelite::Model* model) {
  auto buffer = model->GetPyBuffer();
  std::unique_ptr<treelite::Model> received_model = treelite::Model::CreateFromPyBuffer(buffer);

  ASSERT_EQ(TreeliteToBytes(model), TreeliteToBytes(received_model.get()));
}

}  // anonymous namespace

namespace treelite {

template <typename ThresholdType, typename LeafOutputType>
void PyBufferInterfaceRoundTrip_TreeStump() {
  TypeInfo threshold_type = InferTypeInfoOf<ThresholdType>();
  TypeInfo leaf_output_type = InferTypeInfoOf<LeafOutputType>();
  std::unique_ptr<frontend::ModelBuilder> builder{
      new frontend::ModelBuilder(2, 1, false, threshold_type, leaf_output_type)
  };
  std::unique_ptr<frontend::TreeBuilder> tree{
      new frontend::TreeBuilder(threshold_type, leaf_output_type)
  };
  tree->CreateNode(0);
  tree->CreateNode(1);
  tree->CreateNode(2);
  tree->SetNumericalTestNode(0, 0, "<", frontend::Value::Create<ThresholdType>(0), true, 1, 2);
  tree->SetRootNode(0);
  tree->SetLeafNode(1, frontend::Value::Create<LeafOutputType>(-1));
  tree->SetLeafNode(2, frontend::Value::Create<LeafOutputType>(1));
  builder->InsertTree(tree.get());

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());
}

TEST(PyBufferInterfaceRoundTrip, TreeStump) {
  PyBufferInterfaceRoundTrip_TreeStump<float, float>();
  PyBufferInterfaceRoundTrip_TreeStump<float, uint32_t>();
  PyBufferInterfaceRoundTrip_TreeStump<double, double>();
  PyBufferInterfaceRoundTrip_TreeStump<double, uint32_t>();
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStump<float, double>()), std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStump<double, float>()), std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStump<uint32_t, uint32_t>()), std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStump<uint32_t, float>()), std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStump<uint32_t, double>()), std::runtime_error);
}

template <typename ThresholdType, typename LeafOutputType>
void PyBufferInterfaceRoundTrip_TreeStumpLeafVec() {
  TypeInfo threshold_type = InferTypeInfoOf<ThresholdType>();
  TypeInfo leaf_output_type = InferTypeInfoOf<LeafOutputType>();
  std::unique_ptr<frontend::ModelBuilder> builder{
      new frontend::ModelBuilder(2, 2, true, threshold_type, leaf_output_type)
  };
  std::unique_ptr<frontend::TreeBuilder> tree{
      new frontend::TreeBuilder(threshold_type, leaf_output_type)
  };
  tree->CreateNode(0);
  tree->CreateNode(1);
  tree->CreateNode(2);
  tree->SetNumericalTestNode(0, 0, "<", frontend::Value::Create<ThresholdType>(0), true, 1, 2);
  tree->SetRootNode(0);
  tree->SetLeafVectorNode(1, {frontend::Value::Create<LeafOutputType>(-1),
                              frontend::Value::Create<LeafOutputType>(1)});
  tree->SetLeafVectorNode(2, {frontend::Value::Create<LeafOutputType>(1),
                              frontend::Value::Create<LeafOutputType>(-1)});
  builder->InsertTree(tree.get());

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());
}

TEST(PyBufferInterfaceRoundTrip, TreeStumpLeafVec) {
  PyBufferInterfaceRoundTrip_TreeStumpLeafVec<float, float>();
  PyBufferInterfaceRoundTrip_TreeStumpLeafVec<float, uint32_t>();
  PyBufferInterfaceRoundTrip_TreeStumpLeafVec<double, double>();
  PyBufferInterfaceRoundTrip_TreeStumpLeafVec<double, uint32_t>();
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStumpLeafVec<float, double>()),
               std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStumpLeafVec<double, float>()),
               std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStumpLeafVec<uint32_t, uint32_t>()),
               std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStumpLeafVec<uint32_t, float>()),
               std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStumpLeafVec<uint32_t, double>()),
               std::runtime_error);
}

template <typename ThresholdType, typename LeafOutputType>
void PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit() {
  TypeInfo threshold_type = InferTypeInfoOf<ThresholdType>();
  TypeInfo leaf_output_type = InferTypeInfoOf<LeafOutputType>();
  std::unique_ptr<frontend::ModelBuilder> builder{
      new frontend::ModelBuilder(2, 1, false, threshold_type, leaf_output_type)
  };
  std::unique_ptr<frontend::TreeBuilder> tree{
      new frontend::TreeBuilder(threshold_type, leaf_output_type)
  };
  tree->CreateNode(0);
  tree->CreateNode(1);
  tree->CreateNode(2);
  tree->SetCategoricalTestNode(0, 0, {0, 1}, true, 1, 2);
  tree->SetRootNode(0);
  tree->SetLeafNode(1, frontend::Value::Create<LeafOutputType>(-1));
  tree->SetLeafNode(2, frontend::Value::Create<LeafOutputType>(1));
  builder->InsertTree(tree.get());

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());
}

TEST(PyBufferInterfaceRoundTrip, TreeStumpCategoricalSplit) {
  PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<float, float>();
  PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<float, uint32_t>();
  PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<double, double>();
  PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<double, uint32_t>();
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<float, double>()),
               std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<double, float>()),
               std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<uint32_t, uint32_t>()),
               std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<uint32_t, float>()),
               std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<uint32_t, double>()),
               std::runtime_error);
}

template <typename ThresholdType, typename LeafOutputType>
void PyBufferInterfaceRoundTrip_TreeDepth2() {
  TypeInfo threshold_type = InferTypeInfoOf<ThresholdType>();
  TypeInfo leaf_output_type = InferTypeInfoOf<LeafOutputType>();
  std::unique_ptr<frontend::ModelBuilder> builder{
      new frontend::ModelBuilder(2, 1, false, threshold_type, leaf_output_type)
  };
  builder->SetModelParam("pred_transform", "sigmoid");
  builder->SetModelParam("global_bias", "0.5");
  for (int tree_id = 0; tree_id < 2; ++tree_id) {
    std::unique_ptr<frontend::TreeBuilder> tree{
        new frontend::TreeBuilder(threshold_type, leaf_output_type)
    };
    for (int i = 0; i < 7; ++i) {
      tree->CreateNode(i);
    }
    tree->SetNumericalTestNode(0, 0, "<", frontend::Value::Create<ThresholdType>(0), true, 1, 2);
    tree->SetCategoricalTestNode(1, 0, {0, 1}, true, 3, 4);
    tree->SetCategoricalTestNode(2, 1, {0}, true, 5, 6);
    tree->SetRootNode(0);
    tree->SetLeafNode(3, frontend::Value::Create<LeafOutputType>(-2));
    tree->SetLeafNode(4, frontend::Value::Create<LeafOutputType>(1));
    tree->SetLeafNode(5, frontend::Value::Create<LeafOutputType>(-1));
    tree->SetLeafNode(6, frontend::Value::Create<LeafOutputType>(2));
    builder->InsertTree(tree.get());
  }

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());
}

TEST(PyBufferInterfaceRoundTrip, TreeDepth2) {
  PyBufferInterfaceRoundTrip_TreeDepth2<float, float>();
  PyBufferInterfaceRoundTrip_TreeDepth2<float, uint32_t>();
  PyBufferInterfaceRoundTrip_TreeDepth2<double, double>();
  PyBufferInterfaceRoundTrip_TreeDepth2<double, uint32_t>();
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeDepth2<float, double>()), std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeDepth2<double, float>()), std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeDepth2<uint32_t, uint32_t>()), std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeDepth2<uint32_t, float>()), std::runtime_error);
  ASSERT_THROW((PyBufferInterfaceRoundTrip_TreeDepth2<uint32_t, double>()), std::runtime_error);
}

template <typename ThresholdType, typename LeafOutputType>
void PyBufferInterfaceRoundTrip_DeepFullTree() {
  TypeInfo threshold_type = InferTypeInfoOf<ThresholdType>();
  TypeInfo leaf_output_type = InferTypeInfoOf<LeafOutputType>();
  const int depth = 19;

  std::unique_ptr<frontend::ModelBuilder> builder{
      new frontend::ModelBuilder(3, 1, false, threshold_type, leaf_output_type)
  };
  std::unique_ptr<frontend::TreeBuilder> tree{
      new frontend::TreeBuilder(threshold_type, leaf_output_type)
  };
  for (int level = 0; level <= depth; ++level) {
    for (int i = 0; i < (1 << level); ++i) {
      const int nid = (1 << level) - 1 + i;
      tree->CreateNode(nid);
    }
  }
  for (int level = 0; level <= depth; ++level) {
    for (int i = 0; i < (1 << level); ++i) {
      const int nid = (1 << level) - 1 + i;
      if (level == depth) {
        tree->SetLeafNode(nid, frontend::Value::Create<LeafOutputType>(1));
      } else {
        tree->SetNumericalTestNode(nid, (level % 2), "<", frontend::Value::Create<ThresholdType>(0),
                                   true, 2 * nid + 1, 2 * nid + 2);
      }
    }
  }
  tree->SetRootNode(0);
  builder->InsertTree(tree.get());

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());
}

TEST(PyBufferInterfaceRoundTrip, DeepFullTree) {
  PyBufferInterfaceRoundTrip_DeepFullTree<float, float>();
  PyBufferInterfaceRoundTrip_DeepFullTree<float, uint32_t>();
  PyBufferInterfaceRoundTrip_DeepFullTree<double, double>();
  PyBufferInterfaceRoundTrip_DeepFullTree<double, uint32_t>();
}

}  // namespace treelite
