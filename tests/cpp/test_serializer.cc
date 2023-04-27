/*!
 * Copyright (c) 2020 by Contributors
 * \file test_serializer.cc
 * \author Hyunsu Cho
 * \brief C++ tests for model serializer
 */
#include <gtest/gtest.h>
#include <treelite/tree.h>
#include <treelite/frontend.h>
#include <fmt/format.h>
#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>
#include <cstdint>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <fstream>
#include <stdexcept>

using namespace fmt::literals;

namespace {

inline void TestRoundTrip(treelite::Model* model) {
  for (int i = 0; i < 2; ++i) {
    // Test round trip with in-memory serialization
    auto buffer = model->GetPyBuffer();
    std::unique_ptr<treelite::Model> received_model = treelite::Model::CreateFromPyBuffer(buffer);

    // Use ASSERT_TRUE, since ASSERT_EQ will dump all the raw bytes into a string, potentially
    // causing an OOM error
    ASSERT_TRUE(model->DumpAsJSON(false) == received_model->DumpAsJSON(false));
  }

  for (int i = 0; i < 2; ++i) {
    // Test round trip with serialization to a file stream
    const char* filename = std::tmpnam(nullptr);
    std::unique_ptr<treelite::Model> received_model;
    {
      std::ofstream ofs(filename, std::ios::out | std::ios::binary);
      ASSERT_TRUE(ofs);
      ofs.exceptions(std::ios::failbit | std::ios::badbit);
      model->SerializeToStream(ofs);
    }
    {
      std::ifstream ifs(filename, std::ios::in | std::ios::binary);
      ASSERT_TRUE(ifs);
      ifs.exceptions(std::ios::failbit | std::ios::badbit);
      received_model = treelite::Model::DeserializeFromStream(ifs);
    }

    // Use ASSERT_TRUE, since ASSERT_EQ will dump all the raw bytes into a string, potentially
    // causing an OOM error
    ASSERT_TRUE(model->DumpAsJSON(false) == received_model->DumpAsJSON(false));
  }
}

}  // anonymous namespace

namespace treelite {

template <typename ThresholdType, typename LeafOutputType>
void PyBufferInterfaceRoundTrip_TreeStump() {
  TypeInfo threshold_type = TypeToInfo<ThresholdType>();
  TypeInfo leaf_output_type = TypeToInfo<LeafOutputType>();
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
  tree->SetLeafNode(1, frontend::Value::Create<LeafOutputType>(1));
  tree->SetLeafNode(2, frontend::Value::Create<LeafOutputType>(2));
  builder->InsertTree(tree.get());

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());

  /* Test correctness of JSON dump */
  std::string expected_json_dump_str = fmt::format(R"JSON(
  {{
    "num_feature": 2,
    "task_type": "kBinaryClfRegr",
    "average_tree_output": false,
    "task_param": {{
        "output_type": "float",
        "grove_per_class": false,
        "num_class": 1,
        "leaf_vector_size": 1
    }},
    "model_param": {{
        "pred_transform": "identity",
        "sigmoid_alpha": 1.0,
        "ratio_c": 1.0,
        "global_bias": 0.0
    }},
    "trees": [{{
            "num_nodes": 3,
            "has_categorical_split": false,
            "nodes": [{{
                    "node_id": 0,
                    "split_feature_id": 0,
                    "default_left": true,
                    "split_type": "numerical",
                    "comparison_op": "<",
                    "threshold": {threshold},
                    "left_child": 1,
                    "right_child": 2
                }}, {{
                    "node_id": 1,
                    "leaf_value": {leaf_value0}
                }}, {{
                    "node_id": 2,
                    "leaf_value": {leaf_value1}
                }}]
        }}]
  }}
  )JSON",
    "threshold"_a = static_cast<ThresholdType>(0),
    "leaf_value0"_a = static_cast<LeafOutputType>(1),
    "leaf_value1"_a = static_cast<LeafOutputType>(2)
  );

  rapidjson::Document json_dump;
  json_dump.Parse(model->DumpAsJSON(false).c_str());

  rapidjson::Document expected_json_dump;
  expected_json_dump.Parse(expected_json_dump_str.c_str());
  EXPECT_TRUE(json_dump == expected_json_dump);
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
  TypeInfo threshold_type = TypeToInfo<ThresholdType>();
  TypeInfo leaf_output_type = TypeToInfo<LeafOutputType>();
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
  tree->SetLeafVectorNode(1, {frontend::Value::Create<LeafOutputType>(1),
                              frontend::Value::Create<LeafOutputType>(2)});
  tree->SetLeafVectorNode(2, {frontend::Value::Create<LeafOutputType>(2),
                              frontend::Value::Create<LeafOutputType>(1)});
  builder->InsertTree(tree.get());

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());

  /* Test correctness of JSON dump */
  std::string expected_json_dump_str = fmt::format(R"JSON(
  {{
    "num_feature": 2,
    "task_type": "kMultiClfProbDistLeaf",
    "average_tree_output": true,
    "task_param": {{
        "output_type": "float",
        "grove_per_class": false,
        "num_class": 2,
        "leaf_vector_size": 2
    }},
    "model_param": {{
        "pred_transform": "identity",
        "sigmoid_alpha": 1.0,
        "ratio_c": 1.0,
        "global_bias": 0.0
    }},
    "trees": [{{
            "num_nodes": 3,
            "has_categorical_split": false,
            "nodes": [{{
                    "node_id": 0,
                    "split_feature_id": 0,
                    "default_left": true,
                    "split_type": "numerical",
                    "comparison_op": "<",
                    "threshold": {threshold},
                    "left_child": 1,
                    "right_child": 2
                }}, {{
                    "node_id": 1,
                    "leaf_value": [{leaf_value0}, {leaf_value1}]
                }}, {{
                    "node_id": 2,
                    "leaf_value": [{leaf_value2}, {leaf_value3}]
                }}]
        }}]
  }}
  )JSON",
    "threshold"_a = static_cast<ThresholdType>(0),
    "leaf_value0"_a = static_cast<LeafOutputType>(1),
    "leaf_value1"_a = static_cast<LeafOutputType>(2),
    "leaf_value2"_a = static_cast<LeafOutputType>(2),
    "leaf_value3"_a = static_cast<LeafOutputType>(1)
  );

  rapidjson::Document json_dump;
  json_dump.Parse(model->DumpAsJSON(false).c_str());

  rapidjson::Document expected_json_dump;
  expected_json_dump.Parse(expected_json_dump_str.c_str());
  EXPECT_TRUE(json_dump == expected_json_dump);
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
void PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit(
    const std::vector<uint32_t>& left_categories) {
  TypeInfo threshold_type = TypeToInfo<ThresholdType>();
  TypeInfo leaf_output_type = TypeToInfo<LeafOutputType>();
  std::unique_ptr<frontend::ModelBuilder> builder{
      new frontend::ModelBuilder(2, 1, false, threshold_type, leaf_output_type)
  };
  std::unique_ptr<frontend::TreeBuilder> tree{
      new frontend::TreeBuilder(threshold_type, leaf_output_type)
  };
  tree->CreateNode(0);
  tree->CreateNode(1);
  tree->CreateNode(2);
  tree->SetCategoricalTestNode(0, 0, left_categories, false, 1, 2);
  tree->SetRootNode(0);
  tree->SetLeafNode(1, frontend::Value::Create<LeafOutputType>(2));
  tree->SetLeafNode(2, frontend::Value::Create<LeafOutputType>(3));
  builder->InsertTree(tree.get());

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());

  /* Test correctness of JSON dump */
  std::string categories_list_str;
  {
    std::ostringstream oss;
    rapidjson::OStreamWrapper os_wrapper(oss);
    rapidjson::Writer<rapidjson::OStreamWrapper> writer(os_wrapper);
    writer.StartArray();
    for (auto e : left_categories) {
      writer.Uint(static_cast<unsigned int>(e));
    }
    writer.EndArray();
    categories_list_str = oss.str();
  }
  std::string expected_json_dump_str = fmt::format(R"JSON(
  {{
    "num_feature": 2,
    "task_type": "kBinaryClfRegr",
    "average_tree_output": false,
    "task_param": {{
        "output_type": "float",
        "grove_per_class": false,
        "num_class": 1,
        "leaf_vector_size": 1
    }},
    "model_param": {{
        "pred_transform": "identity",
        "sigmoid_alpha": 1.0,
        "ratio_c": 1.0,
        "global_bias": 0.0
    }},
    "trees": [{{
            "num_nodes": 3,
            "has_categorical_split": true,
            "nodes": [{{
                    "node_id": 0,
                    "split_feature_id": 0,
                    "default_left": false,
                    "split_type": "categorical",
                    "categories_list_right_child": false,
                    "categories_list": {categories_list},
                    "left_child": 1,
                    "right_child": 2
                }}, {{
                    "node_id": 1,
                    "leaf_value": {leaf_value0}
                }}, {{
                    "node_id": 2,
                    "leaf_value": {leaf_value1}
                }}]
        }}]
  }}
  )JSON",
    "leaf_value0"_a = static_cast<LeafOutputType>(2),
    "leaf_value1"_a = static_cast<LeafOutputType>(3),
    "categories_list"_a = categories_list_str
  );

  rapidjson::Document json_dump;
  json_dump.Parse(model->DumpAsJSON(false).c_str());

  rapidjson::Document expected_json_dump;
  expected_json_dump.Parse(expected_json_dump_str.c_str());
  EXPECT_TRUE(json_dump == expected_json_dump);
}

TEST(PyBufferInterfaceRoundTrip, TreeStumpCategoricalSplit) {
  for (const auto& left_categories : std::vector<std::vector<uint32_t>>{ {}, {1}, {0, 1} }) {
    PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<float, float>(left_categories);
    PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<float, uint32_t>(left_categories);
    PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<double, double>(left_categories);
    PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<double, uint32_t>(left_categories);
    ASSERT_THROW(
        (PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<float, double>(left_categories)),
        std::runtime_error);
    ASSERT_THROW(
        (PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<double, float>(left_categories)),
        std::runtime_error);
    ASSERT_THROW(
        (PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<uint32_t, uint32_t>(left_categories)),
        std::runtime_error);
    ASSERT_THROW(
        (PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<uint32_t, float>(left_categories)),
        std::runtime_error);
    ASSERT_THROW(
        (PyBufferInterfaceRoundTrip_TreeStumpCategoricalSplit<uint32_t, double>(left_categories)),
        std::runtime_error);
  }
}

template <typename ThresholdType, typename LeafOutputType>
void PyBufferInterfaceRoundTrip_TreeDepth2() {
  TypeInfo threshold_type = TypeToInfo<ThresholdType>();
  TypeInfo leaf_output_type = TypeToInfo<LeafOutputType>();
  std::unique_ptr<frontend::ModelBuilder> builder{
      new frontend::ModelBuilder(2, 1, false, threshold_type, leaf_output_type)
  };
  builder->SetModelParam("pred_transform", "sigmoid");
  builder->SetModelParam("global_bias", "0.5");
  for (int tree_id = 0; tree_id < 3; ++tree_id) {
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
    tree->SetLeafNode(3, frontend::Value::Create<LeafOutputType>(tree_id + 3));
    tree->SetLeafNode(4, frontend::Value::Create<LeafOutputType>(tree_id + 1));
    tree->SetLeafNode(5, frontend::Value::Create<LeafOutputType>(tree_id + 4));
    tree->SetLeafNode(6, frontend::Value::Create<LeafOutputType>(tree_id + 2));
    builder->InsertTree(tree.get());
  }

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());

  std::string expected_json_dump_str = fmt::format(R"JSON(
  {{
    "num_feature": 2,
    "task_type": "kBinaryClfRegr",
    "average_tree_output": false,
    "task_param": {{
        "output_type": "float",
        "grove_per_class": false,
        "num_class": 1,
        "leaf_vector_size": 1
    }},
    "model_param": {{
        "pred_transform": "sigmoid",
        "sigmoid_alpha": 1.0,
        "ratio_c": 1.0,
        "global_bias": 0.5
    }},
    "trees": [{{
            "num_nodes": 7,
            "has_categorical_split": true,
            "nodes": [{{
                    "node_id": 0,
                    "split_feature_id": 0,
                    "default_left": true,
                    "split_type": "numerical",
                    "comparison_op": "<",
                    "threshold": {threshold},
                    "left_child": 1,
                    "right_child": 2
                }}, {{
                    "node_id": 1,
                    "split_feature_id": 0,
                    "default_left": true,
                    "split_type": "categorical",
                    "categories_list_right_child": false,
                    "categories_list": [0, 1],
                    "left_child": 3,
                    "right_child": 4
                }}, {{
                    "node_id": 2,
                    "split_feature_id": 1,
                    "default_left": true,
                    "split_type": "categorical",
                    "categories_list_right_child": false,
                    "categories_list": [0],
                    "left_child": 5,
                    "right_child": 6
                }}, {{
                    "node_id": 3,
                    "leaf_value": {tree0_leaf3}
                }}, {{
                    "node_id": 4,
                    "leaf_value": {tree0_leaf4}
                }}, {{
                    "node_id": 5,
                    "leaf_value": {tree0_leaf5}
                }}, {{
                    "node_id": 6,
                    "leaf_value": {tree0_leaf6}
                }}]
        }}, {{
            "num_nodes": 7,
            "has_categorical_split": true,
            "nodes": [{{
                    "node_id": 0,
                    "split_feature_id": 0,
                    "default_left": true,
                    "split_type": "numerical",
                    "comparison_op": "<",
                    "threshold": {threshold},
                    "left_child": 1,
                    "right_child": 2
                }}, {{
                    "node_id": 1,
                    "split_feature_id": 0,
                    "default_left": true,
                    "split_type": "categorical",
                    "categories_list_right_child": false,
                    "categories_list": [0, 1],
                    "left_child": 3,
                    "right_child": 4
                }}, {{
                    "node_id": 2,
                    "split_feature_id": 1,
                    "default_left": true,
                    "split_type": "categorical",
                    "categories_list_right_child": false,
                    "categories_list": [0],
                    "left_child": 5,
                    "right_child": 6
                }}, {{
                    "node_id": 3,
                    "leaf_value": {tree1_leaf3}
                }}, {{
                    "node_id": 4,
                    "leaf_value": {tree1_leaf4}
                }}, {{
                    "node_id": 5,
                    "leaf_value": {tree1_leaf5}
                }}, {{
                    "node_id": 6,
                    "leaf_value": {tree1_leaf6}
                }}]
        }}, {{
            "num_nodes": 7,
            "has_categorical_split": true,
            "nodes": [{{
                    "node_id": 0,
                    "split_feature_id": 0,
                    "default_left": true,
                    "split_type": "numerical",
                    "comparison_op": "<",
                    "threshold": {threshold},
                    "left_child": 1,
                    "right_child": 2
                }}, {{
                    "node_id": 1,
                    "split_feature_id": 0,
                    "default_left": true,
                    "split_type": "categorical",
                    "categories_list_right_child": false,
                    "categories_list": [0, 1],
                    "left_child": 3,
                    "right_child": 4
                }}, {{
                    "node_id": 2,
                    "split_feature_id": 1,
                    "default_left": true,
                    "split_type": "categorical",
                    "categories_list_right_child": false,
                    "categories_list": [0],
                    "left_child": 5,
                    "right_child": 6
                }}, {{
                    "node_id": 3,
                    "leaf_value": {tree2_leaf3}
                }}, {{
                    "node_id": 4,
                    "leaf_value": {tree2_leaf4}
                }}, {{
                    "node_id": 5,
                    "leaf_value": {tree2_leaf5}
                }}, {{
                    "node_id": 6,
                    "leaf_value": {tree2_leaf6}
                }}]
        }}]
  }}
  )JSON",
    "threshold"_a = static_cast<ThresholdType>(0),
    "tree0_leaf3"_a = static_cast<LeafOutputType>(3),
    "tree0_leaf4"_a = static_cast<LeafOutputType>(1),
    "tree0_leaf5"_a = static_cast<LeafOutputType>(4),
    "tree0_leaf6"_a = static_cast<LeafOutputType>(2),
    "tree1_leaf3"_a = static_cast<LeafOutputType>(3 + 1),
    "tree1_leaf4"_a = static_cast<LeafOutputType>(1 + 1),
    "tree1_leaf5"_a = static_cast<LeafOutputType>(4 + 1),
    "tree1_leaf6"_a = static_cast<LeafOutputType>(2 + 1),
    "tree2_leaf3"_a = static_cast<LeafOutputType>(3 + 2),
    "tree2_leaf4"_a = static_cast<LeafOutputType>(1 + 2),
    "tree2_leaf5"_a = static_cast<LeafOutputType>(4 + 2),
    "tree2_leaf6"_a = static_cast<LeafOutputType>(2 + 2)
  );

  rapidjson::Document json_dump;
  json_dump.Parse(model->DumpAsJSON(false).c_str());

  rapidjson::Document expected_json_dump;
  expected_json_dump.Parse(expected_json_dump_str.c_str());
  EXPECT_TRUE(json_dump == expected_json_dump);
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
  TypeInfo threshold_type = TypeToInfo<ThresholdType>();
  TypeInfo leaf_output_type = TypeToInfo<LeafOutputType>();
  const int depth = 12;

  std::unique_ptr<frontend::ModelBuilder> builder{
      new frontend::ModelBuilder(3, 1, false, threshold_type, leaf_output_type)
  };
  for (int tree_id = 0; tree_id < 3; ++tree_id) {
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
          tree->SetLeafNode(nid, frontend::Value::Create<LeafOutputType>(tree_id + 1));
        } else {
          tree->SetNumericalTestNode(nid, (level % 2), "<", frontend::Value::Create<ThresholdType>(0),
                                     true, 2 * nid + 1, 2 * nid + 2);
        }
      }
    }
    tree->SetRootNode(0);
    builder->InsertTree(tree.get());
  }

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());
}

TEST(PyBufferInterfaceRoundTrip, DeepFullTree) {
  PyBufferInterfaceRoundTrip_DeepFullTree<float, float>();
  PyBufferInterfaceRoundTrip_DeepFullTree<float, uint32_t>();
  PyBufferInterfaceRoundTrip_DeepFullTree<double, double>();
  PyBufferInterfaceRoundTrip_DeepFullTree<double, uint32_t>();
}

TEST(PyBufferInterfaceRoundTrip, XGBoostBoston) {
  std::unique_ptr<frontend::ModelBuilder> builder{
      new frontend::ModelBuilder(13, 1, false, TypeInfo::kFloat32, TypeInfo::kFloat32)
  };
  using frontend::Value;
  {
    std::unique_ptr<frontend::TreeBuilder> tree{
        new frontend::TreeBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32)
    };
    for (int nid = 0; nid <= 18; ++nid) {
      tree->CreateNode(nid);
    }
    tree->SetRootNode(0);
    tree->SetNumericalTestNode(0, 5, "<", Value::Create<float>(6.67599964f), true, 1, 2);
    tree->SetNumericalTestNode(1, 12, "<", Value::Create<float>(16.0849991f), true, 3, 4);
    tree->SetNumericalTestNode(3, 12, "<", Value::Create<float>(9.71500015f), true, 7, 8);
    tree->SetNumericalTestNode(7, 0, "<", Value::Create<float>(4.72704506f), true, 15, 16);
    tree->SetLeafNode(15, Value::Create<float>(23.1568813f));
    tree->SetLeafNode(16, Value::Create<float>(37.125f));
    tree->SetLeafNode(8, Value::Create<float>(19.625f));
    tree->SetNumericalTestNode(4, 0, "<", Value::Create<float>(7.31708527f), true, 9, 10);
    tree->SetLeafNode(9, Value::Create<float>(16.0535717f));
    tree->SetNumericalTestNode(10, 0, "<", Value::Create<float>(24.9239006f), true, 17, 18);
    tree->SetLeafNode(17, Value::Create<float>(9.96249962f));
    tree->SetLeafNode(18, Value::Create<float>(3.20000005f));
    tree->SetNumericalTestNode(2, 5, "<", Value::Create<float>(7.43700027f), true, 5, 6);
    tree->SetNumericalTestNode(5, 0, "<", Value::Create<float>(8.18112564f), true, 11, 12);
    tree->SetLeafNode(11, Value::Create<float>(31.0228577f));
    tree->SetLeafNode(12, Value::Create<float>(9.30000019f));
    tree->SetNumericalTestNode(6, 0, "<", Value::Create<float>(2.74223518f), true, 13, 14);
    tree->SetLeafNode(13, Value::Create<float>(43.8833313f));
    tree->SetLeafNode(14, Value::Create<float>(10.6999998f));
    builder->InsertTree(tree.get());
  }
  {
    std::unique_ptr<frontend::TreeBuilder> tree{
        new frontend::TreeBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32)
    };
    for (int nid = 0; nid <= 18; ++nid) {
      tree->CreateNode(nid);
    }
    tree->SetRootNode(0);
    tree->SetNumericalTestNode(0, 7, "<", Value::Create<float>(1.37205005f), true, 1, 2);
    tree->SetNumericalTestNode(1, 12, "<", Value::Create<float>(11.4849997f), true, 3, 4);
    tree->SetLeafNode(3, Value::Create<float>(12.3465471f));
    tree->SetNumericalTestNode(4, 0, "<", Value::Create<float>(6.57226992f), true, 7, 8);
    tree->SetLeafNode(7, Value::Create<float>(-2.63571453f));
    tree->SetLeafNode(8, Value::Create<float>(3.13541675f));
    tree->SetNumericalTestNode(2, 12, "<", Value::Create<float>(5.43999958f), true, 5, 6);
    tree->SetNumericalTestNode(5, 7, "<", Value::Create<float>(2.85050011f), true, 9, 10);
    tree->SetLeafNode(9, Value::Create<float>(6.06147718f));
    tree->SetNumericalTestNode(10, 6, "<", Value::Create<float>(11.9499998f), true, 13, 14);
    tree->SetLeafNode(13, Value::Create<float>(-1.27356553f));
    tree->SetLeafNode(14, Value::Create<float>(1.90286708f));
    tree->SetNumericalTestNode(6, 9, "<", Value::Create<float>(278.0f), true, 11, 12);
    tree->SetNumericalTestNode(11, 7, "<", Value::Create<float>(4.36069965f), true, 15, 16);
    tree->SetLeafNode(15, Value::Create<float>(2.3283093f));
    tree->SetLeafNode(16, Value::Create<float>(-0.740369797f));
    tree->SetNumericalTestNode(12, 5, "<", Value::Create<float>(6.67599964f), true, 17, 18);
    tree->SetLeafNode(17, Value::Create<float>(-0.374256492f));
    tree->SetLeafNode(18, Value::Create<float>(-3.15714335f));
    builder->InsertTree(tree.get());
  }
  {
    std::unique_ptr<frontend::TreeBuilder> tree{
        new frontend::TreeBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32)
    };
    for (int nid = 0; nid <= 30; ++nid) {
      tree->CreateNode(nid);
    }
    tree->SetRootNode(0);
    tree->SetNumericalTestNode(0, 10, "<", Value::Create<float>(20.9500008f), true, 1, 2);
    tree->SetNumericalTestNode(1, 7, "<", Value::Create<float>(6.22315025f), true, 3, 4);
    tree->SetNumericalTestNode(3, 8, "<", Value::Create<float>(2.5f), true, 7, 8);
    tree->SetNumericalTestNode(7, 6, "<", Value::Create<float>(62.7999992f), true, 15, 16);
    tree->SetLeafNode(15, Value::Create<float>(1.08206058f));
    tree->SetLeafNode(16, Value::Create<float>(-2.5734961f));
    tree->SetNumericalTestNode(8, 2, "<", Value::Create<float>(3.09500003f), true, 17, 18);
    tree->SetLeafNode(17, Value::Create<float>(2.1601212f));
    tree->SetLeafNode(18, Value::Create<float>(0.435377121f));
    tree->SetNumericalTestNode(4, 5, "<", Value::Create<float>(6.94500017f), true, 9, 10);
    tree->SetNumericalTestNode(9, 5, "<", Value::Create<float>(6.67650032f), true, 19, 20);
    tree->SetLeafNode(19, Value::Create<float>(-0.727042675f));
    tree->SetLeafNode(20, Value::Create<float>(-2.75314689f));
    tree->SetNumericalTestNode(10, 12, "<", Value::Create<float>(4.77499962f), true, 21, 22);
    tree->SetLeafNode(21, Value::Create<float>(-0.468363762f));
    tree->SetLeafNode(22, Value::Create<float>(3.01290059f));
    tree->SetNumericalTestNode(2, 5, "<", Value::Create<float>(6.11900043f), true, 5, 6);
    tree->SetNumericalTestNode(5, 9, "<", Value::Create<float>(385.5f), true, 11, 12);
    tree->SetNumericalTestNode(11, 5, "<", Value::Create<float>(5.51300001f), true, 23, 24);
    tree->SetLeafNode(23, Value::Create<float>(0.224628448f));
    tree->SetLeafNode(24, Value::Create<float>(-2.32530594f));
    tree->SetNumericalTestNode(12, 5, "<", Value::Create<float>(5.91499996f), true, 25, 26);
    tree->SetLeafNode(25, Value::Create<float>(-1.16225815f));
    tree->SetLeafNode(26, Value::Create<float>(0.610342026f));
    tree->SetNumericalTestNode(6, 0, "<", Value::Create<float>(0.448424995f), true, 13, 14);
    tree->SetNumericalTestNode(13, 5, "<", Value::Create<float>(6.45600033f), true, 27, 28);
    tree->SetLeafNode(27, Value::Create<float>(-1.64520073f));
    tree->SetLeafNode(28, Value::Create<float>(-0.275371552f));
    tree->SetNumericalTestNode(14, 0, "<", Value::Create<float>(0.681519985f), true, 29, 30);
    tree->SetLeafNode(29, Value::Create<float>(1.69765615f));
    tree->SetLeafNode(30, Value::Create<float>(-0.246309474f));
    builder->InsertTree(tree.get());
  }
  {
    std::unique_ptr<frontend::TreeBuilder> tree{
        new frontend::TreeBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32)
    };
    for (int nid = 0; nid <= 26; ++nid) {
      tree->CreateNode(nid);
    }
    tree->SetRootNode(0);
    tree->SetNumericalTestNode(0, 5, "<", Value::Create<float>(6.68949986f), true, 1, 2);
    tree->SetNumericalTestNode(1, 5, "<", Value::Create<float>(6.5454998f), true, 3, 4);
    tree->SetNumericalTestNode(3, 9, "<", Value::Create<float>(207.5f), true, 7, 8);
    tree->SetNumericalTestNode(7, 11, "<", Value::Create<float>(377.880005f), true, 15, 16);
    tree->SetLeafNode(15, Value::Create<float>(0.200853109f));
    tree->SetLeafNode(16, Value::Create<float>(3.14392781f));
    tree->SetNumericalTestNode(8, 0, "<", Value::Create<float>(0.085769996f), true, 17, 18);
    tree->SetLeafNode(17, Value::Create<float>(-0.822109044f));
    tree->SetLeafNode(18, Value::Create<float>(0.266653359f));
    tree->SetNumericalTestNode(4, 5, "<", Value::Create<float>(6.56400013f), true, 9, 10);
    tree->SetLeafNode(9, Value::Create<float>(3.40855145f));
    tree->SetNumericalTestNode(10, 7, "<", Value::Create<float>(3.29480004f), true, 19, 20);
    tree->SetLeafNode(19, Value::Create<float>(2.98598123f));
    tree->SetLeafNode(20, Value::Create<float>(0.94572562f));
    tree->SetNumericalTestNode(2, 7, "<", Value::Create<float>(1.95050001f), true, 5, 6);
    tree->SetNumericalTestNode(5, 5, "<", Value::Create<float>(6.86849976f), true, 11, 12);
    tree->SetLeafNode(11, Value::Create<float>(-0.0353970528f));
    tree->SetNumericalTestNode(12, 0, "<", Value::Create<float>(0.943544984f), true, 21, 22);
    tree->SetLeafNode(21, Value::Create<float>(0.761680603f));
    tree->SetLeafNode(22, Value::Create<float>(3.02160382f));
    tree->SetNumericalTestNode(6, 6, "<", Value::Create<float>(50.75f), true, 13, 14);
    tree->SetNumericalTestNode(13, 10, "<", Value::Create<float>(15.5500002f), true, 23, 24);
    tree->SetLeafNode(23, Value::Create<float>(0.743751168f));
    tree->SetLeafNode(24, Value::Create<float>(-0.792990744f));
    tree->SetNumericalTestNode(14, 11, "<", Value::Create<float>(384.794983f), true, 25, 26);
    tree->SetLeafNode(25, Value::Create<float>(0.319963276f));
    tree->SetLeafNode(26, Value::Create<float>(-2.88059473f));
    builder->InsertTree(tree.get());
  }
  {
    std::unique_ptr<frontend::TreeBuilder> tree{
        new frontend::TreeBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32)
    };
    for (int nid = 0; nid <= 24; ++nid) {
      tree->CreateNode(nid);
    }
    tree->SetRootNode(0);
    tree->SetNumericalTestNode(0, 4, "<", Value::Create<float>(0.820500016f), true, 1, 2);
    tree->SetNumericalTestNode(1, 10, "<", Value::Create<float>(17.7000008f), true, 3, 4);
    tree->SetNumericalTestNode(3, 5, "<", Value::Create<float>(6.5255003f), true, 7, 8);
    tree->SetNumericalTestNode(7, 0, "<", Value::Create<float>(0.0687299967f), true, 15, 16);
    tree->SetLeafNode(15, Value::Create<float>(0.206869483f));
    tree->SetLeafNode(16, Value::Create<float>(1.80078018f));
    tree->SetNumericalTestNode(8, 12, "<", Value::Create<float>(3.14499998f), true, 17, 18);
    tree->SetLeafNode(17, Value::Create<float>(-0.923567116f));
    tree->SetLeafNode(18, Value::Create<float>(0.386075258f));
    tree->SetNumericalTestNode(4, 0, "<", Value::Create<float>(0.0301299989f), true, 9, 10);
    tree->SetNumericalTestNode(9, 8, "<", Value::Create<float>(3.5f), true, 19, 20);
    tree->SetLeafNode(19, Value::Create<float>(1.81692481f));
    tree->SetLeafNode(20, Value::Create<float>(0.130609035f));
    tree->SetNumericalTestNode(10, 4, "<", Value::Create<float>(0.708999991f), true, 21, 22);
    tree->SetLeafNode(21, Value::Create<float>(-0.330942363f));
    tree->SetLeafNode(22, Value::Create<float>(1.32937813f));
    tree->SetNumericalTestNode(2, 11, "<", Value::Create<float>(347.565002f), true, 5, 6);
    tree->SetNumericalTestNode(5, 6, "<", Value::Create<float>(97.25f), true, 11, 12);
    tree->SetLeafNode(11, Value::Create<float>(-3.44793344f));
    tree->SetLeafNode(12, Value::Create<float>(-1.2536478f));
    tree->SetNumericalTestNode(6, 0, "<", Value::Create<float>(2.34980488f), true, 13, 14);
    tree->SetNumericalTestNode(13, 0, "<", Value::Create<float>(1.54080999f), true, 23, 24);
    tree->SetLeafNode(23, Value::Create<float>(-0.342328072f));
    tree->SetLeafNode(24, Value::Create<float>(0.655293167f));
    tree->SetLeafNode(14, Value::Create<float>(-1.51396859f));
    builder->InsertTree(tree.get());
  }
  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());
}

TEST(ForwardCompatibility, TreeStump) {
  TypeInfo threshold_type = TypeInfo::kFloat32;
  TypeInfo leaf_output_type = TypeInfo::kFloat32;
  std::unique_ptr<frontend::ModelBuilder> builder{
      new frontend::ModelBuilder(2, 1, false, threshold_type, leaf_output_type)
  };
  constexpr std::size_t num_tree = 3;
  for (std::size_t i = 0; i < num_tree; ++i) {
    std::unique_ptr<frontend::TreeBuilder> tree{
        new frontend::TreeBuilder(threshold_type, leaf_output_type)
    };
    tree->CreateNode(0);
    tree->CreateNode(1);
    tree->CreateNode(2);
    tree->SetNumericalTestNode(0, 0, "<", frontend::Value::Create<float>(0), true, 1, 2);
    tree->SetRootNode(0);
    tree->SetLeafNode(1, frontend::Value::Create<float>(-1));
    tree->SetLeafNode(2, frontend::Value::Create<float>(1));
    builder->InsertTree(tree.get());
  }

  std::unique_ptr<Model> model = builder->CommitModel();

  std::vector<PyBufferFrame> frames = model->GetPyBuffer();
  std::vector<PyBufferFrame> new_frames;

  // Mapping to indicate where to insert new frames
  std::map<std::size_t, PyBufferFrame> frames_to_add;

  std::vector<int64_t> extra_opt_field1{1, 2, 3};
  std::vector<int32_t> extra_opt_field2{100, 200, 300};
  std::vector<double> extra_opt_field3{1.0, -1.0, 1.5};

  /* Locate the frames containing the extension slots */
  std::size_t num_opt_field_per_model_offset = 5 + 1 + 5;
  std::vector<std::size_t> num_opt_field_per_tree_offset{num_opt_field_per_model_offset + 9};
  std::vector<std::size_t> num_opt_field_per_node_offset{num_opt_field_per_tree_offset[0] + 1};
  for (std::size_t i = 1; i < num_tree; ++i) {
    num_opt_field_per_tree_offset.push_back(num_opt_field_per_tree_offset.back() + 10);
    num_opt_field_per_node_offset.push_back(num_opt_field_per_node_offset.back() + 10);
  }

  /* Insert new optional fields to the extension slots */
  frames_to_add[num_opt_field_per_model_offset] = PyBufferFrame{
    extra_opt_field1.data(), "=q",
    sizeof(decltype(extra_opt_field1)::value_type), extra_opt_field1.size()};
  for (std::size_t i : num_opt_field_per_tree_offset) {
    frames_to_add[i] = PyBufferFrame{
      extra_opt_field2.data(), "=l",
      sizeof(decltype(extra_opt_field2)::value_type), extra_opt_field2.size()};
  }
  for (std::size_t i : num_opt_field_per_node_offset) {
    frames_to_add[i] = PyBufferFrame{
      extra_opt_field3.data(), "=d",
      sizeof(decltype(extra_opt_field3)::value_type), extra_opt_field3.size()};
  }

  for (std::size_t i = 0; i < frames.size(); ++i) {
    if (frames_to_add.count(i) > 0) {
      // Increment count field by one
      PyBufferFrame new_cnt_frame = frames[i];
      ++(*static_cast<int32_t*>(new_cnt_frame.buf));
      new_frames.push_back(new_cnt_frame);
      // Insert new optional field
      new_frames.push_back(frames_to_add.at(i));
    } else {
      new_frames.push_back(frames[i]);
    }
  }

  // Ensure that the extra fields don't cause an error when deserializing
  std::unique_ptr<Model> received_model = Model::CreateFromPyBuffer(new_frames);
  ASSERT_TRUE(model->DumpAsJSON(false) == received_model->DumpAsJSON(false));
}

}  // namespace treelite
