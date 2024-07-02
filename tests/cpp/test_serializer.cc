/*!
 * Copyright (c) 2020 by Contributors
 * \file test_serializer.cc
 * \author Hyunsu Cho
 * \brief C++ tests for model serializer
 */
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>
#include <treelite/detail/file_utils.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/error.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

using namespace fmt::literals;

namespace {

inline void TestRoundTrip(treelite::Model* model) {
  for (int i = 0; i < 2; ++i) {
    // Test round trip with in-memory serialization
    auto buffer = model->SerializeToPyBuffer();
    std::unique_ptr<treelite::Model> received_model
        = treelite::Model::DeserializeFromPyBuffer(buffer);

    // Use ASSERT_TRUE, since ASSERT_EQ will dump all the raw bytes into a string, potentially
    // causing an OOM error
    ASSERT_TRUE(model->DumpAsJSON(false) == received_model->DumpAsJSON(false));
  }

  for (int i = 0; i < 2; ++i) {
    // Test round trip with in-memory serialization (via string)
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

  for (int i = 0; i < 2; ++i) {
    // Test round trip with serialization to a file stream
    std::filesystem::path tmpdir = std::filesystem::temp_directory_path();
    std::filesystem::path filepath = tmpdir / (std::string("binary") + std::to_string(i) + ".bin");
    std::unique_ptr<treelite::Model> received_model;
    {
      std::ofstream ofs = treelite::detail::OpenFileForWriteAsStream(filepath);
      model->SerializeToStream(ofs);
    }
    {
      std::ifstream ifs = treelite::detail::OpenFileForReadAsStream(filepath);
      received_model = treelite::Model::DeserializeFromStream(ifs);
    }

    // Use ASSERT_TRUE, since ASSERT_EQ will dump all the raw bytes into a string, potentially
    // causing an OOM error
    ASSERT_TRUE(model->DumpAsJSON(false) == received_model->DumpAsJSON(false));

    std::filesystem::remove(filepath);
  }
}

}  // anonymous namespace

namespace treelite {

template <typename ThresholdType, typename LeafOutputType>
void SerializerRoundTrip_TreeStump() {
  TypeInfo threshold_type = TypeInfoFromType<ThresholdType>();
  TypeInfo leaf_output_type = TypeInfoFromType<LeafOutputType>();
  model_builder::Metadata metadata{2, TaskType::kRegressor, false, 1, {1}, {1, 1}};
  std::unique_ptr<model_builder::ModelBuilder> builder = model_builder::GetModelBuilder(
      threshold_type, leaf_output_type, metadata, model_builder::TreeAnnotation{1, {0}, {0}},
      model_builder::PostProcessorFunc{"identity"}, {0.0});
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

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());

  /* Test correctness of JSON dump */
  std::string expected_json_dump_str = fmt::format(R"JSON(
  {{
    "threshold_type": "{threshold_type}",
    "leaf_output_type": "{leaf_output_type}",
    "num_feature": 2,
    "task_type": "kRegressor",
    "average_tree_output": false,
    "num_target": 1,
    "num_class": [1],
    "leaf_vector_shape": [1, 1],
    "target_id": [0],
    "class_id": [0],
    "postprocessor": "identity",
    "sigmoid_alpha": 1.0,
    "ratio_c": 1.0,
    "base_scores": [0.0],
    "attributes": "{{}}",
    "trees": [{{
            "num_nodes": 3,
            "has_categorical_split": false,
            "nodes": [{{
                    "node_id": 0,
                    "split_feature_id": 0,
                    "default_left": true,
                    "node_type": "numerical_test_node",
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
      "leaf_value1"_a = static_cast<LeafOutputType>(2),
      "threshold_type"_a = TypeInfoToString(TypeInfoFromType<ThresholdType>()),
      "leaf_output_type"_a = TypeInfoToString(TypeInfoFromType<LeafOutputType>()));

  rapidjson::Document json_dump;
  json_dump.Parse(model->DumpAsJSON(false).c_str());

  rapidjson::Document expected_json_dump;
  expected_json_dump.Parse(expected_json_dump_str.c_str());
  EXPECT_TRUE(json_dump == expected_json_dump);
}

TEST(SerializerRoundTrip, TreeStump) {
  SerializerRoundTrip_TreeStump<float, float>();
  SerializerRoundTrip_TreeStump<double, double>();
  ASSERT_THROW((SerializerRoundTrip_TreeStump<float, double>()), treelite::Error);
  ASSERT_THROW((SerializerRoundTrip_TreeStump<double, float>()), treelite::Error);
  ASSERT_THROW((SerializerRoundTrip_TreeStump<std::uint32_t, float>()), treelite::Error);
  ASSERT_THROW((SerializerRoundTrip_TreeStump<std::uint32_t, double>()), treelite::Error);
}

template <typename ThresholdType, typename LeafOutputType>
void SerializerRoundTrip_TreeStumpLeafVec() {
  TypeInfo threshold_type = TypeInfoFromType<ThresholdType>();
  TypeInfo leaf_output_type = TypeInfoFromType<LeafOutputType>();
  model_builder::Metadata metadata{2, TaskType::kMultiClf, true, 1, {2}, {1, 2}};
  model_builder::TreeAnnotation tree_annotation{1, {0}, {-1}};
  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::GetModelBuilder(threshold_type, leaf_output_type, metadata, tree_annotation,
          model_builder::PostProcessorFunc{"identity"}, {0.0, 0.0});
  builder->StartTree();
  builder->StartNode(0);
  builder->NumericalTest(0, 0.0, true, Operator::kLT, 1, 2);
  builder->EndNode();
  builder->StartNode(1);
  builder->LeafVector(
      std::vector<LeafOutputType>{static_cast<LeafOutputType>(1), static_cast<LeafOutputType>(2)});
  builder->EndNode();
  builder->StartNode(2);
  builder->LeafVector(
      std::vector<LeafOutputType>{static_cast<LeafOutputType>(2), static_cast<LeafOutputType>(1)});
  builder->EndNode();
  builder->EndTree();

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());

  /* Test correctness of JSON dump */
  std::string expected_json_dump_str = fmt::format(R"JSON(
  {{
    "threshold_type": "{threshold_type}",
    "leaf_output_type": "{leaf_output_type}",
    "num_feature": 2,
    "task_type": "kMultiClf",
    "average_tree_output": true,
    "num_target": 1,
    "num_class": [2],
    "leaf_vector_shape": [1, 2],
    "target_id": [0],
    "class_id": [-1],
    "postprocessor": "identity",
    "sigmoid_alpha": 1.0,
    "ratio_c": 1.0,
    "base_scores": [0.0, 0.0],
    "attributes": "{{}}",
    "trees": [{{
            "num_nodes": 3,
            "has_categorical_split": false,
            "nodes": [{{
                    "node_id": 0,
                    "split_feature_id": 0,
                    "default_left": true,
                    "node_type": "numerical_test_node",
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
      "leaf_value3"_a = static_cast<LeafOutputType>(1),
      "threshold_type"_a = TypeInfoToString(TypeInfoFromType<ThresholdType>()),
      "leaf_output_type"_a = TypeInfoToString(TypeInfoFromType<LeafOutputType>()));
  rapidjson::Document json_dump;
  json_dump.Parse(model->DumpAsJSON(false).c_str());

  rapidjson::Document expected_json_dump;
  expected_json_dump.Parse(expected_json_dump_str.c_str());
  EXPECT_TRUE(json_dump == expected_json_dump);
}

TEST(SerializerRoundTrip, TreeStumpLeafVec) {
  SerializerRoundTrip_TreeStumpLeafVec<float, float>();
  SerializerRoundTrip_TreeStumpLeafVec<double, double>();
  ASSERT_THROW((SerializerRoundTrip_TreeStumpLeafVec<float, double>()), treelite::Error);
  ASSERT_THROW((SerializerRoundTrip_TreeStumpLeafVec<double, float>()), treelite::Error);
  ASSERT_THROW((SerializerRoundTrip_TreeStumpLeafVec<std::uint32_t, float>()), treelite::Error);
  ASSERT_THROW((SerializerRoundTrip_TreeStumpLeafVec<std::uint32_t, double>()), treelite::Error);
}

template <typename ThresholdType, typename LeafOutputType>
void SerializerRoundTrip_TreeStumpCategoricalSplit(
    std::vector<std::uint32_t> const& left_categories) {
  TypeInfo threshold_type = TypeInfoFromType<ThresholdType>();
  TypeInfo leaf_output_type = TypeInfoFromType<LeafOutputType>();
  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::GetModelBuilder(threshold_type, leaf_output_type,
          model_builder::Metadata{2, TaskType::kRegressor, false, 1, {1}, {1, 1}},
          model_builder::TreeAnnotation{1, {0}, {0}}, model_builder::PostProcessorFunc{"identity"},
          {0.0});
  builder->StartTree();
  builder->StartNode(0);
  builder->CategoricalTest(0, false, left_categories, false, 1, 2);
  builder->EndNode();
  builder->StartNode(1);
  builder->LeafScalar(2);
  builder->EndNode();
  builder->StartNode(2);
  builder->LeafScalar(3);
  builder->EndNode();
  builder->EndTree();

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());

  /* Test correctness of JSON dump */
  std::string category_list_str;
  {
    std::ostringstream oss;
    rapidjson::OStreamWrapper os_wrapper(oss);
    rapidjson::Writer<rapidjson::OStreamWrapper> writer(os_wrapper);
    writer.StartArray();
    for (auto e : left_categories) {
      writer.Uint(static_cast<unsigned int>(e));
    }
    writer.EndArray();
    category_list_str = oss.str();
  }
  std::string expected_json_dump_str = fmt::format(R"JSON(
  {{
    "threshold_type": "{threshold_type}",
    "leaf_output_type": "{leaf_output_type}",
    "num_feature": 2,
    "task_type": "kRegressor",
    "average_tree_output": false,
    "num_target": 1,
    "num_class": [1],
    "leaf_vector_shape": [1, 1],
    "target_id": [0],
    "class_id": [0],
    "postprocessor": "identity",
    "sigmoid_alpha": 1.0,
    "ratio_c": 1.0,
    "base_scores": [0.0],
    "attributes": "{{}}",
    "trees": [{{
            "num_nodes": 3,
            "has_categorical_split": true,
            "nodes": [{{
                    "node_id": 0,
                    "split_feature_id": 0,
                    "default_left": false,
                    "node_type": "categorical_test_node",
                    "category_list_right_child": false,
                    "category_list": {category_list},
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
      "leaf_value1"_a = static_cast<LeafOutputType>(3), "category_list"_a = category_list_str,
      "threshold_type"_a = TypeInfoToString(TypeInfoFromType<ThresholdType>()),
      "leaf_output_type"_a = TypeInfoToString(TypeInfoFromType<LeafOutputType>()));

  rapidjson::Document json_dump;
  json_dump.Parse(model->DumpAsJSON(false).c_str());

  rapidjson::Document expected_json_dump;
  expected_json_dump.Parse(expected_json_dump_str.c_str());
  EXPECT_TRUE(json_dump == expected_json_dump);
}

TEST(SerializerRoundTrip, TreeStumpCategoricalSplit) {
  for (auto const& left_categories : std::vector<std::vector<std::uint32_t>>{{}, {1}, {0, 1}}) {
    SerializerRoundTrip_TreeStumpCategoricalSplit<float, float>(left_categories);
    SerializerRoundTrip_TreeStumpCategoricalSplit<double, double>(left_categories);
    ASSERT_THROW((SerializerRoundTrip_TreeStumpCategoricalSplit<float, double>(left_categories)),
        treelite::Error);
    ASSERT_THROW((SerializerRoundTrip_TreeStumpCategoricalSplit<double, float>(left_categories)),
        treelite::Error);
    ASSERT_THROW((SerializerRoundTrip_TreeStumpCategoricalSplit<std::uint32_t, std::uint32_t>(
                     left_categories)),
        treelite::Error);
    ASSERT_THROW(
        (SerializerRoundTrip_TreeStumpCategoricalSplit<std::uint32_t, float>(left_categories)),
        treelite::Error);
    ASSERT_THROW(
        (SerializerRoundTrip_TreeStumpCategoricalSplit<std::uint32_t, double>(left_categories)),
        treelite::Error);
  }
}

template <typename ThresholdType, typename LeafOutputType>
void SerializerRoundTrip_TreeDepth2() {
  TypeInfo threshold_type = TypeInfoFromType<ThresholdType>();
  TypeInfo leaf_output_type = TypeInfoFromType<LeafOutputType>();
  int const num_tree = 3;
  auto builder = model_builder::GetModelBuilder(threshold_type, leaf_output_type,
      model_builder::Metadata{2, TaskType::kBinaryClf, false, 1, {1}, {1, 1}},
      model_builder::TreeAnnotation{num_tree, {0, 0, 0}, {0, 0, 0}},
      model_builder::PostProcessorFunc{"sigmoid"}, {0.5});
  for (int tree_id = 0; tree_id < num_tree; ++tree_id) {
    builder->StartTree();
    builder->StartNode(0);
    builder->NumericalTest(0, 0.0, true, Operator::kLT, 1, 2);
    builder->EndNode();
    builder->StartNode(1);
    builder->CategoricalTest(0, true, {0, 1}, false, 3, 4);
    builder->EndNode();
    builder->StartNode(2);
    builder->CategoricalTest(1, true, {0}, false, 5, 6);
    builder->EndNode();
    builder->StartNode(3);
    builder->LeafScalar(tree_id + 3);
    builder->EndNode();
    builder->StartNode(4);
    builder->LeafScalar(tree_id + 1);
    builder->EndNode();
    builder->StartNode(5);
    builder->LeafScalar(tree_id + 4);
    builder->EndNode();
    builder->StartNode(6);
    builder->LeafScalar(tree_id + 2);
    builder->EndNode();
    builder->EndTree();
  }

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());

  std::string expected_json_dump_str = fmt::format(R"JSON(
  {{
    "threshold_type": "{threshold_type}",
    "leaf_output_type": "{leaf_output_type}",
    "num_feature": 2,
    "task_type": "kBinaryClf",
    "average_tree_output": false,
    "num_target": 1,
    "num_class": [1],
    "leaf_vector_shape": [1, 1],
    "target_id": [0, 0, 0],
    "class_id": [0, 0, 0],
    "postprocessor": "sigmoid",
    "sigmoid_alpha": 1.0,
    "ratio_c": 1.0,
    "base_scores": [0.5],
    "attributes": "{{}}",
    "trees": [{{
            "num_nodes": 7,
            "has_categorical_split": true,
            "nodes": [{{
                    "node_id": 0,
                    "split_feature_id": 0,
                    "default_left": true,
                    "node_type": "numerical_test_node",
                    "comparison_op": "<",
                    "threshold": {threshold},
                    "left_child": 1,
                    "right_child": 2
                }}, {{
                    "node_id": 1,
                    "split_feature_id": 0,
                    "default_left": true,
                    "node_type": "categorical_test_node",
                    "category_list_right_child": false,
                    "category_list": [0, 1],
                    "left_child": 3,
                    "right_child": 4
                }}, {{
                    "node_id": 2,
                    "split_feature_id": 1,
                    "default_left": true,
                    "node_type": "categorical_test_node",
                    "category_list_right_child": false,
                    "category_list": [0],
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
                    "node_type": "numerical_test_node",
                    "comparison_op": "<",
                    "threshold": {threshold},
                    "left_child": 1,
                    "right_child": 2
                }}, {{
                    "node_id": 1,
                    "split_feature_id": 0,
                    "default_left": true,
                    "node_type": "categorical_test_node",
                    "category_list_right_child": false,
                    "category_list": [0, 1],
                    "left_child": 3,
                    "right_child": 4
                }}, {{
                    "node_id": 2,
                    "split_feature_id": 1,
                    "default_left": true,
                    "node_type": "categorical_test_node",
                    "category_list_right_child": false,
                    "category_list": [0],
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
                    "node_type": "numerical_test_node",
                    "comparison_op": "<",
                    "threshold": {threshold},
                    "left_child": 1,
                    "right_child": 2
                }}, {{
                    "node_id": 1,
                    "split_feature_id": 0,
                    "default_left": true,
                    "node_type": "categorical_test_node",
                    "category_list_right_child": false,
                    "category_list": [0, 1],
                    "left_child": 3,
                    "right_child": 4
                }}, {{
                    "node_id": 2,
                    "split_feature_id": 1,
                    "default_left": true,
                    "node_type": "categorical_test_node",
                    "category_list_right_child": false,
                    "category_list": [0],
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
      "tree2_leaf6"_a = static_cast<LeafOutputType>(2 + 2),
      "threshold_type"_a = TypeInfoToString(TypeInfoFromType<ThresholdType>()),
      "leaf_output_type"_a = TypeInfoToString(TypeInfoFromType<LeafOutputType>()));

  rapidjson::Document json_dump;
  json_dump.Parse(model->DumpAsJSON(false).c_str());

  rapidjson::Document expected_json_dump;
  expected_json_dump.Parse(expected_json_dump_str.c_str());
  EXPECT_TRUE(json_dump == expected_json_dump);
}

TEST(SerializerRoundTrip, TreeDepth2) {
  SerializerRoundTrip_TreeDepth2<float, float>();
  SerializerRoundTrip_TreeDepth2<double, double>();
  ASSERT_THROW((SerializerRoundTrip_TreeDepth2<float, double>()), treelite::Error);
  ASSERT_THROW((SerializerRoundTrip_TreeDepth2<double, float>()), treelite::Error);
  ASSERT_THROW((SerializerRoundTrip_TreeDepth2<std::uint32_t, std::uint32_t>()), treelite::Error);
  ASSERT_THROW((SerializerRoundTrip_TreeDepth2<std::uint32_t, float>()), treelite::Error);
  ASSERT_THROW((SerializerRoundTrip_TreeDepth2<std::uint32_t, double>()), treelite::Error);
}

template <typename ThresholdType, typename LeafOutputType>
void SerializerRoundTrip_DeepFullTree() {
  TypeInfo threshold_type = TypeInfoFromType<ThresholdType>();
  TypeInfo leaf_output_type = TypeInfoFromType<LeafOutputType>();
  int const depth = 12;
  int const num_tree = 3;
  auto builder = model_builder::GetModelBuilder(threshold_type, leaf_output_type,
      model_builder::Metadata{3, TaskType::kBinaryClf, false, 1, {1}, {1, 1}},
      model_builder::TreeAnnotation{num_tree, {0, 0, 0}, {0, 0, 0}},
      model_builder::PostProcessorFunc{"identity"}, {0.0});
  for (int tree_id = 0; tree_id < num_tree; ++tree_id) {
    builder->StartTree();
    for (int level = 0; level <= depth; ++level) {
      for (int i = 0; i < (1 << level); ++i) {
        int const nid = (1 << level) - 1 + i;
        builder->StartNode(nid);
        if (level == depth) {
          builder->LeafScalar(tree_id + 1);
        } else {
          builder->NumericalTest((level % 2), 0.0, true, Operator::kLT, 2 * nid + 1, 2 * nid + 2);
        }
        builder->EndNode();
      }
    }
    builder->EndTree();
  }

  std::unique_ptr<Model> model = builder->CommitModel();
  TestRoundTrip(model.get());
}

TEST(SerializerRoundTrip, DeepFullTree) {
  SerializerRoundTrip_DeepFullTree<float, float>();
  SerializerRoundTrip_DeepFullTree<double, double>();
}

TEST(SerializerForwardCompatibility, TreeStump) {
  constexpr int num_tree = 3;
  auto builder = model_builder::GetModelBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32,
      model_builder::Metadata{2, TaskType::kBinaryClf, false, 1, {1}, {1, 1}},
      model_builder::TreeAnnotation{num_tree, {0, 0, 0}, {0, 0, 0}},
      model_builder::PostProcessorFunc{"sigmoid", {{"sigmoid_alpha", 2.0}}}, {0.0});

  for (int i = 0; i < num_tree; ++i) {
    builder->StartTree();
    builder->StartNode(0);
    builder->NumericalTest(0, 0.0, true, Operator::kLT, 1, 2);
    builder->EndNode();
    builder->StartNode(1);
    builder->LeafScalar(-1.0);
    builder->EndNode();
    builder->StartNode(2);
    builder->LeafScalar(1.0);
    builder->EndNode();
    builder->EndTree();
  }

  std::unique_ptr<Model> model = builder->CommitModel();

  std::vector<PyBufferFrame> frames = model->SerializeToPyBuffer();
  std::vector<PyBufferFrame> new_frames;

  // Mapping to indicate where to insert new frames
  std::map<std::size_t, std::pair<std::string, PyBufferFrame>> frames_to_add;

  std::vector<std::int64_t> extra_opt_field1{1, 2, 3};
  std::vector<std::int32_t> extra_opt_field2{100, 200, 300};
  std::vector<double> extra_opt_field3{1.0, -1.0, 1.5};

  /* Locate the frames containing the extension slots */
  std::size_t num_opt_field_per_model_offset = 5 + 1 + 13;
  std::vector<std::size_t> num_opt_field_per_tree_offset{num_opt_field_per_model_offset + 24};
  std::vector<std::size_t> num_opt_field_per_node_offset{num_opt_field_per_tree_offset[0] + 1};
  for (std::size_t i = 1; i < num_tree; ++i) {
    num_opt_field_per_tree_offset.push_back(num_opt_field_per_tree_offset.back() + 25);
    num_opt_field_per_node_offset.push_back(num_opt_field_per_node_offset.back() + 25);
  }

  /* Insert new optional fields to the extension slots */
  frames_to_add[num_opt_field_per_model_offset] = {"extra_opt_field1",
      {extra_opt_field1.data(), const_cast<char*>("=q"),
          sizeof(decltype(extra_opt_field1)::value_type), extra_opt_field1.size()}};
  for (std::size_t i : num_opt_field_per_tree_offset) {
    frames_to_add[i] = {"extra_opt_field2",
        {extra_opt_field2.data(), const_cast<char*>("=l"),
            sizeof(decltype(extra_opt_field2)::value_type), extra_opt_field2.size()}};
  }
  for (std::size_t i : num_opt_field_per_node_offset) {
    frames_to_add[i] = {"extra_opt_field3",
        {extra_opt_field3.data(), const_cast<char*>("=d"),
            sizeof(decltype(extra_opt_field3)::value_type), extra_opt_field3.size()}};
  }

  for (std::size_t i = 0; i < frames.size(); ++i) {
    if (frames_to_add.count(i) > 0) {
      // Increment count field by one
      PyBufferFrame new_cnt_frame = frames[i];
      ++(*static_cast<std::int32_t*>(new_cnt_frame.buf));
      new_frames.push_back(new_cnt_frame);
      // Insert new optional field
      auto [name, content] = frames_to_add.at(i);
      new_frames.push_back({name.data(), const_cast<char*>("=c"), sizeof(char), name.length()});
      new_frames.push_back(content);
    } else {
      new_frames.push_back(frames[i]);
    }
  }

  // Ensure that the extra fields don't cause an error when deserializing
  std::unique_ptr<Model> received_model = Model::DeserializeFromPyBuffer(new_frames);
  ASSERT_TRUE(model->DumpAsJSON(false) == received_model->DumpAsJSON(false));
}

}  // namespace treelite
