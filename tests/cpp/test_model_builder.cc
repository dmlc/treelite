/*!
 * Copyright (c) 2023 by Contributors
 * \file test_model_builder.cc
 * \author Hyunsu Cho
 * \brief C++ tests for GTIL
 */

#include <gtest/gtest.h>
#include <model_builder/detail/json_parsing.h>
#include <rapidjson/document.h>
#include <treelite/detail/threading_utils.h>
#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

void AssertDocumentValid(rapidjson::Document const& doc) {
  EXPECT_FALSE(doc.HasParseError())
      << "Error when parsing JSON string: offset " << doc.GetErrorOffset() << ", "
      << rapidjson::GetParseError_En(doc.GetParseError());
}

void AssertJSONStringsEqual(std::string const& actual, std::string const& expected) {
  rapidjson::Document actual_parsed, expected_parsed;
  actual_parsed.Parse(actual);
  AssertDocumentValid(actual_parsed);
  expected_parsed.Parse(expected);
  AssertDocumentValid(expected_parsed);
  EXPECT_TRUE(actual_parsed == expected_parsed) << "Expected: " << expected << "\n"
                                                << "Got: " << actual;
}

}  // anonymous namespace

namespace treelite {

TEST(ModelBuilder, OrphanedNodes) {
  model_builder::Metadata metadata{1, TaskType::kBinaryClf, false, 1, {1}, {1, 1}};
  model_builder::TreeAnnotation tree_annotation{1, {0}, {0}};
  model_builder::PostProcessorFunc postprocessor{"sigmoid"};
  std::vector<double> base_scores{0.0};
  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::GetModelBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32, metadata,
          tree_annotation, postprocessor, base_scores);
  builder->StartTree();
  builder->StartNode(0);
  builder->LeafScalar(0.0);
  builder->EndNode();
  builder->StartNode(1);
  builder->LeafScalar(1.0);
  builder->EndNode();
  EXPECT_THROW(builder->EndTree(), Error);
}

TEST(ModelBuilder, InvalidNodeID) {
  model_builder::Metadata metadata{1, TaskType::kBinaryClf, false, 1, {1}, {1, 1}};
  model_builder::TreeAnnotation tree_annotation{1, {0}, {0}};
  model_builder::PostProcessorFunc postprocessor{"sigmoid"};
  std::vector<double> base_scores{0.0};
  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::GetModelBuilder(TypeInfo::kFloat32, TypeInfo::kFloat32, metadata,
          tree_annotation, postprocessor, base_scores);
  builder->StartTree();
  EXPECT_THROW(builder->StartNode(-1), Error);
  builder->StartNode(0);
  EXPECT_THROW(builder->NumericalTest(0, 0.0, true, Operator::kLT, 0, 1), Error);
  EXPECT_THROW(builder->NumericalTest(0, 0.0, true, Operator::kLT, 2, 2), Error);
  EXPECT_THROW(builder->NumericalTest(0, 0.0, true, Operator::kLT, -1, -2), Error);
  EXPECT_THROW(builder->NumericalTest(0, 0.0, true, Operator::kLT, -1, 2), Error);
  EXPECT_THROW(builder->NumericalTest(0, 0.0, true, Operator::kLT, 2, -1), Error);
}

TEST(ModelBuilder, InvalidState) {
  std::unique_ptr<model_builder::ModelBuilder> builder = model_builder::GetModelBuilder(R"(
    {
      "threshold_type": "float32",
      "leaf_output_type": "float32",
      "metadata": {
        "num_feature": 1,
        "task_type": "kMultiClf",
        "average_tree_output": false,
        "num_target": 1,
        "num_class": [2],
        "leaf_vector_shape": [1, 2]
      },
      "tree_annotation": {
        "num_tree": 1,
        "target_id": [0],
        "class_id": [-1]
      },
      "postprocessor": {
        "name": "identity_multiclass"
      },
      "base_scores": [0.0, 0.0]
    }
  )");
  builder->StartTree();
  EXPECT_THROW(builder->StartTree(), Error);
  EXPECT_THROW(builder->Gain(0.0), Error);
  EXPECT_THROW(builder->NumericalTest(0, 0.0, false, Operator::kLT, 1, 2), Error);
  EXPECT_THROW(builder->EndNode(), Error);
  EXPECT_THROW(builder->EndTree(), Error);  // Cannot have an empty tree with 0 nodes
  EXPECT_THROW(builder->CommitModel(), Error);

  builder->StartNode(0);
  EXPECT_THROW(builder->StartTree(), Error);
  EXPECT_THROW(builder->StartNode(1), Error);
  EXPECT_THROW(builder->EndNode(), Error);  // Cannot have an empty node
  EXPECT_THROW(builder->EndTree(), Error);
  EXPECT_THROW(builder->CommitModel(), Error);

  builder->Gain(0.0);
  builder->NumericalTest(0, 0.0, false, Operator::kLT, 1, 2);
  EXPECT_THROW(builder->StartTree(), Error);
  EXPECT_THROW(builder->StartNode(2), Error);
  EXPECT_THROW(builder->EndTree(), Error);
  EXPECT_THROW(builder->CommitModel(), Error);
  EXPECT_THROW(builder->LeafScalar(0.0), Error);  // Cannot change node kind once specified
  EXPECT_THROW(builder->NumericalTest(0, 0.0, false, Operator::kLT, 1, 2), Error);

  builder->Gain(0.0);
  builder->EndNode();
  EXPECT_THROW(builder->StartTree(), Error);
  EXPECT_THROW(builder->Gain(0.0), Error);
  EXPECT_THROW(builder->LeafVector(std::vector<float>{0.0, 1.0}), Error);
  EXPECT_THROW(builder->EndNode(), Error);
  EXPECT_THROW(builder->CommitModel(), Error);
  EXPECT_THROW(builder->EndTree(), Error);  // Did not yet specify nodes 1 and 2

  builder->StartNode(1);
  EXPECT_THROW(builder->LeafScalar(-1.0), Error);  // Wrong leaf shape
  EXPECT_THROW(builder->LeafVector(std::vector<float>{0.0, 1.0, 2.0}), Error);  // Wrong leaf shape
  builder->LeafVector(std::vector<float>{0.0, 1.0});
  builder->EndNode();

  builder->StartNode(2);
  builder->LeafVector(std::vector<float>{1.0, 0.0});
  builder->EndNode();
  builder->EndTree();
  auto model = builder->CommitModel();
  model->DumpAsJSON(true);

  EXPECT_THROW(builder->StartTree(), Error);
  EXPECT_THROW(builder->StartNode(3), Error);
  EXPECT_THROW(builder->Gain(1.0), Error);
  EXPECT_THROW(builder->LeafVector(std::vector<float>{0.5, 0.5}), Error);
  EXPECT_THROW(builder->EndNode(), Error);
  EXPECT_THROW(builder->EndTree(), Error);
  EXPECT_THROW(builder->CommitModel(), Error);
}

TEST(ModelBuilder, NodeMapping) {
  model_builder::Metadata metadata{1, TaskType::kBinaryClf, false, 1, {1}, {1, 1}};
  model_builder::TreeAnnotation tree_annotation{1, {0}, {0}};
  model_builder::PostProcessorFunc postprocessor{"sigmoid"};
  std::vector<double> base_scores{0.0};

  int const n_trial = 10;
  std::vector<std::string> dump(n_trial);
  detail::threading_utils::ThreadConfig config(-1);
  detail::threading_utils::ParallelFor(
      0, n_trial, config, detail::threading_utils::ParallelSchedule::Static(), [&](int i, int) {
        std::unique_ptr<model_builder::ModelBuilder> builder
            = model_builder::GetModelBuilder(TypeInfo::kFloat64, TypeInfo::kFloat64, metadata,
                tree_annotation, postprocessor, base_scores);
        builder->StartTree();
        builder->StartNode(0 + i * 2);
        builder->NumericalTest(0, 0.0, false, Operator::kLT, 1 + i * 2, 2 + i * 2);
        builder->EndNode();
        builder->StartNode(1 + i * 2);
        builder->LeafScalar(-1.0);
        builder->EndNode();
        builder->StartNode(2 + i * 2);
        builder->LeafScalar(1.0);
        builder->EndNode();
        builder->EndTree();
        std::unique_ptr<Model> model = builder->CommitModel();
        dump[i] = model->DumpAsJSON(true);
      });
  detail::threading_utils::ParallelFor(1, n_trial, config,
      detail::threading_utils::ParallelSchedule::Static(),
      [&](int i, int) { TREELITE_CHECK_EQ(dump[0], dump[i]); });
}

TEST(ModelBuilderJSONParsing, Metadata) {
  std::string const json_str = R"(
    {
      "metadata": {
        "num_feature": 4,
        "task_type": "kBinaryClf",
        "average_tree_output": false,
        "num_target": 1,
        "num_class": [1],
        "leaf_vector_shape": [1, 1]
      }
    }
  )";
  rapidjson::Document parsed_json;
  parsed_json.Parse(json_str);
  AssertDocumentValid(parsed_json);

  std::vector<std::int32_t> const expected_num_class{1};
  std::array<std::int32_t, 2> const expected_leaf_vector_shape{1, 1};

  auto metadata = model_builder::detail::json_parse::ParseMetadata(parsed_json, "metadata");
  EXPECT_EQ(metadata.num_feature, 4);
  EXPECT_EQ(metadata.task_type, TaskType::kBinaryClf);
  EXPECT_FALSE(metadata.average_tree_output);
  EXPECT_EQ(metadata.num_target, 1);
  EXPECT_EQ(metadata.num_class, expected_num_class);
  EXPECT_EQ(metadata.leaf_vector_shape, expected_leaf_vector_shape);
}

TEST(ModelBuilderJSONParsing, TreeAnnotation) {
  std::string const json_str = R"(
    {
      "tree_annotation": {
        "num_tree": 2,
        "target_id": [0, 0],
        "class_id": [0, 1]
      }
    }
  )";
  rapidjson::Document parsed_json;
  parsed_json.Parse(json_str);
  AssertDocumentValid(parsed_json);

  std::vector<std::int32_t> const expected_target_id{0, 0};
  std::vector<std::int32_t> const expected_class_id{0, 1};

  auto tree_annotation
      = model_builder::detail::json_parse::ParseTreeAnnotation(parsed_json, "tree_annotation");
  EXPECT_EQ(tree_annotation.target_id, expected_target_id);
  EXPECT_EQ(tree_annotation.class_id, expected_class_id);
}

TEST(ModelBuilderJSONParsing, PostProcessorFunc) {
  std::string const json_str = R"(
    {
      "postprocessor": {
        "name": "sigmoid",
        "config": {
           "sigmoid_alpha": 2.0
        }
      }
    }
  )";
  rapidjson::Document parsed_json;
  parsed_json.Parse(json_str);
  AssertDocumentValid(parsed_json);

  auto postprocessor
      = model_builder::detail::json_parse::ParsePostProcessorFunc(parsed_json, "postprocessor");
  EXPECT_EQ(postprocessor.name, "sigmoid");

  std::map<std::string, treelite::model_builder::PostProcessorConfigParam> const
      expected_postprocessor_config{{"sigmoid_alpha", 2.0}};
  EXPECT_EQ(postprocessor.config, expected_postprocessor_config);
}

TEST(ModelBuilderJSONParsing, Attributes) {
  std::string const json_str = R"(
    {
      "attributes": {
        "foo": "bar",
        "cake": 2,
        "piece": 3.0
      }
    }
  )";
  rapidjson::Document parsed_json;
  parsed_json.Parse(json_str);
  AssertDocumentValid(parsed_json);

  auto attributes = model_builder::detail::json_parse::ParseAttributes(parsed_json, "attributes");
  EXPECT_TRUE(attributes.has_value());

  std::string const expected_attributes_str = R"(
    {
      "foo": "bar",
      "cake": 2,
      "piece": 3.0
    }
  )";
  AssertJSONStringsEqual(attributes.value(), expected_attributes_str);
}

TEST(ModelBuilderJSONParsing, AttributesEmpty) {
  {
    std::string const json_str = R"(
      {
        "attributes": {}
      }
    )";
    rapidjson::Document parsed_json;
    parsed_json.Parse(json_str);
    AssertDocumentValid(parsed_json);

    auto attributes = model_builder::detail::json_parse::ParseAttributes(parsed_json, "attributes");
    EXPECT_TRUE(attributes.has_value());
    std::string const expected_attributes_str = "{}";
    AssertJSONStringsEqual(attributes.value(), expected_attributes_str);
  }
  {
    std::string const json_str = "{}";
    rapidjson::Document parsed_json;
    parsed_json.Parse(json_str);
    AssertDocumentValid(parsed_json);

    auto attributes = model_builder::detail::json_parse::ParseAttributes(parsed_json, "attributes");
    EXPECT_FALSE(attributes.has_value());
  }
}

TEST(ModelBuilderJSONParsing, Combined) {
  std::string const json_str = R"(
    {
      "threshold_type": "float32",
      "leaf_output_type": "float32",
      "metadata": {
        "num_feature": 4,
        "task_type": "kBinaryClf",
        "average_tree_output": false,
        "num_target": 1,
        "num_class": [1],
        "leaf_vector_shape": [1, 1]
      },
      "tree_annotation": {
        "num_tree": 2,
        "target_id": [0, 0],
        "class_id": [0, 1]
      },
      "postprocessor": {
        "name": "sigmoid",
        "config": {
           "sigmoid_alpha": 2.0
        }
      },
      "base_scores": [0.5],
      "attributes": {
        "foo": "bar",
        "cake": 2,
        "piece": 3.0
      }
    }
  )";
  rapidjson::Document parsed_json;
  parsed_json.Parse(json_str);
  AssertDocumentValid(parsed_json);

  namespace json_parse = model_builder::detail::json_parse;

  auto const threshold_type = TypeInfoFromString(
      json_parse::ObjectMemberHandler<std::string>::Get(parsed_json, "threshold_type"));
  auto const leaf_output_type = TypeInfoFromString(
      json_parse::ObjectMemberHandler<std::string>::Get(parsed_json, "leaf_output_type"));
  auto const metadata = json_parse::ParseMetadata(parsed_json, "metadata");
  auto const tree_annotation = json_parse::ParseTreeAnnotation(parsed_json, "tree_annotation");
  auto const postprocessor = json_parse::ParsePostProcessorFunc(parsed_json, "postprocessor");
  auto const base_scores
      = json_parse::ObjectMemberHandler<std::vector<double>>::Get(parsed_json, "base_scores");
  auto const attributes = json_parse::ParseAttributes(parsed_json, "attributes");

  std::vector<std::int32_t> const expected_num_class{1};
  std::array<std::int32_t, 2> const expected_leaf_vector_shape{1, 1};
  std::vector<std::int32_t> const expected_target_id{0, 0};
  std::vector<std::int32_t> const expected_class_id{0, 1};
  std::map<std::string, treelite::model_builder::PostProcessorConfigParam> const
      expected_postprocessor_config
      = {{"sigmoid_alpha", 2.0}};
  std::vector<double> const expected_base_scores{0.5};
  std::string const expected_attributes_str = R"(
    {
      "foo": "bar",
      "cake": 2,
      "piece": 3.0
    }
  )";

  EXPECT_EQ(threshold_type, TypeInfo::kFloat32);
  EXPECT_EQ(leaf_output_type, TypeInfo::kFloat32);
  EXPECT_EQ(metadata.num_feature, 4);
  EXPECT_EQ(metadata.task_type, TaskType::kBinaryClf);
  EXPECT_FALSE(metadata.average_tree_output);
  EXPECT_EQ(metadata.num_target, 1);
  EXPECT_EQ(metadata.num_class, expected_num_class);
  EXPECT_EQ(metadata.leaf_vector_shape, expected_leaf_vector_shape);
  EXPECT_EQ(tree_annotation.target_id, expected_target_id);
  EXPECT_EQ(tree_annotation.class_id, expected_class_id);
  EXPECT_EQ(postprocessor.name, "sigmoid");
  EXPECT_EQ(postprocessor.config, expected_postprocessor_config);
  EXPECT_EQ(base_scores, expected_base_scores);
  EXPECT_TRUE(attributes.has_value());
  AssertJSONStringsEqual(attributes.value(), expected_attributes_str);
}

}  // namespace treelite
