/*!
 * Copyright (c) 2023 by Contributors
 * \file json_importer.cc
 * \brief Function to construct a Treelite model from a JSON string
 * \author Hyunsu Cho
 */

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <treelite/frontend.h>
#include <treelite/tree.h>

#include <cstddef>
#include <cstring>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>

namespace {

template <typename ObjectType>
int ExpectInt(ObjectType const& doc, char const* key) {
  auto itr = doc.FindMember(key);
  TREELITE_CHECK(itr != doc.MemberEnd()) << "Expected key \"" << key << "\" but it does not exist";
  TREELITE_CHECK(itr->value.IsInt()) << "Key \"" << key << "\" must be an int";
  return itr->value.GetInt();
}

template <typename ObjectType>
unsigned int ExpectUint(ObjectType const& doc, char const* key) {
  auto itr = doc.FindMember(key);
  TREELITE_CHECK(itr != doc.MemberEnd()) << "Expected key \"" << key << "\" but it does not exist";
  TREELITE_CHECK(itr->value.IsUint()) << "Key \"" << key << "\" must be an unsigned int";
  return itr->value.GetUint();
}

template <typename ObjectType>
float ExpectFloat(ObjectType const& doc, char const* key) {
  auto itr = doc.FindMember(key);
  TREELITE_CHECK(itr != doc.MemberEnd()) << "Expected key \"" << key << "\" but it does not exist";
  TREELITE_CHECK(itr->value.IsFloat()) << "Key \"" << key << "\" must be a single-precision float";
  return itr->value.GetFloat();
}

template <typename ObjectType>
bool ExpectBool(ObjectType const& doc, char const* key) {
  auto itr = doc.FindMember(key);
  TREELITE_CHECK(itr != doc.MemberEnd()) << "Expected key \"" << key << "\" but it does not exist";
  TREELITE_CHECK(itr->value.IsBool()) << "Key \"" << key << "\" must be a boolean";
  return itr->value.GetBool();
}

template <typename ObjectType>
std::string ExpectString(ObjectType const& doc, char const* key) {
  auto itr = doc.FindMember(key);
  TREELITE_CHECK(itr != doc.MemberEnd()) << "Expected key \"" << key << "\" but it does not exist";
  TREELITE_CHECK(itr->value.IsString()) << "Key \"" << key << "\" must be a string";
  return {itr->value.GetString(), itr->value.GetStringLength()};
}

template <typename ObjectType>
rapidjson::Value::ConstObject ExpectObject(ObjectType const& doc, char const* key) {
  auto itr = doc.FindMember(key);
  TREELITE_CHECK(itr != doc.MemberEnd()) << "Expected key \"" << key << "\" but it does not exist";
  TREELITE_CHECK(itr->value.IsObject()) << "Key \"" << key << "\" must be an object";
  return itr->value.GetObject();
}

template <typename ObjectType>
rapidjson::Value::ConstArray ExpectArray(ObjectType const& doc, char const* key) {
  auto itr = doc.FindMember(key);
  TREELITE_CHECK(itr != doc.MemberEnd()) << "Expected key \"" << key << "\" but it does not exist";
  TREELITE_CHECK(itr->value.IsArray()) << "Key \"" << key << "\" must be an array";
  return itr->value.GetArray();
}

template <typename ObjectType>
void ExpectFloatOptional(ObjectType const& doc, char const* key, float& result) {
  auto itr = doc.FindMember(key);
  if (itr != doc.MemberEnd()) {
    TREELITE_CHECK(itr->value.IsFloat())
        << "Key \"" << key << "\" must be a single-precision float";
    result = itr->value.GetFloat();
  }
}

treelite::TaskParam ParseTaskParam(rapidjson::Value::ConstObject const& object) {
  treelite::TaskParam param{};
  param.output_type = treelite::StringToOutputType(ExpectString(object, "output_type"));
  TREELITE_CHECK(param.output_type != treelite::TaskParam::OutputType::kInt)
      << "Integer output type is not supported";
  param.grove_per_class = ExpectBool(object, "grove_per_class");
  param.num_class = ExpectUint(object, "num_class");
  param.leaf_vector_size = ExpectUint(object, "leaf_vector_size");
  return param;
}

treelite::ModelParam ParseModelParam(rapidjson::Value::ConstObject const& object) {
  treelite::ModelParam param;

  std::string pred_transform = ExpectString(object, "pred_transform");
  constexpr std::size_t max_pred_transform_len = sizeof(param.pred_transform) - 1;
  TREELITE_CHECK_LE(pred_transform.length(), max_pred_transform_len)
      << "pred_transform cannot be longer than " << max_pred_transform_len << " characters";
  std::strncpy(param.pred_transform, pred_transform.c_str(), max_pred_transform_len);

  ExpectFloatOptional(object, "sigmoid_alpha", param.sigmoid_alpha);
  ExpectFloatOptional(object, "ratio_c", param.ratio_c);
  ExpectFloatOptional(object, "global_bias", param.global_bias);

  return param;
}

void ParseInternalNode(
    rapidjson::Value const& node, int new_node_id, treelite::Tree<float, float>& tree) {
  unsigned int const split_index = ExpectUint(node, "split_feature_id");
  bool const default_left = ExpectBool(node, "default_left");
  const treelite::SplitFeatureType split_type
      = treelite::LookupSplitFeatureTypeByName(ExpectString(node, "split_type"));
  if (split_type == treelite::SplitFeatureType::kNumerical) {  // numerical split
    const treelite::Operator op
        = treelite::LookupOperatorByName(ExpectString(node, "comparison_op"));
    float const threshold = ExpectFloat(node, "threshold");
    tree.SetNumericalSplit(new_node_id, split_index, threshold, default_left, op);
  } else if (split_type == treelite::SplitFeatureType::kCategorical) {  // categorical split
    bool const categories_list_right_child = ExpectBool(node, "categories_list_right_child");
    std::vector<std::uint32_t> categories_list;
    for (auto const& e : ExpectArray(node, "categories_list")) {
      TREELITE_CHECK(e.IsUint()) << "Expected an unsigned integer in categories_list field";
      categories_list.push_back(e.GetUint());
    }
    tree.SetCategoricalSplit(
        new_node_id, split_index, default_left, categories_list, categories_list_right_child);
  }
}

void ParseTree(rapidjson::Value::ConstObject const& object, treelite::Tree<float, float>& tree) {
  int const root_id = ExpectInt(object, "root_id");
  TREELITE_CHECK_GE(root_id, 0) << "root_id cannot be negative";
  auto const& nodes = ExpectArray(object, "nodes");
  TREELITE_CHECK(!nodes.Empty()) << "The nodes array must not be empty";
  const std::size_t num_nodes = nodes.Size();

  // Scan through nodes and create a lookup table. This is so that users can specify custom
  // node IDs via field "node_id". Note that the constructed Treelite model objects will use
  // different node IDs than the ones specified by the user.
  std::map<int, rapidjson::Value const&> node_lookup_table;
  for (auto const& e : nodes) {
    TREELITE_CHECK(e.IsObject()) << "All elements in the nodes array must be objects";
    int node_id = ExpectInt(e, "node_id");
    TREELITE_CHECK_GE(node_id, 0) << "node_id cannot be negative";
    node_lookup_table.insert({node_id, e});
  }

  tree.Init();
  // Assign new node IDs so that a breadth-wise traversal would yield
  // the monotonic sequence 0, 1, 2, ...
  std::queue<std::pair<int, int>> Q;  // (old ID, new ID) pair
  Q.emplace(root_id, 0);
  while (!Q.empty()) {
    auto [node_id, new_node_id] = Q.front();
    Q.pop();
    auto const& node = node_lookup_table.at(node_id);
    auto leaf_itr = node.FindMember("leaf_value");
    if (leaf_itr != node.MemberEnd()) {
      // leaf node: leaf value
      if (leaf_itr->value.IsFloat()) {  // leaf value (scalar)
        tree.SetLeaf(new_node_id, leaf_itr->value.GetFloat());
      } else {  // leaf vector
        TREELITE_CHECK(leaf_itr->value.IsArray())
            << "leaf_value field must be either a single float or an array of floats";
        std::vector<float> leaf_vector;
        for (auto const& e : leaf_itr->value.GetArray()) {
          TREELITE_CHECK(e.IsFloat()) << "Detected a non-float element in leaf_value array";
          leaf_vector.push_back(e.GetFloat());
        }
        tree.SetLeafVector(new_node_id, leaf_vector);
      }
    } else {
      // internal (test) node
      tree.AddChilds(new_node_id);
      ParseInternalNode(node, new_node_id, tree);
      int const left_child_id = ExpectInt(node, "left_child");
      int const right_child_id = ExpectInt(node, "right_child");
      Q.emplace(left_child_id, tree.LeftChild(new_node_id));
      Q.emplace(right_child_id, tree.RightChild(new_node_id));
    }

    // Handle metadata
    {
      auto itr = node.FindMember("data_count");
      if (itr != node.MemberEnd()) {
        TREELITE_CHECK(itr->value.IsUint64()) << "data_count must be a valid integer";
        tree.SetDataCount(new_node_id, itr->value.GetUint64());
      }
    }
    {
      auto itr = node.FindMember("sum_hess");
      if (itr != node.MemberEnd()) {
        TREELITE_CHECK(itr->value.IsDouble()) << "sum_hess must be a valid float";
        tree.SetSumHess(new_node_id, itr->value.GetDouble());
      }
    }
    {
      auto itr = node.FindMember("gain");
      if (itr != node.MemberEnd()) {
        TREELITE_CHECK(itr->value.IsDouble()) << "gain must be a valid float";
        tree.SetGain(new_node_id, itr->value.GetDouble());
      }
    }
  }
}

}  // anonymous namespace

namespace treelite::frontend {

std::unique_ptr<treelite::Model> BuildModelFromJSONString(
    char const* json_str, char const* config_json) {
  // config_json unused for now

  rapidjson::Document model_spec;
  model_spec.Parse(json_str);
  TREELITE_CHECK(!model_spec.HasParseError())
      << "Error when parsing JSON config: offset " << model_spec.GetErrorOffset() << ", "
      << rapidjson::GetParseError_En(model_spec.GetParseError());

  TREELITE_CHECK(model_spec.IsObject()) << "JSON string must have an object at the root level";

  std::unique_ptr<Model> model = Model::Create<float, float>();
  model->num_feature = ExpectInt(model_spec, "num_feature");
  model->average_tree_output = ExpectBool(model_spec, "average_tree_output");
  model->task_type = StringToTaskType(ExpectString(model_spec, "task_type"));
  TREELITE_CHECK(model->task_type != TaskType::kMultiClfCategLeaf)
      << "Categorical leafs are not supported";
  model->task_param = ParseTaskParam(ExpectObject(model_spec, "task_param"));
  model->param = ParseModelParam(ExpectObject(model_spec, "model_param"));

  auto const tree_array = ExpectArray(model_spec, "trees");
  auto& trees = std::get<ModelPreset<float, float>>(model->variant_).trees;
  for (auto const& e : tree_array) {
    TREELITE_CHECK(e.IsObject()) << "Expected a JSON object in \"trees\" array";
    trees.emplace_back();
    ParseTree(e.GetObject(), trees.back());
  }

  return model;
}

}  // namespace treelite::frontend
