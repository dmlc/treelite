/*!
 * Copyright (c) 2023 by Contributors
 * \file json_importer.cc
 * \brief Function to construct a Treelite model from a JSON string
 * \author Hyunsu Cho
 */

#include <rapidjson/error/en.h>
#include <rapidjson/document.h>

#include <treelite/tree.h>
#include <treelite/frontend.h>
#include <memory>
#include <string>
#include <utility>
#include <queue>
#include <cstddef>
#include <cstring>

namespace {

template <typename ObjectType>
int ExpectInt(const ObjectType& doc, const char* key) {
  auto itr = doc.FindMember(key);
  TREELITE_CHECK(itr != doc.MemberEnd()) << "Expected key \"" << key << "\" but it does not exist";
  TREELITE_CHECK(itr->value.IsInt()) << "Key \"" << key << "\" must be an int";
  return itr->value.GetInt();
}

template <typename ObjectType>
unsigned int ExpectUint(const ObjectType& doc, const char* key) {
  auto itr = doc.FindMember(key);
  TREELITE_CHECK(itr != doc.MemberEnd()) << "Expected key \"" << key << "\" but it does not exist";
  TREELITE_CHECK(itr->value.IsUint()) << "Key \"" << key << "\" must be an unsigned int";
  return itr->value.GetUint();
}

template <typename ObjectType>
float ExpectFloat(const ObjectType& doc, const char* key) {
  auto itr = doc.FindMember(key);
  TREELITE_CHECK(itr != doc.MemberEnd()) << "Expected key \"" << key << "\" but it does not exist";
  TREELITE_CHECK(itr->value.IsFloat()) << "Key \"" << key << "\" must be a single-precision float";
  return itr->value.GetFloat();
}

template <typename ObjectType>
bool ExpectBool(const ObjectType& doc, const char* key) {
  auto itr = doc.FindMember(key);
  TREELITE_CHECK(itr != doc.MemberEnd()) << "Expected key \"" << key << "\" but it does not exist";
  TREELITE_CHECK(itr->value.IsBool()) << "Key \"" << key << "\" must be a boolean";
  return itr->value.GetBool();
}

template <typename ObjectType>
std::string ExpectString(const ObjectType& doc, const char* key) {
  auto itr = doc.FindMember(key);
  TREELITE_CHECK(itr != doc.MemberEnd()) << "Expected key \"" << key << "\" but it does not exist";
  TREELITE_CHECK(itr->value.IsString()) << "Key \"" << key << "\" must be a string";
  return {itr->value.GetString(), itr->value.GetStringLength()};
}

template <typename ObjectType>
rapidjson::Value::ConstObject ExpectObject(const ObjectType& doc, const char* key) {
  auto itr = doc.FindMember(key);
  TREELITE_CHECK(itr != doc.MemberEnd()) << "Expected key \"" << key << "\" but it does not exist";
  TREELITE_CHECK(itr->value.IsObject()) << "Key \"" << key << "\" must be an object";
  return itr->value.GetObject();
}

template <typename ObjectType>
rapidjson::Value::ConstArray ExpectArray(const ObjectType& doc, const char* key) {
  auto itr = doc.FindMember(key);
  TREELITE_CHECK(itr != doc.MemberEnd()) << "Expected key \"" << key << "\" but it does not exist";
  TREELITE_CHECK(itr->value.IsArray()) << "Key \"" << key << "\" must be an array";
  return itr->value.GetArray();
}

treelite::TaskParam ParseTaskParam(const rapidjson::Value::ConstObject& object) {
  treelite::TaskParam param;
  param.output_type = treelite::StringToOutputType(ExpectString(object, "output_type"));
  TREELITE_CHECK(param.output_type != treelite::TaskParam::OutputType::kInt)
    << "Integer output type is not supported";
  param.grove_per_class = ExpectBool(object, "grove_per_class");
  param.num_class = ExpectUint(object, "num_class");
  param.leaf_vector_size = ExpectUint(object, "leaf_vector_size");
  return param;
}

treelite::ModelParam ParseModelParam(const rapidjson::Value::ConstObject& object) {
  treelite::ModelParam param;

  std::string pred_transform = ExpectString(object, "pred_transform");
  constexpr std::size_t max_pred_transform_len = sizeof(param.pred_transform) - 1;
  TREELITE_CHECK_LE(pred_transform.length(), max_pred_transform_len)
    << "pred_transform cannot be longer than " << max_pred_transform_len << " characters";
  std::strncpy(param.pred_transform, pred_transform.c_str(), max_pred_transform_len);

  param.sigmoid_alpha = ExpectFloat(object, "sigmoid_alpha");
  param.ratio_c = ExpectFloat(object, "ratio_c");
  param.global_bias = ExpectFloat(object, "global_bias");

  return param;
}

void ParseInternalNode(
    const rapidjson::Value& node, int new_node_id, treelite::Tree<float, float>& tree) {
  const std::int64_t split_index = ExpectInt(node, "split_feature_id");
  const bool default_left = ExpectBool(node, "default_left");
  const treelite::SplitFeatureType split_type =
      treelite::LookupSplitFeatureTypeByName(ExpectString(node, "split_type"));
  if (split_type == treelite::SplitFeatureType::kNumerical) {  // numerical split
    const treelite::Operator op =
        treelite::LookupOperatorByName(ExpectString(node, "comparison_op"));
    const float threshold = ExpectFloat(node, "threshold");
    tree.SetNumericalSplit(new_node_id, split_index, threshold, default_left, op);
  } else if (split_type == treelite::SplitFeatureType::kCategorical) {  // categorical split
    const bool categories_list_right_child = ExpectBool(node, "categories_list_right_child");
    std::vector<std::uint32_t> categories_list;
    for (const auto& e : ExpectArray(node, "matching_categories")) {
      TREELITE_CHECK(e.IsUint()) << "Expected an unsigned in matching_categories field";
      categories_list.push_back(e.GetUint());
    }
    tree.SetCategoricalSplit(new_node_id, split_index, default_left, categories_list,
                             categories_list_right_child);
  }
}

void ParseTree(const rapidjson::Value::ConstObject& object, treelite::Tree<float, float>& tree) {
  const auto& nodes = ExpectArray(object, "nodes");
  TREELITE_CHECK(!nodes.Empty()) << "The nodes array must not be empty";
  const std::size_t num_nodes = nodes.Size();

  // TODO(hcho3): Clearly document that node 0 is assumed to be the root node
  tree.Init();

  // Assign node ID's so that a breadth-wise traversal would yield
  // the monotonic sequence 0, 1, 2, ...
  std::queue<std::pair<std::int32_t, int>> Q; // (old ID, new ID) pair
  Q.emplace(0, 0);
  while (!Q.empty()) {
    auto [node_id, new_node_id] = Q.front();
    Q.pop();
    TREELITE_CHECK_LT(node_id, num_nodes) << "node_id must be less than the nodes array size ("
      << num_nodes << ")";
    const auto& node = nodes[node_id];
    TREELITE_CHECK(node.IsObject()) << "The element " << node_id << "of the nodes array must be "
      << " an object";

    auto leaf_itr = node.FindMember("leaf_value");
    if (leaf_itr != node.MemberEnd()) {
      // leaf node: leaf value
      if (leaf_itr->value.IsFloat()) {  // leaf value (scalar)
        tree.SetLeaf(new_node_id, leaf_itr->value.GetFloat());
      } else {  // leaf vector
        TREELITE_CHECK(leaf_itr->value.IsArray())
          << "leaf_value field must be either a single float or an array of floats";
        std::vector<float> leaf_vector;
        for (const auto& e : leaf_itr->value.GetArray()) {
          TREELITE_CHECK(e.IsFloat()) << "Detected a non-float element in leaf_value array";
          leaf_vector.push_back(e.GetFloat());
        }
        tree.SetLeafVector(new_node_id, leaf_vector);
      }
    } else {
      // internal (test) node
      tree.AddChilds(new_node_id);
      ParseInternalNode(node, new_node_id, tree);
      const std::int32_t left_child_id = ExpectInt(node, "left_child");
      const std::int32_t right_child_id = ExpectInt(node, "right_child");
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
    const char* json_str, const char* config_json) {
  // config_json unused for now

  rapidjson::Document model_spec;
  model_spec.Parse(json_str);
  TREELITE_CHECK(!model_spec.HasParseError())
    << "Error when parsing JSON config: offset " << model_spec.GetErrorOffset()
    << ", " << rapidjson::GetParseError_En(model_spec.GetParseError());

  TREELITE_CHECK(model_spec.IsObject()) << "JSON string must have an object at the root level";

  std::unique_ptr<Model> model_ptr = Model::Create<float, float>();
  auto* model = dynamic_cast<treelite::ModelImpl<float, float>*>(model_ptr.get());
  model->num_feature = ExpectInt(model_spec, "num_feature");
  model->average_tree_output = ExpectBool(model_spec, "average_tree_output");
  model->task_type = StringToTaskType(ExpectString(model_spec, "task_type"));
  TREELITE_CHECK(model->task_type != TaskType::kMultiClfCategLeaf)
    << "Categorical leafs are not supported";
  model->task_param = ParseTaskParam(ExpectObject(model_spec, "task_param"));
  model->param = ParseModelParam(ExpectObject(model_spec, "model_param"));

  const auto tree_array = ExpectArray(model_spec, "trees");
  for (const auto& e : tree_array) {
    TREELITE_CHECK(e.IsObject()) << "Expected a JSON object in \"trees\" array";
    model->trees.emplace_back();
    ParseTree(e.GetObject(), model->trees.back());
  }

  return model_ptr;
}

}  // namespace treelite::frontend
