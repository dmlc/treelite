/*!
 * Copyright (c) 2023 by Contributors
 * \file array_builder.cc
 * \brief Model builder to build a Treelite model from a collection of arrays
 * \author Hyunsu Cho
 */

#include <rapidjson/error/en.h>
#include <rapidjson/document.h>

#include <treelite/tree.h>
#include <treelite/frontend.h>

#include <queue>

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

}  // anonymous namespace

namespace treelite::frontend {

std::unique_ptr<treelite::Model> BuildModelFromArrays(
    const Metadata& metadata, const std::int64_t* node_count, const std::int8_t** split_type,
    const std::int8_t** default_left, const std::int64_t** children_left,
    const std::int64_t** children_right, const std::int64_t** split_feature,
    const float** threshold, const float** leaf_value, const std::uint32_t** categories_list,
    const std::int64_t** categories_list_offset_begin,
    const std::int64_t** categories_list_offset_end,
    const std::int8_t** categories_list_right_child) {
  std::unique_ptr<treelite::Model> model_ptr = treelite::Model::Create<float, float>();
  auto* model = dynamic_cast<treelite::ModelImpl<float, float>*>(model_ptr.get());
  model->num_feature = metadata.num_feature;
  model->average_tree_output = metadata.average_tree_output;
  model->task_type = metadata.task_type;
  model->task_param = metadata.task_param;
  model->param = metadata.model_param;

  for (std::int32_t tree_id = 0; tree_id < metadata.num_tree; ++tree_id) {
    model->trees.emplace_back();
    treelite::Tree<float, float>& tree = model->trees.back();
    tree.Init();

    // Assign node ID's so that a breadth-wise traversal would yield
    // the monotonic sequence 0, 1, 2, ...
    std::queue<std::pair<std::int32_t, std::int32_t>> Q;  // (old ID, new ID) pair
    Q.emplace(0, 0);
    while (!Q.empty()) {
      auto [node_id, new_node_id] = Q.front();
      Q.pop();
      const auto split_type_ = static_cast<SplitFeatureType>(split_type[tree_id][node_id]);
      if (split_type_ == SplitFeatureType::kNone) {
        // Leaf node
        const auto leaf_vector_size = metadata.task_param.leaf_vector_size;
        if (leaf_vector_size == 1) {
          tree.SetLeaf(new_node_id, leaf_value[tree_id][node_id]);
        } else {
          std::vector<float> leaf_vector{&leaf_value[tree_id][node_id * leaf_vector_size],
                                         &leaf_value[tree_id][(node_id + 1) * leaf_vector_size]};
          tree.SetLeafVector(new_node_id, leaf_vector);
        }
      } else {
        // internal (test) node
        tree.AddChilds(new_node_id);
        if (split_type_ == SplitFeatureType::kNumerical) {
          // numerical split
          tree.SetNumericalSplit(new_node_id, split_feature[tree_id][node_id],
                                 threshold[tree_id][node_id],
                                 static_cast<bool>(default_left[tree_id][node_id]),
                                 metadata.comparison_op);
        } else {
          // categorical split
          std::vector<std::uint32_t> categories_list_{
              &categories_list[tree_id][categories_list_offset_begin[tree_id][node_id]],
              &categories_list[tree_id][categories_list_offset_end[tree_id][node_id]]
          };
          tree.SetCategoricalSplit(new_node_id, split_feature[tree_id][node_id],
                                   static_cast<bool>(default_left[tree_id][node_id]),
                                   categories_list_, categories_list_right_child[tree_id][node_id]);
        }

        Q.emplace(children_left[tree_id][node_id], tree.LeftChild(new_node_id));
        Q.emplace(children_right[tree_id][node_id], tree.RightChild(new_node_id));
      }
    }
  }

  return model_ptr;
}

Metadata ParseMetadata(const char* metadata_json) {
  Metadata metadata;

  rapidjson::Document metadata_obj;
  metadata_obj.Parse(metadata_json);
  TREELITE_CHECK(!metadata_obj.HasParseError())
    << "Error when parsing JSON config: offset " << metadata_obj.GetErrorOffset()
    << ", " << rapidjson::GetParseError_En(metadata_obj.GetParseError());
  TREELITE_CHECK(metadata_obj.IsObject()) << "JSON string must have an object at the root level";

  metadata.num_tree = ExpectInt(metadata_obj, "num_tree");
  TREELITE_CHECK_GT(metadata.num_tree, 0) << "num_tree must be positive";
  metadata.num_feature = ExpectInt(metadata_obj, "num_feature");
  TREELITE_CHECK_GT(metadata.num_feature, 0) << "num_feature must be positive";
  metadata.average_tree_output = ExpectBool(metadata_obj, "average_tree_output");
  metadata.task_type = StringToTaskType(ExpectString(metadata_obj, "task_type"));
  metadata.comparison_op = LookupOperatorByName(ExpectString(metadata_obj, "comparison_op"));

  auto task_param_obj = ExpectObject(metadata_obj, "task_param");
  metadata.task_param.output_type = TaskParam::OutputType::kFloat;
  metadata.task_param.grove_per_class = ExpectBool(task_param_obj, "grove_per_class");
  metadata.task_param.num_class = ExpectUint(task_param_obj, "num_class");
  metadata.task_param.leaf_vector_size = ExpectUint(task_param_obj, "leaf_vector_size");

  auto model_param_obj = ExpectObject(metadata_obj, "model_param");
  std::string pred_transform = ExpectString(model_param_obj, "pred_transform");
  constexpr std::size_t max_pred_transform_len = sizeof(metadata.model_param.pred_transform) - 1;
  TREELITE_CHECK_LE(pred_transform.length(), max_pred_transform_len)
    << "pred_transform cannot be longer than " << max_pred_transform_len << " characters";
  std::strncpy(metadata.model_param.pred_transform, pred_transform.c_str(), max_pred_transform_len);
  metadata.model_param.sigmoid_alpha = ExpectFloat(model_param_obj, "sigmoid_alpha");
  metadata.model_param.ratio_c = ExpectFloat(model_param_obj, "ratio_c");
  metadata.model_param.global_bias = ExpectFloat(model_param_obj, "global_bias");

  return metadata;
}

}  // namespace treelite::frontend
