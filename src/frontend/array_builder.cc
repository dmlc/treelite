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
#include "../threading_utils/parallel_for.h"

#include <algorithm>
#include <limits>

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
    const std::int8_t** default_left, const std::int32_t** children_left,
    const std::int32_t** children_right, const std::uint32_t** split_feature,
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

  // TODO(hcho3): comparison op must be configurable per-node
  const auto comparison_op = metadata.comparison_op;

  // Access internals of Tree objects directly, since we trust the user to format the
  // arrays the right way
  using ModelAccessor =
      treelite::unsafe::InternalAccessor<treelite::unsafe::HERE_COMES_THE_DRAGON>;

  for (std::int32_t tree_id = 0; tree_id < metadata.num_tree; ++tree_id) {
    model->trees.emplace_back();
    treelite::Tree<float, float>& tree = model->trees.back();
    tree.Init();

    const std::size_t n_nodes = node_count[tree_id];
    auto& nodes = ModelAccessor::GetNodeArray(tree);
    nodes.Resize(n_nodes);
    tree.num_nodes = static_cast<int>(n_nodes);

    // If nthread <= 0, then use all CPU cores in the system
    // TODO(hcho3): nthread must be configurable
    auto thread_config = threading_utils::ConfigureThreadConfig(-1);
    auto sched = threading_utils::ParallelSchedule::Static();
    const auto leaf_vector_size = metadata.task_param.leaf_vector_size;
    TREELITE_LOG(INFO) << "leaf_vector_size = " << leaf_vector_size;

    if (leaf_vector_size > 1) {
      // Special handling for leaf vectors
      // Populating leaf_vector field is straightforward, since all leaf node have the
      // identical leaf vector size
      auto& leaf_vector = ModelAccessor::GetLeafVector(tree);
      leaf_vector.Resize(n_nodes * leaf_vector_size);
      std::copy(&leaf_value[tree_id][0], &leaf_value[tree_id][n_nodes * leaf_vector_size],
                leaf_vector.Data());
      auto& leaf_vector_offset_beg = ModelAccessor::GetLeafVectorBegin(tree);
      auto& leaf_vector_offset_end = ModelAccessor::GetLeafVectorEnd(tree);
      leaf_vector_offset_beg.Resize(n_nodes);
      leaf_vector_offset_end.Resize(n_nodes);
      std::transform(&categories_list_offset_begin[tree_id][0],
                     &categories_list_offset_begin[tree_id][n_nodes],
                     leaf_vector_offset_beg.Data(),
                     [](std::int64_t e) { return static_cast<std::size_t>(e); });
      std::transform(&categories_list_offset_end[tree_id][0],
                     &categories_list_offset_end[tree_id][n_nodes],
                     leaf_vector_offset_end.Data(),
                     [](std::int64_t e) { return static_cast<std::size_t>(e); });
    } else {
      auto& leaf_vector_offset_beg = ModelAccessor::GetLeafVectorBegin(tree);
      auto& leaf_vector_offset_end = ModelAccessor::GetLeafVectorEnd(tree);
      leaf_vector_offset_beg.Resize(n_nodes, 0);
      leaf_vector_offset_end.Resize(n_nodes, 0);
    }

    {
      // Special handling for category list in categorical splits
      auto& categories = ModelAccessor::GetMatchingCategories(tree);
      auto& categories_offset = ModelAccessor::GetMatchingCategoriesOffset(tree);
      const auto n_cat = categories_list_offset_end[tree_id][n_nodes - 1];
      if (n_cat > 0) {
        ModelAccessor::SetCategoricalSplitFlag(tree, true);
        categories.Resize(n_cat);
        std::copy(&categories_list[tree_id][0], &categories_list[tree_id][n_cat],
                  categories.Data());
        categories_offset.Resize(n_nodes + 1);
        std::copy(&categories_list_offset_begin[tree_id][0],
                  &categories_list_offset_begin[tree_id][n_nodes], categories_offset.Data());
        std::copy(&categories_list_offset_end[tree_id][n_nodes - 1],
                  &categories_list_offset_end[tree_id][n_nodes],
                  categories_offset.Data() + static_cast<std::size_t>(n_nodes));
      } else {
        ModelAccessor::SetCategoricalSplitFlag(tree, false);
      }
    }

    TREELITE_LOG(INFO) << "n_nodes = " << n_nodes;

    threading_utils::ParallelFor(std::size_t(0), n_nodes, thread_config, sched,
                                 [&](std::size_t node_id, int) {
      Tree<float, float>::Node& node = nodes[node_id];
      node.Init();
      node.split_type_ = static_cast<SplitFeatureType>(split_type[tree_id][node_id]);
      if (node.split_type_ == SplitFeatureType::kNone) {
        // Leaf node
        node.cleft_ = -1;
        node.cright_ = -1;
        if (leaf_vector_size == 1) {
          (node.info_).leaf_value = leaf_value[tree_id][node_id];
        }  // no need to handle leaf vectors here, since we've already dealt with them earlier
      } else {
        // Internal node
        std::uint32_t split_index = split_feature[tree_id][node_id];
        TREELITE_CHECK_LT(split_index, ((1U << 31U) - 1)) << "split_index too big";
        if (default_left[tree_id][node_id]) {
          split_index |= (1U << 31U);
        }
        node.sindex_ = split_index;
        node.cleft_ = children_left[tree_id][node_id];
        node.cright_ = children_right[tree_id][node_id];
        if (node.split_type_ == SplitFeatureType::kNumerical) {
          (node.info_).threshold = threshold[tree_id][node_id];
          node.cmp_ = comparison_op;
          node.split_type_ = SplitFeatureType::kNumerical;
          node.categories_list_right_child_ = false;
        } else if (node.split_type_ == SplitFeatureType::kCategorical) {
          node.split_type_ = SplitFeatureType::kCategorical;
          node.categories_list_right_child_ = categories_list_right_child[tree_id][node_id];
        }
      }
    });
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
