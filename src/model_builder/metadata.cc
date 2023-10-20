/*!
 * Copyright (c) 2023 by Contributors
 * \file metadata.cc
 * \brief C++ API for constructing Model metadata
 * \author Hyunsu Cho
 */
#include <treelite/enum/task_type.h>
#include <treelite/logging.h>
#include <treelite/model_builder.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace treelite::model_builder {

TreeAnnotation::TreeAnnotation(std::int32_t num_tree, std::vector<std::int32_t> const& target_id,
    std::vector<std::int32_t> const& class_id)
    : num_tree{num_tree}, target_id{target_id}, class_id{class_id} {
  TREELITE_CHECK_EQ(target_id.size(), num_tree)
      << "target_id field must have length equal to num_tree (" << num_tree << ")";
  TREELITE_CHECK_EQ(class_id.size(), num_tree)
      << "class_id field must have length equal to num_tree (" << num_tree << ")";
}

PostProcessorFunc::PostProcessorFunc(std::string const& name) : PostProcessorFunc(name, {}) {}

PostProcessorFunc::PostProcessorFunc(
    std::string const& name, std::map<std::string, PostProcessorConfigParam> const& config)
    : name(name), config(config) {}

Metadata::Metadata(std::int32_t num_feature, TaskType task_type, bool average_tree_output,
    std::int32_t num_target, std::vector<std::int32_t> const& num_class,
    std::array<std::int32_t, 2> const& leaf_vector_shape)
    : num_feature(num_feature),
      task_type(task_type),
      average_tree_output(average_tree_output),
      num_target(num_target),
      num_class(num_class),
      leaf_vector_shape(leaf_vector_shape) {
  TREELITE_CHECK_GT(num_target, 0) << "num_target must be at least 1";
  TREELITE_CHECK_EQ(num_class.size(), num_target)
      << "num_class field must have length equal to num_target (" << num_target << ")";
  if (!std::all_of(num_class.begin(), num_class.end(), [](std::int32_t e) { return e >= 1; })) {
    TREELITE_LOG(FATAL) << "All elements in num_class field must be at least 1.";
  }
  TREELITE_CHECK(leaf_vector_shape[0] == 1 || leaf_vector_shape[0] == num_target)
      << "leaf_vector_shape[0] must be either 1 or num_target (" << num_target << "). "
      << "Currently given: leaf_vector_shape[1] = " << leaf_vector_shape[1];
  std::int32_t const max_num_class = *std::max_element(num_class.begin(), num_class.end());
  TREELITE_CHECK(leaf_vector_shape[1] == 1 || leaf_vector_shape[1] == max_num_class)
      << "leaf_vector_shape[1] must be either 1 or max_num_class (" << max_num_class << "). "
      << "Currently given: leaf_vector_shape[1] = " << leaf_vector_shape[1];
}

}  // namespace treelite::model_builder
