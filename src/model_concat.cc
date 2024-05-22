/*!
 * Copyright (c) 2022-2023 by Contributors
 * \file model_concat.cc
 * \brief Implementation of model concatenation
 * \author Hyunsu Cho
 */

#include <treelite/logging.h>
#include <treelite/tree.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <type_traits>
#include <variant>

namespace treelite {

std::unique_ptr<Model> ConcatenateModelObjects(std::vector<Model const*> const& objs) {
  if (objs.empty()) {
    return {};
  }
  TypeInfo const threshold_type = objs[0]->GetThresholdType();
  TypeInfo const leaf_output_type = objs[0]->GetLeafOutputType();
  std::unique_ptr<Model> concatenated_model = Model::Create(threshold_type, leaf_output_type);
  // Header
  concatenated_model->num_feature = objs[0]->num_feature;
  concatenated_model->task_type = objs[0]->task_type;
  concatenated_model->average_tree_output = objs[0]->average_tree_output;
  // Task parameters
  concatenated_model->num_target = objs[0]->num_target;
  concatenated_model->num_class = objs[0]->num_class.Clone();
  concatenated_model->leaf_vector_shape = objs[0]->leaf_vector_shape.Clone();
  // Model parameters
  concatenated_model->postprocessor = objs[0]->postprocessor;
  concatenated_model->sigmoid_alpha = objs[0]->sigmoid_alpha;
  concatenated_model->ratio_c = objs[0]->ratio_c;
  concatenated_model->base_scores = objs[0]->base_scores.Clone();
  concatenated_model->attributes = objs[0]->attributes;

  std::visit(
      [&objs, &concatenated_model](auto&& first_model_obj) {
        using ModelType = std::remove_const_t<std::remove_reference_t<decltype(first_model_obj)>>;
        TREELITE_CHECK(std::holds_alternative<ModelType>(concatenated_model->variant_));
        auto& concatenated_model_concrete = std::get<ModelType>(concatenated_model->variant_);
        for (std::size_t i = 0; i < objs.size(); ++i) {
          TREELITE_CHECK(std::holds_alternative<ModelType>(objs[i]->variant_))
              << "Model object at index " << i
              << " has a different type than the first model object (at index 0)";
          TREELITE_CHECK_EQ(concatenated_model->num_target, objs[i]->num_target)
              << "Model object at index " << i
              << "has a different num_target than the first model object (at index 0)";
          TREELITE_CHECK(concatenated_model->num_class == objs[i]->num_class)
              << "Model object at index " << i
              << "has a different num_class than the first model object (at index 0)";
          TREELITE_CHECK(concatenated_model->leaf_vector_shape == objs[i]->leaf_vector_shape)
              << "Model object at index " << i
              << "has a different leaf_vector_shape than the first model object (at index 0)";
          auto& casted = std::get<ModelType>(objs[i]->variant_);
          std::transform(casted.trees.begin(), casted.trees.end(),
              std::back_inserter(concatenated_model_concrete.trees),
              [](auto const& tree) { return tree.Clone(); });
          concatenated_model->target_id.Extend(objs[i]->target_id);
          concatenated_model->class_id.Extend(objs[i]->class_id);
        }
      },
      objs[0]->variant_);
  TREELITE_CHECK_EQ(concatenated_model->target_id.Size(), concatenated_model->GetNumTree());
  TREELITE_CHECK_EQ(concatenated_model->class_id.Size(), concatenated_model->GetNumTree());
  return concatenated_model;
}

}  // namespace treelite
