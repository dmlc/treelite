/*!
 * Copyright (c) 2022 by Contributors
 * \file model_concat.cc
 * \brief Implementation of model concatenation
 * \author Hyunsu Cho
 */

#include <treelite/logging.h>
#include <treelite/tree.h>

#include <algorithm>
#include <memory>
#include <type_traits>
#include <variant>

namespace treelite {

std::unique_ptr<Model> ConcatenateModelObjects(std::vector<Model const*> const& objs) {
  if (objs.empty()) {
    return {};
  }
  const TypeInfo threshold_type = objs[0]->GetThresholdType();
  const TypeInfo leaf_output_type = objs[0]->GetLeafOutputType();
  return std::visit(
      [&objs, threshold_type, leaf_output_type](auto&& first_model_obj) {
        using ModelType = std::remove_const_t<std::remove_reference_t<decltype(first_model_obj)>>;
        std::unique_ptr<Model> concatenated_model = Model::Create(threshold_type, leaf_output_type);
        ModelType& concatenated_model_concrete = std::get<ModelType>(concatenated_model->variant_);
        for (std::size_t i = 0; i < objs.size(); ++i) {
          if (!std::holds_alternative<ModelType>(objs[i]->variant_)) {
            TREELITE_LOG(FATAL) << "Model object at index " << i
                                << " has a different type than the first model object (at index 0)";
          }
          auto& casted = std::get<ModelType>(objs[i]->variant_);
          std::transform(casted.trees.begin(), casted.trees.end(),
              std::back_inserter(concatenated_model_concrete.trees),
              [](const auto& tree) { return tree.Clone(); });
        }
        /* Copy model metadata */
        concatenated_model->num_feature = objs[0]->num_feature;
        concatenated_model->task_type = objs[0]->task_type;
        concatenated_model->average_tree_output = objs[0]->average_tree_output;
        concatenated_model->task_param = objs[0]->task_param;
        concatenated_model->param = objs[0]->param;
        return concatenated_model;
      },
      objs[0]->variant_);
}

}  // namespace treelite
