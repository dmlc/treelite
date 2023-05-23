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

namespace treelite {

std::unique_ptr<Model> ConcatenateModelObjects(std::vector<Model const*> const& objs) {
  if (objs.empty()) {
    return std::unique_ptr<Model>();
  }
  const TypeInfo threshold_type = objs[0]->GetThresholdType();
  const TypeInfo leaf_output_type = objs[0]->GetLeafOutputType();
  return objs[0]->Dispatch([&objs, threshold_type, leaf_output_type](auto const& first_model_obj) {
    using ModelType = std::remove_const_t<std::remove_reference_t<decltype(first_model_obj)>>;
    std::unique_ptr<Model> concatenated_model = Model::Create(threshold_type, leaf_output_type);
    auto* concatenated_model_concrete = dynamic_cast<ModelType*>(concatenated_model.get());
    for (std::size_t i = 0; i < objs.size(); ++i) {
      auto* casted = dynamic_cast<const ModelType*>(objs[i]);
      if (casted) {
        std::transform(casted->trees.begin(), casted->trees.end(),
            std::back_inserter(concatenated_model_concrete->trees),
            [](const auto& tree) { return tree.Clone(); });
      } else {
        TREELITE_LOG(FATAL) << "Model object at index " << i
                            << " has a different type than the first model object (at index 0)";
      }
    }
    /* Copy model metadata */
    concatenated_model->num_feature = first_model_obj.num_feature;
    concatenated_model->task_type = first_model_obj.task_type;
    concatenated_model->average_tree_output = first_model_obj.average_tree_output;
    concatenated_model->task_param = first_model_obj.task_param;
    concatenated_model->param = first_model_obj.param;
    return concatenated_model;
  });
}

}  // namespace treelite
