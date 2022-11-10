/*!
 * Copyright (c) 2022 by Contributors
 * \file model_concat.cc
 * \brief Implementation of model concatenation
 * \author Hyunsu Cho
 */

#include <treelite/tree.h>
#include <treelite/logging.h>
#include <algorithm>
#include <memory>
#include <type_traits>

namespace treelite {

std::unique_ptr<Model> ConcatenateModelObjects(const std::vector<const Model*>& objs) {
  if (objs.empty()) {
    return std::unique_ptr<Model>();
  }
  return objs[0]->Dispatch([&objs](const auto& first_model_obj) {
    using ModelType = std::remove_const_t<std::remove_reference_t<decltype(first_model_obj)>>;
    std::unique_ptr<ModelType> concatenated_model = std::make_unique<ModelType>();
    for (std::size_t i = 0; i < objs.size(); ++i) {
      auto* casted = dynamic_cast<const ModelType*>(objs[i]);
      if (casted) {
        std::transform(casted->trees.begin(), casted->trees.end(),
                       std::back_inserter(concatenated_model->trees),
                       [](const auto& tree) { return tree.Clone(); });
      } else {
        TREELITE_LOG(FATAL) << "Model object at index " << i
          << " has a different type than the first model object (at index 0)";
      }
    }
    return std::unique_ptr<Model>(concatenated_model.release());
  });
}

}  // namespace treelite