/*!
 * Copyright (c) 2023 by Contributors
 * \file output_shape.cc
 * \author Hyunsu Cho
 * \brief Compute output shape for GTIL, so that callers can allocate sufficient space
 *        to hold outputs.
 */
#include <treelite/gtil.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace treelite::gtil {

std::vector<std::uint64_t> GetOutputShape(
    Model const& model, std::uint64_t num_row, Configuration const& config) {
  auto const num_tree = model.GetNumTree();
  auto const max_num_class = static_cast<std::uint64_t>(
      *std::max_element(model.num_class.Data(), model.num_class.Data() + model.num_target));
  switch (config.pred_kind) {
  case PredictKind::kPredictDefault:
  case PredictKind::kPredictRaw:
    if (model.num_target > 1) {
      return {static_cast<std::uint64_t>(model.num_target), num_row, max_num_class};
    } else {
      return {1, num_row, max_num_class};
    }
  case PredictKind::kPredictLeafID:
    return {num_row, num_tree};
  case PredictKind::kPredictPerTree:
    return {num_row, num_tree,
        static_cast<std::uint64_t>(model.leaf_vector_shape[0]) * model.leaf_vector_shape[1]};
  default:
    TREELITE_LOG(FATAL) << "Unsupported model type: " << static_cast<int>(config.pred_kind);
    return {};
  }
}

}  // namespace treelite::gtil
