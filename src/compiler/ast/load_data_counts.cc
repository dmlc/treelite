/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file load_data_counts.cc
 * \brief AST manipulation logic to load data counts
 * \author Hyunsu Cho
 */
#include <dmlc/registry.h>
#include <cmath>
#include "./builder.h"

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(load_data_counts);

static void load_data_counts(ASTNode* node, const std::vector<std::vector<size_t>>& counts) {
  if (node->tree_id >= 0 && node->node_id >= 0) {
    node->data_count = counts[node->tree_id][node->node_id];
  }
  for (ASTNode* child : node->children) {
    load_data_counts(child, counts);
  }
}

template <typename ThresholdType, typename LeafOutputType>
void
ASTBuilder<ThresholdType, LeafOutputType>::LoadDataCounts(
    const std::vector<std::vector<size_t>>& counts) {
  load_data_counts(this->main_node, counts);
}

template void ASTBuilder<float, uint32_t>::LoadDataCounts(const std::vector<std::vector<size_t>>&);
template void ASTBuilder<float, float>::LoadDataCounts(const std::vector<std::vector<size_t>>&);
template void ASTBuilder<double, uint32_t>::LoadDataCounts(const std::vector<std::vector<size_t>>&);
template void ASTBuilder<double, double>::LoadDataCounts(const std::vector<std::vector<size_t>>&);

}  // namespace compiler
}  // namespace treelite
