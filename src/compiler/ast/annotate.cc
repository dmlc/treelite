/*!
 * Copyright 2017 by Contributors
 * \file annotate.cc
 * \brief Annotate an AST
 */
#include <cmath>
#include "./builder.h"

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(annotate);

static void annotate(ASTNode* node,
                     const std::vector<std::vector<size_t>>& counts) {
  if (node->tree_id >= 0 && node->node_id >= 0) {
    node->data_count = counts[node->tree_id][node->node_id];
  }
  for (ASTNode* child : node->children) {
    annotate(child, counts);
  }
}

void
ASTBuilder::AnnotateBranches(const std::vector<std::vector<size_t>>& counts) {
  annotate(this->main_node, counts);
}

}  // namespace compiler
}  // namespace treelite
