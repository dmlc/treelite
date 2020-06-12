/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file is_categorical_array.cc
 * \brief AST manipulation logic to determine whether each feature is categorical or not
 * \author Hyunsu Cho
 */
#include <dmlc/registry.h>
#include "./builder.h"

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(is_categorical_array);

static void
scan_thresholds(ASTNode* node, std::vector<bool>* is_categorical) {
  CategoricalConditionNode* cat_cond
    = dynamic_cast<CategoricalConditionNode*>(node);
  if (cat_cond) {
    (*is_categorical)[cat_cond->split_index] = true;
  }
  for (ASTNode* child : node->children) {
    scan_thresholds(child, is_categorical);
  }
}

std::vector<bool> ASTBuilder::GenerateIsCategoricalArray() {
  this->is_categorical = std::vector<bool>(this->num_feature, false);
  scan_thresholds(this->main_node, &this->is_categorical);
  return this->is_categorical;
}

}  // namespace compiler
}  // namespace treelite
