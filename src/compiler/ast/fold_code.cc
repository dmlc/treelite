/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file fold_code.h
 * \brief AST manipulation logic for code folding
 * \author Hyunsu Cho
 */
#include <treelite/logging.h>
#include <limits>
#include <cmath>
#include "./builder.h"

namespace treelite {
namespace compiler {

struct CodeFoldingContext {
  double magnitude_req;
  double log_root_data_count;
  double log_root_sum_hess;
  bool create_new_translation_unit;
  int num_tu;
};

template <typename ThresholdType, typename LeafOutputType>
bool fold_code(ASTNode* node, CodeFoldingContext* context,
               ASTBuilder<ThresholdType, LeafOutputType>* builder) {
  if (node->node_id == 0) {
    if (node->data_count) {
      context->log_root_data_count = std::log(*node->data_count);
    } else {
      context->log_root_data_count = std::numeric_limits<double>::quiet_NaN();
    }
    if (node->sum_hess) {
      context->log_root_sum_hess = std::log(*node->sum_hess);
    } else {
      context->log_root_sum_hess = std::numeric_limits<double>::quiet_NaN();
    }
  }

  if (   (node->data_count && !std::isnan(context->log_root_data_count)
          && context->log_root_data_count - std::log(*node->data_count)
             >= context->magnitude_req)
      || (node->sum_hess && !std::isnan(context->log_root_sum_hess)
          && context->log_root_sum_hess - std::log(*node->sum_hess)
             >= context->magnitude_req) ) {
    // fold the subtree whose root is [node]
    ASTNode* parent_node = node->parent;
    ASTNode* folder_node = nullptr;
    ASTNode* tu_node = nullptr;
    if (context->create_new_translation_unit) {
      tu_node = builder->template AddNode<TranslationUnitNode>(parent_node, context->num_tu++);
      ASTNode* ac = builder->template AddNode<AccumulatorContextNode>(tu_node);
      folder_node = builder->template AddNode<CodeFolderNode>(ac);
      tu_node->children.push_back(ac);
      ac->children.push_back(folder_node);
    } else {
      folder_node = builder->template AddNode<CodeFolderNode>(parent_node);
    }
    size_t node_loc = -1;  // is current node 1st child or 2nd child or so forth
    for (size_t i = 0; i < parent_node->children.size(); ++i) {
      if (parent_node->children[i] == node) {
        node_loc = i;
        break;
      }
    }
    TREELITE_CHECK_NE(node_loc, -1);  // parent should have a link to current node
    parent_node->children[node_loc]
      = context->create_new_translation_unit ? tu_node : folder_node;
    folder_node->children.push_back(node);
    node->parent = folder_node;
    return true;
  } else {
    bool folded_at_least_once = false;
    for (ASTNode* child : node->children) {
      folded_at_least_once |= fold_code(child, context, builder);
    }
    return folded_at_least_once;
  }
}

int count_tu_nodes(ASTNode* node);

template <typename ThresholdType, typename LeafOutputType>
bool
ASTBuilder<ThresholdType, LeafOutputType>::FoldCode(
    double magnitude_req, bool create_new_translation_unit) {
  CodeFoldingContext context{magnitude_req,
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN(),
                             create_new_translation_unit,
                             count_tu_nodes(this->main_node)};
  return fold_code(this->main_node, &context, this);
}

template bool ASTBuilder<float, uint32_t>::FoldCode(double, bool);
template bool ASTBuilder<float, float>::FoldCode(double, bool);
template bool ASTBuilder<double, uint32_t>::FoldCode(double, bool);
template bool ASTBuilder<double, double>::FoldCode(double, bool);

}  // namespace compiler
}  // namespace treelite
