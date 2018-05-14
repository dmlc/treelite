/*!
 * Copyright 2017 by Contributors
 * \file breakup.cc
 * \brief Break up large translation units
 */
#include "./builder.h"

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(breakup);

static int count_tu(ASTNode* node) {
  int accum = (dynamic_cast<TranslationUnitNode*>(node)) ? 1 : 0;
  for (ASTNode* child : node->children) {
    accum += count_tu(child);
  }
  return accum;
}

bool breakup(ASTNode* node, int num_descendant_limit, int* num_tu,
             ASTBuilder* builder) {
  bool flag = false;
  CHECK(node->num_descendant_ast_node.has_value());
  if (dynamic_cast<ConditionNode*>(node)
      && node->num_descendant_ast_node.value() > num_descendant_limit) {
    bool break_here = true;
    for (ASTNode* child : node->children) {
      CHECK(child->num_descendant_ast_node.has_value());
      if (child->num_descendant_ast_node.value() > num_descendant_limit) {
        break_here = false;  // don't break this node; break the child instead
      }
    }
    if (break_here) {
      ASTNode* parent = node->parent;

      int node_idx = -1;
      for (size_t i = 0; i < parent->children.size(); ++i) {
        if (parent->children[i] == node) {
          node_idx = static_cast<int>(i);
          break;
        }
      }
      CHECK_GE(node_idx, 0);

      const int unit_id = (*num_tu)++;
      TranslationUnitNode* tu
        = builder->AddNode<TranslationUnitNode>(parent, unit_id);
      AccumulatorContextNode* ac
        = builder->AddNode<AccumulatorContextNode>(tu);
      parent->children[node_idx] = tu;
      tu->children.push_back(ac);
      ac->children.push_back(node);
      node->parent = ac;
      tu->num_descendant_ast_node = 0;
      ASTNode* n = tu->parent;
      while (n) {
        CHECK(n->num_descendant_ast_node.has_value());
        n->num_descendant_ast_node
          = n->num_descendant_ast_node.value()
            - node->num_descendant_ast_node.value();
        CHECK_GE(n->num_descendant_ast_node.value(), 0);
        n = n->parent;
      }
      flag = true;
    }
  }
  for (ASTNode* child : node->children) {
    CHECK(child->num_descendant_ast_node.has_value());
    if (child->num_descendant_ast_node.value() > 0) {
      flag |= breakup(child, num_descendant_limit, num_tu, builder);
    }
  }
  return flag;
}

void ASTBuilder::BreakUpLargeUnits(int num_descendant_limit) {
  CHECK_GT(num_descendant_limit, 0);
  int num_tu = count_tu(this->main_node);
  while (breakup(this->main_node, num_descendant_limit, &num_tu, this)) {}
}

}  // namespace compiler
}  // namespace treelite
