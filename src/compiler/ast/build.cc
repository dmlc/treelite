/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file build.cc
 * \brief Build AST from a given model
 */
#include <dmlc/registry.h>
#include "./builder.h"

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(build);

void ASTBuilder::BuildAST(const ModelImpl& model) {
  this->output_vector_flag
    = (model.num_output_group > 1 && model.random_forest_flag);
  this->num_feature = model.num_feature;
  this->num_output_group = model.num_output_group;
  this->random_forest_flag = model.random_forest_flag;

  this->main_node = AddNode<MainNode>(nullptr, model.param.global_bias,
                                               model.random_forest_flag,
                                               model.trees.size(),
                                               model.num_feature);
  ASTNode* ac = AddNode<AccumulatorContextNode>(this->main_node);
  this->main_node->children.push_back(ac);
  for (int tree_id = 0; tree_id < model.trees.size(); ++tree_id) {
    ASTNode* tree_head = BuildASTFromTree(model.trees[tree_id], tree_id, ac);
    ac->children.push_back(tree_head);
  }
  this->model_param = model.param.__DICT__();
}

ASTNode* ASTBuilder::BuildASTFromTree(const Tree& tree, int tree_id,
                                      ASTNode* parent) {
  return BuildASTFromTree(tree, tree_id, 0, parent);
}

ASTNode* ASTBuilder::BuildASTFromTree(const Tree& tree, int tree_id, int nid,
                                      ASTNode* parent) {
  ASTNode* ast_node = nullptr;
  if (tree.IsLeaf(nid)) {
    if (this->output_vector_flag) {
      ast_node = AddNode<OutputNode>(parent, tree.LeafVector(nid));
    } else {
      ast_node = AddNode<OutputNode>(parent, tree.LeafValue(nid));
    }
  } else {
    if (tree.SplitType(nid) == SplitFeatureType::kNumerical) {
      ast_node = AddNode<NumericalConditionNode>(parent,
                                                 tree.SplitIndex(nid),
                                                 tree.DefaultLeft(nid),
                                                 false,
                                                 tree.ComparisonOp(nid),
                                                 ThresholdVariant(static_cast<tl_float>(
                                                                      tree.Threshold(nid))));
    } else {
      ast_node = AddNode<CategoricalConditionNode>(parent,
                                                   tree.SplitIndex(nid),
                                                   tree.DefaultLeft(nid),
                                                   tree.LeftCategories(nid),
                                                   tree.MissingCategoryToZero(nid));
    }
    if (tree.HasGain(nid)) {
      dynamic_cast<ConditionNode*>(ast_node)->gain = tree.Gain(nid);
    }
    ast_node->children.push_back(BuildASTFromTree(tree, tree_id, tree.LeftChild(nid), ast_node));
    ast_node->children.push_back(BuildASTFromTree(tree, tree_id, tree.RightChild(nid), ast_node));
  }
  ast_node->node_id = nid;
  ast_node->tree_id = tree_id;
  if (tree.HasDataCount(nid)) {
    ast_node->data_count = tree.DataCount(nid);
  }
  if (tree.HasSumHess(nid)) {
    ast_node->sum_hess = tree.SumHess(nid);
  }

  return ast_node;
}

}  // namespace compiler
}  // namespace treelite
