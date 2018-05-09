/*!
 * Copyright 2017 by Contributors
 * \file build.cc
 * \brief Build AST from a given model
 */
#include "./builder.h"

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(build);

void ASTBuilder::BuildAST(const Model& model) {
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
    ASTNode* tree_head = BuildASTFromTree(model.trees[tree_id], ac);
    /* store tree ID in descendant nodes */
    std::function<void(ASTNode*)> func;
    func = [tree_id, &func](ASTNode* node) -> void {
      node->tree_id = tree_id;
      for (ASTNode* child : node->children) {
        func(child);
      }
    };
    func(tree_head);
    ac->children.push_back(tree_head);
  }
  this->model_param = model.param.__DICT__();
}

ASTNode* ASTBuilder::BuildASTFromTree(const Tree& tree, ASTNode* parent) {
  return BuildASTFromTree(tree, 0, parent);
}

ASTNode* ASTBuilder::BuildASTFromTree(const Tree& tree, int nid, ASTNode* parent) {
  const Tree::Node& node = tree[nid];
  ASTNode* ast_node = nullptr;
  if (node.is_leaf()) {
    if (this->output_vector_flag) {
      ast_node = AddNode<OutputNode>(parent, node.leaf_vector());
    } else {
      ast_node = AddNode<OutputNode>(parent, node.leaf_value());
    }
  } else {
    if (node.split_type() == SplitFeatureType::kNumerical) {
      ast_node = AddNode<NumericalConditionNode>(parent,
                                                 node.split_index(),
                                                 node.default_left(),
                                                 false,
                                                 node.comparison_op(),
                    ThresholdVariant(static_cast<tl_float>(node.threshold())));
    } else {
      ast_node = AddNode<CategoricalConditionNode>(parent,
                                                   node.split_index(),
                                                   node.default_left(),
                                                   node.left_categories());
    }
    ast_node->children.push_back(BuildASTFromTree(tree, node.cleft(), ast_node));
    ast_node->children.push_back(BuildASTFromTree(tree, node.cright(), ast_node));
  }
  ast_node->node_id = nid;

  return ast_node;
}

}  // namespace compiler
}  // namespace treelite
