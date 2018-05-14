/*!
 * Copyright 2017 by Contributors
 * \file builder.h
 * \brief AST Builder class
 */
#ifndef TREELITE_COMPILER_AST_BUILDER_H_
#define TREELITE_COMPILER_AST_BUILDER_H_

#include <treelite/common.h>
#include <treelite/tree.h>
#include <vector>
#include <ostream>
#include "./ast.h"

namespace treelite {
namespace compiler {

// forward declaration
class ASTBuilder;
bool breakup(ASTNode*, int, int*, ASTBuilder*);

class ASTBuilder {
 public:
  ASTBuilder() : output_vector_flag(false), main_node(nullptr),
                 quantize_threshold_flag(false) {}

  void BuildAST(const Model& model);
  void Split(int parallel_comp);
  void QuantizeThresholds();
  void CountDescendant();
  void BreakUpLargeUnits(int num_descendant_limit);
  void AnnotateBranches(const std::vector<std::vector<size_t>>& counts);
  void Serialize(std::ostream* output);
  void Serialize(const std::string& filename);

  inline const ASTNode* GetRootNode() {
    return main_node;
  }

 private:
  friend bool treelite::compiler::breakup(ASTNode*, int, int*, ASTBuilder*);

  template <typename NodeType, typename ...Args>
  NodeType* AddNode(ASTNode* parent, Args&& ...args) {
    std::unique_ptr<NodeType> node
                  = common::make_unique<NodeType>(std::forward<Args>(args)...);
    NodeType* ref = node.get();
    ref->parent = parent;
    nodes.push_back(std::move(node));
    return ref;
  }
  ASTNode* BuildASTFromTree(const Tree& tree, int tree_id, ASTNode* parent);
  ASTNode* BuildASTFromTree(const Tree& tree, int tree_id, int nid,
                            ASTNode* parent);

  // keep tract of all nodes built so far, to prevent memory leak
  std::vector<std::unique_ptr<ASTNode>> nodes;
  bool output_vector_flag;
  bool quantize_threshold_flag;
  int num_feature;
  int num_output_group;
  bool random_forest_flag;
  ASTNode* main_node;
  std::map<std::string, std::string> model_param;
};

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_AST_BUILDER_H_
