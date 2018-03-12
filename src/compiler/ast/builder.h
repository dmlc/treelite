#ifndef TREELITE_COMPILER_AST_BUILDER_H_
#define TREELITE_COMPILER_AST_BUILDER_H_

#include <treelite/common.h>
#include <treelite/tree.h>
#include "./ast.h"

namespace treelite {
namespace compiler {

class ASTBuilder {
 public:
  ASTBuilder() : output_vector_flag(false), main_node(nullptr),
                 quantize_threshold_flag(false) {}

  void Build(const Model& model);
  void Split(int parallel_comp);
  void QuantizeThresholds();
  void CountDescendant();
  void BreakUpLargeUnits(int num_descendant_limit);
  void Dump();

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
  ASTNode* WalkTree(const Tree& tree, ASTNode* parent);
  ASTNode* WalkTree(const Tree& tree, int nid, ASTNode* parent);

  // keep tract of all nodes built so far, to prevent memory leak
  std::vector<std::unique_ptr<ASTNode>> nodes;
  bool output_vector_flag;
  bool quantize_threshold_flag;
  int num_feature;
  ASTNode* main_node;
};

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_AST_BUILDER_H_