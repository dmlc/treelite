/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file builder.h
 * \brief AST Builder class
 */
#ifndef TREELITE_COMPILER_AST_BUILDER_H_
#define TREELITE_COMPILER_AST_BUILDER_H_

#include <treelite/tree.h>
#include <map>
#include <string>
#include <vector>
#include <ostream>
#include <utility>
#include <memory>
#include <cstdint>
#include "./ast.h"

namespace treelite {
namespace compiler {

// forward declarations
template <typename ThresholdType, typename LeafOutputType>
class ASTBuilder;
struct CodeFoldingContext;
template <typename ThresholdType, typename LeafOutputType>
bool fold_code(ASTNode*, CodeFoldingContext*, ASTBuilder<ThresholdType, LeafOutputType>*);

template <typename ThresholdType, typename LeafOutputType>
class ASTBuilder {
 public:
  ASTBuilder() : output_vector_flag(false), quantize_threshold_flag(false), main_node(nullptr) {}

  /* \brief initially build AST from model */
  void BuildAST(const ModelImpl<ThresholdType, LeafOutputType>& model);
  /* \brief generate is_categorical[] array, which tells whether each feature
            is categorical or numerical */
  std::vector<bool> GenerateIsCategoricalArray();
  /*
   * \brief fold rarely visited subtrees into tight loops (don't produce
   *        if/else blocks). Rarity of each node is determined by its
   *        data count and/or hessian sum: any node is "rare" if its data count
   *        or hessian sum is lower than the proscribed threshold.
   * \param magnitude_req all nodes whose data counts are lower than that of
   *                      the root node of the decision tree by [magnitude_req]
   *                      will be folded. To diable folding, set to +inf. If
   *                      hessian sums are available instead of data counts,
   *                      hessian sums will be used as a proxy of data counts
   * \param create_new_translation_unit if true, place folded loops in
   *                                    separate translation units
   * \param whether at least one subtree was folded
   */
  bool FoldCode(double magnitude_req, bool create_new_translation_unit = false);
  /*
   * \brief split prediction function into multiple translation units
   * \param parallel_comp number of translation units
   */
  void Split(int parallel_comp);
  /* \brief replace split thresholds with integers */
  void QuantizeThresholds();
  /* \brief Load data counts from annotation file */
  void LoadDataCounts(const std::vector<std::vector<std::uint64_t>>& counts);
  /*
   * \brief Get a text representation of AST
   */
  std::string GetDump() const;

  inline const ASTNode* GetRootNode() {
    return main_node;
  }

 private:
  friend bool treelite::compiler::fold_code<>(ASTNode*, CodeFoldingContext*,
                                              ASTBuilder<ThresholdType, LeafOutputType>*);

  template <typename NodeType, typename ...Args>
  NodeType* AddNode(ASTNode* parent, Args&& ...args) {
    std::unique_ptr<NodeType> node(new NodeType(std::forward<Args>(args)...));
    NodeType* ref = node.get();
    ref->parent = parent;
    nodes.push_back(std::move(node));
    return ref;
  }

  ASTNode* BuildASTFromTree(const Tree<ThresholdType, LeafOutputType>& tree, int tree_id, int nid,
                            ASTNode* parent);

  // keep tract of all nodes built so far, to prevent memory leak
  std::vector<std::unique_ptr<ASTNode>> nodes;
  bool output_vector_flag;
  bool quantize_threshold_flag;
  int num_feature;
  bool average_output_flag;
  ASTNode* main_node;
  std::vector<bool> is_categorical;
  std::map<std::string, std::string> model_param;
};

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_AST_BUILDER_H_
