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
struct CodeFoldingContext;
bool fold_code(ASTNode*, CodeFoldingContext*, ASTBuilder*);
bool breakup(ASTNode*, int, int*, ASTBuilder*);

class ASTBuilder {
 public:
  ASTBuilder() : output_vector_flag(false), main_node(nullptr),
                 quantize_threshold_flag(false) {}

  /* \brief initially build AST from model */
  void BuildAST(const Model& model);
  /* \brief generate is_categorical[] array, which tells whether each feature
            is categorical or numerical */
  std::vector<bool> GenerateIsCategoricalArray();
  /*
   * \brief fold rarely visited subtrees into tight loops (don't produce
   *        if/else blocks). Rarity of each node is determined by its
   *        data count and/or hessian sum: any node is "rare" if its data count
   *        or hessian sum is lower than the proscribed threshold.
   * \param data_count_magnitude_req all nodes whose data counts are lower
   *                                 than that of the root node of the decision
   *                                 tree by [data_count_magnitude_req] will be
   *                                 folded. To diable folding, set to +inf.
   * \param sum_hess_magnitude_req all nodes whose hessian sums are lower than
   *                               that of the root node of the decision tree
   *                               by [sum_hess_magnitude_req] will be folded.
   *                               To diable folding, set to +inf.
   * \param create_new_translation_unit if true, place folded loops in
   *                                    separate translation units
   * \param whether at least one subtree was folded
   */
  bool FoldCode(double data_count_magnitude_req, double sum_hess_magnitude_req,
                bool create_new_translation_unit = false);
  /*
   * \brief split prediction function into multiple translation units
   * \param parallel_comp number of translation units
   */
  void Split(int parallel_comp);
  /* \brief replace split thresholds with integers */
  void QuantizeThresholds();
  /* \brief call this function before BreakUpLargeTranslationUnits() */
  void CountDescendant();
  /*
   * \brief break up large translation units, to keep 64K bytecode size limit
   *        in Java
   * \param num_descendant_limit max number of AST nodes that are allowed to
   *                             be in each translation unit
   */
  void BreakUpLargeTranslationUnits(int num_descendant_limit);
  /* \brief call this function before BreakUpLargeTranslationUnits() */
  void LoadDataCounts(const std::vector<std::vector<size_t>>& counts);
  /* \brief serialize to output stream. This function uses Protobuf. */
  void Serialize(std::ostream* output, bool binary = true);
  /* \brief serialize to a file. This function uses Protobuf. */
  void Serialize(const std::string& filename, bool binary = true);

  inline const ASTNode* GetRootNode() {
    return main_node;
  }

 private:
  friend bool treelite::compiler::breakup(ASTNode*, int, int*, ASTBuilder*);
  friend bool treelite::compiler::fold_code(ASTNode*, CodeFoldingContext*,
                                            ASTBuilder*);

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
  std::vector<bool> is_categorical;
  std::map<std::string, std::string> model_param;
};

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_AST_BUILDER_H_
