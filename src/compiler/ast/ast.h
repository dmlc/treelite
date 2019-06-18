/*!
 * Copyright 2017 by Contributors
 * \file ast.h
 * \brief Definition for AST classes
 * \author Philip Cho
 */
#ifndef TREELITE_COMPILER_AST_AST_H_
#define TREELITE_COMPILER_AST_AST_H_

#include <dmlc/optional.h>
#include <treelite/base.h>
#include <string>
#include <vector>
#include <utility>

// forward declaration
namespace treelite_ast_protobuf {
class ASTNode;
}  // namespace treelite_ast_protobuf

namespace treelite {
namespace compiler {

class ASTNode {
 public:
  ASTNode* parent;
  std::vector<ASTNode*> children;
  int node_id;
  int tree_id;
  dmlc::optional<size_t> data_count;
  dmlc::optional<double> sum_hess;
  virtual ~ASTNode() = 0;  // force ASTNode to be abstract class
 protected:
  ASTNode() : parent(nullptr), node_id(-1), tree_id(-1) {}
};
inline ASTNode::~ASTNode() {}

class MainNode : public ASTNode {
 public:
  MainNode(tl_float global_bias, bool average_result, int num_tree,
           int num_feature)
    : global_bias(global_bias), average_result(average_result),
      num_tree(num_tree), num_feature(num_feature) {}
  tl_float global_bias;
  bool average_result;
  int num_tree;
  int num_feature;
};

class TranslationUnitNode : public ASTNode {
 public:
  explicit TranslationUnitNode(int unit_id) : unit_id(unit_id) {}
  int unit_id;
};

class QuantizerNode : public ASTNode {
 public:
  explicit QuantizerNode(const std::vector<std::vector<tl_float>>& cut_pts)
    : cut_pts(cut_pts) {}
  explicit QuantizerNode(std::vector<std::vector<tl_float>>&& cut_pts)
    : cut_pts(std::move(cut_pts)) {}
  std::vector<std::vector<tl_float>> cut_pts;
};

class AccumulatorContextNode : public ASTNode {
 public:
  AccumulatorContextNode() {}
};

class CodeFolderNode : public ASTNode {
 public:
  CodeFolderNode() {}
};

class ConditionNode : public ASTNode {
 public:
  ConditionNode(unsigned split_index, bool default_left)
    : split_index(split_index), default_left(default_left) {}
  unsigned split_index;
  bool default_left;
  dmlc::optional<double> gain;
};

union ThresholdVariant {
  tl_float float_val;
  int int_val;
  ThresholdVariant(tl_float val) : float_val(val) {}
  ThresholdVariant(int val) : int_val(val) {}
};

class NumericalConditionNode : public ConditionNode {
 public:
  NumericalConditionNode(unsigned split_index, bool default_left,
                         bool quantized, Operator op,
                         ThresholdVariant threshold)
    : ConditionNode(split_index, default_left),
      quantized(quantized), op(op), threshold(threshold) {}
  bool quantized;
  Operator op;
  ThresholdVariant threshold;
};

class CategoricalConditionNode : public ConditionNode {
 public:
  CategoricalConditionNode(unsigned split_index, bool default_left,
                           const std::vector<uint32_t>& left_categories,
                           bool convert_missing_to_zero)
    : ConditionNode(split_index, default_left),
      left_categories(left_categories),
      convert_missing_to_zero(convert_missing_to_zero) {}
  std::vector<uint32_t> left_categories;
  bool convert_missing_to_zero;
};

class OutputNode : public ASTNode {
 public:
  explicit OutputNode(tl_float scalar)
    : is_vector(false), scalar(scalar) {}
  explicit OutputNode(const std::vector<tl_float>& vector)
    : is_vector(true), vector(vector) {}
  bool is_vector;
  tl_float scalar;
  std::vector<tl_float> vector;
};

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_AST_AST_H_
