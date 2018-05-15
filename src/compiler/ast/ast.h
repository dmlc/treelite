#ifndef TREELITE_COMPILER_AST_AST_H_
#define TREELITE_COMPILER_AST_AST_H_

#include <treelite/base.h>
#include <dmlc/optional.h>

// forward declaration
namespace treelite_ast_protobuf {
class ASTNode;
}  // namespace treelite_ast_protobuf

namespace treelite {
namespace compiler {

/*!
 * \brief get string representation of comparsion operator
 * \param op comparison operator
 * \return string representation
 */
inline std::string OpName(Operator op) {
  switch(op) {
    case Operator::kEQ: return "==";
    case Operator::kLT: return "<";
    case Operator::kLE: return "<=";
    case Operator::kGT: return ">";
    case Operator::kGE: return ">=";
    default: return "";
  }
}

class ASTNode {
 public:
  ASTNode* parent;
  std::vector<ASTNode*> children;
  int node_id;
  int tree_id;
  dmlc::optional<int> num_descendant_ast_node;
  dmlc::optional<size_t> data_count;
  dmlc::optional<double> sum_hess;
  virtual void Serialize(treelite_ast_protobuf::ASTNode* out) = 0;
 protected:
  ASTNode() : parent(nullptr), node_id(-1), tree_id(-1) {}
};

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
  void Serialize(treelite_ast_protobuf::ASTNode* out) override;
};

class TranslationUnitNode : public ASTNode {
 public:
  TranslationUnitNode(int unit_id) : unit_id(unit_id) {}
  int unit_id;
  void Serialize(treelite_ast_protobuf::ASTNode* out) override;
};

class QuantizerNode : public ASTNode {
 public:
  QuantizerNode(const std::vector<std::vector<tl_float>>& cut_pts)
    : cut_pts(cut_pts) {}
  QuantizerNode(std::vector<std::vector<tl_float>>&& cut_pts)
    : cut_pts(std::move(cut_pts)) {}
  std::vector<std::vector<tl_float>> cut_pts;
  void Serialize(treelite_ast_protobuf::ASTNode* out) override;
};

class AccumulatorContextNode : public ASTNode {
 public:
  AccumulatorContextNode() {}
  void Serialize(treelite_ast_protobuf::ASTNode* out) override;
};

class CodeFolderNode : public ASTNode {
 public:
  CodeFolderNode() {}
  void Serialize(treelite_ast_protobuf::ASTNode* out) override;
};

class ConditionNode : public ASTNode {
 public:
  ConditionNode(unsigned split_index, bool default_left)
    : split_index(split_index), default_left(default_left) {}
  unsigned split_index;
  bool default_left;
  dmlc::optional<double> gain;
  virtual void Serialize(treelite_ast_protobuf::ASTNode* out) = 0;
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
  void Serialize(treelite_ast_protobuf::ASTNode* out) override;
};

class CategoricalConditionNode : public ConditionNode {
 public:
  CategoricalConditionNode(unsigned split_index, bool default_left,
                           const std::vector<uint32_t>& left_categories)
    : ConditionNode(split_index, default_left),
      left_categories(left_categories) {}
  std::vector<uint32_t> left_categories;
  void Serialize(treelite_ast_protobuf::ASTNode* out) override;
};

class OutputNode : public ASTNode {
 public:
  OutputNode(tl_float scalar)
    : is_vector(false), scalar(scalar) {}
  OutputNode(const std::vector<tl_float>& vector)
    : is_vector(true), vector(vector) {}
  bool is_vector;
  tl_float scalar;
  std::vector<tl_float> vector;
  void Serialize(treelite_ast_protobuf::ASTNode* out) override;
};

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_AST_AST_H_
