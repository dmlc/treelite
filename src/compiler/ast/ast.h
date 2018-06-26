/*!
 * Copyright 2017 by Contributors
 * \file ast.h
 * \brief Definition for AST classes
 * \author Philip Cho
 */
#ifndef TREELITE_COMPILER_AST_AST_H_
#define TREELITE_COMPILER_AST_AST_H_

#include <string>
#include <vector>

namespace treelite {
namespace compiler {

/*!
 * \brief get string representation of comparsion operator
 * \param op comparison operator
 * \return string representation
 */
inline std::string OpName(Operator op) {
  switch (op) {
    case Operator::kEQ: return "==";
    case Operator::kLT: return "<";
    case Operator::kLE: return "<=";
    case Operator::kGT: return ">";
    case Operator::kGE: return ">=";
    default: return "";
  }
}

/*! \brief enum class to store branch annotation */
enum class BranchHint : uint8_t {
  kNone = 0,     /*!< no hint */
  kLikely = 1,   /*!< condition >50% likely */
  kUnlikely = 2  /*!< condition <50% likely */
};

inline std::string BranchHintName(BranchHint hint) {
  switch (hint) {
    case BranchHint::kNone:
      return "kNone";
    case BranchHint::kLikely:
      return "kLikely";
    case BranchHint::kUnlikely:
      return "kUnlikely";
  }
  LOG(FATAL) << "Unrecognized BranchHint value";
}

class ASTNode {
 public:
  ASTNode* parent;
  std::vector<ASTNode*> children;
  int node_id;
  int tree_id;
  int num_descendant;
  virtual void Dump(int indent) = 0;
};

class MainNode : public ASTNode {
 public:
  MainNode(tl_float global_bias, bool average_result, int num_tree,
           int num_feature)
    : global_bias(global_bias), average_result(average_result),
      num_tree(num_tree), num_feature(num_feature) {}
  void Dump(int indent) override {
    std::cerr << std::string(indent, ' ') << std::boolalpha
              << "MainNode {"
              << "global_bias: " << this->global_bias << ", "
              << "average_result: " << this->average_result << ", "
              << "num_tree: " << this->num_tree << ", "
              << "num_feature: " << this->num_feature << "}"
              << std::endl;
  }
  tl_float global_bias;
  bool average_result;
  int num_tree;
  int num_feature;
};

class TranslationUnitNode : public ASTNode {
 public:
  explicit TranslationUnitNode(int unit_id) : unit_id(unit_id) {}
  void Dump(int indent) override {
    std::cerr << std::string(indent, ' ')
              << "TranslationUnitNode {"
              << "unit_id: " << unit_id << "}"
              << std::endl;
  }
  int unit_id;
};

class QuantizerNode : public ASTNode {
 public:
  QuantizerNode(const std::vector<std::vector<tl_float>>& cut_pts,
                const std::vector<bool>& is_categorical)
    : cut_pts(cut_pts), is_categorical(is_categorical) {}
  QuantizerNode(std::vector<std::vector<tl_float>>&& cut_pts,
                std::vector<bool>&& is_categorical)
    : cut_pts(std::move(cut_pts)), is_categorical(std::move(is_categorical)) {}
  std::vector<std::vector<tl_float>> cut_pts;
  std::vector<bool> is_categorical;
  void Dump(int indent) override {
    std::cerr << std::string(indent, ' ')
              << "QuantizerNode = {}" << std::endl;
  }
};

class AccumulatorContextNode : public ASTNode {
 public:
  AccumulatorContextNode() {}
  void Dump(int indent) override {
    std::cerr << std::string(indent, ' ')
              << "AccumulatorContextNode = {}" << std::endl;
  }
};

class ConditionNode : public ASTNode {
 public:
  ConditionNode(unsigned split_index, bool default_left, BranchHint branch_hint)
    : split_index(split_index), default_left(default_left),
      branch_hint(branch_hint) {}
  unsigned split_index;
  bool default_left;
  BranchHint branch_hint;
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
                         ThresholdVariant threshold,
                         BranchHint branch_hint = BranchHint::kNone)
    : ConditionNode(split_index, default_left, branch_hint),
      quantized(quantized), op(op), threshold(threshold) {}
  void Dump(int indent) override {
    std::cerr << std::string(indent, ' ') << std::boolalpha
              << "NumericalConditionNode {"
              << "split_index: " << this->split_index << ", "
              << "default_left: " << this->default_left << ", "
              << "quantized: " << this->quantized << ", "
              << "op: " << OpName(this->op) << ", "
              << "threshold: " << (quantized ? this->threshold.int_val
                                             : this->threshold.float_val) << ", "
              << "branch_hint: " << BranchHintName(this->branch_hint)
              << "}" << std::endl;
  }
  bool quantized;
  Operator op;
  ThresholdVariant threshold;
};

class CategoricalConditionNode : public ConditionNode {
 public:
  CategoricalConditionNode(unsigned split_index, bool default_left,
                           const std::vector<uint32_t>& left_categories,
                           BranchHint branch_hint = BranchHint::kNone)
    : ConditionNode(split_index, default_left, branch_hint),
      left_categories(left_categories) {}
  void Dump(int indent) override {
    std::ostringstream oss;
    for (uint32_t e : this->left_categories) {
      oss << e << ", ";
    }
    std::cerr << std::string(indent, ' ') << std::boolalpha
              << "CategoricalConditionNode {"
              << "split_index: " << this->split_index << ", "
              << "default_left: " << this->default_left << ", "
              << "left_categories: [" << oss.str() << "], "
              << "branch_hint: " << BranchHintName(this->branch_hint)
              << "}" << std::endl;
  }
  std::vector<uint32_t> left_categories;
};

class OutputNode : public ASTNode {
 public:
  explicit OutputNode(tl_float scalar)
    : is_vector(false), scalar(scalar) {}
  explicit OutputNode(const std::vector<tl_float>& vector)
    : is_vector(true), vector(vector) {}
  void Dump(int indent) override {
    if (this->is_vector) {
      std::ostringstream oss;
      for (tl_float e : this->vector) {
        oss << e << ", ";
      }
      std::cerr << std::string(indent, ' ')
                << "OutputNode {vector: [" << oss.str() << "]}"
                << std::endl;
    } else {
      std::cerr << std::string(indent, ' ')
                << "OutputNode {scalar: " << this->scalar << "}"
                << std::endl;
    }
  }

  bool is_vector;
  tl_float scalar;
  std::vector<tl_float> vector;
};

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_AST_AST_H_
