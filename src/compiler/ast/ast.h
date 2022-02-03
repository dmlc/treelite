/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file ast.h
 * \brief Definition for AST classes
 * \author Hyunsu Cho
 */
#ifndef TREELITE_COMPILER_AST_AST_H_
#define TREELITE_COMPILER_AST_AST_H_

#include <treelite/optional.h>
#include <treelite/base.h>
#include <fmt/format.h>
#include <limits>
#include <string>
#include <vector>
#include <utility>
#include <cstdint>

namespace treelite {
namespace compiler {

class ASTNode {
 public:
  ASTNode* parent;
  std::vector<ASTNode*> children;
  int node_id;
  int tree_id;
  optional<std::uint64_t> data_count;
  optional<double> sum_hess;
  virtual std::string GetDump() const = 0;
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

  std::string GetDump() const override {
    return fmt::format("MainNode {{ global_bias: {}, average_result: {}, num_tree: {}, "
                       "num_feature: {} }}", global_bias, average_result, num_tree, num_feature);
  }
};

class TranslationUnitNode : public ASTNode {
 public:
  explicit TranslationUnitNode(int unit_id) : unit_id(unit_id) {}
  int unit_id;

  std::string GetDump() const override {
    return fmt::format("TranslationUnitNode {{ unit_id: {} }}", unit_id);
  }
};

template <typename ThresholdType>
class QuantizerNode : public ASTNode {
 public:
  explicit QuantizerNode(const std::vector<std::vector<ThresholdType>>& cut_pts)
    : cut_pts(cut_pts) {}
  explicit QuantizerNode(std::vector<std::vector<ThresholdType>>&& cut_pts)
    : cut_pts(std::move(cut_pts)) {}
  std::vector<std::vector<ThresholdType>> cut_pts;

  std::string GetDump() const override {
    std::ostringstream oss;
    for (const auto& vec : cut_pts) {
      oss << "[ ";
      for (const auto& e : vec) {
        oss << e << ", ";
      }
      oss << "], ";
    }
    return fmt::format("QuantizerNode {{ cut_pts: {} }}", oss.str());
  }
};

class AccumulatorContextNode : public ASTNode {
 public:
  AccumulatorContextNode() {}

  std::string GetDump() const override {
    return fmt::format("AccumulatorContextNode {{}}");
  }
};

class CodeFolderNode : public ASTNode {
 public:
  CodeFolderNode() {}

  std::string GetDump() const override {
    return fmt::format("CodeFolderNode {{}}");
  }
};

class ConditionNode : public ASTNode {
 public:
  ConditionNode(unsigned split_index, bool default_left)
    : split_index(split_index), default_left(default_left) {}
  unsigned split_index;
  bool default_left;
  optional<double> gain;

  std::string GetDump() const override {
    if (gain) {
      return fmt::format("ConditionNode {{ split_index: {}, default_left: {}, gain: {} }}",
                         split_index, default_left, *gain);
    } else {
      return fmt::format("ConditionNode {{ split_index: {}, default_left: {} }}",
                         split_index, default_left);
    }
  }
};

template <typename ThresholdType>
union ThresholdVariant {
  ThresholdType float_val;
  int int_val;
  explicit ThresholdVariant(ThresholdType val) : float_val(val) {}
  explicit ThresholdVariant(int val) : int_val(val) {}
};

template <typename ThresholdType>
class NumericalConditionNode : public ConditionNode {
 public:
  NumericalConditionNode(unsigned split_index, bool default_left,
                         bool quantized, Operator op,
                         ThresholdVariant<ThresholdType> threshold)
    : ConditionNode(split_index, default_left),
      quantized(quantized), op(op), threshold(threshold), zero_quantized(-1) {}
  bool quantized;
  Operator op;
  ThresholdVariant<ThresholdType> threshold;
  int zero_quantized;  // quantized value of 0.0f (useful when convert_missing_to_zero is set)

  std::string GetDump() const override {
    return fmt::format("NumericalConditionNode {{ {}, quantized: {}, op: {}, threshold: {}, "
                       "zero_quantized: {} }}",
                       ConditionNode::GetDump(), quantized, OpName(op),
                       (quantized ? fmt::format("{}", threshold.int_val)
                                  : fmt::format("{}", threshold.float_val)),
                       zero_quantized);
  }
};

class CategoricalConditionNode : public ConditionNode {
 public:
  CategoricalConditionNode(unsigned split_index, bool default_left,
                           const std::vector<std::uint32_t>& matching_categories,
                           bool categories_list_right_child)
    : ConditionNode(split_index, default_left),
      matching_categories(matching_categories),
      categories_list_right_child(categories_list_right_child) {}
  std::vector<std::uint32_t> matching_categories;
  bool categories_list_right_child;

  std::string GetDump() const override {
    std::ostringstream oss;
    oss << "[";
    for (const auto& e : matching_categories) {
      oss << e << ", ";
    }
    oss << "]";
    return fmt::format("CategoricalConditionNode {{ {}, matching_categories: {}, "
                       "categories_list_right_child: {} }}",
                       ConditionNode::GetDump(), oss.str(), categories_list_right_child);
  }
};

template <typename LeafOutputType>
class OutputNode : public ASTNode {
 public:
  explicit OutputNode(LeafOutputType scalar)
    : is_vector(false), scalar(scalar) {}
  explicit OutputNode(const std::vector<LeafOutputType>& vector)
    : is_vector(true), vector(vector) {}
  bool is_vector;
  LeafOutputType scalar;
  std::vector<LeafOutputType> vector;

  std::string GetDump() const override {
    if (is_vector) {
      std::ostringstream oss;
      oss << "[";
      for (const auto& e : vector) {
        oss << e << ", ";
      }
      oss << "]";
      return fmt::format("OutputNode {{ is_vector: {}, vector {} }}", is_vector, oss.str());
    } else {
      return fmt::format("OutputNode {{ is_vector: {}, scalar: {} }}", is_vector, scalar);
    }
  }
};

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_AST_AST_H_
