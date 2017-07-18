/*!
 * Copyright 2017 by Contributors
 * \file semantic.h
 * \brief Building blocks for semantic model of tree prediction code
 * \author Philip Cho
 */
#ifndef TREELITE_SEMANTIC_H_
#define TREELITE_SEMANTIC_H_

#include <treelite/tree.h>
#include <algorithm>

namespace treelite {
namespace semantic {

inline std::string OpName(Tree::Operator op) {
  switch(op) {
    case Tree::Operator::kEQ: return "==";
    case Tree::Operator::kLT: return "<";
    case Tree::Operator::kLE: return "<=";
    case Tree::Operator::kGT: return ">";
    case Tree::Operator::kGE: return ">=";
    default: return "";
  }
}

inline void TransformPushBack(std::vector<std::string>* p_dest,
                              const std::vector<std::string>& lines,
                              std::function<std::string(std::string)> func) {
  auto& dest = *p_dest;
  std::transform(lines.begin(), lines.end(), std::back_inserter(dest), func);
}

using common::Cloneable;
using common::DeepCopyUniquePtr;

class FeatureAdapter : public Cloneable {
 public:
  virtual ~FeatureAdapter() = default;
  virtual std::string Compile(bool default_left, unsigned split_index) const = 0;
};

class NumericAdapter : public Cloneable {
 public:
  virtual ~NumericAdapter() = default;
  virtual std::string Compile(tl_float numeric) const = 0;
};

class CodeBlock : public Cloneable {
 public:
  virtual ~CodeBlock() = default;
  virtual std::vector<std::string> Compile() const = 0;
};

class PlainBlock : public CodeBlock {
 public:
  explicit PlainBlock(const std::string& inner_text)
    : inner_text({inner_text}) {}
  explicit PlainBlock(const std::vector<std::string>& inner_text)
    : inner_text(inner_text) {}
  explicit PlainBlock(std::vector<std::string>&& inner_text)
    : inner_text(std::move(inner_text)) {}
  explicit PlainBlock(const PlainBlock& other) = default;
  explicit PlainBlock(PlainBlock&& other) = default;
  Cloneable* clone() const override {
    return new PlainBlock(*this);
  }
  Cloneable* move_clone() override {
    return new PlainBlock(std::move(*this));
  }
  std::vector<std::string> Compile() const override;

 private:
  std::vector<std::string> inner_text;
};

class FunctionEntry {
 protected:
  FunctionEntry() = default;
  inline void Register(const std::string& prototype) {
    registry.push_back(prototype);
  }
 private:
  static std::vector<std::string> registry;
};

class FunctionBlock : public CodeBlock, private FunctionEntry {
 public:
  explicit FunctionBlock(const std::string& prototype,
                         const CodeBlock& body)
    : prototype(prototype), body(body) {
    FunctionEntry::Register(prototype);
  }
  explicit FunctionBlock(std::string&& prototype,
                         CodeBlock&& body)
    : prototype(std::move(prototype)), body(std::move(body)) {
    FunctionEntry::Register(prototype);
  }
  explicit FunctionBlock(const FunctionBlock& other) = default;
  explicit FunctionBlock(FunctionBlock&& other) = default;
  Cloneable* clone() const override {
    return new FunctionBlock(*this);
  }
  Cloneable* move_clone() override {
    return new FunctionBlock(std::move(*this));
  }
  std::vector<std::string> Compile() const override;

 private:
  std::string prototype;
  DeepCopyUniquePtr<CodeBlock> body;
};

class SequenceBlock : public CodeBlock {
 public:
  explicit SequenceBlock() = default;
  explicit SequenceBlock(const SequenceBlock& other) = default;
  explicit SequenceBlock(SequenceBlock&& other) = default;
  Cloneable* clone() const override {
    return new SequenceBlock(*this);
  }
  Cloneable* move_clone() override {
    return new SequenceBlock(std::move(*this));
  }
  std::vector<std::string> Compile() const override;
  void Reserve(size_t size);
  void PushBack(const CodeBlock& block);
  void PushBack(CodeBlock&& block);

 private:
  std::vector<DeepCopyUniquePtr<CodeBlock>> sequence;
};

class Condition : public Cloneable {
 public:
  virtual ~Condition() = default;
  virtual std::string Compile() const = 0;
};

class Accumulator : public Cloneable {
 public:
  virtual ~Accumulator() = default;
  virtual std::string Compile(tl_float leaf_value) const = 0;
};

class SplitCondition : public Condition {
 public:
  explicit SplitCondition(const Tree::Node& node,
                          const FeatureAdapter& feature_adapter,
                          const NumericAdapter& numeric_adapter)
   : split_index(node.split_index()), default_left(node.default_left()),
     op(node.comparison_op()), threshold(node.threshold()),
     feature_adapter(feature_adapter),
     numeric_adapter(numeric_adapter) {}
  explicit SplitCondition(const Tree::Node& node,
                          FeatureAdapter&& feature_adapter,
                          NumericAdapter&& numeric_adapter)
   : split_index(node.split_index()), default_left(node.default_left()),
     op(node.comparison_op()), threshold(node.threshold()),
     feature_adapter(std::move(feature_adapter)),
     numeric_adapter(std::move(numeric_adapter)) {}
  explicit SplitCondition(const SplitCondition& other) = default;
  explicit SplitCondition(SplitCondition&& other) = default;
  Cloneable* clone() const override {
    return new SplitCondition(*this);
  }
  Cloneable* move_clone() override {
    return new SplitCondition(std::move(*this));
  }
  std::string Compile() const override;
 private:
  unsigned split_index;
  bool default_left;
  Tree::Operator op;
  tl_float threshold;
  DeepCopyUniquePtr<FeatureAdapter> feature_adapter;
  DeepCopyUniquePtr<NumericAdapter> numeric_adapter;
};

class SimpleAccumulator : public Accumulator {
 public:
  explicit SimpleAccumulator(const std::string& acc_name,
                             const NumericAdapter& numeric_adapter)
    : acc_name(acc_name), numeric_adapter(numeric_adapter) {}
  explicit SimpleAccumulator(std::string&& acc_name,
                             NumericAdapter&& numeric_adapter)
    : acc_name(std::move(acc_name)),
      numeric_adapter(std::move(numeric_adapter)) {}
  explicit SimpleAccumulator(const SimpleAccumulator& other) = default;
  explicit SimpleAccumulator(SimpleAccumulator&& other) = default;
  Cloneable* clone() const override {
    return new SimpleAccumulator(*this);
  }
  Cloneable* move_clone() override {
    return new SimpleAccumulator(std::move(*this));
  }
  std::string Compile(tl_float leaf_value) const override;

 private:
  std::string acc_name;
  DeepCopyUniquePtr<NumericAdapter> numeric_adapter;
};

class IfElseBlock : public CodeBlock {
 public:
  explicit IfElseBlock(const Condition& condition,
                       const CodeBlock& if_block,
                       const CodeBlock& else_block)
    : condition(condition), if_block(if_block), else_block(else_block) {}
  explicit IfElseBlock(Condition&& condition,
                       CodeBlock&& if_block,
                       CodeBlock&& else_block)
    : condition(std::move(condition)),
      if_block(std::move(if_block)), else_block(std::move(else_block)) {}
  explicit IfElseBlock(const IfElseBlock& other) = default;
  explicit IfElseBlock(IfElseBlock&& other) = default;
  Cloneable* clone() const override {
    return new IfElseBlock(*this);
  }
  Cloneable* move_clone() override {
    return new IfElseBlock(std::move(*this));
  }
  std::vector<std::string> Compile() const override;

 private:
  DeepCopyUniquePtr<Condition> condition;
  DeepCopyUniquePtr<CodeBlock> if_block;
  DeepCopyUniquePtr<CodeBlock> else_block;
};

class SimpleNumeric : public NumericAdapter {
 public:
  explicit SimpleNumeric() = default;
  explicit SimpleNumeric(const SimpleNumeric& other) = default;
  explicit SimpleNumeric(SimpleNumeric&& other) = default;
  Cloneable* clone() const override {
    return new SimpleNumeric(*this);
  }
  Cloneable* move_clone() override {
    return new SimpleNumeric(std::move(*this));
  }
  std::string Compile(tl_float numeric) const override;
};

class DenseFeature : public FeatureAdapter {
 public:
  explicit DenseFeature(const std::string& bitmap_name,
                        const std::string& array_name)
   : array_name(array_name), bitmap_name(bitmap_name) {}
  explicit DenseFeature(const DenseFeature& other) = default;
  explicit DenseFeature(DenseFeature&& other) = default;
  Cloneable* clone() const override {
    return new DenseFeature(*this);
  }
  Cloneable* move_clone() override {
    return new DenseFeature(std::move(*this));
  }
  std::string Compile(bool default_left, unsigned split_index) const override;

 private:
  std::string array_name;
  std::string bitmap_name;
};

class CompressedDenseFeature : public FeatureAdapter {
 public:
  explicit CompressedDenseFeature(const std::string& bitmap_name,
                                  const std::string& accessor_name,
                                  const std::string& array_name)
   : array_name(array_name), bitmap_name(bitmap_name),
     accessor_name(accessor_name) {}
  explicit CompressedDenseFeature(const CompressedDenseFeature& other)
    = default;
  explicit CompressedDenseFeature(CompressedDenseFeature&& other) = default;
  Cloneable* clone() const override {
    return new CompressedDenseFeature(*this);
  }
  Cloneable* move_clone() override {
    return new CompressedDenseFeature(std::move(*this));
  }
  std::string Compile(bool default_left, unsigned split_index) const override;

 private:
  std::string array_name;
  std::string bitmap_name;
  std::string accessor_name;
};

class SparseFeature : public FeatureAdapter {
 public:
  explicit SparseFeature(const std::string& nonzero_name,
                         const std::string& col_ind_name,
                         const std::string& accessor_name)
   : nonzero_name(nonzero_name), col_ind_name(col_ind_name),
     accessor_name(accessor_name) {}
  explicit SparseFeature(const SparseFeature& other) = default;
  explicit SparseFeature(SparseFeature&& other) = default;
  Cloneable* clone() const override {
    return new SparseFeature(*this);
  }
  Cloneable* move_clone() override {
    return new SparseFeature(std::move(*this));
  }
  std::string Compile(bool default_left, unsigned split_index) const override;

 private:
  std::string nonzero_name;
  std::string col_ind_name;
  std::string accessor_name;
};

}  // namespace semantic
}  // namespace treelite
#endif  // TREELITE_SEMANTIC_H_
