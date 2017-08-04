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

enum class LikelyDirection : uint8_t {
  kNone = 0, kLeft = 1, kRight = 2
};

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
  CLONEABLE_BOILERPLATE(PlainBlock)
  std::vector<std::string> Compile() const override;

 private:
  std::vector<std::string> inner_text;
};

class FunctionEntry {
 public:
  static const std::vector<std::string>& GetRegistry() {
    return registry;
  }
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
    FunctionEntry::Register(this->prototype);
  }
  explicit FunctionBlock(std::string&& prototype,
                         CodeBlock&& body)
    : prototype(std::move(prototype)), body(std::move(body)) {
    FunctionEntry::Register(this->prototype);
  }
  CLONEABLE_BOILERPLATE(FunctionBlock)
  std::vector<std::string> Compile() const override;

 private:
  std::string prototype;
  DeepCopyUniquePtr<CodeBlock> body;
};

class SequenceBlock : public CodeBlock {
 public:
  explicit SequenceBlock() = default;
  CLONEABLE_BOILERPLATE(SequenceBlock)
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

class IfElseBlock : public CodeBlock {
 public:
  explicit IfElseBlock(const Condition& condition,
                       const CodeBlock& if_block,
                       const CodeBlock& else_block,
                       LikelyDirection direction = LikelyDirection::kNone)
    : condition(condition), if_block(if_block), else_block(else_block),
      likely_direction(direction) {}
  explicit IfElseBlock(Condition&& condition,
                       CodeBlock&& if_block,
                       CodeBlock&& else_block,
                       LikelyDirection direction = LikelyDirection::kNone)
    : condition(std::move(condition)),
      if_block(std::move(if_block)), else_block(std::move(else_block)),
      likely_direction(direction) {}
  CLONEABLE_BOILERPLATE(IfElseBlock)
  std::vector<std::string> Compile() const override;

 private:
  DeepCopyUniquePtr<Condition> condition;
  DeepCopyUniquePtr<CodeBlock> if_block;
  DeepCopyUniquePtr<CodeBlock> else_block;
  LikelyDirection likely_direction;
};

}  // namespace semantic
}  // namespace treelite
#endif  // TREELITE_SEMANTIC_H_
