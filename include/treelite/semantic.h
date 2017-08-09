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

/*! \brief enum class to store branch annotation */
enum class BranchHint : uint8_t {
  kNone = 0,     /*!< no hint */
  kLikely = 1,   /*!< condition >50% likely */
  kUnlikely = 2  /*!< condition <50% likely */
};

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

using common::Cloneable;
using common::DeepCopyUniquePtr;

class CodeBlock : public Cloneable {
 public:
  virtual ~CodeBlock() = default;
  virtual std::vector<std::string> Compile() const = 0;
};

class TranslationUnit {
 public:
  explicit TranslationUnit(const CodeBlock& preamble, const CodeBlock& body)
    : preamble(preamble), body(body) {}
  explicit TranslationUnit(CodeBlock&& preamble, CodeBlock&& body)
    : preamble(std::move(preamble)), body(std::move(body)) {}
  explicit TranslationUnit(const TranslationUnit& other) = delete;
  explicit TranslationUnit(TranslationUnit&& other) = default;
  std::vector<std::string> Compile(const std::string& header_filename) const;
 private:
  DeepCopyUniquePtr<CodeBlock> preamble;
  DeepCopyUniquePtr<CodeBlock> body;
};

struct SemanticModel {
  std::unique_ptr<CodeBlock> common_header;
  std::vector<std::string> function_registry;
  std::vector<TranslationUnit> units;
};

class PlainBlock : public CodeBlock {
 public:
  explicit PlainBlock()
    : inner_text({}) {}
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

class FunctionBlock : public CodeBlock {
 public:
  explicit FunctionBlock(const std::string& prototype,
                         const CodeBlock& body,
                         std::vector<std::string>* p_function_registry)
    : prototype(prototype), body(body) {
    if (p_function_registry != nullptr) {
      p_function_registry->push_back(this->prototype);
    }
  }
  explicit FunctionBlock(std::string&& prototype,
                         CodeBlock&& body,
                         std::vector<std::string>* p_function_registry)
    : prototype(std::move(prototype)), body(std::move(body)) {
    if (p_function_registry != nullptr) {
      p_function_registry->push_back(this->prototype);
    }
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
                       BranchHint hint = BranchHint::kNone)
    : condition(condition), if_block(if_block), else_block(else_block),
      branch_hint(hint) {}
  explicit IfElseBlock(Condition&& condition,
                       CodeBlock&& if_block,
                       CodeBlock&& else_block,
                       BranchHint hint = BranchHint::kNone)
    : condition(std::move(condition)),
      if_block(std::move(if_block)), else_block(std::move(else_block)),
      branch_hint(hint) {}
  CLONEABLE_BOILERPLATE(IfElseBlock)
  std::vector<std::string> Compile() const override;
 private:
  DeepCopyUniquePtr<Condition> condition;
  DeepCopyUniquePtr<CodeBlock> if_block;
  DeepCopyUniquePtr<CodeBlock> else_block;
  BranchHint branch_hint;
};

}  // namespace semantic
}  // namespace treelite
#endif  // TREELITE_SEMANTIC_H_
