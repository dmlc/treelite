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

using common::Cloneable;
using common::DeepCopyUniquePtr;

/*!
 * \brief fundamental block in semantic model.
 * All code blocks should inherit from this class.
 */
class CodeBlock : public Cloneable {
 public:
  virtual ~CodeBlock() = default;
  virtual std::vector<std::string> Compile() const = 0;
};

/*! \brief translation unit is abstraction of a source file */
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

/*!
 * \brief semantic model consists of a header, function registry, and
 *        a list of translation units
 */
struct SemanticModel {
  struct FunctionEntry {
    std::string prototype;
    bool dll_export;
    FunctionEntry(const std::string& prototype, bool dll_export)
      : prototype(prototype), dll_export(dll_export) {}
  };
  std::unique_ptr<CodeBlock> common_header;
  std::vector<FunctionEntry> function_registry;  // list of function prototypes
  std::vector<TranslationUnit> units;
};

inline std::ostream &operator<<(std::ostream &os,
                                const SemanticModel::FunctionEntry &entry) {
#ifdef _WIN32
  const std::string declspec("__declspec(dllexport) ");
#else
  const std::string declspec("");
#endif
  if (entry.dll_export) {
    os << declspec << entry.prototype << ";\n";
  } else {
    os << entry.prototype << ";\n";
  }
  return os;
}

/*! \brief plain code block containing one or more lines of code */
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

/*!
 * \brief function block with a prototype and code body.
 * Its prototype can optionally be registered with a function registry.
 */
class FunctionBlock : public CodeBlock {
 private:
  using FunctionEntry = SemanticModel::FunctionEntry;

 public:
  explicit FunctionBlock(const std::string& prototype,
                         const CodeBlock& body,
                         std::vector<FunctionEntry>* p_function_registry,
                         bool dll_export = false)
    : prototype(prototype), body(body), dll_export(dll_export) {
    if (p_function_registry != nullptr) {
      p_function_registry->emplace_back(this->prototype, this->dll_export);
    }
  }
  explicit FunctionBlock(std::string&& prototype,
                         CodeBlock&& body,
                         std::vector<FunctionEntry>* p_function_registry,
                         bool dll_export = false)
    : prototype(std::move(prototype)), body(std::move(body)),
      dll_export(dll_export) {
    if (p_function_registry != nullptr) {
      p_function_registry->emplace_back(this->prototype, this->dll_export);
    }
  }
  CLONEABLE_BOILERPLATE(FunctionBlock)
  std::vector<std::string> Compile() const override;
 private:
  std::string prototype;
  bool dll_export;
  DeepCopyUniquePtr<CodeBlock> body;
};

/*! \brief sequence of one or more code blocks */
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

/*! \brief a conditional expression */
class Condition : public Cloneable {
 public:
  virtual ~Condition() = default;
  virtual std::string Compile() const = 0;
};

/*!
 * \brief if-else statement with condition
 * may store a branch hint (>50% or <50% likely)
 */
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
