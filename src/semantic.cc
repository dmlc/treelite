/*!
 * Copyright 2017 by Contributors
 * \file semantic.cc
 * \brief Building blocks for semantic model of tree prediction code
 * \author Philip Cho
 */

#include <treelite/semantic.h>

namespace treelite {
namespace semantic {

std::vector<std::string>
TranslationUnit::Compile(const std::string& header_filename) const {
  std::string header_basename = common::GetBasename(header_filename);
  std::vector<std::string> lines{std::string("#include \"")
                                 + header_basename + "\"", ""};
  auto preamble_lines = preamble->Compile();
  if (preamble_lines.size() > 0) {
    common::TransformPushBack(&lines, preamble_lines,
      [] (std::string line) {
        return line;
      });
    lines.emplace_back();
  }
  common::TransformPushBack(&lines, body->Compile(),
    [] (std::string line) {
      return line;
    });
  return lines;
}

std::vector<std::string>
PlainBlock::Compile() const {
  return inner_text;
}

std::vector<std::string>
FunctionBlock::Compile() const {
  std::vector<std::string> ret{prototype + " {"};
  common::TransformPushBack(&ret, body->Compile(), [] (std::string line) {
    return "  " + line;
  });
  ret.emplace_back("}");

  return ret;
}

std::vector<std::string>
SequenceBlock::Compile() const {
  std::vector<std::string> ret;
  for (const auto& block : sequence) {
    common::TransformPushBack(&ret, block->Compile(), [] (std::string line) {
      return line;
    });
  }
  return ret;
}

void
SequenceBlock::Reserve(size_t size) {
  sequence.reserve(size);
}

void
SequenceBlock::PushBack(const CodeBlock& block) {
  sequence.emplace_back(block);
}

void
SequenceBlock::PushBack(CodeBlock&& block) {
  sequence.push_back(DeepCopyUniquePtr<CodeBlock>(std::move(block)));
}

std::vector<std::string>
IfElseBlock::Compile() const {
  std::vector<std::string> ret;

  if (branch_hint == BranchHint::kNone) {
    ret.push_back(std::string("if (") + condition->Compile() + ") {");
  } else {
    const std::string tag =
                  (branch_hint == BranchHint::kLikely) ? "LIKELY" : "UNLIKELY";
    ret.push_back(std::string("if ( ") + tag + "( "
                                       + condition->Compile() + " ) ) {");
  }
  common::TransformPushBack(&ret, if_block->Compile(), [] (std::string line) {
    return "  " + line;
  });
  ret.emplace_back("} else {");
  common::TransformPushBack(&ret, else_block->Compile(), [] (std::string line) {
    return "  " + line;
  });
  ret.emplace_back("}");
  return ret;
}

}  // namespace semantic
}  // namespace treelite
