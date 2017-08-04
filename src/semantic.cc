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
PlainBlock::Compile() const {
  return inner_text;
}

std::vector<std::string>
FunctionEntry::registry = {};

std::vector<std::string>
FunctionBlock::Compile() const {
  std::vector<std::string> ret{prototype + " {"};
  TransformPushBack(&ret, body->Compile(), [] (std::string line) {
    return "  " + line;
  });
  ret.emplace_back("}");

  return ret;
}

std::vector<std::string>
SequenceBlock::Compile() const {
  std::vector<std::string> ret;
  for (const auto& block : sequence) {
    TransformPushBack(&ret, block->Compile(), [] (std::string line) {
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
  
  if (likely_direction == LikelyDirection::kNone) {
    ret.push_back(std::string("if (") + condition->Compile() + ") {");
  } else {
    const std::string tag =
           (likely_direction == LikelyDirection::kLeft) ? "LIKELY" : "UNLIKELY";
    ret.push_back(std::string("if ( ") + tag + "( "
                                       + condition->Compile() + " ) ) {");
  }
  TransformPushBack(&ret, if_block->Compile(), [] (std::string line) {
    return "  " + line;
  });
  ret.emplace_back("} else {");
  TransformPushBack(&ret, else_block->Compile(), [] (std::string line) {
    return "  " + line;
  });
  ret.emplace_back("}");
  return ret;
}

}  // namespace semantic
}  // namespace treelite
