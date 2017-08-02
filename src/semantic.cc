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

std::string
SplitCondition::Compile() const {
  const std::string bitmap
    = std::string("data[") + std::to_string(split_index) + "].missing != -1";
  if (likely_direction == LikelyDirection::kNone) {
    return ((default_left) ?  (std::string("!(") + bitmap + ") || ")
                            : (std::string(" (") + bitmap + ") && "))
            + numeric_adapter->Compile(op, split_index, threshold);
  } else {
    const std::string tag =
           (likely_direction == LikelyDirection::kLeft) ? "LIKELY" : "UNLIKELY";
    return ((default_left) ?  (tag + "( !(" + bitmap + ") || ")
                            : (tag +  "( (" + bitmap + ") && "))
            + numeric_adapter->Compile(op, split_index, threshold) + ") ";
  }
}

std::string
SimpleAccumulator::Compile(tl_float leaf_value) const {
  std::ostringstream oss;
  oss << acc_name << " += " << leaf_value << ";";
  return oss.str();
}

std::vector<std::string>
IfElseBlock::Compile() const {
  std::vector<std::string> ret{std::string("if (")
                               + condition->Compile() + ") {"};
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

std::string
SimpleNumeric::Compile(Tree::Operator op,
                       unsigned split_index,
                       tl_float numeric) const {
  std::ostringstream oss;
  oss << "data[" << split_index << "].fvalue " << OpName(op) << " " << numeric;
  return oss.str();
}

std::string
QuantizeNumeric::Compile(Tree::Operator op,
                         unsigned split_index,
                         tl_float numeric) const {
  std::ostringstream oss;
  const auto& v = cut_pts[split_index];
  auto loc = std::find(v.begin(), v.end(), numeric);
  CHECK(loc != v.end());
  oss << "data[" << split_index << "].qvalue " << OpName(op) << " "
      << static_cast<size_t>(loc - v.begin()) * 2;
  return oss.str();
}

}  // namespace semantic
}  // namespace treelite
