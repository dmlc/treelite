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
 return feature_adapter->Compile(default_left, split_index)
     + " " + OpName(op) + " "
     + numeric_adapter->Compile(split_index, threshold);
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
SimpleNumeric::Compile(unsigned split_index, tl_float numeric) const {
  std::ostringstream oss;
  oss << numeric;
  return oss.str();
}

std::string
QuantizeNumeric::Compile(unsigned split_index, tl_float numeric) const {
  const auto& v = cut_pts[split_index];
  auto loc = std::find(v.begin(), v.end(), numeric);
  CHECK(loc != v.end());
  return std::to_string(static_cast<size_t>(loc - v.begin()) * 2);
}

std::string
DenseFeature::Compile(bool default_left, unsigned split_index) const {
  const std::string bitmap
    = bitmap_name + "[" + std::to_string(split_index) + "]";
  return ((default_left) ?  (std::string("!") + bitmap + " || ")
                          : (                   bitmap + " && "))
         + array_name + "[" + std::to_string(split_index) + "]";
}

std::string
CompressedDenseFeature::Compile(bool default_left, unsigned split_index) const {
  const std::string bitmap = accessor_name + "(" + bitmap_name + ", "
                             + std::to_string(split_index / 8) + ", "
                             + std::to_string(split_index % 8) + ")";
  return ((default_left) ?  (std::string("!") + bitmap + " || ")
                          : (                   bitmap + " && "))
         + array_name + "[" + std::to_string(split_index) + "]";
}

std::string
SparseFeature::Compile(bool default_left, unsigned split_index) const {
  const std::string bitmap
    = std::string(" (idx = ") + accessor_name + "(" + col_ind_name + ", "
                              + nonzero_len_name + ", "
                              + std::to_string(split_index) + ")) != -1";
  return ((default_left) ?  (std::string("!(") + bitmap + ") || ")
                          : (                    bitmap + " && "))
         + nonzero_name + "[idx]";
}

}  // namespace semantic
}  // namespace treelite
