/*!
 * Copyright (c) 2019-2021 by Contributors
 * \file dump.cc
 * \brief Generate text representation of AST
 */
#include <treelite/logging.h>
#include "./builder.h"

namespace {

void get_dump_from_node(std::ostringstream* oss,
                        const treelite::compiler::ASTNode* node,
                        int indent) {
  (*oss) << std::string(indent, ' ') << node->GetDump() << "\n";
  for (const treelite::compiler::ASTNode* child : node->children) {
    TREELITE_CHECK(child);
    get_dump_from_node(oss, child, indent + 2);
  }
}

}  // anonymous namespace

namespace treelite {
namespace compiler {

template <typename ThresholdType, typename LeafOutputType>
std::string
ASTBuilder<ThresholdType, LeafOutputType>::GetDump() const {
  std::ostringstream oss;
  get_dump_from_node(&oss, this->main_node, 0);
  return oss.str();
}

template std::string ASTBuilder<float, uint32_t>::GetDump() const;
template std::string ASTBuilder<float, float>::GetDump() const;
template std::string ASTBuilder<double, uint32_t>::GetDump() const;
template std::string ASTBuilder<double, double>::GetDump() const;

}  // namespace compiler
}  // namespace treelite
