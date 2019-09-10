/*!
 * Copyright 2019 by Contributors
 * \file dump.cc
 * \brief Generate text representation of AST
 */
#include "./builder.h"

namespace {

void get_dump_from_node(std::ostringstream* oss,
                        const treelite::compiler::ASTNode* node,
                        int indent) {
  (*oss) << std::string(indent, ' ') << node->GetDump() << "\n";
  for (const treelite::compiler::ASTNode* child : node->children) {
    CHECK(child);
    get_dump_from_node(oss, child, indent + 2);
  }
}

}  // namespace anonymous

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(dump);

std::string ASTBuilder::GetDump() const {
  std::ostringstream oss;
  get_dump_from_node(&oss, this->main_node, 0);
  return oss.str();
}

}  // namespace compiler
}  // namespace treelite
