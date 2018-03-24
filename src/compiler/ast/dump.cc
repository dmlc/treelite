#include "./builder.h"

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(dump);

static void dump(ASTNode* node, int indent) {
  node->Dump(indent);
  for (ASTNode* child : node->children) {
    dump(child, indent + 2);
  }
}

void ASTBuilder::Dump() {
  dump(this->main_node, 0);
}

}  // namespace compiler
}  // namespace treelite