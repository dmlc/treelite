#include "./builder.h"

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(count_descendant);

static int count(ASTNode* node) {
  int accum = 0;
  for (ASTNode* child : node->children) {
    accum += count(child) + 1;
  }
  node->num_descendant = accum;
  return accum;
}

void ASTBuilder::CountDescendant() {
  count(this->main_node);
}

}  // namespace compiler
}  // namespace treelite