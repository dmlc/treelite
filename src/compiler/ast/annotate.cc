#include "./builder.h"
#include <cmath>

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(annotate);

static void annotate(ASTNode* node,
                     const std::vector<std::vector<size_t>>& counts) {
  node->data_count = counts[node->tree_id][node->node_id];
  for (ASTNode* child : node->children) {
    annotate(child, counts);
  }
}

void
ASTBuilder::AnnotateBranches(const std::vector<std::vector<size_t>>& counts) {
  annotate(this->main_node, counts);
}

}  // namespace compiler
}  // namespace treelite
