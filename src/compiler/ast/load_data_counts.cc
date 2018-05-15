#include "./builder.h"
#include <cmath>

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(load_data_counts);

static void load_data_counts(ASTNode* node,
                             const std::vector<std::vector<size_t>>& counts) {
  if (node->tree_id >= 0 && node->node_id >= 0) {
    node->data_count = counts[node->tree_id][node->node_id];
  }
  for (ASTNode* child : node->children) {
    load_data_counts(child, counts);
  }
}

void
ASTBuilder::LoadDataCounts(const std::vector<std::vector<size_t>>& counts) {
  load_data_counts(this->main_node, counts);
}

}  // namespace compiler
}  // namespace treelite
