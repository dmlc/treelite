#include "./builder.h"
#include <cmath>

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(annotate);

static void annotate(ASTNode* node,
                     const std::vector<std::vector<size_t>>& counts) {
  const int tree_id = node->tree_id;
  ConditionNode* cond_node;
  if ( (cond_node = dynamic_cast<ConditionNode*>(node)) ) {
    const std::vector<ASTNode*>& children = cond_node->children;
    CHECK_EQ(children.size(), 2);
    CHECK_EQ(tree_id, children[0]->tree_id);
    CHECK_EQ(tree_id, children[1]->tree_id);
    const size_t left_count = counts[tree_id][children[0]->node_id];
    const size_t right_count = counts[tree_id][children[1]->node_id];
    cond_node->branch_hint = (left_count > right_count) ? BranchHint::kLikely
                                                        : BranchHint::kUnlikely;
  }

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