#include "./builder.h"
#include <limits>
#include <cmath>

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(fold_code);

struct CodeFoldingContext {
  double data_count_magnitude_req;
  double sum_hess_magnitude_req;
  double log_root_data_count;
  double log_root_sum_hess;
};

void fold_code(ASTNode* node, CodeFoldingContext* context,
               ASTBuilder* builder) {
  if (node->node_id == 0) {
    if (node->data_count) {
      context->log_root_data_count = std::log(node->data_count.value());
      LOG(INFO) << "log_root_data_count = " << context->log_root_data_count;
    } else {
      context->log_root_data_count = std::numeric_limits<double>::quiet_NaN();
    }
    if (node->sum_hess) {
      context->log_root_sum_hess = std::log(node->sum_hess.value());
      LOG(INFO) << "log_root_sum_hess = " << context->log_root_sum_hess;
    } else {
      context->log_root_sum_hess = std::numeric_limits<double>::quiet_NaN();
    }
  }

  if (   (node->data_count && !std::isnan(context->log_root_data_count)
          && context->log_root_data_count - std::log(node->data_count.value())
             >= context->data_count_magnitude_req)
      || (node->sum_hess && !std::isnan(context->log_root_sum_hess)
          && context->log_root_sum_hess - std::log(node->sum_hess.value())
             >= context->sum_hess_magnitude_req) ) {
    // fold the subtree whose root is [node]
    ASTNode* parent_node = node->parent;
    ASTNode* folder_node = builder->AddNode<CodeFolderNode>(parent_node);
    size_t node_loc = -1;  // is current node 1st child or 2nd child or so forth
    for (size_t i = 0; i < parent_node->children.size(); ++i) {
      if (parent_node->children[i] == node) {
        node_loc = i;
        break;
      }
    }
    CHECK_NE(node_loc, -1);  // parent should have a link to current node
    parent_node->children[node_loc] = folder_node;
    folder_node->children.push_back(node);
    node->parent = folder_node;
  } else {
    for (ASTNode* child : node->children) {
      fold_code(child, context, builder);
    }
  }
}

void ASTBuilder::FoldCode(double data_count_magnitude_req,
                          double sum_hess_magnitude_req) {
  CodeFoldingContext context{data_count_magnitude_req, sum_hess_magnitude_req,
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::quiet_NaN()};
  fold_code(this->main_node, &context, this);
}

}  // namespace compiler
}  // namespace treelite
