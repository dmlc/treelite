/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file quantize.cc
 * \brief Quantize thresholds in condition nodes
 */
#include <treelite/math.h>
#include <dmlc/registry.h>
#include <cmath>
#include "./builder.h"

namespace treelite {
namespace compiler {

DMLC_REGISTRY_FILE_TAG(quantize);

static void
scan_thresholds(ASTNode* node,
                std::vector<std::set<tl_float>>* cut_pts) {
  NumericalConditionNode* num_cond;
  CategoricalConditionNode* cat_cond;
  if ( (num_cond = dynamic_cast<NumericalConditionNode*>(node)) ) {
    CHECK(!num_cond->quantized) << "should not be already quantized";
    const tl_float threshold = num_cond->threshold.float_val;
    if (std::isfinite(threshold)) {
      (*cut_pts)[num_cond->split_index].insert(threshold);
    }
  }
  for (ASTNode* child : node->children) {
    scan_thresholds(child, cut_pts);
  }
}

static void
rewrite_thresholds(ASTNode* node,
                   const std::vector<std::vector<tl_float>>& cut_pts) {
  NumericalConditionNode* num_cond;
  if ( (num_cond = dynamic_cast<NumericalConditionNode*>(node)) ) {
    CHECK(!num_cond->quantized) << "should not be already quantized";
    const tl_float threshold = num_cond->threshold.float_val;
    if (std::isfinite(threshold)) {
      const auto& v = cut_pts[num_cond->split_index];
      auto loc = math::binary_search(v.begin(), v.end(), threshold);
      CHECK(loc != v.end());
      num_cond->threshold.int_val = static_cast<int>(loc - v.begin()) * 2;
      num_cond->quantized = true;
    }  // splits with infinite thresholds will not be quantized
  }
  for (ASTNode* child : node->children) {
    rewrite_thresholds(child, cut_pts);
  }
}

void ASTBuilder::QuantizeThresholds() {
  this->quantize_threshold_flag = true;
  std::vector<std::set<tl_float>> cut_pts;
  std::vector<std::vector<tl_float>> cut_pts_vec;
  cut_pts.resize(this->num_feature);
  cut_pts_vec.resize(this->num_feature);
  scan_thresholds(this->main_node, &cut_pts);
  // convert cut_pts into std::vector
  for (int i = 0; i < this->num_feature; ++i) {
    std::copy(cut_pts[i].begin(), cut_pts[i].end(),
              std::back_inserter(cut_pts_vec[i]));
  }

  /* revise all numerical splits by quantizing thresholds */
  rewrite_thresholds(this->main_node, cut_pts_vec);

  CHECK_EQ(this->main_node->children.size(), 1);
  ASTNode* top_ac_node = this->main_node->children[0];
  CHECK(dynamic_cast<AccumulatorContextNode*>(top_ac_node));
  /* dynamic_cast<> is used here to check node types. This is to ensure
     that we don't accidentally call QuantizeThresholds() twice. */

  ASTNode* quantizer_node = AddNode<QuantizerNode>(this->main_node,
                                                   std::move(cut_pts_vec));
  quantizer_node->children.push_back(top_ac_node);
  top_ac_node->parent = quantizer_node;
  this->main_node->children[0] = quantizer_node;
}

}  // namespace compiler
}  // namespace treelite
