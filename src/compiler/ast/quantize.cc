/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file quantize.cc
 * \brief Quantize thresholds in condition nodes
 */
#include <treelite/math.h>
#include <treelite/logging.h>
#include <set>
#include <cmath>
#include "./builder.h"

namespace treelite {
namespace compiler {

template <typename ThresholdType>
static void
scan_thresholds(ASTNode* node, std::vector<std::set<ThresholdType>>* cut_pts) {
  NumericalConditionNode<ThresholdType>* num_cond;
  if ( (num_cond = dynamic_cast<NumericalConditionNode<ThresholdType>*>(node)) ) {
    TREELITE_CHECK(!num_cond->quantized) << "should not be already quantized";
    const ThresholdType threshold = num_cond->threshold.float_val;
    if (std::isfinite(threshold)) {
      (*cut_pts)[num_cond->split_index].insert(threshold);
    }
  }
  for (ASTNode* child : node->children) {
    scan_thresholds(child, cut_pts);
  }
}

template <typename ThresholdType>
static void
rewrite_thresholds(ASTNode* node, const std::vector<std::vector<ThresholdType>>& cut_pts) {
  NumericalConditionNode<ThresholdType>* num_cond;
  if ( (num_cond = dynamic_cast<NumericalConditionNode<ThresholdType>*>(node)) ) {
    TREELITE_CHECK(!num_cond->quantized) << "should not be already quantized";
    const ThresholdType threshold = num_cond->threshold.float_val;
    if (std::isfinite(threshold)) {
      const auto& v = cut_pts[num_cond->split_index];
      {
        auto loc = math::binary_search(v.begin(), v.end(), threshold);
        TREELITE_CHECK(loc != v.end());
        num_cond->threshold.int_val = static_cast<int>(loc - v.begin()) * 2;
      }
      {
        ThresholdType zero = static_cast<ThresholdType>(0);
        auto loc = std::lower_bound(v.begin(), v.end(), zero);
        num_cond->zero_quantized = static_cast<int>(loc - v.begin()) * 2;
        if (loc != v.end() && zero != *loc) {
          --num_cond->zero_quantized;
        }
      }
      num_cond->quantized = true;
    }  // splits with infinite thresholds will not be quantized
  }
  for (ASTNode* child : node->children) {
    rewrite_thresholds(child, cut_pts);
  }
}

template <typename ThresholdType, typename LeafOutputType>
void
ASTBuilder<ThresholdType, LeafOutputType>::QuantizeThresholds() {
  this->quantize_threshold_flag = true;
  std::vector<std::set<ThresholdType>> cut_pts;
  std::vector<std::vector<ThresholdType>> cut_pts_vec;
  cut_pts.resize(this->num_feature);
  cut_pts_vec.resize(this->num_feature);
  scan_thresholds(this->main_node, &cut_pts);
  // convert cut_pts into std::vector
  for (int i = 0; i < this->num_feature; ++i) {
    std::copy(cut_pts[i].begin(), cut_pts[i].end(), std::back_inserter(cut_pts_vec[i]));
  }

  /* revise all numerical splits by quantizing thresholds */
  rewrite_thresholds(this->main_node, cut_pts_vec);

  TREELITE_CHECK_EQ(this->main_node->children.size(), 1);
  ASTNode* top_ac_node = this->main_node->children[0];
  TREELITE_CHECK(dynamic_cast<AccumulatorContextNode*>(top_ac_node));
  /* dynamic_cast<> is used here to check node types. This is to ensure
     that we don't accidentally call QuantizeThresholds() twice. */

  ASTNode* quantizer_node
    = AddNode<QuantizerNode<ThresholdType>>(this->main_node, std::move(cut_pts_vec));
  quantizer_node->children.push_back(top_ac_node);
  top_ac_node->parent = quantizer_node;
  this->main_node->children[0] = quantizer_node;
}

template void ASTBuilder<float, uint32_t>::QuantizeThresholds();
template void ASTBuilder<float, float>::QuantizeThresholds();
template void ASTBuilder<double, uint32_t>::QuantizeThresholds();
template void ASTBuilder<double, double>::QuantizeThresholds();

}  // namespace compiler
}  // namespace treelite
