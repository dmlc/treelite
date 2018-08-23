/*!
 * Copyright 2017 by Contributors
 * \file ast.h
 * \brief Definition for AST classes
 * \author Philip Cho
 */
#include "ast.h"

#ifdef TREELITE_PROTOBUF_SUPPORT
#include "ast.pb.h"
#endif  // TREELITE_PROTOBUF_SUPPORT

namespace treelite {
namespace compiler {

void ASTNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
#ifdef TREELITE_PROTOBUF_SUPPORT
  out->set_node_id(node_id);
  out->set_tree_id(tree_id);
  if (data_count) {
    out->set_data_count(data_count.value());
  }
  if (sum_hess) {
    out->set_sum_hess(sum_hess.value());
  }
  for (ASTNode* child : children) {
    treelite_ast_protobuf::ASTNode* node = out->add_children();
    child->Serialize(node);
  }
#else  // TREELITE_PROTOBUF_SUPPORT
  LOG(FATAL) << "Treelite was not compiled with Protobuf!";
#endif  // TREELITE_PROTOBUF_SUPPORT
}

void MainNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
#ifdef TREELITE_PROTOBUF_SUPPORT
  ASTNode::Serialize(out);
  treelite_ast_protobuf::MainNode* e = out->mutable_main_variant();
  e->set_global_bias(global_bias);
  e->set_average_result(average_result);
  e->set_num_tree(num_tree);
  e->set_num_feature(num_feature);
#else  // TREELITE_PROTOBUF_SUPPORT
  LOG(FATAL) << "Treelite was not compiled with Protobuf!";
#endif  // TREELITE_PROTOBUF_SUPPORT
}

void TranslationUnitNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
#ifdef TREELITE_PROTOBUF_SUPPORT
  ASTNode::Serialize(out);
  treelite_ast_protobuf::TranslationUnitNode* e
    = out->mutable_translation_unit_variant();
  e->set_unit_id(unit_id);
#else  // TREELITE_PROTOBUF_SUPPORT
  LOG(FATAL) << "Treelite was not compiled with Protobuf!";
#endif  // TREELITE_PROTOBUF_SUPPORT
}

void QuantizerNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
#ifdef TREELITE_PROTOBUF_SUPPORT
  ASTNode::Serialize(out);
  treelite_ast_protobuf::QuantizerNode* e = out->mutable_quantizer_variant();
  for (const auto& cut_pts_per_feature : cut_pts) {
    // serialize list of thresholds, one feature at a time
    treelite_ast_protobuf::FeatureThresholdList* f = e->add_cut_pts();
    google::protobuf::RepeatedField<float>
      feature_threshod_list(cut_pts_per_feature.begin(),
                            cut_pts_per_feature.end());
    f->mutable_cut_pts()->Swap(&feature_threshod_list);
  }
#else  // TREELITE_PROTOBUF_SUPPORT
  LOG(FATAL) << "Treelite was not compiled with Protobuf!";
#endif  // TREELITE_PROTOBUF_SUPPORT
}

void AccumulatorContextNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
#ifdef TREELITE_PROTOBUF_SUPPORT
  ASTNode::Serialize(out);
  out->mutable_accumulator_context_variant();
#else  // TREELITE_PROTOBUF_SUPPORT
  LOG(FATAL) << "Treelite was not compiled with Protobuf!";
#endif  // TREELITE_PROTOBUF_SUPPORT
}

void CodeFolderNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
#ifdef TREELITE_PROTOBUF_SUPPORT
  ASTNode::Serialize(out);
  out->mutable_code_folder_variant();
#else  // TREELITE_PROTOBUF_SUPPORT
  LOG(FATAL) << "Treelite was not compiled with Protobuf!";
#endif  // TREELITE_PROTOBUF_SUPPORT
}

void ConditionNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
#ifdef TREELITE_PROTOBUF_SUPPORT
  ASTNode::Serialize(out);
  treelite_ast_protobuf::ConditionNode* e = out->mutable_condition_variant();
  e->set_split_index(split_index);
  e->set_default_left(default_left);
  if (gain) {
    e->set_gain(gain.value());
  }
#else  // TREELITE_PROTOBUF_SUPPORT
  LOG(FATAL) << "Treelite was not compiled with Protobuf!";
#endif  // TREELITE_PROTOBUF_SUPPORT
}

void NumericalConditionNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
#ifdef TREELITE_PROTOBUF_SUPPORT
  ConditionNode::Serialize(out);
  CHECK_EQ(out->subclasses_case(),
           treelite_ast_protobuf::ASTNode::SubclassesCase::kConditionVariant);
  treelite_ast_protobuf::ConditionNode* e = out->mutable_condition_variant();
  treelite_ast_protobuf::NumericalConditionNode* f
    = e->mutable_numerical_variant();
  f->set_quantized(quantized);
  f->set_op(OpName(op));
  if (quantized) {
    f->set_int_threshold(threshold.int_val);
  } else {
    f->set_float_threshold(threshold.float_val);
  }
#else  // TREELITE_PROTOBUF_SUPPORT
  LOG(FATAL) << "Treelite was not compiled with Protobuf!";
#endif  // TREELITE_PROTOBUF_SUPPORT
}

void CategoricalConditionNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
#ifdef TREELITE_PROTOBUF_SUPPORT
  ConditionNode::Serialize(out);
  CHECK_EQ(out->subclasses_case(),
           treelite_ast_protobuf::ASTNode::SubclassesCase::kConditionVariant);
  treelite_ast_protobuf::ConditionNode* e = out->mutable_condition_variant();
  treelite_ast_protobuf::CategoricalConditionNode* f
    = e->mutable_categorical_variant();
  google::protobuf::RepeatedField<google::protobuf::uint32>
    left_categories_pf(left_categories.begin(), left_categories.end());
  f->mutable_left_categories()->Swap(&left_categories_pf);
#else  // TREELITE_PROTOBUF_SUPPORT
  LOG(FATAL) << "Treelite was not compiled with Protobuf!";
#endif  // TREELITE_PROTOBUF_SUPPORT
}

void OutputNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
#ifdef TREELITE_PROTOBUF_SUPPORT
  ASTNode::Serialize(out);
  treelite_ast_protobuf::OutputNode* e = out->mutable_output_variant();
  e->set_is_vector(is_vector);
  if (is_vector) {
    google::protobuf::RepeatedField<float>
      vector_pf(vector.begin(), vector.end());
    e->mutable_vector()->Swap(&vector_pf);
  } else {
    e->set_scalar(scalar);
  }
#else  // TREELITE_PROTOBUF_SUPPORT
  LOG(FATAL) << "Treelite was not compiled with Protobuf!";
#endif  // TREELITE_PROTOBUF_SUPPORT
}

}  // namespace compiler
}  // namespace treelite
