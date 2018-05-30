#include "ast.h"
#include "ast.pb.h"

namespace treelite {
namespace compiler {

void ASTNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
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
}

void MainNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
  ASTNode::Serialize(out);
  treelite_ast_protobuf::MainNode* e = out->mutable_main_variant();
  e->set_global_bias(global_bias);
  e->set_average_result(average_result);
  e->set_num_tree(num_tree);
  e->set_num_feature(num_feature);
}

void TranslationUnitNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
  ASTNode::Serialize(out);
  treelite_ast_protobuf::TranslationUnitNode* e
    = out->mutable_translation_unit_variant();
  e->set_unit_id(unit_id);
}

void QuantizerNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
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
}

void AccumulatorContextNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
  ASTNode::Serialize(out);
  out->mutable_accumulator_context_variant();
}

void CodeFolderNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
  ASTNode::Serialize(out);
  out->mutable_code_folder_variant();
}

void ConditionNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
  ASTNode::Serialize(out);
  treelite_ast_protobuf::ConditionNode* e = out->mutable_condition_variant();
  e->set_split_index(split_index);
  e->set_default_left(default_left);
  if (gain) {
    e->set_gain(gain.value());
  }
}

void NumericalConditionNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
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
}

void CategoricalConditionNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
  ConditionNode::Serialize(out);
  CHECK_EQ(out->subclasses_case(),
           treelite_ast_protobuf::ASTNode::SubclassesCase::kConditionVariant);
  treelite_ast_protobuf::ConditionNode* e = out->mutable_condition_variant();
  treelite_ast_protobuf::CategoricalConditionNode* f
    = e->mutable_categorical_variant();
  google::protobuf::RepeatedField<google::protobuf::uint32>
    left_categories_pf(left_categories.begin(), left_categories.end());
  f->mutable_left_categories()->Swap(&left_categories_pf);
}

void OutputNode::Serialize(treelite_ast_protobuf::ASTNode* out) {
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
}

}  // namespace compiler
}  // namespace treelite
