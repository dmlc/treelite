/*!
 * Copyright (c) 2023 by Contributors
 * \file field_accessor.cc
 * \author Hyunsu Cho
 * \brief Methods for accessing fields in Treelite model
 */
#include <treelite/detail/serializer.h>
#include <treelite/tree.h>

#include <cstdint>
#include <string>
#include <variant>

namespace treelite {

PyBufferFrame Model::GetHeaderField(std::string const& name) {
  using treelite::detail::serializer::GetPyBufferFromArray;
  using treelite::detail::serializer::GetPyBufferFromScalar;
  using treelite::detail::serializer::GetPyBufferFromString;
  if (name == "major_ver") {
    return GetPyBufferFromScalar(&this->major_ver_);
  } else if (name == "minor_ver") {
    return GetPyBufferFromScalar(&this->minor_ver_);
  } else if (name == "patch_ver") {
    return GetPyBufferFromScalar(&this->patch_ver_);
  } else if (name == "threshold_type") {
    this->threshold_type_ = this->GetThresholdType();
    return GetPyBufferFromScalar(&this->threshold_type_);
  } else if (name == "leaf_output_type") {
    this->leaf_output_type_ = this->GetLeafOutputType();
    return GetPyBufferFromScalar(&this->threshold_type_);
  } else if (name == "num_tree") {
    this->num_tree_ = static_cast<std::uint64_t>(this->GetNumTree());
    return GetPyBufferFromScalar(&this->num_tree_);
  } else if (name == "num_feature") {
    return GetPyBufferFromScalar(&this->num_feature);
  } else if (name == "task_type") {
    return GetPyBufferFromScalar(&this->task_type);
  } else if (name == "average_tree_output") {
    return GetPyBufferFromScalar(&this->average_tree_output);
  } else if (name == "num_target") {
    return GetPyBufferFromScalar(&this->num_target);
  } else if (name == "num_class") {
    return GetPyBufferFromArray(&this->num_class);
  } else if (name == "leaf_vector_shape") {
    return GetPyBufferFromArray(&this->leaf_vector_shape);
  } else if (name == "target_id") {
    return GetPyBufferFromArray(&this->target_id);
  } else if (name == "class_id") {
    return GetPyBufferFromArray(&this->class_id);
  } else if (name == "postprocessor") {
    return GetPyBufferFromString(&this->postprocessor);
  } else if (name == "sigmoid_alpha") {
    return GetPyBufferFromScalar(&this->sigmoid_alpha);
  } else if (name == "ratio_c") {
    return GetPyBufferFromScalar(&this->ratio_c);
  } else if (name == "base_scores") {
    return GetPyBufferFromArray(&this->base_scores);
  } else if (name == "attributes") {
    return GetPyBufferFromString(&this->attributes);
  } else if (name == "num_opt_field_per_model") {
    this->num_opt_field_per_model_ = 0;
    return GetPyBufferFromScalar(&this->num_opt_field_per_model_);
  }
  TREELITE_LOG(FATAL) << "Unknown field: " << name;
  return {};
}

void Model::SetHeaderField(std::string const& name, PyBufferFrame frame) {
  using treelite::detail::serializer::InitArrayFromPyBufferWithCopy;
  using treelite::detail::serializer::InitScalarFromPyBuffer;
  using treelite::detail::serializer::InitStringFromPyBuffer;
  if (name == "major_ver" || name == "minor_ver" || name == "patch_ver" || name == "threshold_type"
      || name == "leaf_output_type" || name == "num_tree" || name == "num_opt_field_per_model") {
    TREELITE_LOG(FATAL) << "Field " << name << " is read-only and cannot be modified";
  } else if (name == "num_feature") {
    InitScalarFromPyBuffer(&this->num_feature, frame);
  } else if (name == "task_type") {
    InitScalarFromPyBuffer(&this->task_type, frame);
  } else if (name == "average_tree_output") {
    InitScalarFromPyBuffer(&this->average_tree_output, frame);
  } else if (name == "num_target") {
    InitScalarFromPyBuffer(&this->num_target, frame);
  } else if (name == "num_class") {
    InitArrayFromPyBufferWithCopy(&this->num_class, frame);
  } else if (name == "leaf_vector_shape") {
    InitArrayFromPyBufferWithCopy(&this->leaf_vector_shape, frame);
  } else if (name == "target_id") {
    InitArrayFromPyBufferWithCopy(&this->target_id, frame);
  } else if (name == "class_id") {
    InitArrayFromPyBufferWithCopy(&this->class_id, frame);
  } else if (name == "postprocessor") {
    InitStringFromPyBuffer(&this->postprocessor, frame);
  } else if (name == "sigmoid_alpha") {
    InitScalarFromPyBuffer(&this->sigmoid_alpha, frame);
  } else if (name == "ratio_c") {
    InitScalarFromPyBuffer(&this->ratio_c, frame);
  } else if (name == "base_scores") {
    InitArrayFromPyBufferWithCopy(&this->base_scores, frame);
  } else if (name == "attributes") {
    InitStringFromPyBuffer(&this->attributes, frame);
  } else {
    TREELITE_LOG(FATAL) << "Unknown field: " << name;
  }
}

PyBufferFrame Model::GetTreeField(std::uint64_t tree_id, std::string const& name) {
  return std::visit(
      [&](auto&& model_preset) {
        return detail::field_accessor::GetTreeFieldImpl(model_preset, tree_id, name);
      },
      this->variant_);
}

void Model::SetTreeField(std::uint64_t tree_id, std::string const& name, PyBufferFrame frame) {
  std::visit(
      [&](auto&& model_preset) {
        detail::field_accessor::SetTreeFieldImpl(model_preset, tree_id, name, frame);
      },
      this->variant_);
}

namespace detail::field_accessor {

template <typename ThresholdType, typename LeafOutputType>
PyBufferFrame GetTreeFieldImpl(ModelPreset<ThresholdType, LeafOutputType>& model_preset,
    std::uint64_t tree_id, std::string const& name) {
  using treelite::detail::serializer::GetPyBufferFromArray;
  using treelite::detail::serializer::GetPyBufferFromScalar;
  Tree<ThresholdType, LeafOutputType>& tree = model_preset.trees[tree_id];
  if (name == "num_nodes") {
    return GetPyBufferFromScalar(&tree.num_nodes);
  } else if (name == "has_categorical_split") {
    return GetPyBufferFromScalar(&tree.has_categorical_split_);
  } else if (name == "node_type") {
    return GetPyBufferFromArray(&tree.node_type_);
  } else if (name == "cleft") {
    return GetPyBufferFromArray(&tree.cleft_);
  } else if (name == "cright") {
    return GetPyBufferFromArray(&tree.cright_);
  } else if (name == "split_index") {
    return GetPyBufferFromArray(&tree.split_index_);
  } else if (name == "default_left") {
    return GetPyBufferFromArray(&tree.default_left_);
  } else if (name == "leaf_value") {
    return GetPyBufferFromArray(&tree.leaf_value_);
  } else if (name == "threshold") {
    return GetPyBufferFromArray(&tree.threshold_);
  } else if (name == "cmp") {
    return GetPyBufferFromArray(&tree.cmp_);
  } else if (name == "category_list_right_child") {
    return GetPyBufferFromArray(&tree.category_list_right_child_);
  } else if (name == "leaf_vector") {
    return GetPyBufferFromArray(&tree.leaf_vector_);
  } else if (name == "leaf_vector_begin") {
    return GetPyBufferFromArray(&tree.leaf_vector_begin_);
  } else if (name == "leaf_vector_end") {
    return GetPyBufferFromArray(&tree.leaf_vector_end_);
  } else if (name == "category_list") {
    return GetPyBufferFromArray(&tree.category_list_);
  } else if (name == "category_list_begin") {
    return GetPyBufferFromArray(&tree.category_list_begin_);
  } else if (name == "category_list_end") {
    return GetPyBufferFromArray(&tree.category_list_end_);
  } else if (name == "data_count") {
    return GetPyBufferFromArray(&tree.data_count_);
  } else if (name == "data_count_present") {
    return GetPyBufferFromArray(&tree.data_count_present_);
  } else if (name == "sum_hess") {
    return GetPyBufferFromArray(&tree.sum_hess_);
  } else if (name == "sum_hess_present") {
    return GetPyBufferFromArray(&tree.sum_hess_present_);
  } else if (name == "gain") {
    return GetPyBufferFromArray(&tree.gain_);
  } else if (name == "gain_present") {
    return GetPyBufferFromArray(&tree.gain_present_);
  } else if (name == "num_opt_field_per_tree") {
    tree.num_opt_field_per_tree_ = 0;
    return GetPyBufferFromScalar(&tree.num_opt_field_per_tree_);
  } else if (name == "num_opt_field_per_node") {
    tree.num_opt_field_per_node_ = 0;
    return GetPyBufferFromScalar(&tree.num_opt_field_per_node_);
  }
  TREELITE_LOG(FATAL) << "Unknown field: " << name;
  return PyBufferFrame{};
}

template <typename ThresholdType, typename LeafOutputType>
void SetTreeFieldImpl(ModelPreset<ThresholdType, LeafOutputType>& model_preset,
    std::uint64_t tree_id, std::string const& name, PyBufferFrame frame) {
  using treelite::detail::serializer::InitArrayFromPyBufferWithCopy;
  using treelite::detail::serializer::InitScalarFromPyBuffer;
  Tree<ThresholdType, LeafOutputType>& tree = model_preset.trees[tree_id];
  if (name == "num_opt_field_per_tree" || name == "num_opt_field_per_node") {
    TREELITE_LOG(FATAL) << "Field " << name << " is read-only and cannot be modified";
  } else if (name == "num_nodes") {
    InitScalarFromPyBuffer(&tree.num_nodes, frame);
  } else if (name == "has_categorical_split") {
    InitScalarFromPyBuffer(&tree.has_categorical_split_, frame);
  } else if (name == "node_type") {
    InitArrayFromPyBufferWithCopy(&tree.node_type_, frame);
  } else if (name == "cleft") {
    InitArrayFromPyBufferWithCopy(&tree.cleft_, frame);
  } else if (name == "cright") {
    InitArrayFromPyBufferWithCopy(&tree.cright_, frame);
  } else if (name == "split_index") {
    InitArrayFromPyBufferWithCopy(&tree.split_index_, frame);
  } else if (name == "default_left") {
    InitArrayFromPyBufferWithCopy(&tree.default_left_, frame);
  } else if (name == "leaf_value") {
    InitArrayFromPyBufferWithCopy(&tree.leaf_value_, frame);
  } else if (name == "threshold") {
    InitArrayFromPyBufferWithCopy(&tree.threshold_, frame);
  } else if (name == "cmp") {
    InitArrayFromPyBufferWithCopy(&tree.cmp_, frame);
  } else if (name == "category_list_right_child") {
    InitArrayFromPyBufferWithCopy(&tree.category_list_right_child_, frame);
  } else if (name == "leaf_vector") {
    InitArrayFromPyBufferWithCopy(&tree.leaf_vector_, frame);
  } else if (name == "leaf_vector_begin") {
    InitArrayFromPyBufferWithCopy(&tree.leaf_vector_begin_, frame);
  } else if (name == "leaf_vector_end") {
    InitArrayFromPyBufferWithCopy(&tree.leaf_vector_end_, frame);
  } else if (name == "category_list") {
    InitArrayFromPyBufferWithCopy(&tree.category_list_, frame);
  } else if (name == "category_list_begin") {
    InitArrayFromPyBufferWithCopy(&tree.category_list_begin_, frame);
  } else if (name == "category_list_end") {
    InitArrayFromPyBufferWithCopy(&tree.category_list_end_, frame);
  } else if (name == "data_count") {
    InitArrayFromPyBufferWithCopy(&tree.data_count_, frame);
  } else if (name == "data_count_present") {
    InitArrayFromPyBufferWithCopy(&tree.data_count_present_, frame);
  } else if (name == "sum_hess") {
    InitArrayFromPyBufferWithCopy(&tree.sum_hess_, frame);
  } else if (name == "sum_hess_present") {
    InitArrayFromPyBufferWithCopy(&tree.sum_hess_present_, frame);
  } else if (name == "gain") {
    InitArrayFromPyBufferWithCopy(&tree.gain_, frame);
  } else if (name == "gain_present") {
    InitArrayFromPyBufferWithCopy(&tree.gain_present_, frame);
  } else {
    TREELITE_LOG(FATAL) << "Unknown field: " << name;
  }
}

}  // namespace detail::field_accessor

}  // namespace treelite
