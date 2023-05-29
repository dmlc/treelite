/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file serializer.cc
 * \brief Implementation for serializers
 * \author Hyunsu Cho
 */

#include <treelite/detail/serializer_mixins.h>
#include <treelite/logging.h>
#include <treelite/tree.h>
#include <treelite/version.h>

#include <iostream>

namespace treelite {

namespace detail::serializer {

template <typename MixIn>
class Serializer {
 public:
  explicit Serializer(MixIn& mixin) : mixin_(mixin) {}

  void SerializeHeader(Model& model) {
    // Header 1
    model.major_ver_ = TREELITE_VER_MAJOR;
    model.minor_ver_ = TREELITE_VER_MINOR;
    model.patch_ver_ = TREELITE_VER_PATCH;
    mixin_.SerializePrimitiveField(&model.major_ver_);
    mixin_.SerializePrimitiveField(&model.minor_ver_);
    mixin_.SerializePrimitiveField(&model.patch_ver_);
    mixin_.SerializePrimitiveField(&model.threshold_type_);
    mixin_.SerializePrimitiveField(&model.leaf_output_type_);

    // Number of trees
    model.num_tree_ = static_cast<std::uint64_t>(model.GetNumTree());
    mixin_.SerializePrimitiveField(&model.num_tree_);

    // Header 2
    mixin_.SerializePrimitiveField(&model.num_feature);
    mixin_.SerializePrimitiveField(&model.task_type);
    mixin_.SerializePrimitiveField(&model.average_tree_output);
    mixin_.SerializeCompositeField(&model.task_param, "T{=B=?xx=I=I}");
    mixin_.SerializeCompositeField(
        &model.param, "T{" _TREELITE_STR(TREELITE_MAX_PRED_TRANSFORM_LENGTH) "s=f=f=f}");

    /* Extension Slot 1: Per-model optional fields -- to be added later */
    model.num_opt_field_per_model_ = 0;
    mixin_.SerializePrimitiveField(&model.num_opt_field_per_model_);
  }

  void SerializeTrees(Model& model) {
    model.Dispatch([&](auto&& concrete_model) {
      TREELITE_CHECK_EQ(concrete_model.trees.size(), model.num_tree_)
          << "Incorrect number of trees in the model";
      for (auto& tree : concrete_model.trees) {
        SerializeTree(tree);
      }
    });
  }

  template <typename ThresholdType, typename LeafOutputType>
  void SerializeTree(Tree<ThresholdType, LeafOutputType>& tree) {
    TREELITE_CHECK_EQ(tree.num_nodes, tree.nodes_.Size()) << "Incorrect number of nodes";
    mixin_.SerializePrimitiveField(&tree.num_nodes);
    mixin_.SerializePrimitiveField(&tree.has_categorical_split_);
    mixin_.SerializeCompositeArray(&tree.nodes_, tree.GetFormatStringForNode());
    mixin_.SerializePrimitiveArray(&tree.leaf_vector_);
    mixin_.SerializePrimitiveArray(&tree.leaf_vector_begin_);
    mixin_.SerializePrimitiveArray(&tree.leaf_vector_end_);
    mixin_.SerializePrimitiveArray(&tree.matching_categories_);
    mixin_.SerializePrimitiveArray(&tree.matching_categories_offset_);

    /* Extension slot 2: Per-tree optional fields -- to be added later */
    tree.num_opt_field_per_tree_ = 0;
    mixin_.SerializePrimitiveField(&tree.num_opt_field_per_tree_);

    /* Extension slot 3: Per-node optional fields -- to be added later */
    tree.num_opt_field_per_node_ = 0;
    mixin_.SerializePrimitiveField(&tree.num_opt_field_per_node_);
  }

 private:
  MixIn& mixin_;
};

template <typename MixIn>
class Deserializer {
 public:
  explicit Deserializer(MixIn& mixin) : mixin_(mixin) {}

  std::unique_ptr<Model> DeserializeHeaderAndCreateModel() {
    // Header 1
    std::int32_t major_ver, minor_ver, patch_ver;
    mixin_.DeserializePrimitiveField(&major_ver);
    mixin_.DeserializePrimitiveField(&minor_ver);
    mixin_.DeserializePrimitiveField(&patch_ver);
    if (major_ver != TREELITE_VER_MAJOR && !(major_ver == 3 && minor_ver == 9)) {
      TREELITE_LOG(FATAL) << "Cannot load model from a different major Treelite version or "
                          << "a version before 3.9.0." << std::endl
                          << "Currently running Treelite version " << TREELITE_VER_MAJOR << "."
                          << TREELITE_VER_MINOR << "." << TREELITE_VER_PATCH << std::endl
                          << "The model checkpoint was generated from Treelite version "
                          << major_ver << "." << minor_ver << "." << patch_ver;
    } else if (major_ver == TREELITE_VER_MAJOR && minor_ver > TREELITE_VER_MINOR) {
      TREELITE_LOG(WARNING)
          << "The model you are loading originated from a newer Treelite version; some "
          << "functionalities may be unavailable." << std::endl
          << "Currently running Treelite version " << TREELITE_VER_MAJOR << "."
          << TREELITE_VER_MINOR << "." << TREELITE_VER_PATCH << std::endl
          << "The model checkpoint was generated from Treelite version " << major_ver << "."
          << minor_ver << "." << patch_ver;
    }
    TypeInfo threshold_type, leaf_output_type;
    mixin_.DeserializePrimitiveField(&threshold_type);
    mixin_.DeserializePrimitiveField(&leaf_output_type);

    std::unique_ptr<Model> model = Model::Create(threshold_type, leaf_output_type);
    model->major_ver_ = major_ver;
    model->minor_ver_ = minor_ver;
    model->patch_ver_ = patch_ver;

    // Number of trees
    mixin_.DeserializePrimitiveField(&model->num_tree_);

    // Header 2
    mixin_.DeserializePrimitiveField(&model->num_feature);
    mixin_.DeserializePrimitiveField(&model->task_type);
    mixin_.DeserializePrimitiveField(&model->average_tree_output);
    mixin_.DeserializeCompositeField(&model->task_param);
    mixin_.DeserializeCompositeField(&model->param);

    /* Extension Slot 1: Per-model optional fields -- to be added later */
    bool const use_opt_field = (major_ver >= 3);
    if (use_opt_field) {
      mixin_.DeserializePrimitiveField(&model->num_opt_field_per_model_);
      // Ignore extra fields; the input is likely from a later version of Treelite
      for (std::int32_t i = 0; i < model->num_opt_field_per_model_; ++i) {
        mixin_.SkipOptionalField();
      }
    } else {
      TREELITE_LOG(FATAL) << "Only Treelite format version 3.x or later is supported.";
    }

    return model;
  }

  void DeserializeTrees(Model& model) {
    model.Dispatch([&](auto&& concrete_model) {
      concrete_model.trees.clear();
      for (std::uint64_t i = 0; i < model.num_tree_; ++i) {
        concrete_model.trees.emplace_back();
        DeserializeTree(concrete_model.trees.back());
      }
    });
  }

  template <typename ThresholdType, typename LeafOutputType>
  void DeserializeTree(Tree<ThresholdType, LeafOutputType>& tree) {
    mixin_.DeserializePrimitiveField(&tree.num_nodes);
    mixin_.DeserializePrimitiveField(&tree.has_categorical_split_);
    mixin_.DeserializeCompositeArray(&tree.nodes_);
    TREELITE_CHECK_EQ(static_cast<std::size_t>(tree.num_nodes), tree.nodes_.Size())
        << "Could not load the correct number of nodes";
    mixin_.DeserializePrimitiveArray(&tree.leaf_vector_);
    mixin_.DeserializePrimitiveArray(&tree.leaf_vector_begin_);
    mixin_.DeserializePrimitiveArray(&tree.leaf_vector_end_);
    mixin_.DeserializePrimitiveArray(&tree.matching_categories_);
    mixin_.DeserializePrimitiveArray(&tree.matching_categories_offset_);

    /* Extension slot 2: Per-tree optional fields -- to be added later */
    mixin_.DeserializePrimitiveField(&tree.num_opt_field_per_tree_);
    // Ignore extra fields; the input is likely from a later version of Treelite
    for (std::int32_t i = 0; i < tree.num_opt_field_per_tree_; ++i) {
      mixin_.SkipOptionalField();
    }

    /* Extension slot 3: Per-node optional fields -- to be added later */
    mixin_.DeserializePrimitiveField(&tree.num_opt_field_per_node_);
    // Ignore extra fields; the input is likely from a later version of Treelite
    for (std::int32_t i = 0; i < tree.num_opt_field_per_node_; ++i) {
      mixin_.SkipOptionalField();
    }
  }

 private:
  MixIn& mixin_;
};

}  // namespace detail::serializer

std::vector<PyBufferFrame> Model::GetPyBuffer() {
  detail::serializer::PyBufferSerializerMixIn mixin{};
  detail::serializer::Serializer<detail::serializer::PyBufferSerializerMixIn> serializer{mixin};
  serializer.SerializeHeader(*this);
  serializer.SerializeTrees(*this);
  return mixin.GetFrames();
}

std::unique_ptr<Model> Model::CreateFromPyBuffer(std::vector<PyBufferFrame> frames) {
  detail::serializer::PyBufferDeserializerMixIn mixin{frames};
  detail::serializer::Deserializer<detail::serializer::PyBufferDeserializerMixIn> deserializer{
      mixin};
  std::unique_ptr<Model> model = deserializer.DeserializeHeaderAndCreateModel();
  deserializer.DeserializeTrees(*model);
  return model;
}

void Model::SerializeToStream(std::ostream& os) {
  detail::serializer::StreamSerializerMixIn mixin{os};
  detail::serializer::Serializer<detail::serializer::StreamSerializerMixIn> serializer{mixin};
  serializer.SerializeHeader(*this);
  serializer.SerializeTrees(*this);
}

std::unique_ptr<Model> Model::DeserializeFromStream(std::istream& is) {
  detail::serializer::StreamDeserializerMixIn mixin{is};
  detail::serializer::Deserializer<detail::serializer::StreamDeserializerMixIn> deserializer{mixin};
  std::unique_ptr<Model> model = deserializer.DeserializeHeaderAndCreateModel();
  deserializer.DeserializeTrees(*model);
  return model;
}

}  // namespace treelite
