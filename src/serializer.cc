/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file serializer.cc
 * \brief Implementation for serializers
 * \author Hyunsu Cho
 */

#include <treelite/detail/serializer_mixins.h>
#include <treelite/enum/operator.h>
#include <treelite/enum/tree_node_type.h>
#include <treelite/logging.h>
#include <treelite/tree.h>
#include <treelite/version.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace treelite {

namespace detail::serializer {

enum class TaskTypeV3 : std::uint8_t {
  kBinaryClfRegr = 0,
  kMultiClfGrovePerClass = 1,
  kMultiClfProbDistLeaf = 2,
  kMultiClfCategLeaf = 3
};

enum class SplitFeatureTypeV3 : std::int8_t { kNone = 0, kNumerical = 1, kCategorical = 2 };

struct TaskParamV3 {
  enum class OutputType : std::uint8_t { kFloat = 0, kInt = 1 };
  OutputType output_type;
  bool grove_per_class;
  unsigned int num_class;
  unsigned int leaf_vector_size;
};

struct ModelParamV3 {
  char pred_transform[256] = {0};
  float sigmoid_alpha;
  float ratio_c;
  float global_bias;
};

template <typename ThresholdType, typename LeafOutputType>
struct NodeV3 {
  union Info {
    LeafOutputType leaf_value;
    ThresholdType threshold;
  };
  std::int32_t cleft_, cright_;
  std::uint32_t sindex_;
  Info info_;
  std::uint64_t data_count_;
  double sum_hess_;
  double gain_;
  SplitFeatureTypeV3 split_type_;
  Operator cmp_;
  bool data_count_present_;
  bool sum_hess_present_;
  bool gain_present_;
  bool categories_list_right_child_;
  inline bool DefaultLeft() const {
    return (sindex_ >> 31U) != 0;
  }
  inline std::uint32_t SplitIndex() const {
    return (sindex_ & ((1U << 31U) - 1U));
  }
  inline bool IsLeaf() const {
    return cleft_ == -1;
  }
  inline LeafOutputType LeafValue() const {
    return info_.leaf_value;
  }
  inline ThresholdType Threshold() const {
    return info_.threshold;
  }
};

template <typename MixIn>
class Serializer {
 public:
  explicit Serializer(std::shared_ptr<MixIn> mixin) : mixin_(mixin) {}

  void SerializeHeader(Model& model) {
    // Header 1
    model.major_ver_ = TREELITE_VER_MAJOR;
    model.minor_ver_ = TREELITE_VER_MINOR;
    model.patch_ver_ = TREELITE_VER_PATCH;
    mixin_->SerializeScalar(&model.major_ver_);
    mixin_->SerializeScalar(&model.minor_ver_);
    mixin_->SerializeScalar(&model.patch_ver_);
    model.threshold_type_ = model.GetThresholdType();
    model.leaf_output_type_ = model.GetLeafOutputType();
    mixin_->SerializeScalar(&model.threshold_type_);
    mixin_->SerializeScalar(&model.leaf_output_type_);

    // Number of trees
    model.num_tree_ = static_cast<std::uint64_t>(model.GetNumTree());
    mixin_->SerializeScalar(&model.num_tree_);

    // Header 2
    mixin_->SerializeScalar(&model.num_feature);
    mixin_->SerializeScalar(&model.task_type);
    mixin_->SerializeScalar(&model.average_tree_output);
    mixin_->SerializeScalar(&model.num_target);
    mixin_->SerializeArray(&model.num_class);
    mixin_->SerializeArray(&model.leaf_vector_shape);
    mixin_->SerializeArray(&model.target_id);
    mixin_->SerializeArray(&model.class_id);
    mixin_->SerializeString(&model.postprocessor);
    mixin_->SerializeScalar(&model.sigmoid_alpha);
    mixin_->SerializeScalar(&model.ratio_c);
    mixin_->SerializeArray(&model.base_scores);
    mixin_->SerializeString(&model.attributes);

    /* Extension Slot 1: Per-model optional fields -- to be added later */
    model.num_opt_field_per_model_ = 0;
    mixin_->SerializeScalar(&model.num_opt_field_per_model_);
  }

  void SerializeTrees(Model& model) {
    std::visit(
        [&](auto&& concrete_model) {
          TREELITE_CHECK_EQ(concrete_model.trees.size(), model.num_tree_)
              << "Incorrect number of trees in the model";
          for (auto& tree : concrete_model.trees) {
            SerializeTree(tree);
          }
        },
        model.variant_);
  }

  template <typename ThresholdType, typename LeafOutputType>
  void SerializeTree(Tree<ThresholdType, LeafOutputType>& tree) {
    mixin_->SerializeScalar(&tree.num_nodes);
    mixin_->SerializeScalar(&tree.has_categorical_split_);
    mixin_->SerializeArray(&tree.node_type_);
    mixin_->SerializeArray(&tree.cleft_);
    mixin_->SerializeArray(&tree.cright_);
    mixin_->SerializeArray(&tree.split_index_);
    mixin_->SerializeArray(&tree.default_left_);
    mixin_->SerializeArray(&tree.leaf_value_);
    mixin_->SerializeArray(&tree.threshold_);
    mixin_->SerializeArray(&tree.cmp_);
    mixin_->SerializeArray(&tree.category_list_right_child_);
    mixin_->SerializeArray(&tree.leaf_vector_);
    mixin_->SerializeArray(&tree.leaf_vector_begin_);
    mixin_->SerializeArray(&tree.leaf_vector_end_);
    mixin_->SerializeArray(&tree.category_list_);
    mixin_->SerializeArray(&tree.category_list_begin_);
    mixin_->SerializeArray(&tree.category_list_end_);

    // Node statistics
    mixin_->SerializeArray(&tree.data_count_);
    mixin_->SerializeArray(&tree.data_count_present_);
    mixin_->SerializeArray(&tree.sum_hess_);
    mixin_->SerializeArray(&tree.sum_hess_present_);
    mixin_->SerializeArray(&tree.gain_);
    mixin_->SerializeArray(&tree.gain_present_);

    /* Extension slot 2: Per-tree optional fields -- to be added later */
    tree.num_opt_field_per_tree_ = 0;
    mixin_->SerializeScalar(&tree.num_opt_field_per_tree_);

    /* Extension slot 3: Per-node optional fields -- to be added later */
    tree.num_opt_field_per_node_ = 0;
    mixin_->SerializeScalar(&tree.num_opt_field_per_node_);
  }

 private:
  std::shared_ptr<MixIn> mixin_;
};

template <typename MixIn>
class Deserializer {
 public:
  explicit Deserializer(std::shared_ptr<MixIn> mixin) : mixin_(mixin) {}

  std::unique_ptr<Model> DeserializeHeaderAndCreateModel() {
    // Header 1
    std::int32_t major_ver, minor_ver, patch_ver;
    mixin_->DeserializeScalar(&major_ver);
    mixin_->DeserializeScalar(&minor_ver);
    mixin_->DeserializeScalar(&patch_ver);
    if (major_ver != TREELITE_VER_MAJOR) {
      TREELITE_CHECK(major_ver == 3 && minor_ver == 9)
          << "Cannot load model from a different major Treelite version or "
          << "a version before 3.9.0." << std::endl
          << "Currently running Treelite version " << TREELITE_VER_MAJOR << "."
          << TREELITE_VER_MINOR << "." << TREELITE_VER_PATCH << std::endl
          << "The model checkpoint was generated from Treelite version " << major_ver << "."
          << minor_ver << "." << patch_ver;
      return DeserializeHeaderAndCreateModelV3(major_ver, minor_ver, patch_ver);
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
    mixin_->DeserializeScalar(&threshold_type);
    mixin_->DeserializeScalar(&leaf_output_type);

    std::unique_ptr<Model> model = Model::Create(threshold_type, leaf_output_type);
    model->major_ver_ = major_ver;
    model->minor_ver_ = minor_ver;
    model->patch_ver_ = patch_ver;

    // Number of trees
    mixin_->DeserializeScalar(&model->num_tree_);

    // Header 2
    mixin_->DeserializeScalar(&model->num_feature);
    mixin_->DeserializeScalar(&model->task_type);
    mixin_->DeserializeScalar(&model->average_tree_output);
    mixin_->DeserializeScalar(&model->num_target);
    mixin_->DeserializeArray(&model->num_class);
    mixin_->DeserializeArray(&model->leaf_vector_shape);
    mixin_->DeserializeArray(&model->target_id);
    mixin_->DeserializeArray(&model->class_id);
    mixin_->DeserializeString(&model->postprocessor);
    mixin_->DeserializeScalar(&model->sigmoid_alpha);
    mixin_->DeserializeScalar(&model->ratio_c);
    mixin_->DeserializeArray(&model->base_scores);
    mixin_->DeserializeString(&model->attributes);

    /* Extension Slot 1: Per-model optional fields -- to be added later */
    mixin_->DeserializeScalar(&model->num_opt_field_per_model_);
    // Ignore extra fields; the input is likely from a later version of Treelite
    for (std::int32_t i = 0; i < model->num_opt_field_per_model_; ++i) {
      mixin_->SkipOptionalField();
    }

    return model;
  }

  void DeserializeTrees(Model& model) {
    if (model.major_ver_ == 3) {
      std::visit(
          [&](auto&& concrete_model) {
            concrete_model.trees.clear();
            for (std::uint64_t i = 0; i < model.num_tree_; ++i) {
              concrete_model.trees.emplace_back();
              DeserializeTreeV3(concrete_model.trees.back());
            }
          },
          model.variant_);
      return;
    }
    std::visit(
        [&](auto&& concrete_model) {
          concrete_model.trees.clear();
          for (std::uint64_t i = 0; i < model.num_tree_; ++i) {
            concrete_model.trees.emplace_back();
            DeserializeTree(concrete_model.trees.back());
          }
        },
        model.variant_);
  }

  template <typename ThresholdType, typename LeafOutputType>
  void DeserializeTree(Tree<ThresholdType, LeafOutputType>& tree) {
    mixin_->DeserializeScalar(&tree.num_nodes);
    mixin_->DeserializeScalar(&tree.has_categorical_split_);
    mixin_->DeserializeArray(&tree.node_type_);
    mixin_->DeserializeArray(&tree.cleft_);
    mixin_->DeserializeArray(&tree.cright_);
    mixin_->DeserializeArray(&tree.split_index_);
    mixin_->DeserializeArray(&tree.default_left_);
    mixin_->DeserializeArray(&tree.leaf_value_);
    mixin_->DeserializeArray(&tree.threshold_);
    mixin_->DeserializeArray(&tree.cmp_);
    mixin_->DeserializeArray(&tree.category_list_right_child_);
    mixin_->DeserializeArray(&tree.leaf_vector_);
    mixin_->DeserializeArray(&tree.leaf_vector_begin_);
    mixin_->DeserializeArray(&tree.leaf_vector_end_);
    mixin_->DeserializeArray(&tree.category_list_);
    mixin_->DeserializeArray(&tree.category_list_begin_);
    mixin_->DeserializeArray(&tree.category_list_end_);

    // Node statistics
    mixin_->DeserializeArray(&tree.data_count_);
    mixin_->DeserializeArray(&tree.data_count_present_);
    mixin_->DeserializeArray(&tree.sum_hess_);
    mixin_->DeserializeArray(&tree.sum_hess_present_);
    mixin_->DeserializeArray(&tree.gain_);
    mixin_->DeserializeArray(&tree.gain_present_);

    /* Extension slot 2: Per-tree optional fields -- to be added later */
    mixin_->DeserializeScalar(&tree.num_opt_field_per_tree_);
    // Ignore extra fields; the input is likely from a later version of Treelite
    for (std::int32_t i = 0; i < tree.num_opt_field_per_tree_; ++i) {
      mixin_->SkipOptionalField();
    }

    /* Extension slot 3: Per-node optional fields -- to be added later */
    mixin_->DeserializeScalar(&tree.num_opt_field_per_node_);
    // Ignore extra fields; the input is likely from a later version of Treelite
    for (std::int32_t i = 0; i < tree.num_opt_field_per_node_; ++i) {
      mixin_->SkipOptionalField();
    }
  }

 private:
  std::shared_ptr<MixIn> mixin_;

  std::unique_ptr<Model> DeserializeHeaderAndCreateModelV3(
      std::int32_t major_ver, std::int32_t minor_ver, std::int32_t patch_ver) {
    TypeInfo threshold_type, leaf_output_type;
    mixin_->DeserializeScalar(&threshold_type);
    mixin_->DeserializeScalar(&leaf_output_type);
    std::unique_ptr<Model> model = Model::Create(threshold_type, leaf_output_type);
    model->major_ver_ = major_ver;
    model->minor_ver_ = minor_ver;
    model->patch_ver_ = patch_ver;

    // Number of trees
    mixin_->DeserializeScalar(&model->num_tree_);

    // Header 2
    mixin_->DeserializeScalar(&model->num_feature);
    TaskTypeV3 task_type;
    mixin_->DeserializeScalar(&task_type);
    TREELITE_CHECK(task_type != TaskTypeV3::kMultiClfCategLeaf)
        << "Task type kMultiClfCategLeaf is no longer supported in Treelite 4.0.";
    mixin_->DeserializeScalar(&model->average_tree_output);

    TaskParamV3 task_param;
    mixin_->DeserializeScalar(&task_param);
    TREELITE_CHECK(task_param.output_type == TaskParamV3::OutputType::kFloat)
        << "Integer outputs are no longer supported in Treelite 4.0.";
    model->num_target = 1;  // All models from 3.x are single-target
    auto const num_class = static_cast<std::int32_t>(task_param.num_class);
    model->num_class = std::vector<std::int32_t>{num_class};
    model->target_id = std::vector<std::int32_t>(model->num_tree_, 0);
    model->class_id = std::vector<std::int32_t>(model->num_tree_, 0);
    model->task_type = TaskType::kRegressor;
    if (task_type == TaskTypeV3::kMultiClfGrovePerClass) {
      TREELITE_CHECK(task_param.grove_per_class) << "Invariant violated";
      model->task_type = TaskType::kMultiClf;
      model->leaf_vector_shape = std::vector<std::int32_t>{1, 1};
      for (std::int32_t i = 0; i < model->num_tree_; ++i) {
        model->class_id[i] = i % num_class;
      }
    } else if (task_type == TaskTypeV3::kMultiClfProbDistLeaf) {
      TREELITE_CHECK(!task_param.grove_per_class) << "Invariant violated";
      model->task_type = TaskType::kMultiClf;
      model->leaf_vector_shape
          = std::vector<std::int32_t>{1, static_cast<std::int32_t>(task_param.num_class)};
      for (std::int32_t i = 0; i < model->num_tree_; ++i) {
        model->class_id[i] = -1;
      }
    } else {
      TREELITE_CHECK(task_type == TaskTypeV3::kBinaryClfRegr && !task_param.grove_per_class)
          << "Invariant violated";
      model->leaf_vector_shape = std::vector<std::int32_t>{1, 1};
    }
    TREELITE_CHECK_EQ(model->leaf_vector_shape[0] * model->leaf_vector_shape[1],
        static_cast<std::int32_t>(task_param.leaf_vector_size))
        << "Invariant violated";

    ModelParamV3 model_param;
    mixin_->DeserializeScalar(&model_param);
    model->postprocessor = std::string(model_param.pred_transform);
    if (model->postprocessor == "max_index") {
      model->postprocessor = "softmax";  // max_index no longer supported; force it to softmax
    }
    if (model->postprocessor == "sigmoid") {  // Heuristic: sigmoid indicates binary classifier
      model->task_type = TaskType::kBinaryClf;
    }
    model->sigmoid_alpha = model_param.sigmoid_alpha;
    model->ratio_c = model_param.ratio_c;
    model->base_scores = std::vector<double>{static_cast<double>(model_param.global_bias)};
    model->attributes = "{}";

    // Extension Slot 1: Per-model optional fields
    mixin_->DeserializeScalar(&model->num_opt_field_per_model_);
    TREELITE_CHECK_EQ(model->num_opt_field_per_model_, 0)
        << "Extension slot 1 must be unused in Treelite 3.x";

    return model;
  }

  template <typename ThresholdType, typename LeafOutputType>
  void DeserializeTreeV3(Tree<ThresholdType, LeafOutputType>& tree) {
    mixin_->DeserializeScalar(&tree.num_nodes);
    mixin_->DeserializeScalar(&tree.has_categorical_split_);
    ContiguousArray<NodeV3<ThresholdType, LeafOutputType>> nodes;
    mixin_->DeserializeArray(&nodes);
    for (std::size_t node_id = 0; node_id < nodes.Size(); ++node_id) {
      NodeV3<ThresholdType, LeafOutputType> const& node = nodes[node_id];
      if (node.IsLeaf()) {  // Leaf node
        tree.node_type_.PushBack(TreeNodeType::kLeafNode);
        tree.leaf_value_.PushBack(node.LeafValue());
        tree.threshold_.PushBack(static_cast<ThresholdType>(0));
      } else {  // Internal node
        if (node.split_type_ == SplitFeatureTypeV3::kNumerical) {
          tree.node_type_.PushBack(TreeNodeType::kNumericalTestNode);
          tree.threshold_.PushBack(node.Threshold());
        } else {
          tree.node_type_.PushBack(TreeNodeType::kCategoricalTestNode);
          tree.threshold_.PushBack(static_cast<ThresholdType>(0));
        }
        tree.leaf_value_.PushBack(static_cast<LeafOutputType>(0));
      }
      tree.cleft_.PushBack(node.cleft_);
      tree.cright_.PushBack(node.cright_);
      tree.split_index_.PushBack(node.SplitIndex());
      tree.default_left_.PushBack(node.DefaultLeft());
      tree.cmp_.PushBack(node.cmp_);
      tree.category_list_right_child_.PushBack(node.categories_list_right_child_);

      tree.data_count_.PushBack(node.data_count_);
      tree.data_count_present_.PushBack(node.data_count_present_);
      tree.sum_hess_.PushBack(node.sum_hess_);
      tree.sum_hess_present_.PushBack(node.sum_hess_present_);
      tree.gain_.PushBack(node.gain_);
      tree.gain_present_.PushBack(node.gain_present_);
    }

    static_assert(sizeof(std::size_t) == sizeof(std::uint64_t), "Wrong size for size_t");
    mixin_->DeserializeArray(&tree.leaf_vector_);
    mixin_->DeserializeArray(&tree.leaf_vector_begin_);
    mixin_->DeserializeArray(&tree.leaf_vector_end_);
    mixin_->DeserializeArray(&tree.category_list_);
    ContiguousArray<std::uint64_t> cat_offsets;
    mixin_->DeserializeArray(&cat_offsets);
    TREELITE_CHECK_EQ(tree.num_nodes + 1, cat_offsets.Size()) << "Invariant violated";
    for (std::size_t node_id = 0; node_id < tree.num_nodes; ++node_id) {
      tree.category_list_begin_.PushBack(cat_offsets[node_id]);
      tree.category_list_end_.PushBack(cat_offsets[node_id + 1]);
    }

    // Extension slot 2: Per-tree optional fields
    mixin_->DeserializeScalar(&tree.num_opt_field_per_tree_);
    TREELITE_CHECK_EQ(tree.num_opt_field_per_tree_, 0)
        << "Extension slot 2 must be unused in Treelite 3.x";

    // Extension slot 3: Per-node optional fields
    mixin_->DeserializeScalar(&tree.num_opt_field_per_node_);
    TREELITE_CHECK_EQ(tree.num_opt_field_per_node_, 0)
        << "Extension slot 3 must be unused in Treelite 3.x";
  }
};

}  // namespace detail::serializer

std::vector<PyBufferFrame> Model::SerializeToPyBuffer() {
  auto mixin = std::make_shared<detail::serializer::PyBufferSerializerMixIn>();
  detail::serializer::Serializer<detail::serializer::PyBufferSerializerMixIn> serializer{mixin};
  serializer.SerializeHeader(*this);
  serializer.SerializeTrees(*this);
  return mixin->GetFrames();
}

std::unique_ptr<Model> Model::DeserializeFromPyBuffer(std::vector<PyBufferFrame> const& frames) {
  auto mixin = std::make_shared<detail::serializer::PyBufferDeserializerMixIn>(frames);
  detail::serializer::Deserializer<detail::serializer::PyBufferDeserializerMixIn> deserializer{
      mixin};
  std::unique_ptr<Model> model = deserializer.DeserializeHeaderAndCreateModel();
  deserializer.DeserializeTrees(*model);
  return model;
}

void Model::SerializeToStream(std::ostream& os) {
  auto mixin = std::make_shared<detail::serializer::StreamSerializerMixIn>(os);
  detail::serializer::Serializer<detail::serializer::StreamSerializerMixIn> serializer{mixin};
  serializer.SerializeHeader(*this);
  serializer.SerializeTrees(*this);
}

std::unique_ptr<Model> Model::DeserializeFromStream(std::istream& is) {
  auto mixin = std::make_shared<detail::serializer::StreamDeserializerMixIn>(is);
  detail::serializer::Deserializer<detail::serializer::StreamDeserializerMixIn> deserializer{mixin};
  std::unique_ptr<Model> model = deserializer.DeserializeHeaderAndCreateModel();
  deserializer.DeserializeTrees(*model);
  return model;
}

}  // namespace treelite
