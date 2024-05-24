/*!
 * Copyright (c) 2023 by Contributors
 * \file model_builder.cc
 * \brief C++ API for constructing Model objects
 * \author Hyunsu Cho
 */
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/model_builder.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>

#include "./detail/json_parsing.h"

namespace treelite::model_builder {

namespace detail {

void ConfigurePostProcessor(Model* model, PostProcessorFunc const& postprocessor) {
  if (postprocessor.name == "sigmoid" || postprocessor.name == "multiclass_ova") {
    model->sigmoid_alpha = 1.0f;
    auto itr = postprocessor.config.find("sigmoid_alpha");
    if (itr != postprocessor.config.end() && std::holds_alternative<double>(itr->second)) {
      model->sigmoid_alpha = static_cast<float>(std::get<double>(itr->second));
    }
  } else if (postprocessor.name == "exponential_standard_ratio") {
    model->ratio_c = 1.0f;
    auto itr = postprocessor.config.find("ratio_c");
    if (itr != postprocessor.config.end() && std::holds_alternative<double>(itr->second)) {
      model->ratio_c = static_cast<float>(std::get<double>(itr->second));
    }
  }
}

enum class ModelBuilderState : std::int8_t {
  kExpectTree,
  kExpectNode,
  kExpectDetail,
  kNodeComplete,
  kModelComplete
};

template <typename ThresholdT, typename LeafOutputT>
class ModelBuilderImpl : public ModelBuilder {
 public:
  ModelBuilderImpl()
      : expected_num_tree_{},
        expected_leaf_size_{},
        model_{Model::Create<ThresholdT, LeafOutputT>()},
        current_tree_{},
        node_id_map_{},
        current_node_key_{},
        current_node_id_{},
        current_state_{ModelBuilderState::kExpectTree},
        metadata_initialized_{false},
        flag_check_orphaned_nodes_{true} {}

  ModelBuilderImpl(Metadata const& metadata, TreeAnnotation const& tree_annotation,
      PostProcessorFunc const& postprocessor, std::vector<double> const& base_scores,
      std::optional<std::string> const& attributes)
      : expected_num_tree_{},
        expected_leaf_size_{},
        model_{Model::Create<ThresholdT, LeafOutputT>()},
        current_tree_{},
        node_id_map_{},
        current_node_key_{},
        current_node_id_{},
        current_state_{ModelBuilderState::kExpectTree},
        metadata_initialized_{false},
        flag_check_orphaned_nodes_{true} {
    InitializeMetadataImpl(metadata, tree_annotation, postprocessor, base_scores, attributes);
  }

  void SetValidationFlag(std::string const& flag, bool value) override {
    if (flag == "check_orphaned_nodes") {
      flag_check_orphaned_nodes_ = value;
    }
  }

  void StartTree() override {
    CheckStateWithDiagnostic("StartTree()", {ModelBuilderState::kExpectTree}, current_state_);

    current_tree_ = Tree<ThresholdT, LeafOutputT>();
    current_tree_.Init();

    current_state_ = ModelBuilderState::kExpectNode;
  }

  void EndTree() override {
    CheckStateWithDiagnostic("EndTree()", {ModelBuilderState::kExpectNode}, current_state_);

    TREELITE_CHECK_GT(current_tree_.num_nodes, 0)
        << "Cannot have an empty tree. Please supply at least one node.";

    std::vector<bool> orphaned(current_tree_.num_nodes, true);
    orphaned[0] = false;  // Root node is by definition not orphaned
    for (std::int32_t i = 0; i < current_tree_.num_nodes; ++i) {
      if (!current_tree_.IsLeaf(i)) {
        // Translate left and right child ID to use internal IDs
        int const left_key = current_tree_.LeftChild(i);
        int const right_key = current_tree_.RightChild(i);
        int cleft, cright;
        try {
          cleft = node_id_map_.at(left_key);
        } catch (std::out_of_range const& ex) {
          TREELITE_LOG(FATAL) << "Node with key " << left_key << " not found";
        }
        try {
          cright = node_id_map_.at(right_key);
        } catch (std::out_of_range const& ex) {
          TREELITE_LOG(FATAL) << "Node with key " << right_key << " not found";
        }
        current_tree_.SetChildren(i, cleft, cright);
        orphaned[cleft] = false;
        orphaned[cright] = false;
      }
    }
    if (flag_check_orphaned_nodes_) {
      auto itr = std::find(orphaned.begin(), orphaned.end(), true);
      if (itr != orphaned.end()) {
        auto orphaned_node_id = *itr;
        for (auto [k, v] : node_id_map_) {
          if (v == orphaned_node_id) {
            TREELITE_LOG(FATAL) << "Node with key " << k << " is orphaned -- it cannot be reached "
                                << "from the root node";
          }
        }
        TREELITE_LOG(FATAL) << "Node at index " << orphaned_node_id << " is orphaned "
                            << "-- it cannot be reached from the root node";
      }
    }

    auto& trees = std::get<ModelPreset<ThresholdT, LeafOutputT>>(model_->variant_).trees;
    trees.push_back(std::move(current_tree_));

    node_id_map_.clear();
    current_state_ = ModelBuilderState::kExpectTree;
  }

  void StartNode(int node_key) override {
    CheckStateWithDiagnostic("StartNode()", {ModelBuilderState::kExpectNode}, current_state_);
    TREELITE_CHECK_GE(node_key, 0) << "Node key cannot be negative";

    int node_id = current_tree_.AllocNode();
    current_node_key_ = node_key;
    current_node_id_ = node_id;
    TREELITE_CHECK_EQ(node_id_map_.count(node_key), 0) << "Key " << node_key << " is duplicated";
    node_id_map_[node_key] = node_id;

    current_state_ = ModelBuilderState::kExpectDetail;
  }

  void EndNode() override {
    CheckStateWithDiagnostic("EndNode()", {ModelBuilderState::kNodeComplete}, current_state_);
    current_state_ = ModelBuilderState::kExpectNode;
  }

  void NumericalTest(std::int32_t split_index, double threshold, bool default_left, Operator cmp,
      int left_child_key, int right_child_key) override {
    CheckStateWithDiagnostic("NumericalTest()", {ModelBuilderState::kExpectDetail}, current_state_);
    TREELITE_CHECK(left_child_key >= 0 && right_child_key >= 0) << "Node key cannot be negative";
    TREELITE_CHECK(current_node_key_ != left_child_key && current_node_key_ != right_child_key)
        << "Duplicated key " << current_node_key_ << " used by a child node";
    TREELITE_CHECK_NE(left_child_key, right_child_key) << "Left and child nodes must be unique";
    if (metadata_initialized_) {
      TREELITE_CHECK_LT(split_index, model_->num_feature)
          << "split_index must be less than num_feature (" << model_->num_feature << ")";
    }

    current_tree_.SetNumericalTest(current_node_id_, split_index, threshold, default_left, cmp);
    // Note: children IDs needs to be later translated into internal IDs
    current_tree_.SetChildren(current_node_id_, left_child_key, right_child_key);

    current_state_ = ModelBuilderState::kNodeComplete;
  }

  void CategoricalTest(std::int32_t split_index, bool default_left,
      std::vector<std::uint32_t> const& category_list, bool category_list_right_child,
      int left_child_key, int right_child_key) override {
    CheckStateWithDiagnostic(
        "CategoricalTest()", {ModelBuilderState::kExpectDetail}, current_state_);
    TREELITE_CHECK(left_child_key >= 0 && right_child_key >= 0) << "Node key cannot be negative";
    TREELITE_CHECK(current_node_key_ != left_child_key && current_node_key_ != right_child_key)
        << "Duplicated key " << current_node_key_ << " used by a child node";
    TREELITE_CHECK_NE(left_child_key, right_child_key) << "Left and child nodes must be unique";
    if (metadata_initialized_) {
      TREELITE_CHECK_LT(split_index, model_->num_feature)
          << "split_index must be less than num_feature (" << model_->num_feature << ")";
    }

    current_tree_.SetCategoricalTest(
        current_node_id_, split_index, default_left, category_list, category_list_right_child);
    // Note: children IDs needs to be later translated into internal IDs
    current_tree_.SetChildren(current_node_id_, left_child_key, right_child_key);

    current_state_ = ModelBuilderState::kNodeComplete;
  }

  void LeafScalar(double leaf_value) override {
    CheckStateWithDiagnostic("LeafScalar()", {ModelBuilderState::kExpectDetail}, current_state_);
    if (metadata_initialized_) {
      TREELITE_CHECK_EQ(expected_leaf_size_, 1)
          << "Cannot call LeafScalar(). Expected leaf output of length " << expected_leaf_size_;
    }

    current_tree_.SetLeaf(current_node_id_, static_cast<ThresholdT>(leaf_value));

    current_state_ = ModelBuilderState::kNodeComplete;
  }

  void LeafVector(std::vector<float> const& leaf_vector) override {
    CheckStateWithDiagnostic("LeafVector()", {ModelBuilderState::kExpectDetail}, current_state_);
    if (metadata_initialized_) {
      TREELITE_CHECK_EQ(expected_leaf_size_, leaf_vector.size())
          << "Expected leaf output of length " << expected_leaf_size_;
    }

    if constexpr (std::is_same_v<LeafOutputT, float>) {
      current_tree_.SetLeafVector(current_node_id_, leaf_vector);
    } else if constexpr (std::is_same_v<LeafOutputT, double>) {
      TREELITE_LOG(FATAL) << "Mismatched type for leaf vector. Expected: float32, Got: float64";
    }

    current_state_ = ModelBuilderState::kNodeComplete;
  }

  void LeafVector(std::vector<double> const& leaf_vector) override {
    CheckStateWithDiagnostic("LeafVector()", {ModelBuilderState::kExpectDetail}, current_state_);
    if (metadata_initialized_) {
      TREELITE_CHECK_EQ(expected_leaf_size_, leaf_vector.size())
          << "Expected leaf output of length " << expected_leaf_size_;
    }

    if constexpr (std::is_same_v<LeafOutputT, float>) {
      TREELITE_LOG(FATAL) << "Mismatched type for leaf vector. Expected: float64, Got: float32";
    } else if constexpr (std::is_same_v<LeafOutputT, double>) {
      current_tree_.SetLeafVector(current_node_id_, leaf_vector);
    }

    current_state_ = ModelBuilderState::kNodeComplete;
  }

  void Gain(double gain) override {
    CheckStateWithDiagnostic("Gain()",
        {ModelBuilderState::kExpectDetail, ModelBuilderState::kNodeComplete}, current_state_);

    current_tree_.SetGain(current_node_id_, gain);
  }

  void DataCount(std::uint64_t data_count) override {
    CheckStateWithDiagnostic("DataCount()",
        {ModelBuilderState::kExpectDetail, ModelBuilderState::kNodeComplete}, current_state_);

    current_tree_.SetDataCount(current_node_id_, data_count);
  }

  void SumHess(double sum_hess) override {
    CheckStateWithDiagnostic("SumHess()",
        {ModelBuilderState::kExpectDetail, ModelBuilderState::kNodeComplete}, current_state_);

    current_tree_.SetSumHess(current_node_id_, sum_hess);
  }

  std::unique_ptr<Model> CommitModel() override {
    CheckStateWithDiagnostic("CommitModel()", {ModelBuilderState::kExpectTree}, current_state_);
    TREELITE_CHECK(metadata_initialized_) << "The model does not yet have a valid metadata. "
                                          << "Please add metadata by calling InitializeMetadata().";
    TREELITE_CHECK_EQ(model_->GetNumTree(), expected_num_tree_)
        << "Expected " << expected_num_tree_ << " trees but only got " << model_->GetNumTree()
        << " trees instead";
    current_state_ = ModelBuilderState::kModelComplete;
    return std::move(model_);
  }

  void InitializeMetadata(Metadata const& metadata, TreeAnnotation const& tree_annotation,
      PostProcessorFunc const& postprocessor, std::vector<double> const& base_scores,
      std::optional<std::string> const& attributes) override {
    InitializeMetadataImpl(metadata, tree_annotation, postprocessor, base_scores, attributes);
  }

 private:
  std::int32_t expected_num_tree_;
  std::int32_t expected_leaf_size_;
  std::unique_ptr<Model> model_;
  Tree<ThresholdT, LeafOutputT> current_tree_;
  std::map<int, int> node_id_map_;  // user-defined ID -> internal ID
  int current_node_key_;  // current node ID (user-defined)
  int current_node_id_;  // current node ID (internal)
  ModelBuilderState current_state_;
  bool metadata_initialized_{false};
  bool flag_check_orphaned_nodes_{true};

  void CheckStateWithDiagnostic(std::string const& func_name,
      std::vector<ModelBuilderState> const& valid_states, ModelBuilderState actual_state) {
    auto error = [&](std::string const& msg) {
      TREELITE_LOG(FATAL) << "Unexpected call to " << func_name << ". " << msg;
    };
    if (std::find(valid_states.begin(), valid_states.end(), actual_state) == valid_states.end()) {
      switch (actual_state) {
      case ModelBuilderState::kExpectTree:
        error("Expected a call to StartTree() or CommitModel().");
        break;
      case ModelBuilderState::kExpectNode:
        error("Expected a call to StartNode() or EndTree().");
      case ModelBuilderState::kExpectDetail:
        error(
            "Expected a call to NumericalTest(), CategoricalTest(), LeafScalar(), LeafVector(), "
            "Gain(), DataCount(), or SumHess().");
        break;
      case ModelBuilderState::kNodeComplete:
        error("Expected a call to EndNode(), Gain(), DataCount(), or SumHess().");
        break;
      case ModelBuilderState::kModelComplete:
      default:
        error("The final model has been already produced with CommitModel().");
        break;
      }
    }
  }

  void InitializeMetadataImpl(Metadata const& metadata, TreeAnnotation const& tree_annotation,
      PostProcessorFunc const& postprocessor, std::vector<double> const& base_scores,
      std::optional<std::string> const& attributes) {
    TREELITE_CHECK(!metadata_initialized_) << "Metadata must be initialized only once";
    std::int32_t const num_tree = tree_annotation.num_tree;
    std::int32_t const num_target = metadata.num_target;

    model_->num_feature = metadata.num_feature;
    model_->task_type = metadata.task_type;
    model_->average_tree_output = metadata.average_tree_output;
    model_->num_target = num_target;
    model_->num_class = metadata.num_class;
    model_->leaf_vector_shape = std::vector<std::int32_t>(
        metadata.leaf_vector_shape.begin(), metadata.leaf_vector_shape.end());

    // Validate target_id and class_id
    for (std::int32_t i = 0; i < num_tree; ++i) {
      if (tree_annotation.target_id[i] >= 0) {
        TREELITE_CHECK_LT(tree_annotation.target_id[i], num_target)
            << "Element " << i << " of target_id is out of range. Revise it to be smaller than "
            << "num_target (" << num_target << ")";
      }
    }
    model_->target_id = tree_annotation.target_id;
    for (std::int32_t i = 0; i < num_tree; ++i) {
      if (tree_annotation.class_id[i] >= 0 && tree_annotation.target_id[i] >= 0) {
        TREELITE_CHECK_LT(
            tree_annotation.class_id[i], metadata.num_class[tree_annotation.target_id[i]])
            << "Element " << i << " of class_id is out of range. Revise it to be smaller than "
            << "num_class[target_id[" << i << "]] ("
            << metadata.num_class[tree_annotation.target_id[i]] << ")";
      }
    }
    model_->class_id = tree_annotation.class_id;

    model_->postprocessor = postprocessor.name;
    detail::ConfigurePostProcessor(model_.get(), postprocessor);

    std::int32_t const max_num_class
        = *std::max_element(metadata.num_class.begin(), metadata.num_class.end());
    TREELITE_CHECK_EQ(base_scores.size(), num_target * max_num_class);
    model_->base_scores = base_scores;
    if (attributes) {
      model_->attributes = attributes.value();
    } else {
      model_->attributes = "{}";
    }

    expected_num_tree_ = tree_annotation.num_tree;
    expected_leaf_size_ = std::accumulate(metadata.leaf_vector_shape.begin(),
        metadata.leaf_vector_shape.end(), std::int32_t(1), std::multiplies<>{});
    metadata_initialized_ = true;
  }
};

}  // namespace detail

std::unique_ptr<ModelBuilder> GetModelBuilder(TypeInfo threshold_type, TypeInfo leaf_output_type,
    Metadata const& metadata, TreeAnnotation const& tree_annotation,
    PostProcessorFunc const& postprocessor, std::vector<double> const& base_scores,
    std::optional<std::string> const& attributes) {
  TREELITE_CHECK(threshold_type == TypeInfo::kFloat32 || threshold_type == TypeInfo::kFloat64)
      << "threshold_type must be either float32 or float64";
  TREELITE_CHECK(leaf_output_type == threshold_type)
      << "threshold_type must be identical to leaf_output_type";
  if (threshold_type == TypeInfo::kFloat32) {
    return std::make_unique<detail::ModelBuilderImpl<float, float>>(
        metadata, tree_annotation, postprocessor, base_scores, attributes);
  } else {
    return std::make_unique<detail::ModelBuilderImpl<double, double>>(
        metadata, tree_annotation, postprocessor, base_scores, attributes);
  }
}

std::unique_ptr<ModelBuilder> GetModelBuilder(TypeInfo threshold_type, TypeInfo leaf_output_type) {
  TREELITE_CHECK(threshold_type == TypeInfo::kFloat32 || threshold_type == TypeInfo::kFloat64)
      << "threshold_type must be either float32 or float64";
  TREELITE_CHECK(leaf_output_type == threshold_type)
      << "threshold_type must be identical to leaf_output_type";
  if (threshold_type == TypeInfo::kFloat32) {
    return std::make_unique<detail::ModelBuilderImpl<float, float>>();
  } else {
    return std::make_unique<detail::ModelBuilderImpl<double, double>>();
  }
}

std::unique_ptr<ModelBuilder> GetModelBuilder(std::string const& json_str) {
  rapidjson::Document parsed_json;
  parsed_json.Parse(json_str);
  TREELITE_CHECK(!parsed_json.HasParseError())
      << "Error when parsing JSON string: offset " << parsed_json.GetErrorOffset() << ", "
      << rapidjson::GetParseError_En(parsed_json.GetParseError());

  namespace json_parse = detail::json_parse;

  auto threshold_type = TypeInfoFromString(
      json_parse::ObjectMemberHandler<std::string>::Get(parsed_json, "threshold_type"));
  auto leaf_output_type = TypeInfoFromString(
      json_parse::ObjectMemberHandler<std::string>::Get(parsed_json, "leaf_output_type"));
  auto metadata = json_parse::ParseMetadata(parsed_json, "metadata");
  auto tree_annotation = json_parse::ParseTreeAnnotation(parsed_json, "tree_annotation");
  auto postprocessor = json_parse::ParsePostProcessorFunc(parsed_json, "postprocessor");
  auto base_scores
      = json_parse::ObjectMemberHandler<std::vector<double>>::Get(parsed_json, "base_scores");
  auto attributes = json_parse::ParseAttributes(parsed_json, "attributes");

  return GetModelBuilder(threshold_type, leaf_output_type, metadata, tree_annotation, postprocessor,
      base_scores, attributes);
}

}  // namespace treelite::model_builder
