/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file sklearn.cc
 * \brief Model loader for scikit-learn models
 * \author Hyunsu Cho
 */
#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/logging.h>
#include <treelite/model_builder.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cstdint>
#include <experimental/mdspan>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>

namespace treelite::model_loader::sklearn {

namespace detail {

namespace stdex = std::experimental;
// Multidimensional array views. Use row-major (C) layout
template <typename ElemT>
using Array2DView = stdex::mdspan<ElemT, stdex::dextents<std::uint64_t, 2>, stdex::layout_right>;

class RandomForestRegressorMixIn {
 public:
  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      int n_targets, [[maybe_unused]] std::int32_t const* n_classes) {
    n_targets_ = n_targets;
    model_builder::Metadata metadata{n_features, TaskType::kRegressor, true,
        static_cast<std::int32_t>(n_targets), std::vector<std::int32_t>(n_targets, 1),
        {n_targets, 1}};
    std::vector<std::int32_t> const target_id(n_trees, (n_targets > 1 ? -1 : 0));
    model_builder::TreeAnnotation tree_annotation{
        n_trees, target_id, std::vector<std::int32_t>(n_trees, 0)};
    model_builder::PostProcessorFunc postprocessor{"identity"};
    std::vector<double> base_scores(n_targets, 0.0);
    builder.InitializeMetadata(metadata, tree_annotation, postprocessor, base_scores, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] std::int32_t const* n_classes) const {
    TREELITE_CHECK_GT(n_targets_, 0)
        << "n_targets not yet initialized. Was HandleMetadata() called?";
    if (n_targets_ == 1) {
      builder.LeafScalar(value[tree_id][node_id]);
    } else {
      std::vector<double> leafvec(
          &value[tree_id][node_id * n_targets_], &value[tree_id][(node_id + 1) * n_targets_]);
      builder.LeafVector(leafvec);
    }
  }

 private:
  int n_targets_{-1};
};

// Note: Here, we will treat binary classifiers as if they are multi-class classifiers with
// n_classes=2.
class RandomForestClassifierMixIn {
 public:
  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      int n_targets, std::int32_t const* n_classes) {
    n_targets_ = n_targets;
    std::vector<std::int32_t> n_classes_(n_classes, n_classes + n_targets);
    if (!std::all_of(n_classes_.begin(), n_classes_.end(), [](auto e) { return e >= 2; })) {
      TREELITE_LOG(FATAL)
          << "All elements in n_classes must be at least 2. "
          << "Note: For sklearn RandomForestClassifier, binary classifiers will have n_classes=2.";
    }
    max_num_class_ = *std::max_element(n_classes_.begin(), n_classes_.end());
    model_builder::Metadata metadata{n_features, TaskType::kMultiClf, true,
        static_cast<std::int32_t>(n_targets), n_classes_, {n_targets, max_num_class_}};
    model_builder::TreeAnnotation tree_annotation{
        n_trees, std::vector<std::int32_t>(n_trees, -1), std::vector<std::int32_t>(n_trees, -1)};
    model_builder::PostProcessorFunc postprocessor{"identity_multiclass"};
    std::vector<double> base_scores(n_targets * max_num_class_, 0.0);
    builder.InitializeMetadata(metadata, tree_annotation, postprocessor, base_scores, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, std::int32_t const* n_classes) const {
    TREELITE_CHECK_GT(n_targets_, 0)
        << "n_targets not yet initialized. Was HandleMetadata() called?";
    TREELITE_CHECK_GT(max_num_class_, 0)
        << "max_num_class not yet initialized. Was HandleMetadata() called?";
    std::vector<double> leafvec(&value[tree_id][node_id * n_targets_ * max_num_class_],
        &value[tree_id][(node_id + 1) * n_targets_ * max_num_class_]);
    // Compute the probability distribution over label classes
    auto leaf_view = Array2DView<double>(leafvec.data(), n_targets_, max_num_class_);
    for (int target_id = 0; target_id < n_targets_; ++target_id) {
      double norm_factor = 0.0;
      for (std::int32_t class_id = 0; class_id < n_classes[target_id]; ++class_id) {
        norm_factor += leaf_view(target_id, class_id);
      }
      for (std::int32_t class_id = 0; class_id < n_classes[target_id]; ++class_id) {
        leaf_view(target_id, class_id) /= norm_factor;
      }
    }
    builder.LeafVector(leafvec);
  }

 private:
  int n_targets_{-1};
  std::int32_t max_num_class_{-1};
};

class IsolationForestMixIn {
 public:
  explicit IsolationForestMixIn(double ratio_c) : ratio_c_{ratio_c} {}

  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      [[maybe_unused]] int n_targets, [[maybe_unused]] std::int32_t const* n_classes) {
    model_builder::Metadata metadata{n_features, TaskType::kIsolationForest, true, 1, {1}, {1, 1}};
    model_builder::TreeAnnotation tree_annotation{
        n_trees, std::vector<std::int32_t>(n_trees, 0), std::vector<std::int32_t>(n_trees, 0)};

    std::ostringstream oss;
    model_builder::PostProcessorFunc postprocessor{
        "exponential_standard_ratio", {{"ratio_c", ratio_c_}}};

    builder.InitializeMetadata(metadata, tree_annotation, postprocessor, {0.0}, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] std::int32_t const* n_classes) const {
    builder.LeafScalar(value[tree_id][node_id]);
  }

 private:
  double ratio_c_;
};

class GradientBoostingRegressorMixIn {
 public:
  explicit GradientBoostingRegressorMixIn(double base_score) : base_score_{base_score} {}

  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      [[maybe_unused]] int n_targets, [[maybe_unused]] std::int32_t const* n_classes) {
    model_builder::Metadata metadata{n_features, TaskType::kRegressor, false, 1, {1}, {1, 1}};
    model_builder::TreeAnnotation tree_annotation{
        n_trees, std::vector<std::int32_t>(n_trees, 0), std::vector<std::int32_t>(n_trees, 0)};
    model_builder::PostProcessorFunc postprocessor{"identity"};
    std::vector<double> base_scores{base_score_};
    builder.InitializeMetadata(metadata, tree_annotation, postprocessor, base_scores, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] std::int32_t const* n_classes) const {
    builder.LeafScalar(value[tree_id][node_id]);
  }

 private:
  double base_score_;
};

class GradientBoostingBinaryClassifierMixIn {
 public:
  explicit GradientBoostingBinaryClassifierMixIn(double base_score) : base_score_(base_score) {}

  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      [[maybe_unused]] int n_targets, std::int32_t const* n_classes) {
    model_builder::Metadata metadata{n_features, TaskType::kBinaryClf, false, 1, {1}, {1, 1}};
    std::vector<std::int32_t> target_id(n_trees, 0);
    std::vector<std::int32_t> class_id(n_trees, 0);
    model_builder::TreeAnnotation tree_annotation{n_trees, target_id, class_id};
    model_builder::PostProcessorFunc postprocessor{"sigmoid"};
    builder.InitializeMetadata(
        metadata, tree_annotation, postprocessor, {base_score_}, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] std::int32_t const* n_classes) const {
    builder.LeafScalar(value[tree_id][node_id]);
  }

 private:
  double base_score_;
};

class GradientBoostingMulticlassClassifierMixIn {
 public:
  explicit GradientBoostingMulticlassClassifierMixIn(std::vector<double> const& base_scores)
      : base_scores_(base_scores) {}

  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      [[maybe_unused]] int n_targets, std::int32_t const* n_classes) {
    model_builder::Metadata metadata{
        n_features, TaskType::kMultiClf, false, 1, {n_classes[0]}, {1, 1}};
    std::vector<std::int32_t> target_id(n_trees, 0);
    std::vector<std::int32_t> class_id(n_trees);
    for (int tree_id = 0; tree_id < n_trees; ++tree_id) {
      class_id[tree_id] = tree_id % n_classes[0];
    }
    model_builder::TreeAnnotation tree_annotation{n_trees, target_id, class_id};
    model_builder::PostProcessorFunc postprocessor{"softmax"};
    builder.InitializeMetadata(
        metadata, tree_annotation, postprocessor, base_scores_, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] std::int32_t const* n_classes) const {
    builder.LeafScalar(value[tree_id][node_id]);
  }

 private:
  std::vector<double> base_scores_;
};

class HistGradientBoostingRegressorMixIn {
 public:
  explicit HistGradientBoostingRegressorMixIn(double base_score) : base_score_{base_score} {}

  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      [[maybe_unused]] int n_targets, [[maybe_unused]] std::int32_t const* n_classes) {
    model_builder::Metadata metadata{n_features, TaskType::kRegressor, false, 1, {1}, {1, 1}};
    model_builder::TreeAnnotation tree_annotation{
        n_trees, std::vector<std::int32_t>(n_trees, 0), std::vector<std::int32_t>(n_trees, 0)};
    model_builder::PostProcessorFunc postprocessor{"identity"};
    std::vector<double> base_scores{base_score_};
    builder.InitializeMetadata(metadata, tree_annotation, postprocessor, base_scores, std::nullopt);
  }

  void HandleLeafNode(model_builder::ModelBuilder& builder, int tree_id, int node_id,
      double const** value, [[maybe_unused]] std::int32_t const* n_classes) const {
    builder.LeafScalar(value[tree_id][node_id]);
  }

 private:
  double base_score_;
};

class HistGradientBoostingBinaryClassifierMixIn {
 public:
  explicit HistGradientBoostingBinaryClassifierMixIn(double base_score) : base_score_{base_score} {}

  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      [[maybe_unused]] int n_targets, [[maybe_unused]] std::int32_t const* n_classes) {
    model_builder::Metadata metadata{n_features, TaskType::kBinaryClf, false, 1, {1}, {1, 1}};
    model_builder::TreeAnnotation tree_annotation{
        n_trees, std::vector<std::int32_t>(n_trees, 0), std::vector<std::int32_t>(n_trees, 0)};
    model_builder::PostProcessorFunc postprocessor{"sigmoid"};
    std::vector<double> base_scores{base_score_};
    builder.InitializeMetadata(metadata, tree_annotation, postprocessor, base_scores, std::nullopt);
  }

 private:
  double base_score_;
};

class HistGradientBoostingMulticlassClassifierMixIn {
 public:
  explicit HistGradientBoostingMulticlassClassifierMixIn(std::vector<double> const& base_scores)
      : base_scores_{base_scores} {}

  void HandleMetadata(model_builder::ModelBuilder& builder, int n_trees, int n_features,
      [[maybe_unused]] int n_targets, std::int32_t const* n_classes) {
    model_builder::Metadata metadata{
        n_features, TaskType::kMultiClf, false, 1, {n_classes[0]}, {1, 1}};
    std::vector<std::int32_t> target_id(n_trees, 0);
    std::vector<std::int32_t> class_id(n_trees);
    for (int tree_id = 0; tree_id < n_trees; ++tree_id) {
      class_id[tree_id] = tree_id % n_classes[0];
    }
    model_builder::TreeAnnotation tree_annotation{n_trees, target_id, class_id};
    model_builder::PostProcessorFunc postprocessor{"softmax"};
    std::vector<double> base_scores{base_scores_};
    builder.InitializeMetadata(metadata, tree_annotation, postprocessor, base_scores, std::nullopt);
  }

 private:
  std::vector<double> base_scores_;
};

template <typename MixIn>
std::unique_ptr<Model> LoadSKLearnModel(MixIn& mixin, int n_trees, int n_features, int n_targets,
    std::int32_t const* n_classes, std::int64_t const* node_count,
    std::int64_t const** children_left, std::int64_t const** children_right,
    std::int64_t const** feature, double const** threshold, double const** value,
    std::int64_t const** n_node_samples, double const** weighted_n_node_samples,
    double const** impurity) {
  TREELITE_CHECK_GT(n_trees, 0) << "n_trees must be at least 1";
  TREELITE_CHECK_GT(n_features, 0) << "n_features must be at least 1";

  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::GetModelBuilder(TypeInfo::kFloat64, TypeInfo::kFloat64);
  mixin.HandleMetadata(*builder, n_trees, n_features, n_targets, n_classes);

  for (int tree_id = 0; tree_id < n_trees; ++tree_id) {
    std::int64_t const total_sample_cnt = n_node_samples[tree_id][0];
    TREELITE_CHECK_LE(
        node_count[tree_id], static_cast<std::int64_t>(std::numeric_limits<int>::max()))
        << "Too many nodes in the tree";
    int const n_nodes = static_cast<int>(node_count[tree_id]);

    builder->StartTree();
    for (int node_id = 0; node_id < n_nodes; ++node_id) {
      int const left_child_id = static_cast<int>(children_left[tree_id][node_id]);
      int const right_child_id = static_cast<int>(children_right[tree_id][node_id]);
      std::int64_t sample_cnt = n_node_samples[tree_id][node_id];
      double const weighted_sample_cnt = weighted_n_node_samples[tree_id][node_id];

      builder->StartNode(node_id);
      if (left_child_id == -1) {  // leaf node
        mixin.HandleLeafNode(*builder, tree_id, node_id, value, n_classes);
      } else {
        std::int64_t const split_index = feature[tree_id][node_id];
        double const split_cond = threshold[tree_id][node_id];
        std::int64_t const left_child_sample_cnt = n_node_samples[tree_id][left_child_id];
        std::int64_t const right_child_sample_cnt = n_node_samples[tree_id][right_child_id];
        double const gain
            = static_cast<double>(sample_cnt)
              * (impurity[tree_id][node_id]
                  - static_cast<double>(left_child_sample_cnt) * impurity[tree_id][left_child_id]
                        / static_cast<double>(sample_cnt)
                  - static_cast<double>(right_child_sample_cnt) * impurity[tree_id][right_child_id]
                        / static_cast<double>(sample_cnt))
              / static_cast<double>(total_sample_cnt);

        TREELITE_CHECK_LE(split_index, std::numeric_limits<std::int32_t>::max())
            << "split_index too large";
        builder->NumericalTest(static_cast<std::int32_t>(split_index), split_cond, true,
            Operator::kLE, left_child_id, right_child_id);
        builder->Gain(gain);
      }
      builder->DataCount(sample_cnt);
      builder->SumHess(weighted_sample_cnt);
      builder->EndNode();
    }
    builder->EndTree();
  }
  return builder->CommitModel();
}

#pragma pack(1)
template <typename FeatureIdT>
struct HistGradientBoostingNode {
  double value;
  std::uint32_t count;
  FeatureIdT feature_idx;
  double num_threshold;
  std::uint8_t missing_go_to_left;
  std::uint32_t left;
  std::uint32_t right;
  double gain;
  std::uint32_t depth;
  std::uint8_t is_leaf;
  std::uint8_t bin_threshold;
  std::uint8_t is_categorical;
  std::uint32_t bitset_idx;
};
template struct HistGradientBoostingNode<std::int32_t>;
template struct HistGradientBoostingNode<std::int64_t>;
#pragma pack()

static_assert(sizeof(HistGradientBoostingNode<std::int32_t>) == 52);
static_assert(sizeof(HistGradientBoostingNode<std::int64_t>) == 56);

template <typename MixIn, typename FeatureIdT>
std::unique_ptr<treelite::Model> LoadHistGradientBoostingImpl(MixIn& mixin, int n_trees,
    int n_features, std::int32_t n_classes, std::int64_t const* node_count,
    HistGradientBoostingNode<FeatureIdT> const** nodes,
    std::uint32_t const** raw_left_cat_bitsets) {
  TREELITE_CHECK_GT(n_trees, 0) << "n_trees must be at least 1";
  TREELITE_CHECK_GT(n_features, 0) << "n_features must be at least 1";

  std::unique_ptr<model_builder::ModelBuilder> builder
      = model_builder::GetModelBuilder(TypeInfo::kFloat64, TypeInfo::kFloat64);
  mixin.HandleMetadata(*builder, n_trees, n_features, 1, &n_classes);

  auto check = [](std::uint32_t const* bitmap, int val, unsigned int row) {
    return (bitmap[8 * row + val / 32] >> (val % 32)) & 1;
  };

  for (int tree_id = 0; tree_id < n_trees; ++tree_id) {
    TREELITE_CHECK_LE(
        node_count[tree_id], static_cast<std::int64_t>(std::numeric_limits<int>::max()))
        << "Too many nodes in the tree";
    int const n_nodes = static_cast<int>(node_count[tree_id]);

    builder->StartTree();
    for (int node_id = 0; node_id < n_nodes; ++node_id) {
      auto const node = nodes[tree_id][node_id];
      std::uint32_t const* left_cat_bitmap = raw_left_cat_bitsets[tree_id];
      int const left_child_id = static_cast<int>(node.left);
      int const right_child_id = static_cast<int>(node.right);
      builder->StartNode(node_id);
      if (left_child_id <= 0) {  // leaf node
        builder->LeafScalar(node.value);
      } else {
        TREELITE_CHECK_LE(node.feature_idx, std::numeric_limits<std::int32_t>::max())
            << "split_index too large";
        auto const split_index = static_cast<std::int32_t>(node.feature_idx);
        bool const default_left = node.missing_go_to_left == 1;

        if (node.is_categorical == 1) {
          std::vector<std::uint32_t> left_categories;
          for (std::uint32_t i = 0; i < 256; ++i) {
            if (check(left_cat_bitmap, i, node.bitset_idx)) {
              left_categories.push_back(i);
            }
          }
          builder->CategoricalTest(
              split_index, default_left, left_categories, false, left_child_id, right_child_id);
        } else {
          double const split_cond = node.num_threshold;
          builder->NumericalTest(split_index, split_cond, default_left, treelite::Operator::kLE,
              left_child_id, right_child_id);
        }
        builder->Gain(node.gain);
      }
      builder->DataCount(node.count);
      builder->EndNode();
    }
    builder->EndTree();
  }
  return builder->CommitModel();
}

template <typename MixIn>
std::unique_ptr<treelite::Model> LoadHistGradientBoosting(MixIn& mixin, int n_trees, int n_features,
    std::int32_t n_classes, std::int64_t const* node_count, void const** nodes,
    int expected_sizeof_node_struct, std::uint32_t const** raw_left_cat_bitsets) {
  if (expected_sizeof_node_struct == sizeof(HistGradientBoostingNode<std::int32_t>)) {
    return LoadHistGradientBoostingImpl(mixin, n_trees, n_features, n_classes, node_count,
        reinterpret_cast<HistGradientBoostingNode<std::int32_t> const**>(nodes),
        raw_left_cat_bitsets);
  } else if (expected_sizeof_node_struct == sizeof(HistGradientBoostingNode<std::int64_t>)) {
    return LoadHistGradientBoostingImpl(mixin, n_trees, n_features, n_classes, node_count,
        reinterpret_cast<HistGradientBoostingNode<std::int64_t> const**>(nodes),
        raw_left_cat_bitsets);
  } else {
    TREELITE_LOG(FATAL) << "Unexpected size for Node struct: " << expected_sizeof_node_struct;
    return {};
  }
}

}  // namespace detail

std::unique_ptr<treelite::Model> LoadRandomForestRegressor(int n_estimators, int n_features,
    int n_targets, std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity) {
  detail::RandomForestRegressorMixIn mixin{};
  return detail::LoadSKLearnModel(mixin, n_estimators, n_features, n_targets, nullptr, node_count,
      children_left, children_right, feature, threshold, value, n_node_samples,
      weighted_n_node_samples, impurity);
}

std::unique_ptr<treelite::Model> LoadIsolationForest(int n_estimators, int n_features,
    std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity, double ratio_c) {
  detail::IsolationForestMixIn mixin{ratio_c};
  std::vector<std::int32_t> n_classes{1};
  return detail::LoadSKLearnModel(mixin, n_estimators, n_features, 1, n_classes.data(), node_count,
      children_left, children_right, feature, threshold, value, n_node_samples,
      weighted_n_node_samples, impurity);
}

std::unique_ptr<treelite::Model> LoadRandomForestClassifier(int n_estimators, int n_features,
    int n_targets, int32_t const* n_classes, std::int64_t const* node_count,
    std::int64_t const** children_left, std::int64_t const** children_right,
    std::int64_t const** feature, double const** threshold, double const** value,
    std::int64_t const** n_node_samples, double const** weighted_n_node_samples,
    double const** impurity) {
  detail::RandomForestClassifierMixIn mixin{};
  return detail::LoadSKLearnModel(mixin, n_estimators, n_features, n_targets, n_classes, node_count,
      children_left, children_right, feature, threshold, value, n_node_samples,
      weighted_n_node_samples, impurity);
}

std::unique_ptr<treelite::Model> LoadGradientBoostingRegressor(int n_iter, int n_features,
    std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity, double const* base_scores) {
  detail::GradientBoostingRegressorMixIn mixin{base_scores[0]};
  return detail::LoadSKLearnModel(mixin, n_iter, n_features, 1, nullptr, node_count, children_left,
      children_right, feature, threshold, value, n_node_samples, weighted_n_node_samples, impurity);
}

std::unique_ptr<treelite::Model> LoadGradientBoostingClassifier(int n_iter, int n_features,
    int n_classes, std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity, double const* base_scores) {
  TREELITE_CHECK_GE(n_classes, 2) << "Number of classes must be at least 2";
  if (n_classes > 2) {
    std::vector<double> base_scores_(base_scores, base_scores + n_classes);
    detail::GradientBoostingMulticlassClassifierMixIn mixin{base_scores_};
    std::vector<std::int32_t> n_classes_{static_cast<std::int32_t>(n_classes)};
    return detail::LoadSKLearnModel(mixin, n_iter * n_classes, n_features, 1, n_classes_.data(),
        node_count, children_left, children_right, feature, threshold, value, n_node_samples,
        weighted_n_node_samples, impurity);
  } else {
    detail::GradientBoostingBinaryClassifierMixIn mixin{base_scores[0]};
    std::vector<std::int32_t> n_classes_{static_cast<std::int32_t>(n_classes)};
    return detail::LoadSKLearnModel(mixin, n_iter, n_features, 1, n_classes_.data(), node_count,
        children_left, children_right, feature, threshold, value, n_node_samples,
        weighted_n_node_samples, impurity);
  }
}

std::unique_ptr<treelite::Model> LoadHistGradientBoostingRegressor(int n_iter, int n_features,
    std::int64_t const* node_count, void const** nodes, int expected_sizeof_node_struct,
    std::uint32_t n_categorical_splits, std::uint32_t const** raw_left_cat_bitsets,
    std::uint32_t const* known_cat_bitsets, std::uint32_t const* known_cat_bitsets_offset_map,
    double const* base_scores) {
  detail::HistGradientBoostingRegressorMixIn mixin{base_scores[0]};
  return detail::LoadHistGradientBoosting(mixin, n_iter, n_features, 1, node_count, nodes,
      expected_sizeof_node_struct, raw_left_cat_bitsets);
}

std::unique_ptr<treelite::Model> LoadHistGradientBoostingClassifier(int n_iter, int n_features,
    int n_classes, int64_t const* node_count, void const** nodes, int expected_sizeof_node_struct,
    uint32_t n_categorical_splits, uint32_t const** raw_left_cat_bitsets,
    uint32_t const* known_cat_bitsets, uint32_t const* known_cat_bitsets_offset_map,
    double const* base_scores) {
  TREELITE_CHECK_GE(n_classes, 2) << "Number of classes must be at least 2";
  if (n_classes > 2) {
    std::vector<double> base_scores_(base_scores, base_scores + n_classes);
    detail::HistGradientBoostingMulticlassClassifierMixIn mixin{base_scores_};
    return detail::LoadHistGradientBoosting(mixin, n_iter * n_classes, n_features, n_classes,
        node_count, nodes, expected_sizeof_node_struct, raw_left_cat_bitsets);
  } else {
    detail::HistGradientBoostingBinaryClassifierMixIn mixin{base_scores[0]};
    return detail::LoadHistGradientBoosting(mixin, n_iter, n_features, n_classes, node_count, nodes,
        expected_sizeof_node_struct, raw_left_cat_bitsets);
  }
}

}  // namespace treelite::model_loader::sklearn
