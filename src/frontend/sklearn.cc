/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file sklearn.cc
 * \brief Frontend for scikit-learn models
 * \author Hyunsu Cho
 */
#include <treelite/frontend.h>
#include <treelite/logging.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <queue>
#include <tuple>

namespace treelite::frontend::sklearn {

namespace detail {

class MixIn {
 public:
  void HandleMetadata(
      treelite::Model* model, int n_features, [[maybe_unused]] int n_classes) const {}

  void HandleLeafNode(int tree_id, std::int64_t node_id, int new_node_id, double const** value,
      [[maybe_unused]] int n_classes, treelite::Tree<double, double>& dest_tree) const {}
};

class RandomForestRegressorMixIn {
 public:
  void HandleMetadata(
      treelite::Model* model, int n_features, [[maybe_unused]] int n_classes) const {
    model->num_feature = n_features;
    model->average_tree_output = true;
    model->task_type = treelite::TaskType::kBinaryClfRegr;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = 1;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(model->param.pred_transform, "identity", sizeof(model->param.pred_transform));
    model->param.global_bias = 0.0f;
  }

  void HandleLeafNode(int tree_id, std::int64_t node_id, int new_node_id, double const** value,
      [[maybe_unused]] int n_classes, treelite::Tree<double, double>& dest_tree) const {
    double const leaf_value = value[tree_id][node_id];
    dest_tree.SetLeaf(new_node_id, leaf_value);
  }
};

class IsolationForestMixIn {
 public:
  explicit IsolationForestMixIn(double ratio_c) : ratio_c_{ratio_c} {}

  void HandleMetadata(
      treelite::Model* model, int n_features, [[maybe_unused]] int n_classes) const {
    model->num_feature = n_features;
    model->average_tree_output = true;
    model->task_type = treelite::TaskType::kBinaryClfRegr;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = 1;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(model->param.pred_transform, "exponential_standard_ratio",
        sizeof(model->param.pred_transform));
    model->param.ratio_c = static_cast<float>(ratio_c_);
  }

  void HandleLeafNode(int tree_id, std::int64_t node_id, int new_node_id, double const** value,
      [[maybe_unused]] int n_classes, treelite::Tree<double, double>& dest_tree) const {
    double const leaf_value = value[tree_id][node_id];
    dest_tree.SetLeaf(new_node_id, leaf_value);
  }

 private:
  double ratio_c_;
};

class RandomForestBinaryClassifierMixIn {
 public:
  void HandleMetadata(
      treelite::Model* model, int n_features, [[maybe_unused]] int n_classes) const {
    model->num_feature = n_features;
    model->average_tree_output = true;
    model->task_type = treelite::TaskType::kBinaryClfRegr;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = 1;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(model->param.pred_transform, "identity", sizeof(model->param.pred_transform));
    model->param.global_bias = 0.0f;
  }

  void HandleLeafNode(int tree_id, std::int64_t node_id, int new_node_id, double const** value,
      [[maybe_unused]] int n_classes, treelite::Tree<double, double>& dest_tree) const {
    // Get counts for each label (+/-) at this leaf node
    double const* leaf_count = &value[tree_id][node_id * 2];
    // Compute the fraction of positive data points at this leaf node
    double const fraction_positive = leaf_count[1] / (leaf_count[0] + leaf_count[1]);
    dest_tree.SetLeaf(new_node_id, fraction_positive);
  }
};

class RandomForestMulticlassClassifierMixIn {
 public:
  void HandleMetadata(
      treelite::Model* model, int n_features, [[maybe_unused]] int n_classes) const {
    model->num_feature = n_features;
    model->average_tree_output = true;
    model->task_type = treelite::TaskType::kMultiClfProbDistLeaf;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = n_classes;
    model->task_param.leaf_vector_size = n_classes;
    std::strncpy(
        model->param.pred_transform, "identity_multiclass", sizeof(model->param.pred_transform));
    model->param.global_bias = 0.0f;
  }

  void HandleLeafNode(int tree_id, std::int64_t node_id, int new_node_id, double const** value,
      [[maybe_unused]] int n_classes, treelite::Tree<double, double>& dest_tree) const {
    // Get counts for each label class at this leaf node
    std::vector<double> prob_distribution(
        &value[tree_id][node_id * n_classes], &value[tree_id][(node_id + 1) * n_classes]);
    // Compute the probability distribution over label classes
    double const norm_factor
        = std::accumulate(prob_distribution.begin(), prob_distribution.end(), 0.0);
    std::for_each(prob_distribution.begin(), prob_distribution.end(),
        [norm_factor](double& e) { e /= norm_factor; });
    dest_tree.SetLeafVector(new_node_id, prob_distribution);
  }
};

class GradientBoostingRegressorMixIn {
 public:
  void HandleMetadata(
      treelite::Model* model, int n_features, [[maybe_unused]] int n_classes) const {
    model->num_feature = n_features;
    model->average_tree_output = false;
    model->task_type = treelite::TaskType::kBinaryClfRegr;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = 1;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(model->param.pred_transform, "identity", sizeof(model->param.pred_transform));
    model->param.global_bias = 0.0f;
  }

  void HandleLeafNode(int tree_id, std::int64_t node_id, int new_node_id, double const** value,
      [[maybe_unused]] int n_classes, treelite::Tree<double, double>& dest_tree) const {
    double const leaf_value = value[tree_id][node_id];
    dest_tree.SetLeaf(new_node_id, leaf_value);
  }
};

class GradientBoostingBinaryClassifierMixIn {
 public:
  void HandleMetadata(
      treelite::Model* model, int n_features, [[maybe_unused]] int n_classes) const {
    model->num_feature = n_features;
    model->average_tree_output = false;
    model->task_type = treelite::TaskType::kBinaryClfRegr;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = 1;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(model->param.pred_transform, "sigmoid", sizeof(model->param.pred_transform));
    model->param.global_bias = 0.0f;
  }

  void HandleLeafNode(int tree_id, std::int64_t node_id, int new_node_id, double const** value,
      [[maybe_unused]] int n_classes, treelite::Tree<double, double>& dest_tree) const {
    double const leaf_value = value[tree_id][node_id];
    dest_tree.SetLeaf(new_node_id, leaf_value);
  }
};

class GradientBoostingMulticlassClassifierMixIn {
 public:
  void HandleMetadata(
      treelite::Model* model, int n_features, [[maybe_unused]] int n_classes) const {
    model->num_feature = n_features;
    model->average_tree_output = false;
    model->task_type = treelite::TaskType::kMultiClfGrovePerClass;
    model->task_param.grove_per_class = true;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = n_classes;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(model->param.pred_transform, "softmax", sizeof(model->param.pred_transform));
    model->param.global_bias = 0.0f;
  }

  void HandleLeafNode(int tree_id, std::int64_t node_id, int new_node_id, double const** value,
      [[maybe_unused]] int n_classes, treelite::Tree<double, double>& dest_tree) const {
    double const leaf_value = value[tree_id][node_id];
    dest_tree.SetLeaf(new_node_id, leaf_value);
  }
};

class HistGradientBoostingRegressorMixIn {
 public:
  explicit HistGradientBoostingRegressorMixIn(double global_bias) : global_bias_{global_bias} {}

  void HandleMetadata(
      treelite::Model* model, int n_features, [[maybe_unused]] int n_classes) const {
    model->num_feature = n_features;
    model->average_tree_output = false;
    model->task_type = treelite::TaskType::kBinaryClfRegr;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = 1;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(model->param.pred_transform, "identity", sizeof(model->param.pred_transform));
    model->param.global_bias = static_cast<float>(global_bias_);
  }

  void HandleLeafNode(int tree_id, std::int64_t node_id, int new_node_id, double const** value,
      [[maybe_unused]] int n_classes, treelite::Tree<double, double>& dest_tree) const {
    double const leaf_value = value[tree_id][node_id];
    dest_tree.SetLeaf(new_node_id, leaf_value);
  }

 private:
  double global_bias_;
};

class HistGradientBoostingBinaryClassifierMixIn {
 public:
  explicit HistGradientBoostingBinaryClassifierMixIn(double global_bias)
      : global_bias_{global_bias} {}

  void HandleMetadata(
      treelite::Model* model, int n_features, [[maybe_unused]] int n_classes) const {
    model->num_feature = n_features;
    model->average_tree_output = false;
    model->task_type = treelite::TaskType::kBinaryClfRegr;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = 1;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(model->param.pred_transform, "sigmoid", sizeof(model->param.pred_transform));
    model->param.global_bias = static_cast<float>(global_bias_);
  }

  void HandleLeafNode(int tree_id, std::int64_t node_id, int new_node_id, double const** value,
      [[maybe_unused]] int n_classes, treelite::Tree<double, double>& dest_tree) const {
    double const leaf_value = value[tree_id][node_id];
    dest_tree.SetLeaf(new_node_id, leaf_value);
  }

 private:
  double global_bias_;
};

template <typename MixIn>
std::unique_ptr<treelite::Model> LoadSKLearnModel(MixIn const& mixin, int n_trees, int n_features,
    int n_classes, [[maybe_unused]] std::int64_t const* node_count,
    std::int64_t const** children_left, std::int64_t const** children_right,
    std::int64_t const** feature, double const** threshold, double const** value,
    std::int64_t const** n_node_samples, double const** weighted_n_node_samples,
    double const** impurity) {
  TREELITE_CHECK_GT(n_trees, 0);
  TREELITE_CHECK_GT(n_features, 0);

  std::unique_ptr<treelite::Model> model = treelite::Model::Create<double, double>();
  mixin.HandleMetadata(model.get(), n_features, n_classes);

  auto& trees = std::get<ModelPreset<double, double>>(model->variant_).trees;
  for (int tree_id = 0; tree_id < n_trees; ++tree_id) {
    trees.emplace_back();
    treelite::Tree<double, double>& tree = trees.back();
    tree.Init();

    // Assign node ID's so that a breadth-wise traversal would yield
    // the monotonic sequence 0, 1, 2, ...
    std::queue<std::pair<std::int64_t, int>> Q;  // (old ID, new ID) pair
    Q.emplace(0, 0);
    const std::int64_t total_sample_cnt = n_node_samples[tree_id][0];
    while (!Q.empty()) {
      std::int64_t node_id;
      int new_node_id;
      std::tie(node_id, new_node_id) = Q.front();
      Q.pop();
      const std::int64_t left_child_id = children_left[tree_id][node_id];
      const std::int64_t right_child_id = children_right[tree_id][node_id];
      const std::int64_t sample_cnt = n_node_samples[tree_id][node_id];
      double const weighted_sample_cnt = weighted_n_node_samples[tree_id][node_id];
      if (left_child_id == -1) {  // leaf node
        mixin.HandleLeafNode(tree_id, node_id, new_node_id, value, n_classes, tree);
      } else {
        const std::int64_t split_index = feature[tree_id][node_id];
        double const split_cond = threshold[tree_id][node_id];
        const std::int64_t left_child_sample_cnt = n_node_samples[tree_id][left_child_id];
        const std::int64_t right_child_sample_cnt = n_node_samples[tree_id][right_child_id];
        double const gain
            = static_cast<double>(sample_cnt)
              * (impurity[tree_id][node_id]
                  - static_cast<double>(left_child_sample_cnt) * impurity[tree_id][left_child_id]
                        / static_cast<double>(sample_cnt)
                  - static_cast<double>(right_child_sample_cnt) * impurity[tree_id][right_child_id]
                        / static_cast<double>(sample_cnt))
              / static_cast<double>(total_sample_cnt);

        tree.AddChilds(new_node_id);
        tree.SetNumericalSplit(new_node_id, split_index, split_cond, true, treelite::Operator::kLE);
        tree.SetGain(new_node_id, gain);
        Q.emplace(left_child_id, tree.LeftChild(new_node_id));
        Q.emplace(right_child_id, tree.RightChild(new_node_id));
      }
      tree.SetDataCount(new_node_id, sample_cnt);
      tree.SetSumHess(new_node_id, weighted_sample_cnt);
    }
  }
  return model;
}

template <typename MixIn>
std::unique_ptr<treelite::Model> LoadHistGradientBoosting(MixIn const& mixin, int n_trees,
    int n_features, int n_classes, [[maybe_unused]] std::int64_t const* node_count,
    std::int64_t const** children_left, std::int64_t const** children_right,
    std::int64_t const** feature, double const** threshold, std::int8_t const** default_left,
    double const** value, std::int64_t const** n_node_samples, double const** gain) {
  TREELITE_CHECK_GT(n_trees, 0);
  TREELITE_CHECK_GT(n_features, 0);

  std::unique_ptr<treelite::Model> model = treelite::Model::Create<double, double>();
  mixin.HandleMetadata(model.get(), n_features, n_classes);

  auto& trees = std::get<ModelPreset<double, double>>(model->variant_).trees;
  for (int tree_id = 0; tree_id < n_trees; ++tree_id) {
    trees.emplace_back();
    treelite::Tree<double, double>& tree = trees.back();
    tree.Init();

    // Assign node ID's so that a breadth-wise traversal would yield
    // the monotonic sequence 0, 1, 2, ...
    std::queue<std::pair<std::int64_t, int>> Q;  // (old ID, new ID) pair
    Q.emplace(0, 0);
    while (!Q.empty()) {
      std::int64_t node_id;
      int new_node_id;
      std::tie(node_id, new_node_id) = Q.front();
      Q.pop();
      const std::int64_t left_child_id = children_left[tree_id][node_id];
      const std::int64_t right_child_id = children_right[tree_id][node_id];
      const std::int64_t sample_cnt = n_node_samples[tree_id][node_id];
      if (left_child_id == -1) {  // leaf node
        mixin.HandleLeafNode(tree_id, node_id, new_node_id, value, n_classes, tree);
      } else {
        const std::int64_t split_index = feature[tree_id][node_id];
        double const split_cond = threshold[tree_id][node_id];
        tree.AddChilds(new_node_id);
        tree.SetNumericalSplit(new_node_id, split_index, split_cond,
            (default_left[tree_id][node_id] == 1), treelite::Operator::kLE);
        tree.SetGain(new_node_id, gain[tree_id][node_id]);
        Q.emplace(left_child_id, tree.LeftChild(new_node_id));
        Q.emplace(right_child_id, tree.RightChild(new_node_id));
      }
      tree.SetDataCount(new_node_id, sample_cnt);
    }
  }
  return model;
}

}  // namespace detail

std::unique_ptr<treelite::Model> LoadRandomForestRegressor(int n_estimators, int n_features,
    std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity) {
  detail::RandomForestRegressorMixIn mixin{};
  return detail::LoadSKLearnModel(mixin, n_estimators, n_features, 1, node_count, children_left,
      children_right, feature, threshold, value, n_node_samples, weighted_n_node_samples, impurity);
}

std::unique_ptr<treelite::Model> LoadIsolationForest(int n_estimators, int n_features,
    std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity, double ratio_c) {
  detail::IsolationForestMixIn mixin{ratio_c};
  return LoadSKLearnModel(mixin, n_estimators, n_features, 1, node_count, children_left,
      children_right, feature, threshold, value, n_node_samples, weighted_n_node_samples, impurity);
}

std::unique_ptr<treelite::Model> LoadRandomForestClassifier(int n_estimators, int n_features,
    int n_classes, std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity) {
  TREELITE_CHECK_GE(n_classes, 2) << "Number of classes must be at least 2";
  if (n_classes == 2) {
    detail::RandomForestBinaryClassifierMixIn mixin;
    return LoadSKLearnModel(mixin, n_estimators, n_features, n_classes, node_count, children_left,
        children_right, feature, threshold, value, n_node_samples, weighted_n_node_samples,
        impurity);
  } else {
    detail::RandomForestMulticlassClassifierMixIn mixin;
    return LoadSKLearnModel(mixin, n_estimators, n_features, n_classes, node_count, children_left,
        children_right, feature, threshold, value, n_node_samples, weighted_n_node_samples,
        impurity);
  }
}

std::unique_ptr<treelite::Model> LoadGradientBoostingRegressor(int n_iter, int n_features,
    std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity) {
  detail::GradientBoostingRegressorMixIn mixin;
  return LoadSKLearnModel(mixin, n_iter, n_features, 1, node_count, children_left, children_right,
      feature, threshold, value, n_node_samples, weighted_n_node_samples, impurity);
}

std::unique_ptr<treelite::Model> LoadGradientBoostingClassifier(int n_iter, int n_features,
    int n_classes, std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity) {
  TREELITE_CHECK_GE(n_classes, 2) << "Number of classes must be at least 2";
  if (n_classes == 2) {
    detail::GradientBoostingBinaryClassifierMixIn mixin;
    return LoadSKLearnModel(mixin, n_iter, n_features, n_classes, node_count, children_left,
        children_right, feature, threshold, value, n_node_samples, weighted_n_node_samples,
        impurity);
  } else {
    detail::GradientBoostingMulticlassClassifierMixIn mixin;
    return LoadSKLearnModel(mixin, n_iter * n_classes, n_features, n_classes, node_count,
        children_left, children_right, feature, threshold, value, n_node_samples,
        weighted_n_node_samples, impurity);
  }
}

std::unique_ptr<treelite::Model> LoadHistGradientBoostingRegressor(int n_iter, int n_features,
    std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    std::int8_t const** default_left, double const** value, std::int64_t const** n_node_samples,
    double const** gain, double const* baseline_prediction) {
  detail::HistGradientBoostingRegressorMixIn mixin{baseline_prediction[0]};
  return detail::LoadHistGradientBoosting(mixin, n_iter, n_features, 1, node_count, children_left,
      children_right, feature, threshold, default_left, value, n_node_samples, gain);
}

std::unique_ptr<treelite::Model> LoadHistGradientBoostingClassifier(int n_iter, int n_features,
    int n_classes, std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    std::int8_t const** default_left, double const** value, std::int64_t const** n_node_samples,
    double const** gain, double const* baseline_prediction) {
  TREELITE_CHECK_GE(n_classes, 2) << "Number of classes must be at least 2";
  if (n_classes == 2) {
    detail::HistGradientBoostingBinaryClassifierMixIn mixin{baseline_prediction[0]};
    return LoadHistGradientBoosting(mixin, n_iter, n_features, n_classes, node_count, children_left,
        children_right, feature, threshold, default_left, value, n_node_samples, gain);
  } else {
    // TODO(hcho3): Add support for multi-class HistGradientBoostingClassifier
    TREELITE_LOG(FATAL) << "HistGradientBoostingClassifier with n_classes > 2 is not supported yet";
    return {};
  }
}

}  // namespace treelite::frontend::sklearn
