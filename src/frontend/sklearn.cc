/*!
 * Copyright (c) 2021 by Contributors
 * \file sklearn.cc
 * \brief Frontend for scikit-learn models
 * \author Hyunsu Cho
 */
#include <treelite/frontend.h>
#include <treelite/tree.h>
#include <memory>
#include <queue>
#include <tuple>

namespace treelite {
namespace frontend {

std::unique_ptr<treelite::Model> LoadSKLearnRandomForestRegressor(
    int n_estimators, int n_features, const int64_t* node_count, const int64_t** children_left,
    const int64_t** children_right, const int64_t** feature, const double** threshold,
    const double** value, const int64_t** n_node_samples, const double** impurity) {
  CHECK_GT(n_estimators, 0);
  CHECK_GT(n_features, 0);

  std::unique_ptr<treelite::Model> model_ptr = treelite::Model::Create<double, double>();
  auto* model = dynamic_cast<treelite::ModelImpl<double, double>*>(model_ptr.get());
  model->num_feature = n_features;
  model->average_tree_output = true;
  model->task_type = treelite::TaskType::kBinaryClfRegr;
  model->task_param.grove_per_class = false;
  model->task_param.output_type = treelite::TaskParameter::OutputType::kFloat;
  model->task_param.num_class = 1;
  model->task_param.leaf_vector_size = 1;
  std::strncpy(model->param.pred_transform, "identity", sizeof(model->param.pred_transform));
  model->param.global_bias = 0.0f;

  for (int tree_id = 0; tree_id < n_estimators; ++tree_id) {
    model->trees.emplace_back();
    treelite::Tree<double, double>& tree = model->trees.back();
    tree.Init();

    // assign node ID's so that a breadth-wise traversal would yield
    // the monotonic sequence 0, 1, 2, ...
    std::queue<std::pair<int64_t, int>> Q;  // (old ID, new ID) pair
    Q.push({0, 0});
    const int64_t total_sample_cnt = n_node_samples[tree_id][0];
    while (!Q.empty()) {
      int64_t node_id;
      int new_node_id;
      std::tie(node_id, new_node_id) = Q.front(); Q.pop();
      const int64_t left_child_id = children_left[tree_id][node_id];
      const int64_t right_child_id = children_right[tree_id][node_id];
      const int64_t sample_cnt = n_node_samples[tree_id][node_id];
      if (left_child_id == -1) {  // leaf node
        const double leaf_value = value[tree_id][node_id];
        tree.SetLeaf(new_node_id, leaf_value);
      } else {
        const int64_t split_index = feature[tree_id][node_id];
        const double split_cond = threshold[tree_id][node_id];
        const int64_t left_child_sample_cnt = n_node_samples[tree_id][left_child_id];
        const int64_t right_child_sample_cnt = n_node_samples[tree_id][right_child_id];
        const double gain = sample_cnt * (
            impurity[tree_id][node_id]
            - left_child_sample_cnt  * impurity[tree_id][left_child_id]  / sample_cnt
            - right_child_sample_cnt * impurity[tree_id][right_child_id] / sample_cnt)
          / total_sample_cnt;

        tree.AddChilds(new_node_id);
        tree.SetNumericalSplit(new_node_id, split_index, split_cond, true, treelite::Operator::kLE);
        tree.SetGain(new_node_id, gain);
        Q.push({left_child_id, tree.LeftChild(new_node_id)});
        Q.push({right_child_id, tree.RightChild(new_node_id)});
      }
      tree.SetDataCount(new_node_id, sample_cnt);
    }
  }
  return model_ptr;
}

std::unique_ptr<treelite::Model> LoadSKLearnRandomForestClassifierBinary(
    int n_estimators, int n_features, int n_classes, const int64_t* node_count,
    const int64_t** children_left, const int64_t** children_right, const int64_t** feature,
    const double** threshold, const double** value, const int64_t** n_node_samples,
    const double** impurity) {
  CHECK_GT(n_estimators, 0);
  CHECK_GT(n_features, 0);

  std::unique_ptr<treelite::Model> model_ptr = treelite::Model::Create<double, double>();
  auto* model = dynamic_cast<treelite::ModelImpl<double, double>*>(model_ptr.get());
  model->num_feature = n_features;
  model->average_tree_output = true;
  model->task_type = treelite::TaskType::kBinaryClfRegr;
  model->task_param.grove_per_class = false;
  model->task_param.output_type = treelite::TaskParameter::OutputType::kFloat;
  model->task_param.num_class = 1;
  model->task_param.leaf_vector_size = 1;
  std::strncpy(model->param.pred_transform, "identity", sizeof(model->param.pred_transform));
  model->param.global_bias = 0.0f;

  for (int tree_id = 0; tree_id < n_estimators; ++tree_id) {
    model->trees.emplace_back();
    treelite::Tree<double, double>& tree = model->trees.back();
    tree.Init();

    // assign node ID's so that a breadth-wise traversal would yield
    // the monotonic sequence 0, 1, 2, ...
    std::queue<std::pair<int64_t, int>> Q;  // (old ID, new ID) pair
    Q.push({0, 0});
    const int64_t total_sample_cnt = n_node_samples[tree_id][0];
    while (!Q.empty()) {
      int64_t node_id;
      int new_node_id;
      std::tie(node_id, new_node_id) = Q.front(); Q.pop();
      const int64_t left_child_id = children_left[tree_id][node_id];
      const int64_t right_child_id = children_right[tree_id][node_id];
      const int64_t sample_cnt = n_node_samples[tree_id][node_id];
      if (left_child_id == -1) {  // leaf node
        // # Get counts for each label (+/-) at this leaf node
        const double* leaf_count = &value[tree_id][node_id * 2];
        // Compute the fraction of positive data points at this leaf node
        const double fraction_positive = leaf_count[1] / (leaf_count[0] + leaf_count[1]);
        tree.SetLeaf(new_node_id, fraction_positive);
      } else {
        const int64_t split_index = feature[tree_id][node_id];
        const double split_cond = threshold[tree_id][node_id];
        const int64_t left_child_sample_cnt = n_node_samples[tree_id][left_child_id];
        const int64_t right_child_sample_cnt = n_node_samples[tree_id][right_child_id];
        const double gain = sample_cnt * (
            impurity[tree_id][node_id]
            - left_child_sample_cnt  * impurity[tree_id][left_child_id]  / sample_cnt
            - right_child_sample_cnt * impurity[tree_id][right_child_id] / sample_cnt)
          / total_sample_cnt;

        tree.AddChilds(new_node_id);
        tree.SetNumericalSplit(new_node_id, split_index, split_cond, true, treelite::Operator::kLE);
        tree.SetGain(new_node_id, gain);
        Q.push({left_child_id, tree.LeftChild(new_node_id)});
        Q.push({right_child_id, tree.RightChild(new_node_id)});
      }
      tree.SetDataCount(new_node_id, sample_cnt);
    }
  }
  return model_ptr;
}

std::unique_ptr<treelite::Model> LoadSKLearnRandomForestClassifierMulticlass(
    int n_estimators, int n_features, int n_classes, const int64_t* node_count,
    const int64_t** children_left, const int64_t** children_right, const int64_t** feature,
    const double** threshold, const double** value, const int64_t** n_node_samples,
    const double** impurity) {
}

std::unique_ptr<treelite::Model> LoadSKLearnRandomForestClassifier(
    int n_estimators, int n_features, int n_classes, const int64_t* node_count,
    const int64_t** children_left, const int64_t** children_right, const int64_t** feature,
    const double** threshold, const double** value, const int64_t** n_node_samples,
    const double** impurity) {
  CHECK_GE(n_classes, 2);
  if (n_classes == 2) {
    return LoadSKLearnRandomForestClassifierBinary(n_estimators, n_features, n_classes, node_count,
        children_left, children_right, feature, threshold, value, n_node_samples, impurity);
  } else {
    return LoadSKLearnRandomForestClassifierMulticlass(n_estimators, n_features, n_classes,
        node_count, children_left, children_right, feature, threshold, value, n_node_samples,
        impurity);
  }
}

}  // namespace frontend
}  // namespace treelite
