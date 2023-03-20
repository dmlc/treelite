/*!
 * Copyright (c) 2021 by Contributors
 * \file sklearn.cc
 * \brief Frontend for scikit-learn models
 * \author Hyunsu Cho
 */
#include <treelite/logging.h>
#include <treelite/frontend.h>
#include <treelite/tree.h>
#include <memory>
#include <queue>
#include <algorithm>
#include <numeric>
#include <tuple>

namespace treelite {
namespace frontend {

template <typename MetaHandlerFunc, typename LeafHandlerFunc>
std::unique_ptr<treelite::Model> LoadSKLearnModel(
    int n_trees, int n_features, int n_classes, const int64_t* node_count,
    const int64_t** children_left, const int64_t** children_right, const int64_t** feature,
    const double** threshold, const double** value, const int64_t** n_node_samples,
    const double** weighted_n_node_samples, const double** impurity, MetaHandlerFunc meta_handler,
    LeafHandlerFunc leaf_handler) {
  TREELITE_CHECK_GT(n_trees, 0);
  TREELITE_CHECK_GT(n_features, 0);

  std::unique_ptr<treelite::Model> model_ptr = treelite::Model::Create<double, double>();
  meta_handler(model_ptr.get(), n_features, n_classes);
  auto* model = dynamic_cast<treelite::ModelImpl<double, double>*>(model_ptr.get());

  for (int tree_id = 0; tree_id < n_trees; ++tree_id) {
    model->trees.emplace_back();
    treelite::Tree<double, double>& tree = model->trees.back();
    tree.Init();

    // Assign node ID's so that a breadth-wise traversal would yield
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
      const double weighted_sample_cnt = weighted_n_node_samples[tree_id][node_id];
      if (left_child_id == -1) {  // leaf node
        leaf_handler(tree_id, node_id, new_node_id, value, n_classes, tree);
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
      tree.SetSumHess(new_node_id, weighted_sample_cnt);
    }
  }
  return model_ptr;
}

template <typename MetaHandlerFunc, typename LeafHandlerFunc>
std::unique_ptr<treelite::Model> LoadSKLearnHistGradientBoosting(
    int n_trees, int n_features, int n_classes, const std::int64_t* node_count,
    const std::int64_t** children_left, const std::int64_t** children_right,
    const std::int64_t** feature, const double** threshold, const std::int8_t** default_left,
    const double** value, const int64_t** n_node_samples, const double** gain,
    MetaHandlerFunc meta_handler, LeafHandlerFunc leaf_handler) {
  TREELITE_CHECK_GT(n_trees, 0);
  TREELITE_CHECK_GT(n_features, 0);

  std::unique_ptr<treelite::Model> model_ptr = treelite::Model::Create<double, double>();
  meta_handler(model_ptr.get(), n_features, n_classes);
  auto* model = dynamic_cast<treelite::ModelImpl<double, double>*>(model_ptr.get());

  for (int tree_id = 0; tree_id < n_trees; ++tree_id) {
    model->trees.emplace_back();
    treelite::Tree<double, double>& tree = model->trees.back();
    tree.Init();

    // Assign node ID's so that a breadth-wise traversal would yield
    // the monotonic sequence 0, 1, 2, ...
    std::queue<std::pair<int64_t, int>> Q;  // (old ID, new ID) pair
    Q.emplace(0, 0);
    while (!Q.empty()) {
      int64_t node_id;
      int new_node_id;
      std::tie(node_id, new_node_id) = Q.front(); Q.pop();
      const int64_t left_child_id = children_left[tree_id][node_id];
      const int64_t right_child_id = children_right[tree_id][node_id];
      const int64_t sample_cnt = n_node_samples[tree_id][node_id];
      if (left_child_id == -1) {  // leaf node
        leaf_handler(tree_id, node_id, new_node_id, value, n_classes, tree);
      } else {
        const int64_t split_index = feature[tree_id][node_id];
        const double split_cond = threshold[tree_id][node_id];
        tree.AddChilds(new_node_id);
        tree.SetNumericalSplit(
            new_node_id, split_index, split_cond, (default_left[tree_id][node_id] == 1),
            treelite::Operator::kLE);
        tree.SetGain(new_node_id, gain[tree_id][node_id]);
        Q.emplace(left_child_id, tree.LeftChild(new_node_id));
        Q.emplace(right_child_id, tree.RightChild(new_node_id));
      }
      tree.SetDataCount(new_node_id, sample_cnt);
    }
  }
  return model_ptr;
}

std::unique_ptr<treelite::Model> LoadSKLearnRandomForestRegressor(
    int n_estimators, int n_features, const int64_t* node_count, const int64_t** children_left,
    const int64_t** children_right, const int64_t** feature, const double** threshold,
    const double** value, const int64_t** n_node_samples, const double** weighted_n_node_samples,
    const double** impurity) {
  auto meta_handler = [](treelite::Model* model, int n_features, int n_classes) {
    model->num_feature = n_features;
    model->average_tree_output = true;
    model->task_type = treelite::TaskType::kBinaryClfRegr;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = 1;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(model->param.pred_transform, "identity", sizeof(model->param.pred_transform));
    model->param.global_bias = 0.0f;
  };
  auto leaf_handler = [](int tree_id, int64_t node_id, int new_node_id, const double** value,
      int n_classes, treelite::Tree<double, double>& dest_tree) {
    const double leaf_value = value[tree_id][node_id];
    dest_tree.SetLeaf(new_node_id, leaf_value);
  };
  return LoadSKLearnModel(n_estimators, n_features, 1, node_count, children_left, children_right,
      feature, threshold, value, n_node_samples, weighted_n_node_samples, impurity, meta_handler,
      leaf_handler);
}

std::unique_ptr<treelite::Model> LoadSKLearnIsolationForest(
    int n_estimators, int n_features, const int64_t* node_count, const int64_t** children_left,
    const int64_t** children_right, const int64_t** feature, const double** threshold,
    const double** value, const int64_t** n_node_samples, const double** weighted_n_node_samples,
    const double** impurity, const double ratio_c) {
  auto meta_handler = [ratio_c](treelite::Model* model, int n_features, int n_classes) {
    model->num_feature = n_features;
    model->average_tree_output = true;
    model->task_type = treelite::TaskType::kBinaryClfRegr;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = 1;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(
      model->param.pred_transform, "exponential_standard_ratio",
      sizeof(model->param.pred_transform));
    model->param.ratio_c = ratio_c;
  };
  auto leaf_handler = [](int tree_id, int64_t node_id, int new_node_id, const double** value,
      int n_classes, treelite::Tree<double, double>& dest_tree) {
    const double leaf_value = value[tree_id][node_id];
    dest_tree.SetLeaf(new_node_id, leaf_value);
  };
  return LoadSKLearnModel(n_estimators, n_features, 1, node_count, children_left, children_right,
      feature, threshold, value, n_node_samples, weighted_n_node_samples, impurity, meta_handler,
      leaf_handler);
}

std::unique_ptr<treelite::Model> LoadSKLearnRandomForestClassifierBinary(
    int n_estimators, int n_features, int n_classes, const int64_t* node_count,
    const int64_t** children_left, const int64_t** children_right, const int64_t** feature,
    const double** threshold, const double** value, const int64_t** n_node_samples,
    const double** weighted_n_node_samples, const double** impurity) {
  auto meta_handler = [](treelite::Model* model, int n_features, int n_classes) {
    model->num_feature = n_features;
    model->average_tree_output = true;
    model->task_type = treelite::TaskType::kBinaryClfRegr;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = 1;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(model->param.pred_transform, "identity", sizeof(model->param.pred_transform));
    model->param.global_bias = 0.0f;
  };
  auto leaf_handler = [](int tree_id, int64_t node_id, int new_node_id, const double** value,
      int n_classes, treelite::Tree<double, double>& dest_tree) {
    // Get counts for each label (+/-) at this leaf node
    const double* leaf_count = &value[tree_id][node_id * 2];
    // Compute the fraction of positive data points at this leaf node
    const double fraction_positive = leaf_count[1] / (leaf_count[0] + leaf_count[1]);
    dest_tree.SetLeaf(new_node_id, fraction_positive);
  };
  return LoadSKLearnModel(n_estimators, n_features, n_classes, node_count, children_left,
      children_right, feature, threshold, value, n_node_samples, weighted_n_node_samples, impurity,
      meta_handler, leaf_handler);
}

std::unique_ptr<treelite::Model> LoadSKLearnRandomForestClassifierMulticlass(
    int n_estimators, int n_features, int n_classes, const int64_t* node_count,
    const int64_t** children_left, const int64_t** children_right, const int64_t** feature,
    const double** threshold, const double** value, const int64_t** n_node_samples,
    const double** weighted_n_node_samples, const double** impurity) {
  auto meta_handler = [](treelite::Model* model, int n_features, int n_classes) {
    model->num_feature = n_features;
    model->average_tree_output = true;
    model->task_type = treelite::TaskType::kMultiClfProbDistLeaf;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = n_classes;
    model->task_param.leaf_vector_size = n_classes;
    std::strncpy(model->param.pred_transform, "identity_multiclass",
                 sizeof(model->param.pred_transform));
    model->param.global_bias = 0.0f;
  };
  auto leaf_handler = [](int tree_id, int64_t node_id, int new_node_id, const double** value,
      int n_classes, treelite::Tree<double, double>& dest_tree) {
    // Get counts for each label class at this leaf node
    std::vector<double> prob_distribution(&value[tree_id][node_id * n_classes],
                                          &value[tree_id][(node_id + 1) * n_classes]);
    // Compute the probability distribution over label classes
    const double norm_factor =
      std::accumulate(prob_distribution.begin(), prob_distribution.end(), 0.0);
    std::for_each(prob_distribution.begin(), prob_distribution.end(), [norm_factor](double& e) {
        e /= norm_factor;
      });
    dest_tree.SetLeafVector(new_node_id, prob_distribution);
  };
  return LoadSKLearnModel(n_estimators, n_features, n_classes, node_count, children_left,
      children_right, feature, threshold, value, n_node_samples, weighted_n_node_samples, impurity,
      meta_handler, leaf_handler);
}

std::unique_ptr<treelite::Model> LoadSKLearnRandomForestClassifier(
    int n_estimators, int n_features, int n_classes, const int64_t* node_count,
    const int64_t** children_left, const int64_t** children_right, const int64_t** feature,
    const double** threshold, const double** value, const int64_t** n_node_samples,
    const double** weighted_n_node_samples, const double** impurity) {
  TREELITE_CHECK_GE(n_classes, 2) << "Number of classes must be at least 2";
  if (n_classes == 2) {
    return LoadSKLearnRandomForestClassifierBinary(n_estimators, n_features, n_classes, node_count,
        children_left, children_right, feature, threshold, value, n_node_samples,
        weighted_n_node_samples, impurity);
  } else {
    return LoadSKLearnRandomForestClassifierMulticlass(n_estimators, n_features, n_classes,
        node_count, children_left, children_right, feature, threshold, value, n_node_samples,
        weighted_n_node_samples, impurity);
  }
}

std::unique_ptr<treelite::Model> LoadSKLearnGradientBoostingRegressor(
    int n_iter, int n_features, const int64_t* node_count, const int64_t** children_left,
    const int64_t** children_right, const int64_t** feature, const double** threshold,
    const double** value, const int64_t** n_node_samples, const double** weighted_n_node_samples,
    const double** impurity) {
  auto meta_handler = [](treelite::Model* model, int n_features, int n_classes) {
    model->num_feature = n_features;
    model->average_tree_output = false;
    model->task_type = treelite::TaskType::kBinaryClfRegr;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = 1;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(model->param.pred_transform, "identity", sizeof(model->param.pred_transform));
    model->param.global_bias = 0.0f;
  };
  auto leaf_handler = [](int tree_id, int64_t node_id, int new_node_id, const double** value,
      int n_classes, treelite::Tree<double, double>& dest_tree) {
    const double leaf_value = value[tree_id][node_id];
    dest_tree.SetLeaf(new_node_id, leaf_value);
  };
  return LoadSKLearnModel(n_iter, n_features, 1, node_count, children_left,
      children_right, feature, threshold, value, n_node_samples, weighted_n_node_samples, impurity,
      meta_handler, leaf_handler);
}

std::unique_ptr<treelite::Model> LoadSKLearnGradientBoostingClassifierBinary(
    int n_iter, int n_features, int n_classes, const int64_t* node_count,
    const int64_t** children_left, const int64_t** children_right, const int64_t** feature,
    const double** threshold, const double** value, const int64_t** n_node_samples,
    const double** weighted_n_node_samples, const double** impurity) {
  auto meta_handler = [](treelite::Model* model, int n_features, int n_classes) {
    model->num_feature = n_features;
    model->average_tree_output = false;
    model->task_type = treelite::TaskType::kBinaryClfRegr;
    model->task_param.grove_per_class = false;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = 1;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(model->param.pred_transform, "sigmoid", sizeof(model->param.pred_transform));
    model->param.global_bias = 0.0f;
  };
  auto leaf_handler = [](int tree_id, int64_t node_id, int new_node_id, const double** value,
      int n_classes, treelite::Tree<double, double>& dest_tree) {
    const double leaf_value = value[tree_id][node_id];
    dest_tree.SetLeaf(new_node_id, leaf_value);
  };
  return LoadSKLearnModel(n_iter, n_features, n_classes, node_count, children_left,
      children_right, feature, threshold, value, n_node_samples, weighted_n_node_samples, impurity,
      meta_handler, leaf_handler);
}

std::unique_ptr<treelite::Model> LoadSKLearnGradientBoostingClassifierMulticlass(
    int n_iter, int n_features, int n_classes, const int64_t* node_count,
    const int64_t** children_left, const int64_t** children_right, const int64_t** feature,
    const double** threshold, const double** value, const int64_t** n_node_samples,
    const double** weighted_n_node_samples, const double** impurity) {
  auto meta_handler = [](treelite::Model* model, int n_features, int n_classes) {
    model->num_feature = n_features;
    model->average_tree_output = false;
    model->task_type = treelite::TaskType::kMultiClfGrovePerClass;
    model->task_param.grove_per_class = true;
    model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
    model->task_param.num_class = n_classes;
    model->task_param.leaf_vector_size = 1;
    std::strncpy(model->param.pred_transform, "softmax", sizeof(model->param.pred_transform));
    model->param.global_bias = 0.0f;
  };
  auto leaf_handler = [](int tree_id, int64_t node_id, int new_node_id, const double** value,
      int n_classes, treelite::Tree<double, double>& dest_tree) {
    const double leaf_value = value[tree_id][node_id];
    dest_tree.SetLeaf(new_node_id, leaf_value);
  };
  return LoadSKLearnModel(n_iter * n_classes, n_features, n_classes, node_count,
      children_left, children_right, feature, threshold, value, n_node_samples,
      weighted_n_node_samples, impurity, meta_handler, leaf_handler);
}

std::unique_ptr<treelite::Model> LoadSKLearnGradientBoostingClassifier(
    int n_iter, int n_features, int n_classes, const int64_t* node_count,
    const int64_t** children_left, const int64_t** children_right, const int64_t** feature,
    const double** threshold, const double** value, const int64_t** n_node_samples,
    const double** weighted_n_node_samples, const double** impurity) {
  TREELITE_CHECK_GE(n_classes, 2) << "Number of classes must be at least 2";
  if (n_classes == 2) {
    return LoadSKLearnGradientBoostingClassifierBinary(n_iter, n_features, n_classes,
        node_count, children_left, children_right, feature, threshold, value, n_node_samples,
        weighted_n_node_samples, impurity);
  } else {
    return LoadSKLearnGradientBoostingClassifierMulticlass(n_iter, n_features, n_classes,
        node_count, children_left, children_right, feature, threshold, value, n_node_samples,
        weighted_n_node_samples, impurity);
  }
}

std::unique_ptr<treelite::Model> LoadSKLearnHistGradientBoostingRegressor(
    int n_iter, int n_features, const std::int64_t* node_count, const std::int64_t** children_left,
    const std::int64_t** children_right, const std::int64_t** feature, const double** threshold,
    const std::int8_t** default_left, const double** value, const std::int64_t** n_node_samples,
    const double** gain, const double* baseline_prediction) {
  const auto global_bias = static_cast<float>(baseline_prediction[0]);
  auto meta_handler = [global_bias](treelite::Model* model, int n_features, int n_classes) {
      model->num_feature = n_features;
      model->average_tree_output = false;
      model->task_type = treelite::TaskType::kBinaryClfRegr;
      model->task_param.grove_per_class = false;
      model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
      model->task_param.num_class = 1;
      model->task_param.leaf_vector_size = 1;
      std::strncpy(model->param.pred_transform, "identity", sizeof(model->param.pred_transform));
      model->param.global_bias = global_bias;
  };
  auto leaf_handler = [](int tree_id, int64_t node_id, int new_node_id, const double** value,
                         int n_classes, treelite::Tree<double, double>& dest_tree) {
      const double leaf_value = value[tree_id][node_id];
      dest_tree.SetLeaf(new_node_id, leaf_value);
  };
  return LoadSKLearnHistGradientBoosting(n_iter, n_features, 1, node_count,
      children_left, children_right, feature, threshold, default_left, value, n_node_samples, gain,
      meta_handler, leaf_handler);
}

std::unique_ptr<treelite::Model> LoadSKLearnHistGradientBoostingClassifierBinary(
    int n_iter, int n_features, int n_classes, const std::int64_t* node_count,
    const std::int64_t** children_left, const std::int64_t** children_right,
    const std::int64_t** feature, const double** threshold, const std::int8_t** default_left,
    const double** value, const std::int64_t** n_node_samples, const double** gain,
    const double* baseline_prediction) {
  const auto global_bias = static_cast<float>(baseline_prediction[0]);
  auto meta_handler = [global_bias](treelite::Model* model, int n_features, int n_classes) {
      model->num_feature = n_features;
      model->average_tree_output = false;
      model->task_type = treelite::TaskType::kBinaryClfRegr;
      model->task_param.grove_per_class = false;
      model->task_param.output_type = treelite::TaskParam::OutputType::kFloat;
      model->task_param.num_class = 1;
      model->task_param.leaf_vector_size = 1;
      std::strncpy(model->param.pred_transform, "sigmoid", sizeof(model->param.pred_transform));
      model->param.global_bias = global_bias;
  };
  auto leaf_handler = [](int tree_id, int64_t node_id, int new_node_id, const double** value,
                         int n_classes, treelite::Tree<double, double>& dest_tree) {
      const double leaf_value = value[tree_id][node_id];
      dest_tree.SetLeaf(new_node_id, leaf_value);
  };
  return LoadSKLearnHistGradientBoosting(n_iter, n_features, n_classes, node_count,
      children_left, children_right, feature, threshold, default_left, value, n_node_samples, gain,
      meta_handler, leaf_handler);
}

std::unique_ptr<treelite::Model> LoadSKLearnHistGradientBoostingClassifier(
    int n_iter, int n_features, int n_classes, const std::int64_t* node_count,
    const std::int64_t** children_left, const std::int64_t** children_right,
    const std::int64_t** feature, const double** threshold, const std::int8_t** default_left,
    const double** value, const std::int64_t** n_node_samples, const double** gain,
    const double* baseline_prediction) {
  TREELITE_CHECK_GE(n_classes, 2) << "Number of classes must be at least 2";
  if (n_classes == 2) {
    return LoadSKLearnHistGradientBoostingClassifierBinary(n_iter, n_features, n_classes,
        node_count, children_left, children_right, feature, threshold, default_left, value,
        n_node_samples, gain, baseline_prediction);
  } else {
    // TODO(hcho3): Add support for multi-class HistGradientBoostingClassifier
    TREELITE_LOG(FATAL) << "HistGradientBoostingClassifier with n_classes > 2 is not supported yet";
    return {};
  }
}

}  // namespace frontend
}  // namespace treelite
