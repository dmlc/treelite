/*!
 * Copyright (c) 2024 by Contributors
 * \file sklearn_bulk.cc
 * \brief Bulk model loader for scikit-learn models - optimized for large forests
 * \author Treelite Contributors
 *
 * This file provides an optimized implementation for loading sklearn RandomForest
 * models that bypasses the node-by-node ModelBuilder API and directly constructs
 * Tree objects with bulk array operations. This significantly reduces overhead
 * for large models with many trees and nodes.
 */
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/tree_node_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/logging.h>
#include <treelite/tree.h>

namespace treelite {

/*!
 * \brief Bulk construct a Tree from sklearn arrays - friend function with direct access
 *
 * This function directly populates the Tree's internal ContiguousArrays in bulk,
 * avoiding the overhead of per-node function calls through the public API.
 *
 * \tparam ThresholdType Type of thresholds (float or double)
 * \tparam LeafOutputType Type of leaf outputs (float or double)
 */
template <typename ThresholdType, typename LeafOutputType>
void BulkConstructTree(Tree<ThresholdType, LeafOutputType>& tree, int n_nodes,
    std::int64_t const* children_left, std::int64_t const* children_right,
    std::int64_t const* feature, double const* threshold, double const* value,
    std::int64_t const* n_node_samples, double const* weighted_n_node_samples,
    double const* impurity, std::int64_t total_sample_cnt, int n_targets, int max_num_class,
    bool is_classifier) {
  // Clear and pre-allocate all arrays at once - key optimization!
  tree.node_type_.Clear();
  tree.cleft_.Clear();
  tree.cright_.Clear();
  tree.split_index_.Clear();
  tree.default_left_.Clear();
  tree.leaf_value_.Clear();
  tree.threshold_.Clear();
  tree.cmp_.Clear();
  tree.category_list_right_child_.Clear();
  tree.leaf_vector_.Clear();
  tree.leaf_vector_begin_.Clear();
  tree.leaf_vector_end_.Clear();
  tree.category_list_.Clear();
  tree.category_list_begin_.Clear();
  tree.category_list_end_.Clear();
  tree.data_count_.Clear();
  tree.data_count_present_.Clear();
  tree.sum_hess_.Clear();
  tree.sum_hess_present_.Clear();
  tree.gain_.Clear();
  tree.gain_present_.Clear();

  // Pre-allocate with exact sizes - avoid dynamic resizing
  tree.node_type_.Reserve(n_nodes);
  tree.cleft_.Reserve(n_nodes);
  tree.cright_.Reserve(n_nodes);
  tree.split_index_.Reserve(n_nodes);
  tree.default_left_.Reserve(n_nodes);
  tree.leaf_value_.Reserve(n_nodes);
  tree.threshold_.Reserve(n_nodes);
  tree.cmp_.Reserve(n_nodes);
  tree.category_list_right_child_.Reserve(n_nodes);
  tree.leaf_vector_begin_.Reserve(n_nodes);
  tree.leaf_vector_end_.Reserve(n_nodes);
  tree.category_list_begin_.Reserve(n_nodes);
  tree.category_list_end_.Reserve(n_nodes);

  // Pre-allocate node statistics arrays
  tree.data_count_.Reserve(n_nodes);
  tree.data_count_present_.Reserve(n_nodes);
  tree.sum_hess_.Reserve(n_nodes);
  tree.sum_hess_present_.Reserve(n_nodes);
  tree.gain_.Reserve(n_nodes);
  tree.gain_present_.Reserve(n_nodes);

  // Calculate leaf vector size for classifiers
  int const leaf_vector_size = n_targets * max_num_class;
  bool const has_leaf_vector = leaf_vector_size > 1;

  // Pre-allocate leaf_vector storage - estimate total size
  if (has_leaf_vector) {
    // Count leaf nodes to estimate leaf_vector size
    int num_leaves = 0;
    for (int i = 0; i < n_nodes; ++i) {
      if (children_left[i] == -1) {
        ++num_leaves;
      }
    }
    tree.leaf_vector_.Reserve(num_leaves * leaf_vector_size);
  }

  // Now populate all arrays in a single pass
  tree.num_nodes = n_nodes;
  tree.has_categorical_split_ = false;

  for (int node_id = 0; node_id < n_nodes; ++node_id) {
    int const left_child = static_cast<int>(children_left[node_id]);
    int const right_child = static_cast<int>(children_right[node_id]);
    bool const is_leaf = (left_child == -1);

    // Set node type
    if (is_leaf) {
      tree.node_type_.PushBack(TreeNodeType::kLeafNode);
    } else {
      tree.node_type_.PushBack(TreeNodeType::kNumericalTestNode);
    }

    // Set children
    tree.cleft_.PushBack(left_child);
    tree.cright_.PushBack(right_child);

    // Set split info (for internal nodes) or defaults (for leaves)
    if (is_leaf) {
      tree.split_index_.PushBack(-1);
      tree.threshold_.PushBack(static_cast<ThresholdType>(0));
      tree.cmp_.PushBack(Operator::kNone);
    } else {
      tree.split_index_.PushBack(static_cast<std::int32_t>(feature[node_id]));
      tree.threshold_.PushBack(static_cast<ThresholdType>(threshold[node_id]));
      tree.cmp_.PushBack(Operator::kLE);
    }

    tree.default_left_.PushBack(true);
    tree.category_list_right_child_.PushBack(false);

    // Handle leaf values
    std::size_t const current_leaf_vec_size = tree.leaf_vector_.Size();
    if (is_leaf && has_leaf_vector) {
      // Leaf node with vector output
      double const* value_ptr = &value[node_id * leaf_vector_size];

      if (is_classifier) {
        // Normalize probabilities for each target
        for (int target_id = 0; target_id < n_targets; ++target_id) {
          double norm_factor = 0.0;
          for (int class_id = 0; class_id < max_num_class; ++class_id) {
            int idx = target_id * max_num_class + class_id;
            norm_factor += value_ptr[idx];
          }
          for (int class_id = 0; class_id < max_num_class; ++class_id) {
            int idx = target_id * max_num_class + class_id;
            LeafOutputType normalized
                = static_cast<LeafOutputType>(norm_factor > 0 ? value_ptr[idx] / norm_factor : 0.0);
            tree.leaf_vector_.PushBack(normalized);
          }
        }
      } else {
        // Just copy the values
        for (int i = 0; i < leaf_vector_size; ++i) {
          tree.leaf_vector_.PushBack(static_cast<LeafOutputType>(value_ptr[i]));
        }
      }
      tree.leaf_vector_begin_.PushBack(current_leaf_vec_size);
      tree.leaf_vector_end_.PushBack(tree.leaf_vector_.Size());
      tree.leaf_value_.PushBack(static_cast<LeafOutputType>(0));
    } else if (is_leaf) {
      // Leaf node with scalar output
      tree.leaf_value_.PushBack(static_cast<LeafOutputType>(value[node_id]));
      tree.leaf_vector_begin_.PushBack(current_leaf_vec_size);
      tree.leaf_vector_end_.PushBack(current_leaf_vec_size);
    } else {
      // Internal node - no leaf value
      tree.leaf_value_.PushBack(static_cast<LeafOutputType>(0));
      tree.leaf_vector_begin_.PushBack(current_leaf_vec_size);
      tree.leaf_vector_end_.PushBack(current_leaf_vec_size);
    }

    // Empty category list for all nodes (no categorical splits)
    std::size_t const current_cat_size = tree.category_list_.Size();
    tree.category_list_begin_.PushBack(current_cat_size);
    tree.category_list_end_.PushBack(current_cat_size);

    // Node statistics
    tree.data_count_.PushBack(static_cast<std::uint64_t>(n_node_samples[node_id]));
    tree.data_count_present_.PushBack(true);
    tree.sum_hess_.PushBack(weighted_n_node_samples[node_id]);
    tree.sum_hess_present_.PushBack(true);

    // Calculate and set gain for internal nodes
    if (!is_leaf) {
      std::int64_t const sample_cnt = n_node_samples[node_id];
      std::int64_t const left_sample_cnt = n_node_samples[left_child];
      std::int64_t const right_sample_cnt = n_node_samples[right_child];
      double const gain = static_cast<double>(sample_cnt)
                          * (impurity[node_id]
                              - static_cast<double>(left_sample_cnt) * impurity[left_child]
                                    / static_cast<double>(sample_cnt)
                              - static_cast<double>(right_sample_cnt) * impurity[right_child]
                                    / static_cast<double>(sample_cnt))
                          / static_cast<double>(total_sample_cnt);
      tree.gain_.PushBack(gain);
      tree.gain_present_.PushBack(true);
    } else {
      tree.gain_.PushBack(0.0);
      tree.gain_present_.PushBack(false);
    }
  }
}

// Explicit instantiation
template void BulkConstructTree<double, double>(Tree<double, double>&, int, std::int64_t const*,
    std::int64_t const*, std::int64_t const*, double const*, double const*, std::int64_t const*,
    double const*, double const*, std::int64_t, int, int, bool);

template void BulkConstructTree<float, float>(Tree<float, float>&, int, std::int64_t const*,
    std::int64_t const*, std::int64_t const*, double const*, double const*, std::int64_t const*,
    double const*, double const*, std::int64_t, int, int, bool);

}  // namespace treelite

namespace treelite::model_loader::sklearn {

/**
 * Load a RandomForestClassifier using bulk construction
 *
 * This is an optimized version that constructs trees in bulk rather than
 * going through the ModelBuilder node-by-node.
 */
std::unique_ptr<treelite::Model> LoadRandomForestClassifier(int n_estimators, int n_features,
    int n_targets, std::int32_t const* n_classes, std::int64_t const* node_count,
    std::int64_t const** children_left, std::int64_t const** children_right,
    std::int64_t const** feature, double const** threshold, double const** value,
    std::int64_t const** n_node_samples, double const** weighted_n_node_samples,
    double const** impurity) {
  TREELITE_CHECK_GT(n_estimators, 0) << "n_estimators must be at least 1";
  TREELITE_CHECK_GT(n_features, 0) << "n_features must be at least 1";

  // Create model with double precision
  auto model = Model::Create<double, double>();

  // Set up model metadata
  // Note: Here, we will treat binary classifiers as if they are multi-class classifiers with
  // n_classes=2.
  model->num_feature = n_features;
  model->task_type = TaskType::kMultiClf;
  model->average_tree_output = true;
  model->num_target = n_targets;

  // Set up n_classes array
  std::vector<std::int32_t> n_classes_vec(n_classes, n_classes + n_targets);
  model->num_class = n_classes_vec;

  // Find max number of classes
  std::int32_t max_num_class = *std::max_element(n_classes_vec.begin(), n_classes_vec.end());

  // Set leaf vector shape
  model->leaf_vector_shape = std::vector<std::int32_t>{n_targets, max_num_class};

  // Set up tree annotation arrays
  model->target_id = std::vector<std::int32_t>(n_estimators, -1);
  model->class_id = std::vector<std::int32_t>(n_estimators, -1);

  // Set postprocessor
  model->postprocessor = "identity_multiclass";

  // Set base scores
  model->base_scores = std::vector<double>(n_targets * max_num_class, 0.0);

  // Get the typed model preset
  auto& preset = std::get<ModelPreset<double, double>>(model->variant_);
  preset.trees.resize(n_estimators);

  // Construct each tree using bulk operations
  for (int tree_id = 0; tree_id < n_estimators; ++tree_id) {
    int const n_nodes = static_cast<int>(node_count[tree_id]);
    std::int64_t const total_sample_cnt = n_node_samples[tree_id][0];

    BulkConstructTree<double, double>(preset.trees[tree_id], n_nodes, children_left[tree_id],
        children_right[tree_id], feature[tree_id], threshold[tree_id], value[tree_id],
        n_node_samples[tree_id], weighted_n_node_samples[tree_id], impurity[tree_id],
        total_sample_cnt, n_targets, max_num_class,
        true);  // is_classifier
  }

  return model;
}

/**
 * Load a RandomForestRegressor using bulk construction
 *
 * This is an optimized version that constructs trees in bulk rather than
 * going through the ModelBuilder node-by-node.
 */
std::unique_ptr<treelite::Model> LoadRandomForestRegressor(int n_estimators, int n_features,
    int n_targets, std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity) {
  TREELITE_CHECK_GT(n_estimators, 0) << "n_estimators must be at least 1";
  TREELITE_CHECK_GT(n_features, 0) << "n_features must be at least 1";

  // Create model with double precision
  auto model = Model::Create<double, double>();

  // Set up model metadata
  model->num_feature = n_features;
  model->task_type = TaskType::kRegressor;
  model->average_tree_output = true;
  model->num_target = n_targets;

  // For regressors, num_class is always 1
  model->num_class = std::vector<std::int32_t>(n_targets, 1);

  // Set leaf vector shape
  model->leaf_vector_shape = std::vector<std::int32_t>{n_targets, 1};

  // Set up tree annotation arrays
  std::vector<std::int32_t> target_id(n_estimators, (n_targets > 1 ? -1 : 0));
  std::vector<std::int32_t> class_id(n_estimators, 0);
  model->target_id = target_id;
  model->class_id = class_id;

  // Set postprocessor
  model->postprocessor = "identity";

  // Set base scores
  model->base_scores = std::vector<double>(n_targets, 0.0);

  // Get the typed model preset
  auto& preset = std::get<ModelPreset<double, double>>(model->variant_);
  preset.trees.resize(n_estimators);

  // Construct each tree using bulk operations
  for (int tree_id = 0; tree_id < n_estimators; ++tree_id) {
    int const n_nodes = static_cast<int>(node_count[tree_id]);
    std::int64_t const total_sample_cnt = n_node_samples[tree_id][0];

    BulkConstructTree<double, double>(preset.trees[tree_id], n_nodes, children_left[tree_id],
        children_right[tree_id], feature[tree_id], threshold[tree_id], value[tree_id],
        n_node_samples[tree_id], weighted_n_node_samples[tree_id], impurity[tree_id],
        total_sample_cnt, n_targets,
        1,  // max_num_class = 1 for regressors
        false);  // is_classifier
  }

  return model;
}

/**
 * Load an IsolationForest using bulk construction
 *
 * This is an optimized version that constructs trees in bulk rather than
 * going through the ModelBuilder node-by-node.
 */
std::unique_ptr<treelite::Model> LoadIsolationForest(int n_estimators, int n_features,
    std::int64_t const* node_count, std::int64_t const** children_left,
    std::int64_t const** children_right, std::int64_t const** feature, double const** threshold,
    double const** value, std::int64_t const** n_node_samples,
    double const** weighted_n_node_samples, double const** impurity, double ratio_c) {
  TREELITE_CHECK_GT(n_estimators, 0) << "n_estimators must be at least 1";
  TREELITE_CHECK_GT(n_features, 0) << "n_features must be at least 1";

  // Create model with double precision
  auto model = Model::Create<double, double>();

  // Set up model metadata
  std::int32_t const n_targets = 1;
  model->num_feature = n_features;
  model->task_type = TaskType::kIsolationForest;
  model->average_tree_output = true;
  model->num_target = n_targets;

  // For isolation forests, num_class is always 1
  model->num_class = std::vector<std::int32_t>(n_targets, 1);

  // Set leaf vector shape
  model->leaf_vector_shape = std::vector<std::int32_t>{n_targets, 1};

  // Set up tree annotation arrays
  model->target_id = std::vector<std::int32_t>(n_estimators, 0);
  model->class_id = std::vector<std::int32_t>(n_estimators, 0);

  // Set postprocessor
  model->postprocessor = "exponential_standard_ratio";
  model->ratio_c = static_cast<float>(ratio_c);

  // Set base scores
  model->base_scores = std::vector<double>{0.0};

  // Get the typed model preset
  auto& preset = std::get<ModelPreset<double, double>>(model->variant_);
  preset.trees.resize(n_estimators);

  // Construct each tree using bulk operations
  for (int tree_id = 0; tree_id < n_estimators; ++tree_id) {
    int const n_nodes = static_cast<int>(node_count[tree_id]);
    std::int64_t const total_sample_cnt = n_node_samples[tree_id][0];

    BulkConstructTree<double, double>(preset.trees[tree_id], n_nodes, children_left[tree_id],
        children_right[tree_id], feature[tree_id], threshold[tree_id], value[tree_id],
        n_node_samples[tree_id], weighted_n_node_samples[tree_id], impurity[tree_id],
        total_sample_cnt, n_targets,
        1,  // max_num_class = 1 for isolation forests
        false);  // is_classifier
  }

  return model;
}

}  // namespace treelite::model_loader::sklearn
