/*!
 * Copyright (c) 2021 by Contributors
 * \file predict.cc
 * \author Hyunsu Cho
 * \brief General Tree Inference Library (GTIL), providing a reference implementation for
 *        predicting with decision trees. GTIL is useful in cases it is infeasible to build the
 *        tree models as native shared libs.
 */
#include <treelite/gtil.h>
#include <treelite/tree.h>
#include <treelite/data.h>
#include <treelite/logging.h>
#include <limits>
#include <vector>
#include <memory>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cfloat>
#include "../threading_utils/parallel_for.h"
#include "./pred_transform.h"

namespace {

using treelite::threading_utils::ThreadConfig;
using PredTransformFuncType = std::size_t (*) (const treelite::Model&, const float*, float*);

template <typename T>
inline int NextNode(float fvalue, T threshold, treelite::Operator op,
                    int left_child, int right_child, int default_child) {
  if (std::isnan(fvalue)) {
    return default_child;
  }
  switch (op) {
    case treelite::Operator::kEQ:
      return (fvalue == threshold) ? left_child : right_child;
    case treelite::Operator::kLT:
      return (fvalue < threshold) ? left_child : right_child;
    case treelite::Operator::kLE:
      return (fvalue <= threshold) ? left_child : right_child;
    case treelite::Operator::kGT:
      return (fvalue > threshold) ? left_child : right_child;
    case treelite::Operator::kGE:
      return (fvalue >= threshold) ? left_child : right_child;
    default:
      TREELITE_CHECK(false) << "Unrecognized comparison operator " << static_cast<int>(op);
      return -1;
  }
}

inline int NextNodeCategorical(float fvalue, const std::vector<uint32_t>& matching_categories,
                               bool categories_list_right_child, int left_child, int right_child,
                               int default_child) {
  if (std::isnan(fvalue)) {
    return default_child;
  }
  bool is_matching_category;
  float max_representable_int = static_cast<float>(uint32_t(1) << FLT_MANT_DIG);
  if (fvalue < 0 || std::fabs(fvalue) > max_representable_int) {
    is_matching_category = false;
  } else {
    const auto category_value = static_cast<uint32_t>(fvalue);
    is_matching_category = (
        std::find(matching_categories.begin(), matching_categories.end(), category_value)
        != matching_categories.end());
  }
  if (categories_list_right_child) {
    return is_matching_category ? right_child : left_child;
  } else {
    return is_matching_category ? left_child : right_child;
  }
}

template <typename ThresholdType, typename LeafOutputType, typename DMatrixType,
          typename OutputFunc>
inline std::size_t PredictImplInner(const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
                                    const DMatrixType* input, float* output,
                                    const ThreadConfig& thread_config, bool pred_transform,
                                    OutputFunc output_func) {
  using TreeType = treelite::Tree<ThresholdType, LeafOutputType>;
  const std::size_t num_row = input->GetNumRow();
  const std::size_t num_col = input->GetNumCol();
  const std::size_t num_tree = model.trees.size();
  const treelite::TaskParam task_param = model.task_param;
  const unsigned num_class = task_param.num_class;
  std::vector<ThresholdType> row_tloc(num_col * thread_config.nthread);
  std::vector<float> sum_tloc(thread_config.nthread * num_row * num_class, 0.0f);
  std::vector<float> sum_tot(num_row * num_class, 0.0f);

  // Query the size of output per input row.
  std::size_t output_size_per_row;
  if (pred_transform) {
    std::vector<float> temp_before_transform(num_class);
    std::vector<float> temp_after_transform(num_class);
    PredTransformFuncType pred_transform_func
        = treelite::gtil::LookupPredTransform(model.param.pred_transform);
    output_size_per_row = pred_transform_func(model, temp_before_transform.data(),
                                              temp_after_transform.data());
  } else {
    output_size_per_row = num_class;
  }

  auto sched = treelite::threading_utils::ParallelSchedule::Static();
  treelite::threading_utils::ParallelFor2D(std::size_t(0), num_row, std::size_t(0), num_tree,
                                           thread_config, sched,
                                           [&](std::size_t row_id, std::size_t tree_id,
                                               int thread_id) {
    ThresholdType* row = &row_tloc[thread_id * num_col];
    // sum_tloc[thread_id, row_id, :]
    float* sum = &sum_tloc[(thread_id * num_row + row_id) * num_class];
    input->FillRow(row_id, row);

    const TreeType& tree = model.trees[tree_id];
    int node_id = 0;
    while (!tree.IsLeaf(node_id)) {
      treelite::SplitFeatureType split_type = tree.SplitType(node_id);
      if (split_type == treelite::SplitFeatureType::kNumerical) {
        node_id = NextNode(row[tree.SplitIndex(node_id)], tree.Threshold(node_id),
                           tree.ComparisonOp(node_id), tree.LeftChild(node_id),
                           tree.RightChild(node_id), tree.DefaultChild(node_id));
      } else if (split_type == treelite::SplitFeatureType::kCategorical) {
        node_id = NextNodeCategorical(row[tree.SplitIndex(node_id)],
                                      tree.MatchingCategories(node_id),
                                      tree.CategoriesListRightChild(node_id),
                                      tree.LeftChild(node_id), tree.RightChild(node_id),
                                      tree.DefaultChild(node_id));
      } else {
        TREELITE_CHECK(false) << "Unrecognized split type: " << static_cast<int>(split_type);
      }
    }
    input->ClearRow(row_id, row);
    output_func(tree, tree_id, node_id, sum);
  });
  // sum_tot[row_id, k] = sum(sum_tloc[:, row_id, k]) for each 0 <= k < [num_class]
  treelite::threading_utils::ParallelFor2D(std::size_t(0), num_row, unsigned(0), num_class,
                                           thread_config, sched,
                                           [&](std::size_t row_id, unsigned k, int) {
    for (int thread_id = 0; thread_id < thread_config.nthread; ++thread_id) {
      sum_tot[row_id * num_class + k] += sum_tloc[(thread_id * num_row + row_id) * num_class + k];
    }
  });
  if (model.average_tree_output) {
    float average_factor;
    if (model.task_type == treelite::TaskType::kMultiClfGrovePerClass) {
      TREELITE_CHECK(task_param.grove_per_class);
      TREELITE_CHECK_EQ(task_param.leaf_vector_size, 1);
      TREELITE_CHECK_GT(task_param.num_class, 1);
      TREELITE_CHECK_EQ(num_tree % task_param.num_class, 0)
        << "Expected the number of trees to be divisible by the number of classes";
      int num_boosting_round = num_tree / static_cast<int>(task_param.num_class);
      average_factor = static_cast<float>(num_boosting_round);
    } else {
      TREELITE_CHECK(model.task_type == treelite::TaskType::kBinaryClfRegr
                     || model.task_type == treelite::TaskType::kMultiClfProbDistLeaf);
      TREELITE_CHECK(task_param.num_class == task_param.leaf_vector_size);
      TREELITE_CHECK(!task_param.grove_per_class);
      average_factor = static_cast<float>(num_tree);
    }
    treelite::threading_utils::ParallelFor2D(std::size_t(0), num_row, unsigned(0), num_class,
                                             thread_config, sched,
                                             [&](std::size_t row_id, unsigned k, int) {
      const std::size_t idx = row_id * num_class + k;
      sum_tot[idx] = sum_tot[idx] / average_factor + model.param.global_bias;
    });
  } else {
    treelite::threading_utils::ParallelFor2D(std::size_t(0), num_row, unsigned(0), num_class,
                                             thread_config, sched,
                                             [&](std::size_t row_id, unsigned k, int) {
      const std::size_t idx = row_id * num_class + k;
      sum_tot[idx] += model.param.global_bias;
    });
  }
  if (pred_transform) {
    PredTransformFuncType pred_transform_func
        = treelite::gtil::LookupPredTransform(model.param.pred_transform);
    treelite::threading_utils::ParallelFor(std::size_t(0), num_row, thread_config, sched,
                                           [&](std::size_t row_id, int) {
      pred_transform_func(model, &sum_tot[row_id * num_class],
                          &output[row_id * output_size_per_row]);
    });
  } else {
    treelite::threading_utils::ParallelFor2D(std::size_t(0), num_row, unsigned(0), num_class,
                                             thread_config, sched,
                                             [&](std::size_t row_id, unsigned k, int) {
      output[row_id * output_size_per_row + k] = sum_tot[row_id * num_class + k];
    });
  }
  return output_size_per_row * num_row;
}

template <typename ThresholdType, typename LeafOutputType, typename DMatrixType>
inline std::size_t PredictImpl(const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
                               const DMatrixType* input, float* output,
                               const ThreadConfig& thread_config, bool pred_transform) {
  using TreeType = treelite::Tree<ThresholdType, LeafOutputType>;
  const treelite::TaskParam task_param = model.task_param;
  if (task_param.num_class > 1) {
    if (task_param.leaf_vector_size > 1) {
      // multi-class classification with random forest
      auto output_logic = [task_param](
          const TreeType& tree, int, int node_id, float* sum) {
        auto leaf_vector = tree.LeafVector(node_id);
        for (unsigned int i = 0; i < task_param.leaf_vector_size; ++i) {
          sum[i] += leaf_vector[i];
        }
      };
      return PredictImplInner(model, input, output, thread_config, pred_transform, output_logic);
    } else {
      // multi-class classification with gradient boosted trees
      auto output_logic = [task_param](
          const TreeType& tree, int tree_id, int node_id, float* sum) {
        sum[tree_id % task_param.num_class] += tree.LeafValue(node_id);
      };
      return PredictImplInner(model, input, output, thread_config, pred_transform, output_logic);
    }
  } else {
    auto output_logic = [task_param](
        const TreeType& tree, int tree_id, int node_id, float* sum) {
      sum[0] += tree.LeafValue(node_id);
    };
    return PredictImplInner(model, input, output, thread_config, pred_transform, output_logic);
  }
}

}  // anonymous namespace

namespace treelite {
namespace gtil {

std::size_t
Predictor::Predict(const DMatrix* input, float* output, int nthread, bool pred_transform) {
  // If nthread <= 0, then use all CPU cores in the system
  auto thread_config = threading_utils::ConfigureThreadConfig(nthread);
  // Check type of DMatrix
  const auto* d1 = dynamic_cast<const DenseDMatrixImpl<float>*>(input);
  const auto* d2 = dynamic_cast<const CSRDMatrixImpl<float>*>(input);
  if (d1) {
    return model_.Dispatch([d1, output, thread_config, pred_transform](const auto& model) {
      return PredictImpl(model, d1, output, thread_config, pred_transform);
    });
  } else if (d2) {
    return model_.Dispatch([d2, output, thread_config, pred_transform](const auto& model) {
      return PredictImpl(model, d2, output, thread_config, pred_transform);
    });
  } else {
    TREELITE_LOG(FATAL) << "DMatrix with float64 data is not supported";
    return 0;
  }
}

std::size_t
Predictor::Predict(const float* input, std::size_t num_row, float* output, int nthread,
                   bool pred_transform) {
  std::unique_ptr<DenseDMatrixImpl<float>> dmat =
      std::make_unique<DenseDMatrixImpl<float>>(
          std::vector<float>(input, input + num_row * model_.num_feature),
          std::numeric_limits<float>::quiet_NaN(),
          num_row,
          model_.num_feature);
  return Predict(dmat.get(), output, nthread, pred_transform);
}

std::size_t
Predictor::QueryResultSize(std::size_t num_row) const {
  return model_.task_param.num_class * num_row;
}

std::size_t
Predictor::QueryResultSize(const DMatrix* input) const {
  return QueryResultSize(input->GetNumRow());
}

}  // namespace gtil
}  // namespace treelite
