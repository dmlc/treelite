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
#include <cmath>
#include <cstddef>
#include <cfloat>
#include "../threading_utils/parallel_for.h"
#include "./pred_transform.h"

namespace {

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
                                    const DMatrixType* input, float* output, int nthread,
                                    bool pred_transform, OutputFunc output_func) {
  using TreeType = treelite::Tree<ThresholdType, LeafOutputType>;
  const size_t num_row = input->GetNumRow();
  const size_t num_col = input->GetNumCol();
  std::vector<ThresholdType> row(num_col);
  const treelite::TaskParam task_param = model.task_param;
  std::vector<float> sum_tloc(task_param.num_class * nthread);
  std::vector<float> sum_tot(task_param.num_class);

  // Query the size of output per input row.
  // This is guaranteed to be at most GetPredictOutputSize(num_row=1).
  std::size_t output_size_per_row;
  if (pred_transform) {
    std::vector<float> temp(treelite::gtil::GetPredictOutputSize(&model, 1));
    PredTransformFuncType pred_transform_func
        = treelite::gtil::LookupPredTransform(model.param.pred_transform);
    output_size_per_row = pred_transform_func(model, sum_tloc.data(), temp.data());
  } else {
    output_size_per_row = task_param.num_class;
  }

  auto sched = treelite::threading_utils::ParallelSchedule::Static();
  for (std::size_t row_id = 0; row_id < num_row; ++row_id) {
    input->FillRow(row_id, row.data());
    std::fill(sum_tloc.begin(), sum_tloc.end(), 0.0f);
    std::fill(sum_tot.begin(), sum_tot.end(), 0.0f);
    const std::size_t num_tree = model.trees.size();
    treelite::threading_utils::ParallelFor(std::size_t(0), num_tree, nthread, sched,
                                           [&](std::size_t tree_id, int thread_id) {
      float* sum = &sum_tloc[thread_id * task_param.num_class];
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
      output_func(tree, tree_id, node_id, sum);
    });
    for (int thread_id = 0; thread_id < nthread; ++thread_id) {
      for (unsigned i = 0; i < task_param.num_class; ++i) {
        sum_tot[i] += sum_tloc[thread_id * task_param.num_class + i];
      }
    }
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
      for (unsigned int i = 0; i < task_param.num_class; ++i) {
        sum_tot[i] /= average_factor;
      }
    }
    for (unsigned int i = 0; i < task_param.num_class; ++i) {
      sum_tot[i] += model.param.global_bias;
    }
    if (pred_transform) {
      PredTransformFuncType pred_transform_func
        = treelite::gtil::LookupPredTransform(model.param.pred_transform);
      pred_transform_func(model, sum_tot.data(), &output[row_id * output_size_per_row]);
    } else {
      for (unsigned int i = 0; i < task_param.num_class; ++i) {
        output[row_id * output_size_per_row + i] = sum_tot[i];
      }
    }
    input->ClearRow(row_id, row.data());
  }
  return output_size_per_row * num_row;
}

template <typename ThresholdType, typename LeafOutputType, typename DMatrixType>
inline std::size_t PredictImpl(const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
                               const DMatrixType* input, float* output, int nthread,
                               bool pred_transform) {
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
      return PredictImplInner(model, input, output, nthread, pred_transform, output_logic);
    } else {
      // multi-class classification with gradient boosted trees
      auto output_logic = [task_param](
          const TreeType& tree, int tree_id, int node_id, float* sum) {
        sum[tree_id % task_param.num_class] += tree.LeafValue(node_id);
      };
      return PredictImplInner(model, input, output, nthread, pred_transform, output_logic);
    }
  } else {
    auto output_logic = [task_param](
        const TreeType& tree, int tree_id, int node_id, float* sum) {
      sum[0] += tree.LeafValue(node_id);
    };
    return PredictImplInner(model, input, output, nthread, pred_transform, output_logic);
  }
}

}  // anonymous namespace

namespace treelite {
namespace gtil {

std::size_t Predict(const Model* model, const DMatrix* input, float* output, int nthread,
                    bool pred_transform) {
  // If nthread <= 0, then use all CPU cores in the system
  if (nthread <= 0) {
    nthread = threading_utils::MaxNumThread();
  }
  // Check type of DMatrix
  const auto* d1 = dynamic_cast<const DenseDMatrixImpl<float>*>(input);
  const auto* d2 = dynamic_cast<const CSRDMatrixImpl<float>*>(input);
  if (d1) {
    return model->Dispatch([d1, output, nthread, pred_transform](const auto& model) {
      return PredictImpl(model, d1, output, nthread, pred_transform);
    });
  } else if (d2) {
    return model->Dispatch([d2, output, nthread, pred_transform](const auto& model) {
      return PredictImpl(model, d2, output, nthread, pred_transform);
    });
  } else {
    TREELITE_LOG(FATAL) << "DMatrix with float64 data is not supported";
    return 0;
  }
}

std::size_t Predict(const Model* model, const float* input, std::size_t num_row, float* output,
                    int nthread, bool pred_transform) {
  std::unique_ptr<DenseDMatrixImpl<float>> dmat =
      std::make_unique<DenseDMatrixImpl<float>>(
          std::vector<float>(input, input + num_row * model->num_feature),
          std::numeric_limits<float>::quiet_NaN(),
          num_row,
          model->num_feature);
  return Predict(model, dmat.get(), output, nthread, pred_transform);
}

std::size_t GetPredictOutputSize(const Model* model, std::size_t num_row) {
  return model->task_param.num_class * num_row;
}

std::size_t GetPredictOutputSize(const Model* model, const DMatrix* input) {
  return GetPredictOutputSize(model, input->GetNumRow());
}

}  // namespace gtil
}  // namespace treelite
