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
#include <cstdint>
#include <cstddef>
#include <cfloat>
#include "../threading_utils/parallel_for.h"
#include "./pred_transform.h"

namespace treelite {

class GTILBridge {
 public:
  template <typename ThresholdType, typename LeafOutputType>
  inline static const typename Tree<ThresholdType, LeafOutputType>::Node*
  GetNode(const Tree<ThresholdType, LeafOutputType>& tree, int nid)  {
    return &tree.nodes_[nid];
  }
};

}  // namespace treelite

namespace {

using treelite::threading_utils::ThreadConfig;
using PredTransformFuncType = std::size_t (*) (const treelite::Model&, const float*, float*);

template <typename ThresholdType>
inline int NextNode(float fvalue, ThresholdType threshold, treelite::Operator op,
                    int left_child, int right_child) {
  switch (op) {
    case treelite::Operator::kLT:
      return (fvalue < threshold) ? left_child : right_child;
    case treelite::Operator::kLE:
      return (fvalue <= threshold) ? left_child : right_child;
    case treelite::Operator::kEQ:
      return (fvalue == threshold) ? left_child : right_child;
    case treelite::Operator::kGT:
      return (fvalue > threshold) ? left_child : right_child;
    case treelite::Operator::kGE:
      return (fvalue >= threshold) ? left_child : right_child;
    default:
      TREELITE_CHECK(false) << "Unrecognized comparison operator " << static_cast<int>(op);
      return -1;
  }
}

inline int NextNodeCategorical(float fvalue, const std::vector<std::uint32_t>& matching_categories,
                               bool categories_list_right_child, int left_child, int right_child) {
  bool is_matching_category;
  auto max_representable_int = static_cast<float>(std::uint32_t(1) << FLT_MANT_DIG);
  if (fvalue < 0 || std::fabs(fvalue) > max_representable_int) {
    is_matching_category = false;
  } else {
    const auto category_value = static_cast<std::uint32_t>(fvalue);
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

constexpr std::size_t kBlockOfRowsSize = 64;

struct BinaryClfRegrOutputLogic {
  template <typename ThresholdType, typename LeafOutputType>
  inline static void PushOutput(const treelite::Tree<ThresholdType, LeafOutputType>& tree,
                                std::size_t, int node_id, float* sum, std::size_t) {
    sum[0] += tree.LeafValue(node_id);
  }
  inline static void ApplyAverageFactor(const treelite::TaskParam& task_param, std::size_t num_tree,
                                        float* output, std::size_t batch_offset,
                                        std::size_t block_size) {
    const auto average_factor = static_cast<float>(num_tree);
    const unsigned num_class = task_param.num_class;
    for (std::size_t i = 0; i < block_size; ++i) {
      for (unsigned k = 0; k < num_class; ++k) {
        output[(batch_offset + i) * num_class + k] /= average_factor;
      }
    }
  }
};

struct MultiClfGrovePerClassOutputLogic {
  template <typename ThresholdType, typename LeafOutputType>
  inline static void PushOutput(const treelite::Tree<ThresholdType, LeafOutputType>& tree,
                                std::size_t tree_id, int node_id, float* sum,
                                std::size_t num_class) {
    sum[tree_id % num_class] += tree.LeafValue(node_id);
  }
  inline static void ApplyAverageFactor(const treelite::TaskParam& task_param, std::size_t num_tree,
                                        float* output, std::size_t batch_offset,
                                        std::size_t block_size) {
    auto num_boosting_round = num_tree / static_cast<std::size_t>(task_param.num_class);
    const auto average_factor = static_cast<float>(num_boosting_round);
    const unsigned num_class = task_param.num_class;
    for (std::size_t i = 0; i < block_size; ++i) {
      for (unsigned k = 0; k < num_class; ++k) {
        output[(batch_offset + i) * num_class + k] /= average_factor;
      }
    }
  }
};

struct MultiClfProbDistLeafOutputLogic {
  template <typename ThresholdType, typename LeafOutputType>
  inline static void PushOutput(const treelite::Tree<ThresholdType, LeafOutputType>& tree,
                                std::size_t tree_id, int node_id, float* sum,
                                std::size_t num_class) {
    auto leaf_vector = tree.LeafVector(node_id);
    for (unsigned int i = 0; i < num_class; ++i) {
      sum[i] += leaf_vector[i];
    }
  }
  inline static void ApplyAverageFactor(const treelite::TaskParam& task_param, std::size_t num_tree,
                                        float* output, std::size_t batch_offset,
                                        std::size_t block_size) {
    return BinaryClfRegrOutputLogic::ApplyAverageFactor(task_param, num_tree, output, batch_offset,
                                                        block_size);
  }
};

template <typename T1, typename T2>
inline T1 DivRoundUp(const T1 a, const T2 b) {
  return static_cast<T1>(std::ceil(static_cast<double>(a) / b));
}

template <typename ThresholdType, typename DMatrixType>
void FVecFill(const std::size_t block_size, const std::size_t batch_offset,
              const DMatrixType* input, const std::size_t fvec_offset, int num_feature,
              std::vector<ThresholdType>& feats) {
  for (std::size_t i = 0; i < block_size; ++i) {
    const std::size_t row_id = batch_offset + i;
    input->FillRow(row_id, &feats[(fvec_offset + i) * num_feature]);
  }
}

template <typename ThresholdType, typename DMatrixType>
void FVecDrop(const std::size_t block_size, const std::size_t batch_offset,
              const DMatrixType* input, const std::size_t fvec_offset, int num_feature,
              std::vector<ThresholdType>& feats) {
  for (std::size_t i = 0; i < block_size; ++i) {
    const std::size_t row_id = batch_offset + i;
    input->ClearRow(row_id, &feats[(fvec_offset + i) * num_feature]);
  }
}

template <typename ThresholdType, typename LeafOutputType, typename DMatrixType>
inline void InitOutPredictions(const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
                               const DMatrixType* input, float* output) {
  const std::size_t num_row = input->GetNumRow();
  const auto num_class = model.task_param.num_class;
  std::fill(output, output + num_row * num_class, model.param.global_bias);
}

template <bool has_categorical, typename OutputLogic, typename ThresholdType,
          typename LeafOutputType>
void PredValueByOneTree(const treelite::Tree<ThresholdType, LeafOutputType>& tree,
                        std::size_t tree_id, const ThresholdType* feats, float* output,
                        std::size_t num_class) {
  int node_id = 0;
  const typename treelite::Tree<ThresholdType, LeafOutputType>::Node* node
    = treelite::GTILBridge::GetNode(tree, node_id);
  while (!node->IsLeaf()) {
    const auto split_index = node->SplitIndex();
    const auto fvalue = feats[split_index];
    if (std::isnan(fvalue)) {
      node_id = node->DefaultChild();
    } else {
      if (has_categorical && node->SplitType() == treelite::SplitFeatureType::kCategorical) {
        node_id = NextNodeCategorical(fvalue, tree.MatchingCategories(node_id),
                                      node->CategoriesListRightChild(), node->LeftChild(),
                                      node->RightChild());
      } else {
        node_id = NextNode(fvalue, node->Threshold(), node->ComparisonOp(), node->LeftChild(),
                           node->RightChild());
      }
    }
    node = treelite::GTILBridge::GetNode(tree, node_id);
  }
  OutputLogic::PushOutput(tree, tree_id, node_id, output, num_class);
}


template <typename OutputLogic, typename ThresholdType, typename LeafOutputType>
void PredictByAllTrees(const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
                       float* output, const std::size_t batch_offset,
                       const std::size_t num_class, const std::vector<ThresholdType>& feats,
                       const std::size_t fvec_offset, const std::size_t block_size) {
  const int num_feature = model.num_feature;
  const std::size_t num_tree = model.trees.size();
  for (std::size_t tree_id = 0; tree_id < num_tree; ++tree_id) {
    const treelite::Tree<ThresholdType, LeafOutputType>& tree = model.trees[tree_id];
    auto has_categorical = tree.HasCategoricalSplit();
    if (has_categorical) {
      for (std::size_t i = 0; i < block_size; ++i) {
        PredValueByOneTree<true, OutputLogic>(tree, tree_id,
                                              &feats[(fvec_offset + i) * num_feature],
                                              &output[(batch_offset + i) * num_class], num_class);
      }
    } else {
      for (std::size_t i = 0; i < block_size; ++i) {
        PredValueByOneTree<false, OutputLogic>(tree, tree_id,
                                               &feats[(fvec_offset + i) * num_feature],
                                               &output[(batch_offset + i) * num_class], num_class);
      }
    }
  }
}


template <std::size_t block_of_rows_size, typename OutputLogic, typename ThresholdType,
          typename LeafOutputType, typename DMatrixType>
void PredictBatchByBlockOfRowsKernel(
    const treelite::ModelImpl<ThresholdType, LeafOutputType>& model, const DMatrixType* input,
    float* output, const ThreadConfig& thread_config) {
  const std::size_t num_row = input->GetNumRow();
  const int num_feature = model.num_feature;
  const auto& task_param = model.task_param;
  std::size_t n_blocks = DivRoundUp(num_row, block_of_rows_size);

  std::vector<ThresholdType> feats(thread_config.nthread * block_of_rows_size * num_feature);
  auto sched = treelite::threading_utils::ParallelSchedule::Static();
  treelite::threading_utils::ParallelFor(std::size_t(0), n_blocks, thread_config, sched,
                                         [&](std::size_t block_id, int thread_id) {
    const std::size_t batch_offset = block_id * block_of_rows_size;
    const std::size_t block_size =
        std::min(num_row - batch_offset, block_of_rows_size);
    const std::size_t fvec_offset = thread_id * block_of_rows_size;

    FVecFill(block_size, batch_offset, input, fvec_offset, num_feature, feats);
    // process block of rows through all trees to keep cache locality
    PredictByAllTrees<OutputLogic>(model, output, batch_offset,
                                   static_cast<std::size_t>(task_param.num_class), feats,
                                   fvec_offset, block_size);
    FVecDrop(block_size, batch_offset, input, fvec_offset, num_feature, feats);
    if (model.average_tree_output) {
      OutputLogic::ApplyAverageFactor(task_param, model.GetNumTree(), output, batch_offset,
                                      block_size);
    }
  });
}

template <typename ThresholdType, typename LeafOutputType, typename DMatrixType>
inline void PredictRaw(const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
                       const DMatrixType* input, float* output, const ThreadConfig& thread_config) {
  InitOutPredictions(model, input, output);

  switch (model.task_type) {
    case treelite::TaskType::kBinaryClfRegr:
      PredictBatchByBlockOfRowsKernel<kBlockOfRowsSize, BinaryClfRegrOutputLogic>(
          model, input, output, thread_config);
      break;
    case treelite::TaskType::kMultiClfGrovePerClass:
      PredictBatchByBlockOfRowsKernel<kBlockOfRowsSize, MultiClfGrovePerClassOutputLogic>(
          model, input, output, thread_config);
      break;
    case treelite::TaskType::kMultiClfProbDistLeaf:
      PredictBatchByBlockOfRowsKernel<kBlockOfRowsSize, MultiClfProbDistLeafOutputLogic>(
          model, input, output, thread_config);
      break;
    case treelite::TaskType::kMultiClfCategLeaf:
    default:
      TREELITE_LOG(FATAL)
      << "Unsupported task type of the tree ensemble model: "
      << static_cast<int>(model.task_type);
  }
}

template <typename ThresholdType, typename LeafOutputType, typename DMatrixType>
inline std::size_t PredTransform(const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
                                 const DMatrixType* input, float* output,
                                 const ThreadConfig& thread_config, bool pred_transform) {
  std::size_t output_size_per_row;
  const auto num_class = model.task_param.num_class;
  std::size_t num_row = input->GetNumRow();
  if (pred_transform) {
    std::vector<float> temp(treelite::gtil::GetPredictOutputSize(&model, num_row));
    PredTransformFuncType pred_transform_func
        = treelite::gtil::LookupPredTransform(model.param.pred_transform);
    auto sched = treelite::threading_utils::ParallelSchedule::Static();
    // Query the size of output per input row.
    output_size_per_row = pred_transform_func(model, &output[0], &temp[0]);
    // Now transform predictions in parallel
    treelite::threading_utils::ParallelFor(std::size_t(0), num_row, thread_config, sched,
                                           [&](std::size_t row_id, int thread_id) {
      pred_transform_func(model, &output[row_id * num_class],
                          &temp[row_id * output_size_per_row]);
    });
    // Copy transformed score back to output
    temp.resize(output_size_per_row * num_row);
    std::copy(temp.begin(), temp.end(), output);
  } else {
    output_size_per_row = model.task_param.num_class;
  }
  return output_size_per_row * num_row;
}

template <typename ThresholdType, typename LeafOutputType, typename DMatrixType>
inline std::size_t PredictImpl(const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
                               const DMatrixType* input, float* output,
                               const ThreadConfig& thread_config, bool pred_transform) {
  PredictRaw(model, input, output, thread_config);
  return PredTransform(model, input, output, thread_config, pred_transform);
}

}  // anonymous namespace

namespace treelite {
namespace gtil {

std::size_t Predict(const Model* model, const DMatrix* input, float* output, int nthread,
                    bool pred_transform) {
  // If nthread <= 0, then use all CPU cores in the system
  auto thread_config = threading_utils::ConfigureThreadConfig(nthread);
  // Check type of DMatrix
  const auto* d1 = dynamic_cast<const DenseDMatrixImpl<float>*>(input);
  const auto* d2 = dynamic_cast<const CSRDMatrixImpl<float>*>(input);
  if (d1) {
    return model->Dispatch([d1, output, thread_config, pred_transform](const auto& model) {
      return PredictImpl(model, d1, output, thread_config, pred_transform);
    });
  } else if (d2) {
    return model->Dispatch([d2, output, thread_config, pred_transform](const auto& model) {
      return PredictImpl(model, d2, output, thread_config, pred_transform);
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
