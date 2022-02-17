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

/*!
 * \brief A dense feature vector.
 */
class FVec {
 public:
  void Init(std::size_t size) {
    data_.resize(size);
    missing_.resize(size);
    std::fill(data_.begin(), data_.end(), std::numeric_limits<float>::quiet_NaN());
    std::fill(missing_.begin(), missing_.end(), true);
    has_missing_ = true;
  }
  template <typename DMatrixType>
  void Fill(const DMatrixType* input, std::size_t row_id) {
    std::size_t feature_count = 0;
    input->FillRow(row_id, &data_[0]);
    for (std::size_t i = 0; i < data_.size(); ++i) {
      if ( !(missing_[i] = std::isnan(data_[i])) ) {
        ++feature_count;
      }
    }
    has_missing_ = data_.size() != feature_count;
  }
  template <typename DMatrixType>
  void Clear(const DMatrixType* input, std::size_t row_id) {
    input->ClearRow(row_id, &data_[0]);
    std::fill(missing_.begin(), missing_.end(), true);
    has_missing_ = true;
  }
  std::size_t Size() const {
    return data_.size();
  }
  float GetFValue(std::size_t i) const {
    return data_[i];
  }
  bool IsMissing(size_t i) const {
    return missing_[i];
  }
  bool HasMissing() const {
    return has_missing_;
  }

 private:
  std::vector<float> data_;
  std::vector<bool> missing_;
  bool has_missing_;
};

template <typename ThresholdType>
inline int NextNode(float fvalue, ThresholdType threshold, treelite::Operator op, int left_child) {
  // XGBoost
  if (op == treelite::Operator::kLT) {
    return left_child + !(fvalue < threshold);
  }
  // LightGBM, sklearn, cuML RF
  if (op == treelite::Operator::kLE) {
    return left_child + !(fvalue <= threshold);
  }
  switch (op) {
    case treelite::Operator::kEQ:
      return left_child + !(fvalue == threshold);
    case treelite::Operator::kGT:
      return left_child + !(fvalue > threshold);
    case treelite::Operator::kGE:
      return left_child + !(fvalue >= threshold);
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

template <typename DMatrixType>
void FVecFill(const std::size_t block_size, const std::size_t batch_offset,
              const DMatrixType* input, const std::size_t fvec_offset, int num_feature,
              std::vector<FVec>& feats) {
  for (std::size_t i = 0; i < block_size; ++i) {
    FVec& vec = feats[fvec_offset + i];
    if (vec.Size() == 0) {
      vec.Init(static_cast<std::size_t>(num_feature));
    }
    const std::size_t row_id = batch_offset + i;
    vec.Fill(input, row_id);
  }
}

template <typename DMatrixType>
void FVecDrop(const std::size_t block_size, const std::size_t batch_offset,
              const DMatrixType* input, const std::size_t fvec_offset, int num_feature,
              std::vector<FVec>& feats) {
  for (std::size_t i = 0; i < block_size; ++i) {
    const std::size_t row_id = batch_offset + i;
    feats[fvec_offset + i].Clear(input, row_id);
  }
}

template <typename ThresholdType, typename LeafOutputType, typename DMatrixType>
inline void InitOutPredictions(const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
                               const DMatrixType* input, float* output) {
  const std::size_t num_row = input->GetNumRow();
  const auto num_class = model.task_param.num_class;
  std::fill(output, output + num_row * num_class, model.param.global_bias);
}

template <bool has_missing, bool has_categorical, typename OutputLogic, typename ThresholdType,
          typename LeafOutputType>
void PredValueByOneTreeImpl(const treelite::Tree<ThresholdType, LeafOutputType>& tree,
                            std::size_t tree_id, const FVec& feats, float* output,
                            std::size_t num_class) {
  int node_id = 0;
  const typename treelite::Tree<ThresholdType, LeafOutputType>::Node* node
    = treelite::GTILBridge::GetNode(tree, node_id);
  while (!node->IsLeaf()) {
    const auto split_index = node->SplitIndex();
    if (has_missing && feats.IsMissing(split_index)) {
      node_id = node->DefaultChild();
    } else {
      const float fvalue = feats.GetFValue(split_index);
      if (has_categorical && node->SplitType() == treelite::SplitFeatureType::kCategorical) {
        node_id = NextNodeCategorical(fvalue, tree.MatchingCategories(node_id),
                                      node->CategoriesListRightChild(), node->LeftChild(),
                                      node->RightChild());
      } else {
        node_id = NextNode(fvalue, node->Threshold(), node->ComparisonOp(), node->LeftChild());
      }
    }
    node = treelite::GTILBridge::GetNode(tree, node_id);
  }
  OutputLogic::PushOutput(tree, tree_id, node_id, output, num_class);
}

template <bool has_categorical, typename OutputLogic, typename ThresholdType,
          typename LeafOutputType>
void PredValueByOneTree(const treelite::Tree<ThresholdType, LeafOutputType>& tree,
                        std::size_t tree_id, const FVec& feats, float* output,
                        std::size_t num_class) {
  if (feats.HasMissing()) {
    PredValueByOneTreeImpl<true, has_categorical, OutputLogic>(
        tree, tree_id, feats, output, num_class);
  } else {
    PredValueByOneTreeImpl<false, has_categorical, OutputLogic>(
        tree, tree_id, feats, output, num_class);
  }
}

template <typename OutputLogic, typename ThresholdType, typename LeafOutputType>
void PredictByAllTrees(const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
                       float* output, const std::size_t batch_offset,
                       const std::size_t num_class, const std::vector<FVec>& feats,
                       const std::size_t fvec_offset, const std::size_t block_size) {
  const int num_feature = model.num_feature;
  const std::size_t num_tree = model.trees.size();
  for (std::size_t tree_id = 0; tree_id < num_tree; ++tree_id) {
    const treelite::Tree<ThresholdType, LeafOutputType>& tree = model.trees[tree_id];
    auto has_categorical = tree.HasCategoricalSplit();
    if (has_categorical) {
      for (std::size_t i = 0; i < block_size; ++i) {
        PredValueByOneTree<true, OutputLogic>(tree, tree_id, feats[fvec_offset + i],
                                              &output[(batch_offset + i) * num_class], num_class);
      }
    } else {
      for (std::size_t i = 0; i < block_size; ++i) {
        PredValueByOneTree<false, OutputLogic>(tree, tree_id, feats[fvec_offset + i],
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

  std::vector<FVec> feats(thread_config.nthread * block_of_rows_size);
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

template <typename OutputLogic, typename ThresholdType, typename LeafOutputType,
          typename DMatrixType>
void PredictBatchTreeParallelKernel(
    const treelite::ModelImpl<ThresholdType, LeafOutputType>& model, const DMatrixType* input,
    float* output, const ThreadConfig& thread_config) {
  const std::size_t num_row = input->GetNumRow();
  const std::size_t num_tree = model.GetNumTree();
  const int num_feature = model.num_feature;
  const auto num_class = model.task_param.num_class;

  FVec feats;
  feats.Init(num_feature);
  std::vector<float> sum_tloc(num_class * thread_config.nthread);
  auto sched = treelite::threading_utils::ParallelSchedule::Static();
  for (std::size_t row_id = 0; row_id < num_row; ++row_id) {
    std::fill(sum_tloc.begin(), sum_tloc.end(), 0.0f);
    feats.Fill(input, row_id);
    treelite::threading_utils::ParallelFor(std::size_t(0), num_tree, thread_config, sched,
                                           [&](std::size_t tree_id, int thread_id) {
      const treelite::Tree<ThresholdType, LeafOutputType>& tree = model.trees[tree_id];
      auto has_categorical = tree.HasCategoricalSplit();
      if (has_categorical) {
        PredValueByOneTree<true, OutputLogic>(tree, tree_id, feats,
                                              &sum_tloc[thread_id * num_class], num_class);
      } else {
        PredValueByOneTree<false, OutputLogic>(tree, tree_id, feats,
                                               &sum_tloc[thread_id * num_class], num_class);
      }
    });
    feats.Clear(input, row_id);
    for (std::uint32_t thread_id = 0; thread_id < thread_config.nthread; ++thread_id) {
      for (unsigned i = 0; i < num_class; ++i) {
        output[row_id * num_class + i] += sum_tloc[thread_id * num_class + i];
      }
    }
  }
  if (model.average_tree_output) {
    for (std::size_t row_id = 0; row_id < num_row; ++row_id) {
      OutputLogic::ApplyAverageFactor(model.task_param, num_tree, output, row_id, 1);
    }
  }
}

template <typename OutputLogic, typename ThresholdType, typename LeafOutputType,
          typename DMatrixType>
void PredictBatchDispatch(
    const treelite::ModelImpl<ThresholdType, LeafOutputType>& model, const DMatrixType* input,
    float* output, const ThreadConfig& thread_config) {
  if (input->GetNumRow() < kBlockOfRowsSize) {
    // Small batch size => tree parallel method
    PredictBatchTreeParallelKernel<OutputLogic>(model, input, output, thread_config);
  } else {
    // Sufficiently large batch size => row parallel method
    PredictBatchByBlockOfRowsKernel<kBlockOfRowsSize, OutputLogic>(
        model, input, output, thread_config);
  }
}

template <typename ThresholdType, typename LeafOutputType, typename DMatrixType>
inline void PredictRaw(const treelite::ModelImpl<ThresholdType, LeafOutputType>& model,
                       const DMatrixType* input, float* output, const ThreadConfig& thread_config) {
  InitOutPredictions(model, input, output);

  switch (model.task_type) {
    case treelite::TaskType::kBinaryClfRegr:
      PredictBatchDispatch<BinaryClfRegrOutputLogic>(model, input, output, thread_config);
      break;
    case treelite::TaskType::kMultiClfGrovePerClass:
      PredictBatchDispatch<MultiClfGrovePerClassOutputLogic>(model, input, output, thread_config);
      break;
    case treelite::TaskType::kMultiClfProbDistLeaf:
      PredictBatchDispatch<MultiClfProbDistLeafOutputLogic>(model, input, output, thread_config);
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
