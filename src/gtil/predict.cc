/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file predict.cc
 * \author Hyunsu Cho
 * \brief General Tree Inference Library (GTIL), providing a reference implementation for
 *        predicting with decision trees.
 */
#include <treelite/detail/threading_utils.h>
#include <treelite/gtil.h>
#include <treelite/logging.h>
#include <treelite/tree.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <experimental/mdspan>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "./postprocessor.h"

namespace treelite::gtil {

namespace stdex = std::experimental;
// Multidimensional array views. Use row-major (C) layout
template <typename ElemT>
using Array1DView = stdex::mdspan<ElemT, stdex::dextents<std::uint64_t, 1>, stdex::layout_right>;
template <typename ElemT>
using Array2DView = stdex::mdspan<ElemT, stdex::dextents<std::uint64_t, 2>, stdex::layout_right>;
template <typename ElemT>
using Array3DView = stdex::mdspan<ElemT, stdex::dextents<std::uint64_t, 3>, stdex::layout_right>;
template <typename ElemT>
using CArray1DView
    = stdex::mdspan<ElemT const, stdex::dextents<std::uint64_t, 1>, stdex::layout_right>;
template <typename ElemT>
using CArray2DView
    = stdex::mdspan<ElemT const, stdex::dextents<std::uint64_t, 2>, stdex::layout_right>;

template <typename InputT>
class DenseMatrixAccessor {
 public:
  DenseMatrixAccessor(InputT const* input, std::uint64_t num_row, std::int32_t num_feature)
      : input_view_(input, num_row, num_feature) {}

  CArray1DView<InputT> GetRow(std::uint64_t row_id, int thread_id) {
    auto row = stdex::submdspan(input_view_, row_id, stdex::full_extent);
    static_assert(std::is_same_v<decltype(row), CArray1DView<InputT>>);
    return row;
  }

 private:
  CArray2DView<InputT> input_view_;
};

template <typename InputT>
class SparseMatrixAccessor {
 public:
  SparseMatrixAccessor(InputT const* data, std::uint64_t const* col_ind,
      std::uint64_t const* row_ptr, std::uint64_t num_row, std::int32_t num_feature,
      detail::threading_utils::ThreadConfig const& thread_config)
      : data_(data, row_ptr[num_row]),
        col_ind_(col_ind, row_ptr[num_row]),
        row_ptr_(row_ptr, num_row + 1),
        dense_row_(thread_config.nthread * num_feature) {
    dense_row_view_ = Array2DView<InputT>(dense_row_.data(), thread_config.nthread, num_feature);
  }

  // This function can safely be called from multiple threads, as long as thread_id is unique.
  CArray1DView<InputT> GetRow(std::uint64_t row_id, int thread_id) {
    auto row = stdex::submdspan(dense_row_view_, thread_id, stdex::full_extent);
    static_assert(std::is_same_v<decltype(row), Array1DView<InputT>>);

    auto data_slice = stdex::submdspan(
        data_, std::pair<std::uint64_t, std::uint64_t>{row_ptr_(row_id), row_ptr_(row_id + 1)});
    auto col_ind_slice = stdex::submdspan(
        col_ind_, std::pair<std::uint64_t, std::uint64_t>{row_ptr_(row_id), row_ptr_(row_id + 1)});
    for (std::uint64_t i = 0; i < row.extent(0); ++i) {
      row[i] = std::numeric_limits<InputT>::quiet_NaN();
    }
    for (std::uint64_t i = 0; i < col_ind_slice.extent(0); ++i) {
      row[col_ind_slice(i)] = data_slice(i);
    }
    return row;
  }

 private:
  CArray1DView<InputT> data_;
  CArray1DView<std::uint64_t> col_ind_;
  CArray1DView<std::uint64_t> row_ptr_;
  // Temporary space to convert sparse rows into dense form
  // Allocate one row per thread
  std::vector<InputT> dense_row_;
  Array2DView<InputT> dense_row_view_;
};

template <typename InputT, typename ThresholdT>
inline int NextNode(
    InputT fvalue, ThresholdT threshold, Operator op, int left_child, int right_child) {
  static_assert(std::is_floating_point_v<InputT>, "Expected floating point type for input");
  bool cond = false;
  switch (op) {
  case Operator::kLT:
    cond = fvalue < threshold;
    break;
  case Operator::kLE:
    cond = fvalue <= threshold;
    break;
  case Operator::kEQ:
    cond = fvalue == threshold;
    break;
  case Operator::kGT:
    cond = fvalue > threshold;
    break;
  case Operator::kGE:
    cond = fvalue >= threshold;
    break;
  default:
    TREELITE_CHECK(false) << "Unrecognized comparison operator " << static_cast<int>(op);
    return -1;
  }
  return (cond ? left_child : right_child);
}

template <typename InputT>
inline int NextNodeCategorical(InputT fvalue, std::vector<std::uint32_t> const& category_list,
    bool category_list_right_child, int left_child, int right_child) {
  static_assert(std::is_floating_point_v<InputT>, "Expected floating point type for input");
  bool category_matched;
  // A valid (integer) category must satisfy two criteria:
  // 1) it must be exactly representable as InputT (float or double)
  // 2) it must fit into uint32_t
  auto max_representable_int
      = std::min(static_cast<InputT>(std::numeric_limits<std::uint32_t>::max()),
          static_cast<InputT>(std::uint64_t(1) << std::numeric_limits<InputT>::digits));
  if (fvalue < 0 || std::fabs(fvalue) > max_representable_int) {
    category_matched = false;
  } else {
    auto const category_value = static_cast<std::uint32_t>(fvalue);
    category_matched = (std::find(category_list.begin(), category_list.end(), category_value)
                        != category_list.end());
  }
  if (category_list_right_child) {
    return category_matched ? right_child : left_child;
  } else {
    return category_matched ? left_child : right_child;
  }
}

template <typename ThresholdT, typename LeafOutputT, typename InputT>
int EvaluateTree(Tree<ThresholdT, LeafOutputT> const& tree, Array1DView<InputT> row) {
  int node_id = 0;
  while (!tree.IsLeaf(node_id)) {
    auto const split_index = tree.SplitIndex(node_id);
    InputT const fvalue = row(split_index);
    if (std::isnan(fvalue)) {
      node_id = tree.DefaultChild(node_id);
    } else {
      if (tree.NodeType(node_id) == treelite::TreeNodeType::kCategoricalTestNode) {
        node_id = NextNodeCategorical(fvalue, tree.CategoryList(node_id),
            tree.CategoryListRightChild(node_id), tree.LeftChild(node_id),
            tree.RightChild(node_id));
      } else {
        node_id = NextNode(fvalue, tree.Threshold(node_id), tree.ComparisonOp(node_id),
            tree.LeftChild(node_id), tree.RightChild(node_id));
      }
    }
  }
  return node_id;
}

template <typename ThresholdT, typename LeafOutputT, typename InputT>
void OutputLeafVector(Model const& model, Tree<ThresholdT, LeafOutputT> const& tree, int tree_id,
    int leaf_id, std::uint64_t row_id, std::int32_t max_num_class,
    Array3DView<InputT> output_view) {
  auto leaf_out = tree.LeafVector(leaf_id);
  if (model.target_id[tree_id] == -1 && model.class_id[tree_id] == -1) {
    std::vector<std::int32_t> const expected_shape{model.num_target, max_num_class};
    TREELITE_CHECK(model.leaf_vector_shape.AsVector() == expected_shape);

    auto leaf_view = Array2DView<LeafOutputT>(leaf_out.data(), model.num_target, max_num_class);
    for (std::int32_t target_id = 0; target_id < model.num_target; ++target_id) {
      for (std::int32_t class_id = 0; class_id < model.num_class[target_id]; ++class_id) {
        output_view(row_id, target_id, class_id)
            += static_cast<InputT>(leaf_view(target_id, class_id));
      }
    }
  } else if (model.target_id[tree_id] == -1) {
    std::vector<std::int32_t> const expected_leaf_shape{model.num_target, 1};
    TREELITE_CHECK(model.leaf_vector_shape.AsVector() == expected_leaf_shape);

    auto leaf_view = Array2DView<LeafOutputT>(leaf_out.data(), model.num_target, 1);
    auto const class_id = model.class_id[tree_id];
    for (std::int32_t target_id = 0; target_id < model.num_target; ++target_id) {
      output_view(row_id, target_id, class_id) += static_cast<InputT>(leaf_view(target_id, 0));
    }
  } else if (model.class_id[tree_id] == -1) {
    std::vector<std::int32_t> const expected_leaf_shape{1, max_num_class};
    TREELITE_CHECK(model.leaf_vector_shape.AsVector() == expected_leaf_shape);

    auto leaf_view = Array2DView<LeafOutputT>(leaf_out.data(), 1, max_num_class);
    auto const target_id = model.target_id[tree_id];
    for (std::int32_t class_id = 0; class_id < model.num_class[target_id]; ++class_id) {
      output_view(row_id, target_id, class_id) += static_cast<InputT>(leaf_view(0, class_id));
    }
  } else {
    std::vector<std::int32_t> const expected_leaf_shape{1, 1};
    TREELITE_CHECK(model.leaf_vector_shape.AsVector() == expected_leaf_shape);

    auto const target_id = model.target_id[tree_id];
    auto const class_id = model.class_id[tree_id];
    output_view(row_id, target_id, class_id) += static_cast<InputT>(leaf_out[0]);
  }
}

template <typename ThresholdT, typename LeafOutputT, typename InputT>
void OutputLeafValue(Model const& model, Tree<ThresholdT, LeafOutputT> const& tree, int tree_id,
    int leaf_id, std::uint64_t row_id, Array3DView<InputT> output_view) {
  auto const target_id = model.target_id[tree_id];
  auto const class_id = model.class_id[tree_id];
  TREELITE_CHECK(target_id != -1 && class_id != -1);

  std::vector<std::int32_t> const expected_leaf_shape{1, 1};
  TREELITE_CHECK(model.leaf_vector_shape.AsVector() == expected_leaf_shape);

  output_view(row_id, target_id, class_id) += static_cast<InputT>(tree.LeafValue(leaf_id));
}

template <typename InputT, typename MatrixAccessorT>
void PredictRaw(Model const& model, MatrixAccessorT accessor, std::uint64_t num_row, InputT* output,
    detail::threading_utils::ThreadConfig const& thread_config) {
  auto max_num_class
      = *std::max_element(model.num_class.Data(), model.num_class.Data() + model.num_target);
  auto output_view = Array3DView<InputT>(output, num_row, model.num_target, max_num_class);
  std::size_t const num_tree = model.GetNumTree();
  std::fill_n(output, output_view.size(), InputT{});  // Fill with 0's
  std::visit(
      [&](auto&& concrete_model) {
        detail::threading_utils::ParallelFor(std::uint64_t(0), num_row, thread_config,
            detail::threading_utils::ParallelSchedule::Static(),
            [&](std::uint64_t row_id, int thread_id) {
              auto row = accessor.GetRow(row_id, thread_id);
              for (std::size_t tree_id = 0; tree_id < num_tree; ++tree_id) {
                auto const& tree = concrete_model.trees[tree_id];
                int const leaf_id = EvaluateTree(tree, row);
                if (tree.HasLeafVector(leaf_id)) {
                  OutputLeafVector(
                      model, tree, tree_id, leaf_id, row_id, max_num_class, output_view);
                } else {
                  OutputLeafValue(model, tree, tree_id, leaf_id, row_id, output_view);
                }
              }
            });
      },
      model.variant_);
  // Apply tree averaging
  if (model.average_tree_output) {
    std::vector<std::size_t> average_factor(model.num_target * max_num_class, 0);
    auto average_factor_view
        = Array2DView<std::size_t>(average_factor.data(), model.num_target, max_num_class);
    for (std::size_t tree_id = 0; tree_id < num_tree; ++tree_id) {
      if (model.target_id[tree_id] < 0 && model.class_id[tree_id] < 0) {
        for (std::int32_t target_id = 0; target_id < model.num_target; ++target_id) {
          for (std::int32_t class_id = 0; class_id < model.num_class[target_id]; ++class_id) {
            average_factor_view(target_id, class_id) += 1;
          }
        }
      } else if (model.target_id[tree_id] < 0) {
        std::int32_t const class_id = model.class_id[tree_id];
        for (std::int32_t target_id = 0; target_id < model.num_target; ++target_id) {
          average_factor_view(target_id, class_id) += 1;
        }
      } else if (model.class_id[tree_id] < 0) {
        std::int32_t const target_id = model.target_id[tree_id];
        for (std::int32_t class_id = 0; class_id < model.num_class[target_id]; ++class_id) {
          average_factor_view(target_id, class_id) += 1;
        }
      } else {
        average_factor_view(model.target_id[tree_id], model.class_id[tree_id]) += 1;
      }
    }
    detail::threading_utils::ParallelFor(std::uint64_t(0), num_row, thread_config,
        detail::threading_utils::ParallelSchedule::Static(), [&](std::uint64_t row_id, int) {
          for (std::int32_t target_id = 0; target_id < model.num_target; ++target_id) {
            for (std::int32_t class_id = 0; class_id < model.num_class[target_id]; ++class_id) {
              output_view(row_id, target_id, class_id)
                  /= static_cast<InputT>(average_factor_view(target_id, class_id));
            }
          }
        });
  }
  // Apply base scores
  auto base_score_view
      = CArray2DView<double>(model.base_scores.Data(), model.num_target, max_num_class);
  detail::threading_utils::ParallelFor(std::uint64_t(0), num_row, thread_config,
      detail::threading_utils::ParallelSchedule::Static(), [&](std::uint64_t row_id, int) {
        for (std::int32_t target_id = 0; target_id < model.num_target; ++target_id) {
          for (std::int32_t class_id = 0; class_id < model.num_class[target_id]; ++class_id) {
            output_view(row_id, target_id, class_id) += base_score_view(target_id, class_id);
          }
        }
      });
}

template <typename InputT>
void ApplyPostProcessor(Model const& model, InputT* output, std::uint64_t num_row,
    Configuration const& pred_config, detail::threading_utils::ThreadConfig const& thread_config) {
  auto postprocessor_func = gtil::GetPostProcessorFunc<InputT>(model.postprocessor);
  auto max_num_class
      = *std::max_element(model.num_class.Data(), model.num_class.Data() + model.num_target);
  auto output_view = Array3DView<InputT>(output, num_row, model.num_target, max_num_class);

  detail::threading_utils::ParallelFor(std::uint64_t(0), num_row, thread_config,
      detail::threading_utils::ParallelSchedule::Static(), [&](std::size_t row_id, int) {
        for (std::int32_t target_id = 0; target_id < model.num_target; ++target_id) {
          auto row = stdex::submdspan(output_view, row_id, target_id, stdex::full_extent);
          static_assert(std::is_same_v<decltype(row), Array1DView<InputT>>);
          postprocessor_func(model, model.num_class[target_id], row.data_handle());
        }
      });
}

template <typename InputT, typename MatrixAccessorT>
void PredictLeaf(Model const& model, MatrixAccessorT accessor, std::uint64_t num_row,
    InputT* output, detail::threading_utils::ThreadConfig const& thread_config) {
  auto const num_tree = model.GetNumTree();
  auto output_view = Array2DView<InputT>(output, num_row, num_tree);
  std::visit(
      [&](auto&& concrete_model) {
        std::size_t const num_tree = concrete_model.trees.size();
        detail::threading_utils::ParallelFor(std::uint64_t(0), num_row, thread_config,
            detail::threading_utils::ParallelSchedule::Static(),
            [&](std::uint64_t row_id, int thread_id) {
              auto row = accessor.GetRow(row_id, thread_id);
              for (std::size_t tree_id = 0; tree_id < num_tree; ++tree_id) {
                auto const& tree = concrete_model.trees[tree_id];
                int const leaf_id = EvaluateTree(tree, row);
                output_view(row_id, tree_id) = leaf_id;
              }
            });
      },
      model.variant_);
}

template <typename InputT, typename MatrixAccessorT>
void PredictScoreByTree(Model const& model, MatrixAccessorT accessor, std::uint64_t num_row,
    InputT* output, detail::threading_utils::ThreadConfig const& thread_config) {
  auto const num_tree = model.GetNumTree();
  auto max_num_class
      = *std::max_element(model.num_class.Data(), model.num_class.Data() + model.num_target);
  auto output_view = Array3DView<InputT>(
      output, num_row, num_tree, model.leaf_vector_shape[0] * model.leaf_vector_shape[1]);
  std::fill_n(output, output_view.size(), InputT{});  // Fill with 0's
  std::visit(
      [&](auto&& concrete_model) {
        std::size_t const num_tree = concrete_model.trees.size();
        detail::threading_utils::ParallelFor(std::uint64_t(0), num_row, thread_config,
            detail::threading_utils::ParallelSchedule::Static(),
            [&](std::uint64_t row_id, int thread_id) {
              auto row = accessor.GetRow(row_id, thread_id);
              for (std::size_t tree_id = 0; tree_id < num_tree; ++tree_id) {
                auto const& tree = concrete_model.trees[tree_id];
                int const leaf_id = EvaluateTree(tree, row);
                if (tree.HasLeafVector(leaf_id)) {
                  auto const leafvec = tree.LeafVector(leaf_id);
                  for (std::size_t i = 0; i < leafvec.size(); ++i) {
                    output_view(row_id, tree_id, i) = leafvec[i];
                  }
                } else {
                  output_view(row_id, tree_id, 0) = tree.LeafValue(leaf_id);
                }
              }
            });
      },
      model.variant_);
}

template <typename InputT, typename MatrixAccessorT>
void PredictImpl(Model const& model, MatrixAccessorT accessor, std::uint64_t num_row,
    InputT* output, Configuration const& config,
    detail::threading_utils::ThreadConfig const& thread_config) {
  if (config.pred_kind == PredictKind::kPredictDefault) {
    PredictRaw(model, accessor, num_row, output, thread_config);
    ApplyPostProcessor(model, output, num_row, config, thread_config);
  } else if (config.pred_kind == PredictKind::kPredictRaw) {
    PredictRaw(model, accessor, num_row, output, thread_config);
  } else if (config.pred_kind == PredictKind::kPredictLeafID) {
    PredictLeaf(model, accessor, num_row, output, thread_config);
  } else if (config.pred_kind == PredictKind::kPredictPerTree) {
    PredictScoreByTree(model, accessor, num_row, output, thread_config);
  } else {
    TREELITE_LOG(FATAL) << "Not implemented";
  }
}

template <typename InputT>
void Predict(Model const& model, InputT const* input, std::uint64_t num_row, InputT* output,
    Configuration const& config) {
  auto thread_config = detail::threading_utils::ThreadConfig(config.nthread);
  auto accessor = DenseMatrixAccessor(input, num_row, model.num_feature);
  PredictImpl(model, accessor, num_row, output, config, thread_config);
}

template <typename InputT>
void PredictSparse(Model const& model, InputT const* data, std::uint64_t const* col_ind,
    std::uint64_t const* row_ptr, std::uint64_t num_row, InputT* output,
    Configuration const& config) {
  auto thread_config = detail::threading_utils::ThreadConfig(config.nthread);
  auto accessor
      = SparseMatrixAccessor(data, col_ind, row_ptr, num_row, model.num_feature, thread_config);
  PredictImpl(model, accessor, num_row, output, config, thread_config);
}

template void Predict<float>(
    Model const&, float const*, std::uint64_t, float*, Configuration const&);
template void Predict<double>(
    Model const&, double const*, std::uint64_t, double*, Configuration const&);
template void PredictSparse<float>(Model const&, float const*, std::uint64_t const*,
    std::uint64_t const*, std::uint64_t, float*, Configuration const&);
template void PredictSparse<double>(Model const&, double const*, std::uint64_t const*,
    std::uint64_t const*, std::uint64_t, double*, Configuration const&);

}  // namespace treelite::gtil
