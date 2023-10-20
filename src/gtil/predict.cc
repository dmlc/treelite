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
        output_view(target_id, row_id, class_id) += leaf_view(target_id, class_id);
      }
    }
  } else if (model.target_id[tree_id] == -1) {
    std::vector<std::int32_t> const expected_leaf_shape{model.num_target, 1};
    TREELITE_CHECK(model.leaf_vector_shape.AsVector() == expected_leaf_shape);

    auto leaf_view = Array2DView<LeafOutputT>(leaf_out.data(), model.num_target, 1);
    auto const class_id = model.class_id[tree_id];
    for (std::int32_t target_id = 0; target_id < model.num_target; ++target_id) {
      output_view(target_id, row_id, class_id) += leaf_view(target_id, 0);
    }
  } else if (model.class_id[tree_id] == -1) {
    std::vector<std::int32_t> const expected_leaf_shape{1, max_num_class};
    TREELITE_CHECK(model.leaf_vector_shape.AsVector() == expected_leaf_shape);

    auto leaf_view = Array2DView<LeafOutputT>(leaf_out.data(), 1, max_num_class);
    auto const target_id = model.target_id[tree_id];
    for (std::int32_t class_id = 0; class_id < model.num_class[target_id]; ++class_id) {
      output_view(target_id, row_id, class_id) += leaf_view(0, class_id);
    }
  } else {
    std::vector<std::int32_t> const expected_leaf_shape{1, 1};
    TREELITE_CHECK(model.leaf_vector_shape.AsVector() == expected_leaf_shape);

    auto const target_id = model.target_id[tree_id];
    auto const class_id = model.class_id[tree_id];
    output_view(target_id, row_id, class_id) += leaf_out[0];
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

  output_view(target_id, row_id, class_id) += tree.LeafValue(leaf_id);
}

template <typename InputT>
void PredictRaw(Model const& model, InputT const* input, std::uint64_t num_row, InputT* output,
    detail::threading_utils::ThreadConfig const& thread_config) {
  auto input_view = CArray2DView<InputT>(input, num_row, model.num_feature);
  auto max_num_class
      = *std::max_element(model.num_class.Data(), model.num_class.Data() + model.num_target);
  auto output_view = Array3DView<InputT>(output, model.num_target, num_row, max_num_class);
  std::size_t const num_tree = model.GetNumTree();
  std::fill_n(output, output_view.size(), InputT{});  // Fill with 0's
  std::visit(
      [&](auto&& concrete_model) {
        detail::threading_utils::ParallelFor(std::uint64_t(0), num_row, thread_config,
            detail::threading_utils::ParallelSchedule::Static(), [&](std::uint64_t row_id, int) {
              auto row = stdex::submdspan(input_view, row_id, stdex::full_extent);
              static_assert(std::is_same_v<decltype(row), CArray1DView<InputT>>);
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
    for (std::int32_t target_id = 0; target_id < model.num_target; ++target_id) {
      detail::threading_utils::ParallelFor(std::uint64_t(0), num_row, thread_config,
          detail::threading_utils::ParallelSchedule::Static(), [&](std::uint64_t row_id, int) {
            for (std::int32_t class_id = 0; class_id < model.num_class[target_id]; ++class_id) {
              output_view(target_id, row_id, class_id)
                  /= static_cast<InputT>(average_factor_view(target_id, class_id));
            }
          });
    }
  }
  // Apply base scores
  auto base_score_view
      = CArray2DView<double>(model.base_scores.Data(), model.num_target, max_num_class);
  for (std::int32_t target_id = 0; target_id < model.num_target; ++target_id) {
    detail::threading_utils::ParallelFor(std::uint64_t(0), num_row, thread_config,
        detail::threading_utils::ParallelSchedule::Static(), [&](std::uint64_t row_id, int) {
          for (std::int32_t class_id = 0; class_id < model.num_class[target_id]; ++class_id) {
            output_view(target_id, row_id, class_id) += base_score_view(target_id, class_id);
          }
        });
  }
}

template <typename InputT>
void ApplyPostProcessor(Model const& model, InputT* output, std::uint64_t num_row,
    Configuration const& pred_config, detail::threading_utils::ThreadConfig const& thread_config) {
  auto postprocessor_func = gtil::GetPostProcessorFunc<InputT>(model.postprocessor);
  auto max_num_class
      = *std::max_element(model.num_class.Data(), model.num_class.Data() + model.num_target);
  auto output_view = Array3DView<InputT>(output, model.num_target, num_row, max_num_class);

  for (std::int32_t target_id = 0; target_id < model.num_target; ++target_id) {
    std::int32_t const num_class = model.num_class[target_id];
    detail::threading_utils::ParallelFor(std::size_t(0), num_row, thread_config,
        detail::threading_utils::ParallelSchedule::Static(), [&](std::size_t row_id, int) {
          auto row = stdex::submdspan(output_view, target_id, row_id, stdex::full_extent);
          static_assert(std::is_same_v<decltype(row), Array1DView<InputT>>);
          postprocessor_func(model, num_class, row.data_handle());
        });
  }
}

template <typename InputT>
void PredictLeaf(Model const& model, InputT const* input, std::uint64_t num_row, InputT* output,
    detail::threading_utils::ThreadConfig const& thread_config) {
  auto const num_tree = model.GetNumTree();
  auto input_view = CArray2DView<InputT>(input, num_row, model.num_feature);
  auto output_view = Array2DView<InputT>(output, num_row, num_tree);
  std::visit(
      [&](auto&& concrete_model) {
        std::size_t const num_tree = concrete_model.trees.size();
        detail::threading_utils::ParallelFor(std::uint64_t(0), num_row, thread_config,
            detail::threading_utils::ParallelSchedule::Static(), [&](std::uint64_t row_id, int) {
              auto row = stdex::submdspan(input_view, row_id, stdex::full_extent);
              static_assert(std::is_same_v<decltype(row), CArray1DView<InputT>>);
              for (std::size_t tree_id = 0; tree_id < num_tree; ++tree_id) {
                auto const& tree = concrete_model.trees[tree_id];
                int const leaf_id = EvaluateTree(tree, row);
                output_view(row_id, tree_id) = leaf_id;
              }
            });
      },
      model.variant_);
}

template <typename InputT>
void PredictScoreByTree(Model const& model, InputT const* input, std::uint64_t num_row,
    InputT* output, detail::threading_utils::ThreadConfig const& thread_config) {
  auto input_view = CArray2DView<InputT>(input, num_row, model.num_feature);
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
            detail::threading_utils::ParallelSchedule::Static(), [&](std::uint64_t row_id, int) {
              auto row = stdex::submdspan(input_view, row_id, stdex::full_extent);
              static_assert(std::is_same_v<decltype(row), CArray1DView<InputT>>);
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

template <typename InputT>
void Predict(Model const& model, InputT const* input, std::uint64_t num_row, InputT* output,
    Configuration const& config) {
  TypeInfo leaf_output_type = model.GetLeafOutputType();
  TypeInfo input_type = TypeInfoFromType<InputT>();
  if (leaf_output_type != input_type) {
    std::string expected = TypeInfoToString(leaf_output_type);
    std::string got = TypeInfoToString(input_type);
    if (got == "invalid") {
      got = typeid(InputT).name();
    }
    TREELITE_LOG(FATAL) << "Incorrect input type passed to GTIL predict(). "
                        << "Expected: " << expected << ", Got: " << got;
  }
  auto thread_config = detail::threading_utils::ThreadConfig(config.nthread);
  if (config.pred_kind == PredictKind::kPredictDefault) {
    PredictRaw(model, input, num_row, output, thread_config);
    ApplyPostProcessor(model, output, num_row, config, thread_config);
  } else if (config.pred_kind == PredictKind::kPredictRaw) {
    PredictRaw(model, input, num_row, output, thread_config);
  } else if (config.pred_kind == PredictKind::kPredictLeafID) {
    PredictLeaf(model, input, num_row, output, thread_config);
  } else if (config.pred_kind == PredictKind::kPredictPerTree) {
    PredictScoreByTree(model, input, num_row, output, thread_config);
  } else {
    TREELITE_LOG(FATAL) << "Not implemented";
  }
}

template void Predict<float>(
    Model const&, float const*, std::uint64_t, float*, Configuration const&);
template void Predict<double>(
    Model const&, double const*, std::uint64_t, double*, Configuration const&);

}  // namespace treelite::gtil
