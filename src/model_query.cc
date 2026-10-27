/*!
 * Copyright (c) 2024-2026 by Contributors
 * \file model_query.cc
 * \author Hyunsu Cho
 * \brief Methods for querying various properties of tree models
 */
#include <algorithm>  // for max
#include <cstdint>
#include <stack>
#include <utility>
#include <variant>
#include <vector>

#include <treelite/tree.h>

namespace {
template <typename ThresholdType, typename LeafOutputType>
std::uint32_t GetDepth(treelite::Tree<ThresholdType, LeafOutputType> const& tree) {
  // Visit all trees nodes in depth-first order
  using NodeDepth = std::pair<std::int32_t, std::uint32_t>;
  std::stack<NodeDepth> st;
  st.push({0, 0u});

  std::uint32_t max_depth = 0;
  while (!st.empty()) {
    auto node_depth = st.top();
    st.pop();
    if (tree.IsLeaf(node_depth.first)) {
      max_depth = std::max(max_depth, node_depth.second);
    } else {
      st.push({tree.LeftChild(node_depth.first), node_depth.second + 1});
      st.push({tree.RightChild(node_depth.first), node_depth.second + 1});
    }
  }
  return max_depth;
}

}  // anonymous namespace

namespace treelite {
std::vector<std::uint32_t> Model::GetTreeDepth() const {
  return std::visit(
      [](auto&& concrete_model) {
        std::vector<std::uint32_t> depth;
        depth.reserve(concrete_model.trees.size());
        for (auto const& tree : concrete_model.trees) {
          depth.push_back(GetDepth(tree));
        }
        return depth;
      },
      variant_);
}
}  // namespace treelite
