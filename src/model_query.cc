/*!
 * Copyright (c) 2024 by Contributors
 * \file model_query.cc
 * \author Hyunsu Cho
 * \brief Methods for querying various properties of tree models
 */
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
  std::stack<int> st;
  st.push(0);
  std::uint32_t max_depth = 0;
  std::uint32_t depth = 1;
  while (!st.empty()) {
    int node_id = st.top();
    st.pop();
    if (tree.IsLeaf(node_id)) {
      --depth;
    } else {
      st.push(tree.LeftChild(node_id));
      st.push(tree.RightChild(node_id));
      ++depth;
    }
    max_depth = std::max(max_depth, depth);
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
