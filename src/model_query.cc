/*!
 * Copyright (c) 2024 by Contributors
 * \file model_query.cc
 * \author Hyunsu Cho
 * \brief Methods for querying various properties of tree models
 */
#include <cstdint>
#include <queue>
#include <utility>
#include <variant>
#include <vector>

#include <treelite/tree.h>

namespace {

template <typename ThresholdType, typename LeafOutputType>
std::uint32_t GetDepth(treelite::Tree<ThresholdType, LeafOutputType> const& tree) {
  // Visit all trees nodes in breath-first order
  std::queue<std::pair<int, std::uint32_t>> q;
  // {current node visiting, depth level of the node}
  q.emplace(0, 0);
  std::uint32_t tree_depth = 0;
  while (!q.empty()) {
    auto [node_id, level] = q.front();
    q.pop();
    if (tree.IsLeaf(node_id)) {
      tree_depth = std::max(tree_depth, level);
    } else {
      q.emplace(tree.LeftChild(node_id), level + 1);
      q.emplace(tree.RightChild(node_id), level + 1);
    }
  }
  return tree_depth;
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
