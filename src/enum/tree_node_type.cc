/*!
 * Copyright (c) 2023 by Contributors
 * \file tree_node_type.cc
 * \author Hyunsu Cho
 * \brief Utilities for NodeType enum
 */

#include <treelite/enum/tree_node_type.h>
#include <treelite/logging.h>

#include <string>

namespace treelite {

std::string TreeNodeTypeToString(TreeNodeType type) {
  switch (type) {
  case TreeNodeType::kLeafNode:
    return "leaf_node";
  case TreeNodeType::kNumericalTestNode:
    return "numerical_test_node";
  case TreeNodeType::kCategoricalTestNode:
    return "categorical_test_node";
  default:
    return "";
  }
}

TreeNodeType TreeNodeTypeFromString(std::string const& name) {
  if (name == "leaf_node") {
    return TreeNodeType::kLeafNode;
  } else if (name == "numerical_test_node") {
    return TreeNodeType::kNumericalTestNode;
  } else if (name == "categorical_test_node") {
    return TreeNodeType::kCategoricalTestNode;
  } else {
    TREELITE_LOG(FATAL) << "Unknown split type: " << name;
    return TreeNodeType::kLeafNode;
  }
}

}  // namespace treelite
