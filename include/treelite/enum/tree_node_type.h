/*!
 * Copyright (c) 2023 by Contributors
 * \file tree_node_type.h
 * \brief Define enum type NodeType
 * \author Hyunsu Cho
 */

#ifndef TREELITE_ENUM_TREE_NODE_TYPE_H_
#define TREELITE_ENUM_TREE_NODE_TYPE_H_

#include <cstdint>
#include <string>

namespace treelite {

/*! \brief Tree node type */
enum class TreeNodeType : std::int8_t {
  kLeafNode = 0,
  kNumericalTestNode = 1,
  kCategoricalTestNode = 2
};

/*! \brief Get string representation of TreeNodeType */
std::string TreeNodeTypeToString(TreeNodeType type);

/*! \brief Get NodeType from string */
TreeNodeType TreeNodeTypeFromString(std::string const& name);

}  // namespace treelite

#endif  // TREELITE_ENUM_TREE_NODE_TYPE_H_
