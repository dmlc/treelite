/*!
 * Copyright (c) 2018 by Contributors
 * \file node_type.h
 * \author Philip Cho
 * \brief template for Node type (used for code folding)
 */

#ifndef TREELITE_COMPILER_JAVA_NODE_TYPE_H_
#define TREELITE_COMPILER_JAVA_NODE_TYPE_H_

namespace treelite {
namespace compiler {
namespace java {

const char* node_type_template =
R"TREELITETEMPLATE(
package {java_package};

public class Node {{
  public boolean default_left;
  public int split_index;
  public {threshold_type} threshold;
  public int left_child;
  public int right_child;

  public Node(boolean default_left, int split_index, {threshold_type} threshold,
              int left_child, int right_child) {{
    this.default_left = default_left;
    this.split_index = split_index;
    this.threshold = threshold;
    this.left_child = left_child;
    this.right_child = right_child;
  }}
}}
)TREELITETEMPLATE";

}  // namespace java
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_JAVA_NODE_TYPE_H_
