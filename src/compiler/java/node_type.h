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
