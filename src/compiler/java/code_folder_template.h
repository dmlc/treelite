const char* eval_loop_template =
R"TREELITETEMPLATE(
nid = 0;
while (nid >= 0) {{  /* negative nid implies leaf */
  fid = {constant_table}.{node_array_name}[nid].split_index;
  if (data[fid].missing.get() == -1) {{
    cond = {constant_table}.{node_array_name}[nid].default_left;
  }} else if (Main.is_categorical[fid]) {{
    tmp = (int)data[fid].fvalue.get();
    cond = (({constant_table}.{cat_bitmap_name}[{constant_table}.{cat_begin_name}[nid] + tmp / 64] >>> (tmp % 64)) & 1) == 1;
  }} else {{
    cond = (data[fid].{data_field}.get() {comp_op} {constant_table}.{node_array_name}[nid].threshold);
  }}
  nid = cond ? {constant_table}.{node_array_name}[nid].left_child : {constant_table}.{node_array_name}[nid].right_child;
}}

{output_switch_statement}
)TREELITETEMPLATE";

const char* code_folder_arrays_template =
R"TREELITETEMPLATE(
public static final Node[] {node_array_name} = {{
{array_nodes}
}};

public static final long[] {cat_bitmap_name} = {{
{array_cat_bitmap}
}};

public static final int[] {cat_begin_name} = {{
{array_cat_begin}
}};
)TREELITETEMPLATE";
