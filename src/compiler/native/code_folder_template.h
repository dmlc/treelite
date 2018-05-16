const char* eval_loop_template =
R"TREELITETEMPLATE(
nid = 0;
while (nid >= 0) {{  /* negative nid implies leaf */
  fid = {node_array_name}[nid].split_index;
  if (data[fid].missing == -1) {{
    cond = {node_array_name}[nid].default_left;
  }} else if (is_categorical[fid]) {{
    tmp = (unsigned int)data[fid].fvalue;
    cond = ({cat_bitmap_name}[{cat_begin_name}[nid] + tmp / 64] >> (tmp % 64)) & 1;
  }} else {{
    cond = (data[fid].{data_field} {comp_op} {node_array_name}[nid].threshold);
  }}
  nid = cond ? {node_array_name}[nid].left_child : {node_array_name}[nid].right_child;
}}

{output_switch_statement}
)TREELITETEMPLATE";

const char* code_folder_arrays_template =
R"TREELITETEMPLATE(
const struct Node {node_array_name}[] = {{
{array_nodes}
}};

const uint64_t {cat_bitmap_name}[] = {{
{array_cat_bitmap}
}};

const size_t {cat_begin_name}[] = {{
{array_cat_begin}
}};
)TREELITETEMPLATE";

const char* code_folder_arrays_declaration_template =
R"TREELITETEMPLATE(
extern const struct Node {node_array_name}[];
extern const uint64_t {cat_bitmap_name}[];
extern const size_t {cat_begin_name}[];
)TREELITETEMPLATE";
