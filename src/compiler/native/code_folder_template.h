/*!
 * Copyright (c) 2018-2020 by Contributors
 * \file code_folder_template.h
 * \author Hyunsu Cho
 * \brief template for evaluation logic for folded code
 */

#ifndef TREELITE_COMPILER_NATIVE_CODE_FOLDER_TEMPLATE_H_
#define TREELITE_COMPILER_NATIVE_CODE_FOLDER_TEMPLATE_H_

namespace treelite {
namespace compiler {
namespace native {

const char* const eval_loop_template =
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

const char* const eval_loop_template_without_categorical_feature =
R"TREELITETEMPLATE(
nid = 0;
while (nid >= 0) {{  /* negative nid implies leaf */
  fid = {node_array_name}[nid].split_index;
  if (data[fid].missing == -1) {{
    cond = {node_array_name}[nid].default_left;
  }} else {{
    cond = (data[fid].{data_field} {comp_op} {node_array_name}[nid].threshold);
  }}
  nid = cond ? {node_array_name}[nid].left_child : {node_array_name}[nid].right_child;
}}

{output_switch_statement}
)TREELITETEMPLATE";

}  // namespace native
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_NATIVE_CODE_FOLDER_TEMPLATE_H_
