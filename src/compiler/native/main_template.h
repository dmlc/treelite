/*!
 * Copyright (c) 2018-2020 by Contributors
 * \file main_template.h
 * \author Hyunsu Cho
 * \brief template for main function
 */

#ifndef TREELITE_COMPILER_NATIVE_MAIN_TEMPLATE_H_
#define TREELITE_COMPILER_NATIVE_MAIN_TEMPLATE_H_

namespace treelite {
namespace compiler {
namespace native {

const char* const query_functions_definition_template =
R"TREELITETEMPLATE(
size_t get_num_output_group(void) {{
  return {num_output_group};
}}

size_t get_num_feature(void) {{
  return {num_feature};
}}

const char* get_pred_transform(void) {{
  return "{pred_transform}";
}}

float get_sigmoid_alpha(void) {{
  return {sigmoid_alpha};
}}

float get_global_bias(void) {{
  return {global_bias};
}}

const char* get_threshold_type(void) {{
  return "{threshold_type_str}";
}}

const char* get_leaf_output_type(void) {{
  return "{leaf_output_type_str}";
}}
)TREELITETEMPLATE";

const char* const main_start_template =
R"TREELITETEMPLATE(
#include "header.h"

{array_is_categorical};

{query_functions_definition}

{pred_transform_function}
{predict_function_signature} {{
)TREELITETEMPLATE";

const char* const main_end_multiclass_template =
R"TREELITETEMPLATE(
  for (int i = 0; i < {num_output_group}; ++i) {{
    result[i] = sum[i]{optional_average_field} + ({leaf_output_type})({global_bias});
  }}
  if (!pred_margin) {{
    return pred_transform(result);
  }} else {{
    return {num_output_group};
  }}
}}
)TREELITETEMPLATE";  // only for multiclass classification

const char* const main_end_template =
R"TREELITETEMPLATE(
  sum = sum{optional_average_field} + ({leaf_output_type})({global_bias});
  if (!pred_margin) {{
    return pred_transform(sum);
  }} else {{
    return sum;
  }}
}}
)TREELITETEMPLATE";

}  // namespace native
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_NATIVE_MAIN_TEMPLATE_H_
