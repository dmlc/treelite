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

const char* const main_start_template =
R"TREELITETEMPLATE(
#include "header.h"

{array_is_categorical};

{get_num_output_group_function_signature} {{
  return {num_output_group};
}}

{get_num_feature_function_signature} {{
  return {num_feature};
}}

{get_pred_transform_function_signature} {{
  return "{pred_transform}";
}}

{get_sigmoid_alpha_function_signature} {{
  return {sigmoid_alpha};
}}

{get_global_bias_function_signature} {{
  return {global_bias};
}}

{get_threshold_type_signature} {{
  return "{threshold_type_str}";
}}

{get_leaf_output_type_signature} {{
  return "{leaf_output_type_str}";
}}

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
