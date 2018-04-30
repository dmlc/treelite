/*!
 * Copyright (c) 2018 by Contributors
 * \file main_template.h
 * \author Philip Cho
 * \brief template for main function
 */

#ifndef TREELITE_COMPILER_NATIVE_MAIN_TEMPLATE_H_
#define TREELITE_COMPILER_NATIVE_MAIN_TEMPLATE_H_

namespace treelite {
namespace compiler {
namespace native {

const char* main_start_template =
R"TREELITETEMPLATE(
#include "header.h"

{get_num_output_group_function_signature} {{
  return {num_output_group};
}}

{get_num_feature_function_signature} {{
  return {num_feature};
}}

{pred_transform_function}
{predict_function_signature} {{
)TREELITETEMPLATE";

const char* main_end_multiclass_template =
R"TREELITETEMPLATE(
  for (int i = 0; i < {num_output_group}; ++i) {{
    result[i] = sum[i]{optional_average_field} + (float)({global_bias});
  }}
  if (!pred_margin) {{
    return pred_transform(result);
  }} else {{
    return {num_output_group};
  }}
}}
)TREELITETEMPLATE";  // only for multiclass classification

const char* main_end_template =
R"TREELITETEMPLATE(
  sum = sum{optional_average_field} + (float)({global_bias});
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
