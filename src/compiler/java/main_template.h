/*!
 * Copyright (c) 2018 by Contributors
 * \file main_template.h
 * \author Philip Cho
 * \brief template for main function
 */

#ifndef TREELITE_COMPILER_JAVA_MAIN_TEMPLATE_H_
#define TREELITE_COMPILER_JAVA_MAIN_TEMPLATE_H_

namespace treelite {
namespace compiler {
namespace java {

const char* main_start_template =
R"TREELITETEMPLATE(
package {java_package};

import java.lang.Math;
import javolution.context.LogContext;
import javolution.context.LogContext.Level;
import ml.dmlc.treelite4j.InferenceEngine;
import ml.dmlc.treelite4j.Data;

public class Main implements InferenceEngine {{
  static {{
    LogContext ctx = LogContext.enter();
    ctx.setLevel(Level.INFO);
  }}
  public int get_num_output_group() {{
    return {num_output_group};
  }}

  public int get_num_feature() {{
    return {num_feature};
  }}

{pred_transform_function}
  {predict_function_signature} {{
  Entry[] data = new Entry[entry.length];
  for (int i=0; i < entry.length; i++)
    data[i] = (Entry) entry[i];
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
  {main_footer}
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
  {main_footer}
}}
)TREELITETEMPLATE";

}  // namespace java
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_JAVA_MAIN_TEMPLATE_H_
