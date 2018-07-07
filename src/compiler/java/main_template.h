const char* main_start_template =
R"TREELITETEMPLATE(
package {java_package};

import java.lang.Math;
import javolution.context.LogContext;
import javolution.context.LogContext.Level;
import ml.dmlc.treelite4j.InferenceEngine;
import ml.dmlc.treelite4j.Data;

public class Main {{
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
  for (int i = 0; i < entry.length; i++) {{
    data[i] = (Entry) entry[i];
  }}
)TREELITETEMPLATE";

const char* main_end_multiclass_template =
R"TREELITETEMPLATE(
    for (int i = 0; i < {num_output_group}; ++i) {{
      sum[i] = sum[i]{optional_average_field} + (float)({global_bias});
    }}
    if (!pred_margin) {{
      return pred_transform(sum);
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
