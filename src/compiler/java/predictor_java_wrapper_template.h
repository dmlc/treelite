/*!
 * Copyright (c) 2017 by Contributors
 * \file predictor_java_wrapper_template.h
 * \author Philip Cho
 * \brief template for wrapper code for generated Java code
 */

#ifndef TREELITE_COMPILER_JAVA_PREDICTOR_JAVA_WRAPPER_TEMPLATE_H_
#define TREELITE_COMPILER_JAVA_PREDICTOR_JAVA_WRAPPER_TEMPLATE_H_

namespace treelite {
namespace compiler {
namespace java {

const char* predictor_java_wrapper_template =
R"TREELITETEMPLATE(
package {java_package};

import ml.dmlc.treelite4j.InferenceEngine;
import ml.dmlc.treelite4j.Data;
public class PredictorJavaWrapper implements InferenceEngine {{
  Main main = new Main();
  public int getNumOutputGroup() {{
    return main.get_num_output_group();
  }}

  public int getNumFeature() {{
    return main.get_num_feature();
  }}

  public float[] predict(Data[] inst, boolean pred_margin) {{
{pred_logic}
  }}
}}
)TREELITETEMPLATE";

const char* pred_logic =
R"TREELITETEMPLATE(
    float[] scores = new float[1];
    scores[0] = main.predict(inst, pred_margin);
    return scores;
)TREELITETEMPLATE";

const char* pred_logic_multiclass =
R"TREELITETEMPLATE(
    int num_output_group = getNumOutputGroup();
    int ret = main.predict_multiclass(inst, pred_margin, scores);
    if (ret != num_output_group) {
      float[] new_scores = new float[ret];
      for (int i = 0; i < ret; ++i) {
        new_scores[i] = scores[i];
      }
      return ret;
    } else {
      return scores;
    }
)TREELITETEMPLATE";

}  // namespace java
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_JAVA_PREDICTOR_JAVA_WRAPPER_TEMPLATE_H_
