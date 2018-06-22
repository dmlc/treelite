/*!
 * Copyright (c) 2018 by Contributors
 * \file inference_engine_interface.h
 * \author James Liu
 * \brief template for Java predictor interface
 */

#ifndef TREELITE_COMPILER_JAVA_INFERENCE_ENGINE_H_
#define TREELITE_COMPILER_JAVA_INFERENCE_ENGINE_H_

namespace treelite {
namespace compiler {
namespace java {

const char* inference_engine_interface =
R"TREELITETEMPLATE(
package ml.dmlc.treelite4j;

public interface InferenceEngine {
    public int get_num_feature();
    public float predict(Data[] data, boolean pred_margin);
}
)TREELITETEMPLATE";

}  // namespace java
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_JAVA_INFERENCE_ENGINE_H_
