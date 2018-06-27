/*!
 * Copyright (c) 2017 by Contributors
 * \file pred_transform.h
 * \author Philip Cho
 * \brief template for pred_transform() function in generated Java code
 */
#ifndef TREELITE_COMPILER_JAVA_PRED_TRANSFORM_H_
#define TREELITE_COMPILER_JAVA_PRED_TRANSFORM_H_

#include <treelite/common.h>
#include <string>

namespace treelite {
namespace compiler {
namespace java {
namespace pred_transform {

inline std::string identity(const Model& model) {
  return
  "  private static float pred_transform(float margin) {\n"
  "    return margin;\n"
  "  }\n";
}

inline std::string sigmoid(const Model& model) {
  const float alpha = model.param.sigmoid_alpha;
  CHECK_GT(alpha, 0.0f) << "sigmoid: alpha must be strictly positive";
  std::ostringstream oss;
  oss << "  private static float pred_transform(float margin) {\n"
      << "    final double alpha = " << common::ToString(alpha) << ";\n"
      << "    return (float)(1.0 / (1.0 + Math.exp(-alpha * margin)));\n"
      << "  }\n";
  return oss.str();
}

inline std::string exponential(const Model& model) {
  return
  "  private static float pred_transform(float margin) {\n"
  "    return (float)Math.exp(margin);\n"
  "  }\n";
}

inline std::string logarithm_one_plus_exp(const Model& model) {
  return
  "  private static float pred_transform(float margin) {\n"
  "    return (float)Math.log1p(Math.exp(margin));\n"
  "  }\n";
}

inline std::string identity_multiclass(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "identity_multiclass: model is not a proper multi-class classifier";
  const int num_class = model.num_output_group;
  std::ostringstream oss;
  oss << "  private static long pred_transform(float[] pred) {\n"
      << "    final long num_class = " << num_class << ";\n"
      << "    return num_class;\n"
      << "  }\n";
  return oss.str();
}

inline std::string max_index(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "max_index: model is not a proper multi-class classifier";
  const int num_class = model.num_output_group;
  std::ostringstream oss;
  oss << "  private static long pred_transform(float[] pred) {\n"
      << "    final int num_class = " << num_class << ";\n"
      << "    int max_index = 0;\n"
      << "    float max_margin = pred[0];\n"
      << "    for (int k = 1; k < num_class; ++k) {\n"
      << "      if (pred[k] > max_margin) {\n"
      << "        max_margin = pred[k];\n"
      << "        max_index = k;\n"
      << "      }\n"
      << "    }\n"
      << "    pred[0] = (float)max_index;\n"
      << "    return 1;\n"
      << "  }\n";
  return oss.str();
}

inline std::string softmax(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "softmax: model is not a proper multi-class classifier";
  const int num_class = model.num_output_group;
  std::ostringstream oss;
  oss << "  private static long pred_transform(float[] pred) {\n"
      << "    final int num_class = " << num_class << ";\n"
      << "    float max_margin = pred[0];\n"
      << "    double norm_const = 0.0;\n"
      << "    double t;\n"
      << "    for (int k = 1; k < num_class; ++k) {\n"
      << "      if (pred[k] > max_margin) {\n"
      << "        max_margin = pred[k];\n"
      << "      }\n"
      << "    }\n"
      << "    for (int k = 0; k < num_class; ++k) {\n"
      << "      t = Math.exp(pred[k] - max_margin);\n"
      << "      norm_const += t;\n"
      << "      pred[k] = (float)t;\n"
      << "    }\n"
      << "    for (int k = 0; k < num_class; ++k) {\n"
      << "      pred[k] /= (float)norm_const;\n"
      << "    }\n"
      << "    return (long)num_class;\n"
      << "  }\n";
  return oss.str();
}

inline std::string multiclass_ova(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "multiclass_ova: model is not a proper multi-class classifier";
  const int num_class = model.num_output_group;
  const float alpha = model.param.sigmoid_alpha;
  CHECK_GT(alpha, 0.0f) << "multiclass_ova: alpha must be strictly positive";
  std::ostringstream oss;
  oss << "  private static long pred_transform(float[] pred) {\n"
      << "    final float alpha = (float)" << common::ToString(alpha) << ";\n"
      << "    final int num_class = " << num_class << ";\n"
      << "    for (int k = 0; k < num_class; ++k) {\n"
      << "      pred[k] = (float)(1.0 / (1.0 + Math.exp(-alpha * pred[k])));\n"
      << "    }\n"
      << "    return (long)num_class;\n"
      << "  }\n";
  return oss.str();
}

}  // namespace pred_transform
}  // namespace java
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_JAVA_PRED_TRANSFORM_H_
