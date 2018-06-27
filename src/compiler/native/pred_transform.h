/*!
 * Copyright (c) 2017 by Contributors
 * \file pred_transform.h
 * \author Philip Cho
 * \brief template for pred_transform() function in generated C code
 */

#ifndef TREELITE_COMPILER_NATIVE_PRED_TRANSFORM_H_
#define TREELITE_COMPILER_NATIVE_PRED_TRANSFORM_H_

#include <treelite/common.h>
#include <string>

namespace treelite {
namespace compiler {
namespace native {
namespace pred_transform {

inline std::string identity(const Model& model) {
  return
  "static inline float pred_transform(float margin) {\n"
  "  return margin;\n"
  "}\n";
}

inline std::string sigmoid(const Model& model) {
  const float alpha = model.param.sigmoid_alpha;
  CHECK_GT(alpha, 0.0f) << "sigmoid: alpha must be strictly positive";
  std::ostringstream oss;
  oss << "static inline float pred_transform(float margin) {\n"
      << "  const float alpha = (float)" << common::ToString(alpha) << ";\n"
      << "  return 1.0f / (1 + expf(-alpha * margin));\n"
      << "}\n";
  return oss.str();
}

inline std::string exponential(const Model& model) {
  return
  "static inline float pred_transform(float margin) {\n"
  "  return expf(margin);\n"
  "}\n";
}

inline std::string logarithm_one_plus_exp(const Model& model) {
  return
  "static inline float pred_transform(float margin) {\n"
  "  return log1pf(expf(margin));\n"
  "}\n";
}

inline std::string identity_multiclass(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "identity_multiclass: model is not a proper multi-class classifier";
  const int num_class = model.num_output_group;
  std::ostringstream oss;
  oss << "static inline size_t pred_transform(float* pred) {\n"
      << "  const size_t num_class = " << num_class << ";\n"
      << "  return num_class;\n"
      << "}\n";
  return oss.str();
}

inline std::string max_index(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "max_index: model is not a proper multi-class classifier";
  const int num_class = model.num_output_group;
  std::ostringstream oss;
  oss << "static inline size_t pred_transform(float* pred) {\n"
      << "  const int num_class = " << num_class << ";\n"
      << "  int max_index = 0;\n"
      << "  float max_margin = pred[0];\n"
      << "  for (int k = 1; k < num_class; ++k) {\n"
      << "    if (pred[k] > max_margin) {\n"
      << "      max_margin = pred[k];\n"
      << "      max_index = k;\n"
      << "    }\n"
      << "  }\n"
      << "  pred[0] = (float)max_index;\n"
      << "  return 1;\n"
      << "}\n";
  return oss.str();
}

inline std::string softmax(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "softmax: model is not a proper multi-class classifier";
  const int num_class = model.num_output_group;
  std::ostringstream oss;
  oss << "static inline size_t pred_transform(float* pred) {\n"
      << "  const int num_class = " << num_class << ";\n"
      << "  float max_margin = pred[0];\n"
      << "  double norm_const = 0.0;\n"
      << "  float t;\n"
      << "  for (int k = 1; k < num_class; ++k) {\n"
      << "    if (pred[k] > max_margin) {\n"
      << "      max_margin = pred[k];\n"
      << "    }\n"
      << "  }\n"
      << "  for (int k = 0; k < num_class; ++k) {\n"
      << "    t = expf(pred[k] - max_margin);\n"
      << "    norm_const += t;\n"
      << "    pred[k] = t;\n"
      << "  }\n"
      << "  for (int k = 0; k < num_class; ++k) {\n"
      << "    pred[k] /= (float)norm_const;\n"
      << "  }\n"
      << "  return (size_t)num_class;\n"
      << "}\n";
  return oss.str();
}

inline std::string multiclass_ova(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "multiclass_ova: model is not a proper multi-class classifier";
  const int num_class = model.num_output_group;
  const float alpha = model.param.sigmoid_alpha;
  CHECK_GT(alpha, 0.0f) << "multiclass_ova: alpha must be strictly positive";
  std::ostringstream oss;
  oss << "static inline size_t pred_transform(float* pred) {\n"
      << "  const float alpha = (float)" << common::ToString(alpha) << ";\n"
      << "  const int num_class = " << num_class << ";\n"
      << "  for (int k = 0; k < num_class; ++k) {\n"
      << "    pred[k] = 1.0f / (1.0f + expf(-alpha * pred[k]));\n"
      << "  }\n"
      << "  return (size_t)num_class;\n"
      << "}\n";
  return oss.str();
}

}  // namespace pred_transform
}  // namespace native
}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_NATIVE_PRED_TRANSFORM_H_
