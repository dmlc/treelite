/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file pred_transform.h
 * \author Hyunsu Cho
 * \brief template for pred_transform() function in generated C code
 */

#ifndef TREELITE_COMPILER_NATIVE_PRED_TRANSFORM_H_
#define TREELITE_COMPILER_NATIVE_PRED_TRANSFORM_H_

#include <dmlc/logging.h>
#include <fmt/format.h>
#include <string>

using namespace fmt::literals;

namespace treelite {
namespace compiler {
namespace native {
namespace pred_transform {

inline std::string identity(const Model& model) {
  return fmt::format(
R"TREELITETEMPLATE(static inline float pred_transform(float margin) {{
  return margin;
}})TREELITETEMPLATE");
}

inline std::string sigmoid(const Model& model) {
  const float alpha = model.param.sigmoid_alpha;
  CHECK_GT(alpha, 0.0f) << "sigmoid: alpha must be strictly positive";
  return fmt::format(
R"TREELITETEMPLATE(static inline float pred_transform(float margin) {{
  const float alpha = (float){alpha};
  return 1.0f / (1 + expf(-alpha * margin));
}})TREELITETEMPLATE",
  "alpha"_a = alpha);
}

inline std::string exponential(const Model& model) {
  return fmt::format(
R"TREELITETEMPLATE(static inline float pred_transform(float margin) {{
  return expf(margin);
}})TREELITETEMPLATE");
}

inline std::string logarithm_one_plus_exp(const Model& model) {
  return fmt::format(
R"TREELITETEMPLATE(static inline float pred_transform(float margin) {{
  return log1pf(expf(margin));
}})TREELITETEMPLATE");
}

inline std::string identity_multiclass(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "identity_multiclass: model is not a proper multi-class classifier";
  return fmt::format(
R"TREELITETEMPLATE(static inline size_t pred_transform(float* pred) {{
  return {num_class};
}})TREELITETEMPLATE",
  "num_class"_a = model.num_output_group);
}

inline std::string max_index(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "max_index: model is not a proper multi-class classifier";
  return fmt::format(
R"TREELITETEMPLATE(static inline size_t pred_transform(float* pred) {{
  const int num_class = {num_class};
  int max_index = 0;
  float max_margin = pred[0];
  for (int k = 1; k < num_class; ++k) {{
    if (pred[k] > max_margin) {{
      max_margin = pred[k];
      max_index = k;
    }}
  }}
  pred[0] = (float)max_index;
  return 1;
}})TREELITETEMPLATE",
    "num_class"_a = model.num_output_group);
}

inline std::string softmax(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "softmax: model is not a proper multi-class classifier";
  return fmt::format(
R"TREELITETEMPLATE(static inline size_t pred_transform(float* pred) {{
  const int num_class = {num_class};
  float max_margin = pred[0];
  double norm_const = 0.0;
  float t;
  for (int k = 1; k < num_class; ++k) {{
    if (pred[k] > max_margin) {{
      max_margin = pred[k];
    }}
  }}
  for (int k = 0; k < num_class; ++k) {{
    t = expf(pred[k] - max_margin);
    norm_const += t;
    pred[k] = t;
  }}
  for (int k = 0; k < num_class; ++k) {{
    pred[k] /= (float)norm_const;
  }}
  return (size_t)num_class;
}})TREELITETEMPLATE",
    "num_class"_a = model.num_output_group);
}

inline std::string multiclass_ova(const Model& model) {
  CHECK(model.num_output_group > 1)
    << "multiclass_ova: model is not a proper multi-class classifier";
  const int num_class = model.num_output_group;
  const float alpha = model.param.sigmoid_alpha;
  CHECK_GT(alpha, 0.0f) << "multiclass_ova: alpha must be strictly positive";
  return fmt::format(
      R"TREELITETEMPLATE(static inline size_t pred_transform(float* pred) {{
  const float alpha = (float){alpha};
  const int num_class = {num_class};
  for (int k = 0; k < num_class; ++k) {{
    pred[k] = 1.0f / (1.0f + expf(-alpha * pred[k]));
  }}
  return (size_t)num_class;
}})TREELITETEMPLATE",
      "num_class"_a = model.num_output_group, "alpha"_a = alpha);
}

}  // namespace pred_transform
}  // namespace native
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_NATIVE_PRED_TRANSFORM_H_
