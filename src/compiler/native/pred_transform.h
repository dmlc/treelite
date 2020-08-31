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
#include "./typeinfo_ctypes.h"

using namespace fmt::literals;

namespace treelite {
namespace compiler {
namespace native {
namespace pred_transform {

inline std::string identity(const Model& model) {
  return fmt::format(
R"TREELITETEMPLATE(static inline {threshold_type} pred_transform({threshold_type} margin) {{
  return margin;
}})TREELITETEMPLATE",
"threshold_type"_a = native::TypeInfoToCTypeString(model.GetThresholdType()));
}

inline std::string sigmoid(const Model& model) {
  const float alpha = model.GetParam().sigmoid_alpha;
  const TypeInfo threshold_type = model.GetThresholdType();
  CHECK_GT(alpha, 0.0f) << "sigmoid: alpha must be strictly positive";
  return fmt::format(
R"TREELITETEMPLATE(static inline {threshold_type} pred_transform({threshold_type} margin) {{
  const {threshold_type} alpha = ({threshold_type}){alpha};
  return ({threshold_type})(1) / (({threshold_type})(1) + {exp}(-alpha * margin));
}})TREELITETEMPLATE",
  "alpha"_a = alpha,
  "threshold_type"_a = native::TypeInfoToCTypeString(threshold_type),
  "exp"_a = native::CExpForTypeInfo(threshold_type));
}

inline std::string exponential(const Model& model) {
  const TypeInfo threshold_type = model.GetThresholdType();
  return fmt::format(
R"TREELITETEMPLATE(static inline {threshold_type} pred_transform({threshold_type} margin) {{
  return {exp}(margin);
}})TREELITETEMPLATE",
  "threshold_type"_a = native::TypeInfoToCTypeString(threshold_type),
  "exp"_a = native::CExpForTypeInfo(threshold_type));
}

inline std::string logarithm_one_plus_exp(const Model& model) {
  const TypeInfo threshold_type = model.GetThresholdType();
  return fmt::format(
R"TREELITETEMPLATE(static inline {threshold_type} pred_transform({threshold_type} margin) {{
  return {log1p}({exp}(margin));
}})TREELITETEMPLATE",
  "threshold_type"_a = native::TypeInfoToCTypeString(threshold_type),
  "exp"_a = native::CExpForTypeInfo(threshold_type),
  "log1p"_a = native::CLog1PForTypeInfo(threshold_type));
}

inline std::string identity_multiclass(const Model& model) {
  CHECK_GT(model.GetNumOutputGroup(), 1)
    << "identity_multiclass: model is not a proper multi-class classifier";
  return fmt::format(
R"TREELITETEMPLATE(static inline size_t pred_transform({threshold_type}* pred) {{
  return {num_class};
}})TREELITETEMPLATE",
  "num_class"_a = model.GetNumOutputGroup(),
  "threshold_type"_a = native::TypeInfoToCTypeString(model.GetThresholdType()));
}

inline std::string max_index(const Model& model) {
  CHECK_GT(model.GetNumOutputGroup(), 1)
    << "max_index: model is not a proper multi-class classifier";
  const TypeInfo threshold_type = model.GetThresholdType();
  return fmt::format(
R"TREELITETEMPLATE(static inline size_t pred_transform({threshold_type}* pred) {{
  const int num_class = {num_class};
  int max_index = 0;
  {threshold_type} max_margin = pred[0];
  for (int k = 1; k < num_class; ++k) {{
    if (pred[k] > max_margin) {{
      max_margin = pred[k];
      max_index = k;
    }}
  }}
  pred[0] = ({threshold_type})max_index;
  return 1;
}})TREELITETEMPLATE",
    "num_class"_a = model.GetNumOutputGroup(),
    "threshold_type"_a = native::TypeInfoToCTypeString(threshold_type));
}

inline std::string softmax(const Model& model) {
  CHECK_GT(model.GetNumOutputGroup(), 1)
    << "softmax: model is not a proper multi-class classifier";
  const TypeInfo threshold_type = model.GetThresholdType();
  return fmt::format(
R"TREELITETEMPLATE(static inline size_t pred_transform({threshold_type}* pred) {{
  const int num_class = {num_class};
  {threshold_type} max_margin = pred[0];
  double norm_const = 0.0;
  {threshold_type} t;
  for (int k = 1; k < num_class; ++k) {{
    if (pred[k] > max_margin) {{
      max_margin = pred[k];
    }}
  }}
  for (int k = 0; k < num_class; ++k) {{
    t = {exp}(pred[k] - max_margin);
    norm_const += t;
    pred[k] = t;
  }}
  for (int k = 0; k < num_class; ++k) {{
    pred[k] /= ({threshold_type})norm_const;
  }}
  return (size_t)num_class;
}})TREELITETEMPLATE",
    "num_class"_a = model.GetNumOutputGroup(),
    "threshold_type"_a = native::TypeInfoToCTypeString(threshold_type),
    "exp"_a = native::CExpForTypeInfo(threshold_type));
}

inline std::string multiclass_ova(const Model& model) {
  CHECK(model.GetNumOutputGroup() > 1)
    << "multiclass_ova: model is not a proper multi-class classifier";
  const int num_class = model.GetNumOutputGroup();
  const float alpha = model.GetParam().sigmoid_alpha;
  const TypeInfo threshold_type = model.GetThresholdType();
  CHECK_GT(alpha, 0.0f) << "multiclass_ova: alpha must be strictly positive";
  return fmt::format(
R"TREELITETEMPLATE(static inline size_t pred_transform({threshold_type}* pred) {{
  const {threshold_type} alpha = ({threshold_type}){alpha};
  const int num_class = {num_class};
  for (int k = 0; k < num_class; ++k) {{
    pred[k] = ({threshold_type})(1) / (({threshold_type})(1) + {exp}(-alpha * pred[k]));
  }}
  return (size_t)num_class;
}})TREELITETEMPLATE",
    "num_class"_a = model.GetNumOutputGroup(), "alpha"_a = alpha,
    "threshold_type"_a = native::TypeInfoToCTypeString(threshold_type),
    "exp"_a = native::CExpForTypeInfo(threshold_type));
}

}  // namespace pred_transform
}  // namespace native
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_NATIVE_PRED_TRANSFORM_H_
