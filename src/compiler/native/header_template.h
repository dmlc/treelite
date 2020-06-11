/*!
 * Copyright (c) 2018 by Contributors
 * \file header_template.h
 * \author Hyunsu Cho
 * \brief template for header
 */

#ifndef TREELITE_COMPILER_NATIVE_HEADER_TEMPLATE_H_
#define TREELITE_COMPILER_NATIVE_HEADER_TEMPLATE_H_

namespace treelite {
namespace compiler {
namespace native {

const char* header_template =
R"TREELITETEMPLATE(
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#if defined(__clang__) || defined(__GNUC__)
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif

union Entry {{
  int missing;
  float fvalue;
  int qvalue;
}};

struct Node {{
  uint8_t default_left;
  unsigned int split_index;
  {threshold_type} threshold;
  int left_child;
  int right_child;
}};

extern const unsigned char is_categorical[];

{dllexport}{get_num_output_group_function_signature};
{dllexport}{get_num_feature_function_signature};
{dllexport}{get_pred_transform_function_signature};
{dllexport}{get_sigmoid_alpha_function_signature};
{dllexport}{get_global_bias_function_signature};
{dllexport}{predict_function_signature};
)TREELITETEMPLATE";

}  // namespace native
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_NATIVE_HEADER_TEMPLATE_H_
