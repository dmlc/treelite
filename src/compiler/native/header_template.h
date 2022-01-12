/*!
 * Copyright (c) 2018-2020 by Contributors
 * \file header_template.h
 * \author Hyunsu Cho
 * \brief template for header
 */

#ifndef TREELITE_COMPILER_NATIVE_HEADER_TEMPLATE_H_
#define TREELITE_COMPILER_NATIVE_HEADER_TEMPLATE_H_

namespace treelite {
namespace compiler {
namespace native {

const char* const query_functions_prototype_template =
R"TREELITETEMPLATE(
{dllexport}size_t get_num_class(void);
{dllexport}size_t get_num_feature(void);
{dllexport}const char* get_pred_transform(void);
{dllexport}float get_sigmoid_alpha(void);
{dllexport}float get_ratio_c(void);
{dllexport}float get_global_bias(void);
{dllexport}const char* get_threshold_type(void);
{dllexport}const char* get_leaf_output_type(void);
)TREELITETEMPLATE";

const char* const header_template =
R"TREELITETEMPLATE(
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
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
  {threshold_type} fvalue;
  int qvalue;
}};

struct Node {{
  uint8_t default_left;
  unsigned int split_index;
  {threshold_type_Node} threshold;
  int left_child;
  int right_child;
}};

extern const unsigned char is_categorical[];

{query_functions_prototype}
{dllexport}{predict_function_signature};
)TREELITETEMPLATE";

}  // namespace native
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_NATIVE_HEADER_TEMPLATE_H_
