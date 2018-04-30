/*!
 * Copyright (c) 2018 by Contributors
 * \file qnode_template.h
 * \author Philip Cho
 * \brief code template for QuantizerNode
 */

#ifndef TREELITE_COMPILER_NATIVE_QNODE_TEMPLATE_H_
#define TREELITE_COMPILER_NATIVE_QNODE_TEMPLATE_H_

namespace treelite {
namespace compiler {
namespace native {

const char* qnode_template =
R"TREELITETEMPLATE(
static const unsigned char is_categorical[] = {{
{array_is_categorical}
}};
static const float threshold[] = {{
{array_threshold}
}};
static const int th_begin[] = {{
{array_th_begin}
}};
static const int th_len[] = {{
{array_th_len}
}};

/*
 * \brief function to convert a feature value into bin index.
 * \param val feature value, in floating-point
 * \param fid feature identifier
 * \return bin index corresponding to given feature value
 */
static inline int quantize(float val, unsigned fid) {{
  const float* array = &threshold[th_begin[fid]];
  int len = th_len[fid];
  int low = 0;
  int high = len;
  int mid;
  float mval;
  if (val < array[0]) {{
    return -10;
  }}
  while (low + 1 < high) {{
    mid = (low + high) / 2;
    mval = array[mid];
    if (val == mval) {{
      return mid * 2;
    }} else if (val < mval) {{
      high = mid;
    }} else {{
      low = mid;
    }}
  }}
  if (array[low] == val) {{
    return low * 2;
  }} else if (high == len) {{
    return len * 2;
  }} else {{
    return low * 2 + 1;
  }}
}}
)TREELITETEMPLATE";

const char* quantize_loop_template =
R"TREELITETEMPLATE(
for (int i = 0; i < {num_feature}; ++i) {{
  if (data[i].missing != -1 && !is_categorical[i]) {{
    data[i].qvalue = quantize(data[i].fvalue, i);
  }}
}}
)TREELITETEMPLATE";

}  // namespace native
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_NATIVE_QNODE_TEMPLATE_H_
