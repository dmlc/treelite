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
static const float threshold[] = {{
{array_threshold}
}};
static const int th_begin[] = {{
{array_th_begin}
}};
static const int th_len[] = {{
{array_th_len}
}};

#include <stdlib.h>

/*
 * \brief function to convert a feature value into bin index.
 * \param val feature value, in floating-point
 * \param fid feature identifier
 * \return bin index corresponding to given feature value
 */
static inline int quantize(float val, unsigned fid) {{
  const size_t offset = th_begin[fid];
  const float* array = &threshold[offset];
  int len = th_len[fid];
  int low = 0;
  int high = len;
  int mid;
  float mval;
  // It is possible th_begin[i] == [total_num_threshold]. This means that
  // all features i, (i+1), ... are not used for any of the splits in the model.
  // So in this case, just return something
  if (offset == {total_num_threshold} || val < array[0]) {{
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
