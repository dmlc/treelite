const char* qnode_template =
R"TREELITETEMPLATE(
private static final float[] threshold = {{
{array_threshold}
}};
private static final int[] th_begin = {{
{array_th_begin}
}};
private static final int[] th_len = {{
{array_th_len}
}};

/*
 * Function to convert a feature value into bin index.
 * @param val feature value, in floating-point
 * @param fid feature identifier
 * @return bin index corresponding to given feature value
 */
private static int quantize(float val, int fid) {{
  final int offset = th_begin[fid];
  final int len = th_len[fid];
  int low = 0;
  int high = len;
  int mid;
  float mval;
  // It is possible th_begin[i] == [total_num_threshold]. This means that
  // all features i, (i+1), ... are not used for any of the splits in the model.
  // So in this case, just return something
  if (offset == {total_num_threshold} || val < threshold[offset]) {{
    return -10;
  }}
  while (low + 1 < high) {{
    mid = (low + high) / 2;
    mval = threshold[offset + mid];
    if (val == mval) {{
      return mid * 2;
    }} else if (val < mval) {{
      high = mid;
    }} else {{
      low = mid;
    }}
  }}
  if (threshold[offset + low] == val) {{
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
  if (data[i].missing.get() != -1 && !is_categorical[i]) {{
    data[i].qvalue.set(quantize(data[i].fvalue.get(), i));
  }}
}}
)TREELITETEMPLATE";
