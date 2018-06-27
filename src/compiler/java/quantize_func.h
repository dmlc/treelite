/*!
 * Copyright (c) 2017 by Contributors
 * \file quantize_func.h
 * \author Philip Cho
 * \brief template for quantize() function in generated Java code
 */
#ifndef TREELITE_COMPILER_JAVA_QUANTIZE_FUNC_H_
#define TREELITE_COMPILER_JAVA_QUANTIZE_FUNC_H_

namespace treelite {
namespace compiler {
namespace java {

const char* quantize_func =
"  private static int quantize(float val, int fid) {\n"
"    final int offset = th_begin[fid];\n"
"    final int len = th_len[fid];\n"
"    int low = 0;\n"
"    int high = len;\n"
"    int mid;\n"
"    float mval;\n"
"    if (val < threshold[offset + 0]) {\n"
"      return -10;\n"
"    }\n"
"    while (low + 1 < high) {\n"
"      mid = (low + high) / 2;\n"
"      mval = threshold[offset + mid];\n"
"      if (val == mval) {\n"
"        return mid * 2;\n"
"      } else if (val < mval) {\n"
"        high = mid;\n"
"      } else {\n"
"        low = mid;\n"
"      }\n"
"    }\n"
"    if (threshold[offset + low] == val) {\n"
"      return low * 2;\n"
"    } else if (high == len) {\n"
"      return len * 2;\n"
"    } else {\n"
"      return low * 2 + 1;\n"
"    }\n"
"  }\n";

}  // namespace java
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_JAVA_QUANTIZE_FUNC_H_
