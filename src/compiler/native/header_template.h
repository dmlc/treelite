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

{get_num_output_group_function_signature};
{get_num_feature_function_signature};
{predict_function_signature};
)TREELITETEMPLATE";
