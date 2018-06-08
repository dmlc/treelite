#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

union Entry {
  int missing;
  float fvalue;
  int qvalue;
};

size_t get_num_output_group(void) {
  return 1;
}

size_t get_num_feature(void) {
  return 127;
}

static inline float pred_transform(float margin) {
  const float alpha = (float)1;
  return 1.0f / (1 + expf(-alpha * margin));
}

float predict(union Entry* data, int pred_margin) {
  float sum = 0.0f;
  if (!(data[29].missing != -1) || (data[29].fvalue < -9.5367432e-07)) {
    if (!(data[56].missing != -1) || (data[56].fvalue < -9.5367432e-07)) {
      if (!(data[60].missing != -1) || (data[60].fvalue < -9.5367432e-07)) {
        sum += (float)1.8989965;
      } else {
        sum += (float)-1.9473684;
      }
    } else {
      if (!(data[21].missing != -1) || (data[21].fvalue < -9.5367432e-07)) {
        sum += (float)1.7837838;
      } else {
        sum += (float)-1.981352;
      }
    }
  } else {
    if (!(data[109].missing != -1) || (data[109].fvalue < -9.5367432e-07)) {
      if (!(data[67].missing != -1) || (data[67].fvalue < -9.5367432e-07)) {
        sum += (float)-1.9854598;
      } else {
        sum += (float)0.93877554;
      }
    } else {
      sum += (float)1.8709677;
    }
  }
  if (!(data[29].missing != -1) || (data[29].fvalue < -9.5367432e-07)) {
    if (!(data[21].missing != -1) || (data[21].fvalue < -9.5367432e-07)) {
      sum += (float)1.1460791;
    } else {
      if (!(data[36].missing != -1) || (data[36].fvalue < -9.5367432e-07)) {
        sum += (float)-6.8799467;
      } else {
        sum += (float)-0.10659159;
      }
    }
  } else {
    if (!(data[109].missing != -1) || (data[109].fvalue < -9.5367432e-07)) {
      if (!(data[39].missing != -1) || (data[39].fvalue < -9.5367432e-07)) {
        sum += (float)-0.093065776;
      } else {
        sum += (float)-1.1526121;
      }
    } else {
      sum += (float)1.0042307;
    }
  }

  sum = sum + (float)(-0);
  if (!pred_margin) {
    return pred_transform(sum);
  } else {
    return sum;
  }
}
