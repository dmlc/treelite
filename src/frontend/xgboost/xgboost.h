/*!
 * Copyright (c) 2020 by Contributors
 * \file xgboost.h
 * \brief Helper functions for loading XGBoost models
 * \author William Hicks
 */
#ifndef TREELITE_FRONTEND_XGBOOST_XGBOOST_H_
#define TREELITE_FRONTEND_XGBOOST_XGBOOST_H_

#include <cmath>

namespace treelite {
namespace details {
struct ProbToMargin {
  static float Sigmoid(float global_bias) {
    return -logf(1.0f / global_bias - 1.0f);
  }
  static float Exponential(float global_bias) {
    return logf(global_bias);
  }
};
}  // namespace details
}  // namespace treelite
#endif  // TREELITE_FRONTEND_XGBOOST_XGBOOST_H_
