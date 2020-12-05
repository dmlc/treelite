/*!
 * Copyright (c) 2020 by Contributors
 * \file xgboost.h
 * \brief Helper functions for loading XGBoost models
 * \author William Hicks
 */
#ifndef TREELITE_FRONTEND_XGBOOST_XGBOOST_H_
#define TREELITE_FRONTEND_XGBOOST_XGBOOST_H_

#include <string>
#include <vector>
#include <cmath>

namespace treelite {

struct ModelParam;  // forward declaration

namespace details {
namespace xgboost {

struct ProbToMargin {
  static float Sigmoid(float global_bias) {
    return -logf(1.0f / global_bias - 1.0f);
  }
  static float Exponential(float global_bias) {
    return logf(global_bias);
  }
};

extern const std::vector<std::string> exponential_objectives;

// set correct prediction transform function, depending on objective function
void SetPredTransform(const std::string& objective_name, ModelParam* param);

// Transform the global bias parameter from probability into margin score
void TransformGlobalBiasToMargin(const std::string& objective_name, ModelParam* param);

enum FeatureType {
  kNumerical = 0,
  kCategorical = 1
};

}  // namespace xgboost
}  // namespace details
}  // namespace treelite
#endif  // TREELITE_FRONTEND_XGBOOST_XGBOOST_H_
