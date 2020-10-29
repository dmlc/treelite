/*!
 * Copyright (c) 2020 by Contributors
 * \file xgboost_util.cc
 * \brief Common utilities for XGBoost frontends
 * \author Hyunsu Cho
 */

#include <treelite/tree.h>
#include <cstring>
#include "xgboost/xgboost.h"

namespace {

inline void SetPredTransformString(const char* value, treelite::ModelParam* param) {
  std::strncpy(param->pred_transform, value, sizeof(param->pred_transform));
}

}  // anonymous namespace

namespace treelite {
namespace details {
namespace xgboost {

const std::vector<std::string> exponential_objectives{
    "count:poisson", "reg:gamma", "reg:tweedie"
};

// set correct prediction transform function, depending on objective function
void SetPredTransform(const std::string& objective_name, ModelParam* param) {
  if (objective_name == "multi:softmax") {
    SetPredTransformString("max_index", param);
  } else if (objective_name == "multi:softprob") {
    SetPredTransformString("softmax", param);
  } else if (objective_name == "reg:logistic" || objective_name == "binary:logistic") {
    SetPredTransformString("sigmoid", param);
    param->sigmoid_alpha = 1.0f;
  } else if (std::find(exponential_objectives.cbegin(), exponential_objectives.cend(),
                       objective_name) != exponential_objectives.cend()) {
    SetPredTransformString("exponential", param);
  } else {
    SetPredTransformString("identity", param);
  }
}

// Transform the global bias parameter from probability into margin score
void TransformGlobalBiasToMargin(ModelParam* param) {
  if (std::strcmp(param->pred_transform, "sigmoid") == 0) {
    param->global_bias = ProbToMargin::Sigmoid(param->global_bias);
  } else if (std::strcmp(param->pred_transform, "exponential") == 0) {
    param->global_bias = ProbToMargin::Exponential(param->global_bias);
  }
}

}  // namespace xgboost
}  // namespace details
}  // namespace treelite
