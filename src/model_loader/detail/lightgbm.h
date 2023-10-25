/*!
 * Copyright (c) 2020-2023 by Contributors
 * \file lightgbm.h
 * \brief Helper functions for loading LightGBM models
 * \author Hyunsu Cho
 */
#ifndef SRC_MODEL_LOADER_DETAIL_LIGHTGBM_H_
#define SRC_MODEL_LOADER_DETAIL_LIGHTGBM_H_

#include <treelite/logging.h>

#include <string>

#include "./string_utils.h"

namespace treelite::model_loader::detail::lightgbm {

/*!
 * \brief Canonicalize the name of an objective function.
 *
 * Some objective functions have many aliases. We use the canonical name to avoid confusion.
 *
 * @param obj_name Name of an objective function
 * @return Canonical name
 */
std::string CanonicalObjective(std::string const& obj_name) {
  if (obj_name == "regression" || obj_name == "regression_l2" || obj_name == "l2"
      || obj_name == "mean_squared_error" || obj_name == "mse" || obj_name == "l2_root"
      || obj_name == "root_mean_squared_error" || obj_name == "rmse") {
    return "regression";
  } else if (obj_name == "regression_l1" || obj_name == "l1" || obj_name == "mean_absolute_error"
             || obj_name == "mae") {
    return "regression_l1";
  } else if (obj_name == "mape" || obj_name == "mean_absolute_percentage_error") {
    return "mape";
  } else if (obj_name == "multiclass" || obj_name == "softmax") {
    return "multiclass";
  } else if (obj_name == "multiclassova" || obj_name == "multiclass_ova" || obj_name == "ova"
             || obj_name == "ovr") {
    return "multiclassova";
  } else if (obj_name == "cross_entropy" || obj_name == "xentropy") {
    return "cross_entropy";
  } else if (obj_name == "cross_entropy_lambda" || obj_name == "xentlambda") {
    return "cross_entropy_lambda";
  } else if (obj_name == "rank_xendcg" || obj_name == "xendcg" || obj_name == "xe_ndcg"
             || obj_name == "xe_ndcg_mart" || obj_name == "xendcg_mart") {
    return "rank_xendcg";
  } else if (obj_name == "huber" || obj_name == "fair" || obj_name == "poisson"
             || obj_name == "quantile" || obj_name == "gamma" || obj_name == "tweedie"
             || obj_name == "binary" || obj_name == "lambdarank" || obj_name == "custom") {
    // These objectives have no aliases
    return obj_name;
  } else {
    TREELITE_LOG(FATAL) << "Unknown objective name: \"" << obj_name << "\"";
    return "";
  }
}

}  // namespace treelite::model_loader::detail::lightgbm

#endif  // SRC_MODEL_LOADER_DETAIL_LIGHTGBM_H_
