/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file pred_transform.cc
 * \brief Library of transform functions to convert margins into predictions
 * \author Hyunsu Cho
 */

#include "pred_transform.h"
#include <string>
#include <unordered_map>

#include "./native/pred_transform.h"

#define PRED_TRANSFORM_FUNC(name) {#name, &(name)}

namespace {

using Model = treelite::Model;
using PredTransformFuncGenerator
  = std::string (*)(const std::string&, const Model&);

/* boilerplate */
#define TREELITE_PRED_TRANSFORM_REGISTRY_DEFAULT_TEMPLATE(FUNC_NAME) \
std::string \
FUNC_NAME(const std::string& backend, const Model& model) { \
  if (backend == "native") { \
    return treelite::compiler::native::pred_transform::FUNC_NAME(model); \
  } else { \
    TREELITE_LOG(FATAL) << "Unrecognized backend: " << backend; \
    return std::string(); \
  } \
}

TREELITE_PRED_TRANSFORM_REGISTRY_DEFAULT_TEMPLATE(identity)
TREELITE_PRED_TRANSFORM_REGISTRY_DEFAULT_TEMPLATE(signed_square)
TREELITE_PRED_TRANSFORM_REGISTRY_DEFAULT_TEMPLATE(hinge)
TREELITE_PRED_TRANSFORM_REGISTRY_DEFAULT_TEMPLATE(sigmoid)
TREELITE_PRED_TRANSFORM_REGISTRY_DEFAULT_TEMPLATE(exponential)
TREELITE_PRED_TRANSFORM_REGISTRY_DEFAULT_TEMPLATE(logarithm_one_plus_exp)
TREELITE_PRED_TRANSFORM_REGISTRY_DEFAULT_TEMPLATE(identity_multiclass)
TREELITE_PRED_TRANSFORM_REGISTRY_DEFAULT_TEMPLATE(max_index)
TREELITE_PRED_TRANSFORM_REGISTRY_DEFAULT_TEMPLATE(softmax)
TREELITE_PRED_TRANSFORM_REGISTRY_DEFAULT_TEMPLATE(multiclass_ova)

const std::unordered_map<std::string, PredTransformFuncGenerator>
pred_transform_db = {
  PRED_TRANSFORM_FUNC(identity),
  PRED_TRANSFORM_FUNC(signed_square),
  PRED_TRANSFORM_FUNC(hinge),
  PRED_TRANSFORM_FUNC(sigmoid),
  PRED_TRANSFORM_FUNC(exponential),
  PRED_TRANSFORM_FUNC(logarithm_one_plus_exp)
};
/*! [pred_transform_db]
  - identity
    Do not transform. The output will be a vector of length
    [number of data points] that contains the margin score for every data point.
  - signed_square
    Apply the function f(x) = sign(x) * (x**2) element-wise to the margin vector. The
    output will be a vector of length [number of data points].
  - hinge
    Apply the function f(x) = (1 if x > 0 else 0) element-wise to the margin vector. The
    output will be a vector of length [number of data points], filled with 0's and 1's.
  - sigmoid
    Apply the sigmoid function element-wise to the margin vector. The output
    will be a vector of length [number of data points] that contains the
    probability of each data point belonging to the positive class.
  - exponential
    Apply the exponential function (exp) element-wise to the margin vector. The
    output will be a vector of length [number of data points].
  - logarithm_one_plus_exp
    Apply the function f(x) = log(1 + exp(x)) element-wise to the margin vector.
    The output will be a vector of length [number of data points].
    [pred_transform_db] */

// prediction transform function for *multi-class classifiers* only
const std::unordered_map<std::string, PredTransformFuncGenerator>
pred_transform_multiclass_db = {
  PRED_TRANSFORM_FUNC(identity_multiclass),
  PRED_TRANSFORM_FUNC(max_index),
  PRED_TRANSFORM_FUNC(softmax),
  PRED_TRANSFORM_FUNC(multiclass_ova)
};
/*! [pred_transform_multiclass_db]
 - identity_multiclass
   do not transform. The output will be a matrix with dimensions
   [number of data points] * [number of classes] that contains the margin score
   for every (data point, class) pair.
 - max_index
   compute the most probable class for each data point and output the class
   index. The output will be a vector of length [number of data points] that
   contains the most likely class of each data point.
 - softmax
   use the softmax function to transform a multi-dimensional vector into a
   proper probability distribution. The output will be a matrix with dimensions
   [number of data points] * [number of classes] that contains the predicted
   probability of each data point belonging to each class.
 - multiclass_ova
   apply the sigmoid function element-wise to the margin matrix. The output will
   be a matrix with dimensions [number of data points] * [number of classes].
    [pred_transform_multiclass_db] */

}  // anonymous namespace

std::string
treelite::compiler::PredTransformFunction(const std::string& backend,
                                          const Model& model) {
  ModelParam param = model.param;
  if (model.task_param.num_class > 1) {  // multi-class classification
    auto it = pred_transform_multiclass_db.find(param.pred_transform);
    if (it == pred_transform_multiclass_db.end()) {
      std::ostringstream oss;
      for (const auto& e : pred_transform_multiclass_db) {
        oss << "'" << e.first << "', ";
      }
      TREELITE_LOG(FATAL) << "Invalid argument given for `pred_transform` parameter. "
                          << "For multi-class classification, you should set "
                          << "`pred_transform` to one of the following: "
                          << "{ " << oss.str() << " }";
    }
    return (it->second)(backend, model);
  } else {
    auto it = pred_transform_db.find(param.pred_transform);
    if (it == pred_transform_db.end()) {
      std::ostringstream oss;
      for (const auto& e : pred_transform_db) {
        oss << "'" << e.first << "', ";
      }
      TREELITE_LOG(FATAL) << "Invalid argument given for `pred_transform` parameter. "
                          << "For any task that is NOT multi-class classification, you "
                          << "should set `pred_transform` to one of the following: "
                          << "{ " << oss.str() << " }";
    }
    return (it->second)(backend, model);
  }
}
