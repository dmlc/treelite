/*!
 * Copyright (c) 2021 by Contributors
 * \file pred_transform.cc
 * \author Hyunsu Cho
 * \brief Functions to post-process prediction results
 */

#include "./pred_transform.h"
#include <treelite/gtil.h>
#include <treelite/tree.h>
#include <dmlc/logging.h>
#include <string>
#include <unordered_map>
#include <cmath>
#include <cstddef>

namespace treelite {
namespace gtil {
namespace pred_transform {

std::size_t identity(const treelite::Model&, const float* in, float* out) {
  *out = *in;
  return 1;
}

std::size_t signed_square(const treelite::Model&, const float* in, float* out) {
  const float margin = *in;
  *out = std::copysign(margin * margin, margin);
  return 1;
}

std::size_t hinge(const treelite::Model&, const float* in, float* out) {
  *out = (*in > 0 ? 1.0f : 0.0f);
  return 1;
}

std::size_t sigmoid(const treelite::Model& model, const float* in, float* out) {
  const float alpha = model.param.sigmoid_alpha;
  CHECK(alpha > 0.0f) << "sigmoid: alpha must be strictly positive";
  *out = 1.0f / (1.0f + std::exp(-alpha * *in));
  return 1;
}

std::size_t exponential(const treelite::Model&, const float* in, float* out) {
  *out = std::exp(*in);
  return 1;
}

std::size_t logarithm_one_plus_exp(const treelite::Model&, const float* in, float* out) {
  *out = std::log1p(std::exp(*in));
  return 1;
}

std::size_t identity_multiclass(const treelite::Model& model, const float* in, float* out) {
  auto num_class = static_cast<std::size_t>(model.task_param.num_class);
  CHECK(num_class > 1) << "model must be a multi-class classifier";
  for (std::size_t i = 0; i < num_class; ++i) {
    out[i] = in[i];
  }
  return num_class;
}

std::size_t max_index(const treelite::Model& model, const float* in, float* out) {
  auto num_class = static_cast<std::size_t>(model.task_param.num_class);
  CHECK(num_class > 1) << "model must be a multi-class classifier";
  std::size_t max_index = 0;
  float max_margin = in[0];
  for (std::size_t i = 1; i < num_class; ++i) {
    if (in[i] > max_margin) {
      max_margin = in[i];
      max_index = i;
    }
  }
  out[0] = static_cast<float>(max_index);
  return 1;
}

std::size_t softmax(const treelite::Model& model, const float* in, float* out) {
  auto num_class = static_cast<std::size_t>(model.task_param.num_class);
  CHECK(num_class > 1) << "model must be a multi-class classifier";
  float max_margin = in[0];
  double norm_const = 0.0;
  float t;
  for (std::size_t i = 1; i < num_class; ++i) {
    if (in[i] > max_margin) {
      max_margin = in[i];
    }
  }
  for (std::size_t i = 0; i < num_class; ++i) {
    t = std::exp(in[i] - max_margin);
    norm_const += t;
    out[i] = t;
  }
  for (std::size_t i = 0; i < num_class; ++i) {
    out[i] /= static_cast<float>(norm_const);
  }
  return num_class;
}

std::size_t multiclass_ova(const treelite::Model& model, const float* in, float* out) {
  auto num_class = static_cast<std::size_t>(model.task_param.num_class);
  CHECK(num_class > 1) << "model must be a multi-class classifier";
  const float alpha = model.param.sigmoid_alpha;
  CHECK(alpha > 0.0f) << "multiclass_ova: alpha must be strictly positive";
  for (std::size_t i = 0; i < num_class; ++i) {
    out[i] = 1.0f / (1.0f + std::exp(-alpha * in[i]));
  }
  return num_class;
}

}  // namespace pred_transform

const std::unordered_map<std::string, PredTransformFuncType> pred_transform_func{
    {"identity", pred_transform::identity},
    {"signed_square", pred_transform::signed_square},
    {"hinge", pred_transform::hinge},
    {"sigmoid", pred_transform::sigmoid},
    {"exponential", pred_transform::exponential},
    {"logarithm_one_plus_exp", pred_transform::logarithm_one_plus_exp},
    {"identity_multiclass", pred_transform::identity_multiclass},
    {"max_index", pred_transform::max_index},
    {"softmax", pred_transform::softmax},
    {"multiclass_ova", pred_transform::multiclass_ova}
};

PredTransformFuncType LookupPredTransform(const std::string& name) {
  return pred_transform_func.at(name);
}

}  // namespace gtil
}  // namespace treelite
