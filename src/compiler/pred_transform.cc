/*!
* Copyright 2017 by Contributors
* \file pred_transform.cc
* \brief Library of transform functions to convert margins into predictions
* \author Philip Cho
*/

#include <treelite/semantic.h>
#include <treelite/tree.h>
#include <string>
#include <unordered_map>
#include "pred_transform.h"

#define PRED_TRANSFORM_FUNC(name) {#name, &(name)}

namespace {

using PlainBlock = treelite::semantic::PlainBlock;
using Model = treelite::Model;
using PredTransformFuncGenerator
  = std::vector<std::string> (*)(const Model&, bool);

std::vector<std::string>
identity(const Model& model, bool batch) {
  if (batch) {
    return {"return ndata;"};
  } else {
    return {"return 1;"};
  }
}

std::vector<std::string>
identity_multiclass(const Model& model, bool batch) {
  CHECK(model.num_output_group > 1)
    << "identity_multiclass: model is not a proper multi-class classifier";
  const int num_class = model.num_output_group;
  if (batch) {
    return {std::string("return ndata * ") + std::to_string(num_class) + ";"};
  } else {
    return {std::string("return ") + std::to_string(num_class) + ";"};
  }
}

std::vector<std::string>
sigmoid(const Model& model, bool batch) {
  const float alpha = model.param.sigmoid_alpha;
  CHECK_GT(alpha, 0.0f) << "sigmoid: alpha must be strictly positive";

  if (batch) {
    return {
      std::string(
      "const float alpha = (float)") + treelite::common::ToString(alpha) + ";",
      "int64_t i;",
      "#pragma omp parallel for schedule(static) num_threads(nthread) \\",
      "   default(none) firstprivate(alpha, ndata) shared(pred) private(i)",
      "for (i = 0; i < ndata; ++i) {",
      "  pred[i] = 1.0f / (1 + expf(-alpha * pred[i]));",
      "}",
      "return ndata;"};
  } else {
    return {
      std::string(
      "const float alpha = (float)")
                         + treelite::common::ToString(alpha) + ";",
      "pred[0] = 1.0f / (1 + expf(-alpha * pred[0]));",
      "return 1;"};
  }
}

std::vector<std::string>
exponential(const Model& model, bool batch) {
  if (batch) {
    return {
      "int64_t i;",
      "#pragma omp parallel for schedule(static) num_threads(nthread) \\",
      "   default(none) firstprivate(ndata) shared(pred) private(i)",
      "for (i = 0; i < ndata; ++i) {",
      "  pred[i] = expf(pred[i]);",
      "}",
      "return ndata;"};
  } else {
    return {"pred[0] = expf(pred[0]);", "return 1;"};
  }
}

std::vector<std::string>
logarithm_one_plus_exp(const Model& model, bool batch) {
  if (batch) {
    return {
      "int64_t i;",
      "#pragma omp parallel for schedule(static) num_threads(nthread) \\",
      "   default(none) firstprivate(ndata) shared(pred) private(i)",
      "for (i = 0; i < ndata; ++i) {",
      "  pred[i] = logf(1.0f + expf(pred[i]));",
      "}",
      "return ndata;"};
  } else {
    return {"pred[0] = logf(1.0f + expf(pred[0]));", "return 1;"};
  }
}

std::vector<std::string>
max_index(const Model& model, bool batch) {
  CHECK(model.num_output_group > 1)
    << "max_index: model is not a proper multi-class classifier";
  const int num_class = model.num_output_group;

  if (batch) {
    return {
      std::string(
      "const int num_class = ") + std::to_string(num_class) + ";",
      "int max_index;",
      "float max_margin;",
      "const float* margin_;",
      "float* tmp;",
      "int64_t i;",
      "tmp = (float*)malloc(ndata * sizeof(float));",
      "#pragma omp parallel for schedule(static) num_threads(nthread) \\",
      "   default(none) firstprivate(num_class, ndata) \\",
      "   private(max_index, max_margin, margin_, i) \\",
      "   shared(pred, tmp)",
      "for (i = 0; i < ndata; ++i) {",
      "  margin_ = &pred[i * num_class];",
      "  max_index = 0;",
      "  max_margin = margin_[0];",
      "  for (int k = 1; k < num_class; ++k) {",
      "    if (margin_[k] > max_margin) {",
      "      max_margin = margin_[k];",
      "      max_index = k;",
      "    }",
      "  }",
      "  tmp[i] = (float)max_index;",
      "}",
      "memcpy(pred, tmp, ndata * sizeof(float));",
      "free(tmp);",
      "return ndata;"};
  } else {
    return {
        std::string(
        "const int num_class = ") + std::to_string(num_class) + ";",
        "int max_index = 0;",
        "float max_margin = pred[0];",
        "for (int k = 1; k < num_class; ++k) {",
        "  if (pred[k] > max_margin) {",
        "    max_margin = pred[k];",
        "    max_index = k;",
        "  }",
        "}",
        "pred[0] = (float)max_index;",
        "return 1;"};
  }
}

std::vector<std::string>
softmax(const Model& model, bool batch) {
  CHECK(model.num_output_group > 1)
    << "softmax: model is not a proper multi-class classifier";
  const int num_class = model.num_output_group;

  if (batch) {
    return {
      std::string(
      "const int num_class = ") + std::to_string(num_class) + ";",
      "float max_margin;",
      "double norm_const;",
      "const float* margin_;",
      "float* out_pred_;",
      "float* tmp;",
      "float t;",
      "int64_t i;",
      "tmp = (float*)malloc(ndata * num_class * sizeof(float));",
      "#pragma omp parallel for schedule(static) num_threads(nthread) \\",
      "   default(none) firstprivate(num_class, ndata) \\",
      "   private(max_margin, norm_const, margin_, out_pred_, i, t) \\",
      "   shared(pred, tmp)",
      "for (i = 0; i < ndata; ++i) {",
      "  margin_ = &pred[i * num_class];",
      "  out_pred_ = &tmp[i * num_class];",
      "  max_margin = margin_[0];",
      "  norm_const = 0.0;",
      "  for (int k = 1; k < num_class; ++k) {",
      "    if (margin_[k] > max_margin) {",
      "      max_margin = margin_[k];",
      "    }",
      "  }",
      "  for (int k = 0; k < num_class; ++k) {",
      "    t = expf(margin_[k] - max_margin);",
      "    norm_const += t;",
      "    out_pred_[k] = t;",
      "  }",
      "  for (int k = 0; k < num_class; ++k) {",
      "    out_pred_[k] /= (float)norm_const;",
      "  }",
      "}",
      "memcpy(pred, tmp, ndata * num_class * sizeof(float));",
      "free(tmp);",
      "return ndata * num_class;"};
  } else {
    return {
      std::string(
      "const int num_class = ") + std::to_string(num_class) + ";",
      "float max_margin = pred[0];",
      "double norm_const = 0.0;",
      "float t;",
      "for (int k = 1; k < num_class; ++k) {",
      "  if (pred[k] > max_margin) {",
      "    max_margin = pred[k];",
      "  }",
      "}",
      "for (int k = 0; k < num_class; ++k) {",
      "  t = expf(pred[k] - max_margin);",
      "  norm_const += t;",
      "  pred[k] = t;",
      "}",
      "for (int k = 0; k < num_class; ++k) {",
      "  pred[k] /= (float)norm_const;",
      "}",
      "return num_class;"};
  }
}

std::vector<std::string>
multiclass_ova(const Model& model, bool batch) {
  CHECK(model.num_output_group > 1)
    << "multiclass_ova: model is not a proper multi-class classifier";
  const int num_class = model.num_output_group;
  const float alpha = model.param.sigmoid_alpha;
  CHECK_GT(alpha, 0.0f) << "multiclass_ova: alpha must be strictly positive";

  if (batch) {
    return {
      std::string(
      "const float alpha = (float)")
                         + treelite::common::ToString(alpha) + ";",
      std::string(
      "const int num_class = ") + std::to_string(num_class) + ";",
      "float* pred_;",
      "int64_t i;",
      "#pragma omp parallel for schedule(static) num_threads(nthread) \\",
      "   default(none) firstprivate(alpha, num_class, ndata) \\",
      "   private(pred_, i) shared(pred)",
      "for (i = 0; i < ndata; ++i) {",
      "  pred_ = &pred[i * num_class];"
      "  for (int k = 0; k < num_class; ++k) {",
      "    pred_[k] = 1.0f / (1 + expf(-alpha * pred_[k]));",
      "  }",
      "}",
      "return ndata * num_class;"};
  } else {
    return {
        std::string(
        "const float alpha = (float)")
                           + treelite::common::ToString(alpha) + ";",
        std::string(
        "const int num_class = ") + std::to_string(num_class) + ";",
        "for (int k = 0; k < num_class; ++k) {",
        "  pred[k] = 1.0f / (1 + expf(-alpha * pred[k]));",
        "}",
        "return num_class;"};
  }
}

const std::unordered_map<std::string, PredTransformFuncGenerator>
pred_transform_db = {
  PRED_TRANSFORM_FUNC(identity),
  PRED_TRANSFORM_FUNC(sigmoid),
  PRED_TRANSFORM_FUNC(exponential),
  PRED_TRANSFORM_FUNC(logarithm_one_plus_exp),
};

// prediction transform function for *multi-class classifiers* only
const std::unordered_map<std::string, PredTransformFuncGenerator>
pred_transform_multiclass_db = {
  PRED_TRANSFORM_FUNC(identity_multiclass),
  PRED_TRANSFORM_FUNC(max_index),
  PRED_TRANSFORM_FUNC(softmax),
  PRED_TRANSFORM_FUNC(multiclass_ova),
};

}  // namespace anonymous


std::vector<std::string>
treelite::compiler::PredTransformFunction(const Model& model, bool batch) {
  if (model.num_output_group > 1) {  // multi-class classification
    auto it = pred_transform_multiclass_db.find(model.param.pred_transform);
    if (it == pred_transform_multiclass_db.end()) {
      std::ostringstream oss;
      for (const auto& e : pred_transform_multiclass_db) {
        oss << "'" << e.first << "', ";
      }
      LOG(FATAL) << "Invalid argument given for `pred_transform` parameter. "
                 << "For multi-class classification, you should set "
                 << "`pred_transform` to one of the following: "
                 << "{ " << oss.str() << " }";
    }
    return (it->second)(model, batch);
  } else {
    auto it = pred_transform_db.find(model.param.pred_transform);
    if (it == pred_transform_db.end()) {
      std::ostringstream oss;
      for (const auto& e : pred_transform_db) {
        oss << "'" << e.first << "', ";
      }
      LOG(FATAL) << "Invalid argument given for `pred_transform` parameter. "
                 << "For any task that is NOT multi-class classification, you "
                 << "should set `pred_transform` to one of the following: "
                 << "{ " << oss.str() << " }";
    }
    return (it->second)(model, batch);
  }
}
