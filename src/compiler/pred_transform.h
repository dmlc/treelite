/*!
* Copyright by 2017 Contributors
* \file pred_transform.h
* \brief tools to define prediction transform function
* \author Philip Cho
*/
#ifndef TREELITE_PRED_TRANSFORM_H_
#define TREELITE_PRED_TRANSFORM_H_

#include <vector>
#include <string>

namespace treelite {
namespace compiler {

inline std::string PredTransformPrototype(bool batch) {
  if (batch) {
    return "size_t pred_transform_batch(float* pred, int64_t ndata, int nthread)";
  } else {
    return "size_t pred_transform(float* pred)";
  }
}

std::vector<std::string> PredTransformFunction(const Model& model, bool batch);

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_PRED_TRANSFORM_H_