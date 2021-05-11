/*!
 * Copyright (c) 2021 by Contributors
 * \file pred_transform.h
 * \author Hyunsu Cho
 * \brief Functions to post-process prediction results
 */

#ifndef TREELITE_GTIL_PRED_TRANSFORM_H_
#define TREELITE_GTIL_PRED_TRANSFORM_H_

#include <string>
#include <cstddef>

namespace treelite {

class Model;

namespace gtil {

using PredTransformFuncType = std::size_t (*) (const treelite::Model&, const float*, float*);

PredTransformFuncType LookupPredTransform(const std::string& name);

}  // namespace gtil
}  // namespace treelite

#endif  // TREELITE_GTIL_PRED_TRANSFORM_H_
