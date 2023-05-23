/*!
 * Copyright (c) 2021 by Contributors
 * \file pred_transform.h
 * \author Hyunsu Cho
 * \brief Functions to post-process prediction results
 */

#ifndef SRC_GTIL_PRED_TRANSFORM_H_
#define SRC_GTIL_PRED_TRANSFORM_H_

#include <cstddef>
#include <string>

namespace treelite {

class Model;

namespace gtil {

using PredTransformFuncType = std::size_t (*)(treelite::Model const&, float const*, float*);

PredTransformFuncType LookupPredTransform(std::string const& name);

}  // namespace gtil
}  // namespace treelite

#endif  // SRC_GTIL_PRED_TRANSFORM_H_
