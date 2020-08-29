/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file pred_transform.h
 * \brief tools to define prediction transform function
 * \author Hyunsu Cho
 */
#ifndef TREELITE_COMPILER_PRED_TRANSFORM_H_
#define TREELITE_COMPILER_PRED_TRANSFORM_H_

#include <treelite/tree.h>
#include <vector>
#include <string>

namespace treelite {
namespace compiler {

std::string PredTransformFunction(const std::string& backend, const Model& model);

}  // namespace compiler
}  // namespace treelite

#endif  // TREELITE_COMPILER_PRED_TRANSFORM_H_
