/*!
 * Copyright (c) 2021 by Contributors
 * \file gtil.h
 * \author Hyunsu Cho
 * \brief General Tree Inference Library (GTIL), providing a reference implementation for
 *        predicting with decision trees. GTIL is useful in cases it is infeasible to build the
 *        tree models as native shared libs.
 */

#ifndef TREELITE_GTIL_H_
#define TREELITE_GTIL_H_

#include <cstddef>

namespace treelite {

class Model;
class DMatrix;

namespace gtil {

// Predict with a DMatrix (can be sparse or dense)
std::size_t Predict(const Model* model, const DMatrix* input, float* output,
                    bool pred_transform = true);
// Predict with 2D dense matrix
std::size_t Predict(const Model* model, const float* input, std::size_t num_row, float* output,
                    bool pred_transform = true);

// Query functions to allocate correct amount of memory for the output
std::size_t GetPredictOutputSize(const Model* model, std::size_t num_row);
std::size_t GetPredictOutputSize(const Model* model, const DMatrix* input);

}  // namespace gtil
}  // namespace treelite

#endif  // TREELITE_GTIL_H_
