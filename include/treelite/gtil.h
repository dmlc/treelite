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

/*!
 * \brief Predict with a DMatrix
 * \param model The model object
 * \param input The data matrix (sparse or dense)
 * \param output Pointer to buffer to store the output. Call GetPredictOutputSize() to get the
 *               amount of buffer you should allocate for this parameter.
 * \param nthread number of CPU threads to use. Set <= 0 to use all CPU cores.
 * \param pred_transform After computing the prediction score, whether to transform it.
 * \return Size of output. This could be smaller than GetPredictOutputSize() but could never be
 *         larger than GetPredictOutputSize().
 */
std::size_t Predict(const Model* model, const DMatrix* input, float* output, int nthread,
                    bool pred_transform);
/*!
 * \brief Predict with a 2D dense matrix
 * \param model The model object
 * \param input The data matrix, laid out in row-major layout
 * \param output Pointer to buffer to store the output. Call GetPredictOutputSize() to get the
 *               amount of buffer you should allocate for this parameter.
 * \param nthread number of CPU threads to use. Set <= 0 to use all CPU cores.
 * \param pred_transform After computing the prediction score, whether to transform it.
 * \return Size of output. This could be smaller than GetPredictOutputSize() but could never be
 *         larger than GetPredictOutputSize().
 */
std::size_t Predict(const Model* model, const float* input, std::size_t num_row, float* output,
                    int nthread, bool pred_transform);

// Query functions to allocate correct amount of memory for the output
std::size_t GetPredictOutputSize(const Model* model, std::size_t num_row);
std::size_t GetPredictOutputSize(const Model* model, const DMatrix* input);

}  // namespace gtil
}  // namespace treelite

#endif  // TREELITE_GTIL_H_
