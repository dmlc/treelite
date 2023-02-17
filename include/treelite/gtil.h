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

struct GTILConfig {
  int nthread{0};  // use all threads by default
  bool pred_transform{true};
  GTILConfig() = default;
  explicit GTILConfig(const char* config_json);
};

/*!
 * \brief Predict with a DMatrix
 * \param model The model object
 * \param input The data matrix (sparse or dense)
 * \param output Pointer to buffer to store the output. Call GetPredictOutputSize() to get the
 *               amount of buffer you should allocate for this parameter.
 * \param config Configuration for GTIL Predictor
 * \return Size of output. This could be smaller than GetPredictOutputSize() but could never be
 *         larger than GetPredictOutputSize().
 */
std::size_t Predict(const Model* model, const DMatrix* input, float* output,
                    const GTILConfig& config);
/*!
 * \brief Predict with a 2D dense array
 * \param model The model object
 * \param input The 2D data array, laid out in row-major layout
 * \param num_row Number of rows in the data matrix.
 * \param output Pointer to buffer to store the output. Call GetPredictOutputSize() to get the
 *               amount of buffer you should allocate for this parameter.
 * \param config Configuration for GTIL Predictor
 * \return Size of output. This could be smaller than GetPredictOutputSize() but could never be
 *         larger than GetPredictOutputSize().
 */
std::size_t Predict(const Model* model, const float* input, std::size_t num_row, float* output,
                    const GTILConfig& config);
/*!
 * \brief Given a batch of data rows, query the necessary size of array to hold predictions for all
 *        data points.
 * \param model Treelite Model object
 * \param num_row Number of rows in the input
 * \param config Configuration for GTIL Predictor
 * \return Size of output buffer that should be allocated
 */
std::size_t GetPredictOutputSize(const Model* model, std::size_t num_row, const GTILConfig& config);
/*!
 * \brief Given a batch of data rows, query the necessary size of array to hold predictions for all
 *        data points.
 * \param model Treelite Model object
 * \param input The input matrix
 * \param config Configuration for GTIL Predictor
 * \return Size of output buffer that should be allocated
 */
std::size_t GetPredictOutputSize(const Model* model, const DMatrix* input,
                                 const GTILConfig& config);

}  // namespace gtil
}  // namespace treelite

#endif  // TREELITE_GTIL_H_
