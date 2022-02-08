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

/*! \brief Predictor class to perform prediction with a Treelite model */
class Predictor {
 public:
  explicit Predictor(const Model& model) : model_(model) {}

  /*!
   * \brief Predict with a DMatrix
   * \param input The data matrix (sparse or dense)
   * \param output Pointer to buffer to store the output. Call GetPredictOutputSize() to get the
   *               amount of buffer you should allocate for this parameter.
   * \param nthread number of CPU threads to use. Set <= 0 to use all CPU cores.
   * \param pred_transform After computing the prediction score, whether to transform it.
   * \return Size of output. This could be smaller than GetPredictOutputSize() but could never be
   *         larger than GetPredictOutputSize().
   */
  std::size_t Predict(const DMatrix* input, float* output, int nthread, bool pred_transform);

  /*!
   * \brief Predict with a 2D dense matrix
   * \param input The data matrix, laid out in row-major layout
   * \param num_row Number of rows in the data matrix.
   * \param output Pointer to buffer to store the output. Call QueryResultSize() to get the
   *               amount of buffer you should allocate for this parameter.
   * \param nthread number of CPU threads to use. Set <= 0 to use all CPU cores.
   * \param pred_transform After computing the prediction score, whether to transform it.
   * \return Size of output. This could be smaller than GetPredictOutputSize() but could never be
   *         larger than GetPredictOutputSize().
   */
  std::size_t Predict(const float* input, std::size_t num_row, float* output, int nthread,
                      bool pred_transform);

  // Query functions to allocate correct amount of memory for the output
  /*!
   * \brief Given a batch of data rows, query the necessary size of array to
   *        hold predictions for all data points.
   * \param num_row Number of rows in the input
   * \return Size of output buffer that should be allocated.
   */
  std::size_t QueryResultSize(std::size_t num_row) const;
  /*!
   * \brief Given a batch of data rows, query the necessary size of array to
   *        hold predictions for all data points.
   * \param num_row The input matrix
   * \return Size of output buffer that should be allocated.
   */
  std::size_t QueryResultSize(const DMatrix* input) const;

 private:
  const Model& model_;
};

}  // namespace gtil
}  // namespace treelite

#endif  // TREELITE_GTIL_H_
