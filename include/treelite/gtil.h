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
#include <cstdint>
#include <vector>

namespace treelite {

class Model;
class DMatrix;

namespace gtil {

/*! \brief Prediction type */
enum class PredictType : std::int8_t {
  /*!
   * \brief Usual prediction method: sum over trees and apply post-processing.
   * Expected output dimensions:
   *  - (num_row,) for model type kBinaryClfRegr
   *  - (num_row,) for model type kMultiClfGrovePerClass, when pred_transform="max_index"
   *  - (num_row, num_class) for model type kMultiClfGrovePerClass / kMultiClfProbDistLeaf
   *
   * \note Most of the time, the dimensions of the prediction output is the same as the return
   * value of \ref GetPredictOutputSize, with one exception: when pred_transform is "max_index",
   * \ref GetPredictOutputSize returns (num_row * num_class), to hold the margin scores. The final
   * output dimension is (num_row,) since the class with the largest margin score is chosen for
   * each data row.
   */
  kPredictDefault = 0,
  /*!
   * \brief Sum over trees, but don't apply post-processing; get raw margin scores instead.
   * Expected output dimensions:
   *  - (num_row,) for model type kBinaryClfRegr
   *  - (num_row, num_class) for model type kMultiClfGrovePerClass / kMultiClfProbDistLeaf
   */
  kPredictRaw = 1,
  /*!
   * \brief Output one (integer) leaf ID per tree.
   * Expected output dimensions: (num_row, num_tree)
   */
  kPredictLeafID = 2,
  /*!
   * \brief Output one or more margin scores per tree.
   * Expected output dimensions:
   *  - (num_row, num_tree) for model type kBinaryClfRegr / kMultiClfGrovePerClass
   *  - (num_row, num_tree, num_class) for model type kMultiClfProbDistLeaf
   */
  kPredictPerTree = 3
};

/*! \brief Configuration class */
struct Configuration {
  int nthread{0};  // use all threads by default
  PredictType pred_type{PredictType::kPredictDefault};
  Configuration() = default;
  explicit Configuration(char const* config_json);
};

/*!
 * \brief Predict with a DMatrix
 * \param model The model object
 * \param input The data matrix (sparse or dense)
 * \param output Pointer to buffer to store the output. Call \ref GetPredictOutputSize to get the
 *               amount of buffer you should allocate for this parameter.
 * \param config Configuration for GTIL Predictor
 * \param output_shape Shape of output. The product of the elements of this array shall be equal to
 *                     the return value.
 * \return Size of output. This could be smaller than \ref GetPredictOutputSize but could never be
 *         larger than \ref GetPredictOutputSize.
 */
std::size_t Predict(Model const* model, DMatrix const* input, float* output,
    Configuration const& config, std::vector<std::size_t>& output_shape);
/*!
 * \brief Predict with a 2D dense array
 * \param model The model object
 * \param input The 2D data array, laid out in row-major layout
 * \param num_row Number of rows in the data matrix.
 * \param output Pointer to buffer to store the output. Call \ref GetPredictOutputSize to get the
 *               amount of buffer you should allocate for this parameter.
 * \param config Configuration for GTIL Predictor
 * \param output_shape Shape of output. The product of the elements of this array shall be equal to
 *                     the return value.
 * \return Size of output. This could be smaller than \ref GetPredictOutputSize but could never be
 *         larger than \ref GetPredictOutputSize.
 */
std::size_t Predict(Model const* model, float const* input, std::size_t num_row, float* output,
    Configuration const& config, std::vector<std::size_t>& output_shape);
/*!
 * \brief Given a batch of data rows, query the necessary size of array to hold predictions for all
 *        data points.
 * \param model Treelite Model object
 * \param num_row Number of rows in the input
 * \param config Configuration for GTIL Predictor
 * \return Size of output buffer that should be allocated
 */
std::size_t GetPredictOutputSize(
    Model const* model, std::size_t num_row, Configuration const& config);
/*!
 * \brief Given a batch of data rows, query the necessary size of array to hold predictions for all
 *        data points.
 * \param model Treelite Model object
 * \param input The input matrix
 * \param config Configuration for GTIL Predictor
 * \return Size of output buffer that should be allocated
 */
std::size_t GetPredictOutputSize(
    Model const* model, DMatrix const* input, Configuration const& config);

}  // namespace gtil
}  // namespace treelite

#endif  // TREELITE_GTIL_H_
