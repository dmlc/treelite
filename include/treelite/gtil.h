/*!
 * Copyright (c) 2021-2023 by Contributors
 * \file gtil.h
 * \author Hyunsu Cho
 * \brief General Tree Inference Library (GTIL), providing a reference implementation for
 *        predicting with decision trees.
 */

#ifndef TREELITE_GTIL_H_
#define TREELITE_GTIL_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace treelite {

class Model;

namespace gtil {

/*! \brief Prediction type */
enum class PredictKind : std::int8_t {
  /*!
   * \brief Usual prediction method: sum over trees and apply post-processing.
   * Expected output dimensions: (num_row, num_target, max_num_class)
   */
  kPredictDefault = 0,
  /*!
   * \brief Sum over trees, but don't apply post-processing; get raw margin scores instead.
   * Expected output dimensions: (num_row, num_target, max_num_class)
   */
  kPredictRaw = 1,
  /*!
   * \brief Output one (integer) leaf ID per tree.
   * Expected output dimensions: (num_row, num_tree)
   */
  kPredictLeafID = 2,
  /*!
   * \brief Output one or more margin scores per tree.
   * Expected output dimensions: (num_row, num_tree, leaf_vector_shape[0] * leaf_vector_shape[1])
   */
  kPredictPerTree = 3
};

/*! \brief Configuration class */
struct Configuration {
  int nthread{0};  // use all threads by default
  PredictKind pred_kind{PredictKind::kPredictDefault};
  Configuration() = default;
  explicit Configuration(std::string const& config_json);
};

/*!
 * \brief Predict with dense data
 * \param model Treelite Model object
 * \param input The 2D data array, laid out in row-major layout
 * \param num_row Number of rows in the data matrix.
 * \param output Pointer to buffer to store the output. Call \ref GetOutputShape to get
 *               the amount of buffer you should allocate for this parameter.
 * \param config Configuration of GTIL predictor
 */
template <typename InputT>
void Predict(Model const& model, InputT const* input, std::uint64_t num_row, InputT* output,
    Configuration const& config);

/*!
 * \brief Predict with sparse data with CSR (compressed sparse row) layout.
 *
 * In the CSR layout, data[row_ptr[i]:row_ptr[i+1]] store the nonzero entries of row i, and
 * col_ind[row_ptr[i]:row_ptr[i+1]] stores the corresponding column indices.
 *
 * \param model Treelite Model object
 * \param data Nonzero elements in the data matrix
 * \param col_ind Feature indices. col_ind[i] indicates the feature index associated with data[i].
 * \param row_ptr Pointer to row headers. Length is [num_row] + 1.
 * \param num_row Number of rows in the data matrix.
 * \param output Pointer to buffer to store the output. Call \ref GetOutputShape to get
 *               the amount of buffer you should allocate for this parameter.
 * \param config Configuration of GTIL predictor
 */
template <typename InputT>
void PredictSparse(Model const& model, InputT const* data, std::uint64_t const* col_ind,
    std::uint64_t const* row_ptr, std::uint64_t num_row, InputT* output,
    Configuration const& config);

/*!
 * \brief Given a data matrix, query the necessary shape of array to hold predictions for all
 *        data points.
 * \param model Treelite Model object
 * \param num_row Number of rows in the input
 * \param config Configuration of GTIL predictor. Set this by calling \ref TreeliteGTILParseConfig.
 * \return Array shape
 */
std::vector<std::uint64_t> GetOutputShape(
    Model const& model, std::uint64_t num_row, Configuration const& config);

extern template void Predict<float>(
    Model const&, float const*, std::uint64_t, float*, Configuration const&);
extern template void Predict<double>(
    Model const&, double const*, std::uint64_t, double*, Configuration const&);
extern template void PredictSparse<float>(Model const&, float const*, std::uint64_t const*,
    std::uint64_t const*, std::uint64_t, float*, Configuration const&);
extern template void PredictSparse<double>(Model const&, double const*, std::uint64_t const*,
    std::uint64_t const*, std::uint64_t, double*, Configuration const&);

}  // namespace gtil
}  // namespace treelite

#endif  // TREELITE_GTIL_H_
