/*!
 * Copyright (c) 2017 by Contributors
 * \file c_api_runtime.h
 * \author Philip Cho
 * \brief C API of treelite, used for interfacing with other languages
 *        This header is used exclusively by the runtime
 */

/* Note: Make sure to use slash-asterisk form of comments in this file
   (like this one). Do not use double-slash (//). */
 
#ifndef TREELITE_C_API_RUNTIME_H_
#define TREELITE_C_API_RUNTIME_H_

#include "c_api_common.h"

/*!
 * \addtogroup opaque_handles
 * opaque handles
 * \{
 */
/*! \brief handle to predictor class */
typedef void* PredictorHandle;
/*! \brief handle to batch of sparse data rows */
typedef void* CSRBatchHandle;
/*! \brief handle to batch of dense data rows */
typedef void* DenseBatchHandle;
/*! \} */

/*!
 * \defgroup predictor
 * Predictor interface
 * \{
 */
/*!
 * \brief assemble a sparse batch
 * \param data feature values
 * \param col_ind feature indices
 * \param row_ptr pointer to row headers
 * \param num_row number of data rows in the batch
 * \param num_col number of columns (features) in the batch
 * \param out handle to sparse batch
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteAssembleSparseBatch(const float* data,
                                             const uint32_t* col_ind,
                                             const size_t* row_ptr,
                                             size_t num_row, size_t num_col,
                                             CSRBatchHandle* out);
/*!
 * \brief delete a sparse batch from memory
 * \param handle sparse batch
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDeleteSparseBatch(CSRBatchHandle handle);
/*!
 * \brief assemble a dense batch
 * \param data feature values
 * \param missing_value value to represent the missing value
 * \param num_row number of data rows in the batch
 * \param num_col number of columns (features) in the batch
 * \param out handle to sparse batch
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteAssembleDenseBatch(const float* data,
                                            float missing_value,
                                            size_t num_row, size_t num_col,
                                            DenseBatchHandle* out);
/*!
 * \brief delete a dense batch from memory
 * \param handle dense batch
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDeleteDenseBatch(DenseBatchHandle handle);

/*!
 * \brief get dimensions of a batch
 * \param handle a batch of rows (must be of type SparseBatch or DenseBatch)
 * \param batch_sparse whether the batch is sparse (true) or dense (false)
 * \param out_num_row used to set number of rows
 * \param out_num_col used to set number of columns
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteBatchGetDimension(void* handle,
                                           int batch_sparse,
                                           size_t* out_num_row,
                                           size_t* out_num_col);

/*!
 * \brief load prediction code into memory.
 * This function assumes that the prediction code has been already compiled into
 * a dynamic shared library object (.so/.dll/.dylib).
 * \param library_path path to library object file containing prediction code
 * \param out handle to predictor
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorLoad(const char* library_path,
                                       PredictorHandle* out);
/*!
 * \brief make predictions on a batch of data rows
 * \param handle predictor
 * \param batch a batch of rows (must be of type SparseBatch or DenseBatch)
 * \param batch_sparse whether batch is sparse (1) or dense (0)
 * \param nthread number of threads to use
 * \param verbose whether to produce extra messages
 * \param pred_margin whether to produce raw margin scores instead of
 *                    transformed probabilities
 * \param out_result resulting output vector; use
 *                   TreelitePredictorQueryResultSize() to allocate sufficient
 *                   space
 * \param out_result_size used to save length of the output vector,
 *                        which is guaranteed to be less than or equal to
 *                        TreelitePredictorQueryResultSize()
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorPredictBatch(PredictorHandle handle,
                                               void* batch,
                                               int batch_sparse,
                                               int nthread,
                                               int verbose,
                                               int pred_margin,
                                               float* out_result,
                                               size_t* out_result_size);

/*!
 * \brief Given a batch of data rows, query the necessary size of array to
 *        hold predictions for all data points.
 * \param handle predictor
 * \param batch a batch of rows (must be of type SparseBatch or DenseBatch)
 * \param batch_sparse whether batch is sparse (1) or dense (0)
 * \param out used to store the length of prediction array
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorQueryResultSize(PredictorHandle handle,
                                                  void* batch,
                                                  int batch_sparse,
                                                  size_t* out);
/*!
 * \brief Get the number of output groups in the loaded model
 * The number is 1 for most tasks;
 * it is greater than 1 for multiclass classifcation.
 * \param handle predictor
 * \param out length of prediction array
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorQueryNumOutputGroup(PredictorHandle handle,
                                                      size_t* out);
/*!
 * \brief delete predictor from memory
 * \param handle predictor to remove
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorFree(PredictorHandle handle);
/*! \} */

#endif  // TREELITE_C_API_RUNTIME_H_
