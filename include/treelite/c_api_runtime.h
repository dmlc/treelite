/*!
 * Copyright (c) 2017 by Contributors
 * \file c_api_runtime.h
 * \author Philip Cho
 * \brief C API of tree-lite, used for interfacing with other languages
 *        This header is used exclusively by the runtime
 */

/* Note: Make sure to use slash-asterisk form of comments in this file
   (like this one). Do not use double-slash (//). */
 
#ifndef TREELITE_C_API_RUNTIME_H_
#define TREELITE_C_API_RUNTIME_H_

#include "c_api_common.h"

/* opaque handles */
typedef void* PredictorHandle;

/***************************************************************************
 * Part 1: predictor interface
 ***************************************************************************/
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
 * \brief make predictions on a given dataset and output raw margin scores
 * \param handle predictor
 * \param dmat data matrix
 * \param nthread number of threads to use
 * \param verbose whether to produce extra messages
 * \param out_result resulting margin vector; use
 *                   TreelitePredictorQueryResultSize() to allocate sufficient
 *                   space. The margin vector is always as long as
 *                   TreelitePredictorQueryResultSize().
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorPredictRaw(PredictorHandle handle,
                                             DMatrixHandle dmat,
                                             int nthread,
                                             int verbose,
                                             float* out_result);
/*!
 * \brief make predictions on a dataset and output probabilities
 * \param handle predictor
 * \param dmat data matrix
 * \param nthread number of threads to use
 * \param verbose whether to produce extra messages
 * \param out_result resulting output probability vector; use
 *                   TreelitePredictorQueryResultSize() to allocate sufficient
 *                   space
 * \param out_result_size used to save length of the output probability vector,
 *                        which is less than or equal to
 *                        TreelitePredictorQueryResultSize()
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorPredict(PredictorHandle handle,
                                          DMatrixHandle dmat,
                                          int nthread,
                                          int verbose,
                                          float* out_result,
                                          size_t* out_result_size);

/*!
 * \brief Given a data matrix, query the necessary size of array to
 *        hold predictions for all data points.
 * \param handle predictor
 * \param dmat data matrix
 * \param out used to store the length of prediction array
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorQueryResultSize(PredictorHandle handle,
                                                  DMatrixHandle dmat,
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

#endif  // TREELITE_C_API_RUNTIME_H_