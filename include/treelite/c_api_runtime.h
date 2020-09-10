/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file c_api_runtime.h
 * \author Hyunsu Cho
 * \brief C API of Treelite, used for interfacing with other languages
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
/*! \} */

/*!
 * \defgroup predictor
 * Predictor interface
 * \{
 */

/*!
 * \brief load prediction code into memory.
 * This function assumes that the prediction code has been already compiled into
 * a dynamic shared library object (.so/.dll/.dylib).
 * \param library_path path to library object file containing prediction code
 * \param num_worker_thread number of worker threads (-1 to use max number)
 * \param out handle to predictor
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorLoad(
    const char* library_path, int num_worker_thread, PredictorHandle* out);
/*!
 * \brief Make predictions on a batch of data rows (synchronously). This
 *        function internally divides the workload among all worker threads.
 * \param handle predictor
 * \param batch the data matrix containing a batch of rows
 * \param verbose whether to produce extra messages
 * \param pred_margin whether to produce raw margin scores instead of
 *                    transformed probabilities
 * \param out_result Resulting output vector. This pointer must point to an array of length
 *                   TreelitePredictorQueryResultSize() and of type
 *                   TreelitePredictorQueryLeafOutputType().
 * \param out_result_size used to save length of the output vector,
 *                        which is guaranteed to be less than or equal to
 *                        TreelitePredictorQueryResultSize()
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorPredictBatch(
    PredictorHandle handle, DMatrixHandle batch, int verbose, int pred_margin,
    void* out_result, size_t* out_result_size);
/*!
 * \brief Given a batch of data rows, query the necessary size of array to
 *        hold predictions for all data points.
 * \param handle predictor
 * \param batch the data matrix containing a batch of rows
 * \param out used to store the length of prediction array
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorQueryResultSize(
    PredictorHandle handle, DMatrixHandle batch, size_t* out);
/*!
 * \brief Get the number of output groups in the loaded model
 * The number is 1 for most tasks;
 * it is greater than 1 for multiclass classifcation.
 * \param handle predictor
 * \param out length of prediction array
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorQueryNumOutputGroup(PredictorHandle handle, size_t* out);
/*!
 * \brief Get the width (number of features) of each instance used to train
 *        the loaded model
 * \param handle predictor
 * \param out number of features
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorQueryNumFeature(PredictorHandle handle, size_t* out);

/*!
 * \brief Get name of post prediction transformation used to train
 *        the loaded model
 * \param handle predictor
 * \param out name of post prediction transformation
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorQueryPredTransform(PredictorHandle handle, const char** out);
/*!
 * \brief Get alpha value of sigmoid transformation used to train
 *        the loaded model
 * \param handle predictor
 * \param out alpha value of sigmoid transformation
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorQuerySigmoidAlpha(PredictorHandle handle, float* out);

/*!
 * \brief Get global bias which adjusting predicted margin scores
 * \param handle predictor
 * \param out global bias value
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorQueryGlobalBias(PredictorHandle handle, float* out);

TREELITE_DLL int TreelitePredictorQueryThresholdType(PredictorHandle handle, const char** out);
TREELITE_DLL int TreelitePredictorQueryLeafOutputType(PredictorHandle handle, const char** out);
/*!
 * \brief delete predictor from memory
 * \param handle predictor to remove
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorFree(PredictorHandle handle);
/*! \} */

#endif  // TREELITE_C_API_RUNTIME_H_
