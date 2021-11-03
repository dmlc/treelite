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
/*! \brief handle to output from predictor */
typedef void* PredictorOutputHandle;

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
 * \brief Make predictions on a batch of data rows (synchronously). This function internally
 *        divides the workload among all worker threads.
 *
 *        Note. This function does not allocate the result vector. Use
 *        TreeliteCreatePredictorOutputVector() convenience function to allocate the vector of
 *        the right length and type.
 *
 *        Note. To access the element values from the output vector, you should cast the opaque
 *        handle (PredictorOutputHandle type) to an appropriate pointer LeafOutputType*, where
 *        the type is either float, double, or uint32_t. So carry out the following steps:
 *        1. Call TreelitePredictorQueryLeafOutputType() to obtain the type of the leaf output.
 *           It will return a string ("float32", "float64", or "uint32") representing the type.
 *        2. Depending on the type string, cast the output handle to float*, double*, or uint32_t*.
 *        3. Now access the array with the casted pointer. The array's length is given by
 *           TreelitePredictorQueryResultSize().
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
    PredictorOutputHandle out_result, size_t* out_result_size);
/*!
 * \brief Convenience function to allocate an output vector that is able to hold the prediction
 *        result for a given data matrix. The vector's length will be identical to
 *        TreelitePredictorQueryResultSize() and its type will be identical to
 *        TreelitePredictorQueryLeafOutputType(). To prevent memory leak, make sure to de-allocate
 *        the vector with TreeliteDeletePredictorOutputVector().
 *
 *        Note. To access the element values from the output vector, you should cast the opaque
 *        handle (PredictorOutputHandle type) to an appropriate pointer LeafOutputType*, where
 *        the type is either float, double, or uint32_t. So carry out the following steps:
 *        1. Call TreelitePredictorQueryLeafOutputType() to obtain the type of the leaf output.
 *           It will return a string ("float32", "float64", or "uint32") representing the type.
 *        2. Depending on the type string, cast the output handle to float*, double*, or uint32_t*.
 *        3. Now access the array with the casted pointer. The array's length is given by
 *           TreelitePredictorQueryResultSize().
 * \param handle predictor
 * \param batch the data matrix containing a batch of rows
 * \param out_output_vector Handle to the newly allocated output vector.
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteCreatePredictorOutputVector(
    PredictorHandle handle, DMatrixHandle batch, PredictorOutputHandle* out_output_vector);

/*!
 * \brief De-allocate an output vector
 * \param handle predictor
 * \param output_vector Output vector to delete from memory
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDeletePredictorOutputVector(
    PredictorHandle handle, PredictorOutputHandle output_vector);

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
 * \brief Get the number classes in the loaded model
 * The number is 1 for most tasks;
 * it is greater than 1 for multiclass classification.
 * \param handle predictor
 * \param out length of prediction array
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorQueryNumClass(PredictorHandle handle, size_t* out);
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
 * \brief Get c value of exponential standard ratio transformation used to train
 *        the loaded model
 * \param handle predictor
 * \param out C value of transformation
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreelitePredictorQueryRatioC(PredictorHandle handle, float* out);

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
