/*!
 * Copyright (c) 2017 by Contributors
 * \file predictor.h
 * \author Philip Cho
 * \brief Load prediction function exported as a shared library
 */
#ifndef TREELITE_PREDICTOR_H_
#define TREELITE_PREDICTOR_H_

#include <treelite/data.h>

namespace treelite {

/*! \brief predictor class: wrapper for optimized prediction code */
class Predictor {
 public:
  /*! \brief data layout. The value -1 signifies the missing value.
      When the "missing" field is set to -1, the "fvalue" field is set to
      NaN (Not a Number), so there is no danger for mistaking between
      missing values and non-missing values. */
  union Entry {
    int missing;
    float fvalue;
    // may contain extra fields later, such as qvalue
  };

  /*! \brief opaque handle types */
  typedef void* QueryFuncHandle;
  typedef void* PredFuncHandle;
  typedef void* PredTransformFuncHandle;
  typedef void* LibraryHandle;

  Predictor();
  ~Predictor();
  /*!
   * \brief load the prediction function from dynamic shared library.
   * \param name name of dynamic shared library (.so/.dll/.dylib).
   */
  void Load(const char* name);
  /*!
   * \brief unload the prediction function
   */
  void Free();
  /*!
   * \brief make predictions on a given dataset and output raw margin scores
   * \param dmat data matrix
   * \param nthread number of threads to use for predicting
   * \param verbose whether to produce extra messages
   * \param out_result resulting margin vector; use QueryResultSize() to
   *                   allocate sufficient space. The margin vector is always
   *                   as long as QueryResultSize().
   */
  void PredictRaw(const DMatrix* dmat, int nthread, int verbose,
                  float* out_result) const;

  /*!
   * \brief make predictions on a given dataset and output probabilities
   * \param dmat data matrix
   * \param nthread number of threads to use for predicting
   * \param verbose whether to produce extra messages
   * \param out_result resulting output probability vector; use
   *                   QueryResultSize() to allocate sufficient space
   * \return length of the output probability vector, which is less than or
   *         equal to QueryResultSize()
   */
  size_t Predict(const DMatrix* dmat, int nthread, int verbose,
                 float* out_result) const;

  /*!
   * \brief Given a data matrix, query the necessary size of array to
   *        hold predictions for all data points.
   * \param dmat data matrix
   * \return length of prediction array
   */
  inline size_t QueryResultSize(const DMatrix* dmat) const {
    CHECK(pred_func_handle_ != nullptr)
      << "A shared library needs to be loaded first using Load()";
    return dmat->num_row * num_output_group_;
  }
  /*!
   * \brief Get the number of output groups in the loaded model
   * The number is 1 for most tasks;
   * it is greater than 1 for multiclass classifcation.
   * \return length of prediction array
   */
  inline size_t QueryNumOutputGroup() const {
    return num_output_group_;
  }

 private:
  LibraryHandle lib_handle_;
  QueryFuncHandle query_func_handle_;
  PredFuncHandle pred_func_handle_;
  PredTransformFuncHandle pred_transform_func_handle_;
  size_t num_output_group_;
};

}  // namespace treelite

#endif  // TREELITE_PREDICTOR_H_
