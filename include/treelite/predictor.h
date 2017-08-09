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
  /*! \brief type alias for prediction function */
  using PredFunc = float (*)(Entry*);

  Predictor();
  ~Predictor();
  /*!
   * \brief load the prediction function from dynamic shared library (.so/.dll).
   *        The library must contain a function with the signature
   * \code
   *        float predict_margin(union Entry*);
   * \endcode
   * \param name name of dynamic shared library (.so/.dll).
   */
  void Load(const char* name);
  /*!
   * \brief unload the prediction function
   */
  void Free();
  /*!
   * \brief make predictions on a given dataset.
   * \param dmat data matrix
   * \param nthread number of threads to use for predicting
   * \param verbose whether to produce extra messages
   * \param out_result used to save predictions
   */
  void Predict(const DMatrix* dmat, int nthread, int verbose,
               float* out_result);

  /*!
   * \brief get prediction function
   * \return function pointer pointing to the prediction function
   */
  inline PredFunc GetPredFunc() const {
    return func_;
  }

 private:
  void* lib_handle_;
  PredFunc func_;
};

}  // namespace treelite

#endif  // TREELITE_PREDICTOR_H_
