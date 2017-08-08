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

class Predictor {
 public:
  // data layout
  union Entry {
    int missing;
    float fvalue;
    // may contain extra fields later, such as qvalue
  };
  using PredFunc = float (*)(Entry*);

  Predictor();
  ~Predictor();
  void Load(const char* name);
  void Free();
  void Predict(const DMatrix* dmat, int nthread, int verbose,
               float* out_result);

  inline PredFunc GetPredFunc() const {
    return func_;
  }

 private:
  void* lib_handle_;
  PredFunc func_;
};

}  // namespace treelite

#endif  // TREELITE_PREDICTOR_H_
