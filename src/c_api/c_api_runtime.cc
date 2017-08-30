/*!
 * Copyright (c) 2017 by Contributors
 * \file c_api_runtime.cc
 * \author Philip Cho
 * \brief C API of tree-lite (runtime portion)
 */
 
#include <treelite/predictor.h>
#include <treelite/c_api_runtime.h>
#include "./c_api_error.h"
#include "./c_api_common.h"

using namespace treelite;

int TreelitePredictorLoad(const char* library_path,
                          PredictorHandle* out) {
  API_BEGIN();
  Predictor* predictor = new Predictor();
  predictor->Load(library_path);
  *out = static_cast<PredictorHandle>(predictor);
  API_END();
}

int TreelitePredictorPredictRaw(PredictorHandle handle,
                                DMatrixHandle dmat,
                                int nthread,
                                int verbose,
                                float* out_result) {
  API_BEGIN();
  const Predictor* predictor_ = static_cast<Predictor*>(handle);
  const DMatrix* dmat_ = static_cast<DMatrix*>(dmat);
  predictor_->PredictRaw(dmat_, nthread, verbose, out_result);
  API_END();
}

int TreelitePredictorPredict(PredictorHandle handle,
                             DMatrixHandle dmat,
                             int nthread,
                             int verbose,
                             float* out_result,
                             size_t* out_result_size) {
  API_BEGIN();
  const Predictor* predictor_ = static_cast<Predictor*>(handle);
  const DMatrix* dmat_ = static_cast<DMatrix*>(dmat);
  *out_result_size = predictor_->Predict(dmat_, nthread, verbose, out_result);
  API_END();
}

int TreelitePredictorQueryResultSize(PredictorHandle handle,
                                     DMatrixHandle dmat,
                                     size_t* out) {
  API_BEGIN();
  const Predictor* predictor_ = static_cast<Predictor*>(handle);
  const DMatrix* dmat_ = static_cast<DMatrix*>(dmat);
  *out = predictor_->QueryResultSize(dmat_);
  API_END();
}

int TreelitePredictorQueryNumOutputGroup(PredictorHandle handle, size_t* out) {
  API_BEGIN();
  const Predictor* predictor_ = static_cast<Predictor*>(handle);
  *out = predictor_->QueryNumOutputGroup();
  API_END();
}

int TreelitePredictorFree(PredictorHandle handle) {
  API_BEGIN();
  delete static_cast<Predictor*>(handle);
  API_END();
}