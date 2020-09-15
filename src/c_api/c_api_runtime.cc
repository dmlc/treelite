/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file c_api_runtime.cc
 * \author Hyunsu Cho
 * \brief C API of treelite (runtime portion)
 */

#include <treelite/predictor.h>
#include <treelite/c_api_runtime.h>
#include <treelite/c_api_error.h>
#include <dmlc/thread_local.h>
#include <string>
#include <cstring>

using namespace treelite;

namespace {

/*! \brief entry to to easily hold returning information */
struct TreeliteRuntimeAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
};

// thread-local store for returning strings
using TreeliteRuntimeAPIThreadLocalStore
  = dmlc::ThreadLocalStore<TreeliteRuntimeAPIThreadLocalEntry>;

}  // anonymous namespace

int TreelitePredictorLoad(const char* library_path, int num_worker_thread, PredictorHandle* out) {
  API_BEGIN();
  auto predictor = std::make_unique<predictor::Predictor>(num_worker_thread);
  predictor->Load(library_path);
  *out = static_cast<PredictorHandle>(predictor.release());
  API_END();
}

int TreelitePredictorPredictBatch(
    PredictorHandle handle, DMatrixHandle batch, int verbose, int pred_margin,
    PredictorOutputHandle out_result, size_t* out_result_size) {
  API_BEGIN();
  const auto* predictor = static_cast<const predictor::Predictor*>(handle);
  const auto* dmat = static_cast<const DMatrix*>(batch);
  const size_t num_feature = predictor->QueryNumFeature();
  const std::string err_msg
    = std::string("Too many columns (features) in the given batch. "
                  "Number of features must not exceed ") + std::to_string(num_feature);
  CHECK_LE(dmat->GetNumCol(), num_feature) << err_msg;
  *out_result_size = predictor->PredictBatch(dmat, verbose, (pred_margin != 0), out_result);
  API_END();
}

int TreeliteCreatePredictorOutputVector(
    PredictorHandle handle, DMatrixHandle batch, PredictorOutputHandle* out_output_vector) {
  API_BEGIN();
  const auto* predictor = static_cast<const predictor::Predictor*>(handle);
  const auto* dmat = static_cast<const DMatrix*>(batch);
  *out_output_vector = predictor->CreateOutputVector(dmat);
  API_END();
}

int TreeliteDeletePredictorOutputVector(
    PredictorHandle handle, PredictorOutputHandle output_vector) {
  API_BEGIN();
  const auto* predictor = static_cast<const predictor::Predictor*>(handle);
  predictor->DeleteOutputVector(output_vector);
  API_END();
}

int TreelitePredictorQueryResultSize(PredictorHandle handle, DMatrixHandle batch, size_t* out) {
  API_BEGIN();
  const auto* predictor = static_cast<const predictor::Predictor*>(handle);
  const auto* dmat = static_cast<const DMatrix*>(batch);
  *out = predictor->QueryResultSize(dmat);
  API_END();
}

int TreelitePredictorQueryNumOutputGroup(PredictorHandle handle, size_t* out) {
  API_BEGIN();
  const auto* predictor = static_cast<const predictor::Predictor*>(handle);
  *out = predictor->QueryNumOutputGroup();
  API_END();
}

int TreelitePredictorQueryNumFeature(PredictorHandle handle, size_t* out) {
  API_BEGIN();
  const auto* predictor = static_cast<const predictor::Predictor*>(handle);
  *out = predictor->QueryNumFeature();
  API_END();
}

int TreelitePredictorQueryPredTransform(PredictorHandle handle, const char** out) {
  API_BEGIN()
  const auto* predictor = static_cast<const predictor::Predictor*>(handle);
  auto pred_transform = predictor->QueryPredTransform();
  std::string& ret_str = TreeliteRuntimeAPIThreadLocalStore::Get()->ret_str;
  ret_str = pred_transform;
  *out = ret_str.c_str();
  API_END();
}

int TreelitePredictorQuerySigmoidAlpha(PredictorHandle handle, float* out) {
  API_BEGIN()
  const auto* predictor = static_cast<const predictor::Predictor*>(handle);
  *out = predictor->QuerySigmoidAlpha();
  API_END();
}

int TreelitePredictorQueryGlobalBias(PredictorHandle handle, float* out) {
  API_BEGIN()
  const auto* predictor = static_cast<const predictor::Predictor*>(handle);
  *out = predictor->QueryGlobalBias();
  API_END();
}

int TreelitePredictorQueryThresholdType(PredictorHandle handle, const char** out) {
  API_BEGIN()
  const auto* predictor = static_cast<const predictor::Predictor*>(handle);
  std::string& ret_str = TreeliteRuntimeAPIThreadLocalStore::Get()->ret_str;
  ret_str = TypeInfoToString(predictor->QueryThresholdType());
  *out = ret_str.c_str();
  API_END();
}

int TreelitePredictorQueryLeafOutputType(PredictorHandle handle, const char** out) {
  API_BEGIN()
  const auto* predictor = static_cast<const predictor::Predictor*>(handle);
  std::string& ret_str = TreeliteRuntimeAPIThreadLocalStore::Get()->ret_str;
  ret_str = TypeInfoToString(predictor->QueryLeafOutputType());
  *out = ret_str.c_str();
  API_END();
}

int TreelitePredictorFree(PredictorHandle handle) {
  API_BEGIN();
  delete static_cast<predictor::Predictor*>(handle);
  API_END();
}
