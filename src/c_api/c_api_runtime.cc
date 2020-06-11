/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file c_api_runtime.cc
 * \author Hyunsu Cho
 * \brief C API of treelite (runtime portion)
 */

#include <treelite/predictor.h>
#include <treelite/c_api_runtime.h>
#include <dmlc/thread_local.h>
#include <string>
#include <cstring>
#include "./c_api_error.h"

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

int TreeliteAssembleSparseBatch(const float* data,
                                const uint32_t* col_ind,
                                const size_t* row_ptr,
                                size_t num_row, size_t num_col,
                                CSRBatchHandle* out) {
  API_BEGIN();
  CSRBatch* batch = new CSRBatch();
  batch->data = data;
  batch->col_ind = col_ind;
  batch->row_ptr = row_ptr;
  batch->num_row = num_row;
  batch->num_col = num_col;
  *out = static_cast<CSRBatchHandle>(batch);
  API_END();
}

int TreeliteDeleteSparseBatch(CSRBatchHandle handle) {
  API_BEGIN();
  delete static_cast<CSRBatch*>(handle);
  API_END();
}

int TreeliteAssembleDenseBatch(const float* data, float missing_value,
                               size_t num_row, size_t num_col,
                               DenseBatchHandle* out) {
  API_BEGIN();
  DenseBatch* batch = new DenseBatch();
  batch->data = data;
  batch->missing_value = missing_value;
  batch->num_row = num_row;
  batch->num_col = num_col;
  *out = static_cast<DenseBatchHandle>(batch);
  API_END();
}

int TreeliteDeleteDenseBatch(DenseBatchHandle handle) {
  API_BEGIN();
  delete static_cast<DenseBatch*>(handle);
  API_END();
}

int TreeliteBatchGetDimension(void* handle,
                              int batch_sparse,
                              size_t* out_num_row,
                              size_t* out_num_col) {
  API_BEGIN();
  if (batch_sparse) {
    const CSRBatch* batch_ = static_cast<CSRBatch*>(handle);
    *out_num_row = batch_->num_row;
    *out_num_col = batch_->num_col;
  } else {
    const DenseBatch* batch_ = static_cast<DenseBatch*>(handle);
    *out_num_row = batch_->num_row;
    *out_num_col = batch_->num_col;
  }
  API_END();
}

int TreelitePredictorLoad(const char* library_path,
                          int num_worker_thread,
                          PredictorHandle* out) {
  API_BEGIN();
  Predictor* predictor = new Predictor(num_worker_thread);
  predictor->Load(library_path);
  *out = static_cast<PredictorHandle>(predictor);
  API_END();
}

int TreelitePredictorPredictBatch(PredictorHandle handle,
                                  void* batch,
                                  int batch_sparse,
                                  int verbose,
                                  int pred_margin,
                                  float* out_result,
                                  size_t* out_result_size) {
  API_BEGIN();
  Predictor* predictor_ = static_cast<Predictor*>(handle);
  const size_t num_feature = predictor_->QueryNumFeature();
  const std::string err_msg
    = std::string("Too many columns (features) in the given batch. "
                  "Number of features must not exceed ")
      + std::to_string(num_feature);
  if (batch_sparse) {
    const CSRBatch* batch_ = static_cast<CSRBatch*>(batch);
    CHECK_LE(batch_->num_col, num_feature) << err_msg;
    *out_result_size = predictor_->PredictBatch(batch_, verbose,
                                               (pred_margin != 0), out_result);
  } else {
    const DenseBatch* batch_ = static_cast<DenseBatch*>(batch);
    CHECK_LE(batch_->num_col, num_feature) << err_msg;
    *out_result_size = predictor_->PredictBatch(batch_, verbose,
                                               (pred_margin != 0), out_result);
  }
  API_END();
}

int TreelitePredictorPredictInst(PredictorHandle handle,
                                 union TreelitePredictorEntry* inst,
                                 int pred_margin,
                                 float* out_result, size_t* out_result_size) {
  API_BEGIN();
  Predictor* predictor_ = static_cast<Predictor*>(handle);
  *out_result_size
    = predictor_->PredictInst(inst, (pred_margin != 0), out_result);
  API_END();
}

int TreelitePredictorQueryResultSize(PredictorHandle handle,
                                     void* batch,
                                     int batch_sparse,
                                     size_t* out) {
  API_BEGIN();
  const Predictor* predictor_ = static_cast<Predictor*>(handle);
  if (batch_sparse) {
    const CSRBatch* batch_ = static_cast<CSRBatch*>(batch);
    *out = predictor_->QueryResultSize(batch_);
  } else {
    const DenseBatch* batch_ = static_cast<DenseBatch*>(batch);
    *out = predictor_->QueryResultSize(batch_);
  }
  API_END();
}

int TreelitePredictorQueryResultSizeSingleInst(PredictorHandle handle,
                                               size_t* out) {
  API_BEGIN();
  const Predictor* predictor_ = static_cast<Predictor*>(handle);
  *out = predictor_->QueryResultSizeSingleInst();
  API_END();
}

int TreelitePredictorQueryNumOutputGroup(PredictorHandle handle, size_t* out) {
  API_BEGIN();
  const Predictor* predictor_ = static_cast<Predictor*>(handle);
  *out = predictor_->QueryNumOutputGroup();
  API_END();
}

int TreelitePredictorQueryNumFeature(PredictorHandle handle, size_t* out) {
  API_BEGIN();
  const Predictor* predictor_ = static_cast<Predictor*>(handle);
  *out = predictor_->QueryNumFeature();
  API_END();
}

int TreelitePredictorQueryPredTransform(PredictorHandle handle, const char** out) {
  API_BEGIN()
  const Predictor* predictor_ = static_cast<Predictor*>(handle);
  auto pred_transform = predictor_->QueryPredTransform();
  std::string& ret_str = TreeliteRuntimeAPIThreadLocalStore::Get()->ret_str;
  ret_str = pred_transform;
  *out = ret_str.c_str();
  API_END();
}

int TreelitePredictorQuerySigmoidAlpha(PredictorHandle handle, float* out) {
  API_BEGIN()
  const Predictor* predictor_ = static_cast<Predictor*>(handle);
  *out = predictor_->QuerySigmoidAlpha();
  API_END();
}

int TreelitePredictorQueryGlobalBias(PredictorHandle handle, float* out) {
  API_BEGIN()
  const Predictor* predictor_ = static_cast<Predictor*>(handle);
  *out = predictor_->QueryGlobalBias();
  API_END();
}

int TreelitePredictorFree(PredictorHandle handle) {
  API_BEGIN();
  delete static_cast<Predictor*>(handle);
  API_END();
}
