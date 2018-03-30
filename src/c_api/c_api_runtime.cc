/*!
 * Copyright (c) 2017 by Contributors
 * \file c_api_runtime.cc
 * \author Philip Cho
 * \brief C API of treelite (runtime portion)
 */
 
#include <treelite/predictor.h>
#include <treelite/c_api_runtime.h>
#include "./c_api_error.h"

using namespace treelite;

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
                          int include_master_thread,
                          PredictorHandle* out) {
  API_BEGIN();
  Predictor* predictor = new Predictor(num_worker_thread,
                                       (bool)include_master_thread);
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
  if (batch_sparse) {
    const CSRBatch* batch_ = static_cast<CSRBatch*>(batch);
    *out_result_size = predictor_->PredictBatch(batch_, verbose,
                                               (pred_margin != 0), out_result);
  } else {
    const DenseBatch* batch_ = static_cast<DenseBatch*>(batch);
    *out_result_size = predictor_->PredictBatch(batch_, verbose,
                                               (pred_margin != 0), out_result);
  }
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
