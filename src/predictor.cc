/*!
 * Copyright (c) 2017 by Contributors
 * \file predictor.cc
 * \author Philip Cho
 * \brief Load prediction function exported as a shared library
 */

#include <treelite/predictor.h>
#include <treelite/omp.h>
#include <dmlc/logging.h>
#include <dmlc/timer.h>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <functional>
#include "common/math.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

inline treelite::Predictor::LibraryHandle OpenLibrary(const char* name) {
#ifdef _WIN32
  HMODULE handle = LoadLibraryA(name);
#else
  void* handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
#endif
  return static_cast<treelite::Predictor::LibraryHandle>(handle);
}

inline void CloseLibrary(treelite::Predictor::LibraryHandle handle) {
#ifdef _WIN32
  FreeLibrary(static_cast<HMODULE>(handle));
#else
  dlclose(static_cast<void*>(handle));
#endif
}

template <typename HandleType>
inline HandleType LoadFunction(treelite::Predictor::LibraryHandle lib_handle,
                               const char* name) {
#ifdef _WIN32
  FARPROC func_handle = GetProcAddress(static_cast<HMODULE>(lib_handle), name);
#else
  void* func_handle = dlsym(static_cast<void*>(lib_handle), name);
#endif
  return static_cast<HandleType>(func_handle);
}

template <typename PredFunc>
inline void PredLoop(const treelite::CSRBatch* batch, int nthread, int verbose,
                     float* out_pred, PredFunc func) {
  std::vector<treelite::Predictor::Entry> inst(nthread * batch->num_col, {-1});
  CHECK(sizeof(size_t) < sizeof(int64_t)
        || batch->num_row
           <= static_cast<size_t>(std::numeric_limits<int64_t>::max()));
  const int64_t num_row = static_cast<int64_t>(batch->num_row);
  const size_t num_col = batch->num_col;
  const float* data = batch->data;
  const uint32_t* col_ind = batch->col_ind;
  const size_t* row_ptr = batch->row_ptr;
  #pragma omp parallel for schedule(static) num_threads(nthread) \
    default(none) firstprivate(num_row, num_col, data, col_ind, row_ptr) \
    shared(inst, func, out_pred)
  for (int64_t rid = 0; rid < num_row; ++rid) {
    const int tid = omp_get_thread_num();
    const size_t off = num_col * tid;
    const size_t ibegin = row_ptr[rid];
    const size_t iend = row_ptr[rid + 1];
    for (size_t i = ibegin; i < iend; ++i) {
      inst[off + col_ind[i]].fvalue = data[i];
    }
    func(rid, &inst[off], out_pred);
    for (size_t i = ibegin; i < iend; ++i) {
      inst[off + col_ind[i]].missing = -1;
    }
  }
}

template <typename PredFunc>
inline void PredLoop(const treelite::DenseBatch* batch, int nthread,
                     int verbose, float* out_pred, PredFunc func) {
  const bool nan_missing
                      = treelite::common::math::CheckNAN(batch->missing_value);
  std::vector<treelite::Predictor::Entry> inst(nthread * batch->num_col, {-1});
  CHECK(sizeof(size_t) < sizeof(int64_t)
        || batch->num_row
           <= static_cast<size_t>(std::numeric_limits<int64_t>::max()));
  const int64_t num_row = static_cast<int64_t>(batch->num_row);
  const size_t num_col = batch->num_col;
  const float missing_value = batch->missing_value;
  const float* data = batch->data;
  const float* row;
  #pragma omp parallel for schedule(static) num_threads(nthread) \
    default(none) \
    firstprivate(num_row, num_col, data, missing_value, nan_missing) \
    private(row) shared(inst, func, out_pred)
  for (int64_t rid = 0; rid < num_row; ++rid) {
    const int tid = omp_get_thread_num();
    const size_t off = num_col * tid;
    row = &data[rid * num_col];
    for (size_t j = 0; j < num_col; ++j) {
      if (treelite::common::math::CheckNAN(row[j])) {
        CHECK(nan_missing)
          << "The missing_value argument must be set to NaN if there is any "
          << "NaN in the matrix.";
      } else if (nan_missing || row[j] != missing_value) {
        inst[off + j].fvalue = row[j];
      }
    }
    func(rid, &inst[off], out_pred);
    for (size_t j = 0; j < num_col; ++j) {
      if (inst[off + j].missing != -1) {
        inst[off + j].missing = -1;
      }
    }
  }
}

template <typename BatchType>
inline size_t PredictBatch_(const BatchType* batch, int nthread, int verbose,
                            bool pred_margin, size_t num_output_group,
                          treelite::Predictor::PredFuncHandle pred_func_handle,
       treelite::Predictor::PredTransformFuncHandle pred_transform_func_handle,
                            size_t query_result_size, float* out_pred) {
  CHECK(pred_func_handle != nullptr)
    << "A shared library needs to be loaded first using Load()";
  const int max_thread = omp_get_max_threads();
  nthread = (nthread == 0) ? max_thread : std::min(nthread, max_thread);

  if (verbose > 0) {
    LOG(INFO) << "Begin prediction";
  }
  double tstart = dmlc::GetTime();

  /* Pass the correct prediction function to PredLoop.
     We also need to specify how the function should be called. */
  if (num_output_group > 1) {
    using PredFunc = void (*)(treelite::Predictor::Entry*, float*);
    PredFunc pred_func = reinterpret_cast<PredFunc>(pred_func_handle);
    PredLoop(batch, nthread, verbose, out_pred,
      [pred_func, num_output_group]
      (int64_t rid, treelite::Predictor::Entry* inst, float* out_pred) {
        pred_func(inst, &out_pred[rid * num_output_group]);
      });
  } else {
    using PredFunc = float (*)(treelite::Predictor::Entry*);
    PredFunc pred_func = reinterpret_cast<PredFunc>(pred_func_handle);
    PredLoop(batch, nthread, verbose, out_pred,
      [pred_func]
      (int64_t rid, treelite::Predictor::Entry* inst, float* out_pred) {
        out_pred[rid] = pred_func(inst);
      });
  }
  if (verbose > 0) {
    LOG(INFO) << "Finished prediction in "
              << dmlc::GetTime() - tstart << " sec";
  }

  if (pred_margin) {
    return query_result_size;
  } else {
    using PredTransformFunc = size_t(*)(float*, int64_t, int);
    PredTransformFunc pred_transform_func
      = reinterpret_cast<PredTransformFunc>(pred_transform_func_handle);
    return pred_transform_func(out_pred, batch->num_row, nthread);
  }
}

}  // namespace anonymous

namespace treelite {

Predictor::Predictor() : lib_handle_(nullptr),
                         query_func_handle_(nullptr),
                         pred_func_handle_(nullptr),
                         pred_transform_func_handle_(nullptr) {}
Predictor::~Predictor() {
  Free();
}

void
Predictor::Load(const char* name) {
  lib_handle_ = OpenLibrary(name);
  CHECK(lib_handle_ != nullptr)
    << "Failed to load dynamic shared library `" << name << "'";

  /* 1. query # of output groups */
  query_func_handle_ = LoadFunction<QueryFuncHandle>(lib_handle_,
                                                     "get_num_output_group");
  using QueryFunc = size_t (*)(void);
  QueryFunc query_func = reinterpret_cast<QueryFunc>(query_func_handle_);
  CHECK(query_func != nullptr)
    << "Dynamic shared library `" << name
    << "' does not contain valid get_num_output_group() function";
  num_output_group_ = query_func();

  /* 2. load appropriate function for margin prediction */
  CHECK_GT(num_output_group_, 0) << "num_output_group cannot be zero";
  if (num_output_group_ > 1) {   // multi-class classification
    pred_func_handle_ = LoadFunction<PredFuncHandle>(lib_handle_,
                                                  "predict_margin_multiclass");
    using PredFunc = void (*)(Entry*, float*);
    PredFunc pred_func = reinterpret_cast<PredFunc>(pred_func_handle_);
    CHECK(pred_func != nullptr)
      << "Dynamic shared library `" << name
      << "' does not contain valid predict_margin_multiclass() function";
  } else {                      // everything else
    pred_func_handle_ = LoadFunction<PredFuncHandle>(lib_handle_,
                                                     "predict_margin");
    using PredFunc = float (*)(Entry*);
    PredFunc pred_func = reinterpret_cast<PredFunc>(pred_func_handle_);
    CHECK(pred_func != nullptr)
      << "Dynamic shared library `" << name
      << "' does not contain valid predict_margin() function";
  }

  /* 3. load prediction transform function */
  pred_transform_func_handle_
    = LoadFunction<PredTransformFuncHandle>(lib_handle_,
                                            "pred_transform_batch");
  using PredTransformFunc = size_t (*)(float*, int64_t, int);
  PredTransformFunc pred_transform_func
    = reinterpret_cast<PredTransformFunc>(pred_transform_func_handle_);
  CHECK(pred_transform_func != nullptr)
    << "Dynamic shared library `" << name
    << "' does not contain valid pred_transform_batch() function";
}

void
Predictor::Free() {
  CloseLibrary(lib_handle_);
}

size_t
Predictor::PredictBatch(const CSRBatch* batch, int nthread, int verbose,
                        bool pred_margin, float* out_result) const {
  return PredictBatch_(batch, nthread, verbose, pred_margin, num_output_group_,
                       pred_func_handle_, pred_transform_func_handle_,
                       QueryResultSize(batch), out_result);
}

size_t
Predictor::PredictBatch(const DenseBatch* batch, int nthread, int verbose,
                        bool pred_margin, float* out_result) const {
  return PredictBatch_(batch, nthread, verbose, pred_margin, num_output_group_,
                       pred_func_handle_, pred_transform_func_handle_,
                       QueryResultSize(batch), out_result);
}

}  // namespace treelite
