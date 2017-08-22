/*!
 * Copyright (c) 2017 by Contributors
 * \file predictor.cc
 * \author Philip Cho
 * \brief Load prediction function exported as a shared library
 */

#include <treelite/predictor.h>
#include <dmlc/logging.h>
#include <dmlc/timer.h>
#include <omp.h>
#include <cstdint>
#include <limits>
#include <functional>

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
inline void PredLoopInternal(const treelite::DMatrix* dmat,
                             size_t rbegin, size_t rend, int nthread,
                             treelite::Predictor::Entry* inst,
                             float* out_pred, PredFunc func) {
  CHECK_LE(rbegin, rend);
  CHECK_LT(static_cast<int64_t>(rend), std::numeric_limits<int64_t>::max());
  const int64_t rbegin_i = static_cast<int64_t>(rbegin);
  const int64_t rend_i = static_cast<int64_t>(rend);
  #pragma omp parallel for schedule(static) num_threads(nthread)
  for (int64_t rid = rbegin_i; rid < rend_i; ++rid) {
    const int tid = omp_get_thread_num();
    const size_t off = dmat->num_col * tid;
    const size_t ibegin = dmat->row_ptr[rid];
    const size_t iend = dmat->row_ptr[rid + 1];
    for (size_t i = ibegin; i < iend; ++i) {
      inst[off + dmat->col_ind[i]].fvalue = dmat->data[i];
    }
    func(rid, &inst[off], out_pred);
    for (size_t i = ibegin; i < iend; ++i) {
      inst[off + dmat->col_ind[i]].missing = -1;
    }
  }
}

template <typename PredFunc>
inline void PredLoop(const treelite::DMatrix* dmat, int nthread, int verbose,
                     float* out_pred, PredFunc func) {
  // interval to display progress
  const size_t pstep = (dmat->num_row + 19) / 20;
  std::vector<treelite::Predictor::Entry> inst(nthread * dmat->num_col, {-1});
  for (size_t rbegin = 0; rbegin < dmat->num_row; rbegin += pstep) {
    const size_t rend = std::min(rbegin + pstep, dmat->num_row);
    PredLoopInternal(dmat, rbegin, rend, nthread, &inst[0], out_pred, func);
    if (verbose > 0) {
      LOG(INFO) << rend << " of " << dmat->num_row << " rows processed";
    }
  }
}

}  // namespace anonymous

namespace treelite {

Predictor::Predictor() : lib_handle_(nullptr),
                         query_func_handle_(nullptr),
                         pred_func_handle_(nullptr) {}
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
}

void
Predictor::Free() {
  CloseLibrary(lib_handle_);
}

void
Predictor::Predict(const DMatrix* dmat, int nthread, int verbose,
                   float* out_pred) const {
  CHECK(pred_func_handle_ != nullptr)
    << "A shared library needs to be loaded first using Load()";
  const int max_thread = omp_get_max_threads();
  nthread = (nthread == 0) ? max_thread : std::min(nthread, max_thread);

  if (verbose > 0) {
    LOG(INFO) << "Begin prediction";
  }
  double tstart = dmlc::GetTime();

  /* Pass the correct prediction function to PredLoop.
     We also need to specify how the function should be called. */
  if (num_output_group_ > 1) {
    using PredFunc = void (*)(Entry*, float*);
    PredFunc pred_func = reinterpret_cast<PredFunc>(pred_func_handle_);
    const size_t num_output_group = num_output_group_;
    PredLoop(dmat, nthread, verbose, out_pred,
      [pred_func, num_output_group]
      (int64_t rid, Entry* inst, float* out_pred) {
        pred_func(inst, &out_pred[rid * num_output_group]);
      });
  } else {
    using PredFunc = float (*)(Entry*);
    PredFunc pred_func = reinterpret_cast<PredFunc>(pred_func_handle_);
    PredLoop(dmat, nthread, verbose, out_pred,
      [pred_func](int64_t rid, Entry* inst, float* out_pred) {
        out_pred[rid] = pred_func(inst);
      });
  }
  if (verbose > 0) {
    LOG(INFO) << "Finished prediction in "
              << dmlc::GetTime() - tstart << " sec";
  }
}

}  // namespace treelite
