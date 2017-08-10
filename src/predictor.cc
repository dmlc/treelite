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

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

inline void PredLoop(treelite::Predictor::PredFunc func,
                     const treelite::DMatrix* dmat,
                     size_t rbegin, size_t rend, int nthread,
                     treelite::Predictor::Entry* inst,
                     float* out_pred) {
  #pragma omp parallel for schedule(static) num_threads(nthread)
  for (size_t rid = rbegin; rid < rend; ++rid) {
    const int tid = omp_get_thread_num();
    const size_t off = dmat->num_col * tid;
    const size_t ibegin = dmat->row_ptr[rid];
    const size_t iend = dmat->row_ptr[rid + 1];
    for (size_t i = ibegin; i < iend; ++i) {
      inst[off + dmat->col_ind[i]].fvalue = dmat->data[i];
    }
    out_pred[rid] = func(&inst[off]);
    for (size_t i = ibegin; i < iend; ++i) {
      inst[off + dmat->col_ind[i]].missing = -1;
    }
  }
}

}  // namespace anonymous

namespace treelite {

Predictor::Predictor() : lib_handle_(nullptr), func_(nullptr) {}
Predictor::~Predictor() {
  Free();
}

#ifdef _WIN32
void
Predictor::Load(const char* name) {
  lib_handle_ = static_cast<void*>(LoadLibraryA(wname));
  CHECK(lib_handle_ != nullptr)
    << "Failed to load dynamic shared library `" << name << "'";
  func_ = reinterpret_cast<PredFunc>(GetProcAddress(lib_handle_,
                                                    "predict_margin"));
  CHECK(func_ != nullptr)
    << "Dynamic shared library `" << name
    << "' does not contain predict_margin() function";
}

void
Predictor::Free() {
  FreeLibrary(lib_handle_);
}
#else
void
Predictor::Load(const char* name) {
  lib_handle_ = static_cast<void*>(dlopen(name, RTLD_LAZY | RTLD_LOCAL));
  CHECK(lib_handle_ != nullptr)
    << "Failed to load dynamic shared library `" << name << "'";
  func_ = reinterpret_cast<PredFunc>(dlsym(lib_handle_, "predict_margin"));
  CHECK(func_ != nullptr)
    << "Dynamic shared library `" << name
    << "' does not contain predict_margin() function";
}

void
Predictor::Free() {
  dlclose(lib_handle_);
}
#endif

void
Predictor::Predict(const DMatrix* dmat, int nthread, int verbose,
                   float* out_pred) const {
  CHECK(func_ != nullptr)
    << "The predict_margin() function needs to be loaded first.";
  const int max_thread = omp_get_max_threads();
  nthread = (nthread == 0) ? max_thread : std::min(nthread, max_thread);
  std::vector<Entry> inst(nthread * dmat->num_col, {-1});
  const size_t pstep = (dmat->num_row + 99) / 100;
      // interval to display progress

  if (verbose > 0) {
    LOG(INFO) << "Begin prediction";
  }
  double tstart = dmlc::GetTime();
  for (size_t rbegin = 0; rbegin < dmat->num_row; rbegin += pstep) {
    const size_t rend = std::min(rbegin + pstep, dmat->num_row);
    PredLoop(func_, dmat, rbegin, rend, nthread, &inst[0], out_pred);
    if (verbose > 0) {
      LOG(INFO) << rend << " of " << dmat->num_row << " rows processed";
    }
  }
  if (verbose > 0) {
    LOG(INFO) << "Finished prediction in "
              << dmlc::GetTime() - tstart << " sec";
  }
}

}  // namespace treelite
