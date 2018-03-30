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
#include <type_traits>
#include "common/math.h"
#include "thread_pool/thread_pool.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

struct InputToken {
  bool sparse;
  const void* batch;
  bool pred_margin;
  size_t num_output_group;
  treelite::Predictor::PredFuncHandle pred_func_handle;
  treelite::Predictor::PredTransformFuncHandle pred_transform_func_handle;
  size_t rbegin, rend;
  float* out_pred;
};

struct OutputToken {
  size_t query_result_size;
};

using PredThreadPool
  = treelite::ThreadPool<InputToken, OutputToken, treelite::Predictor>;

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
inline void PredLoop(const treelite::CSRBatch* batch,
                     size_t rbegin, size_t rend,
                     float* out_pred, PredFunc func) {
  std::vector<treelite::Predictor::Entry> inst(batch->num_col, {-1});
  CHECK(rbegin < rend && rend <= batch->num_row);
  CHECK(sizeof(size_t) < sizeof(int64_t)
     || (rbegin <= static_cast<size_t>(std::numeric_limits<int64_t>::max())
        && rend <= static_cast<size_t>(std::numeric_limits<int64_t>::max())));
  const int64_t rbegin_ = static_cast<int64_t>(rbegin);
  const int64_t rend_ = static_cast<int64_t>(rend);
  const size_t num_col = batch->num_col;
  const float* data = batch->data;
  const uint32_t* col_ind = batch->col_ind;
  const size_t* row_ptr = batch->row_ptr;
  for (int64_t rid = rbegin_; rid < rend_; ++rid) {
    const size_t ibegin = row_ptr[rid];
    const size_t iend = row_ptr[rid + 1];
    for (size_t i = ibegin; i < iend; ++i) {
      inst[col_ind[i]].fvalue = data[i];
    }
    func(rid, &inst[0], out_pred);
    for (size_t i = ibegin; i < iend; ++i) {
      inst[col_ind[i]].missing = -1;
    }
  }
}

template <typename PredFunc>
inline void PredLoop(const treelite::DenseBatch* batch,
                     size_t rbegin, size_t rend,
                     float* out_pred, PredFunc func) {
  const bool nan_missing
                      = treelite::common::math::CheckNAN(batch->missing_value);
  std::vector<treelite::Predictor::Entry> inst(batch->num_col, {-1});
  CHECK(rbegin < rend && rend <= batch->num_row);
  CHECK(sizeof(size_t) < sizeof(int64_t)
     || (rbegin <= static_cast<size_t>(std::numeric_limits<int64_t>::max())
        && rend <= static_cast<size_t>(std::numeric_limits<int64_t>::max())));
  const int64_t rbegin_ = static_cast<int64_t>(rbegin);
  const int64_t rend_ = static_cast<int64_t>(rend);
  const size_t num_col = batch->num_col;
  const float missing_value = batch->missing_value;
  const float* data = batch->data;
  const float* row;
  for (int64_t rid = rbegin_; rid < rend_; ++rid) {
    row = &data[rid * num_col];
    for (size_t j = 0; j < num_col; ++j) {
      if (treelite::common::math::CheckNAN(row[j])) {
        CHECK(nan_missing)
          << "The missing_value argument must be set to NaN if there is any "
          << "NaN in the matrix.";
      } else if (nan_missing || row[j] != missing_value) {
        inst[j].fvalue = row[j];
      }
    }
    func(rid, &inst[0], out_pred);
    for (size_t j = 0; j < num_col; ++j) {
      inst[j].missing = -1;
    }
  }
}

template <typename BatchType>
inline size_t PredictBatch_(const BatchType* batch,
                            bool pred_margin, size_t num_output_group,
                          treelite::Predictor::PredFuncHandle pred_func_handle,
       treelite::Predictor::PredTransformFuncHandle pred_transform_func_handle,
                            size_t rbegin, size_t rend,
                            size_t query_result_size, float* out_pred) {
  CHECK(pred_func_handle != nullptr)
    << "A shared library needs to be loaded first using Load()";
  /* Pass the correct prediction function to PredLoop.
     We also need to specify how the function should be called. */
  if (num_output_group > 1) {
    using PredFunc = void (*)(treelite::Predictor::Entry*, float*);
    PredFunc pred_func = reinterpret_cast<PredFunc>(pred_func_handle);
    PredLoop(batch, rbegin, rend, out_pred,
      [pred_func, num_output_group]
      (int64_t rid, treelite::Predictor::Entry* inst, float* out_pred) {
        pred_func(inst, &out_pred[rid * num_output_group]);
      });
  } else {
    using PredFunc = float (*)(treelite::Predictor::Entry*);
    PredFunc pred_func = reinterpret_cast<PredFunc>(pred_func_handle);
    PredLoop(batch, rbegin, rend, out_pred,
      [pred_func]
      (int64_t rid, treelite::Predictor::Entry* inst, float* out_pred) {
        out_pred[rid] = pred_func(inst);
      });
  }
  if (pred_margin) {
    return query_result_size;
  } else {
    LOG(FATAL) << "pred_transform not supported";
    return 0;
  }
}

}  // namespace anonymous

namespace treelite {

Predictor::Predictor(int num_worker_thread,
                     bool include_master_thread)
                       : lib_handle_(nullptr),
                         query_func_handle_(nullptr),
                         pred_func_handle_(nullptr),
                         pred_transform_func_handle_(nullptr),
                         thread_pool_handle_(nullptr),
                         include_master_thread_(include_master_thread),
                         num_worker_thread_(num_worker_thread) {}
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

  if (num_worker_thread_ == -1) {
    num_worker_thread_ = std::thread::hardware_concurrency() - 1;
  }
  thread_pool_handle_ = static_cast<ThreadPoolHandle>(
      new PredThreadPool(num_worker_thread_, this,
                         [](SpscQueue<InputToken>* incoming_queue,
                            SpscQueue<OutputToken>* outgoing_queue,
                            const treelite::Predictor* predictor) {
      InputToken input;
      while (incoming_queue->Pop(&input)) {
        size_t query_result_size;
        const size_t rbegin = input.rbegin;
        const size_t rend = input.rend;
        if (input.sparse) {
          const CSRBatch* batch = static_cast<const CSRBatch*>(input.batch);
          query_result_size
            = PredictBatch_(batch, input.pred_margin, input.num_output_group,
                            input.pred_func_handle,
                            input.pred_transform_func_handle,
                            rbegin, rend,
                            predictor->QueryResultSize(batch, rbegin, rend),
                            input.out_pred);
        } else {
          const DenseBatch* batch = static_cast<const DenseBatch*>(input.batch);
          query_result_size
            = PredictBatch_(batch, input.pred_margin, input.num_output_group,
                            input.pred_func_handle,
                            input.pred_transform_func_handle,
                            rbegin, rend,
                            predictor->QueryResultSize(batch, rbegin, rend),
                            input.out_pred);
        }
        outgoing_queue->Push(OutputToken{query_result_size});
      }
    }));

  /* 3. load prediction transform function */
  pred_transform_func_handle_
    = LoadFunction<PredTransformFuncHandle>(lib_handle_,
                                            "pred_transform");
  using PredTransformFunc = size_t (*)(float*, int64_t, int);
  PredTransformFunc pred_transform_func
    = reinterpret_cast<PredTransformFunc>(pred_transform_func_handle_);
  CHECK(pred_transform_func != nullptr)
    << "Dynamic shared library `" << name
    << "' does not contain valid pred_transform() function";
}

void
Predictor::Free() {
  CloseLibrary(lib_handle_);
  delete static_cast<PredThreadPool*>(thread_pool_handle_);
}

template <typename BatchType>
static inline
std::vector<size_t> SplitBatch(const BatchType* batch, size_t nthread) {
  const size_t num_row = batch->num_row;
  CHECK_LE(nthread, num_row);
  const size_t portion = num_row / nthread;
  const size_t remainder = num_row % nthread;
  std::vector<size_t> workload(nthread, portion);
  std::vector<size_t> row_ptr(nthread + 1, 0);
  for (size_t i = 0; i < remainder; ++i) {
    ++workload[i];
  }
  size_t accum = 0;
  for (size_t i = 0; i < nthread; ++i) {
    accum += workload[i];
    row_ptr[i + 1] = accum;
  }
  return row_ptr;
}

template <typename BatchType>
inline size_t
Predictor::PredictBatchBase_(const BatchType* batch,
                             bool pred_margin, float* out_result) {
  static_assert(   std::is_same<BatchType, DenseBatch>::value
                || std::is_same<BatchType, CSRBatch>::value,
                "PredictBatchBase_: unrecognized batch type");
  const double tstart = dmlc::GetTime();
  PredThreadPool* pool = static_cast<PredThreadPool*>(thread_pool_handle_);
  InputToken request{std::is_same<BatchType, CSRBatch>::value,
                     static_cast<const void*>(batch), pred_margin,
                     num_output_group_, pred_func_handle_,
                     pred_transform_func_handle_, 0, batch->num_row,
                     out_result};
  OutputToken response;
  const int nthread = num_worker_thread_;
  const std::vector<size_t> row_ptr
    = SplitBatch(batch, nthread + (int)(include_master_thread_));
  for (int tid = 0; tid < nthread; ++tid) {
    request.rbegin = row_ptr[tid];
    request.rend = row_ptr[tid + 1];
    pool->SubmitTask(tid, request);
  }
  size_t total_size = 0;
  if (include_master_thread_) {
    const size_t rbegin = row_ptr[nthread];
    const size_t rend = row_ptr[nthread + 1];
    const size_t query_result_size
      = PredictBatch_(batch, pred_margin, num_output_group_,
                      pred_func_handle_, pred_transform_func_handle_,
                      rbegin, rend, QueryResultSize(batch, rbegin, rend),
                      out_result);
    total_size += query_result_size;
  }
  for (int tid = 0; tid < nthread; ++tid) {
    if (pool->WaitForTask(tid, &response)) {
      total_size += response.query_result_size;
    }
  }
  const double tend = dmlc::GetTime();
  LOG(INFO) << "Treelite: Finished prediction in "
            << tend - tstart << " sec";
  return total_size;
}

size_t
Predictor::PredictBatch(const CSRBatch* batch,
                        bool pred_margin, float* out_result) {
  return PredictBatchBase_(batch, pred_margin, out_result);
}

size_t
Predictor::PredictBatch(const DenseBatch* batch,
                        bool pred_margin, float* out_result) {
  return PredictBatchBase_(batch, pred_margin, out_result);
}

}  // namespace treelite
