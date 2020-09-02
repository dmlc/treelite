/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file predictor.cc
 * \author Hyunsu Cho
 * \brief Load prediction function exported as a shared library
 */

#include <treelite/predictor.h>
#include <treelite/math.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include <dmlc/timer.h>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <fstream>
#include <limits>
#include <functional>
#include <type_traits>
#include "thread_pool/thread_pool.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

enum class InputType : uint8_t {
  kSparseBatch = 0, kDenseBatch = 1
};

struct InputToken {
  InputType input_type;
  const void* data;  // pointer to input data
  bool pred_margin;  // whether to store raw margin or transformed scores
  size_t num_feature;
    // # features (columns) accepted by the tree ensemble model
  size_t num_output_group;
    // size of output per instance (row)
  treelite::Predictor::PredFuncHandle pred_func_handle;
  size_t rbegin, rend;
    // range of instances (rows) assigned to each worker
  float* out_pred;
    // buffer to store output from each worker
};

struct OutputToken {
  size_t query_result_size;
};

using PredThreadPool = treelite::ThreadPool<InputToken, OutputToken, treelite::Predictor>;

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
inline size_t PredLoop(const treelite::CSRBatch* batch, size_t num_feature,
                       size_t rbegin, size_t rend,
                       float* out_pred, PredFunc func) {
  CHECK_LE(batch->num_col, num_feature);
  std::vector<TreelitePredictorEntry> inst(
    std::max(batch->num_col, num_feature), {-1});
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
  size_t total_output_size = 0;
  for (int64_t rid = rbegin_; rid < rend_; ++rid) {
    const size_t ibegin = row_ptr[rid];
    const size_t iend = row_ptr[rid + 1];
    for (size_t i = ibegin; i < iend; ++i) {
      inst[col_ind[i]].fvalue = data[i];
    }
    total_output_size += func(rid, &inst[0], out_pred);
    for (size_t i = ibegin; i < iend; ++i) {
      inst[col_ind[i]].missing = -1;
    }
  }
  return total_output_size;
}

template <typename PredFunc>
inline size_t PredLoop(const treelite::DenseBatch* batch, size_t num_feature,
                       size_t rbegin, size_t rend,
                       float* out_pred, PredFunc func) {
  const bool nan_missing = treelite::math::CheckNAN(batch->missing_value);
  CHECK_LE(batch->num_col, num_feature);
  std::vector<TreelitePredictorEntry> inst(
    std::max(batch->num_col, num_feature), {-1});
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
  size_t total_output_size = 0;
  for (int64_t rid = rbegin_; rid < rend_; ++rid) {
    row = &data[rid * num_col];
    for (size_t j = 0; j < num_col; ++j) {
      if (treelite::math::CheckNAN(row[j])) {
        CHECK(nan_missing)
          << "The missing_value argument must be set to NaN if there is any "
          << "NaN in the matrix.";
      } else if (nan_missing || row[j] != missing_value) {
        inst[j].fvalue = row[j];
      }
    }
    total_output_size += func(rid, &inst[0], out_pred);
    for (size_t j = 0; j < num_col; ++j) {
      inst[j].missing = -1;
    }
  }
  return total_output_size;
}

template <typename BatchType>
inline size_t PredictBatch_(const BatchType* batch, bool pred_margin,
                            size_t num_feature, size_t num_output_group,
                            treelite::Predictor::PredFuncHandle pred_func_handle,
                            size_t rbegin, size_t rend,
                            size_t expected_query_result_size, float* out_pred) {
  CHECK(pred_func_handle != nullptr)
    << "A shared library needs to be loaded first using Load()";
  /* Pass the correct prediction function to PredLoop.
     We also need to specify how the function should be called. */
  size_t query_result_size;
    // Dimension of output vector:
    // can be either [num_data] or [num_class]*[num_data].
    // Note that size of prediction may be smaller than out_pred (this occurs
    // when pred_function is set to "max_index").
  if (num_output_group > 1) {  // multi-class classification task
    using PredFunc = size_t (*)(TreelitePredictorEntry*, int, float*);
    PredFunc pred_func = reinterpret_cast<PredFunc>(pred_func_handle);
    query_result_size =
     PredLoop(batch, num_feature, rbegin, rend, out_pred,
      [pred_func, num_output_group, pred_margin]
      (int64_t rid, TreelitePredictorEntry* inst, float* out_pred) -> size_t {
        return pred_func(inst, static_cast<int>(pred_margin),
                         &out_pred[rid * num_output_group]);
      });
  } else {                     // every other task
    using PredFunc = float (*)(TreelitePredictorEntry*, int);
    PredFunc pred_func = reinterpret_cast<PredFunc>(pred_func_handle);
    query_result_size =
     PredLoop(batch, num_feature, rbegin, rend, out_pred,
      [pred_func, pred_margin]
      (int64_t rid, TreelitePredictorEntry* inst, float* out_pred) -> size_t {
        out_pred[rid] = pred_func(inst, static_cast<int>(pred_margin));
        return 1;
      });
  }
  return query_result_size;
}

}  // anonymous namespace

namespace treelite {

Predictor::Predictor(int num_worker_thread)
                       : lib_handle_(nullptr),
                         num_output_group_query_func_handle_(nullptr),
                         num_feature_query_func_handle_(nullptr),
                         pred_func_handle_(nullptr),
                         thread_pool_handle_(nullptr),
                         num_worker_thread_(num_worker_thread) {}
Predictor::~Predictor() {
  Free();
}

void
Predictor::Load(const char* name) {
  lib_handle_ = OpenLibrary(name);
  if (lib_handle_ == nullptr) {
    LOG(FATAL) << "Failed to load dynamic shared library `" << name << "'";
  }

  /* 1. query # of output groups */
  num_output_group_query_func_handle_
    = LoadFunction<QueryFuncHandle>(lib_handle_, "get_num_output_group");
  using UnsignedQueryFunc = size_t (*)(void);
  auto uint_query_func
    = reinterpret_cast<UnsignedQueryFunc>(num_output_group_query_func_handle_);
  CHECK(uint_query_func != nullptr)
    << "Dynamic shared library `" << name
    << "' does not contain valid get_num_output_group() function";
  num_output_group_ = uint_query_func();

  /* 2. query # of features */
  num_feature_query_func_handle_
    = LoadFunction<QueryFuncHandle>(lib_handle_, "get_num_feature");
  uint_query_func = reinterpret_cast<UnsignedQueryFunc>(num_feature_query_func_handle_);
  CHECK(uint_query_func != nullptr)
    << "Dynamic shared library `" << name
    << "' does not contain valid get_num_feature() function";
  num_feature_ = uint_query_func();
  CHECK_GT(num_feature_, 0) << "num_feature cannot be zero";

  /* 3. query # of pred_transform name */
  pred_transform_query_func_handle_
    = LoadFunction<QueryFuncHandle>(lib_handle_, "get_pred_transform");
  using StringQueryFunc = const char* (*)(void);
  auto str_query_func =
      reinterpret_cast<StringQueryFunc>(pred_transform_query_func_handle_);
  if (str_query_func == nullptr) {
    LOG(INFO) << "Dynamic shared library `" << name
              << "' does not contain valid get_pred_transform() function";
    pred_transform_ = "unknown";
  } else {
    pred_transform_ = str_query_func();
  }

  /* 4. query # of sigmoid_alpha */
  sigmoid_alpha_query_func_handle_
    = LoadFunction<QueryFuncHandle>(lib_handle_, "get_sigmoid_alpha");
  using FloatQueryFunc = float (*)(void);
  auto float_query_func =
      reinterpret_cast<FloatQueryFunc>(sigmoid_alpha_query_func_handle_);
  if (float_query_func == nullptr) {
    LOG(INFO) << "Dynamic shared library `" << name
              << "' does not contain valid get_sigmoid_alpha() function";
    sigmoid_alpha_ = NAN;
  } else {
    sigmoid_alpha_ = float_query_func();
  }

  /* 5. query # of global_bias */
  global_bias_query_func_handle_
    = LoadFunction<QueryFuncHandle>(lib_handle_, "get_global_bias");
  float_query_func = reinterpret_cast<FloatQueryFunc>(global_bias_query_func_handle_);
  if (float_query_func == nullptr) {
    LOG(INFO) << "Dynamic shared library `" << name
              << "' does not contain valid get_global_bias() function";
    global_bias_ = NAN;
  } else {
    global_bias_ = float_query_func();
  }

  /* 6. load appropriate function for margin prediction */
  CHECK_GT(num_output_group_, 0) << "num_output_group cannot be zero";
  if (num_output_group_ > 1) {   // multi-class classification
    pred_func_handle_ = LoadFunction<PredFuncHandle>(lib_handle_,
                                                     "predict_multiclass");
    using PredFunc = size_t (*)(TreelitePredictorEntry*, int, float*);
    PredFunc pred_func = reinterpret_cast<PredFunc>(pred_func_handle_);
    CHECK(pred_func != nullptr)
      << "Dynamic shared library `" << name
      << "' does not contain valid predict_multiclass() function";
  } else {                      // everything else
    pred_func_handle_ = LoadFunction<PredFuncHandle>(lib_handle_, "predict");
    using PredFunc = float (*)(TreelitePredictorEntry*, int);
    PredFunc pred_func = reinterpret_cast<PredFunc>(pred_func_handle_);
    CHECK(pred_func != nullptr)
      << "Dynamic shared library `" << name
      << "' does not contain valid predict() function";
  }

  if (num_worker_thread_ == -1) {
    num_worker_thread_ = std::thread::hardware_concurrency();
  }
  thread_pool_handle_ = static_cast<ThreadPoolHandle>(
      new PredThreadPool(num_worker_thread_ - 1, this,
                         [](SpscQueue<InputToken>* incoming_queue,
                            SpscQueue<OutputToken>* outgoing_queue,
                            const Predictor* predictor) {
      InputToken input;
      while (incoming_queue->Pop(&input)) {
        size_t query_result_size;
        const size_t rbegin = input.rbegin;
        const size_t rend = input.rend;
        switch (input.input_type) {
         case InputType::kSparseBatch:
          {
            const CSRBatch* batch = static_cast<const CSRBatch*>(input.data);
            query_result_size
              = PredictBatch_(batch, input.pred_margin, input.num_feature,
                              input.num_output_group, input.pred_func_handle,
                              rbegin, rend,
                              predictor->QueryResultSize(batch, rbegin, rend),
                              input.out_pred);
          }
          break;
         case InputType::kDenseBatch:
          {
            const DenseBatch* batch = static_cast<const DenseBatch*>(input.data);
            query_result_size
              = PredictBatch_(batch, input.pred_margin, input.num_feature,
                              input.num_output_group, input.pred_func_handle,
                              rbegin, rend,
                              predictor->QueryResultSize(batch, rbegin, rend),
                              input.out_pred);
          }
          break;
        }
        outgoing_queue->Push(OutputToken{query_result_size});
      }
    }));
}

void
Predictor::Free() {
  CloseLibrary(lib_handle_);
  delete static_cast<PredThreadPool*>(thread_pool_handle_);
}

template <typename BatchType>
static inline
std::vector<size_t> SplitBatch(const BatchType* batch, size_t split_factor) {
  const size_t num_row = batch->num_row;
  CHECK_LE(split_factor, num_row);
  const size_t portion = num_row / split_factor;
  const size_t remainder = num_row % split_factor;
  std::vector<size_t> workload(split_factor, portion);
  std::vector<size_t> row_ptr(split_factor + 1, 0);
  for (size_t i = 0; i < remainder; ++i) {
    ++workload[i];
  }
  size_t accum = 0;
  for (size_t i = 0; i < split_factor; ++i) {
    accum += workload[i];
    row_ptr[i + 1] = accum;
  }
  return row_ptr;
}

template <typename BatchType>
inline size_t
Predictor::PredictBatchBase_(const BatchType* batch, int verbose,
                             bool pred_margin, float* out_result) {
  static_assert(std::is_same<BatchType, DenseBatch>::value
                || std::is_same<BatchType, CSRBatch>::value,
                "PredictBatchBase_: unrecognized batch type");
  const double tstart = dmlc::GetTime();
  PredThreadPool* pool = static_cast<PredThreadPool*>(thread_pool_handle_);
  const InputType input_type
    = std::is_same<BatchType, CSRBatch>::value
      ? InputType::kSparseBatch : InputType::kDenseBatch;
  InputToken request{input_type, static_cast<const void*>(batch), pred_margin,
                     num_feature_, num_output_group_, pred_func_handle_,
                     0, batch->num_row, out_result};
  OutputToken response;
  CHECK_GT(batch->num_row, 0);
  const int nthread = std::min(num_worker_thread_,
                               static_cast<int>(batch->num_row));
  const std::vector<size_t> row_ptr = SplitBatch(batch, nthread);
  for (int tid = 0; tid < nthread - 1; ++tid) {
    request.rbegin = row_ptr[tid];
    request.rend = row_ptr[tid + 1];
    pool->SubmitTask(tid, request);
  }
  size_t total_size = 0;
  {
    // assign work to master
    const size_t rbegin = row_ptr[nthread - 1];
    const size_t rend = row_ptr[nthread];
    const size_t query_result_size
      = PredictBatch_(batch, pred_margin, num_feature_, num_output_group_,
                      pred_func_handle_,
                      rbegin, rend, QueryResultSize(batch, rbegin, rend),
                      out_result);
    total_size += query_result_size;
  }
  for (int tid = 0; tid < nthread - 1; ++tid) {
    if (pool->WaitForTask(tid, &response)) {
      total_size += response.query_result_size;
    }
  }
  // re-shape output if total_size < dimension of out_result
  if (total_size < QueryResultSize(batch, 0, batch->num_row)) {
    CHECK_GT(num_output_group_, 1);
    CHECK_EQ(total_size % batch->num_row, 0);
    const size_t query_size_per_instance = total_size / batch->num_row;
    CHECK_GT(query_size_per_instance, 0);
    CHECK_LT(query_size_per_instance, num_output_group_);
    for (size_t rid = 0; rid < batch->num_row; ++rid) {
      for (size_t k = 0; k < query_size_per_instance; ++k) {
        out_result[rid * query_size_per_instance + k]
          = out_result[rid * num_output_group_ + k];
      }
    }
  }
  const double tend = dmlc::GetTime();
  if (verbose > 0) {
    LOG(INFO) << "Treelite: Finished prediction in " << tend - tstart << " sec";
  }
  return total_size;
}

size_t
Predictor::PredictBatch(const CSRBatch* batch, int verbose,
                        bool pred_margin, float* out_result) {
  return PredictBatchBase_(batch, verbose, pred_margin, out_result);
}

size_t
Predictor::PredictBatch(const DenseBatch* batch, int verbose,
                        bool pred_margin, float* out_result) {
  return PredictBatchBase_(batch, verbose, pred_margin, out_result);
}

}  // namespace treelite
