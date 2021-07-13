/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file predictor.cc
 * \author Hyunsu Cho
 * \brief Load prediction function exported as a shared library
 */

#include <treelite/predictor.h>
#include <treelite/math.h>
#include <treelite/data.h>
#include <treelite/typeinfo.h>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <fstream>
#include <limits>
#include <functional>
#include <type_traits>
#include <chrono>
#include "thread_pool/thread_pool.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

inline double GetTime(void) {
  return std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

struct InputToken {
  const treelite::DMatrix* dmat;  // input data
  bool pred_margin;  // whether to store raw margin or transformed scores
  const treelite::predictor::PredFunction* pred_func_;
  size_t rbegin, rend;  // range of instances (rows) assigned to each worker
  PredictorOutputHandle out_pred;  // buffer to store output from each worker
};

struct OutputToken {
  size_t query_result_size;
};

using PredThreadPool
  = treelite::predictor::ThreadPool<InputToken, OutputToken, treelite::predictor::Predictor>;

template <typename ElementType, typename ThresholdType, typename LeafOutputType, typename PredFunc>
inline size_t PredLoop(const treelite::CSRDMatrixImpl<ElementType>* dmat, int num_feature,
                       size_t rbegin, size_t rend, LeafOutputType* out_pred, PredFunc func) {
  TREELITE_CHECK_LE(dmat->num_col, static_cast<size_t>(num_feature));
  std::vector<treelite::predictor::Entry<ThresholdType>> inst(
    std::max(dmat->num_col, static_cast<size_t>(num_feature)), {-1});
  TREELITE_CHECK(rbegin < rend && rend <= dmat->num_row);
  const ElementType* data = dmat->data.data();
  const uint32_t* col_ind = dmat->col_ind.data();
  const size_t* row_ptr = dmat->row_ptr.data();
  size_t total_output_size = 0;
  for (size_t rid = rbegin; rid < rend; ++rid) {
    const size_t ibegin = row_ptr[rid];
    const size_t iend = row_ptr[rid + 1];
    for (size_t i = ibegin; i < iend; ++i) {
      inst[col_ind[i]].fvalue = static_cast<ThresholdType>(data[i]);
    }
    total_output_size += func(rid, &inst[0], out_pred);
    for (size_t i = ibegin; i < iend; ++i) {
      inst[col_ind[i]].missing = -1;
    }
  }
  return total_output_size;
}

template <typename ElementType, typename ThresholdType, typename LeafOutputType, typename PredFunc>
inline size_t PredLoop(const treelite::DenseDMatrixImpl<ElementType>* dmat, int num_feature,
                       size_t rbegin, size_t rend, LeafOutputType* out_pred, PredFunc func) {
  const bool nan_missing = treelite::math::CheckNAN(dmat->missing_value);
  TREELITE_CHECK_LE(dmat->num_col, static_cast<size_t>(num_feature));
  std::vector<treelite::predictor::Entry<ThresholdType>> inst(
      std::max(dmat->num_col, static_cast<size_t>(num_feature)), {-1});
  TREELITE_CHECK(rbegin < rend && rend <= dmat->num_row);
  const size_t num_col = dmat->num_col;
  const ElementType missing_value = dmat->missing_value;
  const ElementType* data = dmat->data.data();
  const ElementType* row = nullptr;
  size_t total_output_size = 0;
  for (size_t rid = rbegin; rid < rend; ++rid) {
    row = &data[rid * num_col];
    for (size_t j = 0; j < num_col; ++j) {
      if (treelite::math::CheckNAN(row[j])) {
        TREELITE_CHECK(nan_missing)
          << "The missing_value argument must be set to NaN if there is any NaN in the matrix.";
      } else if (nan_missing || row[j] != missing_value) {
        inst[j].fvalue = static_cast<ThresholdType>(row[j]);
      }
    }
    total_output_size += func(rid, &inst[0], out_pred);
    for (size_t j = 0; j < num_col; ++j) {
      inst[j].missing = -1;
    }
  }
  return total_output_size;
}

template <typename ElementType>
class PredLoopDispatcherWithDenseDMatrix {
 public:
  template <typename ThresholdType, typename LeafOutputType, typename PredFunc>
  inline static size_t Dispatch(
      const treelite::DMatrix* dmat, ThresholdType, int num_feature, size_t rbegin, size_t rend,
      LeafOutputType* out_pred, PredFunc func) {
    const auto* dmat_ = static_cast<const treelite::DenseDMatrixImpl<ElementType>*>(dmat);
    return PredLoop<ElementType, ThresholdType, LeafOutputType, PredFunc>(
        dmat_, num_feature, rbegin, rend, out_pred, func);
  }
};

template <typename ElementType>
class PredLoopDispatcherWithCSRDMatrix {
 public:
  template <typename ThresholdType, typename LeafOutputType, typename PredFunc>
  inline static size_t Dispatch(
      const treelite::DMatrix* dmat, ThresholdType, int num_feature, size_t rbegin, size_t rend,
      LeafOutputType* out_pred, PredFunc func) {
    const auto* dmat_ = static_cast<const treelite::CSRDMatrixImpl<ElementType>*>(dmat);
    return PredLoop<ElementType, ThresholdType, LeafOutputType, PredFunc>(
        dmat_, num_feature, rbegin, rend, out_pred, func);
  }
};

template <typename ThresholdType, typename LeafOutputType, typename PredFunc>
inline size_t PredLoop(const treelite::DMatrix* dmat, ThresholdType test_val, int num_feature,
                       size_t rbegin, size_t rend, LeafOutputType* out_pred, PredFunc func) {
  treelite::DMatrixType dmat_type = dmat->GetType();
  switch (dmat_type) {
  case treelite::DMatrixType::kDense: {
    return treelite::DispatchWithTypeInfo<PredLoopDispatcherWithDenseDMatrix>(
        dmat->GetElementType(), dmat, test_val, num_feature, rbegin, rend, out_pred, func);
  }
  case treelite::DMatrixType::kSparseCSR: {
    return treelite::DispatchWithTypeInfo<PredLoopDispatcherWithCSRDMatrix>(
        dmat->GetElementType(), dmat, test_val, num_feature, rbegin, rend, out_pred, func);
  }
  default:
    TREELITE_LOG(FATAL) << "Unrecognized data matrix type: " << static_cast<int>(dmat_type);
    return 0;
  }
}

}  // anonymous namespace

namespace treelite {
namespace predictor {

SharedLibrary::SharedLibrary() : handle_(nullptr), libpath_() {}

SharedLibrary::~SharedLibrary() {
  if (handle_) {
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle_));
#else
    dlclose(static_cast<void*>(handle_));
#endif
  }
}

void
SharedLibrary::Load(const char* libpath) {
#ifdef _WIN32
  HMODULE handle = LoadLibraryA(libpath);
#else
  void* handle = dlopen(libpath, RTLD_LAZY | RTLD_LOCAL);
#endif
  TREELITE_CHECK(handle) << "Failed to load dynamic shared library `" << libpath << "'";
  handle_ = static_cast<LibraryHandle>(handle);
  libpath_ = std::string(libpath);
}

SharedLibrary::FunctionHandle
SharedLibrary::LoadFunction(const char* name) const {
#ifdef _WIN32
  FARPROC func_handle = GetProcAddress(static_cast<HMODULE>(handle_), name);
#else
  void* func_handle = dlsym(static_cast<void*>(handle_), name);
#endif
  TREELITE_CHECK(func_handle)
    << "Dynamic shared library `" << libpath_ << "' does not contain a function " << name << "().";
  return static_cast<SharedLibrary::FunctionHandle>(func_handle);
}

template <typename HandleType>
HandleType
SharedLibrary::LoadFunctionWithSignature(const char* name) const {
  auto func_handle = reinterpret_cast<HandleType>(LoadFunction(name));
  TREELITE_CHECK(func_handle) << "Dynamic shared library `" << libpath_
                              << "' does not contain a function " << name
                              << "() with the requested signature";
  return func_handle;
}

template <typename ThresholdType, typename LeafOutputType>
class PredFunctionInitDispatcher {
 public:
  inline static std::unique_ptr<PredFunction> Dispatch(
      const SharedLibrary& library, int num_feature, int num_class) {
    return std::make_unique<PredFunctionImpl<ThresholdType, LeafOutputType>>(
        library, num_feature, num_class);
  }
};

std::unique_ptr<PredFunction>
PredFunction::Create(
    TypeInfo threshold_type, TypeInfo leaf_output_type, const SharedLibrary& library,
    int num_feature, int num_class) {
  return DispatchWithModelTypes<PredFunctionInitDispatcher>(
      threshold_type, leaf_output_type, library, num_feature, num_class);
}

template <typename ThresholdType, typename LeafOutputType>
PredFunctionImpl<ThresholdType, LeafOutputType>::PredFunctionImpl(
    const SharedLibrary& library, int num_feature, int num_class) {
  TREELITE_CHECK_GT(num_class, 0) << "num_class cannot be zero";
  if (num_class > 1) {  // multi-class classification
    handle_ = library.LoadFunction("predict_multiclass");
  } else {  // everything else
    handle_ = library.LoadFunction("predict");
  }
  num_feature_ = num_feature;
  num_class_ = num_class;
}

template <typename ThresholdType, typename LeafOutputType>
TypeInfo
PredFunctionImpl<ThresholdType, LeafOutputType>::GetThresholdType() const {
  return TypeToInfo<ThresholdType>();
}

template <typename ThresholdType, typename LeafOutputType>
TypeInfo
PredFunctionImpl<ThresholdType, LeafOutputType>::GetLeafOutputType() const {
  return TypeToInfo<LeafOutputType>();
}

template <typename ThresholdType, typename LeafOutputType>
size_t
PredFunctionImpl<ThresholdType, LeafOutputType>::PredictBatch(
    const DMatrix* dmat, size_t rbegin, size_t rend, bool pred_margin,
    PredictorOutputHandle out_pred) const {
  /* Pass the correct prediction function to PredLoop.
     We also need to specify how the function should be called. */
  size_t result_size;
  // Dimension of output vector:
  // can be either [num_data] or [num_class]*[num_data].
  // Note that size of prediction may be smaller than out_pred (this occurs
  // when pred_function is set to "max_index").
  TREELITE_CHECK(rbegin < rend && rend <= dmat->GetNumRow());
  if (num_class_ > 1) {  // multi-class classification
    using PredFunc = size_t (*)(Entry<ThresholdType>*, int, LeafOutputType*);
    auto pred_func = reinterpret_cast<PredFunc>(handle_);
    TREELITE_CHECK(pred_func) << "The predict_multiclass() function has incorrect signature.";
    auto pred_func_wrapper
      = [pred_func, num_class = num_class_, pred_margin]
            (int64_t rid, Entry<ThresholdType>* inst, LeafOutputType* out_pred) -> size_t {
          return pred_func(inst, static_cast<int>(pred_margin),
                           &out_pred[rid * num_class]);
        };
    result_size = PredLoop(dmat, static_cast<ThresholdType>(0), num_feature_, rbegin, rend,
                           static_cast<LeafOutputType*>(out_pred), pred_func_wrapper);
  } else {  // everything else
    using PredFunc = LeafOutputType (*)(Entry<ThresholdType>*, int);
    auto pred_func = reinterpret_cast<PredFunc>(handle_);
    TREELITE_CHECK(pred_func) << "The predict() function has incorrect signature.";
    auto pred_func_wrapper
      = [pred_func, pred_margin]
            (int64_t rid, Entry<ThresholdType>* inst, LeafOutputType* out_pred) -> size_t {
          out_pred[rid] = pred_func(inst, static_cast<int>(pred_margin));
          return 1;
        };
    result_size = PredLoop(dmat, static_cast<ThresholdType>(0), num_feature_, rbegin, rend,
                           static_cast<LeafOutputType*>(out_pred), pred_func_wrapper);
  }
  return result_size;
}

Predictor::Predictor(int num_worker_thread)
                       : pred_func_(nullptr),
                         thread_pool_handle_(nullptr),
                         num_class_(0),
                         num_feature_(0),
                         sigmoid_alpha_(std::numeric_limits<float>::quiet_NaN()),
                         global_bias_(std::numeric_limits<float>::quiet_NaN()),
                         num_worker_thread_(num_worker_thread),
                         threshold_type_(TypeInfo::kInvalid),
                         leaf_output_type_(TypeInfo::kInvalid) {}
Predictor::~Predictor() {
  if (thread_pool_handle_) {
    Free();
  }
}

void
Predictor::Load(const char* libpath) {
  lib_.Load(libpath);

  using UnsignedQueryFunc = size_t (*)();
  using StringQueryFunc = const char* (*)();
  using FloatQueryFunc = float (*)();

  /* 1. query # of output groups */
  auto* num_class_query_func
    = lib_.LoadFunctionWithSignature<UnsignedQueryFunc>("get_num_class");
  num_class_ = num_class_query_func();

  /* 2. query # of features */
  auto* num_feature_query_func
    = lib_.LoadFunctionWithSignature<UnsignedQueryFunc>("get_num_feature");
  num_feature_ = num_feature_query_func();
  TREELITE_CHECK_GT(num_feature_, 0) << "num_feature cannot be zero";

  /* 3. query # of pred_transform name */
  auto* pred_transform_query_func
    = lib_.LoadFunctionWithSignature<StringQueryFunc>("get_pred_transform");
  pred_transform_ = pred_transform_query_func();

  /* 4. query # of sigmoid_alpha */
  auto* sigmoid_alpha_query_func
    = lib_.LoadFunctionWithSignature<FloatQueryFunc>("get_sigmoid_alpha");
  sigmoid_alpha_ = sigmoid_alpha_query_func();

  /* 5. query # of global_bias */
  auto* global_bias_query_func = lib_.LoadFunctionWithSignature<FloatQueryFunc>("get_global_bias");
  global_bias_ = global_bias_query_func();

  /* 6. Query the data type for thresholds and leaf outputs */
  auto* threshold_type_query_func
    = lib_.LoadFunctionWithSignature<StringQueryFunc>("get_threshold_type");
  threshold_type_ = GetTypeInfoByName(threshold_type_query_func());
  auto* leaf_output_type_query_func
    = lib_.LoadFunctionWithSignature<StringQueryFunc>("get_leaf_output_type");
  leaf_output_type_ = GetTypeInfoByName(leaf_output_type_query_func());

  /* 7. load appropriate function for margin prediction */
  TREELITE_CHECK_GT(num_class_, 0) << "num_class cannot be zero";
  pred_func_ = PredFunction::Create(
      threshold_type_, leaf_output_type_, lib_,
      static_cast<int>(num_feature_), static_cast<int>(num_class_));

  if (num_worker_thread_ == -1) {
    num_worker_thread_ = static_cast<int>(std::thread::hardware_concurrency());
  }
  thread_pool_handle_ = static_cast<ThreadPoolHandle>(
      new PredThreadPool(num_worker_thread_ - 1, this,
                         [](SpscQueue<InputToken>* incoming_queue,
                            SpscQueue<OutputToken>* outgoing_queue,
                            const Predictor* predictor) {
      predictor->exception_catcher_.Run([&]() {
        InputToken input;
        while (incoming_queue->Pop(&input)) {
          const size_t rbegin = input.rbegin;
          const size_t rend = input.rend;
          size_t query_result_size
              = predictor->pred_func_->PredictBatch(
                  input.dmat, rbegin, rend, input.pred_margin, input.out_pred);
          outgoing_queue->Push(OutputToken{query_result_size});
        }
      });
    }));
}

void
Predictor::Free() {
  delete static_cast<PredThreadPool*>(thread_pool_handle_);
}

static inline
std::vector<size_t> SplitBatch(const DMatrix* dmat, size_t split_factor) {
  const size_t num_row = dmat->GetNumRow();
  TREELITE_CHECK_LE(split_factor, num_row);
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

template <typename LeafOutputType>
class ShrinkResultToFit {
 public:
  inline static void Dispatch(
      size_t num_row, size_t query_size_per_instance, size_t num_class,
      PredictorOutputHandle out_result);
};

template <typename LeafOutputType>
class AllocateOutputVector {
 public:
  inline static PredictorOutputHandle Dispatch(size_t size);
};

template <typename LeafOutputType>
class DeallocateOutputVector {
 public:
  inline static void Dispatch(PredictorOutputHandle output_vector);
};

size_t
Predictor::PredictBatch(
    const DMatrix* dmat, int verbose, bool pred_margin, PredictorOutputHandle out_result) const {
  const double tstart = GetTime();

  const size_t num_row = dmat->GetNumRow();
  auto* pool = static_cast<PredThreadPool*>(thread_pool_handle_);
  InputToken request{dmat, pred_margin, pred_func_.get(), 0, num_row, out_result};
  OutputToken response;
  TREELITE_CHECK_GT(num_row, 0);
  const int nthread = std::min(num_worker_thread_, static_cast<int>(num_row));
  const std::vector<size_t> row_ptr = SplitBatch(dmat, nthread);
  for (int tid = 0; tid < nthread - 1; ++tid) {
    request.rbegin = row_ptr[tid];
    request.rend = row_ptr[tid + 1];
    pool->SubmitTask(tid, request);
  }
  size_t total_size = 0;
  {
    // assign work to the main thread
    const size_t rbegin = row_ptr[nthread - 1];
    const size_t rend = row_ptr[nthread];
    const size_t query_result_size
      = pred_func_->PredictBatch(dmat, rbegin, rend, pred_margin, out_result);
    total_size += query_result_size;
  }
  for (int tid = 0; tid < nthread - 1; ++tid) {
    if (pool->WaitForTask(tid, &response)) {
      total_size += response.query_result_size;
    }
  }
  // re-shape output if total_size < dimension of out_result
  if (total_size < QueryResultSize(dmat, 0, num_row)) {
    TREELITE_CHECK_GT(num_class_, 1);
    TREELITE_CHECK_EQ(total_size % num_row, 0);
    const size_t query_size_per_instance = total_size / num_row;
    TREELITE_CHECK_GT(query_size_per_instance, 0);
    TREELITE_CHECK_LT(query_size_per_instance, num_class_);
    DispatchWithTypeInfo<ShrinkResultToFit>(
        leaf_output_type_, num_row, query_size_per_instance, num_class_, out_result);
  }
  const double tend = GetTime();
  if (verbose > 0) {
    TREELITE_LOG(INFO) << "Treelite: Finished prediction in " << tend - tstart << " sec";
  }
  return total_size;
}

PredictorOutputHandle
Predictor::CreateOutputVector(const DMatrix* dmat) const {
  const size_t output_vector_size = this->QueryResultSize(dmat);
  return DispatchWithTypeInfo<AllocateOutputVector>(leaf_output_type_, output_vector_size);
}

void
Predictor::DeleteOutputVector(PredictorOutputHandle output_vector) const {
  DispatchWithTypeInfo<DeallocateOutputVector>(leaf_output_type_, output_vector);
}

template <typename LeafOutputType>
void
ShrinkResultToFit<LeafOutputType>::Dispatch(
    size_t num_row, size_t query_size_per_instance, size_t num_class,
    PredictorOutputHandle out_result) {
  auto* out_result_ = static_cast<LeafOutputType*>(out_result);
  for (size_t rid = 0; rid < num_row; ++rid) {
    for (size_t k = 0; k < query_size_per_instance; ++k) {
      out_result_[rid * query_size_per_instance + k] = out_result_[rid * num_class + k];
    }
  }
}

template <typename LeafOutputType>
PredictorOutputHandle
AllocateOutputVector<LeafOutputType>::Dispatch(size_t size) {
  return static_cast<PredictorOutputHandle>(new LeafOutputType[size]);
}

template <typename LeafOutputType>
void
DeallocateOutputVector<LeafOutputType>::Dispatch(PredictorOutputHandle output_vector) {
  delete[] (static_cast<LeafOutputType*>(output_vector));
}

}  // namespace predictor
}  // namespace treelite
