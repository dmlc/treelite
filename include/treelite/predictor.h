/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file predictor.h
 * \author Hyunsu Cho
 * \brief Load prediction function exported as a shared library
 */
#ifndef TREELITE_PREDICTOR_H_
#define TREELITE_PREDICTOR_H_

#include <treelite/logging.h>
#include <treelite/typeinfo.h>
#include <treelite/c_api_runtime.h>
#include <treelite/data.h>
#include <string>
#include <memory>
#include <mutex>
#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#endif  // _WIN32

namespace treelite {
namespace predictor {

/*!
 * \brief OMP Exception class catches, saves and rethrows exception from OMP blocks
 */
class OMPException {
 private:
  // exception_ptr member to store the exception
  std::exception_ptr omp_exception_;
  // mutex to be acquired during catch to set the exception_ptr
  std::mutex mutex_;

 public:
  /*!
   * \brief Parallel OMP blocks should be placed within Run to save exception
   */
  template <typename Function, typename... Parameters>
  void Run(Function f, Parameters... params) {
    try {
      f(params...);
    } catch (treelite::Error &ex) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!omp_exception_) {
        omp_exception_ = std::current_exception();
      }
    } catch (std::exception &ex) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!omp_exception_) {
        omp_exception_ = std::current_exception();
      }
    }
  }

  /*!
   * \brief should be called from the main thread to rethrow the exception
   */
  void Rethrow() {
    if (this->omp_exception_) std::rethrow_exception(this->omp_exception_);
  }
};

/*! \brief data layout. The value -1 signifies the missing value.
    When the "missing" field is set to -1, the "fvalue" field is set to
    NaN (Not a Number), so there is no danger for mistaking between
    missing values and non-missing values. */
template <typename ElementType>
union Entry {
  int missing;
  ElementType fvalue;
  // may contain extra fields later, such as qvalue
};

class SharedLibrary {
 public:
#ifdef _WIN32
  using LibraryHandle = HMODULE;
  using FunctionHandle = FARPROC;
#else  // _WIN32
  using LibraryHandle = void*;
  using FunctionHandle = void*;
#endif  // _WIN32
  SharedLibrary();
  ~SharedLibrary();
  void Load(const char* libpath);
  FunctionHandle LoadFunction(const char* name) const;
  template<typename HandleType>
  HandleType LoadFunctionWithSignature(const char* name) const;

 private:
  LibraryHandle handle_;
  std::string libpath_;
};

class PredFunction {
 public:
  static std::unique_ptr<PredFunction> Create(TypeInfo threshold_type, TypeInfo leaf_output_type,
                                              const SharedLibrary& library, int num_feature,
                                              int num_class);
  PredFunction() = default;
  virtual ~PredFunction() = default;
  virtual TypeInfo GetThresholdType() const = 0;
  virtual TypeInfo GetLeafOutputType() const = 0;
  virtual size_t PredictBatch(const DMatrix* dmat, size_t rbegin, size_t rend, bool pred_margin,
                              PredictorOutputHandle out_pred) const = 0;
};

template<typename ThresholdType, typename LeafOutputType>
class PredFunctionImpl : public PredFunction {
 public:
  using PredFuncHandle = void*;
  PredFunctionImpl(const SharedLibrary& library, int num_feature, int num_class);
  TypeInfo GetThresholdType() const override;
  TypeInfo GetLeafOutputType() const override;
  size_t PredictBatch(const DMatrix* dmat, size_t rbegin, size_t rend, bool pred_margin,
                      PredictorOutputHandle out_pred) const override;

 private:
  PredFuncHandle handle_;
  int num_feature_;
  int num_class_;
};

/*! \brief predictor class: wrapper for optimized prediction code */
class Predictor {
 public:
  /*! \brief opaque handle types */
  typedef void* ThreadPoolHandle;

  explicit Predictor(int num_worker_thread = -1);
  ~Predictor();
  /*!
   * \brief load the prediction function from dynamic shared library.
   * \param libpath path of dynamic shared library (.so/.dll/.dylib).
   */
  void Load(const char* libpath);
  /*!
   * \brief unload the prediction function
   */
  void Free();
  /*!
   * \brief Make predictions on a batch of data rows (synchronously). This
   *        function internally divides the workload among all worker threads.
   * \param dmat a batch of rows
   * \param verbose whether to produce extra messages
   * \param pred_margin whether to produce raw margin scores instead of
   *                    transformed probabilities
   * \param out_result Resulting output vector. This pointer must point to an array of length
   *                   QueryResultSize() and of type QueryLeafOutputType().
   * \return length of the output vector, which is guaranteed to be less than
   *         or equal to QueryResultSize()
   */
  size_t PredictBatch(
      const DMatrix* dmat, int verbose, bool pred_margin, PredictorOutputHandle out_result) const;
  /*!
   * \brief Given a batch of data rows, query the necessary size of array to
   *        hold predictions for all data points.
   * \param dmat a batch of rows
   * \return length of prediction array
   */
  inline size_t QueryResultSize(const DMatrix* dmat) const {
    TREELITE_CHECK(pred_func_) << "A shared library needs to be loaded first using Load()";
    return dmat->GetNumRow() * num_class_;
  }
  /*!
   * \brief Given a batch of data rows, query the necessary size of array to
   *        hold predictions for all data points.
   * \param dmat a batch of rows
   * \param rbegin beginning of range of rows
   * \param rend end of range of rows
   * \return length of prediction array
   */
  inline size_t QueryResultSize(const DMatrix* dmat, size_t rbegin, size_t rend) const {
    TREELITE_CHECK(pred_func_) << "A shared library needs to be loaded first using Load()";
    TREELITE_CHECK(rbegin < rend && rend <= dmat->GetNumRow());
    return (rend - rbegin) * num_class_;
  }
  /*!
   * \brief Get the number of classes in the loaded model
   * The number is 1 for most tasks;
   * it is greater than 1 for multiclass classification.
   * \return length of prediction array
   */
  inline size_t QueryNumClass() const {
    return num_class_;
  }
  /*!
   * \brief Get the width (number of features) of each instance used to train
   *        the loaded model
   * \return number of features
   */
  inline size_t QueryNumFeature() const {
    return num_feature_;
  }
  /*!
   * \brief Get name of post prediction transformation used to train the loaded model
   * \return name of prediction transformation
   */
  inline std::string QueryPredTransform() const {
    return pred_transform_;
  }
  /*!
   * \brief Get alpha value in sigmoid transformation used to train the loaded model
   * \return alpha value in sigmoid transformation
   */
  inline float QuerySigmoidAlpha() const {
    return sigmoid_alpha_;
  }
  /*!
   * \brief Get c value in exponential standard ratio used to train the loaded model
   * \return c value in exponential standard ratio transformation
   */
  inline float QueryRatioC() const {
    return ratio_c_;
  }
  /*!
   * \brief Get global bias which adjusting predicted margin scores
   * \return global bias
   */
  inline float QueryGlobalBias() const {
    return global_bias_;
  }
  /*!
   * \brief Get the type of the split thresholds
   * \return type of the split thresholds
   */
  inline TypeInfo QueryThresholdType() const {
    return threshold_type_;
  }
  /*!
   * \brief Get the type of the leaf outputs
   * \return type of the leaf outputs
   */
  inline TypeInfo QueryLeafOutputType() const {
    return leaf_output_type_;
  }
  /*!
   * \brief Create an output vector suitable to hold prediction result for a given data matrix
   * \param dmat a data matrix
   * \return Opaque handle to the allocated output vector
   */
  PredictorOutputHandle CreateOutputVector(const DMatrix* dmat) const;
  /*!
   * \brief Free an output vector from memory
   * \param output_vector Opaque handle to the output vector
   */
  void DeleteOutputVector(PredictorOutputHandle output_vector) const;

 private:
  SharedLibrary lib_;
  std::unique_ptr<PredFunction> pred_func_;
  ThreadPoolHandle thread_pool_handle_;
  size_t num_class_;
  size_t num_feature_;
  std::string pred_transform_;
  float sigmoid_alpha_;
  float ratio_c_;
  float global_bias_;
  int num_worker_thread_;
  TypeInfo threshold_type_;
  TypeInfo leaf_output_type_;

  mutable OMPException exception_catcher_;
};

}  // namespace predictor
}  // namespace treelite

#endif  // TREELITE_PREDICTOR_H_
