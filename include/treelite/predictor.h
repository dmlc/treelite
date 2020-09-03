/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file predictor.h
 * \author Hyunsu Cho
 * \brief Load prediction function exported as a shared library
 */
#ifndef TREELITE_PREDICTOR_H_
#define TREELITE_PREDICTOR_H_

#include <dmlc/logging.h>
#include <treelite/typeinfo.h>
#include <treelite/data.h>
#include <string>
#include <cstdint>

namespace treelite {
namespace predictor {

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

class PredictorOutput {
 public:
  virtual size_t GetNumRow() const = 0;
  virtual size_t GetNumOutputGroup() const = 0;

  PredictorOutput() = default;
  virtual ~PredictorOutput() = default;

  static std::unique_ptr<PredictorOutput> Create(
      TypeInfo leaf_output_type, size_t num_row, size_t num_output_group);
};

template<typename LeafOutputType>
class PredictorOutputImpl : public PredictorOutput {
 private:
  std::vector<LeafOutputType> preds_;
  size_t num_row_;
  size_t num_output_group_;

  friend class PredictorOutput;

 public:
  size_t GetNumRow() const override;
  size_t GetNumOutputGroup() const override;
  std::vector<LeafOutputType>& GetPreds();
  const std::vector<LeafOutputType>& GetPreds() const;

  PredictorOutputImpl(size_t num_row, size_t num_output_group);
};

class SharedLibrary {
 public:
  using LibraryHandle = void*;
  using FunctionHandle = void*;
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
                                              int num_output_group);
  PredFunction() = default;
  virtual ~PredFunction() = default;
  virtual TypeInfo GetThresholdType() const = 0;
  virtual TypeInfo GetLeafOutputType() const = 0;
  virtual size_t PredictBatch(
      const DMatrix* dmat, size_t rbegin, size_t rend, bool pred_margin,
      PredictorOutput* out_pred) const = 0;
};

template<typename ThresholdType, typename LeafOutputType>
class PredFunctionImpl : public PredFunction {
 public:
  using PredFuncHandle = void*;
  PredFunctionImpl(const SharedLibrary& library, int num_feature, int num_output_group);
  TypeInfo GetThresholdType() const override;
  TypeInfo GetLeafOutputType() const override;
  size_t PredictBatch(
      const DMatrix* dmat, size_t rbegin, size_t rend, bool pred_margin,
      PredictorOutput* out_pred) const override;

 private:
  PredFuncHandle handle_;
  int num_feature_;
  int num_output_group_;
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
   * \param out_result resulting output vector
   * \return length of the output vector, which is guaranteed to be less than
   *         or equal to QueryResultSize()
   */
  size_t PredictBatch(
      const DMatrix* dmat, int verbose, bool pred_margin, PredictorOutput* out_result) const;
  /*!
   * \brief Allocate a buffer space that's sufficient to hold predicton for a given data matrix.
   *        The size of the buffer is given by QueryResultSize().
   * \param dmat a batch of rows
   * \return Newly allocated buffer space
   */
  std::unique_ptr<PredictorOutput> AllocateOutputBuffer(const DMatrix* dmat) const;
  /*!
   * \brief Given a batch of data rows, query the necessary size of array to
   *        hold predictions for all data points.
   * \param dmat a batch of rows
   * \return length of prediction array
   */
  inline size_t QueryResultSize(const DMatrix* dmat) const {
    CHECK(pred_func_) << "A shared library needs to be loaded first using Load()";
    return dmat->GetNumRow() * num_output_group_;
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
    CHECK(pred_func_) << "A shared library needs to be loaded first using Load()";
    CHECK(rbegin < rend && rend <= dmat->GetNumRow());
    return (rend - rbegin) * num_output_group_;
  }
  /*!
   * \brief Get the number of output groups in the loaded model
   * The number is 1 for most tasks;
   * it is greater than 1 for multiclass classifcation.
   * \return length of prediction array
   */
  inline size_t QueryNumOutputGroup() const {
    return num_output_group_;
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
   * \brief Get global bias which adjusting predicted margin scores
   * \return global bias
   */
  inline float QueryGlobalBias() const {
    return global_bias_;
  }

 private:
  SharedLibrary lib_;
  std::unique_ptr<PredFunction> pred_func_;
  ThreadPoolHandle thread_pool_handle_;
  size_t num_output_group_;
  size_t num_feature_;
  std::string pred_transform_;
  float sigmoid_alpha_;
  float global_bias_;
  int num_worker_thread_;
  TypeInfo threshold_type_;
  TypeInfo leaf_output_type_;
};

}  // namespace predictor
}  // namespace treelite

#endif  // TREELITE_PREDICTOR_H_
