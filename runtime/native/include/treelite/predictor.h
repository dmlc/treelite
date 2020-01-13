/*!
 * Copyright (c) 2017 by Contributors
 * \file predictor.h
 * \author Philip Cho
 * \brief Load prediction function exported as a shared library
 */
#ifndef TREELITE_PREDICTOR_H_
#define TREELITE_PREDICTOR_H_

#include <dmlc/logging.h>
#include <treelite/entry.h>
#include <cstdint>

namespace treelite {

namespace common {
namespace filesystem {
class TemporaryDirectory;  // forward declaration
}
}

/*! \brief sparse batch in Compressed Sparse Row (CSR) format */
struct CSRBatch {
  /*! \brief feature values */
  const float* data;
  /*! \brief feature indices */
  const uint32_t* col_ind;
  /*! \brief pointer to row headers; length of [num_row] + 1 */
  const size_t* row_ptr;
  /*! \brief number of rows */
  size_t num_row;
  /*! \brief number of columns (i.e. # of features used) */
  size_t num_col;
};

/*! \brief dense batch */
struct DenseBatch {
  /*! \brief feature values */
  const float* data;
  /*! \brief value representing the missing value (usually nan) */
  float missing_value;
  /*! \brief number of rows */
  size_t num_row;
  /*! \brief number of columns (i.e. # of features used) */
  size_t num_col;
};

/*! \brief predictor class: wrapper for optimized prediction code */
class Predictor {
 public:
  /*! \brief opaque handle types */
  typedef void* QueryFuncHandle;
  typedef void* PredFuncHandle;
  typedef void* LibraryHandle;
  typedef void* ThreadPoolHandle;

  Predictor(int num_worker_thread = -1);
  ~Predictor();
  /*!
   * \brief load the prediction function from dynamic shared library.
   * \param name name of dynamic shared library (.so/.dll/.dylib).
   */
  void Load(const char* name);
  /*!
   * \brief unload the prediction function
   */
  void Free();

  /*!
   * \brief Make predictions on a batch of data rows (synchronously). This
   *        function internally divides the workload among all worker threads.
   * \param batch a batch of rows
   * \param verbose whether to produce extra messages
   * \param pred_margin whether to produce raw margin scores instead of
   *                    transformed probabilities
   * \param out_result resulting output vector; use
   *                   QueryResultSize() to allocate sufficient space
   * \return length of the output vector, which is guaranteed to be less than
   *         or equal to QueryResultSize()
   */
  size_t PredictBatch(const CSRBatch* batch, int verbose,
                      bool pred_margin, float* out_result);
  size_t PredictBatch(const DenseBatch* batch, int verbose,
                      bool pred_margin, float* out_result);
  /*!
   * \brief Make predictions on a single data row (synchronously). The work
   *        will be scheduled to the calling thread.
   * \param inst single data row
   * \param pred_margin whether to produce raw margin scores instead of
   *                    transformed probabilities
   * \param out_result resulting output vector; use
   *                   QueryResultSizeSingleInst() to allocate sufficient space
   * \return length of the output vector, which is guaranteed to be less than
   *         or equal to QueryResultSizeSingleInst()
   */
  size_t PredictInst(TreelitePredictorEntry* inst, bool pred_margin,
                     float* out_result);

  /*!
   * \brief Given a batch of data rows, query the necessary size of array to
   *        hold predictions for all data points.
   * \param batch a batch of rows
   * \return length of prediction array
   */
  inline size_t QueryResultSize(const CSRBatch* batch) const {
    CHECK(pred_func_handle_ != nullptr)
      << "A shared library needs to be loaded first using Load()";
    return batch->num_row * num_output_group_;
  }
  /*!
   * \brief Given a batch of data rows, query the necessary size of array to
   *        hold predictions for all data points.
   * \param batch a batch of rows
   * \return length of prediction array
   */
  inline size_t QueryResultSize(const DenseBatch* batch) const {
    CHECK(pred_func_handle_ != nullptr)
      << "A shared library needs to be loaded first using Load()";
    return batch->num_row * num_output_group_;
  }
  /*!
   * \brief Given a batch of data rows, query the necessary size of array to
   *        hold predictions for all data points.
   * \param batch a batch of rows
   * \param rbegin beginning of range of rows
   * \param rend end of range of rows
   * \return length of prediction array
   */
  inline size_t QueryResultSize(const CSRBatch* batch,
                                size_t rbegin, size_t rend) const {
    CHECK(pred_func_handle_ != nullptr)
      << "A shared library needs to be loaded first using Load()";
    CHECK(rbegin < rend && rend <= batch->num_row);
    return (rend - rbegin) * num_output_group_;
  }
  /*!
   * \brief Given a batch of data rows, query the necessary size of array to
   *        hold predictions for all data points.
   * \param batch a batch of rows
   * \param rbegin beginning of range of rows
   * \param rend end of range of rows
   * \return length of prediction array
   */
  inline size_t QueryResultSize(const DenseBatch* batch,
                                size_t rbegin, size_t rend) const {
    CHECK(pred_func_handle_ != nullptr)
      << "A shared library needs to be loaded first using Load()";
    CHECK(rbegin < rend && rend <= batch->num_row);
    return (rend - rbegin) * num_output_group_;
  }
  /*!
   * \brief Query the necessary size of array to hold the prediction for a
   *        single data row
   * \return length of prediction array
   */
  inline size_t QueryResultSizeSingleInst() const {
    CHECK(pred_func_handle_ != nullptr)
      << "A shared library needs to be loaded first using Load()";
    return num_output_group_;
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
  LibraryHandle lib_handle_;
  QueryFuncHandle num_output_group_query_func_handle_;
  QueryFuncHandle num_feature_query_func_handle_;
  QueryFuncHandle pred_transform_query_func_handle_;
  QueryFuncHandle sigmoid_alpha_query_func_handle_;
  QueryFuncHandle global_bias_query_func_handle_;
  PredFuncHandle pred_func_handle_;
  ThreadPoolHandle thread_pool_handle_;
  size_t num_output_group_;
  size_t num_feature_;
  std::string pred_transform_;
  float sigmoid_alpha_;
  float global_bias_;
  int num_worker_thread_;

  bool using_remote_lib_;  // load lib from remote location?
  // information for temporary file to cache remote lib
  std::unique_ptr<common::filesystem::TemporaryDirectory> tempdir_;
  std::string temp_libfile_;

  template <typename BatchType>
  size_t PredictBatchBase_(const BatchType* batch, int verbose,
                           bool pred_margin, float* out_result);
};

}  // namespace treelite

#endif  // TREELITE_PREDICTOR_H_
