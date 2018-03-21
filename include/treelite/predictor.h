/*!
 * Copyright (c) 2017 by Contributors
 * \file predictor.h
 * \author Philip Cho
 * \brief Load prediction function exported as a shared library
 */
#ifndef TREELITE_PREDICTOR_H_
#define TREELITE_PREDICTOR_H_

#include <dmlc/logging.h>
#include <cstdint>

namespace treelite {

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
  /*! \brief data layout. The value -1 signifies the missing value.
      When the "missing" field is set to -1, the "fvalue" field is set to
      NaN (Not a Number), so there is no danger for mistaking between
      missing values and non-missing values. */
  union Entry {
    int missing;
    float fvalue;
    // may contain extra fields later, such as qvalue
  };

  /*! \brief opaque handle types */
  typedef void* QueryFuncHandle;
  typedef void* PredFuncHandle;
  typedef void* PredTransformFuncHandle;
  typedef void* LibraryHandle;
  typedef void* ThreadPoolHandle;

  Predictor();
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
   * \param pred_margin whether to produce raw margin scores instead of
   *                    transformed probabilities
   * \param out_result resulting output vector; use
   *                   QueryResultSize() to allocate sufficient space
   * \return length of the output vector, which is guaranteed to be less than
   *         or equal to QueryResultSize()
   */
  size_t PredictBatch(const CSRBatch* batch,
                      bool pred_margin, float* out_result);
  size_t PredictBatch(const DenseBatch* batch,
                      bool pred_margin, float* out_result);

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
  inline size_t QueryResultSize(const DenseBatch* batch) const {
    CHECK(pred_func_handle_ != nullptr)
      << "A shared library needs to be loaded first using Load()";
    return batch->num_row * num_output_group_;
  }
  inline size_t QueryResultSize(const CSRBatch* batch,
                                size_t rbegin, size_t rend) const {
    CHECK(pred_func_handle_ != nullptr)
      << "A shared library needs to be loaded first using Load()";
    CHECK(rbegin < rend && rend <= batch->num_row);
    return (rend - rbegin) * num_output_group_;
  }
  inline size_t QueryResultSize(const DenseBatch* batch,
                                size_t rbegin, size_t rend) const {
    CHECK(pred_func_handle_ != nullptr)
      << "A shared library needs to be loaded first using Load()";
    CHECK(rbegin < rend && rend <= batch->num_row);
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

 private:
  LibraryHandle lib_handle_;
  QueryFuncHandle query_func_handle_;
  PredFuncHandle pred_func_handle_;
  PredTransformFuncHandle pred_transform_func_handle_;
  ThreadPoolHandle thread_pool_handle_;
  size_t num_output_group_;
};

}  // namespace treelite

#endif  // TREELITE_PREDICTOR_H_
