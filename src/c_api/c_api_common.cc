/*!
 * Copyright (c) 2017 by Contributors
 * \file c_api_common.cc
 * \author Philip Cho
 * \brief C API of tree-lite (this file is used by both runtime and main package)
 */

#include <treelite/data.h>
#include <treelite/c_api_common.h>
#include "./c_api_common.h"
#include "./c_api_error.h"
#include "../common/math.h"

using namespace treelite;

#if _MSC_VER
const char* TreeliteVarsallBatPath() {
  return TREELITE_MSVC_VARSALL_BAT;
}
#endif

int TreeliteDMatrixCreateFromFile(const char* path,
                                  const char* format,
                                  int nthread,
                                  int verbose,
                                  DMatrixHandle* out) {
  API_BEGIN();
  *out = static_cast<DMatrixHandle>(DMatrix::Create(path, format,
                                    nthread, verbose));
  API_END();
}

int TreeliteDMatrixCreateFromCSR(const float* data,
                                 const unsigned* col_ind,
                                 const size_t* row_ptr,
                                 size_t num_row,
                                 size_t num_col,
                                 DMatrixHandle* out) {
  API_BEGIN();
  DMatrix* dmat = new DMatrix();
  dmat->Clear();
  auto& data_ = dmat->data;
  auto& col_ind_ = dmat->col_ind;
  auto& row_ptr_ = dmat->row_ptr;
  data_.reserve(row_ptr[num_row]);
  col_ind_.reserve(row_ptr[num_row]);
  row_ptr_.reserve(num_row + 1);
  for (size_t i = 0; i < num_row; ++i) {
    const size_t jbegin = row_ptr[i];
    const size_t jend = row_ptr[i + 1];
    for (size_t j = jbegin; j < jend; ++j) {
      if (!common::math::CheckNAN(data[j])) {  // skip NaN
        data_.push_back(data[j]);
        CHECK_LT(col_ind[j], std::numeric_limits<uint32_t>::max())
          << "feature index too big to fit into uint32_t";
        col_ind_.push_back(static_cast<uint32_t>(col_ind[j]));
      }
    }
    row_ptr_.push_back(data_.size());
  }
  data_.shrink_to_fit();
  col_ind_.shrink_to_fit();
  dmat->num_row = num_row;
  dmat->num_col = num_col;
  dmat->nelem = data_.size();  // some nonzeros may have been deleted as NAN

  *out = static_cast<DMatrixHandle>(dmat);
  API_END();
}

int TreeliteDMatrixCreateFromMat(const float* data,
                                 size_t num_row,
                                 size_t num_col,
                                 float missing_value,
                                 DMatrixHandle* out) {
  const bool nan_missing = common::math::CheckNAN(missing_value);
  API_BEGIN();
  CHECK_LT(num_col, std::numeric_limits<uint32_t>::max())
    << "num_col argument is too big";
  DMatrix* dmat = new DMatrix();
  dmat->Clear();
  auto& data_ = dmat->data;
  auto& col_ind_ = dmat->col_ind;
  auto& row_ptr_ = dmat->row_ptr;
  // make an educated guess for initial sizes,
  // so as to present initial wave of allocation
  const size_t guess_size
    = std::min(std::min(num_row * num_col, num_row * 1000),
               static_cast<size_t>(64 * 1024 * 1024));
  data_.reserve(guess_size);
  col_ind_.reserve(guess_size);
  row_ptr_.reserve(num_row + 1);
  const float* row = &data[0];  // points to beginning of each row
  for (size_t i = 0; i < num_row; ++i, row += num_col) {
    for (size_t j = 0; j < num_col; ++j) {
      if (common::math::CheckNAN(row[j])) {
        CHECK(nan_missing)
          << "The missing_value argument must be set to NaN if there is any "
          << "NaN in the matrix.";
      } else if (nan_missing || row[j] != missing_value) {
        // row[j] is a valid entry
        data_.push_back(row[j]);
        col_ind_.push_back(static_cast<uint32_t>(j));
      }
    }
    row_ptr_.push_back(data_.size());
  }
  data_.shrink_to_fit();
  col_ind_.shrink_to_fit();
  dmat->num_row = num_row;
  dmat->num_col = num_col;
  dmat->nelem = data_.size();  // some nonzeros may have been deleted as NaN

  *out = static_cast<DMatrixHandle>(dmat);
  API_END();
}

int TreeliteDMatrixGetDimension(DMatrixHandle handle,
                                size_t* out_num_row,
                                size_t* out_num_col,
                                size_t* out_nelem) {
  API_BEGIN();
  const DMatrix* dmat = static_cast<DMatrix*>(handle);
  *out_num_row = dmat->num_row;
  *out_num_col = dmat->num_col;
  *out_nelem = dmat->nelem;
  API_END();
}

int TreeliteDMatrixGetPreview(DMatrixHandle handle,
                              const char** out_preview) {
  API_BEGIN();
  const DMatrix* dmat = static_cast<DMatrix*>(handle);
  std::string& ret_str = TreeliteAPIThreadLocalStore::Get()->ret_str;
  std::ostringstream oss;
  for (size_t i = 0; i < 25; ++i) {
    const size_t row_ind =
      std::upper_bound(&dmat->row_ptr[0], &dmat->row_ptr[dmat->num_row + 1], i)
        - &dmat->row_ptr[0] - 1;
    oss << "  (" << row_ind << ", " << dmat->col_ind[i] << ")\t"
        << dmat->data[i] << "\n";
  }
  oss << "  :\t:\n";
  for (size_t i = dmat->nelem - 25; i < dmat->nelem; ++i) {
    const size_t row_ind =
      std::upper_bound(&dmat->row_ptr[0], &dmat->row_ptr[dmat->num_row + 1], i)
      - &dmat->row_ptr[0] - 1;
    oss << "  (" << row_ind << ", " << dmat->col_ind[i] << ")\t"
      << dmat->data[i] << "\n";
  }
  ret_str = oss.str();
  *out_preview = ret_str.c_str();
  API_END();
}

int TreeliteDMatrixFree(DMatrixHandle handle) {
  API_BEGIN();
  delete static_cast<DMatrix*>(handle);
  API_END();
}