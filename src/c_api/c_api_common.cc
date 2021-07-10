/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file c_api_common.cc
 * \author Hyunsu Cho
 * \brief C API of treelite (this file is used by both runtime and main package)
 */

#include <treelite/thread_local.h>
#include <treelite/logging.h>
#include <treelite/data.h>
#include <treelite/c_api_common.h>
#include <treelite/c_api_error.h>

using namespace treelite;

/*! \brief entry to to easily hold returning information */
struct TreeliteAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
};

// define threadlocal store for returning information
using TreeliteAPIThreadLocalStore = ThreadLocalStore<TreeliteAPIThreadLocalEntry>;

int TreeliteRegisterLogCallback(void (*callback)(const char*)) {
  API_BEGIN();
  LogCallbackRegistry* registry = LogCallbackRegistryStore::Get();
  registry->Register(callback);
  API_END();
}

int TreeliteDMatrixCreateFromCSR(
    const void* data, const char* data_type_str, const uint32_t* col_ind, const size_t* row_ptr,
    size_t num_row, size_t num_col, DMatrixHandle* out) {
  API_BEGIN();
  TypeInfo data_type = GetTypeInfoByName(data_type_str);
  std::unique_ptr<DMatrix> matrix
    = CSRDMatrix::Create(data_type, data, col_ind, row_ptr, num_row, num_col);
  *out = static_cast<DMatrixHandle>(matrix.release());
  API_END();
}

int TreeliteDMatrixCreateFromMat(
    const void* data, const char* data_type_str, size_t num_row, size_t num_col,
    const void* missing_value, DMatrixHandle* out) {
  API_BEGIN();
  TypeInfo data_type = GetTypeInfoByName(data_type_str);
  std::unique_ptr<DMatrix> matrix
    = DenseDMatrix::Create(data_type, data, missing_value, num_row, num_col);
  *out = static_cast<DMatrixHandle>(matrix.release());
  API_END();
}

int TreeliteDMatrixGetDimension(DMatrixHandle handle,
                                size_t* out_num_row,
                                size_t* out_num_col,
                                size_t* out_nelem) {
  API_BEGIN();
  const DMatrix* dmat = static_cast<DMatrix*>(handle);
  *out_num_row = dmat->GetNumRow();
  *out_num_col = dmat->GetNumCol();
  *out_nelem = dmat->GetNumElem();
  API_END();
}

int TreeliteDMatrixFree(DMatrixHandle handle) {
  API_BEGIN();
  delete static_cast<DMatrix*>(handle);
  API_END();
}
