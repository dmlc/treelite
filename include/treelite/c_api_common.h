/*!
 * Copyright (c) 2017-2020 by Contributors
 * \file c_api_common.h
 * \author Hyunsu Cho
 * \brief C API of Treelite, used for interfacing with other languages
 *        This header is used by both the runtime and the main package
 */

#ifndef TREELITE_C_API_COMMON_H_
#define TREELITE_C_API_COMMON_H_

#ifdef __cplusplus
#define TREELITE_EXTERN_C extern "C"
#include <cstdio>
#include <cstdint>
#else
#define TREELITE_EXTERN_C
#include <stdio.h>
#include <stdint.h>
#endif

/* special symbols for DLL library on Windows */
#if defined(_MSC_VER) || defined(_WIN32)
#define TREELITE_DLL TREELITE_EXTERN_C __declspec(dllexport)
#else
#define TREELITE_DLL TREELITE_EXTERN_C
#endif

/*! \brief handle to a data matrix */
typedef void* DMatrixHandle;

/*!
 * \brief display last error; can be called by multiple threads
 * Note. Each thread will get the last error occured in its own context.
 * \return error string
 */
TREELITE_DLL const char* TreeliteGetLastError(void);

/*!
 * \brief register callback function for LOG(INFO) messages -- helpful messages
 *        that are not errors.
 * Note: this function can be called by multiple threads. The callback function
 *       will run on the thread that registered it
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteRegisterLogCallback(void (*callback)(const char*));

/*!
 * \defgroup dmatrix
 * Data matrix interface
 * \{
 */
/*!
 * \brief create a sparse DMatrix from a file
 * \param path file path
 * \param format file format
 * \param nthread number of threads to use
 * \param verbose whether to produce extra messages
 * \param out the created DMatrix
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixCreateFromFile(
    const char* path, const char* format, int nthread, int verbose, DMatrixHandle* out);
/*!
 * \brief create DMatrix from a (in-memory) CSR matrix
 * \param data feature values
 * \param data_type Type of data elements
 * \param col_ind feature indices
 * \param row_ptr pointer to row headers
 * \param num_row number of rows
 * \param num_col number of columns
 * \param out the created DMatrix
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixCreateFromCSR(
    const void* data, const char* data_type, const uint32_t* col_ind, const size_t* row_ptr,
    size_t num_row, size_t num_col, DMatrixHandle* out);
/*!
 * \brief create DMatrix from a (in-memory) dense matrix
 * \param data feature values
 * \param data_type Type of data elements
 * \param num_row number of rows
 * \param num_col number of columns
 * \param missing_value value to represent missing value
 * \param out the created DMatrix
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixCreateFromMat(
    const void* data, const char* data_type, size_t num_row, size_t num_col,
    const void* missing_value, DMatrixHandle* out);
/*!
 * \brief get dimensions of a DMatrix
 * \param handle handle to DMatrix
 * \param out_num_row used to set number of rows
 * \param out_num_col used to set number of columns
 * \param out_nelem used to set number of nonzero entries
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixGetDimension(DMatrixHandle handle,
                                             size_t* out_num_row,
                                             size_t* out_num_col,
                                             size_t* out_nelem);

/*!
 * \brief delete DMatrix from memory
 * \param handle handle to DMatrix
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixFree(DMatrixHandle handle);
/*! \} */

#endif  // TREELITE_C_API_COMMON_H_
