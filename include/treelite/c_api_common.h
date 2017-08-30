/*!
 * Copyright (c) 2017 by Contributors
 * \file c_api_common.h
 * \author Philip Cho
 * \brief C API of tree-lite, used for interfacing with other languages
 *        This header is used by both the runtime and the main package
 */

/* Note: Make sure to use slash-asterisk form of comments in this file
   (like this one). Do not use double-slash (//). */
 
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

/* opaque handles */
typedef void* DMatrixHandle;

/*!
 * \brief display last error; can be called by different threads
 * \return error string
 */
TREELITE_DLL const char* TreeliteGetLastError();

/***************************************************************************
 * Part 1: data matrix interface                                           *
 ***************************************************************************/
/*!
 * \brief create DMatrix from a file
 * \param path file path
 * \param format file format
 * \param nthread number of threads to use
 * \param verbose whether to produce extra messages
 * \param out the created DMatrix
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixCreateFromFile(const char* path,
                                               const char* format,
                                               int nthread,
                                               int verbose,
                                               DMatrixHandle* out);
/*!
 * \brief create DMatrix from a (in-memory) CSR matrix
 * \param data feature values
 * \param col_ind feature indices
 * \param row_ptr pointer to row headers
 * \param num_row number of rows
 * \param num_col number of columns
 * \param out the created DMatrix
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixCreateFromCSR(const float* data,
                                              const unsigned* col_ind,
                                              const size_t* row_ptr,
                                              size_t num_row,
                                              size_t num_col,
                                              DMatrixHandle* out);
/*!
 * \brief create DMatrix from a (in-memory) dense matrix
 * \param data feature values
 * \param num_row number of rows
 * \param num_col number of columns
 * \param missing_value value to represent missing value
 * \param out the created DMatrix
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixCreateFromMat(const float* data,
                                              size_t num_row,
                                              size_t num_col,
                                              float missing_value,
                                              DMatrixHandle* out);
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
 * \brief produce a human-readable preview of a DMatrix
 * Will print first and last 25 non-zero entries, along with their locations
 * \param handle handle to DMatrix
 * \param out_preview used to save the address of the string literal
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixGetPreview(DMatrixHandle handle,
                                           const char** out_preview);
/*!
 * \brief delete DMatrix from memory
 * \param handle handle to DMatrix
 * \return 0 for success, -1 for failure
 */
TREELITE_DLL int TreeliteDMatrixFree(DMatrixHandle handle);

#endif  // TREELITE_C_API_COMMON_H_