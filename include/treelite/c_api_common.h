/*!
 * Copyright (c) 2017 by Contributors
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

#endif  // TREELITE_C_API_COMMON_H_
