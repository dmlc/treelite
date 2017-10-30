/*!
 *  Copyright (c) 2017 by Contributors
 * \file c_api_error.h
 * \author Philip Cho
 * \brief Error handling for C API.
 */
#ifndef TREELITE_C_API_C_API_ERROR_H_
#define TREELITE_C_API_C_API_ERROR_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <treelite/c_api_common.h>

/*! \brief macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*! \brief every function starts with API_BEGIN();
     and finishes with API_END() or API_END_HANDLE_ERROR */
#define API_END()                                    \
     } catch(dmlc::Error &_except_) {                \
       return TreeliteAPIHandleException(_except_);  \
     }                                               \
     return 0
/*!
 * \brief every function starts with API_BEGIN();
 *   and finishes with API_END() or API_END_HANDLE_ERROR()
 *   "Finalize" contains procedure to cleanup states when an error happens
 */
#define API_END_HANDLE_ERROR(Finalize)          \
     } catch(dmlc::Error &_except_) {           \
       Finalize;                                \
       return TreeliteAPIHandleException(_except_);  \
     }                                          \
     return 0

/*!
 * \brief Set the last error message needed by C API
 * \param msg The error message to set.
 */
void TreeliteAPISetLastError(const char* msg);
/*!
 * \brief handle exception thrown out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
inline int TreeliteAPIHandleException(const dmlc::Error &e) {
  TreeliteAPISetLastError(e.what());
  return -1;
}
#endif  // TREELITE_C_API_C_API_ERROR_H_
