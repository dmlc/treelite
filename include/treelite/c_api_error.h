/*!
 * Copyright (c) 2017-2023 by Contributors
 * \file c_api_error.h
 * \author Hyunsu Cho
 * \brief Error handling for C API.
 */
#ifndef TREELITE_C_API_ERROR_H_
#define TREELITE_C_API_ERROR_H_

#include <stdexcept>

/*! \brief Macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*! \brief Every function starts with API_BEGIN();
     and finishes with API_END() or API_END_HANDLE_ERROR */
#define API_END()                                \
  }                                              \
  catch (std::exception & _except_) {            \
    return TreeliteAPIHandleException(_except_); \
  }                                              \
  return 0
/*!
 * \brief Every function starts with API_BEGIN();
 *   and finishes with API_END() or API_END_HANDLE_ERROR()
 *   "Finalize" contains procedure to cleanup states when an error happens
 */
#define API_END_HANDLE_ERROR(Finalize)           \
  }                                              \
  catch (std::exception & _except_) {            \
    Finalize;                                    \
    return TreeliteAPIHandleException(_except_); \
  }                                              \
  return 0

/*!
 * \brief Set the last error message needed by C API
 * \param msg Error message to set.
 */
void TreeliteAPISetLastError(char const* msg);
/*!
 * \brief handle Exception thrown out
 * \param e Exception object
 * \return The return value of API after exception is handled
 */
inline int TreeliteAPIHandleException(std::exception const& e) {
  TreeliteAPISetLastError(e.what());
  return -1;
}
#endif  // TREELITE_C_API_ERROR_H_
