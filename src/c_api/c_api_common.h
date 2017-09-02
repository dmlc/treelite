/*!
 * Copyright (c) 2017 by Contributors
 * \file c_api_common.h
 * \author Philip Cho
 * \brief C API of tree-lite (this file is used by both runtime and main package)
 */

#ifndef TREELITE_C_API_C_API_COMMON_H_
#define TREELITE_C_API_C_API_COMMON_H_

#include <dmlc/thread_local.h>

/*! \brief entry to to easily hold returning information */
struct TreeliteAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
};

// define threadlocal store for returning information
using TreeliteAPIThreadLocalStore
  = dmlc::ThreadLocalStore<TreeliteAPIThreadLocalEntry>;

#endif  // TREELITE_C_API_C_API_COMMON_H_
