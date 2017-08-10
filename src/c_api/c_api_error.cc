/*!
 *  Copyright (c) 2017 by Contributors
 * \file c_api_error.cc
 * \author Philip Cho
 * \brief C error handling
 */
#include <dmlc/thread_local.h>
#include "./c_api_error.h"

struct TreeliteAPIErrorEntry {
  std::string last_error;
};

typedef dmlc::ThreadLocalStore<TreeliteAPIErrorEntry> TreeliteAPIErrorStore;

const char* TreeliteGetLastError() {
  return TreeliteAPIErrorStore::Get()->last_error.c_str();
}

void TreeliteAPISetLastError(const char* msg) {
  TreeliteAPIErrorStore::Get()->last_error = msg;
}
