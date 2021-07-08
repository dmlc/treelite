/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file c_api_error.cc
 * \author Hyunsu Cho
 * \brief C error handling
 */
#include <treelite/thread_local.h>
#include <treelite/c_api_error.h>
#include <string>

struct TreeliteAPIErrorEntry {
  std::string last_error;
};

using TreeliteAPIErrorStore = treelite::ThreadLocalStore<TreeliteAPIErrorEntry>;

const char* TreeliteGetLastError() {
  return TreeliteAPIErrorStore::Get()->last_error.c_str();
}

void TreeliteAPISetLastError(const char* msg) {
  TreeliteAPIErrorStore::Get()->last_error = msg;
}
