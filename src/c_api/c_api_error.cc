/*!
 * Copyright (c) 2017-2023 by Contributors
 * \file c_api_error.cc
 * \author Hyunsu Cho
 * \brief C error handling
 */
#include <treelite/c_api.h>
#include <treelite/c_api_error.h>
#include <treelite/thread_local.h>
#include <treelite/version.h>

#include <string>

namespace treelite::c_api {

struct APIErrorEntry {
  std::string last_error;
  std::string version_str;
};

using APIErrorStore = ThreadLocalStore<APIErrorEntry>;

}  // namespace treelite::c_api

char const* TreeliteGetLastError() {
  return treelite::c_api::APIErrorStore::Get()->last_error.c_str();
}

void TreeliteAPISetLastError(char const* msg) {
  treelite::c_api::APIErrorStore::Get()->last_error = msg;
}

char const* TreeliteQueryTreeliteVersion() {
  auto& version_str = treelite::c_api::APIErrorStore::Get()->version_str;
  version_str = TREELITE_VERSION_STR;
  return version_str.c_str();
}

char const* TREELITE_VERSION = "TREELITE_VERSION_" TREELITE_VERSION_STR;
