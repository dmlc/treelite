/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file c_api_error.cc
 * \author Hyunsu Cho
 * \brief C error handling
 */
#include <treelite/thread_local.h>
#include <treelite/c_api_error.h>
#include <treelite/version.h>
#include <string>
#include <sstream>

// Stringify an integer macro constant
#define STR_IMPL_(x) #x
#define STR(x) STR_IMPL_(x)

namespace {

struct TreeliteAPIErrorEntry {
  std::string last_error;
  std::string version_str;
};

using TreeliteAPIErrorStore = treelite::ThreadLocalStore<TreeliteAPIErrorEntry>;

}  // anonymous namespace

const char* TreeliteGetLastError() {
  return TreeliteAPIErrorStore::Get()->last_error.c_str();
}

void TreeliteAPISetLastError(const char* msg) {
  TreeliteAPIErrorStore::Get()->last_error = msg;
}

const char* TreeliteQueryTreeliteVersion() {
  std::ostringstream oss;
  oss << TREELITE_VER_MAJOR << "." << TREELITE_VER_MINOR << "." <<  TREELITE_VER_PATCH;
  std::string& version_str = TreeliteAPIErrorStore::Get()->version_str;
  version_str = oss.str();
  return version_str.c_str();
}

const char* TREELITE_VERSION = "TREELITE_VERSION_" STR(TREELITE_VER_MAJOR) "."
    STR(TREELITE_VER_MINOR) "." STR(TREELITE_VER_PATCH);
