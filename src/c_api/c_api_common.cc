/*!
 * Copyright (c) 2017 by Contributors
 * \file c_api_common.cc
 * \author Philip Cho
 * \brief C API of tree-lite (this file is used by both runtime and main package)
 */

#include <treelite/logging.h>
#include <treelite/c_api_common.h>
#include "./c_api_error.h"

using namespace treelite;

int TreeliteOpenMPSupported() {
#if OPENMP_SUPPORT
  return 1;
#else
  return 0;
#endif
}

#if _MSC_VER
const char* TreeliteVarsallBatPath() {
  return TREELITE_MSVC_VARSALL_BAT;
}
#endif

int TreeliteRegisterLogCallback(void (*callback)(const char*)) {
  API_BEGIN();
  LogCallbackRegistry* registry = LogCallbackRegistryStore::Get();
  registry->Register(callback);
  API_END();
}
