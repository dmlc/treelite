/*!
 * Copyright (c) 2017 by Contributors
 * \file c_api_common.cc
 * \author Hyunsu Cho
 * \brief C API of treelite (this file is used by both runtime and main package)
 */

#include <treelite/logging.h>
#include <treelite/c_api_common.h>
#include "./c_api_error.h"

using namespace treelite;

int TreeliteRegisterLogCallback(void (*callback)(const char*)) {
  API_BEGIN();
  LogCallbackRegistry* registry = LogCallbackRegistryStore::Get();
  registry->Register(callback);
  API_END();
}
