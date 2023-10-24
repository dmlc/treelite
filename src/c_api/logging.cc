/*!
 * Copyright (c) 2023 by Contributors
 * \file logging.cc
 * \author Hyunsu Cho
 * \brief C API for logging functions
 */

#include <treelite/c_api.h>
#include <treelite/c_api_error.h>
#include <treelite/logging.h>

int TreeliteRegisterLogCallback(void (*callback)(char const*)) {
  API_BEGIN();
  auto* registry = treelite::LogCallbackRegistryStore::Get();
  registry->RegisterCallBackLogInfo(callback);
  API_END();
}

int TreeliteRegisterWarningCallback(void (*callback)(char const*)) {
  API_BEGIN();
  auto* registry = treelite::LogCallbackRegistryStore::Get();
  registry->RegisterCallBackLogWarning(callback);
  API_END();
}
