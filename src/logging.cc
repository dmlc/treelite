/*!
 * Copyright (c) 2017-2023 by Contributors
 * \file logging.cc
 * \author Hyunsu Cho
 * \brief logging facility for treelite
 */

#include <treelite/logging.h>

namespace treelite {

void LogMessage::Log(const std::string& msg) {
  const LogCallbackRegistry *registry = LogCallbackRegistryStore::Get();
  auto callback = registry->GetCallbackLogInfo();
  callback(msg.c_str());
}

void LogMessageWarning::Log(const std::string& msg) {
  const LogCallbackRegistry *registry = LogCallbackRegistryStore::Get();
  auto callback = registry->GetCallbackLogWarning();
  callback(msg.c_str());
}

}  // namespace treelite
