/*!
 * Copyright (c) 2017-2023 by Contributors
 * \file logging.cc
 * \author Hyunsu Cho
 * \brief logging facility for treelite
 */

#include <treelite/logging.h>

namespace treelite {

void LogMessage::Log(std::string const& msg) {
  LogCallbackRegistry const* registry = LogCallbackRegistryStore::Get();
  auto callback = registry->GetCallbackLogInfo();
  callback(msg.c_str());
}

void LogMessageWarning::Log(std::string const& msg) {
  LogCallbackRegistry const* registry = LogCallbackRegistryStore::Get();
  auto callback = registry->GetCallbackLogWarning();
  callback(msg.c_str());
}

}  // namespace treelite
