/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file logging.cc
 * \author Hyunsu Cho
 * \brief logging facility for treelite
 */

#include <treelite/logging.h>

// Override logging mechanism
void treelite::LogMessage::Log(const std::string& msg) {
  const treelite::LogCallbackRegistry* registry = treelite::LogCallbackRegistryStore::Get();
  auto callback = registry->Get();
  callback(msg.c_str());
}
