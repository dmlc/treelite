/*!
 * Copyright 2017 by Contributors
 * \file logging.h
 * \brief logging facility for treelite
 * \author Philip Cho
 */
#ifndef TREELITE_LOGGING_H_
#define TREELITE_LOGGING_H_

#include <dmlc/thread_local.h>
#include <iostream>

namespace treelite {

class LogCallbackRegistry {
 public:
  using Callback = void (*)(const char*);
  LogCallbackRegistry()
    : log_callback_([] (const char* msg) { std::cerr << msg << std::endl; }) {}
  inline void Register(Callback log_callback) {
    this->log_callback_ = log_callback;
  }
  inline Callback Get() const {
    return log_callback_;
  }
 private:
  Callback log_callback_;
};

using LogCallbackRegistryStore = dmlc::ThreadLocalStore<LogCallbackRegistry>;

}  // namespace treelite

#endif  // TREELITE_LOGGING_H_
