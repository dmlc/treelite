/*!
 * Copyright (c) 2017-2021 by Contributors
 * \file logging.h
 * \brief logging facility for Treelite
 * \author Hyunsu Cho
 */
#ifndef TREELITE_LOGGING_H_
#define TREELITE_LOGGING_H_

#include <treelite/thread_local.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <memory>
#include <cstdio>
#include <ctime>

namespace treelite {

/*!
 * \brief Exception class that will be thrown by Treelite
 */
struct Error : public std::runtime_error {
  explicit Error(const std::string& s) : std::runtime_error(s) {}
};

template <typename X, typename Y>
std::unique_ptr<std::string> LogCheckFormat(const X& x, const Y& y) {
  std::ostringstream os;
  os << " (" << x << " vs. " << y << ") ";
  /* CHECK_XX(x, y) requires x and y can be serialized to string. Use CHECK(x OP y) otherwise. */
  return std::make_unique<std::string>(os.str());
}

#if defined(__GNUC__) || defined(__clang__)
#define TREELITE_ALWAYS_INLINE inline __attribute__((__always_inline__))
#elif defined(_MSC_VER)
#define TREELITE_ALWAYS_INLINE __forceinline
#else
#define TREELITE_ALWAYS_INLINE inline
#endif

#define DEFINE_CHECK_FUNC(name, op)                                                        \
  template <typename X, typename Y>                                                        \
  TREELITE_ALWAYS_INLINE std::unique_ptr<std::string> LogCheck##name(const X& x, const Y& y) { \
    if (x op y) return nullptr;                                                            \
    return LogCheckFormat(x, y);                                                           \
  }                                                                                        \
  TREELITE_ALWAYS_INLINE std::unique_ptr<std::string> LogCheck##name(int x, int y) {           \
    return LogCheck##name<int, int>(x, y);                                                 \
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
DEFINE_CHECK_FUNC(_LT, <)
DEFINE_CHECK_FUNC(_GT, >)
DEFINE_CHECK_FUNC(_LE, <=)
DEFINE_CHECK_FUNC(_GE, >=)
DEFINE_CHECK_FUNC(_EQ, ==)
DEFINE_CHECK_FUNC(_NE, !=)
#pragma GCC diagnostic pop


#define CHECK_BINARY_OP(name, op, x, y)                  \
  if (auto __treelite__log__err = treelite::LogCheck##name(x, y))  \
      ::treelite::LogMessageFatal(__FILE__, __LINE__).stream() \
        << "Check failed: " << #x " " #op " " #y << *__treelite__log__err << ": "
#define CHECK(x)                                               \
  if (!(x))                                                    \
    treelite::LogMessageFatal(__FILE__, __LINE__).stream()     \
      << "Check failed: " #x << ": "
#define CHECK_LT(x, y) CHECK_BINARY_OP(_LT, <, x, y)
#define CHECK_GT(x, y) CHECK_BINARY_OP(_GT, >, x, y)
#define CHECK_LE(x, y) CHECK_BINARY_OP(_LE, <=, x, y)
#define CHECK_GE(x, y) CHECK_BINARY_OP(_GE, >=, x, y)
#define CHECK_EQ(x, y) CHECK_BINARY_OP(_EQ, ==, x, y)
#define CHECK_NE(x, y) CHECK_BINARY_OP(_NE, !=, x, y)

#define LOG_INFO treelite::LogMessage(__FILE__, __LINE__)
#define LOG_ERROR LOG_INFO
#define LOG_FATAL treelite::LogMessageFatal(__FILE__, __LINE__)
#define LOG(severity) LOG_##severity.stream()

class DateLogger {
 public:
  DateLogger() {
#if defined(_MSC_VER)
    _tzset();
#endif  // defined(_MSC_VER)
  }
  const char* HumanDate() {
#if defined(_MSC_VER)
    _strtime_s(buffer_, sizeof(buffer_));
#else  // defined(_MSC_VER)
    time_t time_value = std::time(nullptr);
    struct tm* pnow;
#if !defined(_WIN32)
    struct tm now;
    pnow = localtime_r(&time_value, &now);
#else  // !defined(_WIN32)
    pnow = std::localtime(&time_value);  // NOLINT(*)
#endif  // !defined(_WIN32)
    std::snprintf(buffer_, sizeof(buffer_), "%02d:%02d:%02d",
                  pnow->tm_hour, pnow->tm_min, pnow->tm_sec);
#endif  // defined(_MSC_VER)
    return buffer_;
  }

 private:
  char buffer_[9];
};

class LogMessageFatal {
 public:
  LogMessageFatal(const char* file, int line) {
    log_stream_ << "[" << pretty_date_.HumanDate() << "] " << file << ":" << line << ": ";
  }
  LogMessageFatal(const LogMessageFatal&) = delete;
  void operator=(const LogMessageFatal&) = delete;

  std::ostringstream& stream() {
    return log_stream_;
  }
  ~LogMessageFatal() noexcept(false) {
    throw Error(log_stream_.str());
  }

 private:
  std::ostringstream log_stream_;
  DateLogger pretty_date_;
};

class LogMessage {
 public:
  LogMessage(const char* file, int line) {
    log_stream_ << "[" << DateLogger().HumanDate() << "] " << file << ":"
                << line << ": ";
  }
  ~LogMessage() {
    Log(log_stream_.str());
  }
  std::ostream& stream() { return log_stream_; }
  static void Log(const std::string& msg);

 private:
  std::ostringstream log_stream_;
};

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

using LogCallbackRegistryStore = ThreadLocalStore<LogCallbackRegistry>;

}  // namespace treelite

#endif  // TREELITE_LOGGING_H_
