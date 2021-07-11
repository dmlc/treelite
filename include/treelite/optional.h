/*!
 * Copyright (c) 2021 by Contributors
 * \file optional.h
 * \brief Backport of std::optional from C++17
 * \author Hyunsu Cho
 */

#ifndef TREELITE_OPTIONAL_H_
#define TREELITE_OPTIONAL_H_

namespace treelite {

template <typename T>
class optional {  // C++17: Switch to std::optional
 public:
  optional() : empty_{}, has_value_{false} {}

  explicit optional(const T& input_value) : value_{input_value}, has_value_{true} {}
  optional(optional<T>&& other) : has_value_{other} {
    if (other) {
      value_ = *other;
    }
  }

  ~optional() {
    if (has_value_) {
      value_.~T();
    } else {
      empty_.~empty_byte();
    }
  }

  explicit operator bool() const {
    return has_value_;
  }
  T& operator*() {
    return value_;
  }
  const T& operator*() const {
    return value_;
  }
  T* operator->() {
    return &value_;
  }
  optional& operator=(const T& new_value) {
    value_ = new_value;
    has_value_ = true;
    return *this;
  }

 private:
  struct empty_byte {};
  union {
    empty_byte empty_;
    T value_;
  };
  bool has_value_;
};

}  // namespace treelite

#endif  // TREELITE_OPTIONAL_H_
