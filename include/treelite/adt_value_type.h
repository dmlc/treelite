/*!
 * Copyright 2019 by Contributors
 * \file adt_value_type.h
 * \brief Define an ADT (Abstract Data Type) for Treelite. We use type erasure so that we can
 *        accommodate multiple data types for thresholds and leaf values without littering the
 *        whole codebase with templates.
 * \author Philip Cho
 */

#ifndef TREELITE_ADT_VALUE_TYPE_H
#define TREELITE_ADT_VALUE_TYPE_H

#include <cstdint>
#include <type_traits>
#include <vector>
#include <dmlc/logging.h>
#include "adt_value_type_c_api.h"

namespace treelite {

namespace ADT {

class Value;

/*
 * RTTI (Run-Time Type Information), together with type erasure, is inspired by
 * https://github.com/dmlc/xgboost/blob/d2e1e4d/include/xgboost/json.h.
 */

class ValueImpl {
 public:
  enum class ValueKind : int8_t {
    kInt32 = TreeliteValueType::kTreeliteInt32,
    kFloat32 = TreeliteValueType::kTreeliteFloat32,
    kFloat64 = TreeliteValueType::kTreeliteFloat64,
  };

  explicit ValueImpl(ValueKind kind) : kind_(kind) {}
  virtual ~ValueImpl() = default;

  ValueKind Type() const {
    return kind_;
  }

  virtual bool operator==(const ValueImpl& rhs) const = 0;
  virtual ValueImpl& operator=(const ValueImpl& rhs) = 0;
  virtual bool operator<(const ValueImpl& rhs) const = 0;
  virtual std::string ToString() const = 0;
  virtual bool IsFinite() const = 0;
  virtual std::shared_ptr<ValueImpl> Clone(const void* new_data) = 0;

  std::string TypeStr() const;

 private:
  ValueKind kind_;
};

template <typename T>
bool IsA(const ValueImpl* value) {
  return T::isClassOf(value);
}

template <typename T, typename U>
T* Cast(U* value) {
  if (IsA<T>(value)) {
    return dynamic_cast<T*>(value);
  } else {
    LOG(FATAL) << "Invalid cast, from " + value->TypeStr() + " to " + T().TypeStr();
  }
  return dynamic_cast<T*>(value);  // suppress compiler warning.
}

class Int32Value : public ValueImpl {
 private:
  int32_t val_;
 public:
  Int32Value() : ValueImpl(ValueKind::kInt32) {}
  template <typename Integer,
    typename std::enable_if<
      std::is_same<Integer, int32_t>::value ||
      std::is_same<Integer, const int32_t>::value>::type* = nullptr>
  Int32Value(Integer val) : ValueImpl(ValueKind::kInt32), val_(val) {}

  const int32_t& GetValue() &&     { return val_; }
  const int32_t& GetValue() const& { return val_; }
  int32_t&       GetValue() &      { return val_; }

  bool operator==(const ValueImpl& rhs) const override;
  bool operator<(const ValueImpl& rhs) const override;
  Int32Value& operator=(const ValueImpl& rhs) override;
  std::string ToString() const override;
  bool IsFinite() const override;
  std::shared_ptr<ValueImpl> Clone(const void* new_data) override;

  static bool isClassOf(const ValueImpl* value) {
    return value->Type() == ValueKind::kInt32;
  }
};

class Float32Value : public ValueImpl {
 private:
  float val_;
 public:
  Float32Value() : ValueImpl(ValueKind::kFloat32) {}
  template <typename Float,
    typename std::enable_if<
      std::is_same<Float, float>::value ||
      std::is_same<Float, const float>::value>::type* = nullptr>
  Float32Value(Float val) : ValueImpl(ValueKind::kFloat32), val_(val) {}

  const float& GetValue() &&     { return val_; }
  const float& GetValue() const& { return val_; }
  float&       GetValue() &      { return val_; }

  bool operator==(const ValueImpl& rhs) const override;
  bool operator<(const ValueImpl& rhs) const override;
  Float32Value& operator=(const ValueImpl& rhs) override;
  std::string ToString() const override;
  bool IsFinite() const override;
  std::shared_ptr<ValueImpl> Clone(const void* new_data) override;

  static bool isClassOf(const ValueImpl* value) {
    return value->Type() == ValueKind::kFloat32;
  }
};

class Float64Value : public ValueImpl {
 private:
  double val_;
 public:
  Float64Value() : ValueImpl(ValueKind::kFloat64) {}
  template <typename Float,
    typename std::enable_if<
      std::is_same<Float, double>::value ||
      std::is_same<Float, const double>::value>::type* = nullptr>
  Float64Value(Float val) : ValueImpl(ValueKind::kFloat64), val_(val) {}

  const double& GetValue() &&     { return val_; }
  const double& GetValue() const& { return val_; }
  double&       GetValue() &      { return val_; }

  bool operator==(const ValueImpl& rhs) const override;
  bool operator<(const ValueImpl& rhs) const override;
  Float64Value& operator=(const ValueImpl& rhs) override;
  std::string ToString() const override;
  bool IsFinite() const override;
  std::shared_ptr<ValueImpl> Clone(const void* new_data) override;

  static bool isClassOf(const ValueImpl* value) {
    return value->Type() == ValueKind::kFloat64;
  }
};

class Value {
 public:
  Value() : ptr_(nullptr) {}
  Value(const Value& other) : ptr_(other.ptr_) {}
  Value(Value&& other) : ptr_(std::move(other.ptr_)) {}

  Value& operator=(const Value& other) = default;
  Value& operator=(Value&& other) {
    ptr_ = std::move(other.ptr_);
    return *this;
  }

  bool operator==(const Value& rhs) const {
    return *ptr_ == *(rhs.ptr_);
  }
  bool operator<(const Value& rhs) const {
    return *ptr_ < *(rhs.ptr_);
  }

  explicit Value(Int32Value val) : ptr_(new Int32Value(std::move(val))) {}
  Value& operator=(Int32Value val) {
    ptr_.reset(new Int32Value(std::move(val)));
    return *this;
  }

  explicit Value(Float32Value val) : ptr_(new Float32Value(std::move(val))) {}
  Value& operator=(Float32Value val) {
    ptr_.reset(new Float32Value(std::move(val)));
    return *this;
  }

  explicit Value(Float64Value val) : ptr_(new Float64Value(std::move(val))) {}
  Value& operator=(Float64Value val) {
    ptr_.reset(new Float64Value(std::move(val)));
    return *this;
  }

  static Value ReadFromBuffer(const void* data, TreeliteValueType type) {
    switch (type) {
      case TreeliteValueType::kTreeliteInt32:
        return Value(*static_cast<const int32_t*>(data));
      case TreeliteValueType::kTreeliteFloat32:
        return Value(*static_cast<const float*>(data));
      case TreeliteValueType::kTreeliteFloat64:
        return Value(*static_cast<const double*>(data));
      default:
        LOG(FATAL) << "Unrecognized value type: " << static_cast<int>(type);
        return Value();
    }
  }

  static std::vector<Value> ReadArrayFromBuffer(const void* data, TreeliteValueType type,
                                                size_t len) {
    std::vector<Value> vec;
    switch (type) {
      case TreeliteValueType::kTreeliteInt32: {
        const int32_t* ptr = static_cast<const int32_t*>(data);
        for (size_t i = 0; i < len; ++i) {
          vec.push_back(Value(ptr[i]));
        }
        break;
      }
      case TreeliteValueType::kTreeliteFloat32: {
        const float* ptr = static_cast<const float*>(data);
        for (size_t i = 0; i < len; ++i) {
          vec.push_back(Value(ptr[i]));
        }
        break;
      }
      case TreeliteValueType::kTreeliteFloat64: {
        const double* ptr = static_cast<const double*>(data);
        for (size_t i = 0; i < len; ++i) {
          vec.push_back(Value(ptr[i]));
        }
        break;
      }
      default:
        LOG(FATAL) << "Unrecognized value type: " << static_cast<int>(type);
    }
    return vec;
  }

  const ValueImpl& GetValue() const& { return *ptr_; }
  const ValueImpl& GetValue() &&     { return *ptr_; }
  ValueImpl&       GetValue() &      { return *ptr_; }

  inline std::string ToString() const { return ptr_->ToString(); }
  inline bool IsFinite() const { return ptr_->IsFinite(); }

  template <typename T>
  inline Value Clone(T new_value) const {
    Value retval;
    retval.ptr_ = ptr_->Clone(static_cast<const void*>(&new_value));
    return retval;
  }

 private:
  std::shared_ptr<ValueImpl> ptr_;
};

template <typename T>
inline bool IsA(const Value x) {
  const auto& v = x.GetValue();
  return IsA<T>(&v);
}

inline bool IsFloat(const Value x) {
  return IsA<Float32Value>(x) || IsA<Float64Value>(x);
}

inline bool IsIntegral(const Value x) {
  return IsA<Int32Value>(x);
}

namespace detail {
template <typename T,
  typename std::enable_if<std::is_same<T, Int32Value>::value>::type* = nullptr>
int32_t& GetImpl(T& val) {
  return val.GetValue();
}

template <typename T,
  typename std::enable_if<std::is_same<T, Float32Value>::value>::type* = nullptr>
float& GetImpl(T& val) {
  return val.GetValue();
}

template <typename T,
  typename std::enable_if<std::is_same<T, Float64Value>::value>::type* = nullptr>
double& GetImpl(T& val) {
  return val.GetValue();
}

template <typename T,
  typename std::enable_if<std::is_same<T, const Int32Value>::value>::type* = nullptr>
const int32_t& GetImpl(T& val) {
  return val.GetValue();
}

template <typename T,
  typename std::enable_if<std::is_same<T, const Float32Value>::value>::type* = nullptr>
const float& GetImpl(T& val) {
  return val.GetValue();
}

template <typename T,
  typename std::enable_if<std::is_same<T, const Float64Value>::value>::type* = nullptr>
const double& GetImpl(T& val) {
  return val.GetValue();
}

}  // namespace detail

template <typename T, typename U>
auto get(U& adt_value) -> decltype(detail::GetImpl(*Cast<T>(&adt_value.GetValue())))& { // NOLINT
  auto& value = *Cast<T>(&adt_value.GetValue());
  return detail::GetImpl(value);
}

}  // namespace ADT
}  // namespace treelite

#endif  // TREELITE_ADT_VALUE_TYPE_H
