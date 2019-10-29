/*!
 * Copyright 2019 by Contributors
 * \file adt_value_type.cc
 * \brief Define an ADT (Abstract Data Type) for Treelite. We use type erasure so that we can
 *        accommodate multiple data types for thresholds and leaf values without littering the
 *        whole codebase with templates.
 * \author Philip Cho
 */

/*
 * RTTI (Run-Time Type Information), together with type erasure, is inspired by
 * https://github.com/dmlc/xgboost/blob/d2e1e4d/include/xgboost/json.h.
 */

#include <treelite/adt_value_type.h>
#include <fmt/format.h>
#include <limits>
#include <unordered_map>

namespace treelite {
namespace ADT {

std::string ValueImpl::TypeStr() const {
  switch (kind_) {
    case ValueKind::kInt32:   return "Int32";  break;
    case ValueKind::kFloat32: return "Float32";  break;
    case ValueKind::kFloat64: return "Float64";  break;
  }
  return "";
}

bool Int32Value::operator==(const ValueImpl& rhs) const {
  if (!IsA<Int32Value>(&rhs)) {
    return false;
  }
  return val_ == Cast<const Int32Value>(&rhs)->GetValue();
}

bool Int32Value::operator<(const ValueImpl& rhs) const {
  if (!IsA<Int32Value>(&rhs)) {
    LOG(FATAL) << "Invalid comparison between " + this->TypeStr() + " and " + rhs.TypeStr();
  }
  return val_ < Cast<const Int32Value>(&rhs)->GetValue();
}

Int32Value& Int32Value::operator=(const ValueImpl& rhs) {
  const Int32Value* casted = Cast<const Int32Value>(&rhs);
  val_ = casted->GetValue();
  return *this;
}

std::string Int32Value::ToString() const {
  return fmt::format("{:d}", val_);
}

bool Int32Value::IsFinite() const {
  return std::isfinite(val_);
}

std::shared_ptr<ValueImpl> Int32Value::Clone(const void* data) {
  int32_t val = *static_cast<const int32_t*>(data);
  return std::make_shared<Int32Value>(val);
}

bool Float32Value::operator==(const ValueImpl& rhs) const {
  if (!IsA<Float32Value>(&rhs)) {
    return false;
  }
  return val_ == Cast<const Float32Value>(&rhs)->GetValue();
}

bool Float32Value::operator<(const ValueImpl& rhs) const {
  if (!IsA<Float32Value>(&rhs)) {
    LOG(FATAL) << "Invalid comparison between " + this->TypeStr() + " and " + rhs.TypeStr();
  }
  return val_ < Cast<const Float32Value>(&rhs)->GetValue();
}

Float32Value& Float32Value::operator=(const ValueImpl& rhs) {
  const Float32Value* casted = Cast<const Float32Value>(&rhs);
  val_ = casted->GetValue();
  return *this;
}

std::string Float32Value::ToString() const {
  return fmt::format("{:.{}g}", val_, std::numeric_limits<decltype(val_)>::digits10 + 2);
}

bool Float32Value::IsFinite() const {
  return std::isfinite(val_);
}

std::shared_ptr<ValueImpl> Float32Value::Clone(const void* data) {
  float val = *static_cast<const float*>(data);
  return std::make_shared<Float32Value>(val);
}

bool Float64Value::operator==(const ValueImpl& rhs) const {
  if (!IsA<Float64Value>(&rhs)) {
    return false;
  }
  return val_ == Cast<const Float64Value>(&rhs)->GetValue();
}

bool Float64Value::operator<(const ValueImpl& rhs) const {
  if (!IsA<Float64Value>(&rhs)) {
    LOG(FATAL) << "Invalid comparison between " + this->TypeStr() + " and " + rhs.TypeStr();
  }
  return val_ < Cast<const Float64Value>(&rhs)->GetValue();
}

Float64Value& Float64Value::operator=(const ValueImpl& rhs) {
  const Float64Value* casted = Cast<const Float64Value>(&rhs);
  val_ = casted->GetValue();
  return *this;
}

std::string Float64Value::ToString() const {
  return fmt::format("{:.{}g}", val_, std::numeric_limits<decltype(val_)>::digits10 + 2);
}

bool Float64Value::IsFinite() const {
  return std::isfinite(val_);
}

std::shared_ptr<ValueImpl> Float64Value::Clone(const void* data) {
  double val = *static_cast<const double*>(data);
  return std::make_shared<Float64Value>(val);
}

const std::unordered_map<std::string, TreeliteValueType> ValueTypeNameTable{
  {"Int32", kTreeliteInt32},
  {"Float32", kTreeliteFloat32},
  {"Float64", kTreeliteFloat64}
};

}  // namespace ADT
}  // namespace treelite
