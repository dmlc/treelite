/*!
 * Copyright (c) 2024 by Contributors
 * \file sax_adapters.cc
 * \brief Adapters to connect RapidJSON and nlohmann/json with the delegated handler
 * \author Hyunsu Cho
 */

#include "./sax_adapters.h"

#include <algorithm>
#include <string>

#include <nlohmann/json.hpp>

#include "./delegated_handler.h"

namespace treelite::model_loader::detail::xgboost {

/******************************************************************************
 * RapidJSONAdapter
 * ***************************************************************************/

bool RapidJSONAdapter::Null() {
  return handler_->Null();
}

bool RapidJSONAdapter::Bool(bool b) {
  return handler_->Bool(b);
}

bool RapidJSONAdapter::Int(int i) {
  return handler_->Int64(static_cast<std::int64_t>(i));
}

bool RapidJSONAdapter::Uint(unsigned int u) {
  return handler_->Uint64(static_cast<std::uint64_t>(u));
}

bool RapidJSONAdapter::Int64(std::int64_t i) {
  return handler_->Int64(i);
}

bool RapidJSONAdapter::Uint64(std::uint64_t u) {
  return handler_->Uint64(u);
}

bool RapidJSONAdapter::Double(double d) {
  return handler_->Double(d);
}

bool RapidJSONAdapter::RawNumber(char const* str, std::size_t length, bool copy) {
  TREELITE_LOG(FATAL) << "RawNumber() not implemented";
  return false;
}

bool RapidJSONAdapter::String(char const* str, std::size_t length, bool) {
  return handler_->String(std::string{str, length});
}

bool RapidJSONAdapter::StartObject() {
  return handler_->StartObject();
}

bool RapidJSONAdapter::Key(char const* str, std::size_t length, bool copy) {
  return handler_->Key(std::string{str, length});
}

bool RapidJSONAdapter::EndObject(std::size_t) {
  return handler_->EndObject();
}

bool RapidJSONAdapter::StartArray() {
  return handler_->StartArray();
}

bool RapidJSONAdapter::EndArray(std::size_t) {
  return handler_->EndArray();
}

/******************************************************************************
 * NlohmannJSONAdapter
 * ***************************************************************************/

bool NlohmannJSONAdapter::null() {
  return handler_->Null();
}

bool NlohmannJSONAdapter::boolean(bool val) {
  return handler_->Bool(val);
}

bool NlohmannJSONAdapter::number_integer(std::int64_t val) {
  return handler_->Int64(val);
}

bool NlohmannJSONAdapter::number_unsigned(std::uint64_t val) {
  return handler_->Uint64(val);
}

bool NlohmannJSONAdapter::number_float(double val, std::string const&) {
  return handler_->Double(val);
}

bool NlohmannJSONAdapter::string(std::string& val) {
  return handler_->String(val);
}

bool NlohmannJSONAdapter::binary(nlohmann::json::binary_t& val) {
  static_assert(sizeof(char) == sizeof(std::uint8_t), "char must be 1 byte");
  std::string s;
  s.resize(val.size());
  std::transform(std::begin(val), std::end(val), std::begin(s),
      [](std::uint8_t e) -> char { return static_cast<char>(e); });
  return handler_->String(s);
}

bool NlohmannJSONAdapter::start_object(std::size_t) {
  return handler_->StartObject();
}

bool NlohmannJSONAdapter::end_object() {
  return handler_->EndObject();
}

bool NlohmannJSONAdapter::start_array(std::size_t) {
  return handler_->StartArray();
}

bool NlohmannJSONAdapter::end_array() {
  return handler_->EndArray();
}

bool NlohmannJSONAdapter::key(std::string& val) {
  return handler_->Key(val);
}

bool NlohmannJSONAdapter::parse_error(
    std::size_t position, std::string const& last_token, nlohmann::json::exception const& ex) {
  TREELITE_LOG(ERROR) << "Parsing error at token " << position << ": " << ex.what();
  return false;
}

}  // namespace treelite::model_loader::detail::xgboost
