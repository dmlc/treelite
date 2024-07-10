/*!
 * Copyright (c) 2024 by Contributors
 * \file sax_adapters.cc
 * \brief Adapters to connect RapidJSON and nlohmann/json with the delegated handler
 * \author Hyunsu Cho
 */

#include "./sax_adapters.h"

#include "./delegated_handler.h"

namespace treelite::model_loader::detail::xgboost {

RapidJSONAdapter::RapidJSONAdapter(std::shared_ptr<DelegatedHandler> handler)
    : handler_{std::move(handler)} {}

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

bool RapidJSONAdapter::String(char const* str, std::size_t length, bool copy) {
  return handler_->String(str, length, copy);
}

bool RapidJSONAdapter::StartObject() {
  return handler_->StartObject();
}

bool RapidJSONAdapter::Key(char const* str, std::size_t length, bool copy) {
  return handler_->Key(str, length, copy);
}

bool RapidJSONAdapter::EndObject(std::size_t memberCount) {
  return handler_->EndObject(memberCount);
}

bool RapidJSONAdapter::StartArray() {
  return handler_->StartArray();
}

bool RapidJSONAdapter::EndArray(std::size_t elementCount) {
  return handler_->EndArray(elementCount);
}

}  // namespace treelite::model_loader::detail::xgboost
