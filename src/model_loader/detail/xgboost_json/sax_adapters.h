/*!
 * Copyright (c) 2024 by Contributors
 * \file sax_adapters.h
 * \brief Adapters to connect RapidJSON and nlohmann/json with the delegated handler
 * \author Hyunsu Cho
 */

#ifndef SRC_MODEL_LOADER_DETAIL_XGBOOST_JSON_SAX_ADAPTERS_H_
#define SRC_MODEL_LOADER_DETAIL_XGBOOST_JSON_SAX_ADAPTERS_H_

#include <rapidjson/reader.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace treelite::model_loader::detail::xgboost {

class DelegatedHandler;

class RapidJSONAdapter {
 public:
  explicit RapidJSONAdapter(std::shared_ptr<DelegatedHandler> handler);
  bool Null();
  bool Bool(bool b);
  bool Int(int i);
  bool Uint(unsigned u);
  bool Int64(std::int64_t i);
  bool Uint64(std::uint64_t u);
  bool Double(double d);
  bool RawNumber(char const* str, std::size_t length, bool copy);
  bool String(char const* str, std::size_t length, bool copy);
  bool StartObject();
  bool Key(char const* str, std::size_t length, bool copy);
  bool EndObject(std::size_t);
  bool StartArray();
  bool EndArray(std::size_t);

 private:
  std::shared_ptr<DelegatedHandler> handler_;
};

class NlohmannUBJSONAdapter {};

}  // namespace treelite::model_loader::detail::xgboost

#endif  // SRC_MODEL_LOADER_DETAIL_XGBOOST_JSON_SAX_ADAPTERS_H_
