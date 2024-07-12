/*!
 * Copyright (c) 2024 by Contributors
 * \file sax_adapters.h
 * \brief Adapters to connect RapidJSON and nlohmann/json with the delegated handler
 * \author Hyunsu Cho
 */

#ifndef SRC_MODEL_LOADER_DETAIL_XGBOOST_JSON_SAX_ADAPTERS_H_
#define SRC_MODEL_LOADER_DETAIL_XGBOOST_JSON_SAX_ADAPTERS_H_

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace treelite::model_loader::detail::xgboost {

class DelegatedHandler;

/*!
 * \brief Adapter for SAX parser from RapidJSON
 */
class RapidJSONAdapter {
 public:
  explicit RapidJSONAdapter(std::shared_ptr<DelegatedHandler> handler)
      : handler_{std::move(handler)} {}
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

/*!
 * \brief Adapter for SAX parser from nlohmann/json
 */
class NlohmannJSONAdapter {
 public:
  explicit NlohmannJSONAdapter(std::shared_ptr<DelegatedHandler> handler)
      : handler_{std::move(handler)} {}
  bool null();
  bool boolean(bool val);
  bool number_integer(std::int64_t val);
  bool number_unsigned(std::uint64_t val);
  bool number_float(double val, std::string const&);
  bool string(std::string& val);
  bool binary(nlohmann::json::binary_t& val);
  bool start_object(std::size_t);
  bool end_object();
  bool start_array(std::size_t);
  bool end_array();
  bool key(std::string& val);
  bool parse_error(
      std::size_t position, std::string const& last_token, nlohmann::json::exception const& ex);

 private:
  std::shared_ptr<DelegatedHandler> handler_;
};

}  // namespace treelite::model_loader::detail::xgboost

#endif  // SRC_MODEL_LOADER_DETAIL_XGBOOST_JSON_SAX_ADAPTERS_H_
