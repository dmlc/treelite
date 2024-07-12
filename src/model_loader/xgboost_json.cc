/*!
 * Copyright (c) 2020-2024 by Contributors
 * \file xgboost_json.cc
 * \brief Model loader for XGBoost model (JSON)
 * \author Hyunsu Cho, William Hicks
 */

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include <treelite/detail/file_utils.h>
#include <treelite/logging.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>

#include "detail/xgboost_json/delegated_handler.h"
#include "detail/xgboost_json/sax_adapters.h"

namespace {

template <typename StreamType, typename ErrorHandlerFunc>
std::unique_ptr<treelite::Model> ParseStream(std::unique_ptr<StreamType> input_stream,
    ErrorHandlerFunc error_handler, rapidjson::Document const& config);

}  // anonymous namespace

namespace treelite::model_loader {

std::unique_ptr<treelite::Model> LoadXGBoostModelJSON(
    std::string const& filename, std::string const& config_json) {
  char read_buffer[65536];

  FILE* fp = treelite::detail::OpenFileForReadAsFilePtr(filename);

  auto input_stream
      = std::make_unique<rapidjson::FileReadStream>(fp, read_buffer, sizeof(read_buffer));
  auto error_handler = [fp](std::size_t offset) -> std::string {
    std::size_t cur = (offset >= 50 ? (offset - 50) : 0);
    std::fseek(fp, cur, SEEK_SET);
    int c;
    std::ostringstream oss, oss2;
    for (int i = 0; i < 100; ++i) {
      c = std::fgetc(fp);
      if (c == EOF) {
        break;
      }
      oss << static_cast<char>(c);
      if (cur == offset) {
        oss2 << "^";
      } else {
        oss2 << "~";
      }
      ++cur;
    }
    std::fclose(fp);
    return oss.str() + "\n" + oss2.str();
  };
  rapidjson::Document parsed_config;
  parsed_config.Parse(config_json);
  TREELITE_CHECK(!parsed_config.HasParseError())
      << "Error when parsing JSON config: offset " << parsed_config.GetErrorOffset() << ", "
      << rapidjson::GetParseError_En(parsed_config.GetParseError());
  auto parsed_model = ParseStream(std::move(input_stream), error_handler, parsed_config);
  std::fclose(fp);
  return parsed_model;
}

std::unique_ptr<treelite::Model> LoadXGBoostModelFromJSONString(
    std::string_view json_str, std::string const& config_json) {
  auto input_stream = std::make_unique<rapidjson::MemoryStream>(json_str.data(), json_str.length());
  auto error_handler = [json_str](std::size_t offset) -> std::string {
    std::size_t cur = (offset >= 50 ? (offset - 50) : 0);
    std::ostringstream oss, oss2;
    for (int i = 0; i < 100; ++i) {
      if (!json_str[cur]) {
        break;
      }
      oss << json_str[cur];
      if (cur == offset) {
        oss2 << "^";
      } else {
        oss2 << "~";
      }
      ++cur;
    }
    return oss.str() + "\n" + oss2.str();
  };
  rapidjson::Document parsed_config;
  parsed_config.Parse(config_json);
  TREELITE_CHECK(!parsed_config.HasParseError())
      << "Error when parsing JSON config: offset " << parsed_config.GetErrorOffset() << ", "
      << rapidjson::GetParseError_En(parsed_config.GetParseError());
  return ParseStream(std::move(input_stream), error_handler, parsed_config);
}

}  // namespace treelite::model_loader

namespace {

template <typename StreamType, typename ErrorHandlerFunc>
std::unique_ptr<treelite::Model> ParseStream(std::unique_ptr<StreamType> input_stream,
    ErrorHandlerFunc error_handler, rapidjson::Document const& parsed_config) {
  treelite::model_loader::detail::xgboost::HandlerConfig handler_config;
  if (parsed_config.IsObject()) {
    auto itr = parsed_config.FindMember("allow_unknown_field");
    if (itr != parsed_config.MemberEnd() && itr->value.IsBool()) {
      handler_config.allow_unknown_field = itr->value.GetBool();
    }
  }

  std::shared_ptr<treelite::model_loader::detail::xgboost::DelegatedHandler> handler
      = treelite::model_loader::detail::xgboost::DelegatedHandler::create(handler_config);
  auto adapter
      = std::make_unique<treelite::model_loader::detail::xgboost::RapidJSONAdapter>(handler);
  rapidjson::Reader reader;

  rapidjson::ParseResult result
      = reader.Parse<rapidjson::ParseFlag::kParseNanAndInfFlag>(*input_stream, *adapter);
  if (!result) {
    auto const error_code = result.Code();
    std::size_t const offset = result.Offset();
    std::string diagnostic = error_handler(offset);
    TREELITE_LOG(FATAL) << "Provided JSON could not be parsed as XGBoost model. "
                        << "Parsing error at offset " << offset << ": "
                        << rapidjson::GetParseError_En(error_code) << "\n"
                        << diagnostic;
  }
  treelite::model_loader::detail::xgboost::ParsedXGBoostModel parsed = handler->get_result();
  auto model = parsed.builder->CommitModel();

  // Apply Dart weights
  if (!parsed.weight_drop.empty()) {
    auto& trees = std::get<treelite::ModelPreset<float, float>>(model->variant_).trees;
    TREELITE_CHECK_EQ(trees.size(), parsed.weight_drop.size());
    for (std::size_t i = 0; i < trees.size(); ++i) {
      for (int nid = 0; nid < trees[i].num_nodes; ++nid) {
        if (trees[i].IsLeaf(nid)) {
          trees[i].SetLeaf(
              nid, static_cast<float>(parsed.weight_drop[i] * trees[i].LeafValue(nid)));
        }
      }
    }
  }
  return model;
}

}  // anonymous namespace
