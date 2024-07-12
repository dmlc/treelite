/*!
 * Copyright (c) 2024 by Contributors
 * \file xgboost_ubjson.cc
 * \brief Model loader for XGBoost model (UBJSON)
 * \author Hyunsu Cho
 */

#include <fstream>
#include <memory>
#include <string>
#include <string_view>

#include <treelite/detail/file_utils.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

#include <nlohmann/json.hpp>

#include "detail/xgboost_json/delegated_handler.h"
#include "detail/xgboost_json/sax_adapters.h"

namespace {

template <typename InputType>
std::unique_ptr<treelite::Model> ParseStream(
    InputType&& input_stream, nlohmann::json const& parsed_config);

}  // anonymous namespace

namespace treelite::model_loader {

std::unique_ptr<treelite::Model> LoadXGBoostModelUBJSON(
    std::string const& filename, std::string const& config_json) {
  nlohmann::json parsed_config = nlohmann::json::parse(config_json);
  std::ifstream ifs = treelite::detail::OpenFileForReadAsStream(filename);
  return ParseStream(ifs, parsed_config);
}

std::unique_ptr<treelite::Model> LoadXGBoostModelFromUBJSONString(
    std::basic_string_view<std::uint8_t> ubjson_str, std::string const& config_json) {
  nlohmann::json parsed_config = nlohmann::json::parse(config_json);
  return ParseStream(ubjson_str, parsed_config);
}

}  // namespace treelite::model_loader

namespace {

template <typename InputType>
std::unique_ptr<treelite::Model> ParseStream(
    InputType&& input_stream, nlohmann::json const& parsed_config) {
  treelite::model_loader::detail::xgboost::HandlerConfig handler_config;
  if (parsed_config.is_object()) {
    auto itr = parsed_config.find("allow_unknown_field");
    if (itr != parsed_config.end() && itr->is_boolean()) {
      handler_config.allow_unknown_field = itr->template get<bool>();
    }
  }

  std::shared_ptr<treelite::model_loader::detail::xgboost::DelegatedHandler> handler
      = treelite::model_loader::detail::xgboost::DelegatedHandler::create(handler_config);
  auto adapter
      = std::make_unique<treelite::model_loader::detail::xgboost::NlohmannJSONAdapter>(handler);
  TREELITE_CHECK(nlohmann::json::sax_parse(
      input_stream, adapter.get(), nlohmann::json::input_format_t::ubjson));

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
