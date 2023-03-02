/*!
 * Copyright (c) 2023 by Contributors
 * \file config.cc
 * \author Hyunsu Cho
 * \brief Configuration handling logic for General Tree Inference Library (GTIL)
 */
#include <treelite/gtil.h>
#include <treelite/logging.h>
#include <rapidjson/document.h>
#include <string>

namespace treelite {
namespace gtil {

Configuration::Configuration(const char* config_json) {
  rapidjson::Document parsed_config;
  parsed_config.Parse(config_json);

  if (parsed_config.IsObject()) {
    auto itr = parsed_config.FindMember("predict_type");
    if (itr != parsed_config.MemberEnd() && itr->value.IsString()) {
      auto value = std::string(itr->value.GetString());
      if (value == "default") {
        this->pred_type = PredictType::kPredictDefault;
      } else if (value == "raw") {
        this->pred_type = PredictType::kPredictRaw;
      } else if (value == "leaf_id") {
        this->pred_type = PredictType::kPredictLeafID;
      } else if (value == "score_per_tree") {
        this->pred_type = PredictType::kPredictPerTree;
      } else {
        TREELITE_LOG(FATAL) << "Unknown prediction type: " << value;
      }
    } else {
      TREELITE_LOG(FATAL) << "The field \"predict_type\" must be specified";
    }
    itr = parsed_config.FindMember("nthread");
    if (itr != parsed_config.MemberEnd()) {
      TREELITE_CHECK(itr->value.IsInt()) << "nthread must be an integer";
      this->nthread = itr->value.GetInt();
    }
  } else {
    TREELITE_LOG(FATAL) << "The JSON string must be a valid JSON object";
  }
}

}  // namespace gtil
}  // namespace treelite
