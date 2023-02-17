/*!
 * Copyright (c) 2023 by Contributors
 * \file config.cc
 * \author Hyunsu Cho
 * \brief Configuration handling logic for General Tree Inference Library (GTIL)
 */
#include <treelite/gtil.h>
#include <rapidjson/document.h>

namespace treelite {
namespace gtil {

GTILConfig::GTILConfig(const char* config_json) {
  rapidjson::Document parsed_config;
  parsed_config.Parse(config_json);

  if (parsed_config.IsObject()) {
    auto itr = parsed_config.FindMember("pred_transform");
    if (itr != parsed_config.MemberEnd() && itr->value.IsBool()) {
      this->pred_transform = itr->value.GetBool();
    }
    itr = parsed_config.FindMember("nthread");
    if (itr != parsed_config.MemberEnd() && itr->value.IsInt()) {
      this->nthread = itr->value.GetInt();
    }
  }
}

}  // namespace gtil
}  // namespace treelite
