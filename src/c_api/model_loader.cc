/*!
 * Copyright (c) 2023 by Contributors
 * \file model_loader.cc
 * \author Hyunsu Cho
 * \brief C API for frontend functions
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <treelite/c_api.h>
#include <treelite/c_api_error.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

#include "./c_api_utils.h"

int TreeliteLoadXGBoostModelLegacyBinary(
    char const* filename, [[maybe_unused]] char const* config_json, TreeliteModelHandle* out) {
  // config_json is unused for now
  API_BEGIN();
  std::unique_ptr<treelite::Model> model
      = treelite::model_loader::LoadXGBoostModelLegacyBinary(filename);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModelLegacyBinaryFromMemoryBuffer(void const* buf, std::uint64_t len,
    [[maybe_unused]] char const* config_json, TreeliteModelHandle* out) {
  // config_json is unused for now
  API_BEGIN();
  std::unique_ptr<treelite::Model> model
      = treelite::model_loader::LoadXGBoostModelLegacyBinary(buf, len);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModel(
    char const* filename, char const* config_json, TreeliteModelHandle* out) {
  TREELITE_LOG(WARNING) << "TreeliteLoadXGBoostModel() is deprecated. Please use "
                        << "TreeliteLoadXGBoostModelJSON() instead.";
  return TreeliteLoadXGBoostModelJSON(filename, config_json, out);
}

int TreeliteLoadXGBoostModelFromString(
    char const* json_str, std::size_t length, char const* config_json, TreeliteModelHandle* out) {
  TREELITE_LOG(WARNING) << "TreeliteLoadXGBoostModelFromString() is deprecated. Please use "
                        << "TreeliteLoadXGBoostModelFromJSONString() instead.";
  return TreeliteLoadXGBoostModelFromJSONString(json_str, length, config_json, out);
}

int TreeliteLoadXGBoostModelJSON(
    char const* filename, char const* config_json, TreeliteModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<treelite::Model> model
      = treelite::model_loader::LoadXGBoostModelJSON(filename, config_json);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModelFromJSONString(
    char const* json_str, std::size_t length, char const* config_json, TreeliteModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<treelite::Model> model = treelite::model_loader::LoadXGBoostModelFromJSONString(
      std::string_view{json_str, length}, config_json);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModelUBJSON(
    char const* filename, char const* config_json, TreeliteModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<treelite::Model> model
      = treelite::model_loader::LoadXGBoostModelUBJSON(filename, config_json);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModelFromUBJSONString(std::uint8_t const* ubjson_str, std::size_t length,
    char const* config_json, TreeliteModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<treelite::Model> model = treelite::model_loader::LoadXGBoostModelFromUBJSONString(
      std::basic_string_view<std::uint8_t>{ubjson_str, length}, config_json);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteDetectXGBoostFormat(char const* filename, char const** out_str) {
  API_BEGIN();
  std::string& ret_str = treelite::c_api::ReturnValueStore::Get()->ret_str;
  ret_str = treelite::model_loader::DetectXGBoostFormat(filename);
  *out_str = ret_str.c_str();
  API_END();
}

int TreeliteLoadLightGBMModel(
    char const* filename, [[maybe_unused]] char const* config_json, TreeliteModelHandle* out) {
  // config_json is unused for now
  API_BEGIN();
  std::unique_ptr<treelite::Model> model = treelite::model_loader::LoadLightGBMModel(filename);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

TREELITE_DLL int TreeliteLoadLightGBMModelFromString(
    char const* model_str, [[maybe_unused]] char const* config_json, TreeliteModelHandle* out) {
  // config_json is unused for now
  API_BEGIN();
  std::unique_ptr<treelite::Model> model
      = treelite::model_loader::LoadLightGBMModelFromString(model_str);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}
