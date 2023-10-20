/*!
 * Copyright (c) 2023 by Contributors
 * \file model_loader.cc
 * \author Hyunsu Cho
 * \brief C API for frontend functions
 */

#include <treelite/c_api.h>
#include <treelite/c_api_error.h>
#include <treelite/model_loader.h>
#include <treelite/tree.h>

#include <cstddef>
#include <cstdint>

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
  API_BEGIN();
  std::unique_ptr<treelite::Model> model
      = treelite::model_loader::LoadXGBoostModel(filename, config_json);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteLoadXGBoostModelFromString(
    char const* json_str, std::uint64_t length, char const* config_json, TreeliteModelHandle* out) {
  API_BEGIN();
  std::unique_ptr<treelite::Model> model
      = treelite::model_loader::LoadXGBoostModelFromString(json_str, length, config_json);
  *out = static_cast<TreeliteModelHandle>(model.release());
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
