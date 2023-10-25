/*!
 * Copyright (c) 2023 by Contributors
 * \file serializer.cc
 * \author Hyunsu Cho
 * \brief C API for functions to serialize model objects
 */

#include <treelite/c_api.h>
#include <treelite/c_api_error.h>
#include <treelite/detail/file_utils.h>
#include <treelite/tree.h>

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>

#include "./c_api_utils.h"

int TreeliteSerializeModelToFile(TreeliteModelHandle handle, char const* filename) {
  API_BEGIN();
  std::ofstream ofs = treelite::detail::OpenFileForWriteAsStream(filename);
  auto* model_ = static_cast<treelite::Model*>(handle);
  model_->SerializeToStream(ofs);
  API_END();
}

int TreeliteDeserializeModelFromFile(char const* filename, TreeliteModelHandle* out) {
  API_BEGIN();
  std::ifstream ifs = treelite::detail::OpenFileForReadAsStream(filename);
  std::unique_ptr<treelite::Model> model = treelite::Model::DeserializeFromStream(ifs);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}

int TreeliteSerializeModelToBytes(
    TreeliteModelHandle handle, char const** out_bytes, std::size_t* out_bytes_len) {
  API_BEGIN();
  std::ostringstream oss;
  oss.exceptions(std::ios::failbit | std::ios::badbit);  // Throw exception on failure
  auto* model_ = static_cast<treelite::Model*>(handle);
  model_->SerializeToStream(oss);

  std::string& ret_str = treelite::c_api::ReturnValueStore::Get()->ret_str;
  ret_str = oss.str();
  *out_bytes = ret_str.data();
  *out_bytes_len = ret_str.length();
  API_END();
}

int TreeliteDeserializeModelFromBytes(
    char const* bytes, std::size_t bytes_len, TreeliteModelHandle* out) {
  API_BEGIN();
  std::istringstream iss(std::string(bytes, bytes_len));
  iss.exceptions(std::ios::failbit | std::ios::badbit);  // Throw exception on failure
  std::unique_ptr<treelite::Model> model = treelite::Model::DeserializeFromStream(iss);
  *out = static_cast<TreeliteModelHandle>(model.release());
  API_END();
}
