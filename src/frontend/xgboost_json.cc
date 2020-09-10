/*!
 * Copyright (c) 2020 by Contributors
 * \file xgboost_json.cc
 * \brief Frontend for xgboost model
 * \author Hyunsu Cho
 */

#include <dmlc/registry.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <treelite/tree.h>
#include <treelite/frontend.h>
#include <rapidjson/document.h>

namespace {

inline std::unique_ptr<treelite::Model> ParseStream(dmlc::Stream* fi);
inline std::string ReadWholeStream(dmlc::Stream* fi);

}  // anonymous namespace

namespace treelite {
namespace frontend {

DMLC_REGISTRY_FILE_TAG(xgboost_json);

std::unique_ptr<treelite::Model> LoadXGBoostJSONModel(const char* filename) {
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(filename, "r"));
  return ParseStream(fi.get());
}

std::unique_ptr<treelite::Model> LoadXGBoostJSONModel(std::string json_str) {
  dmlc::MemoryStringStream fs(&json_str);
  return ParseStream(&fs);
}

}  // namespace frontend
}  // namespace treelite

namespace {

inline std::unique_ptr<treelite::Model> ParseStream(dmlc::Stream* fi) {
  std::string json_str = ReadWholeStream(fi);
  LOG(INFO) << json_str;

  std::unique_ptr<treelite::Model> model = treelite::Model::Create<float, float>();
  return model;
}

inline std::string ReadWholeStream(dmlc::Stream* fi) {
  std::string str;
  size_t total_read = 0;
  size_t size = 4096;

  while (true) {
    str.resize(total_read + size);
    size_t read = fi->Read(&str[total_read], size);
    total_read += read;
    if (read < size) {
      break;
    }
    size *= 2;
  }
  str.resize(total_read);

  return str;
}

}  // anonymous namespace
