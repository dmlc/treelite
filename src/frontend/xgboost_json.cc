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

inline treelite::Model ParseStream(dmlc::Stream* fi);
inline std::string ReadWholeStream(dmlc::Stream* fi);

}  // anonymous namespace

namespace treelite {
namespace frontend {

DMLC_REGISTRY_FILE_TAG(xgboost_json);

void LoadXGBoostJSONModel(const char* filename, Model* out) {
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(filename, "r"));
  *out = std::move(ParseStream(fi.get()));
}

void LoadXGBoostJSONModel(std::string json_str, Model* out) {
  dmlc::MemoryStringStream fs(&json_str);
  *out = std::move(ParseStream(&fs));
}

}  // namespace frontend
}  // namespace treelite

namespace {

inline treelite::Model ParseStream(dmlc::Stream* fi) {
  std::string json_str = ReadWholeStream(fi);
  LOG(INFO) << json_str;

  treelite::Model model;
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
