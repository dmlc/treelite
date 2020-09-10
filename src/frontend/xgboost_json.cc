/*!
 * Copyright (c) 2020 by Contributors
 * \file xgboost_json.cc
 * \brief Frontend for xgboost model
 * \author Hyunsu Cho
 */

#include <cstdio>
#include <iostream>
#include <memory>
#include <stack>
#include <string_view>

#include <dmlc/registry.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <treelite/tree.h>
#include <treelite/frontend.h>
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

namespace {

template<typename StreamType> treelite::Model ParseStream(
    std::unique_ptr<StreamType> input_stream
);

}  // anonymous namespace

namespace treelite {
namespace frontend {

DMLC_REGISTRY_FILE_TAG(xgboost_json);

void LoadXGBoostJSONModel(const char* filename, Model* out) {
  char readBuffer[65536];

#ifdef _WIN32
  FILE * fp = fopen(filename, "rb");
#else
  FILE * fp = fopen(filename, "r");
#endif

  auto input_stream = std::make_unique<rapidjson::FileReadStream> (
      fp,
      readBuffer,
      sizeof(readBuffer)
  );
  *out = std::move(ParseStream(std::move(input_stream)));
}

/* void LoadXGBoostJSONModelString(std::string json_str, Model* out) {
  dmlc::MemoryStringStream fs(&json_str);
  *out = std::move(ParseStream(&fs));
} */

}  // namespace frontend
}  // namespace treelite

namespace {

class BaseHandler;

class DelegatedHandler {
public:
  void pop_delegate();
  void push_delegate(std::shared_ptr<BaseHandler> new_delegate);
};

class BaseHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, BaseHandler> {
public:
  BaseHandler(){};
  virtual bool Null() { return false; }
  virtual bool Bool(bool b) { return false; }
  virtual bool Int(int i) { return false; }
  virtual bool Uint(unsigned u) { return false; }
  virtual bool Int64(int64_t i) { return false; }
  virtual bool Uint64(uint64_t u) { return false; }
  virtual bool Double(double d) { return false; }
  virtual bool String(const char *str, std::size_t length, bool copy) {
    return false;
  }
  virtual bool StartObject() { return false; }
  virtual bool Key(const char *str, std::size_t length, bool copy) {
    return false;
  }
  virtual bool EndObject() {
    if (auto parent = get_delegator()) {
      parent->pop_delegate();
      return true;
    } else {
      return false;
    }
  }
  virtual bool StartArray() { return false; }
  bool EndArray(std::size_t elementCount) { return false; }

protected:
  std::shared_ptr<DelegatedHandler> get_delegator() { return delegator.lock(); }

private:
  std::weak_ptr<DelegatedHandler> delegator;
};

class XGBoostModelHandler : public BaseHandler {
public:
  bool Key(const char *str, std::size_t length, bool copy) {
    cur_key = std::string_view(str, length);
    return false;
  }

private:
  std::string_view cur_key = "";
};

class RootHandler : public BaseHandler {
  virtual bool StartObject() {
    if (auto parent = get_delegator()) {
      parent->push_delegate();
      return true;
    } else {
      return false;
    }
  }
};

class DelegatedHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, DelegatedHandler> {

public:
  DelegatedHandler() : delegates({std::make_shared<RootHandler>()}) {}

  void push_delegate(std::shared_ptr<BaseHandler> new_delegate) {
    delegates.push(new_delegate);
  };
  void pop_delegate() { delegates.pop(); };
  bool Null() { return delegates.top()->Null(); }
  bool Bool(bool b) { return delegates.top()->Bool(b); }
  bool Int(int i) { return delegates.top()->Int(i); }
  bool Uint(unsigned u) { return delegates.top()->Uint(u); }
  bool Int64(int64_t i) { return delegates.top()->Int64(i); }
  bool Uint64(uint64_t u) { return delegates.top()->Uint64(u); }
  bool Double(double d) { return delegates.top()->Double(d); }
  bool String(const char *str, std::size_t length, bool copy) {
    return delegates.top()->String(str, length, copy);
  }
  bool StartObject() { return delegates.top()->StartObject(); }
  bool Key(const char *str, std::size_t length, bool copy) {
    return delegates.top()->Key(str, length, copy);
  }
  bool EndObject(std::size_t memberCount) {
    return delegates.top()->EndObject(memberCount);
  }
  bool StartArray() { return delegates.top()->StartArray(); }
  bool EndArray(std::size_t elementCount) {
    return delegates.top()->StartArray();
  }

private:
  std::stack<std::shared_ptr<BaseHandler>> delegates;
};

template<typename StreamType>
treelite::Model ParseStream(std::unique_ptr<StreamType> input_stream) {

  DelegatedHandler handler;
  rapidjson::Reader reader;

  reader.Parse(*input_stream, handler);

  treelite::Model model;
  return model;
}


}  // anonymous namespace
