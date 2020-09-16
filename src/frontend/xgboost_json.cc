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
#include <string>
#include <utility>

#include <fmt/format.h>

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

class Delegator {
public:
  virtual void pop_delegate() = 0;
  virtual void push_delegate(std::shared_ptr<BaseHandler> new_delegate) = 0;
};

class BaseHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, BaseHandler> {
public:
  BaseHandler(std::weak_ptr<Delegator> parent_delegator)
      : delegator{parent_delegator} {};
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
    set_cur_key(str, length);
    return true;
  }
  virtual bool EndObject(std::size_t memberCount) { return false; }
  virtual bool StartArray() { return false; }
  bool EndArray(std::size_t elementCount) { return false; }

protected:
  std::shared_ptr<Delegator> get_delegator() { return delegator.lock(); }

  template <typename HandlerType, typename... ArgsTypes>
  bool push_handler(ArgsTypes &... args) {
    if (auto parent = get_delegator()) {
      parent->push_delegate(std::make_shared<HandlerType>(parent, args...));
      return true;
    } else {
      return false;
    }
  }

  template <typename HandlerType, typename... ArgsTypes>
  bool push_key_handler(std::string key, ArgsTypes &... args) {
    if (check_cur_key(key)) {
      push_handler<HandlerType, ArgsTypes...>(args...);
      return true;
    } else {
      return false;
    }
  }

  bool pop_handler() {
    if (auto parent = get_delegator()) {
      parent->pop_delegate();
      return true;
    } else {
      return false;
    }
  }

  void set_cur_key(const char *str, std::size_t length) {
    cur_key = fmt::string_view(str, length);
  }

  const fmt::string_view get_cur_key() { return cur_key; }

  bool check_cur_key(const std::string &query_key) {
    return cur_key.compare(
        fmt::string_view{query_key.c_str(), query_key.size()});
  }

  template <typename ValueType>
  bool assign_value(const std::string &key, ValueType &output,
                    ValueType &&value) {
    if (check_cur_key(key)) {
      output = value;
      return true;
    } else {
      return false;
    }
  }

  template <typename ValueType>
  bool assign_value(const std::string &key, ValueType &output,
                    ValueType &value) {
    return assign_value(key, output, std::forward<ValueType>(value));
  }

private:
  std::weak_ptr<Delegator> delegator;
  fmt::string_view cur_key;
};

template <typename OutputType> class OutputHandler : public BaseHandler {
public:
  OutputHandler(std::weak_ptr<Delegator> parent_delegator, OutputType &output)
      : BaseHandler{parent_delegator}, m_output{output} {};
  OutputHandler(std::weak_ptr<Delegator> parent_delegator,
                OutputType &&output) = delete;

protected:
  OutputType &m_output;
};

template <typename ElemType, typename HandlerType = BaseHandler>
class ArrayHandler : public OutputHandler<std::vector<ElemType>> {
public:
  bool Bool(ElemType b) {
    this->m_output.push_back(b);
    return true;
  }
  bool Int(ElemType i) {
    this->m_output.push_back(i);
    return true;
  }
  bool Uint(ElemType u) {
    this->m_output.push_back(u);
    return true;
  }
  bool Int64(ElemType i) {
    this->m_output.push_back(i);
    return true;
  }
  bool Uint64(ElemType u) {
    this->m_output.push_back(u);
    return true;
  }
  bool Double(ElemType d) {
    this->m_output.push_back(d);
    return true;
  }

  template <typename StringType = ElemType,
            typename std::enable_if<
                std::is_same<StringType, fmt::string_view>::value>::type>
  bool String(const char *str, std::size_t length, bool copy) {
    this->m_output.push_back(ElemType{str, length});
  }

  bool StartObject() {
    this->m_output.push_back(ElemType{});
    return this->template push_handler<HandlerType, ElemType>(
        this->m_output.back());
  }

  bool EndArray(std::size_t elementCount) { return this->pop_handler(); }
};

struct LearnerTrainParam {
  LearnerTrainParam() : n_gpus{0}, gpu_id{0} {};
  int seed;
  bool seed_per_iteration;
  fmt::string_view dsplit;
  fmt::string_view tree_method;
  bool disable_default_eval_metric;
  double base_score;
  unsigned num_feature;
  unsigned num_class;
  unsigned gpu_id;
  unsigned n_gpus;
};

class LearnerTrainParamHandler : public OutputHandler<LearnerTrainParam> {
  bool Bool(bool b) {
    return (this->assign_value("seed_per_iteration",
                               m_output.seed_per_iteration, b) ||
            this->assign_value("disable_default_eval_metric",
                               m_output.disable_default_eval_metric, b));
  }
  bool Int(int i) { return this->assign_value("seed", m_output.seed, i); }
  bool Uint(unsigned u) {
    return (this->assign_value("seed", m_output.seed, static_cast<int>(u)) ||
            this->assign_value("num_feature", m_output.num_feature, u) ||
            this->assign_value("num_class", m_output.num_class, u) ||
            this->assign_value("gpu_id", m_output.gpu_id, u) ||
            this->assign_value("n_gpus", m_output.n_gpus, u));
  }
  bool Double(double d) {
    return this->assign_value("base_score", m_output.base_score, d);
  }
  bool EndObject(std::size_t memberCount) { return this->pop_handler(); };
};

struct GBTreeTrainParam {
  std::vector<fmt::string_view> updater_seq;
  fmt::string_view process_type;
  fmt::string_view predictor;
};

class GBTreeTrainParamHandler : public OutputHandler<GBTreeTrainParam> {
  bool Uint(unsigned u) { return check_cur_key("num_parallel_tree"); }
  bool String(const char *str, std::size_t length, bool copy) {
    return (this->assign_value("process_type", m_output.process_type,
                               fmt::string_view{str, length}) ||
            this->assign_value("predictor", m_output.predictor,
                               fmt::string_view{str, length}));
  }
  bool StartArray() {
    return this->push_key_handler<ArrayHandler<fmt::string_view>,
                                  std::vector<fmt::string_view>>(
        "updater_seq", m_output.updater_seq);
  }
};

struct TreeTrainParam {};
class TreeTrainParamHandler : public OutputHandler<TreeTrainParam> {};
struct TreeUpdater {};
class TreeUpdaterHandler : public OutputHandler<TreeUpdater> {};

struct GBTreeModelParam {};
class GBTreeModelParamHandler : public OutputHandler<GBTreeModelParam> {};

struct TreeParam {};
class TreeParamHandler : public OutputHandler<TreeParam> {};

struct Node {};
class NodeHandler : public OutputHandler<Node> {};

struct NodeStat {};
class NodeStatHandler : public OutputHandler<NodeStat> {};

struct RegTree {
  TreeParam tree_param;
  std::vector<Node> nodes;
  std::vector<NodeStat> stats;
};
class RegTreeHandler : public OutputHandler<RegTree> {};

struct GBTreeModel {
  GBTreeModelParam model_param;
  std::vector<RegTree> trees;
  std::vector<int> tree_info;
};
class GBTreeModelHandler : public OutputHandler<GBTreeModel> {};

struct GradientBooster {
  GBTreeTrainParam gbtree_train_param;
  TreeTrainParam updater_train_param;
  std::vector<TreeUpdater> updaters;
  GBTreeModel model;
};
class GradientBoosterHandler : public OutputHandler<GradientBooster> {
public:
  bool Uint(unsigned u) { return check_cur_key("num_boosting_round"); }
  bool String(const char *str, std::size_t length, bool copy) {
    fmt::string_view name{str, length};
    if (!name.compare("GBTree")) {
      LOG(ERROR) << "Only GBTree-type boosters are currently supported.";
      return false;
    } else {
      return true;
    }
  }
  bool Key(const char *str, std::size_t length, bool copy) {
    set_cur_key(str, length);
    return true;
  }
  bool StartObject() {
    if (this->push_key_handler<GBTreeTrainParamHandler, GBTreeTrainParam>(
            "gbtree_train_param", m_output.gbtree_train_param) ||
        this->push_key_handler<TreeTrainParamHandler, TreeTrainParam>(
            "updater_train_param", m_output.updater_train_param) ||
        this->push_key_handler<GBTreeModelHandler, GBTreeModel>(
            "model", m_output.model)) {
      return true;
    } else {
      LOG(ERROR) << "Key \""
                 << std::string{get_cur_key().data(), get_cur_key().size()}
                 << "\" not recognized. Is this a GBTree-type booster?";
      return false;
    }
  }
  bool EndObject(std::size_t memberCount) { return this->pop_handler(); }
  bool StartArray() {
    return this->push_key_handler<ArrayHandler<TreeUpdater, TreeUpdaterHandler>,
                                  std::vector<TreeUpdater>>("updaters",
                                                            m_output.updaters);
  }
};

struct Objective {};
class ObjectiveHandler : public OutputHandler<Objective> {};

class LearnerHandler : public OutputHandler<treelite::Model> {
public:
  bool StartArray() {
    return this->push_key_handler<ArrayHandler<fmt::string_view>,
                                  std::vector<fmt::string_view>>("eval_metrics",
                                                                 eval_metrics);
  }

  bool StartObject() {
    return (this->push_key_handler<LearnerTrainParamHandler, LearnerTrainParam>(
                "learner_train_param", learner_train_param) ||
            this->push_key_handler<GradientBoosterHandler, GradientBooster>(
                "gradient_booster", gradient_booster) ||
            this->push_key_handler<ObjectiveHandler, Objective>("objective",
                                                                objective));
  }

  bool EndObject(std::size_t memberCount) {
    // TODO: eval_metrics
    return this->pop_handler();
  }

private:
  std::vector<fmt::string_view> eval_metrics;
  LearnerTrainParam learner_train_param;
  GradientBooster gradient_booster;
  Objective objective;
};

class XGBoostModelHandler : public OutputHandler<treelite::Model> {
public:
  bool StartArray() {
    return push_key_handler<ArrayHandler<unsigned>, std::vector<unsigned>>(
        "version", version);
  }

  bool StartObject() {
    return push_key_handler<LearnerHandler, treelite::Model>("learner",
                                                             m_output);
  }

  bool EndObject(std::size_t memberCount) {
    // TODO: version validation
    return this->pop_handler();
  }

private:
  std::vector<unsigned> version;
};

class RootHandler : public OutputHandler<treelite::Model> {
  bool StartObject() {
    return push_handler<XGBoostModelHandler, treelite::Model>(m_output);
  }
};

class DelegatedHandler
    : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, DelegatedHandler>,
      Delegator {

public:
  DelegatedHandler() : delegates{}, result{} {
    push_delegate(std::make_shared<RootHandler>(result));
  };

  void push_delegate(std::shared_ptr<BaseHandler> new_delegate) override {
    delegates.push(new_delegate);
  };
  void pop_delegate() override { delegates.pop(); };
  treelite::Model get_result() { return std::move(result); }
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
    return delegates.top()->EndArray(elementCount);
  }

private:
  std::stack<std::shared_ptr<BaseHandler>> delegates;
  treelite::Model result;
};

template<typename StreamType>
treelite::Model ParseStream(std::unique_ptr<StreamType> input_stream) {

  DelegatedHandler handler;
  rapidjson::Reader reader;

  reader.Parse(*input_stream, handler);

  return handler.get_result();
}


}  // anonymous namespace
